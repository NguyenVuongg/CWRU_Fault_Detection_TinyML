# -*- coding: utf-8 -*-
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
# pyright: reportReturnType=false, reportOperatorIssue=false, reportIndexIssue=false
# (pandas-stubs báo sai ở các lệnh pandas hợp lệ như Series.rename("tên"), groupby().agg(...);
#  đã kiểm tra bằng cách chạy thật, không ảnh hưởng kết quả.)
"""Reusable utilities for the three-method MLP experiment.

This module contains data loading/auditing, file-level grouped CV, model
training, metrics/bootstrap, SNR robustness and parity-check helpers.
The notebook should contain only experiment configuration, calls to these
functions, result display and plotting.

Metrics kept by the notebook: file-level Accuracy and Macro-F1 (mean/std over
seeds, bootstrap CI by file, paired deltas), training loss/accuracy curves and
the feature-group ablation. The SNR sweep, parity-check and detailed per-class
helpers below are no longer called by the notebook but are kept for reuse.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import dsp, features_full, io_utils, models


# ---------------------------------------------------------------------------
# Generic file and table utilities
# ---------------------------------------------------------------------------
def file_stem(value: Any) -> str:
    normalized = str(value).replace("\\", "/")
    return os.path.splitext(normalized.rsplit("/", 1)[-1])[0]


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_table(features_dir: str | os.PathLike, stem: str) -> tuple[pd.DataFrame, str]:
    """Prefer Parquet; if CSV and Parquet both exist, require numeric parity."""
    features_dir = os.fspath(features_dir)
    csv_path = os.path.join(features_dir, stem + ".csv")
    parquet_path = os.path.join(features_dir, stem + ".parquet")
    have_csv, have_parquet = os.path.isfile(csv_path), os.path.isfile(parquet_path)
    if not (have_csv or have_parquet):
        raise FileNotFoundError(f"Không có {stem}.csv/.parquet trong {features_dir}")

    if have_csv and have_parquet:
        csv_df = pd.read_csv(csv_path)
        parquet_df = pd.read_parquet(parquet_path)
        if csv_df.shape != parquet_df.shape or list(csv_df.columns) != list(parquet_df.columns):
            raise ValueError(
                f"{stem}: CSV và Parquet khác cấu trúc ({csv_df.shape} vs {parquet_df.shape}). "
                "Xoá bản cũ hoặc chạy lại notebook tạo bảng đặc trưng."
            )
        numeric_cols = parquet_df.select_dtypes(include="number").columns
        if not np.allclose(
            csv_df[numeric_cols].to_numpy(float),
            parquet_df[numeric_cols].to_numpy(float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"{stem}: CSV và Parquet có giá trị khác nhau; một bản đã lỗi thời. "
                "Xoá bản cũ hoặc chạy lại notebook tạo bảng đặc trưng."
            )
        return parquet_df, parquet_path
    if have_parquet:
        return pd.read_parquet(parquet_path), parquet_path
    return pd.read_csv(csv_path), csv_path


def load_and_audit_feature_tables(
    features_dir: str | os.PathLike,
    feature_files: Mapping[str, str],
    methods: Sequence[str],
    primary_method: str,
    class_names: Sequence[str],
    n_splits: int,
    target_fs_hz: float,
    *,
    group_col: str = "file_id",
    label_col: str = "label",
    strat_col: str = "class_label",
    expected_feature_count: int = 28,
    expected_groups: tuple[int, int, int] = (11, 10, 7),
) -> dict[str, Any]:
    """Load all DSP feature tables and run the strict inventory gate."""
    if primary_method not in methods:
        raise ValueError(f"primary_method={primary_method!r} không nằm trong methods.")
    if set(methods) != set(feature_files):
        raise ValueError("methods và feature_files phải có cùng tập khóa.")

    feature_tables: dict[str, pd.DataFrame] = {}
    table_paths: dict[str, str] = {}
    for method in methods:
        df, path = load_feature_table(features_dir, feature_files[method])
        required = [group_col, label_col, strat_col, "window_idx"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{method}: thiếu cột bắt buộc {missing}.")
        df[group_col] = df[group_col].astype(str)
        df[label_col] = df[label_col].astype(str)
        df[strat_col] = df[strat_col].astype(str)
        feature_tables[method] = df.sort_values([group_col, "window_idx"]).reset_index(drop=True)
        table_paths[method] = path

    table_sha256 = {method: sha256_file(path) for method, path in table_paths.items()}
    reference = feature_tables[primary_method]
    feature_cols = features_full.feature_columns(
        reference, expected_count=expected_feature_count, strict_prefix=True
    )
    time_cols = [c for c in feature_cols if c.startswith("time_")]
    order_cols = [c for c in feature_cols if c.startswith("order_")]
    env_cols = [c for c in feature_cols if c.startswith("envelope_")]
    if (len(time_cols), len(order_cols), len(env_cols)) != expected_groups:
        raise ValueError(
            f"Nhóm đặc trưng phải là {expected_groups}, hiện có "
            f"{(len(time_cols), len(order_cols), len(env_cols))}."
        )

    expected_dsp_method = {
        "hybrid": "hybrid",
        "hilbert_offline": "hilbert_offline",
        "hilbert_block": "hilbert_block",
    }
    shared_meta_cols = [
        group_col, "window_idx", "start_idx", label_col, strat_col,
        "load_hp", "fault_diameter_mils", "rpm", "rpm_source",
    ]
    for method, df in feature_tables.items():
        this_feature_cols = features_full.feature_columns(
            df, expected_count=expected_feature_count, strict_prefix=True
        )
        if this_feature_cols != feature_cols:
            raise ValueError(f"{method}: danh sách/thứ tự feature khác bảng {primary_method}.")
        if "dsp_method" not in df.columns:
            raise ValueError(f"{method}: thiếu metadata dsp_method; hãy build lại bảng từ notebook 05.")
        actual_methods = set(df["dsp_method"].dropna().astype(str).unique())
        expected_method = expected_dsp_method.get(method, method)
        if actual_methods != {expected_method}:
            raise ValueError(
                f"{method}: dsp_method={actual_methods}, cần {expected_method}."
            )
        key_cols = [group_col, "window_idx"]
        if len(df) != len(reference) or not np.array_equal(
            df[key_cols].to_numpy(), reference[key_cols].to_numpy()
        ):
            raise ValueError(f"{method}: tập file/cửa sổ khác bảng {primary_method}.")

        missing_meta = [c for c in shared_meta_cols if c not in df.columns or c not in reference.columns]
        if missing_meta:
            raise ValueError(f"{method}: thiếu metadata để đối chiếu: {missing_meta}.")
        try:
            pd.testing.assert_frame_equal(
                reference[shared_meta_cols].reset_index(drop=True),
                df[shared_meta_cols].reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=0.0,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(f"{method}: metadata/file/window không khớp {primary_method}: {exc}") from exc

        values = df[feature_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{method}: bảng feature chứa NaN/Inf.")
        shared_cols = time_cols + order_cols
        if not np.allclose(
            df[shared_cols].to_numpy(float),
            reference[shared_cols].to_numpy(float),
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                f"{method}: đặc trưng Time/Order khác bảng {primary_method}; "
                "các bảng được tạo từ dữ liệu/phiên bản khác nhau. Tạo lại cả ba."
            )

    meta_cols = [
        c for c in ["load_hp", "fault_diameter_mils", "rpm", "rpm_source",
                    "source_sample_rate_hz", "source_sample_rate_source"]
        if c in reference.columns
    ]
    aggregations: dict[str, tuple[str, str]] = {
        label_col: (label_col, "first"),
        strat_col: (strat_col, "first"),
        "n_windows": ("window_idx", "size"),
    }
    aggregations.update({c: (c, "first") for c in meta_cols})
    file_meta = reference.groupby(group_col, sort=True).agg(**aggregations).reset_index()

    for col in (label_col, strat_col):
        if (reference.groupby(group_col)[col].nunique() != 1).any():
            raise ValueError(f"Có file mang nhiều giá trị {col}.")

    files_per_class = file_meta.groupby(strat_col).size().rename("Số file")
    files_per_label = (
        file_meta.groupby(label_col).size().reindex(pd.Index(class_names)).rename("Số file")
    )
    if files_per_label.isna().any():
        raise ValueError(f"Thiếu nhãn {list(files_per_label[files_per_label.isna()].index)}.")
    if files_per_class.empty or int(files_per_class.min()) < int(n_splits):
        smallest = 0 if files_per_class.empty else int(files_per_class.min())
        raise ValueError(
            f"Điều kiện kiểm kê không thoả: lớp ít file nhất có {smallest} < k={n_splits}. "
            "Giao thức quy định giảm k."
        )

    # Duplicate signal fingerprints based on the first 20 windows' Time features.
    fingerprint: dict[str, str] = {}
    for file_id, group in reference.groupby(group_col):
        block = np.round(
            group.sort_values("window_idx")[time_cols].to_numpy(dtype=np.float64)[:20], 8
        )
        fingerprint[str(file_id)] = hashlib.sha256(block.tobytes()).hexdigest()
    by_fingerprint: dict[str, list[str]] = {}
    for file_id, fingerprint_value in fingerprint.items():
        by_fingerprint.setdefault(fingerprint_value, []).append(file_id)
    duplicates = [items for items in by_fingerprint.values() if len(items) > 1]
    if duplicates:
        raise ValueError(
            f"Các file có tín hiệu trùng nhau (đặc trưng thời gian giống hệt): {duplicates}. "
            "Kiểm tra bước nạp .mat (file có nhiều biến *_DE_time?)."
        )

    provenance_review = file_meta.copy()
    condition = pd.Series(False, index=provenance_review.index)
    if "source_sample_rate_hz" in provenance_review:
        condition |= provenance_review["source_sample_rate_hz"] != float(target_fs_hz)
    if "rpm_source" in provenance_review:
        condition |= provenance_review["rpm_source"].astype(str).map(
            lambda value: re.search(
                "measured|file", str(value), flags=re.IGNORECASE
            ) is None
        )
    provenance_review = provenance_review.loc[condition].copy()

    all_files = sorted(file_meta[group_col].astype(str).tolist())
    strat_by_file = file_meta.set_index(group_col)[strat_col]
    label_by_file = file_meta.set_index(group_col)[label_col]

    return {
        "feature_tables": feature_tables,
        "table_paths": table_paths,
        "table_sha256": table_sha256,
        "feature_cols": feature_cols,
        "time_cols": time_cols,
        "order_cols": order_cols,
        "env_cols": env_cols,
        "file_meta": file_meta,
        "files_per_class": files_per_class,
        "files_per_label": files_per_label,
        "all_files": all_files,
        "strat_by_file": strat_by_file,
        "label_by_file": label_by_file,
        "provenance_review": provenance_review,
        "fingerprints": fingerprint,
        "duplicate_groups": duplicates,
    }


# ---------------------------------------------------------------------------
# File-level stratified group folds
# ---------------------------------------------------------------------------
def _load_manifest_folds(
    manifest_path: str | os.PathLike,
    seeds: Sequence[int],
    all_files: Sequence[str],
    strat_by_file: pd.Series,
    strict_hash: bool,
) -> tuple[dict[int, dict[str, int]] | None, str]:
    path = os.fspath(manifest_path)
    if not os.path.isfile(path):
        return None, f"không có {os.path.basename(path)}"

    sidecar = path + ".sha256"
    if os.path.isfile(sidecar):
        with open(sidecar, encoding="ascii") as stream:
            expected = stream.read().split()[0]
        if sha256_file(path) != expected:
            message = f"SHA-256 của {os.path.basename(path)} không khớp file .sha256"
            if strict_hash:
                raise ValueError(message)
            warnings.warn(message)

    manifest = pd.read_csv(path)
    if "file_id" not in manifest.columns:
        if "file_path" not in manifest.columns:
            return None, "manifest thiếu cả file_id và file_path"
        manifest["file_id"] = manifest["file_path"].map(file_stem)
    manifest["file_id"] = manifest["file_id"].astype(str)

    required_fold_cols = [f"fold_seed_{seed}" for seed in seeds]
    missing = [c for c in required_fold_cols if c not in manifest.columns]
    if missing:
        return None, f"manifest thiếu cột {missing[:3]}..."
    if set(manifest["file_id"]) != set(all_files) or manifest["file_id"].duplicated().any():
        return None, "tập file của manifest fold khác bảng đặc trưng hoặc file_id bị trùng"
    if "class_label" in manifest.columns:
        mismatch = (
            manifest.set_index("file_id")["class_label"]
            .astype(str)
            .reindex(pd.Index(all_files))
            != strat_by_file.reindex(pd.Index(all_files)).astype(str)
        )
        if mismatch.any():
            return None, "class_label trong manifest khác bảng đặc trưng"

    fold_assignments: dict[int, dict[str, int]] = {}
    for seed in seeds:
        seed_assignments = manifest.set_index("file_id")[f"fold_seed_{seed}"].astype(int)
        fold_assignments[int(seed)] = {
            str(file_id): int(fold_id)
            for file_id, fold_id in seed_assignments.items()
        }
    return fold_assignments, os.path.basename(path)


def _compute_fold_assignment(
    seed: int,
    all_files: Sequence[str],
    strat_by_file: pd.Series,
    n_splits: int,
) -> dict[str, int]:
    labels = strat_by_file.reindex(pd.Index(all_files)).astype("category").cat.codes.to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    assignments: dict[str, int] = {}
    for fold_idx, (_, test_idx) in enumerate(
        splitter.split(np.zeros(len(all_files)), labels)
    ):
        for idx in test_idx:
            assignments[str(all_files[idx])] = int(fold_idx)
    return assignments


def validate_group_folds(
    folds: Mapping[int, Sequence[tuple[Sequence[str], Sequence[str]]]],
    all_files: Sequence[str],
    files_per_class: pd.Series,
    label_by_file: pd.Series,
    strat_by_file: pd.Series,
    class_names: Sequence[str],
    n_splits: int,
) -> None:
    all_files_sorted = sorted(map(str, all_files))
    exact_one_per_strat_class = bool((files_per_class == n_splits).all())
    for seed, splits in folds.items():
        if len(splits) != n_splits:
            raise AssertionError(f"Seed {seed}: có {len(splits)} fold, cần {n_splits}.")
        tested = [str(file_id) for _, test_files in splits for file_id in test_files]
        if sorted(tested) != all_files_sorted:
            raise AssertionError(f"Seed {seed}: mỗi file phải được test đúng một lần.")
        for train_files, test_files in splits:
            if set(train_files) & set(test_files):
                raise AssertionError(f"Seed {seed}: file nằm ở cả Train và Test.")
            if set(label_by_file.reindex(pd.Index(test_files)).astype(str)) != set(class_names):
                raise AssertionError(f"Seed {seed}: một fold Test thiếu nhãn 4 lớp.")
            if exact_one_per_strat_class:
                expected = sorted(files_per_class.index.astype(str))
                actual = sorted(strat_by_file.reindex(pd.Index(test_files)).astype(str))
                if actual != expected:
                    raise AssertionError(
                        f"Seed {seed}: fold Test không có đúng 1 file mỗi lớp chi tiết."
                    )


def build_group_folds(
    file_meta: pd.DataFrame,
    features_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    seeds: Sequence[int],
    n_splits: int,
    class_names: Sequence[str],
    *,
    fold_source_preference: str = "auto",
    strict_manifest_hash: bool = True,
    group_col: str = "file_id",
    label_col: str = "label",
    strat_col: str = "class_label",
) -> dict[str, Any]:
    if fold_source_preference not in {"auto", "manifest", "compute"}:
        raise ValueError("fold_source_preference phải là auto, manifest hoặc compute.")
    all_files: list[str] = sorted(str(file_id) for file_id in file_meta[group_col])
    strat_by_file = file_meta.set_index(group_col)[strat_col].astype(str)
    label_by_file = file_meta.set_index(group_col)[label_col].astype(str)
    files_per_class = file_meta.groupby(strat_col).size().rename("Số file")

    manifest_folds, manifest_note = (None, "tắt theo cấu hình")
    if fold_source_preference != "compute":
        manifest_folds, manifest_note = _load_manifest_folds(
            os.path.join(os.fspath(features_dir), "manifest_with_folds.csv"),
            seeds, all_files, strat_by_file, strict_manifest_hash,
        )
    if fold_source_preference == "manifest" and manifest_folds is None:
        raise RuntimeError(f"Yêu cầu dùng fold của manifest nhưng: {manifest_note}")

    use_manifest = manifest_folds is not None
    fold_source = "manifest_with_folds.csv" if use_manifest else "tính tại chỗ (StratifiedKFold)"
    folds: dict[int, list[tuple[list[str], list[str]]]] = {}
    for seed in seeds:
        assignment: dict[str, int] = (
            manifest_folds[int(seed)] if use_manifest
            else _compute_fold_assignment(int(seed), all_files, strat_by_file, n_splits)
        )
        fold_ids = sorted(set(assignment.values()))
        folds[int(seed)] = [
            (
                [file_id for file_id in all_files if assignment[file_id] != fold_id],
                [file_id for file_id in all_files if assignment[file_id] == fold_id],
            )
            for fold_id in fold_ids
        ]

    validate_group_folds(
        folds, all_files, files_per_class, label_by_file, strat_by_file,
        class_names, n_splits,
    )

    fold_audit = pd.DataFrame([
        {
            "seed": seed,
            "fold": fold_idx,
            "file_id": file_id,
            "label": label_by_file[file_id],
            "class_label": strat_by_file[file_id],
        }
        for seed, splits in folds.items()
        for fold_idx, (_, test_files) in enumerate(splits, start=1)
        for file_id in test_files
    ])
    os.makedirs(output_dir, exist_ok=True)
    fold_audit.to_csv(os.path.join(os.fspath(output_dir), "folds_used.csv"), index=False)

    fold_summary_rows = []
    for fold_idx, (train_files, test_files) in enumerate(folds[int(seeds[0])], start=1):
        row: dict[str, Any] = {
            "fold": fold_idx,
            "n_train_files": len(train_files),
            "n_test_files": len(test_files),
        }
        for class_name in class_names:
            row[f"test_{class_name}"] = int((label_by_file.reindex(test_files) == class_name).sum())
            row[f"train_{class_name}"] = int((label_by_file.reindex(train_files) == class_name).sum())
        fold_summary_rows.append(row)

    return {
        "folds": folds,
        "fold_source": fold_source,
        "manifest_note": manifest_note,
        "fold_audit": fold_audit,
        "fold_summary": pd.DataFrame(fold_summary_rows),
        "all_files": all_files,
        "strat_by_file": strat_by_file,
        "label_by_file": label_by_file,
    }


# ---------------------------------------------------------------------------
# MLP fitting and file-level predictions
# ---------------------------------------------------------------------------
def aggregate_file_predictions(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    group_col: str = "file_id",
    label_col: str = "label",
) -> list[dict[str, Any]]:
    """Convert window probabilities to one prediction per source file."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) != len(test_df):
        raise ValueError("probabilities phải có shape (số cửa sổ, số lớp).")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("Số cột xác suất không khớp class_names.")

    file_values = test_df[group_col].astype(str).to_numpy()
    rows = []
    for file_id in pd.unique(file_values):
        indices = np.flatnonzero(file_values == file_id)
        truths = test_df[label_col].iloc[indices].astype(str).unique()
        if len(truths) != 1:
            raise ValueError(f"File {file_id!r} có nhiều nhãn thực.")
        mean_probability = probabilities[indices].mean(axis=0)
        rows.append({
            "file_id": file_id,
            "y_true": truths[0],
            "y_pred": class_names[int(np.argmax(mean_probability))],
            "n_windows": int(len(indices)),
            **{f"p_{class_name}": float(mean_probability[i])
               for i, class_name in enumerate(class_names)},
        })
    return rows


def _fit_one_fold(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    seed: int,
    fold_id: int,
    class_names: Sequence[str],
    class_to_id: Mapping[str, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mlp_kwargs: Mapping[str, Any],
    use_class_weight: bool,
    label_col: str = "label",
):
    import tensorflow as tf

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[list(feature_cols)].to_numpy(dtype=np.float32))
    y_series = train_df[label_col].astype(str).map(class_to_id)
    if bool(y_series.isna().any()):
        unknown = sorted(train_df.loc[y_series.isna(), label_col].astype(str).unique())
        raise ValueError(f"Train chứa nhãn không có trong class_to_id: {unknown}.")
    y_train = y_series.to_numpy(dtype=np.int32)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed) * 100 + int(fold_id))
    model = models.build_primary_mlp(
        input_dim=len(feature_cols), n_classes=len(class_names), **dict(mlp_kwargs)
    )
    models.compile_classifier(model, learning_rate=learning_rate, n_classes=len(class_names))
    fit_kwargs: dict[str, Any] = {}
    if use_class_weight:
        fit_kwargs["class_weight"] = models.compute_class_weights(y_train, len(class_names))
    history = model.fit(
        x_train, y_train, epochs=int(epochs), batch_size=int(batch_size),
        verbose=0, **fit_kwargs,
    )
    return model, scaler, history.history


def run_cross_validation(
    method: str,
    table: pd.DataFrame,
    feature_cols: Sequence[str],
    seeds: Sequence[int],
    folds: Mapping[int, Sequence[tuple[Sequence[str], Sequence[str]]]],
    class_names: Sequence[str],
    class_to_id: Mapping[str, int],
    mlp_kwargs: Mapping[str, Any],
    *,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    use_class_weight: bool = False,
    variant: str = "All 28 features",
    keep_state: bool = False,
    group_col: str = "file_id",
    label_col: str = "label",
) -> dict[str, Any]:
    """Run grouped CV for one DSP method; all preprocessing is fit on Train only."""
    import tensorflow as tf

    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    fold_state: dict[tuple[str, int, int], tuple[list[np.ndarray], StandardScaler]] = {}
    start_time = time.time()

    for seed in seeds:
        if int(seed) not in folds:
            raise KeyError(f"Không có fold assignment cho seed {seed}.")
        for fold_id, (train_files, test_files) in enumerate(folds[int(seed)], start=1):
            train_file_set, test_file_set = set(train_files), set(test_files)
            train_df = table[table[group_col].astype(str).isin(train_file_set)]
            test_df = table[table[group_col].astype(str).isin(test_file_set)].reset_index(drop=True)
            if train_file_set & test_file_set:
                raise RuntimeError("File bị chia giữa Train và Test.")
            if train_df.empty or test_df.empty:
                raise RuntimeError(f"{method}, seed={seed}, fold={fold_id}: Train/Test rỗng.")

            model, scaler, history = _fit_one_fold(
                train_df, feature_cols, int(seed), fold_id, class_names, class_to_id,
                epochs, batch_size, learning_rate, mlp_kwargs, use_class_weight,
                label_col=label_col,
            )
            losses = history.get("loss", [])
            accuracies = history.get("accuracy", [np.nan] * len(losses))
            for epoch_no, loss_value in enumerate(losses, start=1):
                curve_rows.append({
                    "method": method,
                    "variant": variant,
                    "seed": int(seed),
                    "fold": fold_id,
                    "epoch": epoch_no,
                    "loss": float(loss_value),
                    "accuracy": float(accuracies[epoch_no - 1]) if epoch_no - 1 < len(accuracies) else np.nan,
                })

            x_test = scaler.transform(test_df[list(feature_cols)].to_numpy(dtype=np.float32))
            probabilities = model.predict(x_test, verbose=0)
            for row in aggregate_file_predictions(test_df, probabilities, class_names,
                                                  group_col=group_col, label_col=label_col):
                row.update(method=method, variant=variant, seed=int(seed), fold=fold_id)
                rows.append(row)

            if keep_state:
                fold_state[(method, int(seed), fold_id)] = (
                    [np.array(weight, copy=True) for weight in model.get_weights()], scaler
                )
            del model
        print(f"  {method:20s} seed {seed} xong ({time.time() - start_time:5.0f}s)")

    tf.keras.backend.clear_session()
    return {
        "predictions": pd.DataFrame(rows),
        "fold_state": fold_state,
        "learning_curves": curve_rows,
        "elapsed_seconds": time.time() - start_time,
    }


# ---------------------------------------------------------------------------
# Training loss / accuracy curves
# ---------------------------------------------------------------------------
def summarize_learning_curves(
    curve_rows: Iterable[Mapping[str, Any]],
    group_cols: Sequence[str] = ("method", "variant"),
) -> pd.DataFrame:
    """Train loss/accuracy per epoch: mean and std over models (seed x fold)."""
    curves = pd.DataFrame(list(curve_rows))
    if curves.empty:
        raise ValueError("Không có learning curve để tổng hợp.")
    return (
        curves.groupby([*group_cols, "epoch"], sort=False)
        .agg(n_models=("loss", "size"),
             loss_mean=("loss", "mean"), loss_std=("loss", "std"),
             acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"))
        .reset_index()
    )


def final_epoch_table(
    curves: pd.DataFrame,
    group_cols: Sequence[str] = ("method",),
    tail: int = 10,
) -> pd.DataFrame:
    """Convergence diagnostics on Train only: epoch-1 and last-epoch loss/accuracy and the
    relative loss drop over the last `tail` epochs (descriptive, not for epoch selection)."""
    rows = []
    for key, group in curves.groupby(list(group_cols), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        last_epoch = int(group["epoch"].max())
        first, last = group[group["epoch"] == 1], group[group["epoch"] == last_epoch]
        mean_loss = group.groupby("epoch")["loss"].mean()
        back = min(int(tail), last_epoch - 1)
        drop = (
            100.0 * (mean_loss.loc[last_epoch - back] - mean_loss.loc[last_epoch])
            / mean_loss.loc[last_epoch - back]
        ) if back > 0 else np.nan
        rows.append({
            **dict(zip(group_cols, key_tuple)),
            "n_models": int(last["loss"].size),
            "loss_epoch_1": float(first["loss"].mean()),
            "loss_last_mean": float(last["loss"].mean()),
            "loss_last_std": float(last["loss"].std()),
            "acc_last_mean": float(last["accuracy"].mean()),
            "acc_last_std": float(last["accuracy"].std()),
            "loss_drop_last_epochs_pct": float(drop),
            "tail_epochs": back,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# File-level metrics and paired bootstrap
# ---------------------------------------------------------------------------
def make_bootstrap_indices(n_files: int, n_boot: int, seed: int) -> np.ndarray:
    if n_files <= 0 or n_boot <= 0:
        raise ValueError("n_files và n_boot phải dương.")
    return np.random.default_rng(seed).integers(0, n_files, size=(n_boot, n_files))


def replicate_arrays(
    pred_df: pd.DataFrame,
    all_files: Sequence[str],
    class_names: Sequence[str],
    rep_cols: Sequence[str] = ("seed",),
) -> tuple[np.ndarray, np.ndarray, list[tuple[Any, ...]]]:
    class_index = {name: i for i, name in enumerate(class_names)}
    true_replicates, pred_replicates, keys = [], [], []
    if pred_df.empty:
        raise ValueError("Không có dự đoán để tính metric.")
    for key, group in pred_df.groupby(list(rep_cols), sort=True):
        indexed = (
            group.assign(file_id=group["file_id"].astype(str))
            .set_index("file_id")
            .reindex(pd.Index(all_files))
        )
        if bool(indexed["y_true"].isna().any()) or bool(indexed["y_pred"].isna().any()):
            raise ValueError(f"Bản sao {key}: thiếu dự đoán cho một số file.")
        unknown = (set(indexed["y_true"].astype(str)) | set(indexed["y_pred"].astype(str))) - set(class_index)
        if unknown:
            raise ValueError(f"Bản sao {key}: nhãn không hợp lệ {sorted(unknown)}.")
        true_replicates.append(indexed["y_true"].astype(str).map(class_index).to_numpy(dtype=np.int32))
        pred_replicates.append(indexed["y_pred"].astype(str).map(class_index).to_numpy(dtype=np.int32))
        keys.append(key if isinstance(key, tuple) else (key,))
    return np.asarray(true_replicates), np.asarray(pred_replicates), keys


def confusion_counts(true_ids: np.ndarray, pred_ids: np.ndarray, n_classes: int) -> np.ndarray:
    """T, P: (..., n_files) -> (..., K, K), row=true and col=predicted."""
    true_ids, pred_ids = np.asarray(true_ids), np.asarray(pred_ids)
    if true_ids.shape != pred_ids.shape:
        raise ValueError("true_ids và pred_ids phải cùng shape.")
    if ((true_ids < 0) | (true_ids >= n_classes) | (pred_ids < 0) | (pred_ids >= n_classes)).any():
        raise ValueError("Class id ngoài phạm vi.")
    lead_shape = true_ids.shape[:-1]
    n_files = true_ids.shape[-1]
    n_replicates = int(np.prod(lead_shape)) if lead_shape else 1
    codes = (true_ids * n_classes + pred_ids).reshape(n_replicates, n_files)
    offsets = (np.arange(n_replicates) * n_classes * n_classes)[:, None]
    return np.bincount(
        (codes + offsets).ravel(), minlength=n_replicates * n_classes * n_classes
    ).reshape(*lead_shape, n_classes, n_classes)


def accuracy_from_cm(confusion: np.ndarray) -> np.ndarray:
    confusion = np.asarray(confusion)
    denominator = confusion.sum(axis=(-2, -1))
    return np.divide(
        np.trace(confusion, axis1=-2, axis2=-1), denominator,
        out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0,
    )


def macro_f1_from_cm(confusion: np.ndarray) -> np.ndarray:
    confusion = np.asarray(confusion)
    tp = np.diagonal(confusion, axis1=-2, axis2=-1).astype(float)
    fp = confusion.sum(axis=-2) - tp
    fn = confusion.sum(axis=-1) - tp
    denominator = 2 * tp + fp + fn
    present = (confusion.sum(axis=-1) + confusion.sum(axis=-2)) > 0
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    n_present = present.sum(axis=-1)
    return np.divide(
        (f1 * present).sum(axis=-1), n_present,
        out=np.full_like(n_present, np.nan, dtype=float), where=n_present > 0,
    )


def bootstrap_stats(
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    bootstrap_indices: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    confusion = confusion_counts(true_ids[:, bootstrap_indices], pred_ids[:, bootstrap_indices], n_classes)
    return accuracy_from_cm(confusion).mean(axis=0), macro_f1_from_cm(confusion).mean(axis=0)


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Không thể tính percentile CI từ dữ liệu rỗng hoặc không hữu hạn.")
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize_conditions(
    pred_df: pd.DataFrame,
    cond_col: str,
    order: Sequence[str],
    reference: str,
    all_files: Sequence[str],
    class_names: Sequence[str],
    bootstrap_indices: np.ndarray,
    *,
    rep_cols: Sequence[str] = ("seed",),
    alpha: float = 0.05,
    margin: float | None = None,
) -> dict[str, Any]:
    """Summarize conditions; generic delta uses reference minus each other condition."""
    n_classes = len(class_names)
    n_files = len(all_files)
    boots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    confusion_matrices: dict[str, np.ndarray] = {}
    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for condition in order:
        subset = pred_df[pred_df[cond_col] == condition]
        true_ids, pred_ids, keys = replicate_arrays(subset, all_files, class_names, rep_cols)
        confusion_per_rep = confusion_counts(true_ids, pred_ids, n_classes)
        accuracy_per_rep = accuracy_from_cm(confusion_per_rep)
        f1_per_rep = macro_f1_from_cm(confusion_per_rep)
        
        replicate_seed = np.asarray([str(key[0]) for key in keys])
        unique_seeds = np.unique(replicate_seed)
        accuracy_by_seed = np.asarray([accuracy_per_rep[replicate_seed == seed].mean() for seed in unique_seeds])
        f1_by_seed = np.asarray([f1_per_rep[replicate_seed == seed].mean() for seed in unique_seeds])

        for seed, accuracy, f1 in zip(unique_seeds, accuracy_by_seed, f1_by_seed):
            per_seed_rows.append({
                cond_col: condition, "seed": int(seed),
                "accuracy": float(accuracy), "macro_f1": float(f1),
            })

        boots[condition] = bootstrap_stats(true_ids, pred_ids, bootstrap_indices, n_classes)
        confusion_matrices[condition] = confusion_per_rep.sum(axis=0)
        accuracy_ci = percentile_ci(boots[condition][0], alpha)
        f1_ci = percentile_ci(boots[condition][1], alpha)
        summary_rows.append({
            cond_col: condition,
            "accuracy_mean": float(accuracy_by_seed.mean()),
            "accuracy_std_seeds": float(accuracy_by_seed.std(ddof=1)) if len(accuracy_by_seed) > 1 else np.nan,
            "accuracy_ci_lo": accuracy_ci[0], "accuracy_ci_hi": accuracy_ci[1],
            "macro_f1_mean": float(f1_by_seed.mean()),
            "macro_f1_std_seeds": float(f1_by_seed.std(ddof=1)) if len(f1_by_seed) > 1 else np.nan,
            "macro_f1_ci_lo": f1_ci[0], "macro_f1_ci_hi": f1_ci[1],
            "n_files": n_files, "n_seeds": int(len(unique_seeds)),
        })

    summary_df = pd.DataFrame(summary_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    if reference not in set(summary_df[cond_col]):
        raise ValueError(f"Không tìm thấy reference={reference!r} trong {cond_col}.")
    reference_row = summary_df[summary_df[cond_col] == reference].iloc[0]
    delta_rows = []
    for condition in order:
        if condition == reference:
            continue
        condition_row = summary_df[summary_df[cond_col] == condition].iloc[0]
        entry: dict[str, Any] = {"reference": reference, "versus": condition}
        for metric_idx, metric_name in enumerate(("accuracy", "macro_f1")):
            delta_values = boots[reference][metric_idx] - boots[condition][metric_idx]
            ci = percentile_ci(delta_values, alpha)
            entry[f"delta_{metric_name}"] = float(
                reference_row[f"{metric_name}_mean"] - condition_row[f"{metric_name}_mean"]
            )
            entry[f"delta_{metric_name}_ci_lo"], entry[f"delta_{metric_name}_ci_hi"] = ci
        if margin is not None:
            entry["non_inferior_accuracy"] = bool(entry["delta_accuracy_ci_lo"] > -margin)
            entry["non_inferior_macro_f1"] = bool(entry["delta_macro_f1_ci_lo"] > -margin)
        delta_rows.append(entry)

    return {
        "summary": summary_df,
        "delta": pd.DataFrame(delta_rows),
        "per_seed": per_seed_df,
        "cm": confusion_matrices,
    }


def paired_method_deltas(
    pred_df: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    methods: Sequence[str],
    all_files: Sequence[str],
    class_names: Sequence[str],
    bootstrap_indices: np.ndarray,
    *,
    rep_cols: Sequence[str] = ("seed",),
    alpha: float = 0.05,
    margin: float | None = None,
) -> pd.DataFrame:
    """Candidate minus reference, paired by identical replicate and file bootstrap."""
    n_classes = len(class_names)
    arrays: dict[str, tuple[np.ndarray, np.ndarray, list[tuple[Any, ...]]]] = {}
    for method in methods:
        subset = pred_df[pred_df["method"] == method]
        arrays[method] = replicate_arrays(subset, all_files, class_names, rep_cols)

    rows = []
    for candidate, reference in pairs:
        if candidate not in arrays or reference not in arrays:
            raise ValueError(f"Cặp phương pháp không hợp lệ: {candidate} / {reference}.")
        candidate_true, candidate_pred, candidate_keys = arrays[candidate]
        reference_true, reference_pred, reference_keys = arrays[reference]
        if candidate_keys != reference_keys or not np.array_equal(candidate_true, reference_true):
            raise ValueError(
                f"{candidate} và {reference}: replicate hoặc thứ tự nhãn file không khớp."
            )

        boot_candidate = bootstrap_stats(candidate_true, candidate_pred, bootstrap_indices, n_classes)
        boot_reference = bootstrap_stats(reference_true, reference_pred, bootstrap_indices, n_classes)
        delta_accuracy = boot_candidate[0] - boot_reference[0]
        delta_f1 = boot_candidate[1] - boot_reference[1]
        accuracy_ci = percentile_ci(delta_accuracy, alpha)
        f1_ci = percentile_ci(delta_f1, alpha)
        cm_candidate = confusion_counts(candidate_true, candidate_pred, n_classes)
        cm_reference = confusion_counts(reference_true, reference_pred, n_classes)
        point_accuracy = float(accuracy_from_cm(cm_candidate).mean() - accuracy_from_cm(cm_reference).mean())
        point_f1 = float(macro_f1_from_cm(cm_candidate).mean() - macro_f1_from_cm(cm_reference).mean())

        row: dict[str, Any] = {
            "candidate": candidate, "reference": reference,
            "delta_accuracy": point_accuracy,
            "delta_accuracy_ci_lo": accuracy_ci[0], "delta_accuracy_ci_hi": accuracy_ci[1],
            "delta_macro_f1": point_f1,
            "delta_macro_f1_ci_lo": f1_ci[0], "delta_macro_f1_ci_hi": f1_ci[1],
            "n_replicates": len(candidate_keys), "n_files": len(all_files),
        }
        if margin is not None and candidate == "hybrid" and reference == "hilbert_offline":
            row["non_inferior_accuracy"] = bool(accuracy_ci[0] > -margin)
            row["non_inferior_macro_f1"] = bool(f1_ci[0] > -margin)
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_recall(
    pred_df: pd.DataFrame,
    cond_col: str,
    order: Sequence[str],
    class_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for condition in order:
        subset = pred_df[pred_df[cond_col] == condition]
        for class_name in class_names:
            class_rows = subset[subset["y_true"] == class_name]
            rows.append({
                cond_col: condition,
                "class": class_name,
                "n_correct": int((class_rows["y_pred"] == class_name).sum()),
                "n_total": int(len(class_rows)),
                "n_files": int(class_rows["file_id"].nunique()),
                "recall": float((class_rows["y_pred"] == class_name).mean()) if len(class_rows) else np.nan,
            })
    return pd.DataFrame(rows)


def file_error_table(pred_df: pd.DataFrame, cond_col: str) -> pd.DataFrame:
    data = pred_df.assign(wrong=pred_df["y_true"] != pred_df["y_pred"])
    wrong = data[data["wrong"]]
    most_common = (
        wrong.groupby([cond_col, "file_id"])["y_pred"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("dự_đoán_hay_gặp")
    )
    table = (
        data.groupby([cond_col, "file_id", "y_true"])
        .agg(n_sai=("wrong", "sum"), n_seed=("wrong", "size"))
        .reset_index()
    )
    table = table[table["n_sai"] > 0].merge(
        most_common.reset_index(), on=[cond_col, "file_id"], how="left"
    )
    return table.sort_values([cond_col, "n_sai"], ascending=[True, False]).reset_index(drop=True)


def evaluate_clean_predictions(
    predictions: pd.DataFrame,
    methods: Sequence[str],
    method_comparisons: Sequence[tuple[str, str]],
    primary_method: str,
    all_files: Sequence[str],
    class_names: Sequence[str],
    bootstrap_indices: np.ndarray,
    *,
    alpha: float = 0.05,
    noninferiority_margin: float | None = None,
) -> dict[str, Any]:
    main = summarize_conditions(
        predictions, "method", methods, primary_method, all_files, class_names,
        bootstrap_indices, alpha=alpha,
    )
    return {
        "main": main,
        "summary": main["summary"],
        "metrics_by_seed": main["per_seed"],
        "delta_table": paired_method_deltas(
            predictions, method_comparisons, methods, all_files, class_names,
            bootstrap_indices, alpha=alpha, margin=noninferiority_margin,
        ),
        "recall_table": per_class_recall(predictions, "method", methods, class_names),
        "error_table": file_error_table(predictions, "method"),
    }


def detailed_metrics_by_seed(
    pred_df: pd.DataFrame,
    method: str,
    all_files: Sequence[str],
    class_names: Sequence[str],
) -> pd.DataFrame:
    true_ids, pred_ids, keys = replicate_arrays(
        pred_df[pred_df["method"] == method], all_files, class_names, ("seed",)
    )
    confusion = confusion_counts(true_ids, pred_ids, len(class_names)).astype(float)
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    support, predicted = confusion.sum(axis=-1), confusion.sum(axis=-2)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    f1 = np.divide(
        2 * true_positive, support + predicted,
        out=np.zeros_like(true_positive), where=(support + predicted) > 0,
    )
    accuracy = accuracy_from_cm(confusion)
    rows = []
    for replicate_idx, key in enumerate(keys):
        wrong_files = [
            str(all_files[idx]) for idx in np.flatnonzero(true_ids[replicate_idx] != pred_ids[replicate_idx])
        ]
        rows.append({
            "method": method,
            "seed": int(str(key[0])),
            "accuracy": accuracy[replicate_idx],
            "macro_f1": macro_f1_from_cm(confusion[replicate_idx]),
            "macro_precision": precision[replicate_idx].mean(),
            "balanced_accuracy": recall[replicate_idx].mean(),
            **{f"recall_{name}": recall[replicate_idx, idx] for idx, name in enumerate(class_names)},
            **{f"f1_{name}": f1[replicate_idx, idx] for idx, name in enumerate(class_names)},
            "n_files_sai": len(wrong_files),
            "files_sai": ", ".join(wrong_files) if wrong_files else "-",
        })
    return pd.DataFrame(rows)



def build_detailed_metrics_table(
    predictions: pd.DataFrame,
    methods: Sequence[str],
    all_files: Sequence[str],
      class_names: Sequence[str],
    metrics_by_seed: pd.DataFrame,
) -> pd.DataFrame:

    details = pd.concat(
        [
            detailed_metrics_by_seed(
                predictions, method, all_files, class_names
            )
            for method in methods
        ],
        ignore_index=True,
    )

    checked = details.merge(
        metrics_by_seed,
        on=["method", "seed"],
        suffixes=("", "_ref"),
        how="left",
    )

    if len(checked) != len(details):
        raise AssertionError(
            "Bảng chi tiết không ghép đủ với bảng metric theo seed."
        )

    # Chuẩn hóa dữ liệu về kiểu số trước khi so sánh
    for column in ["accuracy", "macro_f1"]:
        ref_column = f"{column}_ref"

        checked[column] = pd.to_numeric(
            checked[column], errors="coerce"
        )
        checked[ref_column] = pd.to_numeric(
            checked[ref_column], errors="coerce"
        )

        # Kiểm tra dữ liệu thiếu hoặc không hợp lệ
        if checked[[column, ref_column]].isna().any().any():
            raise ValueError(
                f"Cột {column} chứa giá trị thiếu hoặc không phải số. "
                f"Kiểm tra metrics_by_seed."
            )

        # Chuyển thành NumPy float array
        values = checked[column].to_numpy(dtype=np.float64)
        reference = checked[ref_column].to_numpy(dtype=np.float64)

        if not np.allclose(
            values,
            reference,
            rtol=1e-6,
            atol=1e-8,
        ):
            max_error = np.max(np.abs(values - reference))
            raise AssertionError(
                f"Bảng chi tiết không khớp metric theo seed: {column}. "
                f"Sai lệch lớn nhất = {max_error:.8g}"
            )

    return details



# ---------------------------------------------------------------------------
# SNR robustness and clean-feature parity
# ---------------------------------------------------------------------------
def find_raw_manifest(
    features_dir: str | os.PathLike,
    stage1_dir: str | os.PathLike,
    project_root: str | os.PathLike,
    all_files: Sequence[str],
    class_names: Sequence[str],
) -> tuple[pd.DataFrame | None, str]:
    candidates = [
        os.path.join(os.fspath(features_dir), "manifest_filtered.csv"),
        os.path.join(os.fspath(features_dir), "manifest.csv"),
    ]
    manifest_path = next((path for path in candidates if os.path.isfile(path)), None)
    if manifest_path is None:
        return None, "không thấy manifest_filtered.csv/manifest.csv"
    raw = pd.read_csv(manifest_path)
    required = [
        "file_path", "label", "fault_diameter_mils", "resolved_sample_rate_hz",
        "resolved_sample_rate_source", "rpm", "rpm_source",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        return None, f"manifest thiếu cột {missing}"

    stage1_dir = os.path.abspath(os.fspath(stage1_dir))
    project_root = os.path.abspath(os.fspath(project_root))

    def resolve_raw_path(value: Any) -> str:
        raw_path = os.path.expandvars(os.path.expanduser(str(value)))
        path_candidates = [raw_path] if os.path.isabs(raw_path) else [
            os.path.abspath(os.path.join(stage1_dir, raw_path)),
            os.path.abspath(os.path.join(project_root, raw_path)),
            os.path.abspath(raw_path),
        ]
        return next((candidate for candidate in path_candidates if os.path.isfile(candidate)), path_candidates[0])

    raw["file_id"] = raw["file_path"].map(file_stem)
    raw = raw[
        raw["file_id"].isin(set(all_files)) & raw["label"].astype(str).isin(class_names)
    ].copy()
    raw["file_path"] = raw["file_path"].map(resolve_raw_path)
    missing_path_count = int((~raw["file_path"].map(os.path.isfile)).sum())
    missing_ids = sorted(set(all_files) - set(raw["file_id"]))
    if missing_path_count or missing_ids or raw["file_id"].duplicated().any():
        return None, (
            f"không ghép đủ dữ liệu raw: {missing_path_count} đường dẫn thiếu, "
            f"{len(missing_ids)} file chưa có trong manifest hoặc file_id trùng"
        )
    return raw.reset_index(drop=True), manifest_path


def check_dsp_methods(
    methods: Sequence[str],
    target_fs_hz: float,
    band_hz: Sequence[float],
    lp_cutoff_hz: float,
    bandpass_order: int,
    lowpass_order: int,
    envelope_method_kwargs: Mapping[str, Mapping[str, Any]],
) -> None:
    probe = np.random.default_rng(0).standard_normal(5003)
    for method in methods:
        try:
            kwargs = cast("dict[str, Any]", dict(envelope_method_kwargs[method]))
            envelope = dsp.envelope_by_method(
                probe, target_fs_hz, band=band_hz, method=method,
                lp_cutoff=lp_cutoff_hz, bandpass_order=bandpass_order,
                lowpass_order=lowpass_order, **kwargs,
            )
            if len(envelope) != len(probe) or not np.isfinite(envelope).all():
                raise ValueError("Envelope không cùng chiều dài hoặc có NaN/Inf.")
        except Exception as exc:
            raise RuntimeError(
                f"dsp.envelope_by_method không hỗ trợ method={method!r}: "
                "kiểm tra src/dsp.py và cấu hình Hilbert Block."
            ) from exc


def build_feature_table_for(
    raw_manifest: pd.DataFrame,
    method: str,
    loader: Callable,
    *,
    band_hz: Sequence[float],
    window_size: int,
    stride: int,
    warmup_samples: int,
    target_fs_hz: float,
    lp_cutoff_hz: float,
    bandpass_order: int,
    lowpass_order: int,
    envelope_method_kwargs: Mapping[str, Mapping[str, Any]],
    group_col: str = "file_id",
    label_col: str = "label",
) -> pd.DataFrame:
    kwargs = cast("dict[str, Any]", dict(envelope_method_kwargs[method]))
    table = features_full.build_full_feature_table(
        raw_manifest,
        band_hz=band_hz,
        load_de_signal_fn=loader,
        window_size=window_size,
        stride=stride,
        warmup_samples=warmup_samples,
        target_fs_hz=target_fs_hz,
        lp_cutoff_hz=lp_cutoff_hz,
        bandpass_order=bandpass_order,
        lowpass_order=lowpass_order,
        envelope_method=method,
        envelope_kwargs=kwargs,
    )
    if table.empty:
        raise RuntimeError(f"Không tạo được feature cho method={method}.")
    table[group_col] = table[group_col].astype(str)
    table[label_col] = table[label_col].astype(str)
    return table.sort_values([group_col, "window_idx"]).reset_index(drop=True)


def run_parity_check(
    raw_manifest: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    methods: Sequence[str],
    all_files: Sequence[str],
    label_by_file: pd.Series,
    feature_cols: Sequence[str],
    *,
    band_hz: Sequence[float],
    window_size: int,
    stride: int,
    warmup_samples: int,
    target_fs_hz: float,
    lp_cutoff_hz: float,
    bandpass_order: int,
    lowpass_order: int,
    envelope_method_kwargs: Mapping[str, Mapping[str, Any]],
) -> None:
    available = set(raw_manifest["file_id"].astype(str))
    chosen = [file_id for file_id in all_files if label_by_file[file_id] == "Normal"]
    chosen += [
        sorted(file_id for file_id in all_files if label_by_file[file_id] == class_name)[0]
        for class_name in label_by_file.unique() if class_name != "Normal"
    ]
    chosen = [file_id for file_id in chosen if file_id in available]
    subset = raw_manifest[raw_manifest["file_id"].astype(str).isin(chosen)].reset_index(drop=True)
    problems = []
    for method in methods:
        rebuilt = build_feature_table_for(
            subset, method, io_utils.load_de_signal_resampled,
            band_hz=band_hz, window_size=window_size, stride=stride,
            warmup_samples=warmup_samples, target_fs_hz=target_fs_hz,
            lp_cutoff_hz=lp_cutoff_hz, bandpass_order=bandpass_order,
            lowpass_order=lowpass_order, envelope_method_kwargs=envelope_method_kwargs,
        )
        saved = feature_tables[method]
        saved = saved[saved["file_id"].astype(str).isin(chosen)].reset_index(drop=True)
        key_cols = ["file_id", "window_idx"]
        same_keys = (
            len(rebuilt) == len(saved)
            and np.array_equal(rebuilt[key_cols].to_numpy(), saved[key_cols].to_numpy())
        )
        if not same_keys:
            problems.append(f"{method}: số cửa sổ/khóa khác bảng ({len(rebuilt)} vs {len(saved)})")
            continue
        difference = ~np.isclose(
            rebuilt[list(feature_cols)].to_numpy(float),
            saved[list(feature_cols)].to_numpy(float),
            rtol=1e-7,
            atol=1e-9,
        )
        if difference.any():
            bad_files = sorted(rebuilt.loc[difference.any(axis=1), "file_id"].unique())
            problems.append(f"{method}: đặc trưng khác bảng ở {bad_files}")
    if problems:
        raise RuntimeError(
            "Parity check THẤT BẠI — bảng đặc trưng không khớp pipeline hiện tại:\n  - "
            + "\n  - ".join(problems)
            + "\nChạy lại notebook tạo bảng đặc trưng (và tạo lại manifest) rồi chạy lại."
        )
    print(f"Parity check: ĐẠT trên {len(chosen)} file × {len(methods)} phương pháp.")


def add_awgn(signal: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - np.mean(signal)
    signal_power = np.mean(centered ** 2)
    noise = np.random.default_rng(seed).standard_normal(signal.shape)
    noise_power = np.mean(noise ** 2)
    if signal_power <= 0 or noise_power <= 0:
        return signal.copy()
    noise *= np.sqrt(signal_power / (10.0 ** (float(snr_db) / 10.0)) / noise_power)
    return signal + noise


def awgn_seed(snr_db: float, draw: int, file_id: str) -> int:
    text = f"CWRU-AWGN-v2|{snr_db}|{draw}|{file_id}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def make_awgn_loader(snr_db: float, draw: int) -> Callable:
    def loader(filepath, source_fs_hz, target_fs_hz):
        signal = io_utils.load_de_signal_resampled(filepath, source_fs_hz, target_fs_hz)
        return add_awgn(signal, snr_db, awgn_seed(snr_db, draw, file_stem(filepath)))
    return loader


def run_noise_sweep(
    raw_manifest: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    methods: Sequence[str],
    method_comparisons: Sequence[tuple[str, str]],
    folds: Mapping[int, Sequence[tuple[Sequence[str], Sequence[str]]]],
    fold_state: Mapping[tuple[str, int, int], tuple[list[np.ndarray], StandardScaler]],
    feature_cols: Sequence[str],
    class_names: Sequence[str],
    class_to_id: Mapping[str, int],
    all_files: Sequence[str],
    bootstrap_indices: np.ndarray,
    primary_method: str,
    mlp_kwargs: Mapping[str, Any],
    *,
    snr_levels_db: Sequence[float],
    n_noise_draws: int,
    noninferiority_margin: float | None,
    alpha: float,
    band_hz: Sequence[float],
    window_size: int,
    stride: int,
    warmup_samples: int,
    target_fs_hz: float,
    lp_cutoff_hz: float,
    bandpass_order: int,
    lowpass_order: int,
    envelope_method_kwargs: Mapping[str, Mapping[str, Any]],
    run_parity: bool = True,
) -> dict[str, Any]:
    """Build noisy feature tables and evaluate saved clean models without refitting."""
    import tensorflow as tf

    check_dsp_methods(
        methods, target_fs_hz, band_hz, lp_cutoff_hz,
        bandpass_order, lowpass_order, envelope_method_kwargs,
    )
    if run_parity:
        run_parity_check(
            raw_manifest, feature_tables, methods, all_files,
            pd.Series({file_id: group["label"].iloc[0]
                       for file_id, group in raw_manifest.groupby("file_id")}),
            feature_cols,
            band_hz=band_hz, window_size=window_size, stride=stride,
            warmup_samples=warmup_samples, target_fs_hz=target_fs_hz,
            lp_cutoff_hz=lp_cutoff_hz, bandpass_order=bandpass_order,
            lowpass_order=lowpass_order, envelope_method_kwargs=envelope_method_kwargs,
        )

    noise_tables: dict[tuple[str, float, int], pd.DataFrame] = {}
    for snr_db in snr_levels_db:
        for draw in range(n_noise_draws):
            for method in methods:
                print(f"Trích đặc trưng nhiễu: {method:20s} SNR={snr_db:>3} dB, lần rút {draw}")
                noisy_table = build_feature_table_for(
                    raw_manifest, method, make_awgn_loader(float(snr_db), draw),
                    band_hz=band_hz, window_size=window_size, stride=stride,
                    warmup_samples=warmup_samples, target_fs_hz=target_fs_hz,
                    lp_cutoff_hz=lp_cutoff_hz, bandpass_order=bandpass_order,
                    lowpass_order=lowpass_order, envelope_method_kwargs=envelope_method_kwargs,
                )
                clean_table = feature_tables[method]
                keys_noisy = noisy_table[["file_id", "window_idx"]].astype(str).to_numpy()
                keys_clean = clean_table[["file_id", "window_idx"]].astype(str).to_numpy()
                if not np.array_equal(keys_noisy, keys_clean):
                    raise RuntimeError(f"{method}: bảng nhiễu và bảng sạch không cùng tập cửa sổ.")
                noise_tables[(method, float(snr_db), draw)] = noisy_table

    noise_rows: list[dict[str, Any]] = []
    for method in methods:
        evaluation_model = models.build_primary_mlp(
            input_dim=len(feature_cols), n_classes=len(class_names), **dict(mlp_kwargs)
        )
        for seed, splits in folds.items():
            for fold_id, (_, test_files) in enumerate(splits, start=1):
                state_key = (method, int(seed), fold_id)
                if state_key not in fold_state:
                    raise KeyError(f"Thiếu model/scaler đã lưu cho {state_key}.")
                weights, scaler = fold_state[state_key]
                evaluation_model.set_weights(weights)
                test_file_set = set(test_files)
                for snr_db in snr_levels_db:
                    for draw in range(n_noise_draws):
                        table = noise_tables[(method, float(snr_db), draw)]
                        test_df = table[table["file_id"].astype(str).isin(test_file_set)].reset_index(drop=True)
                        x_test = scaler.transform(test_df[list(feature_cols)].to_numpy(dtype=np.float32))
                        probabilities = evaluation_model.predict(x_test, verbose=0)
                        for row in aggregate_file_predictions(test_df, probabilities, class_names):
                            row.update(method=method, seed=int(seed), fold=fold_id,
                                       snr_db=float(snr_db), draw=draw)
                            noise_rows.append(row)
        del evaluation_model
        tf.keras.backend.clear_session()

    noise_predictions = pd.DataFrame(noise_rows)
    summaries, deltas = [], []
    for snr_db in snr_levels_db:
        subset = noise_predictions[noise_predictions["snr_db"] == float(snr_db)]
        result = summarize_conditions(
            subset, "method", methods, primary_method, all_files, class_names,
            bootstrap_indices, rep_cols=("seed", "draw"), alpha=alpha,
        )
        summaries.append(result["summary"].assign(snr_db=float(snr_db)))
        deltas.append(paired_method_deltas(
            subset, method_comparisons, methods, all_files, class_names,
            bootstrap_indices, rep_cols=("seed", "draw"), alpha=alpha,
            margin=noninferiority_margin,
        ).assign(snr_db=float(snr_db)))

    return {
        "predictions": noise_predictions,
        "summary": pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        "delta": pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame(),
    }
