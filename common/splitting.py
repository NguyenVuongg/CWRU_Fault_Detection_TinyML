# -*- coding: utf-8 -*-
"""
common/splitting.py
====================
File-level split utilities aligned with the README protocol:
- file_id is the grouping key;
- each file belongs to exactly one fold in a seed;
- train/val/test are created from the same fixed file groups without leaking windows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from .config import CLASS_SCHEME_DEFAULT, LABEL_COL_BY_SCHEME, make_class_label

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover - optional dependency
    StratifiedGroupKFold = None


def resolve_label_col(feature_df: pd.DataFrame, label_col=None,
                      scheme: str = CLASS_SCHEME_DEFAULT) -> str:
    """Chọn cột nhãn theo sơ đồ lớp đã chốt ở mục 1.1."""
    if label_col is not None:
        if label_col not in feature_df.columns:
            raise ValueError(
                f"Không có cột nhãn '{label_col}' trong bảng đặc trưng. "
                f"Các cột đang có: {list(feature_df.columns)[:12]}..."
            )
        return label_col

    preferred = LABEL_COL_BY_SCHEME[scheme]
    if preferred in feature_df.columns:
        return preferred

    fallback = LABEL_COL_BY_SCHEME["4class"]
    if fallback in feature_df.columns:
        print(
            f"[resolve_label_col] CẢNH BÁO: bảng đặc trưng không có cột "
            f"'{preferred}' nên tạm dùng '{fallback}' (4 lớp). Đề cương mục "
            f"1.1 chốt bài toán chính là 10 lớp — hãy build lại bảng bằng "
            f"features_full.build_full_feature_table() phiên bản mới."
        )
        return fallback

    raise ValueError(
        f"Bảng đặc trưng không có cả '{preferred}' lẫn '{fallback}'."
    )


def _group_file_labels(df: pd.DataFrame, label_col: str, group_col: str = "file_id") -> pd.DataFrame:
    if group_col not in df.columns:
        raise ValueError("DataFrame cần có cột 'file_id' để phân chia dữ liệu theo file.")
    if label_col not in df.columns:
        raise ValueError(f"DataFrame cần có cột nhãn '{label_col}' cho stratified split.")

    file_meta = df[[group_col, label_col]].drop_duplicates()
    inconsistent = file_meta[group_col].duplicated(keep=False)
    if inconsistent.any():
        bad_groups = sorted(file_meta.loc[inconsistent, group_col].astype(str).unique())
        raise ValueError(
            f"Mỗi {group_col} phải có đúng một nhãn; không nhất quán: {bad_groups[:10]}."
        )
    return file_meta.reset_index(drop=True)

def _manifest_class_label(label: Any, fault_diameter: Any) -> str:
    if pd.isna(label):
        raise ValueError("Manifest có nhãn thiếu, không thể tạo class_label.")
    label_text = str(label)
    if label_text == "Normal":
        return make_class_label(label_text)
    if pd.isna(fault_diameter):
        return make_class_label(label_text)
    if isinstance(fault_diameter, (int, float, np.integer, np.floating)):
        return make_class_label(label_text, float(fault_diameter))
    if isinstance(fault_diameter, str):
        return make_class_label(label_text, fault_diameter)
    raise TypeError(
        f"fault_diameter_mils có kiểu không được hỗ trợ: {type(fault_diameter).__name__}."
    )

def stratified_group_kfold_splits(
    df: pd.DataFrame,
    *,
    n_splits: int | None = None,
    seed: int = 42,
    label_col: str | None = None,
    group_col: str = "file_id",
):
    """Trả về list các fold file-level, mỗi fold chứa các file thuộc cùng một phân hoạch k-fold.
    Dùng đúng chuẩn README: nhóm = file_id, mỗi file xuất hiện đúng một lần trong mỗi seed.
    """
    label_col = resolve_label_col(df, label_col)
    file_meta = _group_file_labels(df, label_col, group_col=group_col)

    if len(file_meta) < 2:
        raise ValueError("Cần ít nhất 2 file để k-fold group split.")

    if n_splits is None:
        class_counts = file_meta[label_col].value_counts()
        n_splits = int(class_counts.min())
    n_splits = int(min(max(2, n_splits), len(file_meta)))

    class_counts = file_meta[label_col].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Stratified group k-fold cần mỗi lớp có ít nhất 2 file.")
    n_splits = int(min(n_splits, class_counts.min()))

    if StratifiedGroupKFold is None:
        raise ImportError("scikit-learn StratifiedGroupKFold là bắt buộc cho giao thức đánh giá.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = file_meta[group_col].to_numpy()
    labels = file_meta[label_col].to_numpy()
    folds = []
    assigned = []
    for _, test_index in cv.split(file_meta, labels, groups):
        test_files = set(file_meta.iloc[test_index][group_col].tolist())
        assigned.extend(test_files)
        folds.append(df[df[group_col].isin(test_files)].reset_index(drop=True))
    if len(assigned) != len(set(assigned)) or set(assigned) != set(file_meta[group_col]):
        raise RuntimeError("Group k-fold phải gán mỗi file đúng một lần làm fold kiểm tra.")
    return folds


def add_stratified_group_fold_assignments(
    manifest: pd.DataFrame,
    *,
    seeds: Iterable[int] = range(42, 52),
    n_splits: int | None = None,
    label_col: str | None = None,
    group_col: str = "file_id",
) -> pd.DataFrame:
    """Thêm cột fold_seed_<seed>; mọi dòng/cửa sổ của một file nhận cùng fold."""
    result = manifest.copy()
    if group_col not in result.columns:
        if "file_path" not in result.columns:
            raise ValueError(f"Manifest cần '{group_col}' hoặc 'file_path'.")
        result[group_col] = result["file_path"].map(lambda value: Path(str(value)).stem)
    if label_col is None:
        if "class_label" in result.columns:
            label_col = "class_label"
        elif {"label", "fault_diameter_mils"}.issubset(result.columns):
            result["class_label"] = [
                _manifest_class_label(row.label, row.fault_diameter_mils)
                for row in result[["label", "fault_diameter_mils"]].itertuples(index=False)
            ]
            label_col = "class_label"
        else:
            raise ValueError("Manifest cần class_label hoặc label + fault_diameter_mils.")

    file_meta = _group_file_labels(result, label_col, group_col)
    class_counts = file_meta[label_col].value_counts()
    if class_counts.empty or class_counts.min() < 2:
        raise ValueError(
            "Không thể tạo stratified group k-fold: mỗi lớp cần ít nhất 2 file."
        )
    effective_splits = int(class_counts.min() if n_splits is None else n_splits)
    if effective_splits < 2 or effective_splits > int(class_counts.min()):
        raise ValueError(
            f"n_splits phải thuộc [2, {int(class_counts.min())}], nhận {effective_splits}."
        )
    if StratifiedGroupKFold is None:
        raise ImportError("scikit-learn StratifiedGroupKFold là bắt buộc cho giao thức đánh giá.")

    for seed in seeds:
        fold_col = f"fold_seed_{int(seed)}"
        if fold_col in result.columns:
            raise ValueError(f"Cột {fold_col} đã tồn tại; không ghi đè assignment.")
        cv = StratifiedGroupKFold(
            n_splits=effective_splits, shuffle=True, random_state=int(seed)
        )
        assignments: dict[str, int] = {}
        groups = file_meta[group_col].to_numpy()
        labels = file_meta[label_col].to_numpy()
        for fold_idx, (_, test_idx) in enumerate(cv.split(file_meta, labels, groups)):
            for file_id in file_meta.iloc[test_idx][group_col]:
                assignments[str(file_id)] = fold_idx
        if len(assignments) != len(file_meta):
            raise RuntimeError(f"Seed {seed}: một số file không có fold assignment.")
        result[fold_col] = result[group_col].map(
            lambda file_id: assignments[str(file_id)]
        ).astype("int64")

    return result


def save_manifest_with_folds(
    manifest: pd.DataFrame,
    output_path,
    *,
    seeds: Iterable[int] = range(42, 52),
    n_splits: int | None = None,
    label_col: str | None = None,
    group_col: str = "file_id",
) -> tuple[pd.DataFrame, str]:
    """Lưu manifest kèm fold assignments và sidecar SHA-256 của chính nội dung CSV."""
    result = add_stratified_group_fold_assignments(
        manifest, seeds=seeds, n_splits=n_splits,
        label_col=label_col, group_col=group_col,
    )
    csv_bytes = result.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(csv_bytes).hexdigest()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(csv_bytes)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="ascii"
    )
    return result, digest


def file_based_split_single_seed(df: pd.DataFrame, val_ratio: float = 0.15,
                                 test_ratio: float = 0.15, random_state: int = 42,
                                 stratify_col=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cắt train/val/test theo file_id cho một seed duy nhất; mỗi file nằm trong một tập duy nhất."""
    stratify_col = resolve_label_col(df, stratify_col)
    file_label = _group_file_labels(df, stratify_col).set_index("file_id")[stratify_col]
    rng = np.random.RandomState(random_state)

    train_files, val_files, test_files = [], [], []
    for _, group in file_label.groupby(file_label):
        ids = group.index.to_numpy().copy()
        rng.shuffle(ids)
        n_total = len(ids)

        if n_total <= 1:
            train_files.extend(ids)
            continue

        n_test = max(1, int(round(n_total * test_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        if n_total - n_test - n_val < 1:
            n_val = 0
            n_test = max(1, n_total - 1)

        test_files.extend(ids[:n_test])
        val_files.extend(ids[n_test:n_test + n_val])
        train_files.extend(ids[n_test + n_val:])

    train_df = df[df["file_id"].isin(train_files)].reset_index(drop=True)
    val_df = df[df["file_id"].isin(val_files)].reset_index(drop=True)
    test_df = df[df["file_id"].isin(test_files)].reset_index(drop=True)
    return train_df, val_df, test_df


def generate_n_seed_splits(df: pd.DataFrame, n_seeds: int = 5,
                           val_ratio: float = 0.15, test_ratio: float = 0.15,
                           start_seed: int = 42, stratify_col=None):
    """Generator N seed, mỗi lần sinh train/val/test file-level.
    Mục tiêu là thống nhất khung so sánh giữa các phương án mà không rò rỉ cửa sổ.
    """
    if "file_id" not in df.columns:
        raise ValueError("DataFrame cần có cột 'file_id' để phân chia dữ liệu.")

    for i in range(n_seeds):
        seed = start_seed + i
        train_df, val_df, test_df = file_based_split_single_seed(
            df,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=seed,
            stratify_col=stratify_col,
        )
        yield seed, train_df, val_df, test_df
