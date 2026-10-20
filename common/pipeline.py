# -*- coding: utf-8 -*-
"""
common/pipeline.py
====================
"""

from pathlib import Path
import pandas as pd

from . import config as cfg
from . import io_utils, splitting, synthetic

MANIFEST_CACHE_REQUIRED_COLUMNS = {
    "file_path",
    "label",
    "load_hp",
    "fault_diameter_mils",
    "or_position",
    "sensor_location",
    "declared_sample_rate_khz",
    "file_sample_rate_override_hz",
    "sample_rate_mat_hz",
    "n_samples_DE",
    "rpm_from_file",
    "rpm",
    "rpm_source",
    "read_error",
    "warnings",
    "has_warning",
    "resolved_sample_rate_hz",
    "resolved_sample_rate_source",
    "record_duration_sec",
    "manifest_schema_version",
}

SCOPE_EXCLUDE_WARNING_KEYWORDS = (
    "VONG_BI_NTN",
    "OR_NGOAI_PHAM_VI",
    "OR_THIEU_VI_TRI",
    "THIEU_NHAN",
    "THIEU_TAI",
    "DUONG_KINH_LA",
    "NGOAI_PHAM_VI_CAM_BIEN",
    "NGOAI_PHAM_VI_TAN_SO_KHAI_BAO",
    "THIEU_TIN_HIEU_DE",
    "LOI_DOC_FILE",
    "FS_KHONG_DANG_TIN_CAY",
    "FS_CONFLICT_METADATA",
)


def _validate_manifest_contract(manifest: pd.DataFrame, *, stage: str) -> None:
    if manifest is None or not isinstance(manifest, pd.DataFrame):
        raise AssertionError(f"{stage}: manifest phải là pandas.DataFrame.")
    if manifest.empty:
        raise AssertionError(f"{stage}: manifest rỗng - không thể chạy tiếp.")

    required = {
        "sanity": [
            "file_path",
            "label",
            "load_hp",
            "fault_diameter_mils",
            "warnings",
            "has_warning",
            "resolved_sample_rate_hz",
            "resolved_sample_rate_source",
            "rpm",
            "rpm_source",
        ],
        "scope": [
            "file_path",
            "label",
            "load_hp",
            "warnings",
            "resolved_sample_rate_hz",
            "resolved_sample_rate_source",
            "rpm",
            "rpm_source",
        ],
    }

    missing = [c for c in required[stage] if c not in manifest.columns]
    if missing:
        raise AssertionError(
            f"{stage}: manifest thiếu cột bắt buộc {missing}. "
            "Manifest phải đã qua run_sanity_checks() / apply_scope_filter() đúng chuẩn."
        )

    if stage == "scope" and "resolved_sample_rate_hz" in manifest.columns:
        missing_fs = manifest["resolved_sample_rate_hz"].isna().sum()
        if missing_fs:
            raise AssertionError(
                f"{stage}: có {missing_fs} file thiếu resolved_sample_rate_hz. "
                "Manifest đã qua scope filter nhưng vẫn còn file chưa xác định fs thực."
            )


def _validate_cache_schema(manifest: pd.DataFrame) -> None:
    missing = sorted(MANIFEST_CACHE_REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(
            "manifest cache không hợp lệ: thiếu cột "
            f"{missing}. Tái tạo manifest mới từ dữ liệu gốc."
        )
    versions = pd.to_numeric(manifest["manifest_schema_version"], errors="coerce")
    if versions.isna().any() or not versions.eq(io_utils.MANIFEST_SCHEMA_VERSION).all():
        raise ValueError(
            "manifest cache dùng schema cũ hoặc version không hợp lệ. "
            "Tái tạo manifest từ metadata và thư mục nguồn."
        )


def _normalize_manifest_file_paths(manifest: pd.DataFrame) -> pd.DataFrame:
    if "file_path" not in manifest.columns:
        return manifest
    manifest = manifest.copy()

    def normalize_path(path_value):
        if pd.isna(path_value):
            return path_value
        p = Path(str(path_value))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        return str(p)

    manifest["file_path"] = manifest["file_path"].map(normalize_path)
    return manifest


def get_manifest(use_synthetic: bool, real_data_root, synthetic_data_root,
                 output_dir, force_rebuild: bool = False) -> pd.DataFrame:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tables_dir / "manifest.csv"

    if manifest_path.exists() and not force_rebuild:
        manifest = pd.read_csv(manifest_path)
        try:
            _validate_cache_schema(manifest)
            manifest = _normalize_manifest_file_paths(manifest)
            if not manifest["file_path"].map(lambda p: isinstance(p, str) and Path(p).exists()).all():
                raise ValueError("manifest cache chứa file_path không tồn tại.")
            return manifest
        except ValueError:
            print(
                f"[CACHE] Manifest tại {manifest_path} thiếu schema cần thiết hoặc file_path không hợp lệ; "
                "tái sinh lại từ dữ liệu gốc."
            )

    if use_synthetic:
        data_root = synthetic.build_synthetic_dataset(Path(synthetic_data_root))
    else:
        data_root = Path(real_data_root)
        if not data_root.exists():
            raise FileNotFoundError(
                f"real_data_root '{data_root}' không tồn tại."
            )

    manifest = io_utils.build_manifest(data_root)
    manifest = _normalize_manifest_file_paths(manifest)
    _validate_manifest_contract(manifest, stage="sanity")
    manifest.to_csv(manifest_path, index=False)
    return manifest


def pick_file(manifest, label, load_hp, diameter_mils=None, or_position=None):
    df = manifest.copy()
    df = df.loc[(df["label"] == label) & (df["load_hp"] == load_hp)]

    if diameter_mils is not None:
        df = df.loc[df["fault_diameter_mils"] == diameter_mils]

    if label == "OR":
        if or_position is not None:
            df = df.loc[df["or_position"] == or_position]
        else:
            df = df.loc[df["or_position"] == "Centered"]

    if len(df) == 0:
        raise ValueError(
            f"Không tìm thấy file khớp label={label}, load_hp={load_hp}, "
            f"diameter_mils={diameter_mils}, or_position={or_position} trong manifest."
        )

    return str(df.iloc[0]["file_path"])


def _assert_scope_filtered(manifest: pd.DataFrame) -> None:
    _validate_manifest_contract(manifest, stage="scope")
    warning_text = manifest["warnings"].fillna("").astype(str)
    blocked = warning_text[warning_text.str.contains("|".join(SCOPE_EXCLUDE_WARNING_KEYWORDS), regex=True)]
    if not blocked.empty:
        raise AssertionError("manifest đã qua apply_scope_filter() nhưng vẫn còn file vi phạm phạm vi")


def _assert_feature_contract(feature_df: pd.DataFrame) -> None:
    from . import features_full
    required_metadata = (
        "file_id",
        "window_idx",
        "start_idx",
        "label",
        "class_label",
        "load_hp",
        "fault_diameter_mils",
        "rpm",
        "rpm_source",
        "source_sample_rate_hz",
        "source_sample_rate_source",
        "feature_sample_rate_hz",
    )
    missing_metadata = [c for c in required_metadata if c not in feature_df.columns]
    if missing_metadata:
        raise AssertionError(f"feature contract sai: bảng đặc trưng thiếu metadata bắt buộc {missing_metadata}.")

    feature_cols = features_full.feature_columns(
        feature_df,
        expected_count=features_full.EXPECTED_FEATURE_COUNTS["realized_total"],
        strict_prefix=True,
    )
    if len(feature_cols) != features_full.EXPECTED_FEATURE_COUNTS["realized_total"]:
        raise AssertionError("feature contract sai: số cột đặc trưng không khớp.")


def run_preprocessing_pipeline(
    *,
    real_data_root: Path,
    output_dir: Path,
    band_hz: tuple = cfg.RESONANCE_BAND_HZ,
    lp_cutoff_hz: float,
    target_fs_hz: float = 12000,
    window_size: int = 2048,
    stride: int = 1024,
    envelope_method: str = "hybrid",
    envelope_kwargs: dict | None = None,
    fold_seeds=range(42, 52),
    fold_n_splits: int | None = None,
    use_synthetic: bool = False,
    synthetic_data_root: Path | None = None,
    force_rebuild_manifest: bool = False,
):
    from . import io_utils, features_full

    manifest_full = get_manifest(
        use_synthetic=use_synthetic,
        real_data_root=real_data_root,
        synthetic_data_root=synthetic_data_root,
        output_dir=output_dir,
        force_rebuild=force_rebuild_manifest,
    )
    _validate_manifest_contract(manifest_full, stage="sanity")

    relevant_rows = manifest_full["label"].isin(cfg.SCOPE["labels"])
    unknown_fs = manifest_full.loc[
        relevant_rows & manifest_full["resolved_sample_rate_hz"].isna(),
        ["file_path", "label"],
    ]
    if not unknown_fs.empty:
        raise ValueError(
            "Không xác định được sampling rate đáng tin cậy cho file thuộc "
            f"phạm vi: {unknown_fs.to_dict(orient='records')[:10]}"
        )

    manifest = io_utils.apply_scope_filter(manifest_full)
    _assert_scope_filtered(manifest)

    manifest, manifest_hash = splitting.save_manifest_with_folds(
        manifest,
        Path(output_dir) / "tables" / "manifest_with_folds.csv",
        seeds=fold_seeds,
        n_splits=fold_n_splits,
    )
    print(f"[MANIFEST] SHA-256 {manifest_hash}")

    required_cols = ("resolved_sample_rate_hz", "label", "load_hp", "file_path")
    missing = [c for c in required_cols if c not in manifest.columns]
    if missing:
        raise AssertionError(f"Manifest sau apply_scope_filter() vẫn thiếu cột bắt buộc {missing}.")

    features_df = features_full.build_full_feature_table(
        manifest,
        band_hz=band_hz,
        load_de_signal_fn=io_utils.load_de_signal_resampled,
        window_size=window_size,
        stride=stride,
        target_fs_hz=target_fs_hz,
        lp_cutoff_hz=lp_cutoff_hz,
        envelope_method=envelope_method,
        envelope_kwargs=envelope_kwargs,
    )
    _assert_feature_contract(features_df)

    return manifest, features_df