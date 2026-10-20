# -*- coding: utf-8 -*-
"""
common/io_utils.py
===================
Đọc file .mat CWRU, dựng bảng manifest, chạy sanity check.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import config as cfg


KNOWN_SOURCE_CATEGORIES = {
    "12k_Drive_End_Bearing_Fault_Data",
    "24k_Drive_End_Bearing_Fault_Data",
    "48k_Drive_End_Bearing_Fault_Data",
    "12k_Fan_End_Bearing_Fault_Data",
    "24k_Fan_End_Bearing_Fault_Data",
    "48k_Fan_End_Bearing_Fault_Data",
}
# File-level rate evidence supplied/verified during manifest review.
# This is deliberately a separate catalogue, not a claim that the rate is encoded
# in the filename or directory. Keep entries only when there is reliable evidence.
FILE_SAMPLE_RATE_OVERRIDES_HZ = {
    # 97_Normal_0.mat: 243938 mẫu. Hình dạng phổ (trục chu kỳ/mẫu) khớp với
    # 98/99/100 (48 kHz): tương quan log-PSD 0.93 ở hệ số trục x1, chỉ 0.59 (x0.5)
    # và 0.37 (x2). Nghĩa là file này ~5.08 s @ 48 kHz, KHÔNG phải 10.16 s @ 24 kHz.
    # Cảnh báo DURATION_UNEXPECTED_AT_RESOLVED_FS là chủ ý (không bị scope filter loại).
    "97_Normal_0.mat": 48000.0,
}
MANIFEST_SCHEMA_VERSION = 3

def select_de_key(mat, filepath) -> str:
    """Chọn biến '*_DE_time' thuộc ĐÚNG file này.

    Một số file CWRU chứa nhiều biến DE (vd. 99_Normal_2.mat có cả X098_DE_time
    và X099_DE_time). Lấy biến đầu tiên/cuối cùng là sai: so khớp số hiệu trong
    tên biến với số đứng đầu tên file ('99_Normal_2' -> X099_DE_time).
    """
    keys = [k for k in mat if not str(k).startswith("__") and str(k).endswith("_DE_time")]
    if not keys:
        raise KeyError(f"Không tìm thấy biến '_DE_time' trong {filepath}")
    if len(keys) == 1:
        return keys[0]
    match = re.match(r"(\d+)", Path(filepath).stem)
    if match:
        number = int(match.group(1))
        exact = [k for k in keys if re.fullmatch(rf"X0*{number}_DE_time", str(k))]
        if len(exact) == 1:
            return exact[0]
    raise KeyError(
        f"{filepath} có nhiều biến DE {keys} và không xác định được biến thuộc file này."
    )


def resolve_rpm_metadata(rpm_from_file=None, load_hp=None, *, rpm_nominal_map=None) -> dict:
    """Tạo meta RPM theo README: measured nếu có sẵn từ file, hoặc imputed:<quy tắc> nếu dùng giá trị định danh theo tải."""
    rpm_nominal_map = cfg.NOMINAL_RPM_BY_LOAD if rpm_nominal_map is None else rpm_nominal_map
    rpm_nominal = None if load_hp is None else rpm_nominal_map.get(load_hp)

    if rpm_from_file is not None and not pd.isna(rpm_from_file):
        rpm = float(rpm_from_file)
        if rpm > 0 and (rpm_nominal is None or abs(rpm - rpm_nominal) <= 20.0):
            return {"rpm": rpm, "rpm_source": "measured"}

    if rpm_nominal is not None:
        return {"rpm": float(rpm_nominal), "rpm_source": "imputed:nominal_by_load"}

    return {"rpm": float("nan"), "rpm_source": "imputed:manual"}


def parse_metadata_from_filename(filepath: Path) -> dict:
    """Parse label/load/source folder metadata without guessing from sibling folders.

    A Normal file placed at ``data/raw/Normal`` does NOT inherit a 48 kHz rate
    merely because a 48k source folder exists elsewhere. Source rate is resolved
    later from explicit source-folder evidence, embedded MAT metadata, a reviewed
    file-level override, or a clearly marked duration-based inference.
    """
    filepath = Path(filepath)
    parts = filepath.parts
    name = filepath.name
    stem = filepath.stem

    label = None
    diameter_mils = None
    or_position = None
    load_hp = None
    source_category = None
    sensor_location = None
    declared_sample_rate_khz = None
    file_sample_rate_override_hz = None

    # Source category is considered explicit only when it is an actual path component.
    for part in parts:
        if part in KNOWN_SOURCE_CATEGORIES:
            source_category = part
            break

    if source_category is not None:
        if "Fan_End" in source_category:
            sensor_location = "FE"
        elif "Drive_End" in source_category:
            sensor_location = "DE"
        if source_category.startswith("48k"):
            declared_sample_rate_khz = 48
        elif source_category.startswith("24k"):
            declared_sample_rate_khz = 24
        elif source_category.startswith("12k"):
            declared_sample_rate_khz = 12

    label_candidates = {"B", "IR", "OR", "Normal"}
    for part in parts:
        if part in label_candidates:
            label = part
            break
    # Many curated CWRU layouts put the label in the filename instead of a directory.
    if label is None:
        match_label = re.search(r"(?:^|_)(Normal|IR|OR|B)(?:_|$)", stem, flags=re.IGNORECASE)
        if match_label:
            label = match_label.group(1).upper() if match_label.group(1).upper() != "NORMAL" else "Normal"

    # Apply only exact file-level overrides from the reviewed inventory.
    # Do not turn this override into a fictitious source_category/declared rate.
    override = FILE_SAMPLE_RATE_OVERRIDES_HZ.get(name)
    if override is not None:
        file_sample_rate_override_hz = float(override)

    if label == "Normal" and sensor_location is None:
        sensor_location = "DE"

    if label in ("B", "IR", "OR") and label in parts:
        idx = parts.index(label)
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            diameter_mils = int(parts[idx + 1])
    elif label in ("B", "IR", "OR"):
        match_diam = re.search(r"(?:^|_)(?:B|IR|OR)_(\d{3})(?:_|$)", stem)
        if match_diam:
            diameter_mils = int(match_diam.group(1))

    if label == "OR":
        or_position_map = {"@3": "Orthogonal", "@6": "Centered", "@12": "Opposite"}
        for part in parts:
            if part in or_position_map:
                or_position = or_position_map[part]
                break
        if or_position is None:
            embedded_match = re.search(r"@(3|6|12)(?=_|$)", stem)
            if embedded_match:
                or_position = or_position_map[f"@{embedded_match.group(1)}"]

    load_match = re.search(r"_(\d+)$", stem)
    if load_match:
        load_hp = int(load_match.group(1))

    return {
        "load_hp": load_hp,
        "label": label,
        "fault_diameter_mils": diameter_mils,
        "or_position": or_position,
        "source_category": source_category,
        "sensor_location": sensor_location,
        "declared_sample_rate_khz": declared_sample_rate_khz,
        "file_sample_rate_override_hz": file_sample_rate_override_hz,
    }


def inspect_mat_file(filepath: Path) -> dict:
    result: dict[str, object] = {
        "n_samples_DE": None, "n_samples_FE": None, "n_samples_BA": None,
        "rpm_from_file": None, "sample_rate_mat_hz": None, "read_error": None,
    }
    try:
        mat = dict(loadmat(str(filepath)))
    except Exception as exc:
        result["read_error"] = str(exc)
        return result

    try:
        de_key = select_de_key(mat, filepath)
    except KeyError:
        de_key = None

    fs_candidates = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if key == de_key:
            result["n_samples_DE"] = int(arr.size)
        elif key.endswith("_FE_time"):
            result["n_samples_FE"] = int(arr.size)
        elif key.endswith("_BA_time"):
            result["n_samples_BA"] = int(arr.size)
        elif key.endswith("RPM"):
            rpm_arr = arr.ravel()
            if rpm_arr.size > 0:
                result["rpm_from_file"] = float(rpm_arr[0])

        # Synthetic/test files can carry explicit acquisition metadata in the MAT.
        # Common names include sampling_rate_hz, sample_rate_hz, X999_fs_Hz.
        key_l = str(key).lower()
        if ("fs_hz" in key_l or "sampling_rate_hz" in key_l or "sample_rate_hz" in key_l):
            vals = arr.ravel()
            if vals.size == 1:
                try:
                    rate = float(vals[0])
                    if np.isfinite(rate) and rate > 0:
                        fs_candidates.append(rate)
                except (TypeError, ValueError):
                    pass

    if fs_candidates:
        unique_rates = sorted(set(float(v) for v in fs_candidates))
        result["sample_rate_mat_hz"] = unique_rates[0] if len(unique_rates) == 1 else np.nan
        if len(unique_rates) > 1:
            result["read_error"] = (
                "MAT file chứa nhiều trường sampling-rate không nhất quán: "
                f"{unique_rates}"
            )
    return result



def load_de_signal(filepath: Path):
    filepath_str = str(filepath)
    if filepath_str.endswith('.npy'):
        return np.load(filepath_str)

    try:
        mat = dict(loadmat(filepath_str))
    except Exception as e:
        raise IOError(f"Không thể đọc file {filepath_str}: {e}")

    return np.asarray(mat[select_de_key(mat, filepath)]).ravel()


def load_de_signal_resampled(filepath: Path, source_fs_hz, target_fs_hz: float):
    from fractions import Fraction
    from scipy.signal import resample_poly

    x = load_de_signal(filepath)

    if source_fs_hz is None or not np.isfinite(float(source_fs_hz)):
        raise ValueError(
            f"Không xác định được sampling rate thực đáng tin cậy cho {filepath}."
        )

    source_fs_hz = float(source_fs_hz)
    target_fs_hz = float(target_fs_hz)
    if source_fs_hz <= 0.0 or target_fs_hz <= 0.0:
        raise ValueError("Sampling rate nguồn và đích phải lớn hơn 0.")
    if x.ndim != 1 or not len(x):
        raise ValueError(
            f"Tín hiệu DE phải là mảng 1D không rỗng: {filepath}."
        )

    if np.isclose(source_fs_hz, target_fs_hz, rtol=0.0, atol=1e-9):
        return np.asarray(x)

    frac = Fraction(int(round(target_fs_hz)), int(round(source_fs_hz))).limit_denominator(1000)
    result = resample_poly(x, frac.numerator, frac.denominator)
    expected_length = int(np.ceil(len(x) * target_fs_hz / source_fs_hz))
    if len(result) != expected_length:
        raise RuntimeError(
            f"Resampling length mismatch for {filepath}: got {len(result)}, "
            f"expected {expected_length}."
        )
    return result


def _as_finite_rate(value):
    if value is None or pd.isna(value):
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    return rate if np.isfinite(rate) and rate > 0 else None


def _resolve_sampling_rate_hz(
    n_samples,
    candidates,
    expected_duration,
    tolerance,
    *,
    declared_rate_hz=None,
    override_rate_hz=None,
    mat_rate_hz=None,
):
    """Resolve source fs while preserving evidence/provenance.

    Priority is not used to hide contradictions: if multiple explicit metadata
    sources disagree, the rate remains unresolved and the caller emits a warning.
    When no explicit metadata exists, inference is allowed only if exactly one
    candidate rate implies a recording duration inside the configured window.
    """
    if n_samples is None or pd.isna(n_samples) or float(n_samples) <= 0:
        return None, "unresolved:no_sample_count", "FS_KHONG_DANG_TIN_CAY"

    explicit = []
    for source, value in (
        ("source_category", declared_rate_hz),
        ("file_catalog:user_reviewed", override_rate_hz),
        ("embedded_mat_metadata", mat_rate_hz),
    ):
        rate = _as_finite_rate(value)
        if rate is not None:
            explicit.append((source, rate))

    valid_candidates = {float(v) for v in candidates}
    invalid = [(source, rate) for source, rate in explicit if rate not in valid_candidates]
    if invalid:
        return None, "unresolved:rate_outside_candidate_set", "FS_KHONG_DANG_TIN_CAY"

    distinct_rates = sorted({rate for _, rate in explicit})
    if len(distinct_rates) > 1:
        details = ", ".join(f"{src}={rate:g}" for src, rate in explicit)
        return None, f"unresolved:conflicting_metadata[{details}]", "FS_CONFLICT_METADATA"

    n = int(float(n_samples))
    if explicit:
        rate = distinct_rates[0]
        # When metadata sources agree, retain the most direct provenance.
        source_order = {
            "embedded_mat_metadata": 0,
            "file_catalog:user_reviewed": 1,
            "source_category": 2,
        }
        source = sorted(
            (src for src, value in explicit if value == rate),
            key=lambda src: source_order.get(src, 99),
        )[0]
        duration = n / rate
        warning = None
        if abs(duration - float(expected_duration)) > float(tolerance):
            warning = "DURATION_UNEXPECTED_AT_RESOLVED_FS"
        return rate, source, warning

    low = float(expected_duration) - float(tolerance)
    high = float(expected_duration) + float(tolerance)
    plausible = []
    for candidate in candidates:
        rate = float(candidate)
        duration = n / rate
        if low <= duration <= high:
            plausible.append((rate, duration))

    if len(plausible) == 1:
        rate, _ = plausible[0]
        return rate, "inferred:sample_count_and_expected_duration", "FS_INFERRED_FROM_DURATION"
    if len(plausible) > 1:
        rates = [int(rate) for rate, _ in plausible]
        return None, f"unresolved:ambiguous_duration_candidates{rates}", "FS_KHONG_DANG_TIN_CAY"
    return None, "unresolved:duration_does_not_identify_rate", "FS_KHONG_DANG_TIN_CAY"


def run_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    warnings_list: list[list[str]] = [[] for _ in range(len(df))]
    resolved_fs_list: list = [None] * len(df)
    resolved_fs_sources: list[str] = ["unresolved"] * len(df)
    record_duration_list: list = [np.nan] * len(df)
    rpm_values: list = [np.nan] * len(df)
    rpm_sources: list[str] = ["imputed:manual"] * len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        read_error = row.get("read_error")
        if pd.notna(read_error):
            warnings_list[i].append(f"LOI_DOC_FILE: {read_error}")

        n = row.get("n_samples_DE")
        if n is None or pd.isna(n):
            warnings_list[i].append("THIEU_TIN_HIEU_DE")
        declared_khz = row.get("declared_sample_rate_khz")
        declared_hz = None if declared_khz is None or pd.isna(declared_khz) else float(declared_khz) * 1000.0
        resolved_fs, fs_source, fs_note = _resolve_sampling_rate_hz(
            n,
            cfg.CANDIDATE_SAMPLING_RATES_HZ,
            cfg.EXPECTED_DURATION_SEC,
            cfg.DURATION_TOLERANCE_SEC,
            declared_rate_hz=declared_hz,
            override_rate_hz=row.get("file_sample_rate_override_hz"),
            mat_rate_hz=row.get("sample_rate_mat_hz"),
        )
        resolved_fs_list[i] = resolved_fs
        resolved_fs_sources[i] = fs_source
        if resolved_fs is not None and n is not None and not pd.isna(n):
            record_duration_list[i] = float(n) / float(resolved_fs)
        if fs_note is not None:
            warnings_list[i].append(fs_note)

        load_hp = row.get("load_hp")
        rpm_meta = resolve_rpm_metadata(row.get("rpm_from_file"), load_hp)
        rpm_values[i] = rpm_meta["rpm"]
        rpm_sources[i] = rpm_meta["rpm_source"]

        rpm_file = row.get("rpm_from_file")
        if load_hp in cfg.NOMINAL_RPM_BY_LOAD and rpm_file is not None and not pd.isna(rpm_file):
            rpm_nominal = cfg.NOMINAL_RPM_BY_LOAD[load_hp]
            if abs(float(rpm_file) - rpm_nominal) > 20:
                warnings_list[i].append("RPM_LECH")

        diam = row.get("fault_diameter_mils")
        if diam is not None and not pd.isna(diam):
            diam = int(diam)
            if diam in cfg.NTN_FAULT_DIAMETERS_MILS:
                warnings_list[i].append("VONG_BI_NTN")
            elif diam not in cfg.SKF_VALID_FAULT_DIAMETERS_MILS:
                warnings_list[i].append("DUONG_KINH_LA")

        label = row.get("label")
        or_pos = row.get("or_position")
        if label == "OR":
            if or_pos is None or (isinstance(or_pos, float) and pd.isna(or_pos)):
                warnings_list[i].append("OR_THIEU_VI_TRI")
            elif cfg.SCOPE["outer_race_position"].lower() not in str(or_pos).lower():
                warnings_list[i].append("OR_NGOAI_PHAM_VI")

        if label is None or (isinstance(label, float) and pd.isna(label)):
            warnings_list[i].append("THIEU_NHAN")
        if load_hp is None or (isinstance(load_hp, float) and pd.isna(load_hp)):
            warnings_list[i].append("THIEU_TAI")

        sensor_location = row.get("sensor_location")
        target_sensor = cfg.SCOPE["sensor_location"]
        if pd.notna(sensor_location) and sensor_location != target_sensor:
            warnings_list[i].append("NGOAI_PHAM_VI_CAM_BIEN")

        # Scope is based on source evidence by file, not on a sibling-folder heuristic.
        # Normal DE signals can have mixed source rates (12/24/48 kHz) because they
        # will be explicitly resampled to the common target rate. Fault files in this
        # study are scoped to 12 kHz source signals.
        resolved_fs = resolved_fs_list[i]
        if label == "Normal":
            if sensor_location != "DE":
                pass
        elif resolved_fs is not None and not np.isclose(
            float(resolved_fs), float(cfg.SCOPE["sampling_rate_hz"]), rtol=0, atol=1e-6
        ):
            warnings_list[i].append("NGOAI_PHAM_VI_TAN_SO_KHAI_BAO")

    df["warnings"] = ["; ".join(w) if w else "" for w in warnings_list]
    df["has_warning"] = df["warnings"] != ""
    df["resolved_sample_rate_hz"] = resolved_fs_list
    df["resolved_sample_rate_source"] = resolved_fs_sources
    df["record_duration_sec"] = record_duration_list
    df["rpm"] = rpm_values
    df["rpm_source"] = rpm_sources
    return df


def apply_scope_filter(manifest: pd.DataFrame,
                        exclude_warning_keywords: list[str] | None = None) -> pd.DataFrame:
    if exclude_warning_keywords is None:
        exclude_warning_keywords = [
            "VONG_BI_NTN", "OR_NGOAI_PHAM_VI", "OR_THIEU_VI_TRI",
            "THIEU_NHAN", "THIEU_TAI", "DUONG_KINH_LA",
            "NGOAI_PHAM_VI_CAM_BIEN", "NGOAI_PHAM_VI_TAN_SO_KHAI_BAO",
            "THIEU_TIN_HIEU_DE", "LOI_DOC_FILE",
            "FS_KHONG_DANG_TIN_CAY", "FS_CONFLICT_METADATA",
        ]

    df = manifest.copy()
    warnings_text = df["warnings"].fillna("")
    exclude_mask = pd.Series(False, index=df.index)

    for kw in exclude_warning_keywords:
        kw_mask = warnings_text.str.contains(kw, regex=False)
        exclude_mask = exclude_mask | kw_mask

    kept = df.loc[~exclude_mask].reset_index(drop=True)
    return kept


def build_manifest(data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root).resolve()
    mat_files = sorted(data_root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"Không tìm thấy file .mat nào trong {data_root}.")

    rows = []
    for fp in mat_files:
        resolved_fp = fp.resolve()
        meta = parse_metadata_from_filename(resolved_fp)
        content = inspect_mat_file(resolved_fp)
        rows.append({"file_path": str(resolved_fp), **meta, **content})

    manifest = run_sanity_checks(pd.DataFrame(rows))
    manifest["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    return manifest