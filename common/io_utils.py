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
    "48k_Drive_End_Bearing_Fault_Data",
    "12k_Fan_End_Bearing_Fault_Data",
}

def parse_metadata_from_filename(filepath: Path) -> dict:
    parts = filepath.parts
    name = filepath.stem

    label = None
    diameter_mils = None
    or_position = None
    load_hp = None
    source_category = None
    sensor_location = None
    declared_sample_rate_khz = None

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
        elif source_category.startswith("12k"):
            declared_sample_rate_khz = 12

    label_candidates = {"B", "IR", "OR", "Normal"}
    for part in parts:
        if part in label_candidates:
            label = part
            break

    if label == "Normal":
        sensor_location = "DE"

    if label in ("B", "IR", "OR") and label in parts:
        idx = parts.index(label)
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            diameter_mils = int(parts[idx + 1])

    if label == "OR":
        or_position_map = {"@3": "Orthogonal", "@6": "Centered", "@12": "Opposite"}
        for part in parts:
            if part in or_position_map:
                or_position = or_position_map[part]
                break
        if or_position is None:
            embedded_match = re.search(r"@(3|6|12)(?=_|$)", name)
            if embedded_match:
                or_position = or_position_map[f"@{embedded_match.group(1)}"]

    load_match = re.search(r"_(\d+)$", name)
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
    }


def inspect_mat_file(filepath: Path) -> dict:
    result: dict[str, object] = {
        "n_samples_DE": None, "n_samples_FE": None, "n_samples_BA": None,
        "rpm_from_file": None, "read_error": None,
    }
    try:
        mat = dict(loadmat(str(filepath)))
    except Exception as exc:
        result["read_error"] = str(exc)
        return result

    for key in mat.keys():
        if key.startswith("__"):
            continue
        if key.endswith("_DE_time"):
            result["n_samples_DE"] = int(np.asarray(mat[key]).size)
        elif key.endswith("_FE_time"):
            result["n_samples_FE"] = int(np.asarray(mat[key]).size)
        elif key.endswith("_BA_time"):
            result["n_samples_BA"] = int(np.asarray(mat[key]).size)
        elif key.endswith("RPM"):
            rpm_arr = np.asarray(mat[key]).ravel()
            if rpm_arr.size > 0:
                result["rpm_from_file"] = float(rpm_arr[0])
    return result


def load_de_signal(filepath: Path):
    filepath_str = str(filepath)
    if filepath_str.endswith('.npy'):
        return np.load(filepath_str)

    try:
        mat = dict(loadmat(filepath_str))
    except Exception as e:
        raise IOError(f"Không thể đọc file {filepath_str}: {e}")

    for key in mat.keys():
        if key.endswith("_DE_time"):
            return np.asarray(mat[key]).ravel()

    raise KeyError(f"Không tìm thấy biến '_DE_time' trong {filepath_str}")


def load_de_signal_resampled(filepath: Path, source_fs_hz, target_fs_hz: float):
    from fractions import Fraction
    from scipy.signal import resample_poly

    x = load_de_signal(filepath)

    if source_fs_hz is None or (isinstance(source_fs_hz, float) and np.isnan(source_fs_hz)):
        raise ValueError("Không xác định được sampling rate thực.")

    source_fs_hz = float(source_fs_hz)
    target_fs_hz = float(target_fs_hz)

    if round(source_fs_hz) == round(target_fs_hz):
        return x

    frac = Fraction(int(round(target_fs_hz)), int(round(source_fs_hz))).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator)


def _resolve_sampling_rate_hz(n_samples, target_rate_hz, candidates,
                               expected_duration, tolerance):
    if n_samples is None or pd.isna(n_samples):
        return None

    durations = {rate: n_samples / rate for rate in candidates}
    plausible = [rate for rate, dur in durations.items()
                 if abs(dur - expected_duration) <= tolerance]

    if not plausible:
        return None
    if target_rate_hz in plausible:
        return float(target_rate_hz)
    return float(min(plausible, key=lambda r: abs(r - target_rate_hz)))


def run_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    warnings_list: list[list[str]] = [[] for _ in range(len(df))]
    resolved_fs_list: list = [None] * len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        read_error = row.get("read_error")
        if pd.notna(read_error):
            warnings_list[i].append(f"LOI_DOC_FILE: {read_error}")

        n = row.get("n_samples_DE")
        if n is None or pd.isna(n):
            warnings_list[i].append("THIEU_TIN_HIEU_DE")
        else:
            declared_khz = row.get("declared_sample_rate_khz")
            target_rate_hz = (
                declared_khz * 1000 if pd.notna(declared_khz)
                else cfg.SCOPE["sampling_rate_hz"]
            )

            durations = {rate: n / rate for rate in cfg.CANDIDATE_SAMPLING_RATES_HZ}
            plausible = {rate: dur for rate, dur in durations.items()
                         if abs(dur - cfg.EXPECTED_DURATION_SEC) <= cfg.DURATION_TOLERANCE_SEC}

            resolved_fs_list[i] = _resolve_sampling_rate_hz(
                n, target_rate_hz, cfg.CANDIDATE_SAMPLING_RATES_HZ,
                cfg.EXPECTED_DURATION_SEC, cfg.DURATION_TOLERANCE_SEC,
            )

            if target_rate_hz not in plausible and plausible:
                warnings_list[i].append("NGHI_NGO_SAMPLING_RATE")
            elif not plausible:
                warnings_list[i].append("THOI_LUONG_BAT_THUONG")

        load_hp = row.get("load_hp")
        rpm_file = row.get("rpm_from_file")
        if load_hp in cfg.NOMINAL_RPM_BY_LOAD and rpm_file is not None and not pd.isna(rpm_file):
            rpm_nominal = cfg.NOMINAL_RPM_BY_LOAD[load_hp]
            if abs(rpm_file - rpm_nominal) > 20:
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

        if label is None:
            warnings_list[i].append("THIEU_NHAN")
        if load_hp is None:
            warnings_list[i].append("THIEU_TAI")

        sensor_location = row.get("sensor_location")
        target_sensor = cfg.SCOPE["sensor_location"]
        if pd.notna(sensor_location) and sensor_location != target_sensor:
            warnings_list[i].append("NGOAI_PHAM_VI_CAM_BIEN")

        declared_rate = row.get("declared_sample_rate_khz")
        target_rate = cfg.SCOPE["sampling_rate_hz"] / 1000
        if pd.notna(declared_rate) and declared_rate != target_rate:
            warnings_list[i].append("NGOAI_PHAM_VI_TAN_SO_KHAI_BAO")

    df["warnings"] = ["; ".join(w) if w else "" for w in warnings_list]
    df["has_warning"] = df["warnings"] != ""
    df["resolved_sample_rate_hz"] = resolved_fs_list
    return df


def apply_scope_filter(manifest: pd.DataFrame,
                        exclude_warning_keywords: list[str] | None = None) -> pd.DataFrame:
    if exclude_warning_keywords is None:
        exclude_warning_keywords = [
            "VONG_BI_NTN", "OR_NGOAI_PHAM_VI", "OR_THIEU_VI_TRI",
            "THIEU_NHAN", "THIEU_TAI", "DUONG_KINH_LA",
            "NGOAI_PHAM_VI_CAM_BIEN", "NGOAI_PHAM_VI_TAN_SO_KHAI_BAO",
            "THIEU_TIN_HIEU_DE", "LOI_DOC_FILE",
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
    data_root = Path(data_root)
    mat_files = sorted(data_root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"Không tìm thấy file .mat nào trong {data_root}.")

    rows = []
    for fp in mat_files:
        meta = parse_metadata_from_filename(fp)
        content = inspect_mat_file(fp)
        rows.append({"file_path": str(fp), **meta, **content})

    return run_sanity_checks(pd.DataFrame(rows))