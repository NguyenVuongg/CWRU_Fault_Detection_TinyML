# -*- coding: utf-8 -*-
"""
common/io_utils.py
===================
Đọc file .mat CWRU, dựng bảng manifest, chạy sanity check (mục 0.1).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from typing import Any

from scipy.io import loadmat

from . import config as cfg


def parse_metadata_from_filename(filepath: Path) -> dict:
    """
    VÍ DỤ MẪU — chỉnh lại cho khớp cấu trúc dữ liệu thật của bạn.

    Giả định ví dụ: .../<load_hp>hp/<label>_<diameter_mils>_<or_position?>.mat
    File .mat gốc từ CWRU Bearing Data Center chỉ có tên số (vd 105.mat) —
    không tự chứa metadata. Nếu dùng bản gốc, thay hàm này bằng cách merge
    một file lookup CSV bạn tự tạo theo bảng tra cứu chính thức của CWRU.
    """
    name = filepath.stem
    parent = filepath.parent.name

    load_match = re.search(r"(\d+)\s*hp", parent, re.IGNORECASE)
    load_hp = int(load_match.group(1)) if load_match else None

    label = None
    diameter_mils = None
    or_position = None

    if name.lower().startswith("normal"):
        label = "Normal"
    else:
        parts = name.split("_")
        prefix = parts[0].upper()
        if prefix in ("IR", "OR", "B"):
            label = prefix
        if len(parts) > 1 and parts[1].isdigit():
            diameter_mils = int(parts[1])
        if label == "OR" and len(parts) > 2:
            or_position = parts[2]

    return {
        "load_hp": load_hp,
        "label": label,
        "fault_diameter_mils": diameter_mils,
        "or_position": or_position,
    }


def inspect_mat_file(filepath: Path) -> dict:
    # Báo cho Pylance biết value của dict có thể là bất kỳ kiểu gì (Any)
    result: dict[str, Any] = {
        "n_samples_DE": None, "n_samples_FE": None, "n_samples_BA": None,
        "rpm_from_file": None, "read_error": None,
    }
    try:
        mat = loadmat(str(filepath))
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
    """Đọc thẳng mảng tín hiệu DE (dùng ở các notebook phân tích tín hiệu)."""
    mat = loadmat(str(filepath))
    for key in mat.keys():
        if key.endswith("_DE_time"):
            return np.asarray(mat[key]).ravel()
    raise KeyError(f"Không tìm thấy biến '..._DE_time' trong {filepath}")


def run_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    warnings_list = [[] for _ in range(len(df))]

    # Thay df.iterrows() bằng df.to_dict('records') để tránh lỗi typing của Pandas Series
    for i, row in enumerate(df.to_dict('records')):
        n = row.get("n_samples_DE")
        if n is None or pd.isna(n):
            continue

        dur_12k = n / 12000.0
        dur_48k = n / 48000.0
        ok_12k = abs(dur_12k - cfg.EXPECTED_DURATION_SEC) <= cfg.DURATION_TOLERANCE_SEC
        ok_48k = abs(dur_48k - cfg.EXPECTED_DURATION_SEC) <= cfg.DURATION_TOLERANCE_SEC

        if ok_48k and not ok_12k:
            warnings_list[i].append(
                f"NGHI_NGO_SAMPLING_RATE: n_samples={n} -> {dur_12k:.1f}s nếu "
                f"12kHz (bất thường), {dur_48k:.1f}s nếu 48kHz (hợp lý)."
            )
        elif not ok_12k and not ok_48k:
            warnings_list[i].append(
                f"THOI_LUONG_BAT_THUONG: n_samples={n} không khớp ~10s ở cả "
                f"12kHz ({dur_12k:.1f}s) lẫn 48kHz ({dur_48k:.1f}s)."
            )

        load_hp = row.get("load_hp")
        rpm_file = row.get("rpm_from_file")
        if load_hp in cfg.NOMINAL_RPM_BY_LOAD and rpm_file is not None and not pd.isna(rpm_file):
            rpm_nominal = cfg.NOMINAL_RPM_BY_LOAD[load_hp]
            if abs(rpm_file - rpm_nominal) > 20:
                warnings_list[i].append(
                    f"RPM_LECH: RPM file ({rpm_file:.0f}) lệch >20 so với "
                    f"danh định tải {load_hp}HP ({rpm_nominal})."
                )

        diam = row.get("fault_diameter_mils")
        if diam is not None and not pd.isna(diam):
            diam = int(diam)
            if diam in cfg.NTN_FAULT_DIAMETERS_MILS:
                warnings_list[i].append(
                    f"VONG_BI_NTN: đường kính {diam} mils dùng vòng bi NTN, "
                    f"KHÔNG dùng hình học SKF 6205 để tính BPFO/BPFI/BSF."
                )
            elif diam not in cfg.SKF_VALID_FAULT_DIAMETERS_MILS:
                warnings_list[i].append(f"DUONG_KINH_LA: {diam} mils không rõ nguồn gốc.")

        label = row.get("label")
        or_pos = row.get("or_position")
        if label == "OR":
            if or_pos is None or (isinstance(or_pos, float) and pd.isna(or_pos)):
                warnings_list[i].append("OR_THIEU_VI_TRI: nhãn OR nhưng không rõ vị trí lỗi.")
            elif cfg.SCOPE["outer_race_position"].lower() not in str(or_pos).lower():
                warnings_list[i].append(
                    f"OR_NGOAI_PHAM_VI: vị trí '{or_pos}' khác phạm vi đã chốt "
                    f"('{cfg.SCOPE['outer_race_position']}')."
                )

        if label is None:
            warnings_list[i].append("THIEU_NHAN: không parse được nhãn lỗi.")
        if load_hp is None:
            warnings_list[i].append("THIEU_TAI: không parse được mức tải.")

    df["warnings"] = ["; ".join(w) if w else "" for w in warnings_list]
    df["has_warning"] = df["warnings"] != ""
    return df


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
