# -*- coding: utf-8 -*-
"""
common/config.py
=================
Hằng số và công thức dùng chung cho toàn bộ dự án.
"""

from typing import Any, List, Optional, SupportsFloat, SupportsIndex, Union
import numpy as np

# ---------------------------------------------------------------------------
# Hình học vòng bi SKF 6205-2RS JEM (Drive-End)
# ---------------------------------------------------------------------------
SKF_6205_GEOMETRY = {
    "n_balls": 9,
    "ball_diameter_in": 0.3126,
    "pitch_diameter_in": 1.537,
    "contact_angle_deg": 0.0,
}

SKF_VALID_FAULT_DIAMETERS_MILS = {7, 14, 21}
NTN_FAULT_DIAMETERS_MILS = {28, 40}

def _validate_rpm(rpm, caller: str) -> float:
    r = float(rpm)
    if r != r:
        raise ValueError(f"{caller}: rpm là NaN.")
    if r <= 0.0:
        raise ValueError(f"{caller}: rpm phải > 0, nhận {rpm!r}.")
    return r

def bearing_fault_frequencies(rpm: float, geometry: dict = SKF_6205_GEOMETRY) -> dict:
    f_r = _validate_rpm(rpm, "bearing_fault_frequencies") / 60.0
    n = geometry["n_balls"]
    bd = geometry["ball_diameter_in"]
    pd = geometry["pitch_diameter_in"]
    phi = np.deg2rad(geometry["contact_angle_deg"])
    ratio = (bd / pd) * np.cos(phi)

    return {
        "f_rot": f_r,
        "BPFO": (n / 2.0) * f_r * (1 - ratio),
        "BPFI": (n / 2.0) * f_r * (1 + ratio),
        "BSF": (pd / (2.0 * bd)) * f_r * (1 - ratio ** 2),
        "FTF": (f_r / 2.0) * (1 - ratio),
    }

def hz_to_order(freq_hz, rpm: float):
    f_r = _validate_rpm(rpm, "hz_to_order") / 60.0
    return np.asarray(freq_hz) / f_r


# ---------------------------------------------------------------------------
# Phạm vi dữ liệu đã chốt
# ---------------------------------------------------------------------------
SCOPE_LABELS: List[str] = ["Normal", "IR", "OR", "B"]
SCOPE_LOADS_HP: List[int] = [0, 1, 2, 3]

SCOPE = {
    "sensor_location": "DE",
    "sampling_rate_hz": 12000,
    "outer_race_position": "Centered",
    "labels": SCOPE_LABELS,
    "loads_hp": SCOPE_LOADS_HP,
}

_REQUIRED_SCOPE_KEYS = {
    "sensor_location", "sampling_rate_hz", "outer_race_position",
    "labels", "loads_hp",
}
if set(SCOPE) != _REQUIRED_SCOPE_KEYS:
    raise ValueError("SCOPE sai tập khóa.")

# ---------------------------------------------------------------------------
# Sơ đồ 10 LỚP là bài toán CHÍNH
# ---------------------------------------------------------------------------
CLASSES_10 = [
    "Normal",
    "IR_007", "IR_014", "IR_021",
    "OR_007", "OR_014", "OR_021",
    "B_007", "B_014", "B_021",
]

CLASSES_4 = ["Normal", "IR", "OR", "B"]

LABEL_COL_BY_SCHEME = {"10class": "class_label", "4class": "label"}
CLASSES_BY_SCHEME = {"10class": CLASSES_10, "4class": CLASSES_4}
CLASS_SCHEME_DEFAULT = "10class"

def make_class_label(
    label: Any,
    fault_diameter_mils: Optional[Union[SupportsFloat, SupportsIndex, str]] = None,
) -> str:
    if label is None:
        raise ValueError("label = None: không suy ra được nhãn 10 lớp.")

    label = str(label).strip()
    if label == "Normal":
        return "Normal"

    if label not in SCOPE_LABELS:
        raise ValueError(f"label='{label}' không thuộc phạm vi đã chốt {SCOPE_LABELS}.")

    if fault_diameter_mils is None:
        raise ValueError(f"label='{label}' là nhãn lỗi nên BẮT BUỘC có fault_diameter_mils.")

    try:
        d_val = float(fault_diameter_mils)
    except (TypeError, ValueError):
        d_val = float("nan")
    if d_val != d_val:
        raise ValueError(f"fault_diameter_mils không hợp lệ.")

    diameter = int(round(d_val))
    if diameter not in SKF_VALID_FAULT_DIAMETERS_MILS:
        raise ValueError(f"fault_diameter_mils={diameter} ngoài phạm vi SKF.")

    class_label = f"{label}_{diameter:03d}"
    if class_label not in CLASSES_10:
        raise ValueError(f"'{class_label}' không thuộc CLASSES_10.")
    return class_label


NOMINAL_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
CANDIDATE_SAMPLING_RATES_HZ = [12000, 24000, 48000]

EXPECTED_DURATION_SEC = 10.0
DURATION_TOLERANCE_SEC = 3.0

# ---------------------------------------------------------------------------
# Envelope + cửa sổ 
# ---------------------------------------------------------------------------
RESONANCE_BAND_HZ = (2300.0, 3800.0)
RESONANCE_FC_HZ = (RESONANCE_BAND_HZ[0] + RESONANCE_BAND_HZ[1]) / 2.0

# Ngưỡng lowpass envelope — CHỐT 750 Hz
LP_CUTOFF_HZ = 750.0

LP_CUTOFF_HZ_NARROW = 500.0
LP_CUTOFF_HZ_WIDE = LP_CUTOFF_HZ

WINDOW_SIZE_RAW = 2048
STRIDE_RAW = 1024

ENV_DECIM = 2
WINDOW_SIZE_ENV = WINDOW_SIZE_RAW // ENV_DECIM
STRIDE_ENV = STRIDE_RAW // ENV_DECIM
WARMUP_SAMPLES = WINDOW_SIZE_RAW