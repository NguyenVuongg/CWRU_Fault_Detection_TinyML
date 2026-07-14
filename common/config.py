# -*- coding: utf-8 -*-
"""
common/config.py
=================
Hằng số và công thức dùng chung cho toàn bộ notebook Giai đoạn 0.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Hình học vòng bi SKF 6205-2RS JEM (Drive-End) — chỉ áp dụng cho lỗi
# đường kính 0.007" / 0.014" / 0.021"
# ---------------------------------------------------------------------------
SKF_6205_GEOMETRY = {
    "n_balls": 9,
    "ball_diameter_in": 0.3126,
    "pitch_diameter_in": 1.537,
    "contact_angle_deg": 0.0,
}

SKF_VALID_FAULT_DIAMETERS_MILS = {7, 14, 21}

# Đường kính lỗi dùng vòng bi NTN tương đương — KHÔNG dùng SKF_6205_GEOMETRY
# để tính BPFO/BPFI/BSF cho các file này nếu chưa xác minh lại hình học NTN.
NTN_FAULT_DIAMETERS_MILS = {28, 40}


def bearing_fault_frequencies(rpm: float, geometry: dict = SKF_6205_GEOMETRY) -> dict:
    """Tính BPFO/BPFI/BSF/FTF (Hz) theo công thức chuẩn vòng bi rãnh sâu."""
    f_r = rpm / 60.0
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
    """Order = f_Hz / f_rot."""
    f_r = rpm / 60.0
    return np.asarray(freq_hz) / f_r


# ---------------------------------------------------------------------------
# Phạm vi dữ liệu đã chốt (mục 0.1)
# ---------------------------------------------------------------------------
SCOPE = {
    "sensor_position": "DE",
    "sampling_rate_hz": 12000,
    "outer_race_position": "Centered",
    "labels": ["Normal", "IR", "OR", "B"],
    "loads_hp": [0, 1, 2, 3],
}

NOMINAL_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

EXPECTED_DURATION_SEC = 10.0
DURATION_TOLERANCE_SEC = 3.0
