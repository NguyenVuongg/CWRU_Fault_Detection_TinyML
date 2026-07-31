# -*- coding: utf-8 -*-
"""
common/order.py
=================
[Giai đoạn 1 — mục 1.4, Nhóm B] Trích biên độ tại hài bậc 1–3 của các tần
số mục tiêu (f_rot, BPFO, BPFI, BSF, FTF) từ phổ FFT/Order.

Tách riêng khỏi dsp.py vì đây là logic "trích đặc trưng" (feature
engineering), không phải "xử lý tín hiệu" (signal processing) thuần túy.
"""

import numpy as np

from .config import bearing_fault_frequencies, SKF_6205_GEOMETRY


def amplitude_near_frequency(freqs, mag, target_freq_hz, search_width_hz=2.0):
    """
    Biên độ LỚN NHẤT trong cửa sổ [target - width, target + width] quanh
    tần số mục tiêu — bù trừ sai số nhỏ giữa RPM danh định (dùng để tính
    target_freq_hz) và RPM thực tế của từng file. KHÔNG lấy đúng 1 bin FFT
    tại target_freq_hz vì gần như chắc chắn lệch do rời rạc hóa tần số
    (frequency resolution = fs / n_samples).
    """
    freqs = np.asarray(freqs)
    mag = np.asarray(mag)
    mask = (freqs >= target_freq_hz - search_width_hz) & (freqs <= target_freq_hz + search_width_hz)
    if not np.any(mask):
        return 0.0
    return float(np.max(mag[mask]))


def extract_order_features(freqs, mag, rpm, target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                            harmonics=(1, 2, 3), search_width_hz=2.0,
                            geometry=SKF_6205_GEOMETRY, prefix="order"):
    """
    Trích đặc trưng Nhóm B (mục 1.4): biên độ tại hài bậc 1..len(harmonics)
    của từng tần số mục tiêu trong target_names.

    Số chiều trả về = len(target_names) * len(harmonics) — đúng công thức
    "= số tần số mục tiêu × số hài" ghi trong đề cương.

    Trả về dict, khóa dạng "order_BPFO_h1", "order_BPFO_h2", ... — tiền tố
    "order_" giúp dễ lọc nhóm đặc trưng khi phân tích Feature Importance
    ở Giai đoạn 2.2.

    LƯU Ý: nếu target_names bao gồm các tên KHÔNG có trong kết quả của
    bearing_fault_frequencies() (vd gõ nhầm), hàm sẽ raise KeyError ngay,
    không âm thầm bỏ qua — tránh lỗi khó phát hiện trong bể đặc trưng lớn.
    """
    fault_freqs = bearing_fault_frequencies(rpm, geometry)
    feats = {}
    for name in target_names:
        if name not in fault_freqs:
            raise KeyError(
                f"'{name}' không có trong kết quả bearing_fault_frequencies() "
                f"— các tên hợp lệ: {list(fault_freqs.keys())}"
            )
        base_freq = fault_freqs[name]
        for h in harmonics:
            key = f"{prefix}_{name}_h{h}"
            feats[key] = amplitude_near_frequency(freqs, mag, base_freq * h, search_width_hz)
    return feats
