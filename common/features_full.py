# -*- coding: utf-8 -*-
"""
common/features_full.py
=========================
[Giai đoạn 1 — mục 1.4] Bể đặc trưng đa miền đầy đủ: Nhóm A (thời gian),
Nhóm B (phổ Order), Nhóm C (phổ Envelope).

KHÁC với common/features.py (chỉ có 5 đặc trưng đơn giản, dùng minh họa
RQ1/H1 ở Giai đoạn 0, notebook 06) — file này là bể đặc trưng CHÍNH THỨC
dùng cho Feature Selection (mục 2.2) và huấn luyện mô hình (Giai đoạn 2).
"""

import numpy as np
from scipy.stats import kurtosis, skew

from . import dsp
from .config import bearing_fault_frequencies, SKF_6205_GEOMETRY
from .order import extract_order_features


# ============================================================================
# NHÓM A — Miền thời gian (11 đặc trưng, số chiều CỐ ĐỊNH theo mục 1.4)
# ============================================================================
def extract_time_domain_features(x, prefix="time"):
    """11 đặc trưng miền thời gian: Mean, Std, RMS, Peak, Kurtosis, Skewness,
    Variance, Crest Factor, Shape Factor, Impulse Factor, Margin Factor.

    Công thức chuẩn dùng trong phân tích rung động công nghiệp:
        Crest Factor   = Peak / RMS
        Shape Factor   = RMS  / mean(|x|)
        Impulse Factor = Peak / mean(|x|)
        Margin Factor  = Peak / mean(sqrt(|x|))^2
    """
    x = np.asarray(x, dtype=np.float64)
    mean_abs = np.mean(np.abs(x))
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(x)))

    eps = 1e-12  # tránh chia cho 0 trên các đoạn tín hiệu gần như hằng số
    feats = {
        "mean": np.mean(x),
        "std": np.std(x),
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis(x),
        "skewness": skew(x),
        "variance": np.var(x),
        "crest_factor": peak / (rms + eps),
        "shape_factor": rms / (mean_abs + eps),
        "impulse_factor": peak / (mean_abs + eps),
        "margin_factor": peak / (mean_sqrt_abs ** 2 + eps),
    }
    return {f"{prefix}_{k}": v for k, v in feats.items()}


# ============================================================================
# NHÓM B — Phổ Order (số chiều = số tần số mục tiêu × số hài)
# ============================================================================
def extract_order_domain_features(x, fs, rpm, target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                                   harmonics=(1, 2, 3), search_width_hz=2.0,
                                   geometry=SKF_6205_GEOMETRY):
    """Wrapper: FFT tín hiệu thô rồi trích biên độ tại hài của tần số mục
    tiêu (dùng common/order.py). Xem docstring order.extract_order_features."""
    freqs, mag = dsp.compute_fft(x, fs)
    return extract_order_features(
        freqs, mag, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry, prefix="order",
    )


# ============================================================================
# NHÓM C — Phổ Envelope (Square-Law) — tập trung vào tần số ĐỘNG HỌC LỖI,
# không gồm f_rot (mất cân bằng trục không phải cơ chế điều biên bởi lỗi)
# ============================================================================
def extract_envelope_features(x, fs, rpm, band_hz, lp_cutoff_hz=500,
                               target_names=("BPFO", "BPFI", "BSF"),
                               harmonics=(1, 2, 3), search_width_hz=2.0,
                               geometry=SKF_6205_GEOMETRY):
    """
    band_hz: dải bandpass đã CHỐT ở Giai đoạn 0 (notebook 03/05) — bắt buộc
    truyền vào tường minh, KHÔNG có giá trị mặc định ở đây, để tránh vô
    tình dùng nhầm dải ví dụ chưa qua kiểm chứng của Giai đoạn 0.
    """
    envelope = dsp.square_law_envelope(x, fs, band=band_hz, lp_cutoff=lp_cutoff_hz)
    freqs, mag = dsp.compute_fft(envelope - envelope.mean(), fs)
    return extract_order_features(
        freqs, mag, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry, prefix="envelope",
    )


# ============================================================================
# GHÉP CẢ 3 NHÓM
# ============================================================================
def extract_full_feature_vector(x, fs, rpm, band_hz,
                                 order_target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                                 envelope_target_names=("BPFO", "BPFI", "BSF"),
                                 harmonics=(1, 2, 3), search_width_hz=2.0,
                                 geometry=SKF_6205_GEOMETRY):
    """
    Trả về 1 dict gộp cả 3 nhóm — mỗi key có tiền tố rõ ràng (time_/order_/
    envelope_) để lọc theo nhóm khi phân tích Feature Importance (mục 2.2).

    Số chiều mặc định = 11 (Nhóm A) + 4×3 (Nhóm B) + 3×3 (Nhóm C) = 32.
    Số chiều THẬT phải được đếm lại sau khi chốt band_hz/target_names cuối
    cùng (đúng nguyên tắc mục 1.4: chốt logic trước, đếm số chiều thật sau,
    KHÔNG cố định trước một con số rồi tìm cách khớp).
    """
    feats = {}
    feats.update(extract_time_domain_features(x))
    feats.update(extract_order_domain_features(
        x, fs, rpm, target_names=order_target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry,
    ))
    feats.update(extract_envelope_features(
        x, fs, rpm, band_hz=band_hz, target_names=envelope_target_names,
        harmonics=harmonics, search_width_hz=search_width_hz, geometry=geometry,
    ))
    return feats


def build_full_feature_table(manifest_df, band_hz, load_de_signal_fn, **kwargs):
    """
    Chạy extract_full_feature_vector() cho TOÀN BỘ file trong manifest_df,
    trả về DataFrame (mỗi dòng 1 file, cột file_id/label/load_hp + các
    đặc trưng). Dùng lại ở notebook Giai đoạn 1 (feature pool) và làm input
    trực tiếp cho common/training.py ở Giai đoạn 2.

    load_de_signal_fn: hàm nhận file_path (str) -> mảng tín hiệu, thường là
    io_utils.load_de_signal. Truyền vào thay vì import cứng để tránh phụ
    thuộc vòng giữa features_full.py và io_utils.py.
    """
    import pandas as pd
    from pathlib import Path
    from .config import NOMINAL_RPM_BY_LOAD, SCOPE

    rows = []
    for _, row in manifest_df.iterrows():
        if row.get("label") is None or (isinstance(row.get("label"), float)):
            continue
        load_hp = row.get("load_hp")
        rpm = NOMINAL_RPM_BY_LOAD.get(load_hp)
        if rpm is None:
            continue

        x = load_de_signal_fn(Path(row["file_path"]))
        feats = extract_full_feature_vector(
            x, fs=SCOPE["sampling_rate_hz"], rpm=rpm, band_hz=band_hz, **kwargs
        )
        rows.append({
            "file_id": row["file_path"],
            "label": row["label"],
            "load_hp": load_hp,
            "fault_diameter_mils": row.get("fault_diameter_mils"),
            **feats,
        })

    return pd.DataFrame(rows)
