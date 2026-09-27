# -*- coding: utf-8 -*-
"""
common/features_dynamic.py
=========================
Trích xuất đặc trưng động dựa trên band_hz cố định và hỗ trợ
3 phương pháp giải điều chế: Square-Law, Hilbert, và Hybrid DSP.

Quy ước RQ:
    - method='hilbert' là alias của baseline nhân quả 'hilbert_fir'.
    - Mọi phương pháp dùng chung dsp.BENCHMARK_LP_CUTOFF_HZ.
    - Envelope phải được tính trên tín hiệu nguyên trước khi cắt cửa sổ.
"""

import warnings
import numpy as np
import pandas as pd

from .config import (NOMINAL_RPM_BY_LOAD, RESONANCE_BAND_HZ,  # noqa: F401
                     WARMUP_SAMPLES)
from . import dsp, features_full
from .dsp import hybrid_envelope  # noqa: F401


# --- HÀM CHỌN PHƯƠNG PHÁP GIẢI ĐIỀU CHẾ ---
def apply_envelope_method(window_data, fs, band_hz, method='square_law',
                          lp_cutoff=dsp.BENCHMARK_LP_CUTOFF_HZ,
                          bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                          lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER, **kwargs):
    """Uỷ quyền cho dsp.envelope_by_method() với chung cấu hình lọc."""
    return dsp.envelope_by_method(
        window_data, fs=fs, band=band_hz, method=method, lp_cutoff=lp_cutoff,
        bandpass_order=bandpass_order, lowpass_order=lowpass_order, **kwargs)


# --- HÀM TRÍCH XUẤT ĐẶC TRƯNG ĐỘNG (TƯƠNG THÍCH FEATURES_FULL) ---
def extract_features_dynamic_from_signals(signals_series, band_hz, fs=12000,
                                          rpm=1750, load_hp_series=None,
                                          rpm_series=None,
                                          method='square_law',
                                          window_size=2048, stride=1024,
                                          warmup_samples=None,
                                          envelope_kwargs=None):
    """
    Trích đặc trưng từ các TÍN HIỆU NGUYÊN (chưa cắt cửa sổ), với
    phương pháp giải điều chế được chỉ định.

    SỐ CHIỀU THỰC TẾ LÀ 28 (Nhóm B 12->10, Nhóm C 9->7).
    Envelope được tính MỘT LẦN trên toàn bộ tín hiệu rồi mới cắt thành cửa sổ.
    """
    signals_list = [np.asarray(s, dtype=np.float64) for s in signals_series]
    n_signals = len(signals_list)

    if rpm_series is not None:
        rpm_list = [float(v) for v in rpm_series]
        if len(rpm_list) != n_signals:
            raise ValueError("rpm_series phải cùng độ dài với signals_series")
    elif load_hp_series is not None:
        load_list = list(load_hp_series)
        if len(load_list) != n_signals:
            raise ValueError("load_hp_series phải cùng độ dài với signals_series")
        rpm_list = [float(NOMINAL_RPM_BY_LOAD.get(load, rpm)) for load in load_list]
    else:
        rpm_list = [float(rpm)] * n_signals

    warmup_samples = WARMUP_SAMPLES if warmup_samples is None else int(warmup_samples)
    envelope_kwargs = dict(envelope_kwargs or {})

    rows = []
    n_skipped = 0
    for signal_idx, (sig, current_rpm) in enumerate(zip(signals_list, rpm_list)):
        num_windows = (len(sig) - window_size) // stride + 1
        
        if num_windows <= 0 or (num_windows - 1) * stride < warmup_samples:
            n_skipped += 1
            continue

        env_full = apply_envelope_method(sig, fs, band_hz, method,
                                         **envelope_kwargs)

        for i in range(num_windows):
            s0 = i * stride
            if s0 < warmup_samples:
                continue
            
            x_window = sig[s0:s0 + window_size]
            env_window = env_full[s0:s0 + window_size]

            feats = {}
            feats.update(features_full.extract_time_domain_features(x_window))
            feats.update(features_full.extract_order_domain_features(
                x_window, fs=fs, rpm=current_rpm))
            feats.update(features_full.extract_envelope_features_from_envelope(
                env_window, fs=fs, rpm=current_rpm))
            
            rows.append({"signal_idx": signal_idx, "window_idx": i,
                         "start_idx": s0, **feats})

    if n_skipped:
        print(f"[CẢNH BÁO] Bỏ qua {n_skipped}/{n_signals} tín hiệu do quá ngắn hoặc rơi vào vùng warmup.")

    return pd.DataFrame(rows)


def extract_features_dynamic(windows_series, band_hz, fs=12000, rpm=1750,
                              load_hp_series=None, method='square_law'):
    """
    [DEPRECATED — không dùng cho kết quả RQ trong báo cáo.]
    """
    warnings.warn(
        "extract_features_dynamic() lọc envelope trên TỪNG CỬA SỔ đã cắt "
        "sẵn -> reset trạng thái bộ lọc nhân quả ở mỗi cửa sổ (transient). "
        "Hãy chuyển sang extract_features_dynamic_from_signals()",
        DeprecationWarning,
        stacklevel=2,
    )

    features_list = []
    windows_list = list(windows_series)
    
    if load_hp_series is not None:
        load_hp_list = list(load_hp_series)
        if len(load_hp_list) != len(windows_list):
            raise ValueError("load_hp_series phải cùng độ dài với windows_series")
    else:
        load_hp_list = None

    for i, window in enumerate(windows_list):
        w_raw = np.array(window, dtype=np.float32)

        if load_hp_list is not None:
            current_rpm = NOMINAL_RPM_BY_LOAD.get(load_hp_list[i], rpm)
        else:
            current_rpm = rpm

        time_feats = features_full.extract_time_domain_features(w_raw)
        order_feats = features_full.extract_order_domain_features(w_raw, fs=fs, rpm=current_rpm)

        env_signal = apply_envelope_method(w_raw, fs, band_hz, method)
        env_feats = features_full.extract_envelope_features_from_envelope(env_signal, fs=fs, rpm=current_rpm)

        all_feats = {}
        all_feats.update(time_feats)
        all_feats.update(order_feats)
        all_feats.update(env_feats)

        features_list.append(all_feats)

    return pd.DataFrame(features_list)