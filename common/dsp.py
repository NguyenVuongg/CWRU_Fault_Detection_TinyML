# -*- coding: utf-8 -*-
"""
common/dsp.py
=============
Các hàm xử lý tín hiệu dùng chung: lọc, Envelope Demodulation, FFT.
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(x, fs, low_hz, high_hz, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass") # type: ignore
    return filtfilt(b, a, x)


def lowpass_filter(x, fs, cutoff_hz, order=2):
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="lowpass") # type: ignore
    return filtfilt(b, a, x)


def square_law_envelope(x, fs, band=(1000, 4000), lp_cutoff=500, lp_order=2):
    """Envelope[n] = LowPass( (BandPass(x))^2 ) — đúng công thức mục 1.3.

    band mặc định là VÍ DỤ — dải thật cần chọn qua quan sát phổ FFT trên
    file có lỗi rõ ràng (xem notebook 03_frequency_domain.ipynb) kết hợp
    tài liệu kỹ thuật SKF 6205."""
    filtered = bandpass_filter(x, fs, band[0], band[1])
    squared = filtered ** 2
    return lowpass_filter(squared, fs, lp_cutoff, order=lp_order)


def hilbert_envelope(x, fs, band=(1000, 4000)):
    """Envelope qua Hilbert Transform — CHỈ dùng đối chiếu/kiểm tra chéo ở
    Giai đoạn 0, KHÔNG dùng trong pipeline triển khai MCU (Square-Law thay
    thế nó, xem RQ4/H4 — so sánh tài nguyên thật để ở Giai đoạn 3)."""
    filtered = bandpass_filter(x, fs, band[0], band[1])
    return np.abs(np.asarray(hilbert(filtered)))