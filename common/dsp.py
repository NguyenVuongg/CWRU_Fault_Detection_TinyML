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
    
    """Dùng filtfilt (zero-phase) cho offline feature extraction. 
    Khi triển khai MCU thật (Giai đoạn 3), thay bằng lfilter (causal)."""
    
    return filtfilt(b, a, x)


def lowpass_filter(x, fs, cutoff_hz, order=2):
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="lowpass") # type: ignore
    return filtfilt(b, a, x)
#   Dùng filtfilt (zero-phase) cho offline feature extraction. Khi triển khai MCU thật (Giai đoạn 3), thay bằng lfilter (causal).

def square_law_envelope(x, fs, band=(2300, 4000), lp_cutoff=500, lp_order=2):
    """Envelope[n] = LowPass( (BandPass(x))^2 ) — đúng công thức mục 1.3.

    band mặc định là VÍ DỤ — dải thật cần chọn qua quan sát phổ FFT trên
    file có lỗi rõ ràng (xem notebook 03_frequency_domain.ipynb) kết hợp
    tài liệu kỹ thuật SKF 6205."""
    filtered = bandpass_filter(x, fs, band[0], band[1])
    squared = filtered ** 2
    return lowpass_filter(squared, fs, lp_cutoff, order=lp_order)


def hilbert_envelope(x, fs, band=(2300  , 3800)):
    """Envelope[n] = LowPass( (BandPass(x))² )

    band=(2300, 4000) đã chốt qua thực nghiệm Giai đoạn 1:
      - Chặn f_rot (~30 Hz) khỏi lọt vào envelope
      - Giữ vùng cộng hưởng cơ học chính của CWRU DE 12kHz
      - Đã kiểm chứng chéo trên IR/OR/B ở các đường kính 7-21 mils

    Dùng filtfilt (zero-phase) cho offline feature extraction.
    Khi triển khai MCU (Giai đoạn 3), thay bằng lfilter (causal)."""
    
    filtered = bandpass_filter(x, fs, band[0], band[1])
    return np.abs(np.asarray(hilbert(filtered)))

def compute_fft(x, fs):
    """
    Tính toán phổ biên độ (FFT) của tín hiệu rời rạc.
    
    Args:
        x (array-like): Mảng tín hiệu đầu vào.
        fs (float): Tần số lấy mẫu (Hz).
        
    Returns:
        tuple: (freqs, mag)
            - freqs: Mảng các giá trị tần số (Hz).
            - mag: Mảng biên độ (magnitude) tương ứng.
    """
    N = len(x)
    
    # Dùng rfft (Real FFT) tối ưu hơn cho tín hiệu thực so với fft thông thường
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft_values = np.fft.rfft(x)
    
    # Tính biên độ (Magnitude) và chuẩn hóa năng lượng theo N
    mag = np.abs(fft_values) * 2.0 / N
    
    # Thành phần DC (tần số 0 Hz) và thành phần Nyquist (nếu N chẵn) không nhân 2
    mag[0] /= 2.0
    if N % 2 == 0:
        mag[-1] /= 2.0
        
    return freqs, mag

def goertzel(x, target_freqs, fs):
    """Goertzel algorithm — tính biên độ tại M tần số mục tiêu.
    
    Args:
        x: tín hiệu đầu vào (1D array)
        target_freqs: list các tần số cần đo (Hz)
        fs: tần số lấy mẫu (Hz)
    
    Returns:
        dict {freq_hz: magnitude}
    """
    N = len(x)
    result = {}
    for f in target_freqs:
        k = int(0.5 + N * f / fs)
        omega = 2.0 * np.pi * k / N
        coeff = 2.0 * np.cos(omega)
        s_prev, s_prev2 = 0.0, 0.0
        for n in range(N):
            s = x[n] + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        mag = np.sqrt(s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2)
        result[f] = mag * 2.0 / N
    return result