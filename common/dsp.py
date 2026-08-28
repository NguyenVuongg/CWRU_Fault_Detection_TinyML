# -*- coding: utf-8 -*-
"""
common/dsp.py
=============
Các hàm xử lý tín hiệu dùng chung: lọc, Envelope Demodulation, FFT.
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, stft, sosfilt
from scipy.signal import lfilter
import numpy as np

import scipy.signal as sig
import scipy.stats as stats


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

def square_law_envelope(x, fs, band, lp_cutoff, order=4):
    """
    Trích xuất đường bao bằng Square-Law.
    Hiệu năng bám sát Hilbert nhưng chỉ dùng mảng float32 thực.
    """
    nyq = 0.5 * fs
    
    # 1. BỘ LỌC BANDPASS (Bắt dải cộng hưởng)
    low = band[0] / nyq
    high = band[1] / nyq
    b_bp, a_bp = butter(order, [low, high], btype='bandpass') # type: ignore
    
    # Ép kiểu mảng để Pylance không nhầm lẫn x_bandpassed là tuple
    x_bandpassed = np.asarray(lfilter(b_bp, a_bp, x))
    
    # 2. SQUARE-LAW (Bình phương tín hiệu)
    x_squared = x_bandpassed ** 2
    
    # 3. KHỬ DC OFFSET
    x_squared_ac = x_squared - np.mean(x_squared)
    
    # 4. BỘ LỌC LOWPASS
    lp = lp_cutoff / nyq
    b_lp, a_lp = butter(order, lp, btype='lowpass') # type: ignore
    
    # Ép kiểu đầu ra để file 02_choose_bandpass có thể gọi .mean()
    envelope = np.asarray(lfilter(b_lp, a_lp, x_squared_ac))
    
    return envelope


def hilbert_envelope(x, fs, band, bandpass_order=4):
    filtered = bandpass_filter(x, fs, band[0], band[1], order=bandpass_order)
    return np.abs(np.asarray(hilbert(filtered)))


def hybrid_envelope(x, fs, band, bandpass_order=4, alpha=1.0, beta=0.375):
    """
    Tiền xử lý Lai ghép (Hybrid Demodulation) tối ưu cho MCU:
    1. Causal Bandpass Filter (IIR Butterworth)
    2. IIR All-Pass Phase Splitter (Tạo tín hiệu Trực giao I/Q)
    3. Alpha-Max Plus Beta-Min (Xấp xỉ Envelope tuyến tính, không dùng hàm sqrt)
    
    Args:
        x: mảng tín hiệu thô 1D.
        fs: tần số lấy mẫu thực (Hz).
        band: (low_hz, high_hz) dải cộng hưởng.
        bandpass_order: bậc Butterworth cho bandpass (mặc định 4).
        alpha, beta: Hệ số xấp xỉ (mặc định 1.0 và 3/8 cho sai số max ~6.8%).
    """
    nyq = 0.5 * fs
    
    # ==========================================================
    # BƯỚC 1: Lọc dải cộng hưởng (Causal Bandpass)
    # ==========================================================
    low, high = band[0] / nyq, band[1] / nyq
    sos_bp = butter(bandpass_order, [low, high], btype="bandpass", output="sos")
    x_bp = np.asarray(sosfilt(sos_bp, x))
    
    # ==========================================================
    # BƯỚC 2: Mạng trực giao IIR All-Pass (Tạo I và Q)
    # ==========================================================
    # Tính tần số trung tâm của dải cộng hưởng (Radians)
    fc = (band[0] + band[1]) / 2.0
    w_c = 2.0 * np.pi * fc / fs
    
    # Thiết kế cặp IIR lệch pha 90 độ:
    # Nhánh I (In-phase): Trễ 1 mẫu (Delay z^-1)
    # Nhánh Q (Quadrature): Bộ lọc All-pass bậc 1, H(z) = (a + z^-1)/(1 + a*z^-1)
    # Giải phương trình pha để lệch đúng 90 độ tại tần số trung tâm w_c:
    a_coeff = -1.0 / (np.sin(w_c) + np.cos(w_c))
    
    # Khởi tạo mảng (Mô phỏng bộ nhớ đệm trên MCU)
    N = len(x_bp)
    I = np.zeros(N)
    
    # Tính nhánh I: I[n] = x_bp[n-1]
    I[1:] = x_bp[:-1]
    
    # Tính nhánh Q qua phương trình sai phân: Q[n] + a*Q[n-1] = a*x[n] + x[n-1]
    # Dùng lfilter để tính toán véc-tơ nhanh trên Python (ánh xạ đúng logic C/C++)
    b_ap = np.array([a_coeff, 1.0])
    a_ap = np.array([1.0, a_coeff])
    Q = lfilter(b_ap, a_ap, x_bp)
    
    # ==========================================================
    # BƯỚC 3: Xấp xỉ Alpha-Max Plus Beta-Min
    # ==========================================================
    abs_I = np.abs(I)
    abs_Q = np.abs(Q)
    
    max_val = np.maximum(abs_I, abs_Q)
    min_val = np.minimum(abs_I, abs_Q)
    
    # Tính đường bao mà không cần dùng hàm np.sqrt(I^2 + Q^2)
    envelope = alpha * max_val + beta * min_val
    
    # ==========================================================
    # BƯỚC 4: Khử DC Offset
    # ==========================================================
    envelope_ac = envelope - np.mean(envelope)
    
    return np.asarray(envelope_ac, dtype=np.float64)


def compute_fft(x, fs, apply_window=False):
    """
    Tính toán phổ biên độ (FFT) của tín hiệu rời rạc.
 
    Args:
        x (array-like): Mảng tín hiệu đầu vào.
        fs (float): Tần số lấy mẫu (Hz).
        apply_window (bool): Nếu True, nhân tín hiệu với cửa sổ Hann trước
            FFT để giảm rò rỉ phổ (spectral leakage), có bù hệ số suy giảm
            biên độ do cửa sổ gây ra. Mặc định False để KHÔNG thay đổi
            hành vi/giá trị đặc trưng đã dùng trong pipeline hiện tại -
            chỉ bật khi đã đánh giá lại ảnh hưởng lên các ngưỡng/kết quả
            đã có.
 
    Returns:
        tuple: (freqs, mag)
            - freqs: Mảng các giá trị tần số (Hz).
            - mag: Mảng biên độ (magnitude) tương ứng.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
 
    window_gain = 1.0
    if apply_window:
        win = np.hanning(N)
        window_gain = win.mean() if win.mean() > 0 else 1.0
        x = x * win
 
    # Dùng rfft (Real FFT) tối ưu hơn cho tín hiệu thực so với fft thông thường
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_values = np.fft.rfft(x)
 
    # Tính biên độ (Magnitude) và chuẩn hóa năng lượng theo N
    mag = np.abs(fft_values) * 2.0 / N
 
    # Thành phần DC (tần số 0 Hz) và thành phần Nyquist (nếu N chẵn) không nhân 2
    mag[0] /= 2.0
    if N % 2 == 0:
        mag[-1] /= 2.0
 
    if apply_window:
        mag = mag / window_gain
 
    return freqs, mag
 