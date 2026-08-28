# -*- coding: utf-8 -*-
"""
common/dsp.py
=============
Các hàm xử lý tín hiệu dùng chung: lọc, Envelope Demodulation, FFT.

LỊCH SỬ SỬA ĐỔI QUAN TRỌNG (đọc trước khi dùng):
  - square_law_envelope(): trước đây dùng CHUNG 1 tham số `order` cho cả
    bandpass và lowpass, khiến lowpass vô tình chạy ở bậc 4 thay vì bậc 2
    như đã CHỐT ở mục 1.3 đề cương. Nay tách thành `bandpass_order` và
    `lowpass_order` (mặc định 4 và 2) — đúng công thức
    Envelope[n] = LowPass_bậc2((BandPass_bậc4(x))^2).
  - Toàn bộ bộ lọc chuyển từ dạng hệ số (b, a) sang dạng `sos`
    (second-order sections) — ổn định số học hơn nhiều so với (b, a) ở
    bậc >=4, đặc biệt khi tần số cắt gần dải Nyquist.
  - hilbert_envelope() KHÔNG còn giá trị mặc định cho `band` — bắt buộc
    truyền tường minh, cùng nguyên tắc đã áp dụng cho square_law_envelope
    (tránh âm thầm dùng lại dải tần đã chốt cho 1 cấu hình cụ thể ở Giai
    đoạn 0 tại những chỗ gọi hàm khác chưa xác nhận lại dải đó có hợp lý
    không).
  - QUAN TRỌNG VỀ CÁCH GỌI square_law_envelope() TRONG PIPELINE CHÍNH
    THỨC: hàm này dùng lọc causal (sosfilt, mô phỏng đúng bộ lọc 1 chiều
    chạy trên MCU thật) với trạng thái khởi tạo = 0. Gọi hàm 1 LẦN cho
    TOÀN BỘ tín hiệu của 1 file (không lặp lại trên từng cửa sổ nhỏ) để
    tránh transient khởi động lặp lại ở đầu mỗi cửa sổ — xem
    features_full.build_full_feature_table(), nơi đã áp dụng đúng cách
    gọi này.
"""

import numpy as np
from scipy.signal import butter, hilbert
from scipy.signal import sosfilt, sosfiltfilt


def bandpass_filter(x, fs, low_hz, high_hz, order=4):
    """Bandpass zero-phase (sosfiltfilt) — dùng cho phân tích offline
    (Hilbert envelope, khảo sát Giai đoạn 0). KHÔNG dùng hàm này để mô
    phỏng lọc causal trên MCU — xem square_law_envelope()."""
    nyq = fs / 2.0
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def lowpass_filter(x, fs, cutoff_hz, order=2):
    """Lowpass zero-phase (sosfiltfilt) — dùng cho phân tích offline."""
    nyq = fs / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x)


def square_law_envelope(x, fs, band, lp_cutoff, bandpass_order=4, lowpass_order=2):
    """
    Trích xuất đường bao bằng Square-Law, tối ưu hóa cho Edge AI/MCU:
        Envelope[n] = LowPass_bậc(lowpass_order)( (BandPass_bậc(bandpass_order)(x))^2 )

    Mặc định bandpass_order=4, lowpass_order=2 — ĐÚNG mục 1.3 đề cương
    ("...rồi qua lọc thông thấp IIR Butterworth bậc 2..."). Trước đây 2 bộ
    lọc dùng chung 1 tham số `order=4`, khiến lowpass chạy sai bậc.

    Dùng `sosfilt` (causal - một chiều, không padding, không nhìn tương
    lai) để mô phỏng đúng bộ lọc chạy trên MCU thật. Vì lọc causal có
    trạng thái nội bộ khởi tạo bằng 0 mỗi lần gọi, hàm này PHẢI được gọi
    1 LẦN trên toàn bộ tín hiệu dài (1 file), rồi cắt cửa sổ trên kết quả
    đã lọc — KHÔNG gọi lặp lại trên từng cửa sổ ngắn (sẽ tạo transient
    khởi động giả ở đầu mỗi cửa sổ, làm méo biên độ/pha đường bao).

    Args:
        x: mảng tín hiệu 1D (khuyến nghị: toàn bộ chiều dài 1 file).
        fs: tần số lấy mẫu thực (Hz) của `x`.
        band: (low_hz, high_hz) dải cộng hưởng đã chốt.
        lp_cutoff: tần số cắt lowpass (Hz).
        bandpass_order: bậc Butterworth cho bandpass (mặc định 4).
        lowpass_order: bậc Butterworth cho lowpass (mặc định 2).
    """
    nyq = 0.5 * fs

    # 1. BANDPASS (bắt dải cộng hưởng) - causal
    low, high = band[0] / nyq, band[1] / nyq
    sos_bp = butter(bandpass_order, [low, high], btype="bandpass", output="sos")
    # np.asarray(...) ở đây không đổi giá trị lúc chạy (sosfilt không truyền
    # zi nên luôn trả về 1 mảng), chỉ để Pylance không suy luận nhầm kiểu
    # trả về là tuple (do sosfilt có overload trả (y, zf) khi có zi).
    x_bandpassed = np.asarray(sosfilt(sos_bp, x))

    # 2. SQUARE-LAW (bình phương tín hiệu)
    x_squared = x_bandpassed ** 2

    # 3. KHỬ DC OFFSET (thành phần A(t)^2/2 trung bình sinh ra bởi phép bình phương)
    x_squared_ac = x_squared - np.mean(x_squared)

    # 4. LOWPASS (tách sóng mang, giữ lại điều biên) - causal, bậc 2 mặc định
    lp = lp_cutoff / nyq
    sos_lp = butter(lowpass_order, lp, btype="lowpass", output="sos")
    envelope = np.asarray(sosfilt(sos_lp, x_squared_ac))

    return np.asarray(envelope, dtype=np.float64)


def hilbert_envelope(x, fs, band, bandpass_order=4):
    """Envelope[n] = |Hilbert( BandPass(x) )|.

    `band` BẮT BUỘC truyền tường minh - KHÔNG còn giá trị mặc định (trước
    đây default=(2300, 3800), gây rủi ro âm thầm tái sử dụng dải đã chốt
    cho 1 cấu hình cụ thể ở những lần gọi khác chưa xác nhận lại)."""
    filtered = bandpass_filter(x, fs, band[0], band[1], order=bandpass_order)
    return np.abs(np.asarray(hilbert(filtered)))


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