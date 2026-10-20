# -*- coding: utf-8 -*-
"""
common/dsp.py
=============
Lọc, giải điều chế đường bao (Envelope Demodulation) và FFT dùng chung.

=============================================================================
GIAO THỨC SO SÁNH CÔNG BẰNG (đọc trước khi chạy bất kỳ thực nghiệm RQ2/RQ3)
=============================================================================
Toàn bộ đường ống được đưa vào bảng benchmark phải NHÂN QUẢ (causal), vì
mục tiêu của đề tài là chi phí/hiệu năng TRÊN MCU chạy realtime.

QUY ƯỚC BENCHMARK:
    - Mặc định mọi bộ lọc trong module này là NHÂN QUẢ (sosfilt).
    zero_phase=True chỉ dành cho minh họa/vẽ phổ offline.
  - hilbert_envelope() (scipy, phi nhân quả) giữ lại CHỈ để tham chiếu
    offline / vẽ hình — KHÔNG đưa vào bảng RQ2/RQ3.
    - Phương pháp benchmark và tham chiếu offline dùng CHUNG:
    bandpass_order, lp_cutoff, lowpass_order và cách khử DC. Xem các hằng
    BENCHMARK_* và envelope_by_method() — dispatcher DUY NHẤT.
"""

import numpy as np
from scipy.signal import butter, filtfilt, group_delay, hilbert, lfilter, sos2tf, sosfilt

from .config import ENV_DECIM, LP_CUTOFF_HZ, RESONANCE_BAND_HZ

# ---------------------------------------------------------------------------
# Cấu hình dùng CHUNG cho cả 2 phương pháp benchmark (RQ2/RQ3).
# Đổi ở ĐÂY, không hardcode rải rác trong từng hàm/notebook.
# ---------------------------------------------------------------------------
BENCHMARK_BANDPASS_ORDER = 4
BENCHMARK_LOWPASS_ORDER = 2
BENCHMARK_LP_CUTOFF_HZ = float(LP_CUTOFF_HZ)  # nguồn duy nhất: config.LP_CUTOFF_HZ
BENCHMARK_BAND_HZ = tuple(float(v) for v in RESONANCE_BAND_HZ)  # config.RESONANCE_BAND_HZ

# ---------------------------------------------------------------------------
# HỆ SỐ ALPHA-MAX PLUS BETA-MIN
# ---------------------------------------------------------------------------
# Xấp xỉ |I + jQ| ~= alpha*max(|I|,|Q|) + beta*min(|I|,|Q|), tránh sqrt.
#
# Có hai cặp hệ số chuẩn, tối ưu cho hai tiêu chí khác nhau. Đo trên toàn
# cung |I|>=|Q| cho các sai số sau:
#
#     cặp (0.947,     0.392)      -> sai số max 5.32% , trung bình 1.99%
#     cặp (0.9475436, 0.3924856)  -> sai số max 5.25% , trung bình 2.00%
#     cặp (0.9604339, 0.3978247)  -> sai số max 3.96% , trung bình 2.41%
#     cặp (1.0,       0.5)        -> sai số max 11.80%, trung bình 8.68%
#
# CHỌN cặp MINMAX làm mặc định. Lý do: đại lượng đề cương cam kết là SAI SỐ
# BIÊN ĐỘ TỐI ĐA, và sai số của xấp xỉ này dao động theo pha tức thời của
# tín hiệu — tức nó tự biến thành một thành phần điều biên giả nằm ĐÚNG
# trên đường bao, chỗ ta đang đi tìm vạch BPFO/BPFI. Ghim đỉnh sai số xuống
# 3.96% quan trọng hơn hạ sai số trung bình, dù trung bình có nhích nhẹ.

ALPHA_MAX_MINMAX = 0.960433870103   # tối ưu sai số TỐI ĐA (mặc định)
BETA_MIN_MINMAX = 0.397824734759
ALPHA_MAX_MSE = 0.947543636291      # tối ưu sai số TRUNG BÌNH BÌNH PHƯƠNG
BETA_MIN_MSE = 0.392485603          # hệ số tối ưu theo tiêu chí MSE


def _resolve_band(band):
    """band=None -> lấy dải cộng hưởng đã chốt ở config (mục 1.2).

    Để mọi hàm envelope có CÙNG một dải mặc định, thay vì mỗi notebook tự
    truyền một dải rồi so sánh các phương pháp trên những dải khác nhau.
    """
    if band is None:
        return BENCHMARK_BAND_HZ
    if len(band) != 2 or float(band[0]) >= float(band[1]):
        raise ValueError(f"band phải là (low_hz, high_hz) với low < high, nhận {band!r}.")
    return band

# Chế độ chọn hệ số chuẩn hóa cho nhánh Q của Hybrid.
# "center" giữ hành vi cũ; "minimax" giảm sai lệch tương đối lớn nhất trên band.
HYBRID_Q_MODES = ("center", "minimax")
HYBRID_Q_MODE_DEFAULT = "minimax"


def hybrid_sin_wc(fs, band=None, q_mode=HYBRID_Q_MODE_DEFAULT, fc_hz=None):
    """Tính sin(w_c) cho bộ sai phân I/Q của Hybrid.

    center: dùng tâm số học của band (hành vi cũ).
    minimax: lấy trung điểm giữa min/max sin(w) trên band để tối thiểu hóa
    sai lệch tương đối lớn nhất |sin(w)/sin(w_c) - 1|.
    Nếu truyền fc_hz, tần số này được ưu tiên hơn q_mode tự động.
    """
    try:
        fs = float(fs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fs phải là số dương, nhận {fs!r}.") from exc
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs phải là số dương hữu hạn, nhận {fs!r}.")
    if q_mode not in HYBRID_Q_MODES:
        raise ValueError(f"q_mode={q_mode!r} không hợp lệ; chọn {HYBRID_Q_MODES}.")

    band = _resolve_band(band)
    low_hz, high_hz = float(band[0]), float(band[1])
    if not (0.0 < low_hz < high_hz < fs / 2.0):
        raise ValueError(f"band phải thỏa 0 < low < high < fs/2; band={band}, fs={fs}.")

    if fc_hz is not None:
        try:
            fc = float(fc_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fc_hz phải là số hợp lệ, nhận {fc_hz!r}.") from exc
        if not np.isfinite(fc) or not 0.0 < fc < fs / 2.0:
            raise ValueError(f"fc_hz phải nằm trong (0, fs/2), nhận {fc_hz!r}.")
        return float(np.sin(2.0 * np.pi * fc / fs))

    if q_mode == "center":
        # Giữ nguyên phép tính của mode center để đối chứng với phiên bản cũ.
        fc = (band[0] + band[1]) / 2.0
        w_c = 2.0 * np.pi * fc / fs
        return float(np.sin(w_c))

    edge_sines = (
        float(np.sin(2.0 * np.pi * low_hz / fs)),
        float(np.sin(2.0 * np.pi * high_hz / fs)),
    )
    max_sine = max(edge_sines)
    if low_hz <= fs / 4.0 <= high_hz:
        max_sine = 1.0
    min_sine = min(edge_sines)
    sin_wc = 0.5 * (max_sine + min_sine)
    if not np.isfinite(sin_wc) or sin_wc <= 0.0:
        raise ValueError(f"Không xác định được sin(w_c) hợp lệ cho band={band}, fs={fs}.")
    return float(sin_wc)


# ---------------------------------------------------------------------------
# CHI PHÍ TRÊN MCU của RIÊNG khối giải điều chế (RQ2/RQ3, mục 1.2 và 3.1)
# ---------------------------------------------------------------------------
# CHỮ "RIÊNG" LÀ QUAN TRỌNG: các phương pháp dùng CHUNG bandpass
# Butterworth bậc 4 và lowpass bậc 2 (chi phí xem SHARED_STAGE_COST bên
# dưới). Hai khối đó triệt tiêu khi so sánh, nên
# bảng dưới chỉ đếm phần KHÁC NHAU. Cộng thêm SHARED_STAGE_COST nếu cần
# con số tuyệt đối cho toàn đường ống.
#
# Đây là ƯỚC LƯỢNG PHÂN TÍCH để định hướng, KHÔNG thay thế số đo thật
# bằng DWT->CYCCNT ở Giai đoạn 3 (xem mcu_export.latency_from_dwt_cycles).
# Cách đếm biquad cho chi phí benchmark:
# butter(N, [f1, f2], btype="bandpass") KHÔNG cho bộ lọc bậc N. N là bậc của
# NGUYÊN MẪU lowpass; phép biến đổi lowpass -> bandpass nhân đôi số cực, nên
# bậc thực tế là 2N. Kiểm chứng bằng số:
#     butter(4, [2300, 3800], btype="bandpass", output="sos").shape == (4, 6)
#     butter(2, 750,          btype="lowpass",  output="sos").shape == (1, 6)
# Mỗi biquad dạng transposed direct-form II (a0 chuẩn hóa = 1) tốn 5 nhân +
# 4 cộng/mẫu, nên:
#     bandpass bậc 4 -> 4 biquad -> 20 nhân + 16 cộng/mẫu
#     lowpass  bậc 2 -> 1 biquad ->  5 nhân +  4 cộng/mẫu
# Sai số này KHÔNG đổi thứ hạng RQ2 (khối chung triệt tiêu khi so sánh), nhưng
# nó làm HỎNG con số TUYỆT ĐỐI ở mục 3.1 — đúng con số sẽ đem đối chiếu với
# đo thật bằng DWT->CYCCNT. Chiều sai lại có lợi cho đề tài (khai thấp khối
# chung khiến 3 phép nhân dôi ra của Hybrid trông nhỏ hơn thực tế), nên càng
# phải sửa trước khi đưa vào báo cáo.
SHARED_STAGE_COST = {
    "bandpass_biquads": 4, "bandpass_mults_per_sample": 20, "bandpass_adds_per_sample": 16,
    "lowpass_biquads": 1, "lowpass_mults_per_sample": 5, "lowpass_adds_per_sample": 4,
}

ENVELOPE_MCU_COST = {
    "square_law": {
        "mults_per_sample": 1,
        "adds_per_sample": 0,
        "sqrt_per_sample": 0,
        "state_words": 0,
        "extra_latency_samples": 0,
        "note": (
            "Bình phương tín hiệu sau bandpass rồi lowpass để lấy đường bao. "
            "Đường bao mang đơn vị bình phương của tín hiệu."
        ),
    },
    "hybrid": {
        "mults_per_sample": 3,        # 1 chuẩn hóa Q + 2 hệ số alpha/beta
        "adds_per_sample": 2,         # 1 trừ (sai phân) + 1 cộng (alpha-max)
        "sqrt_per_sample": 0,
        "state_words": 3,
        "extra_latency_samples": 1,
        "note": (
            "Q[n]=(x[n]-x[n-2])*k là 1 trừ + 1 nhân hằng số; Alpha-Max là "
            "2 nhân + 1 cộng + 1 phép so sánh (max/min) + 2 phép lấy trị tuyệt "
            "đối (xóa bit dấu). Buffer 3 mẫu, trễ 1 mẫu, KHÔNG sqrt, KHÔNG FFT. "
            "Đây là luận điểm chi phí chính của đề tài: giữ được độ lệch "
            "pha 90 độ giống Hilbert với chi phí thấp."
        ),
    },
    "hilbert_offline": {
        "mults_per_sample": None,
        "adds_per_sample": None,
        "sqrt_per_sample": None,
        "state_words": None,
        "extra_latency_samples": 0,
        "note": (
            "KHÔNG TRIỂN KHAI ĐƯỢC trên MCU realtime: scipy.signal.hilbert đòi "
            "FFT/IFFT N điểm trên toàn khối và filtfilt chạy ngược thời gian. "
            "Chỉ dùng làm tham chiếu offline — không đưa vào bảng RQ2/RQ3."
        ),
    },
}


def envelope_mcu_cost(method: str):
    """Tra chi phí triển khai của riêng khối giải điều chế (xem ENVELOPE_MCU_COST)."""
    method = ENVELOPE_METHOD_ALIASES.get(method, method)
    if method not in ENVELOPE_MCU_COST:
        raise ValueError(
            f"method='{method}' không hợp lệ. Chọn một trong "
            f"{tuple(ENVELOPE_MCU_COST)}."
        )
    return dict(ENVELOPE_MCU_COST[method])


# ===========================================================================
# BỘ LỌC CƠ SỞ
# ===========================================================================
def bandpass_filter(x, fs, low_hz, high_hz, order=BENCHMARK_BANDPASS_ORDER,
                    zero_phase=False):
    """Butterworth bandpass.

    zero_phase=False (MẶC ĐỊNH MỚI): lọc NHÂN QUẢ bằng sosfilt — đúng với
        những gì MCU thực thi được, có trễ pha thật.
    zero_phase=True: filtfilt (lọc 2 chiều, triệt tiêu pha). CHỈ dùng cho
        phân tích/vẽ hình offline. TUYỆT ĐỐI KHÔNG dùng trong bảng so sánh
        RQ2/RQ3 (xem lý do ở docstring đầu module).

    Hàm trả về bộ lọc nhân quả mặc định; zero_phase chỉ dành cho phân tích
    offline.
    """
    nyq = fs / 2.0
    wn = [low_hz / nyq, high_hz / nyq]
    if zero_phase:
        b, a = butter(order, wn, btype="bandpass")  # type: ignore
        return np.asarray(filtfilt(b, a, x))
    sos = butter(order, wn, btype="bandpass", output="sos")
    return np.asarray(sosfilt(sos, x))


def lowpass_filter(x, fs, cutoff_hz, order=BENCHMARK_LOWPASS_ORDER,
                   zero_phase=False):
    """Butterworth lowpass. Xem ghi chú zero_phase ở bandpass_filter()."""
    nyq = fs / 2.0
    # Từ chối cutoff không hợp lệ để tránh tạo bộ lọc gần như thông suốt.
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError(
            f"lowpass_filter: cutoff_hz phải là số dương hữu hạn, nhận {cutoff_hz!r}."
        )
    if cutoff_hz >= nyq:
        raise ValueError(
            f"lowpass_filter: cutoff_hz={cutoff_hz} Hz >= Nyquist={nyq} Hz "
            f"(fs={fs} Hz). Hãy giảm cutoff hoặc tăng tần số lấy mẫu."
        )
    wn = np.clip(cutoff_hz / nyq, 1e-4, 0.99)
    if zero_phase:
        b, a = butter(order, wn, btype="lowpass")  # type: ignore
        return np.asarray(filtfilt(b, a, x))
    sos = butter(order, wn, btype="lowpass", output="sos")
    return np.asarray(sosfilt(sos, x))


def dc_blocker(x, pole=0.995):
    """Khử DC bằng bộ lọc nhân quả bậc một, không dùng trung bình toàn file."""
    x = np.asarray(x, dtype=np.float64)
    pole = float(pole)
    if not np.isfinite(pole) or not 0.0 <= pole < 1.0:
        raise ValueError(f"pole phải hữu hạn và thuộc [0, 1), nhận {pole!r}.")
    if x.ndim != 1:
        raise ValueError(f"dc_blocker yêu cầu tín hiệu 1D, nhận shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("dc_blocker không chấp nhận NaN hoặc Inf.")
    return np.asarray(lfilter([1.0, -1.0], [1.0, -pole], x), dtype=np.float64)


def _finish_envelope(env_raw, fs, lp_cutoff, lowpass_order, remove_dc,
                     dc_blocker_pole):
    env = lowpass_filter(env_raw, fs, lp_cutoff, order=lowpass_order)
    if remove_dc:
        env = dc_blocker(env, pole=dc_blocker_pole)
    return np.asarray(env, dtype=np.float64)


# ===========================================================================
# 1. SQUARE-LAW (baseline)
# ===========================================================================
def square_law_envelope(x, fs, band=None, lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                        bandpass_order=BENCHMARK_BANDPASS_ORDER,
                        lowpass_order=BENCHMARK_LOWPASS_ORDER,
                        remove_dc=True, order=None, dc_blocker_pole=0.995):
    """Đường bao Square-Law: bình phương tín hiệu đã bandpass rồi lowpass."""
    if order is not None:
        bandpass_order = order
    band = _resolve_band(band)

    x_bp = bandpass_filter(x, fs, band[0], band[1], order=bandpass_order,
                           zero_phase=False)
    x_squared = x_bp ** 2
    return _finish_envelope(
        x_squared, fs, lp_cutoff, lowpass_order, remove_dc, dc_blocker_pole
    )


# ===========================================================================
# 2. HILBERT
# ===========================================================================
def hilbert_envelope(x, fs, band=None, bandpass_order=BENCHMARK_BANDPASS_ORDER,
                     lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                     lowpass_order=BENCHMARK_LOWPASS_ORDER,
                     remove_dc=True, zero_phase=False, dc_blocker_pole=0.995,
                     align_delay=True):
    """[CHỈ THAM CHIẾU OFFLINE — KHÔNG ĐƯA VÀO BẢNG RQ2/RQ3]

    scipy.signal.hilbert() là phép biến đổi theo khối, phi nhân quả, chỉ dùng
    làm tham chiếu offline. Mặc định BPF/LPF đều causal sosfilt, cùng cấu hình
    với Hybrid. Đường bao được trễ một mẫu để cân bằng trễ I/Q của Hybrid.
    zero_phase=True chỉ dành cho khảo sát offline riêng, không dùng để so sánh
    trực tiếp trong pipeline chung.
    """
    band = _resolve_band(band)
    filtered = bandpass_filter(x, fs, band[0], band[1], order=bandpass_order,
                               zero_phase=zero_phase)
    env = np.abs(np.asarray(hilbert(filtered)))
    if align_delay and len(env):
        aligned = np.zeros_like(env)
        aligned[1:] = env[:-1]
        env = aligned
    if zero_phase:
        env = lowpass_filter(env, fs, lp_cutoff, order=lowpass_order,
                             zero_phase=True)
        if remove_dc:
            env = dc_blocker(env, pole=dc_blocker_pole)
        return np.asarray(env, dtype=np.float64)
    return _finish_envelope(
        env, fs, lp_cutoff, lowpass_order, remove_dc, dc_blocker_pole
    )


# ===========================================================================
# 3. HYBRID (đóng góp của đề tài)
# ===========================================================================
def hybrid_envelope(x, fs, band=None, bandpass_order=BENCHMARK_BANDPASS_ORDER,
                    alpha=ALPHA_MAX_MINMAX, beta=BETA_MIN_MINMAX,
                    lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                    lowpass_order=BENCHMARK_LOWPASS_ORDER,
                    remove_dc=True, dc_blocker_pole=0.995,
                    q_mode=HYBRID_Q_MODE_DEFAULT, fc_hz=None):
    """Hybrid Envelope Demodulation — không FFT, không sqrt, luôn ổn định.
    
        I[n] = x_bp[n-1]
        Q[n] = (x_bp[n] - x_bp[n-2]) / (2*sin(w_c))

    Vì (1 - z^{-2}) = z^{-1} * (z - z^{-1}), ta có

        Q(w)/I(w) = j * sin(w) / sin(w_c)

    -> tỉ số THUẦN ẢO: lệch pha CHÍNH XÁC 90° tại MỌI tần số (không chỉ
       tại w_c), và |Q/I| = 1 đúng tại w = w_c.
    -> Đây là biến thể nhân quả của công thức phi nhân quả
       Q[n] = (x[n+1] - x[n-1]) / (2*sin(w_c)): giống hệt nhau, chỉ trễ
       thêm 1 mẫu cho CẢ HAI nhánh, nên triển khai được trên MCU (chi phí:
       1 phép trừ + 1 phép nhân hằng số + buffer 3 mẫu).
    -> BẮT BUỘC trễ I đúng 1 mẫu. Nếu để I[n] = x_bp[n], tỉ số trở thành
       e^{-jw} * j*sin(w)/sin(w_c), tức sai số pha là w TRỌN (không phải
    w/2): trên dải 2300-3800 Hz sai số pha đạt khoảng 60° ... 120°.

    ---------------------------------------------------------------------
    GIỚI HẠN CÒN LẠI — PHẢI ghi rõ trong báo cáo
    ---------------------------------------------------------------------
    Biên độ nhánh Q tỉ lệ sin(w)/sin(w_c): bằng 1 đúng tại w_c nhưng
    nghiêng dần về hai mép dải cộng hưởng (đáp ứng của bộ VI PHÂN, không
    phải Hilbert thật). Dải càng rộng, độ nghiêng càng lớn -> đường bao bị
    điều biên nhẹ theo tần số tức thời. Vì vậy:
      - KHÔNG được mô tả phương pháp này là "Hilbert" trong báo cáo;
      - kiến trúc IIR All-Pass trực giao trong đề cương vẫn là ĐÍCH CUỐI
        (pha ~90° VÀ biên độ phẳng trên toàn dải). Sai phân trung tâm ở
        đây là mốc trung gian đúng về pha, dùng để chạy tiếp RQ2/RQ3 ngay.

    Alpha-Max Plus Beta-Min: |I + jQ| ~= alpha*max(|I|,|Q|) + beta*min(...)
    Mặc định dùng cặp (ALPHA_MAX_MINMAX, BETA_MIN_MINMAX) = (0.9604, 0.3978)
    -> sai số biên độ tối đa 3.96% so với sqrt, đúng với cam kết "< ~4%".
    Mặc định dùng cặp tối ưu sai số tối đa ALPHA_MAX_MINMAX và
    BETA_MIN_MINMAX; cặp tối ưu MSE được giữ để đối chiếu.
    """
    band = _resolve_band(band)
    x = np.asarray(x)
    if len(x) < 3: # len(x) < 3 -> không đủ mẫu cho sai phân trung tâm
        raise ValueError(
            f"hybrid_envelope cần >= 3 mẫu cho sai phân trung tâm (nhận {len(x)})."
        )

    sin_wc = hybrid_sin_wc(fs, band=band, q_mode=q_mode, fc_hz=fc_hz)
    if abs(sin_wc) < 1e-8:  # phòng chia 0 khi fc gần DC hoặc Nyquist
        sin_wc = 1e-8

    # BƯỚC 1: Bandpass nhân quả (cùng bậc với 2 phương pháp còn lại)
    x_bp = bandpass_filter(x, fs, band[0], band[1], order=bandpass_order,
                           zero_phase=False).astype(np.float64)

    # BƯỚC 2: I/Q bằng sai phân TRUNG TÂM dạng nhân quả (trễ 1 mẫu cả 2 nhánh)
    n_samples = len(x_bp)
    i_ch = np.zeros(n_samples, dtype=np.float64)
    q_ch = np.zeros(n_samples, dtype=np.float64)
    i_ch[1:] = x_bp[:-1]                                    # I[n] = x_bp[n-1]
    q_ch[2:] = (x_bp[2:] - x_bp[:-2]) / (2.0 * sin_wc)      # Q[n] = ...

    # BƯỚC 3: Alpha-Max Plus Beta-Min (không sqrt)
    abs_i, abs_q = np.abs(i_ch), np.abs(q_ch)
    env_raw = alpha * np.maximum(abs_i, abs_q) + beta * np.minimum(abs_i, abs_q)

    # BƯỚC 4: LPF và DC blocker chung với nhánh FFT Hilbert.
    return _finish_envelope(
        env_raw, fs, lp_cutoff, lowpass_order, remove_dc, dc_blocker_pole
    )


# ===========================================================================
# DISPATCHER DUY NHẤT cho RQ2/RQ3
# ===========================================================================
BENCHMARK_ENVELOPE_METHODS = ("square_law", "hybrid", "hilbert_offline")
ENVELOPE_METHOD_ALIASES = {
    "squarelaw": "square_law",
    "square-law": "square_law",
}

ENVELOPE_GROUP_DELAY_SAMPLES = {
    "square_law": 0,
    "hybrid": 1,                                          # sai phân trung tâm
    "hilbert_offline": 1,                                 # trễ nhân tạo để căn với Hybrid
}


def envelope_group_delay(method: str):
    """Trễ nhóm (mẫu) CỦA RIÊNG khối tạo tín hiệu trực giao.

    Đây là trục CHI PHÍ (cạnh MACC, SRAM trong ENVELOPE_MCU_COST), không
    phải trễ tổng của pipeline. KHÔNG dùng số này để dịch mẫu/căn chỉnh khi
    so sánh với tham chiếu zero-phase — dùng total_causal_delay_samples()
    hoặc measure_causal_delay_samples() cho việc đó (xem cảnh báo phía trên
    ENVELOPE_GROUP_DELAY_SAMPLES).
    """
    method = ENVELOPE_METHOD_ALIASES.get(method, method)
    if method not in ENVELOPE_GROUP_DELAY_SAMPLES:
        raise ValueError(f"method='{method}' không hợp lệ.")
    return ENVELOPE_GROUP_DELAY_SAMPLES[method]


def _iir_group_delay_samples(sos, freq_hz, fs):
    """Trễ nhóm (mẫu) của một bộ lọc IIR (dạng sos) tại MỘT tần số, dùng
    scipy.signal.group_delay. freq_hz=0 được kẹp lên 1e-3 Hz vì group_delay
    có thể không xác định đúng tại DC tuyệt đối cho một số bộ lọc.

    scipy chưa kèm type-stub cho group_delay(); Pylance suy luận kiểu của
    `w` là `int` từ giá trị mặc định 512, dù API thật (xem docstring scipy)
    nhận cả int (số điểm lưới) lẫn mảng tần số cụ thể. Truyền [f] để lấy
    ĐÚNG một tần số (không xấp xỉ qua lưới) là dùng đúng API, nên chỉ cần
    bỏ qua cảnh báo kiểu tĩnh ở dòng này, không đổi logic.
    """
    b, a = sos2tf(sos)
    f = max(float(freq_hz), 1e-3)
    _, gd = group_delay((b, a), w=[f], fs=fs)  # pyright: ignore[reportArgumentType]
    return float(gd[0])


def total_causal_delay_samples(method: str, fs, band=None,
                               lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                               bandpass_order=BENCHMARK_BANDPASS_ORDER,
                               lowpass_order=BENCHMARK_LOWPASS_ORDER,
                               mod_freq_hz=0.0):
    """Trễ nhóm TỔNG (mẫu) của toàn bộ pipeline envelope nhân quả: BPF +
    khối trực giao + LPF envelope. Dùng số này để
    CĂN CHỈNH (dịch mẫu) khi so đường bao nhân quả với tham chiếu zero-phase
    (hilbert_offline) — đúng việc mà envelope_group_delay() bị cấm làm.

    Cách tính:
      - Trễ BPF lấy tại tần số trung tâm dải cộng hưởng fc=(band[0]+band[1])/2,
        vì năng lượng sóng mang tập trung quanh đó.
            - Trễ khối trực giao lấy đúng số nguyên mẫu (0 hoặc 1), vì với Hybrid
                (sai phân trung tâm), trễ này là HẰNG SỐ tuyệt đối.
      - Trễ LPF lấy tại `mod_freq_hz` (mặc định gần 0 Hz), vì đó là dải tần
        chứa các đặc trưng bao hình (BPFO/BPFI/BSF/f_rot). Nếu biết trước
        tần số lỗi mục tiêu, hãy truyền đúng giá trị đó để có số chính xác
        hơn cho trường hợp cụ thể.

    GIỚI HẠN: BPF và LPF là Butterworth IIR nên trễ nhóm PHỤ THUỘC TẦN SỐ
    (ví dụ BPF dao động ~6-10 mẫu trong dải thông, LPF ~3.5-4.3 mẫu quanh
    100-500 Hz). Một số nguyên duy nhất không bù được hoàn toàn cho mọi
    thành phần tần số — phần méo pha dư (dispersion) vẫn còn sau khi dịch.
    Với số đo không phụ thuộc mô hình lọc, dùng
    measure_causal_delay_samples() đo trực tiếp trên dữ liệu thật.
    """
    method = ENVELOPE_METHOD_ALIASES.get(method, method)
    band = _resolve_band(band)
    fc = (band[0] + band[1]) / 2.0
    nyq = fs / 2.0

    sos_bp = butter(bandpass_order, [band[0] / nyq, band[1] / nyq],
                    btype="bandpass", output="sos")
    bp_delay = _iir_group_delay_samples(sos_bp, fc, fs)

    if method == "square_law":
        block_delay = 0.0
    elif method in ("hybrid", "hilbert_offline"):
        block_delay = 1.0
    else:
        raise ValueError(
            f"method='{method}' không hợp lệ. Chọn một trong "
            f"{BENCHMARK_ENVELOPE_METHODS} hoặc 'hilbert_offline'."
        )

    sos_lp = butter(lowpass_order, lp_cutoff / nyq, btype="lowpass", output="sos")
    lp_delay = _iir_group_delay_samples(sos_lp, mod_freq_hz, fs)

    return bp_delay + block_delay + lp_delay


def measure_causal_delay_samples(x, fs, method: str, band=None,
                                 lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                                 bandpass_order=BENCHMARK_BANDPASS_ORDER,
                                 lowpass_order=BENCHMARK_LOWPASS_ORDER,
                                 warmup_samples=2048, max_lag_samples=100):
    """Đo trễ THỰC (mẫu) bằng cross-correlation giữa đường bao nhân quả của
    `method` và tham chiếu zero-phase (hilbert_envelope, filtfilt) trên
    CHÍNH tín hiệu x. Không phụ thuộc mô hình lọc — khuyến nghị dùng số này
    (thay vì total_causal_delay_samples) cho hình vẽ/bảng đưa vào báo cáo,
    vì trễ nhóm IIR không phải hằng số theo tần số còn cross-correlation đo
    trực tiếp trên phổ thật của tín hiệu.

    Quét lag từ 0..max_lag_samples, chọn lag cho tương quan Pearson lớn
    nhất giữa đường bao nhân quả (dịch sớm lên `lag` mẫu) và tham chiếu.
    Cả hai nhánh dùng remove_dc=False (Pearson bất biến với DC, không ảnh
    hưởng).

    Trả về (best_lag_samples: int, corr_tai_best_lag: float).
    """
    band = _resolve_band(band)
    x = np.asarray(x, dtype=np.float64)

    n_min = warmup_samples + max_lag_samples + 100
    if len(x) < n_min:
        raise ValueError(
            f"Tín hiệu quá ngắn ({len(x)} mẫu) để đo trễ đáng tin cậy "
            f"(cần >= {n_min} mẫu). Dùng đoạn dài hơn hoặc giảm max_lag_samples."
        )

    env_causal = envelope_by_method(
        x, fs, band=band, method=method, lp_cutoff=lp_cutoff,
        bandpass_order=bandpass_order, lowpass_order=lowpass_order,
        remove_dc=False)
    env_ref = hilbert_envelope(
        x, fs, band=band, bandpass_order=bandpass_order, lp_cutoff=lp_cutoff,
        lowpass_order=lowpass_order, remove_dc=False, zero_phase=False)

    a = env_causal[warmup_samples:]
    b = env_ref[warmup_samples:]
    a = a - a.mean()
    b = b - b.mean()
    n = min(len(a), len(b)) - max_lag_samples
    if n <= 10:
        raise ValueError(
            "Đoạn tín hiệu sau warmup_samples quá ngắn so với max_lag_samples."
        )

    best_lag, best_corr = 0, -2.0
    for lag in range(0, max_lag_samples + 1):
        corr = float(np.corrcoef(a[lag:lag + n], b[:n])[0, 1])
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr


def decimate_envelope(env, fs, factor=ENV_DECIM, lp_cutoff=BENCHMARK_LP_CUTOFF_HZ):
    """Giảm mẫu envelope bằng cách lấy 1 trên `factor` mẫu.

    KHÔNG cần bộ lọc chống chồng phổ bổ sung: envelope đã qua lowpass
    lp_cutoff, chỉ cần lp_cutoff < fs/(2*factor) là an toàn — hàm kiểm tra
    điều kiện này và raise nếu vi phạm. Lấy mẫu thưa trực tiếp cũng đúng với
    cách MCU hạ tốc độ dòng envelope: không thêm bộ lọc, không thêm trễ.

    ĐỘ DÀI được cắt về bội số của factor TRƯỚC khi lấy mẫu thưa, để số mẫu
    ra đúng bằng len(env) // factor với MỌI độ dài đầu vào. Nếu để nguyên
    env[::factor] (= ceil) thì với một số độ dài lẻ, nhánh envelope có thêm
    đúng một cửa sổ so với nhánh raw và hai tập khóa (file_id, window_idx)
    lệch nhau.

    Trả về (env_decimated, fs_decimated).
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor phải >= 1, nhận {factor}.")
    fs_out = float(fs) / factor
    if lp_cutoff is not None and lp_cutoff >= fs_out / 2.0:
        raise ValueError(
            f"lp_cutoff={lp_cutoff} Hz >= Nyquist sau giảm mẫu ({fs_out / 2:.1f} "
            f"Hz): sẽ chồng phổ. Giảm lp_cutoff hoặc giảm factor."
        )
    env = np.asarray(env)
    n = len(env) - len(env) % factor
    return env[:n:factor], fs_out


def envelope_by_method(x, fs, band=None, method="hybrid",
                       lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                       bandpass_order=BENCHMARK_BANDPASS_ORDER,
                       lowpass_order=BENCHMARK_LOWPASS_ORDER,
                       **kwargs):
    """Chọn Hybrid hoặc tham chiếu FFT Hilbert với cùng BPF/LPF/DC blocker."""
    method = ENVELOPE_METHOD_ALIASES.get(method, method)
    band = _resolve_band(band)

    if method == "square_law":
        return square_law_envelope(
            x, fs, band=band, lp_cutoff=lp_cutoff,
            bandpass_order=bandpass_order, lowpass_order=lowpass_order, **kwargs)
    if method == "hybrid":
        return hybrid_envelope(
            x, fs, band=band, lp_cutoff=lp_cutoff,
            bandpass_order=bandpass_order, lowpass_order=lowpass_order, **kwargs)
    if method == "hilbert_offline":
        # Dùng cùng cấu hình lọc với các nhánh còn lại.
        return hilbert_envelope(
            x, fs, band=band, bandpass_order=bandpass_order,
            lp_cutoff=lp_cutoff, lowpass_order=lowpass_order, **kwargs)

    raise ValueError(
        f"method='{method}' không hợp lệ. Chọn một trong {BENCHMARK_ENVELOPE_METHODS}."
    )


# ===========================================================================
# FFT
# ===========================================================================
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
    n_samples = len(x)

    window_gain = 1.0
    if apply_window:
        win = np.hanning(n_samples)
        window_gain = win.mean() if win.mean() > 0 else 1.0
        x = x * win

    # Dùng rfft (Real FFT) tối ưu hơn cho tín hiệu thực so với fft thông thường
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    fft_values = np.fft.rfft(x)

    # Tính biên độ (Magnitude) và chuẩn hóa năng lượng theo N
    mag = np.abs(fft_values) * 2.0 / n_samples

    # Thành phần DC (0 Hz) và Nyquist (nếu N chẵn) không nhân 2
    mag[0] /= 2.0
    if n_samples % 2 == 0:
        mag[-1] /= 2.0

    if apply_window:
        mag = mag / window_gain

    return freqs, mag