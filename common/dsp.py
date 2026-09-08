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

Cấu hình CŨ (đã sửa) so sánh khập khiễng:
  - hilbert_envelope() lọc bằng filtfilt (zero-phase, phi nhân quả) VÀ dùng
    scipy.signal.hilbert() (biến đổi theo KHỐI dựa trên FFT).
  - square_law_envelope() / hybrid_envelope() lọc bằng lfilter/sosfilt
    (nhân quả, có trễ pha thật).

=> Baseline Hilbert được lợi BA lần, không chỉ về pha:
     (1) filtfilt không trễ pha — điều KHÔNG tồn tại trên MCU realtime;
     (2) filtfilt lọc 2 lượt nên đáp ứng biên độ là |H|^2: Butterworth
         order=4 trở thành order=8 HIỆU DỤNG -> dải chặn sắc hơn hẳn, tức
         baseline được lợi cả về ĐỘ CHỌN LỌC chứ không chỉ pha;
     (3) scipy.signal.hilbert() cần toàn bộ khối tín hiệu (FFT N điểm), nên
         không phải một bộ lọc chạy theo từng mẫu và không phản ánh chi phí
         thật trên MCU.
   Mọi kết luận kiểu "Hybrid/Square-Law nhiễu hơn Hilbert" rút ra từ cấu
   hình cũ đều KHÔNG có giá trị so sánh.

QUY ƯỚC MỚI:
  - Mặc định mọi bộ lọc trong module này là NHÂN QUẢ (sosfilt/lfilter).
    zero_phase=True chỉ dành cho minh họa/vẽ phổ offline.
  - Baseline Hilbert dùng cho benchmark là hilbert_envelope_fir(): FIR
    Hilbert transformer NHÂN QUẢ, độ trễ nhóm cố định (numtaps-1)/2 mẫu,
    triển khai được trên MCU và PHẢI được tính chi phí (numtaps MAC/mẫu +
    numtaps buffer) vào bảng RQ2.
  - hilbert_envelope() (scipy, phi nhân quả) giữ lại CHỈ để tham chiếu
    offline / vẽ hình — KHÔNG đưa vào bảng RQ2/RQ3.
  - Ba phương pháp benchmark dùng CHUNG: bandpass_order, lp_cutoff,
    lowpass_order và cách khử DC. Xem các hằng BENCHMARK_* và
    envelope_by_method() — dispatcher DUY NHẤT, để không tồn tại 2 cấu
    hình "công bằng" khác nhau ở 2 module.
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, lfilter, sosfilt

from .config import ENV_DECIM, LP_CUTOFF_HZ, RESONANCE_BAND_HZ

# ---------------------------------------------------------------------------
# Cấu hình dùng CHUNG cho cả 3 phương pháp benchmark (RQ2/RQ3).
# Đổi ở ĐÂY, không hardcode rải rác trong từng hàm/notebook.
# ---------------------------------------------------------------------------
BENCHMARK_BANDPASS_ORDER = 4
BENCHMARK_LOWPASS_ORDER = 2
BENCHMARK_LP_CUTOFF_HZ = float(LP_CUTOFF_HZ)  # nguồn duy nhất: config.LP_CUTOFF_HZ
BENCHMARK_HILBERT_NUMTAPS = 65  # lẻ -> trễ nhóm (65-1)/2 = 32 mẫu (nguyên)
BENCHMARK_BAND_HZ = tuple(float(v) for v in RESONANCE_BAND_HZ)  # config.RESONANCE_BAND_HZ

# ---------------------------------------------------------------------------
# HỆ SỐ ALPHA-MAX PLUS BETA-MIN
# ---------------------------------------------------------------------------
# Xấp xỉ |I + jQ| ~= alpha*max(|I|,|Q|) + beta*min(|I|,|Q|), tránh sqrt.
#
# CÓ HAI cặp hệ số "chuẩn" và chúng tối ưu HAI tiêu chí KHÁC NHAU — bản cũ
# dùng cặp MSE nhưng docstring lại hứa "sai số biên độ < ~4%", là con số
# của cặp minmax. Đo lại bằng số trên toàn cung |I|>=|Q|:
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
BETA_MIN_MSE = 0.392485603          # (để lại để tái lập số liệu bản cũ)


def _resolve_band(band):
    """band=None -> lấy dải cộng hưởng đã chốt ở config (mục 1.2).

    Để mọi hàm envelope có CÙNG một dải mặc định, thay vì mỗi notebook tự
    truyền một dải rồi so sánh 3 phương pháp trên 3 dải khác nhau.
    """
    if band is None:
        return BENCHMARK_BAND_HZ
    if len(band) != 2 or float(band[0]) >= float(band[1]):
        raise ValueError(f"band phải là (low_hz, high_hz) với low < high, nhận {band!r}.")
    return band


# ---------------------------------------------------------------------------
# CHI PHÍ TRÊN MCU của RIÊNG khối giải điều chế (RQ2/RQ3, mục 1.2 và 3.1)
# ---------------------------------------------------------------------------
# CHỮ "RIÊNG" LÀ QUAN TRỌNG: cả ba phương pháp dùng CHUNG bandpass
# Butterworth bậc 4 và lowpass bậc 2 (chi phí xem SHARED_STAGE_COST bên
# dưới). Hai khối đó triệt tiêu khi so sánh, nên
# bảng dưới chỉ đếm phần KHÁC NHAU. Cộng thêm SHARED_STAGE_COST nếu cần
# con số tuyệt đối cho toàn đường ống.
#
# Đây là ƯỚC LƯỢNG PHÂN TÍCH để định hướng, KHÔNG thay thế số đo thật
# bằng DWT->CYCCNT ở Giai đoạn 3 (xem mcu_export.latency_from_dwt_cycles).
# CẢNH BÁO ĐẾM BIQUAD (ĐÃ SỬA — bản cũ đếm THIẾU ĐÚNG 2 LẦN ở nhánh bandpass):
# butter(N, [f1, f2], btype="bandpass") KHÔNG cho bộ lọc bậc N. N là bậc của
# NGUYÊN MẪU lowpass; phép biến đổi lowpass -> bandpass nhân đôi số cực, nên
# bậc thực tế là 2N. Kiểm chứng bằng số:
#     butter(4, [2300, 3800], btype="bandpass", output="sos").shape == (4, 6)
#     butter(2, 750,          btype="lowpass",  output="sos").shape == (1, 6)
# Mỗi biquad dạng transposed direct-form II (a0 chuẩn hóa = 1) tốn 5 nhân +
# 4 cộng/mẫu, nên:
#     bandpass bậc 4 -> 4 biquad -> 20 nhân + 16 cộng/mẫu   (bản cũ: 2/10/8)
#     lowpass  bậc 2 -> 1 biquad ->  5 nhân +  4 cộng/mẫu   (bản cũ ĐÚNG)
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
            "Chỉ 1 phép nhân x*x. Rẻ nhất tuyệt đối. Đánh đổi: đường bao có "
            "đơn vị BÌNH PHƯƠNG của x (khác 2 phương pháp kia) và phép bình "
            "phương sinh thành phần tần số kép 2*fc phải dựa vào lowpass lọc "
            "bỏ. Bật take_sqrt=True thì phải cộng thêm 1 sqrt/mẫu."
        ),
    },
    "hilbert_fir": {
        "mults_per_sample": 16 + 2,   # 16 (FIR dùng đối xứng lẻ) + I^2, Q^2
        "adds_per_sample": 32 + 1,
        "sqrt_per_sample": 1,
        "state_words": BENCHMARK_HILBERT_NUMTAPS,
        "extra_latency_samples": (BENCHMARK_HILBERT_NUMTAPS - 1) // 2,
        "note": (
            "FIR Type III 65 tap. Ngây thơ là 65 MAC/mẫu, nhưng h[n]=0 với mọi "
            "n chẵn nên chỉ còn 32 tap khác 0; thêm tính phản đối xứng "
            "h[n] = -h[64-n] thì ghép cặp còn 16 nhân + 32 cộng/trừ. Vẫn phải "
            "giữ delay line 65 từ và chịu trễ nhóm 32 mẫu. Cộng sqrt(I^2+Q^2) "
            "— đúng phép toán mà Alpha-Max sinh ra để né."
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
            "Đây là luận điểm chi phí chính của đề tài: rẻ xấp xỉ Square-Law "
            "nhưng giữ được độ lệch pha 90 độ giống Hilbert."
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


def envelope_mcu_cost(method):
    """Tra chi phí triển khai của riêng khối giải điều chế (xem
    ENVELOPE_MCU_COST). Nhận cả alias 'hilbert' như envelope_by_method()."""
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

    Đã bỏ chuỗi ký tự lơ lửng giữa thân hàm ở bản cũ — nó nằm SAU câu lệnh
    nên không phải docstring, chỉ là một biểu thức string bị bỏ đi.
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
    # ĐÃ SỬA: bản cũ np.clip(..., 1e-4, 0.99) ÂM THẦM kẹp tần số cắt. Nếu gọi
    # lowpass_filter trên tín hiệu ĐÃ giảm mẫu (vd fs=1500 -> nyq=750) với
    # lp_cutoff=750 thì wn=1.0 bị kẹp về 0.99: bộ lọc gần như thông suốt,
    # không lỗi, không cảnh báo, đường bao ra sai hoàn toàn. Sau khi nâng
    # LP_CUTOFF_HZ 500 -> 750, biên an toàn đó hẹp đi, nên chuyển thành lỗi
    # tường minh thay vì kẹp im.
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError(
            f"lowpass_filter: cutoff_hz phải là số dương hữu hạn, nhận {cutoff_hz!r}."
        )
    if cutoff_hz >= nyq:
        raise ValueError(
            f"lowpass_filter: cutoff_hz={cutoff_hz} Hz >= Nyquist={nyq} Hz "
            f"(fs={fs} Hz). Bản cũ kẹp im về 0.99*Nyquist khiến bộ lọc gần "
            f"như thông suốt mà không báo gì."
        )
    wn = np.clip(cutoff_hz / nyq, 1e-4, 0.99)
    if zero_phase:
        b, a = butter(order, wn, btype="lowpass")  # type: ignore
        return np.asarray(filtfilt(b, a, x))
    sos = butter(order, wn, btype="lowpass", output="sos")
    return np.asarray(sosfilt(sos, x))


# ===========================================================================
# 1. SQUARE-LAW (baseline nhẹ)
# ===========================================================================
def square_law_envelope(x, fs, band=None, lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                        bandpass_order=BENCHMARK_BANDPASS_ORDER,
                        lowpass_order=BENCHMARK_LOWPASS_ORDER,
                        remove_dc=True, take_sqrt=False, order=None):
    """Đường bao bằng Square-Law (bình phương + lowpass). Nhân quả sẵn.

    Thay đổi so với bản cũ:
      - Tách bandpass_order và lowpass_order. Bản cũ dùng CHUNG order=4 cho
        cả bandpass VÀ lowpass, trong khi hybrid_envelope() dùng lowpass
        bậc 2 -> hai phương pháp bị so sánh với bộ lọc mượt khác bậc nhau.
        Tham số `order` cũ vẫn nhận được (alias cho bandpass_order) để
        không phá code gọi cũ.
      - lp_cutoff có giá trị mặc định dùng chung BENCHMARK_LP_CUTOFF_HZ.

    take_sqrt (mặc định False — GIỮ NGUYÊN số liệu đã có):
        Square-Law không lấy căn nên đường bao có ĐƠN VỊ BÌNH PHƯƠNG của x,
        khác Hilbert/Hybrid (đơn vị của x). Với RQ3, dải động khác nhau này
        ảnh hưởng trực tiếp tới lượng tử hóa INT8 (mục 2.3). Khi so sánh
        đầu vào AI giữa 3 phương pháp, nên bật take_sqrt=True để đồng nhất
        đơn vị, và ghi rõ trong báo cáo rằng phép sqrt này là chi phí phải
        trả (đúng thứ mà Alpha-Max sinh ra để né).
    """
    if order is not None:
        bandpass_order = order
    band = _resolve_band(band)

    x_bp = bandpass_filter(x, fs, band[0], band[1], order=bandpass_order,
                           zero_phase=False)

    x_squared = x_bp ** 2
    if take_sqrt:
        env = np.sqrt(np.maximum(
            lowpass_filter(x_squared, fs, lp_cutoff, order=lowpass_order), 0.0))
    else:
        # Khử DC TRƯỚC lowpass giữ đúng hành vi bản cũ.
        env = lowpass_filter(x_squared - np.mean(x_squared), fs, lp_cutoff,
                             order=lowpass_order)

    if remove_dc:
        env = env - np.mean(env)
    return np.asarray(env, dtype=np.float64)


# ===========================================================================
# 2. HILBERT
# ===========================================================================
def hilbert_envelope(x, fs, band=None, bandpass_order=BENCHMARK_BANDPASS_ORDER,
                     lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                     lowpass_order=BENCHMARK_LOWPASS_ORDER,
                     remove_dc=True, zero_phase=True):
    """[CHỈ THAM CHIẾU OFFLINE — KHÔNG ĐƯA VÀO BẢNG RQ2/RQ3]

    scipy.signal.hilbert() là biến đổi theo KHỐI dựa trên FFT (phi nhân
    quả): nó cần toàn bộ đoạn tín hiệu, nên không tồn tại phiên bản
    "chạy theo mẫu" tương ứng trên MCU. Dùng zero_phase=True ở bandpass để
    tái lập đúng đường bao lý tưởng kinh điển (mục đích duy nhất: vẽ hình
    và làm mốc "trần trên" cho chất lượng đường bao).

    Baseline Hilbert dùng để BENCHMARK là hilbert_envelope_fir().

    ---------------------------------------------------------------------
    ĐÃ SỬA: bản cũ KHÔNG lowpass và KHÔNG khử DC
    ---------------------------------------------------------------------
    Ba phương pháp kia đều kết thúc bằng lowpass + khử DC, riêng hàm này
    trả thẳng |analytic|. Đo trên cùng một tín hiệu thử:
        square_law               : mean = +7.7e-18, min = -0.605
        hilbert_fir              : mean = +6.2e-17, min = -0.997
        hybrid                   : mean = +3.0e-17, min = -0.968
        hilbert_offline (bản cũ) : mean = +1.000,   min = +0.239   <-- lệch
    Tức "mốc trần trên" nằm ở THANG KHÁC với chính thứ nó làm mốc: vẽ chồng
    lên nhau thì lệch trục, còn tính chỉ số so sánh (tương quan, SNR, dải
    động cho INT8 ở mục 2.3) thì ra số vô nghĩa. Nay dùng CHUNG lp_cutoff /
    lowpass_order / remove_dc với ba phương pháp còn lại.

    zero_phase=True giữ đúng bản sắc "offline lý tưởng": cả bandpass lẫn
    lowpass đều chạy filtfilt. Lưu ý filtfilt cho đáp ứng |H|^2 nên suy
    giảm gấp đôi (tính bằng dB) so với nhánh nhân quả — đó là chủ ý, vì hàm
    này là tham chiếu CHẤT LƯỢNG, không phải ứng viên triển khai.
    """
    band = _resolve_band(band)
    filtered = bandpass_filter(x, fs, band[0], band[1], order=bandpass_order,
                               zero_phase=zero_phase)
    env = np.abs(np.asarray(hilbert(filtered)))
    env = lowpass_filter(env, fs, lp_cutoff, order=lowpass_order,
                         zero_phase=zero_phase)
    if remove_dc:
        env = env - np.mean(env)
    return np.asarray(env, dtype=np.float64)


def design_fir_hilbert(numtaps=BENCHMARK_HILBERT_NUMTAPS):
    """Thiết kế FIR Hilbert transformer (Type III, phản đối xứng).

    Đáp ứng xung lý tưởng h[n] = 2/(pi*n) với n lẻ, 0 với n chẵn; cắt cửa
    sổ Hamming để giảm gợn dải thông.

    numtaps PHẢI LẺ để độ trễ nhóm (numtaps-1)/2 là SỐ NGUYÊN mẫu — nhờ đó
    việc căn chỉnh nhánh I chỉ cần một buffer dịch số nguyên trên MCU,
    không phải nội suy phân số mẫu.

    Chi phí MCU (phải ghi vào bảng RQ2): numtaps phép MAC/mẫu + buffer
    numtaps mẫu — đây chính là con số mà kiến trúc Hybrid nhắm tới việc
    cắt giảm.
    """
    if numtaps % 2 == 0:
        raise ValueError(
            f"numtaps phải LẺ để trễ nhóm là số nguyên mẫu (nhận {numtaps})."
        )
    M = (numtaps - 1) // 2
    n = np.arange(-M, M + 1)
    h = np.zeros(numtaps, dtype=np.float64)
    odd = (n % 2) != 0
    h[odd] = 2.0 / (np.pi * n[odd])
    return h * np.hamming(numtaps)


def hilbert_envelope_fir(x, fs, band=None, bandpass_order=BENCHMARK_BANDPASS_ORDER,
                         numtaps=BENCHMARK_HILBERT_NUMTAPS,
                         lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                         lowpass_order=BENCHMARK_LOWPASS_ORDER,
                         remove_dc=True):
    """[BASELINE HILBERT DÙNG CHO BENCHMARK — nhân quả, chạy được trên MCU]

    Pipeline:
      1. Bandpass Butterworth NHÂN QUẢ (cùng bậc với 2 phương pháp còn lại)
      2. Q = FIR Hilbert transformer nhân quả (trễ nhóm M = (numtaps-1)/2)
         I = x_bp trễ ĐÚNG M mẫu -> I/Q căn chỉnh, lệch pha ~90° trên toàn
         dải thông (khác hẳn xấp xỉ sai phân, xem hybrid_envelope)
      3. Biên độ THẬT: sqrt(I^2 + Q^2)  (baseline có sqrt — chi phí này là
         thứ Alpha-Max muốn loại bỏ)
      4. Lowpass + khử DC dùng CHUNG cấu hình với Square-Law/Hybrid

    Lưu ý về M mẫu đầu: mọi phương pháp nhân quả đều có đoạn transient đầu
    (bandpass IIR, FIR Hilbert, lowpass). Khi cắt cửa sổ nên bỏ vài trăm
    mẫu đầu file hoặc tính đường bao MỘT LẦN cho cả file rồi mới cắt (đúng
    cách features_full.build_full_feature_table đang làm).
    """
    band = _resolve_band(band)
    x_bp = bandpass_filter(x, fs, band[0], band[1], order=bandpass_order,
                           zero_phase=False)
    h = design_fir_hilbert(numtaps)
    M = (numtaps - 1) // 2

    q = np.asarray(lfilter(h, [1.0], x_bp), dtype=np.float64)

    i_delayed = np.zeros_like(x_bp, dtype=np.float64)
    if M < len(x_bp):
        i_delayed[M:] = x_bp[:len(x_bp) - M]

    env = np.sqrt(i_delayed ** 2 + q ** 2)
    env = lowpass_filter(env, fs, lp_cutoff, order=lowpass_order)
    if remove_dc:
        env = env - np.mean(env)
    return np.asarray(env, dtype=np.float64)


# ===========================================================================
# 3. HYBRID (đóng góp của đề tài)
# ===========================================================================
def hybrid_envelope(x, fs, band=None, bandpass_order=BENCHMARK_BANDPASS_ORDER,
                    alpha=ALPHA_MAX_MINMAX, beta=BETA_MIN_MINMAX,
                    lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                    lowpass_order=BENCHMARK_LOWPASS_ORDER,
                    remove_dc=True):
    """Hybrid Envelope Demodulation — không FFT, không sqrt, luôn ổn định.

    ---------------------------------------------------------------------
    LỖI TOÁN HỌC CỦA BẢN CŨ (đã sửa)
    ---------------------------------------------------------------------
    Bản cũ dùng sai phân LÙI bậc 1 và tự nhận là "xấp xỉ đúng lệch pha 90°
    tại tần số trung tâm w_c":

        I[n] = x_bp[n]
        Q[n] = (x_bp[n] - x_bp[n-1]) / (2*sin(w_c))

    Khẳng định đó SAI. Với H_Q(w) = (1 - e^{-jw}) / (2*sin(w_c)):

        1 - e^{-jw} = 2*sin(w/2) * e^{j*(90° - w/2)}

      -> lệch pha giữa Q và I là 90° - w/2 tại MỌI w, kể cả tại chính w_c.
         Sai số pha = w_c/2, KHÔNG triệt tiêu ở đâu cả.
         Ví dụ dải đã chốt 2300, 3800 Hz @ fs=12 kHz: fc=3050 Hz -> w_c=90°
         -> đo được lệch pha 45.00° tại fc (sai số 45.00°), 60.00°/30.00° ở
            hai mép dải (sai số 30.00°/60.00°), biên độ |Q|/|I| hụt còn
            0.500 ... 0.866.
      -> hệ số chuẩn hóa cũng sai: |1 - e^{-j*w_c}| = 2*sin(w_c/2), nhưng
         code chia cho 2*sin(w_c) -> biên độ Q bị nhân thêm
         sin(w_c/2)/sin(w_c) = 1/(2*cos(w_c/2)); với w_c=90° là 0.707,
         tức Q hụt ~29%.
    Hệ quả: luận điểm "không méo dạng, không trễ pha" của đề cương bị phá
    vỡ, và đường bao Hybrid bị méo dạng có hệ thống.

    ---------------------------------------------------------------------
    BẢN SỬA: sai phân TRUNG TÂM, viết ở dạng NHÂN QUẢ (trễ đúng 1 mẫu)
    ---------------------------------------------------------------------
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
       w/2 như bản cũ): đo trên dải 2300-3800 Hz được +30.00° ... -30.00°
       (sai số 60° ... 120°, đổi cả DẤU ở giữa dải), tệ hơn cả bản cũ. Xem notebook 06 (bảng 06_iq_phase_amplitude.csv) để chạy lại phép đo này.

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
    Bản cũ dùng cặp (0.947, 0.392) là cặp tối ưu MSE, sai số tối đa thực tế
    5.32% — tức docstring cũ hứa một đằng, hệ số làm một nẻo. Xem phần
    ALPHA_MAX_* ở đầu file để biết vì sao chọn tiêu chí sai số TỐI ĐA.
    """
    band = _resolve_band(band)
    x = np.asarray(x)
    if len(x) < 3:
        raise ValueError(
            f"hybrid_envelope cần >= 3 mẫu cho sai phân trung tâm (nhận {len(x)})."
        )

    fc = (band[0] + band[1]) / 2.0
    w_c = 2.0 * np.pi * fc / fs
    sin_wc = np.sin(w_c)
    if abs(sin_wc) < 1e-8:  # phòng chia 0 khi fc ~ 0 hoặc fc ~ fs/2
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

    # BƯỚC 4: Lowpass nhân quả — DÙNG CHUNG lp_cutoff với Square-Law/Hilbert.
    # Bản cũ tự đặt lp_hz = (band[1]-band[0])/2 (vd 1000 Hz) trong khi
    # Square-Law dùng LP_CUTOFF_HZ -> đường bao 2 phương pháp bị làm mượt khác
    # nhau, thêm một nguồn so sánh không công bằng nữa.
    env = lowpass_filter(env_raw, fs, lp_cutoff, order=lowpass_order)

    # BƯỚC 5: Khử DC
    if remove_dc:
        env = env - np.mean(env)
    return np.asarray(env, dtype=np.float64)


# ===========================================================================
# DISPATCHER DUY NHẤT cho RQ2/RQ3
# ===========================================================================
BENCHMARK_ENVELOPE_METHODS = ("square_law", "hilbert_fir", "hybrid")
# "squarelaw"/"square-law" là cách viết trong notebook giai đoạn 1 (tên file
# đầu ra features_mlp_squarelaw.parquet, cột dsp_method). Gom về tên chuẩn
# "square_law" để code cũ không chết vì ValueError khi đi qua
# envelope_by_method(), thay vì bắt mọi notebook đổi chuỗi.
ENVELOPE_METHOD_ALIASES = {
    "hilbert": "hilbert_fir",
    "squarelaw": "square_law",
    "square-law": "square_law",
}


# Trễ nhóm RIÊNG của khối tạo tín hiệu trực giao. KHÔNG tính chuỗi
# bandpass/lowpass vì cả ba phương pháp dùng chung nên phần đó triệt tiêu khi
# so sánh.
#
# KHÔNG dùng để bù trễ ở bất kỳ đâu trong đường dữ liệu: trễ nhóm là thuộc
# tính của phương pháp lấy đường bao, không phải sai số. Đường bao được giữ
# nguyên đúng như ra khỏi khối DSP. Bảng giá trị này chỉ để BÁO CÁO độ trễ
# như một trục chi phí (cạnh MACC và SRAM) và để ghi nhãn hình vẽ.
#
# Đặc trưng miền tần số không bị ảnh hưởng vì |FFT| bất biến với dịch
# thời gian, nên việc giữ nguyên trễ không làm lệch bảng đặc trưng.
ENVELOPE_GROUP_DELAY_SAMPLES = {
    "square_law": 0,
    "hilbert_fir": (BENCHMARK_HILBERT_NUMTAPS - 1) // 2,  # 32 mẫu = 2.67 ms
    "hybrid": 1,                                          # sai phân trung tâm
    "hilbert_offline": 0,                                 # filtfilt: zero-phase
}


def envelope_group_delay(method):
    """Trễ nhóm (mẫu) của khối trực giao, chấp nhận cả alias tên phương pháp."""
    method = ENVELOPE_METHOD_ALIASES.get(method, method)
    if method not in ENVELOPE_GROUP_DELAY_SAMPLES:
        raise ValueError(f"method='{method}' không hợp lệ.")
    return ENVELOPE_GROUP_DELAY_SAMPLES[method]


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
    fs_out = float(fs) / factor
    if lp_cutoff is not None and lp_cutoff >= fs_out / 2.0:
        raise ValueError(
            f"lp_cutoff={lp_cutoff} Hz >= Nyquist sau giảm mẫu ({fs_out / 2:.1f} "
            f"Hz): sẽ chồng phổ. Giảm lp_cutoff hoặc giảm factor."
        )
    env = np.asarray(env)
    n = len(env) - len(env) % factor
    return env[:n:factor], fs_out


def envelope_by_method(x, fs, band=None, method="square_law",
                       lp_cutoff=BENCHMARK_LP_CUTOFF_HZ,
                       bandpass_order=BENCHMARK_BANDPASS_ORDER,
                       lowpass_order=BENCHMARK_LOWPASS_ORDER,
                       take_sqrt=False, **kwargs):
    """Chọn phương pháp giải điều chế với CÙNG MỘT cấu hình lọc.

    method:
      - "square_law"      : Square-Law (nhân quả)
      - "hilbert_fir"     : FIR Hilbert nhân quả  <- baseline benchmark
      - "hybrid"          : Hybrid sai phân trung tâm + Alpha-Max
      - "hilbert_offline" : scipy.hilbert + filtfilt — CHỈ vẽ hình/tham
                            chiếu, không dùng cho bảng kết quả

    "hilbert" là ALIAS của "hilbert_fir": code/notebook cũ gọi
    method='hilbert' sẽ TỰ ĐỘNG chuyển sang baseline nhân quả, thay vì âm
    thầm tiếp tục so sánh với bản filtfilt phi nhân quả.

    ---------------------------------------------------------------------
    take_sqrt — ĐÃ NÂNG THÀNH THAM SỐ CHUNG (trước đây là lỗi CHẶN RQ3)
    ---------------------------------------------------------------------
    Bản cũ chỉ chuyển tiếp take_sqrt qua **kwargs nên nó CHỈ chạy được với
    square_law; gọi envelope_by_method(..., method="hilbert_fir",
    take_sqrt=True) ném thẳng TypeError. Trong khi đó chính docstring của
    square_law_envelope lại dặn "khi so sánh đầu vào AI giữa 3 phương pháp,
    nên bật take_sqrt=True". Tức là dispatcher không chạy nổi đúng thí
    nghiệm mà nó khuyến nghị.

    Ý nghĩa đúng của cờ này là ĐỒNG NHẤT THỨ NGUYÊN, và chỉ square_law mới
    lệch thứ nguyên:
        square_law           -> đường bao mang đơn vị x^2  (cần sqrt)
        hilbert_fir, hybrid  -> đã là đơn vị của x         (sqrt vô nghĩa)
        hilbert_offline      -> đã là đơn vị của x (|analytic|)
    Nên take_sqrt=True CHỈ tác động lên square_law và là KHÔNG-OP với ba
    nhánh còn lại. Cố ý không raise, để vòng lặp so sánh truyền cùng một bộ
    tham số cho cả ba phương pháp mà không phải rẽ nhánh if/else.

    Cách chạy đúng thí nghiệm RQ3 (cả 3 cùng đơn vị của x):
        for m in BENCHMARK_ENVELOPE_METHODS:
            env = envelope_by_method(x, fs, method=m, take_sqrt=True)
    """
    method = ENVELOPE_METHOD_ALIASES.get(method, method)
    band = _resolve_band(band)

    if method == "square_law":
        return square_law_envelope(
            x, fs, band=band, lp_cutoff=lp_cutoff,
            bandpass_order=bandpass_order, lowpass_order=lowpass_order,
            take_sqrt=take_sqrt, **kwargs)
    if method == "hilbert_fir":
        return hilbert_envelope_fir(
            x, fs, band=band, lp_cutoff=lp_cutoff,
            bandpass_order=bandpass_order, lowpass_order=lowpass_order, **kwargs)
    if method == "hybrid":
        return hybrid_envelope(
            x, fs, band=band, lp_cutoff=lp_cutoff,
            bandpass_order=bandpass_order, lowpass_order=lowpass_order, **kwargs)
    if method == "hilbert_offline":
        # ĐÃ SỬA: nhánh này trước đây nuốt im lp_cutoff/lowpass_order/kwargs,
        # nên tham chiếu offline chạy ở một thang hoàn toàn khác 3 nhánh kia.
        return hilbert_envelope(
            x, fs, band=band, bandpass_order=bandpass_order,
            lp_cutoff=lp_cutoff, lowpass_order=lowpass_order, **kwargs)

    raise ValueError(
        f"method='{method}' không hợp lệ. Chọn một trong "
        f"{BENCHMARK_ENVELOPE_METHODS} (benchmark) hoặc 'hilbert_offline' "
        f"(chỉ tham chiếu offline)."
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
