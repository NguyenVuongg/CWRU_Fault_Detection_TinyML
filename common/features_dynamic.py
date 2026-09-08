# -*- coding: utf-8 -*-
"""
common/features_dynamic.py
=========================
Trích xuất đặc trưng động dựa trên band_hz cố định và hỗ trợ
3 phương pháp giải điều chế: Square-Law, Hilbert, và Hybrid DSP.

Mọi phương pháp giải điều chế ở đây đều đi qua dsp.envelope_by_method()
— dispatcher DUY NHẤT của codebase — để cả 3 phương pháp dùng CHUNG một
cấu hình lọc nhân quả (bậc bandpass/lowpass, lp_cutoff, cách khử DC).
KHÔNG định nghĩa lại bản riêng ở module này.

HAI ĐIỂM ĐÃ SỬA (đọc trước khi trích dẫn kết quả RQ3):
  1. method='hilbert' trước đây gọi dsp.hilbert_envelope() — filtfilt
     (zero-phase, phi nhân quả) + scipy.signal.hilbert() (FFT theo khối).
     Đầu vào AI của nhánh Hilbert vì thế được lợi cả về pha LẪN độ chọn
     lọc (filtfilt lọc 2 lượt -> |H|^2), không phản ánh được thứ MCU chạy
     được. Nay 'hilbert' là alias của baseline NHÂN QUẢ 'hilbert_fir'.
  2. lp_cutoff trước đây hardcode 500 Hz cho square_law nhưng hybrid tự
     dùng (band[1]-band[0])/2 bên trong -> đường bao 2 phương pháp bị làm
     mượt ở 2 tần số cắt khác nhau. Nay dùng chung dsp.BENCHMARK_LP_CUTOFF_HZ.

LỖI THỨ BA ĐÃ SỬA — RESET TRẠNG THÁI LỌC THEO CỬA SỔ:
  extract_features_dynamic() (bản cũ) nhận vào các CỬA SỔ ĐÃ CẮT SẴN và
  gọi apply_envelope_method() trên từng cửa sổ nhỏ. Mọi bộ lọc trong
  đường ống đều NHÂN QUẢ (sosfilt/lfilter/FIR) nên mỗi lần gọi là một lần
  khởi tạo trạng thái về 0: đầu mỗi cửa sổ dính transient giả tạo —
  riêng FIR Hilbert 65 tap làm sai 32 mẫu đầu (bằng đúng trễ nhóm), còn
  bandpass IIR bậc 4 lê thêm vài chục mẫu. Đây đúng là lỗi mà
  features_full.build_full_feature_table() đã phải tránh bằng cách tính
  envelope MỘT LẦN cho cả file rồi mới cắt cửa sổ (xem docstring hàm đó).
    -> Hàm đúng là extract_features_dynamic_from_signals(): nhận TÍN HIỆU
       NGUYÊN chưa cắt, tự tính envelope một lần rồi cắt cửa sổ ĐỒNG BỘ
       cho cả tín hiệu thô lẫn envelope.
    -> extract_features_dynamic() (nhận cửa sổ sẵn) giữ lại ở trạng thái
       DEPRECATED, phát DeprecationWarning mỗi lần gọi. KHÔNG dùng nó cho
       kết quả RQ3 đưa vào báo cáo.

Ghi chú về Hybrid: bản cũ của dsp.hybrid_envelope() dùng sai phân LÙI bậc 1
và tự nhận "đúng 90° tại tần số trung tâm" — khẳng định đó sai (sai số pha
bằng w_c/2). Đã chuyển sang sai phân TRUNG TÂM dạng nhân quả; xem
docstring dsp.hybrid_envelope() để biết chứng minh và giới hạn còn lại.
"""

import warnings

import numpy as np
import pandas as pd

# Import TƯƠNG ĐỐI. Bản cũ viết "from common.config import ..." (tuyệt đối),
# chỉ chạy được khi thư mục cha của common/ nằm trên sys.path — tức file này
# gãy import ngay khi package được cài đặt thật hoặc đổi tên, trong khi 13
# file còn lại trong common/ đều dùng import tương đối.
from .config import (NOMINAL_RPM_BY_LOAD, RESONANCE_BAND_HZ,  # noqa: F401
                     WARMUP_SAMPLES)
from . import dsp, features_full
from .dsp import hybrid_envelope  # noqa: F401 — re-export để notebook 05 và nơi khác import từ đây vẫn hoạt động


# --- HÀM CHỌN PHƯƠNG PHÁP GIẢI ĐIỀU CHẾ ---
def apply_envelope_method(window_data, fs, band_hz, method='square_law',
                          lp_cutoff=dsp.BENCHMARK_LP_CUTOFF_HZ,
                          bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                          lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER, **kwargs):
    """Uỷ quyền cho dsp.envelope_by_method() — không tự định nghĩa cấu hình
    lọc riêng ở đây, để không tồn tại 2 "giao thức công bằng" lệch nhau.

    method: 'square_law' | 'hilbert_fir' (alias 'hilbert') | 'hybrid'
            | 'hilbert_offline' (chỉ tham chiếu, KHÔNG dùng cho bảng kết quả)

    LƯU Ý: dù tên tham số là window_data, hàm này nên được gọi trên TÍN
    HIỆU NGUYÊN (xem extract_features_dynamic_from_signals), không phải
    trên từng cửa sổ đã cắt — trừ khi cố ý nghiên cứu transient.
    """
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

    SỐ CHIỀU THỰC TẾ LÀ 28, không phải 32 như bản cũ ghi: order.py gộp các
    cột hài rơi trùng bin FFT lại với nhau (Nhóm B 12->10, Nhóm C 9->7).
    Xem features_full.EXPECTED_FEATURE_COUNTS để biết lý do.

    KHẮC PHỤC LỖI CỦA extract_features_dynamic() BẢN CŨ: envelope được
    tính MỘT LẦN trên toàn bộ tín hiệu rồi mới cắt thành cửa sổ — bộ lọc
    nhân quả giữ nguyên trạng thái xuyên suốt, không còn transient giả
    tạo ở đầu mỗi cửa sổ. Cửa sổ được cắt ĐỒNG BỘ (cùng start_idx) cho cả
    tín hiệu thô (Nhóm A/B) lẫn envelope (Nhóm C) — đúng cách
    features_full.build_full_feature_table() đang làm.

    Args:
        signals_series: iterable các mảng tín hiệu nguyên (mỗi phần tử =
            1 file / 1 đoạn ghi liên tục), CHƯA cắt cửa sổ.
        band_hz: dải bandpass cộng hưởng. Dùng đúng dải đã chốt ở mục 1.2:
            config.RESONANCE_BAND_HZ = (2300.0, 3800.0). Truyền dải khác
            chỉ khi đang CỐ Ý khảo sát độ nhạy theo dải, và khi đó phải
            đổi ĐỒNG THỜI cho cả 3 phương pháp, nếu không RQ2 mất hiệu lực.
        rpm, load_hp_series, rpm_series: RPM dùng cho tần số mục tiêu của
            từng tín hiệu, thứ tự ưu tiên:
              1. rpm_series — RPM THỰC của từng tín hiệu (vd lấy từ cột
                 rpm_from_file của manifest). KHUYẾN NGHỊ khi có.
              2. load_hp_series — mức tải của từng tín hiệu, tra
                 NOMINAL_RPM_BY_LOAD (RPM danh định).
              3. rpm — scalar dùng chung cho mọi tín hiệu.
        method: 'square_law' | 'hilbert_fir' (alias 'hilbert') | 'hybrid'
            — đi qua dsp.envelope_by_method() với cấu hình lọc chung.
        window_size, stride: kích thước cửa sổ và bước trượt (mẫu).
            Mặc định 2048/1024 (chồng 50%) — khớp build_full_feature_table
            và ngân sách realtime ở profiling.realtime_budget().
        warmup_samples: số mẫu đầu tín hiệu bị loại (quá độ khởi động cơ
            khí + xác lập bộ lọc IIR nhân quả). Mặc định None ->
            config.WARMUP_SAMPLES, ĐÚNG BẰNG ngưỡng của
            features_full.build_full_feature_table() để hai bảng có CÙNG
            tập cửa sổ. Truyền 0 để lấy lại hành vi cũ (không loại gì).
        envelope_kwargs: dict tham số bổ sung chuyển tiếp cho
            apply_envelope_method() -> dsp.envelope_by_method().
            BẮT BUỘC cho RQ3: envelope_kwargs={"take_sqrt": True} khi
            method="square_law", vì square_law_envelope() mặc định trả
            LP(x^2), tức ĐƠN VỊ BÌNH PHƯƠNG của x, còn hilbert_fir/hybrid
            trả |A| (đơn vị của x). Không đồng nhất đơn vị thì Nhóm C lệch
            cả thang đo lẫn DẢI ĐỘNG giữa 3 bảng, và dải động là đúng thứ
            quyết định sai số lượng tử hóa INT8 ở mục 2.3.

    Returns:
        pd.DataFrame, mỗi dòng = 1 cửa sổ. Ba cột METADATA đi TRƯỚC
        (signal_idx, window_idx, start_idx), rồi 28 cột đặc trưng (tiền tố
        time_/order_/envelope_). Cả ba đã được khai trong
        features_full.METADATA_COLUMNS.
          - signal_idx: chỉ số của tín hiệu trong signals_series. ĐÂY là
            khóa để ghép file_id/label/load_hp. TUYỆT ĐỐI không suy ra từ
            số dòng: tín hiệu quá ngắn bị bỏ GIỮA DÒNG nên số cửa sổ của
            mỗi tín hiệu không đều nhau.
          - Lấy X huấn luyện bằng features_full.feature_columns(df,
            expected_count=28), KHÔNG dùng df.values trực tiếp —
            window_idx/start_idx lọt vào X là rò rỉ vị trí.
    """
    signals_list = [np.asarray(s, dtype=np.float64) for s in signals_series]
    n_signals = len(signals_list)

    if rpm_series is not None:
        rpm_list = [float(v) for v in rpm_series]
        if len(rpm_list) != n_signals:
            raise ValueError(
                f"rpm_series (len={len(rpm_list)}) phải cùng độ dài với "
                f"signals_series (len={n_signals})"
            )
    elif load_hp_series is not None:
        load_list = list(load_hp_series)
        if len(load_list) != n_signals:
            raise ValueError(
                f"load_hp_series (len={len(load_list)}) phải cùng độ dài "
                f"với signals_series (len={n_signals})"
            )
        rpm_list = [float(NOMINAL_RPM_BY_LOAD.get(load, rpm)) for load in load_list]
    else:
        rpm_list = [float(rpm)] * n_signals

    warmup_samples = WARMUP_SAMPLES if warmup_samples is None else int(warmup_samples)
    envelope_kwargs = dict(envelope_kwargs or {})

    rows = []
    n_skipped = 0
    for signal_idx, (sig, current_rpm) in enumerate(zip(signals_list, rpm_list)):
        num_windows = (len(sig) - window_size) // stride + 1
        # (num_windows - 1) * stride = start_idx của cửa sổ CUỐI. Nếu ngay
        # cửa sổ cuối vẫn nằm trong vùng warmup thì tín hiệu này không còn
        # cửa sổ nào hợp lệ -> bỏ hẳn, thay vì trả cửa sổ dính transient.
        if num_windows <= 0 or (num_windows - 1) * stride < warmup_samples:
            n_skipped += 1
            continue

        # Envelope tính MỘT LẦN trên cả tín hiệu — bộ lọc nhân quả giữ
        # trạng thái xuyên suốt; transient chỉ còn ở ĐẦU FILE, và đó đúng
        # là phần bị warmup_samples loại bỏ ở vòng lặp dưới.
        env_full = apply_envelope_method(sig, fs, band_hz, method,
                                         **envelope_kwargs)

        for i in range(num_windows):
            s0 = i * stride
            # Bỏ cửa sổ chạm vùng quá độ đầu tín hiệu — CÙNG NGƯỠNG với
            # features_full.build_full_feature_table(). i GIỮ NGUYÊN chỉ số
            # gốc để hai bảng đối chiếu được theo window_idx.
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
            # Metadata đi TRƯỚC đặc trưng. Không có 3 cột này thì người gọi
            # không thể ghép file_id/label an toàn (xem mục Returns).
            rows.append({"signal_idx": signal_idx, "window_idx": i,
                         "start_idx": s0, **feats})

    if n_skipped:
        print(f"[CẢNH BÁO] {n_skipped}/{n_signals} tín hiệu không đủ cửa sổ "
              f"hợp lệ (ngắn hơn window_size={window_size} mẫu, hoặc mọi "
              f"cửa sổ đều nằm trong warmup_samples={warmup_samples}) -> bị "
              f"bỏ qua. Dùng cột 'signal_idx' để biết tín hiệu nào CÒN trong "
              f"bảng — số dòng mỗi tín hiệu KHÔNG đều nhau.")

    return pd.DataFrame(rows)


def extract_features_dynamic(windows_series, band_hz, fs=12000, rpm=1750,
                              load_hp_series=None, method='square_law'):
    """
    [DEPRECATED — xem "LỖI THỨ BA" ở docstring module. KHÔNG dùng cho kết
    quả RQ3 trong báo cáo.]

    Rút đặc trưng (28 chiều thực tế, xem features_full.EXPECTED_FEATURE_COUNTS)
    dựa trên band_hz cố định và phương pháp giải điều chế được chỉ định.

    VÌ SAO DEPRECATED: hàm này nhận các CỬA SỔ ĐÃ CẮT SẴN và lọc envelope
    trên TỪNG CỬA SỔ -> mọi bộ lọc nhân quả (bandpass/lowpass IIR, FIR
    Hilbert) bị reset trạng thái ở mỗi cửa sổ, gây transient giả tạo ở đầu
    mỗi cửa sổ (FIR Hilbert 65 tap làm sai 32 mẫu đầu). Hãy dùng
    extract_features_dynamic_from_signals() (nhận tín hiệu nguyên, tính
    envelope một lần rồi mới cắt cửa sổ) hoặc
    features_full.build_full_feature_table(envelope_method=...).
    Hàm giữ lại chỉ để notebook cũ không gãy import, và phát
    DeprecationWarning mỗi lần gọi.

    load_hp_series: (khuyến nghị LUÔN truyền khi dùng cho LOLO)
        pd.Series cùng chỉ số/độ dài với windows_series, chứa load_hp của
        TỪNG window. Khi có, RPM thực tế được tra theo NOMINAL_RPM_BY_LOAD
        cho từng window riêng lẻ, thay vì dùng chung 1 giá trị `rpm` mặc
        định — QUAN TRỌNG vì trong 1 fold LOLO, train_df/val_df thường
        chứa nhiều load khác nhau trộn lẫn.

        Nếu để None, dùng `rpm` scalar cho mọi window (hành vi cũ - chỉ
        nên dùng khi chắc chắn windows_series đồng nhất 1 load duy nhất).
    """
    warnings.warn(
        "extract_features_dynamic() lọc envelope trên TỪNG CỬA SỔ đã cắt "
        "sẵn -> reset trạng thái bộ lọc nhân quả ở mỗi cửa sổ (transient "
        "giả tạo; FIR Hilbert 65 tap làm sai 32 mẫu đầu mỗi cửa sổ). Hãy "
        "chuyển sang extract_features_dynamic_from_signals() hoặc "
        "features_full.build_full_feature_table(envelope_method=...). "
        "Không dùng hàm này cho kết quả RQ3 trong báo cáo.",
        DeprecationWarning,
        stacklevel=2,
    )

    features_list = []

    windows_list = list(windows_series)
    if load_hp_series is not None:
        load_hp_list = list(load_hp_series)
        if len(load_hp_list) != len(windows_list):
            raise ValueError(
                f"load_hp_series (len={len(load_hp_list)}) phải cùng độ dài "
                f"với windows_series (len={len(windows_list)})"
            )
    else:
        load_hp_list = None

    for i, window in enumerate(windows_list):
        w_raw = np.array(window, dtype=np.float32)

        # Tra RPM đúng theo load thực tế của window này, thay vì hardcode
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
