# -*- coding: utf-8 -*-
"""
common/features_full.py
=========================
"""

import numpy as np
from scipy.stats import kurtosis, skew

from pathlib import Path

from . import dsp
from .config import (bearing_fault_frequencies, make_class_label,
                     SKF_6205_GEOMETRY,
                     NOMINAL_RPM_BY_LOAD, SCOPE, WARMUP_SAMPLES)
from .order import extract_order_features


# ============================================================================
# ĐỊNH DANH FILE GỐC cho File-based Split / LOLO
# ============================================================================
def make_file_id(file_path) -> str:
    """Định danh FILE GỐC — KHÔNG được chứa chỉ số cửa sổ.

    Đây là khóa mà splitting.file_based_split() dùng để tách train/test và
    là thứ duy nhất ngăn rò rỉ dữ liệu giữa các cửa sổ trượt CHỒNG LẤN
    của cùng một file. Mọi thông tin "cửa sổ thứ mấy" phải nằm ở cột
    riêng (window_idx/start_idx), TUYỆT ĐỐI không ghép vào file_id.
    """
    return Path(file_path).stem


# ============================================================================
# NHÓM A - Miền thời gian (11 đặc trưng, số chiều CỐ ĐỊNH theo mục 1.4)
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
# NHÓM B - Phổ Order (số chiều = số tần số mục tiêu x số hài)
# ============================================================================
def extract_order_domain_features(x, fs, rpm, target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                                   harmonics=(1, 2, 3), search_width_hz=3.0,
                                   geometry=SKF_6205_GEOMETRY):
    """Wrapper: FFT tín hiệu thô rồi trích biên độ tại hài của tần số mục
    tiêu (dùng common/order.py). Xem docstring order.extract_order_features."""
    freqs, mag = dsp.compute_fft(x, fs)
    return extract_order_features(
        freqs, mag, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry, prefix="order",
    )


# ============================================================================
# NHÓM C - Phổ Envelope (Square-Law) - tập trung vào tần số ĐỘNG HỌC LỖI,
# không gồm f_rot (mất cân bằng trục không phải cơ chế điều biên bởi lỗi)
# ============================================================================
def extract_envelope_features_from_envelope(envelope, fs, rpm,
                                             target_names=("BPFO", "BPFI", "BSF"),
                                             harmonics=(1, 2, 3), search_width_hz=3.0,
                                             geometry=SKF_6205_GEOMETRY):
    """
    [BỔ SUNG] Tính đặc trưng Nhóm C từ 1 đoạn envelope ĐÃ TÍNH SẴN - không
    gọi lại dsp.square_law_envelope() bên trong hàm này.

    Dùng khi envelope được tính 1 LẦN cho toàn bộ file rồi cắt cửa sổ từ
    đó (xem build_full_feature_table) - tránh reset trạng thái bộ lọc IIR
    causal ở mỗi cửa sổ nhỏ, vốn gây méo biên độ/pha giả tạo (xem docstring
    đầu file).
    """
    envelope = np.asarray(envelope, dtype=np.float64)
    freqs, mag = dsp.compute_fft(envelope - envelope.mean(), fs)
    return extract_order_features(
        freqs, mag, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry, prefix="envelope",
    )


def extract_envelope_features(x, fs, rpm, band_hz,
                               lp_cutoff_hz=dsp.BENCHMARK_LP_CUTOFF_HZ,
                               bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                               lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER,
                               target_names=("BPFO", "BPFI", "BSF"),
                               harmonics=(1, 2, 3), search_width_hz=3.0,
                               geometry=SKF_6205_GEOMETRY,
                               *, method="square_law", envelope_kwargs=None):
    """Tính Nhóm C bằng cách TỰ tính envelope từ tín hiệu thô `x`.

    Dùng khi chưa có sẵn envelope (gọi độc lập / test nhanh). Khi envelope
    đã có sẵn (đường chính thức build_full_feature_table()), dùng thẳng
    extract_envelope_features_from_envelope() thay vì hàm này.

    ĐÃ SỬA: trước đây hàm này gọi CỐ ĐỊNH dsp.square_law_envelope(), không
    có cách nào chọn phương pháp khác dù tên tham số bandpass_order/
    lowpass_order gợi ý có thể cấu hình - đây là bẫy cho bất kỳ ai gọi hàm
    này trực tiếp để thử Hybrid/Hilbert-FIR. Nay đi qua
    dsp.envelope_by_method() - dispatcher DUY NHẤT - như mọi nơi khác
    trong codebase.

    method: 'square_law' (mặc định, GIỮ NGUYÊN hành vi cũ) | 'hilbert_fir'
        | 'hybrid'. Xem dsp.envelope_by_method() để biết các lựa chọn.
    envelope_kwargs: dict tham số bổ sung chuyển tiếp cho
        dsp.envelope_by_method(), vd {"take_sqrt": True} khi
        method="square_law" để đồng nhất đơn vị với hilbert_fir/hybrid
        (xem docstring build_full_feature_table về envelope_kwargs).

    bandpass_order/lowpass_order: ĐÃ ĐỔI default từ literal 4/2 sang
    dsp.BENCHMARK_BANDPASS_ORDER/dsp.BENCHMARK_LOWPASS_ORDER - cùng giá
    trị số nên KHÔNG đổi hành vi, chỉ tránh 2 số hardcode rời khỏi nguồn
    chân lý duy nhất ở dsp.py.
    """
    envelope = dsp.envelope_by_method(
        x, fs, band=band_hz, method=method, lp_cutoff=lp_cutoff_hz,
        bandpass_order=bandpass_order, lowpass_order=lowpass_order,
        **(envelope_kwargs or {}),
    )
    return extract_envelope_features_from_envelope(
        envelope, fs, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry,
    )


# ============================================================================
# GHÉP CẢ 3 NHÓM
# ============================================================================
def extract_full_feature_vector(x, fs, rpm, band_hz, envelope_window=None,
                                 order_target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                                 envelope_target_names=("BPFO", "BPFI", "BSF"),
                                 harmonics=(1, 2, 3), search_width_hz=3.0,
                                 geometry=SKF_6205_GEOMETRY,
                                 lp_cutoff_hz=dsp.BENCHMARK_LP_CUTOFF_HZ,
                                 bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                                 lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER,
                                 envelope_method="square_law", envelope_kwargs=None):
    """
    Trả về 1 dict gộp cả 3 nhóm - mỗi key có tiền tố rõ ràng (time_/order_/
    envelope_) để lọc theo nhóm khi phân tích Feature Importance (mục 2.2).

    Args:
        envelope_window: nếu được truyền vào (mảng envelope ĐÃ TÍNH SẴN,
            cùng độ dài/vị trí với `x`), Nhóm C dùng thẳng mảng này qua
            extract_envelope_features_from_envelope() thay vì tự tính lại
            từ `x` - đây là đường dùng trong build_full_feature_table()
            (envelope đã được tính đúng 1 lần cho cả file qua
            dsp.envelope_by_method(method=envelope_method) ở đó). Nếu để
            None (mặc định), Nhóm C tự tính envelope từ `x` theo
            `envelope_method`/`envelope_kwargs` bên dưới (phù hợp gọi độc
            lập / test nhanh 1 cửa sổ).

        envelope_method, envelope_kwargs: CHỈ có tác dụng khi
            envelope_window=None (nhánh tự tính envelope). Forward xuống
            extract_envelope_features() -> dsp.envelope_by_method(). Mặc
            định "square_law" GIỮ NGUYÊN hành vi cũ.
            ĐÃ SỬA: trước đây 2 tham số này không tồn tại, nhánh tự tính
            LUÔN LUÔN ra Square-Law bất kể ý định gọi hàm là gì - xem
            docstring extract_envelope_features().

    Số chiều mặc định = 11 (Nhóm A) + 10 (Nhóm B) + 7 (Nhóm C) = 28.
    Nhóm B là 10 chứ không phải 4x3=12, Nhóm C là 7 chứ không phải 3x3=9: ở
    độ phân giải 5.86 Hz/bin của cửa sổ 2048 mẫu, hai cặp hài (BPFO_h2 với
    BSF_h3, BPFO_h3 với BPFI_h2) không tách được nên order.py gộp mỗi cặp
    thành một cột, cố định cho mọi tải. Xem docstring common/order.py.
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
    if envelope_window is not None:
        feats.update(extract_envelope_features_from_envelope(
            envelope_window, fs, rpm, target_names=envelope_target_names,
            harmonics=harmonics, search_width_hz=search_width_hz, geometry=geometry,
        ))
    else:
        feats.update(extract_envelope_features(
            x, fs, rpm, band_hz=band_hz, lp_cutoff_hz=lp_cutoff_hz,
            bandpass_order=bandpass_order, lowpass_order=lowpass_order,
            target_names=envelope_target_names, harmonics=harmonics,
            search_width_hz=search_width_hz, geometry=geometry,
            method=envelope_method, envelope_kwargs=envelope_kwargs,
        ))
    return feats


def build_full_feature_table(manifest_df, band_hz, load_de_signal_fn=None,
                              window_size=2048, stride=1024,
                              warmup_samples=None,
                              target_fs_hz=None,
                              lp_cutoff_hz=dsp.BENCHMARK_LP_CUTOFF_HZ,
                              bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                              lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER,
                              envelope_method="square_law",
                              envelope_kwargs=None, **kwargs):
    """
    Chạy extract_full_feature_vector() bằng CỬA SỔ TRƯỢT cho TOÀN BỘ file.
    Tạo ra hàng ngàn dòng dữ liệu để mô hình MLP có đủ không gian mẫu học tập.

    Args:
        load_de_signal_fn: callable(file_path, source_fs_hz, target_fs_hz)
            -> np.ndarray. MẶC ĐỊNH dùng io_utils.load_de_signal_resampled
            (đọc tín hiệu DE + resample về target_fs_hz). CHỮ KÝ ĐÃ ĐỔI so
            với bản trước (trước đây chỉ nhận `file_path`) - nếu bạn có
            hàm đọc tùy chỉnh, cập nhật chữ ký cho khớp.
        target_fs_hz: tần số lấy mẫu CHUNG mà mọi file sẽ được resample về
            trước khi trích đặc trưng. Mặc định lấy từ
            config.SCOPE["sampling_rate_hz"] (12000Hz). ĐẢM BẢO band_hz,
            lp_cutoff_hz, window_size đều được thiết kế cho đúng tần số
            này.

        envelope_method: "square_law" (mặc định, giữ nguyên hành vi cũ),
            "hilbert_fir" hoặc "hybrid". Đi qua dsp.envelope_by_method() để
            CẢ 3 phương pháp dùng chung một cấu hình lọc nhân quả (bậc
            bandpass/lowpass, lp_cutoff, cách khử DC) — điều kiện bắt buộc
            để bảng RQ2/RQ3 so sánh được với nhau.
        envelope_kwargs: dict tham số bổ sung chuyển tiếp cho
            dsp.envelope_by_method(). BẮT BUỘC dùng cho RQ3:
            envelope_kwargs={"take_sqrt": True} khi envelope_method=
            "square_law". Mặc định square_law_envelope() trả LP(x^2), tức
            ĐƠN VỊ BÌNH PHƯƠNG của x, còn hilbert_fir/hybrid trả |A| (đơn
            vị của x). Nhóm C (envelope_*) vì thế lệch nhau cả thang đo lẫn
            DẢI ĐỘNG giữa 3 bảng đặc trưng, và dải động là đúng thứ quyết
            định sai số lượng tử hóa INT8 ở mục 2.3 - so sánh 3 phương pháp
            làm đầu vào AI mà không đồng nhất đơn vị thì chênh lệch đo được
            lẫn cả tác động của phép bình phương.

    manifest_df BẮT BUỘC đã qua run_sanity_checks() (có cột
    'resolved_sample_rate_hz') - nếu build từ io_utils.build_manifest()
    thì mặc định đã có cột này.

    RPM dùng để tính tần số mục tiêu (BPFO/BPFI/BSF/FTF): ƯU TIÊN cột
    'rpm_from_file' (RPM thực đọc từ từng file .mat); nếu thiếu, hoặc lệch
    danh định quá 20 RPM (ngưỡng RPM_LECH), thì fallback về
    NOMINAL_RPM_BY_LOAD theo tải và in cảnh báo.

    ------------------------------------------------------------------
    HỢP ĐỒNG VỀ CỘT ĐẦU RA (đừng phá vỡ)
    ------------------------------------------------------------------
      - file_id    : định danh FILE GỐC (make_file_id) — khóa chia
                     train/test của file_based_split()/LOLO.
      - window_idx : chỉ số cửa sổ trong file (0..n-1).
      - start_idx  : vị trí mẫu bắt đầu của cửa sổ (để truy vết).

    LỖI RÒ RỈ DỮ LIỆU ĐÃ SỬA: bản cũ ghi
        file_id = f"{Path(file_path).name}_win_{i}"
    khiến MỖI CỮA SỔ thành một "file" riêng biệt. file_based_split() chia
    theo file_id nên các cửa sổ CHỒNG LẤN của cùng 1 file bị rải cả vào
    train và test -> File-based Split thoái hóa thành Random Window Split.
    Hệ quả: RQ1/H1 bị vô hiệu, và vì cả hai nhánh thí nghiệm đều rò rỉ
    như nhau, kết quả gần nhau sẽ dẫn tới kết luận NGƯỢC rằng "không có
    rò rỉ dữ liệu" — tức dùng chính thí nghiệm bị rò rỉ để phủ nhận rò rỉ.
    """
    import pandas as pd
    from pathlib import Path

    if load_de_signal_fn is None:
        from . import io_utils
        load_de_signal_fn = io_utils.load_de_signal_resampled

    target_fs_hz = float(target_fs_hz or SCOPE.get("sampling_rate_hz", 12000))
    # Số mẫu đầu tín hiệu bị loại (quá độ khởi động + xác lập lọc IIR nhân
    # quả). Truyền 0 để giữ nguyên hành vi cũ.
    warmup_samples = WARMUP_SAMPLES if warmup_samples is None else int(warmup_samples)

    if "resolved_sample_rate_hz" not in manifest_df.columns:
        raise ValueError(
            "manifest_df thiếu cột 'resolved_sample_rate_hz' - manifest này "
            "chưa qua io_utils.run_sanity_checks(). KHÔNG được đoán fs mặc "
            "định (đây chính là bug đã sửa: trước đây fallback im lặng về "
            "12000Hz kể cả cho file Normal baseline thực chất 48kHz)."
        )

    rows = []
    n_skipped_no_fs = 0
    # file_id -> file_path đầu tiên dùng nó. Nếu 2 file khác nhau cho ra
    # cùng file_id, khóa chia train/test sẽ GHÉP nhầm 2 file thành 1 — phải
    # dừng ngay thay vì âm thầm chạy tiếp.
    file_id_registry: dict[str, str] = {}

    for _, row in manifest_df.iterrows():
        if row.get("label") is None or (isinstance(row.get("label"), float)):
            continue

        load_hp = row.get("load_hp")

        # RPM cho tần số mục tiêu BPFO/BPFI/BSF/FTF — ƯU TIÊN RPM THỰC đọc
        # từ file (cột rpm_from_file do io_utils.inspect_mat_file() trích từ
        # trường ...RPM của .mat), vì RPM thực tế của từng file lệch danh
        # định vài RPM, còn amplitude_near_frequency() chỉ bù được trong cửa
        # sổ ±search_width_hz quanh tần số mục tiêu. Trước đây hàm này chỉ
        # dùng RPM DANH ĐỊNH theo tải và bỏ phí cột rpm_from_file đã có sẵn
        # trong manifest.
        #
        # Nếu rpm_from_file lệch danh định quá 20 RPM (cùng ngưỡng với check
        # RPM_LECH của io_utils.run_sanity_checks) thì không rõ giá trị nào
        # đáng tin -> quay về danh định và in cảnh báo để kiểm tra, thay vì
        # âm thầm tin một trong hai. File không có trường RPM hợp lệ
        # (None/NaN/<=0) cũng dùng danh định.
        rpm_nominal = NOMINAL_RPM_BY_LOAD.get(load_hp)
        rpm_file = row.get("rpm_from_file")
        if rpm_file is not None and not pd.isna(rpm_file) and float(rpm_file) > 0:
            rpm_file = float(rpm_file)
            if rpm_nominal is not None and abs(rpm_file - rpm_nominal) > 20:
                print(
                    f"[CẢNH BÁO] {row['file_path']}: rpm_from_file="
                    f"{rpm_file:.0f} lệch >20 so với danh định tải "
                    f"{load_hp}HP ({rpm_nominal}) — dùng RPM DANH ĐỊNH. "
                    f"Kiểm tra lại cảnh báo RPM_LECH của file này."
                )
                rpm = float(rpm_nominal)
            else:
                rpm = rpm_file
        elif rpm_nominal is not None:
            rpm = float(rpm_nominal)
        else:
            rpm = 1750.0

        # Định danh file gốc + kiểm tra trùng NGAY TRƯỚC khi trích đặc
        # trưng (bước đắt nhất), thay vì phát hiện muộn sau hàng giờ chạy.
        file_id = make_file_id(row["file_path"])
        first_seen_path = file_id_registry.setdefault(file_id, str(row["file_path"]))
        if first_seen_path != str(row["file_path"]):
            raise ValueError(
                f"file_id bị trùng: '{file_id}' sinh ra từ 2 file khác nhau:\n"
                f"  - {first_seen_path}\n  - {row['file_path']}\n"
                f"file_id phải DUY NHẤT cho mỗi file gốc, nếu không "
                f"file_based_split()/LOLO sẽ ghép nhầm 2 file thành 1 nhóm. "
                f"Sửa make_file_id() để gồm thêm thư mục phân loại (ví dụ "
                f"nhãn/đường kính/vị trí OR) khi mở rộng phạm vi dữ liệu."
            )

        resolved_fs = row.get("resolved_sample_rate_hz")
        if resolved_fs is None or pd.isna(resolved_fs):
            # Không đủ căn cứ xác định fs thực (THOI_LUONG_BAT_THUONG) -
            # bỏ qua file này thay vì âm thầm đoán fs sai, chỉ log lại để
            # bạn kiểm tra thủ công thay vì mất dấu hoàn toàn.
            n_skipped_no_fs += 1
            print(f"[CẢNH BÁO] Bỏ qua {row['file_path']}: không xác định "
                  f"được resolved_sample_rate_hz (kiểm tra thủ công).")
            continue

        # Đọc + resample về đúng target_fs_hz chung cho toàn bộ dataset -
        # xử lý đúng rủi ro Normal baseline 48kHz nêu ở mục 0.1 đề cương.
        x_full = load_de_signal_fn(Path(row["file_path"]), resolved_fs, target_fs_hz)
        fs = target_fs_hz  # đồng nhất tuyệt đối cho MỌI file sau resample
        
        envelope_full = dsp.envelope_by_method(
            x_full, fs, band=band_hz, method=envelope_method,
            lp_cutoff=lp_cutoff_hz, bandpass_order=bandpass_order,
            lowpass_order=lowpass_order, **(envelope_kwargs or {}),
        )

        # Áp dụng Cửa sổ trượt (Sliding Window) - cùng chỉ số cắt cho cả
        # tín hiệu thô và envelope đã tính sẵn.
        num_windows = (len(x_full) - window_size) // stride + 1
        if num_windows <= 0:
            continue

        for i in range(num_windows):
            start_idx = i * stride
            # Bỏ cửa sổ chạm vùng quá độ đầu tín hiệu. i GIỮ NGUYÊN chỉ số
            # gốc để còn đối chiếu được với nhánh CNN (notebook 05).
            if start_idx < warmup_samples:
                continue
            end_idx = start_idx + window_size
            x_window = x_full[start_idx:end_idx]
            env_window = envelope_full[start_idx:end_idx]

            feats = extract_full_feature_vector(
                x_window, fs=fs, rpm=rpm, band_hz=band_hz,
                envelope_window=env_window,
                lp_cutoff_hz=lp_cutoff_hz, bandpass_order=bandpass_order,
                lowpass_order=lowpass_order, **kwargs
            )

            rows.append({
                "file_id": file_id,
                "window_idx": i,
                "start_idx": start_idx,
                "label": row["label"],
                # Nhãn của bài toán CHÍNH (10 lớp, mục 1.1). Sinh ngay tại
                # đây thay vì để từng notebook tự ghép label + đường kính:
                # ghép tay ở nhiều nơi là cách chắc chắn nhất để có hai
                # notebook định nghĩa lớp lệch nhau ("IR_7" vs "IR_007").
                "class_label": make_class_label(
                    row["label"], row.get("fault_diameter_mils")),
                "load_hp": load_hp,
                "fault_diameter_mils": row.get("fault_diameter_mils"),
                **feats,
            })

    if n_skipped_no_fs:
        print(f"[TÓM TẮT] Đã bỏ qua {n_skipped_no_fs} file do không xác "
              f"định được sampling rate thực - xem log CẢNH BÁO ở trên.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HỢP ĐỒNG VỀ CỘT: đâu là metadata, đâu là đặc trưng
# ---------------------------------------------------------------------------
# build_full_feature_table() trả về bảng gồm hai nhóm cột:
#   - Nhóm metadata (liệt kê dưới đây): dùng để chia tập, gộp nhóm, kiểm
#     toán. TUYỆT ĐỐI không được đưa vào X khi huấn luyện.
#   - Nhóm đặc trưng: mọi cột còn lại, do extract_full_feature_vector()
#     sinh ra với tiền tố "time_", "order_", "envelope_".
#
# window_idx và start_idx là hai cột MỚI thêm khi sửa lỗi rò rỉ dự liệu. Nếu
# notebook dựng danh sách đặc trưng bằng cách loại trừ thủ công, hai cột này
# rất dễ bị tính lẫn thành đặc trưng: mô hình khi đó học được "cửa sổ này nằm
# ở đâu trong file" - một dạng rò rỉ vị trí - và số chiều phồng từ 28 lên 30.
METADATA_COLUMNS = (
    "file_id",
    "window_idx",
    "start_idx",
    # features_dynamic.extract_features_dynamic_from_signals() gắn thêm cột
    # này: chỉ số tín hiệu trong signals_series, là KHÓA GHÉP file_id/label
    # cho bảng đó. Là metadata, KHÔNG phải đặc trưng — phải khai ở đây để
    # feature_columns(strict_prefix=True) không nhầm nó thành đặc trưng.
    "signal_idx",
    "label",
    # Nhãn 10 lớp (mục 1.1) - là NHÃN, không phải đặc trưng. Nếu quên khai
    # vào đây thì feature_columns(strict_prefix=True) sẽ báo lỗi thiếu tiền
    # tố, và tệ hơn là notebook liệt kê tay có thể đưa thẳng nhãn vào X.
    "class_label",
    "load_hp",
    "fault_diameter_mils",
    # Notebook 04 gắn thêm cột này khi build 3 bảng song song (square_law /
    # hilbert_fir / hybrid) cho RQ3. Đây là NHÃN MÔ TẢ, không phải đặc trưng:
    # nếu không khai vào hợp đồng thì feature_columns(strict_prefix=True) sẽ
    # báo lỗi thiếu tiền tố, còn nếu notebook tự liệt kê tay thì 'dsp_method'
    # có nguy cơ bị đưa vào X khi huấn luyện.
    "dsp_method",
)

FEATURE_PREFIXES = ("time_", "order_", "envelope_")

# ---------------------------------------------------------------------------
# SỐ CHIỀU ĐẶC TRƯNG: đề cương ghi 32, code sinh ra 28 — CÓ CHỦ Ý
# ---------------------------------------------------------------------------
# Mục 1.3 của đề cương đếm 11 + 12 + 9 = 32 chiều. Con số đó giả định mọi
# hài đều đọc được riêng biệt, nhưng ở độ phân giải thực tế thì KHÔNG:
#
#     FFT 2048 điểm @ 12000 Hz -> độ phân giải 5.86 Hz/bin
#     BSF_h3  = 7.0702 order  vs  BPFO_h2 = 7.1696 order -> cách 2.98 Hz
#     BPFO_h3 = 10.7543 order vs  BPFI_h2 = 10.8304 order -> cách 2.28 Hz
#
# Cả hai khoảng cách đều NHỎ HƠN MỘT BIN, nên hai "đặc trưng" trong mỗi
# cặp thực chất đọc cùng một bin. order.py vì vậy gộp chúng thành một cột
# (Nhóm B: 12 -> 10, Nhóm C: 9 -> 7), ra 11 + 10 + 7 = 28.
#
# ĐÂY KHÔNG PHẢI CHUYỆN TỐI ƯU BỀ MẶT ĐẶC TRƯNG — giữ 32 cột sẽ TẠO RÒ
# RỈ cho đánh giá LOLO. Hai cột đọc cùng một bin sẽ bằng nhau CHÍNH XÁC chỉ
# ở những mức tải mà hai hài rơi trùng bin, và khác nhau ở các mức tải
# khác. Tức phép kiểm tra "hai cột này có bằng nhau không" tự mã hóa MỨC
# TẢI — đúng biến mà LOLO đang cố giữ kín. Mô hình học được "đây là tải
# nào" thay vì "đây là lỗi gì", và độ chính xác LOLO báo cáo sẽ cao giả tạo.
#
# => GIỮ 28. Cần sửa lại con số 32 ở mục 1.3 của đề cương, kèm giải
#    thích trên, chứ KHÔNG sửa code để ép đủ 32.
EXPECTED_FEATURE_COUNTS = {
    "proposal_time": 11, "proposal_order": 12, "proposal_envelope": 9,
    "proposal_total": 32,
    "realized_time": 11, "realized_order": 10, "realized_envelope": 7,
    "realized_total": 28,
    "reason": (
        "FFT 2048 @ 12 kHz = 5.86 Hz/bin; BSF_h3 cách BPFO_h2 2.98 Hz và "
        "BPFO_h3 cách BPFI_h2 2.28 Hz -> dưới 1 bin, nên order.py gộp mỗi "
        "cặp thành 1 cột. Giữ riêng sẽ rò rỉ mức tải vào LOLO."
    ),
}


def feature_columns(feature_df, expected_count=None, strict_prefix=True):
    """
    Trả về danh sách cột đặc trưng của bảng do build_full_feature_table() tạo.

    Dùng hàm này thay cho việc tự viết [c for c in df.columns if c not in
    [...]] trong notebook: danh sách metadata chỉ khai báo ở một chỗ duy
    nhất, nên khi bảng thêm cột mới thì notebook không âm thầm sai.

    Args:
        expected_count: nếu truyền số (ví dụ 28), hàm raise ValueError khi số
            cột đặc trưng khác kỳ vọng. Nên truyền để chặn sai sót sớm.
        strict_prefix: nếu True, raise ValueError khi có cột không thuộc
            metadata mà cũng không mang tiền tố đặc trưng nào - đó là dấu
            hiệu bảng đã đổi cấu trúc.

    Returns:
        list[str] tên cột đặc trưng, giữ nguyên thứ tự trong DataFrame.
    """
    meta = set(METADATA_COLUMNS)
    cols = [c for c in feature_df.columns if c not in meta]

    if strict_prefix:
        la = [c for c in cols if not c.startswith(FEATURE_PREFIXES)]
        if la:
            raise ValueError(
                "Các cột sau không phải metadata nhưng cũng không mang tiền "
                f"tố đặc trưng {FEATURE_PREFIXES}: {la}. Hãy bổ sung chúng vào "
                "METADATA_COLUMNS nếu đó là cột mô tả, hoặc kiểm tra lại "
                "extract_full_feature_vector()."
            )

    if expected_count is not None and len(cols) != expected_count:
        raise ValueError(
            f"Kỳ vọng {expected_count} cột đặc trưng nhưng tìm thấy "
            f"{len(cols)}. Danh sách thực tế: {cols}"
        )

    return cols