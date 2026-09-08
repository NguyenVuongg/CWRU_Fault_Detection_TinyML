# -*- coding: utf-8 -*-
"""
common/config.py
=================
Hằng số và công thức dùng chung cho toàn bộ notebook Giai đoạn 0.
"""

from typing import Any, List, Optional, SupportsFloat, SupportsIndex, Union

import numpy as np

# ---------------------------------------------------------------------------
# Hình học vòng bi SKF 6205-2RS JEM (Drive-End) — chỉ áp dụng cho lỗi
# đường kính 0.007" / 0.014" / 0.021"
# ---------------------------------------------------------------------------
SKF_6205_GEOMETRY = {
    "n_balls": 9,
    "ball_diameter_in": 0.3126,
    "pitch_diameter_in": 1.537,
    "contact_angle_deg": 0.0,
}

SKF_VALID_FAULT_DIAMETERS_MILS = {7, 14, 21}

# Đường kính lỗi dùng vòng bi NTN tương đương — KHÔNG dùng SKF_6205_GEOMETRY
# để tính BPFO/BPFI/BSF cho các file này nếu chưa xác minh lại hình học NTN.
NTN_FAULT_DIAMETERS_MILS = {28, 40}


def _validate_rpm(rpm, caller: str) -> float:
    """Chặn rpm phi vật lý TRƯỚC khi nó kịp biến thành inf/NaN trong đặc trưng.

    Vì sao cần: rpm=0 -> hz_to_order() chia cho 0 -> trả về inf; rpm=NaN ->
    mọi order thành NaN; rpm<0 -> order âm. Cả ba trường hợp đều KHÔNG raise,
    numpy chỉ phát RuntimeWarning (thường bị tắt), nên giá trị hỏng chảy
    thẳng vào bảng đặc trưng Nhóm B/C rồi vào tập huấn luyện. Chỉ cần MỘT
    file CWRU thiếu khóa RPM hoặc parse hỏng là đủ đầu độc âm thầm cả LOLO.
    """
    r = float(rpm)
    if r != r:  # NaN
        raise ValueError(
            f"{caller}: rpm là NaN — kiểm tra khóa RPM trong file .mat "
            f"(xem io_utils.parse_metadata_from_filename / run_sanity_checks)."
        )
    if r <= 0.0:
        raise ValueError(
            f"{caller}: rpm phải > 0, nhận {rpm!r}. Tốc độ 0 hoặc âm khiến "
            f"order = f/f_rot thành inf hoặc âm và âm thầm làm hỏng đặc trưng."
        )
    return r


def bearing_fault_frequencies(rpm: float, geometry: dict = SKF_6205_GEOMETRY) -> dict:
    """Tính BPFO/BPFI/BSF/FTF (Hz) theo công thức chuẩn vòng bi rãnh sâu."""
    f_r = _validate_rpm(rpm, "bearing_fault_frequencies") / 60.0
    n = geometry["n_balls"]
    bd = geometry["ball_diameter_in"]
    pd = geometry["pitch_diameter_in"]
    phi = np.deg2rad(geometry["contact_angle_deg"])
    ratio = (bd / pd) * np.cos(phi)

    return {
        "f_rot": f_r,
        "BPFO": (n / 2.0) * f_r * (1 - ratio),
        "BPFI": (n / 2.0) * f_r * (1 + ratio),
        "BSF": (pd / (2.0 * bd)) * f_r * (1 - ratio ** 2),
        "FTF": (f_r / 2.0) * (1 - ratio),
    }


def hz_to_order(freq_hz, rpm: float):
    """Order = f_Hz / f_rot. Chặn rpm phi vật lý — xem _validate_rpm()."""
    f_r = _validate_rpm(rpm, "hz_to_order") / 60.0
    return np.asarray(freq_hz) / f_r


# ---------------------------------------------------------------------------
# Phạm vi dữ liệu đã chốt (mục 0.1)
# ---------------------------------------------------------------------------
# LƯU Ý VỀ TÊN KHÓA: khóa này PHẢI trùng tên cột "sensor_location" do
# io_utils.build_manifest() sinh ra, vì run_sanity_checks() so sánh
# trực tiếp hai bên. Bản cũ khai báo "sensor_position" trong khi
# io_utils đọc cfg.SCOPE.get("sensor_location", "DE") — KeyError bị
# .get() che đi nên check #6 luôn âm thầm chạy bằng fallback "DE":
# đúng số một cách tình cờ hôm nay, nhưng nếu đổi phạm vi sang "FE"
# thì config đổi mà check vẫn lọc theo "DE" và không báo gì cả.
# Tách hai danh sách này ra hằng số CÓ KIỂU RÕ RÀNG. Lý do không thuần
# thẩm mỹ: SCOPE là dict trộn nhiều loại giá trị (str, int, list), nên
# trình kiểm kiểu suy ra SCOPE["labels"] có kiểu hợp
#     str | int | list[str] | list[int]
# Và khi đó `label not in SCOPE["labels"]` là LỖI KIỂU, vì toán tử `in`
# không dùng được với int. Hai hằng số dưới đây là CÙNG MỘT object với
# SCOPE["labels"] / SCOPE["loads_hp"] — không nhân bản nguồn chân lý, chỉ
# khác là đã có kiểu tường minh để dùng trực tiếp.
SCOPE_LABELS: List[str] = ["Normal", "IR", "OR", "B"]
SCOPE_LOADS_HP: List[int] = [0, 1, 2, 3]

SCOPE = {
    "sensor_location": "DE",
    "sampling_rate_hz": 12000,
    "outer_race_position": "Centered",
    # 4 nhãn GỐC parse được từ tên thư mục/tên file CWRU. Đây KHÔNG phải
    # danh sách lớp của bài toán phân loại: sơ đồ 10 LỚP (mục 1.1) được
    # suy ra từ (labels x fault_diameter_mils) qua make_class_label().
    "labels": SCOPE_LABELS,
    "loads_hp": SCOPE_LOADS_HP,
}

# Chốt cứng tập khóa của SCOPE: mọi nơi đọc SCOPE đều index trực tiếp
# (không dùng .get có fallback), nên nếu ai gõ sai hoặc đổi tên khóa
# thì phải nổ NGAY tại import, thay vì để một check âm thầm chạy
# bằng giá trị mặc định và cho ra kết quả sai không dấu vết.
_REQUIRED_SCOPE_KEYS = {
    "sensor_location", "sampling_rate_hz", "outer_race_position",
    "labels", "loads_hp",
}
if set(SCOPE) != _REQUIRED_SCOPE_KEYS:
    raise ValueError(
        "SCOPE sai tập khóa. Thiếu: "
        f"{sorted(_REQUIRED_SCOPE_KEYS - set(SCOPE))}; "
        f"lạ: {sorted(set(SCOPE) - _REQUIRED_SCOPE_KEYS)}. "
        "Nếu đổi tên khóa, phải cập nhật cả io_utils.run_sanity_checks() "
        "và _REQUIRED_SCOPE_KEYS này."
    )

# ---------------------------------------------------------------------------
# Sơ đồ lớp (mục 1.1) — 10 LỚP là bài toán CHÍNH, 4 lớp là phân tích BỔ TRỢ
# ---------------------------------------------------------------------------
# Đề cương chốt bài toán chính là phân loại 10 lớp: Normal + {IR, OR, B} x
# {7, 14, 21} mils. Gộp cả 3 đường kính vào một nhãn "IR" (sơ đồ 4 lớp) làm
# mất chiều "mức độ nghiêm trọng" — vốn là thứ quyết định giá trị thực tế của
# chẩn đoán — nên 4 lớp chỉ dùng để ĐỐI CHIẾU phụ, không phải kết quả chính.
#
# Thứ tự trong CLASSES_10 là thứ tự chỉ số nhãn nguyên (0..9) dùng cho
# sparse_categorical_crossentropy và cho tf.one_hot ở models.make_sparse_macro_f1.
# ĐÃ CHỐT: không đổi thứ tự này, vì mọi model .tflite đã export và mọi
# confusion matrix đã lưu đều gắn với đúng thứ tự này.
CLASSES_10 = [
    "Normal",
    "IR_007", "IR_014", "IR_021",
    "OR_007", "OR_014", "OR_021",
    "B_007", "B_014", "B_021",
]

CLASSES_4 = ["Normal", "IR", "OR", "B"]

# Tên cột nhãn tương ứng trong bảng đặc trưng (features_full.py sinh cả hai):
#   "class_label" -> 10 lớp (chính)   |   "label" -> 4 lớp (bổ trợ)
LABEL_COL_BY_SCHEME = {"10class": "class_label", "4class": "label"}
CLASSES_BY_SCHEME = {"10class": CLASSES_10, "4class": CLASSES_4}
CLASS_SCHEME_DEFAULT = "10class"


def make_class_label(
    label: Any,
    fault_diameter_mils: Optional[Union[SupportsFloat, SupportsIndex, str]] = None,
) -> str:
    """Ghép (nhãn gốc, đường kính lỗi) -> nhãn 10 lớp, vd ("IR", 7) -> "IR_007".

    Nổ lỗi thay vì trả về giá trị đoán bừa, vì nhãn sai sẽ âm thầm chảy vào
    tập train và làm sai toàn bộ kết quả LOLO mà không có dấu vết nào.

    - "Normal" không có đường kính -> giữ nguyên "Normal".
    - Nhãn lỗi BẮT BUỘC có đường kính thuộc SKF_VALID_FAULT_DIAMETERS_MILS.
      File 28/40 mils (vòng bi NTN) đã bị loại khỏi phạm vi từ mục 0.1 nên
      không được phép đi tới đây.
    """
    if label is None:
        raise ValueError("label = None: không suy ra được nhãn 10 lớp.")

    label = str(label).strip()
    if label == "Normal":
        return "Normal"

    if label not in SCOPE_LABELS:
        raise ValueError(
            f"label='{label}' không thuộc phạm vi đã chốt {SCOPE_LABELS}."
        )

    # Chặn None TƯỜNG MINH trước khi gọi float(). Bản cũ dựa vào việc
    # float(None) ném TypeError rồi bắt lại — chạy đúng, nhưng trình kiểm
    # kiểu báo lỗi vì None không thỏa ConvertibleToFloat, và nó báo ĐÚNG:
    # dùng exception để xử lý một ca đã biết chắc là None thì vừa chậm hơn
    # vừa che mất ý định. Sau phép kiểm tra này kiểu được thu hẹp còn
    # SupportsFloat | SupportsIndex | str, đều hợp lệ với float().
    if fault_diameter_mils is None:
        raise ValueError(
            f"label='{label}' là nhãn lỗi nên BẮT BUỘC có fault_diameter_mils, "
            f"nhận được None."
        )

    # Vẫn giữ try/except cho các ca KHÔNG phải None mà vẫn không đổi được
    # sang số: chuỗi rác "abc", pandas.NA, numpy.datetime64, v.v.
    try:
        d_val = float(fault_diameter_mils)
    except (TypeError, ValueError):
        d_val = float("nan")
    if d_val != d_val:  # NaN — dùng cách này để config.py không phải import pandas
        raise ValueError(
            f"label='{label}' là nhãn lỗi nên BẮT BUỘC có fault_diameter_mils "
            f"đổi được sang số, nhận được {fault_diameter_mils!r}."
        )

    diameter = int(round(d_val))
    if diameter not in SKF_VALID_FAULT_DIAMETERS_MILS:
        raise ValueError(
            f"fault_diameter_mils={diameter} ngoài {sorted(SKF_VALID_FAULT_DIAMETERS_MILS)}. "
            f"File 28/40 mils dùng vòng bi NTN, đã bị loại ở mục 0.1 "
            f"(io_utils.apply_scope_filter -> VONG_BI_NTN)."
        )

    class_label = f"{label}_{diameter:03d}"
    if class_label not in CLASSES_10:
        raise ValueError(f"'{class_label}' không thuộc CLASSES_10 = {CLASSES_10}.")
    return class_label


NOMINAL_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
# Danh sách các tần số lấy mẫu cần test (Hz)
CANDIDATE_SAMPLING_RATES_HZ = [12000, 24000, 48000]

EXPECTED_DURATION_SEC = 10.0
DURATION_TOLERANCE_SEC = 3.0

# ---------------------------------------------------------------------------
# Envelope + cửa sổ — NGUỒN DUY NHẤT cho notebook 03/04/05/06 và Giai đoạn 2
# ---------------------------------------------------------------------------
# Dải cộng hưởng đã CHỐT cho toàn bộ RQ2 (mục 1.2). Trước đây con số này
# nằm rải rác trong docstring của dsp.py (2300-3800 Hz) và
# features_dynamic.py (2300-3800 Hz) mà không khớp đề cương, nên mỗi
# notebook truyền một dải khác nhau là so sánh 3 phương pháp envelope trên
# 3 dải khác nhau -> RQ2 mất hiệu lực. Từ đây mọi nơi lấy MỘT nguồn này.
#
# fc là tần số dùng để chuẩn hóa hệ số của hybrid envelope (dsp.hybrid_envelope):
# đặt fc = trung điểm dải để sai lệch biên độ |Q/I| đối xứng ở hai biên
# (fs=12kHz, fc=3kHz -> |Q/I| = 1.000 tại fc và 0.866 tại cả 2300 và 3800 Hz).
RESONANCE_BAND_HZ = (2300.0, 3800.0)

# DẪN XUẤT, không gõ tay: dsp.hybrid_envelope() tự tính fc = trung điểm dải.
# Nếu để cứng 3000.0 thì khi ai đó đổi RESONANCE_BAND_HZ, hằng số này lệch âm
# thầm và mọi chỗ trích dẫn nó sẽ khác hệ số thực sự đang chạy trong dsp.
RESONANCE_FC_HZ = (RESONANCE_BAND_HZ[0] + RESONANCE_BAND_HZ[1]) / 2.0

# Ngưỡng lowpass envelope — CHỐT 750 Hz (đã đổi từ 500 Hz).
#
# Đánh đổi đã kiểm tra bằng số (Butterworth bậc 2, vạch cao nhất trong phạm
# vi Nhóm C là BPFI hài bậc 3 = 486.6 Hz ở 0HP/1797rpm):
#     500 Hz -> BPFI_h3 suy giảm -2.78 dB   (biên độ còn ~72.6%)
#     750 Hz -> BPFI_h3 suy giảm -0.71 dB   (biên độ còn ~92.1%)
#
# VÌ SAO ĐỔI — đây là lỗi làm lệch KẾT QUẢ, không phải chuyện thẩm mỹ:
# 486.6 Hz "nằm dưới 500 Hz" nghe như còn dư địa, nhưng Butterworth bậc 2 đã
# -3 dB ĐÚNG TẠI tần số cắt, nên ở 0.97*fc nó vẫn cắt mất -2.78 dB. Tệ hơn,
# rpm đổi theo tải nên BPFI_h3 trượt 486.6 -> 468.4 Hz khi đi từ 0HP tới 3HP,
# kéo độ suy giảm trượt theo -2.78 -> -2.48 dB:
#     0HP 1797rpm  486.56 Hz  -2.7775 dB
#     1HP 1772rpm  479.79 Hz  -2.6630 dB
#     2HP 1750rpm  473.83 Hz  -2.5639 dB
#     3HP 1730rpm  468.42 Hz  -2.4751 dB
# Chênh 0.3024 dB đó ĐƠN ĐIỆU theo tải: bộ lọc tự in một "chữ ký tải" lên
# đúng cột envelope_BPFI_h3, chẳng liên quan gì tới vật lý vòng bi. Với RQ1
# (LOLO) đó là kênh rò rỉ thật — mô hình có thể suy ra tải từ riêng cột này.
# Ở 750 Hz suy giảm chỉ còn -0.71 dB và độ trượt theo tải gần như biến mất.
#
# Cả hai ngưỡng đều thấp hơn Nyquist của envelope sau giảm mẫu
# (12000/2/2 = 3000 Hz) nên KHÔNG bên nào gây chồng phổ.
#
# HỆ QUẢ PHẢI NHỚ: mọi số liệu envelope tính bằng 500 Hz đã hết hiệu lực và
# phải chạy lại; mục 1.3 của đề cương cần sửa 500 -> 750 Hz.
#
# Đổi ngưỡng thì đổi Ở ĐÂY (một dòng) — đừng truyền tay lp_cutoff rời rạc ở
# từng notebook, vì đó là cách các con số bắt đầu lệch nhau.
LP_CUTOFF_HZ = 750.0

# Hai hằng số dưới CHỈ để tái lập/đối chiếu, KHÔNG phải nguồn chân lý:
#   NARROW = ngưỡng cũ, giữ lại để chạy lại đúng số liệu trước khi đổi.
#   WIDE   = alias của giá trị đang dùng (giữ tên cũ cho code cũ khỏi vỡ).
LP_CUTOFF_HZ_NARROW = 500.0
LP_CUTOFF_HZ_WIDE = LP_CUTOFF_HZ

WINDOW_SIZE_RAW = 2048
STRIDE_RAW = 1024

# Envelope được GIẢM MẪU 1/ENV_DECIM trước khi cắt cửa sổ cho CNN. Nhờ đó
# 1024 mẫu @6kHz phủ đúng 170.67 ms = 2048 mẫu @12kHz, nên hai nhánh CNN của
# RQ4 có cùng thời lượng, cùng số cửa sổ, và cửa sổ thứ k của hai nhánh trùng
# khít cùng một khoảng thời gian (so sánh theo cặp được).
# Trước khi sửa: env 1024 mẫu @12kHz = 85.33 ms (13.8 chu kỳ BPFI) so với raw
# 170.67 ms (27.7 chu kỳ), và env có 9465 cửa sổ so với raw 4703 — hai
# confound cùng lúc, RQ4/H4 không kết luận được.
ENV_DECIM = 2
WINDOW_SIZE_ENV = WINDOW_SIZE_RAW // ENV_DECIM
STRIDE_ENV = STRIDE_RAW // ENV_DECIM

# Bỏ mọi cửa sổ chạm vùng đầu tín hiệu: quá độ khởi động cơ khí + thời gian
# xác lập của chuỗi lọc IIR NHÂN QUẢ (bandpass bậc 4 + lowpass bậc 2, chạy
# một lần trên cả file). Tính theo MẪU ở fs gốc; nhánh envelope quy đổi bằng
# WARMUP_SAMPLES // ENV_DECIM nên hai nhánh bỏ đúng cùng khoảng thời gian.
WARMUP_SAMPLES = WINDOW_SIZE_RAW
