# -*- coding: utf-8 -*-
"""
common/io_utils.py
===================
Đọc file .mat CWRU, dựng bảng manifest, chạy sanity check (mục 0.1).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import config as cfg


# Toàn bộ category gốc thật sự tồn tại trong data/raw/ (đã xác nhận qua
# manifest thật) — chỉ "12k_Drive_End_Bearing_Fault_Data" nằm trong phạm vi
# đã chốt; 2 category kia LUÔN tồn tại sẵn trong dữ liệu tải về gốc từ CWRU
# (không phải lỗi), nhưng phải bị loại tường minh, không dựa vào hiệu ứng
# phụ của check sampling-rate/vị-trí.
KNOWN_SOURCE_CATEGORIES = {
    "12k_Drive_End_Bearing_Fault_Data",
    "48k_Drive_End_Bearing_Fault_Data",
    "12k_Fan_End_Bearing_Fault_Data",
}


def parse_metadata_from_filename(filepath: Path) -> dict:
    """
    Parser KHỚP ĐÚNG cấu trúc thư mục thật đang dùng — data/raw/ có ĐỦ CẢ
    4 NHÓM sau (không chỉ nhóm trong phạm vi):

        data/raw/
            12k_Drive_End_Bearing_Fault_Data/   <- TRONG PHẠM VI
                B/007/118_0.mat            <id>_<load_hp>.mat
                B/014/.../ ..._<load>.mat
                IR/007/.../ ..._<load>.mat
                OR/007/@3/..._<load>.mat   @3=Orthogonal, @6=Centered, @12=Opposite
                OR/007/@6/..._<load>.mat
                OR/014/197@6_0.mat        (014 chỉ có Centered -> gắn "@6"
                                            thẳng vào tên file, không có
                                            thư mục @6 riêng)
            48k_Drive_End_Bearing_Fault_Data/   <- NGOÀI PHẠM VI (đã chốt chỉ 12kHz)
                (cùng cấu trúc con như trên)
            12k_Fan_End_Bearing_Fault_Data/     <- NGOÀI PHẠM VI (đã chốt chỉ Drive-End)
                (cùng cấu trúc con như trên)
            Normal/
                97_Normal_0.mat            <id>_Normal_<load_hp>.mat

    Quy tắc:
      - Nhãn lỗi (B/IR/OR/Normal) = tên 1 thư mục cha nào đó trong đường dẫn.
      - Đường kính lỗi (chỉ B/IR/OR) = tên thư mục NGAY SAU thư mục nhãn
        (vd "007" -> 7 mils).
      - Vị trí Outer Race (chỉ OR) = thư mục dạng "@3"/"@6"/"@12" trong
        đường dẫn, hoặc nhúng thẳng trong tên file (vd "197@6_0.mat", bắt
        bằng regex có lookahead để không khớp nhầm số khác), map sang
        Orthogonal/Centered/Opposite.
      - Mức tải (load_hp) = SỐ CUỐI CÙNG trong tên file, ngay sau dấu "_"
        cuối (đúng cho cả 2 dạng "118_0.mat" và "97_Normal_0.mat").
      - source_category = tên thư mục gốc khớp KNOWN_SOURCE_CATEGORIES ở
        trên (None nếu là Normal, vì Normal không thuộc 3 category này).
      - sensor_location = "DE"/"FE" suy ra từ source_category; Normal coi
        như tương thích DE (cấu trúc thư mục hiện tại không tách DE/FE
        riêng cho Normal).
      - declared_sample_rate_khz = 12/48 suy ra từ tên source_category
        (tần số CWRU KHAI BÁO qua tên thư mục — khác với tần số THỰC ĐO
        qua n_samples/thời lượng ở run_sanity_checks; 2 giá trị này có
        thể lệch nhau, như trường hợp Normal đã phát hiện).
    """
    parts = filepath.parts
    name = filepath.stem  # "118_0" hoặc "97_Normal_0"

    label = None
    diameter_mils = None
    or_position = None
    load_hp = None
    source_category = None
    sensor_location = None
    declared_sample_rate_khz = None

    for part in parts:
        if part in KNOWN_SOURCE_CATEGORIES:
            source_category = part
            break

    if source_category is not None:
        if "Fan_End" in source_category:
            sensor_location = "FE"
        elif "Drive_End" in source_category:
            sensor_location = "DE"
        if source_category.startswith("48k"):
            declared_sample_rate_khz = 48
        elif source_category.startswith("12k"):
            declared_sample_rate_khz = 12

    label_candidates = {"B", "IR", "OR", "Normal"}
    for part in parts:
        if part in label_candidates:
            label = part
            break

    if label == "Normal":
        sensor_location = "DE"  # baseline không tách DE/FE trong cấu trúc hiện tại

    if label in ("B", "IR", "OR") and label in parts:
        idx = parts.index(label)
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            diameter_mils = int(parts[idx + 1])

    if label == "OR":
        or_position_map = {"@3": "Orthogonal", "@6": "Centered", "@12": "Opposite"}
        for part in parts:
            if part in or_position_map:
                or_position = or_position_map[part]
                break
        if or_position is None:
            # Một số đường kính (vd 014) nhúng vị trí NGAY TRONG TÊN FILE
            # (dạng "197@6_0.mat") thay vì tách thư mục con "@6/" riêng như
            # 007/021 ("OR/007/@6/130_0.mat") — cùng 1 bộ dữ liệu nhưng đặt
            # tên không đồng nhất giữa các đường kính. Bắt thêm trường hợp
            # này bằng regex có lookahead (?=_|$) để không khớp nhầm số
            # khác đứng liền kề trong tên file.
            embedded_match = re.search(r"@(3|6|12)(?=_|$)", name)
            if embedded_match:
                or_position = or_position_map[f"@{embedded_match.group(1)}"]

    load_match = re.search(r"_(\d+)$", name)
    if load_match:
        load_hp = int(load_match.group(1))

    return {
        "load_hp": load_hp,
        "label": label,
        "fault_diameter_mils": diameter_mils,
        "or_position": or_position,
        "source_category": source_category,
        "sensor_location": sensor_location,
        "declared_sample_rate_khz": declared_sample_rate_khz,
    }


def inspect_mat_file(filepath: Path) -> dict:
    result: dict[str, object] = {
        "n_samples_DE": None, "n_samples_FE": None, "n_samples_BA": None,
        "rpm_from_file": None, "read_error": None,
    }
    try:
        # Ép kiểu rõ ràng thành dict để Pylance hiểu đây là dictionary
        mat = dict(loadmat(str(filepath)))
    except Exception as exc:
        result["read_error"] = str(exc)
        return result

    for key in mat.keys():
        if key.startswith("__"):
            continue
        if key.endswith("_DE_time"):
            result["n_samples_DE"] = int(np.asarray(mat[key]).size)
        elif key.endswith("_FE_time"):
            result["n_samples_FE"] = int(np.asarray(mat[key]).size)
        elif key.endswith("_BA_time"):
            result["n_samples_BA"] = int(np.asarray(mat[key]).size)
        elif key.endswith("RPM"):
            rpm_arr = np.asarray(mat[key]).ravel()
            if rpm_arr.size > 0:
                result["rpm_from_file"] = float(rpm_arr[0])
    return result


def load_de_signal(filepath: Path):
    """Đọc thẳng mảng tín hiệu DE (dùng ở các notebook phân tích tín hiệu)."""
    filepath_str = str(filepath)

    # Xử lý file numpy (dành cho Normal baseline đã downsample — nếu bạn
    # tự tạo file .npy này ở bước riêng của Giai đoạn 1, KHÔNG phải qua
    # pipeline.get_manifest(); xem ghi chú ở common/pipeline.py)
    if filepath_str.endswith('.npy'):
        return np.load(filepath_str)

    # Xử lý file .mat
    try:
        mat = dict(loadmat(filepath_str))
    except Exception as e:
        raise IOError(f"Không thể đọc file {filepath_str}: {e}")

    for key in mat.keys():
        if key.endswith("_DE_time"):
            return np.asarray(mat[key]).ravel()

    raise KeyError(f"Không tìm thấy biến '..._DE_time' trong {filepath_str}")


def load_de_signal_resampled(filepath: Path, source_fs_hz, target_fs_hz: float):
    """
    [BỔ SUNG] Đọc tín hiệu DE rồi resample về ĐÚNG 1 tần số lấy mẫu chung
    (target_fs_hz) cho TOÀN BỘ dataset, dựa trên `source_fs_hz` — PHẢI là
    giá trị lấy từ cột `resolved_sample_rate_hz` của manifest (xem
    run_sanity_checks/_resolve_sampling_rate_hz), KHÔNG phải
    `declared_sample_rate_khz` (suy từ tên thư mục, có thể sai).

    Đây là bước "xử lý sampling rate của Normal baseline trước khi đưa
    vào manifest chính thức" mà mục 0.1 đề cương yêu cầu nhưng trước đó
    chưa có code nào thực hiện — features_full.build_full_feature_table()
    trước đây LUÔN giả định fs=12000Hz cho mọi file (kể cả Normal baseline
    thực chất 48kHz), khiến toàn bộ đặc trưng Order/Envelope của lớp
    Normal bị tính sai trục tần số.

    Dùng resample_poly (FIR đa pha) thay vì scipy.signal.resample (dựa
    trên FFT) vì resample_poly ổn định hơn với tỉ lệ hữu tỉ đơn giản như
    48000/12000 = 4 (downsample nguyên lần) và không giả định tín hiệu
    tuần hoàn (tránh méo ở 2 đầu đoạn tín hiệu).

    Raises:
        ValueError: nếu source_fs_hz là None/NaN — nghĩa là
            _resolve_sampling_rate_hz() không xác định được fs thực cho
            file này (đã bị gắn cảnh báo THOI_LUONG_BAT_THUONG). KHÔNG
            được đoán fs trong trường hợp này — phải kiểm tra thủ công.
    """
    from fractions import Fraction
    from scipy.signal import resample_poly

    x = load_de_signal(filepath)

    if source_fs_hz is None or (isinstance(source_fs_hz, float) and np.isnan(source_fs_hz)):
        raise ValueError(
            f"Không xác định được sampling rate thực (resolved_sample_rate_hz) "
            f"cho file {filepath} — kiểm tra cảnh báo THOI_LUONG_BAT_THUONG "
            f"trong manifest trước khi trích đặc trưng cho file này."
        )

    source_fs_hz = float(source_fs_hz)
    target_fs_hz = float(target_fs_hz)

    if round(source_fs_hz) == round(target_fs_hz):
        return x

    frac = Fraction(int(round(target_fs_hz)), int(round(source_fs_hz))).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator)


def _resolve_sampling_rate_hz(n_samples, target_rate_hz, candidates,
                               expected_duration, tolerance):
    """
    [BỔ SUNG] Xác định tần số lấy mẫu THỰC cần dùng cho mọi tính toán DSP
    (bandpass/lowpass/FFT/envelope) của 1 file — khác với
    `declared_sample_rate_khz`, vốn chỉ suy ra từ TÊN THƯ MỤC và có thể
    sai (đúng trường hợp Normal baseline 48kHz đã nêu ở mục 0.1 đề cương:
    "Sampling rate thật của file Normal baseline... Đặc trưng Order/tần số
    của lớp Normal lệch hoàn toàn so với 3 lớp lỗi" nếu dùng nhầm rate).

    Quy tắc chọn (dựa trên thời lượng thực đo n_samples/rate ~ 10s):
      1. Nếu chính target_rate_hz (declared, hoặc mặc định SCOPE cho
         Normal) đã cho thời lượng hợp lý -> tin tưởng nó, dùng luôn.
      2. Nếu không, nhưng có (>=1) rate khác trong danh sách candidates
         cho thời lượng hợp lý -> đây là rate THỰC (declared bị sai) -> 
         dùng rate hợp lý gần target_rate_hz nhất.
      3. Nếu không có rate nào cho thời lượng hợp lý -> trả về None
         (không đủ căn cứ; dòng này đã được gắn cảnh báo
         THOI_LUONG_BAT_THUONG, cần bạn kiểm tra thủ công thay vì đoán).

    Trả về None nếu n_samples không xác định (NaN) hoặc không tìm được
    rate hợp lý nào — KHÔNG bao giờ trả về 1 con số đoán mò không có căn cứ.
    """
    if n_samples is None or pd.isna(n_samples):
        return None

    durations = {rate: n_samples / rate for rate in candidates}
    plausible = [rate for rate, dur in durations.items()
                 if abs(dur - expected_duration) <= tolerance]

    if not plausible:
        return None
    if target_rate_hz in plausible:
        return float(target_rate_hz)
    return float(min(plausible, key=lambda r: abs(r - target_rate_hz)))


def run_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột cảnh báo — KHÔNG tự xóa dòng nào, chỉ đánh dấu.

    Dùng enumerate(df.iterrows()) thay vì chỉ số trả về từ iterrows() —
    chỉ số của iterrows() là NHÃN index của DataFrame (kiểu Hashable, có
    thể không liên tục nếu df là kết quả lọc/subset từ DataFrame khác),
    trong khi warnings_list được dựng theo VỊ TRÍ (0..len(df)-1). Nếu
    df không có RangeIndex mặc định liên tục, dùng thẳng chỉ số iterrows()
    sẽ ghi cảnh báo NHẦM DÒNG. enumerate() luôn cho chỉ số vị trí đúng.

    QUAN TRỌNG: không dùng `continue` khi thiếu n_samples_DE — dòng bị
    thiếu DE chỉ bỏ qua đúng phần check thời lượng/sampling rate, các
    check RPM/NTN/OR-vị-trí/thiếu-nhãn/cảm-biến/tần-số-khai-báo bên dưới
    (không phụ thuộc gì vào việc đọc được DE hay không) vẫn phải chạy.
    """
    df = df.copy()
    warnings_list: list[list[str]] = [[] for _ in range(len(df))]
    # [BỔ SUNG] fs THỰC của từng file — dùng bởi io_utils.load_de_signal_resampled()
    # và features_full.build_full_feature_table() thay vì đoán/mặc định 12000Hz.
    resolved_fs_list: list = [None] * len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        # --- 0. File có đọc được không? ---
        read_error = row.get("read_error")
        if pd.notna(read_error):
            warnings_list[i].append(f"LOI_DOC_FILE: không đọc được file .mat ({read_error}).")

        # --- 1. Sampling rate / thời lượng (chỉ chạy được nếu có n_samples_DE) ---
        n = row.get("n_samples_DE")
        if n is None or pd.isna(n):
            warnings_list[i].append(
                "THIEU_TIN_HIEU_DE: không tìm thấy kênh '_DE_time' trong file .mat "
                "(không thể kiểm tra sampling rate/thời lượng cho dòng này)."
            )
        else:
            # Test nhiều mức tần số phổ biến (cfg.CANDIDATE_SAMPLING_RATES_HZ,
            # vd [12000, 24000, 48000]) — KHÔNG chỉ 12k/48k, vì thực tế đã
            # gặp file cho ra thời lượng hợp lý ở mức KHÁC (24kHz) mà nếu
            # chỉ test 2 mức mặc định sẽ không phát hiện được.
            #
            # So với đúng tần số MÀ FILE TỰ KHAI BÁO (declared_sample_rate_khz
            # qua source_category) — KHÔNG so với 1 mốc "nominal" cố định
            # toàn cục. Normal không có source_category riêng nên fallback
            # về tần số MỤC TIÊU của phạm vi (cfg.SCOPE["sampling_rate_hz"])
            # — đây chính là cách phát hiện Normal thực chất ở 48kHz dù
            # không "khai báo" tần số nào qua tên thư mục. Nhờ so theo TỪNG
            # FILE thay vì 1 mốc chung, 1 file 48k_Drive_End thật sự đúng
            # 48kHz (đã NẰM SẴN trong thư mục khai báo 48kHz) sẽ KHÔNG bị
            # bắt ở đây — việc loại nó khỏi phạm vi là việc của check
            # NGOAI_PHAM_VI_TAN_SO_KHAI_BAO bên dưới, tách bạch đúng 2 mối
            # quan tâm khác nhau: "nội dung có khớp cái file tự nhận là gì
            # không" vs "cái file tự nhận có nằm trong phạm vi đề tài không".
            declared_khz = row.get("declared_sample_rate_khz")
            # LƯU Ý: declared_khz đọc từ cột pandas — với các dòng không có
            # source_category (vd Normal), giá trị gốc là None nhưng pandas
            # có thể ép kiểu cả cột thành float khi trộn với các dòng có số
            # (12/48), biến None thành NaN. `NaN is not None` là True trong
            # Python — dùng `is not None` ở đây sẽ lọt qua, tính NaN*1000 =
            # NaN, in ra "nankHz" (đã xảy ra thật). Phải dùng pd.notna().
            target_rate_hz = (
                declared_khz * 1000 if pd.notna(declared_khz)
                else cfg.SCOPE["sampling_rate_hz"]
            )

            durations = {rate: n / rate for rate in cfg.CANDIDATE_SAMPLING_RATES_HZ}
            plausible = {rate: dur for rate, dur in durations.items()
                         if abs(dur - cfg.EXPECTED_DURATION_SEC) <= cfg.DURATION_TOLERANCE_SEC}
            duration_summary = ", ".join(f"{r/1000:.0f}kHz->{d:.1f}s" for r, d in durations.items())

            resolved_fs_list[i] = _resolve_sampling_rate_hz(
                n, target_rate_hz, cfg.CANDIDATE_SAMPLING_RATES_HZ,
                cfg.EXPECTED_DURATION_SEC, cfg.DURATION_TOLERANCE_SEC,
            )

            if target_rate_hz not in plausible and plausible:
                plausible_str = "/".join(f"{r/1000:.0f}kHz" for r in plausible)
                warnings_list[i].append(
                    f"NGHI_NGO_SAMPLING_RATE: n_samples={n} ({duration_summary}). "
                    f"Rate hợp lý nhất: {plausible_str} — KHÁC rate mà file tự "
                    f"nhận ({target_rate_hz/1000:.0f}kHz, qua thư mục nguồn/phạm "
                    f"vi mục tiêu)."
                )
            elif not plausible:
                warnings_list[i].append(
                    f"THOI_LUONG_BAT_THUONG: n_samples={n} không khớp ~10s ở bất "
                    f"kỳ rate phổ biến nào đã kiểm tra ({duration_summary}) — "
                    f"kiểm tra file này thủ công (có thể bị cắt ngắn/hỏng, hoặc "
                    f"dùng sampling rate không nằm trong danh sách đã test)."
                )

        # --- 2. RPM lệch so với danh định --- (độc lập với DE ở trên)
        load_hp = row.get("load_hp")
        rpm_file = row.get("rpm_from_file")
        if load_hp in cfg.NOMINAL_RPM_BY_LOAD and rpm_file is not None and not pd.isna(rpm_file):
            rpm_nominal = cfg.NOMINAL_RPM_BY_LOAD[load_hp]
            if abs(rpm_file - rpm_nominal) > 20:
                warnings_list[i].append(
                    f"RPM_LECH: RPM file ({rpm_file:.0f}) lệch >20 so với "
                    f"danh định tải {int(load_hp)}HP ({rpm_nominal})."
                )

        # --- 3. Vòng bi NTN vs SKF --- (độc lập với DE ở trên)
        diam = row.get("fault_diameter_mils")
        if diam is not None and not pd.isna(diam):
            diam = int(diam)
            if diam in cfg.NTN_FAULT_DIAMETERS_MILS:
                warnings_list[i].append(
                    f"VONG_BI_NTN: đường kính {diam} mils dùng vòng bi NTN, "
                    f"KHÔNG dùng hình học SKF 6205 để tính BPFO/BPFI/BSF."
                )
            elif diam not in cfg.SKF_VALID_FAULT_DIAMETERS_MILS:
                warnings_list[i].append(f"DUONG_KINH_LA: {diam} mils không rõ nguồn gốc.")

        # --- 4. Vị trí Outer Race --- (độc lập với DE ở trên)
        label = row.get("label")
        or_pos = row.get("or_position")
        if label == "OR":
            if or_pos is None or (isinstance(or_pos, float) and pd.isna(or_pos)):
                warnings_list[i].append("OR_THIEU_VI_TRI: nhãn OR nhưng không rõ vị trí lỗi.")
            elif cfg.SCOPE["outer_race_position"].lower() not in str(or_pos).lower():
                warnings_list[i].append(
                    f"OR_NGOAI_PHAM_VI: vị trí '{or_pos}' khác phạm vi đã chốt "
                    f"('{cfg.SCOPE['outer_race_position']}')."
                )

        # --- 5. Thiếu nhãn / thiếu tải --- (độc lập với DE ở trên)
        if label is None:
            warnings_list[i].append("THIEU_NHAN: không parse được nhãn lỗi.")
        if load_hp is None:
            warnings_list[i].append("THIEU_TAI: không parse được mức tải.")

        # --- 6. Vị trí cảm biến (DE vs FE) --- (tường minh, không dựa vào
        #     hiệu ứng phụ của check khác — file Fan-End đúng RPM và đúng
        #     vị trí OR có thể lọt qua với 0 cảnh báo nếu thiếu check này)
        sensor_location = row.get("sensor_location")
        # KHÔNG dùng .get(..., "DE"): fallback im lặng biến một khóa
        # cấu hình sai thành "check luôn đúng". Index trực tiếp để
        # KeyError nổ ngay khi tên khóa lệch với config.SCOPE.
        target_sensor = cfg.SCOPE["sensor_location"]
        # pd.notna() thay vì "is not None" — cùng lý do đã sửa ở check 1/7:
        # cột pandas có thể trả NaN cho các dòng vốn là None khi bị ép kiểu
        # chung với dòng khác (chưa từng biểu hiện lỗi vì Normal luôn gán
        # sẵn "DE" ở trên, nhưng sửa cho nhất quán/an toàn về sau).
        if pd.notna(sensor_location) and sensor_location != target_sensor:
            warnings_list[i].append(
                f"NGOAI_PHAM_VI_CAM_BIEN: file thuộc vị trí cảm biến "
                f"'{sensor_location}', phạm vi đã chốt chỉ dùng '{target_sensor}'."
            )

        # --- 7. Tần số lấy mẫu KHAI BÁO (qua tên thư mục nguồn) --- (tường
        #     minh, khác với check thời lượng THỰC ĐO ở mục 1 — nếu thiếu
        #     check này, nhóm 48k_Drive_End chỉ bị loại "may rủi" nhờ ăn
        #     theo check thời lượng, không phải quyết định phạm vi rõ ràng)
        declared_rate = row.get("declared_sample_rate_khz")
        # BUG ĐÃ SỬA: SCOPE không có key "target_sample_rate_khz" (chỉ có
        # "sampling_rate_hz"), nên .get(...) trước đây LUÔN rơi về fallback
        # hardcode "12", không thực sự đọc từ SCOPE — vô hại về số (khớp
        # đúng 12 hiện tại) nhưng sẽ âm thầm sai nếu SCOPE["sampling_rate_hz"]
        # từng đổi. Tính trực tiếp từ SCOPE để bám theo cấu hình thật.
        target_rate = cfg.SCOPE["sampling_rate_hz"] / 1000

        # pd.notna() thay vì "is not None" — ĐÂY chính là chỗ gây ra bug
        # Normal bị loại nhầm khỏi manifest ở 2 lượt trước: declared_rate
        # của Normal là NaN (không phải None) sau khi qua DataFrame.
        if pd.notna(declared_rate) and declared_rate != target_rate:
            warnings_list[i].append(
                f"NGOAI_PHAM_VI_TAN_SO_KHAI_BAO: file thuộc nhóm khai báo "
                f"{declared_rate}kHz, phạm vi đã chốt chỉ dùng {target_rate}kHz."
            )

    df["warnings"] = ["; ".join(w) if w else "" for w in warnings_list]
    df["has_warning"] = df["warnings"] != ""
    df["resolved_sample_rate_hz"] = resolved_fs_list
    return df


def apply_scope_filter(manifest: pd.DataFrame,
                        exclude_warning_keywords: list[str] | None = None) -> pd.DataFrame:
    """
    [Dùng ở Giai đoạn 1, trước khi trích đặc trưng] Loại khỏi manifest các
    file KHÔNG thuộc phạm vi đã chốt (mục 0.1/0.4) — vì các cảnh báo ở
    Giai đoạn 0 chỉ ĐÁNH DẤU, không tự loại: nếu không lọc ở đây, bể đặc
    trưng Giai đoạn 1 sẽ tính BPFO/BPFI/BSF SAI cho file vòng bi NTN (hình
    học khác SKF 6205), gộp nhầm vị trí Outer Race ngoài Centered, hoặc
    trộn lẫn dữ liệu Fan-End/48kHz vào tập được coi là "Drive-End 12kHz".

    exclude_warning_keywords mặc định loại các từ khóa RỦI RO PHÁ VỠ GIẢ
    ĐỊNH VẬT LÝ / NGOÀI PHẠM VI ĐÃ CHỐT, cộng với các file không đọc được
    nội dung (không thể dùng bất kể phạm vi). KHÔNG mặc định loại
    RPM_LECH hay NGHI_NGO_SAMPLING_RATE — đó là cảnh báo cần bạn tự xem
    xét, có thể vẫn dùng được nếu RPM/sampling rate thật đã được xác nhận
    qua giá trị đọc trực tiếp từ file, chỉ lệch nhẹ so với bảng danh định.
    Normal baseline (48kHz thật) rơi đúng vào trường hợp này — cố ý KHÔNG
    tự động loại, để bạn quyết định resample hay xử lý riêng ở Giai đoạn 1.

    In ra rõ loại bao nhiêu dòng, vì lý do gì, để không mất kiểm soát.
    """
    if exclude_warning_keywords is None:
        exclude_warning_keywords = [
            # Ngoài phạm vi / phá vỡ giả định vật lý -> loại tường minh
            "VONG_BI_NTN", "OR_NGOAI_PHAM_VI", "OR_THIEU_VI_TRI",
            "THIEU_NHAN", "THIEU_TAI", "DUONG_KINH_LA",
            "NGOAI_PHAM_VI_CAM_BIEN", "NGOAI_PHAM_VI_TAN_SO_KHAI_BAO",
            # Không đọc được nội dung -> không thể dùng bất kể phạm vi
            "THIEU_TIN_HIEU_DE", "LOI_DOC_FILE",
        ]

    df = manifest.copy()
    warnings_text = df["warnings"].fillna("")
    exclude_mask = pd.Series(False, index=df.index)

    print("Áp dụng bộ lọc phạm vi (mục 0.1/0.4):")
    for kw in exclude_warning_keywords:
        kw_mask = warnings_text.str.contains(kw, regex=False)
        n = int(kw_mask.sum())
        print(f"  - Loại {n} file có cảnh báo '{kw}'")
        exclude_mask = exclude_mask | kw_mask

    kept = df.loc[~exclude_mask].reset_index(drop=True)
    print(f"Tổng: loại {int(exclude_mask.sum())}/{len(df)} file — còn lại {len(kept)} file.")
    return kept


def build_manifest(data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root)
    mat_files = sorted(data_root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"Không tìm thấy file .mat nào trong {data_root}.")

    rows = []
    for fp in mat_files:
        meta = parse_metadata_from_filename(fp)
        content = inspect_mat_file(fp)
        rows.append({"file_path": str(fp), **meta, **content})

    return run_sanity_checks(pd.DataFrame(rows))