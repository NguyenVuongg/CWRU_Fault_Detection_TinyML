# -*- coding: utf-8 -*-
"""
common/order.py
=================
[Giai đoạn 1 — mục 1.4, Nhóm B] Biên độ tại hài bậc 1–3 của các tần số mục
tiêu (f_rot, BPFO, BPFI, BSF, FTF) trên phổ FFT/Order.

Tách riêng khỏi dsp.py vì đây là logic "trích đặc trưng", không phải "xử lý
tín hiệu" thuần túy.

===========================================================================
GIỚI HẠN ĐỘ PHÂN GIẢI — VÀ VÌ SAO PHẢI GỘP CỘT (ảnh hưởng trực tiếp LOLO)
===========================================================================
Đặc trưng tính trên cửa sổ 2048 mẫu @12kHz -> 1 bin = 5.86 Hz. Hai cặp hài
của hai HỌ tần số khác nhau nằm gần nhau hơn 1 bin nên không phân giải được:

    BSF  h3 =  7.0701 * f_rot   vs  BPFO h2 =  7.1696 * f_rot   (cách 0.0995)
    BPFO h3 = 10.7544 * f_rot   vs  BPFI h2 = 10.8305 * f_rot   (cách 0.0761)

Ở f_rot ~ 29 Hz, hai khoảng cách này chỉ là 2.98 Hz và 2.28 Hz. Zero-padding
không giúp gì: bề rộng búp chính vẫn là fs/N. Muốn tách thật cần cửa sổ 8192
mẫu (682 ms) -> chỉ còn ~29 cửa sổ/file, không đủ mẫu huấn luyện.

Vấn đề không nằm ở chỗ hai cột trùng nhau (cộng tuyến hoàn hảo thì vô hại về
độ chính xác, chỉ làm nhiễu Feature Importance ở mục 2.2). Vấn đề là bin rơi
vào đâu lại phụ thuộc f_rot: hai cột TRÙNG NHAU ở tải này nhưng KHÁC NHAU ở
tải khác. Tức "hai cột có bằng nhau hay không" tự mã hóa TẢI — đúng biến mà
Leave-One-Load-Out đang đo, một kênh rò rỉ thật.

Vì vậy mỗi cặp được GỘP THÀNH MỘT cột, cố định cho MỌI file và MỌI tải:
plan_harmonic_groups() chốt nhóm theo TỈ SỐ hình học (độc lập rpm), không
theo rpm của từng file.

    Nhóm B: 12 -> 10 cột      Nhóm C: 9 -> 7 cột      Tổng: 32 -> 28 chiều
"""

from functools import lru_cache

import numpy as np

from .config import (bearing_fault_frequencies, NOMINAL_RPM_BY_LOAD,
                     SKF_6205_GEOMETRY)

# rpm tham chiếu để chốt nhóm gộp. Khoảng cách hai vạch tỉ lệ với f_rot, nên
# rpm LỚN NHẤT trong phạm vi là trường hợp DỄ phân giải nhất: chỉ gộp những
# cặp không phân giải được ngay cả ở trường hợp tốt nhất.
MERGE_REFERENCE_RPM = float(max(NOMINAL_RPM_BY_LOAD.values()))

_GRAY_WARNED: set = set()


def amplitude_near_frequencies(freqs, mag, target_freqs_hz, search_width_hz=3.0):
    """Biên độ LỚN NHẤT trong HỢP các cửa sổ [f - w, f + w] quanh từng tần số
    mục tiêu — bù sai số nhỏ giữa rpm dùng để tính target và rpm thực của
    file. Cửa sổ tự nới tới >= nửa bin nên không bao giờ rỗng.
    """
    freqs = np.asarray(freqs)
    mag = np.asarray(mag)

    if len(freqs) >= 2:
        min_safe = float(freqs[1] - freqs[0]) / 2.0 * 1.05  # +5% biên làm tròn
        search_width_hz = max(search_width_hz, min_safe)

    mask = np.zeros(len(freqs), dtype=bool)
    for f in np.atleast_1d(target_freqs_hz):
        mask |= (freqs >= f - search_width_hz) & (freqs <= f + search_width_hz)
    return float(np.max(mag[mask])) if mask.any() else 0.0


def amplitude_near_frequency(freqs, mag, target_freq_hz, search_width_hz=3.0):
    """Trường hợp 1 tần số của amplitude_near_frequencies() — giữ cho code cũ."""
    return amplitude_near_frequencies(freqs, mag, [target_freq_hz], search_width_hz)


def target_orders(target_names, harmonics=(1, 2, 3), geometry=SKF_6205_GEOMETRY):
    """[(name, harmonic, order)] với order = f_target / f_rot — độc lập rpm."""
    # f_rot = 1 Hz  =>  giá trị trả về CHÍNH LÀ order
    ratios = bearing_fault_frequencies(60.0, geometry)
    out = []
    for name in target_names:
        if name not in ratios:
            raise KeyError(
                f"'{name}' không có trong bearing_fault_frequencies() — "
                f"các tên hợp lệ: {list(ratios)}"
            )
        out.extend((name, h, float(ratios[name]) * h) for h in harmonics)
    return out


@lru_cache(maxsize=64)
def _plan(freq_resolution_hz, target_names, harmonics, geometry_items,
          reference_rpm, warn_factor):
    geometry = dict(geometry_items)
    f_rot = reference_rpm / 60.0
    items = sorted(target_orders(target_names, harmonics, geometry), key=lambda t: t[2])

    groups, gray = [[items[0]]], []
    for prev, cur in zip(items, items[1:]):
        gap_hz = (cur[2] - prev[2]) * f_rot
        if gap_hz < freq_resolution_hz:
            groups[-1].append(cur)
        else:
            if gap_hz < warn_factor * freq_resolution_hz:
                gray.append((f"{prev[0]}_h{prev[1]}", f"{cur[0]}_h{cur[1]}",
                             round(gap_hz, 2)))
            groups.append([cur])
    return tuple(tuple(g) for g in groups), tuple(gray)


def plan_harmonic_groups(freq_resolution_hz,
                         target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                         harmonics=(1, 2, 3), geometry=SKF_6205_GEOMETRY,
                         reference_rpm=MERGE_REFERENCE_RPM, warn_factor=1.5):
    """Gộp các (tần số mục tiêu, hài) cách nhau < 1 bin.

    Trả về (groups, gray):
      - groups: tuple các nhóm, mỗi nhóm là tuple (name, harmonic, order).
        Danh sách này GIỐNG NHAU cho mọi file/mọi tải vì chốt theo tỉ số
        hình học + reference_rpm, không theo rpm từng file.
      - gray: các cặp cách 1..warn_factor bin — chưa gộp nhưng sát ngưỡng.
        Nếu đổi window_size/fs/hình học mà danh sách này không rỗng thì phải
        xem lại, vì đó là chỗ va chạm bin có thể quay lại theo tải.
    """
    return _plan(float(freq_resolution_hz), tuple(target_names), tuple(harmonics),
                 tuple(sorted(geometry.items())), float(reference_rpm),
                 float(warn_factor))


def _group_key(prefix, group, decl_index):
    """Nhóm 1 phần tử -> tên cũ ('order_BPFO_h1'). Nhóm gộp -> nối bằng
    '_and_' theo thứ tự KHAI BÁO của target_names ('order_BPFO_h2_and_BSF_h3').
    """
    labels = [f"{n}_h{h}"
              for n, h, _ in sorted(group, key=lambda t: (decl_index[t[0]], t[1]))]
    return f"{prefix}_" + "_and_".join(labels)


def extract_order_features(freqs, mag, rpm, target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                           harmonics=(1, 2, 3), search_width_hz=3.0,
                           geometry=SKF_6205_GEOMETRY, prefix="order",
                           merge_unresolvable=True,
                           reference_rpm=MERGE_REFERENCE_RPM):
    """
    Đặc trưng Nhóm B (mục 1.4): biên độ tại hài bậc 1..N của từng tần số mục
    tiêu. Tiền tố "order_"/"envelope_" giúp lọc theo nhóm khi phân tích
    Feature Importance ở mục 2.2.

    ĐÃ SỬA search_width_hz: 2.0 -> 3.0 để khớp features_full (cả
    extract_order_domain_features lẫn extract_envelope_features_from_envelope
    đều mặc định 3.0). Ở cửa sổ 2048 @12kHz hai giá trị này TRÙNG kết quả vì
    đều bị sàn an toàn kéo lên 3.0762 Hz (= 0.5*1.05*độ phân giải 5.8594 Hz),
    nên số liệu hiện có KHÔNG đổi. Nhưng ở cửa sổ 8192 mà chính docstring đầu
    file đề xuất (độ phân giải 1.4648 Hz, sàn chỉ 0.7690 Hz) thì 2.0 và 3.0
    cho ra HAI bề rộng tìm kiếm khác nhau -> Nhóm B và Nhóm C được trích bằng
    hai quy tắc khác nhau trong cùng một vector đặc trưng. Chốt về một số.

    Khóa: "order_BPFO_h1", ... Các cặp KHÔNG phân giải được ở độ phân giải
    của `freqs` được gộp thành một khóa, ví dụ "order_BPFO_h2_and_BSF_h3"
    (lý do: xem docstring đầu file). Số chiều vì thế là 10 (Nhóm B) và 7
    (Nhóm C) thay vì 12 và 9.

    merge_unresolvable=False trả lại đúng hành vi cũ (mỗi hài một cột, có cột
    trùng nhau theo tải) — chỉ dùng khi cần tái lập kết quả cũ.

    Tên không có trong bearing_fault_frequencies() làm hàm raise KeyError
    ngay, không âm thầm bỏ qua.
    """
    fault_freqs = bearing_fault_frequencies(rpm, geometry)
    decl_index = {n: i for i, n in enumerate(target_names)}
    freqs = np.asarray(freqs)

    if merge_unresolvable and len(freqs) >= 2:
        resolution = float(freqs[1] - freqs[0])
        groups, gray = plan_harmonic_groups(resolution, target_names, harmonics,
                                            geometry, reference_rpm)
        warn_key = (prefix, round(resolution, 4), gray)
        if gray and warn_key not in _GRAY_WARNED:
            _GRAY_WARNED.add(warn_key)  # in MỘT lần, không in theo từng cửa sổ
            print(f"[CẢNH BÁO] {prefix}: có cặp hài sát ngưỡng phân giải "
                  f"({resolution:.2f} Hz/bin): {list(gray)} — chưa gộp. Kiểm "
                  f"tra lại nếu đã đổi window_size/fs/hình học.")
    else:
        groups = tuple((item,) for item in
                       target_orders(target_names, harmonics, geometry))

    feats = {}
    for group in groups:
        target_hz = [fault_freqs[n] * h for n, h, _ in group]
        feats[_group_key(prefix, group, decl_index)] = amplitude_near_frequencies(
            freqs, mag, target_hz, search_width_hz)
    return feats
