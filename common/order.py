# -*- coding: utf-8 -*-
"""
common/order.py
=================
[Giai đoạn 1 — mục 1.4, Nhóm B] Biên độ tại hài bậc 1–3 của các tần số mục
tiêu (f_rot, BPFO, BPFI, BSF, FTF) trên phổ FFT/Order.

Tách riêng khỏi dsp.py vì đây là logic "trích đặc trưng", không phải "xử lý
tín hiệu" thuần túy.

===========================================================================
GIỚI HẠN ĐỘ PHÂN GIẢI — VÀ VÌ SAO PHẢI GỘP CỘT
===========================================================================
Đặc trưng tính trên cửa sổ 2048 mẫu @12kHz -> 1 bin = 5.86 Hz. Hai cặp hài
của hai HỌ tần số khác nhau nằm gần nhau hơn 1 bin nên không phân giải được:

    BSF  h3 =  7.0701 * f_rot   vs  BPFO h2 =  7.1696 * f_rot   (cách 0.0995)
    BPFO h3 = 10.7544 * f_rot   vs  BPFI h2 = 10.8305 * f_rot   (cách 0.0761)

Hệ thống gộp chung dữ liệu của mọi mức tải (0, 1, 2, 3 HP) vào một pool 
duy nhất, nên các vạch phổ này phải đủ bao quát để không bị ảnh hưởng bởi 
sự xê dịch RPM nhẹ giữa các tải. Vì vậy mỗi cặp được GỘP THÀNH MỘT cột, 
cố định cho MỌI file và MỌI tải: plan_harmonic_groups() chốt nhóm theo 
TỈ SỐ hình học (độc lập rpm), không theo rpm của từng file.

    Nhóm B: 12 -> 10 cột      Nhóm C: 9 -> 7 cột      Tổng: 32 -> 28 chiều
"""

from functools import lru_cache

import numpy as np

from .config import (bearing_fault_frequencies, NOMINAL_RPM_BY_LOAD,
                     SKF_6205_GEOMETRY)

MERGE_REFERENCE_RPM = float(max(NOMINAL_RPM_BY_LOAD.values()))

_GRAY_WARNED: set = set()

def amplitude_near_frequencies(freqs, mag, target_freqs_hz, search_width_hz=3.0):
    freqs = np.asarray(freqs)
    mag = np.asarray(mag)

    if len(freqs) >= 2:
        min_safe = float(freqs[1] - freqs[0]) / 2.0 * 1.05
        search_width_hz = max(search_width_hz, min_safe)

    mask = np.zeros(len(freqs), dtype=bool)
    for f in np.atleast_1d(target_freqs_hz):
        mask |= (freqs >= f - search_width_hz) & (freqs <= f + search_width_hz)
    return float(np.max(mag[mask])) if mask.any() else 0.0

def amplitude_near_frequency(freqs, mag, target_freq_hz, search_width_hz=3.0):
    return amplitude_near_frequencies(freqs, mag, [target_freq_hz], search_width_hz)

def target_orders(target_names, harmonics=(1, 2, 3), geometry=SKF_6205_GEOMETRY):
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
    return _plan(float(freq_resolution_hz), tuple(target_names), tuple(harmonics),
                 tuple(sorted(geometry.items())), float(reference_rpm),
                 float(warn_factor))

def _group_key(prefix, group, decl_index):
    labels = [f"{n}_h{h}"
              for n, h, _ in sorted(group, key=lambda t: (decl_index[t[0]], t[1]))]
    return f"{prefix}_" + "_and_".join(labels)

def extract_order_features(freqs, mag, rpm, target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                           harmonics=(1, 2, 3), search_width_hz=3.0,
                           geometry=SKF_6205_GEOMETRY, prefix="order",
                           merge_unresolvable=True,
                           reference_rpm=MERGE_REFERENCE_RPM):
    fault_freqs = bearing_fault_frequencies(rpm, geometry)
    decl_index = {n: i for i, n in enumerate(target_names)}
    freqs = np.asarray(freqs)

    if merge_unresolvable and len(freqs) >= 2:
        resolution = float(freqs[1] - freqs[0])
        groups, gray = plan_harmonic_groups(resolution, target_names, harmonics,
                                            geometry, reference_rpm)
        warn_key = (prefix, round(resolution, 4), gray)
        if gray and warn_key not in _GRAY_WARNED:
            _GRAY_WARNED.add(warn_key)
            print(f"[CẢNH BÁO] {prefix}: có cặp hài sát ngưỡng phân giải "
                  f"({resolution:.2f} Hz/bin): {list(gray)} — chưa gộp.")
    else:
        groups = tuple((item,) for item in
                       target_orders(target_names, harmonics, geometry))

    feats = {}
    for group in groups:
        target_hz = [fault_freqs[n] * h for n, h, _ in group]
        feats[_group_key(prefix, group, decl_index)] = amplitude_near_frequencies(
            freqs, mag, target_hz, search_width_hz)
    return feats