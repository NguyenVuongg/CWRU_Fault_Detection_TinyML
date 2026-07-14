# -*- coding: utf-8 -*-
"""
common/synthetic.py
=====================
Sinh tín hiệu / file .mat GIẢ LẬP — dùng khi chưa có dữ liệu CWRU thật,
để bạn chạy thử toàn bộ 8 notebook và xem trước hình dạng đầu ra.

QUAN TRỌNG: mọi con số/kết luận rút ra từ dữ liệu giả lập chỉ có giá trị
kiểm tra logic code, KHÔNG được dùng làm kết quả báo cáo chính thức.
Mỗi notebook đều có cờ USE_SYNTHETIC_DATA ở cell đầu tiên — đổi thành
False và trỏ DATA_ROOT vào dữ liệu thật khi đã sẵn sàng.
"""

import shutil
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import lfilter

from . import config as cfg


def make_synthetic_signal(fs, duration_sec, rpm, fault_freq_hz=None,
                           carrier_hz=2500, noise_std=0.3, seed=None):
    """
    Tín hiệu giả: rung nền (mất cân bằng trục tại f_rot) + nhiễu trắng, và
    nếu có fault_freq_hz thì cộng thêm chuỗi xung điều biên bởi sóng mang
    cộng hưởng — mô phỏng đúng cơ chế vật lý của lỗi vòng bi thật.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(0, duration_sec, 1 / fs)
    x = noise_std * rng.randn(len(t))
    f_rot = rpm / 60.0
    x += 0.5 * np.sin(2 * np.pi * f_rot * t)

    if fault_freq_hz is not None and fault_freq_hz > 0:
        period = 1.0 / fault_freq_hz
        impulse_train = np.zeros_like(t)
        for it in np.arange(0, duration_sec, period):
            idx = int(it * fs)
            if idx < len(impulse_train):
                impulse_train[idx] = 1.0
        decay = np.exp(-np.arange(200) / 15.0) * np.sin(2 * np.pi * carrier_hz * np.arange(200) / fs)
        x += 1.5 * lfilter(decay, [1.0], impulse_train)

    return t, x


# Tần số lỗi giả lập gần đúng cho mỗi nhãn tại RPM danh định — chỉ dùng để
# tạo dữ liệu demo có "hình dạng" hợp lý, KHÔNG phải số liệu CWRU thật.
_FAULT_LABEL_TO_FREQ_KEY = {"IR": "BPFI", "OR": "BPFO", "B": "BSF"}


def build_synthetic_dataset(root: Path, loads=(0, 1, 2, 3), seed=0,
                             diameters_mils=(7, 14, 21), duration_sec=10.0):
    """
    Tạo bộ file .mat giả lập đầy đủ 4 nhãn x nhiều tải, cấu trúc thư mục
    <root>/<load>hp/<label>_<diameter>[_<or_position>].mat — khớp với
    parse_metadata_from_filename() mẫu trong common/io_utils.py.
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    rng_seed = seed
    fs = cfg.SCOPE["sampling_rate_hz"]

    for load in loads:
        folder = root / f"{load}hp"
        folder.mkdir(exist_ok=True)
        rpm = cfg.NOMINAL_RPM_BY_LOAD[load]
        fault_freqs = cfg.bearing_fault_frequencies(rpm)

        # Normal
        _, x = make_synthetic_signal(fs, duration_sec, rpm, seed=rng_seed)
        savemat(str(folder / "Normal.mat"), {"X999_DE_time": x.reshape(-1, 1),
                                              "X999RPM": np.array([[rpm]])})
        rng_seed += 1

        # IR / OR / B tại từng đường kính
        for label, freq_key in _FAULT_LABEL_TO_FREQ_KEY.items():
            for diam in diameters_mils:
                _, x = make_synthetic_signal(
                    fs, duration_sec, rpm, fault_freq_hz=fault_freqs[freq_key], seed=rng_seed,
                )
                suffix = "_Centered" if label == "OR" else ""
                fname = f"{label}_{diam:03d}{suffix}.mat"
                savemat(str(folder / fname), {"X999_DE_time": x.reshape(-1, 1),
                                              "X999RPM": np.array([[rpm]])})
                rng_seed += 1

    return root


def build_edge_case_dataset(root: Path):
    """
    Tạo 5 file .mat GIẢ LẬP, mỗi file cố ý gài đúng 1 loại lỗi mà
    common/io_utils.run_sanity_checks() phải bắt được. Dùng để TỰ KIỂM
    CHỨNG pipeline (notebook 01, mục cuối) hoạt động đúng, không dùng cho
    phân tích ở các notebook khác.

    Trả về (root, danh_sách_từ_khóa_cảnh_báo_mong_đợi).
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    fs_correct = cfg.SCOPE["sampling_rate_hz"]

    cases = [
        # (load_hp, filename, fs_thật_để_sinh_tín_hiệu, rpm, mô_tả)
        (0, "Normal", 48000, 1797),                  # (A) sampling rate thật 48kHz
        (0, "OR_007_Orthogonal", fs_correct, 1797),   # (B) OR ngoài phạm vi đã chốt
        (0, "B_028", fs_correct, 1797),               # (C) đường kính 28 mils -> NTN
        (1, "IR_014", fs_correct, 1650),              # (D) RPM lệch xa danh định (đúng 1797/1772/1750/1730)
    ]
    for load_hp, fname, fs, rpm in cases:
        folder = root / f"{load_hp}hp"
        folder.mkdir(exist_ok=True)
        _, x = make_synthetic_signal(fs, 10.0, rpm, seed=hash(fname) % 1000)
        savemat(str(folder / f"{fname}.mat"), {"X999_DE_time": x.reshape(-1, 1),
                                                "X999RPM": np.array([[rpm]])})

    expected_warning_keywords = [
        "NGHI_NGO_SAMPLING_RATE", "OR_NGOAI_PHAM_VI", "VONG_BI_NTN", "RPM_LECH",
    ]
    return root, expected_warning_keywords
