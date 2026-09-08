# -*- coding: utf-8 -*-
"""
common/synthetic.py
=====================
"""

import hashlib
import shutil
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import lfilter

from . import config as cfg


def stable_seed(text, modulo: int = 2 ** 31 - 1) -> int:
    """Seed ỔN ĐỊNH GIỮA CÁC PHIÊN sinh từ một chuỗi.

    hash() của Python bị salt ngẫu nhiên theo từng tiến trình (xem
    PYTHONHASHSEED), nên hash(fname) % 1000 tạo ra dữ liệu giả KHÁC
    NHAU mỗi lần chạy: một ca kiểm thử vừa thất bại sẽ không tái lập
    được, và bug của run_sanity_checks() có thể biến mất khi chạy lại.
    md5 cho cùng một giá trị mãi mãi trên mọi máy, mọi phiên bản.

    Dải rộng 2^31-1 thay vì 1000 để giảm nguy cơ hai tên file khác
    nhau nhận cùng seed (sinh ra hai file trùng khít nội dung, làm
    thí nghiệm tự kiểm chứng mất hiệu lực).
    """
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


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
        x += 1.5 * np.asarray(lfilter(decay, [1.0], impulse_train))

    return t, x


# Tần số lỗi giả lập gần đúng cho mỗi nhãn tại RPM danh định — chỉ dùng để
# tạo dữ liệu demo có "hình dạng" hợp lý, KHÔNG phải số liệu CWRU thật.
_FAULT_LABEL_TO_FREQ_KEY = {"IR": "BPFI", "OR": "BPFO", "B": "BSF"}

# Cấu trúc thư mục khớp ĐÚNG dữ liệu thật đang dùng (xem parser trong
# common/io_utils.py): <root>/12k_Drive_End_Bearing_Fault_Data/...
_TOP_FOLDER = "12k_Drive_End_Bearing_Fault_Data"


def build_synthetic_dataset(root: Path, loads=(0, 1, 2, 3), seed=0,
                             diameters_mils=(7, 14, 21), duration_sec=10.0):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    top = root / _TOP_FOLDER
    top.mkdir(parents=True)

    rng_seed = seed
    file_id = 100
    fs = cfg.SCOPE["sampling_rate_hz"]

    for load in loads:
        rpm = cfg.NOMINAL_RPM_BY_LOAD[load]
        fault_freqs = cfg.bearing_fault_frequencies(rpm)

        # Normal — đặt trực tiếp dưới root/, NGANG HÀNG với _TOP_FOLDER,
        # không lồng bên trong (xem giải thích trong docstring ở trên).
        _, x = make_synthetic_signal(fs, duration_sec, rpm, seed=rng_seed)
        normal_dir = root / "Normal"
        normal_dir.mkdir(parents=True, exist_ok=True)
        savemat(str(normal_dir / f"{file_id}_Normal_{load}.mat"),
                {"X999_DE_time": x.reshape(-1, 1), "X999RPM": np.array([[rpm]])})
        rng_seed += 1
        file_id += 1

        # IR / OR / B tại từng đường kính
        for label, freq_key in _FAULT_LABEL_TO_FREQ_KEY.items():
            for diam in diameters_mils:
                _, x = make_synthetic_signal(
                    fs, duration_sec, rpm, fault_freq_hz=fault_freqs[freq_key], seed=rng_seed,
                )
                if label == "OR":
                    fault_dir = top / label / f"{diam:03d}" / "@6"
                else:
                    fault_dir = top / label / f"{diam:03d}"
                fault_dir.mkdir(parents=True, exist_ok=True)
                savemat(str(fault_dir / f"{file_id}_{load}.mat"),
                        {"X999_DE_time": x.reshape(-1, 1), "X999RPM": np.array([[rpm]])})
                rng_seed += 1
                file_id += 1

    return root


def build_edge_case_dataset(root: Path):
    """
    Tạo 6 file .mat GIẢ LẬP, mỗi file cố ý gài đúng 1 loại lỗi mà
    common/io_utils.run_sanity_checks() phải bắt được — cấu trúc thư mục
    khớp ĐÚNG dữ liệu thật (xem build_synthetic_dataset()). Dùng để TỰ
    KIỂM CHỨNG pipeline (notebook 01, mục cuối), không dùng cho phân tích
    ở các notebook khác.

    Trả về (root, danh_sách_từ_khóa_cảnh_báo_mong_đợi).
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    top = root / _TOP_FOLDER
    top.mkdir(parents=True)
    fs_correct = cfg.SCOPE["sampling_rate_hz"]

    def save(rel_dir: str, fname: str, fs: float, rpm: float, category: str = _TOP_FOLDER):
        d = root / category / rel_dir
        d.mkdir(parents=True, exist_ok=True)
        _, x = make_synthetic_signal(fs, 10.0, rpm, seed=stable_seed(fname))
        savemat(str(d / f"{fname}.mat"), {"X999_DE_time": x.reshape(-1, 1),
                                           "X999RPM": np.array([[rpm]])})

    # (A) sampling rate thật là 48kHz dù thư mục Normal/ không khai báo tần
    # số nào qua tên (đúng cấu trúc thật: Normal/ ngang hàng với _TOP_FOLDER,
    # KHÔNG lồng bên trong — category="" để save() đặt file trực tiếp dưới
    # root/, không qua _TOP_FOLDER. Đây chính là ca thật cần test: file
    # KHÔNG có source_category, buộc phải đi qua nhánh fallback fs của
    # io_utils.run_sanity_checks (không phải qua nhánh so khớp declared_khz
    # như khi lồng nhầm trong _TOP_FOLDER).
    save("Normal", "500_Normal_0", fs=48000, rpm=1797, category="")
    # (B) OR ngoài phạm vi đã chốt (Orthogonal, không phải Centered)
    save("OR/007/@3", "501_0", fs=fs_correct, rpm=1797)
    # (C) đường kính 28 mils -> vòng bi NTN
    save("B/028", "502_0", fs=fs_correct, rpm=1797)
    # (D) RPM lệch xa danh định (đúng phải là 1772 cho tải 1HP)
    save("IR/014", "503_1", fs=fs_correct, rpm=1650)
    # (E) mọi thứ hợp lệ (đúng tần số, đúng RPM, đúng đường kính) NHƯNG đặt
    #     ở Fan-End thay vì Drive-End -> cô lập đúng 1 biến, chỉ nên kích
    #     hoạt NGOAI_PHAM_VI_CAM_BIEN, không kèm cảnh báo nào khác.
    save("IR/007", "504_0", fs=fs_correct, rpm=1797,
         category="12k_Fan_End_Bearing_Fault_Data")
    # (F) đặt ở 48k_Drive_End, và sinh ĐÚNG 480.000 mẫu thật (10.0s @
    #     48kHz) -- KHÔNG được để lệch thời lượng, nếu không sẽ dính thêm
    #     NGHI_NGO_SAMPLING_RATE/THOI_LUONG_BAT_THUONG, không còn "sạch" để
    #     cô lập riêng NGOAI_PHAM_VI_TAN_SO_KHAI_BAO. Cách cô lập: check
    #     thời lượng ở io_utils.py giờ so với đúng tần số KHAI BÁO của
    #     chính category này (48kHz) chứ không mù quáng thử 12k/48k, nên
    #     file "thật" 48kHz nằm đúng thư mục 48k sẽ pass check đó êm.
    save("B/007", "505_0", fs=48000, rpm=1797,
         category="48k_Drive_End_Bearing_Fault_Data")

    expected_warning_keywords = [
        "NGHI_NGO_SAMPLING_RATE", "OR_NGOAI_PHAM_VI", "VONG_BI_NTN", "RPM_LECH",
        "NGOAI_PHAM_VI_CAM_BIEN", "NGOAI_PHAM_VI_TAN_SO_KHAI_BAO",
    ]
    return root, expected_warning_keywords