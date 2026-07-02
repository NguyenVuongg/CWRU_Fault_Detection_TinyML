import os
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Dict

from src import config
from src.dsp_utils import extract_time_features, extract_freq_features, get_freq_feature_names


class CWRUDataPipeline:
    """
    QUAN TRỌNG: Class này chỉ xử lý 1 FILE thành 1 DataFrame đầy đủ
    (không tự chia train/val/test bên trong).

    Lý do: nếu chia theo thời gian trong 1 file (70% đầu -> train,
    15% giữa -> val, 15% cuối -> test), các window ở ranh giới sẽ
    chồng lấp lẫn nhau (do OVERLAP=0.5) -> data leakage giữa các tập.

    Quyết định file nào thuộc train/val/test phải làm TRƯỚC khi gọi
    class này, ở cấp độ file (xem main.py / split_files()).
    """

    def __init__(self):
        self.step_size = int(config.WINDOW_SIZE * (1 - config.OVERLAP))
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        self.freq_feature_names = get_freq_feature_names(config.FREQ_BAND_EDGES)

    def _calculate_fault_freqs(self, motor_load_hp: int) -> Dict[str, float]:
        """Tính tần số lỗi (Hz) dựa trên RPM thực tế của từng mức tải.

        Quan trọng: BPFI/BPFO/BSF thay đổi theo RPM, nên phải tính
        riêng cho từng file theo đúng motor_load_hp của file đó,
        không dùng chung 1 giá trị cho mọi file.
        """
        rpm = config.LOAD_TO_RPM.get(motor_load_hp, 1797)
        fr = rpm / 60.0  # Tần số quay (Hz)
        return {
            'BPFI': config.BEARING_GEOMETRY['BPFI'] * fr,
            'BPFO': config.BEARING_GEOMETRY['BPFO'] * fr,
            'BSF': config.BEARING_GEOMETRY['BSF'] * fr
        }

    def _load_raw_signal(self, file_path: str) -> np.ndarray:
        """Đọc tín hiệu Drive-End từ file .mat."""
        mat_data = sio.loadmat(file_path)
        matched = [k for k in mat_data.keys() if '_DE_time' in k]
        if len(matched) == 0:
            raise ValueError(f"Không tìm thấy biến '_DE_time' trong {file_path}")
        if len(matched) > 1:
            logging.warning(f"{file_path}: tìm thấy {len(matched)} biến '_DE_time', dùng biến đầu: {matched}")
        return mat_data[matched[0]].flatten()

    def process_single_file(self, file_path: str, label: int, motor_load_hp: int) -> pd.DataFrame:
        """
        Xử lý TOÀN BỘ 1 file .mat thành 1 DataFrame feature.
        Mỗi dòng = 1 window. Không chia tập ở đây.
        """
        try:
            raw_signal = self._load_raw_signal(file_path)
        except Exception as e:
            logging.error(f"Lỗi khi đọc {file_path}: {e}")
            return pd.DataFrame()

        fault_freqs = self._calculate_fault_freqs(motor_load_hp)

        features_list = []
        for start in range(0, len(raw_signal) - config.WINDOW_SIZE + 1, self.step_size):
            segment = raw_signal[start: start + config.WINDOW_SIZE]
            segment = segment - np.mean(segment)  # Khử nhiễu DC

            time_feats = extract_time_features(segment)
            freq_feats = extract_freq_features(
                segment, config.FS, config.WINDOW_SIZE,
                config.FREQ_BAND_EDGES, fault_freqs
            )

            features_list.append(time_feats + freq_feats + [label, motor_load_hp])

        cols = ['RMS', 'Peak', 'Kurtosis', 'Skewness', 'Crest_Factor', 'Shape_Factor']
        cols += self.freq_feature_names
        cols += ['Label', 'Load']

        logging.info(f"Đã xử lý: {os.path.basename(file_path)} -> {len(features_list)} windows")
        return pd.DataFrame(features_list, columns=cols)


def split_files(file_list, train_ratio=None, val_ratio=None, seed=None):
    """
    Chia danh sách FILE (không phải window) thành train/val/test.
    Chia THỦ CÔNG theo từng nhóm (label, load) — không dùng sklearn
    train_test_split với stratify 2 lần liên tiếp, vì khi số file mỗi
    nhóm nhỏ (ví dụ 4 file/nhóm), lần chia thứ 2 (val/test trên phần
    còn lại của mỗi nhóm) dễ không đủ mẫu để stratify -> lỗi ValueError.

    Cách làm: với mỗi nhóm (label, load), xáo trộn rồi cắt theo tỉ lệ,
    đảm bảo mỗi tập có ít nhất 1 file nếu nhóm đủ lớn.

    file_list: list các tuple (filename, label, load_hp)
    Trả về: (train_files, val_files, test_files) — cùng định dạng.
    """
    import random
    from collections import defaultdict

    train_ratio = train_ratio or config.TRAIN_RATIO
    val_ratio = val_ratio or config.VAL_RATIO
    seed = seed if seed is not None else config.RANDOM_SEED

    rng = random.Random(seed)

    groups = defaultdict(list)
    for item in file_list:
        _, label, load = item
        groups[(label, load)].append(item)

    train_files, val_files, test_files = [], [], []
    small_groups = []

    for key, items in groups.items():
        items = items.copy()
        rng.shuffle(items)
        n = len(items)

        if n < 3:
            # Nhóm quá nhỏ để chia đủ 3 tập -> đưa hết vào train,
            # ghi log để người dùng biết dataset thiếu file ở nhóm này
            train_files.extend(items)
            small_groups.append((key, n))
            continue

        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        # Đảm bảo còn ít nhất 1 file cho test
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_files.extend(items[:n_train])
        val_files.extend(items[n_train:n_train + n_val])
        test_files.extend(items[n_train + n_val:])

    if small_groups:
        logging.warning(
            f"{len(small_groups)} nhóm (label,load) có < 3 file, đã đưa "
            f"toàn bộ vào TRAIN (không chia val/test cho nhóm đó): "
            f"{small_groups}"
        )

    return train_files, val_files, test_files