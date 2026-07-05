import os
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Dict, List, Tuple

from src import config
from src.dsp_utils import extract_time_features, extract_freq_features, get_freq_feature_names

class CWRUDataPipeline:
    def __init__(self, demod_type: str = 'hilbert'):
        self.step_size = int(config.WINDOW_SIZE * (1 - config.OVERLAP))
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        self.freq_feature_names = get_freq_feature_names(config.FREQ_BAND_EDGES)
        self.demod_type = demod_type  # 'hilbert' hoặc 'square_law'

    def _calculate_fault_freqs(self, motor_load_hp: int) -> Dict[str, float]:
        rpm = config.LOAD_TO_RPM.get(motor_load_hp, 1797)
        fr = rpm / 60.0
        return {
            'BPFI': config.BEARING_GEOMETRY['BPFI'] * fr,
            'BPFO': config.BEARING_GEOMETRY['BPFO'] * fr,
            'BSF': config.BEARING_GEOMETRY['BSF'] * fr
        }

    def _load_raw_signal(self, file_path: str) -> np.ndarray:
        mat_data = sio.loadmat(file_path)
        matched = [k for k in mat_data.keys() if '_DE_time' in k]
        if not matched:
            raise ValueError(f"Không tìm thấy biến '_DE_time' trong {file_path}")
        return mat_data[matched[0]].flatten()

    def process_single_file(self, file_path: str, label: int, motor_load_hp: int) -> pd.DataFrame:
        try:
            raw_signal = self._load_raw_signal(file_path)
        except Exception as e:
            logging.error(f"Lỗi đọc {file_path}: {e}")
            return pd.DataFrame()

        fault_freqs = self._calculate_fault_freqs(motor_load_hp)
        features_list = []

        for start in range(0, len(raw_signal) - config.WINDOW_SIZE + 1, self.step_size):
            segment = raw_signal[start : start + config.WINDOW_SIZE]
            segment = segment - np.mean(segment)

            time_feats = extract_time_features(segment)
            freq_feats = extract_freq_features(
                segment, config.FS, config.WINDOW_SIZE,
                config.FREQ_BAND_EDGES, fault_freqs,
                demod_type=self.demod_type
            )
            features_list.append(time_feats + freq_feats + [label, motor_load_hp])

        cols = ['RMS', 'Peak', 'Kurtosis', 'Skewness', 'Crest_Factor', 'Shape_Factor']
        cols += self.freq_feature_names + ['Label', 'Load']
        return pd.DataFrame(features_list, columns=cols)


# ── CÁC HÀM PHÂN CHIA DỮ LIỆU THỰC NGHIỆM ─────────────────────────
def split_files_random(file_list, train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO, seed=config.RANDOM_SEED):
    """Chia File-Random Split (Dành cho Kịch bản A2/D2)."""
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in file_list:
        groups[(item[1], item[2])].append(item)

    train_files, val_files, test_files = [], [], []
    for key, items in groups.items():
        items = items.copy()
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_train + n_val >= n: n_train = max(1, n - 2); n_val = 1
        train_files.extend(items[:n_train])
        val_files.extend(items[n_train:n_train + n_val])
        test_files.extend(items[n_train + n_val:])
    return train_files, val_files, test_files


def split_files_lolo(file_list, test_load_hp: int) -> Tuple[List, List]:
    """
    Chia Leave-One-Load-Out (Dành cho A3, B, C, D3).
    Giấu nguyên 1 mức tải làm tập TEST độc lập.
    Trả về: (train_val_files, test_files)
    """
    train_val_files = [f for f in file_list if f[2] != test_load_hp]
    test_files = [f for f in file_list if f[2] == test_load_hp]
    return train_val_files, test_files