# -*- coding: utf-8 -*-
"""
common/features.py
===================
Trích đặc trưng đơn giản + sliding window — dùng cho thực nghiệm minh họa
ở mục 0.3 (KHÔNG phải bể đặc trưng đầy đủ của Giai đoạn 1, xem mục 1.4).
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def make_sliding_windows(signal, window_size, overlap_ratio, file_id, label):
    """Cắt tín hiệu dài thành các sliding window có overlap, gắn kèm
    file_id — cột này chính là thứ File-based Split dùng để tách."""
    step = max(int(window_size * (1 - overlap_ratio)), 1)
    starts = range(0, len(signal) - window_size, step)
    rows = []
    for start in starts:
        window = signal[start: start + window_size]
        rows.append({"file_id": file_id, "label": label, "start_idx": start, "window": window})
    return pd.DataFrame(rows)


def extract_simple_features(window):
    return {
        "rms": np.sqrt(np.mean(window ** 2)),
        "kurtosis": kurtosis(window),
        "skewness": skew(window),
        "peak": np.max(np.abs(window)),
        "std": np.std(window),
    }


def build_feature_table(windows_df):
    feat_rows = windows_df["window"].apply(extract_simple_features)
    feat_df = pd.DataFrame(list(feat_rows))
    return pd.concat(
        [windows_df[["file_id", "label", "start_idx"]].reset_index(drop=True),
         feat_df.reset_index(drop=True)], axis=1,
    )
