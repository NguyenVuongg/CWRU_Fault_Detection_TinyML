# -*- coding: utf-8 -*-
"""
common/features.py
===================
Trích đặc trưng đơn giản + sliding window.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def make_sliding_windows(signal, window_size, overlap_ratio, file_id, label,
                         warmup_samples=0):
    """Cắt tín hiệu dài thành các sliding window có overlap, gắn kèm file_id.

    HỢP ĐỒNG VỀ file_id: đây là định danh FILE GỐC và KHÔNG được chứa chỉ
    số cửa sổ. Quá trình chia File-based Split tách tập theo đúng cột này.
    Thông tin "cửa sổ thứ mấy" nằm ở window_idx, vị trí mẫu nằm ở start_idx.

    window_idx và start_idx là METADATA: không được đưa vào X khi huấn luyện.
    """
    step = max(int(window_size * (1 - overlap_ratio)), 1)
    starts = range(0, len(signal) - window_size + 1, step)
    rows = []
    
    for window_idx, start in enumerate(starts):
        if start < warmup_samples:
            continue
        window = signal[start: start + window_size]
        rows.append({"file_id": file_id, "label": label,
                     "window_idx": window_idx, "start_idx": start,
                     "window": window})
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
        [windows_df[["file_id", "label", "window_idx", "start_idx"]].reset_index(drop=True),
         feat_df.reset_index(drop=True)], axis=1,
    )