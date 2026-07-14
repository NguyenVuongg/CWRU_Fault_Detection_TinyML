# -*- coding: utf-8 -*-
"""
common/splitting.py
====================
Random Window Split vs File-based Split (mục 0.3) + cấu trúc Leave-One-
Load-Out (mục 2.1 — định nghĩa 1 lần ở đây, dùng lại nguyên vẹn ở Giai
đoạn 1-2 để đảm bảo nhất quán xuyên suốt đề tài).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def random_window_split(feature_df, test_ratio=0.3, seed=42):
    """SAI CÁCH LÀM (cố ý) — chia ngẫu nhiên theo dòng window, bỏ qua
    file_id. Windows từ cùng 1 file có thể rơi cả vào train và test."""
    rng = np.random.RandomState(seed)
    idx = feature_df.index.to_numpy().copy()
    rng.shuffle(idx)
    n_test = int(len(idx) * test_ratio)
    return feature_df.loc[idx[n_test:]], feature_df.loc[idx[:n_test]]


def file_based_split(feature_df, test_ratio=0.3, seed=42):
    """ĐÚNG CÁCH LÀM — chia theo file_id trước. Không file nào vừa ở
    train vừa ở test."""
    rng = np.random.RandomState(seed)
    file_ids = np.array(feature_df["file_id"].unique().tolist(), dtype=object)
    rng.shuffle(file_ids)
    n_test_files = max(1, int(len(file_ids) * test_ratio))
    test_files = set(file_ids[:n_test_files])
    train_mask = ~feature_df["file_id"].isin(test_files)
    return feature_df[train_mask], feature_df[~train_mask]


def run_split_comparison_experiment(feature_df, feature_cols, seed=42):
    """Huấn luyện Random Forest với 2 cách chia, trả về bảng so sánh —
    bằng chứng thực nghiệm sơ bộ cho RQ1/H1."""
    results = {}
    for name, split_fn in [
        ("Random Window Split (SAI)", random_window_split),
        ("File-based Split (ĐÚNG)", file_based_split),
    ]:
        train_df, test_df = split_fn(feature_df, seed=seed)
        X_train, y_train = train_df[feature_cols], train_df["label"]
        X_test, y_test = test_df[feature_cols], test_df["label"]

        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
    return pd.DataFrame(results).T


def generate_lolo_folds(loads=(0, 1, 2, 3)):
    """Cấu trúc 4 fold LOLO đúng mục 2.1. Trả về list dict
    {'test_load': X, 'trainval_loads': [...]}. IMPORT LẠI hàm này ở
    Giai đoạn 1-2, không viết lại lần thứ hai."""
    return [{"test_load": t, "trainval_loads": [l for l in loads if l != t]} for t in loads]
