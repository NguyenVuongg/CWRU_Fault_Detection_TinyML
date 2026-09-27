# -*- coding: utf-8 -*-
"""
common/splitting.py
====================
Chia dữ liệu File-based kết hợp lặp N-seed trên toàn bộ các mức tải để hạn chế 
rò rỉ dữ liệu và đảm bảo độ tin cậy thống kê (mean ± std).
"""

import numpy as np
import pandas as pd

from .config import CLASS_SCHEME_DEFAULT, LABEL_COL_BY_SCHEME

def resolve_label_col(feature_df: pd.DataFrame, label_col=None,
                      scheme: str = CLASS_SCHEME_DEFAULT) -> str:
    """Chọn cột nhãn theo sơ đồ lớp đã chốt ở mục 1.1."""
    if label_col is not None:
        if label_col not in feature_df.columns:
            raise ValueError(
                f"Không có cột nhãn '{label_col}' trong bảng đặc trưng. "
                f"Các cột đang có: {list(feature_df.columns)[:12]}..."
            )
        return label_col

    preferred = LABEL_COL_BY_SCHEME[scheme]
    if preferred in feature_df.columns:
        return preferred

    fallback = LABEL_COL_BY_SCHEME["4class"]
    if fallback in feature_df.columns:
        print(
            f"[resolve_label_col] CẢNH BÁO: bảng đặc trưng không có cột "
            f"'{preferred}' nên tạm dùng '{fallback}' (4 lớp). Đề cương mục "
            f"1.1 chốt bài toán chính là 10 lớp — hãy build lại bảng bằng "
            f"features_full.build_full_feature_table() phiên bản mới."
        )
        return fallback

    raise ValueError(
        f"Bảng đặc trưng không có cả '{preferred}' lẫn '{fallback}'."
    )

def file_based_split_single_seed(df: pd.DataFrame, val_ratio: float = 0.15, 
                                 test_ratio: float = 0.15, random_state: int = 42, 
                                 stratify_col=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cắt Train/Val/Test theo file_id cho một seed duy nhất, đảm bảo stratify theo nhãn 
    và file không bị vắt ngang qua các tập."""
    stratify_col = resolve_label_col(df, stratify_col)
    rng = np.random.RandomState(random_state)
    
    file_label = (df[["file_id", stratify_col]]
                  .drop_duplicates("file_id")
                  .set_index("file_id")[stratify_col])
    
    train_files, val_files, test_files = [], [], []
    
    for _, group in file_label.groupby(file_label):
        ids = group.index.to_numpy().copy()
        rng.shuffle(ids)
        n_total = len(ids)
        
        if n_total <= 1:
            # Giữ file duy nhất ở train để tránh mất trắng nhãn đó trong lúc huấn luyện
            train_files.extend(ids) 
        else:
            n_test = max(1, int(round(n_total * test_ratio)))
            n_val = max(1, int(round(n_total * val_ratio)))
            
            # Đảm bảo luôn chừa lại ít nhất 1 file cho tập Train
            if n_total - n_test - n_val < 1:
                n_val = 0
                n_test = max(1, n_total - 1)
                
            test_files.extend(ids[:n_test])
            val_files.extend(ids[n_test:n_test+n_val])
            train_files.extend(ids[n_test+n_val:])
            
    train_df = df[df["file_id"].isin(train_files)].reset_index(drop=True)
    val_df = df[df["file_id"].isin(val_files)].reset_index(drop=True)
    test_df = df[df["file_id"].isin(test_files)].reset_index(drop=True)
    
    return train_df, val_df, test_df

def generate_n_seed_splits(df: pd.DataFrame, n_seeds: int = 5, 
                           val_ratio: float = 0.15, test_ratio: float = 0.15,
                           start_seed: int = 42, stratify_col=None):
    """Hàm generator gọi file_based_split_single_seed N lần với các seed khác nhau, 
    trả về danh sách các tập (Train, Val, Test) tương ứng với từng seed."""
    if "file_id" not in df.columns:
        raise ValueError("DataFrame cần có cột 'file_id' để phân chia dữ liệu.")
        
    for i in range(n_seeds):
        seed = start_seed + i
        train_df, val_df, test_df = file_based_split_single_seed(
            df, val_ratio=val_ratio, test_ratio=test_ratio, 
            random_state=seed, stratify_col=stratify_col
        )
        yield seed, train_df, val_df, test_df