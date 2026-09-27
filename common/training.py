# -*- coding: utf-8 -*-
"""
common/training.py
====================
Orchestrator huấn luyện/đánh giá estimator qua N lần lặp (N seeds). 
Tổng hợp kết quả thống kê (mean ± std) và tự động trích xuất mô hình tốt nhất 
để xuất xuống MCU.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .splitting import resolve_label_col

def run_n_seed_evaluation(estimator_factory, splits_generator, feature_cols, label_col=None):
    """Vòng lặp chính. Khởi tạo và fit mô hình mới trên từng bộ Train/Val của mỗi seed. 
    Đánh giá trên tập Test tương ứng và trả về list mô hình cùng kết quả metric."""
    models_list = []
    metrics_list = []
    
    for seed, train_df, val_df, test_df in splits_generator:
        label_col_resolved = resolve_label_col(train_df, label_col)
        
        # Khởi tạo mô hình mới cho mỗi seed
        estimator = estimator_factory()
        
        X_train, y_train = train_df[feature_cols], train_df[label_col_resolved]
        X_test, y_test = test_df[feature_cols], test_df[label_col_resolved]
        
        # Fit estimator (Giả định tuân theo interface scikit-learn. Nếu dùng Keras, 
        # bạn có thể cần gói model lại hoặc truyền thêm tham số epoch/validation_data BÊN NGOÀI)
        estimator.fit(X_train, y_train)
        
        # Đánh giá trên tập Test không nhìn thấy
        y_pred = estimator.predict(X_test)
        
        # Nếu model Keras trả về xác suất softmax, cần argmax (bạn có thể xử lý trong adapter/wrapper)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        
        models_list.append(estimator)
        metrics_list.append({
            "seed": seed,
            "accuracy": acc,
            "f1_macro": f1
        })
        
    return models_list, metrics_list

def format_summary(metrics_list: list, metric: str = "f1_macro") -> str:
    """Tổng hợp list các accuracy hoặc macro-F1 của N seed thành định dạng 'mean ± std'."""
    if not metrics_list:
        return "N/A"
    
    values = [m[metric] for m in metrics_list]
    mean_val = np.mean(values)
    std_val = np.std(values)
    return f"{mean_val:.4f} ± {std_val:.4f}"

def extract_best_model(models_list: list, metrics_list: list, metric: str = "f1_macro"):
    """So sánh kết quả của N seed, chọn ra 1 mô hình có metric trên tập Test 
    cao nhất để đại diện mang đi lượng tử hóa INT8."""
    if not models_list or not metrics_list:
        raise ValueError("Danh sách mô hình hoặc metrics trống. Hãy chạy run_n_seed_evaluation trước.")
        
    values = [m[metric] for m in metrics_list]
    best_idx = int(np.argmax(values))
    best_seed = metrics_list[best_idx]['seed']
    best_score = values[best_idx]
    
    print(f"[extract_best_model] Đã chọn mô hình đại diện từ seed {best_seed} với {metric} = {best_score:.4f}")
    return models_list[best_idx], best_seed