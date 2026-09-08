# -*- coding: utf-8 -*-
"""
common/training.py
====================
[Giai đoạn 2 — mục 2.1] Orchestrator chạy đánh giá qua 4 fold LOLO.

Dùng lại NGUYÊN VẸN `splitting.generate_lolo_folds()` và
`splitting.file_based_split()` đã viết và test ở Giai đoạn 0 — không viết
lại lần thứ hai, tránh lệch cấu trúc giữa các giai đoạn (đúng cam kết ghi
trong common/splitting.py và notebooks/giai_doan_0/07_lolo_visualization.ipynb).

Cung cấp 2 mức API:
  - iterate_lolo_splits(): generator mức THẤP, trả về train/val/test
    DataFrame của từng fold — dùng khi cần vòng lặp huấn luyện tùy chỉnh
    (vd Keras .fit() với callback, early stopping — xem Giai đoạn 2.3).
  - run_lolo_evaluation(): API mức CAO, nhận vào 1 "estimator_factory"
    kiểu scikit-learn (.fit/.predict) — phù hợp Random Forest baseline
    (mục 2.1 khuyến nghị chạy trước) hoặc bất kỳ estimator nào tuân theo
    interface này (kể cả Keras model bọc qua lớp adapter, xem models.py).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .splitting import file_based_split, generate_lolo_folds, resolve_label_col


def iterate_lolo_splits(feature_df: pd.DataFrame, load_col: str = "load_hp",
                         val_ratio: float = 0.2, seed: int = 42,
                         loads=(0, 1, 2, 3), stratify_col=None):
    """
    Generator: với mỗi fold LOLO, yield (fold_info, train_df, val_df, test_df).

    feature_df BẮT BUỘC có cột 'file_id' (đúng tên, không đổi được qua
    tham số) — khớp với output của features_full.build_full_feature_table()
    và với chữ ký hàm splitting.file_based_split().
    """
    if "file_id" not in feature_df.columns:
        raise ValueError(
            "feature_df cần có cột 'file_id' để file_based_split() hoạt "
            "động đúng — xem features_full.build_full_feature_table()."
        )

    for fold in generate_lolo_folds(loads=loads):
        test_load = fold["test_load"]
        trainval_loads = fold["trainval_loads"]

        test_df = feature_df[feature_df[load_col] == test_load].reset_index(drop=True)
        trainval_df = feature_df[feature_df[load_col].isin(trainval_loads)].reset_index(drop=True)

        # stratify_col=None -> file_based_split tự lấy cột nhãn 10 lớp
        # ("class_label"), đúng "Stratified File-based Split 80/20" của mục 2.1.
        train_df, val_df = file_based_split(trainval_df, test_ratio=val_ratio,
                                            seed=seed, stratify_col=stratify_col)

        fold_info = {
            "fold_name": f"test_load_{test_load}",
            "test_load": test_load,
            "trainval_loads": trainval_loads,
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        }
        yield fold_info, train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


def run_lolo_evaluation(feature_df: pd.DataFrame, feature_cols, estimator_factory,
                         label_col=None, load_col: str = "load_hp",
                         val_ratio: float = 0.2, seed: int = 42, loads=(0, 1, 2, 3),
                         use_val_for_fit: bool = False):
    """
    Chạy 1 estimator qua 4 fold LOLO, trả về (per_fold_df, summary_dict).

    estimator_factory: callable() -> object có .fit(X, y) và .predict(X)
    kiểu scikit-learn. Gọi lại 1 lần MỚI cho mỗi fold (không dùng chung 1
    instance đã fit từ fold trước).

    use_val_for_fit=False (mặc định): chỉ fit trên train_df, val_df CHỈ
    dùng để bạn tự chọn hyperparameter BÊN NGOÀI hàm này (vd grid search
    trước khi gọi run_lolo_evaluation với estimator_factory đã cố định
    hyperparameter tốt nhất) — đúng quy trình mục 2.1: "lựa chọn siêu
    tham số dựa trên tập Validation, đánh giá trên tập Test".

    label_col=None (mặc định) -> dùng cột "class_label" của sơ đồ 10 LỚP,
    đúng bài toán chính đã chốt ở mục 1.1. Truyền label_col="label" khi muốn
    chạy phân tích bổ trợ 4 lớp để đối chiếu.
    """
    # label_col=None -> "class_label" (10 lớp, mục 1.1). Cột nhãn được dùng
    # luôn làm cột stratify của bước chia Train/Val bên trong mỗi fold.
    label_col = resolve_label_col(feature_df, label_col)

    rows = []
    for fold_info, train_df, val_df, test_df in iterate_lolo_splits(
        feature_df, load_col=load_col, val_ratio=val_ratio, seed=seed, loads=loads,
        stratify_col=label_col,
    ):
        fit_df = pd.concat([train_df, val_df], ignore_index=True) if use_val_for_fit else train_df

        estimator = estimator_factory()
        estimator.fit(fit_df[feature_cols], fit_df[label_col])
        y_pred = estimator.predict(test_df[feature_cols])

        rows.append({
            **fold_info,
            "accuracy": accuracy_score(test_df[label_col], y_pred),
            "f1_macro": f1_score(test_df[label_col], y_pred, average="macro"),
        })

    per_fold_df = pd.DataFrame(rows)
    summary = {
        "accuracy_mean": per_fold_df["accuracy"].mean(),
        "accuracy_std": per_fold_df["accuracy"].std(),
        "f1_macro_mean": per_fold_df["f1_macro"].mean(),
        "f1_macro_std": per_fold_df["f1_macro"].std(),
        "n_folds": len(per_fold_df),
    }
    return per_fold_df, summary


def format_summary(summary: dict, metric: str = "accuracy") -> str:
    """In dạng 'mean ± std' đúng chuẩn báo cáo mục 2.1."""
    return f"{summary[f'{metric}_mean']:.4f} ± {summary[f'{metric}_std']:.4f}"
