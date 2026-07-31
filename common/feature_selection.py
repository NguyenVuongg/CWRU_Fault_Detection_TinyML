# -*- coding: utf-8 -*-
"""
common/feature_selection.py
=============================
[Giai đoạn 2 — mục 2.2] Rút gọn đặc trưng giải thích được.

  - Random Forest Importance (tiêu chí chính để chọn top-k)
  - Permutation Importance (đối chiếu, giảm thiên lệch của RF Importance
    với đặc trưng có nhiều giá trị khác nhau)
  - Jaccard stability giữa các fold LOLO (xác nhận bộ đặc trưng chọn ra
    không phải nhiễu ngẫu nhiên của riêng 1 fold)
"""

import numpy as np
import pandas as pd
from typing import cast
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch


def compute_rf_importance(X: pd.DataFrame, y, seed: int = 42, n_estimators: int = 300) -> pd.Series:
    """Random Forest Importance, sắp xếp giảm dần."""
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X, y)
    return pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)


def compute_permutation_importance(X: pd.DataFrame, y, seed: int = 42,
                                    n_repeats: int = 10, n_estimators: int = 300) -> pd.Series:
    """Permutation Importance — huấn luyện RF riêng rồi đo importance bằng
    cách xáo trộn từng cột, đối chiếu với compute_rf_importance()."""
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X, y)
    # Không truyền `scoring=` (mặc định None) -> sklearn LUÔN trả về 1 Bunch
    # duy nhất (không phải dict[str, Bunch], chỉ xảy ra khi `scoring` là
    # list/dict nhiều scorer — không phải trường hợp ở đây). sklearn không
    # tự khai báo kiểu trả về nên Pylance suy luận thành Union — ép kiểu
    # tường minh bằng cast() vì ta biết chắc nhánh runtime nào xảy ra.
    result = cast(Bunch, permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=seed))
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def select_top_k_features(importance_series: pd.Series, k: int = 30) -> list:
    """Trả về danh sách tên k đặc trưng có importance cao nhất."""
    return importance_series.head(k).index.tolist()


def jaccard_index(set_a, set_b) -> float:
    """Jaccard = |A ∩ B| / |A ∪ B|."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_stability_across_folds(list_of_feature_sets) -> pd.DataFrame:
    """
    Ma trận Jaccard giữa mọi cặp fold — đo độ ổn định của bộ đặc trưng
    chọn ra qua các fold LOLO (mục 2.2). Giá trị gần 1 = ổn định cao,
    gần 0 = bộ đặc trưng chọn ra khác nhau nhiều giữa các fold (đáng ngờ).
    """
    n = len(list_of_feature_sets)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = jaccard_index(list_of_feature_sets[i], list_of_feature_sets[j])
    labels = [f"fold_{i+1}" for i in range(n)]
    return pd.DataFrame(mat, index=labels, columns=labels)


def select_features_per_lolo_fold(feature_df: pd.DataFrame, feature_cols: list[str],
                                   label_col: str = "label", load_col: str = "load_hp",
                                   loads=(0, 1, 2, 3), k: int = 30, seed: int = 42):
    """
    Chạy Feature Selection RIÊNG cho từng fold LOLO (fit RF Importance chỉ
    trên phần Train/Val của fold đó, KHÔNG dùng Test — tránh rò rỉ thông
    tin chọn đặc trưng từ tập Test), rồi tính Jaccard stability giữa các
    fold. Trả về (dict fold_name -> list top-k features, DataFrame Jaccard).
    """
    from .training import iterate_lolo_splits

    selected_by_fold = {}
    for fold_info, train_df, val_df, _test_df in iterate_lolo_splits(
        feature_df, load_col=load_col, loads=loads,
    ):
        fit_df: pd.DataFrame = pd.concat([train_df, val_df], ignore_index=True)
        importance = compute_rf_importance(fit_df.loc[:, feature_cols], fit_df[label_col], seed=seed)
        selected_by_fold[fold_info["fold_name"]] = select_top_k_features(importance, k=k)

    jaccard_df = jaccard_stability_across_folds(list(selected_by_fold.values()))
    return selected_by_fold, jaccard_df