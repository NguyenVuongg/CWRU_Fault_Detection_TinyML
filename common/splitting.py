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

from .config import CLASS_SCHEME_DEFAULT, LABEL_COL_BY_SCHEME

# TỶ LỆ CHIA ĐÃ CHỐT (mục 2.1: "Stratified File-based Split 80/20"). Để ở
# đây một chỗ để cả hai hàm chia dùng CÙNG một tỷ lệ: RQ1 so sánh hai
# CÁCH CHIA, nên nếu một bên 70/30 và bên kia 80/20 thì chính kích thước
# tập train đã là một confound.
TEST_RATIO_DEFAULT = 0.2


def resolve_label_col(feature_df: pd.DataFrame, label_col=None,
                      scheme: str = CLASS_SCHEME_DEFAULT) -> str:
    """Chọn cột nhãn theo sơ đồ lớp đã chốt ở mục 1.1.

    label_col=None -> lấy "class_label" (10 lớp, bài toán CHÍNH). Nếu bảng
    đặc trưng được sinh bằng phiên bản features_full.py cũ (chưa có cột
    này) thì lùi về "label" 4 lớp VÀ CẢNH BÁO, thay vì nổ lỗi — nhưng để
    báo cáo đúng đề cương thì phải build lại bảng đặc trưng.
    """
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


def random_window_split(feature_df: pd.DataFrame,
                         test_ratio: float = TEST_RATIO_DEFAULT,
                         seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SAI CÁCH LÀM (cố ý) — chia ngẫu nhiên theo dòng window, bỏ qua
    file_id. Windows từ cùng 1 file có thể rơi cả vào train và test."""
    rng = np.random.RandomState(seed)
    idx = feature_df.index.to_numpy().copy()
    rng.shuffle(idx)
    n_test = int(len(idx) * test_ratio)
    train_df: pd.DataFrame = feature_df.loc[idx[n_test:]]
    test_df: pd.DataFrame = feature_df.loc[idx[:n_test]]
    return train_df, test_df


def file_based_split(feature_df: pd.DataFrame,
                      test_ratio: float = TEST_RATIO_DEFAULT,
                      seed: int = 42, stratify_col=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ĐÚNG CÁCH LÀM — "Stratified File-based Split 80/20" của mục 2.1:
    chia theo file_id trước, có stratify theo `stratify_col`. Không file nào
    vừa ở train vừa ở test.

    stratify_col=None -> tự lấy cột nhãn của sơ đồ 10 lớp ("class_label").
    Stratify theo 10 lớp là BẮT BUỘC với bài toán chính: nếu chỉ stratify
    theo 4 nhãn gốc thì một lớp con như B_021 (chỉ ~4 file, mỗi tải 1 file)
    hoàn toàn có thể vắng mặt ở tập train của một fold.
    """
    stratify_col = resolve_label_col(feature_df, stratify_col)
    rng = np.random.RandomState(seed)

    file_label = (feature_df[["file_id", stratify_col]]
                  .drop_duplicates("file_id")
                  .set_index("file_id")[stratify_col])

    test_files: list = []
    for _, group in file_label.groupby(file_label):
        ids = group.index.to_numpy().copy()
        rng.shuffle(ids)
        if len(ids) <= 1:
            n_test = 0  # giữ file duy nhất ở train, tránh mất trắng nhãn đó
        else:
            n_test = max(1, int(round(len(ids) * test_ratio)))
            n_test = min(n_test, len(ids) - 1)  # luôn chừa >=1 file ở train
        test_files.extend(ids[:n_test])

    test_files_set = set(test_files)
    train_mask = ~feature_df["file_id"].isin(test_files_set)
    train_df: pd.DataFrame = feature_df.loc[train_mask].reset_index(drop=True)
    test_df: pd.DataFrame = feature_df.loc[~train_mask].reset_index(drop=True)
    return train_df, test_df


def run_split_comparison_experiment(feature_df, feature_cols, seed=42,
                                    test_ratio: float = TEST_RATIO_DEFAULT,
                                    label_col=None):
    """Huấn luyện Random Forest với 2 cách chia, trả về bảng so sánh —
    bằng chứng thực nghiệm sơ bộ cho RQ1/H1.

    Cả hai cách chia nhận CÙNG test_ratio và CÙNG cột nhãn, nên biến duy
    nhất còn lại đúng là "chia theo dòng hay chia theo file".
    """
    label_col = resolve_label_col(feature_df, label_col)
    results = {}
    for name, split_fn in [
        ("Random Window Split", random_window_split),
        ("File-based Split", file_based_split),
    ]:
        train_df, test_df = split_fn(feature_df, test_ratio=test_ratio, seed=seed)
        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_test, y_test = test_df[feature_cols], test_df[label_col]

        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
    return pd.DataFrame(results).T.reset_index(names=["split_method"])


def generate_lolo_folds(loads=(0, 1, 2, 3)):
    """Cấu trúc 4 fold LOLO đúng mục 2.1. Trả về list dict
    {'test_load': X, 'trainval_loads': [...]}."""
    return [{"test_load": t, "trainval_loads": [l for l in loads if l != t]} for t in loads]