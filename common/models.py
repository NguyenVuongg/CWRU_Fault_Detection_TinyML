# -*- coding: utf-8 -*-
"""
common/models.py
================
MLP 28 đặc trưng cho dự án (đã gỡ toàn bộ CNN 1D).

SỐ LỚP: mặc định lấy từ config.CLASS_SCHEME_DEFAULT ("4class" -> 4 lớp
Normal / IR / OR / B). Muốn chạy 10 lớp thì đổi CLASS_SCHEME_DEFAULT trong
config, hoặc truyền n_classes=N_CLASSES_10 và dùng cột nhãn 10 lớp.

Kiến trúc chốt: 28 -> 32 -> 16 -> n_classes (softmax).
Ablation nhóm đặc trưng chỉ đổi input_dim (11 / 21 / 28), giữ nguyên 32 -> 16.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Model, Input, layers, optimizers, regularizers
from sklearn.metrics import accuracy_score, f1_score

from .config import CLASSES_4, CLASSES_10, CLASSES_BY_SCHEME, CLASS_SCHEME_DEFAULT

N_CLASSES_4 = len(CLASSES_4)
N_CLASSES_10 = len(CLASSES_10)
N_CLASSES_DEFAULT = len(CLASSES_BY_SCHEME[CLASS_SCHEME_DEFAULT])
PRIMARY_FEATURE_COUNT = 28


# ============================================================================
# MLP
# ============================================================================
def build_mlp(
    input_dim: int,
    n_classes: int = N_CLASSES_DEFAULT,
    hidden_units: tuple = (32, 16),
    dropout_rate: float = 0.1,
    l2_reg: float = 1e-4,
    norm_mean=None,
    norm_variance=None,
    name: str = "mlp",
    warn_if_unscaled: bool = True,
) -> Model:
    """MLP gọn cho bảng đặc trưng thủ công.

    - Dropout chỉ đặt sau lớp ẩn ĐẦU TIÊN (không tốn gì lúc suy luận).
    - Không BatchNorm: z-score ở đầu vào là đủ, INT8 đơn giản hơn.
    - Đầu ra softmax để tính xác suất / F1 trực tiếp; argmax trên MCU.
    """
    import warnings

    if input_dim <= 0:
        raise ValueError("input_dim phải dương.")

    inputs = Input(shape=(input_dim,), name="features")

    if (norm_mean is None) != (norm_variance is None):
        raise ValueError("norm_mean và norm_variance phải được truyền cùng nhau.")
    if norm_mean is not None:
        safe_mean = np.array(norm_mean, dtype=np.float32).reshape(-1)
        safe_var = np.array(norm_variance, dtype=np.float32).reshape(-1)
        if len(safe_mean) != input_dim or len(safe_var) != input_dim:
            raise ValueError(f"Thông số normalization phải có đúng {input_dim} phần tử.")
        if not np.all(np.isfinite(safe_mean)) or not np.all(np.isfinite(safe_var)):
            raise ValueError("Thông số normalization phải hữu hạn.")
        if np.any(safe_var < 0):
            raise ValueError("norm_variance không được âm.")
        x = layers.Normalization(
            mean=safe_mean, variance=safe_var, name="feature_norm"
        )(inputs)
    else:
        if warn_if_unscaled:
            warnings.warn(
                "build_mlp() nhận thẳng đặc trưng CHƯA chuẩn hóa. "
                "Nếu bạn tự chuẩn hóa bằng sklearn bên ngoài thì có thể bỏ qua.",
                stacklevel=2,
            )
        x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units, activation="relu", name=f"dense_{units}_{i}",
            kernel_regularizer=regularizers.l2(l2_reg),
        )(x)
        if i == 0 and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="dropout_0")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return Model(inputs, outputs, name=name)


def build_primary_mlp(n_classes: int = N_CLASSES_DEFAULT, **kwargs) -> Model:
    """MLP chính: mặc định 28 đặc trưng đã Z-score bằng thống kê Train của fold.

    Cho ablation nhóm đặc trưng, truyền input_dim=11 (Time), 21 (Time+Order)
    hoặc 28 (đầy đủ); phần ẩn giữ nguyên 32 -> 16.
    """
    input_dim = int(kwargs.pop("input_dim", PRIMARY_FEATURE_COUNT))
    return build_mlp(
        input_dim=input_dim,
        n_classes=n_classes,
        name=kwargs.pop("name", f"mlp_primary_{input_dim}"),
        warn_if_unscaled=False,
        **kwargs,
    )


def build_linear_baseline(input_dim: int = PRIMARY_FEATURE_COUNT,
                          n_classes: int = N_CLASSES_DEFAULT,
                          norm_mean=None, norm_variance=None,
                          l2_reg: float = 1e-4,
                          name: str = "linear_baseline") -> Model:
    """Softmax regression (không lớp ẩn) để kiểm tra ceiling effect."""
    inputs = Input(shape=(input_dim,), name="features")
    if (norm_mean is None) != (norm_variance is None):
        raise ValueError("norm_mean và norm_variance phải được truyền cùng nhau.")
    x = inputs
    if norm_mean is not None:
        x = layers.Normalization(
            mean=np.array(norm_mean, dtype=np.float32).reshape(-1),
            variance=np.array(norm_variance, dtype=np.float32).reshape(-1),
            name="feature_norm",
        )(x)
    outputs = layers.Dense(
        n_classes, activation="softmax", name="output",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    return Model(inputs, outputs, name=name)


# ============================================================================
# Compile / tiện ích huấn luyện
# ============================================================================
def make_sparse_macro_f1(n_classes: int, name: str = "f1_macro"):
    """Metric F1 macro dùng được với nhãn NGUYÊN (sparse labels)."""
    base = getattr(tf.keras.metrics, "F1Score", None)
    if base is None:
        return None
    n_classes = int(n_classes)

    class SparseMacroF1(base):
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.one_hot(
                tf.reshape(tf.cast(y_true, tf.int32), [-1]), depth=n_classes
            )
            return super().update_state(y_true, y_pred, sample_weight)

    return SparseMacroF1(average="macro", name=name)


def compile_classifier(model: Model, learning_rate: float = 1e-3,
                       n_classes: int | None = None) -> Model:
    """Compile chuẩn cho bài toán phân loại (Adam + sparse cross-entropy)."""
    if n_classes is None:
        n_classes = int(model.output_shape[-1])

    metric_list = ["accuracy"]
    f1_metric = make_sparse_macro_f1(n_classes)
    if f1_metric is not None:
        metric_list.append(f1_metric)
    else:
        print("[compile_classifier] CẢNH BÁO: bản TensorFlow này không có tf.keras.metrics.F1Score.")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=metric_list,
    )
    return model


def compute_class_weights(y_train, n_classes: int | None = None) -> dict:
    """Class weight 'balanced' tính CHỈ từ nhãn Train của fold hiện tại.
    Truyền vào model.fit(..., class_weight=...). Cần vì Normal ít file hơn
    hẳn các lớp lỗi."""
    y = np.asarray(y_train, dtype=np.int64)
    k = int(n_classes) if n_classes is not None else int(y.max()) + 1
    counts = np.bincount(y, minlength=k).astype(np.float64)
    present = counts > 0
    n_present = int(present.sum())
    return {
        int(c): float(len(y) / (n_present * counts[c]))
        for c in range(k) if present[c]
    }

def count_params(model: Model) -> dict:
    # Sử dụng .num_elements() của TensorFlow để đếm số phần tử thay vì dùng Numpy
    trainable = sum(w.shape.num_elements() or 0 for w in model.trainable_weights)
    non_trainable = sum(w.shape.num_elements() or 0 for w in model.non_trainable_weights)
    return {"trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "total_params": trainable + non_trainable}

# ============================================================================
# Hàm bổ trợ cho giao thức đánh giá (mục 6, README) — chuẩn hóa + CI bootstrap
# ============================================================================
def compute_norm_stats(train_df: pd.DataFrame, feature_cols: list):
    """Mean/variance CHỈ từ Train của fold hiện tại — không tính trên toàn bộ bảng.
    Dùng làm norm_mean/norm_variance truyền vào build_primary_mlp()."""
    X = train_df[feature_cols].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    var = np.maximum(X.var(axis=0), 1e-12)  # chặn variance=0 làm Normalization chia 0
    return mean.astype(np.float32), var.astype(np.float32)


def _file_level_labels(y_true, y_pred, file_ids):
    """Gộp nhãn cửa sổ -> nhãn file theo đa số cửa sổ/file."""
    file_true, file_pred = {}, {}
    for fid, yt, yp in zip(np.asarray(file_ids), y_true, y_pred):
        file_true.setdefault(fid, []).append(int(yt))
        file_pred.setdefault(fid, []).append(int(yp))
    uniq = np.array(sorted(file_true))
    ft = np.array([np.bincount(file_true[f]).argmax() for f in uniq])
    fp = np.array([np.bincount(file_pred[f]).argmax() for f in uniq])
    return uniq, ft, fp


def _metric(ft, fp, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(ft, fp))
    return float(f1_score(ft, fp, average="macro", zero_division=0))


def bootstrap_ci_by_file(y_true, y_pred, file_ids, *, metric: str = "f1_macro",
                         seed: int = 42, n_boot: int = 2000, alpha: float = 0.05):
    """CI bootstrap THEO FILE (mục 6 README: 'không tính std trên cửa sổ').
    Nhận mảng y_true/y_pred/file_ids cùng độ dài (mức cửa sổ), gộp theo đa số
    cửa sổ/file rồi bootstrap resample FILE. point_estimate tính trên mẫu gốc."""
    uniq, ft, fp = _file_level_labels(y_true, y_pred, file_ids)
    n_files = len(uniq)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_files, size=n_files)
        boot[b] = _metric(ft[idx], fp[idx], metric)

    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point_estimate": float(_metric(ft, fp, metric)),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "n_files": n_files, "n_boot": n_boot}