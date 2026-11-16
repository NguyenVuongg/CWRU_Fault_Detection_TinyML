# -*- coding: utf-8 -*-
"""
src/models.py
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
    
    


def export_for_stm32cubeai(
    model: Model,
    output_dir: str,
    feature_cols: list,
    norm_mean=None,
    norm_variance=None,
    class_names=None,
    model_name: str = "bearing_mlp",
):
    """
    Export TensorFlow/Keras MLP sang HDF5 cho STM32Cube.AI.

    Xuất:
        - bearing_mlp.h5
        - normalization_stats.npz
        - metadata.json

    Có kiểm tra lại mô hình sau khi lưu.
    """

    import json
    from pathlib import Path

    # ============================================================
    # 1. KIỂM TRA MODEL
    # ============================================================

    if (norm_mean is None) != (norm_variance is None):
        raise ValueError(
            "norm_mean và norm_variance phải được truyền cùng nhau."
        )

    input_dim_value = model.input_shape[-1]
    output_dim_value = model.output_shape[-1]

    if input_dim_value is None or output_dim_value is None:
        raise ValueError("Input/output dimension không xác định.")

    input_dim = int(input_dim_value)
    n_classes = int(output_dim_value)

    feature_cols = [str(c) for c in feature_cols]

    if len(feature_cols) != input_dim:
        raise ValueError(
            f"Số feature = {len(feature_cols)}, "
            f"nhưng model cần {input_dim}."
        )

    # ============================================================
    # 2. XÁC ĐỊNH CLASS
    # ============================================================

    if class_names is None:
        if n_classes == N_CLASSES_4:
            class_names = CLASSES_4
        elif n_classes == N_CLASSES_10:
            class_names = CLASSES_10
        else:
            class_names = [
                f"class_{i}" for i in range(n_classes)
            ]

    class_names = [str(c) for c in class_names]

    if len(class_names) != n_classes:
        raise ValueError("Số class không khớp output model.")

    # ============================================================
    # 3. CHUẨN BỊ THÔNG SỐ Z-SCORE
    # ============================================================

    mean_arr: np.ndarray | None = None
    var_arr: np.ndarray | None = None

    if norm_mean is not None and norm_variance is not None:

        mean_arr = np.asarray(
            norm_mean, dtype=np.float32
        ).reshape(-1)

        var_arr = np.asarray(
            norm_variance, dtype=np.float32
        ).reshape(-1)

        if len(mean_arr) != input_dim or len(var_arr) != input_dim:
            raise ValueError("Kích thước Z-score không hợp lệ.")

        if (
            not np.all(np.isfinite(mean_arr))
            or not np.all(np.isfinite(var_arr))
            or np.any(var_arr <= 0)
        ):
            raise ValueError("Mean/variance không hợp lệ.")

    # Kiểm tra Normalization layer bên trong model
    has_embedded_norm = any(
        isinstance(
            layer,
            tf.keras.layers.Normalization
        )
        for layer in model.layers
    )

    if has_embedded_norm:
        norm_mode = "embedded_in_model"
    elif mean_arr is not None and var_arr is not None:
        norm_mode = "external_zscore"
    else:
        norm_mode = "not_recorded"

    # ============================================================
    # 4. TẠO THƯ MỤC EXPORT
    # ============================================================

    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_name).stem

    model_path = export_dir / f"{model_name}.h5"

    # ============================================================
    # 5. EXPORT MODEL HDF5
    # ============================================================

    # Không sử dụng include_optimizer=False ở đây:
    # Tham số này không có trong type signature của Keras hiện tại.
    model.save(str(model_path))

    print(f"[EXPORT] Model saved: {model_path}")

    # ============================================================
    # 6. EXPORT THÔNG SỐ Z-SCORE
    # ============================================================

    scaler_file = None

    # Kiểm tra CẢ HAI biến để Pylance nhận diện đúng kiểu dữ liệu.
    if mean_arr is not None and var_arr is not None:

        scaler_path = export_dir / "normalization_stats.npz"

        mean_values = np.asarray(mean_arr, dtype=np.float32)
        variance_values = np.asarray(var_arr, dtype=np.float32)

        std_values = np.sqrt(variance_values)

        np.savez_compressed(
            str(scaler_path),
            mean=mean_values,
            variance=variance_values,
            std=std_values,
        )

        scaler_file = scaler_path.name

        print(f"[EXPORT] Normalization saved: {scaler_path}")

    # ============================================================
    # 7. EXPORT METADATA JSON
    # ============================================================

    metadata = {
        "model_name": model_name,
        "format": "Keras HDF5",

        "input_dim": input_dim,
        "output_dim": n_classes,

        "input_shape": [None, input_dim],
        "output_shape": [None, n_classes],

        "architecture": "MLP_28_32_16_N",
        "parameters": int(model.count_params()),

        "feature_names": feature_cols,
        "class_names": class_names,

        "output_activation": "softmax",
        "prediction_method": "argmax",

        "normalization": {
            "mode": norm_mode,
            "formula": "(x - mean) / sqrt(variance)",

            "mean": (
                mean_arr.tolist()
                if mean_arr is not None else None
            ),

            "variance": (
                var_arr.tolist()
                if var_arr is not None else None
            ),

            "scaler_file": scaler_file,

            "apply_externally": (
                norm_mode == "external_zscore"
            ),
        },
    }

    metadata_path = export_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"[EXPORT] Metadata saved: {metadata_path}")

    # ============================================================
    # 8. KIỂM TRA MODEL SAU KHI EXPORT
    # ============================================================

    loaded_model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )

    dummy_input = np.zeros(
        (2, input_dim),
        dtype=np.float32,
    )

    # Sử dụng predict() thay vì gọi model(...)
    # để tránh lỗi overload của Pylance.
    pred_original = np.asarray(
        model.predict(dummy_input, verbose=0),
        dtype=np.float32,
    )

    pred_loaded = np.asarray(
        loaded_model.predict(dummy_input, verbose=0),
        dtype=np.float32,
    )

    if not np.allclose(
        pred_original,
        pred_loaded,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Model sau khi load không khớp model ban đầu!"
        )

    print("[EXPORT] Model verification: PASSED")

    # ============================================================
    # 9. HIỂN THỊ KẾT QUẢ
    # ============================================================

    print("\n" + "=" * 55)
    print("          STM32Cube.AI MODEL EXPORT")
    print("=" * 55)

    print(f"Model       : {model_path}")
    print(f"Input       : {input_dim}")
    print(f"Output      : {n_classes}")
    print(f"Parameters  : {model.count_params()}")
    print(f"Normalization: {norm_mode}")

    print("-" * 55)
    print("Verification: PASSED")
    print("Export      : SUCCESS")
    print("=" * 55)

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "scaler_file": scaler_file,
    }
