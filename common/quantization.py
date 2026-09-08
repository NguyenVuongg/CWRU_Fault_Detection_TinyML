# -*- coding: utf-8 -*-
"""
common/quantization.py
=====================
Giai đoạn 2 — mục 2.4
Post-training full-integer quantization: Float32 -> INT8 bằng TensorFlow Lite.

Nguyên tắc:
1. Representative dataset CHỈ lấy từ X_train của từng LOLO fold.
2. Calibration dùng mẫu ngẫu nhiên, có seed để tái lập.
3. Input/output của mô hình được đặt ở INT8.
4. Khi mô phỏng input quantization, phải clip về đúng miền INT8 trước khi cast.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score


def make_representative_dataset_fn(
    X_train: np.ndarray,
    n_samples: int = 100,
    seed: int = 42,
):
    """Tạo representative dataset chỉ từ TRAIN của fold hiện tại."""
    X_train = np.asarray(X_train, dtype=np.float32)
    if len(X_train) == 0:
        raise ValueError("X_train is empty.")

    rng = np.random.RandomState(seed)
    n = min(n_samples, len(X_train))
    indices = rng.choice(len(X_train), size=n, replace=False)
    samples = X_train[indices]

    def representative_dataset():
        for sample in samples:
            # MLP: (32,) -> (1, 32)
            # CNN1D: (L, 1) -> (1, L, 1)
            yield [sample[np.newaxis, ...].astype(np.float32)]

    return representative_dataset


def quantize_model_int8(
    model,
    X_train: np.ndarray,
    n_representative_samples: int = 100,
    seed: int = 42,
) -> bytes:
    """Full-integer INT8 post-training quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset_fn(
        X_train,
        n_samples=n_representative_samples,
        seed=seed,
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def model_bytes_to_file(tflite_bytes: bytes, output_path) -> None:
    with open(output_path, "wb") as f:
        f.write(tflite_bytes)


def get_tflite_size(tflite_bytes: bytes) -> dict:
    size_bytes = len(tflite_bytes)
    return {
        "tflite_size_bytes": int(size_bytes),
        "tflite_size_kb": float(size_bytes / 1024.0),
    }


def get_quantization_info(tflite_bytes: bytes) -> dict:
    """Lấy dtype/scale/zero-point của input và output."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    return {
        "input_dtype": str(inp["dtype"]),
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zero),
        "output_dtype": str(out["dtype"]),
        "output_scale": float(out_scale),
        "output_zero_point": int(out_zero),
    }


def evaluate_tflite_model(
    tflite_bytes: bytes,
    X_test: np.ndarray,
    y_test,
) -> dict:
    """Evaluate an INT8 TFLite model on float32 test inputs."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    in_scale, in_zero_point = input_detail["quantization"]

    if in_scale == 0:
        raise ValueError("Invalid input quantization scale = 0.")

    input_dtype = input_detail["dtype"]
    qmin = np.iinfo(input_dtype).min
    qmax = np.iinfo(input_dtype).max

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)
    y_pred = []

    for i in range(len(X_test)):
        x = X_test[i:i + 1]
        x_q = np.round(x / in_scale + in_zero_point)
        # IMPORTANT: cast without clipping can wrap out-of-range values.
        x_q = np.clip(x_q, qmin, qmax).astype(input_dtype)

        interpreter.set_tensor(input_detail["index"], x_q)
        interpreter.invoke()
        out = interpreter.get_tensor(output_detail["index"])
        y_pred.append(int(np.argmax(out[0])))

    acc = accuracy_score(y_test, y_pred)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "y_pred": y_pred,
    }


def compare_float_vs_int8(
    keras_model,
    tflite_bytes: bytes,
    X_test: np.ndarray,
    y_test,
    label_names=None,
) -> dict:
    """Compare Float32 Keras vs INT8 TFLite on the same Test set."""
    float_pred_proba = keras_model.predict(X_test, verbose=0)
    float_pred = np.argmax(float_pred_proba, axis=1)

    if label_names is not None:
        y_test_idx = np.asarray([
            label_names.index(y) if isinstance(y, str) else int(y)
            for y in y_test
        ])
    else:
        y_test_idx = np.asarray(y_test, dtype=int)

    float_acc = accuracy_score(y_test_idx, float_pred)
    float_f1 = f1_score(y_test_idx, float_pred, average="macro")
    int8_result = evaluate_tflite_model(tflite_bytes, X_test, y_test_idx)

    # "Accuracy Drop" là đại lượng mục 2.3 yêu cầu báo cáo. Báo cáo kèm CẢ F1
    # macro vì với sơ đồ 10 lớp, lượng tự hóa có thể làm sụp riêng một lớp
    # thiểu số mà accuracy tổng gần như không đổi.
    return {
        "float32_accuracy": float(float_acc),
        "int8_accuracy": float(int8_result["accuracy"]),
        "delta_accuracy": float(float_acc - int8_result["accuracy"]),
        "float32_f1_macro": float(float_f1),
        "int8_f1_macro": float(int8_result["f1_macro"]),
        "delta_f1_macro": float(float_f1 - int8_result["f1_macro"]),
    }


# ============================================================================
# Mục 2.3 — khảo sát nhu cầu chia tỷ lệ theo KÊNH (per-channel scaling)
# ============================================================================
def diagnose_input_dynamic_range(X, feature_names=None, top_k: int = 8) -> dict:
    """Đo độ lệch dải động giữa các chiều đầu vào — căn cứ để trả lời câu
    hỏi "có cần per-channel scaling cho nhánh CNN-envelope không?" (mục 2.3).

    VẤN ĐỀ: full-integer PTQ dùng MỘT cặp (scale, zero_point) DUY NHẤT cho
    cả tensor đầu vào. Scale đó bị quyết định bởi chiều có biên độ LỚN
    nhất, nên mọi chiều nhỏ hơn nhiều bậc độ lớn sẽ bị lượng tự về cùng
    vài mức nguyên, tức mất trắng thông tin. Đường bao sau lowpass 500 Hz
    có biên độ nhỏ hẳn tín hiệu thô nên nhánh envelope là nhánh dễ trúng
    bẫy này nhất.

    Trả về dict có `range_ratio` = (dải động lớn nhất) / (dải động nhỏ
    nhất) và `n_effective_levels_min` = số mức INT8 mà chiều "yếu" nhất còn
    dùng được trong tổng 256 mức. Diễn giải để viết vào báo cáo:

      - n_effective_levels_min >= ~16  -> một scale chung là đủ, không cần
        can thiệp; ghi nhận như một kết quả khảo sát âm tính.
      - n_effective_levels_min < ~4   -> phải xử lý. Với đầu vào MLP: chuẩn
        hóa Z-score theo từng đặc trưng (đã có ở mục 1.3, fit trên Train của
        từng fold) đã đủ đưa mọi chiều về cùng thang. Với CNN 1 kênh: nhân
        đường bao với một hệ số cố định (hoặc chuẩn hóa theo file) trước
        khi đưa vào mô hình — KHÔNG thể "bật per-channel" ở đầu vào vì
        TFLite chỉ per-channel cho TRỌNG SỐ conv, không cho tensor input.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        raise ValueError("X is empty.")

    flat = X.reshape(len(X), -1)
    col_min = flat.min(axis=0)
    col_max = flat.max(axis=0)
    col_range = col_max - col_min

    global_absmax = float(np.max(np.abs(flat)))
    # scale mà TFLite sẽ chọn cho toàn tensor (đối xứng, 8 bit có dấu)
    global_scale = global_absmax / 127.0 if global_absmax > 0 else 0.0
    effective_levels = (col_range / global_scale) if global_scale > 0 else np.zeros_like(col_range)

    nonzero = col_range[col_range > 0]
    range_ratio = float(nonzero.max() / nonzero.min()) if nonzero.size else float("inf")

    order = np.argsort(effective_levels)
    names = list(feature_names) if feature_names is not None else [f"dim_{i}" for i in range(flat.shape[1])]
    weakest = [
        {"name": names[i] if i < len(names) else f"dim_{i}",
         "index": int(i),
         "range": float(col_range[i]),
         "effective_int8_levels": float(effective_levels[i])}
        for i in order[:top_k]
    ]

    n_min = float(effective_levels.min()) if effective_levels.size else 0.0
    if n_min >= 16:
        verdict = "OK — một scale chung là đủ, không cần can thiệp"
    elif n_min >= 4:
        verdict = "CẢNH BÁO — nên chuẩn hóa đầu vào và báo cáo kèm Accuracy Drop"
    else:
        verdict = "XẤU — phải chuẩn hóa/định lại thang đầu vào trước khi PTQ"

    return {
        "n_dims": int(flat.shape[1]),
        "global_absmax": global_absmax,
        "global_input_scale": float(global_scale),
        "range_ratio": range_ratio,
        "n_effective_levels_min": n_min,
        "n_effective_levels_median": float(np.median(effective_levels)) if effective_levels.size else 0.0,
        "weakest_dims": weakest,
        "verdict": verdict,
    }