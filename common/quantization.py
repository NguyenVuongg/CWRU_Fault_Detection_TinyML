# -*- coding: utf-8 -*-
"""
common/quantization.py
========================
[Giai đoạn 2 — mục 2.4] Lượng tử hóa MLP từ Float32 sang INT8 qua
TensorFlow Lite (post-training quantization, dùng representative dataset
từ tập Train).
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


def make_representative_dataset_fn(X_train: np.ndarray, n_samples: int = 100, seed: int = 42):
    """
    Trả về generator function theo đúng interface TFLiteConverter yêu cầu
    (representative_dataset). Lấy mẫu ngẫu nhiên (không lặp) từ X_train —
    ĐÚNG mục 2.4: 'dùng representative dataset từ tập Train'.
    """
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(X_train))
    idx = rng.choice(len(X_train), size=n, replace=False)
    samples = X_train[idx].astype(np.float32)

    def representative_dataset():
        for i in range(len(samples)):
            yield [samples[i:i + 1]]

    return representative_dataset


def quantize_model_int8(keras_model, X_train: np.ndarray, n_representative_samples: int = 100,
                         seed: int = 42) -> bytes:
    """
    Chuyển keras_model (đã train xong, Float32) sang TFLite INT8 đầy đủ
    (input/output/weights đều INT8 — phù hợp triển khai MCU không FPU).
    Trả về bytes của file .tflite (dùng model_bytes_to_file() để lưu ra đĩa).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset_fn(
        X_train, n_samples=n_representative_samples, seed=seed,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def model_bytes_to_file(tflite_bytes: bytes, output_path) -> None:
    with open(output_path, "wb") as f:
        f.write(tflite_bytes)


def evaluate_tflite_model(tflite_bytes: bytes, X_test: np.ndarray, y_test) -> dict:
    """
    Chạy inference bằng TFLite Interpreter trên tập Test (đã lượng tử hóa),
    tự xử lý bước quantize input (float -> int8) và dequantize output theo
    đúng scale/zero_point mà converter đã chọn — KHÔNG tự ép kiểu thủ công
    theo giả định cố định, vì scale/zero_point khác nhau tùy mô hình.

    LƯU Ý BẢO TRÌ: `tf.lite.Interpreter` đã bị đánh dấu deprecated kể từ
    TF 2.20+ (khuyến nghị chuyển sang package `ai_edge_litert`), nhưng vẫn
    hoạt động bình thường (chỉ in cảnh báo) tới ít nhất TF 2.21 — bản đã
    test khi viết file này. Nếu sau này nâng cấp TF và hàm này bắt đầu lỗi
    thật (không chỉ cảnh báo), đổi `tf.lite.Interpreter` thành
    `ai_edge_litert.interpreter.Interpreter` (interface tương thích).
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    in_scale, in_zero_point = input_detail["quantization"]
    y_pred = []

    for i in range(len(X_test)):
        x = X_test[i:i + 1].astype(np.float32)
        if in_scale != 0:
            x_q = (x / in_scale + in_zero_point).round().astype(input_detail["dtype"])
        else:
            x_q = x.astype(input_detail["dtype"])

        interpreter.set_tensor(input_detail["index"], x_q)
        interpreter.invoke()
        out = interpreter.get_tensor(output_detail["index"])
        y_pred.append(int(np.argmax(out[0])))

    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc, "y_pred": y_pred}


def compare_float_vs_int8(keras_model, tflite_bytes: bytes, X_test: np.ndarray, y_test,
                           label_names=None) -> dict:
    """
    So sánh accuracy Float32 (Keras gốc) vs INT8 (TFLite) trên CÙNG tập
    Test — đúng yêu cầu mục 2.4: 'Kiểm tra chênh lệch accuracy trước/sau
    lượng tử hóa'.
    """
    float_pred_proba = keras_model.predict(X_test, verbose=0)
    float_pred = np.argmax(float_pred_proba, axis=1)

    if label_names is not None:
        y_test_idx = np.array([label_names.index(y) for y in y_test])
    else:
        y_test_idx = np.asarray(y_test)

    float_acc = accuracy_score(y_test_idx, float_pred)
    int8_result = evaluate_tflite_model(tflite_bytes, X_test, y_test_idx)

    return {
        "float32_accuracy": float_acc,
        "int8_accuracy": int8_result["accuracy"],
        "delta_accuracy": float_acc - int8_result["accuracy"],
    }
