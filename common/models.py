# -*- coding: utf-8 -*-
"""
common/models.py
==================
[Giai đoạn 2 — mục 2.3] Định nghĩa kiến trúc MLP & CNN1D.

Kiến trúc đã CHỐT trong đề cương ("Chốt kiến trúc trước khi code, không đổi
giữa chừng") — file này chỉ có 1 phiên bản duy nhất của mỗi kiến trúc,
KHÔNG thêm biến thể, đúng tinh thần mục 2.3.

Cần cài đặt tensorflow (nhóm phụ thuộc tùy chọn 'dl' trong pyproject.toml):
    pip install -e ".[dl]"

LƯU Ý IMPORT: dùng `keras` độc lập (package `keras`, Keras 3, chạy trên
backend TensorFlow đã cài) thay vì `tensorflow.keras` — cách import cũ
khiến Pylance/pyright báo warning "could not be resolved from source" do
cách TensorFlow tiêm module keras vào lúc chạy (runtime injection), static
type checker không lần theo được. `keras` độc lập có nguồn rõ ràng, chạy
identical vì cùng dùng backend TensorFlow đã cài (tensorflow-cpu kéo theo
keras như một dependency).
"""

from keras import Model, Sequential, Input, layers, optimizers


def build_mlp(input_dim: int, n_classes: int = 4, name: str = "mlp_lightweight") -> Model:
    """
    MLP siêu gọn — ĐÚNG kiến trúc mục 2.3:
        input(input_dim) -> Dense(32, relu) -> Dense(16, relu) -> Dense(n_classes, softmax)

    input_dim = số đặc trưng đã chọn ở Giai đoạn 2.2 (thường ~30, nhưng
    PHẢI truyền số THẬT sau khi chạy feature_selection.select_top_k_features(),
    không hardcode 30 ở đây).
    """
    model = Sequential([
        Input(shape=(input_dim,), name="features"),
        layers.Dense(32, activation="relu", name="dense_32"),
        layers.Dense(16, activation="relu", name="dense_16"),
        layers.Dense(n_classes, activation="softmax", name="output"),
    ], name=name)
    return model


def build_cnn1d(window_size: int = 2048, n_classes: int = 4,
                 conv1_filters: int = 16, conv1_kernel: int = 64, conv1_stride: int = 2,
                 pool_size: int = 4,
                 conv2_filters: int = 32, conv2_kernel: int = 16,
                 name: str = "cnn1d_shallow") -> Model:
    """
    CNN1D nông (baseline đối chứng, học trực tiếp từ tín hiệu thô) — ĐÚNG
    cấu trúc tầng mục 2.3:
        input(window_size, 1) -> Conv1D -> MaxPooling1D -> Conv1D
                               -> GlobalAveragePooling1D -> Dense(n_classes, softmax)

    Số filter/kernel/stride/pool KHÔNG được nêu cụ thể trong đề cương —
    đây là giá trị mặc định HỢP LÝ cho tín hiệu rung động 2048 mẫu, có thể
    chỉnh qua tham số. Nếu bạn tinh chỉnh (tune) các số này qua tập
    Validation, GHI RÕ vào báo cáo (mục 2.3 chỉ chốt CẤU TRÚC TẦNG, không
    chốt cứng các con số này).
    """
    model = Sequential([
        Input(shape=(window_size, 1), name="raw_signal"),
        layers.Conv1D(conv1_filters, conv1_kernel, strides=conv1_stride,
                       activation="relu", padding="same", name="conv1d_1"),
        layers.MaxPooling1D(pool_size=pool_size, name="maxpool"),
        layers.Conv1D(conv2_filters, conv2_kernel, activation="relu",
                       padding="same", name="conv1d_2"),
        layers.GlobalAveragePooling1D(name="global_avg_pool"),
        layers.Dense(n_classes, activation="softmax", name="output"),
    ], name=name)
    return model


def compile_classifier(model: Model, learning_rate: float = 1e-3) -> Model:
    """Compile chuẩn cho bài toán phân loại 4 lớp (Normal/IR/OR/B), dùng
    chung cho cả MLP và CNN1D để đảm bảo so sánh công bằng (mục 2.3)."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model: Model) -> dict:
    """Đếm tham số — dùng đối chiếu NHANH giữa MLP/CNN1D trước khi có số
    đo tài nguyên thật qua STM32Cube.AI ở Giai đoạn 3 (mục 2.3: 'chênh
    lệch tài nguyên được ghi nhận như quan sát đi kèm')."""
    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    non_trainable = sum(int(w.numpy().size) for w in model.non_trainable_weights)
    return {"trainable_params": trainable, "non_trainable_params": non_trainable,
            "total_params": trainable + non_trainable}