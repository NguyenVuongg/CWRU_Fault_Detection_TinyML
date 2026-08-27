# -*- coding: utf-8 -*-
"""
common/models.py
==================
[Giai đoạn 2 — mục 2.3] Định nghĩa kiến trúc MLP
Cần cài đặt tensorflow (nhóm phụ thuộc tùy chọn 'dl' trong pyproject.toml):
    pip install -e ".[dl]"
"""

from tensorflow.keras import Model, Sequential, Input, layers, optimizers

def build_mlp(input_dim: int, n_classes: int = 4, name: str = "mlp_lightweight") -> Model:
    """
    MLP siêu gọn — ĐÚNG kiến trúc mục 2.3 (Có bổ sung Dropout chống Overfitting)
    """
    model = Sequential([
        Input(shape=(input_dim,), name="features"),
        layers.Dense(32, activation="relu", name="dense_32"),
        layers.Dropout(0.2, name="dropout_1"), # Tắt ngẫu nhiên 20% nơ-ron
        layers.Dense(16, activation="relu", name="dense_16"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(n_classes, activation="softmax", name="output"),
    ], name=name)
    return model

# CNN1D — học từ tín hiệu THÔ (raw signal)

def build_cnn1d_raw(
    window_size: int = 2048,
    n_classes: int = 4,
    conv1_filters: int = 8,
    conv1_kernel: int = 64,
    conv1_stride: int = 2,
    pool_size: int = 4,
    conv2_filters: int = 16,
    conv2_kernel: int = 16,
    name: str = "cnn1d_raw"
) -> Model:
    model = Sequential([
        Input(shape=(window_size, 1), name="raw_signal"),
        layers.Conv1D(
            conv1_filters,
            conv1_kernel,
            strides=conv1_stride,
            activation="relu",
            padding="same",
            name="conv1d_1"
        ),
        layers.MaxPooling1D(pool_size=pool_size, name="maxpool"),
        layers.Conv1D(
            conv2_filters,
            conv2_kernel,
            activation="relu",
            padding="same",
            name="conv1d_2"
        ),
        layers.GlobalAveragePooling1D(name="global_avg_pool"),
        layers.Dense(n_classes, activation="softmax", name="output"),
    ], name=name)
    return model

def build_cnn1d_env(
    window_size: int = 1024,
    n_classes: int = 4,
    conv1_filters: int = 8,
    conv1_kernel: int = 32,
    pool_size: int = 2,
    conv2_filters: int = 16,
    conv2_kernel: int = 8,
    name: str = "cnn1d_envelope"
) -> Model: 
    model = Sequential([
        Input(shape=(window_size, 1), name="envelope"),
        
        # Block 1: Bỏ strides=2 để không làm mất thông tin đỉnh sóng
        layers.Conv1D(
            conv1_filters,
            conv1_kernel,
            strides=1,
            padding="same",
            use_bias=False,
            name="conv1d_1"
        ),
        layers.BatchNormalization(name="bn_1"),
        layers.Activation("relu", name="relu_1"),
        layers.MaxPooling1D(pool_size=pool_size, name="maxpool_1"),
        
        # Block 2
        layers.Conv1D(
            conv2_filters,
            conv2_kernel,
            padding="same",
            use_bias=False,
            name="conv1d_2"
        ),
        layers.BatchNormalization(name="bn_2"),
        layers.Activation("relu", name="relu_2"),
        
        # Output
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
    return {"trainable_params": trainable, 
            "non_trainable_params": non_trainable,
            "total_params": trainable + non_trainable}