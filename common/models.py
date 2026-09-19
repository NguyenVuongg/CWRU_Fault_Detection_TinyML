# -*- coding: utf-8 -*-
"""
common/models.py
==================
[Giai đoạn 2 — mục 2.2-2.3] Kiến trúc MLP + hai nhánh CNN 1D.

Cần cài đặt tensorflow (nhóm phụ thuộc tùy chọn 'dl' trong pyproject.toml):
    pip install -e ".[dl]"

SỐ LỚP: mặc định là 10 lớp (Normal + {IR,OR,B} x {7,14,21} mils) đúng bài
toán CHÍNH của mục 1.1. Muốn chạy phân tích bổ trợ 4 lớp thì truyền
n_classes=len(CLASSES_4) và dùng cột nhãn "label" thay cho "class_label".
"""

import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input, layers, optimizers, regularizers

from .config import CLASSES_4, CLASSES_10, WINDOW_SIZE_ENV, WINDOW_SIZE_RAW

# Số lớp mặc định của mọi kiến trúc lấy từ config để tránh hard-code.
N_CLASSES_DEFAULT = len(CLASSES_10)
N_CLASSES_4 = len(CLASSES_4)

def build_mlp(
    input_dim: int,
    n_classes: int = N_CLASSES_DEFAULT,
    hidden_units: tuple = (24, 12),
    dropout_rate: float = 0.2,
    l2_reg: float = 1e-4,
    norm_mean=None,
    norm_variance=None,
    name: str = "mlp_lightweight",
) -> Model:
    """
    MLP siêu gọn cho bảng đặc trưng thủ công.

    input_dim: số chiều đặc trưng THỰC TẾ của bảng đặc trưng, KHÔNG phải
    hằng số viết cứng. Sau khi gộp các cột hài rơi trùng bin FFT, bể đặc
    trưng ra 28 chiều — xem features_full.EXPECTED_FEATURE_COUNTS và lấy
    số này bằng len(feature_cols).
    """
    import warnings
    import numpy as np

    inputs = Input(shape=(input_dim,), name="features")

    if norm_mean is not None and norm_variance is not None:
        # Ép kiểu tường minh về float32 và làm phẳng về 1D (shape: (input_dim,))
        safe_mean = np.array(norm_mean, dtype=np.float32).reshape(-1)
        safe_var = np.array(norm_variance, dtype=np.float32).reshape(-1)

        x = layers.Normalization(
            mean=safe_mean, variance=safe_var, name="feature_norm"
        )(inputs)
    else:
        warnings.warn(
            "build_mlp() không nhận norm_mean/norm_variance — model nhận "
            "thẳng đặc trưng CHƯA chuẩn hóa. Nếu bạn tự StandardScaler bên "
            "ngoài model thì đây là chủ ý, bỏ qua cảnh báo. Nếu không, "
            "thang đo lệch nhau giữa nhóm Time/Order/Envelope sẽ làm huấn "
            "luyện không ổn định.",
            stacklevel=2,
        )
        x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units, activation="relu", name=f"dense_{units}_{i}",
            kernel_regularizer=regularizers.l2(l2_reg),
        )(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return Model(inputs, outputs, name=name)

# ============================================================================
# CNN 1D — hai nhánh của RQ4 (raw signal vs envelope)
# ============================================================================
# Mục 2.2 đòi hai nhánh dùng khối Conv/Pooling/BatchNorm GIỐNG HỆT NHAU để
# biến duy nhất còn lại là DẠNG ĐẦU VÀO (raw 2048 mẫu @12kHz vs envelope
# 1024 mẫu @6kHz, cùng phủ 170.67 ms).
#
# Hai nhánh dùng cùng cấu hình Conv/Pooling/BatchNorm. Stride 1 giữ lại
# thông tin ở dải cộng hưởng 2-4 kHz.
CNN_BLOCK_CONFIG = {
    "conv1_filters": 8,
    "conv1_kernel": 32,
    "conv1_stride": 1,
    "pool_size": 2,
    "conv2_filters": 16,
    "conv2_kernel": 8,
}


def build_cnn1d(window_size: int, n_classes: int = N_CLASSES_DEFAULT, *,
                input_name: str = "signal", name: str = "cnn1d",
                **block_overrides) -> Model:
    """Bộ khung CNN 1D DÙNG CHUNG cho cả hai nhánh của RQ4.

    Đây là NGUỒN DUY NHẤT định nghĩa các khối Conv/Pool/BN. build_cnn1d_raw()
    và build_cnn1d_env() chỉ là hai lời gọi mỏng vào đây với window_size khác
    nhau, nên hai nhánh không thể lệch kiến trúc do sửa một bên mà quên bên
    kia. Muốn đổi kiến trúc thì đổi CNN_BLOCK_CONFIG (áp dụng cho cả hai).

    LƯU Ý KHI DIỄN GIẢI KẾT QUẢ: kernel giống nhau tính theo MẪU, nhưng hai
    nhánh có fs khác nhau, nên trường thu nhận tính theo THỜI GIAN khác nhau
    (kernel 32 mẫu = 2.67 ms ở nhánh raw @12kHz, = 5.33 ms ở nhánh envelope
    @6kHz). Đây là hệ quả TRỰC TIẾP của việc envelope đã giảm mẫu, tức là
    một phần của chính biến đang xét, không phải confound thêm vào. Cũng vì
    vậy nhánh raw có MACC gấp đôi nhánh envelope (~1.57 MMAC vs ~0.79 MMAC)
    — chênh lệch này là MỘT KẾT QUẢ của RQ4 (mục 2.3: "chênh lệch tài nguyên
    được ghi nhận như quan sát đi kèm"), không phải lỗi thiết kế.
    """
    unknown = set(block_overrides) - set(CNN_BLOCK_CONFIG)
    if unknown:
        raise TypeError(
            f"Tham số khối không hợp lệ: {sorted(unknown)}. "
            f"Cho phép: {sorted(CNN_BLOCK_CONFIG)}."
        )
    cfg = {**CNN_BLOCK_CONFIG, **block_overrides}

    model = Sequential([
        Input(shape=(window_size, 1), name=input_name),

        # Block 1 — stride 1 để không làm mất mẫu ở dải cộng hưởng
        layers.Conv1D(
            cfg["conv1_filters"],
            cfg["conv1_kernel"],
            strides=cfg["conv1_stride"],
            padding="same",
            use_bias=False,          # BatchNorm ngay sau -> bias là dư thừa
            name="conv1d_1",
        ),
        layers.BatchNormalization(name="bn_1"),
        layers.Activation("relu", name="relu_1"),
        layers.MaxPooling1D(pool_size=cfg["pool_size"], name="maxpool_1"),

        # Block 2
        layers.Conv1D(
            cfg["conv2_filters"],
            cfg["conv2_kernel"],
            padding="same",
            use_bias=False,
            name="conv1d_2",
        ),
        layers.BatchNormalization(name="bn_2"),
        layers.Activation("relu", name="relu_2"),

        # Output
        layers.GlobalAveragePooling1D(name="global_avg_pool"),
        layers.Dense(n_classes, activation="softmax", name="output"),
    ], name=name)
    return model


def build_cnn1d_raw(window_size: int = WINDOW_SIZE_RAW,
                    n_classes: int = N_CLASSES_DEFAULT,
                    name: str = "cnn1d_raw", **block_overrides) -> Model:
    """Nhánh RQ4 học từ TÍN HIỆU THÔ: 2048 mẫu @12kHz = 170.67 ms.

    Mọi block_overrides truyền vào đây PHẢI truyền y hệt cho
    build_cnn1d_env(), nếu không hai nhánh lệch kiến trúc và RQ4 mất hiệu
    lực — dùng assert_identical_cnn_blocks() để kiểm tra lại trước khi train.
    """
    return build_cnn1d(window_size, n_classes=n_classes,
                       input_name="raw_signal", name=name, **block_overrides)


def build_cnn1d_env(window_size: int = WINDOW_SIZE_ENV,
                    n_classes: int = N_CLASSES_DEFAULT,
                    name: str = "cnn1d_envelope", **block_overrides) -> Model:
    """Nhánh RQ4 học từ ĐƯỜNG BAO đã giảm mẫu: 1024 mẫu @6kHz = 170.67 ms.

    Cùng thời lượng, cùng số cửa sổ, cùng khối Conv/Pool/BN với nhánh raw —
    xem config.ENV_DECIM và build_cnn1d().
    """
    return build_cnn1d(window_size, n_classes=n_classes,
                       input_name="envelope", name=name, **block_overrides)


def _comparable_block_configs(model: Model):
    """Lấy cấu hình từng lớp, bỏ 'name' (tên lớp không ảnh hưởng kiến trúc)
    và bỏ InputLayer (chỗ DUY NHẤT hai nhánh được phép khác nhau)."""
    out = []
    for layer in model.layers:
        if isinstance(layer, layers.InputLayer):
            continue
        cfg = dict(layer.get_config())
        cfg.pop("name", None)
        out.append((type(layer).__name__, cfg))
    return out


def cnn_block_differences(model_a: Model, model_b: Model) -> list:
    """Liệt kê mọi khác biệt kiến trúc giữa 2 model, BỎ QUA lớp input.

    Trả về list rỗng nghĩa là hai nhánh thỏa đúng yêu cầu "khối Conv/Pooling/
    BatchNorm giống hệt nhau" của mục 2.2.
    """
    a = _comparable_block_configs(model_a)
    b = _comparable_block_configs(model_b)
    diffs = []

    if len(a) != len(b):
        diffs.append(
            f"số lớp (không tính input) khác nhau: {len(a)} vs {len(b)}"
        )

    for i, ((type_a, cfg_a), (type_b, cfg_b)) in enumerate(zip(a, b)):
        if type_a != type_b:
            diffs.append(f"lớp #{i}: kiểu khác nhau ({type_a} vs {type_b})")
            continue
        for key in sorted(set(cfg_a) | set(cfg_b)):
            va, vb = cfg_a.get(key), cfg_b.get(key)
            if va != vb:
                diffs.append(f"lớp #{i} ({type_a}): {key} = {va!r} vs {vb!r}")

    return diffs


def assert_identical_cnn_blocks(model_a: Model, model_b: Model) -> None:
    """Nổ lỗi nếu 2 nhánh CNN không dùng khối giống hệt nhau (mục 2.2).

    Gọi hàm này NGAY TRƯỚC khi train cặp raw/envelope. Chi phí gần như bằng
    0 và nó chặn đúng loại lỗi âm thầm nhất: một người sửa siêu tham số của
    một nhánh rồi báo cáo kết quả như thể hai nhánh vẫn công bằng.
    """
    diffs = cnn_block_differences(model_a, model_b)
    if diffs:
        raise AssertionError(
            "Hai nhánh CNN KHÔNG dùng khối giống hệt nhau, vi phạm mục 2.2 "
            "(chỉ dạng đầu vào được phép khác):\n  - "
            + "\n  - ".join(diffs)
            + f"\nĐầu vào: {model_a.input_shape} vs {model_b.input_shape}"
        )


def make_sparse_macro_f1(n_classes: int, name: str = "f1_macro"):
    """Metric F1 macro dùng được với nhãn NGUYÊN (sparse labels).

    VÌ SAO CẦN BỌC LẠI: tf.keras.metrics.F1Score gốc đòi y_true CÙNG SHAPE
    với y_pred, tức dạng one-hot (batch, n_classes). Truyền thẳng nó vào
    .compile() khi loss là sparse_categorical_crossentropy (y_true là số
    nguyên) sẽ gây lỗi shape hoặc — tệ hơn — ra một con số vô nghĩa mà vẫn
    in ra bình thường. Nên phải one-hot lại y_true trong update_state().

    VÌ SAO CẦN F1 MACRO: nó trọng số ĐỀU cho cả 10 lớp nên phát hiện được ca
    mô hình bỏ rơi một lớp thiểu số trong khi accuracy vẫn cao. Với sơ đồ 10
    lớp thì rủi ro này CAO HƠN HẲN so với 4 lớp: mỗi lớp lỗi chỉ còn ~1/9 số
    file lỗi, nên một mô hình bỏ rơi trọn lớp B_021 vẫn có thể đạt accuracy
    ~0.9. Đây cũng chính là đại lượng training.run_lolo_evaluation() và
    splitting.run_split_comparison_experiment() đang báo cáo, nên theo dõi
    cùng một metric xuyên suốt thì đường hội tụ mới đối chiếu được với
    bảng LOLO cuối cùng.

    Trả về None nếu bản TensorFlow không có F1Score (< 2.13) để không làm
    gãy pipeline. Lưu ý: lớp này định nghĩa cục bộ nên khi load lại model
    đã lưu, hãy compile lại bằng compile_classifier() thay vì dựa vào
    custom_objects.
    """
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
    """Compile chuẩn cho bài toán phân loại 10 lớp (Normal + {IR,OR,B} x
    {7,14,21} mils, mục 1.1), dùng chung cho cả MLP và CNN1D để đảm bảo so
    sánh công bằng (mục 2.3).

    Theo dõi CẢ accuracy VÀ f1_macro: phạm vi dữ liệu đã chốt lệch lớp
    (Normal chỉ có 4 file, mỗi lớp lỗi có ~4 file), nên accuracy một mình
    có thể cao trong khi một lớp bị bỏ rơi hoàn toàn. History sẽ có khóa
    'f1_macro' và 'val_f1_macro' -> dùng EarlyStopping(monitor="val_f1_macro",
    mode="max") ở Giai đoạn 2.3 thay cho val_accuracy.

    n_classes: mặc định suy ra từ shape lớp output.
    """
    if n_classes is None:
        n_classes = int(model.output_shape[-1])

    metric_list = ["accuracy"]
    f1_metric = make_sparse_macro_f1(n_classes)
    if f1_metric is not None:
        metric_list.append(f1_metric)
    else:
        print("[compile_classifier] CẢNH BÁO: bản TensorFlow này không có "
              "tf.keras.metrics.F1Score (cần >= 2.13); chỉ theo dõi accuracy "
              "trong lúc train. F1 macro cuối cùng vẫn được tính qua sklearn "
              "ở training.py/splitting.py.")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=metric_list,
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
