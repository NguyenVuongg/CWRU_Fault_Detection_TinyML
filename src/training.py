# -*- coding: utf-8 -*-
"""File-level stratified group k-fold evaluation for the primary MLP."""

from __future__ import annotations

import inspect
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from .config import (
    CLASS_SCHEME_DEFAULT,
    CLASSES_BY_SCHEME,
    LABEL_COL_BY_SCHEME,
)
from .splitting import resolve_label_col, stratified_group_kfold_splits
from .models import (
    compile_classifier,
    compute_norm_stats,
    compute_class_weights,
)

def _predict_scores(estimator, X, n_classes: int) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(X))
        estimator_classes = getattr(estimator, "classes_", None)
        if estimator_classes is not None and scores.ndim == 2:
            aligned = np.zeros((len(X), n_classes), dtype=np.float64)
            for source_col, class_id in enumerate(estimator_classes):
                aligned[:, int(class_id)] = scores[:, source_col]
            return aligned
    else:
        predict = estimator.predict
        parameters = inspect.signature(predict).parameters
        scores = np.asarray(predict(X, verbose=0) if "verbose" in parameters else predict(X))

    if scores.ndim == 1:
        return scores.astype(np.int64, copy=False)
    if scores.ndim != 2 or scores.shape[0] != len(X):
        raise ValueError(f"predict trả shape không hợp lệ: {scores.shape}.")
    if scores.shape[1] == 1:
        return scores[:, 0].astype(np.int64, copy=False)
    if scores.shape[1] > n_classes:
        raise ValueError(
            f"predict trả {scores.shape[1]} lớp, nhưng LabelEncoder có {n_classes}."
        )
    if scores.shape[1] < n_classes:
        aligned = np.zeros((len(X), n_classes), dtype=np.float64)
        aligned[:, :scores.shape[1]] = scores
        scores = aligned
    return scores


def _file_level_predictions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    group_col: str,
    label_col: str,
    encoded_labels: np.ndarray,
    n_classes: int,
) -> pd.DataFrame:
    rows = frame.reset_index(drop=True)
    if len(scores) != len(rows):
        raise ValueError("Số dự đoán không khớp số cửa sổ trong fold.")
    scores_are_probabilities = scores.ndim == 2
    if scores_are_probabilities:
        if scores.shape[1] != n_classes:
            raise ValueError("Số cột xác suất không khớp số lớp.")
        window_predictions = np.argmax(scores, axis=1)
    else:
        window_predictions = scores.astype(np.int64, copy=False)

    result = []
    groups = rows[group_col].to_numpy()
    for file_id in pd.unique(groups):
        indices = np.flatnonzero(groups == file_id)
        true_values = np.unique(encoded_labels[indices])
        if len(true_values) != 1:
            raise ValueError(f"File {file_id!r} có nhiều nhãn thực.")
        if scores_are_probabilities:
            file_prediction = int(np.argmax(np.mean(scores[indices], axis=0)))
        else:
            counts = np.bincount(window_predictions[indices], minlength=n_classes)
            file_prediction = int(np.argmax(counts))
        result.append({
            group_col: file_id,
            "true_label_id": int(true_values[0]),
            "predicted_label_id": file_prediction,
            "true_label": str(rows.iloc[indices[0]][label_col]),
        })
    return pd.DataFrame(result)


def _prepare_keras_estimator(estimator, n_classes: int, learning_rate: float):
    """Validate and compile a freshly created Keras classifier for this fold.

    The evaluation routine owns the compile protocol so every fold uses the
    same loss, optimizer, and metrics. The factory should BUILD a fresh model;
    it does not need to compile it first.
    """
    output_shape = getattr(estimator, "output_shape", None)
    compile_method = getattr(estimator, "compile", None)
    fit_method = getattr(estimator, "fit", None)

    if output_shape is None or not callable(compile_method) or not callable(fit_method):
        raise TypeError(
            "estimator_factory phải trả về một Keras Model có output_shape, "
            "compile() và fit(). Hãy kiểm tra factory trong notebook/script."
        )

    # Keras Functional models normally expose (None, n_classes).
    if isinstance(output_shape, list):
        raise ValueError("Mô hình phải có đúng một đầu ra phân loại.")
    if not isinstance(output_shape, tuple) or len(output_shape) < 2:
        raise ValueError(f"output_shape không hợp lệ: {output_shape!r}.")

    actual_classes = output_shape[-1]
    if actual_classes is None or int(actual_classes) != int(n_classes):
        raise ValueError(
            f"Mô hình có {actual_classes} đầu ra, nhưng dữ liệu có "
            f"{n_classes} lớp. Kiểm tra n_classes khi gọi build_primary_mlp()."
        )

    # Compile consistently here, even if the factory returned a precompiled
    # model, to guarantee the same optimizer/loss/metrics for every fold.
    compile_classifier(
        estimator,
        learning_rate=learning_rate,
        n_classes=n_classes,
    )
    return estimator


def run_stratified_group_kfold_evaluation(
    estimator_factory,
    feature_df: pd.DataFrame,
    feature_cols,
    *,
    n_splits: int | None = None,
    seeds: Iterable[int] = range(42, 52),
    label_col: str | None = None,
    group_col: str = "file_id",
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
    learning_rate: float = 1e-3,
):
    """Fit one estimator per fold; score out-of-fold predictions per source file.

    The model is compiled inside this function using compile_classifier(), so
    estimator_factory only needs to return a fresh, uncompiled Keras model.
    The default label column follows config.CLASS_SCHEME_DEFAULT (currently
    the 4-class task uses "label"). Feature normalization is fitted on each
    fold's training windows only.

    Returns models, per-seed file-level metrics, all file predictions, and the
    fitted LabelEncoder (whose classes_ define output-index-to-label mapping).
    """
    if not feature_cols:
        raise ValueError("feature_cols không được rỗng.")
    if group_col not in feature_df.columns:
        raise ValueError(f"feature_df thiếu cột nhóm '{group_col}'.")
    if label_col is None:
        # Use the configured task: CLASS_SCHEME_DEFAULT="4class" -> "label".
        label_col = LABEL_COL_BY_SCHEME[CLASS_SCHEME_DEFAULT]
        if label_col not in feature_df.columns:
            raise ValueError(
                f"Cấu hình CLASS_SCHEME_DEFAULT={CLASS_SCHEME_DEFAULT!r} yêu cầu "
                f"cột nhãn {label_col!r}, nhưng bảng đặc trưng hiện có các cột "
                f"{list(feature_df.columns)}. Hãy truyền label_col rõ ràng hoặc "
                "sửa cấu hình cho đúng bài toán đã chốt."
            )
    else:
        label_col = resolve_label_col(
            feature_df, label_col, scheme=CLASS_SCHEME_DEFAULT
        )
    if feature_df[label_col].isna().any():
        raise ValueError(f"Cột nhãn '{label_col}' có giá trị thiếu.")
    feature_values = feature_df[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Feature matrix chứa NaN hoặc Inf; kiểm tra lại trích xuất đặc trưng "
            "trước khi fit."
        )

    label_strings = feature_df[label_col].astype(str)
    expected_classes = set(CLASSES_BY_SCHEME[CLASS_SCHEME_DEFAULT])
    observed_classes = set(label_strings.unique())
    missing_classes = sorted(expected_classes - observed_classes)
    unexpected_classes = sorted(observed_classes - expected_classes)
    if missing_classes or unexpected_classes:
        raise ValueError(
            f"Nhãn dữ liệu không khớp cấu hình {CLASS_SCHEME_DEFAULT!r}. "
            f"Thiếu lớp: {missing_classes}; nhãn ngoài phạm vi: {unexpected_classes}. "
            f"Cấu hình yêu cầu: {list(CLASSES_BY_SCHEME[CLASS_SCHEME_DEFAULT])}."
        )

    encoder = LabelEncoder()
    encoded_y = np.asarray(encoder.fit_transform(label_strings), dtype=np.int64)
    encoded_frame = feature_df.copy()
    encoded_frame["_encoded_label"] = pd.Series(
        encoded_y, index=encoded_frame.index, dtype="int64"
    )
    

    models = []
    metric_rows = []
    prediction_rows = []
    all_file_ids = set(feature_df[group_col])

    for seed in seeds:
        folds = stratified_group_kfold_splits(
            encoded_frame,
            n_splits=n_splits,
            seed=int(seed),
            label_col="_encoded_label",
            group_col=group_col,
        )
        seed_predictions = []
        for fold_idx, test_df in enumerate(folds):
            test_files = set(test_df[group_col])
            train_df = encoded_frame.loc[
                ~encoded_frame[group_col].isin(test_files)
            ].copy()
            if train_df.empty or test_df.empty:
                raise ValueError(f"Seed {seed}, fold {fold_idx}: Train/Test rỗng.")
            if set(train_df[group_col]) & test_files:
                raise RuntimeError("File bị chia giữa Train và Test trong cùng fold.")

            X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
            X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
            y_train = train_df["_encoded_label"].to_numpy(dtype=np.int64)

            # 1. Lấy thông số chuẩn hóa tĩnh từ tập Train để nhúng thẳng vào mạng (cho MCU)
            norm_mean, norm_var = compute_norm_stats(train_df, feature_cols)
            
            # 2. Tạo model mới (factory chỉ cần build, không cần compile).
            estimator = estimator_factory(
                norm_mean=norm_mean,
                norm_variance=norm_var,
            )
            estimator = _prepare_keras_estimator(
                estimator,
                n_classes=len(encoder.classes_),
                learning_rate=learning_rate,
            )

            # 3. Tính class_weights chỉ trên Train của fold hiện tại.
            c_weights = compute_class_weights(y_train, len(encoder.classes_))

            # 4. Huấn luyện mô hình.
            estimator.fit(
                X_train,
                y_train,
                class_weight=c_weights,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )
            
            # ... (phần predict_scores và file_predictions GIỮ NGUYÊN) ...
            scores = _predict_scores(estimator, X_test, len(encoder.classes_))
            file_predictions = _file_level_predictions(
                test_df,
                scores,
                group_col=group_col,
                label_col=label_col,
                encoded_labels=test_df["_encoded_label"].to_numpy(dtype=np.int64),
                n_classes=len(encoder.classes_),
            )
            file_predictions["seed"] = int(seed)
            file_predictions["fold"] = fold_idx
            seed_predictions.append(file_predictions)
            
            # 5. Lưu lại model thay vì scaler
            models.append({
                "seed": int(seed),
                "fold": fold_idx,
                "norm_mean": norm_mean,
                "norm_variance": norm_var,
                "estimator": estimator,
            })

        seed_result = pd.concat(seed_predictions, ignore_index=True)
        seed_files = set(seed_result[group_col])
        if seed_files != all_file_ids or seed_result[group_col].duplicated().any():
            raise RuntimeError(
                f"Seed {seed}: mỗi file phải xuất hiện đúng một lần ở tập kiểm tra."
            )
        truth = seed_result["true_label_id"].to_numpy(dtype=np.int64)
        predicted = seed_result["predicted_label_id"].to_numpy(dtype=np.int64)
        matrix = confusion_matrix(
            truth, predicted, labels=np.arange(len(encoder.classes_))
        )
        metric_rows.append({
            "seed": int(seed),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(
                truth, predicted, labels=np.arange(len(encoder.classes_)),
                average="macro", zero_division=0,
            )),
            "confusion_matrix": matrix,
            "class_labels": encoder.classes_.tolist(),
        })
        prediction_rows.append(seed_result)

    return models, metric_rows, pd.concat(prediction_rows, ignore_index=True), encoder


def format_summary(metrics_list: list, metric: str = "macro_f1") -> str:
    """Tổng hợp kết quả theo seed dưới dạng mean ± population std."""
    if not metrics_list:
        return "N/A"
    values = [float(row[metric]) for row in metrics_list]
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"
