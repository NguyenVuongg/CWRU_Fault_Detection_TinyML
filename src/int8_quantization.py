# -*- coding: utf-8 -*-
"""Fold-safe INT8 export and evaluation for the CWRU MLP.

The float model used by the CV pipeline consumes StandardScaler-transformed
features.  Keep that contract for the INT8 model as well: quantize z-scores,
and apply the training-fold scaler once before quantizing each input tensor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def balanced_representative_indices(
    frame: pd.DataFrame,
    *,
    n_samples: int = 256,
    seed: int = 42,
    group_col: str = "file_id",
    label_col: str = "label",
    load_col: str = "load_hp",
) -> np.ndarray:
    """Choose train windows round-robin across files, labels, and available loads."""
    if frame.empty or n_samples <= 0:
        raise ValueError("Representative source and n_samples must be non-empty/positive.")
    if group_col not in frame or label_col not in frame:
        raise ValueError(f"Representative frame must include {group_col!r} and {label_col!r}.")
    rng = np.random.default_rng(int(seed))
    work = frame.reset_index(drop=True).copy()
    if load_col not in work:
        work[load_col] = "all"
    # Shuffle within each source-file stratum, then take one row from every
    # available (class, load, file) stratum before taking second windows.
    strata: list[np.ndarray] = []
    for _, group in work.groupby([label_col, load_col, group_col], dropna=False, sort=True):
        ids = group.index.to_numpy(dtype=np.int64, copy=True)
        rng.shuffle(ids)
        strata.append(ids)
    rng.shuffle(strata)
    selected: list[int] = []
    offsets = np.zeros(len(strata), dtype=np.int64)
    while len(selected) < min(int(n_samples), len(work)):
        added = False
        for i, ids in enumerate(strata):
            if offsets[i] < len(ids):
                selected.append(int(ids[offsets[i]]))
                offsets[i] += 1
                added = True
                if len(selected) >= min(int(n_samples), len(work)):
                    break
        if not added:
            break
    return np.asarray(selected, dtype=np.int64)


def _tflite_probabilities(interpreter, input_detail, output_detail, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]
    if in_scale <= 0 or out_scale <= 0:
        raise ValueError("TFLite input/output quantization scale must be positive.")
    q_dtype = input_detail["dtype"]
    q_limits = np.iinfo(q_dtype)
    quantized = np.rint(X / in_scale + in_zero)
    clipped = np.clip(quantized, q_limits.min, q_limits.max).astype(q_dtype)
    probabilities = np.empty((len(X), output_detail["shape"][-1]), dtype=np.float32)
    for i, row in enumerate(clipped):
        interpreter.set_tensor(input_detail["index"], row.reshape(1, -1))
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0]
        if np.issubdtype(output_detail["dtype"], np.integer):
            probabilities[i] = (raw.astype(np.float32) - float(out_zero)) * float(out_scale)
        else:
            probabilities[i] = raw.astype(np.float32)
    dequantized_input = (clipped.astype(np.float32) - float(in_zero)) * float(in_scale)
    return probabilities, dequantized_input


def _per_file_rows(frame: pd.DataFrame, probabilities: np.ndarray, class_names: Sequence[str], *,
                   seed: int, fold: int, version: str, group_col: str, label_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = frame[group_col].astype(str).to_numpy()
    for file_id in pd.unique(groups):
        idx = np.flatnonzero(groups == file_id)
        truths = frame.iloc[idx][label_col].astype(str).unique()
        if len(truths) != 1:
            raise ValueError(f"File {file_id!r} has multiple ground-truth labels.")
        mean_p = np.asarray(probabilities[idx], dtype=np.float64).mean(axis=0)
        rows.append({
            "file_id": file_id, "y_true": truths[0],
            "y_pred": str(class_names[int(np.argmax(mean_p))]),
            "n_windows": int(len(idx)), "seed": int(seed), "fold": int(fold),
            "version": version,
            **{f"p_{name}": float(mean_p[j]) for j, name in enumerate(class_names)},
        })
    return rows


def run_int8_cross_validation(
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    folds: Mapping[int, Sequence[tuple[Sequence[str], Sequence[str]]]],
    class_names: Sequence[str],
    class_to_id: Mapping[str, int],
    mlp_kwargs: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    representative_count: int = 256,
    representative_seed: int = 24042,
    group_col: str = "file_id",
    label_col: str = "label",
) -> dict[str, Any]:
    """Evaluate Keras FP32, TFLite FP32, and TFLite INT8 with file-level OOF folds.

    Each fold fits its StandardScaler and representative set from train files
    only. The test windows are transformed once and passed to the three models.
    """
    import tensorflow as tf
    from . import models

    required = {group_col, label_col, *feature_cols}
    missing = sorted(required - set(feature_df.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    if not feature_df[list(feature_cols)].apply(lambda col: np.isfinite(col.to_numpy(dtype=float)).all()).all():
        raise ValueError("Feature table contains NaN or Inf.")

    file_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    quant_rows: list[dict[str, Any]] = []
    model_sizes: list[dict[str, Any]] = []
    versions = ("keras_float32", "tflite_float32", "tflite_int8")

    for seed in seeds:
        if int(seed) not in folds:
            raise KeyError(f"No file folds for seed {seed}.")
        for fold_id, (train_files, test_files) in enumerate(folds[int(seed)], start=1):
            train_set, test_set = set(map(str, train_files)), set(map(str, test_files))
            if train_set & test_set:
                raise RuntimeError("A source file appears in both train and test.")
            train_df = feature_df[feature_df[group_col].astype(str).isin(train_set)].reset_index(drop=True)
            test_df = feature_df[feature_df[group_col].astype(str).isin(test_set)].reset_index(drop=True)
            if train_df.empty or test_df.empty:
                raise RuntimeError(f"Empty train/test data for seed={seed}, fold={fold_id}.")

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[list(feature_cols)].to_numpy(dtype=np.float32)).astype(np.float32)
            X_test_raw = test_df[list(feature_cols)].to_numpy(dtype=np.float32)
            X_test = scaler.transform(X_test_raw).astype(np.float32)
            y_train = train_df[label_col].astype(str).map(class_to_id)
            if y_train.isna().any():
                raise ValueError("Training labels are not present in class_to_id.")
            y_train = y_train.to_numpy(dtype=np.int32)

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(int(seed) * 100 + int(fold_id))
            model = models.build_primary_mlp(
                input_dim=len(feature_cols), n_classes=len(class_names), **dict(mlp_kwargs)
            )
            models.compile_classifier(model, learning_rate=learning_rate, n_classes=len(class_names))
            model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)

            keras_prob = np.asarray(model.predict(X_test, verbose=0), dtype=np.float32)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_float = converter.convert()
            float_interpreter = tf.lite.Interpreter(model_content=tflite_float)
            float_interpreter.allocate_tensors()
            float_in = float_interpreter.get_input_details()[0]
            float_out = float_interpreter.get_output_details()[0]
            tflite_float_prob = np.empty_like(keras_prob)
            for i, sample in enumerate(X_test):
                float_interpreter.set_tensor(float_in["index"], sample.reshape(1, -1).astype(float_in["dtype"]))
                float_interpreter.invoke()
                tflite_float_prob[i] = float_interpreter.get_tensor(float_out["index"])[0]

            rep_idx = balanced_representative_indices(
                train_df, n_samples=representative_count,
                seed=int(representative_seed) + int(seed) * 100 + fold_id,
                group_col=group_col, label_col=label_col,
            )
            rep_scaled = scaler.transform(train_df.iloc[rep_idx][list(feature_cols)].to_numpy(dtype=np.float32)).astype(np.float32)

            def representative_dataset():
                for row in rep_scaled:
                    yield [row.reshape(1, -1)]

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            tflite_int8 = converter.convert()
            int8_interpreter = tf.lite.Interpreter(model_content=tflite_int8)
            int8_interpreter.allocate_tensors()
            int8_in = int8_interpreter.get_input_details()[0]
            int8_out = int8_interpreter.get_output_details()[0]
            int8_prob, X_test_dequant = _tflite_probabilities(int8_interpreter, int8_in, int8_out, X_test)

            probs_by_version = {
                "keras_float32": keras_prob,
                "tflite_float32": tflite_float_prob,
                "tflite_int8": int8_prob,
            }
            for version, probabilities in probs_by_version.items():
                window_pred = np.argmax(probabilities, axis=1)
                for j, (_, item) in enumerate(test_df.iterrows()):
                    row = {
                        "file_id": str(item[group_col]), "window_idx": int(item.get("window_idx", j)),
                        "y_true": str(item[label_col]), "y_pred": str(class_names[int(window_pred[j])]),
                        "seed": int(seed), "fold": int(fold_id), "version": version,
                    }
                    row.update({f"p_{name}": float(probabilities[j, k]) for k, name in enumerate(class_names)})
                    window_rows.append(row)
                file_rows.extend(_per_file_rows(
                    test_df, probabilities, class_names, seed=int(seed), fold=fold_id,
                    version=version, group_col=group_col, label_col=label_col,
                ))

            y_test_ids = test_df[label_col].astype(str).map(class_to_id).to_numpy(dtype=np.int32)
            for version, probabilities in probs_by_version.items():
                y_pred = np.argmax(probabilities, axis=1)
                quant_rows.append({
                    "seed": int(seed), "fold": int(fold_id), "version": version,
                    "top1_agreement_vs_keras": float(np.mean(y_pred == np.argmax(keras_prob, axis=1))),
                    "mean_probability_error_vs_keras": float(np.abs(probabilities - keras_prob).mean()),
                    "p95_probability_error_vs_keras": float(np.percentile(np.abs(probabilities - keras_prob), 95)),
                    "max_probability_error_vs_keras": float(np.abs(probabilities - keras_prob).max()),
                    "window_accuracy": float(accuracy_score(y_test_ids, y_pred)),
                    "window_macro_f1": float(f1_score(y_test_ids, y_pred, labels=np.arange(len(class_names)), average="macro", zero_division=0)),
                })

            in_scale, in_zero = int8_in["quantization"]
            q_limits = np.iinfo(int8_in["dtype"])
            ideal_q = np.rint(X_test / in_scale + in_zero)
            clipped_q = np.clip(ideal_q, q_limits.min, q_limits.max)
            feature_abs_error = np.abs(X_test_dequant - X_test)
            for j, feature in enumerate(feature_cols):
                quant_rows.append({
                    "seed": int(seed), "fold": int(fold_id), "version": "input_quantization_feature",
                    "feature": str(feature), "input_scale": float(in_scale), "input_zero_point": int(in_zero),
                    "input_min": float(X_test[:, j].min()), "input_max": float(X_test[:, j].max()),
                    "input_mae": float(feature_abs_error[:, j].mean()),
                    "input_p95_abs_error": float(np.percentile(feature_abs_error[:, j], 95)),
                    "quantized_min": float(X_test_dequant[:, j].min()),
                    "quantized_max": float(X_test_dequant[:, j].max()),
                    "saturation_fraction": float(np.mean((ideal_q[:, j] < q_limits.min) | (ideal_q[:, j] > q_limits.max))),
                })
            model_sizes.append({
                "seed": int(seed), "fold": int(fold_id),
                "tflite_float32_bytes": len(tflite_float), "tflite_int8_bytes": len(tflite_int8),
                "int8_input_scale": float(in_scale), "int8_input_zero_point": int(in_zero),
                "int8_output_scale": float(int8_out["quantization"][0]),
                "int8_output_zero_point": int(int8_out["quantization"][1]),
                "representative_count": int(len(rep_idx)),
                "representative_files": int(train_df.iloc[rep_idx][group_col].nunique()),
            })
            del model
            tf.keras.backend.clear_session()

    file_predictions = pd.DataFrame(file_rows)
    window_predictions = pd.DataFrame(window_rows)
    quantization_diagnostics = pd.DataFrame(quant_rows)
    sizes = pd.DataFrame(model_sizes)
    metrics_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for level, predictions in (("file", file_predictions), ("window", window_predictions)):
        for (version, seed), group in predictions.groupby(["version", "seed"], sort=True):
            truth = group["y_true"].map(class_to_id).to_numpy(dtype=np.int32)
            pred = group["y_pred"].map(class_to_id).to_numpy(dtype=np.int32)
            cm = confusion_matrix(truth, pred, labels=np.arange(len(class_names)))
            row: dict[str, Any] = {
                "level": level, "version": version, "seed": int(seed),
                "accuracy": float(accuracy_score(truth, pred)),
                "macro_f1": float(f1_score(truth, pred, labels=np.arange(len(class_names)), average="macro", zero_division=0)),
            }
            precision = precision_score(truth, pred, labels=np.arange(len(class_names)), average=None, zero_division=0)
            recall = recall_score(truth, pred, labels=np.arange(len(class_names)), average=None, zero_division=0)
            for class_id, class_name in enumerate(class_names):
                row[f"precision_{class_name}"] = float(precision[class_id])
                row[f"recall_{class_name}"] = float(recall[class_id])
            metrics_rows.append(row)
            for i, true_name in enumerate(class_names):
                for j, predicted_name in enumerate(class_names):
                    confusion_rows.append({
                        "level": level, "version": version, "seed": int(seed), "true": true_name,
                        "predicted": predicted_name, "count": int(cm[i, j]),
                    })
    return {
        "file_predictions": file_predictions,
        "window_predictions": window_predictions,
        "metrics_by_seed": pd.DataFrame(metrics_rows),
        "confusion_by_seed": pd.DataFrame(confusion_rows),
        "quantization_diagnostics": quantization_diagnostics,
        "model_sizes": sizes,
    }


def export_standardized_int8(
    model,
    X_train_standardized: np.ndarray,
    representative_frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    feature_cols: Sequence[str],
    class_names: Sequence[str],
    scaler_mean: Sequence[float],
    scaler_scale: Sequence[float],
    seed: int = 42,
    representative_count: int = 256,
) -> dict[str, Any]:
    """Export a separate INT8 artifact whose model input is standardized float features."""
    import tensorflow as tf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = balanced_representative_indices(
        representative_frame, n_samples=representative_count, seed=seed
    )
    reps = np.asarray(X_train_standardized[indices], dtype=np.float32)

    def representative_dataset():
        for row in reps:
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    model_path = output_dir / "mlp_4class_hybrid_standardized_int8.tflite"
    model_path.write_bytes(model_bytes)

    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise RuntimeError(f"Expected INT8 I/O, got {inp['dtype']} -> {out['dtype']}.")
    info = {
        "model_path": str(model_path.resolve()), "model_size_bytes": int(len(model_bytes)),
        "input_dtype": str(inp["dtype"]), "input_shape": inp["shape"].tolist(),
        "input_scale": float(inp["quantization"][0]), "input_zero_point": int(inp["quantization"][1]),
        "output_dtype": str(out["dtype"]), "output_shape": out["shape"].tolist(),
        "output_scale": float(out["quantization"][0]), "output_zero_point": int(out["quantization"][1]),
        "feature_columns_in_input_order": list(feature_cols),
        "class_names_and_output_order": list(class_names),
        "input_preprocessing": "Apply exactly one fold-independent final StandardScaler: (x_raw - scaler_mean) / scaler_scale, then quantize the standardized vector using input_scale/input_zero_point. Do not pass raw features and do not standardize twice.",
        "scaler_mean": [float(v) for v in scaler_mean], "scaler_scale": [float(v) for v in scaler_scale],
        "representative_seed": int(seed), "representative_count": int(len(indices)),
        "representative_source": "Final-fit training feature table only; selected round-robin across class/load/source-file strata.",
        "representative_files": int(representative_frame.iloc[indices]["file_id"].nunique()),
    }
    rep_manifest = representative_frame.iloc[indices].copy()
    manifest_cols = [c for c in ("file_id", "window_idx", "label", "class_label", "load_hp") if c in rep_manifest]
    rep_manifest[manifest_cols].to_csv(output_dir / "representative_samples.csv", index=False)
    (output_dir / "quantization_config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"model_bytes": model_bytes, "interpreter": interpreter, "input_detail": inp, "output_detail": out, "metadata": info}
