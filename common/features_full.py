# -*- coding: utf-8 -*-
"""
common/features_full.py
=========================
"""

import numpy as np
from scipy.stats import kurtosis, skew
from pathlib import Path

from . import dsp
from .config import (bearing_fault_frequencies, make_class_label,
                     SKF_6205_GEOMETRY,
                     SCOPE, WARMUP_SAMPLES)
from .order import extract_order_features


# ============================================================================
# ĐỊNH DANH FILE GỐC cho File-based Split
# ============================================================================
def make_file_id(file_path) -> str:
    """Định danh FILE GỐC — KHÔNG được chứa chỉ số cửa sổ.
    Đây là khóa mà splitting.file_based_split() dùng để tách train/test và
    ngăn rò rỉ dữ liệu giữa các cửa sổ trượt CHỒNG LẤN của cùng một file.
    """
    return Path(file_path).stem


# ============================================================================
# NHÓM A - Miền thời gian (11 đặc trưng)
# ============================================================================
def extract_time_domain_features(x, prefix="time"):
    """11 đặc trưng miền thời gian cơ bản."""
    x = np.asarray(x, dtype=np.float64)
    mean_abs = np.mean(np.abs(x))
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(x)))

    eps = 1e-12
    feats = {
        "mean": np.mean(x),
        "std": np.std(x),
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis(x),
        "skewness": skew(x),
        "variance": np.var(x),
        "crest_factor": peak / (rms + eps),
        "shape_factor": rms / (mean_abs + eps),
        "impulse_factor": peak / (mean_abs + eps),
        "margin_factor": peak / (mean_sqrt_abs ** 2 + eps),
    }
    return {f"{prefix}_{k}": v for k, v in feats.items()}


# ============================================================================
# NHÓM B - Phổ Order 
# ============================================================================
def extract_order_domain_features(x, fs, rpm, target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                                   harmonics=(1, 2, 3), search_width_hz=3.0,
                                   geometry=SKF_6205_GEOMETRY):
    freqs, mag = dsp.compute_fft(x, fs)
    return extract_order_features(
        freqs, mag, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry, prefix="order",
    )


# ============================================================================
# NHÓM C - Phổ Envelope
# ============================================================================
def extract_envelope_features_from_envelope(envelope, fs, rpm,
                                             target_names=("BPFO", "BPFI", "BSF"),
                                             harmonics=(1, 2, 3), search_width_hz=3.0,
                                             geometry=SKF_6205_GEOMETRY):
    envelope = np.asarray(envelope, dtype=np.float64)
    freqs, mag = dsp.compute_fft(envelope, fs)
    return extract_order_features(
        freqs, mag, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry, prefix="envelope",
    )


def extract_envelope_features(x, fs, rpm, band_hz,
                               lp_cutoff_hz=dsp.BENCHMARK_LP_CUTOFF_HZ,
                               bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                               lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER,
                               target_names=("BPFO", "BPFI", "BSF"),
                               harmonics=(1, 2, 3), search_width_hz=3.0,
                               geometry=SKF_6205_GEOMETRY,
                               *, method="hybrid", envelope_kwargs=None):
    envelope = dsp.envelope_by_method(
        x, fs, band=band_hz, method=method, lp_cutoff=lp_cutoff_hz,
        bandpass_order=bandpass_order, lowpass_order=lowpass_order,
        **(envelope_kwargs or {}),
    )
    envelope, envelope_fs = dsp.decimate_envelope(
        envelope, fs, factor=dsp.ENV_DECIM, lp_cutoff=lp_cutoff_hz
    )
    return extract_envelope_features_from_envelope(
        envelope, envelope_fs, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry,
    )


# ============================================================================
# GHÉP CẢ 3 NHÓM
# ============================================================================
def extract_full_feature_vector(x, fs, rpm, band_hz, envelope_window=None,
                                 envelope_fs=None,
                                 order_target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                                 envelope_target_names=("BPFO", "BPFI", "BSF"),
                                 harmonics=(1, 2, 3), search_width_hz=3.0,
                                 geometry=SKF_6205_GEOMETRY,
                                 lp_cutoff_hz=dsp.BENCHMARK_LP_CUTOFF_HZ,
                                 bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                                 lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER,
                                 envelope_method="hybrid", envelope_kwargs=None):
    feats = {}
    feats.update(extract_time_domain_features(x))
    feats.update(extract_order_domain_features(
        x, fs, rpm, target_names=order_target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry,
    ))
    
    if envelope_window is not None:
        feats.update(extract_envelope_features_from_envelope(
            envelope_window, fs if envelope_fs is None else envelope_fs,
            rpm, target_names=envelope_target_names,
            harmonics=harmonics, search_width_hz=search_width_hz, geometry=geometry,
        ))
    else:
        feats.update(extract_envelope_features(
            x, fs, rpm, band_hz=band_hz, lp_cutoff_hz=lp_cutoff_hz,
            bandpass_order=bandpass_order, lowpass_order=lowpass_order,
            target_names=envelope_target_names, harmonics=harmonics,
            search_width_hz=search_width_hz, geometry=geometry,
            method=envelope_method, envelope_kwargs=envelope_kwargs,
        ))
    return feats


def _validate_feature_band(band_hz):
    try:
        band = tuple(float(value) for value in band_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"band_hz phải bằng dải cố định {dsp.BENCHMARK_BAND_HZ}, nhận {band_hz!r}."
        ) from exc
    if band != dsp.BENCHMARK_BAND_HZ:
        raise ValueError(
            f"Bảng đặc trưng chính dùng band cố định {dsp.BENCHMARK_BAND_HZ} Hz; "
            f"nhận {band} Hz."
        )
    return dsp.BENCHMARK_BAND_HZ


def build_full_feature_table(manifest_df, band_hz=dsp.BENCHMARK_BAND_HZ,
                              load_de_signal_fn=None,
                              window_size=2048, stride=1024,
                              warmup_samples=None,
                              target_fs_hz=None,
                              lp_cutoff_hz=dsp.BENCHMARK_LP_CUTOFF_HZ,
                              bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                              lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER,
                              envelope_method="hybrid",
                              envelope_kwargs=None, **kwargs):
    import pandas as pd
    from pathlib import Path

    if load_de_signal_fn is None:
        from . import io_utils
        load_de_signal_fn = io_utils.load_de_signal_resampled

    target_fs_hz = float(target_fs_hz or SCOPE.get("sampling_rate_hz", 12000))
    warmup_samples = WARMUP_SAMPLES if warmup_samples is None else int(warmup_samples)
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size và stride phải là số nguyên dương.")
    if window_size % dsp.ENV_DECIM or stride % dsp.ENV_DECIM:
        raise ValueError(
            f"window_size và stride phải chia hết cho ENV_DECIM={dsp.ENV_DECIM}."
        )

    band_hz = _validate_feature_band(band_hz)
    required_manifest_columns = (
        "resolved_sample_rate_hz", "resolved_sample_rate_source", "rpm", "rpm_source",
    )
    missing_manifest_columns = [
        column for column in required_manifest_columns
        if column not in manifest_df.columns
    ]
    if missing_manifest_columns:
        raise ValueError(
            "manifest_df thiếu cột bắt buộc "
            f"{missing_manifest_columns}; feature build chỉ đọc metadata từ manifest."
        )

    rows = []
    file_id_registry: dict[str, str] = {}

    for _, row in manifest_df.iterrows():
        if row.get("label") is None or (isinstance(row.get("label"), float)):
            continue

        load_hp = row.get("load_hp")
        rpm_value = row.get("rpm")
        rpm_source = row.get("rpm_source")
        if rpm_value is None or pd.isna(rpm_value):
            raise ValueError(
                f"Manifest thiếu RPM hợp lệ cho {row.get('file_path')}; "
                "hãy xử lý RPM trong bước lập manifest."
            )
        rpm = float(rpm_value)
        if not np.isfinite(rpm) or rpm <= 0.0:
            raise ValueError(
                f"Manifest có RPM không hợp lệ ({rpm_value!r}) "
                f"cho {row.get('file_path')}."
            )
        if rpm_source is None or pd.isna(rpm_source) or not str(rpm_source).strip():
            raise ValueError(
                f"Manifest thiếu rpm_source cho {row.get('file_path')}."
            )

        file_id = make_file_id(row["file_path"])
        first_seen_path = file_id_registry.setdefault(file_id, str(row["file_path"]))
        if first_seen_path != str(row["file_path"]):
            raise ValueError(f"file_id bị trùng: '{file_id}'")

        resolved_fs = row.get("resolved_sample_rate_hz")
        if resolved_fs is None or pd.isna(resolved_fs) or not np.isfinite(float(resolved_fs)):
            raise ValueError(
                f"Không xác định được sampling rate đáng tin cậy cho {row['file_path']}."
            )

        x_full = load_de_signal_fn(Path(row["file_path"]), resolved_fs, target_fs_hz)
        x_full = np.asarray(x_full, dtype=np.float64)
        if x_full.ndim != 1 or not len(x_full):
            raise ValueError(
                f"Tín hiệu DE phải là mảng 1D không rỗng: {row['file_path']}."
            )
        fs = target_fs_hz  
        
        envelope_full = dsp.envelope_by_method(
            x_full, fs, band=band_hz, method=envelope_method,
            lp_cutoff=lp_cutoff_hz, bandpass_order=bandpass_order,
            lowpass_order=lowpass_order, **(envelope_kwargs or {}),
        )
        if len(envelope_full) != len(x_full):
            raise ValueError(
                f"Envelope length {len(envelope_full)} không khớp tín hiệu "
                f"length {len(x_full)} tại {row['file_path']}."
            )
        envelope_decimated, envelope_fs = dsp.decimate_envelope(
            envelope_full, fs, factor=dsp.ENV_DECIM, lp_cutoff=lp_cutoff_hz
        )

        num_windows = (len(x_full) - window_size) // stride + 1
        if num_windows <= 0:
            continue

        for i in range(num_windows):
            start_idx = i * stride
            if start_idx < warmup_samples:
                continue
            end_idx = start_idx + window_size
            x_window = x_full[start_idx:end_idx]
            env_start = start_idx // dsp.ENV_DECIM
            env_end = end_idx // dsp.ENV_DECIM
            env_window = envelope_decimated[env_start:env_end]
            if len(env_window) != window_size // dsp.ENV_DECIM:
                raise ValueError(
                    f"Envelope window length {len(env_window)} không đúng "
                    f"{window_size // dsp.ENV_DECIM} tại {row['file_path']}."
                )

            feats = extract_full_feature_vector(
                x_window, fs=fs, rpm=rpm, band_hz=band_hz,
                envelope_window=env_window, envelope_fs=envelope_fs,
                lp_cutoff_hz=lp_cutoff_hz, bandpass_order=bandpass_order,
                lowpass_order=lowpass_order, **kwargs
            )

            rows.append({
                "file_id": file_id,
                "window_idx": i,
                "start_idx": start_idx,
                "label": row["label"],
                "class_label": make_class_label(
                    row["label"], row.get("fault_diameter_mils")),
                "load_hp": load_hp,
                "fault_diameter_mils": row.get("fault_diameter_mils"),
                "rpm": rpm,
                "rpm_source": str(rpm_source),
                # Preserve acquisition-rate provenance for auditability; these
                # columns are metadata and must never be used as model features.
                "source_sample_rate_hz": float(resolved_fs),
                "source_sample_rate_source": str(row.get("resolved_sample_rate_source")),
                "feature_sample_rate_hz": float(target_fs_hz),
                **feats,
            })

    return pd.DataFrame(rows)


METADATA_COLUMNS = (
    "file_id",
    "window_idx",
    "start_idx",
    "signal_idx",
    "label",
    "class_label",
    "load_hp",
    "fault_diameter_mils",
    "rpm",
    "rpm_source",
    "source_sample_rate_hz",
    "source_sample_rate_source",
    "feature_sample_rate_hz",
    "dsp_method",
)

FEATURE_PREFIXES = ("time_", "order_", "envelope_")

EXPECTED_FEATURE_COUNTS = {
    "proposal_time": 11, "proposal_order": 12, "proposal_envelope": 9,
    "proposal_total": 32,
    "realized_time": 11, "realized_order": 10, "realized_envelope": 7,
    "realized_total": 28,
    "reason": (
        "FFT 2048 @ 12 kHz = 5.86 Hz/bin; BSF_h3 cách BPFO_h2 2.98 Hz và "
        "BPFO_h3 cách BPFI_h2 2.28 Hz -> dưới 1 bin, nên order.py gộp mỗi "
        "cặp thành 1 cột. Việc gộp này đảm bảo tính ổn định của mô hình qua "
        "các mức tải khác nhau trong cùng một tập dữ liệu."
    ),
}

def feature_columns(feature_df, expected_count=None, strict_prefix=True):
    meta = set(METADATA_COLUMNS)
    cols = [c for c in feature_df.columns if c not in meta]

    if strict_prefix:
        la = [c for c in cols if not c.startswith(FEATURE_PREFIXES)]
        if la:
            raise ValueError(
                "Các cột sau không phải metadata nhưng cũng không mang tiền "
                f"tố đặc trưng {FEATURE_PREFIXES}: {la}."
            )

    if expected_count is not None and len(cols) != expected_count:
        raise ValueError(
            f"Kỳ vọng {expected_count} cột đặc trưng nhưng tìm thấy "
            f"{len(cols)}. Danh sách thực tế: {cols}"
        )

    return cols