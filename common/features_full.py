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
                     NOMINAL_RPM_BY_LOAD, SCOPE, WARMUP_SAMPLES)
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
    freqs, mag = dsp.compute_fft(envelope - envelope.mean(), fs)
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
                               *, method="square_law", envelope_kwargs=None):
    envelope = dsp.envelope_by_method(
        x, fs, band=band_hz, method=method, lp_cutoff=lp_cutoff_hz,
        bandpass_order=bandpass_order, lowpass_order=lowpass_order,
        **(envelope_kwargs or {}),
    )
    return extract_envelope_features_from_envelope(
        envelope, fs, rpm, target_names=target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry,
    )


# ============================================================================
# GHÉP CẢ 3 NHÓM
# ============================================================================
def extract_full_feature_vector(x, fs, rpm, band_hz, envelope_window=None,
                                 order_target_names=("f_rot", "BPFO", "BPFI", "BSF"),
                                 envelope_target_names=("BPFO", "BPFI", "BSF"),
                                 harmonics=(1, 2, 3), search_width_hz=3.0,
                                 geometry=SKF_6205_GEOMETRY,
                                 lp_cutoff_hz=dsp.BENCHMARK_LP_CUTOFF_HZ,
                                 bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                                 lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER,
                                 envelope_method="square_law", envelope_kwargs=None):
    feats = {}
    feats.update(extract_time_domain_features(x))
    feats.update(extract_order_domain_features(
        x, fs, rpm, target_names=order_target_names, harmonics=harmonics,
        search_width_hz=search_width_hz, geometry=geometry,
    ))
    
    if envelope_window is not None:
        feats.update(extract_envelope_features_from_envelope(
            envelope_window, fs, rpm, target_names=envelope_target_names,
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


def build_full_feature_table(manifest_df, band_hz, load_de_signal_fn=None,
                              window_size=2048, stride=1024,
                              warmup_samples=None,
                              target_fs_hz=None,
                              lp_cutoff_hz=dsp.BENCHMARK_LP_CUTOFF_HZ,
                              bandpass_order=dsp.BENCHMARK_BANDPASS_ORDER,
                              lowpass_order=dsp.BENCHMARK_LOWPASS_ORDER,
                              envelope_method="square_law",
                              envelope_kwargs=None, **kwargs):
    import pandas as pd
    from pathlib import Path

    if load_de_signal_fn is None:
        from . import io_utils
        load_de_signal_fn = io_utils.load_de_signal_resampled

    target_fs_hz = float(target_fs_hz or SCOPE.get("sampling_rate_hz", 12000))
    warmup_samples = WARMUP_SAMPLES if warmup_samples is None else int(warmup_samples)

    if "resolved_sample_rate_hz" not in manifest_df.columns:
        raise ValueError("manifest_df thiếu cột 'resolved_sample_rate_hz'.")

    rows = []
    n_skipped_no_fs = 0
    file_id_registry: dict[str, str] = {}

    for _, row in manifest_df.iterrows():
        if row.get("label") is None or (isinstance(row.get("label"), float)):
            continue

        load_hp = row.get("load_hp")
        rpm_nominal = NOMINAL_RPM_BY_LOAD.get(load_hp)
        rpm_file = row.get("rpm_from_file")
        
        if rpm_file is not None and not pd.isna(rpm_file) and float(rpm_file) > 0:
            rpm_file = float(rpm_file)
            if rpm_nominal is not None and abs(rpm_file - rpm_nominal) > 20:
                rpm = float(rpm_nominal)
            else:
                rpm = rpm_file
        elif rpm_nominal is not None:
            rpm = float(rpm_nominal)
        else:
            rpm = 1750.0

        file_id = make_file_id(row["file_path"])
        first_seen_path = file_id_registry.setdefault(file_id, str(row["file_path"]))
        if first_seen_path != str(row["file_path"]):
            raise ValueError(f"file_id bị trùng: '{file_id}'")

        resolved_fs = row.get("resolved_sample_rate_hz")
        if resolved_fs is None or pd.isna(resolved_fs):
            n_skipped_no_fs += 1
            continue

        x_full = load_de_signal_fn(Path(row["file_path"]), resolved_fs, target_fs_hz)
        fs = target_fs_hz  
        
        envelope_full = dsp.envelope_by_method(
            x_full, fs, band=band_hz, method=envelope_method,
            lp_cutoff=lp_cutoff_hz, bandpass_order=bandpass_order,
            lowpass_order=lowpass_order, **(envelope_kwargs or {}),
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
            env_window = envelope_full[start_idx:end_idx]

            feats = extract_full_feature_vector(
                x_window, fs=fs, rpm=rpm, band_hz=band_hz,
                envelope_window=env_window,
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