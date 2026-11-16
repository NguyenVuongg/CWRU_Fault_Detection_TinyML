"""
src/dsp_benchmark.py
=======================
Benchmark offline: ứng viên (Hybrid nhân quả, hoặc FFT-Hilbert theo khối cho MCU) so với
FFT-Hilbert toàn file (tham chiếu ngoại tuyến).
Phạm vi: DSP Fidelity + Robustness. Không có Square-Law, ML, chu kỳ CPU, Flash/SRAM, AWGN.

Thiết kế
--------
* Module thuần: không import `src.dsp`. Notebook tạo hai envelope và truyền vào cùng hàm
  giảm mẫu (`decimate`), nên module kiểm thử được độc lập.
* Mọi tham số nằm trong `BenchConfig`.
* Các cột/khóa mang chữ "hybrid" (vd. `hybrid_peak_amplitude`, `lag_samples_hybrid_vs_hilbert`) là
  cột của ỨNG VIÊN; cột `method` ghi ứng viên nào ('Hybrid', 'Hilbert-MCU', ...) và
  `reference_method` ghi tham chiếu. Tên cột giữ nguyên để notebook cũ chạy tiếp.

Quy ước
-------
* lag > 0 : Hybrid trễ hơn Hilbert. Lag đo TRƯỚC căn chỉnh; lag_µs = lag_mẫu / fs × 1e6.
* Pearson, NRMSE: sau căn chỉnh, NRMSE không khớp gain = rms(Hy − Hi) / rms(Hi).
* Peak Amplitude Error (dB) = 20·log10(A_Hybrid / A_Hilbert) trên phổ CHƯA chuẩn hóa,
  cùng đoạn dữ liệu, cùng cửa sổ Hann. Không có đỉnh tham chiếu đáng tin cậy -> NaN (không phải 0)
  và cột `peak_invalid_reason` nói rõ lý do.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import signal

# ----------------------------------------------------------------------------
# Hằng số
# ----------------------------------------------------------------------------
METHOD_NAME = 'Hybrid'
REFERENCE_NAME = 'FFT Hilbert'
FAULT_TYPE_ORDER = ('Normal', 'IR', 'OR', 'Ball')
LOAD_ORDER = (0, 1, 2, 3)


@dataclass(frozen=True)
class BenchConfig:
    fs: float                              # fs sau resample (Hz); dùng đổi lag mẫu -> µs
    warmup: int = 0                        # mẫu bỏ đầu (fs đầu vào)
    edge_trim: int = 0                     # mẫu bỏ cuối (Hilbert theo khối)
    max_lag_samples: int = 100             # tìm lag trong ±N mẫu
    fmax_hz: float = 500.0                 # dải phổ envelope
    search_width_hz: float = 3.0           # tìm đỉnh ±3 Hz quanh tần số lý thuyết
    floor_half_width_hz: float = 15.0      # vùng lấy nền phổ (loại trừ cửa sổ tìm đỉnh)
    peak_snr_min: float = 3.0              # đỉnh nổi bật: ≥ 3 × nền lân cận
    rel_peak_floor: float = 1e-3           # và ≥ 1e-3 × đỉnh lớn nhất của phổ Hilbert
    max_freq_resolution_hz: float = 1.0    # độ phân giải phổ tối đa để tin đỉnh
    min_bins_per_target: float = 10.0      # và ≤ f_mục_tiêu / 10
    harmonics: tuple = (1, 2, 3)
    fault_names: tuple = ('BPFO', 'BPFI', 'BSF', 'FTF')

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d['harmonics'], d['fault_names'] = list(self.harmonics), list(self.fault_names)
        return d


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------
def _finite_1d(values: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError(f'{name} rỗng.')
    if not np.isfinite(a).all():
        raise ValueError(f'{name} có NaN/Inf.')
    return a


def _rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() > 1e-20 and b.std() > 1e-20:
        return float(np.corrcoef(a, b)[0, 1])
    return np.nan


def samples_to_us(lag_samples: float, fs_hz: float) -> float:
    """lag_µs = lag_mẫu / fs × 1e6."""
    return float(lag_samples) / float(fs_hz) * 1e6


# ----------------------------------------------------------------------------
# Phân loại lỗi: Normal / IR / OR / Ball
# ----------------------------------------------------------------------------
_META_COLS = ('fault_type', 'fault_location', 'fault_class', 'bearing_fault_type')


def classify_fault_type(label: Any, filename: Any = '', meta: Mapping | None = None) -> str:
    """'Normal' | 'IR' | 'OR' | 'Ball' | 'Unknown'. Ưu tiên cột metadata, rồi nhãn, rồi tên file.

    Nhận 'Normal', 'IR007', 'OR007@6', 'B014', 'IR_021', 'Inner Race', 'outer', 'ball'...
    """
    texts = []
    if meta is not None and hasattr(meta, 'get'):
        texts += [str(meta[c]) for c in _META_COLS if c in meta and not pd.isna(meta[c])]
    texts += [str(label), str(filename)]
    for raw in texts:
        t = raw.strip().lower()
        if not t:
            continue
        if 'normal' in t or 'healthy' in t or t in {'n', 'baseline'}:
            return 'Normal'
        if 'inner' in t or re.search(r'(^|[^a-z])ir(?=\d|[^a-z]|$)', t):
            return 'IR'
        if 'outer' in t or re.search(r'(^|[^a-z])or(?=\d|[^a-z]|$)', t):
            return 'OR'
        if 'ball' in t or re.search(r'(^|[^a-z])b(?=\d|[^a-z]|$)', t):
            return 'Ball'
    return 'Unknown'


# ----------------------------------------------------------------------------
# 1. Dạng sóng: lag, căn chỉnh, Pearson, NRMSE
# ----------------------------------------------------------------------------
def estimate_lag(hybrid: np.ndarray, hilbert: np.ndarray, cfg: BenchConfig) -> int:
    """Lag (mẫu) tối đa hóa tương quan chéo; lag > 0: hybrid[i+lag] ≈ hilbert[i]."""
    lo, hi = cfg.warmup, len(hilbert) - cfg.edge_trim
    if lo < 0 or cfg.edge_trim < 0 or hi - lo <= 2 * cfg.max_lag_samples + 2:
        raise ValueError('Khoảng warm-up/trim không đủ để ước lượng lag.')
    a = hybrid[lo:hi] - hybrid[lo:hi].mean()
    b = hilbert[lo:hi] - hilbert[lo:hi].mean()
    cc = signal.correlate(a, b, mode='full', method='fft')
    lags = signal.correlation_lags(len(a), len(b), mode='full')
    m = np.abs(lags) <= cfg.max_lag_samples
    lag = int(lags[m][np.argmax(cc[m])])
    if abs(lag) == cfg.max_lag_samples:
        raise ValueError(f'Lag chạm biên tìm kiếm ±{cfg.max_lag_samples} mẫu.')
    return lag


def compare_waveforms(hybrid: ArrayLike, hilbert: ArrayLike, cfg: BenchConfig) -> dict:
    """Một dòng chỉ số: lag (trước căn chỉnh), Pearson và NRMSE (sau căn chỉnh, không khớp gain)."""
    hy, hi = _finite_1d(hybrid, 'hybrid'), _finite_1d(hilbert, 'hilbert')
    if hy.size != hi.size:
        raise ValueError('Hybrid và Hilbert phải cùng độ dài.')
    lag = estimate_lag(hy, hi, cfg)
    i0 = max(cfg.warmup, cfg.warmup - lag)
    i1 = min(len(hi) - cfg.edge_trim, len(hi) - cfg.edge_trim - lag)
    if i1 <= i0:
        raise ValueError('Không còn vùng chồng lấp sau căn chỉnh.')
    a, b = hy[i0 + lag:i1 + lag], hi[i0:i1]
    return {
        'pearson_after_alignment': _pearson(a, b),
        'nrmse_no_gain_fit_after_alignment': _rms(a - b) / max(_rms(b), 1e-12),
        'lag_samples_hybrid_vs_hilbert': int(lag),
        'lag_us_hybrid_vs_hilbert': samples_to_us(lag, cfg.fs),
        'n_samples_compared': int(a.size),
    }


# ----------------------------------------------------------------------------
# 2. Phổ envelope: sai số toàn dải + sai số đỉnh tại tần số lỗi
# ----------------------------------------------------------------------------
def amplitude_spectrum(x: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Phổ biên độ một phía, cửa sổ Hann, bỏ trung bình, KHÔNG chuẩn hóa."""
    if x.size < 8:
        raise ValueError('Cần ≥ 8 mẫu để tính phổ.')
    x = x - x.mean()
    w = np.hanning(x.size)
    amp = np.abs(np.fft.rfft(x * w)) * (2.0 / w.sum())
    amp[0] *= 0.5
    if x.size % 2 == 0:
        amp[-1] *= 0.5
    return np.fft.rfftfreq(x.size, d=1.0 / fs_hz), amp


def find_local_peak(freqs: np.ndarray, amp: np.ndarray, target: float, cfg: BenchConfig):
    """(f_đỉnh, biên độ đỉnh, nền).

    Nền là trung vị các bin trong ±floor_half_width_hz nhưng LOẠI toàn bộ
    cửa sổ tìm đỉnh ±search_width_hz. Nếu không còn bin nền hợp lệ thì trả
    floor=NaN; không lấy chính vùng đỉnh làm nền thay thế.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    amp = np.asarray(amp, dtype=np.float64)
    if freqs.ndim != 1 or amp.ndim != 1 or freqs.size != amp.size or not freqs.size:
        raise ValueError("freqs và amp phải là hai vector 1D cùng độ dài, không rỗng.")

    d = np.abs(freqs - float(target))
    idx = np.flatnonzero(d <= cfg.search_width_hz)
    if idx.size == 0:  # cửa sổ tìm đỉnh hẹp hơn một bin: lấy bin gần nhất
        idx = np.array([int(np.argmin(d))])
    j = int(idx[np.argmax(amp[idx])])

    floor_mask = (d <= cfg.floor_half_width_hz) & (d > cfg.search_width_hz)
    floor_bins = amp[floor_mask]
    floor = float(np.median(floor_bins)) if floor_bins.size else np.nan
    return float(freqs[j]), float(amp[j]), floor


def compare_spectra(hy_dec, hi_dec, fs_env: float, faults: Mapping[str, float], cfg: BenchConfig,
                    warmup_env: int = 0, trim_env: int = 0):
    """Trả về (chỉ số phổ toàn dải, danh sách dòng đỉnh, phổ để vẽ).

    Hai phổ lấy từ CÙNG đoạn, cùng cửa sổ, cùng lưới tần số, nên biên độ so sánh trực tiếp được.
    """
    hy, hi = _finite_1d(hy_dec, 'hybrid_dec'), _finite_1d(hi_dec, 'hilbert_dec')
    if hy.size != hi.size:
        raise ValueError('Envelope giảm mẫu phải cùng độ dài.')
    end = hy.size - trim_env
    if warmup_env < 0 or trim_env < 0 or end - warmup_env < 8:
        raise ValueError('Đoạn envelope sau trim quá ngắn để tính phổ.')
    seg_hy, seg_hi = hy[warmup_env:end], hi[warmup_env:end]
    freqs, s_hy = amplitude_spectrum(seg_hy, fs_env)
    _, s_hi = amplitude_spectrum(seg_hi, fs_env)
    fmax = min(cfg.fmax_hz, fs_env / 2.0)
    band = freqs <= fmax
    pk_hy, pk_hi = float(s_hy[band].max()), float(s_hi[band].max())
    if pk_hy <= 1e-20 or pk_hi <= 1e-20:
        raise ValueError('Phổ envelope không có đỉnh khác 0 trong dải.')
    df = float(fs_env / seg_hy.size)

    # --- sai số phổ toàn dải (0 < f ≤ fmax) ---
    m = band & (freqs > 0)
    r, h = s_hi[m], s_hy[m]
    r_ref, h_ref = r / pk_hi, h / pk_hi            # cùng chuẩn tham chiếu: giữ chênh lệch biên độ
    r_own, h_own = r / pk_hi, h / pk_hy            # mỗi phổ theo đỉnh riêng: chỉ so hình dạng
    file_metrics = {
        'envspec_nrmse_ref_norm': _rms(h_ref - r_ref) / max(_rms(r_ref), 1e-12),
        'envspec_nrmse_own_norm': _rms(h_own - r_own) / max(_rms(r_own), 1e-12),
        'envspec_pearson': _pearson(h, r),
        'spectrum_resolution_hz': df,
        'spectrum_fs_hz': float(fs_env),
        'spectrum_fmax_hz': float(fmax),
    }

    # --- sai số đỉnh tại BPFO/BPFI/BSF/FTF và hài ---
    peaks = []
    for comp in cfg.fault_names:
        if comp not in faults:
            continue
        for k in cfg.harmonics:
            tgt = float(faults[comp]) * int(k)
            row = {'fault_component': comp, 'harmonic': int(k), 'target_frequency_hz': tgt,
                   'freq_resolution_hz': df, 'reference_peak_frequency_hz': np.nan,
                   'hybrid_peak_frequency_hz': np.nan, 'reference_peak_amplitude': np.nan,
                   'hybrid_peak_amplitude': np.nan, 'reference_local_floor': np.nan,
                   'reference_peak_snr': np.nan, 'reference_peak_prominent': False,
                   'peak_valid': False, 'peak_invalid_reason': '',
                   'peak_amplitude_error_db': np.nan, 'abs_peak_amplitude_error_db': np.nan,
                   'peak_frequency_error_hz': np.nan, 'abs_peak_frequency_error_hz': np.nan}
            if not np.isfinite(tgt) or tgt <= 0 or tgt > fmax:
                row['peak_invalid_reason'] = 'out_of_band'
                peaks.append(row); continue
            rf, ra, floor = find_local_peak(freqs, s_hi, tgt, cfg)
            hf, ha, _ = find_local_peak(freqs, s_hy, tgt, cfg)
            snr = ra / floor if np.isfinite(floor) and floor > 0 else np.nan
            prominent = bool(np.isfinite(snr) and snr >= cfg.peak_snr_min and ra >= cfg.rel_peak_floor * pk_hi)
            row.update(reference_peak_frequency_hz=rf, hybrid_peak_frequency_hz=hf,
                       reference_peak_amplitude=ra, hybrid_peak_amplitude=ha,
                       reference_local_floor=floor, reference_peak_snr=snr, reference_peak_prominent=prominent)
            if not np.isfinite(floor) or floor <= 0:
                row['peak_invalid_reason'] = 'no_valid_floor_bins'
            elif not prominent:
                row['peak_invalid_reason'] = 'ref_not_prominent'
            elif not (df <= cfg.max_freq_resolution_hz and df <= tgt / cfg.min_bins_per_target):
                row['peak_invalid_reason'] = 'low_resolution'
            elif not (np.isfinite(ha) and ha > 0):
                row['peak_invalid_reason'] = 'hybrid_peak_missing'
            else:
                db = 20.0 * np.log10(ha / ra)
                row.update(peak_valid=True, peak_invalid_reason='ok',
                           peak_amplitude_error_db=float(db), abs_peak_amplitude_error_db=abs(float(db)),
                           peak_frequency_error_hz=hf - rf, abs_peak_frequency_error_hz=abs(hf - rf))
            peaks.append(row)

    spectra = {'frequencies_hz': freqs, 'fmax_hz': float(fmax), 'hybrid_raw': s_hy, 'hilbert_raw': s_hi,
               'hybrid_norm_own': s_hy / pk_hy, 'hilbert_norm_own': s_hi / pk_hi}
    return file_metrics, peaks, spectra


# ----------------------------------------------------------------------------
# 3. Chạy một file
# ----------------------------------------------------------------------------
Decimator = Callable[[np.ndarray, float], tuple]


def evaluate_case(case: dict, hybrid_env, hilbert_env, cfg: BenchConfig, decimate: Decimator,
                  method_name: str = METHOD_NAME, reference_name: str = REFERENCE_NAME) -> dict:
    """Một file -> {'wave': dict, 'peaks': list[dict], 'spectra': dict}.

    `case` là dict thông tin file; khóa 'faults' (tên -> Hz) dùng cho đỉnh lỗi, các khóa còn lại
    được ghi nguyên vào CSV. `decimate(env, fs)` -> (env_giảm_mẫu, fs_env), ví dụ dsp.decimate_envelope.
    `hybrid_env` là envelope của ỨNG VIÊN (tên tham số giữ nguyên để tương thích); truyền
    `method_name` để ghi đúng tên ứng viên khi đó không phải Hybrid.
    """
    hy, hi = _finite_1d(hybrid_env, 'hybrid'), _finite_1d(hilbert_env, 'hilbert')
    wave = compare_waveforms(hy, hi, cfg)

    hy_d, fs_a = decimate(hy, cfg.fs)
    hi_d, fs_b = decimate(hi, cfg.fs)
    if not np.isclose(float(fs_a), float(fs_b)):
        raise ValueError('fs envelope không đồng nhất giữa hai phương pháp.')
    fs_env = float(fs_a)
    warm = int(np.ceil(cfg.warmup * fs_env / cfg.fs))
    trim = int(np.ceil(cfg.edge_trim * fs_env / cfg.fs))
    spec_metrics, peaks, spectra = compare_spectra(hy_d, hi_d, fs_env, case['faults'], cfg, warm, trim)

    meta = {k: v for k, v in case.items() if k != 'faults'}
    base = {**meta, 'method': method_name, 'reference_method': reference_name}
    return {'wave': {**base, 'fs_hz': cfg.fs, **wave, **spec_metrics},
            'peaks': [{**base, **p} for p in peaks],
            'spectra': spectra}


def evaluate_candidates(case: dict, candidates: Mapping[str, np.ndarray], reference_env,
                        cfg: BenchConfig, decimate: Decimator,
                        reference_name: str = REFERENCE_NAME) -> dict:
    """Một file, NHIỀU ứng viên cùng một tham chiếu: {tên_ứng_viên: kết quả evaluate_case}.

    Ví dụ: candidates={'Hybrid': env_hy, 'Hilbert-MCU': env_hb}, reference_env = FFT-Hilbert
    toàn file. Cùng tham chiếu, cùng cfg -> các dòng so sánh trực tiếp được (thiết kế ghép cặp).
    """
    if not candidates:
        raise ValueError('candidates rỗng.')
    return {name: evaluate_case(case, env, reference_env, cfg, decimate,
                                method_name=name, reference_name=reference_name)
            for name, env in candidates.items()}


# ----------------------------------------------------------------------------
# 4. Bảng tổng hợp
# ----------------------------------------------------------------------------
def _method_label(df: pd.DataFrame, override: str | None = None) -> str:
    """Tên ứng viên cho cột `method` của bảng tổng hợp. Bảng trộn nhiều ứng viên -> phải truyền
    `by=['method', ...]` hoặc lọc trước, nên ở đây báo lỗi thay vì gộp nhầm."""
    if override is not None:
        return override
    if 'method' in df.columns:
        names = df['method'].dropna().unique()
        if len(names) == 1:
            return str(names[0])
        if len(names) > 1:
            raise ValueError(f'Bảng có nhiều ứng viên {list(names)}; lọc theo từng ứng viên '
                             'hoặc dùng by=[\'method\', ...] qua wave_summary_by_method().')
    return METHOD_NAME

def _p95(s: pd.Series) -> float:
    s = s.dropna()
    return float(np.percentile(s, 95)) if len(s) else np.nan


def wave_summary(wave: pd.DataFrame, by: list | None = None, method_name: str | None = None) -> pd.DataFrame:
    """Fidelity tổng hợp. by=None: toàn bộ; by=['fault_type'] hoặc ['load_hp'] cho Robustness.
    Bảng chỉ được chứa MỘT ứng viên (cột `method`); nhiều ứng viên -> wave_summary_by_method()."""
    g = wave.groupby(by, dropna=False) if by else wave.assign(_all='all').groupby('_all')
    out = pd.DataFrame(g.agg(
        n_files=('case_id', 'nunique'),
        pearson_mean=('pearson_after_alignment', 'mean'),
        pearson_std=('pearson_after_alignment', 'std'),
        pearson_min=('pearson_after_alignment', 'min'),
        nrmse_mean=('nrmse_no_gain_fit_after_alignment', 'mean'),
        nrmse_median=('nrmse_no_gain_fit_after_alignment', 'median'),
        nrmse_std=('nrmse_no_gain_fit_after_alignment', 'std'),
        nrmse_max=('nrmse_no_gain_fit_after_alignment', 'max'),
        lag_samples_abs_median=('lag_samples_hybrid_vs_hilbert', lambda s: float(np.median(np.abs(s)))),
        lag_us_mean=('lag_us_hybrid_vs_hilbert', 'mean'),
        lag_us_abs_median=('lag_us_hybrid_vs_hilbert', lambda s: float(np.median(np.abs(s)))),
        envspec_nrmse_ref_norm_mean=('envspec_nrmse_ref_norm', 'mean'),
        envspec_nrmse_own_norm_mean=('envspec_nrmse_own_norm', 'mean'),
        envspec_pearson_mean=('envspec_pearson', 'mean'),
    ))
    out = out.reset_index(drop=not by)
    out.insert(0, 'method', _method_label(wave, method_name))
    return out


def wave_summary_by_method(wave: pd.DataFrame, by: list | None = None) -> pd.DataFrame:
    """wave_summary() cho từng ứng viên trong cột `method`, nối thành một bảng."""
    if 'method' not in wave.columns:
        return wave_summary(wave, by)
    parts = [wave_summary(pd.DataFrame(g), by) for _, g in wave.groupby('method', sort=False)]
    return pd.concat(parts, ignore_index=True)


def robustness_table(wave: pd.DataFrame, group_col: str, order: Iterable,
                     method_name: str | None = None) -> pd.DataFrame:
    """Pearson/NRMSE theo nhóm đã biết (Normal/IR/OR/Ball hoặc 0–3 HP).

    Nhóm không có file vẫn hiện với n_files = 0; giá trị ngoài `order` (vd. Unknown) bị loại.
    """
    order = list(order)
    w = wave.copy()
    if group_col == 'load_hp':
        w[group_col] = pd.to_numeric(w[group_col], errors='coerce')
    label = _method_label(w, method_name)
    s = wave_summary(pd.DataFrame(w[w[group_col].isin(order)]), [group_col], method_name=label).set_index(group_col)
    s = pd.DataFrame(s.reindex(order))
    s['n_files'] = s['n_files'].fillna(0).astype(int)
    s['method'] = label
    s.index.name = group_col
    return s.reset_index()


def peak_summary(peaks: pd.DataFrame, by: list) -> pd.DataFrame:
    """Thống kê chỉ trên đỉnh hợp lệ (peak_valid). NaN không bị tính là 0."""
    out = pd.DataFrame(peaks.groupby(by, as_index=False, dropna=False).agg(
        n_rows=('case_id', 'size'), n_files=('case_id', 'nunique'),
        n_prominent=('reference_peak_prominent', 'sum'), n_valid_peak=('peak_valid', 'sum'),
        peak_error_db_median_signed=('peak_amplitude_error_db', 'median'),
        peak_error_db_mean_signed=('peak_amplitude_error_db', 'mean'),
        abs_peak_error_db_median=('abs_peak_amplitude_error_db', 'median'),
        abs_peak_error_db_mean=('abs_peak_amplitude_error_db', 'mean'),
        abs_peak_error_db_p95=('abs_peak_amplitude_error_db', _p95),
        abs_peak_freq_error_hz_median=('abs_peak_frequency_error_hz', 'median'),
        abs_peak_freq_error_hz_p95=('abs_peak_frequency_error_hz', _p95),
    ))
    out['note'] = np.where(out['n_valid_peak'] == 0, 'no valid reference peak', '')
    return out