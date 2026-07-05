import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert, butter, sosfiltfilt
from typing import List, Dict, Sequence, Any, cast
from src import config

def extract_time_features(segment: np.ndarray) -> List[float]:
    """Trích xuất 6 đặc trưng thống kê miền thời gian."""
    eps = 1e-10
    rms = float(np.sqrt(np.mean(segment**2)))
    peak = float(np.max(np.abs(segment)))
    kurt = float(kurtosis(segment, fisher=False))
    skewness = float(skew(segment))
    crest_factor = float(peak / (rms + eps))
    mean_abs = float(np.mean(np.abs(segment)))
    shape_factor = float(rms / (mean_abs + eps))
    return [rms, peak, kurt, skewness, crest_factor, shape_factor]

# ── BỘ ĐÔI DEMODULATION ──────────────────────────────────────────
def get_hilbert_envelope(segment: np.ndarray) -> np.ndarray:
    """Tách đường bao bằng biến đổi Hilbert."""
    analytic_signal = np.asarray(hilbert(segment))
    return np.abs(analytic_signal)

def get_square_law_envelope(segment: np.ndarray, fs: int = config.FS) -> np.ndarray:
    """Tách đường bao bằng Square-Law Demodulation (Dùng lọc SOS ổn định số học)."""
    nyq = 0.5 * fs
    
    # 1. Bandpass IIR Butterworth (2kHz - 5kHz) dùng Second-Order Sections (SOS)
    low = config.DEMOD_BANDPASS_LOW / nyq
    high = config.DEMOD_BANDPASS_HIGH / nyq
    sos_bp = cast(Any, butter(2, [low, high], btype='bandpass', output='sos'))
    filtered = sosfiltfilt(sos_bp, segment)
    
    # 2. Rectification (Bình phương phi tuyến)
    rectified = filtered ** 2
    
    # 3. Lowpass IIR Butterworth (< 500Hz) dùng SOS
    cutoff = config.DEMOD_LOWPASS_CUTOFF / nyq
    sos_lp = cast(Any, butter(2, cutoff, btype='lowpass', output='sos'))
    envelope = sosfiltfilt(sos_lp, rectified)
    return envelope

# ── TRÍCH XUẤT ĐẶC TRƯNG MIỀN TẦN SỐ ─────────────────────────────
def extract_freq_features(segment: np.ndarray, fs: int, window_size: int,
                          freq_band_edges: Sequence[float],
                          fault_freqs: Dict[str, float],
                          demod_type: str = 'none') -> List[float]:
    window = np.hanning(window_size)
    fft_mag = np.abs(np.fft.rfft(segment * window)) * 2 / window_size
    freqs = np.fft.rfftfreq(window_size, d=1 / fs)

    freq_feats: List[float] = []
    for i in range(len(freq_band_edges) - 1):
        f_lo, f_hi = freq_band_edges[i], freq_band_edges[i + 1]
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energy = float(np.sum(fft_mag[mask] ** 2)) if mask.any() else 0.0
        freq_feats.append(band_energy)

    def get_amp_at_freq(mag_array: np.ndarray, target_f: float, tol: float = 2.0) -> float:
        idx = np.where((freqs >= target_f - tol) & (freqs <= target_f + tol))[0]
        return float(np.max(mag_array[idx])) if len(idx) > 0 else 0.0

    freq_feats.append(get_amp_at_freq(fft_mag, fault_freqs['BPFI']))
    freq_feats.append(get_amp_at_freq(fft_mag, fault_freqs['BPFO']))
    freq_feats.append(get_amp_at_freq(fft_mag, fault_freqs['BSF']))

    if demod_type == 'hilbert':
        env = get_hilbert_envelope(segment)
    elif demod_type == 'square_law':
        env = get_square_law_envelope(segment, fs)
    else:
        env = None

    if env is not None:
        env_mag = np.abs(np.fft.rfft((env - np.mean(env)) * window)) * 2 / window_size
        freq_feats.append(get_amp_at_freq(env_mag, fault_freqs['BPFI']))
        freq_feats.append(get_amp_at_freq(env_mag, fault_freqs['BPFO']))
        freq_feats.append(get_amp_at_freq(env_mag, fault_freqs['BSF']))
    else:
        freq_feats.extend([0.0, 0.0, 0.0])

    return freq_feats

def get_freq_feature_names(freq_band_edges: Sequence[float]) -> List[str]:
    names = [f"Energy_{freq_band_edges[i]}_{freq_band_edges[i+1]}Hz" for i in range(len(freq_band_edges) - 1)]
    names += ['Amp_BPFI', 'Amp_BPFO', 'Amp_BSF']
    names += ['Env_Amp_BPFI', 'Env_Amp_BPFO', 'Env_Amp_BSF']
    return names
