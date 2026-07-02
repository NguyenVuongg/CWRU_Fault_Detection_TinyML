import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Dict, Sequence


def extract_time_features(segment: np.ndarray) -> List[float]:
    """Trích xuất 6 đặc trưng thống kê miền thời gian."""
    eps = 1e-10  # Tránh lỗi chia cho 0

    rms = np.sqrt(np.mean(segment**2))
    peak = np.max(np.abs(segment))
    kurt = kurtosis(segment, fisher=False)
    skewness = skew(segment)
    crest_factor = peak / (rms + eps)

    mean_abs = np.mean(np.abs(segment))
    shape_factor = rms / (mean_abs + eps)

    return [rms, peak, kurt, skewness, crest_factor, shape_factor]


def extract_freq_features(segment: np.ndarray, fs: int, window_size: int,
                           freq_band_edges: Sequence[float],
                           fault_freqs: Dict[str, float]) -> List[float]:
    """
    Trích xuất đặc trưng miền tần số.

    freq_band_edges: danh sách mốc tần số (Hz) để chia dải năng lượng,
                      ví dụ [0, 50, 100, 200, 400, 800, 1600, 3200, 6000].
                      Không dùng linear-binning đều nhau, vì fault
                      frequency (BPFI/BPFO/BSF) đều nằm dưới 200Hz —
                      linear-binning sẽ nén hết vùng đó vào 1 bin duy nhất,
                      không phân biệt được các loại lỗi.
    """
    # 1. Áp dụng Hanning Window
    window = np.hanning(window_size)
    windowed_segment = segment * window

    # 2. Tính RFFT (chỉ lấy nửa phổ dương)
    fft_mag = np.abs(np.fft.rfft(windowed_segment)) * 2 / window_size
    freqs = np.fft.rfftfreq(window_size, d=1 / fs)

    freq_feats = []

    # 3. Năng lượng theo dải tần số THEO MỐC HZ THỰC (không theo index)
    for i in range(len(freq_band_edges) - 1):
        f_lo, f_hi = freq_band_edges[i], freq_band_edges[i + 1]
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energy = np.sum(fft_mag[mask] ** 2) if mask.any() else 0.0
        freq_feats.append(band_energy)

    # 4. Biên độ tại các tần số lỗi (dung sai ±2.0 Hz — phù hợp với
    #    độ phân giải FFT 11.7 Hz/bin ở fs=12000, window=1024)
    def get_amp_at_freq(target_f: float, tol: float = 2.0) -> float:
        idx = np.where((freqs >= target_f - tol) & (freqs <= target_f + tol))[0]
        return np.max(fft_mag[idx]) if len(idx) > 0 else 0.0

    freq_feats.append(get_amp_at_freq(fault_freqs['BPFI']))
    freq_feats.append(get_amp_at_freq(fault_freqs['BPFO']))
    freq_feats.append(get_amp_at_freq(fault_freqs['BSF']))  # Bật lại — cần thiết cho Ball fault

    return freq_feats


def get_freq_feature_names(freq_band_edges: Sequence[float]) -> List[str]:
    """Sinh tên cột tương ứng với extract_freq_features, để dataset.py
    không phải hard-code lại số lượng band."""
    names = []
    for i in range(len(freq_band_edges) - 1):
        names.append(f"Energy_{freq_band_edges[i]}_{freq_band_edges[i+1]}Hz")
    names += ['Amp_BPFI', 'Amp_BPFO', 'Amp_BSF']
    return names
