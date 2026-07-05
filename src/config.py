# ============================================================
# CONFIG — Motor Fault Detection (CWRU, 12kHz)
# ============================================================

# 1. Cấu hình Xử lý tín hiệu (DSP)
FS = 12000               # Tần số lấy mẫu (Hz) — CWRU 12kHz Drive End
WINDOW_SIZE = 1024        # Kích thước cửa sổ cắt segment (N). Fs/N = 12000/1024 ~ 11.7 Hz/bin
OVERLAP = 0.5             # Tỉ lệ chồng lấp (50%)

# Dải lọc Bandpass cho Square-Law Demodulation (Vùng cộng hưởng cơ học CWRU)
DEMOD_BANDPASS_LOW = 2000.0   # Hz
DEMOD_BANDPASS_HIGH = 5000.0  # Hz
DEMOD_LOWPASS_CUTOFF = 500.0  # Hz (Đủ lấy đường bao chứa BPFI/BPFO < 200Hz)

# Dải năng lượng miền tần số — LOG-SPACED (không phải linear)
FREQ_BAND_EDGES = [0.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6000.0]  # Hz, 8 dải

# 2. Thông số hình học vòng bi SKF 6205 (Drive End)
# Các hệ số này nhân với tần số quay của trục (fr) sẽ ra tần số lỗi
BEARING_GEOMETRY = {
    'BPFI': 5.4152,  # Lỗi vòng trong (Inner Race)
    'BPFO': 3.5848,  # Lỗi vòng ngoài (Outer Race)
    'BSF': 2.357     # Lỗi con lăn (Ball)
}

# 3. Thông số vận hành (Ánh xạ Tải HP sang Tốc độ quay RPM)
LOAD_TO_RPM = {
    0: 1797,
    1: 1772,
    2: 1750,
    3: 1730
}

# 4. Cấu hình chia Train/Val/Test — CHIA THEO FILE, không theo thời gian trong file
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# 5. Cấu hình Thư mục
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "figures"

# 6. Nhãn
LABEL_NAMES = {0: "Normal", 1: "Inner_Race", 2: "Outer_Race", 3: "Ball"}
