from pathlib import Path
import re
import warnings

import numpy as np
import scipy.io


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/CWRU-dataset"
SEG_LEN = 1024
STEP = 512
LOADS = {0, 1, 2, 3}
SOURCE_DIRS = {"48k_Drive_End_Bearing_Fault_Data", "Normal"}

LABELS = {
    "Normal": 0,
    "B": 1,
    "IR": 2,
    "OR": 3,
}
LABEL_NAMES = {label: name for name, label in LABELS.items()}

# Same scope as the original FILE_MAP:
# normal, IR/B faults with 007/014/021 inch sizes, and OR faults at @6.
FAULT_SIZES = {"007", "014", "021"}
OR_POSITIONS = {"@6"}


def get_sampling_rate(source_dir):
    if source_dir.startswith("48k"):
        return 48000
    if source_dir.startswith("12k"):
        return 12000
    return None


def load_mat_data(filepath):
    """Read drive-end signal plus small file-level metadata from a .mat file."""
    mat = scipy.io.loadmat(filepath)
    de_key = [key for key in mat.keys() if "DE_time" in key]
    if not de_key:
        return None, None

    rpm_key = next((key for key in mat.keys() if key.endswith("RPM")), None)
    rpm = None
    if rpm_key is not None:
        rpm = int(np.asarray(mat[rpm_key]).ravel()[0])

    signal = mat[de_key[0]].ravel().astype(np.float32, copy=False)
    return signal, rpm


def segment_signal(signal, seg_len=SEG_LEN, step=STEP):
    """Split a signal into overlapping windows."""
    segments, windows = [], []
    for start in range(0, len(signal) - seg_len + 1, step):
        segments.append(signal[start : start + seg_len])
        windows.append((start, start + seg_len))
    return np.asarray(segments, dtype=np.float32), windows


def get_load(filepath):
    match = re.search(r"_(\d+)\.mat$", filepath.name)
    if not match:
        return None
    return int(match.group(1))


def infer_label(filepath):
    parts = filepath.relative_to(DATA_DIR).parts

    if "Normal" in parts:
        return LABELS["Normal"]

    size = next((part for part in parts if part in FAULT_SIZES), None)
    if size is None:
        return None

    if "IR" in parts:
        return LABELS["IR"]
    if "B" in parts:
        return LABELS["B"]
    if "OR" in parts:
        has_or_position = any(part in OR_POSITIONS for part in parts) or any(
            position in filepath.stem for position in OR_POSITIONS
        )
        if has_or_position:
            return LABELS["OR"]

    return None


def infer_file_metadata(filepath, label):
    parts = filepath.relative_to(DATA_DIR).parts
    source_dir = parts[0]
    fault_type = LABEL_NAMES[label]
    fault_size = next((part for part in parts if part in FAULT_SIZES), None)
    or_position = next((part for part in parts if part in OR_POSITIONS), None)

    if or_position is None and fault_type == "OR":
        or_position = next(
            (position for position in OR_POSITIONS if position in filepath.stem),
            None,
        )

    return {
        "relative_path": str(filepath.relative_to(DATA_DIR)),
        "source_dir": source_dir,
        "sampling_rate_hz": get_sampling_rate(source_dir),
        "load": get_load(filepath),
        "label": label,
        "label_name": LABEL_NAMES[label],
        "fault_type": fault_type,
        "fault_size": fault_size,
        "or_position": or_position,
    }

def iter_dataset_files():
    for filepath in sorted(DATA_DIR.rglob("*.mat")):
        parts = filepath.relative_to(DATA_DIR).parts
        if not parts or parts[0] not in SOURCE_DIRS:
            continue

        load = get_load(filepath)
        if load not in LOADS:
            continue

        label = infer_label(filepath)
        if label is None:
            if "OR" in parts:
                warnings.warn(
                    "Skipping OR file outside allowed positions "
                    f"{sorted(OR_POSITIONS)}: {filepath.relative_to(DATA_DIR)}",
                    stacklevel=2,
                )
            continue

        yield filepath, label


def build_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    x_all, y_all, metadata = [], [], []

    for filepath, label in iter_dataset_files():
        signal, rpm = load_mat_data(filepath)
        if signal is None:
            print(f"Skipping {filepath.relative_to(DATA_DIR)}: no DE_time signal")
            continue

        segments, windows = segment_signal(signal)
        if len(segments) == 0:
            print(f"Skipping {filepath.relative_to(DATA_DIR)}: signal too short")
            continue

        labels = np.full(len(segments), label, dtype=np.int64)
        file_metadata = infer_file_metadata(filepath, label)
        file_metadata["rpm"] = rpm

        for segment_index, (start, end) in enumerate(windows):
            metadata.append(
                {
                    **file_metadata,
                    "segment_index": segment_index,
                    "start": start,
                    "end": end,
                }
            )

        x_all.append(segments)
        y_all.append(labels)
        print(
            f"{filepath.relative_to(DATA_DIR)}: "
            f"{len(segments)} segments, label={label}"
        )

    if not x_all:
        raise RuntimeError(
            f"No matching .mat files were loaded from {DATA_DIR}. "
            "Check DATA_DIR, LOADS, FAULT_SIZES, and OR_POSITIONS."
        )

    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return x, y, metadata


def main():
    x, y, metadata = build_dataset()
    print(f"\nTotal: X={x.shape}, y={y.shape}")
    print(f"Dtypes: X={x.dtype}, y={y.dtype}")
    print(f"Label distribution: {np.unique(y, return_counts=True)}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Metadata fields: {tuple(metadata[0].keys())}")


if __name__ == "__main__":
    main()
