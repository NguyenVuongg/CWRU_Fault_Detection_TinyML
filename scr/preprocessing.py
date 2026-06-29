"""
preprocessing.py
────────────────
Chuẩn hóa tín hiệu, cân bằng lớp và chia tập dữ liệu CWRU.

Hai chiến lược split:
  A. Random split     — baseline nhanh (80/10/10)
  B. Cross-load split — train load 0-2, test load 3
                        (khắt khe hơn, phân biệt với nghiên cứu trước)
"""

from __future__ import annotations

import json
import numpy as np
from sklearn.model_selection import train_test_split

# ── Gọi từ dataset.py ─────────────────────────────────────────────
from scr.dataset import build_dataset, LABEL_NAMES

RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════
# 1. NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def normalize_per_sample(X: np.ndarray) -> np.ndarray:
    """
    Z-score normalize từng segment độc lập.

    Công thức: x' = (x - mean(x)) / (std(x) + ε)

    Ưu điểm: loại bỏ bias offset giữa các file thu âm khác nhau.
    Phổ biến nhất trong các bài báo CWRU với raw signal input.
    """
    mean = X.mean(axis=1, keepdims=True)          # (N, 1)
    std  = X.std(axis=1, keepdims=True) + 1e-8    # (N, 1)
    return ((X - mean) / std).astype(np.float32, copy=False)


def normalize_global(X: np.ndarray,
                     mean: float | None = None,
                     std:  float | None = None
                     ) -> tuple[np.ndarray, float, float]:
    """
    Z-score normalize toàn bộ dataset theo global mean/std.

    Dùng khi muốn giữ biên độ tương đối giữa các file.
    QUAN TRỌNG: chỉ fit mean/std trên tập train, rồi apply lên val/test.

    Returns:
        X_norm : normalized array
        mean   : dùng lại cho val/test
        std    : dùng lại cho val/test
    """
    if mean is None:
        mean = X.mean()
    if std is None:
        std = X.std() + 1e-8
    X_norm = ((X - mean) / std).astype(np.float32, copy=False)
    return X_norm, float(mean), float(std)


# ══════════════════════════════════════════════════════════════════
# 2. CLASS BALANCING
# ══════════════════════════════════════════════════════════════════

def undersample_to_balance(X: np.ndarray,
                           y: np.ndarray,
                           metadata: list[dict],
                           seed: int = RANDOM_SEED
                           ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Giảm mẫu các lớp đa số về bằng lớp thiểu số (Normal).

    Ưu điểm: không tạo dữ liệu giả, phổ biến trong benchmark CWRU.
    Nhược điểm: mất ~⅔ dữ liệu fault. Dùng khi ưu tiên đơn giản.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    idx_keep = []
    for cls in classes:
        idx_cls = np.where(y == cls)[0]
        idx_sampled = rng.choice(idx_cls, size=min_count, replace=False)
        idx_keep.append(idx_sampled)

    idx_keep = np.concatenate(idx_keep)
    idx_keep = rng.permutation(idx_keep)      # shuffle

    meta_balanced = [metadata[i] for i in idx_keep]
    return X[idx_keep], y[idx_keep], meta_balanced


def get_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Tính class weights để dùng trong loss function.

    Dùng khi KHÔNG muốn undersample — chỉ truyền weights vào
    model.fit(class_weight=...) hoặc loss tùy chỉnh.

    Công thức: w_i = N / (n_classes × count_i)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_total   = len(y)
    n_classes = len(classes)
    weights   = {int(cls): n_total / (n_classes * cnt)
                 for cls, cnt in zip(classes, counts)}
    return weights


# ══════════════════════════════════════════════════════════════════
# 3A. RANDOM SPLIT  (baseline)
# ══════════════════════════════════════════════════════════════════

def random_split(X: np.ndarray,
                 y: np.ndarray,
                 metadata: list[dict] | None = None,
                 train_ratio: float = 0.80,
                 val_ratio:   float = 0.10,
                 seed: int = RANDOM_SEED
                 ) -> tuple:
    """
    Chia ngẫu nhiên theo tỷ lệ 80 / 10 / 10 (train / val / test).

    Dùng stratify=y để đảm bảo tỷ lệ nhãn đều nhau ở mỗi tập.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train + val ratio phải < 1.0"
    if metadata is not None and len(metadata) != len(y):
        raise ValueError("metadata length must match y length")

    indices = np.arange(len(y))
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices,
        test_size=1 - train_ratio,
        stratify=y,
        random_state=seed,
    )
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=1 - val_size_adjusted,
        stratify=y_temp,
        random_state=seed,
    )
    split = (X_train, X_val, X_test, y_train, y_val, y_test)
    if metadata is None:
        return split

    return (
        *split,
        [metadata[i] for i in idx_train],
        [metadata[i] for i in idx_val],
        [metadata[i] for i in idx_test],
    )


# ══════════════════════════════════════════════════════════════════
# 3B. CROSS-LOAD SPLIT  (điểm khác biệt với nghiên cứu trước)
# ══════════════════════════════════════════════════════════════════

def cross_load_split(X: np.ndarray,
                     y: np.ndarray,
                     metadata: list[dict],
                     test_loads:  set[int] = {3},
                     val_loads:   set[int] = {2},
                     ) -> tuple:
    """
    Chia theo điều kiện tải:
        Train : load 0, 1          (điều kiện huấn luyện)
        Val   : load 2             (điều kiện kiểm tra trong quá trình train)
        Test  : load 3             (điều kiện chưa thấy bao giờ)

    Ý nghĩa: đánh giá khả năng tổng quát hóa sang điều kiện tải mới.
    Đây là kịch bản thực tế hơn random split và là điểm mạnh
    của đề tài so với nhiều nghiên cứu chỉ dùng random split.
    """
    loads = np.array([m["load"] for m in metadata])

    test_mask  = np.isin(loads, list(test_loads))
    val_mask   = np.isin(loads, list(val_loads))
    train_mask = ~test_mask & ~val_mask

    return (
        X[train_mask], X[val_mask],  X[test_mask],
        y[train_mask], y[val_mask],  y[test_mask],
        [m for m, keep in zip(metadata, train_mask) if keep],
        [m for m, keep in zip(metadata, val_mask) if keep],
        [m for m, keep in zip(metadata, test_mask) if keep],
    )


# ══════════════════════════════════════════════════════════════════
# 4. SAVE / LOAD .npz
# ══════════════════════════════════════════════════════════════════

def save_splits(path: str,
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                meta_train=None, meta_val=None, meta_test=None) -> None:
    """Lưu toàn bộ split vào 1 file .npz để tái sử dụng."""
    payload = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    if meta_train is not None:
        payload["metadata_train_json"] = json.dumps(meta_train)
        payload["metadata_val_json"] = json.dumps(meta_val)
        payload["metadata_test_json"] = json.dumps(meta_test)

    np.savez_compressed(path, **payload)
    print(f"Saved → {path}")


def load_splits(path: str) -> tuple:
    """Load lại split từ file .npz."""
    with np.load(path) as data:
        split = (data["X_train"], data["X_val"],  data["X_test"],
                 data["y_train"], data["y_val"],  data["y_test"])

        if "metadata_train_json" not in data.files:
            return split

        return (
            *split,
            json.loads(str(data["metadata_train_json"])),
            json.loads(str(data["metadata_val_json"])),
            json.loads(str(data["metadata_test_json"])),
        )


# ══════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ══════════════════════════════════════════════════════════════════

def print_split_summary(X_train, X_val, X_test,
                        y_train, y_val, y_test) -> None:
    total = len(y_train) + len(y_val) + len(y_test)
    for name, X, y in [("Train", X_train, y_train),
                        ("Val",   X_val,   y_val),
                        ("Test",  X_test,  y_test)]:
        classes, counts = np.unique(y, return_counts=True)
        dist = " | ".join(
            f"{LABEL_NAMES[c]}:{n}" for c, n in zip(classes, counts)
        )
        print(f"  {name:5s}  {X.shape}  [{dist}]  ({100*len(y)/total:.1f}%)")


# ══════════════════════════════════════════════════════════════════
# MAIN — chạy thử cả 2 chiến lược split
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Load dataset ──────────────────────────────────────────────
    print("Loading dataset...")
    X, y, metadata = build_dataset()

    # ── Normalize (per-sample z-score) ────────────────────────────
    X = normalize_per_sample(X)
    print(f"\nNormalized: mean≈{X.mean():.4f}  std≈{X.std():.4f}")

    # ── Cân bằng lớp ──────────────────────────────────────────────
    X_bal, y_bal, meta_bal = undersample_to_balance(X, y, metadata)
    classes, counts = np.unique(y_bal, return_counts=True)
    print(f"\nAfter balancing: {dict(zip(classes, counts))}")
    print(f"  (Removed {len(y) - len(y_bal)} segments)")

    # ── Class weights (alternative to undersampling) ───────────────
    weights = get_class_weights(y)
    print(f"\nClass weights (no undersampling): {weights}")

    # ── Strategy A: Random split ───────────────────────────────────
    print("\n── Strategy A: Random Split (80/10/10) ──────────────────")
    splits_A = random_split(X_bal, y_bal, meta_bal)
    print_split_summary(*splits_A[:6])
    save_splits("splits_random.npz", *splits_A)

    # ── Strategy B: Cross-load split ──────────────────────────────
    print("\n── Strategy B: Cross-Load Split ─────────────────────────")
    print("   Train: load 0,1 | Val: load 2 | Test: load 3")
    splits_B = cross_load_split(X, y, metadata)   # dùng X chưa undersample
    print_split_summary(*splits_B[:6])
    save_splits("splits_crossload.npz", *splits_B)
