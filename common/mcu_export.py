# -*- coding: utf-8 -*-
"""
common/mcu_export.py
======================
[Giai đoạn 3 — mục 3.1-3.2] Chuẩn bị model để đo tài nguyên thật trên
ARM Cortex-M qua STM32Cube.AI.
"""

import re
from pathlib import Path

import numpy as np


def export_tflite_for_mcu(tflite_bytes: bytes, output_path, model_tag: str = "model") -> Path:
    """Lưu bytes .tflite ra đĩa với tên rõ ràng — sẵn sàng để kéo-thả vào
    STM32Cube.AI Studio hoặc upload lên ST Edge AI Developer Cloud."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"Đã lưu [{model_tag}]: {output_path.resolve()} ({len(tflite_bytes)} bytes)")
    return output_path


def save_signal_sample_csv(X: np.ndarray, output_path, n_samples: int = 20, seed: int = 42) -> Path:
    """
    Lưu vài mẫu tín hiệu/đặc trưng THẬT ra CSV — dùng làm input mẫu khi
    validate model trên STM32Cube.AI/ST Edge AI Cloud (tùy chọn `-vi
    test_data.csv` của CLI `stedgeai validate`, xem cuối file), thay vì
    chỉ validate bằng dữ liệu ngẫu nhiên mặc định của công cụ.
    """
    import pandas as pd
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    pd.DataFrame(X[idx]).to_csv(output_path, index=False, header=False)
    print(f"Đã lưu {n} mẫu vào: {output_path.resolve()}")
    return output_path


# ============================================================================
# Parser cho dòng tóm tắt của CLI `stedgeai`
# ============================================================================
_SUMMARY_PATTERN = re.compile(
    r"macc=([\d,]+)(?:/([\d,]+))?.*?"
    r"weights=([\d,]+)(?:/([\d,]+))?.*?"
    r"activations=(?:--|([\d,]+))(?:/([\d,]+))?.*?"
    r"io=(?:--|([\d,]+))(?:/([\d,]+))?",
    re.IGNORECASE,
)


def _parse_int(s):
    return int(s.replace(",", "")) if s else None


def parse_stedgeai_cli_summary(summary_text: str) -> dict:
    """
    Trích macc/weights(Flash)/activations(RAM)/io từ dòng tóm tắt CLI
    `stedgeai`. Nhận CẢ 2 dạng: chỉ có 1 số ("macc=369672") hoặc dạng
    "model/c-model" có 2 số phân cách bởi "/" (số sau optimize).

    Trả về dict — lấy số THỨ HAI (sau optimize) nếu có, vì đó là số dùng
    triển khai thật; nếu chỉ có 1 số thì dùng số đó.
    """
    match = _SUMMARY_PATTERN.search(summary_text.replace("\n", " "))
    if not match:
        raise ValueError(
        )
    macc1, macc2, w1, w2, a1, a2, io1, io2 = match.groups()

    def _pick(second, first):
        second_val = _parse_int(second)
        return second_val if second_val is not None else _parse_int(first)

    return {
        "macc": _pick(macc2, macc1),
        "flash_weights_bytes": _pick(w2, w1),
        "ram_activations_bytes": _pick(a2, a1),
        "io_bytes": _pick(io2, io1),
    }