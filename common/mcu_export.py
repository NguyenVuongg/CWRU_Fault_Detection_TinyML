# -*- coding: utf-8 -*-
"""
common/mcu_export.py
======================
Chuẩn bị model INT8 để đo tài nguyên thật trên ARM Cortex-M qua STM32Cube.AI.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

def export_tflite_for_mcu(tflite_bytes: bytes, output_path, model_tag: str = "model") -> Path:
    """Lưu bytes .tflite ra đĩa với tên rõ ràng — sẵn sàng để nạp vào STM32Cube.AI."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"Đã lưu [{model_tag}]: {output_path.resolve()} ({len(tflite_bytes)} bytes)")
    return output_path

def save_signal_sample_csv(X: np.ndarray, output_path, n_samples: int = 20, seed: int = 42) -> Path:
    """Lưu vài mẫu tín hiệu/đặc trưng THẬT ra CSV làm input test trên board STM32."""
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
    """Trích xuất macc/weights(Flash)/activations(RAM)/io từ STM32Cube.AI CLI."""
    match = _SUMMARY_PATTERN.search(summary_text.replace("\n", " "))
    if not match:
        raise ValueError(
            f"Không parse được dòng tóm tắt stedgeai — kiểm tra lại format output:\n{summary_text[:300]}"
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

# ============================================================================
# Tính Năng lượng: E = P x t
# ============================================================================
def estimate_energy_mj(latency_ms: float, power_mw: float) -> dict:
    """Tính năng lượng mỗi lần suy luận theo công thức E = P x t."""
    latency_ms = float(latency_ms)
    power_mw = float(power_mw)
    if latency_ms < 0 or power_mw < 0:
        raise ValueError(f"latency_ms={latency_ms}, power_mw={power_mw} phải >= 0")

    energy_uj = power_mw * latency_ms
    return {
        "latency_ms": latency_ms,
        "power_mw": power_mw,
        "energy_uj": float(energy_uj),
        "energy_mj": float(energy_uj / 1000.0),
        "inferences_per_second": float(1000.0 / latency_ms) if latency_ms > 0 else float("inf"),
    }

def latency_from_dwt_cycles(cycles: int, clock_mhz: float) -> float:
    """Đổi số chu kỳ đọc từ STM32 DWT->CYCCNT sang milligiây."""
    cycles = int(cycles)
    clock_mhz = float(clock_mhz)
    if clock_mhz <= 0:
        raise ValueError(f"clock_mhz phải > 0, nhận được {clock_mhz}.")
    return cycles / (clock_mhz * 1e3)

def summarize_mcu_cost(model_tag: str,
                       stedgeai_summary_text: Optional[str] = None,
                       latency_ms: Optional[float] = None,
                       power_mw: Optional[float] = None,
                       tflite_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """Gói toàn bộ chỉ số tĩnh (Flash/RAM/MACC) và động (Latency/Energy) thành một dict."""
    row: Dict[str, Any] = {"model_tag": model_tag}

    if tflite_bytes is not None:
        row["tflite_size_bytes"] = len(tflite_bytes)

    if stedgeai_summary_text:
        row.update(parse_stedgeai_cli_summary(stedgeai_summary_text))

    if latency_ms is not None and power_mw is not None:
        row.update(estimate_energy_mj(latency_ms, power_mw))
    elif latency_ms is not None:
        row["latency_ms"] = float(latency_ms)
        row["energy_uj"] = None
        print(f"[summarize_mcu_cost] {model_tag}: thiếu power_mw nên chưa tính được Energy.")

    return row