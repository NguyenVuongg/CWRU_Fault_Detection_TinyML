# -*- coding: utf-8 -*-
"""
common/mcu_export.py
======================
[Giai đoạn 3 — mục 3.1-3.2] Chuẩn bị model để đo tài nguyên thật trên
ARM Cortex-M qua STM32Cube.AI.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

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
# mục 3.2 — NĂNG LƯỢNG TIÊU THỤ MỖI LẦN SUY LUẬN: E = P x t
# ============================================================================
def estimate_energy_mj(latency_ms: float, power_mw: float) -> dict:
    """Tính năng lượng mỗi lần suy luận theo đúng công thức mục 3.2: E = P x t.

    latency_ms: độ trễ ĐO THẬT trên STM32F4 bằng bộ đếm chu kỳ DWT
                (DWT->CYCCNT), KHÔNG phải số ước lượng từ MACC.
    power_mw:   công suất trung bình của MCU trong lúc suy luận (mW). Lấy
                từ datasheet theo đúng điểm hoạt động (điện áp, tần số
                clock, trạng thái FPU/cache) hoặc đo bằng đồng hồ hiện số /
                mạch shunt.

    Trả về cả mJ và uJ vì với mô hình cỡ này năng lượng mỗi lần suy luận
    thường rơi vào khoảng chục-trăm uJ, ghi bằng mJ sẽ ra 0.0xx khó đọc.

    LƯU Ý KHI SO SÁNH: chỉ được so E giữa các model đo ở CÙNG một điểm
    hoạt động (cùng board, cùng điện áp, cùng tần số clock). Đổi clock vừa
    đổi t vừa đổi P nên hai số E ở hai điểm clock khác nhau không so
    được với nhau.
    """
    latency_ms = float(latency_ms)
    power_mw = float(power_mw)
    if latency_ms < 0 or power_mw < 0:
        raise ValueError(
            f"latency_ms và power_mw phải >= 0, nhận được "
            f"latency_ms={latency_ms}, power_mw={power_mw}."
        )

    # mW x ms = uJ  (1e-3 W x 1e-3 s = 1e-6 J)
    energy_uj = power_mw * latency_ms
    return {
        "latency_ms": latency_ms,
        "power_mw": power_mw,
        "energy_uj": float(energy_uj),
        "energy_mj": float(energy_uj / 1000.0),
        "inferences_per_second": float(1000.0 / latency_ms) if latency_ms > 0 else float("inf"),
    }


def latency_from_dwt_cycles(cycles: int, clock_mhz: float) -> float:
    """Đổi số chu kỳ đọc từ DWT->CYCCNT sang milligiây (mục 3.2).

    Tách riêng thành hàm để hệ số quy đổi chỉ viết ở một chỗ: ghi sai
    clock_mhz là cách nhanh nhất để toàn bộ bảng latency/energy sai theo
    cùng một hệ số mà vẫn trông hợp lý.
    """
    cycles = int(cycles)
    clock_mhz = float(clock_mhz)
    if clock_mhz <= 0:
        raise ValueError(f"clock_mhz phải > 0, nhận được {clock_mhz}.")
    return cycles / (clock_mhz * 1e3)   # cycles / (MHz * 1e6) * 1e3 ms


def summarize_mcu_cost(model_tag: str,
                       stedgeai_summary_text: Optional[str] = None,
                       latency_ms: Optional[float] = None,
                       power_mw: Optional[float] = None,
                       tflite_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """Gói 1 dòng của bảng chi phí triển khai ở Giai đoạn 3.

    Ghép chỉ số TĨNH (mục 3.1 — Flash/SRAM/MACC từ STM32Cube.AI) với chỉ số
    ĐỘNG (mục 3.2 — Latency đo bằng DWT, Energy = P x t) vào cùng một dict
    để ghi trực tiếp thành DataFrame. Mọi tham số đo đều tùy chọn — đo
    được tới đâu điền tới đó, phần chưa đo để None thay vì điền số ước
    lượng trông như số đo thật.

    ---------------------------------------------------------------------
    ĐÃ SỬA 5 lỗi Pylance reportArgumentType
    ---------------------------------------------------------------------
    1-4) Bốn tham số khai kiểu "x: str = None" / "float = None" /
         "bytes = None". Chạy thì vẫn chạy, nhưng kiểu KHAI BÁO không chứa
         None, nên: (a) báo lỗi ngay tại chỗ khai; (b) tệ hơn, trong thân
         hàm trình kiểm tra tưởng biến CHẮC CHẮN có giá trị nên KHÔNG còn
         cảnh báo khi ta quên kiểm tra None. Đúng phải là Optional[...].
    5)   row = {"model_tag": model_tag} khiến row bị suy ra là dict[str, str]
         (vì model_tag: str), nên mọi dòng row[...] = <số> phía dưới đều là
         gán sai kiểu. Panel chỉ báo dòng row["tflite_size_bytes"] = len(...)
         (int), nhưng row["latency_ms"] = float(...) và row["energy_uj"] =
         None cũng cùng bản chất. Bảng chi phí này CỐ Ý trộn
         str + int + float + None, nên cách đúng là khai tường minh
         Dict[str, Any] (và đổi kiểu trả về dict -> Dict[str, Any] cho
         khớp), không phải cast từng dòng.
    """
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
        print(f"[summarize_mcu_cost] {model_tag}: thiếu power_mw nên chưa tính "
              f"được Energy (mục 3.2 yêu cầu E = P x t).")

    return row