# -*- coding: utf-8 -*-
"""
common/mcu_export.py
======================
[Giai đoạn 3 — mục 3.1-3.2] Chuẩn bị model để đo tài nguyên thật trên
ARM Cortex-M qua STM32Cube.AI / ST Edge AI Developer Cloud.

MINH BẠCH VỀ MỨC ĐỘ CHẮC CHẮN:
  - `export_tflite_for_mcu()`, `save_signal_sample_csv()`: 100% chạy được,
    đã test — chỉ là thao tác file cục bộ, không phụ thuộc dịch vụ ngoài.
  - `parse_stedgeai_cli_summary()`: parser cho ĐÚNG định dạng dòng tóm tắt
    mà CLI `stedgeai` in ra (xem docstring — có dẫn nguồn), đã test bằng
    dữ liệu mẫu tự tạo khớp định dạng đó.
  - Phần "quy trình thao tác" ở cuối file: mô tả các bước dựa trên tài
    liệu công khai của ST tại thời điểm viết (đầu 2026) — công cụ này có
    thể đổi giao diện/API theo thời gian, TỰ XÁC NHẬN LẠI trên
    https://stedgeai-dc.st.com/ hoặc `stedgeai --help` trước khi phụ thuộc
    vào chi tiết cụ thể (tên tham số CLI, endpoint REST API...).
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
# Định dạng đã xác nhận qua tài liệu công khai của ST (command_line_interface,
# stedgeai-dc.st.com), ví dụ 1 dòng tóm tắt thật:
#   "model/c-model: macc=369,672/369,688 +16(+0.0%) weights=18,288/18,288 "
#   "activations=--/6,032 io=--/2,111"
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
            "Không parse được summary_text theo định dạng mong đợi. "
            "Dán nguyên văn dòng '...: macc=... weights=... activations=... io=...' "
            "mà lệnh `stedgeai validate`/`analyze` in ra. Nếu ST đã đổi định dạng, "
            "chỉnh lại _SUMMARY_PATTERN cho khớp."
        )
    macc1, macc2, w1, w2, a1, a2, io1, io2 = match.groups()
    return {
        "macc": _parse_int(macc2) or _parse_int(macc1),
        "flash_weights_bytes": _parse_int(w2) or _parse_int(w1),
        "ram_activations_bytes": _parse_int(a2) or _parse_int(a1),
        "io_bytes": _parse_int(io2) or _parse_int(io1),
    }


# ============================================================================
# HƯỚNG DẪN THAO TÁC — đọc kỹ trước khi dùng, xác nhận lại với tài liệu ST
# ============================================================================
MANUAL_WORKFLOW_NOTES = """
Có 3 cách lấy số đo Flash/RAM/MACC thật cho Bảng 3a/3b (mục 4.1), xếp theo
độ dễ dùng (không phải độ chính xác — cả 3 cho cùng 1 nguồn số liệu):

1) WEB UI (dễ nhất, không cài gì) — https://stedgeai-dc.st.com/
   - Đăng nhập bằng tài khoản myST (bạn đã có quyền truy cập).
   - Kéo-thả file .tflite xuất từ export_tflite_for_mcu().
   - Chọn STM32 target (F4/G4/H7...), chạy "Analyze" (static analysis —
     bắt buộc theo mục 3.2) để lấy Flash/RAM/MACC.
   - Tùy chọn: chạy "Benchmark" (board farm) để lấy latency thật trên
     board vật lý — thực nghiệm MỞ RỘNG, không bắt buộc (mục 0.4).
   - Xuất báo cáo (nút export trên UI) — dán dòng tóm tắt vào
     parse_stedgeai_cli_summary() ở trên, hoặc đọc trực tiếp trên UI.

2) CLI `stedgeai` (nếu cài STM32Cube.AI / ST Edge AI Core cục bộ) — cú
   pháp tham khảo từ tài liệu ST (XÁC NHẬN LẠI bằng `stedgeai --help` vì
   có thể đổi theo phiên bản):
       stedgeai validate -m <model.tflite> --target stm32
       stedgeai validate -m <model.tflite> --target stm32h7 --desc serial  # đo trên board thật
   CLI in ra dòng tóm tắt dạng "macc=... weights=... activations=... io=...".
   Copy dòng đó, đưa vào parse_stedgeai_cli_summary(dong_do_copy).

3) REST API (để tự động hóa toàn bộ Giai đoạn 3 nếu chạy nhiều model) —
   ST Edge AI Developer Cloud có expose REST API cho việc này, nhưng
   file này KHÔNG hardcode endpoint/schema cụ thể vì mình không xác nhận
   được chi tiết auth/endpoint hiện tại một cách đáng tin cậy. Tra cứu
   tài liệu API chính thức trên chính trang stedgeai-dc.st.com (thường có
   mục "API"/"REST API" sau khi đăng nhập) trước khi viết code gọi API.

Dù dùng cách nào, LƯU LẠI dòng tóm tắt gốc (raw text) vào file .txt cùng
thư mục với .tflite tương ứng — vừa làm bằng chứng thực nghiệm cho báo
cáo (mục 4.3: minh bạch hóa phạm vi nghiên cứu), vừa để parse lại bằng
parse_stedgeai_cli_summary() bất cứ lúc nào không cần chạy lại tool.
"""


def print_manual_workflow_notes():
    print(MANUAL_WORKFLOW_NOTES)
