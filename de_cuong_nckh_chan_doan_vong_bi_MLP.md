# Hệ thống Chẩn đoán Lỗi Vòng bi Động cơ bằng TinyML: Tối ưu hóa Tiền xử lý Lai ghép và So sánh MLP vs 1D CNN trên Vi điều khiển Tài nguyên Thấp

**Dataset:** CWRU Drive-End 12kHz

---

## TỔNG QUAN ĐỀ TÀI

**Bối cảnh & khoảng trống nghiên cứu:**

- CWRU là benchmark phổ biến nhất cho chẩn đoán lỗi vòng bi. Tuy nhiên, phần lớn công trình công bố độ chính xác cao (>99%) dựa trên cách chia dữ liệu bị rò rỉ (random window split trên tín hiệu liên tục).
- Đồng thời, giải điều chế bằng biến đổi Hilbert theo lối FFT (ví dụ `scipy.signal.hilbert`) đòi hỏi đệm theo khối và bộ nhớ số phức, không phù hợp với xử lý theo dòng mẫu trên MCU giá rẻ như STM32; bản nhân quả khả thi trên MCU là FIR Hilbert (Type III).
- Các phương pháp thay thế nhẹ hơn như Square-Law lại vướng nhược điểm trễ pha (group delay) do bộ lọc thông thấp và suy giảm biên độ xung, đồng thời đường bao mang đơn vị bình phương nên dải động rộng — bất lợi khi lượng tử hóa INT8.
- Hiện tại, rất ít công trình đo đạc chi phí của toàn bộ đường ống (tiền xử lý DSP + trích xuất đặc trưng + suy luận AI) trên phần cứng thực tế để giải quyết bài toán đánh đổi này.

**Giá trị cốt lõi của đề tài:**

Xây dựng một pipeline hoàn chỉnh với hai đóng góp:

1. **Kiến trúc tiền xử lý DSP lai ghép (Hybrid Pre-processing):** Đề xuất khối tạo tín hiệu trực giao nhân quả, không FFT, kết hợp xấp xỉ Alpha-Max Plus Beta-Min để loại bỏ hoàn toàn hàm căn bậc hai (sqrt). Khối trực giao được triển khai theo hai bậc và báo cáo trung thực cả hai:
   - (a) Bộ vi phân trung tâm chuẩn hóa Q[n] = (x[n] − x[n−2]) / (2·sin ωc) kèm nhánh trễ bù I[n] = x[n−1] — chỉ 1 phép trừ + 1 phép nhân mỗi mẫu, lệch pha đúng 90° tại mọi tần số nhưng biên độ nghiêng theo sinω / sinωc.
   - (b) Cặp lọc IIR All-Pass trực giao nhằm làm phẳng biên độ trong dải.
2. **Kiểm định và tối ưu tài nguyên TinyML:** Áp dụng giao thức kiểm định chống rò rỉ (File-based Split + LOLO) để đánh giá trung thực hai phương pháp tiếp cận (MLP đặc trưng thủ công vs 1D CNN) trên cùng nền tảng nhúng.

---

## MỤC TIÊU CỤ THỂ

1. Áp dụng quy trình kiểm định hai tầng: File-based Split (chống rò rỉ) + Leave-One-Load-Out (LOLO) để đo khả năng tổng quát hóa qua điều kiện tải.
2. Thiết kế và cấu trúc hóa hệ thống giải điều chế trực giao nhân quả (vi phân trung tâm chuẩn hóa là thực nghiệm chính; IIR All-Pass là mở rộng có điều kiện) kết hợp xấp xỉ Alpha-Max cho bài toán nhận diện tần số lỗi cơ khí.
3. Xây dựng và so sánh hai mô hình phân loại: MLP với 3 phương pháp DSP khác nhau và mở rộng mô hình 1DCNN sang hướng sử dụng đường bao envelope làm tín hiệu vào thay vì tín hiệu thô.
4. Đánh giá tác động của tín hiệu bao hình từ DSP Hybrid lên độ chính xác và tốc độ suy luận của mô hình AI.
5. Cung cấp các benchmark về Flash, SRAM, chu kỳ lệnh (MACC), độ trễ suy luận (Latency) và năng lượng tiêu thụ cho toàn bộ pipeline (tiền xử lý DSP + trích xuất đặc trưng của MLP + suy luận AI), với hai mức đo rõ ràng: phân tích tĩnh qua STM32Cube.AI và đo động trên board STM32F4 thật (nếu có thể).

---

## PHẠM VI & GIỚI HẠN

- Chỉ sử dụng tập dữ liệu Drive-End 12kHz (xử lý nội suy mẫu file Normal 48kHz về 12kHz).
- MLP sử dụng bộ đặc trưng cố định. CNN sử dụng kiến trúc đồng bộ giữa hai biến thể (raw/envelope) để cô lập biến số kích thước đầu vào.
- LOLO đo khả năng tổng quát hóa qua điều kiện tải vật lý hiện có, không qua vòng bi vật lý độc lập.
- Tài nguyên phần cứng được đo ở hai mức: (i) phân tích tĩnh qua STM32Cube.AI (Flash, SRAM, MACC); (ii) đo động trên board STM32F4 thật cho Latency và năng lượng. Kết quả DSP ở Giai đoạn 3.1 được benchmark bằng mã C chạy trên board; môi trường Python dùng để phát triển/đối chiếu thuật toán.
- Giới hạn đã biết của khối trực giao bậc (a): tỉ số biên độ |Q|/|I| nghiêng theo sin ω / sin ωc — mở rộng nếu muốn biên độ phẳng toàn dải thì phải dùng bậc (b).
- Đặc trưng miền Order yêu cầu biết RPM tại thời điểm suy luận (xem mục 1.3 về nguồn RPM); đây là giả định vận hành của hệ thống, không phải đại lượng hệ tự ước lượng.

---

## 1. TỐI ƯU HÓA KIẾN TRÚC GIẢI ĐIỀU CHẾ LAI GHÉP (Hybrid Demodulation Architecture)

- **Tạo tín hiệu trực giao (I/Q):** Dùng bộ vi phân trung tâm chuẩn hóa dạng nhân quả (Q[n] = (x[n] - x[n-2]) / (2·sin ωc), I[n] = x[n-1]) để có lệch pha đúng 90° với trễ nhóm chỉ 1 mẫu và không cần FFT, giữ hình thái xung va đập sắc nét hơn Square-Law; hướng nâng cấp có điều kiện là cặp lọc IIR All-Pass trực giao khi cần biên độ phẳng toàn dải.
- **Xấp xỉ biên độ:** Ứng dụng công thức Alpha-Max Plus Beta-Min (a·max(|I|,|Q|) + β·min(|I|,|Q|), hệ số α, β tối ưu theo sai số biên độ cực đại) để tính biên độ đường bao, bỏ qua phép toán căn bậc hai tốn kém chu kỳ ALU của vi điều khiển.

---

## 2. QUY TRÌNH KIỂM ĐỊNH HAI TẦNG, TÁCH BIỆT RÕ CHỨC NĂNG

- **File-based Splitting:** Phân chia Train/Val/Test tuyệt đối theo nhãn ID file trước khi cắt cửa sổ. Định nghĩa chốt: file_id = tên file .mat gốc (stem), TUYỆT ĐỐI không chứa chỉ số cửa sổ; chỉ số cửa sổ nằm ở hai cột riêng window_idx và start_idx. Hàm dựng bảng đặc trưng chặn lỗi ghép nhầm chỉ số cửa sổ vào file_id bằng kiểm tra trùng file_id giữa các tập.
- **Leave-One-Load-Out (LOLO):** Huấn luyện trên 3 mức tải, kiểm tra trên mức tải bị giấu để đánh giá độ vững (robustness) thực tế.

---

## 3. SO SÁNH HAI PHƯƠNG PHÁP TIẾP CẬN TRÊN CÙNG NỀN TẢNG NHÚNG

- **MLP với đặc trưng thủ công (32 chiều):** Kết hợp kiến thức miền động học vòng bi (Time + Order + Envelope), cực kỳ nhẹ tài nguyên, thích hợp cho MCU siêu nhỏ chạy pin.
- **1D CNN học end-to-end:** Khảo sát khả năng tự trích xuất đặc trưng của mạng nơ-ron sâu với hai biến thể đầu vào (raw 2048 mẫu, envelope 1024 mẫu).

---

## 4. PHÂN TÍCH ĐỊNH LƯỢNG TÀI NGUYÊN NHÚNG (Embedded Resource Profiling)

- Cung cấp bằng chứng thực nghiệm về sự đánh đổi giữa độ chính xác, không gian lưu trữ và thời gian phản hồi của toàn bộ pipeline — bao gồm cả khối trích xuất 32 đặc trưng của MLP (FFT, phổ Order, phổ Envelope) — trên lõi kiến trúc ARM Cortex-M.

---

## RESEARCH QUESTIONS (RQ) & HYPOTHESES (H)

### RQ1 — Tính trung thực của phương pháp kiểm định
**RQ1:** Mức độ chênh lệch độ chính xác giữa Random Window Split và File-based Split + LOLO là bao nhiêu trên tập dữ liệu CWRU?

### RQ2 — Hiệu năng của tiền xử lý DSP Lai ghép (Hybrid DSP)
**RQ2:** Hệ thống DSP lai ghép (trực giao nhân quả bậc (a) + Alpha-Max) tối ưu hóa chu kỳ lệnh (MACC) và SRAM như thế nào so với FIR Hilbert nhân quả và Square-Law trên vi xử lý nhúng, khi cả ba dùng chung một cấu hình lọc nhân quả?

### RQ3 — Tác động của DSP Lai ghép lên Mô hình AI
**RQ3:** Tín hiệu đường bao sinh ra từ DSP lai ghép tác động thế nào đến độ chính xác phân loại (F1-Score) và độ hội tụ của mô hình MLP và 1D CNN?

### RQ4 — So sánh hai biến thể 1D CNN
**RQ4:** 1D CNN_raw vs 1D CNN_envelope khác biệt ra sao về độ chính xác (LOLO test), cấu hình tài nguyên (Flash/RAM/MACC), thời gian suy luận (Inference Latency) và mức tiêu thụ năng lượng trên STM32F4?

### Bảng tổng hợp RQ ↔ Thực nghiệm ↔ Metric

| RQ  | Thực nghiệm (Experiment) | Metric đo lường |
|-----|---------------------------|-----------------|
| RQ1 | Random Window Split vs. File-based Split + LOLO | Accuracy, F1, ΔAccuracy |
| RQ2 | DSP: FIR Hilbert (nhân quả) vs. Square-Law vs. Hybrid trực giao bậc (a) | RAM (bytes), MACC/mẫu, Phase Error (độ) — N/A cho Square-Law, Độ phẳng biên độ trong dải, Latency |
| RQ3 | Đánh giá tác động của nguồn đầu vào DSP (Hybrid vs Square-Law) lên MLP và CNN | Accuracy, F1 (LOLO test), số epoch hội tụ, AUC learning curve |
| RQ4 | CNN1D (raw) vs. CNN1D (env) | Accuracy, F1, Flash (KB), RAM (KB), Inference Latency (ms), Energy (mJ) |

---

## GIAI ĐOẠN 1 — TIỀN XỬ LÝ DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG

### 1.1. Chuẩn bị và Đồng bộ hóa Dữ liệu

- Sử dụng dữ liệu CWRU Drive-End 12kHz.
- Xử lý Baseline: Áp dụng Polyphase Filter (resample_poly, có lọc chống xếp phổ) để hạ tần số lấy mẫu các file Normal từ 48kHz về 12kHz.
- Bộ nhãn phân loại (10 lớp): Normal + 3 vị trí lỗi (IR, OR, B) × 3 mức đường kính lỗi (7, 14, 21 mil). Phân tích phụ gộp 4 lớp (Normal/IR/OR/B) được báo cáo bổ sung để đối chiếu với các công trình dùng 4 lớp.
- Thiết lập Split: Phân lập Train/Val/Test tuyệt đối theo nhãn ID file trước khi cắt cửa sổ (ID file = stem của file .mat gốc, không chứa chỉ số cửa sổ).
- Cửa sổ trượt: chốt kích thước cửa sổ 2048 mẫu và bước trượt (stride) cố định, ghi lại trong bảng đặc trưng qua window_idx và start_idx.

### 1.2. Kỹ thuật Giải điều chế Lai ghép (Hybrid Demodulation)

- **Xác định dải cộng hưởng:** Khảo sát phổ tần số rồi chốt hằng số dải băng thông cho phần cứng.
- **Dịch pha trực giao (bậc a — thực nghiệm chính):** Tách tín hiệu thành In-phase (I) và Quadrature (Q) bằng vi phân trung tâm chuẩn hóa dạng nhân quả:
  - I[n] = x[n-1],
  - Q[n] = (x[n] - x[n-2]) / (2·sin ωc),
  - cho Q/I = j·sin ω / sin ωc.
  - Baseline đối chứng là FIR Hilbert Type III nhân quả: pha –90.00°, |H| ∈ [0.9985, 1.0013] (biên độ phẳng).
- **Bậc (b) — mở rộng có điều kiện:** cặp lọc IIR All-Pass trực giao để làm phẳng biên độ toàn dải; chỉ triển khai nếu bậc (a) cho thấy méo biên độ ảnh hưởng đáng kể đến F1 ở RQ3.
- **Xấp xỉ đường bao:** Tính toán đường bao tuyến tính thông qua bộ hệ số Alpha-Max Plus Beta-Min tối ưu (α, β chọn theo tiêu chí sai số biên độ cực đại).

### 1.3. Trích xuất đặc trưng và Chuẩn hóa

- **Order-domain Normalization:** Chuyển trục Hz sang trục bậc quay (Order) dựa trên RPM. Nguồn RPM: trong thí nghiệm, RPM lấy theo tải danh định của từng file CWRU (~1797/1772/1750/1730 RPM cho tải 0/1/2/3 HP) và được ghi rõ là dữ liệu đầu vào của hệ thống; khi triển khai thực tế, hệ thống giả định có tín hiệu tốc độ (encoder/cảm biến) hoặc cấu hình tốc độ vận hành cố định — hệ thống KHÔNG tự ước lượng RPM từ rung động.
- **Bộ đặc trưng MLP (32 chiều):** Tổng hợp 11 đặc trưng thời gian, 12 đặc trưng phổ Order, và 9 đặc trưng phổ Envelope.
- **Chuẩn hóa đặc trưng MLP:** Z-score (standardization) trên 32 đặc trưng, tham số (mean, std) chỉ fit trên tập Train của từng Fold rồi áp dụng cho Val/Test — tránh rò rỉ thống kê giữa các tập.
- **Dữ liệu chuỗi CNN:** Trích xuất mảng thô (2048 mẫu) và mảng đường bao (1024 mẫu), chuẩn hóa Z-score độc lập cho mỗi Fold (fit trên Train).
- **Giải thích lựa chọn envelope 1024 mẫu:** sau LPF 750 Hz, tín hiệu đường bao được decimate hệ số 2 (12 kHz → 6 kHz) về 1024 mẫu — giữ độ dự trữ lấy mẫu rộng so với băng thông 750 Hz nhằm bảo toàn độ phân giải thời gian của xung va đập (biên độ xung là đặc trưng quan trọng của lỗi vòng bi). Việc decimate mạnh hơn (hệ số 8–10) là hướng tối ưu bổ sung, được ghi nhận như công việc tương lai.

---

## GIAI ĐOẠN 2 — XÂY DỰNG MÔ HÌNH & KIỂM ĐỊNH LOLO

### 2.1. Cấu trúc Leave-One-Load-Out (LOLO Cross-Validation)

- Fold 1: Train/Val = Tải {1,2,3} | Test = Tải {0}
- Fold 2: Train/Val = Tải {0,2,3} | Test = Tải {1}
- Fold 3: Train/Val = Tải {0,1,3} | Test = Tải {2}
- Fold 4: Train/Val = Tải {0,1,2} | Test = Tải {3}

Trong từng Pool Train/Val, áp dụng Stratified File-based Split để phân chia 80/20.

### 2.2. Kiến trúc Mô hình Trí tuệ nhân tạo

- **MLP:** Input 32 chiều (đã chuẩn hóa Z-score theo Fold), thiết kế mạng nông tối giản, tập trung vào tốc độ thực thi.
- **1D CNN:** Chuẩn hóa các khối Conv, Pooling và BatchNorm giữa hai biến thể CNN (raw) và CNN (envelope) nhằm cô lập biến số kích thước, đảm bảo sự so sánh công bằng về mặt kiến trúc.

### 2.3. Lượng tử hóa mô hình (Model Quantization)

- Chuyển đổi toàn bộ mô hình (MLP, CNN) từ Float32 sang định dạng INT8 thông qua phương pháp Post-training Quantization (TensorFlow Lite).
- Dữ liệu hiệu chuẩn (calibration set): chỉ lấy từ tập Train của từng Fold, tránh rò rỉ gián tiếp từ Val/Test vào tham số lượng tử hóa.
- Đánh giá suy hao: Đo lường mức độ suy giảm độ chính xác (Accuracy Drop) trước và sau lượng tử hóa.
- Điểm cần chú trọng đặc thù: đầu vào của CNN envelope là tín hiệu đường bao có biên độ nhỏ (bản chất của thành phần điều chế sau khi gỡ sóng mang), nhạy cảm với sai số lượng tử hóa INT8; cần khảo sát per-channel scaling cho nhánh này. MLP không chịu rủi ro này trực tiếp vì đầu vào là 32 đặc trưng đã chuẩn hóa Z-score (dải động đã được nén về đơn vị).

---

## GIAI ĐOẠN 3 — THỰC CHỨNG TÀI NGUYÊN NHÚNG

Nguyên tắc đo lường thống nhất: mọi số liệu báo cáo thuộc một trong hai mức rõ ràng — (i) phân tích tĩnh qua STM32Cube.AI (Flash, SRAM, MACC); (ii) đo động trên board STM32F4 thật (Latency, Energy). Python chỉ dùng trong giai đoạn phát triển thuật toán, không dùng làm nguồn số liệu benchmark.

### 3.1. Đo lường hiệu năng khối tiền xử lý DSP

- Triển khai mã C của ba phương án giải điều chế (FIR Hilbert nhân quả, Square-Law, Hybrid trực giao bậc (a)) và chạy micro-benchmark trên board STM32F4 (đếm chu kỳ qua DWT cycle counter), đo trong cùng chế độ nhân quả.
- Đối chiếu chi phí tài nguyên (SRAM, MACC/mẫu) giữa phương pháp DSP lai ghép và các phương pháp kinh điển.

### 3.2. Đo đạc tài nguyên AI, độ trễ và năng lượng

- Nạp các mô hình `.tflite` vào phần mềm phân tích STM32Cube.AI Studio.
- **Tài nguyên tĩnh:** Kết xuất các báo cáo tài nguyên cấp phát (Flash, SRAM) và mức tiêu hao phép tính (Static MACC) cho mục tiêu MCU STM32F4.
- **Hiệu năng động:** Đo thời gian thực thi (Inference Latency, ms) cho toàn bộ một chu kỳ chẩn đoán trên board thật, tách riêng và cộng gộp: (i) tiền xử lý DSP; (ii) trích xuất 32 đặc trưng (đối với MLP — gồm FFT, phổ Order, phổ Envelope); (iii) suy luận mô hình.
- **Năng lượng (mJ/chu kỳ chẩn đoán):** ước lượng theo công thức E = P × t, với t đo từ số chu kỳ xung nhịp × chu kỳ clock, và P lấy theo một trong hai cách (ghi rõ trong báo cáo): (a) dòng tiêu thụ danh định của STM32F4 ở tần số hoạt động và điện áp cấp cụ thể từ datasheet; hoặc (b) đo dòng thực tế bằng thiết bị đo công suất trên board. Không suy ra mJ chỉ từ clock cycles mà thiếu giả định công suất.

---

## GIAI ĐOẠN 4 — HOÀN THIỆN BÁO CÁO KẾT QUẢ

### Các bảng đánh giá định lượng mục tiêu

- **Bảng 1:** Tính trung thực của giao thức đánh giá (Random Split vs. File-based LOLO).
- **Bảng 2:** Đánh giá độ trễ và tài nguyên khối DSP (FIR Hilbert nhân quả vs. Square-Law vs. Hybrid trực giao bậc (a)) — tất cả đo trong cùng chế độ nhân quả; cột Phase Error ghi N/A cho Square-Law; MACC của FIR Hilbert báo cáo cả hai giá trị (có/không khai thác đối xứng).
- **Bảng 3:** So sánh toàn diện hiệu năng LOLO, tài nguyên lưu trữ (Flash/RAM), thời gian đáp ứng (Latency) và năng lượng tiêu thụ trên STM32F4 cho toàn pipeline (MLP bao gồm trích xuất đặc trưng vs. CNN raw vs. CNN envelope).

### So sánh (SOTA Comparison) & Minh bạch hóa

- Đối chiếu mức tiết kiệm tài nguyên của giải pháp đề xuất với các hệ thống Edge AI hiện hữu.
- Làm rõ giới hạn của phương pháp: kết quả DSP đo trên board thật (không phải profiler mô phỏng), giới hạn vật lý của CWRU trong bài toán tổng quát hóa, giả định về nguồn RPM cho đặc trưng Order, và giả định công suất khi quy đổi năng lượng.