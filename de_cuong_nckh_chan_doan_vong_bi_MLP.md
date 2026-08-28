# TỔNG QUAN DỰ ÁN NGHIÊN CỨU KHOA HỌC

## Tên đề tài
**Hệ thống Chẩn đoán Lỗi Vòng bi Động cơ bằng TinyML: Tối ưu hóa Tiền xử lý Lai ghép và So sánh MLP vs 1D CNN trên Vi điều khiển Tài nguyên Thấp**

Dataset: CWRU Drive-End 12kHz

---

## TỔNG QUAN ĐỀ TÀI

**Bối cảnh & khoảng trống nghiên cứu:** 
CWRU là benchmark phổ biến nhất cho chẩn đoán lỗi vòng bi. Tuy nhiên, phần lớn công trình công bố độ chính xác cao (>99%) dựa trên cách chia dữ liệu bị rò rỉ (random window split trên tín hiệu liên tục). Đồng thời, các phép biến đổi miền tần số truyền thống để giải điều chế (như Hilbert Transform) làm tăng đáng kể chi phí tính toán FFT và bộ nhớ số phức khi triển khai trên MCU giá rẻ như STM32. Các phương pháp thay thế nhẹ hơn như Square-Law lại vướng nhược điểm trễ pha (group delay) lớn và suy giảm biên độ xung do phụ thuộc vào bộ lọc thông thấp. Hiện tại, rất ít công trình đo đạc chi phí của *toàn bộ* đường ống (tiền xử lý DSP + suy luận AI) trên phần cứng thực tế để giải quyết bài toán đánh đổi này.

**Giá trị cốt lõi của đề tài:** 
Xây dựng một pipeline hoàn chỉnh với hai đóng góp đột phá:
1. **Kiến trúc tiền xử lý DSP lai ghép (Hybrid Pre-processing):** Đề xuất kết hợp mạng lọc IIR All-Pass trực giao (Quadrature Filter) và thuật toán xấp xỉ Alpha-Max Plus Beta-Min nhằm loại bỏ hoàn toàn tính toán FFT và hàm căn bậc hai (sqrt), khắc phục triệt để nhược điểm trễ pha của Square-Law.
2. **Kiểm định và tối ưu tài nguyên TinyML:** Áp dụng giao thức kiểm định chống rò rỉ (File-based Split + LOLO) để đánh giá trung thực hai phương pháp tiếp cận (MLP đặc trưng thủ công vs 1D CNN) trên cùng nền tảng nhúng.

**Mục tiêu cụ thể:**
1. Áp dụng quy trình kiểm định hai tầng: File-based Split (chống rò rỉ) + Leave-One-Load-Out (LOLO) để đo khả năng tổng quát hóa qua điều kiện tải.
2. Thiết kế và cấu trúc hóa hệ thống giải điều chế IIR trực giao kết hợp xấp xỉ Alpha-Max cho bài toán nhận diện tần số lỗi cơ khí.
3. Xây dựng và so sánh hai mô hình phân loại: MLP (32 đặc trưng cố định) và 1D CNN (raw và envelope).
4. Đánh giá tác động của tín hiệu bao hình (độ nhiễu thấp, không trễ pha từ DSP lai ghép) lên độ chính xác và tốc độ suy luận của mô hình AI.
5. Cung cấp các benchmark thực tế về Flash, SRAM, chu kỳ lệnh (MACC), độ trễ suy luận (Latency) và năng lượng tiêu thụ trên MCU STM32F4 thông qua STM32Cube.AI.

**Phạm vi & giới hạn:**
*   Chỉ sử dụng tập dữ liệu Drive-End 12kHz (xử lý nội suy mẫu file Normal 48kHz về 12kHz).
*   MLP sử dụng bộ đặc trưng cố định (không chạy Feature Selection). CNN sử dụng kiến trúc đồng bộ giữa hai biến thể (raw/envelope) để cô lập biến số kích thước đầu vào.
*   LOLO đo khả năng tổng quát hóa qua điều kiện tải vật lý hiện có, không qua vòng bi vật lý độc lập.
*   Tài nguyên phần cứng được phân tích tĩnh (static analysis) thông qua bộ công cụ STM32Cube.AI kết hợp đo lường thời gian thực thi thực tế.

---

## MỤC ĐÍCH TRỌNG TÂM

**1. Tối ưu hóa kiến trúc giải điều chế lai ghép (Hybrid Demodulation Architecture):**
*   **Tạo tín hiệu trực giao ($I/Q$):** Sử dụng bộ lọc IIR All-Pass để giữ độ trễ nhóm cực thấp, duy trì hình thái xung va đập sắc nét hơn Square-Law.
*   **Xấp xỉ biên độ:** Ứng dụng công thức Alpha-Max Plus Beta-Min ($\alpha\max(|I|,|Q|)+\beta\min(|I|,|Q|)$) để tính biên độ đường bao, bỏ qua phép toán căn bậc hai tốn kém chu kỳ ALU của vi điều khiển.

**2. Quy trình kiểm định hai tầng, tách biệt rõ chức năng:**
*   **File-based Splitting:** Phân chia Train/Val/Test theo file ID nguyên bản nhằm ngăn chặn sự tương quan của các cửa sổ trượt (sliding window) rơi vào nhiều tập khác nhau.
*   **Leave-One-Load-Out (LOLO):** Huấn luyện trên 3 mức tải, kiểm tra trên mức tải bị giấu để đánh giá độ vững (robustness) thực tế.

**3. So sánh hai phương pháp tiếp cận trên cùng nền tảng nhúng:**
*   **MLP với đặc trưng thủ công (32 chiều):** Kết hợp kiến thức miền động học vòng bi (Time + Order + Envelope), cực kỳ nhẹ tài nguyên, thích hợp cho MCU siêu nhỏ chạy pin.
*   **1D CNN học end-to-end:** Khảo sát khả năng tự trích xuất đặc trưng của mạng nơ-ron sâu với hai biến thể đầu vào (raw 2048 mẫu, envelope 1024 mẫu).

**4. Phân tích định lượng tài nguyên nhúng (Embedded Resource Profiling):**
*   Cung cấp bằng chứng thực nghiệm về sự đánh đổi giữa độ chính xác, không gian lưu trữ và thời gian phản hồi của cả hai khối (tiền xử lý DSP và suy luận AI) trên lõi kiến trúc ARM Cortex-M.

---

## RESEARCH QUESTIONS (RQ) & HYPOTHESES (H)

### RQ1 — Tính trung thực của phương pháp kiểm định
**RQ1:** Mức độ chênh lệch độ chính xác giữa Random Window Split và File-based Split + LOLO là bao nhiêu trên tập dữ liệu CWRU?
**H1:** Giao thức Random Window Split sẽ sinh ra độ chính xác cao ảo tưởng do rò rỉ dữ liệu giữa các cửa sổ chồng lấp, khác biệt đáng kể so với phương pháp kiểm định nghiêm ngặt LOLO.

### RQ2 — Hiệu năng của tiền xử lý DSP Lai ghép (Hybrid DSP)
**RQ2:** Hệ thống DSP lai ghép (IIR All-Pass + Alpha-Max) tối ưu hóa chu kỳ lệnh (MACC) và SRAM như thế nào so với biến đổi Hilbert truyền thống và Square-Law trên vi xử lý nhúng?
**H2:** Kiến trúc lai ghép sẽ giảm thiểu đáng kể chi phí RAM/MACC so với Hilbert (do loại bỏ FFT và số phức) và giữ nguyên được hình thái/biên độ xung so với hiện tượng trễ pha của Square-Law.

### RQ3 — Tác động của DSP Lai ghép lên Mô hình AI
**RQ3:** Tín hiệu đường bao sinh ra từ DSP lai ghép tác động thế nào đến độ chính xác phân loại (F1-Score) và độ hội tụ của mô hình MLP và 1D CNN?
**H3:** Đường bao có SNR cao và không méo dạng từ DSP lai ghép sẽ giúp mô hình (đặc biệt là CNN envelope) đạt độ chính xác cao hơn và hội tụ nhanh hơn so với đầu vào từ Square-Law.

### RQ4 — So sánh kiến trúc MLP vs. 1D CNN
**RQ4:** MLP (32 đặc trưng thủ công) và 1D CNN (raw/envelope) khác biệt ra sao về độ chính xác (LOLO test), cấu hình tài nguyên (Flash/RAM/MACC), thời gian suy luận (Inference Latency) và mức tiêu thụ năng lượng trên STM32F4?
**H4:** MLP đạt độ chính xác tương đương CNN nhưng chiếm dụng ít tài nguyên nhất. CNN (envelope) sẽ tối ưu hơn CNN (raw) về bộ nhớ do đầu vào nhỏ, tạo ra sự cân bằng cho bài toán TinyML, đồng thời MLP sẽ chiếm ưu thế tuyệt đối về độ trễ và khả năng tiết kiệm năng lượng.

### Bảng tổng hợp RQ ↔ Thực nghiệm ↔ Metric

| RQ | Thực nghiệm (Experiment) | Metric đo lường |
|---|---|---|
| RQ1 | Random Window Split vs. File-based Split + LOLO | Accuracy, F1, ΔAccuracy |
| RQ2 | DSP: Hilbert vs. Square-Law vs. Hybrid IIR | RAM (bytes), MACC, Phase Error, Latency |
| RQ3 | Đánh giá tác động của nguồn đầu vào DSP lên AI | Accuracy, F1 (LOLO test) |
| RQ4 | MLP (32 đặc trưng) vs. CNN1D (raw) vs. CNN1D (env) | Accuracy, Flash (KB), RAM (KB), Inference Latency (ms), Energy (mJ) |

---

## GIAI ĐOẠN 1 — TIỀN XỬ LÝ DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG

### 1.1. Chuẩn bị và Đồng bộ hóa Dữ liệu
*   Sử dụng dữ liệu CWRU Drive-End 12kHz.
*   **Xử lý Baseline:** Áp dụng Polyphase Filter (`resample_poly`) để hạ tần số lấy mẫu (downsample) các file Normal từ 48kHz về 12kHz, ngăn chặn hiện tượng sai lệch phổ tần số.
*   **Thiết lập Split:** Phân lập Train/Val/Test tuyệt đối theo nhãn ID file trước khi cắt cửa sổ.

### 1.2. Kỹ thuật Giải điều chế Lai ghép (Hybrid Demodulation)
*   **Xác định dải cộng hưởng:** Tự động hóa dò tìm tần số sóng mang tối ưu bằng thuật toán Fast Kurtogram ngoại tuyến (Offline), chốt hằng số dải băng thông cho phần cứng.
*   **Dịch pha trực giao:** Sử dụng cặp lọc IIR All-Pass tách tín hiệu thành In-phase ($I$) và Quadrature ($Q$).
*   **Xấp xỉ đường bao:** Tính toán đường bao tuyến tính thông qua bộ hệ số Alpha-Max Plus Beta-Min tối ưu.

### 1.3. Trích xuất đặc trưng và Chuẩn hóa
*   **Order-domain Normalization:** Chuyển trục Hz sang trục bậc quay (Order) dựa trên biến thiên RPM, giúp các tần số lỗi BPFO/BPFI/BSF đứng yên bất kể điều kiện tải.
*   **Bể đặc trưng MLP (32 chiều):** Tổng hợp 11 đặc trưng thời gian, 12 đặc trưng phổ Order, và 9 đặc trưng phổ Envelope. Tần số lồng (FTF) được bổ sung để khép kín không gian chẩn đoán.
*   **Dữ liệu chuỗi CNN:** Trích xuất mảng thô (2048 mẫu) và mảng đường bao (1024 mẫu), chuẩn hóa Z-score độc lập cho mỗi Fold.

---

## GIAI ĐOẠN 2 — XÂY DỰNG MÔ HÌNH & KIỂM ĐỊNH LOLO

### 2.1. Cấu trúc Leave-One-Load-Out (LOLO Cross-Validation)
*   **Fold 1:** Train/Val = Tải {1,2,3} | Test = Tải {0}
*   **Fold 2:** Train/Val = Tải {0,2,3} | Test = Tải {1}
*   **Fold 3:** Train/Val = Tải {0,1,3} | Test = Tải {2}
*   **Fold 4:** Train/Val = Tải {0,1,2} | Test = Tải {3}
*   Trong từng Pool Train/Val, áp dụng Stratified File-based Split để phân chia 80/20.

### 2.2. Kiến trúc Mô hình Trí tuệ nhân tạo
*   **MLP:** Input 32 chiều, thiết kế mạng nông tối giản, tập trung vào tốc độ thực thi.
*   **1D CNN:** Chuẩn hóa các khối Conv, Pooling và BatchNorm giữa hai biến thể CNN (raw) và CNN (envelope) nhằm cô lập biến số kích thước, đảm bảo sự so sánh công bằng về mặt kiến trúc.

### 2.3. Lượng tử hóa mô hình (Model Quantization)
*   Chuyển đổi toàn bộ mô hình (MLP, CNN) từ Float32 sang định dạng INT8 thông qua phương pháp Post-training Quantization (sử dụng TensorFlow Lite).
*   **Đánh giá suy hao:** Đo lường mức độ suy giảm độ chính xác (Accuracy Drop) trước và sau lượng tử hóa, đặc biệt chú trọng vào ảnh hưởng của lượng tử hóa INT8 đối với tín hiệu đầu vào Envelope có biên độ nhỏ (do đặc tính của xấp xỉ Alpha-Max).

---

## GIAI ĐOẠN 3 — THỰC CHỨNG TÀI NGUYÊN NHÚNG

### 3.1. Đo lường hiệu năng khối tiền xử lý DSP
*   Đánh giá tĩnh và động (Micro-benchmarking) thông qua mã C/C++ hoặc cấu hình Profile trên Python để đối chiếu chi phí tài nguyên (SRAM, chu kỳ MACC) giữa phương pháp DSP lai ghép mới và các phương pháp kinh điển.

### 3.2. Đo đạc tài nguyên AI, độ trễ và năng lượng
*   Nạp các mô hình `.tflite` vào phần mềm phân tích STM32Cube.AI Studio.
*   **Tài nguyên tĩnh:** Kết xuất các báo cáo tài nguyên cấp phát (Flash, SRAM) và mức tiêu hao phép tính (Static MACC) cho mục tiêu MCU STM32F4.
*   **Hiệu năng động:** Đo lường thời gian thực thi (Inference Latency - ms) cho một chu kỳ chẩn đoán và ước lượng mức tiêu thụ năng lượng (Energy Consumption - mJ/inference) dựa trên số chu kỳ xung nhịp (Clock cycles), nhằm chứng minh khả năng hoạt động độc lập trên thiết bị IoT/Edge dùng pin.

---

## GIAI ĐOẠN 4 — HOÀN THIỆN BÁO CÁO KẾT QUẢ

### Các bảng đánh giá định lượng mục tiêu
*   **Bảng 1:** Tính trung thực của giao thức đánh giá (Random Split vs. File-based LOLO).
*   **Bảng 2:** Đánh giá độ trễ và tài nguyên khối DSP (Hilbert vs. Square-Law vs. Hybrid IIR).
*   **Bảng 3:** So sánh toàn diện hiệu năng LOLO, tài nguyên lưu trữ (Flash/RAM), thời gian đáp ứng (Latency) và năng lượng tiêu thụ trên STM32F4 (MLP vs. CNN raw vs. CNN envelope).

### So sánh (SOTA Comparison) & Minh bạch hóa
*   Đối chiếu mức tiết kiệm tài nguyên của giải pháp đề xuất với các hệ thống Edge AI hiện hữu.
*   Làm rõ giới hạn của phương pháp: kết quả DSP được Benchmark từ công cụ profiler, giới hạn vật lý của CWRU trong bài toán tổng quát hóa.