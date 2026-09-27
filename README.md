# Hệ thống Chẩn đoán Lỗi Vòng bi Động cơ bằng TinyML: Tối ưu hóa Tiền xử lý Lai ghép và So sánh MLP vs 1D CNN trên Vi điều khiển Tài nguyên Thấp

**Dataset:** CWRU Drive-End 12kHz

## TỔNG QUAN ĐỀ TÀI

**Bối cảnh & khoảng trống nghiên cứu:**

* Giải điều chế bằng biến đổi Hilbert theo lối FFT (ví dụ `scipy.signal.hilbert`) đòi hỏi đệm theo khối và bộ nhớ số phức, không phù hợp với xử lý theo dòng mẫu trên MCU giá rẻ như STM32; bản nhân quả khả thi trên MCU là FIR Hilbert (Type III).

* Các phương pháp thay thế nhẹ hơn như Square-Law lại vướng nhược điểm trễ pha (group delay) do bộ lọc thông thấp và suy giảm biên độ xung, đồng thời đường bao mang đơn vị bình phương nên dải động rộng — bất lợi khi lượng tử hóa INT8.

* Hiện tại, rất ít công trình đo đạc chi phí của toàn bộ đường ống (tiền xử lý DSP + trích xuất đặc trưng + suy luận AI) trên phần cứng thực tế để giải quyết bài toán đánh đổi này cho các dòng vi điều khiển tài nguyên thấp.

**Giá trị cốt lõi của đề tài:**

Xây dựng một pipeline hoàn chỉnh với hai đóng góp:

1. **Kiến trúc tiền xử lý DSP lai ghép (Hybrid Pre-processing):** Đề xuất khối tạo tín hiệu trực giao nhân quả, không FFT, kết hợp xấp xỉ Alpha-Max Plus Beta-Min để loại bỏ hoàn toàn hàm căn bậc hai (sqrt). Khối trực giao được triển khai theo hai bậc và báo cáo trung thực cả hai:

   * (a) Bộ vi phân trung tâm chuẩn hóa Q\[n\] = (x\[n\] − x\[n−2\]) / (2·sin ωc) kèm nhánh trễ bù I\[n\] = x\[n−1\] — chỉ 1 phép trừ + 1 phép nhân mỗi mẫu, lệch pha đúng 90° tại mọi tần số nhưng biên độ nghiêng theo sinω / sinωc.

   * (b) Cặp lọc IIR All-Pass trực giao nhằm làm phẳng biên độ trong dải.

2. **Kiểm định và tối ưu tài nguyên TinyML:** Áp dụng giao thức kiểm định chống rò rỉ (File-based Split) kết hợp lặp N-seed để đánh giá thống kê khách quan, từ đó so sánh thực chứng hai phương pháp tiếp cận (MLP đặc trưng thủ công vs 1D CNN) trên cùng nền tảng nhúng STM32F4.

## MỤC TIÊU CỤ THỂ

1. Thiết kế và cấu trúc hóa hệ thống giải điều chế trực giao nhân quả (vi phân trung tâm chuẩn hóa là thực nghiệm chính; IIR All-Pass là mở rộng có điều kiện) kết hợp xấp xỉ Alpha-Max cho bài toán nhận diện tần số lỗi cơ khí.

2. Áp dụng quy trình phân chia dữ liệu File-based Split và lặp lại với N seed ngẫu nhiên nhằm đảm bảo tính tin cậy thống kê (mean ± std) của thuật toán học máy, thay thế cho việc thử nghiệm đơn lẻ.

3. Xây dựng và so sánh hai mô hình phân loại: MLP với 3 phương pháp DSP khác nhau và mở rộng mô hình 1D CNN sang hướng sử dụng đường bao envelope làm tín hiệu vào thay vì tín hiệu thô.

4. Đánh giá tác động của tín hiệu bao hình từ DSP Hybrid lên độ chính xác và tốc độ suy luận của mô hình AI.

5. Cung cấp các benchmark về Flash, SRAM, chu kỳ lệnh (MACC), độ trễ suy luận (Latency) và năng lượng tiêu thụ cho toàn bộ pipeline, với hai mức đo: phân tích tĩnh qua STM32Cube.AI và đo động trên board STM32F4 thật.

## PHẠM VI & GIỚI HẠN

* Chỉ sử dụng tập dữ liệu Drive-End 12kHz (xử lý nội suy mẫu file Normal 48kHz về 12kHz), gom chung dữ liệu từ các mức tải (0, 1, 2, 3 HP) để huấn luyện.

* Chiến lược đánh giá: Sử dụng phương pháp lặp N seed trên PC để lấy kết quả thống kê độ chính xác (mean ± std). Sau đó, **chỉ trích xuất một mô hình đại diện duy nhất** (tốt nhất hoặc trung vị) của mỗi phương pháp DSP để lượng tử hóa và nạp xuống MCU nhằm đo đạc tài nguyên. Tài nguyên phần cứng phụ thuộc vào kiến trúc mạng, không phụ thuộc vào trọng số.

* MLP sử dụng bộ đặc trưng cố định. CNN sử dụng kiến trúc đồng bộ giữa hai biến thể (raw/envelope) để cô lập biến số kích thước đầu vào.

* Tài nguyên phần cứng được đo ở hai mức: (i) phân tích tĩnh qua STM32Cube.AI (Flash, SRAM, MACC); (ii) đo động trên board STM32F4 thật cho Latency và năng lượng.

* Giới hạn đã biết của khối trực giao bậc (a): tỉ số biên độ |Q|/|I| nghiêng theo sin ω / sin ωc — mở rộng nếu muốn biên độ phẳng toàn dải thì phải dùng bậc (b).

* Đặc trưng miền Order yêu cầu biết RPM tại thời điểm suy luận; đây là giả định vận hành của hệ thống (lấy từ cảm biến hoặc cấu hình cố định), hệ thống không tự ước lượng RPM.

## 1. TỐI ƯU HÓA KIẾN TRÚC GIẢI ĐIỀU CHẾ LAI GHÉP (Hybrid Demodulation Architecture)

* **Tạo tín hiệu trực giao (I/Q):** Dùng bộ vi phân trung tâm chuẩn hóa dạng nhân quả (Q\[n\] = (x\[n\] - x\[n-2\]) / (2·sin ωc), I\[n\] = x\[n-1\]) để có lệch pha đúng 90° với trễ nhóm chỉ 1 mẫu và không cần FFT. Hướng nâng cấp có điều kiện là cặp lọc IIR All-Pass trực giao.

* **Xấp xỉ biên độ:** Ứng dụng công thức Alpha-Max Plus Beta-Min (a·max(|I|,|Q|) + β·min(|I|,|Q|)) để tính biên độ đường bao, bỏ qua phép toán căn bậc hai tốn kém chu kỳ ALU của vi điều khiển.

## 2. QUY TRÌNH KIỂM ĐỊNH DỮ LIỆU ĐẢM BẢO TIN CẬY THỐNG KÊ

* **File-based Splitting:** Phân chia Train/Val/Test tuyệt đối theo nhãn ID file trước khi cắt cửa sổ để ngăn chặn rò rỉ dữ liệu. Định nghĩa chốt: file_id = tên file .mat gốc (stem), TUYỆT ĐỐI không chứa chỉ số cửa sổ.

* **Lặp N Seed:** Để tránh rủi ro "may rủi" khi chia tập dữ liệu nhỏ (\~32 file), quy trình chia File-based được lặp lại N lần (ví dụ N=5) với các seed ngẫu nhiên khác nhau. Kết quả mô hình báo cáo dưới dạng **Trung bình ± Độ lệch chuẩn (Mean ± Std)**.

## 3. SO SÁNH HAI PHƯƠNG PHÁP TIẾP CẬN TRÊN CÙNG NỀN TẢNG NHÚNG

* **MLP với đặc trưng thủ công:** Kết hợp kiến thức miền động học vòng bi (Time + Order + Envelope), cực kỳ nhẹ tài nguyên, thích hợp cho MCU siêu nhỏ.

* **1D CNN học end-to-end:** Khảo sát khả năng tự trích xuất đặc trưng của mạng nơ-ron sâu với hai biến thể đầu vào (raw 2048 mẫu, envelope 1024 mẫu).

## RESEARCH QUESTIONS (RQ) & HYPOTHESES (H)

### RQ1 — Hiệu năng của tiền xử lý DSP Lai ghép (Hybrid DSP)

**RQ1:** Hệ thống DSP lai ghép (trực giao nhân quả bậc (a) + Alpha-Max) tối ưu hóa chu kỳ lệnh (MACC) và SRAM như thế nào so với FIR Hilbert nhân quả và Square-Law trên vi xử lý nhúng, khi cả ba dùng chung một cấu hình lọc nhân quả?

### RQ2 — Tác động của DSP Lai ghép lên Mô hình AI

**RQ2:** Tín hiệu đường bao sinh ra từ DSP lai ghép tác động thế nào đến độ chính xác phân loại (Accuracy, F1-Score qua N seed) và độ hội tụ của mô hình MLP và 1D CNN trên tập Test?

### RQ3 — So sánh hai biến thể 1D CNN và giới hạn tài nguyên Edge AI

**RQ3:** 1D CNN_raw vs 1D CNN_envelope khác biệt ra sao về độ chính xác (mean ± std), cấu hình tài nguyên (Flash/RAM/MACC), thời gian suy luận (Inference Latency) và mức tiêu thụ năng lượng khi triển khai thực tế trên STM32F4?

### Bảng tổng hợp RQ ↔ Thực nghiệm ↔ Metric

| RQ | Thực nghiệm (Experiment) | Metric đo lường | 
 | ----- | ----- | ----- | 
| RQ1 | DSP: FIR Hilbert (nhân quả) vs. Square-Law vs. Hybrid trực giao bậc (a) | RAM (bytes), MACC/mẫu, Phase Error (độ) — N/A cho Square-Law, Latency | 
| RQ2 | Đánh giá tác động của nguồn đầu vào DSP (Hybrid vs Square-Law) lên MLP và CNN | Accuracy/F1 (Mean ± Std qua N seed), số epoch hội tụ | 
| RQ3 | CNN1D (raw) vs. CNN1D (env) trên MCU | Accuracy/F1, Flash (KB), RAM (KB), Inference Latency (ms), Energy (mJ) | 

## GIAI ĐOẠN 1 — TIỀN XỬ LÝ DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG

### 1.1. Chuẩn bị và Đồng bộ hóa Dữ liệu

* Xử lý Baseline: Áp dụng Polyphase Filter để hạ tần số lấy mẫu các file Normal từ 48kHz về 12kHz.

* Bộ nhãn phân loại (10 lớp): Normal + 3 vị trí lỗi (IR, OR, B) × 3 mức đường kính lỗi (7, 14, 21 mil).

* Cửa sổ trượt: chốt kích thước cửa sổ 2048 mẫu và bước trượt (stride) cố định.

### 1.2. Kỹ thuật Giải điều chế Lai ghép (Hybrid Demodulation)

* Xác định dải cộng hưởng và chốt dải băng thông cho phần cứng.

* Dịch pha trực giao (bậc a — thực nghiệm chính): I\[n\] = x\[n-1\], Q\[n\] = (x\[n\] - x\[n-2\]) / (2·sin ωc).

* Xấp xỉ đường bao tuyến tính qua hệ số Alpha-Max Plus Beta-Min tối ưu.

### 1.3. Trích xuất đặc trưng và Chuẩn hóa

* Đặc trưng MLP: Tổng hợp 11 đặc trưng thời gian, 10 đặc trưng phổ Order, và 7 đặc trưng phổ Envelope. Chuẩn hóa Z-score với tham số (mean, std) fit độc lập trên tập Train.

* Dữ liệu chuỗi CNN: Trích xuất mảng thô (2048 mẫu) và mảng đường bao (1024 mẫu - lý do: tín hiệu đường bao đã qua decimate hệ số 2 sau LPF 750Hz để tối ưu tính toán).

## GIAI ĐOẠN 2 — XÂY DỰNG MÔ HÌNH

### 2.1. Cấu trúc File-based Split với N seed

* Gom toàn bộ dữ liệu tải (0, 1, 2, 3 HP) làm một tập pool duy nhất.

* Áp dụng Stratified File-based Split (đảm bảo tỷ lệ các lớp) để chia Train/Val/Test (vd: 70/15/15 hoặc 80/10/10) theo file ID.

* Lặp lại toàn bộ quá trình chia tách và huấn luyện N lần (N=5) với N seed ngẫu nhiên. Trích xuất chỉ số thống kê (Mean ± Std) cho báo cáo hàn lâm.

### 2.2. Lượng tử hóa mô hình (Model Quantization)

* **Tuyển chọn:** Sau khi có kết quả từ N seed, chọn ra 1 mô hình đại diện tốt nhất của mỗi phương pháp để mang xuống phần cứng.

* Chuyển đổi mô hình đại diện từ Float32 sang định dạng INT8 thông qua Post-training Quantization (TensorFlow Lite).

* Dữ liệu hiệu chuẩn (calibration set): Lấy từ tập Train tương ứng của seed đó.

* Khảo sát đặc thù: Đánh giá sự nhạy cảm của biến thể CNN envelope với sai số lượng tử do biên độ đầu vào rất nhỏ.

## GIAI ĐOẠN 3 — THỰC CHỨNG TÀI NGUYÊN NHÚNG

### 3.1. Đo lường hiệu năng khối tiền xử lý DSP

* Triển khai mã C của ba phương án giải điều chế (FIR Hilbert nhân quả, Square-Law, Hybrid trực giao) trên board STM32F4.

* Đo MACC/mẫu, chu kỳ lệnh (qua DWT cycle counter) và RAM cấp phát trong cùng một chế độ nhân quả.

### 3.2. Đo đạc tài nguyên AI

* Nạp các mô hình `.tflite` (1 mô hình đại diện/phương pháp) vào STM32Cube.AI Studio.

* **Tài nguyên tĩnh:** Báo cáo không gian bộ nhớ (Flash, SRAM) và phép tính (Static MACC).

* **Hiệu năng động:** Đo thời gian thực thi (Inference Latency, ms) cho toàn bộ một chu kỳ chẩn đoán (bao gồm DSP + trích xuất đặc trưng + Suy luận) trên STM32F4.

* **Năng lượng:** Tính toán công suất tiêu thụ mJ dựa trên dòng tiêu thụ và thời gian đo đạc thực tế hoặc danh định.

## GIAI ĐOẠN 4 — HOÀN THIỆN BÁO CÁO KẾT QUẢ

### Các bảng đánh giá định lượng mục tiêu

* **Bảng 1:** Đánh giá độ trễ và tài nguyên khối DSP (FIR Hilbert nhân quả vs. Square-Law vs. Hybrid trực giao bậc (a)) — tất cả đo trong cùng chế độ nhân quả.

* **Bảng 2:** So sánh toàn diện hiệu năng (Mean ± Std của Accuracy/F1), tài nguyên lưu trữ (Flash/RAM), thời gian đáp ứng (Latency) và năng lượng tiêu thụ trên STM32F4 cho toàn pipeline (MLP vs. CNN raw vs. CNN envelope).

### So sánh

* Đối chiếu mức tiết kiệm tài nguyên của giải pháp đề xuất với các hệ thống Edge AI hiện hữu.

* Làm rõ giới hạn của phương pháp: kết quả DSP đo trên board thật, giới hạn vật lý của CWRU, giả định về nguồn RPM cho đặc trưng Order.