# TỔNG QUAN DỰ ÁN NGHIÊN CỨU KHOA HỌC

## Tên đề tài
**Hệ thống Chẩn đoán Lỗi Vòng bi Động cơ bằng TinyML: So sánh MLP và 1D CNN trên Vi điều khiển STM32 Tài nguyên Thấp**

Dataset: CWRU Drive-End 12kHz

---

## TỔNG QUAN ĐỀ TÀI

**Bối cảnh & khoảng trống nghiên cứu:** CWRU là benchmark phổ biến nhất cho chẩn đoán lỗi vòng bi, nhưng phần lớn công trình công bố độ chính xác > 99% dựa trên cách chia train/test bị rò rỉ dữ liệu (random window split trên tín hiệu liên tục) — vấn đề đã được ghi nhận rộng rãi. Đồng thời, các phép biến đổi miền tần số truyền thống (Hilbert Transform, FFT toàn dải) làm tăng đáng kể chi phí tính toán/bộ nhớ khi triển khai trên MCU giá rẻ như STM32, trong khi ít công trình đo đúng chi phí của *toàn bộ* pipeline (cả bước tiền xử lý DSP, không chỉ bước suy luận mô hình).

**Giá trị cốt lõi của đề tài:** việc xây dựng **một pipeline hoàn chỉnh** — kiểm định chống rò rỉ dữ liệu, đánh giá tổng quát hóa qua điều kiện tải (LOLO), và tối ưu tài nguyên có chủ đích cho triển khai TinyML. Đề tài đặt trọng tâm vào việc **so sánh hai phương pháp tiếp cận** trên cùng một nền tảng nhúng:
1. **MLP với đặc trưng trích xuất thủ công** (Time + Order + Envelope) — tận dụng kiến thức miền, tối ưu tài nguyên.
2. **1D CNN** (raw và envelope) — đánh giá khả năng triển khai trên MCU.

**Mục tiêu cụ thể:**
1. Quy trình kiểm định hai tầng: File-based Split (chống rò rỉ) + Leave-One-Load-Out — LOLO (đo tổng quát hóa qua tải chưa từng thấy).
2. Xây dựng và so sánh hai mô hình:
   - **MLP** (32 đặc trưng cố định: Time + Order + Envelope)
   - **1D CNN** (hai biến thể: raw signal 2048 mẫu và envelope 1024 mẫu)
3. Square-Law Demodulation thay Hilbert Transform, Goertzel thay FFT toàn dải — đánh giá tài nguyên DSP trên MCU.
4. Đánh giá tài nguyên (Flash/RAM/MACC) và độ chính xác của cả hai mô hình trên STM32F4 qua STM32Cube.AI.

**Phạm vi & giới hạn:**
- **Pipeline chính:** So sánh MLP (32 đặc trưng) với 1D CNN (raw và envelope) trên cùng tập dữ liệu và cùng giao thức kiểm định LOLO.
- MLP sử dụng đặc trưng cố định (không Feature Selection). CNN sử dụng tín hiệu đầu vào đã chuẩn hóa.
- Đo tài nguyên chủ yếu qua static analysis (STM32Cube.AI).
- **LOLO đo tổng quát hóa qua điều kiện tải, không qua vòng bi vật lý độc lập** — vì CWRU dùng chung 1 vòng bi qua nhiều mức tải liên tiếp.
- Order-domain Normalization không có ablation study riêng.

---

## MỤC ĐÍCH TRỌNG TÂM

**1. Quy trình kiểm định hai tầng, tách biệt rõ chức năng:**
- **File-based Splitting** (chống rò rỉ dữ liệu): phân chia Train/Val/Test theo file gốc, ngăn các đoạn tín hiệu gần trùng lặp (do sliding window overlap) rơi vào cả hai tập.
- **Leave-One-Load-Out — LOLO** (đánh giá tổng quát hóa): huấn luyện trên 3 mức tải, kiểm định độc lập trên mức tải bị giấu.

**2. So sánh hai phương pháp tiếp cận trên cùng nền tảng nhúng:**
- **MLP với đặc trưng thủ công (32 chiều):** Tận dụng kiến thức miền động học vòng bi (Time + Order + Envelope), độ chính xác cao, cực kỳ nhẹ (dự kiến < 2KB Flash, < 2KB RAM).
- **1D CNN học end-to-end:** Hai biến thể (raw 2048 mẫu, envelope 1024 mẫu). Đánh giá khả năng học đặc trưng tự động và mức độ đánh đổi tài nguyên khi triển khai trên STM32F4.

**3. Tối ưu tài nguyên DSP trên MCU — hai luận điểm độc lập:**
- **Envelope Demodulation nhẹ**: Square-Law (`LowPass((bandpass(x))²)`) thay Hilbert Transform — chỉ dùng bộ lọc IIR miền thời gian, không cần FFT, không cần buffer số phức.
- **Goertzel cho tần số mục tiêu**: thay FFT toàn dải khi số tần số mục tiêu M nhỏ hơn log₂(N); tiết kiệm RAM đáng kể (không cần buffer số phức, bảng twiddle factor).

**4. Đánh giá toàn diện tài nguyên trên MCU:**
- Phân tích Flash/RAM/MACC của MLP và 1D CNN (cả raw và envelope) trên STM32F4.
- Cung cấp bằng chứng thực nghiệm về sự đánh đổi giữa độ chính xác và tài nguyên, giúp định hướng lựa chọn mô hình cho các ứng dụng TinyML.

---

## RESEARCH QUESTIONS (RQ) & HYPOTHESES (H)

### RQ1 — Tính trung thực của phương pháp kiểm định

**RQ1:** Mức độ chênh lệch giữa độ chính xác đo được qua Random Window Split so với File-based Split + LOLO là bao nhiêu, trên cùng một tập dữ liệu CWRU?

**H1:** Random Window Split sẽ cho độ chính xác cao hơn đáng kể (ảo) so với File-based Split + LOLO, do khả năng xuất hiện các cửa sổ có mức tương quan rất cao ở cả tập huấn luyện và tập kiểm tra khi chia ngẫu nhiên theo window.

*(Tương ứng Bảng 1, mục 4.1)*

### RQ2 — So sánh MLP vs. 1D CNN về độ chính xác và tài nguyên

**RQ2:** MLP với 32 đặc trưng thủ công và 1D CNN (raw và envelope) khác nhau như thế nào về độ chính xác phân loại (trên tập test LOLO) và chi phí tài nguyên (Flash/RAM/MACC) khi triển khai trên STM32F4?

**H2:** 
- MLP sẽ có độ chính xác tương đương hoặc cao hơn CNN do đặc trưng đã được tối ưu theo kiến thức miền, trong khi tiêu tốn ít tài nguyên hơn đáng kể.
- CNN (envelope) sẽ có tài nguyên thấp hơn CNN (raw) do kích thước đầu vào nhỏ hơn (1024 vs 2048), nhưng có thể độ chính xác thấp hơn do mất thông tin.
- CNN (raw) sẽ có độ chính xác cao nhất nhưng tiêu tốn nhiều tài nguyên nhất.

*(Tương ứng Bảng 2a và 2b, mục 4.1)*

### RQ3 — Chi phí DSP: Square-Law vs. Hilbert Transform

**RQ3:** Square-Law Demodulation tiết kiệm được bao nhiêu thời gian thực thi và RAM so với Hilbert Transform khi triển khai trên lõi ARM Cortex-M?

**H3:** Square-Law Demodulation sẽ nhanh hơn và nhẹ RAM hơn đáng kể so với Hilbert Transform, do loại bỏ hoàn toàn nhu cầu tính FFT/IFFT và buffer số phức.

*(Tương ứng Bảng 3a, mục 4.1)*

### RQ4 — Chi phí DSP: Goertzel vs. FFT toàn dải

**RQ4:** Với số lượng tần số mục tiêu M đã chốt (M = 21), Goertzel có mang lại lợi ích cả về tốc độ tính toán (MACC) lẫn RAM so với FFT toàn dải, hay chỉ một trong hai?

**H4:** Với M = 21 và N = 2048, Goertzel (O(M·N) ≈ 43K MACC) sẽ có MACC tương đương hoặc lớn hơn FFT (O(N log₂N) ≈ 22K MACC) nhưng tiết kiệm RAM đáng kể (không cần buffer số phức, bảng twiddle factor).

*(Tương ứng Bảng 3b, mục 4.1)*

### Bảng tổng hợp RQ ↔ Thực nghiệm ↔ Metric

| RQ | Thực nghiệm (Experiment) | Metric |
|---|---|---|
| RQ1 | Random Window Split vs. File-based Split + LOLO | Accuracy, F1, ΔAccuracy |
| RQ2 | MLP (32 features) vs. CNN1D (raw) vs. CNN1D (envelope) | Accuracy, F1, Flash, RAM, MACC |
| RQ3 | Square-Law vs. Hilbert Transform | Latency (ms), RAM (bytes) |
| RQ4 | Goertzel vs. FFT toàn dải | MACC, RAM (bytes) |

---

## GIAI ĐOẠN 0 — KHẢO SÁT VÀ HIỂU BỘ DỮ LIỆU

*Mục tiêu: hiểu đúng bản chất dữ liệu và các phương pháp trước khi thiết kế pipeline chính thức — mọi quyết định ở Giai đoạn 1–4 đều bắt nguồn từ những gì quan sát được ở đây.*

### 0.1. Khảo sát cấu trúc bộ dữ liệu

- Phân tích cấu trúc thư mục và các file `.mat` (vị trí cảm biến DE/FE/BA, tần số lấy mẫu 12kHz/48kHz).
- Thống kê số lượng mẫu theo loại lỗi và mức tải (đếm số file và độ dài file cho từng tổ hợp nhãn × tải).
- Xác định các thông số đo: tần số lấy mẫu, RPM danh định, loại và vị trí cảm biến, đường kính lỗi (EDM-seeded: 7/14/21/28 mils), vị trí lỗi Outer Race (6h/3h/12h).
- Làm rõ ý nghĩa từng điều kiện đo trong CWRU — bao gồm các điểm dễ gây sai sót nếu không kiểm tra kỹ:

| Điểm cần xác minh | Rủi ro nếu bỏ qua |
|---|---|
| Sampling rate thật của file Normal baseline (nhiều nguồn không thống nhất 12kHz hay 48kHz) | Đặc trưng Order/tần số của lớp Normal lệch hoàn toàn so với 3 lớp lỗi |
| Đường kính lỗi 0.028/ inch dùng vòng bi NTN, khác SKF 6205 dùng cho 0.007/0.014/0.021 inch | Công thức BPFO/BPFI/BSF tính theo SKF 6205 sẽ sai cho các file NTN nếu đưa chung vào bể dữ liệu |
| Outer Race có 3 vị trí lỗi (6h/3h/12h) với đặc tính điều biên khác nhau | Gộp cả 3 vào 1 nhãn "OR" tạo ra lớp không đồng nhất nội tại |

**Quyết định phạm vi dữ liệu (áp dụng cho toàn bộ đề tài):** chỉ dùng Drive-End 12kHz, chỉ dùng vị trí Outer Race Centered/6 o'clock; xác minh và xử lý sampling rate của Normal baseline trước khi đưa vào manifest chính thức.

**Sanity check:** nếu `duration_sec_at_12k` cho ra con số bất thường (ví dụ ~40s cho file lẽ ra chỉ ghi ~10s) trong khi `duration_sec_at_48k` hợp lý, đó là dấu hiệu file thực chất được ghi ở 48kHz.

**Kết quả mong đợi:**
- Bảng thống kê Dataset (số file, số mẫu mỗi lớp/tải, độ dài file).
- Sơ đồ cấu trúc Dataset (thư mục → điều kiện đo → nhãn).
- Danh mục điều kiện thí nghiệm (bảng RPM/tải, bảng đường kính lỗi ↔ loại vòng bi).

### 0.2. Phân tích tín hiệu rung động

Khảo sát tín hiệu trong miền thời gian và miền tần số, gồm:

- So sánh dạng sóng miền thời gian giữa Normal, Inner Race, Outer Race, Ball (cùng 1 tải) — quan sát biên độ và tính xung (impulsiveness).
- FFT của từng loại lỗi; so sánh phổ của cùng 1 loại lỗi giữa các mức tải khác nhau — quan sát các đỉnh phổ dịch chuyển theo tải.
- Phổ Envelope (sau Square-Law Demodulation) — so sánh giữa các lớp lỗi, quan sát đỉnh tại BPFO/BPFI/BSF.
- Chuyển trục Hz sang Order và quan sát lại: các đỉnh lỗi có "đứng yên" tại đúng vị trí Order bất kể tải hay không.

**Câu hỏi cần trả lời:**
- Load ảnh hưởng thế nào đến phổ rung động?
- Vì sao cần Order-domain Normalization? (minh chứng bằng hình: đỉnh phổ dịch chuyển ở trục Hz nhưng đứng yên ở trục Order)
- Vì sao cần Envelope Analysis? (tín hiệu thô/phổ thô thường không lộ rõ BPFO/BPFI do bị điều biên bởi cộng hưởng cơ khí; phổ Envelope mới bộc lộ rõ các đỉnh này)

**Kết quả mong đợi:** bộ hình minh họa (chồng phổ thời gian, chồng phổ FFT, chồng phổ Envelope, chồng phổ Order) — dùng trực tiếp làm minh chứng trực quan cho phần Cơ sở lý thuyết của báo cáo.

### 0.3. Phân tích phương pháp chia dữ liệu

Khảo sát ảnh hưởng của các chiến lược chia dữ liệu bằng thực nghiệm nhỏ và hình minh họa, trước khi chốt giao thức kiểm định chính thức:

- **Random Window Split**: minh họa các sliding window chồng lấp (overlap) từ cùng 1 file có thể rơi vào cả train và test.
- **File-based Split**: minh họa chia theo file gốc, loại bỏ khả năng 2 window gần như trùng lặp rơi vào 2 tập khác nhau.
- **Leave-One-Load-Out (LOLO)**: minh họa từng fold giữ 1 mức tải làm test độc lập với 3 tải còn lại dùng train/val.

**Hình minh họa cần dựng:**
- Sliding Window: 2 cửa sổ liền kề, overlap 50%, thể hiện độ tương quan cao giữa chúng.
- Data Leakage: minh họa 2 window gần như giống hệt bị tách vào 2 tập khác nhau khi random split.
- Quy trình LOLO: sơ đồ 4 fold, mỗi fold giữ đúng 1 mức tải làm test.

**Thực nghiệm nhỏ đề xuất:** huấn luyện thử một mô hình đơn giản (Random Forest) bằng Random Window Split để quan sát con số accuracy ảo, đối chiếu với kết quả khi chạy qua File-based Split + LOLO. Đây là bằng chứng thực nghiệm sơ bộ cho RQ1/H1.

**Mục tiêu:** làm rõ sự khác nhau giữa chống rò rỉ dữ liệu (File-based Split) và đánh giá khả năng tổng quát hóa (LOLO).

### 0.4. Hình thành giả thuyết nghiên cứu

Từ các quan sát ở 0.1–0.3, đề tài hình thành 4 câu hỏi nghiên cứu (RQ1–RQ4) và giả thuyết tương ứng, trình bày chi tiết ở phần **RESEARCH QUESTIONS & HYPOTHESES** — đây là cầu nối chuyển từ giai đoạn khảo sát sang giai đoạn thiết kế/thực thi pipeline chính thức (Giai đoạn 1–4).

---

## GIAI ĐOẠN 1 — TIỀN XỬ LÝ DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG

*Mục tiêu: biến đổi tín hiệu rung động thời gian thô thành dạng phù hợp cho cả hai mô hình (MLP và CNN).*

### 1.1. Chuẩn bị dữ liệu & thiết kế kiểm định chống rò rỉ

Dataset CWRU Drive-End 12kHz, 4 nhãn (Normal, Inner Race, Outer Race, Ball) — phạm vi dữ liệu đã chốt ở Giai đoạn 0.1.

**Quy tắc bắt buộc:** phân lập dữ liệu tuyệt đối theo file gốc — không chia trên sliding window thuộc cùng 1 file. Đây là bước chống leakage, thực hiện **trước khi** cắt sliding window.

**Cấu trúc split thực tế** — tổ chức trực tiếp theo cấu trúc LOLO:

- Với mỗi fold LOLO (giữ 1 mức tải làm Test), pool 3 tải còn lại được chia tiếp theo file thành Train (~80%) / Val (~20%).
- **Chỉ stratify theo `label`** trong bước chia Train/Val này — *không* stratify theo `(label, load)`.

Kết quả: mỗi fold có 3 tập Train/Val/Test riêng, tổng cộng 4 fold cho 4 mức tải.

### 1.2. Order-domain Normalization

Chuyển trục tần số Hz sang trục bậc quay (Order), dựa trên RPM danh định của từng file (`Order = f_Hz / f_rot`). Nhờ đó BPFI/BPFO/BSF được neo tại giá trị Order cố định bất kể mức tải, giúp model nhận diện lỗi ổn định hơn khi tốc độ động cơ thay đổi.

Đây là Order Normalization đơn giản (chia trục tần số cho RPM danh định), không phải COT đầy đủ — vì CWRU chỉ có RPM danh định cố định mỗi file, không có tín hiệu tốc độ tức thời.

### 1.3. Giải điều chế Envelope nhúng (Square-Law Demodulation)

Envelope[n] = LowPassFilter( (x[n] * h_bandpass[n])² )

Tín hiệu qua lọc dải thông quanh vùng cộng hưởng cơ học (dải cố định 2300–3800 Hz), bình phương, rồi qua lọc thông thấp IIR Butterworth bậc 2 để tách sóng mang thông tin điều biên. Toàn bộ chuỗi xử lý chỉ dùng bộ lọc miền thời gian — không cần FFT/IFFT, không cần buffer số phức.

Envelope được sử dụng cho:
- **MLP:** làm đầu vào để trích xuất đặc trưng phổ Envelope (Nhóm C).
- **1D CNN (envelope):** làm đầu vào trực tiếp cho mô hình (1024 mẫu).

### 1.4. Bể đặc trưng cho MLP — 32 chiều cố định

**Lý do chốt cứng 32 đặc trưng:** Việc chọn đặc trưng bằng Random Forest Importance không đảm bảo tính ổn định giữa các fold LOLO, đồng thời làm tăng độ phức tạp của pipeline. Do đó, MLP sử dụng bộ 32 đặc trưng cố định dựa trên kiến thức miền động học vòng bi, đảm bảo tính giải thích được và nhất quán.

| Nhóm | Nội dung | Số chiều |
|---|---|---|
| **A — Miền thời gian** | Mean, Std, RMS, Peak, Kurtosis, Skewness, Variance, Crest Factor, Shape Factor, Impulse Factor, Margin Factor | **11** |
| **B — Phổ Order** | Biên độ tại hài bậc 1–3 của f_rot, BPFO, BPFI, BSF (4 tần số × 3 hài) | **12** |
| **C — Phổ Envelope** | Biên độ tại hài bậc 1–3 của BPFO, BPFI, BSF (3 tần số × 3 hài) | **9** |
| **Tổng** | | **32** |

**Chuẩn hóa:** StandardScaler fit trên tập Train của mỗi fold, transform Val/Test tương ứng. Lưu `scaler.pkl` riêng cho từng fold.

### 1.5. Chuẩn bị dữ liệu cho 1D CNN

- **CNN (raw):** Sử dụng sliding window với window_size = 2048, stride = 1024 (50% overlap). Đầu vào là tín hiệu thô (đã chuẩn hóa mean=0, std=1).
- **CNN (envelope):** Sử dụng sliding window với window_size = 1024, stride = 512 (50% overlap). Đầu vào là tín hiệu envelope (Square-Law, đã chuẩn hóa mean=0, std=1).

Cả hai đều giữ nguyên `file_id` để đảm bảo chia train/val/test không bị rò rỉ theo giao thức LOLO.

---

## GIAI ĐOẠN 2 — XÂY DỰNG MÔ HÌNH, KIỂM ĐỊNH LOLO & ĐÁNH GIÁ TÀI NGUYÊN

*Mục tiêu: xây dựng và so sánh MLP và 1D CNN về độ chính xác và tài nguyên trên STM32.*

### 2.1. Kiểm định Leave-One-Load-Out (LOLO Cross-Validation)

Đây là bước **định nghĩa chính thức** cấu trúc split áp dụng cho cả MLP và CNN:

Fold 1: Train/Val = tải {1,2,3} | Test = tải {0}
Fold 2: Train/Val = tải {0,2,3} | Test = tải {1}
Fold 3: Train/Val = tải {0,1,3} | Test = tải {2}
Fold 4: Train/Val = tải {0,1,2} | Test = tải {3}


Trong mỗi fold, toàn bộ dữ liệu thuộc mức tải được giữ lại sẽ tạo thành tập Test. Ba mức tải còn lại được chia tiếp theo file gốc thành tập Train và Validation (80/20). Mỗi mô hình được huấn luyện trên tập Train, lựa chọn siêu tham số dựa trên tập Validation và đánh giá trên tập Test. Kết quả cuối cùng được báo cáo dưới dạng **trung bình ± độ lệch chuẩn** của bốn fold.

### 2.2. Kiến trúc mô hình MLP

**MLP siêu gọn** (đã chốt): Input(32) → Dense(32, relu) → Dense(16, relu) → Dense(4, softmax)

- **Tổng tham số:** 1.652.
- **Mục tiêu:** Đánh giá khả năng tổng quát hóa qua LOLO và đo tài nguyên trên STM32.

### 2.3. Kiến trúc 1D CNN

**CNN siêu nhẹ** (đã chốt):

| Tầng | Tham số | Đầu ra |
|---|---|---|
| Input (raw) | shape=(2048, 1) | (2048, 1) |
| Conv1D | filters=8, kernel=16, stride=4 | (512, 8) |
| MaxPooling1D | pool_size=4 | (128, 8) |
| Conv1D | filters=16, kernel=8 | (128, 16) |
| GlobalAveragePooling1D | - | (16) |
| Dense | 4 classes, softmax | (4) |

**Hai biến thể đầu vào:**
1. **CNN1D (raw):** Input shape = (2048, 1)
2. **CNN1D (envelope):** Input shape = (1024, 1)

**Mục tiêu:** So sánh độ chính xác và tài nguyên với MLP, đánh giá khả năng triển khai trên STM32F4.

### 2.4. Lượng tử hóa mô hình

Chuyển cả MLP và CNN từ Float32 sang INT8 qua TensorFlow Lite (post-training quantization, dùng representative dataset từ tập Train). Kiểm tra chênh lệch accuracy trước/sau lượng tử hóa trên tập Test của từng fold.

---

## GIAI ĐOẠN 3 — TỐI ƯU DSP NHÚNG & KIỂM CHỨNG TARGET VALIDATION

*Mục tiêu: kiểm chứng hiệu năng thực tế trên lõi ARM Cortex-M.*

### 3.1. Chiến lược tối ưu DSP nhúng

- **Luận điểm A — Square-Law vs Hilbert Transform:** Đo thời gian thực thi (ms) và RAM (bytes) trên Python (timeit) kết hợp lý thuyết.
- **Luận điểm B — Goertzel vs FFT toàn dải:** Tính MACC lý thuyết và RAM cho cả 2 phương án với M = 21 tần số mục tiêu và N = 2048:
  - Goertzel: O(M·N) ≈ 43.008 MACC
  - FFT: O(N log₂N) ≈ 22.528 MACC
  - Kết luận dự kiến: Goertzel **không nhanh hơn** FFT về MACC, nhưng **tiết kiệm RAM** (không cần buffer số phức, bảng twiddle factor).

### 3.2. Đo đạc hiệu năng thực trên phần cứng

Nạp các model `.tflite` (MLP, CNN raw, CNN envelope) vào STM32Cube.AI Studio, chạy Analyze để lấy Flash/SRAM/MACC (static analysis). So sánh tài nguyên của ba mô hình trên cùng vi điều khiển STM32F4.

### 3.3. Mô phỏng kiến trúc giao tiếp (SIL Streaming Simulation)

Cổng COM ảo: Node A (Edge Emulation) đọc tuần tự CWRU, chạy DSP + INT8 Inference, gửi mã chẩn đoán qua UART; Node B (HMI) giải mã, hiển thị, ghi log. Giữ ở mức proof-of-concept.

---

## GIAI ĐOẠN 4 — HOÀN THIỆN BÁO CÁO NCKH & CÔNG BỐ KẾT QUẢ

### 4.1. Các bảng so sánh định lượng

- **Bảng 1**: Random Window Split vs. File-based Split + LOLO (RQ1) — Accuracy, F1, ΔAccuracy.
- **Bảng 2a**: So sánh độ chính xác LOLO của MLP vs. CNN (raw) vs. CNN (envelope) — Accuracy, F1 (mean ± std).
- **Bảng 2b**: So sánh tài nguyên trên STM32F4 của MLP vs. CNN (raw) vs. CNN (envelope) — Flash (KB), RAM (KB), MACC.
- **Bảng 3a**: Square-Law vs. Hilbert Transform (RQ3) — Latency (ms), RAM (bytes).
- **Bảng 3b**: Goertzel vs. FFT toàn dải (RQ4) — MACC, RAM (bytes).

### 4.2. So sánh với công trình nghiên cứu hiện hữu (SOTA Comparison)

Mở rộng phạm vi so sánh sang **TinyML/Edge AI deployment nói chung**, không giới hạn cứng vào STM32. Làm nổi bật:
- Mức tiết kiệm tài nguyên của MLP (dự kiến < 2KB Flash, < 2KB RAM).
- Khả năng triển khai 1D CNN trên STM32F4 với mức đánh đổi chấp nhận được.
- Tính trung thực trong phương pháp kiểm định (File-based Split + LOLO).

### 4.3. Minh bạch hóa phạm vi nghiên cứu

- Tách biệt rõ: MLP (pipeline chính) và 1D CNN (thực nghiệm đối chứng trọng tâm).
- Kết quả đo DSP (RQ3, RQ4) dựa trên lý thuyết và Python timeit (không phải đo trên MCU thật).
- Giới hạn của LOLO trên CWRU và giới hạn của Order-domain Normalization được nêu rõ trong phần Kết luận/Hướng phát triển.

---

## Ghi chú định hướng trình bày

Đây là một đề tài **so sánh hai phương pháp tiếp cận (MLP với đặc trưng thủ công và 1D CNN học end-to-end) trên cùng một pipeline kiểm định chống rò rỉ dữ liệu và đánh giá tổng quát hóa qua điều kiện tải, hướng đến triển khai trên MCU tài nguyên thấp**.

**Tên gợi ý (tiếng Anh):**
*"Leakage-Free Comparison of Feature-Based MLP and End-to-End 1D CNN for Bearing Fault Diagnosis on Resource-Constrained Edge AI"*

**Điểm nhấn:**
- Pipeline kiểm định trung thực (File-based Split + LOLO).
- So sánh toàn diện MLP vs. 1D CNN (raw và envelope) về độ chính xác và tài nguyên.
- MLP siêu nhẹ với 32 đặc trưng cố định (dự kiến < 2KB Flash, < 2KB RAM).
- 1D CNN (raw và envelope) đánh giá khả năng học end-to-end và mức đánh đổi tài nguyên.
- Square-Law và Goertzel tối ưu DSP cho MCU tài nguyên thấp.
