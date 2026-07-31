# TỔNG QUAN DỰ ÁN NGHIÊN CỨU KHOA HỌC (BẢN CHỐT CUỐI CÙNG)

## Tên đề tài
**Hệ thống Chẩn đoán Lỗi Vòng bi Động cơ bằng TinyML: Tối ưu hóa Đặc trưng Miền Order và Envelope trên Vi điều khiển STM32 Tài nguyên Thấp**

Dataset: CWRU Drive-End 12kHz

---

## TỔNG QUAN ĐỀ TÀI

**Bối cảnh & khoảng trống nghiên cứu:** CWRU là benchmark phổ biến nhất cho chẩn đoán lỗi vòng bi, nhưng phần lớn công trình công bố độ chính xác > 99% dựa trên cách chia train/test bị rò rỉ dữ liệu (random window split trên tín hiệu liên tục) — vấn đề đã được ghi nhận rộng rãi trong nhiều nghiên cứu gần đây, không phải trường hợp cá biệt. Đồng thời, các phép biến đổi miền tần số truyền thống (Hilbert Transform, FFT toàn dải) làm tăng đáng kể chi phí tính toán/bộ nhớ khi triển khai trên MCU giá rẻ như STM32, trong khi ít công trình đo đúng chi phí của *toàn bộ* pipeline (cả bước tiền xử lý DSP, không chỉ bước suy luận mô hình).

**Giá trị cốt lõi của đề tài:** không nằm ở từng kỹ thuật riêng lẻ (Square-Law, Goertzel, Feature Selection...), mà ở việc xây dựng **một pipeline hoàn chỉnh** — kiểm định chống rò rỉ dữ liệu, đánh giá tổng quát hóa qua điều kiện tải (LOLO), và tối ưu tài nguyên có chủ đích cho triển khai TinyML.

**Mục tiêu cụ thể:**
1. Quy trình kiểm   định hai tầng: File-based Split (chống rò rỉ) + Leave-One-Load-Out — LOLO (đo tổng quát hóa qua tải chưa từng thấy).
2. Square-Law Demodulation thay Hilbert Transform, Goertzel thay FFT toàn dải — đánh giá tài nguyên thật (Flash/RAM/MACC) qua STM32Cube.AI.
3. Kiểm chứng thực nghiệm: MLP trên đặc trưng chọn lọc theo kiến thức miền có đạt/vượt CNN1D học trực tiếp từ tín hiệu thô hay không.

**Phạm vi & giới hạn:**
- Chỉ đối chứng MLP (đặc trưng chọn lọc) vs. CNN1D nông (tín hiệu thô) — không mở rộng sang các kiến trúc deep learning khác.
- Bandpass cho Square-Law Demodulation cố định, chọn qua quan sát phổ — không dùng phương pháp dò tự động (Kurtogram).
- Đo tài nguyên chủ yếu qua static analysis (STM32Cube.AI); đo trên board STM32 thật là thực nghiệm mở rộng, không bắt buộc.
- **LOLO đo tổng quát hóa qua điều kiện tải, không qua vòng bi vật lý độc lập** — vì CWRU dùng chung 1 vòng bi qua nhiều mức tải liên tiếp.
- Order-domain Normalization không có ablation study riêng đo mức cải thiện cụ thể (giới hạn thời gian của một đồ án NCKH) — trình bày như lựa chọn thiết kế có cơ sở lý thuyết.

---

## ĐÓNG GÓP KHOA HỌC TRỌNG TÂM (NOVELTY & CONTRIBUTIONS)

**1. Quy trình kiểm định hai tầng, tách biệt rõ chức năng** — nhầm lẫn phổ biến trong nhiều công trình cùng lĩnh vực là gộp hai việc dưới đây làm một:
- **File-based Splitting** (chống rò rỉ dữ liệu): phân chia Train/Val/Test theo file gốc, ngăn các đoạn tín hiệu gần trùng lặp (do sliding window overlap) rơi vào cả hai tập.
- **Leave-One-Load-Out — LOLO** (đánh giá tổng quát hóa): huấn luyện trên 3 mức tải, kiểm định độc lập trên mức tải bị giấu.

**2. Tối ưu hóa đặc trưng hướng triển khai nhúng (Explainable Feature Selection):** rút gọn bể đặc trưng đa miền xuống ~30 đặc trưng trọng yếu bằng Random Forest Importance (đối chiếu Permutation Importance), kiểm tra độ ổn định qua chỉ số Jaccard giữa các fold LOLO.

**3. Tối ưu tài nguyên DSP trên MCU — hai luận điểm độc lập:**
- **Envelope Demodulation nhẹ**: Square-Law (`LowPass((bandpass(x))²)`) thay Hilbert Transform — chỉ dùng bộ lọc IIR miền thời gian, không cần FFT.
- **Goertzel cho tần số mục tiêu**: thay FFT toàn dải khi số tần số mục tiêu M nhỏ hơn log₂(N); nếu M vượt ngưỡng, chỉ còn lợi ích RAM — số liệu MACC thật quyết định luận điểm được trình bày.

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
- Đặc trưng nào có khả năng phân biệt lỗi tốt? (quan sát sơ bộ, làm cơ sở đối chiếu với Random Forest Importance chạy chính thức ở Giai đoạn 2)

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

**Thực nghiệm nhỏ đề xuất:** huấn luyện thử một mô hình đơn giản (ví dụ Random Forest) bằng Random Window Split để tự tay quan sát con số accuracy ảo, đối chiếu với kết quả cùng mô hình khi chạy qua File-based Split + LOLO. Đây chính là bằng chứng thực nghiệm sơ bộ cho RQ1/H1 (kết quả chính thức, đầy đủ vẫn báo cáo ở Giai đoạn 4).

**Mục tiêu:** làm rõ sự khác nhau giữa chống rò rỉ dữ liệu (File-based Split) và đánh giá khả năng tổng quát hóa (LOLO) — đây là cơ sở để chốt giao thức kiểm định cho toàn bộ đề tài (Đóng góp khoa học #1).

### 0.4. Hình thành giả thuyết nghiên cứu

Từ các quan sát ở 0.1–0.3, đề tài hình thành 5 câu hỏi nghiên cứu (RQ1–RQ5) và giả thuyết tương ứng, trình bày chi tiết ở phần **RESEARCH QUESTIONS & HYPOTHESES** ngay sau đây — đây là cầu nối chuyển từ giai đoạn khảo sát sang giai đoạn thiết kế/thực thi pipeline chính thức (Giai đoạn 1–4).

---

## RESEARCH QUESTIONS (RQ) & HYPOTHESES (H)

### RQ1 — Tính trung thực của phương pháp kiểm định

**RQ1:** Mức độ chênh lệch giữa độ chính xác đo được qua Random Window Split so với File-based Split + LOLO là bao nhiêu, trên cùng một tập dữ liệu CWRU?

**H1:** Random Window Split sẽ cho độ chính xác cao hơn đáng kể (ảo) so với File-based Split + LOLO, do khả năng xuất hiện các cửa sổ có mức tương quan rất cao ở cả tập huấn luyện và tập kiểm tra khi chia ngẫu nhiên theo window.

*(Tương ứng Bảng 1, mục 4.1 — thực nghiệm sơ bộ đã chạy thử ở mục 0.3)*

### RQ2 — Feature engineering (MLP) vs. Deep Learning thuần túy (CNN1D)

**RQ2:** Trên bài toán chẩn đoán lỗi vòng bi công nghiệp quy mô nhỏ, một mô hình MLP dựa trên đặc trưng miền kỹ thuật (Order + Envelope, đã chọn lọc) có đạt độ chính xác và độ ổn định (qua các fold LOLO) tương đương hoặc vượt trội so với một mô hình CNN1D học trực tiếp từ tín hiệu thô hay không?

**H2:** Một mô hình MLP sử dụng tập đặc trưng được thiết kế theo kiến thức miền kỹ thuật có khả năng đạt hiệu năng cạnh tranh với CNN1D, đồng thời có độ lệch chuẩn giữa các fold LOLO thấp hơn (ổn định hơn qua các điều kiện tải khác nhau), do CNN1D dễ overfit hơn trên tập dữ liệu công nghiệp quy mô nhỏ.

*(Tương ứng mục 2.3)*

### RQ3 — Đánh đổi giữa Feature Selection và tài nguyên/độ chính xác

**RQ3:** Việc rút gọn bể đặc trưng đầy đủ xuống ~30 đặc trưng (qua Random Forest Importance) ảnh hưởng thế nào đến độ chính xác phân loại và chi phí tài nguyên (RAM/MACC) trên MCU?

**H3:** Giảm số chiều đặc trưng sẽ làm giảm đáng kể RAM/MACC, trong khi độ chính xác giảm không đáng kể — thậm chí có thể tăng nhẹ nếu loại bỏ được đặc trưng nhiễu góp phần overfitting.

*(Tương ứng Bảng 2, mục 4.1)*

### RQ4 — Chi phí DSP: Square-Law vs. Hilbert Transform

**RQ4:** Square-Law Demodulation tiết kiệm được bao nhiêu thời gian thực thi và RAM so với Hilbert Transform khi triển khai trên lõi ARM Cortex-M?

**H4:** Square-Law Demodulation sẽ nhanh hơn và nhẹ RAM hơn đáng kể so với Hilbert Transform, do loại bỏ hoàn toàn nhu cầu tính FFT/IFFT và buffer số phức.

*(Tương ứng Bảng 3a, mục 4.1)*

### RQ5 — Chi phí DSP: Goertzel vs. FFT toàn dải

**RQ5:** Với số lượng tần số mục tiêu M đã chốt sau bước Feature Selection, Goertzel có mang lại lợi ích cả về tốc độ tính toán (MACC) lẫn RAM so với FFT toàn dải, hay chỉ một trong hai?

**H5:** Nếu M > log₂(N), Goertzel sẽ không nhanh hơn FFT về MACC/compute cycles, nhưng vẫn tiết kiệm RAM đáng kể (không cần buffer số phức, bảng twiddle factor). Đây là giả thuyết có cơ sở lý thuyết định lượng rõ ràng (độ phức tạp O(M·N) so với O(N log N)), kết quả gần như có thể dự đoán trước, chỉ cần xác nhận bằng số đo thật.

*(Tương ứng Bảng 3b, mục 4.1)*

### Bảng tổng hợp RQ ↔ Thực nghiệm ↔ Metric

| RQ | Thực nghiệm (Experiment) | Metric |
|---|---|---|
| RQ1 | Random Window Split vs. File-based Split + LOLO | Accuracy, F1, ΔAccuracy |
| RQ2 | MLP vs. CNN1D (qua LOLO) | Accuracy, F1, Std giữa các fold LOLO |
| RQ3 | Bể đặc trưng đầy đủ vs. ~30 đặc trưng | Accuracy, RAM, MACC, Flash |
| RQ4 | Square-Law vs. Hilbert Transform | Latency, RAM |
| RQ5 | Goertzel vs. FFT toàn dải | MACC, RAM |

### Sơ đồ luồng pipeline tổng thể

```
Giai đoạn 0: EDA (khảo sát cấu trúc, tín hiệu, phương pháp chia dữ liệu)
 ↓
Leakage-free protocol (File-based Split + LOLO)
 ↓
Order-domain Normalization
 ↓
Envelope Demodulation (Square-Law)
 ↓
Feature Pool → Feature Selection (~30 features)
 ↓
MLP (đối chứng CNN1D)
 ↓
INT8 Quantization
 ↓
STM32 (STM32Cube.AI — Flash/RAM/MACC)
```

---

## GIAI ĐOẠN 1 — TIỀN XỬ LÝ DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG ĐA MIỀN

*Mục tiêu: biến đổi tín hiệu rung động thời gian thô thành vector đặc trưng bất biến theo tải, phù hợp vi xử lý tài nguyên giới hạn.*

### 1.1. Chuẩn bị dữ liệu & thiết kế kiểm định chống rò rỉ

Dataset CWRU Drive-End 12kHz, 4 nhãn (Normal, Inner Race, Outer Race, Ball) — phạm vi dữ liệu đã chốt ở Giai đoạn 0.1.

**Quy tắc bắt buộc:** phân lập dữ liệu tuyệt đối theo file gốc — không chia trên sliding window thuộc cùng 1 file. Đây là bước chống leakage, thực hiện **trước khi** cắt sliding window.

**Cấu trúc split thực tế** — không dùng tập Test cố định 3 phần tách rời khỏi toàn bộ pipeline, mà tổ chức trực tiếp theo cấu trúc LOLO (định nghĩa đầy đủ ở mục 2.1):

- Với mỗi fold LOLO (giữ 1 mức tải làm Test), pool 3 tải còn lại được chia tiếp theo file thành Train (~80%) / Val (~20%).
- **Chỉ stratify theo `label`** trong bước chia Train/Val này — *không* stratify theo `(label, load)`, vì bản thân LOLO đã cố định load của Test là 1 giá trị duy nhất tách biệt khỏi pool Train/Val, khiến việc stratify theo load trở nên vô nghĩa (thậm chí mâu thuẫn logic với chính cấu trúc LOLO).

Kết quả: mỗi fold có 3 tập Train/Val/Test riêng, tổng cộng 4 fold cho 4 mức tải.

### 1.2. Order-domain Normalization

*(Đổi tên từ "Order-Tracking" — tên gọi trước dễ gây hiểu nhầm với Computed Order Tracking (COT) đầy đủ, một kỹ thuật cần tín hiệu tachometer/encoder mà CWRU không có.)*

Chuyển trục tần số Hz sang trục bậc quay (Order), dựa trên RPM danh định của từng file (`Order = f_Hz / f_rot`). Nhờ đó BPFI/BPFO/BSF (và FTF nếu sử dụng) được neo tại giá trị Order cố định bất kể mức tải, giúp model nhận diện lỗi ổn định hơn khi tốc độ động cơ thay đổi — bằng chứng trực quan cho lợi ích này đã khảo sát ở Giai đoạn 0.2.

Đây là Order Normalization đơn giản (chia trục tần số cho RPM danh định), không phải COT đầy đủ — vì CWRU chỉ có RPM danh định cố định mỗi file, không có tín hiệu tốc độ tức thời đồng thời. Ghi rõ ràng buộc này trong báo cáo để tránh overclaim phương pháp. Đề tài không thực hiện ablation riêng đo mức cải thiện cụ thể (đã nêu ở phần Phạm vi & giới hạn).

### 1.3. Giải điều chế Envelope nhúng (Square-Law Demodulation)

```
Envelope[n] = LowPassFilter( (x[n] * h_bandpass[n])² )
```

Tín hiệu qua lọc dải thông quanh vùng cộng hưởng cơ học (dải cố định, chọn qua quan sát phổ ở Giai đoạn 0.2), bình phương, rồi qua lọc thông thấp IIR Butterworth bậc 2 để tách sóng mang thông tin điều biên. Toàn bộ chuỗi xử lý chỉ dùng bộ lọc miền thời gian — không cần FFT/IFFT, không cần buffer số phức.

Lưu ý kỹ thuật: phép bình phương sinh ra hài bậc 2 của tần số sóng mang — bộ lọc thông thấp sau đó cần đủ dốc (order đủ cao) để loại bỏ thành phần này, chỉ giữ lại đúng đường bao thật.

### 1.4. Bể đặc trưng đa miền — cấu trúc, số chiều xác nhận sau khi cài đặt

**Nguyên tắc:** không cố định trước một con số cụ thể rồi tìm cách khớp — chốt cấu trúc/logic tính đặc trưng trước, cài đặt, rồi đếm số chiều thật và trình bày kèm bảng breakdown.

| Nhóm | Nội dung | Số chiều (điền sau khi cài đặt) |
|---|---|---|
| A — Miền thời gian | Kurtosis, RMS, Skewness, Variance, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, Peak, Mean, Std | 11 (cố định) |
| B — Phổ Order | Biên độ tại hài bậc 1–3 của các tần số mục tiêu (f_rot, BPFO, BPFI, BSF, FTF) | *(= số tần số mục tiêu × số hài, điền số thật; có thể loại FTF nếu biên độ quá yếu/không ổn định — cần ghi rõ lý do nếu bỏ)* |
| C — Phổ Envelope | Biên độ giải điều chế (qua Square-Law) tại các tần số động học vòng bi | *(điền số thật)* |

Chuẩn hóa: StandardScaler fit trên tập Train của mỗi fold, transform Val/Test tương ứng. Lưu `scaler.pkl` riêng cho từng fold để export sang STM32.

---

## GIAI ĐOẠN 2 — XÂY DỰNG MÔ HÌNH, KIỂM ĐỊNH LOLO & TỐI ƯU HÓA

*Mục tiêu: đạt độ tổng quát hóa cao trên tải lạ, cô đọng mô hình xuống mức siêu nhẹ.*

### 2.1. Kiểm định Leave-One-Load-Out (LOLO Cross-Validation)

Đây là bước **định nghĩa chính thức** cấu trúc split đã mô tả ở mục 1.1:

```
Fold 1: Train/Val = tải {1,2,3}  |  Test = tải {0}
Fold 2: Train/Val = tải {0,2,3}  |  Test = tải {1}
Fold 3: Train/Val = tải {0,1,3}  |  Test = tải {2}
Fold 4: Train/Val = tải {0,1,2}  |  Test = tải {3}
```

Trong mỗi fold, toàn bộ dữ liệu thuộc mức tải được giữ lại sẽ tạo thành tập Test. Ba mức tải còn lại được chia tiếp theo file gốc thành tập Train và Validation, đảm bảo không xảy ra rò rỉ dữ liệu giữa các tập.

Mỗi mô hình được huấn luyện trên tập Train, lựa chọn siêu tham số dựa trên tập Validation và đánh giá trên tập Test tương ứng. Kết quả cuối cùng được báo cáo dưới dạng giá trị **trung bình ± độ lệch chuẩn** của bốn fold LOLO. Nếu một fold có kết quả thấp hơn đáng kể so với các fold còn lại, nguyên nhân sẽ được phân tích trong phần Thảo luận.

Khuyến nghị chạy Random Forest baseline trước tiên (nhanh, ít nhạy tham số) để xác nhận toàn bộ pipeline Giai đoạn 1 hoạt động đúng, trước khi chuyển sang MLP/CNN1D.

### 2.2. Rút gọn đặc trưng giải thích được (Explainable Feature Selection)

Áp dụng Random Forest Importance trên bể đặc trưng đầy đủ ở mỗi fold, lọc ra ~30 đặc trưng tối ưu nhất. Đối chiếu bằng Permutation Importance để kiểm chứng thêm. Đo độ ổn định của tập đặc trưng chọn ra qua **chỉ số Jaccard** giữa các fold LOLO.

Tỷ trọng đóng góp của phổ Envelope trong bộ đặc trưng cuối là con số cần đo, trình bày như kết quả thực nghiệm sau khi chạy xong pipeline.

### 2.3. Đối chứng kiến trúc mô hình (Model Ablation)

**Chốt kiến trúc trước khi code, không đổi giữa chừng:**

- **MLP siêu gọn**: input = số đặc trưng đã chọn (~30) → Dense(32, relu) → Dense(16, relu) → Dense(4, softmax).
- **CNN1D nông** (baseline đối chứng, học trực tiếp từ tín hiệu thô): input = window 2048 mẫu → Conv1D → MaxPooling1D → Conv1D → GlobalAveragePooling1D → Dense(4, softmax).

Mục tiêu kiểm chứng giả thuyết H2 qua LOLO, không phải kết luận tiên nghiệm. Chênh lệch tài nguyên (RAM/Flash/MACC) giữa 2 kiến trúc được ghi nhận như một quan sát đi kèm, không phải một phần của hypothesis chính.

### 2.4. Lượng tử hóa mô hình

Chuyển MLP từ Float32 sang INT8 qua TensorFlow Lite (post-training quantization, dùng representative dataset từ tập Train). Kiểm tra chênh lệch accuracy trước/sau lượng tử hóa trên tập Test của fold tương ứng trước khi đưa sang Giai đoạn 3.

---

## GIAI ĐOẠN 3 — TỐI ƯU DSP NHÚNG & KIỂM CHỨNG TARGET VALIDATION

*Mục tiêu: kiểm chứng hiệu năng thực tế trên lõi ARM Cortex-M mà không phụ thuộc chi phí chế tạo phần cứng cơ khí.*

### 3.1. Chiến lược tối ưu DSP nhúng — hai luận điểm tách biệt, đo bằng số thật

- **Luận điểm A — Square-Law Demodulation thay Hilbert Transform**: đo thời gian thực thi (ms) và RAM (byte) trên cùng 1 lõi ARM Cortex-M qua STM32Cube.AI. *(Xem lưu ý về giả định triển khai Hilbert Transform ở mục 1.3.)*
- **Luận điểm B — Goertzel thay FFT toàn dải**: đo MACC thật cho cả 2 phương án với số lượng tần số mục tiêu M cụ thể đã chốt sau Giai đoạn 2.2. Nếu M > log₂(N), trình bày trung thực: Goertzel tiết kiệm RAM nhưng không nhất thiết nhanh hơn về compute cycles.

### 3.2. Đo đạc hiệu năng thực trên phần cứng

Nạp model `.tflite` vào STM32Cube.AI / ST Edge AI Studio, chạy Analyze lấy Flash/SRAM/MACC (static analysis — bắt buộc). Nếu điều kiện phòng lab cho phép, đo latency thực trên board STM32 vật lý (F407/G474/H7) qua ST Edge AI Developer Cloud — thực nghiệm mở rộng, không bắt buộc.

### 3.3. Mô phỏng kiến trúc giao tiếp (SIL Streaming Simulation)

Cổng COM ảo (com0com/pty): Node A (Edge Emulation) đọc tuần tự CWRU, chạy DSP + INT8 Inference, gửi mã chẩn đoán qua khung UART; Node B (HMI) giải mã, hiển thị, ghi log. Ghi rõ ranh giới: đây là mô phỏng tầng giao thức phần mềm, không phải tín hiệu điện thực. Giữ ở mức proof-of-concept, không đầu tư hoàn thiện.

---

## GIAI ĐOẠN 4 — HOÀN THIỆN BÁO CÁO NCKH & CÔNG BỐ KẾT QUẢ

### 4.1. Các bảng so sánh định lượng

- **Bảng 1**: Random Window Split (>99% ảo) vs. File-based Split + LOLO (độ chính xác thực tế). *(RQ1/H1)*
- **Bảng 2**: Bể đặc trưng đầy đủ (Full) vs. ~30 đặc trưng (sau Feature Selection) — RAM/MACC giảm bao nhiêu %, F1-Score thay đổi thế nào. *(RQ3/H3)*
- **Bảng 3a**: Square-Law Demodulation vs. Hilbert Transform — thời gian thực thi (ms) trên ARM Cortex-M. *(RQ4/H4)*
- **Bảng 3b**: Goertzel (M tần số đã chốt) vs. FFT toàn dải — RAM và MACC. *(RQ5/H5)*
- **Bảng bổ sung**: MLP vs. CNN1D qua 4 fold LOLO — accuracy/F1 trung bình ± độ lệch chuẩn. *(RQ2/H2)*

### 4.2. So sánh với công trình nghiên cứu hiện hữu (SOTA Comparison)

Mở rộng phạm vi so sánh sang **TinyML/Edge AI deployment nói chung**, không giới hạn cứng vào STM32. Làm nổi bật mức tiết kiệm tài nguyên bộ nhớ và tính trung thực trong phương pháp kiểm định.

### 4.3. Minh bạch hóa phạm vi nghiên cứu

Tách biệt rõ: kết quả đo phần cứng thật (nếu có) vs. static analysis vs. mô phỏng giao thức. Mọi con số cụ thể chỉ nêu sau khi đã đo/chạy thực nghiệm. Nêu rõ giới hạn của LOLO trên CWRU và giới hạn của Order-domain Normalization trong phần Kết luận/Hướng phát triển.

---

## Ghi chú định hướng trình bày
Đây không phải một đề tài "train CNN rồi deploy", mà là một **pipeline chẩn đoán lỗi vòng bi được kiểm định chống rò rỉ dữ liệu, đánh giá tổng quát hóa qua điều kiện tải, và tối ưu có chủ đích cho triển khai Edge AI tài nguyên thấp**. *"Leakage-Free Lightweight Bearing Fault Diagnosis on Resource-Constrained Edge AI using Order-Domain and Envelope Features"*.