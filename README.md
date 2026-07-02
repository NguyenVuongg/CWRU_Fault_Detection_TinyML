# Hệ thống chẩn đoán lỗi vòng bi động cơ bằng TinyML trên vi điều khiển STM32

## Giới thiệu

Đây là dự án nghiên cứu xây dựng hệ thống chẩn đoán lỗi vòng bi động cơ sử dụng **Edge AI** trên nền tảng **vi điều khiển STM32**.

Dự án sử dụng tín hiệu rung từ bộ dữ liệu **Case Western Reserve University (CWRU) Bearing Dataset**, kết hợp các kỹ thuật **xử lý tín hiệu số (DSP)**, **trích xuất đặc trưng**, **trí tuệ nhân tạo (AI)** và **triển khai mô hình lên thiết bị nhúng (Embedded AI)**.

Mục tiêu của dự án không chỉ đạt độ chính xác cao trong chẩn đoán lỗi mà còn tối ưu kích thước mô hình và tài nguyên phần cứng để có thể triển khai trên các dòng vi điều khiển STM32.

---

# Mục tiêu

* Xây dựng quy trình xử lý tín hiệu rung phục vụ chẩn đoán lỗi vòng bi.
* Trích xuất các đặc trưng miền thời gian và miền tần số.
* Huấn luyện mô hình AI có độ chính xác cao.
* Tối ưu mô hình cho hệ thống Edge AI/ TinyML.
* Đánh giá khả năng triển khai trên vi điều khiển STM32.
* Mô phỏng hệ thống hoạt động mà không cần phần cứng thực.

---

# Quy trình thực hiện

```
Bộ dữ liệu CWRU
        │
        ▼
Tiền xử lý dữ liệu
        │
        ▼
Chia tín hiệu thành các đoạn (Segmentation)
        │
        ▼
Biến đổi FFT
        │
        ▼
Trích xuất đặc trưng
        │
        ▼
Chuẩn hóa dữ liệu
        │
        ▼
Huấn luyện mô hình AI
        │
        ▼
Đánh giá mô hình
        │
        ▼
Lượng tử hóa (Quantization)
        │
        ▼
Triển khai lên STM32
```

---

# Cấu trúc thư mục

```text
BearingFaultDiagnosis/
│
├── data/
│   ├── raw/                  # Dữ liệu CWRU gốc (.mat)
│   ├── processed/            # Dữ liệu sau khi chia segment
│   └── features/             # Vector đặc trưng
│
├── src/
│   ├── preprocessing.py      # Đọc và tiền xử lý dữ liệu
│   ├── segmentation.py       # Chia tín hiệu thành các cửa sổ
│   ├── fft.py                # Biến đổi Fourier nhanh
│   ├── feature_extraction.py # Trích xuất đặc trưng
│   ├── normalization.py      # Chuẩn hóa dữ liệu
│   └── utils.py              # Hàm hỗ trợ
│
├── models/                   #     Mô hình đã huấn luyện
│
├── notebooks/                # Notebook phục vụ nghiên cứu
│
├── figures/                  # Hình ảnh minh họa
│
├── results/                  # Kết quả đánh giá
│
├── train.py                  # Huấn luyện mô hình
├── evaluate.py               # Đánh giá mô hình
├── requirements.txt
└── README.md
```

---

# Giai đoạn 1 - Tiền xử lý dữ liệu và xử lý tín hiệu số (DSP)

## Mục tiêu

Biến đổi tín hiệu rung thô thành các vector đặc trưng có kích thước nhỏ nhưng vẫn chứa đầy đủ thông tin phục vụ chẩn đoán.

## Các bước thực hiện

### 1. Đọc dữ liệu

* Đọc các file `.mat` từ bộ dữ liệu CWRU.
* Phân loại theo từng trạng thái lỗi.

### 2. Tiền xử lý

* Loại bỏ giá trị DC (nếu cần).
* Chuẩn hóa biên độ tín hiệu.
* Kiểm tra chất lượng dữ liệu.

### 3. Chia tín hiệu (Segmentation)

Chia tín hiệu dài thành nhiều đoạn có độ dài cố định nhằm tăng số lượng mẫu huấn luyện.

Ví dụ:

* Window Size = 2048 mẫu
* Overlap = 50%

### 4. Biến đổi FFT

Chuyển tín hiệu từ miền thời gian sang miền tần số bằng thuật toán FFT.

Mục đích:

* Phân tích phổ tần số.
* Phục vụ trích xuất đặc trưng miền tần số.

### 5. Trích xuất đặc trưng

#### Đặc trưng miền thời gian

* RMS
* Variance
* Skewness
* Kurtosis
* Crest Factor
* Shape Factor

#### Đặc trưng miền tần số

* Biên độ phổ FFT
* Biên độ tại tần số BPFI
* Biên độ tại tần số BPFO

### 6. Chuẩn hóa dữ liệu

Chuẩn hóa vector đặc trưng bằng:

* Standard Scaler
* hoặc Min-Max Scaler

---

# Giai đoạn 2 - Xây dựng mô hình AI

## Các mô hình dự kiến

* MLP
* CNN 1D
* Random Forest (tham khảo)
* SVM (tham khảo)

## Tiêu chí đánh giá

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Đồng thời đánh giá:

* Kích thước mô hình
* Số lượng tham số
* Thời gian suy luận

---

# Giai đoạn 3 - Triển khai Edge AI

## Nền tảng

* STM32
* TensorFlow Lite
* STM32Cube.AI
* ST Edge AI Core
* ST Edge AI Developer Cloud

## Mục tiêu

Đánh giá khả năng triển khai mô hình trên vi điều khiển thông qua:

* Dung lượng Flash
* Bộ nhớ RAM
* Số phép toán (MACC)
* Thời gian suy luận

---

# Mô phỏng hệ thống

Do không sử dụng phần cứng thực, dự án tiến hành mô phỏng:

* Luồng dữ liệu từ bộ dữ liệu CWRU theo dạng tuần tự.
* Giao tiếp UART bằng cổng COM ảo.
* Thiết bị Edge được mô phỏng bằng chương trình Python.
* HMI hoặc ứng dụng Android nhận kết quả chẩn đoán.

---

# Giai đoạn 4 - Đánh giá kết quả

Các chỉ tiêu đánh giá gồm:

* Accuracy
* Precision
* Recall
* F1-score
* Model Size
* Flash Usage
* RAM Usage
* Inference Time
* MACC

Kết quả sẽ được so sánh với các công trình nghiên cứu trước nhằm đánh giá tính khả thi của việc triển khai trên thiết bị Edge.

---

# Yêu cầu môi trường

* Python 3.11 trở lên

Các thư viện chính:

* NumPy
* SciPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow
* TensorFlow Lite
* PySerial

Cài đặt:

```bash
pip install -r requirements.txt
```

---

# Bộ dữ liệu

Sử dụng **Case Western Reserve University (CWRU) Bearing Dataset**.

Sau khi tải về, đặt toàn bộ file `.mat` vào thư mục:

```text
data/raw/
```

---

# Hướng phát triển

* Thu thập dữ liệu từ cảm biến rung thực tế.
* Triển khai trên kit STM32.
* Tối ưu bằng TinyML.
* Mở rộng sang nhiều bộ dữ liệu vòng bi khác.
* Xây dựng giao diện giám sát trực quan.
* Kết nối với ứng dụng Android hoặc Web Dashboard.

---

# Ghi chú

Đây là dự án phục vụ nghiên cứu khoa học và học thuật.

Một số nội dung như luồng dữ liệu thời gian thực và giao tiếp UART được mô phỏng bằng phần mềm nhằm đánh giá khả năng triển khai Edge AI trong điều kiện không có phần cứng thực.
