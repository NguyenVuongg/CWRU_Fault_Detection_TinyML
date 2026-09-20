# Tóm tắt Giai đoạn 1: Tiền xử lý và chuẩn bị dữ liệu

## 1. Phạm vi

Giai đoạn 1 gồm sáu notebook trong `notebooks/giai_doan_1_tien_xu_ly`, nhằm chuyển dữ liệu CWRU thô thành manifest sạch, cấu hình DSP thống nhất, bảng đặc trưng cho MLP, cửa sổ tín hiệu cho CNN và các phép kiểm định đường bao phục vụ TinyML.

## 2. Kết quả chính

- Đã rà soát 161 file đầu vào và loại 121 file không phù hợp scope. Manifest cuối còn **40 file**.
- Tập 40 file cân bằng theo nhãn và tải: B, IR, OR, Normal; mỗi nhãn có 3 file ở mỗi mức tải, riêng Normal có 1 file ở mỗi mức tải theo cấu trúc dữ liệu hiện tại.
- Đã sao chép/tổ chức 40 file vào `data/clean/` và tạo `manifest_clean.csv`.
- Sampling rate thực được lấy từ `resolved_sample_rate_hz`, không dùng một bảng override thủ công cho từng file. Phân bố fs gốc:
  - 36 file ở 12 kHz;
  - 3 file ở 48 kHz;
  - 1 file ở 24 kHz.
- Các file khác 12 kHz được resample trong RAM về target **12 kHz** trước khi trích đặc trưng hoặc tạo cửa sổ; file `.mat` gốc không bị thay đổi.

## 3. Nội dung từng notebook

### 01 - Build manifest

- Đọc/build manifest từ dữ liệu raw và áp dụng bộ lọc phạm vi nghiên cứu.
- Loại các nhóm cảnh báo liên quan đến vòng bi NTN, vị trí OR ngoài phạm vi, cảm biến ngoài phạm vi, tần số khai báo ngoài phạm vi và các metadata/tín hiệu thiếu.
- Xuất `manifest.csv`, `manifest_filtered.csv`, `01_warnings.csv` và hình phân bố dữ liệu.
- Kiểm tra uniqueness của `file_path`, phân bố nhãn theo tải và các trường bắt buộc.

Kết quả ghi trong notebook: **121/161 file bị loại, còn 40 file**. Sau lọc vẫn có cảnh báo metadata ở 4 file Normal do `rpm_from_file` bị thiếu; notebook ghi nhận các giá trị này không chặn pipeline sau vì RPM danh định theo tải được lấy từ `common/config.py`.

### 02 - Organize clean data

- Tổ chức 40 file đã lọc vào `data/clean/`.
- Tạo `manifest_clean.csv` với `resolved_sample_rate_hz` làm nguồn duy nhất cho fs.
- Có kiểm tra các file Normal có fs suy luận thay vì fs khai báo.

Điểm cần lưu ý: notebook đánh dấu **4 file có fs suy luận**, cần xác minh thủ công trước khi coi sampling rate là hoàn toàn chắc chắn. Đây là rủi ro dữ liệu quan trọng hơn cảnh báo RPM thiếu vì fs tác động trực tiếp đến bandpass, resample và tần số đặc trưng.

### 03 - Choose bandpass

- Dùng FFT và tần số đặc trưng vòng bi để khảo sát vùng cộng hưởng trên các nhãn/tải đại diện.
- Chốt cấu hình dùng chung:
  - bandpass: **2300-3800 Hz**;
  - tần số trung tâm: **3050 Hz**;
  - lowpass envelope: **750 Hz**;
  - bandpass Butterworth bậc 4, lowpass bậc 2.
- Lưu cấu hình vào `bandpass_config.json` để notebook 04, 05 và 06 dùng chung, tránh lệch hằng số.
- Kiểm tra cho thấy BPFI bậc 3 khoảng 486.6 Hz bị suy giảm khoảng **-0.71 dB** tại lowpass 750 Hz; mức này được chấp nhận để giữ thành phần và hạn chế méo.
- Notebook cũng ghi nhận một số thành phần hài bị gộp do độ phân giải FFT khoảng 5.86 Hz/bin; vì vậy bandpass được xem là dải thỏa hiệp chung, không phải cộng hưởng tối ưu riêng cho từng nhãn lỗi.

### 04 - Build feature table

- Xây dựng bảng đặc trưng MLP 28 chiều cho ba phương pháp đường bao:
  - Square-Law;
  - Hilbert FIR;
  - Hybrid.
- Xuất Parquet và CSV tương ứng, cùng alias `features_mlp.parquet`/`features_mlp.csv` dùng cho giai đoạn 2.
- Đưa resample ra thành bước rõ ràng trong notebook; các đặc trưng được tính sau khi đưa tín hiệu về 12 kHz.
- Dùng cùng bandpass, lowpass, cửa sổ 2048 mẫu, stride 1024 và warmup 2048 mẫu.

### 05 - Prepare CNN windows

- Tạo cửa sổ raw dài **2048 mẫu @12 kHz**, tương đương 170.67 ms.
- Tạo cửa sổ envelope dài **1024 mẫu @6 kHz**, cũng tương đương 170.67 ms sau decimation 1/2.
- Dùng chung `file_id`, `window_idx`, warmup, stride và quy tắc resample với nhánh MLP để hai nhánh có thể so sánh và chia tập theo file/LOLO nhất quán.
- Tạo cửa sổ envelope cho cả Square-Law, Hilbert FIR và Hybrid.
- Notebook kiểm tra khóa `(file_id, window_idx)` giữa các phương pháp và kiểm tra số cửa sổ theo file. Log cho thấy có **4623 cặp file_id/cửa sổ** dùng chung giữa các nhánh.
- Xuất Parquet đầy đủ, metadata CSV và một tập sample CSV để kiểm tra.

### 06 - DSP method benchmark

- Kiểm định khối tạo I/Q và độ nghiêng biên độ trong dải 2300-3800 Hz.
- Kết quả chính:
  - Hybrid: sai số pha cực đại khoảng **0.0024 độ**, tỉ số Q/I từ **0.9139 đến 1.0000**, độ nghiêng biên độ thấp nhất **0.6582**.
  - FIR Hilbert 65 tap: sai số pha cực đại khoảng **0.0024 độ**, Q/I từ **0.9989 đến 1.0016**, độ nghiêng thấp nhất **0.6854**.
  - Square-Law không có khối I/Q nên pha và Q/I được để trống đúng về mặt ý nghĩa; độ nghiêng thấp nhất **0.4861**.
  - Hilbert FFT offline được giữ làm đối chứng, không phải phương án triển khai MCU.
- Xuất `06_iq_phase_amplitude.csv`, các bảng phổ envelope và hình so sánh waveform/spectrum/ablation.
- Kết luận kỹ thuật: Hybrid là phép vi phân gần cầu phương quanh tần số trung tâm, không phải Hilbert lý tưởng; FIR Hilbert cho Q/I phẳng hơn trong toàn dải nhưng cần chi phí FIR 65 tap.

## 4. Tính nhất quán xuyên suốt

- Các notebook 03-06 đọc bandpass và lowpass từ artifact/config thay vì gõ lại độc lập.
- MLP và CNN dùng cùng target fs, warmup, file ID và quy tắc chia cửa sổ.
- Ba phương pháp envelope đi qua các đường xử lý DSP chuẩn hóa tương ứng; Hilbert FFT chỉ dùng offline để đối chứng.
- Nhánh Square-Law giữ nguyên đại lượng `LP(x^2)`, còn Hilbert/Hybrid dùng biên độ envelope. Vì vậy giai đoạn 2 phải chuẩn hóa đầu vào riêng theo phương pháp, không dùng chung một hằng số scale.

## 5. Vấn đề và việc cần làm trước Giai đoạn 2

1. **Xác minh fs suy luận của 4 file Normal.** Đây là việc ưu tiên cao nhất; nếu fs sai, toàn bộ tần số vật lý và đặc trưng của các file này có thể bị lệch.
2. **Quyết định cách xử lý RPM thiếu ở 2 file Normal.** Hiện pipeline dùng RPM danh định theo tải trong config; cần ghi rõ đây là giả định trong báo cáo và kiểm tra độ nhạy nếu cần.
3. **Chuẩn hóa metadata notebook.** Notebook 03 thiếu `id` ở một cell; notebook 04 thiếu `id` ở phần lớn cell; notebook 06 thiếu `id` ở nhiều cell. Notebook 02 có output nhưng các code cell chưa có `execution_count`. Nên chạy lại và lưu notebook theo một chuẩn thống nhất để tăng khả năng tái lập.
4. **Tái chạy toàn bộ pipeline sau khi xác minh fs.** Cần cập nhật các artifact phụ thuộc fs: `bandpass_config.json`, feature tables, CNN windows và benchmark nếu metadata thay đổi.
5. **Không xem các hình ablation là bằng chứng định lượng cuối cùng.** Chúng hỗ trợ kiểm tra trực quan; đánh giá cuối cần thực hiện ở giai đoạn 2 bằng split theo file/LOLO và metric đã định trước.

## 6. Danh mục artifact chính

- Manifest: `outputs/tables/manifest.csv`, `manifest_filtered.csv`, `manifest_clean.csv`, `01_warnings.csv`.
- DSP config: `outputs/tables/bandpass_config.json`, `windows_cnn_config.json`.
- MLP: `features_mlp*.parquet` và `features_mlp*.csv`.
- CNN: `windows_cnn_raw.parquet`, `windows_cnn_env*.parquet`, các file `*_meta.csv` và `*_sample.csv`.
- DSP benchmark: `06_iq_phase_amplitude.csv`, `06_env_spectrum_*.csv` và các hình trong `outputs/figures/`.

## 7. Trạng thái kết thúc

Giai đoạn 1 đã hoàn thành phần lớn pipeline tiền xử lý và đã tạo đủ artifact để bắt đầu huấn luyện/đánh giá ở Giai đoạn 2. Tuy nhiên, cần hoàn tất xác minh 4 fs suy luận, ghi rõ giả định RPM của Normal và chuẩn hóa metadata notebook trước khi chốt kết quả như một pipeline tái lập hoàn toàn.
