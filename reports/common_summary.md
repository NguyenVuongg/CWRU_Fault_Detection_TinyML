# Tóm tắt thư viện `common/`

## 1. Vai trò tổng thể

Thư mục `common/` là lớp thư viện dùng chung của toàn bộ pipeline chẩn đoán hư hỏng vòng bi CWRU. Nó tập trung các quy ước khoa học và kỹ thuật để bảo đảm:

- cấu hình thí nghiệm có một nguồn duy nhất;
- sampling rate, phạm vi dữ liệu và nhãn được kiểm tra trước khi xử lý;
- DSP được so sánh trên cùng giao thức nhân quả;
- đặc trưng có schema cố định và không đưa metadata vào mô hình;
- chia dữ liệu theo file và LOLO để hạn chế rò rỉ giữa các cửa sổ chồng lấn;
- mô hình, lượng tử hóa và chi phí MCU có thể tái lập.

Luồng phụ thuộc chính:

```text
config
  -> io_utils -> pipeline -> dsp/order -> features_full -> splitting/training
  -> models -> quantization -> mcu_export
```

`features.py` và `features_dynamic.py` cung cấp các đường trích đặc trưng đơn giản hoặc động cho các mục khảo sát/chuyên biệt. `synthetic.py` tạo dữ liệu giả và ca kiểm thử biên.

## 2. Tóm tắt theo module

### 2.1 `common/config.py` — cấu hình và cơ sở vật lý

**Ý nghĩa:** Nguồn chân lý cho hình học vòng bi SKF 6205, phạm vi dữ liệu, sơ đồ lớp, RPM danh định, tần số lấy mẫu, dải cộng hưởng, bộ lọc envelope và kích thước cửa sổ.

**Hàm:**

- `_validate_rpm(rpm, caller)`: kiểm tra RPM dương, hữu hạn trước khi tính tần số/order.
- `bearing_fault_frequencies(rpm, geometry)`: tính `f_rot`, BPFO, BPFI, BSF và FTF theo hình học vòng bi.
- `hz_to_order(freq_hz, rpm)`: đổi tần số Hz sang order theo tần số quay.
- `make_class_label(label, fault_diameter_mils)`: ánh xạ nhãn gốc và đường kính lỗi thành 10 lớp chính, ví dụ `IR_007`.

**Hằng số quan trọng:** `SCOPE`, `CLASSES_10`, `CLASSES_4`, `NOMINAL_RPM_BY_LOAD`, `CANDIDATE_SAMPLING_RATES_HZ`, `RESONANCE_BAND_HZ`, `LP_CUTOFF_HZ`, `WINDOW_SIZE_RAW`, `WINDOW_SIZE_ENV`, `WARMUP_SAMPLES`.

**Nguyên tắc:** Không dùng hình học SKF cho các đường kính NTN 28/40 mils; không suy đoán RPM hoặc sampling rate khi dữ liệu không đủ căn cứ.

### 2.2 `common/io_utils.py` — đọc dữ liệu và lập manifest

**Ý nghĩa:** Chuyển dữ liệu `.mat` thô thành manifest có metadata, kiểm tra tính hợp lệ và nạp tín hiệu DE với sampling rate đúng.

**Hàm:**

- `parse_metadata_from_filename(filepath)`: phân tích nhãn, đường kính, vị trí OR, tải, nguồn dữ liệu, sensor và sampling rate khai báo từ đường dẫn/tên file.
- `inspect_mat_file(filepath)`: đọc cấu trúc `.mat`, lấy số mẫu các kênh và RPM.
- `load_de_signal(filepath)`: nạp tín hiệu `_DE_time` từ `.mat` hoặc `.npy`.
- `load_de_signal_resampled(filepath, source_fs_hz, target_fs_hz)`: nạp tín hiệu DE và resample bằng `resample_poly` về tần số chung.
- `_resolve_sampling_rate_hz(...)`: suy ra sampling rate thực từ số mẫu và thời lượng kỳ vọng; trả `None` nếu không đủ căn cứ.
- `run_sanity_checks(df)`: thêm cảnh báo về thời lượng/sampling rate, RPM, vòng bi NTN, vị trí OR, nhãn, tải, sensor và tần số khai báo.
- `apply_scope_filter(manifest, exclude_warning_keywords)`: loại các file ngoài phạm vi nghiên cứu hoặc không thể sử dụng.
- `build_manifest(data_root)`: quét `.mat`, ghép metadata và kết quả kiểm tra thành manifest hoàn chỉnh.

**Đầu ra chính:** manifest có `resolved_sample_rate_hz`, `warnings`, `has_warning` và các metadata cần cho các bước sau.

### 2.3 `common/pipeline.py` — điều phối và bảo vệ contract

**Ý nghĩa:** Tách việc đọc/cache manifest khỏi orchestration preprocessing; ngăn cache cũ, manifest sai schema và feature table sai chiều đi tiếp.

**Hàm:**

- `_validate_manifest_contract(manifest, stage)`: kiểm tra kiểu dữ liệu, schema và điều kiện theo stage `sanity` hoặc `scope`.
- `_validate_cache_schema(manifest)`: kiểm tra cache manifest có đủ cột tối thiểu.
- `get_manifest(...)`: đọc cache hợp lệ hoặc rebuild manifest từ dữ liệu thật/dữ liệu giả.
- `pick_file(...)`: chọn file đại diện theo nhãn, tải, đường kính và vị trí OR.
- `_assert_scope_filtered(manifest)`: xác nhận manifest đã lọc đúng phạm vi và không còn cảnh báo loại trừ.
- `_assert_feature_contract(feature_df)`: xác nhận metadata và số feature khớp contract của `features_full.py`.
- `run_preprocessing_pipeline(...)`: orchestration chính: manifest -> scope filter -> đọc/resample -> trích đặc trưng -> kiểm tra contract.

**Nguyên tắc:** Raw manifest có thể còn dòng ngoài phạm vi; điều kiện sampling rate không thiếu chỉ được khóa chặt sau `apply_scope_filter()`.

### 2.4 `common/dsp.py` — DSP envelope và FFT

**Ý nghĩa:** Cài đặt ba phương pháp giải điều chế có thể so sánh công bằng: Square-Law, Hilbert FIR nhân quả và Hybrid; đồng thời cung cấp lọc, giảm mẫu và FFT.

**Lưu ý về kiến trúc Hybrid:** kiến trúc lai ghép hiện tại mới chỉ cài đặt **bậc (a)** là khối sai phân trung tâm nhân quả kết hợp với Alpha-Max Plus Beta-Min. **Bậc (b) - IIR All-Pass - chưa được cài đặt.** Với tần số trung tâm góc `\(\omega_c\)`, logic cốt lõi của bậc (a) là:

$$
I[n] = x[n-1], \qquad Q[n] = \frac{x[n] - x[n-2]}{2\sin(\omega_c)}.
$$

Sau đó biên độ được xấp xỉ bằng Alpha-Max Plus Beta-Min, không dùng FFT hoặc căn bậc hai. Vì vậy không được mô tả Hybrid hiện tại như một kiến trúc đã bao gồm IIR All-Pass.

**Hàm:**

- `_resolve_band(band)`: lấy dải cộng hưởng mặc định hoặc kiểm tra dải truyền vào.
- `envelope_mcu_cost(method)`: trả chi phí ước lượng của khối envelope trên MCU.
- `bandpass_filter(...)`: Butterworth bandpass; mặc định nhân quả bằng `sosfilt`.
- `lowpass_filter(...)`: Butterworth lowpass; kiểm tra cutoff dưới Nyquist.
- `square_law_envelope(...)`: bình phương tín hiệu sau bandpass rồi lowpass; có tùy chọn lấy căn để đồng nhất đơn vị.
- `hilbert_envelope(...)`: envelope Hilbert offline bằng FFT/zero-phase, chỉ làm tham chiếu.
- `design_fir_hilbert(numtaps)`: thiết kế FIR Hilbert Type III, số tap phải lẻ.
- `hilbert_envelope_fir(...)`: baseline Hilbert nhân quả triển khai được trên MCU.
- `hybrid_envelope(...)`: tạo I/Q bằng sai phân trung tâm nhân quả và Alpha-Max Plus Beta-Min, không FFT/không sqrt.
- `envelope_group_delay(method)`: trả trễ nhóm của khối giải điều chế.
- `decimate_envelope(env, fs, factor, lp_cutoff)`: giảm mẫu envelope và kiểm tra chống chồng phổ.
- `envelope_by_method(...)`: dispatcher duy nhất, ép các phương pháp dùng chung cấu hình benchmark.
- `compute_fft(x, fs, apply_window)`: tính phổ biên độ một phía bằng real FFT.

**Nguyên tắc:** `hilbert_envelope()` không được đưa vào benchmark MCU; benchmark dùng `hilbert_envelope_fir()`. Các phương pháp phải dùng cùng bandpass, lowpass, cutoff và cách khử DC.

### 2.5 `common/order.py` — đặc trưng phổ order

**Ý nghĩa:** Đo biên độ quanh các tần số động học và các hài của chúng, đồng thời xử lý giới hạn độ phân giải FFT.

**Hàm:**

- `amplitude_near_frequencies(...)`: lấy biên độ lớn nhất trong các cửa sổ quanh nhiều tần số mục tiêu.
- `amplitude_near_frequency(...)`: phiên bản một tần số.
- `target_orders(target_names, harmonics, geometry)`: tạo danh sách tần số mục tiêu theo order, độc lập RPM.
- `_plan(...)`: lập kế hoạch nhóm các vạch phổ không phân giải được.
- `plan_harmonic_groups(...)`: cố định các nhóm gộp theo độ phân giải, hình học và RPM tham chiếu.
- `_group_key(prefix, group, decl_index)`: tạo tên cột đặc trưng ổn định.
- `extract_order_features(...)`: trích đặc trưng Nhóm B hoặc Nhóm C từ phổ.

**Ý nghĩa phương pháp:** Với cửa sổ 2048 mẫu tại 12 kHz, độ phân giải là khoảng 5.86 Hz/bin; vì vậy một số hài được gộp cố định. Đây là lý do số feature thực tế là 28 thay vì 32 và tránh mã hóa tải vào LOLO.

### 2.6 `common/features.py` — đặc trưng đơn giản

**Ý nghĩa:** API tối giản cho khảo sát ban đầu, không phải pipeline feature đầy đủ chính thức.

**Hàm:**

- `make_sliding_windows(...)`: cắt tín hiệu thành cửa sổ chồng lấn, giữ `file_id`, `window_idx` và `start_idx`.
- `extract_simple_features(window)`: tính RMS, kurtosis, skewness, peak và độ lệch chuẩn.
- `build_feature_table(windows_df)`: áp dụng đặc trưng đơn giản cho toàn bộ bảng cửa sổ.

**Lưu ý:** Khi đánh giá mô hình chính phải dùng `features_full.py`, vì module này chỉ phục vụ minh họa/khảo sát.

### 2.7 `common/features_full.py` — bảng đặc trưng chính thức

**Ý nghĩa:** Tạo vector 28 đặc trưng gồm miền thời gian, phổ order và phổ envelope, kèm metadata chống leakage.

**Hàm:**

- `make_file_id(file_path)`: tạo định danh file gốc, không chứa chỉ số cửa sổ.
- `extract_time_domain_features(x, prefix)`: tính 11 đặc trưng thống kê miền thời gian.
- `extract_order_domain_features(...)`: FFT tín hiệu và gọi `order.extract_order_features()`.
- `extract_envelope_features_from_envelope(...)`: tính đặc trưng envelope từ envelope đã tính sẵn.
- `extract_envelope_features(...)`: tự tạo envelope rồi trích đặc trưng; dùng cho gọi độc lập/test.
- `extract_full_feature_vector(...)`: ghép ba nhóm đặc trưng thành một vector có prefix.
- `build_full_feature_table(...)`: đọc/resample từng file, tính envelope một lần cho toàn file, cắt cửa sổ đồng bộ và tạo bảng feature.
- `feature_columns(feature_df, expected_count, strict_prefix)`: lấy đúng các cột đặc trưng, loại metadata và kiểm tra số chiều/prefix.

**Contract:** metadata gồm `file_id`, `window_idx`, `start_idx`, nhãn, tải và đường kính; không được đưa vào `X`. Số chiều thực tế: 11 time + 10 order + 7 envelope = 28.

### 2.8 `common/features_dynamic.py` — đặc trưng động theo phương pháp DSP

**Ý nghĩa:** Tạo các bảng đặc trưng để so sánh Square-Law, Hilbert FIR và Hybrid trong RQ3.

**Hàm:**

- `apply_envelope_method(...)`: ủy quyền cho `dsp.envelope_by_method()`.
- `extract_features_dynamic_from_signals(...)`: nhận tín hiệu nguyên, tính envelope một lần trên toàn file, sau đó cắt cửa sổ đồng bộ raw/envelope.
- `extract_features_dynamic(...)`: API cũ nhận cửa sổ đã cắt; được giữ để tương thích nhưng deprecated vì mỗi cửa sổ khởi tạo lại trạng thái lọc và tạo transient giả.

**Nguyên tắc:** RQ3 phải đồng nhất đơn vị envelope; với Square-Law cần dùng `envelope_kwargs={"take_sqrt": True}` khi so sánh với Hilbert/Hybrid.

### 2.9 `common/splitting.py` — chia dữ liệu và thiết kế đánh giá

**Ý nghĩa:** Bảo vệ đánh giá khỏi leakage do cửa sổ chồng lấn và chuẩn hóa cấu trúc LOLO.

**Hàm:**

- `resolve_label_col(feature_df, label_col, scheme)`: chọn cột nhãn 10 lớp chính hoặc 4 lớp bổ trợ.
- `random_window_split(...)`: chia ngẫu nhiên theo dòng; giữ làm baseline minh họa, không phải phương pháp đúng.
- `file_based_split(...)`: chia theo file, stratify theo nhãn, bảo đảm file không xuất hiện ở cả train và test.
- `run_split_comparison_experiment(...)`: so sánh thực nghiệm Random Window Split và File-based Split bằng Random Forest.
- `generate_lolo_folds(loads)`: sinh bốn fold Leave-One-Load-Out.

### 2.10 `common/training.py` — đánh giá LOLO

**Ý nghĩa:** Orchestrator huấn luyện/đánh giá estimator qua các fold LOLO, dùng lại quy tắc chia trong `splitting.py`.

**Hàm:**

- `iterate_lolo_splits(...)`: generator trả về thông tin fold cùng train/validation/test DataFrame.
- `run_lolo_evaluation(...)`: fit estimator mới trên từng fold và trả metric từng fold cùng summary.
- `format_summary(summary, metric)`: định dạng kết quả theo `mean ± std`.

**Metric chính:** accuracy và macro-F1; macro-F1 cần thiết vì sơ đồ 10 lớp có thể làm một lớp bị bỏ rơi dù accuracy tổng vẫn cao.

### 2.11 `common/models.py` — kiến trúc mô hình

**Ý nghĩa:** Định nghĩa MLP cho bảng feature và hai CNN 1D đối sánh raw/envelope.

**Hàm:**

- `build_mlp(...)`: tạo MLP nhẹ, tùy chọn tích hợp lớp Normalization.
- `build_cnn1d(...)`: nguồn duy nhất tạo block Conv1D/BatchNorm/Pooling dùng chung.
- `build_cnn1d_raw(...)`: CNN cho tín hiệu raw.
- `build_cnn1d_env(...)`: CNN cho envelope đã giảm mẫu.
- `_comparable_block_configs(...)`: lấy cấu hình các block, bỏ khác biệt tên/input.
- `cnn_block_differences(model_a, model_b)`: liệt kê khác biệt kiến trúc giữa hai nhánh.
- `assert_identical_cnn_blocks(model_a, model_b)`: chặn so sánh không công bằng nếu hai nhánh khác block.
- `make_sparse_macro_f1(n_classes, name)`: tạo macro-F1 cho nhãn sparse nguyên.
- `compile_classifier(model, learning_rate, n_classes)`: compile chuẩn với sparse cross-entropy, accuracy và macro-F1.
- `count_params(model)`: đếm trainable, non-trainable và tổng số tham số.

### 2.12 `common/quantization.py` — lượng tử hóa INT8

**Ý nghĩa:** Chuyển model Float32 sang TFLite full-integer INT8 và đo ảnh hưởng đến chất lượng.

**Hàm:**

- `make_representative_dataset_fn(X_train, n_samples, seed)`: tạo calibration dataset chỉ từ train của fold.
- `quantize_model_int8(...)`: lượng tử hóa post-training, ép input/output INT8.
- `model_bytes_to_file(...)`: ghi bytes TFLite ra file.
- `get_tflite_size(...)`: trả kích thước model theo bytes/KB.
- `get_quantization_info(...)`: đọc dtype, scale và zero-point input/output.
- `evaluate_tflite_model(...)`: lượng giá model INT8 trên test float32 sau khi quantize input đúng cách.
- `compare_float_vs_int8(...)`: so sánh accuracy và macro-F1 giữa Keras Float32 và TFLite INT8.
- `diagnose_input_dynamic_range(...)`: đo chênh lệch dải động và số mức INT8 hữu hiệu để quyết định có cần chuẩn hóa đầu vào.

**Nguyên tắc:** Representative dataset không được lấy từ validation/test; input quantization phải clip trước khi cast để tránh wrap-around.

**Đính chính về per-channel scaling:** TFLite không hỗ trợ per-channel cho input tensor, do đó hàm `diagnose_input_dynamic_range` được dùng để cảnh báo và yêu cầu chuẩn hóa Z-score từ trước.

### 2.13 `common/mcu_export.py` — xuất và tổng hợp chỉ số MCU

**Ý nghĩa:** Chuẩn bị model/dữ liệu cho STM32Cube.AI hoặc ST Edge AI và hợp nhất chi phí tĩnh/động.

**Hàm:**

- `export_tflite_for_mcu(...)`: ghi model `.tflite` ra file triển khai.
- `save_signal_sample_csv(...)`: lưu mẫu tín hiệu/feature thật làm input kiểm thử trên công cụ ST.
- `_parse_int(s)`: đổi chuỗi số có dấu phẩy thành integer.
- `parse_stedgeai_cli_summary(summary_text)`: parse MACC, Flash weights, RAM activations và I/O từ output CLI.
- `estimate_energy_mj(latency_ms, power_mw)`: tính năng lượng theo `E = P × t`, trả cả µJ và mJ.
- `latency_from_dwt_cycles(cycles, clock_mhz)`: đổi số chu kỳ DWT/CYCCNT thành millisecond.
- `summarize_mcu_cost(...)`: gộp model size, chi phí STM32Cube.AI, latency và energy thành một dòng báo cáo.

**Nguyên tắc:** MACC là chỉ số tĩnh; latency phải đo trên MCU; energy chỉ so sánh được khi board, điện áp và clock giống nhau.

**Yêu cầu bắt buộc:** Phải ghi rõ nguồn gốc của $P$ (công suất) khi tính Năng lượng, và báo cáo MACC của FIR Hilbert ở 2 chế độ (có/không đối xứng).

### 2.14 `common/synthetic.py` — dữ liệu giả và kiểm thử biên

**Ý nghĩa:** Tạo dữ liệu mô phỏng có tính tái lập để kiểm tra pipeline mà không thay thế dữ liệu CWRU thật.

**Hàm:**

- `stable_seed(text, modulo)`: tạo seed ổn định bằng MD5 giữa các phiên chạy.
- `make_synthetic_signal(...)`: sinh tín hiệu rung gồm mất cân bằng, nhiễu và xung lỗi điều biên bởi cộng hưởng.
- `build_synthetic_dataset(...)`: tạo cây thư mục và các file `.mat` giả tương thích parser thật.
- `build_edge_case_dataset(...)`: tạo các file cố ý chứa lỗi sampling rate, vị trí OR, NTN, RPM, sensor và source frequency để kiểm chứng sanity checks.

**Giới hạn:** Dữ liệu giả chỉ dùng smoke test/unit test và minh họa; không dùng để kết luận hiệu năng trên CWRU.

### 2.15 `common/__init__.py`

Module khởi tạo package `common`. File này không chứa thuật toán chính; mục đích là cho phép import các module dùng chung theo cấu trúc package Python.

## 3. Các nguyên tắc thiết kế

1. **Sampling rate:** `resolved_sample_rate_hz` là sampling rate thực được xác định từ dữ liệu; không thay bằng `declared_sample_rate_khz` hoặc giá trị đoán.
2. **Phạm vi dữ liệu:** raw manifest được kiểm tra trước, sau đó `apply_scope_filter()` loại dữ liệu ngoài phạm vi trước khi trích đặc trưng.
3. **DSP công bằng:** các phương pháp envelope dùng chung bandpass, lowpass, cutoff và quy tắc khử DC; Hilbert FFT chỉ là tham chiếu offline.
4. **Không leakage:** `file_id` là file gốc; `window_idx` và `start_idx` chỉ là metadata; chia train/test phải theo file.
5. **Số chiều feature:** 28 là số chiều thực tế sau khi gộp các hài không phân giải được ở cửa sổ 2048 mẫu @12 kHz.
6. **Đánh giá:** LOLO giữ một tải làm test; báo cáo cả accuracy và macro-F1.
7. **TinyML:** calibration INT8 dùng train fold; latency/energy phải dựa trên đo MCU, không thay bằng MACC ước lượng.

## 4. Tổng kết

`common/` không chỉ là tập hàm tiện ích mà là lớp kiểm soát tính hợp lệ của nghiên cứu. `config.py`, `io_utils.py`, `pipeline.py`, `features_full.py`, `order.py` và `splitting.py` tạo thành lõi chống sai lệch dữ liệu; `dsp.py` và `features_dynamic.py` định nghĩa giao thức so sánh envelope; `models.py`, `quantization.py`, `training.py` và `mcu_export.py` nối kết quả khoa học với triển khai TinyML; `synthetic.py` hỗ trợ kiểm thử có tái lập.
