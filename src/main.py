from fileinput import filename
import os
import re
import pickle
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.dataset import CWRUDataPipeline, split_files

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_dataset_config_automatically():
    """
    Quét thư mục data/raw/.
    Tự động gán Label dựa trên tên thư mục chứa file (Normal, IR, B, OR).
    Trích xuất HP từ đuôi tên file (VD: 118_0.mat -> 0 HP).
    """
    file_list = []
    
    for root, dirs, files in os.walk(config.RAW_DATA_DIR):
        for filename in files:
            if filename.endswith('.mat'):
                filepath = os.path.join(root, filename)
                
                # Cắt nhỏ đường dẫn thành các thư mục (VD: ['12k_Drive_End', 'B', '007'])
                # Dùng cách này sẽ bắt chính xác 100% thư mục tên là 'B'
                path_parts = [p.upper() for p in os.path.normpath(root).split(os.sep)]
                
                # 1. Xác định Label
                label = -1
                if 'NORMAL' in path_parts:
                    label = 0
                elif 'IR' in path_parts:
                    label = 1
                elif 'OR' in path_parts:
                    label = 2  # Trong config: 2 là Outer_Race
                elif 'B' in path_parts:
                    label = 3  # Trong config: 3 là Ball
                
                # 2. Xác định Load HP
                hp = -1
                # Ưu tiên 1: Bắt số sau dấu gạch dưới ở đuôi file (Như trong ảnh của bạn)
                hp_match = re.search(r"_(\d+)\.mat$", filename)
                if hp_match:
                    hp = int(hp_match.group(1))
                elif label == 0:
                    # Ưu tiên 2 (Dự phòng): Chỉ dùng cho file Normal gốc không có đuôi _0
                    if '97' in filename: hp = 0
                    elif '98' in filename: hp = 1
                    elif '99' in filename: hp = 2
                    elif '100' in filename: hp = 3
                        
                if label != -1 and hp != -1:
                    file_list.append((filepath, label, hp))
                else:
                    logging.warning(f"Bỏ qua: {filepath} (Label: {label}, HP: {hp})")

    return file_list


def build_dataframe_for_files(pipeline: CWRUDataPipeline, file_list) -> pd.DataFrame:
    """Xử lý danh sách file thành 1 DataFrame gộp (mỗi dòng = 1 window)."""
    dfs = []
    for file_path, label, hp in file_list:
        df = pipeline.process_single_file(file_path, label, hp)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    logging.info("Khởi động pipeline trích xuất đặc trưng...")
    pipeline = CWRUDataPipeline()

    # ── Bước 1: Quét file, gán nhãn ──────────────────────────
    file_list = get_dataset_config_automatically()
    if not file_list:
        logging.error("Không tìm thấy file nào đúng định dạng trong data/raw/!")
        return
    logging.info(f"Tìm thấy {len(file_list)} file hợp lệ.")

    # ── Bước 2: Chia TRAIN/VAL/TEST THEO FILE (trước khi xử lý) ──
    # Đây là điểm sửa quan trọng nhất: quyết định file nào vào tập
    # nào PHẢI làm trước khi cắt window, để không có window nào của
    # cùng 1 file bị rải vào 2 tập khác nhau.
    train_files, val_files, test_files = split_files(file_list)
    logging.info(f"Chia file: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # ── Bước 3: Xử lý từng tập thành feature DataFrame ────────
    logging.info("Đang xử lý tập TRAIN...")
    df_train = build_dataframe_for_files(pipeline, train_files)
    logging.info("Đang xử lý tập VAL...")
    df_val = build_dataframe_for_files(pipeline, val_files)
    
    logging.info("Đang xử lý tập TEST...")
    df_test = build_dataframe_for_files(pipeline, test_files)

    # =====================================================================
    # --- FIX BUG: PHÂN BỔ LẠI DỮ LIỆU BỊ THIẾU TỪ TRAIN SANG VAL & TEST ---
    # Lý do: Các nhóm có < 3 file (như Normal) bị đẩy toàn bộ vào Train. 
    # Ta sẽ cắt trực tiếp các dòng (window) từ DataFrame Train sang Val và Test.
    # =====================================================================
    # Tìm các Label có trong Train nhưng bị thiếu trong Val
    missing_labels_in_val = set(df_train['Label'].unique()) - set(df_val['Label'].unique())
    
    for lbl in list(missing_labels_in_val):
        # Lấy toàn bộ data của nhãn bị thiếu ra khỏi Train
        lbl_data = df_train[df_train['Label'] == lbl]
        n_total = len(lbl_data)
        
        # Tính toán số lượng cần chia (Ví dụ: Val 15%, Test 15%)
        n_val = int(n_total * config.VAL_RATIO)
        n_test = int(n_total * config.TEST_RATIO)
        
        # Cắt data thành 3 phần
        val_add = lbl_data.iloc[:n_val]
        test_add = lbl_data.iloc[n_val:n_val+n_test]
        train_keep = lbl_data.iloc[n_val+n_test:]
        
        # Cập nhật lại các DataFrame
        df_train = df_train[df_train['Label'] != lbl] # Xóa nhãn này khỏi train hiện tại
        
        # Nối lại data đã chia vào đúng các tập
        df_train = pd.concat([df_train, train_keep], ignore_index=True)
        df_val = pd.concat([df_val, val_add], ignore_index=True)
        df_test = pd.concat([df_test, test_add], ignore_index=True)

    if df_train.empty:
        logging.error("Không có dữ liệu train nào được xử lý.")
        return

    # Xáo trộn thứ tự dòng trong train (không ảnh hưởng đến việc
    # file nào thuộc tập nào, chỉ xáo trộn thứ tự window để training
    # ổn định hơn)
    df_train = df_train.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # ── Bước 4: Chuẩn hóa feature — FIT trên TRAIN, transform VAL/TEST ──
    feature_cols = [c for c in df_train.columns if c not in ['Label', 'Load']]

    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    df_train.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "train_features.csv"), index=False)
    df_val.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "val_features.csv"), index=False)
    df_test.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "test_features.csv"), index=False)

    with open(os.path.join(config.PROCESSED_DATA_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # ── Bước 5: In báo cáo tóm tắt — kiểm tra nhanh chất lượng dataset ──
    logging.info("HOÀN TẤT!")
    print("\n" + "=" * 60)
    print("TÓM TẮT DATASET")
    print("=" * 60)
    for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        print(f"\n{name}: {len(df)} windows")
        print(df['Label'].map(config.LABEL_NAMES).value_counts().to_string())

    print(f"\nĐã lưu:")
    print(f"  {config.PROCESSED_DATA_DIR}/train_features.csv")
    print(f"  {config.PROCESSED_DATA_DIR}/val_features.csv")
    print(f"  {config.PROCESSED_DATA_DIR}/test_features.csv")
    print(f"  {config.PROCESSED_DATA_DIR}/scaler.pkl")


if __name__ == "__main__":
    main()