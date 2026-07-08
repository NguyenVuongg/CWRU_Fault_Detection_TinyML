import os
import re
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, cast

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping

from src import config
from src.dataset import CWRUDataPipeline, split_files_random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_dataset_files() -> List[Tuple[str, int, int]]:
    file_list: List[Tuple[str, int, int]] = []
    for root, _, files in os.walk(config.RAW_DATA_DIR):
        for filename in files:
            if filename.endswith('.mat'):
                filepath = os.path.join(root, filename)
                path_parts = [p.upper() for p in os.path.normpath(root).split(os.sep)]
                
                label = -1
                if 'NORMAL' in path_parts: label = 0
                elif 'IR' in path_parts: label = 1
                elif 'OR' in path_parts: label = 2
                elif 'B' in path_parts: label = 3
                
                hp = -1
                hp_match = re.search(r"_(\d+)\.mat$", filename)
                if hp_match: hp = int(hp_match.group(1))
                elif label == 0:
                    if '97' in filename: hp = 0
                    elif '98' in filename: hp = 1
                    elif '99' in filename: hp = 2
                    elif '100' in filename: hp = 3
                        
                if label != -1 and hp != -1:
                    file_list.append((filepath, label, hp))
    return file_list


def extract_or_load_dataset(file_list: List[Tuple[str, int, int]], demod_type: str = 'hilbert', force_rebuild: bool = False) -> pd.DataFrame:
    cache_path = os.path.join(config.PROCESSED_DATA_DIR, f"full_dataset_{demod_type}.csv")
    if not force_rebuild and os.path.exists(cache_path):
        logging.info(f"[{demod_type.upper()}] Đang tải dataset từ cache: {cache_path}")
        return pd.read_csv(cache_path)
    
    logging.info(f"[{demod_type.upper()}] Đang chạy DSP Pipeline trích xuất đặc trưng từ file .mat...")
    pipeline = CWRUDataPipeline(demod_type=demod_type)
    dfs: List[pd.DataFrame] = []
    for filepath, label, hp in file_list:
        df = pipeline.process_single_file(filepath, label, hp)
        if not df.empty:
            df['Filepath'] = filepath
            dfs.append(df)
            
    full_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not full_df.empty:
        full_df.to_csv(cache_path, index=False)
    return full_df


def build_mlp_model(input_dim: int, num_classes: int = 4) -> Sequential:
    """Kiến trúc MLP chuẩn Keras 3 (Dùng Input layer riêng biệt)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_eval_mlp(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, run_name: str) -> Tuple[float, float, Sequential]:
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=config.RANDOM_SEED, stratify=y_train)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    model = build_mlp_model(X_tr_scaled.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    
    model.fit(X_tr_scaled, y_tr, validation_data=(X_val_scaled, y_val),
              epochs=100, batch_size=32, callbacks=[es], verbose="0")
    
    y_pred = np.argmax(model.predict(X_test_scaled, verbose="0"), axis=1)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average='macro'))
    
    logging.info(f"[{run_name:25}] -> Acc: {acc:.4f} | F1: {f1:.4f}")
    return acc, f1, model


def main() -> None:
    file_list = get_dataset_files()
    if not file_list:
        logging.error("Không tìm thấy file nào trong data/raw/!")
        return
        
    logging.info("="*70)
    logging.info("BẮT ĐẦU LUỒNG THỰC NGHIỆM NCKH (16 RUNS MLP + RFE)")
    logging.info("="*70)
    
    df_hilbert = extract_or_load_dataset(file_list, demod_type='hilbert')
    df_sqlaw   = extract_or_load_dataset(file_list, demod_type='square_law')
    
    feature_cols_all = [c for c in df_hilbert.columns if c not in ['Label', 'Load', 'Filepath']]
    results: List[Dict[str, Any]] = []

    # GIAI ĐOẠN A
    logging.info("\n--- [GIAI ĐOẠN A] KHẢO SÁT BASELINE & VẠCH TRẦN LEAKAGE ---")
    tr_df, ts_df = train_test_split(df_hilbert, test_size=0.2, random_state=config.RANDOM_SEED, stratify=df_hilbert['Label'])
    acc, f1, _ = train_and_eval_mlp(tr_df[feature_cols_all], tr_df['Label'], ts_df[feature_cols_all], ts_df['Label'], "A1. Window-Random Split")
    results.append({'Stage': 'A', 'Name': 'A1. Window-Random Split', 'Features': len(feature_cols_all), 'Acc': acc, 'F1': f1})

    tr_files, _, ts_files = split_files_random(file_list)
    tr_filepaths, ts_filepaths = [f[0] for f in tr_files], [f[0] for f in ts_files]
    tr_df = df_hilbert[df_hilbert['Filepath'].isin(tr_filepaths)]
    ts_df = df_hilbert[df_hilbert['Filepath'].isin(ts_filepaths)]
    acc, f1, _ = train_and_eval_mlp(tr_df[feature_cols_all], tr_df['Label'], ts_df[feature_cols_all], ts_df['Label'], "A2. File-Random Split")
    results.append({'Stage': 'A', 'Name': 'A2. File-Random Split', 'Features': len(feature_cols_all), 'Acc': acc, 'F1': f1})

    a3_accs: List[float] = []
    for test_hp in [0, 1, 2, 3]:
        tr_df = df_hilbert[df_hilbert['Load'] != test_hp]
        ts_df = df_hilbert[df_hilbert['Load'] == test_hp]
        acc, f1, _ = train_and_eval_mlp(tr_df[feature_cols_all], tr_df['Label'], ts_df[feature_cols_all], ts_df['Label'], f"A3/B1. Hilbert LOLO (HP={test_hp})")
        a3_accs.append(acc)
    results.append({'Stage': 'A/B', 'Name': 'A3/B1. Hilbert LOLO Avg', 'Features': len(feature_cols_all), 'Acc': float(np.mean(a3_accs)), 'F1': float(np.mean(a3_accs))})

    # GIAI ĐOẠN B
    logging.info("\n--- [GIAI ĐOẠN B] ĐỐI CHỨNG DEMODULATION (SQUARE-LAW LOLO) ---")
    b2_accs: List[float] = []
    for test_hp in [0, 1, 2, 3]:
        tr_df = df_sqlaw[df_sqlaw['Load'] != test_hp]
        ts_df = df_sqlaw[df_sqlaw['Load'] == test_hp]
        acc, f1, _ = train_and_eval_mlp(tr_df[feature_cols_all], tr_df['Label'], ts_df[feature_cols_all], ts_df['Label'], f"B2. Square-Law LOLO (HP={test_hp})")
        b2_accs.append(acc)
    results.append({'Stage': 'B', 'Name': 'B2. Square-Law LOLO Avg', 'Features': len(feature_cols_all), 'Acc': float(np.mean(b2_accs)), 'F1': float(np.mean(b2_accs))})

    # GIAI ĐOẠN C
    logging.info("\n--- [GIAI ĐOẠN C] FEATURE SELECTION (TOP-30 RFE) & JACCARD ---")
    fold_top_features: List[set] = []
    for test_hp in [0, 1, 2, 3]:
        tr_df = df_sqlaw[df_sqlaw['Load'] != test_hp]
        rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
        rf.fit(tr_df[feature_cols_all], tr_df['Label'])
        imp_series = pd.Series(rf.feature_importances_, index=feature_cols_all).sort_values(ascending=False)
        fold_top_features.append(set(imp_series.head(30).index))

    jaccard_scores: List[float] = []
    for i in range(len(fold_top_features)):
        for j in range(i + 1, len(fold_top_features)):
            inter = len(fold_top_features[i].intersection(fold_top_features[j]))
            union = len(fold_top_features[i].union(fold_top_features[j]))
            jaccard_scores.append(inter / union)
    logging.info(f"Chỉ số ổn định đặc trưng (Jaccard Similarity Index trung bình): {np.mean(jaccard_scores)*100:.2f}%")

    overall_rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
    overall_rf.fit(df_sqlaw[feature_cols_all], df_sqlaw['Label'])
    top_30_cols = list(pd.Series(overall_rf.feature_importances_, index=feature_cols_all).sort_values(ascending=False).head(30).index)

    c3_accs: List[float] = []
    for test_hp in [0, 1, 2, 3]:
        tr_df = df_sqlaw[df_sqlaw['Load'] != test_hp]
        ts_df = df_sqlaw[df_sqlaw['Load'] == test_hp]
        acc, f1, _ = train_and_eval_mlp(tr_df[top_30_cols], tr_df['Label'], ts_df[top_30_cols], ts_df['Label'], f"C3. Top-30 LOLO (HP={test_hp})")
        c3_accs.append(acc)
    results.append({'Stage': 'C/D', 'Name': 'C3/D3. Top-30 LOLO Avg', 'Features': len(top_30_cols), 'Acc': float(np.mean(c3_accs)), 'F1': float(np.mean(c3_accs))})

    # GIAI ĐOẠN D
    logging.info("\n--- [GIAI ĐOẠN D] XÁC NHẬN CẤU HÌNH CUỐI CÙNG ---")
    tr_df, ts_df = train_test_split(df_sqlaw, test_size=0.2, random_state=config.RANDOM_SEED, stratify=df_sqlaw['Label'])
    acc, f1, _ = train_and_eval_mlp(tr_df[top_30_cols], tr_df['Label'], ts_df[top_30_cols], ts_df['Label'], "D1. Window-Random (Top-30)")
    results.append({'Stage': 'D', 'Name': 'D1. Window-Random (Top-30)', 'Features': len(top_30_cols), 'Acc': acc, 'F1': f1})

    tr_df = df_sqlaw[df_sqlaw['Filepath'].isin(tr_filepaths)]
    ts_df = df_sqlaw[df_sqlaw['Filepath'].isin(ts_filepaths)]
    acc, f1, final_deploy_model = train_and_eval_mlp(tr_df[top_30_cols], tr_df['Label'], ts_df[top_30_cols], ts_df['Label'], "D2. File-Random (Top-30)")
    results.append({'Stage': 'D', 'Name': 'D2. File-Random (Top-30)', 'Features': len(top_30_cols), 'Acc': acc, 'F1': f1})

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    final_deploy_model.save(os.path.join(config.MODELS_DIR, "final_edge_model.h5"))
    with open(os.path.join(config.MODELS_DIR, "top_30_features.pkl"), 'wb') as f:
        pickle.dump(top_30_cols, f)
    logging.info(f"Đã lưu mô hình triển khai: {config.MODELS_DIR}/final_edge_model.h5")

    print("\n" + "="*80)
    print("BẢNG TỔNG KẾT THỰC NGHIỆM ĐƯA VÀO BÁO CÁO NCKH")
    print("="*80)
    res_df = pd.DataFrame(results)
    res_df['Acc'] = (res_df['Acc'] * 100).round(2).astype(str) + '%'
    res_df['F1'] = (res_df['F1'] * 100).round(2).astype(str) + '%'
    print(res_df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
