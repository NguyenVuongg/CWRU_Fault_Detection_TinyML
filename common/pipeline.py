# -*- coding: utf-8 -*-
"""
common/pipeline.py
====================
Helper dùng chung để 8 notebook không phải lặp lại logic "lấy manifest ở
đâu". Mỗi notebook có thể chạy ĐỘC LẬP (không cần chạy 01 trước) nhờ hàm
get_manifest() tự động: dùng lại manifest đã lưu trong outputs/ nếu có,
nếu chưa có thì tự sinh dữ liệu giả lập (khi USE_SYNTHETIC_DATA=True) hoặc
quét DATA_ROOT thật rồi build manifest mới.
"""

from pathlib import Path

import pandas as pd

from . import io_utils, synthetic


def get_manifest(use_synthetic: bool, real_data_root, synthetic_data_root,
                  output_dir, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Trả về DataFrame manifest, ưu tiên đọc cache từ
    <output_dir>/tables/manifest.csv nếu đã tồn tại và force_rebuild=False.

    use_synthetic=True  -> nếu cần build mới, tự sinh dữ liệu giả lập vào
                            synthetic_data_root rồi build manifest từ đó.
    use_synthetic=False -> build manifest trực tiếp từ real_data_root
                            (bạn phải tự đảm bảo thư mục này đã có file .mat
                            thật và parse_metadata_from_filename() đã được
                            chỉnh đúng cấu trúc — xem common/io_utils.py).
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tables_dir / "manifest.csv"

    if manifest_path.exists() and not force_rebuild:
        return pd.read_csv(manifest_path)

    if use_synthetic:
        data_root = synthetic.build_synthetic_dataset(Path(synthetic_data_root))
    else:
        data_root = Path(real_data_root)
        if not data_root.exists():
            raise FileNotFoundError(
                f"real_data_root '{data_root}' không tồn tại. Chỉnh đường dẫn "
                f"hoặc đặt USE_SYNTHETIC_DATA=True để chạy thử bằng dữ liệu giả."
            )

    manifest = io_utils.build_manifest(data_root)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def pick_file(manifest: pd.DataFrame, label: str, load_hp: int, diameter_mils=None):
    """Chọn 1 file đại diện khớp điều kiện, trả về đường dẫn (str)."""
    df = manifest[(manifest["label"] == label) & (manifest["load_hp"] == load_hp)]
    if diameter_mils is not None:
        df = df[df["fault_diameter_mils"] == diameter_mils]
    if len(df) == 0:
        raise ValueError(
            f"Không tìm thấy file khớp label={label}, load_hp={load_hp}, "
            f"diameter_mils={diameter_mils} trong manifest."
        )
    return df.iloc[0]["file_path"]
