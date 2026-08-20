"""
用途：建立或讀取 MuseChat 候選音樂負樣本快取。
輸入：依程式內路徑設定讀取本專案資料或前一階段輸出。
輸出：依程式內 OUTPUT_DIR、results 或 checkpoints 設定寫出結果。
執行：建議在 repo 根目錄執行，避免相對路徑錯誤。
"""

import glob, json, os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
H5_DIR    = PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_NPY = CACHE_DIR / "mc_neg_bank.npy"
CACHE_IDS = CACHE_DIR / "mc_neg_bank_ids.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

h5_files = sorted(glob.glob(os.path.join(str(H5_DIR), "*.h5")))
feats = {}

for h5_path in tqdm(h5_files):
    with h5py.File(h5_path, "r") as f:
        if "pairs" not in f:
            continue
        for key in f["pairs"].keys():
            try:
                cand = f[f"pairs/{key}/candidate_music_all_seq"][:, 0, :].astype(np.float32)
                feats[key] = cand.mean(axis=0)   # (768,)
            except Exception:
                pass

pair_ids  = sorted(feats.keys())
arr       = np.stack([feats[k] for k in pair_ids])   # (N, 768)
np.save(CACHE_NPY, arr)
with open(CACHE_IDS, "w") as f:
    json.dump(pair_ids, f)

print(f"完成：{len(pair_ids)} 筆，存至 {CACHE_NPY}")
