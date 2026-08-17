"""
mc_neg_bank.py — 預計算 candidate music CLS mean pool 並存成快取
執行一次後，dataset.py 改為從快取讀取，避免每次讀 28MB HDF5
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
