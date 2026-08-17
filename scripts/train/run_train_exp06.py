# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
run_train_exp06.py — 模態消融：w/o Text (TEXT_CLIP)

移除對話文字特徵（CLIP Text 512D）。
prefix 縮減為 [VIDEO][LTP][MUSIC]（3 tokens）。
text_proj 投影層不被呼叫。
build_prompt() 移除 "Context: [TEXT_CLIP] {user_text}" 行（即不使用 t3）。

學術意義：
  驗證對話文字（用戶在推薦對話中的文字表達）的必要性。
  預期：移除文字後推薦指標下降，但因還有 P_ltp 補足部分用戶偏好信號，
  降幅可能小於 MuseChat w/o Text。
"""

import os, sys, json, logging
import numpy as np

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"]     = "disabled"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, TrainConfig
from train import train

H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
OUTPUT_DIR  = str(PROJECT_ROOT / "checkpoints" / "exp_06")
CACHE_DIR   = str(PROJECT_ROOT / "cache")
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"

LOG_PATH = os.path.join(OUTPUT_DIR, "train.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger = logging.getLogger("run_train_exp06")

RESUME_CHECKPOINT = os.path.join(OUTPUT_DIR, "epoch_9")  # 從 epoch 9 接續
LTP_H5 = {"hybrid": r"data/user_profiling/stage5_output/preference_vectors.h5"}
LTP_MODE = "hybrid"

# ★ 核心設定：移除 Text
ACTIVE_MODALITIES = ["video", "ltp", "music"]   # 不含 "text"

model_cfg = ModelConfig(
    llama_model_name      = LLAMA_MODEL,
    video_dim=768, music_dim=768, text_dim=512, ltp_dim=256,
    num_candidates=1, lora_rank=32, lora_alpha=16,
    lambda_rank=0.5, lambda_gen=0.5,
    active_modalities     = ACTIVE_MODALITIES,
    rank_special_token    = "[RANK]",
)

train_cfg = TrainConfig(
    data_dir=H5_DIR, json_dir=JSON_DIR,
    pair_index_cache=os.path.join(CACHE_DIR, "pair_index.json"),
    song_bank_cache =os.path.join(CACHE_DIR, "song_bank"),
    output_dir=OUTPUT_DIR,
    train_ratio=0.90, val_ratio=0.05, test_ratio=0.05, split_seed=42,
    micro_batch_size=4, accumulation_steps=16, eval_batch_size=8,
    num_epochs=10,
    learning_rate=2e-4, lora_learning_rate=2e-4, weight_decay=5e-4,
    warmup_ratio=0.03, use_bf16=True,
    music_pool_size=500, pointwise_eval_batch_size=32,
    best_model_metric="recall@1", ranking_loss_type="bpr",
    resume_checkpoint=RESUME_CHECKPOINT,
)


def load_ltp_dict(h5_path, mode, cache_path=None):
    import h5py
    if cache_path:
        npy, ids = cache_path + f"_{mode}.npy", cache_path + f"_{mode}_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids) as f: vid_list = json.load(f)
            return {v: arr[i] for i, v in enumerate(vid_list)}
    out = {}
    with h5py.File(h5_path, "r") as f:
        grp = f["preference_vectors"]
        for k in grp.keys(): out[k] = grp[k][:].astype(np.float32)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        ]
    )
    logger.info("Log: %s", LOG_PATH)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("=" * 60)
    print(f"  消融實驗 exp_06：w/o Text (TEXT_CLIP)")
    print(f"  active_modalities = {ACTIVE_MODALITIES}")
    print(f"  prefix_len = {len(ACTIVE_MODALITIES)}（4→3）")
    print(f"  注意：build_prompt() 需在 dataset.py 中條件性移除 TEXT_CLIP 行")
    print("=" * 60)
    ltp_dict = load_ltp_dict(LTP_H5[LTP_MODE], LTP_MODE, os.path.join(CACHE_DIR, "ltp"))
    train(model_config=model_cfg, train_config=train_cfg, ltp_dict=ltp_dict)
