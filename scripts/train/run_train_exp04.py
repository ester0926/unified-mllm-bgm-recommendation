"""
用途：設定並啟動 exp_04 的訓練流程。
輸入：data/、cache/ 與 checkpoints/ 中的特徵、LTP 向量和資料切分。
輸出：新的訓練 checkpoint、log 與必要的中間結果。
執行：建議在 repo 根目錄執行，並先確認 config.py 的資料路徑。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os, sys, json, logging
import numpy as np
from typing import Dict, Optional

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"]     = "disabled"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, TrainConfig
from train import train

H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
OUTPUT_DIR  = str(PROJECT_ROOT / "checkpoints" / "exp_04")
CACHE_DIR   = str(PROJECT_ROOT / "cache")
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"

LOG_PATH = os.path.join(OUTPUT_DIR, "train.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger = logging.getLogger("run_train_exp04")

RESUME_CHECKPOINT = None  # 從頭訓練

LTP_H5 = {
    "hybrid": r"data/user_profiling/stage5_output/preference_vectors.h5",
}
# exp_04 雖然移除 LTP，但 dataset 仍需載入（用於 ltp_feat 欄位），
# projectors.py 會因 active_modalities 不含 "ltp" 而略過 ltp_proj
LTP_MODE = "hybrid"

# ★ 核心設定：移除 P_ltp
ACTIVE_MODALITIES = ["video", "text", "music"]   # 不含 "ltp"

model_cfg = ModelConfig(
    llama_model_name      = LLAMA_MODEL,
    video_dim=768, music_dim=768, text_dim=512, ltp_dim=256,
    num_candidates=1,
    lora_rank=32, lora_alpha=16,
    lambda_rank=0.5, lambda_gen=0.5,
    # active_modalities 驅動 projectors.py 和 unified_mllm.py 的動態行為
    active_modalities     = ACTIVE_MODALITIES,
    # multimodal_prefix_len 由 active_modalities 自動計算（= 3）
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
        npy = cache_path + f"_{mode}.npy"
        ids = cache_path + f"_{mode}_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids) as f:
                vid_list = json.load(f)
            d = {v: arr[i] for i, v in enumerate(vid_list)}
            logger.info(f"[LTP] 快取載入：{len(d)} 筆，模式={mode}")
            return d
    out = {}
    with h5py.File(h5_path, "r") as f:
        grp = f["preference_vectors"]
        for k in grp.keys():
            out[k] = grp[k][:].astype(np.float32)
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
    print(f"  消融實驗 exp_04：w/o P_ltp")
    print(f"  active_modalities = {ACTIVE_MODALITIES}")
    print(f"  prefix_len = {len(ACTIVE_MODALITIES)}（4→3）")
    print(f"  Vocab: 32000 + 5 = 32005（特殊 token 數量不變）")
    print("=" * 60)

    ltp_dict = load_ltp_dict(LTP_H5[LTP_MODE], LTP_MODE,
                             os.path.join(CACHE_DIR, "ltp"))
    train(model_config=model_cfg, train_config=train_cfg, ltp_dict=ltp_dict)
