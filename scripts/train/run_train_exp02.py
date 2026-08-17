# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
run_train_exp02.py — 消融實驗 A_explicit：P_ltp 使用 explicit_only 向量

消融設計：
  exp_01（主實驗）：LTP_MODE = "hybrid"      → 已完成，R@1=30.82%
  exp_02（本腳本）：LTP_MODE = "explicit_only" → 只用顯性語義偏好（CLIP-Text 512D→256D）
  exp_03          ：LTP_MODE = "implicit_only"  → 只用隱性行為偏好（AST 768D→256D）

消融目的：
  驗證 hybrid P_ltp 的設計選擇是否有效：
  若 explicit_only 或 implicit_only 效能顯著低於 hybrid，
  即說明兩種偏好表示的融合在本研究中是必要的。

修正清單（相比 run_train.py）：
  ★ P1 fix：epoch/step loss 透過 train.py 的 logger 記錄（請確認 train.py 使用 logger 而非 print）
  ★ P6 fix：VAL_POOL_SIZE 建議改為 100（需 train.py 支援此參數）
  ★ 其他：RESUME_CHECKPOINT=None（從頭訓練），OUTPUT_DIR=exp_02，LTP_MODE=explicit_only
  ★ vocab 注釋修正：5 個 special tokens（[VIDEO][MUSIC][LTP][TEXT_CLIP][RANK]）→ 32005

執行：
  python run_train_exp02.py
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, Optional

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"]     = "disabled"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, TrainConfig
from train import train

# ─────────────────────────────────────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────────────────────────────────────

H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
# ★ exp_02：獨立輸出目錄，不覆蓋 exp_01
OUTPUT_DIR  = str(PROJECT_ROOT / "checkpoints" / "exp_02")
CACHE_DIR   = str(PROJECT_ROOT / "cache")   # 快取共用
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"

LOG_PATH = os.path.join(OUTPUT_DIR, "train.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger("run_train_exp02")

# ★ 從頭訓練（消融實驗必須獨立訓練，不可接續 exp_01 的 checkpoint）
RESUME_CHECKPOINT = os.path.join(OUTPUT_DIR, "epoch_3")   # ← 取消注釋並確認 checkpoint 路徑正確
# RESUME_CHECKPOINT = None   # 從頭訓練

# ── P_ltp 路徑 ───────────────────────────────────────────────────────────────
LTP_H5 = {
    "hybrid":        str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors.h5"),
    "explicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_explicit_only.h5"),
    "implicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_implicit_only.h5"),
}

# ★ exp_02 消融設定：只用顯性語義偏好
LTP_MODE = "explicit_only"


# ─────────────────────────────────────────────────────────────────────────────
# P_ltp 載入
# ─────────────────────────────────────────────────────────────────────────────

def load_ltp_dict(h5_path: str, mode: str, cache_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    import h5py
    if cache_path:
        npy = cache_path + f"_{mode}.npy"
        ids = cache_path + f"_{mode}_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids) as f:
                vid_list = json.load(f)
            d = {v: arr[i] for i, v in enumerate(vid_list)}
            logger.info(f"[LTP] 快取載入：{len(d)} 筆，dim=256，模式={mode}")
            return d

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"找不到 P_ltp 檔案：{h5_path}")

    logger.info(f"[LTP] 載入 {mode}：{h5_path}")
    ltp_dict = {}
    with h5py.File(h5_path, "r") as f:
        grp = f["preference_vectors"]
        for k in grp.keys():
            ltp_dict[k] = grp[k][:].astype(np.float32)

    logger.info(f"[LTP] 載入完成：{len(ltp_dict)} 筆，dim=256，模式={mode}")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        vid_list = list(ltp_dict.keys())
        arr      = np.stack([ltp_dict[v] for v in vid_list])
        np.save(cache_path + f"_{mode}.npy", arr)
        with open(cache_path + f"_{mode}_ids.json", "w") as f:
            json.dump(vid_list, f)

    return ltp_dict


# ─────────────────────────────────────────────────────────────────────────────
# 模型設定（與 exp_01 完全相同，確保公平比較）
# ─────────────────────────────────────────────────────────────────────────────

model_cfg = ModelConfig(
    llama_model_name  = LLAMA_MODEL,
    video_dim         = 768,
    music_dim         = 768,
    text_dim          = 512,
    ltp_dim           = 256,
    num_candidates    = 1,
    lora_rank         = 32,
    lora_alpha        = 16,
    lambda_rank       = 0.5,
    lambda_gen        = 0.5,
    multimodal_prefix_len = 4,
    # ★ 修正注釋：vocab = 32000 + 5 special tokens = 32005
    # special tokens: [VIDEO][MUSIC][LTP][TEXT_CLIP][RANK]
    rank_special_token = "[RANK]",
)


# ─────────────────────────────────────────────────────────────────────────────
# 訓練設定
# ─────────────────────────────────────────────────────────────────────────────

train_cfg = TrainConfig(
    data_dir          = H5_DIR,
    json_dir          = JSON_DIR,
    pair_index_cache  = os.path.join(CACHE_DIR, "pair_index.json"),
    song_bank_cache   = os.path.join(CACHE_DIR, "song_bank"),
    output_dir        = OUTPUT_DIR,
    train_ratio       = 0.90,
    val_ratio         = 0.05,
    test_ratio        = 0.05,
    split_seed        = 42,
    micro_batch_size  = 4,
    accumulation_steps= 16,
    eval_batch_size   = 8,
    num_epochs        = 10,
    learning_rate     = 2e-4,
    lora_learning_rate= 2e-4,
    weight_decay      = 5e-4,
    warmup_ratio      = 0.03,
    use_bf16          = True,
    use_fp16          = False,
    use_gradient_checkpointing = True,
    music_pool_size   = 500,
    pointwise_eval_batch_size = 32,
    best_model_metric = "recall@1",
    ranking_loss_type = "bpr",
    resume_checkpoint = RESUME_CHECKPOINT,
    # ★ P6 fix（建議）：若 TrainConfig / train.py 支援，改用 100-pool 提高鑑別力
    # val_pool_size   = 100,   # ← 取消注釋並確認 train.py 支援此參數
)


# ─────────────────────────────────────────────────────────────────────────────
# 啟動
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        ]
    )
    logger.info("Log 檔案路徑: %s", LOG_PATH)

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("  消融實驗 exp_02：explicit_only P_ltp")
    print("=" * 60)
    print(f"  輸出目錄    : {OUTPUT_DIR}")
    print(f"  LTP 模式    : {LTP_MODE}（僅顯性語義偏好）")
    print(f"  RESUME      : {RESUME_CHECKPOINT}（從頭訓練）")
    print(f"  損失權重    : λ_rank={model_cfg.lambda_rank}, λ_gen={model_cfg.lambda_gen}")
    print(f"  Vocab 大小  : 32000 + 5 = 32005")
    print("=" * 60)

    ltp_dict = None
    try:
        ltp_dict = load_ltp_dict(
            h5_path    = LTP_H5[LTP_MODE],
            mode       = LTP_MODE,
            cache_path = os.path.join(CACHE_DIR, "ltp"),
        )
    except Exception as e:
        logger.error(f"[LTP] 載入失敗：{e}")
        logger.warning("[LTP] 退回零向量模式")

    train(model_config=model_cfg, train_config=train_cfg, ltp_dict=ltp_dict)
