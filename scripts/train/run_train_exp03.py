"""
用途：設定並啟動 exp_03 的訓練流程。
輸入：data/、cache/ 與 checkpoints/ 中的特徵、LTP 向量和資料切分。
輸出：新的訓練 checkpoint、log 與必要的中間結果。
執行：建議在 repo 根目錄執行，並先確認 config.py 的資料路徑。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
# ★ exp_03：獨立輸出目錄
OUTPUT_DIR  = str(PROJECT_ROOT / "checkpoints" / "exp_03")
CACHE_DIR   = str(PROJECT_ROOT / "cache")   # 快取共用
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"

LOG_PATH = os.path.join(OUTPUT_DIR, "train.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger("run_train_exp03")

# ★ 從頭訓練
# RESUME_CHECKPOINT = None
RESUME_CHECKPOINT = os.path.join(OUTPUT_DIR, "epoch_7")  # 從 epoch 4 接續

LTP_H5 = {
    "hybrid":        str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors.h5"),
    "explicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_explicit_only.h5"),
    "implicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_implicit_only.h5"),
}

# ★ exp_03 消融設定：只用隱性行為偏好
LTP_MODE = "implicit_only"


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
    rank_special_token = "[RANK]",
)

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
    # val_pool_size   = 100,   # ← P6 fix：取消注釋以提高鑑別力
)


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
    print("  消融實驗 exp_03：implicit_only P_ltp")
    print("=" * 60)
    print(f"  輸出目錄    : {OUTPUT_DIR}")
    print(f"  LTP 模式    : {LTP_MODE}（僅隱性行為偏好）")
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
