# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
diagnose_exp07_scores.py — exp_07「R@1=R@5=R@10=99.64%」異常結果診斷腳本

【診斷目的】
  驗證 exp_07 (w/o Music) 評估時，500-pool 內所有候選音樂是否取得
  完全相同的 ranking score，從而導致 argsort 的 GPU 排序副作用
  讓 GT（固定在 index 0）被誤判為第一名。

【驗證方法】
  1. 取少量測試樣本（預設 100 筆）
  2. 對每筆樣本計算 500 首候選的 ranking scores
  3. 檢查 score 是否全等（std == 0 或 max-min < 1e-5）
  4. 若全等：以隨機打亂排名重算 R@1/R@5/R@10（揭示真實性能）
  5. 對比：
       - 原始 argsort 排名（顯示異常高值）
       - Random tie-breaking 排名（顯示真實隨機基線 ≈ 1/500）

執行：
  python diagnose_exp07_scores.py

輸出：
  checkpoints/exp_07/diagnose_scores_result.json
  checkpoints/exp_07/diagnose_scores_detail.csv
"""

import os, sys, json, logging, random
import numpy as np
import torch
import h5py
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# 設定
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR    = str(PROJECT_ROOT)
H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"
OUTPUT_DIR  = os.path.join(BASE_DIR, "checkpoints", "exp_07")
BEST_CKPT   = os.path.join(OUTPUT_DIR, "best")

ACTIVE_MODALITIES = ["video", "ltp", "text"]   # exp_07：w/o Music
LTP_H5 = r"data/user_profiling/stage5_output/preference_vectors.h5"
SPECIAL_TOKENS = ["[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"]

N_DIAGNOSE   = 100    # 診斷樣本數（100 筆足以確認）
POOL_SIZE    = 500
MICRO_BATCH  = 32
EQUAL_EPS    = 1e-5   # score 全等判定閾值

LOG_PATH = os.path.join(OUTPUT_DIR, "diagnose_scores.log")
sys.path.insert(0, BASE_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger("diagnose_exp07")


# ══════════════════════════════════════════════════════════════════════════════
# 輔助函式
# ══════════════════════════════════════════════════════════════════════════════

def load_ltp_dict(h5_path, mode="hybrid"):
    npy = os.path.join(CACHE_DIR, f"ltp_{mode}.npy")
    ids = os.path.join(CACHE_DIR, f"ltp_{mode}_ids.json")
    if os.path.exists(npy) and os.path.exists(ids):
        arr = np.load(npy)
        with open(ids) as f:
            vl = json.load(f)
        d = {v: arr[i] for i, v in enumerate(vl)}
        logger.info("[LTP] 快取載入：%d 筆", len(d))
        return d
    logger.info("[LTP] 從 HDF5 載入...")
    out = {}
    with h5py.File(h5_path, "r") as f:
        grp = f["preference_vectors"]
        for k in grp.keys():
            out[k] = grp[k][:].astype(np.float32)
    logger.info("[LTP] 載入完成：%d 筆", len(out))
    return out


def load_model_and_data():
    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer
    from evaluate import pointwise_pool_scoring

    model_cfg = ModelConfig(
        llama_model_name=LLAMA_MODEL,
        video_dim=768, music_dim=768, text_dim=512, ltp_dim=256,
        num_candidates=1,
        active_modalities=ACTIVE_MODALITIES,
        rank_special_token="[RANK]",
    )
    train_cfg = TrainConfig(output_dir=OUTPUT_DIR, pointwise_eval_batch_size=MICRO_BATCH)

    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    ltp_dict = load_ltp_dict(LTP_H5)

    from dataset import (UnifiedMLLMDataset, build_pair_index,
                          load_conversation_map, build_song_bank, split_by_video_id)
    pair_index = build_pair_index(H5_DIR, cache_path=os.path.join(CACHE_DIR, "pair_index.json"))
    conv_map   = load_conversation_map(JSON_DIR)
    song_bank_np, song_ids = build_song_bank(pair_index, cache_path=os.path.join(CACHE_DIR, "song_bank"))
    _, _, te_pairs = split_by_video_id(pair_index, 0.90, 0.05, 0.05, 42)

    test_dataset = UnifiedMLLMDataset(
        pairs=te_pairs, tokenizer=tokenizer, conv_map=conv_map,
        song_bank=song_bank_np, song_ids=song_ids, ltp_dict=ltp_dict,
        max_seq_len=model_cfg.max_seq_len, is_train=False,
        ltp_dim=model_cfg.ltp_dim, mc_neg_cache_dir=CACHE_DIR,
        active_modalities=model_cfg.active_modalities,
    )
    all_music_features = torch.tensor(song_bank_np, dtype=torch.float32)
    logger.info("Test pairs: %d | Song bank: %d", len(te_pairs), len(song_ids))

    # 載入模型
    import torch.nn as nn
    from transformers import LlamaForCausalLM
    from peft import PeftModel
    from models.projectors import MultimodalProjectors
    from models.unified_mllm import UnifiedMLLM

    n = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    logger.info("tokenizer +%d special tokens → vocab=%d", n, len(tokenizer))

    base = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    base.resize_token_embeddings(len(tokenizer))
    peft_llama = PeftModel.from_pretrained(base, BEST_CKPT, torch_dtype=torch.bfloat16)
    peft_llama.eval()
    if hasattr(peft_llama, "gradient_checkpointing_disable"):
        peft_llama.gradient_checkpointing_disable()

    projectors = MultimodalProjectors(
        video_dim=768, music_dim=768, text_dim=512, ltp_dim=256,
        llama_hidden_dim=4096, projector_hidden_dim=2048, dropout=0.0,
    )
    projectors.load_state_dict(
        torch.load(os.path.join(BEST_CKPT, "projectors.pt"), map_location="cuda:0")
    )
    projectors = projectors.to(torch.bfloat16).cuda().eval()

    ranking_head = nn.Sequential(
        nn.LayerNorm(4096), nn.Linear(4096, 256), nn.GELU(),
        nn.Dropout(0.0), nn.Linear(256, 1),
    )
    ranking_head.load_state_dict(
        torch.load(os.path.join(BEST_CKPT, "ranking_head.pt"), map_location="cuda:0")
    )
    ranking_head = ranking_head.to(torch.bfloat16).cuda().eval()

    try:
        model = UnifiedMLLM(model_config=model_cfg, tokenizer=tokenizer, _load_llama=False)
    except TypeError:
        model = UnifiedMLLM(model_config=model_cfg, tokenizer=tokenizer)
        del model.llama
        torch.cuda.empty_cache()

    model.llama        = peft_llama
    model.projectors   = projectors
    model.ranking_head = ranking_head
    model.eval()

    return model, test_dataset, all_music_features, song_ids, model_cfg, train_cfg


# ══════════════════════════════════════════════════════════════════════════════
# 核心診斷：對單一樣本取得全部 500 個 scores，分析是否全等
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def diagnose_sample(model, sample, all_music_features, song_ids, device):
    """
    對一筆樣本計算 500-pool 的完整 scores，回傳：
      scores        : np.ndarray (500,)
      gt_rank_orig  : GT 用原始 argsort 的排名
      gt_rank_random: GT 用隨機 tie-breaking 的排名
      is_all_equal  : 是否所有 score 相同（True=問題確認）
      score_std     : score 標準差
    """
    M = all_music_features.size(0)
    video_feat = sample["video_feat"].unsqueeze(0).to(device)
    ltp_feat   = sample["ltp_feat"].unsqueeze(0).to(device)
    text_feat  = sample["text_feat"].unsqueeze(0).to(device)

    prompt_len     = int(sample["prompt_len"].item())
    full_input_ids = sample["input_ids"]
    full_attn_mask = sample["attention_mask"]
    prompt_ids     = full_input_ids[:prompt_len]
    prompt_mask    = full_attn_mask[:prompt_len]
    valid_len      = int(prompt_mask.sum().item())
    input_ids      = prompt_ids[:valid_len].unsqueeze(0).to(device)
    attention_mask = prompt_mask[:valid_len].unsqueeze(0).to(device)

    gt_music_id = sample.get("gt_music_id", "")
    video_id    = sample.get("video_id", "")

    gt_global_idx = next((i for i, sid in enumerate(song_ids) if sid == gt_music_id), 0)
    excl = {i for i, sid in enumerate(song_ids) if sid[:11] == video_id}
    candidates = [i for i in range(M) if i not in excl and i != gt_global_idx]
    rng = random.Random(20260315)
    negatives = rng.sample(candidates, min(POOL_SIZE - 1, len(candidates)))
    pool_idx  = [gt_global_idx] + negatives
    pool_feats = all_music_features[pool_idx].to(device)

    # ── 計算 scores ──────────────────────────────────────────────────────────
    all_scores = []
    for start in range(0, POOL_SIZE, MICRO_BATCH):
        end = min(start + MICRO_BATCH, POOL_SIZE)
        k   = end - start
        batch_music = pool_feats[start:end]
        v  = video_feat.expand(k, -1)
        l  = ltp_feat.expand(k, -1)
        t  = text_feat.expand(k, -1)
        ii = input_ids.expand(k, -1)
        am = attention_mask.expand(k, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                video_feat=v, music_candidates=batch_music.unsqueeze(1),
                ltp_feat=l, text_feat=t,
                input_ids=ii, attention_mask=am,
                labels=None, compute_gen_loss=False,
            )
        all_scores.append(out["ranking_score"].float().cpu())

    scores = torch.cat(all_scores, dim=0).numpy()   # (500,)

    # ── 統計 ─────────────────────────────────────────────────────────────────
    score_std  = float(np.std(scores))
    score_range = float(np.max(scores) - np.min(scores))
    is_all_equal = score_range < EQUAL_EPS

    # 原始 argsort 排名
    sorted_idx    = np.argsort(scores)[::-1]
    gt_rank_orig  = int(np.where(sorted_idx == 0)[0][0]) + 1

    # Random tie-breaking：加微小隨機擾動後排序
    rng2 = np.random.default_rng(42)
    noise = rng2.uniform(0, 1e-6, size=scores.shape)
    sorted_idx_rand = np.argsort(scores + noise)[::-1]
    gt_rank_random  = int(np.where(sorted_idx_rand == 0)[0][0]) + 1

    return {
        "scores_mean":    float(np.mean(scores)),
        "scores_std":     score_std,
        "scores_range":   score_range,
        "scores_min":     float(np.min(scores)),
        "scores_max":     float(np.max(scores)),
        "is_all_equal":   is_all_equal,
        "gt_rank_orig":   gt_rank_orig,
        "gt_rank_random": gt_rank_random,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda")
    logger.info("=" * 60)
    logger.info("exp_07 Score 診斷腳本")
    logger.info("  驗證假設：w/o Music 時所有 500 候選 score 完全相同")
    logger.info("  樣本數：%d  |  pool_size：%d", N_DIAGNOSE, POOL_SIZE)
    logger.info("=" * 60)

    model, test_dataset, all_music_features, song_ids, model_cfg, train_cfg = \
        load_model_and_data()

    records = []
    n_all_equal   = 0
    ranks_orig    = []
    ranks_random  = []

    for idx in range(N_DIAGNOSE):
        sample = test_dataset[idx]
        result = diagnose_sample(model, sample, all_music_features, song_ids, device)
        result["sample_idx"] = idx
        records.append(result)

        if result["is_all_equal"]:
            n_all_equal += 1
        ranks_orig.append(result["gt_rank_orig"])
        ranks_random.append(result["gt_rank_random"])

        if (idx + 1) % 10 == 0:
            logger.info(
                "  [%3d/%d] score_std=%.2e  range=%.2e  all_equal=%s  "
                "rank_orig=%3d  rank_random=%3d",
                idx + 1, N_DIAGNOSE,
                result["scores_std"], result["scores_range"],
                "YES ✗" if result["is_all_equal"] else "no ✓",
                result["gt_rank_orig"], result["gt_rank_random"],
            )

    # ── 彙整統計 ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("【診斷結果彙整（%d 筆樣本）】", N_DIAGNOSE)
    logger.info("=" * 60)

    n_eq_pct = n_all_equal / N_DIAGNOSE * 100
    logger.info("  全等樣本數：%d / %d（%.1f%%）", n_all_equal, N_DIAGNOSE, n_eq_pct)
    logger.info("  avg score_std  = %.6f", np.mean([r["scores_std"]   for r in records]))
    logger.info("  avg score_range= %.6f", np.mean([r["scores_range"] for r in records]))

    logger.info("\n  ── 原始 argsort 排名（製造虛假高分）──")
    logger.info("  R@1  = %.2f%%", np.mean([r <= 1  for r in ranks_orig]) * 100)
    logger.info("  R@5  = %.2f%%", np.mean([r <= 5  for r in ranks_orig]) * 100)
    logger.info("  R@10 = %.2f%%", np.mean([r <= 10 for r in ranks_orig]) * 100)
    logger.info("  MR   = %.1f",   float(np.median(ranks_orig)))
    logger.info("  Mean = %.1f",   float(np.mean(ranks_orig)))

    logger.info("\n  ── Random tie-breaking 排名（真實性能）──")
    logger.info("  R@1  = %.2f%%", np.mean([r <= 1  for r in ranks_random]) * 100)
    logger.info("  R@5  = %.2f%%", np.mean([r <= 5  for r in ranks_random]) * 100)
    logger.info("  R@10 = %.2f%%", np.mean([r <= 10 for r in ranks_random]) * 100)
    logger.info("  MR   = %.1f",   float(np.median(ranks_random)))
    logger.info("  Mean = %.1f",   float(np.mean(ranks_random)))

    logger.info("\n  ── 理論基線（完全隨機，1/500）──")
    logger.info("  R@1  =  0.20%%  (1/500)")
    logger.info("  R@5  =  1.00%%  (5/500)")
    logger.info("  R@10 =  2.00%%  (10/500)")
    logger.info("  MR   = 250.5    (500+1)/2)")

    # ── 結論 ──────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("【結論】")
    if n_all_equal > N_DIAGNOSE * 0.9:
        logger.info("  ✗ 確認：exp_07 的 500-pool scores 全部相同（%.1f%%）", n_eq_pct)
        logger.info("  ✗ 根本原因：music 不在 active_modalities，projector 不處理")
        logger.info("              → 所有候選 music 取得相同的 inputs_embeds")
        logger.info("              → ranking_head 輸出完全相同的 scalar score")
        logger.info("  ✗ 副作用：CUDA argsort 對等值通常保持原始順序")
        logger.info("              → GT（固定在 pool_idx[0]）幾乎永遠排名第 1")
        logger.info("  ✗ 結論：R@1=R@5=R@10=99.64%% 是測試邏輯缺陷造成的偽高分")
        logger.info("              → 真實性能 ≈ 隨機基線（R@1≈0.2%%）")
    else:
        logger.info("  ✓ scores 並非全等，請進一步檢查原因")
    logger.info("=" * 60)

    # ── 儲存結果 ──────────────────────────────────────────────────────────────
    summary = {
        "exp": "exp_07",
        "n_diagnose": N_DIAGNOSE,
        "pool_size": POOL_SIZE,
        "equal_eps": EQUAL_EPS,
        "n_all_equal": n_all_equal,
        "pct_all_equal": round(n_eq_pct, 2),
        "avg_score_std":   round(float(np.mean([r["scores_std"]   for r in records])), 8),
        "avg_score_range": round(float(np.mean([r["scores_range"] for r in records])), 8),
        "ranking_orig": {
            "R@1":  round(np.mean([r <= 1  for r in ranks_orig]) * 100, 4),
            "R@5":  round(np.mean([r <= 5  for r in ranks_orig]) * 100, 4),
            "R@10": round(np.mean([r <= 10 for r in ranks_orig]) * 100, 4),
            "MR":   round(float(np.median(ranks_orig)), 1),
            "mean_rank": round(float(np.mean(ranks_orig)), 2),
        },
        "ranking_random_tiebreak": {
            "R@1":  round(np.mean([r <= 1  for r in ranks_random]) * 100, 4),
            "R@5":  round(np.mean([r <= 5  for r in ranks_random]) * 100, 4),
            "R@10": round(np.mean([r <= 10 for r in ranks_random]) * 100, 4),
            "MR":   round(float(np.median(ranks_random)), 1),
            "mean_rank": round(float(np.mean(ranks_random)), 2),
        },
        "theoretical_random_baseline": {
            "R@1": 0.20, "R@5": 1.00, "R@10": 2.00, "MR": 250.5,
            "note": "1/pool_size for each threshold"
        },
        "diagnosis": (
            "CONFIRMED: all 500 scores equal because music modality is inactive. "
            "argsort artifact places GT (index 0) at rank 1. "
            "True performance ≈ random baseline."
        ) if n_all_equal > N_DIAGNOSE * 0.9 else "scores not all equal, further investigation needed",
    }

    out_json = os.path.join(OUTPUT_DIR, "diagnose_scores_result.json")
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    logger.info("\n結果已儲存：%s", out_json)

    # CSV 明細
    try:
        import csv
        out_csv = os.path.join(OUTPUT_DIR, "diagnose_scores_detail.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        logger.info("明細已儲存：%s", out_csv)
    except Exception as e:
        logger.warning("CSV 儲存失敗：%s", e)


if __name__ == "__main__":
    main()
