# Auto-added: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = PROJECT_ROOT / "scripts" / "diagnostics"
for _p in [str(PROJECT_ROOT), str(DIAGNOSTICS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

"""
run_eval_500pool_ltp_control.py
================================
Matched / Shuffled / Random LTP 對照實驗（inference-time，無需重新訓練）

實驗目的（對應 TISMIR 補強事項 RQ5）：
  驗證 Long-Term Preference（LTP）對排序效能的貢獻是否真實有效，
  而非模型忽略 LTP token 或對任意輸入給出相同分數。

三組條件（同一模型 exp_01、同一 500-pool 種子）：
  matched  — 正常使用每個 video 對應的 LTP 向量（= 標準 exp_01）
  shuffled — 將測試集內所有 video 的 LTP 指派隨機打亂
             （video A 拿到 video B 的 LTP；保留向量分布但破壞配對語義）
  random   — 每個 sample 使用獨立的標準高斯雜訊向量
             （完全無使用者偏好資訊的 baseline）

解讀：
  matched >> shuffled ≈ random → LTP 確實傳遞了語義偏好資訊（最佳結果）
  matched ≈ shuffled > random  → 模型利用了 LTP 分布特性而非語義內容
  matched ≈ shuffled ≈ random  → 模型未有效使用 LTP（需進一步檢查）

輸出：
  results/main_eval/exp_01/ltp_control/
    ltp_matched_ranking.csv
    ltp_shuffled_ranking.csv
    ltp_random_ranking.csv
    ltp_control_summary.json   ← 三組比較摘要表

使用方式：
  VSCode Run 或：
  python scripts/eval_main/run_eval_500pool_ltp_control.py
"""

import csv
import datetime as _dt
import gc
import json
import logging
import os
import random
import traceback

import h5py
import numpy as np
import torch


# =============================================================================
# 路徑設定
# =============================================================================

BASE_DIR    = str(PROJECT_ROOT)
H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
LLAMA_MODEL = "meta-llama/Llama-2-7b-hf"

SPECIAL_TOKENS = ["[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"]

# =============================================================================
# 使用者設定（按需調整）
# =============================================================================

EXP_NAME             = "exp_01"
CKPT_NAME            = "best"
LTP_DIM              = 256        # hybrid = 256
POOL_SIZE            = 500
CANDIDATE_POOL_SEED  = 20260315   # 與 run_eval_500pool_detailed.py 一致
SHUFFLE_SEED         = 42
RANDOM_SEED          = 1234
POINTWISE_BATCH_SIZE = 16
TIEBREAK_NOISE       = True
TIEBREAK_SEED        = 42
MAX_SAMPLES: int | None = None    # None = 全測試集；整數 = smoke test

LTP_H5 = {
    "hybrid":        str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors.h5"),
    "explicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_explicit_only.h5"),
    "implicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_implicit_only.h5"),
}

EXP_TO_LTP_MODE = {
    "exp_01": "hybrid",        "exp_02": "explicit_only",
    "exp_03": "implicit_only", "exp_04": "hybrid",
    "exp_05": "hybrid",        "exp_06": "hybrid",
    "exp_07": "hybrid",
}

EXP_TO_MODALITIES = {
    "exp_01": ["video", "ltp", "text", "music"],
    "exp_02": ["video", "ltp", "text", "music"],
    "exp_03": ["video", "ltp", "text", "music"],
    "exp_04": ["video", "text", "music"],
    "exp_05": ["ltp",   "text", "music"],
    "exp_06": ["video", "ltp",  "music"],
    "exp_07": ["video", "ltp",  "text"],
}


# =============================================================================
# Logging
# =============================================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("ltp_control")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    for h in [logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_path, mode="a", encoding="utf-8")]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# =============================================================================
# LTP 載入
# =============================================================================

def load_ltp_dict(
    h5_path: str,
    mode: str,
    cache_path: str | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    if cache_path:
        npy = cache_path + f"_{mode}.npy"
        ids = cache_path + f"_{mode}_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids, encoding="utf-8") as f:
                video_ids = json.load(f)
            out = {v: arr[i] for i, v in enumerate(video_ids)}
            if logger:
                logger.info("[LTP] loaded from cache: %d entries (%s)", len(out), mode)
            return out
    if logger:
        logger.info("[LTP] loading HDF5: %s", h5_path)
    out: dict = {}
    with h5py.File(h5_path, "r") as f:
        grp: h5py.Group = f["preference_vectors"]  # type: ignore[assignment]
        for k in grp.keys():
            out[k] = grp[k][()].astype(np.float32)  # type: ignore[index, union-attr]
    return out


# =============================================================================
# LTP 控制條件建構
# =============================================================================

def build_shuffled_ltp(
    ltp_dict: dict,
    test_video_ids: list[str],
    seed: int,
    ltp_dim: int,
    logger: logging.Logger | None = None,
) -> dict:
    """
    將測試集內 video 的 LTP 指派隨機打亂（近似 derangement）。

    - 有 LTP 記錄的 test video → 拿到另一個 test video 的 LTP
    - 無 LTP 記錄的 test video → 零向量
    保留整體向量分布，但破壞配對語義。
    """
    available   = [v for v in test_video_ids if v in ltp_dict]
    unavailable = [v for v in test_video_ids if v not in ltp_dict]

    # 打亂直到沒有 fixed point（至多 10 次，幾乎總是 1-2 次即可）
    shuffled_targets = available[:]
    rng_py = random.Random(seed)
    for _ in range(10):
        rng_py.shuffle(shuffled_targets)
        if all(a != b for a, b in zip(available, shuffled_targets)):
            break

    shuffled_ltp: dict = {}
    for src_vid, tgt_vid in zip(available, shuffled_targets):
        shuffled_ltp[src_vid] = ltp_dict[tgt_vid]

    zero_vec = np.zeros(ltp_dim, dtype=np.float32)
    for v in unavailable:
        shuffled_ltp[v] = zero_vec

    n_fp = sum(a == b for a, b in zip(available, shuffled_targets))
    if logger:
        logger.info("[LTP-shuffled] %d available, %d unavailable, fixed-points=%d",
                    len(available), len(unavailable), n_fp)
    return shuffled_ltp


def get_ltp_feat(
    sample: dict,
    condition: str,
    ltp_dict_shuffled: dict,
    ltp_dim: int,
    random_rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    """
    依條件回傳 ltp_feat tensor，shape = (1, ltp_dim)。

    matched  : 使用 dataset 已載入的正確 LTP（sample["ltp_feat"]）
    shuffled : 使用 ltp_dict_shuffled[video_id]
    random   : 每個 sample 獨立標準高斯雜訊
    """
    if condition == "matched":
        return sample["ltp_feat"].unsqueeze(0).to(device)

    video_id: str = sample.get("video_id", "")

    if condition == "shuffled":
        vec = ltp_dict_shuffled.get(video_id, np.zeros(ltp_dim, dtype=np.float32))
        return torch.from_numpy(vec).unsqueeze(0).to(device)

    # condition == "random"
    vec = random_rng.standard_normal(ltp_dim).astype(np.float32)
    return torch.from_numpy(vec).unsqueeze(0).to(device)


# =============================================================================
# 模型與資料集載入
# =============================================================================

def build_test_data(model_config, train_config, tokenizer, ltp_dict, logger):
    from dataset import (
        UnifiedMLLMDataset,
        build_pair_index,
        build_song_bank,
        load_conversation_map,
        split_by_video_id,
    )
    pair_index = build_pair_index(
        H5_DIR, cache_path=os.path.join(CACHE_DIR, "pair_index.json"))
    conv_map = load_conversation_map(JSON_DIR)
    song_bank_np, song_ids = build_song_bank(
        pair_index, cache_path=os.path.join(CACHE_DIR, "song_bank"))
    _, _, test_pairs = split_by_video_id(
        pair_index,
        train_config.train_ratio,
        train_config.val_ratio,
        train_config.test_ratio,
        train_config.split_seed,
    )
    test_dataset = UnifiedMLLMDataset(
        pairs=test_pairs,
        tokenizer=tokenizer,
        conv_map=conv_map,
        song_bank=song_bank_np,
        song_ids=song_ids,
        ltp_dict=ltp_dict,
        max_seq_len=model_config.max_seq_len,
        is_train=False,
        ltp_dim=model_config.ltp_dim,
        mc_neg_cache_dir=CACHE_DIR,
        active_modalities=model_config.active_modalities,
    )
    test_video_ids = [p[1][:11] for p in test_pairs]
    logger.info("Test pairs=%d | song bank=%d", len(test_pairs), len(song_ids))
    return test_dataset, torch.tensor(song_bank_np, dtype=torch.float32), song_ids, test_video_ids


def load_model(ckpt_dir: str, model_config, tokenizer, logger):
    import torch.nn as nn
    from peft import PeftModel
    from transformers import LlamaForCausalLM
    from models.projectors import MultimodalProjectors
    from models.unified_mllm import UnifiedMLLM

    logger.info("Loading checkpoint: %s", ckpt_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    base = LlamaForCausalLM.from_pretrained(
        model_config.llama_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base.resize_token_embeddings(len(tokenizer))
    peft_llama = PeftModel.from_pretrained(
        base, ckpt_dir, torch_dtype=torch.bfloat16)
    peft_llama.eval()
    _disable_gc = getattr(peft_llama, "gradient_checkpointing_disable", None)
    if callable(_disable_gc):
        _disable_gc()

    projectors = MultimodalProjectors(
        video_dim=model_config.video_dim,
        music_dim=model_config.music_dim,
        text_dim=model_config.text_dim,
        ltp_dim=model_config.ltp_dim,
        llama_hidden_dim=model_config.llama_hidden_dim,
        projector_hidden_dim=model_config.projector_hidden_dim,
        dropout=0.0,
    )
    projectors.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "projectors.pt"),
                   map_location="cuda:0", weights_only=False))
    projectors = projectors.to(torch.bfloat16).cuda().eval()

    ranking_head = torch.nn.Sequential(
        nn.LayerNorm(model_config.llama_hidden_dim),
        nn.Linear(model_config.llama_hidden_dim, 256),
        nn.GELU(), nn.Dropout(0.0), nn.Linear(256, 1),
    )
    ranking_head.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "ranking_head.pt"),
                   map_location="cuda:0", weights_only=False))
    ranking_head = ranking_head.to(torch.bfloat16).cuda().eval()

    # UnifiedMLLM 組裝（不呼叫 __init__，避免重新載入 LLaMA）
    model = UnifiedMLLM.__new__(UnifiedMLLM)
    nn.Module.__init__(model)
    model.config                = model_config
    model.tokenizer             = tokenizer
    model.num_candidates        = 1
    model.active_modalities     = getattr(
        model_config, "active_modalities", ["video", "ltp", "text", "music"])
    model.multimodal_prefix_len = len(model.active_modalities)
    model.rank_token_id         = tokenizer.convert_tokens_to_ids(
        getattr(model_config, "rank_special_token", "[RANK]"))
    model.llama                 = peft_llama
    model.projectors            = projectors
    model.ranking_head          = ranking_head
    model.eval()
    return model


# =============================================================================
# 評估核心
# =============================================================================

def sample_prompt_tensors(sample: dict, device: torch.device):
    prompt_len  = int(sample["prompt_len"].item())
    prompt_ids  = sample["input_ids"][:prompt_len]
    prompt_mask = sample["attention_mask"][:prompt_len]
    valid_len   = int(prompt_mask.sum().item())
    return (prompt_ids[:valid_len].unsqueeze(0).to(device),
            prompt_mask[:valid_len].unsqueeze(0).to(device))


def rank_from_scores(
    scores: torch.Tensor,
    noise_rng: np.random.Generator,
    add_noise: bool = True,
    noise_scale: float = 1e-6,
):
    scores_np   = scores.float().cpu().numpy()
    sort_scores = (scores_np + noise_rng.uniform(0, noise_scale, scores_np.shape)
                   if add_noise else scores_np)
    sorted_idx  = np.argsort(sort_scores)[::-1]
    rank        = int(np.where(sorted_idx == 0)[0][0]) + 1
    top1_idx    = int(sorted_idx[0])
    return rank, top1_idx, scores_np


@torch.no_grad()
def eval_one_condition(
    condition: str,
    model,
    test_dataset,
    all_music_features: torch.Tensor,
    all_music_ids: list,
    ltp_dict_shuffled: dict,
    ltp_dim: int,
    device: torch.device,
    pool_size: int,
    candidate_seed: int,
    max_samples: int | None,
    tiebreak_noise: bool,
    tiebreak_seed: int,
    random_seed: int,
    micro_batch: int,
    logger: logging.Logger,
) -> tuple[list[dict], dict]:
    from evaluate import pointwise_pool_scoring
    from tqdm import tqdm

    model.eval()
    n_eval     = len(test_dataset) if max_samples is None else min(max_samples, len(test_dataset))
    total_music = all_music_features.size(0)

    noise_rng  = np.random.default_rng(tiebreak_seed)
    random_rng = np.random.default_rng(random_seed)

    id_to_index: dict = {sid: i for i, sid in enumerate(all_music_ids)}
    # map: target_music_id (11 chars) → set of global indices sharing that target
    vid_to_indices: dict = {}
    for i, sid in enumerate(all_music_ids):
        vid_to_indices.setdefault(sid[:11], set()).add(i)

    rows: list[dict] = []
    n_allequal = 0

    for idx in tqdm(range(n_eval), desc=f"[{condition:8s}]"):
        sample      = test_dataset[idx]
        video_id    = sample.get("video_id", "")
        gt_pair_key = sample.get("gt_music_id", "")  # 23-char pair_key

        video_feat = sample["video_feat"].unsqueeze(0).to(device)
        text_feat  = sample["text_feat"].unsqueeze(0).to(device)

        # ── LTP：依條件切換 ──────────────────────────────────────────────
        ltp_feat = get_ltp_feat(
            sample=sample,
            condition=condition,
            ltp_dict_shuffled=ltp_dict_shuffled,
            ltp_dim=ltp_dim,
            random_rng=random_rng,
            device=device,
        )

        input_ids, attention_mask = sample_prompt_tensors(sample, device)

        gt_global_idx = id_to_index.get(gt_pair_key)
        if gt_global_idx is None:
            logger.warning("GT not in song bank: %s", gt_pair_key)
            continue

        excluded   = vid_to_indices.get(video_id, set())
        candidates = [i for i in range(total_music)
                      if i != gt_global_idx and i not in excluded]
        pool_rng   = random.Random(candidate_seed + idx)
        negatives  = pool_rng.sample(candidates, min(pool_size - 1, len(candidates)))
        pool_idx   = [gt_global_idx] + negatives
        pool_feats = all_music_features[pool_idx].to(device)

        scores = pointwise_pool_scoring(
            model=model,
            video_feat=video_feat,
            ltp_feat=ltp_feat,
            text_feat=text_feat,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_music_features=pool_feats,
            micro_batch_size=micro_batch,
            device=device,
        )

        rank, top1_pool_idx, scores_np = rank_from_scores(
            scores, noise_rng, add_noise=tiebreak_noise)
        score_range = float(scores_np.max() - scores_np.min())
        if score_range < 1e-5:
            n_allequal += 1

        rows.append({
            "sample_idx":   idx,
            "video_id":     video_id,
            "gt_pair_key":  gt_pair_key,
            "condition":    condition,
            "rank":         rank,
            "R@1":          int(rank <= 1),
            "R@5":          int(rank <= 5),
            "R@10":         int(rank <= 10),
            "pool_size":    pool_size,
            "gt_score":     float(scores_np[0]),
            "top1_score":   float(scores_np[top1_pool_idx]),
            "score_gap":    float(scores_np[top1_pool_idx] - scores_np[0]),
            "score_range":  score_range,
            "score_std":    float(scores_np.std()),
        })

    ranks = np.array([r["rank"] for r in rows], dtype=np.float64)
    summary = {
        "condition":    condition,
        "recall@1":     float(np.mean([r["R@1"]  for r in rows])),
        "recall@5":     float(np.mean([r["R@5"]  for r in rows])),
        "recall@10":    float(np.mean([r["R@10"] for r in rows])),
        "median_rank":  float(np.median(ranks)),
        "mean_rank":    float(np.mean(ranks)),
        "num_samples":  len(rows),
        "pool_size":    pool_size,
        "n_allequal":   n_allequal,
        "pct_allequal": float(n_allequal / max(len(rows), 1) * 100),
    }
    logger.info(
        "[%s] R@1=%.4f  R@5=%.4f  R@10=%.4f  MR=%.1f",
        condition, summary["recall@1"], summary["recall@5"],
        summary["recall@10"], summary["median_rank"],
    )
    return rows, summary


# =============================================================================
# CSV 輸出
# =============================================================================

def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_summary_from_csv(csv_path: str, condition: str) -> dict | None:
    """從已存在的 CSV 重建 summary，用於斷點續練。回傳 None 表示 CSV 不存在或為空。"""
    if not os.path.exists(csv_path):
        return None
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    ranks      = np.array([float(r["rank"])  for r in rows])
    r1         = np.array([float(r["R@1"])   for r in rows])
    r5         = np.array([float(r["R@5"])   for r in rows])
    r10        = np.array([float(r["R@10"])  for r in rows])
    n_allequal = sum(1 for r in rows if float(r.get("score_range", 1)) == 0.0)
    pool_size  = int(float(rows[0]["pool_size"])) if rows else POOL_SIZE
    return {
        "condition":    condition,
        "recall@1":     float(np.mean(r1)),
        "recall@5":     float(np.mean(r5)),
        "recall@10":    float(np.mean(r10)),
        "median_rank":  float(np.median(ranks)),
        "mean_rank":    float(np.mean(ranks)),
        "num_samples":  len(rows),
        "pool_size":    pool_size,
        "n_allequal":   n_allequal,
        "pct_allequal": float(n_allequal / max(len(rows), 1) * 100),
        "csv_path":     csv_path,
    }


# =============================================================================
# 比較摘要與自動解讀
# =============================================================================

def build_comparison_summary(summaries: list[dict]) -> dict:
    by_cond  = {s["condition"]: s for s in summaries}
    matched  = by_cond.get("matched",  {})
    shuffled = by_cond.get("shuffled", {})
    random_  = by_cond.get("random",   {})

    def delta(a: dict, b: dict, key: str) -> float | None:
        if a.get(key) is not None and b.get(key) is not None:
            return round(float(a[key]) - float(b[key]), 4)
        return None

    comparison = {
        "conditions": summaries,
        "delta_matched_minus_shuffled": {
            "R@1":  delta(matched, shuffled, "recall@1"),
            "R@5":  delta(matched, shuffled, "recall@5"),
            "R@10": delta(matched, shuffled, "recall@10"),
            "MR":   delta(matched, shuffled, "median_rank"),
        },
        "delta_matched_minus_random": {
            "R@1":  delta(matched, random_, "recall@1"),
            "R@5":  delta(matched, random_, "recall@5"),
            "R@10": delta(matched, random_, "recall@10"),
            "MR":   delta(matched, random_, "median_rank"),
        },
        "delta_shuffled_minus_random": {
            "R@1":  delta(shuffled, random_, "recall@1"),
            "R@5":  delta(shuffled, random_, "recall@5"),
            "R@10": delta(shuffled, random_, "recall@10"),
            "MR":   delta(shuffled, random_, "median_rank"),
        },
    }

    # 自動解讀
    r1_ms = comparison["delta_matched_minus_shuffled"].get("R@1") or 0.0
    r1_mr = comparison["delta_matched_minus_random"].get("R@1") or 0.0

    if r1_ms > 0.02 and r1_mr > 0.02:
        interp = (
            "SUPPORTS LTP VALIDITY: matched significantly outperforms both shuffled and "
            f"random (ΔR@1 vs shuffled={r1_ms:+.4f}, vs random={r1_mr:+.4f}). "
            "The model effectively utilises semantically matched long-term preference vectors."
        )
    elif r1_ms <= 0.01 and r1_mr > 0.02:
        interp = (
            f"PARTIAL: matched ≈ shuffled but both > random (ΔR@1 vs shuffled={r1_ms:+.4f}, "
            f"vs random={r1_mr:+.4f}). The model responds to LTP distribution but not to "
            "specific matched semantics. Possible explanation: LTP vectors cluster by "
            "genre/style so shuffled still conveys meaningful distributional information."
        )
    elif abs(r1_ms) <= 0.01 and abs(r1_mr) <= 0.01:
        interp = (
            "WEAK LTP USAGE: all three conditions perform similarly "
            f"(ΔR@1 vs shuffled={r1_ms:+.4f}, vs random={r1_mr:+.4f}). "
            "The model may not be effectively utilising the LTP pathway. "
            "Consider inspecting LTP projection weights or increasing LTP loss weight."
        )
    else:
        interp = (
            f"MIXED RESULT: ΔR@1 matched-shuffled={r1_ms:+.4f}, "
            f"matched-random={r1_mr:+.4f}. Manual inspection recommended."
        )

    comparison["interpretation"] = interp
    return comparison


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ltp_mode    = EXP_TO_LTP_MODE[EXP_NAME]
    active_mods = EXP_TO_MODALITIES[EXP_NAME]

    out_dir = os.path.join(BASE_DIR, "results", "main_eval", EXP_NAME, "ltp_control")
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logger(os.path.join(out_dir, f"{EXP_NAME}_ltp_control.log"))
    logger.info("=" * 72)
    logger.info(
        "LTP Control Experiment | exp=%s ckpt=%s pool=%d mods=%s",
        EXP_NAME, CKPT_NAME, POOL_SIZE, active_mods,
    )
    logger.info(
        "Conditions: matched / shuffled(seed=%d) / random(seed=%d)",
        SHUFFLE_SEED, RANDOM_SEED,
    )
    logger.info("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for model inference.")

    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer

    model_cfg = ModelConfig(
        llama_model_name=LLAMA_MODEL,
        video_dim=768, music_dim=768, text_dim=512, ltp_dim=LTP_DIM,
        num_candidates=1, active_modalities=active_mods,
        music_token_offset=3, rank_special_token="[RANK]",
    )
    train_cfg = TrainConfig(
        output_dir=os.path.join(BASE_DIR, "checkpoints", EXP_NAME),
        pointwise_eval_batch_size=POINTWISE_BATCH_SIZE,
        music_pool_size=POOL_SIZE,
    )
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", EXP_NAME, CKPT_NAME)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    # ── 載入 tokenizer & LTP ─────────────────────────────────────────────
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    ltp_dict_matched = load_ltp_dict(
        LTP_H5[ltp_mode], ltp_mode,
        cache_path=os.path.join(CACHE_DIR, "ltp"),
        logger=logger,
    )

    # ── 載入資料集 ────────────────────────────────────────────────────────
    test_dataset, all_music_features, all_music_ids, test_video_ids = build_test_data(
        model_cfg, train_cfg, tokenizer, ltp_dict_matched, logger,
    )

    # ── 建立 shuffled LTP 對照字典 ────────────────────────────────────────
    logger.info("Building shuffled LTP mapping (seed=%d)...", SHUFFLE_SEED)
    ltp_dict_shuffled = build_shuffled_ltp(
        ltp_dict=ltp_dict_matched,
        test_video_ids=test_video_ids,
        seed=SHUFFLE_SEED,
        ltp_dim=LTP_DIM,
        logger=logger,
    )
    n_cov = sum(1 for v in test_video_ids if v in ltp_dict_matched)
    logger.info("  test videos with LTP: %d/%d (%.1f%%)",
                n_cov, len(test_video_ids), 100 * n_cov / max(len(test_video_ids), 1))

    # ── 三個條件逐一評估（支援斷點續練）────────────────────────────────
    start = _dt.datetime.now()
    all_summaries: list[dict] = []

    # 若三個 CSV 均已存在，跳過模型推理直接重建摘要
    all_conditions   = ["matched", "shuffled", "random"]
    pending          = [c for c in all_conditions
                        if not os.path.exists(
                            os.path.join(out_dir, f"ltp_{c}_ranking.csv"))]
    if not pending:
        logger.info("[RESUME] All 3 conditions already have CSV output; "
                    "skipping model loading and re-building summary.")
        for condition in all_conditions:
            csv_path = os.path.join(out_dir, f"ltp_{condition}_ranking.csv")
            summary  = load_summary_from_csv(csv_path, condition)
            assert summary is not None, f"Unexpected empty CSV for condition={condition}"
            logger.info("[RESUME] %s: R@1=%.4f  R@5=%.4f  R@10=%.4f  MR=%.1f",
                        condition.upper(),
                        summary["recall@1"], summary["recall@5"],
                        summary["recall@10"], summary["median_rank"])
            all_summaries.append(summary)
    else:
        # 有未完成的 condition，才需要載入模型
        model = load_model(ckpt_dir, model_cfg, tokenizer, logger)

        for condition in all_conditions:
            csv_path = os.path.join(out_dir, f"ltp_{condition}_ranking.csv")

            # 斷點續練：CSV 已存在則跳過此 condition 的推理
            resumed = load_summary_from_csv(csv_path, condition)
            if resumed is not None:
                logger.info("[RESUME] Condition %s already done (%d samples), skipping.",
                            condition.upper(), resumed["num_samples"])
                all_summaries.append(resumed)
                continue

            logger.info("\n--- Condition: %s ---", condition.upper())
            rows, summary = eval_one_condition(
                condition=condition,
                model=model,
                test_dataset=test_dataset,
                all_music_features=all_music_features,
                all_music_ids=list(all_music_ids),
                ltp_dict_shuffled=ltp_dict_shuffled,
                ltp_dim=LTP_DIM,
                device=device,
                pool_size=POOL_SIZE,
                candidate_seed=CANDIDATE_POOL_SEED,
                max_samples=MAX_SAMPLES,
                tiebreak_noise=TIEBREAK_NOISE,
                tiebreak_seed=TIEBREAK_SEED,
                random_seed=RANDOM_SEED,
                micro_batch=POINTWISE_BATCH_SIZE,
                logger=logger,
            )
            write_csv(csv_path, rows)
            summary["csv_path"] = csv_path
            all_summaries.append(summary)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── 比較摘要 ─────────────────────────────────────────────────────────
    comparison = build_comparison_summary(all_summaries)
    comparison.update({
        "exp_name":    EXP_NAME,
        "checkpoint":  CKPT_NAME,
        "pool_size":   POOL_SIZE,
        "ltp_mode":    ltp_mode,
        "ltp_dim":     LTP_DIM,
        "shuffle_seed":  SHUFFLE_SEED,
        "random_seed":   RANDOM_SEED,
        "started_at":    start.isoformat(timespec="seconds"),
        "finished_at":   _dt.datetime.now().isoformat(timespec="seconds"),
    })

    summary_path = os.path.join(out_dir, "ltp_control_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # ── 終端摘要輸出 ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  LTP Control Experiment — {EXP_NAME}")
    print("=" * 65)
    print(f"  {'Condition':<12} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MR':>7}")
    print("  " + "-" * 43)
    for s in all_summaries:
        print(f"  {s['condition']:<12} {s['recall@1']:>7.4f} {s['recall@5']:>7.4f}"
              f" {s['recall@10']:>7.4f} {s['median_rank']:>7.1f}")
    print("  " + "-" * 43)
    dm = comparison["delta_matched_minus_shuffled"]
    dr = comparison["delta_matched_minus_random"]
    print(f"  {'Δ(M-S)':12} {dm['R@1']:>+7.4f} {dm['R@5']:>+7.4f}"
          f" {dm['R@10']:>+7.4f} {dm['MR']:>+7.1f}")
    print(f"  {'Δ(M-R)':12} {dr['R@1']:>+7.4f} {dr['R@5']:>+7.4f}"
          f" {dr['R@10']:>+7.4f} {dr['MR']:>+7.1f}")
    print(f"\n  Interpretation:\n  {comparison['interpretation']}")
    print(f"\n  Saved to: {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        sys.exit(1)
