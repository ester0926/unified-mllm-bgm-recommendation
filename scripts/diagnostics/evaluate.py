# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
evaluate.py — 評估模組（Pointwise v2 Plan B）

Plan B 主要改動（相比原 Pointwise v2）：
  - evaluate_with_music_pool：改用 prompt-only input_ids（含 [RANK]）
    原版直接傳整段 full input_ids，[RANK] 不在正確位置（在 response 中間）
    修正後用 sample["prompt_len"] 截到 [RANK] 結尾，與 validation 邏輯一致
  - pointwise_pool_evaluate_loader：新增 [RANK] sanity check
  - pointwise_pool_scoring：不需修改，模型 forward 自己找 [RANK] 位置

ranking score 路徑（Plan B）：
  input_ids（含 [RANK]）→ LLaMA → last_hidden → [RANK] hidden → ranking_head → score
  [RANK] 在 causal mask 下可看到 prefix 全部模態 + tokenized t3 prompt 文字
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 排序指標（與 v2 相同）
# ─────────────────────────────────────────────────────────────────────────────

def compute_ranking_metrics(
    scores: torch.Tensor,
    gt_idx: int = 0,
) -> Dict[str, float]:
    sorted_indices = torch.argsort(scores, descending=True)
    rank = (sorted_indices == gt_idx).nonzero(as_tuple=True)[0].item() + 1
    return {
        "recall@1":  1.0 if rank <= 1  else 0.0,
        "recall@5":  1.0 if rank <= 5  else 0.0,
        "recall@10": 1.0 if rank <= 10 else 0.0,
        "rank": rank,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise Pool Scoring（Plan B 版，不需修改 scoring 邏輯）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def pointwise_pool_scoring(
    model,
    video_feat: torch.Tensor,           # (1, video_dim)
    ltp_feat: torch.Tensor,             # (1, ltp_dim)
    text_feat: torch.Tensor,            # (1, text_dim)
    input_ids: torch.Tensor,            # (1, prompt_len)，含 [RANK]
    attention_mask: torch.Tensor,       # (1, prompt_len)
    pool_music_features: torch.Tensor,  # (pool_size, music_dim)
    micro_batch_size: int = 32,
    device: torch.device = None,
) -> torch.Tensor:
    """
    對 pool_size 首候選音樂，每首獨立計算 relevance score。

    Plan B：input_ids 含 [RANK] token（在 prompt 末尾），
    model.forward() 自動找到 [RANK] 位置並提取其隱藏狀態作為 ranking score。

    Returns:
        scores : (pool_size,)
    """
    if device is None:
        device = video_feat.device

    pool_size = pool_music_features.size(0)
    all_scores = []

    for start in range(0, pool_size, micro_batch_size):
        end     = min(start + micro_batch_size, pool_size)
        batch_k = end - start

        batch_music = pool_music_features[start:end].to(device)   # (k, 768)

        v  = video_feat.expand(batch_k, -1)
        l  = ltp_feat.expand(batch_k, -1)
        t  = text_feat.expand(batch_k, -1)
        ii = input_ids.expand(batch_k, -1)
        am = attention_mask.expand(batch_k, -1)

        music_input = batch_music.unsqueeze(1)   # (k, 1, 768)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                video_feat       = v,
                music_candidates = music_input,
                ltp_feat         = l,
                text_feat        = t,
                input_ids        = ii,
                attention_mask   = am,
                labels           = None,
                compute_gen_loss = False,
            )

        batch_scores = outputs["ranking_score"].float().cpu()
        all_scores.append(batch_scores)

    return torch.cat(all_scores, dim=0)   # (pool_size,)


# ─────────────────────────────────────────────────────────────────────────────
# 500-pool 正式評估（Plan B 版）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_music_pool(
    model,
    test_dataset,
    all_music_features: torch.Tensor,   # (M, music_dim)
    all_music_ids: List[str],
    device: torch.device,
    train_config,
    model_config,
    tokenizer=None,
) -> Dict[str, float]:
    """
    正式 500-pool 測試集評估（Plan B，Pointwise，與 MuseChat 公平比較）。

    Plan B 修正：
      - 使用 prompt-only input_ids（含 [RANK]），不傳全序列
      - 與 pointwise_pool_evaluate_loader validation 邏輯完全一致
      - [RANK] 在 prompt 末尾，model.forward() 正確找到並提取 ranking score
    """
    model.eval()
    pool_size      = train_config.music_pool_size
    micro_batch_sz = getattr(train_config, "pointwise_eval_batch_size", 32)
    M = all_music_features.size(0)

    all_recalls_1, all_recalls_5, all_recalls_10, all_ranks = [], [], [], []

    for idx in tqdm(range(len(test_dataset)),
                    desc=f"Test ({pool_size}-pool | pointwise | Plan B)"):
        sample = test_dataset[idx]

        video_feat = sample["video_feat"].unsqueeze(0).to(device)
        ltp_feat   = sample["ltp_feat"].unsqueeze(0).to(device)
        text_feat  = sample["text_feat"].unsqueeze(0).to(device)

        # ★ Plan B 修正：prompt-only slicing（含 [RANK]）
        # 原版直接用 full input_ids，[RANK] 位置不在 prompt 末尾
        prompt_len     = int(sample["prompt_len"].item())
        full_input_ids = sample["input_ids"]          # (max_seq_len,)
        full_attn_mask = sample["attention_mask"]     # (max_seq_len,)

        # 截到 prompt_len（含 [RANK]），再去掉 padding
        prompt_ids  = full_input_ids[:prompt_len]     # (prompt_len,)
        prompt_mask = full_attn_mask[:prompt_len]     # (prompt_len,)
        valid_len   = int(prompt_mask.sum().item())
        input_ids      = prompt_ids[:valid_len].unsqueeze(0).to(device)
        attention_mask = prompt_mask[:valid_len].unsqueeze(0).to(device)

        gt_music_id = sample.get("gt_music_id", "")
        video_id    = sample.get("video_id", "")

        # ── 建構 500 首音樂池（GT 固定在 index 0）────────────────────────────
        excl = {i for i, sid in enumerate(all_music_ids) if sid[:11] == video_id}
        gt_global_idx = next(
            (i for i, sid in enumerate(all_music_ids) if sid == gt_music_id), 0
        )
        candidates = [i for i in range(M) if i not in excl and i != gt_global_idx]
        rng = random.Random(20260315 + idx)
        negatives = rng.sample(candidates, min(pool_size - 1, len(candidates)))
        pool_idx  = [gt_global_idx] + negatives
        pool_feats = all_music_features[pool_idx].to(device)

        # ── Pointwise scoring ──────────────────────────────────────────────────
        scores = pointwise_pool_scoring(
            model=model,
            video_feat=video_feat,
            ltp_feat=ltp_feat,
            text_feat=text_feat,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_music_features=pool_feats,
            micro_batch_size=micro_batch_sz,
            device=device,
        )

        m = compute_ranking_metrics(scores, gt_idx=0)
        all_recalls_1.append(m["recall@1"])
        all_recalls_5.append(m["recall@5"])
        all_recalls_10.append(m["recall@10"])
        all_ranks.append(m["rank"])

    results = {
        "recall@1":    float(np.mean(all_recalls_1)),
        "recall@5":    float(np.mean(all_recalls_5)),
        "recall@10":   float(np.mean(all_recalls_10)),
        "median_rank": float(np.median(all_ranks)),
        "mean_rank":   float(np.mean(all_ranks)),
        "num_samples": len(all_ranks),
        "pool_size":   pool_size,
        "scoring":     "pointwise Plan B ([RANK] token readout)",
    }
    logger.info(
        f"[Test/{pool_size}-pool|Plan B] "
        f"R@1={results['recall@1']:.4f} | "
        f"R@5={results['recall@5']:.4f} | "
        f"R@10={results['recall@10']:.4f} | "
        f"MR={results['median_rank']:.1f}"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Validation Loop 評估（供 train.py 訓練中呼叫）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def pointwise_pool_evaluate_loader(
    model,
    data_loader: DataLoader,
    all_music_features: torch.Tensor,
    all_music_ids: List[str],
    device: torch.device,
    train_config,
    model_config,
    val_pool_size: int = 50,
    max_samples: int = 500,
    val_seed: int = 20260315,
) -> Dict[str, float]:
    """
    訓練中的 validation 評估（Plan B）。

    Plan B 不需修改 prompt-only 邏輯，因為 prompt_len 已包含 [RANK]，
    截出的 prompt_ids 末尾自然是 [RANK]，model.forward() 會正確找到它。

    額外加入 [RANK] 存在性檢查，方便 debug tokenizer 設定。
    """
    import hashlib
    from collections import defaultdict

    # ── 關閉 gradient checkpointing（eval 加速）──────────────────────────────
    gc_was_enabled = False
    try:
        if hasattr(model.llama, "gradient_checkpointing_disable"):
            model.llama.gradient_checkpointing_disable()
            gc_was_enabled = True
    except Exception:
        pass

    model.eval()
    micro_batch_sz = getattr(train_config, "pointwise_eval_batch_size", 50)
    M = all_music_features.size(0)

    # 預建查找字典
    music_id_to_idx = {sid: j for j, sid in enumerate(all_music_ids)}
    video_to_indices = defaultdict(set)
    for j, sid in enumerate(all_music_ids):
        video_to_indices[sid[:11]].add(j)

    all_recalls_1, all_recalls_5, all_recalls_10, all_ranks = [], [], [], []
    samples_done = 0
    rank_token_warning_shown = False   # 只顯示一次 [RANK] 缺失警告

    for batch in tqdm(data_loader, desc=f"Val ({val_pool_size}-pool | Plan B)", leave=False):
        B = batch["video_feat"].size(0)

        for i in range(B):
            gt_music_id = batch["gt_music_id"][i] if isinstance(batch.get("gt_music_id"), list) else ""
            video_id    = batch["video_id"][i]    if isinstance(batch.get("video_id"),    list) else ""

            video_feat = batch["video_feat"][i:i+1].to(device)
            ltp_feat   = batch["ltp_feat"][i:i+1].to(device)
            text_feat  = batch["text_feat"][i:i+1].to(device)

            # ── Prompt-only slicing（Plan B：含 [RANK]）────────────────────
            prompt_len     = int(batch["prompt_len"][i].item())
            full_input_ids = batch["input_ids"][i]
            full_attn_mask = batch["attention_mask"][i]

            prompt_ids  = full_input_ids[:prompt_len]
            prompt_mask = full_attn_mask[:prompt_len]
            valid_len   = int(prompt_mask.sum().item())
            prompt_ids  = prompt_ids[:valid_len].unsqueeze(0).to(device)
            prompt_mask = prompt_mask[:valid_len].unsqueeze(0).to(device)

            # ★ [RANK] 存在性檢查（第一筆時驗證，確認 tokenizer 設定正確）
            if not rank_token_warning_shown and hasattr(model, "rank_token_id"):
                rank_in_prompt = (prompt_ids == model.rank_token_id).any()
                if not rank_in_prompt:
                    logger.warning(
                        "[Val] [RANK] token (id=%d) not found in prompt_ids. "
                        "Check tokenizer.add_special_tokens() and dataset.build_prompt(). "
                        "P_ltp/text ablation Δ will remain 0 if [RANK] is missing.",
                        model.rank_token_id,
                    )
                    rank_token_warning_shown = True

            # ── O(1) 查找 GT index ─────────────────────────────────────────
            gt_global_idx = music_id_to_idx.get(gt_music_id, 0)

            # ── stable hash ───────────────────────────────────────────────
            stable_key  = f"{video_id}__{gt_music_id}".encode("utf-8")
            stable_hash = int(hashlib.md5(stable_key).hexdigest()[:8], 16)
            stable_seed = val_seed + (stable_hash % 10 ** 6)
            rng_neg     = random.Random(stable_seed)

            # ── pool 建構 ──────────────────────────────────────────────────
            excl       = video_to_indices.get(video_id, set())
            candidates = [j for j in range(M) if j not in excl and j != gt_global_idx]
            negatives  = rng_neg.sample(candidates, min(val_pool_size - 1, len(candidates)))
            pool_idx   = [gt_global_idx] + negatives
            pool_feats = all_music_features[pool_idx].to(device)

            scores = pointwise_pool_scoring(
                model=model,
                video_feat=video_feat,
                ltp_feat=ltp_feat,
                text_feat=text_feat,
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                pool_music_features=pool_feats,
                micro_batch_size=micro_batch_sz,
                device=device,
            )

            m = compute_ranking_metrics(scores, gt_idx=0)
            all_recalls_1.append(m["recall@1"])
            all_recalls_5.append(m["recall@5"])
            all_recalls_10.append(m["recall@10"])
            all_ranks.append(m["rank"])
            samples_done += 1

            del video_feat, ltp_feat, text_feat, prompt_ids, prompt_mask, pool_feats, scores

        torch.cuda.empty_cache()

    # gradient checkpointing 恢復
    if gc_was_enabled:
        try:
            model.llama.gradient_checkpointing_enable()
        except Exception:
            pass

    return {
        "recall@1":    float(np.mean(all_recalls_1)),
        "recall@5":    float(np.mean(all_recalls_5)),
        "recall@10":   float(np.mean(all_recalls_10)),
        "median_rank": float(np.median(all_ranks)),
        "pool_size":   val_pool_size,
        "num_samples": samples_done,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BERTScore（與 v2 相同）
# ─────────────────────────────────────────────────────────────────────────────

def compute_bert_score(
    generated: list,
    references: list,
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> Dict[str, float]:
    if not generated or not references:
        return {"bertscore_f1": 0.0}
    try:
        import importlib
        hf_evaluate = importlib.import_module("evaluate")
        bs = hf_evaluate.load("bertscore")
        res = bs.compute(
            predictions=generated,
            references=references,
            model_type=model_type,
            lang="en",
        )
        return {
            "bertscore_precision": float(np.mean(res["precision"])),
            "bertscore_recall":    float(np.mean(res["recall"])),
            "bertscore_f1":        float(np.mean(res["f1"])),
        }
    except Exception as e:
        logger.warning(f"BERTScore 失敗: {e}")
        return {"bertscore_f1": 0.0}