"""
用途：使用較相似的 hard negative 候選音樂執行 500-pool 評估。
輸入：已訓練 checkpoint、測試集特徵、候選 pool 與 LTP/cache 資料。
輸出：ranking、generation、指標摘要或逐筆評估檔。
執行：建議在 repo 根目錄執行，必要資料請先由 Zenodo 解壓到對應資料夾。
"""

from pathlib import Path
import sys

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = PROJECT_ROOT / "scripts" / "diagnostics"

for _p in [str(PROJECT_ROOT), str(DIAGNOSTICS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


import csv
import datetime as _dt
import gc
import json
import logging
import os
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch


# =============================================================================
# 路徑設定：若資料夾位置不同，只需要調整本區塊
# =============================================================================

BASE_DIR    = str(PROJECT_ROOT)
H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
LLAMA_MODEL = "meta-llama/Llama-2-7b-hf"

SPECIAL_TOKENS = ["[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"]

# =============================================================================
# 使用前可調整的設定
# =============================================================================
#
# EXP_NAME：要評估的實驗 checkpoint，通常使用 exp_01。
# hard-negative 分析通常以最佳模型最有參考價值。
# POOL_SIZE：候選池大小，包含 1 首 GT 與其餘 hard negatives；論文設定為 500。
# TOP_K_HARD：先取音訊特徵最相近的前 K 首音樂。
# 若 TOP_K_HARD 大於 POOL_SIZE - 1，會從前 K 首中抽樣。
# 例如 TOP_K_HARD=1000、POOL_SIZE=500 時，會從前 1000 首抽 499 首。
# 若要嚴格使用前 N 首，請設為 POOL_SIZE - 1。
# HARDNEG_SEED：當需要抽樣時，用此 seed 固定抽樣結果。
# 若 TOP_K_HARD 等於 POOL_SIZE - 1，這個 seed 不會影響結果。
# MAX_SAMPLES：None 表示跑完整測試集；小數字可用於快速檢查。
# =============================================================================

EXP_NAME             = "exp_01"
CKPT_NAME            = "best"
POOL_SIZE            = 500
TOP_K_HARD           = 499      # must be >= POOL_SIZE - 1
HARDNEG_SEED         = 42
POINTWISE_BATCH_SIZE = 32
INJECT_TITLE         = True
TIEBREAK_NOISE       = True
TIEBREAK_SEED        = 42
MAX_SAMPLES          = None

LTP_H5 = {
    "hybrid":        str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors.h5"),
    "explicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_explicit_only.h5"),
    "implicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_implicit_only.h5"),
}

EXP_TO_LTP_MODE = {
    "exp_01": "hybrid",
    "exp_02": "explicit_only",
    "exp_03": "implicit_only",
    "exp_04": "hybrid",
    "exp_05": "hybrid",
    "exp_06": "hybrid",
    "exp_07": "hybrid",
}

EXP_TO_MODALITIES = {
    "exp_01": ["video", "ltp", "text", "music"],
    "exp_02": ["video", "ltp", "text", "music"],
    "exp_03": ["video", "ltp", "text", "music"],
    "exp_04": ["video", "text", "music"],
    "exp_05": ["ltp", "text", "music"],
    "exp_06": ["video", "ltp", "music"],
    "exp_07": ["video", "ltp", "text"],
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(log_path: str):
    logger = logging.getLogger("eval_hardneg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# =============================================================================
# 音樂特徵庫與 hard-negative 索引
# =============================================================================

def load_song_bank_normed(cache_dir: str, logger):
    """讀取 song_bank.npy，並回傳 L2 正規化後的音樂特徵。"""
    npy = os.path.join(cache_dir, "song_bank.npy")
    ids = os.path.join(cache_dir, "song_bank_ids.json")
    if not os.path.exists(npy) or not os.path.exists(ids):
        raise FileNotFoundError(
            f"song_bank cache not found at {cache_dir}. "
            "Run build_song_bank() (dataset.py) first."
        )
    arr = np.load(npy).astype(np.float32)          # (N, 768)
    with open(ids, encoding="utf-8") as f:
        pair_keys = json.load(f)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    arr_normed = arr / norms                        # (N, 768), unit vectors
    logger.info("[SongBank] Loaded %d pairs (768-dim), L2-normalised.", len(pair_keys))
    return arr_normed, pair_keys


def build_hardneg_pool(
    gt_idx: int,
    video_id: str,
    song_bank_normed: np.ndarray,
    pair_keys: list,
    top_k: int,
    pool_size: int,
    hardneg_rng: random.Random,
) -> tuple[list[int], bool]:
    """
    回傳 pool_size - 1 個 hard-negative 索引；回傳清單不包含 gt_idx。

    參數：
        gt_idx：GT pair 在 song_bank 中的索引。
        video_id：查詢影片 ID，也就是測試樣本 pair_key[:11]。
        song_bank_normed：L2 正規化後的 song bank 特徵，形狀為 (N, 768)。
        pair_keys：與 song_bank 列順序對齊的 pair_key 字串清單。
        top_k：過濾前先取出的最近鄰數量。
        pool_size：包含 1 首 GT 與其餘 hard negatives 的候選池大小。
        hardneg_rng：從 top_k 抽樣時使用的隨機產生器。

    回傳：
        (neg_indices, used_random_fill)
        neg_indices：長度為 pool_size - 1 的負樣本索引清單。
        used_random_fill：有效 hard negatives 不足時為 True。
    """
    n_neg_needed     = pool_size - 1
    gt_target_mid    = pair_keys[gt_idx][:11]   # pair_key[:11] = target_music_id = GT = video_id

    # Cosine similarity：L2 正規化後以內積計算相似度
    query_vec = song_bank_normed[gt_idx]    # (768,)
    sims      = song_bank_normed @ query_vec  # (N,)

    # 依相似度由高到低排序，並跳過 GT 自己
    ranked = np.argsort(sims)[::-1]         # 由高到低

    # 收集前 top_k 個有效 hard negatives
    # 排除規則（pair_key = {target_music_id}_{candidate_music_id}）：
    #   (a) 與 GT 完全相同的 pair
    #   (b) target_music_id 與 GT 相同的 pair
    #       這些 pair 的 song_bank 特徵相同，
    #       只是同一首 GT 音樂的不同 pair_key，因此必須排除。
    valid_hard = []
    for i in ranked:
        if len(valid_hard) >= top_k:
            break
        pk = pair_keys[i]
        if i == gt_idx:
            continue                         # (a) 與 GT 完全相同的 pair
        if pk[:11] == gt_target_mid:
            continue                         # (b) same target_music (same GT, different candidate)
        valid_hard.append(int(i))

    used_random_fill = len(valid_hard) < n_neg_needed

    if len(valid_hard) >= n_neg_needed:
        if top_k == n_neg_needed:
            selected = valid_hard[:n_neg_needed]
        else:
            # 從 top_k hard negatives 中抽出 pool_size-1 首
            selected = hardneg_rng.sample(valid_hard, n_neg_needed)
    else:
        # 備用處理：使用有效 hard negatives，不足時以其他候選補齊
        # 以目前資料量來看，這種補候選情況應該很少發生
        used_as_hard = set(valid_hard)
        fill_pool = [
            int(i) for i in range(len(pair_keys))
            if i != gt_idx
            and pair_keys[i][:11] != gt_target_mid
            and i not in used_as_hard
        ]
        n_fill = n_neg_needed - len(valid_hard)
        fill   = hardneg_rng.sample(fill_pool, min(n_fill, len(fill_pool)))
        selected = valid_hard + fill

    return selected, used_random_fill


# =============================================================================
# LTP
# =============================================================================

def load_ltp_dict(h5_path, mode, cache_path=None, logger=None):
    if cache_path:
        npy = cache_path + f"_{mode}.npy"
        ids = cache_path + f"_{mode}_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids, encoding="utf-8") as f:
                video_ids = json.load(f)
            out = {v: arr[i] for i, v in enumerate(video_ids)}
            if logger:
                logger.info("[LTP] cache loaded: %d items (%s)", len(out), mode)
            return out
    if logger:
        logger.info("[LTP] loading HDF5: %s", h5_path)
    out = {}
    with h5py.File(h5_path, "r") as f:
        grp: h5py.Group = f["preference_vectors"]  # type: ignore[assignment]
        for k in grp.keys():
            out[k] = grp[k][()].astype(np.float32)  # type: ignore[index, union-attr]
    return out


# =============================================================================
# 資料集與模型載入，與 run_eval_500pool_detailed.py 相同
# =============================================================================

def build_test_data(model_config, train_config, tokenizer, ltp_dict, logger):
    from dataset import (
        UnifiedMLLMDataset,
        build_pair_index,
        build_song_bank,
        load_conversation_map,
        split_by_video_id,
    )
    pair_index  = build_pair_index(H5_DIR, cache_path=os.path.join(CACHE_DIR, "pair_index.json"))
    conv_map    = load_conversation_map(JSON_DIR)
    song_bank_np, song_ids = build_song_bank(pair_index, cache_path=os.path.join(CACHE_DIR, "song_bank"))
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
    logger.info("Test pairs=%d | song bank=%d", len(test_pairs), len(song_ids))
    return test_dataset, torch.tensor(song_bank_np, dtype=torch.float32), song_ids, conv_map


def load_model(ckpt_dir, model_config, tokenizer, logger):
    import torch.nn as nn
    from peft import PeftModel
    from transformers import LlamaForCausalLM
    from models.projectors import MultimodalProjectors
    from models.unified_mllm import UnifiedMLLM

    logger.info("Loading checkpoint: %s", ckpt_dir)
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    logger.info("Tokenizer added %d special tokens; vocab=%d", added, len(tokenizer))

    base = LlamaForCausalLM.from_pretrained(
        model_config.llama_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base.resize_token_embeddings(len(tokenizer))
    peft_llama = PeftModel.from_pretrained(base, ckpt_dir, torch_dtype=torch.bfloat16)
    peft_llama.eval()
    # 使用 getattr 避免型別檢查工具對 hasattr 的誤判
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
                   map_location="cuda:0", weights_only=False)
    )
    projectors = projectors.to(torch.bfloat16).cuda().eval()

    ranking_head = torch.nn.Sequential(
        torch.nn.LayerNorm(model_config.llama_hidden_dim),
        torch.nn.Linear(model_config.llama_hidden_dim, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.0),
        torch.nn.Linear(256, 1),
    )
    ranking_head.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "ranking_head.pt"),
                   map_location="cuda:0", weights_only=False)
    )
    ranking_head = ranking_head.to(torch.bfloat16).cuda().eval()

    model = UnifiedMLLM.__new__(UnifiedMLLM)
    torch.nn.Module.__init__(model)
    model.config               = model_config
    model.tokenizer            = tokenizer
    model.num_candidates       = 1
    model.active_modalities    = getattr(model_config, "active_modalities",
                                         ["video", "ltp", "text", "music"])
    model.multimodal_prefix_len = len(model.active_modalities)
    model.rank_token_id        = tokenizer.convert_tokens_to_ids(
                                     getattr(model_config, "rank_special_token", "[RANK]"))
    model.llama         = peft_llama
    model.projectors    = projectors
    model.ranking_head  = ranking_head
    model.eval()
    return model


# =============================================================================
# Prompt 輔助函式，保留與 detailed eval 一致的版本
# =============================================================================

def sample_prompt_tensors(sample, device):
    prompt_len    = int(sample["prompt_len"].item())
    full_input_ids = sample["input_ids"]
    full_attn_mask = sample["attention_mask"]
    prompt_ids    = full_input_ids[:prompt_len]
    prompt_mask   = full_attn_mask[:prompt_len]
    valid_len     = int(prompt_mask.sum().item())
    return (
        prompt_ids[:valid_len].unsqueeze(0).to(device),
        prompt_mask[:valid_len].unsqueeze(0).to(device),
    )



# =============================================================================
# 使用 hard-negative 候選池進行 ranking 評估
# =============================================================================

def rank_from_scores(scores, add_noise=True, rng=None, noise_scale=1e-6):
    scores_np      = scores.float().cpu().numpy()
    scores_for_sort = scores_np
    if add_noise:
        if rng is None:
            rng = np.random.default_rng(42)
        scores_for_sort = scores_np + rng.uniform(0, noise_scale, size=scores_np.shape)
    sorted_indices = np.argsort(scores_for_sort)[::-1]
    rank           = int(np.where(sorted_indices == 0)[0][0]) + 1
    top1_pool_idx  = int(sorted_indices[0])
    return rank, top1_pool_idx, scores_np


@torch.no_grad()
def eval_ranking_hardneg(
    model,
    test_dataset,
    tokenizer,
    all_music_features,
    all_music_ids,
    song_bank_normed,
    device,
    active_modalities,
    conv_t3,
    conv_t4,
    train_cfg,
    pool_size,
    top_k_hard,
    hardneg_seed,
    max_samples,
    tiebreak_noise,
    tiebreak_seed,
    logger,
):
    from evaluate import pointwise_pool_scoring
    from tqdm import tqdm

    model.eval()
    n_eval        = len(test_dataset) if max_samples is None else min(max_samples, len(test_dataset))
    micro_batch   = getattr(train_cfg, "pointwise_eval_batch_size", 32)
    tiebreak_rng  = np.random.default_rng(tiebreak_seed)
    hardneg_rng   = random.Random(hardneg_seed)

    id_to_index   = {sid: i for i, sid in enumerate(all_music_ids)}

    rows            = []
    n_allequal      = 0
    n_random_fill   = 0

    for idx in tqdm(range(n_eval), desc=f"HardNeg Ranking ({pool_size}-pool)"):
        sample      = test_dataset[idx]
        video_id    = sample.get("video_id", "")
        gt_pair_key = sample.get("gt_music_id", "")  # 23-char pair_key

        video_feat = sample["video_feat"].unsqueeze(0).to(device)
        ltp_feat   = sample["ltp_feat"].unsqueeze(0).to(device)
        text_feat  = sample["text_feat"].unsqueeze(0).to(device)

        # Ranking 階段使用 dataset 已 tokenize 的 prompt，不注入歌名；
        # 這與 run_eval_500pool_detailed.py 的設定一致。
        t3_text = conv_t3.get(video_id, "")   # kept for future generation step
        _ = conv_t4.get(video_id, "")         # t4_ref unused in ranking; silence lint
        input_ids, attention_mask = sample_prompt_tensors(sample, device)

        gt_global_idx = id_to_index.get(gt_pair_key)
        if gt_global_idx is None:
            logger.warning("GT pair_key not found in song bank: %s (skipping)", gt_pair_key)
            continue

        # 建立 hard-negative 候選池
        # video_id = pair_key[:11] = target_music_id，也就是 GT 音樂
        # build_hardneg_pool 會排除 target_music_id 與 video_id 相同的 pair
        neg_indices, used_fill = build_hardneg_pool(
            gt_idx           = gt_global_idx,
            video_id         = video_id,  # = target_music_id = pair_key[:11]
            song_bank_normed = song_bank_normed,
            pair_keys        = all_music_ids,
            top_k            = top_k_hard,
            pool_size        = pool_size,
            hardneg_rng      = hardneg_rng,
        )
        if used_fill:
            n_random_fill += 1
            logger.warning(
                "Sample %d (video=%s): not enough hard negatives; used random fill.", idx, video_id
            )

        # GT 固定放在 index 0，與 random-pool 評估一致
        pool_idx   = [gt_global_idx] + neg_indices
        pool_feats = all_music_features[pool_idx].to(device)

        scores = pointwise_pool_scoring(
            model                = model,
            video_feat           = video_feat,
            ltp_feat             = ltp_feat,
            text_feat            = text_feat,
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pool_music_features  = pool_feats,
            micro_batch_size     = micro_batch,
            device               = device,
        )

        rank, top1_pool_idx, scores_np = rank_from_scores(
            scores, add_noise=tiebreak_noise, rng=tiebreak_rng
        )
        score_range = float(scores_np.max() - scores_np.min())
        score_std   = float(scores_np.std())
        if score_range < 1e-5:
            n_allequal += 1

        top1_global_idx = pool_idx[top1_pool_idx]

        # GT 在 hard-negative 候選池中的最近鄰排名
        # 用來表示 GT 與最相似干擾候選之間的距離
        gt_vec = song_bank_normed[gt_global_idx]
        neg_sims = song_bank_normed[neg_indices] @ gt_vec   # (499,)
        # 有多少比例的 hard negatives 比 GT 更接近查詢特徵
        # 越接近 1.0 的負樣本代表越困難。
        avg_neg_sim = float(neg_sims.mean())
        max_neg_sim = float(neg_sims.max())

        rows.append({
            "sample_idx":               idx,
            "video_id":                 video_id,       # = target_music_id = pair_key[:11]
            "gt_pair_key":              gt_pair_key,    # full 23-char pair_key
            "gt_target_music_id":       gt_pair_key[:11],   # the GT music (= video_id)
            "gt_candidate_music_id":    gt_pair_key[12:],   # the training negative
            "top1_pair_key":            all_music_ids[top1_global_idx],
            "top1_target_music_id":     all_music_ids[top1_global_idx][:11],
            "top1_is_gt":               int(rank == 1),
            "rank":                     rank,
            "R@1":                      int(rank <= 1),
            "R@5":                      int(rank <= 5),
            "R@10":                     int(rank <= 10),
            "pool_size":                pool_size,
            "pool_type":                "hard_negative",
            "top_k_hard":               top_k_hard,
            "used_random_fill":         int(used_fill),
            "gt_score":                 float(scores_np[0]),
            "top1_score":               float(scores_np[top1_pool_idx]),
            "score_gap_top1_minus_gt":  float(scores_np[top1_pool_idx] - scores_np[0]),
            "score_range":              score_range,
            "score_std":                score_std,
            "n_equal_to_gt_score":      int(np.isclose(scores_np, scores_np[0], atol=1e-7).sum()),
            "avg_neg_cosim":            avg_neg_sim,
            "max_neg_cosim":            max_neg_sim,
        })

    ranks = np.array([r["rank"] for r in rows], dtype=np.float64)
    summary = {
        "recall@1":          float(np.mean([r["R@1"]   for r in rows])),
        "recall@5":          float(np.mean([r["R@5"]   for r in rows])),
        "recall@10":         float(np.mean([r["R@10"]  for r in rows])),
        "median_rank":       float(np.median(ranks)),
        "mean_rank":         float(np.mean(ranks)),
        "num_samples":       len(rows),
        "pool_size":         pool_size,
        "pool_type":         "hard_negative_cosim",
        "top_k_hard":        top_k_hard,
        "hardneg_seed":      hardneg_seed,
        "scoring":           "pointwise [RANK] token + cosine-sim hard negatives",
        "tiebreak_noise":    bool(tiebreak_noise),
        "tiebreak_seed":     int(tiebreak_seed),
        "n_allequal_scores": int(n_allequal),
        "pct_allequal":      float(n_allequal / max(len(rows), 1) * 100),
        "n_random_fill":     int(n_random_fill),
        "avg_score_range":   float(np.mean([r["score_range"]    for r in rows])),
        "avg_score_std":     float(np.mean([r["score_std"]      for r in rows])),
        "avg_neg_cosim":     float(np.mean([r["avg_neg_cosim"]  for r in rows])),
        "max_neg_cosim":     float(np.mean([r["max_neg_cosim"]  for r in rows])),
    }
    logger.info(
        "HardNeg Ranking: R@1=%.4f R@5=%.4f R@10=%.4f MR=%.1f allequal=%.2f%% "
        "avg_neg_cosim=%.4f max_neg_cosim=%.4f",
        summary["recall@1"], summary["recall@5"], summary["recall@10"],
        summary["median_rank"], summary["pct_allequal"],
        summary["avg_neg_cosim"], summary["max_neg_cosim"],
    )
    return rows, summary


# =============================================================================
# 參考對話對照表
# =============================================================================

def load_reference_maps():
    conv_t3 = {}
    conv_t4 = {}
    for jf in Path(JSON_DIR).glob("**/*.json"):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            convs = data.get("conversations", [])
            if len(convs) < 4:
                continue
            video_id        = jf.parent.name
            conv_t3[video_id] = convs[2].get("value", "").strip()
            conv_t4[video_id] = convs[3].get("value", "").strip()
        except Exception:
            continue
    return conv_t3, conv_t4


# =============================================================================
# 與既有 random-pool 結果比較
# =============================================================================

def load_random_pool_summary(base_dir: str, exp_name: str, ckpt: str, pool_size: int) -> dict:
    """
    嘗試讀取既有 random-pool summary JSON，用於結果比較。
    若找不到檔案則回傳空字典。
    """
    # run_eval_500pool_detailed.py 的標準輸出路徑
    summary_path = os.path.join(
        base_dir, "checkpoints", exp_name, "detailed_eval",
        f"{exp_name}_{ckpt}_{pool_size}pool_summary.json",
    )
    # 也檢查 results/main_eval，也就是 run_eval_500pool.py 的輸出位置
    alt_path = os.path.join(
        base_dir, "results", "main_eval", exp_name, "detailed_eval",
        f"{exp_name}_{ckpt}_{pool_size}pool_ranking_samples.csv",
    )
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("ranking", {})

    # 備用處理：若沒有 summary JSON，改從 CSV 計算
    if os.path.exists(alt_path):
        import csv as _csv
        r1, r5, r10, ranks = [], [], [], []
        with open(alt_path, encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                r1.append(int(row.get("R@1", 0)))
                r5.append(int(row.get("R@5", 0)))
                r10.append(int(row.get("R@10", 0)))
                ranks.append(int(row.get("rank", 0)))
        if ranks:
            return {
                "recall@1":    float(np.mean(r1)),
                "recall@5":    float(np.mean(r5)),
                "recall@10":   float(np.mean(r10)),
                "median_rank": float(np.median(ranks)),
                "mean_rank":   float(np.mean(ranks)),
                "num_samples": len(ranks),
                "source":      alt_path,
            }
    return {}


# =============================================================================
# CSV WRITER
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 主程式入口
# =============================================================================

def main():
    exp_name       = EXP_NAME
    ltp_mode       = EXP_TO_LTP_MODE[exp_name]
    active_mods    = EXP_TO_MODALITIES[exp_name]

    # 輸出資料夾
    out_dir = os.path.join(BASE_DIR, "results", "main_eval", exp_name, "hardneg_eval")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(os.path.join(out_dir, f"{exp_name}_hardneg_eval.log"))

    logger.info("=" * 72)
    logger.info(
        "Hard-negative 500-pool eval | exp=%s ckpt=%s pool=%d top_k=%d modalities=%s",
        exp_name, CKPT_NAME, POOL_SIZE, TOP_K_HARD, active_mods,
    )
    logger.info("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for model inference.")

    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer

    model_cfg = ModelConfig(
        llama_model_name   = LLAMA_MODEL,
        video_dim          = 768,
        music_dim          = 768,
        text_dim           = 512,
        ltp_dim            = 256,
        num_candidates     = 1,
        active_modalities  = active_mods,
        music_token_offset = 3,
        rank_special_token = "[RANK]",
    )
    train_cfg = TrainConfig(
        output_dir              = os.path.join(BASE_DIR, "checkpoints", exp_name),
        pointwise_eval_batch_size = POINTWISE_BATCH_SIZE,
        music_pool_size         = POOL_SIZE,
    )

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", exp_name, CKPT_NAME)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    # 載入已 L2 正規化的 song bank，用於 cosine similarity
    song_bank_normed, sb_pair_keys = load_song_bank_normed(CACHE_DIR, logger)

    # 載入 tokenizer、LTP 與測試資料集
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    ltp_dict = load_ltp_dict(
        LTP_H5[ltp_mode], ltp_mode,
        cache_path=os.path.join(CACHE_DIR, "ltp"), logger=logger,
    )
    test_dataset, all_music_features, all_music_ids, _ = build_test_data(
        model_cfg, train_cfg, tokenizer, ltp_dict, logger
    )

    # 確認 song_bank_normed 與 all_music_ids 順序一致
    if sb_pair_keys != list(all_music_ids):
        # 依 all_music_ids 順序重建正規化矩陣
        logger.warning(
            "song_bank_normed order differs from all_music_ids; re-indexing."
        )
        pk_to_idx = {pk: i for i, pk in enumerate(sb_pair_keys)}
        reorder   = [pk_to_idx[pk] for pk in all_music_ids]
        song_bank_normed = song_bank_normed[reorder]

    conv_t3, conv_t4 = load_reference_maps()
    model = load_model(ckpt_dir, model_cfg, tokenizer, logger)

    start = _dt.datetime.now()

    ranking_rows, ranking_summary = eval_ranking_hardneg(
        model               = model,
        test_dataset        = test_dataset,
        tokenizer           = tokenizer,
        all_music_features  = all_music_features,
        all_music_ids       = list(all_music_ids),
        song_bank_normed    = song_bank_normed,
        device              = device,
        active_modalities   = active_mods,
        conv_t3             = conv_t3,
        conv_t4             = conv_t4,
        train_cfg           = train_cfg,
        pool_size           = POOL_SIZE,
        top_k_hard          = TOP_K_HARD,
        hardneg_seed        = HARDNEG_SEED,
        max_samples         = MAX_SAMPLES,
        tiebreak_noise      = TIEBREAK_NOISE,
        tiebreak_seed       = TIEBREAK_SEED,
        logger              = logger,
    )

    # 與 random-pool baseline 比較
    random_summary = load_random_pool_summary(BASE_DIR, exp_name, CKPT_NAME, POOL_SIZE)

    comparison = {}
    if random_summary:
        comparison = {
            "random_pool_R@1":          random_summary.get("recall@1"),
            "random_pool_R@5":          random_summary.get("recall@5"),
            "random_pool_R@10":         random_summary.get("recall@10"),
            "random_pool_MR":           random_summary.get("median_rank"),
            "hardneg_pool_R@1":         ranking_summary["recall@1"],
            "hardneg_pool_R@5":         ranking_summary["recall@5"],
            "hardneg_pool_R@10":        ranking_summary["recall@10"],
            "hardneg_pool_MR":          ranking_summary["median_rank"],
            "delta_R@1_hardneg_minus_random":  (
                ranking_summary["recall@1"] - random_summary["recall@1"]
                if random_summary.get("recall@1") is not None else None
            ),
            "delta_R@5_hardneg_minus_random":  (
                ranking_summary["recall@5"] - random_summary["recall@5"]
                if random_summary.get("recall@5") is not None else None
            ),
            "delta_R@10_hardneg_minus_random": (
                ranking_summary["recall@10"] - random_summary["recall@10"]
                if random_summary.get("recall@10") is not None else None
            ),
        }
        logger.info(
            "Pool comparison | Random: R@1=%.4f R@5=%.4f R@10=%.4f | "
            "HardNeg: R@1=%.4f R@5=%.4f R@10=%.4f | "
            "ΔR@1=%.4f ΔR@5=%.4f ΔR@10=%.4f",
            comparison["random_pool_R@1"],   comparison["random_pool_R@5"],
            comparison["random_pool_R@10"],
            comparison["hardneg_pool_R@1"],  comparison["hardneg_pool_R@5"],
            comparison["hardneg_pool_R@10"],
            comparison["delta_R@1_hardneg_minus_random"],
            comparison["delta_R@5_hardneg_minus_random"],
            comparison["delta_R@10_hardneg_minus_random"],
        )

    # 儲存輸出
    prefix        = f"{exp_name}_{CKPT_NAME}_{POOL_SIZE}pool_hardneg_top{TOP_K_HARD}"
    ranking_csv   = os.path.join(out_dir, f"{prefix}_ranking_samples.csv")
    summary_path  = os.path.join(out_dir, f"{prefix}_summary.json")

    write_csv(ranking_csv, ranking_rows)

    full_summary = {
        "exp_name":         exp_name,
        "checkpoint":       CKPT_NAME,
        "checkpoint_dir":   ckpt_dir,
        "ltp_mode":         ltp_mode,
        "active_modalities": active_mods,
        "pool_size":        POOL_SIZE,
        "pool_type":        "hard_negative_cosim",
        "top_k_hard":       TOP_K_HARD,
        "hardneg_seed":     HARDNEG_SEED,
        "max_samples":      MAX_SAMPLES,
        "started_at":       start.isoformat(timespec="seconds"),
        "finished_at":      _dt.datetime.now().isoformat(timespec="seconds"),
        "ranking":          ranking_summary,
        "comparison_vs_random_pool": comparison,
        "outputs": {
            "ranking_csv": ranking_csv,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved ranking CSV: %s", ranking_csv)
    logger.info("Saved summary JSON: %s", summary_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 60)
    print(f"Hard-negative 500-pool eval complete ({exp_name})")
    print(f"  R@1 = {ranking_summary['recall@1']:.4f}")
    print(f"  R@5 = {ranking_summary['recall@5']:.4f}")
    print(f"  R@10= {ranking_summary['recall@10']:.4f}")
    print(f"  MR  = {ranking_summary['median_rank']:.1f}")
    if comparison:
        print(f"\n  ΔR@1  vs random-500: {comparison['delta_R@1_hardneg_minus_random']:+.4f}")
        print(f"  ΔR@5  vs random-500: {comparison['delta_R@5_hardneg_minus_random']:+.4f}")
        print(f"  ΔR@10 vs random-500: {comparison['delta_R@10_hardneg_minus_random']:+.4f}")
    print(f"\n  Avg cosine sim of hard negatives to GT: {ranking_summary['avg_neg_cosim']:.4f}")
    print(f"  Max cosine sim of hard negatives to GT: {ranking_summary['max_neg_cosim']:.4f}")
    print(f"\nOutputs saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        sys.exit(1)
