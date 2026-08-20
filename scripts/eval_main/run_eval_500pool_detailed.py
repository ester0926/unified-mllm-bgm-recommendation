"""
用途：執行主實驗詳細評估，輸出逐筆 ranking 與生成結果。
輸入：已訓練 checkpoint、測試集特徵、候選 pool 與 LTP/cache 資料。
輸出：ranking、generation、指標摘要或逐筆評估檔。
執行：建議在 repo 根目錄執行，必要資料請先由 Zenodo 解壓到對應資料夾。
"""

from pathlib import Path
import sys

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
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
from pathlib import Path

import h5py
import numpy as np
import torch


BASE_DIR = str(PROJECT_ROOT)
H5_DIR = r"data/optimized_musechat_features_float16_v3"
JSON_DIR = r"data/musechat_json"
CACHE_DIR = os.path.join(BASE_DIR, "cache")
LLAMA_MODEL = "meta-llama/Llama-2-7b-hf"

SPECIAL_TOKENS = ["[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"]
EXP_NAMES = [f"exp_{i:02d}" for i in range(1, 8)]


# =============================================================================
# 使用前可調整的設定
# =============================================================================
#
# EXP_NAME:
#   "all" runs exp_01 to exp_07.
#   Or set one experiment, e.g. "exp_01".
#
# MAX_SAMPLES / MAX_GEN_SAMPLES：
#   None 表示完整測試集；小數字如 10 可用於快速檢查。
#
# KEEP_PER_SAMPLE_INFOLM：
#   論文輸出需保留逐筆 InfoLM。此計算較耗時，
#   因為 torchmetrics 預設回傳整體分數，
#   因此此處逐筆計算單一樣本的 InfoLM。
# =============================================================================

EXP_NAME = "all"
CKPT_NAME = "best"
POOL_SIZE = 500
PROMPT_VARIANT = "original"  # original / simple / strict / simple_v2 / strict_v2 / faithful
RESULT_TAG = ""              # optional suffix, e.g. "pool100" or "prompt_strict"
MAX_SAMPLES = None
MAX_GEN_SAMPLES = None
POINTWISE_BATCH_SIZE = 32
INJECT_TITLE = True
TIEBREAK_NOISE = True
TIEBREAK_SEED = 42
CANDIDATE_POOL_SEED = 20260315
KEEP_PER_SAMPLE_INFOLM = True

PROMPT_VARIANT_RULES = {
    "original": {
        "purpose": "MuseChat-aligned baseline prompt used by the main experiments.",
        "construction_rules": [
            "Reuse dataset.build_prompt() exactly.",
            "Training uses Candidate: [MUSIC] without title injection.",
            "Inference may inject Candidate: title by artist; [MUSIC].",
            "No explicit faithfulness or unsupported-claim constraints are added.",
        ],
    },
    "simple": {
        "purpose": "Original simple robustness prompt kept for backward compatibility.",
        "construction_rules": [
            "Use one short task sentence.",
            "Keep modality labels but remove detailed grounding constraints.",
            "Designed as an under-specified prompt stress test.",
        ],
    },
    "strict": {
        "purpose": "Original strict robustness prompt kept for backward compatibility.",
        "construction_rules": [
            "Add an explicit instruction not to invent unsupported details.",
            "Use modality labels and request a concise grounded reason.",
            "This version may over-constrain generation and reduce fluency.",
        ],
    },
    "simple_v2": {
        "purpose": "Revised simple prompt with controlled brevity and minimal constraints.",
        "construction_rules": [
            "Instruction block is 25-45 English words.",
            "Use only one positive instruction and no negative prohibitions.",
            "List available modalities using short labels.",
            "Ask for one concise recommendation reason.",
        ],
    },
    "strict_v2": {
        "purpose": "Revised strict prompt with bounded constraints to avoid over-restriction.",
        "construction_rules": [
            "Instruction block is 70-100 English words.",
            "Use exactly four grounding rules.",
            "Mention unavailable-modality avoidance without requiring excessive refusal.",
            "Allow cautious general wording when evidence is partial.",
        ],
    },
    "faithful": {
        "purpose": "Model-specific faithful multimodal prompt for explanation faithfulness tests.",
        "construction_rules": [
            "Instruction block is 100-140 English words.",
            "Explicitly maps video, music, text prompt, and LTP claims to their available sources.",
            "Forbid visual, audio, prompt, or long-term preference claims when the corresponding modality is absent.",
            "Require concise explanations grounded only in available inputs.",
        ],
    },
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

LTP_H5 = {
    "hybrid": r"data/user_profiling/stage5_output/preference_vectors.h5",
    "explicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_explicit_only.h5"),
    "implicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_implicit_only.h5"),
}


def setup_logger(log_path: str):
    logger = logging.getLogger("eval_detailed")
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
        grp = f["preference_vectors"]
        for k in grp.keys():
            out[k] = grp[k][:].astype(np.float32)
    return out


def build_test_data(model_config, train_config, tokenizer, ltp_dict, logger):
    from dataset import (
        UnifiedMLLMDataset,
        build_pair_index,
        build_song_bank,
        load_conversation_map,
        split_by_video_id,
    )

    pair_index = build_pair_index(H5_DIR, cache_path=os.path.join(CACHE_DIR, "pair_index.json"))
    conv_map = load_conversation_map(JSON_DIR)
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
    if hasattr(peft_llama, "gradient_checkpointing_disable"):
        peft_llama.gradient_checkpointing_disable()

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
        torch.load(os.path.join(ckpt_dir, "projectors.pt"), map_location="cuda:0")
    )
    projectors = projectors.to(torch.bfloat16).cuda().eval()

    ranking_head = nn.Sequential(
        nn.LayerNorm(model_config.llama_hidden_dim),
        nn.Linear(model_config.llama_hidden_dim, 256),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(256, 1),
    )
    ranking_head.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "ranking_head.pt"), map_location="cuda:0")
    )
    ranking_head = ranking_head.to(torch.bfloat16).cuda().eval()

    # 建立輕量版 UnifiedMLLM 容器，不呼叫原始 __init__。
    # 若呼叫 __init__ 會先載入完整 LLaMA，
    # 之後又被 PEFT checkpoint 取代，速度慢且容易造成 OOM。
    model = UnifiedMLLM.__new__(UnifiedMLLM)
    nn.Module.__init__(model)
    model.config = model_config
    model.tokenizer = tokenizer
    model.num_candidates = 1
    model.active_modalities = getattr(model_config, "active_modalities", ["video", "ltp", "text", "music"])
    model.multimodal_prefix_len = len(model.active_modalities)
    model.rank_token_id = tokenizer.convert_tokens_to_ids(getattr(model_config, "rank_special_token", "[RANK]"))
    model.llama = peft_llama
    model.projectors = projectors
    model.ranking_head = ranking_head
    model.eval()
    return model


def sample_prompt_tensors(sample, device):
    prompt_len = int(sample["prompt_len"].item())
    full_input_ids = sample["input_ids"]
    full_attn_mask = sample["attention_mask"]
    prompt_ids = full_input_ids[:prompt_len]
    prompt_mask = full_attn_mask[:prompt_len]
    valid_len = int(prompt_mask.sum().item())
    return (
        prompt_ids[:valid_len].unsqueeze(0).to(device),
        prompt_mask[:valid_len].unsqueeze(0).to(device),
    )


def build_prompt_variant(active_modalities, variant="original", music_title=None, music_artist=None):
    from dataset import build_prompt

    if variant == "original":
        return build_prompt(
            active_modalities=active_modalities,
            music_title=music_title,
            music_artist=music_artist,
        )

    parts = []
    if variant == "simple":
        parts.append("<s>[INST] Recommend background music for this short video.\n")
    elif variant == "strict":
        parts.append(
            "<s>[INST] You are evaluating whether one candidate track is suitable. "
            "Use only the provided modality tokens and the user's request. "
            "Do not invent visual, music, or preference details that are not supported.\n"
        )
    elif variant == "simple_v2":
        parts.append(
            "<s>[INST] Recommend this candidate as background music. "
            "Use the available inputs below and give one concise reason for the match.\n"
        )
    elif variant == "strict_v2":
        parts.append(
            "<s>[INST] Evaluate whether this candidate track fits the short video. "
            "Follow four rules: ground claims in the listed inputs; avoid details from "
            "modalities that are not listed; prefer cautious wording when evidence is "
            "partial; keep the recommendation reason concise and specific.\n"
        )
    elif variant == "faithful":
        available = ", ".join(active_modalities)
        parts.append(
            "<s>[INST] You are a faithful multimodal music recommendation assistant. "
            f"Available inputs are: {available}. Generate one concise recommendation "
            "reason using only these inputs. Mention visual scenes only when Video is "
            "listed. Mention genre, rhythm, tempo, mood, instruments, title, or artist "
            "only when Candidate music information is listed. Mention the current user "
            "request only when Request text is listed. Mention long-term taste or "
            "history only when Long-term preference is listed. If evidence is missing, "
            "use general wording instead of inventing details.\n"
        )
    else:
        raise ValueError(f"Unknown PROMPT_VARIANT: {variant}")

    if "video" in active_modalities:
        parts.append("Video: [VIDEO]\n")

    if "music" in active_modalities:
        if music_title:
            if music_artist:
                parts.append(f"Candidate: {music_title} by {music_artist}; [MUSIC]\n")
            else:
                parts.append(f"Candidate: {music_title}; [MUSIC]\n")
        else:
            parts.append("Candidate: [MUSIC]\n")

    if "ltp" in active_modalities:
        if variant == "simple":
            parts.append("Preference: [LTP]\n")
        else:
            parts.append("Long-term user preference: [LTP]\n")

    if "text" in active_modalities:
        if variant == "simple":
            parts.append("Request: [TEXT_CLIP] {user_text}\n")
        else:
            parts.append("Current user request: [TEXT_CLIP] {user_text}\n")

    if variant == "simple":
        parts.append("Is this candidate suitable? [/INST] [RANK] ")
    elif variant == "simple_v2":
        parts.append("Give one short suitability judgment. [/INST] [RANK] ")
    elif variant == "faithful":
        parts.append("Answer with a grounded recommendation reason. [/INST] [RANK] ")
    else:
        parts.append(
            "Return a recommendation judgment and concise reason grounded in the inputs. "
            "Does this candidate best fit this video? [/INST] [RANK] "
        )
    return "".join(parts)


def prompt_tensors_for_variant(
    sample,
    tokenizer,
    device,
    active_modalities,
    prompt_variant,
    t3_text="",
    t4_ref="",
    inject_title=False,
):
    if prompt_variant == "original" and not inject_title:
        return sample_prompt_tensors(sample, device)

    music_title = None
    music_artist = None
    if inject_title and t4_ref:
        from dataset import extract_music_title
        music_title, music_artist = extract_music_title(t4_ref)

    prompt_tmpl = build_prompt_variant(
        active_modalities=active_modalities,
        variant=prompt_variant,
        music_title=music_title,
        music_artist=music_artist,
    )
    # 使用指定欄位替換，不用 str.format()。有些歌名或藝人名
    # 可能包含像 {rap} 的大括號，str.format() 會誤判成欄位名稱，
    # 導致 KeyError。
    full_prompt = prompt_tmpl.replace("{user_text}", t3_text)
    enc = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def rank_from_scores(scores, add_noise=True, rng=None, noise_scale=1e-6):
    scores_np = scores.float().cpu().numpy()
    scores_for_sort = scores_np
    if add_noise:
        if rng is None:
            rng = np.random.default_rng(42)
        scores_for_sort = scores_np + rng.uniform(0, noise_scale, size=scores_np.shape)

    sorted_indices = np.argsort(scores_for_sort)[::-1]
    rank = int(np.where(sorted_indices == 0)[0][0]) + 1
    top1_pool_index = int(sorted_indices[0])
    return rank, top1_pool_index, scores_np


@torch.no_grad()
def eval_ranking_detailed(
    model,
    test_dataset,
    tokenizer,
    all_music_features,
    all_music_ids,
    device,
    active_modalities,
    prompt_variant,
    conv_t3,
    train_cfg,
    pool_size,
    max_samples,
    tiebreak_noise,
    tiebreak_seed,
    candidate_pool_seed,
    logger,
):
    from evaluate import pointwise_pool_scoring

    model.eval()
    n_eval = len(test_dataset) if max_samples is None else min(max_samples, len(test_dataset))
    micro_batch_size = getattr(train_cfg, "pointwise_eval_batch_size", 32)
    total_music = all_music_features.size(0)
    tiebreak_rng = np.random.default_rng(tiebreak_seed)
    rows = []
    n_allequal = 0

    id_to_index = {sid: i for i, sid in enumerate(all_music_ids)}
    video_to_indices = {}
    for i, sid in enumerate(all_music_ids):
        video_to_indices.setdefault(sid[:11], set()).add(i)

    from tqdm import tqdm

    for idx in tqdm(range(n_eval), desc=f"Ranking ({pool_size}-pool)"):
        sample = test_dataset[idx]
        video_id = sample.get("video_id", "")
        gt_music_id = sample.get("gt_music_id", "")

        video_feat = sample["video_feat"].unsqueeze(0).to(device)
        ltp_feat = sample["ltp_feat"].unsqueeze(0).to(device)
        text_feat = sample["text_feat"].unsqueeze(0).to(device)
        input_ids, attention_mask = prompt_tensors_for_variant(
            sample=sample,
            tokenizer=tokenizer,
            device=device,
            active_modalities=active_modalities,
            prompt_variant=prompt_variant,
            t3_text=conv_t3.get(video_id, ""),
            t4_ref="",
            inject_title=False,
        )

        gt_global_idx = id_to_index.get(gt_music_id)
        if gt_global_idx is None:
            raise KeyError(f"GT music id not found in song bank: {gt_music_id}")

        excluded = video_to_indices.get(video_id, set())
        candidates = [
            i for i in range(total_music)
            if i != gt_global_idx and i not in excluded
        ]
        rng_pool = random.Random(candidate_pool_seed + idx)
        negatives = rng_pool.sample(candidates, min(pool_size - 1, len(candidates)))
        pool_idx = [gt_global_idx] + negatives
        pool_feats = all_music_features[pool_idx].to(device)

        scores = pointwise_pool_scoring(
            model=model,
            video_feat=video_feat,
            ltp_feat=ltp_feat,
            text_feat=text_feat,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_music_features=pool_feats,
            micro_batch_size=micro_batch_size,
            device=device,
        )

        rank, top1_pool_index, scores_np = rank_from_scores(
            scores,
            add_noise=tiebreak_noise,
            rng=tiebreak_rng,
        )
        score_range = float(scores_np.max() - scores_np.min())
        score_std = float(scores_np.std())
        if score_range < 1e-5:
            n_allequal += 1

        top1_global_idx = pool_idx[top1_pool_index]
        rows.append({
            "sample_idx": idx,
            "video_id": video_id,
            "gt_music_id": gt_music_id,
            "top1_music_id": all_music_ids[top1_global_idx],
            "top1_is_gt": int(rank == 1),
            "rank": rank,
            "R@1": int(rank <= 1),
            "R@5": int(rank <= 5),
            "R@10": int(rank <= 10),
            "pool_size": pool_size,
            "prompt_variant": prompt_variant,
            "gt_score": float(scores_np[0]),
            "top1_score": float(scores_np[top1_pool_index]),
            "score_gap_top1_minus_gt": float(scores_np[top1_pool_index] - scores_np[0]),
            "score_range": score_range,
            "score_std": score_std,
            "n_equal_to_gt_score": int(np.isclose(scores_np, scores_np[0], atol=1e-7).sum()),
        })

    ranks = np.array([r["rank"] for r in rows], dtype=np.float64)
    summary = {
        "recall@1": float(np.mean([r["R@1"] for r in rows])),
        "recall@5": float(np.mean([r["R@5"] for r in rows])),
        "recall@10": float(np.mean([r["R@10"] for r in rows])),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "num_samples": len(rows),
        "pool_size": pool_size,
        "scoring": "pointwise Plan B ([RANK] token readout) + random tiebreak",
        "tiebreak_noise": bool(tiebreak_noise),
        "tiebreak_seed": int(tiebreak_seed),
        "candidate_pool_seed": int(candidate_pool_seed),
        "n_allequal_scores": int(n_allequal),
        "pct_allequal": float(n_allequal / max(len(rows), 1) * 100),
        "avg_score_range": float(np.mean([r["score_range"] for r in rows])),
        "avg_score_std": float(np.mean([r["score_std"] for r in rows])),
    }
    logger.info(
        "Ranking: R@1=%.4f R@5=%.4f R@10=%.4f MR=%.1f allequal=%.2f%%",
        summary["recall@1"],
        summary["recall@5"],
        summary["recall@10"],
        summary["median_rank"],
        summary["pct_allequal"],
    )
    return rows, summary


def load_reference_maps():
    cache_path = Path(CACHE_DIR) / "reference_maps_t3_t4.json"
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            return cached.get("conv_t3", {}), cached.get("conv_t4", {}), {}
        except Exception:
            pass

    conv_t3 = {}
    conv_t4 = {}
    meta = {}
    for jf in Path(JSON_DIR).glob("**/*.json"):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            convs = data.get("conversations", [])
            if len(convs) < 4:
                continue
            video_id = jf.parent.name
            conv_t3[video_id] = convs[2].get("value", "").strip()
            conv_t4[video_id] = convs[3].get("value", "").strip()
            meta[video_id] = data
        except Exception:
            continue
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({"conv_t3": conv_t3, "conv_t4": conv_t4}, f, ensure_ascii=False)
    temp_path.replace(cache_path)
    return conv_t3, conv_t4, meta


@torch.no_grad()
def generate_one(model, sample, tokenizer, device, active_modalities, prompt_variant, t3_text, t4_ref, inject_title):
    bf16 = torch.bfloat16
    video_feat = sample["video_feat"].unsqueeze(0).to(device, dtype=bf16)
    ltp_feat = sample["ltp_feat"].unsqueeze(0).to(device, dtype=bf16)
    text_feat = sample["text_feat"].unsqueeze(0).to(device, dtype=bf16)
    pos_music = sample["pos_music_feat"].unsqueeze(0).unsqueeze(0).to(device, dtype=bf16)
    input_ids, attention_mask = prompt_tensors_for_variant(
        sample=sample,
        tokenizer=tokenizer,
        device=device,
        active_modalities=active_modalities,
        prompt_variant=prompt_variant,
        t3_text=t3_text,
        t4_ref=t4_ref,
        inject_title=inject_title,
    )

    music_title = None
    music_artist = None
    if inject_title and t4_ref:
        from dataset import extract_music_title
        music_title, music_artist = extract_music_title(t4_ref)

    rank_token_id = getattr(model, "rank_token_id", None)
    if rank_token_id is None or (input_ids == rank_token_id).sum().item() == 0:
        return "", True, music_title, music_artist

    generated_ids, _ = model.generate(
        video_feat=video_feat,
        music_candidates=pos_music,
        ltp_feat=ltp_feat,
        text_feat=text_feat,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=getattr(model.config, "max_new_tokens", 128),
        do_sample=False,
    )
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    decoded = decoded.replace("[RANK]", "").strip()
    return decoded, not bool(decoded), music_title, music_artist


def add_bertscore_to_rows(rows, logger):
    valid_idx = [i for i, r in enumerate(rows) if r["generated_text"].strip() and r["reference_text"].strip()]
    if not valid_idx:
        return {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
        }
    from bert_score import score as bert_score

    hyps = [rows[i]["generated_text"] for i in valid_idx]
    refs = [rows[i]["reference_text"] for i in valid_idx]
    p, r, f1 = bert_score(
        hyps,
        refs,
        model_type="microsoft/deberta-xlarge-mnli",
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    for j, i in enumerate(valid_idx):
        rows[i]["bertscore_precision"] = float(p[j])
        rows[i]["bertscore_recall"] = float(r[j])
        rows[i]["bertscore_f1"] = float(f1[j])

    return {
        "bertscore_precision": float(p.mean()),
        "bertscore_recall": float(r.mean()),
        "bertscore_f1": float(f1.mean()),
    }


def infolm_score(hyps, refs, information_measure):
    from torchmetrics.functional.text.infolm import infolm

    kwargs = dict(
        model_name_or_path="bert-base-uncased",
        idf=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
        batch_size=16,
    )
    if information_measure == "ab_divergence":
        kwargs.update(alpha=0.5, beta=0.5)
    return infolm(hyps, refs, information_measure=information_measure, **kwargs)


def add_infolm_to_rows(rows, per_sample, logger):
    valid_idx = [i for i, r in enumerate(rows) if r["generated_text"].strip() and r["reference_text"].strip()]
    if not valid_idx:
        return {
            "infolm_ab_divergence": None,
            "infolm_l2_distance": None,
            "infolm_fisher_rao": None,
        }

    hyps = [rows[i]["generated_text"] for i in valid_idx]
    refs = [rows[i]["reference_text"] for i in valid_idx]

    summary = {}
    for key, measure in [
        ("infolm_ab_divergence", "ab_divergence"),
        ("infolm_l2_distance", "l2_distance"),
        ("infolm_fisher_rao", "fisher_rao_distance"),
    ]:
        try:
            summary[key] = float(infolm_score(hyps, refs, measure))
        except Exception as exc:
            logger.warning("Aggregate InfoLM %s failed: %s", key, exc)
            summary[key] = None

    if not per_sample:
        return summary

    logger.info("Computing per-sample InfoLM for %d valid rows", len(valid_idx))
    from tqdm import tqdm

    for i in tqdm(valid_idx, desc="Per-sample InfoLM"):
        hyp = [rows[i]["generated_text"]]
        ref = [rows[i]["reference_text"]]
        for key, measure in [
            ("infolm_ab_divergence", "ab_divergence"),
            ("infolm_l2_distance", "l2_distance"),
            ("infolm_fisher_rao", "fisher_rao_distance"),
        ]:
            try:
                rows[i][key] = float(infolm_score(hyp, ref, measure))
            except Exception:
                rows[i][key] = None
    return summary


@torch.no_grad()
def eval_generation_detailed(
    model,
    test_dataset,
    tokenizer,
    device,
    active_modalities,
    prompt_variant,
    max_samples,
    inject_title,
    per_sample_infolm,
    logger,
):
    conv_t3, conv_t4, _ = load_reference_maps()
    n_eval = len(test_dataset) if max_samples is None else min(max_samples, len(test_dataset))
    rows = []
    fallback_count = 0

    from tqdm import tqdm

    for idx in tqdm(range(n_eval), desc="Generation"):
        sample = test_dataset[idx]
        video_id = sample.get("video_id", "")
        gt_music_id = sample.get("gt_music_id", "")
        t3_text = conv_t3.get(video_id, "")
        t4_ref = conv_t4.get(video_id, "")
        if not t4_ref:
            continue
        try:
            generated, is_fallback, music_title, music_artist = generate_one(
                model,
                sample,
                tokenizer,
                device,
                active_modalities,
                prompt_variant,
                t3_text,
                t4_ref,
                inject_title,
            )
        except Exception as exc:
            logger.warning("Generation failed at idx=%s video=%s: %s", idx, video_id, exc)
            generated, is_fallback, music_title, music_artist = "", True, None, None
        fallback_count += int(is_fallback)
        rows.append({
            "sample_idx": idx,
            "video_id": video_id,
            "gt_music_id": gt_music_id,
            "music_title": music_title,
            "music_artist": music_artist,
            "prompt_variant": prompt_variant,
            "user_text": t3_text,
            "generated_text": generated,
            "reference_text": t4_ref,
            "is_fallback": int(is_fallback),
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "infolm_ab_divergence": None,
            "infolm_l2_distance": None,
            "infolm_fisher_rao": None,
        })

    summary = {
        "n_generated": len(rows),
        "n_valid": int(sum(1 for r in rows if r["generated_text"].strip())),
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(len(rows), 1)),
        "inject_title": bool(inject_title),
    }
    summary.update(add_bertscore_to_rows(rows, logger))
    summary.update(add_infolm_to_rows(rows, per_sample=per_sample_infolm, logger=logger))
    logger.info(
        "Generation: valid=%d/%d fallback=%.2f%% BERTScore-F1=%s InfoLM-L2=%s",
        summary["n_valid"],
        summary["n_generated"],
        summary["fallback_rate"] * 100,
        summary.get("bertscore_f1"),
        summary.get("infolm_l2_distance"),
    )
    return rows, summary


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_rows(ranking_rows, generation_rows):
    by_idx = {r["sample_idx"]: dict(r) for r in ranking_rows}
    for grow in generation_rows:
        base = by_idx.setdefault(grow["sample_idx"], {})
        for key, value in grow.items():
            if key not in base:
                base[key] = value
            elif key in {"video_id", "gt_music_id", "sample_idx"}:
                continue
            else:
                base[key] = value
    return [by_idx[k] for k in sorted(by_idx)]


class EvalSettings:
    def __init__(self):
        self.exp = EXP_NAME
        self.ckpt = CKPT_NAME
        self.pool_size = POOL_SIZE
        self.prompt_variant = PROMPT_VARIANT
        self.result_tag = RESULT_TAG
        self.max_samples = MAX_SAMPLES
        self.max_gen_samples = MAX_GEN_SAMPLES
        self.pointwise_batch_size = POINTWISE_BATCH_SIZE
        self.inject_title = INJECT_TITLE
        self.tiebreak_noise = TIEBREAK_NOISE
        self.tiebreak_seed = TIEBREAK_SEED
        self.candidate_pool_seed = CANDIDATE_POOL_SEED
        self.per_sample_infolm = KEEP_PER_SAMPLE_INFOLM


def run_one_exp(exp_name, settings):
    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer

    output_dir = os.path.join(BASE_DIR, "checkpoints", exp_name)
    detail_dir = os.path.join(output_dir, "detailed_eval")
    os.makedirs(detail_dir, exist_ok=True)
    logger = setup_logger(os.path.join(detail_dir, f"eval_detailed_{exp_name}.log"))

    ltp_mode = EXP_TO_LTP_MODE[exp_name]
    active_modalities = EXP_TO_MODALITIES[exp_name]
    ckpt_dir = os.path.join(output_dir, settings.ckpt)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    logger.info("=" * 72)
    logger.info(
        "Detailed eval | exp=%s ckpt=%s pool=%d prompt=%s tag=%s candidate_seed=%d modalities=%s",
        exp_name,
        settings.ckpt,
        settings.pool_size,
        settings.prompt_variant,
        settings.result_tag,
        settings.candidate_pool_seed,
        active_modalities,
    )
    logger.info("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This evaluation expects CUDA because the model is loaded on cuda:0.")

    model_cfg = ModelConfig(
        llama_model_name=LLAMA_MODEL,
        video_dim=768,
        music_dim=768,
        text_dim=512,
        ltp_dim=256,
        num_candidates=1,
        active_modalities=active_modalities,
        music_token_offset=3,
        rank_special_token="[RANK]",
    )
    train_cfg = TrainConfig(
        output_dir=output_dir,
        pointwise_eval_batch_size=settings.pointwise_batch_size,
        music_pool_size=settings.pool_size,
    )

    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    ltp_dict = load_ltp_dict(LTP_H5[ltp_mode], ltp_mode, cache_path=os.path.join(CACHE_DIR, "ltp"), logger=logger)
    test_dataset, all_music_features, all_music_ids, _ = build_test_data(model_cfg, train_cfg, tokenizer, ltp_dict, logger)
    conv_t3, _, _ = load_reference_maps()
    model = load_model(ckpt_dir, model_cfg, tokenizer, logger)

    start = _dt.datetime.now()
    ranking_rows, ranking_summary = [], {}
    generation_rows, generation_summary = [], {}

    ranking_rows, ranking_summary = eval_ranking_detailed(
        model=model,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        all_music_features=all_music_features,
        all_music_ids=all_music_ids,
        device=device,
        active_modalities=active_modalities,
        prompt_variant=settings.prompt_variant,
        conv_t3=conv_t3,
        train_cfg=train_cfg,
        pool_size=settings.pool_size,
        max_samples=settings.max_samples,
        tiebreak_noise=settings.tiebreak_noise,
        tiebreak_seed=settings.tiebreak_seed,
        candidate_pool_seed=settings.candidate_pool_seed,
        logger=logger,
    )

    generation_rows, generation_summary = eval_generation_detailed(
        model=model,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        active_modalities=active_modalities,
        prompt_variant=settings.prompt_variant,
        max_samples=settings.max_gen_samples if settings.max_gen_samples is not None else settings.max_samples,
        inject_title=settings.inject_title,
        per_sample_infolm=settings.per_sample_infolm,
        logger=logger,
    )

    merged_rows = merge_rows(ranking_rows, generation_rows)
    tag = f"_{settings.result_tag}" if settings.result_tag else ""
    if not settings.result_tag and settings.prompt_variant != "original":
        tag = f"_prompt_{settings.prompt_variant}"
    prefix = f"{exp_name}_{settings.ckpt}_{settings.pool_size}pool{tag}"
    ranking_csv = os.path.join(detail_dir, f"{prefix}_ranking_samples.csv")
    generation_csv = os.path.join(detail_dir, f"{prefix}_generation_samples.csv")
    merged_csv = os.path.join(detail_dir, f"{prefix}_samples_merged.csv")
    merged_jsonl = os.path.join(detail_dir, f"{prefix}_samples_merged.jsonl")
    summary_path = os.path.join(detail_dir, f"{prefix}_summary.json")

    write_csv(ranking_csv, ranking_rows)
    write_csv(generation_csv, generation_rows)
    write_csv(merged_csv, merged_rows)
    write_jsonl(merged_jsonl, merged_rows)

    summary = {
        "exp_name": exp_name,
        "checkpoint": settings.ckpt,
        "checkpoint_dir": ckpt_dir,
        "ltp_mode": ltp_mode,
        "active_modalities": active_modalities,
        "pool_size": settings.pool_size,
        "prompt_variant": settings.prompt_variant,
        "prompt_variant_rule": PROMPT_VARIANT_RULES.get(settings.prompt_variant, {}),
        "result_tag": settings.result_tag,
        "candidate_pool_seed": settings.candidate_pool_seed,
        "max_samples": settings.max_samples,
        "max_gen_samples": settings.max_gen_samples,
        "started_at": start.isoformat(timespec="seconds"),
        "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "ranking": ranking_summary,
        "generation": generation_summary,
        "outputs": {
            "ranking_csv": ranking_csv,
            "generation_csv": generation_csv,
            "merged_csv": merged_csv,
            "merged_jsonl": merged_jsonl,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved merged samples: %s", merged_csv)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return summary


def main():
    settings = EvalSettings()
    if settings.exp != "all" and settings.exp not in EXP_NAMES:
        raise ValueError(f"EXP_NAME must be 'all' or one of {EXP_NAMES}, got: {settings.exp}")
    if settings.prompt_variant not in PROMPT_VARIANT_RULES:
        raise ValueError(f"PROMPT_VARIANT must be one of: {', '.join(PROMPT_VARIANT_RULES)}")

    exp_list = EXP_NAMES if settings.exp == "all" else [settings.exp]
    summaries = []
    for exp_name in exp_list:
        summaries.append(run_one_exp(exp_name, settings))

    combined_path = os.path.join(
        BASE_DIR,
        "checkpoints",
        f"detailed_eval_summary_{settings.exp}_{settings.pool_size}pool"
        f"{('_' + settings.result_tag) if settings.result_tag else ''}"
        f"{('_prompt_' + settings.prompt_variant) if (not settings.result_tag and settings.prompt_variant != 'original') else ''}.json",
    )
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Combined summary saved: {combined_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        sys.exit(1)
