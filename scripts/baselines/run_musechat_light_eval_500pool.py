"""
用途：執行 baseline 模型的訓練或評估流程。
輸入：已訓練 checkpoint、測試集特徵、候選 pool 與 LTP/cache 資料。
輸出：ranking、generation、指標摘要或逐筆評估檔。
執行：建議在 repo 根目錄執行，必要資料請先由 Zenodo 解壓到對應資料夾。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import csv
import datetime as _dt
import gc
import importlib.util
import json
import logging
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_main import run_eval_500pool_detailed as core
from config import TrainConfig
from dataset import build_pair_index, build_song_bank, extract_music_title, split_by_video_id


# =============================================================================
# 使用前可調整的設定
# =============================================================================

MUSECHAT_DIR = r"external/musechat"
MVT_CKPT_PATH = os.path.join(MUSECHAT_DIR, "checkpoints", "mvt_best.pt")

# 設為 None 時會自動尋找 checkpoints/generator_epoch*/training_state.pt。
# 範例：r"external/musechat\checkpoints\generator_epoch3"
GENERATOR_CKPT_DIR = None

POOL_SIZE = 500
MAX_SAMPLES = None
# False 表示重用既有 ranking CSV，只重新產生 GT/Top-1 文字。
# 若 ranking CSV 不存在，仍會自動執行 ranking。
RUN_RANKING = False
RUN_GT_GENERATION = True
RUN_TOP1_GENERATION = True
KEEP_PER_SAMPLE_INFOLM = True

CANDIDATE_POOL_SEED = 20260315
TIEBREAK_NOISE = True
TIEBREAK_SEED = 42

TARGET_ENCODE_BATCH_SIZE = 4096
SEGMENT_BATCH_SIZE = 1
GEN_MAX_NEW_TOKENS = 128

OUTPUT_DIR = os.path.join(core.BASE_DIR, "checkpoints", "musechat_light", "detailed_eval")
OUTPUT_PREFIX = f"musechat_light_{POOL_SIZE}pool"


def setup_logger():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger("musechat_light_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    log_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


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


def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    spec.loader.exec_module(module)
    return module


def load_musechat_modules():
    """從 external/musechat 載入 baseline 模組，避免覆蓋本專案的 config.py。"""
    old_config = sys.modules.get("config")
    old_sys_path = list(sys.path)
    muse_config = load_module_from_file(
        "musechat_light_config",
        os.path.join(MUSECHAT_DIR, "config.py"),
    )
    try:
        sys.modules["config"] = muse_config
        if MUSECHAT_DIR not in sys.path:
            sys.path.insert(0, MUSECHAT_DIR)
        mvt_module = load_module_from_file(
            "musechat_light_mvt_fusion",
            os.path.join(MUSECHAT_DIR, "models", "mvt_fusion.py"),
        )
        sentence_module = load_module_from_file(
            "musechat_light_sentence_generator",
            os.path.join(MUSECHAT_DIR, "models", "sentence_generator.py"),
        )
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config
        sys.path[:] = old_sys_path
    return muse_config, mvt_module, sentence_module


def load_mvt_model(device, logger):
    if not os.path.exists(MVT_CKPT_PATH):
        raise FileNotFoundError(f"MuseChat-light MVT checkpoint not found: {MVT_CKPT_PATH}")

    muse_config, mvt_module, sentence_module = load_musechat_modules()
    cfg = muse_config.MuseChatConfig()
    model = mvt_module.MVTFusionModule(cfg=cfg.mvt).to(device)
    state = torch.load(MVT_CKPT_PATH, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("Loaded MuseChat-light MVT checkpoint: %s", MVT_CKPT_PATH)
    return model, cfg, sentence_module


class H5FeatureReader:
    def __init__(self):
        self.handles = {}

    def get_handle(self, h5_path):
        if h5_path not in self.handles:
            self.handles[h5_path] = h5py.File(h5_path, "r")
        return self.handles[h5_path]

    def read_mvt_query_features(self, h5_path, pair_key):
        grp = self.get_handle(h5_path)[f"pairs/{pair_key}"]
        video_all = torch.from_numpy(grp["video_features_all"][:].astype(np.float32))
        candidate_all = torch.from_numpy(grp["candidate_music_all_seq"][:].astype(np.float32))
        text_seq = torch.from_numpy(grp["text_features"][:].astype(np.float32))
        return video_all, candidate_all, text_seq

    def close(self):
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self.handles.clear()


@torch.no_grad()
def encode_target_bank(model, song_bank_tensor, device, logger):
    outputs = []
    for start in range(0, song_bank_tensor.size(0), TARGET_ENCODE_BATCH_SIZE):
        batch = song_bank_tensor[start:start + TARGET_ENCODE_BATCH_SIZE].to(device)
        encoded = model.encode_target(batch)
        outputs.append(encoded.detach().cpu())
    target = F.normalize(torch.cat(outputs, dim=0), dim=-1)
    logger.info("Encoded target music bank: %s", tuple(target.shape))
    return target


@torch.no_grad()
def encode_query(model, reader, h5_path, pair_key, device):
    video_all, candidate_all, text_seq = reader.read_mvt_query_features(h5_path, pair_key)
    outputs = []
    for start in range(0, video_all.size(0), SEGMENT_BATCH_SIZE):
        end = min(start + SEGMENT_BATCH_SIZE, video_all.size(0))
        video = video_all[start:end].to(device)
        candidate = candidate_all[start:end].to(device)
        text = text_seq.unsqueeze(0).expand(end - start, -1, -1).to(device)
        outputs.append(model(video, candidate, text).detach().cpu())
    query = torch.cat(outputs, dim=0).mean(dim=0)
    return F.normalize(query, dim=-1).to(device)


def rank_from_scores(scores, tiebreak_rng):
    scores_np = np.asarray(scores, dtype=np.float64)
    scores_for_sort = scores_np
    if TIEBREAK_NOISE:
        scores_for_sort = scores_np + tiebreak_rng.uniform(0, 1e-8, size=scores_np.shape)
    sorted_indices = np.argsort(scores_for_sort)[::-1]
    rank = int(np.where(sorted_indices == 0)[0][0]) + 1
    top1_pool_index = int(sorted_indices[0])
    return rank, top1_pool_index, scores_np


def summarize_ranking(rows):
    ranks = np.array([float(r["rank"]) for r in rows], dtype=np.float64)
    return {
        "recall@1": float(np.mean([int(r["R@1"]) for r in rows])),
        "recall@5": float(np.mean([int(r["R@5"]) for r in rows])),
        "recall@10": float(np.mean([int(r["R@10"]) for r in rows])),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "num_samples": len(rows),
        "pool_size": POOL_SIZE,
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "tiebreak_noise": bool(TIEBREAK_NOISE),
        "tiebreak_seed": int(TIEBREAK_SEED),
    }


def build_shared_data(logger):
    train_cfg = TrainConfig()
    pair_index = build_pair_index(core.H5_DIR, cache_path=os.path.join(core.CACHE_DIR, "pair_index.json"))
    song_bank_np, song_ids = build_song_bank(pair_index, cache_path=os.path.join(core.CACHE_DIR, "song_bank"))
    _, _, test_pairs = split_by_video_id(
        pair_index,
        train_cfg.train_ratio,
        train_cfg.val_ratio,
        train_cfg.test_ratio,
        train_cfg.split_seed,
    )
    conv_t3, conv_t4, _ = core.load_reference_maps()
    logger.info("Shared data: test_pairs=%d song_bank=%d", len(test_pairs), len(song_ids))
    return test_pairs, torch.tensor(song_bank_np, dtype=torch.float32), song_ids, conv_t3, conv_t4


def run_ranking(model, test_pairs, target_bank, song_ids, logger):
    reader = H5FeatureReader()
    id_to_index = {sid: i for i, sid in enumerate(song_ids)}
    video_to_indices = {}
    for i, sid in enumerate(song_ids):
        video_to_indices.setdefault(sid[:11], set()).add(i)

    n_eval = len(test_pairs) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(test_pairs))
    tiebreak_rng = np.random.default_rng(TIEBREAK_SEED)
    rows = []

    from tqdm import tqdm

    try:
        for idx in tqdm(range(n_eval), desc=f"MuseChat-light ranking ({POOL_SIZE}-pool)"):
            h5_path, pair_key = test_pairs[idx]
            video_id = pair_key[:11]
            gt_music_id = pair_key
            gt_global_idx = id_to_index.get(gt_music_id)
            if gt_global_idx is None:
                raise KeyError(f"GT music id not found in song bank: {gt_music_id}")

            query = encode_query(model, reader, h5_path, pair_key, target_bank.device)
            excluded = video_to_indices.get(video_id, set())
            candidates = [
                i for i in range(target_bank.size(0))
                if i != gt_global_idx and i not in excluded
            ]
            rng_pool = random.Random(CANDIDATE_POOL_SEED + idx)
            negatives = rng_pool.sample(candidates, min(POOL_SIZE - 1, len(candidates)))
            pool_idx = [gt_global_idx] + negatives
            pool_feats = target_bank[pool_idx]
            scores = (query.unsqueeze(0) @ pool_feats.T).squeeze(0).cpu().numpy()
            rank, top1_pool_index, scores_np = rank_from_scores(scores, tiebreak_rng)
            top1_global_idx = pool_idx[top1_pool_index]

            score_range = float(scores_np.max() - scores_np.min())
            rows.append({
                "sample_idx": idx,
                "video_id": video_id,
                "gt_music_id": gt_music_id,
                "top1_music_id": song_ids[top1_global_idx],
                "top1_is_gt": int(rank == 1),
                "rank": rank,
                "R@1": int(rank <= 1),
                "R@5": int(rank <= 5),
                "R@10": int(rank <= 10),
                "pool_size": POOL_SIZE,
                "baseline": "musechat_light",
                "gt_score": float(scores_np[0]),
                "top1_score": float(scores_np[top1_pool_index]),
                "score_gap_top1_minus_gt": float(scores_np[top1_pool_index] - scores_np[0]),
                "score_range": score_range,
                "score_std": float(scores_np.std()),
                "n_equal_to_gt_score": int(np.isclose(scores_np, scores_np[0], atol=1e-7).sum()),
            })
    finally:
        reader.close()

    summary = summarize_ranking(rows)
    logger.info(
        "MuseChat-light ranking: R@1=%.4f R@5=%.4f R@10=%.4f MedianRank=%.1f",
        summary["recall@1"],
        summary["recall@5"],
        summary["recall@10"],
        summary["median_rank"],
    )
    return rows, summary


def normalize_for_match(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"\s+", " ", text).strip()


def mentions_value(generated_text, value):
    value = normalize_for_match(value)
    if not value:
        return None
    generated = normalize_for_match(generated_text)
    if value in generated:
        return True
    compact_value = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    compact_generated = re.sub(r"[\W_]+", "", generated, flags=re.UNICODE)
    return bool(compact_value and compact_value in compact_generated)


def title_consistency_flags(generated_text, music_title, music_artist):
    title_hit = mentions_value(generated_text, music_title)
    artist_hit = mentions_value(generated_text, music_artist)
    checked = title_hit is not None or artist_hit is not None
    if not checked:
        consistency = None
    elif title_hit is False or artist_hit is False:
        consistency = False
    else:
        consistency = True
    return {
        "generated_mentions_music_title": "" if title_hit is None else int(title_hit),
        "generated_mentions_music_artist": "" if artist_hit is None else int(artist_hit),
        "title_consistency": "" if consistency is None else int(consistency),
        "needs_manual_review": 1 if consistency is False else 0,
    }


def find_generator_checkpoint():
    if GENERATOR_CKPT_DIR:
        return GENERATOR_CKPT_DIR if os.path.exists(os.path.join(GENERATOR_CKPT_DIR, "training_state.pt")) else None
    ckpt_root = Path(MUSECHAT_DIR) / "checkpoints"
    candidates = sorted(
        [p for p in ckpt_root.glob("generator_epoch*") if (p / "training_state.pt").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def load_generator(sentence_module, muse_cfg, device, logger):
    ckpt_dir = find_generator_checkpoint()
    if not ckpt_dir:
        logger.warning(
            "No MuseChat sentence-generator checkpoint found. "
            "GT and Top-1 generation metrics will be skipped."
        )
        return None, None, None

    tokenizer = sentence_module.build_tokenizer(muse_cfg.gen.llm_model_name)
    model = sentence_module.SentenceGenerator(cfg=muse_cfg.gen, tokenizer=tokenizer)
    try:
        model.llm.resize_token_embeddings(len(tokenizer))
        logger.info("Resized generator token embeddings to %d before loading LoRA.", len(tokenizer))
    except Exception as exc:
        base_model = getattr(model.llm, "base_model", None)
        if base_model is not None and hasattr(base_model, "model"):
            base_model.model.resize_token_embeddings(len(tokenizer))
            logger.info("Resized generator base token embeddings to %d before loading LoRA.", len(tokenizer))
        else:
            raise RuntimeError("Unable to resize MuseChat generator token embeddings.") from exc
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    state = torch.load(state_path, map_location="cpu")
    model.music_proj.load_state_dict(state["music_proj_state"])
    model.music_proj = model.music_proj.to(device)

    lora_dir = os.path.join(ckpt_dir, "lora_weights")
    if os.path.isdir(lora_dir):
        if hasattr(model.llm, "load_adapter"):
            adapter_name = "musechat_light_eval"
            model.llm.load_adapter(lora_dir, adapter_name=adapter_name, is_trainable=False)
            if hasattr(model.llm, "set_adapter"):
                model.llm.set_adapter(adapter_name)
            logger.info("Loaded LoRA adapter from %s", lora_dir)
        else:
            logger.warning("PEFT load_adapter is unavailable; LoRA weights were not loaded.")
    else:
        logger.warning("No lora_weights directory found under %s", ckpt_dir)

    model.eval()
    logger.info("Loaded MuseChat sentence-generator checkpoint: %s", ckpt_dir)
    return model, tokenizer, ckpt_dir


def title_string_from_reference(t4_text):
    title, artist = extract_music_title(t4_text)
    if title and artist:
        return f"{title} by {artist}", title, artist
    if title:
        return title, title, artist
    return None, title, artist


@torch.no_grad()
def generate_with_musechat(generator, music_feat, title_string):
    device = next(generator.music_proj.parameters()).device
    return generator.generate(
        music_avg=music_feat.unsqueeze(0).to(device),
        music_title=title_string,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        temperature=0.1,
        do_sample=False,
    ).strip()


def summarize_generation(rows):
    def mean_metric(key):
        vals = []
        for row in rows:
            value = row.get(key)
            try:
                vals.append(float(value))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else None

    checked = [r for r in rows if str(r.get("title_consistency", "")) != ""]
    consistent = [r for r in checked if int(r.get("title_consistency", 0)) == 1]
    return {
        "num_samples": len(rows),
        "n_valid": int(sum(1 for r in rows if str(r.get("generated_text", "")).strip())),
        "fallback_count": int(sum(int(r.get("is_fallback", 0)) for r in rows)),
        "fallback_rate": float(sum(int(r.get("is_fallback", 0)) for r in rows) / max(len(rows), 1)),
        "title_consistency_checked": len(checked),
        "title_consistency_rate": float(len(consistent) / max(len(checked), 1)),
        "bertscore_precision": mean_metric("bertscore_precision"),
        "bertscore_recall": mean_metric("bertscore_recall"),
        "bertscore_f1": mean_metric("bertscore_f1"),
        "infolm_ab_divergence": mean_metric("infolm_ab_divergence"),
        "infolm_l2_distance": mean_metric("infolm_l2_distance"),
        "infolm_fisher_rao": mean_metric("infolm_fisher_rao"),
    }


def add_text_metrics(rows, logger):
    summary = {}
    summary.update(core.add_bertscore_to_rows(rows, logger))
    summary.update(core.add_infolm_to_rows(rows, per_sample=KEEP_PER_SAMPLE_INFOLM, logger=logger))
    summary.update(summarize_generation(rows))
    return summary


def run_generation(generator, ranking_rows, song_bank_tensor, song_ids, conv_t3, conv_t4, mode, logger):
    id_to_index = {sid: i for i, sid in enumerate(song_ids)}
    rows = []
    from tqdm import tqdm

    for rank_row in tqdm(ranking_rows, desc=f"MuseChat-light {mode} generation"):
        sample_idx = int(rank_row["sample_idx"])
        query_video_id = rank_row["video_id"]
        gt_music_id = rank_row["gt_music_id"]
        reference_text = conv_t4.get(query_video_id, "")
        user_text = conv_t3.get(query_video_id, "")

        if mode == "gt_conditioned":
            generation_music_id = gt_music_id
            title_reference_text = reference_text
            title_source = "query_gt_reference"
        elif mode == "top1_end_to_end":
            generation_music_id = rank_row["top1_music_id"]
            top1_video_id = str(generation_music_id)[:11]
            title_reference_text = conv_t4.get(top1_video_id, "") or reference_text
            title_source = "top1_reference" if conv_t4.get(top1_video_id, "") else "query_reference_fallback"
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        music_index = id_to_index.get(generation_music_id)
        title_string, music_title, music_artist = title_string_from_reference(title_reference_text)
        if music_index is None or not reference_text:
            generated_text = ""
            is_fallback = True
        else:
            try:
                generated_text = generate_with_musechat(generator, song_bank_tensor[music_index], title_string)
                is_fallback = not bool(generated_text)
            except Exception as exc:
                logger.warning(
                    "MuseChat-light generation failed: mode=%s sample=%s music=%s error=%s",
                    mode,
                    sample_idx,
                    generation_music_id,
                    exc,
                )
                generated_text = ""
                is_fallback = True

        rows.append({
            "sample_idx": sample_idx,
            "video_id": query_video_id,
            "gt_music_id": gt_music_id,
            "generation_music_id": generation_music_id,
            "top1_music_id": rank_row.get("top1_music_id", ""),
            "top1_is_gt": rank_row.get("top1_is_gt", ""),
            "rank": rank_row.get("rank", ""),
            "R@1": rank_row.get("R@1", ""),
            "R@5": rank_row.get("R@5", ""),
            "R@10": rank_row.get("R@10", ""),
            "pool_size": POOL_SIZE,
            "baseline": "musechat_light",
            "generation_mode": mode,
            "title_source": title_source,
            "music_title": music_title,
            "music_artist": music_artist,
            "user_text": user_text,
            "generated_text": generated_text,
            "reference_text": reference_text,
            "title_reference_text": title_reference_text,
            **title_consistency_flags(generated_text, music_title, music_artist),
            "is_fallback": int(is_fallback),
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "infolm_ab_divergence": None,
            "infolm_l2_distance": None,
            "infolm_fisher_rao": None,
        })

    raw_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{mode}_generation_samples_raw.csv")
    write_csv(raw_path, rows)
    logger.info("Saved raw MuseChat-light %s generations: %s", mode, raw_path)

    metric_summary = add_text_metrics(rows, logger)
    csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{mode}_generation_samples.csv")
    jsonl_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{mode}_generation_samples.jsonl")
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    logger.info("Saved MuseChat-light %s generations: %s", mode, csv_path)
    return rows, metric_summary, {"csv": csv_path, "jsonl": jsonl_path, "raw_csv": raw_path}


def main():
    logger = setup_logger()
    started_at = _dt.datetime.now()
    logger.info("Re-implemented MuseChat-light evaluation")
    logger.info("MuseChat dir: %s", MUSECHAT_DIR)
    logger.info("Pool size=%d candidate_seed=%d", POOL_SIZE, CANDIDATE_POOL_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This evaluation expects CUDA for MuseChat-light and text metrics.")

    test_pairs, song_bank_tensor, song_ids, conv_t3, conv_t4 = build_shared_data(logger)
    mvt_model, muse_cfg, sentence_module = load_mvt_model(device, logger)
    target_bank = encode_target_bank(mvt_model, song_bank_tensor, device, logger).to(device)

    ranking_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ranking_samples.csv")
    ranking_jsonl_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ranking_samples.jsonl")
    ranking_rows = []
    ranking_summary = {"skipped": True}
    if RUN_RANKING or not os.path.exists(ranking_path):
        ranking_rows, ranking_summary = run_ranking(mvt_model, test_pairs, target_bank, song_ids, logger)
        write_csv(ranking_path, ranking_rows)
        write_jsonl(ranking_jsonl_path, ranking_rows)
    else:
        with open(ranking_path, encoding="utf-8-sig", newline="") as f:
            ranking_rows = list(csv.DictReader(f))
        ranking_summary = summarize_ranking(ranking_rows)

    del mvt_model
    torch.cuda.empty_cache()
    gc.collect()

    generation_summary = {
        "gt_conditioned": {"skipped": True},
        "top1_end_to_end": {"skipped": True},
    }
    generation_outputs = {}

    if RUN_GT_GENERATION or RUN_TOP1_GENERATION:
        generator, _, generator_ckpt = load_generator(sentence_module, muse_cfg, device, logger)
        if generator is not None:
            if RUN_GT_GENERATION:
                _, gt_summary, gt_outputs = run_generation(
                    generator,
                    ranking_rows,
                    song_bank_tensor,
                    song_ids,
                    conv_t3,
                    conv_t4,
                    "gt_conditioned",
                    logger,
                )
                generation_summary["gt_conditioned"] = gt_summary
                generation_outputs["gt_conditioned"] = gt_outputs
            if RUN_TOP1_GENERATION:
                _, top1_summary, top1_outputs = run_generation(
                    generator,
                    ranking_rows,
                    song_bank_tensor,
                    song_ids,
                    conv_t3,
                    conv_t4,
                    "top1_end_to_end",
                    logger,
                )
                generation_summary["top1_end_to_end"] = top1_summary
                generation_outputs["top1_end_to_end"] = top1_outputs
        else:
            generator_ckpt = None

    summary = {
        "baseline": "musechat_light",
        "paper_alignment": {
            "recommendation_module": (
                "MVT-Fusion over video, user prompt text, and first-round candidate music; "
                "target music encoded by AST CLS projection."
            ),
            "sentence_generator": (
                "Vicuna-7B + LoRA + linear music projection, using MuseChat-style inference "
                "prompt with music title and music feature token when a checkpoint is available."
            ),
            "data_alignment": (
                "Uses the unified thesis HDF5 directory, video-level split seed, song bank, "
                "and candidate-pool sampling rule from run_eval_500pool_detailed.py."
            ),
        },
        "musechat_dir": MUSECHAT_DIR,
        "mvt_checkpoint": MVT_CKPT_PATH,
        "generator_checkpoint": generator_ckpt if "generator_ckpt" in locals() else None,
        "pool_size": POOL_SIZE,
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "max_samples": MAX_SAMPLES,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "ranking": ranking_summary,
        "generation": generation_summary,
        "outputs": {
            "ranking_csv": ranking_path,
            "ranking_jsonl": ranking_jsonl_path,
            **generation_outputs,
        },
    }
    summary_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary: %s", summary_path)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
