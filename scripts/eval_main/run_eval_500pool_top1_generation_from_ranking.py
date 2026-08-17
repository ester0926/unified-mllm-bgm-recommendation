# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Top-1 end-to-end generation from existing 500-pool ranking outputs.

Usage:
  Open this file in VSCode and click Run.

This script does NOT rerun ranking. It reads the per-sample ranking CSV produced
by run_eval_500pool_detailed.py, retrieves each sample's recorded top-1 music,
then generates an explanation for that top-1 recommendation.

Outputs are written next to the original detailed_eval files with a `_top1_`
suffix so the original GT-conditioned generation files remain untouched.
"""

import csv
import datetime as _dt
import gc
import json
import os
import re
import unicodedata

import torch

from scripts.eval_main import run_eval_500pool_detailed as core


# =============================================================================
# USER SETTINGS
# =============================================================================

EXP_NAME = "all"          # "all" or one exp, e.g. "exp_01"
CKPT_NAME = "best"
POOL_SIZE = 500
PROMPT_VARIANT = "original"
RANKING_RESULT_TAG = ""   # "" for exp_01_best_500pool_ranking_samples.csv
OUTPUT_TAG = "top1"
MAX_SAMPLES = None        # None means all rows in the existing ranking CSV
INJECT_TITLE = True
KEEP_PER_SAMPLE_INFOLM = True


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_rows_by_sample_idx(ranking_rows, generation_rows):
    by_idx = {str(r["sample_idx"]): dict(r) for r in ranking_rows}
    for grow in generation_rows:
        key = str(grow["sample_idx"])
        base = by_idx.setdefault(key, {})
        for field, value in grow.items():
            if field == "sample_idx":
                base[field] = key
            elif field in {"video_id", "gt_music_id"} and base.get(field):
                continue
            else:
                base[field] = value
    return [by_idx[k] for k in sorted(by_idx, key=lambda x: int(float(x)))]


def ranking_prefix(exp_name):
    tag = f"_{RANKING_RESULT_TAG}" if RANKING_RESULT_TAG else ""
    return f"{exp_name}_{CKPT_NAME}_{POOL_SIZE}pool{tag}"


def output_prefix(exp_name):
    tag = f"_{OUTPUT_TAG}" if OUTPUT_TAG else "_top1"
    return f"{exp_name}_{CKPT_NAME}_{POOL_SIZE}pool{tag}"


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def to_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def mean_metric(rows, key):
    vals = [to_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def normalize_for_match(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mentions_value(generated_text, value):
    value = normalize_for_match(value)
    if not value:
        return None
    generated = normalize_for_match(generated_text)
    if value in generated:
        return True
    # Some titles include punctuation or spacing variants. Keep this conservative:
    # only compare an alphanumeric/punctuation-stripped form as a fallback.
    compact_value = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    compact_generated = re.sub(r"[\W_]+", "", generated, flags=re.UNICODE)
    if compact_value and compact_value in compact_generated:
        return True
    return False


def title_consistency_flags(generated_text, music_title, music_artist):
    title_hit = mentions_value(generated_text, music_title)
    artist_hit = mentions_value(generated_text, music_artist)
    checked = title_hit is not None or artist_hit is not None
    if not checked:
        consistency = None
    elif title_hit is False:
        consistency = False
    elif artist_hit is False:
        consistency = False
    else:
        consistency = True
    return {
        "generated_mentions_top1_title": "" if title_hit is None else int(title_hit),
        "generated_mentions_top1_artist": "" if artist_hit is None else int(artist_hit),
        "title_consistency": "" if consistency is None else int(consistency),
        "needs_manual_review": 1 if consistency is False else 0,
    }


def summarize_generation_rows(rows):
    def summarize_subset(items):
        checked = [r for r in items if str(r.get("title_consistency", "")) != ""]
        consistent = [r for r in checked if to_int(r.get("title_consistency")) == 1]
        review = [r for r in items if to_int(r.get("needs_manual_review")) == 1]
        return {
            "num_samples": len(items),
            "fallback_count": sum(to_int(r.get("is_fallback")) for r in items),
            "fallback_rate": sum(to_int(r.get("is_fallback")) for r in items) / max(len(items), 1),
            "title_consistency_checked": len(checked),
            "title_consistency_rate": len(consistent) / max(len(checked), 1),
            "needs_manual_review_count": len(review),
            "needs_manual_review_rate": len(review) / max(len(items), 1),
            "bertscore_precision": mean_metric(items, "bertscore_precision"),
            "bertscore_recall": mean_metric(items, "bertscore_recall"),
            "bertscore_f1": mean_metric(items, "bertscore_f1"),
            "infolm_ab_divergence": mean_metric(items, "infolm_ab_divergence"),
            "infolm_l2_distance": mean_metric(items, "infolm_l2_distance"),
            "infolm_fisher_rao": mean_metric(items, "infolm_fisher_rao"),
        }

    correct = [r for r in rows if to_int(r.get("top1_is_gt")) == 1]
    incorrect = [r for r in rows if to_int(r.get("top1_is_gt")) == 0]
    return {
        "all": summarize_subset(rows),
        "top1_correct_only": summarize_subset(correct),
        "top1_incorrect_only": summarize_subset(incorrect),
    }


@torch.no_grad()
def generate_top1(
    model,
    sample,
    tokenizer,
    device,
    active_modalities,
    prompt_variant,
    top1_music_feat,
    t3_text,
    title_ref_text,
    inject_title,
):
    bf16 = torch.bfloat16
    video_feat = sample["video_feat"].unsqueeze(0).to(device, dtype=bf16)
    ltp_feat = sample["ltp_feat"].unsqueeze(0).to(device, dtype=bf16)
    text_feat = sample["text_feat"].unsqueeze(0).to(device, dtype=bf16)
    music_feat = top1_music_feat.unsqueeze(0).unsqueeze(0).to(device, dtype=bf16)

    input_ids, attention_mask = core.prompt_tensors_for_variant(
        sample=sample,
        tokenizer=tokenizer,
        device=device,
        active_modalities=active_modalities,
        prompt_variant=prompt_variant,
        t3_text=t3_text,
        t4_ref=title_ref_text,
        inject_title=inject_title,
    )

    music_title = None
    music_artist = None
    if inject_title and title_ref_text:
        from dataset import extract_music_title
        music_title, music_artist = extract_music_title(title_ref_text)

    rank_token_id = getattr(model, "rank_token_id", None)
    if rank_token_id is None or (input_ids == rank_token_id).sum().item() == 0:
        return "", True, music_title, music_artist

    generated_ids, _ = model.generate(
        video_feat=video_feat,
        music_candidates=music_feat,
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


def run_one_exp(exp_name):
    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer
    from tqdm import tqdm

    ltp_mode = core.EXP_TO_LTP_MODE[exp_name]
    active_modalities = core.EXP_TO_MODALITIES[exp_name]
    output_dir = os.path.join(core.BASE_DIR, "checkpoints", exp_name)
    ckpt_dir = os.path.join(output_dir, CKPT_NAME)
    detail_dir = os.path.join(output_dir, "detailed_eval")

    rank_prefix = ranking_prefix(exp_name)
    ranking_csv = os.path.join(detail_dir, f"{rank_prefix}_ranking_samples.csv")
    if not os.path.exists(ranking_csv):
        raise FileNotFoundError(f"Missing ranking CSV. Run run_eval_500pool_detailed.py first: {ranking_csv}")

    out_prefix = output_prefix(exp_name)
    generation_csv = os.path.join(detail_dir, f"{out_prefix}_generation_samples.csv")
    raw_generation_csv = os.path.join(detail_dir, f"{out_prefix}_generation_samples_raw.csv")
    merged_csv = os.path.join(detail_dir, f"{out_prefix}_samples_merged.csv")
    merged_jsonl = os.path.join(detail_dir, f"{out_prefix}_samples_merged.jsonl")
    summary_path = os.path.join(detail_dir, f"{out_prefix}_summary.json")
    log_path = os.path.join(detail_dir, f"eval_top1_generation_{exp_name}.log")
    logger = core.setup_logger(log_path)

    logger.info("=" * 72)
    logger.info("Top-1 generation from existing ranking | exp=%s ckpt=%s pool=%d", exp_name, CKPT_NAME, POOL_SIZE)
    logger.info("Ranking CSV: %s", ranking_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is expected for LLaMA evaluation.")

    model_cfg = ModelConfig(
        llama_model_name=core.LLAMA_MODEL,
        video_dim=768,
        music_dim=768,
        text_dim=512,
        ltp_dim=256,
        num_candidates=1,
        active_modalities=active_modalities,
        music_token_offset=3,
        rank_special_token="[RANK]",
    )
    train_cfg = TrainConfig(output_dir=output_dir, pointwise_eval_batch_size=32, music_pool_size=POOL_SIZE)

    tokenizer = LlamaTokenizer.from_pretrained(core.LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    ltp_dict = core.load_ltp_dict(
        core.LTP_H5[ltp_mode],
        ltp_mode,
        cache_path=os.path.join(core.CACHE_DIR, "ltp"),
        logger=logger,
    )
    test_dataset, all_music_features, all_music_ids, _ = core.build_test_data(
        model_cfg, train_cfg, tokenizer, ltp_dict, logger
    )
    conv_t3, conv_t4, _ = core.load_reference_maps()
    model = core.load_model(ckpt_dir, model_cfg, tokenizer, logger)

    id_to_index = {sid: i for i, sid in enumerate(all_music_ids)}
    ranking_rows = read_csv(ranking_csv)
    if MAX_SAMPLES is not None:
        ranking_rows = ranking_rows[:MAX_SAMPLES]

    generation_rows = []
    fallback_count = 0
    missing_top1_count = 0
    started_at = _dt.datetime.now()

    for rank_row in tqdm(ranking_rows, desc=f"Top-1 generation {exp_name}"):
        sample_idx = to_int(rank_row["sample_idx"])
        sample = test_dataset[sample_idx]
        query_video_id = rank_row["video_id"]
        gt_music_id = rank_row["gt_music_id"]
        top1_music_id = rank_row["top1_music_id"]
        top1_video_id = str(top1_music_id)[:11]

        t3_text = conv_t3.get(query_video_id, "")
        reference_text = conv_t4.get(query_video_id, "")
        top1_reference_text = conv_t4.get(top1_video_id, "")
        title_ref_text = top1_reference_text if top1_reference_text else reference_text
        title_source = "top1_reference" if top1_reference_text else "query_reference_fallback"

        top1_index = id_to_index.get(top1_music_id)
        if top1_index is None:
            missing_top1_count += 1
            generated_text, is_fallback, music_title, music_artist = "", True, None, None
        else:
            top1_music_feat = all_music_features[top1_index]
            try:
                generated_text, is_fallback, music_title, music_artist = generate_top1(
                    model=model,
                    sample=sample,
                    tokenizer=tokenizer,
                    device=device,
                    active_modalities=active_modalities,
                    prompt_variant=PROMPT_VARIANT,
                    top1_music_feat=top1_music_feat,
                    t3_text=t3_text,
                    title_ref_text=title_ref_text,
                    inject_title=INJECT_TITLE,
                )
            except Exception as exc:
                logger.warning(
                    "Top-1 generation failed: sample_idx=%s video=%s top1=%s error=%s",
                    sample_idx,
                    query_video_id,
                    top1_music_id,
                    exc,
                )
                generated_text, is_fallback, music_title, music_artist = "", True, None, None

        fallback_count += int(is_fallback)
        consistency = title_consistency_flags(generated_text, music_title, music_artist)
        generation_rows.append({
            "sample_idx": str(sample_idx),
            "video_id": query_video_id,
            "gt_music_id": gt_music_id,
            "top1_music_id": top1_music_id,
            "top1_video_id": top1_video_id,
            "top1_is_gt": to_int(rank_row.get("top1_is_gt")),
            "rank": to_int(rank_row.get("rank")),
            "R@1": to_int(rank_row.get("R@1")),
            "R@5": to_int(rank_row.get("R@5")),
            "R@10": to_int(rank_row.get("R@10")),
            "pool_size": to_int(rank_row.get("pool_size"), POOL_SIZE),
            "generation_mode": "top1_end_to_end_from_existing_ranking",
            "prompt_variant": PROMPT_VARIANT,
            "inject_title": int(INJECT_TITLE),
            "title_source": title_source,
            "music_title": music_title,
            "music_artist": music_artist,
            "user_text": t3_text,
            "generated_text": generated_text,
            "reference_text": reference_text,
            "top1_reference_text": top1_reference_text,
            **consistency,
            "is_fallback": int(is_fallback),
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "infolm_ab_divergence": None,
            "infolm_l2_distance": None,
            "infolm_fisher_rao": None,
        })

    # Save raw generations before expensive metrics and merging. This protects a
    # long run from losing all generated text if a later metric or merge step fails.
    write_csv(raw_generation_csv, generation_rows)
    logger.info("Saved raw top1 generation samples before metrics: %s", raw_generation_csv)

    generation_summary = {
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(len(generation_rows), 1)),
        "missing_top1_count": int(missing_top1_count),
        "reference_definition": "query GT t4 reference; Top-1 incorrect samples therefore reflect ranking and generation jointly",
    }
    generation_summary.update(core.add_bertscore_to_rows(generation_rows, logger))
    generation_summary.update(core.add_infolm_to_rows(generation_rows, per_sample=KEEP_PER_SAMPLE_INFOLM, logger=logger))
    split_summary = summarize_generation_rows(generation_rows)

    write_csv(generation_csv, generation_rows)
    logger.info("Saved metric-enriched top1 generation samples: %s", generation_csv)

    merged_rows = merge_rows_by_sample_idx(ranking_rows, generation_rows)
    write_csv(merged_csv, merged_rows)
    write_jsonl(merged_jsonl, merged_rows)

    summary = {
        "exp_name": exp_name,
        "checkpoint": CKPT_NAME,
        "pool_size": POOL_SIZE,
        "prompt_variant": PROMPT_VARIANT,
        "ranking_source_csv": ranking_csv,
        "generation_mode": "top1_end_to_end_from_existing_ranking",
        "max_samples": MAX_SAMPLES,
        "inject_title": bool(INJECT_TITLE),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "generation": generation_summary,
        "split_generation": split_summary,
        "outputs": {
            "generation_csv": generation_csv,
            "raw_generation_csv": raw_generation_csv,
            "merged_csv": merged_csv,
            "merged_jsonl": merged_jsonl,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved top1 generation samples: %s", generation_csv)
    logger.info("Saved top1 merged samples: %s", merged_csv)
    logger.info("Saved top1 summary: %s", summary_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return summary


def main():
    exp_names = core.EXP_NAMES if EXP_NAME == "all" else [EXP_NAME]
    summaries = {}
    for exp_name in exp_names:
        summaries[exp_name] = run_one_exp(exp_name)

    tag_parts = [EXP_NAME, f"{POOL_SIZE}pool"]
    if OUTPUT_TAG:
        tag_parts.append(OUTPUT_TAG)
    if PROMPT_VARIANT:
        tag_parts.append(f"prompt_{PROMPT_VARIANT}")
    if RANKING_RESULT_TAG:
        tag_parts.append(f"ranking_{RANKING_RESULT_TAG}")
    summary_tag = "_".join(tag_parts)
    all_summary_path = os.path.join(
        core.BASE_DIR,
        "checkpoints",
        f"top1_generation_summary_{summary_tag}.json",
    )
    with open(all_summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Saved: {all_summary_path}")


if __name__ == "__main__":
    main()
