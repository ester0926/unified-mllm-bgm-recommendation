# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Top-1 counterfactual preference generation for explanation faithfulness.

Usage:
  Open this file in VSCode and click Run.

This is the end-to-end counterpart of run_preference_counterfactual_generation.py.
It reads existing 500-pool ranking outputs and generates explanations for the
actual Top-1 recommendation, while replacing only the natural-language prompt.
The precomputed text feature remains unchanged, so this remains a prompt-level
preference sensitivity test rather than a full reranking test.
"""

import csv
import json
import os
import random
from datetime import datetime

import torch

from scripts.eval_main import run_eval_500pool_detailed as core
from scripts.eval_main.run_eval_500pool_top1_generation_from_ranking import (
    title_consistency_flags,
    to_int,
)


# =============================================================================
# USER SETTINGS
# =============================================================================

EXP_NAME = "exp_01"
CKPT_NAME = "best"
POOL_SIZE = 500
RANKING_RESULT_TAG = ""
N_SAMPLES = 200
SAMPLE_SEED = 20260516
PROMPT_VARIANT = "original"
INJECT_TITLE = True

PREFERENCE_VARIANTS = {
    "original": None,
    "cf_upbeat_electronic": (
        "I prefer upbeat electronic background music with a bright mood, clear "
        "rhythm, energetic beats, and a modern dance or pop feeling."
    ),
    "cf_lyrical_piano": (
        "I prefer lyrical piano background music with a slow tempo, soft acoustic "
        "texture, gentle melody, and a calm sentimental mood."
    ),
}

OUTPUT_DIR = os.path.join(core.BASE_DIR, "checkpoints", "faithfulness_analysis")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "preference_counterfactual_generations_top1.csv")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "preference_counterfactual_generations_top1.jsonl")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "preference_counterfactual_generation_top1_summary.json")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ranking_csv_path():
    tag = f"_{RANKING_RESULT_TAG}" if RANKING_RESULT_TAG else ""
    prefix = f"{EXP_NAME}_{CKPT_NAME}_{POOL_SIZE}pool{tag}"
    return os.path.join(core.BASE_DIR, "checkpoints", EXP_NAME, "detailed_eval", f"{prefix}_ranking_samples.csv")


def choose_ranking_rows(ranking_rows, n_samples, seed):
    rng = random.Random(seed)
    n = min(n_samples, len(ranking_rows))
    return sorted(rng.sample(ranking_rows, n), key=lambda r: to_int(r.get("sample_idx")))


@torch.no_grad()
def generate_for_preference(
    model,
    sample,
    tokenizer,
    device,
    active_modalities,
    top1_music_feat,
    user_text,
    title_ref_text,
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
        prompt_variant=PROMPT_VARIANT,
        t3_text=user_text,
        t4_ref=title_ref_text,
        inject_title=INJECT_TITLE,
    )

    music_title = None
    music_artist = None
    if INJECT_TITLE and title_ref_text:
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


def main():
    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer
    from tqdm import tqdm

    ensure_dir(OUTPUT_DIR)
    logger = core.setup_logger(os.path.join(OUTPUT_DIR, "preference_counterfactual_generation_top1.log"))
    started_at = datetime.now()

    ranking_csv = ranking_csv_path()
    if not os.path.exists(ranking_csv):
        raise FileNotFoundError(f"Missing ranking CSV: {ranking_csv}")

    ltp_mode = core.EXP_TO_LTP_MODE[EXP_NAME]
    active_modalities = core.EXP_TO_MODALITIES[EXP_NAME]
    output_dir = os.path.join(core.BASE_DIR, "checkpoints", EXP_NAME)
    ckpt_dir = os.path.join(output_dir, CKPT_NAME)

    logger.info("Top-1 preference counterfactual generation")
    logger.info("exp=%s ckpt=%s n_samples=%d seed=%d", EXP_NAME, CKPT_NAME, N_SAMPLES, SAMPLE_SEED)
    logger.info("ranking_csv=%s", ranking_csv)

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
    test_dataset, all_music_features, all_music_ids, _ = core.build_test_data(model_cfg, train_cfg, tokenizer, ltp_dict, logger)
    conv_t3, conv_t4, _ = core.load_reference_maps()
    model = core.load_model(ckpt_dir, model_cfg, tokenizer, logger)

    id_to_index = {sid: i for i, sid in enumerate(all_music_ids)}
    selected_rows = choose_ranking_rows(read_csv(ranking_csv), N_SAMPLES, SAMPLE_SEED)
    rows = []
    fallback_count = 0
    missing_top1_count = 0

    for rank_row in tqdm(selected_rows, desc="Top-1 preference CF generation"):
        sample_idx = to_int(rank_row["sample_idx"])
        sample = test_dataset[sample_idx]
        video_id = rank_row["video_id"]
        gt_music_id = rank_row["gt_music_id"]
        top1_music_id = rank_row["top1_music_id"]
        top1_video_id = str(top1_music_id)[:11]
        original_t3 = conv_t3.get(video_id, "")
        reference_text = conv_t4.get(video_id, "")
        top1_reference_text = conv_t4.get(top1_video_id, "")
        title_ref_text = top1_reference_text if top1_reference_text else reference_text
        title_source = "top1_reference" if top1_reference_text else "query_reference_fallback"

        top1_index = id_to_index.get(top1_music_id)
        if top1_index is None:
            missing_top1_count += 1
            logger.warning("Missing top1 music id: sample_idx=%s top1=%s", sample_idx, top1_music_id)
            continue
        top1_music_feat = all_music_features[top1_index]

        for variant_name, variant_prompt in PREFERENCE_VARIANTS.items():
            user_text = original_t3 if variant_prompt is None else variant_prompt
            try:
                generated_text, is_fallback, music_title, music_artist = generate_for_preference(
                    model=model,
                    sample=sample,
                    tokenizer=tokenizer,
                    device=device,
                    active_modalities=active_modalities,
                    top1_music_feat=top1_music_feat,
                    user_text=user_text,
                    title_ref_text=title_ref_text,
                )
            except Exception as exc:
                logger.warning("generation failed: sample_idx=%s variant=%s error=%s", sample_idx, variant_name, exc)
                generated_text, is_fallback, music_title, music_artist = "", True, None, None

            fallback_count += int(is_fallback)
            consistency = title_consistency_flags(generated_text, music_title, music_artist)
            rows.append({
                "sample_idx": str(sample_idx),
                "video_id": video_id,
                "gt_music_id": gt_music_id,
                "top1_music_id": top1_music_id,
                "top1_video_id": top1_video_id,
                "top1_is_gt": to_int(rank_row.get("top1_is_gt")),
                "rank": to_int(rank_row.get("rank")),
                "preference_variant": variant_name,
                "generation_mode": "top1_prompt_level_preference_counterfactual",
                "prompt_variant": PROMPT_VARIANT,
                "title_source": title_source,
                "user_text": user_text,
                "original_user_text": original_t3,
                "generated_text": generated_text,
                "reference_text": reference_text,
                "top1_reference_text": top1_reference_text,
                "music_title": music_title,
                "music_artist": music_artist,
                **consistency,
                "is_fallback": int(is_fallback),
                "scope_note": "top1_generation_prompt_text_changed_only_precomputed_text_feature_unchanged",
            })

    write_csv(OUTPUT_CSV, rows)
    write_jsonl(OUTPUT_JSONL, rows)

    summary = {
        "exp_name": EXP_NAME,
        "checkpoint": CKPT_NAME,
        "pool_size": POOL_SIZE,
        "ranking_source_csv": ranking_csv,
        "generation_mode": "top1_prompt_level_preference_counterfactual",
        "n_samples": len(selected_rows),
        "n_rows": len(rows),
        "preference_variants": PREFERENCE_VARIANTS,
        "sample_seed": SAMPLE_SEED,
        "prompt_variant": PROMPT_VARIANT,
        "inject_title": INJECT_TITLE,
        "fallback_count": fallback_count,
        "missing_top1_count": missing_top1_count,
        "scope_note": "Prompt-level preference sensitivity; ranking and precomputed text features are unchanged.",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "csv": OUTPUT_CSV,
            "jsonl": OUTPUT_JSONL,
        },
    }
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved: %s", OUTPUT_CSV)
    logger.info("Saved: %s", OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
