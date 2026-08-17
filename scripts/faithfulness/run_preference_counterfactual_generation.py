# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Counterfactual preference generation for explanation faithfulness.

Usage:
  Open this file in VSCode and click Run.

This script tests whether generated explanations respond to changed natural
language user preferences. It keeps the exp_01 checkpoint and candidate music
fixed, then replaces the user prompt with controlled counterfactual preference
statements.

Important scope:
  The current dataset stores precomputed text features, so this script changes
  only the LLM prompt text, not the CLIP/text embedding used by the model. It is
  therefore a prompt-level preference sensitivity test for explanations, not a
  full reranking test with newly embedded counterfactual preferences.
"""

import csv
import json
import os
import random
from datetime import datetime

import torch

from scripts.eval_main import run_eval_500pool_detailed as core


# =============================================================================
# USER SETTINGS
# =============================================================================

EXP_NAME = "exp_01"
CKPT_NAME = "best"
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "preference_counterfactual_generations.csv")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "preference_counterfactual_generations.jsonl")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "preference_counterfactual_generation_summary.json")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def choose_sample_indices(n_total, n_samples, seed):
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), min(n_samples, n_total)))


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


@torch.no_grad()
def generate_for_preference(model, sample, tokenizer, device, active_modalities, t3_text, t4_ref):
    bf16 = torch.bfloat16
    video_feat = sample["video_feat"].unsqueeze(0).to(device, dtype=bf16)
    ltp_feat = sample["ltp_feat"].unsqueeze(0).to(device, dtype=bf16)
    text_feat = sample["text_feat"].unsqueeze(0).to(device, dtype=bf16)
    music_feat = sample["pos_music_feat"].unsqueeze(0).unsqueeze(0).to(device, dtype=bf16)

    input_ids, attention_mask = core.prompt_tensors_for_variant(
        sample=sample,
        tokenizer=tokenizer,
        device=device,
        active_modalities=active_modalities,
        prompt_variant=PROMPT_VARIANT,
        t3_text=t3_text,
        t4_ref=t4_ref,
        inject_title=INJECT_TITLE,
    )

    music_title = None
    music_artist = None
    if INJECT_TITLE and t4_ref:
        from dataset import extract_music_title
        music_title, music_artist = extract_music_title(t4_ref)

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
    logger = core.setup_logger(os.path.join(OUTPUT_DIR, "preference_counterfactual_generation.log"))
    started_at = datetime.now()

    ltp_mode = core.EXP_TO_LTP_MODE[EXP_NAME]
    active_modalities = core.EXP_TO_MODALITIES[EXP_NAME]
    output_dir = os.path.join(core.BASE_DIR, "checkpoints", EXP_NAME)
    ckpt_dir = os.path.join(output_dir, CKPT_NAME)

    logger.info("Preference counterfactual generation")
    logger.info("exp=%s ckpt=%s n_samples=%d seed=%d", EXP_NAME, CKPT_NAME, N_SAMPLES, SAMPLE_SEED)

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
    train_cfg = TrainConfig(output_dir=output_dir, pointwise_eval_batch_size=32, music_pool_size=500)

    tokenizer = LlamaTokenizer.from_pretrained(core.LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    ltp_dict = core.load_ltp_dict(
        core.LTP_H5[ltp_mode],
        ltp_mode,
        cache_path=os.path.join(core.CACHE_DIR, "ltp"),
        logger=logger,
    )
    test_dataset, _, _, _ = core.build_test_data(model_cfg, train_cfg, tokenizer, ltp_dict, logger)
    conv_t3, conv_t4, _ = core.load_reference_maps()
    model = core.load_model(ckpt_dir, model_cfg, tokenizer, logger)

    sample_indices = choose_sample_indices(len(test_dataset), N_SAMPLES, SAMPLE_SEED)
    rows = []
    fallback_count = 0

    for pos, sample_idx in enumerate(tqdm(sample_indices, desc="Preference CF generation"), start=1):
        sample = test_dataset[sample_idx]
        video_id = sample.get("video_id", "")
        gt_music_id = sample.get("gt_music_id", "")
        original_t3 = conv_t3.get(video_id, "")
        t4_ref = conv_t4.get(video_id, "")

        for variant_name, variant_prompt in PREFERENCE_VARIANTS.items():
            user_text = original_t3 if variant_prompt is None else variant_prompt
            try:
                generated_text, is_fallback, music_title, music_artist = generate_for_preference(
                    model=model,
                    sample=sample,
                    tokenizer=tokenizer,
                    device=device,
                    active_modalities=active_modalities,
                    t3_text=user_text,
                    t4_ref=t4_ref,
                )
            except Exception as exc:
                logger.warning(
                    "generation failed: sample_idx=%s video=%s variant=%s error=%s",
                    sample_idx,
                    video_id,
                    variant_name,
                    exc,
                )
                generated_text, is_fallback, music_title, music_artist = "", True, None, None
            fallback_count += int(is_fallback)
            rows.append({
                "sample_idx": str(sample_idx),
                "video_id": video_id,
                "gt_music_id": gt_music_id,
                "preference_variant": variant_name,
                "user_text": user_text,
                "original_user_text": original_t3,
                "generated_text": generated_text,
                "reference_text": t4_ref,
                "music_title": music_title,
                "music_artist": music_artist,
                "is_fallback": int(is_fallback),
                "scope_note": "prompt_text_changed_only_precomputed_text_feature_unchanged",
            })
        if pos % 25 == 0:
            logger.info("Generated %d/%d sampled instances", pos, len(sample_indices))

    write_csv(OUTPUT_CSV, rows)
    write_jsonl(OUTPUT_JSONL, rows)

    summary = {
        "exp_name": EXP_NAME,
        "checkpoint": CKPT_NAME,
        "n_samples": len(sample_indices),
        "n_rows": len(rows),
        "preference_variants": PREFERENCE_VARIANTS,
        "sample_seed": SAMPLE_SEED,
        "prompt_variant": PROMPT_VARIANT,
        "inject_title": INJECT_TITLE,
        "fallback_count": fallback_count,
        "scope_note": "This is a prompt-level preference sensitivity test because text features are precomputed.",
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
