# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Counterfactual modality-removal generation for explanation faithfulness.

Usage:
  Open this file in VSCode and click Run.

This script fixes the exp_01 best checkpoint and generates explanations for the
same sampled test instances under six inference-time conditions:
  full, wo_video, wo_audio_feature_only, wo_audio_all, wo_prompt, wo_ltp

Unlike exp_04/05/06/07, this does not load separately trained ablation models.
It keeps the exp_01 model fixed and removes one input modality at inference
time, which matches the counterfactual faithfulness analysis design.
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

CONDITIONS = [
    "full",
    "wo_video",
    "wo_audio_feature_only",
    "wo_audio_all",
    "wo_prompt",
    "wo_ltp",
]

CONDITION_DESCRIPTIONS = {
    "full": "all modalities available; title/artist injection follows INJECT_TITLE",
    "wo_video": "video feature is zeroed; title/artist injection follows INJECT_TITLE",
    "wo_audio_feature_only": "music feature is zeroed, but title/artist metadata is kept",
    "wo_audio_all": "music feature is zeroed and title/artist metadata injection is disabled",
    "wo_prompt": "text prompt feature and user text are removed; title/artist metadata is kept",
    "wo_ltp": "long-term preference feature is zeroed; title/artist metadata is kept",
}

OUTPUT_DIR = os.path.join(core.BASE_DIR, "checkpoints", "faithfulness_analysis")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "counterfactual_generations.csv")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "counterfactual_generations.jsonl")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "counterfactual_generation_summary.json")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def choose_sample_indices(n_total, n_samples, seed):
    rng = random.Random(seed)
    n = min(n_samples, n_total)
    return sorted(rng.sample(range(n_total), n))


def apply_condition(sample, condition, device):
    bf16 = torch.bfloat16
    video_feat = sample["video_feat"].unsqueeze(0).to(device, dtype=bf16)
    ltp_feat = sample["ltp_feat"].unsqueeze(0).to(device, dtype=bf16)
    text_feat = sample["text_feat"].unsqueeze(0).to(device, dtype=bf16)
    music_feat = sample["pos_music_feat"].unsqueeze(0).unsqueeze(0).to(device, dtype=bf16)

    if condition == "wo_video":
        video_feat = torch.zeros_like(video_feat)
    elif condition in {"wo_audio", "wo_audio_feature_only", "wo_audio_all"}:
        music_feat = torch.zeros_like(music_feat)
    elif condition == "wo_prompt":
        text_feat = torch.zeros_like(text_feat)
    elif condition == "wo_ltp":
        ltp_feat = torch.zeros_like(ltp_feat)
    elif condition != "full":
        raise ValueError(f"Unknown condition: {condition}")

    return video_feat, ltp_feat, text_feat, music_feat


@torch.no_grad()
def generate_condition(model, sample, tokenizer, device, condition, active_modalities, t3_text, t4_ref):
    video_feat, ltp_feat, text_feat, music_feat = apply_condition(sample, condition, device)
    prompt_user_text = "" if condition == "wo_prompt" else t3_text
    inject_title_for_condition = INJECT_TITLE and condition != "wo_audio_all"

    input_ids, attention_mask = core.prompt_tensors_for_variant(
        sample=sample,
        tokenizer=tokenizer,
        device=device,
        active_modalities=active_modalities,
        prompt_variant=PROMPT_VARIANT,
        t3_text=prompt_user_text,
        t4_ref=t4_ref,
        inject_title=inject_title_for_condition,
    )

    rank_token_id = getattr(model, "rank_token_id", None)
    if rank_token_id is None or (input_ids == rank_token_id).sum().item() == 0:
        return "", True, inject_title_for_condition

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
    return decoded, not bool(decoded), inject_title_for_condition


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


def main():
    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer

    ensure_dir(OUTPUT_DIR)
    logger = core.setup_logger(os.path.join(OUTPUT_DIR, "counterfactual_generation.log"))
    started_at = datetime.now()

    exp_name = EXP_NAME
    ltp_mode = core.EXP_TO_LTP_MODE[exp_name]
    active_modalities = core.EXP_TO_MODALITIES[exp_name]
    output_dir = os.path.join(core.BASE_DIR, "checkpoints", exp_name)
    ckpt_dir = os.path.join(output_dir, CKPT_NAME)

    logger.info("Counterfactual faithfulness generation")
    logger.info("exp=%s ckpt=%s n_samples=%d seed=%d", exp_name, CKPT_NAME, N_SAMPLES, SAMPLE_SEED)

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

    for pos, sample_idx in enumerate(sample_indices, start=1):
        sample = test_dataset[sample_idx]
        video_id = sample.get("video_id", "")
        gt_music_id = sample.get("gt_music_id", "")
        t3_text = conv_t3.get(video_id, "")
        t4_ref = conv_t4.get(video_id, "")

        music_title, music_artist = None, None
        if t4_ref:
            from dataset import extract_music_title
            music_title, music_artist = extract_music_title(t4_ref)

        logger.info("[%d/%d] sample_idx=%d video=%s", pos, len(sample_indices), sample_idx, video_id)
        for condition in CONDITIONS:
            try:
                generated_text, is_fallback, inject_title_used = generate_condition(
                    model=model,
                    sample=sample,
                    tokenizer=tokenizer,
                    device=device,
                    condition=condition,
                    active_modalities=active_modalities,
                    t3_text=t3_text,
                    t4_ref=t4_ref,
                )
            except Exception as exc:
                logger.warning(
                    "generation failed: sample_idx=%s video=%s condition=%s error=%s",
                    sample_idx,
                    video_id,
                    condition,
                    exc,
                )
                generated_text, is_fallback, inject_title_used = "", True, INJECT_TITLE and condition != "wo_audio_all"
            fallback_count += int(is_fallback)
            rows.append({
                "sample_idx": sample_idx,
                "video_id": video_id,
                "gt_music_id": gt_music_id,
                "condition": condition,
                "condition_description": CONDITION_DESCRIPTIONS.get(condition, ""),
                "generated_text": generated_text,
                "reference_text": t4_ref,
                "user_text": t3_text,
                "music_title": music_title,
                "music_artist": music_artist,
                "inject_title_used": int(inject_title_used),
                "is_fallback": int(is_fallback),
            })

    write_csv(OUTPUT_CSV, rows)
    write_jsonl(OUTPUT_JSONL, rows)

    summary = {
        "exp_name": exp_name,
        "checkpoint": CKPT_NAME,
        "n_samples": len(sample_indices),
        "n_rows": len(rows),
        "conditions": CONDITIONS,
        "condition_descriptions": CONDITION_DESCRIPTIONS,
        "sample_seed": SAMPLE_SEED,
        "prompt_variant": PROMPT_VARIANT,
        "inject_title": INJECT_TITLE,
        "fallback_count": fallback_count,
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
