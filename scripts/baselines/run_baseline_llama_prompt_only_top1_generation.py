# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
VSCode-run baseline: LLaMA/Vicuna prompting-only generation.

This baseline removes learned multimodal projectors, LTP vectors, and model-side
recommendation reasoning. It reuses an existing Top-1 recommendation list only
to decide which song title/artist is shown to the base LLM, then asks the base
LLM to generate a recommendation reason from text prompt + selected song
metadata.

Use this as an LLM-only explanation baseline. It is not a full 500-candidate
LLM retrieval baseline, because scoring 500 candidates with a 7B LLM for every
test sample would be prohibitively slow and difficult to keep comparable.
"""

import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from scripts.eval_main import run_eval_500pool_detailed as core
from dataset import extract_music_title


BASE_DIR = str(PROJECT_ROOT)
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints", "baseline_llama_prompt_only", "detailed_eval")

BASE_MODEL = core.LLAMA_MODEL
SOURCE_EXP = "exp_01"
SOURCE_TAG = "top1_prompt_original"
POOL_SIZE = 500
MAX_SAMPLES = None
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_paths() -> Tuple[str, str]:
    detailed_dir = os.path.join(BASE_DIR, "checkpoints", SOURCE_EXP, "detailed_eval")
    merged = os.path.join(
        detailed_dir,
        f"{SOURCE_EXP}_best_{POOL_SIZE}pool_{SOURCE_TAG}_samples_merged.csv",
    )
    ranking = os.path.join(
        detailed_dir,
        f"{SOURCE_EXP}_best_{POOL_SIZE}pool_ranking_samples.csv",
    )
    if not os.path.exists(merged):
        raise FileNotFoundError(f"Missing source merged CSV: {merged}")
    if not os.path.exists(ranking):
        raise FileNotFoundError(f"Missing source ranking CSV: {ranking}")
    return merged, ranking


def load_title_lookup(rows: List[dict]) -> Dict[str, Tuple[str, str]]:
    lookup: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        music_id = row.get("top1_music_id") or row.get("gt_music_id")
        text = (
            row.get("top1_reference_text")
            or row.get("reference_text")
            or row.get("generated_text")
            or ""
        )
        title, artist = extract_music_title(text)
        if music_id and (title or artist):
            lookup[music_id] = (title or "the selected song", artist or "the artist")
    return lookup


def clean_generation(text: str) -> str:
    text = text.strip()
    markers = [
        "Recommendation:",
        "### Assistant:",
        "Assistant:",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
    return re.sub(r"\s+", " ", text)


def build_prompt(user_text: str, title: str, artist: str) -> str:
    return (
        "### User:\n"
        f"{user_text}\n\n"
        "### Recommender:\n"
        f"Music title: {title}; Artist: {artist}.\n"
        "Generate one concise recommendation reason explaining why this music fits the video.\n"
        "Recommendation:"
    )


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=TEMPERATURE > 0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if TEMPERATURE > 0:
        gen_kwargs["temperature"] = TEMPERATURE
    output_ids = model.generate(
        **inputs,
        **gen_kwargs,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return clean_generation(tokenizer.decode(new_tokens, skip_special_tokens=True))


def title_consistency(generated_text: str, title: str, artist: str) -> int:
    text = generated_text.lower()
    title_ok = bool(title) and title.lower() in text
    artist_ok = bool(artist) and artist.lower() in text
    return int(title_ok or artist_ok)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_merged, source_ranking = source_paths()
    source_rows = read_csv(source_merged)
    ranking_rows = read_csv(source_ranking)
    title_lookup = load_title_lookup(source_rows)

    if MAX_SAMPLES is not None:
        source_rows = source_rows[:MAX_SAMPLES]
        ranking_rows = ranking_rows[:MAX_SAMPLES]

    print(f"Loading base LLM: {BASE_MODEL}")
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    by_sample = {str(r["sample_idx"]): r for r in source_rows}
    rows = []
    for i, rrow in enumerate(ranking_rows):
        sample_idx = str(rrow["sample_idx"])
        source = by_sample.get(sample_idx, {})
        if i % 50 == 0:
            print(f"[{i}/{len(ranking_rows)}] sample={sample_idx}")

        top1_music_id = rrow.get("top1_music_id") or source.get("top1_music_id") or source.get("gt_music_id")
        title, artist = title_lookup.get(top1_music_id, ("the selected song", "the artist"))
        user_text = source.get("user_text") or source.get("t3_text") or "Please recommend suitable background music for my video."
        reference_text = source.get("reference_text") or ""
        prompt = build_prompt(user_text, title, artist)
        generated_text = generate_one(model, tokenizer, prompt)

        rows.append(
            {
                "sample_idx": sample_idx,
                "video_id": rrow.get("video_id") or source.get("video_id"),
                "gt_music_id": rrow.get("gt_music_id") or source.get("gt_music_id"),
                "top1_music_id": top1_music_id,
                "rank": rrow.get("rank"),
                "R@1": rrow.get("R@1"),
                "R@5": rrow.get("R@5"),
                "R@10": rrow.get("R@10"),
                "music_title": title,
                "music_artist": artist,
                "user_text": user_text,
                "generated_text": generated_text,
                "reference_text": reference_text,
                "title_or_artist_consistent": title_consistency(generated_text, title, artist),
            }
        )

    logger = core.setup_logger(os.path.join(OUTPUT_DIR, "llama_prompt_only_generation.log"))
    bert_summary = core.add_bertscore_to_rows(rows, logger)
    infolm_summary = core.add_infolm_to_rows(rows, per_sample=core.KEEP_PER_SAMPLE_INFOLM, logger=logger)

    n = len(rows)
    valid = [r for r in rows if r["generated_text"].strip() and r["reference_text"].strip()]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": "llama_prompt_only_top1_generation",
        "base_model": BASE_MODEL,
        "source_exp": SOURCE_EXP,
        "source_tag": SOURCE_TAG,
        "pool_size": POOL_SIZE,
        "n": n,
        "n_valid": len(valid),
        "title_or_artist_consistency_rate": sum(int(r["title_or_artist_consistent"]) for r in rows) / n if n else None,
    }
    summary.update(bert_summary)
    summary.update(infolm_summary)

    samples_csv = os.path.join(OUTPUT_DIR, "llama_prompt_only_top1_generation_samples.csv")
    summary_json = os.path.join(OUTPUT_DIR, "llama_prompt_only_top1_generation_summary.json")
    write_csv(samples_csv, rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved: {samples_csv}")
    print(f"Saved: {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
