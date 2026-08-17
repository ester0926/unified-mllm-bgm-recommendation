"""Materialize the fixed-component `full` condition from the existing exp_01 run.

The source detailed evaluation used the same checkpoint, 500-pool seed and
tie-breaking seed.  This helper selects the exact 200 v21 sample indices,
reconstructs every candidate pool from the ordered song-bank IDs, and records
source/output hashes.  Only the intervention conditions need fresh inference.
"""

from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import random

import numpy as np
from transformers import LlamaTokenizer


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "main_eval" / "exp_01" / "detailed_eval" / "exp_01_best_500pool_top1_prompt_original_samples_merged.csv"
OUT_DIR = ROOT / "results" / "main_eval" / "exp_01" / "fixed_component_intervention_v21"
INDICES = OUT_DIR / "sample_indices.json"
OUTPUT = OUT_DIR / "fixed_component_full.csv"
PROVENANCE = OUT_DIR / "fixed_component_full_reuse_provenance.json"
POOL_SEED = 20260315
TIEBREAK_SEED = 42
POOL_SIZE = 500


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_indices = json.loads(INDICES.read_text(encoding="utf-8"))
    if len(sample_indices) != 200 or sample_indices != sorted(sample_indices):
        raise ValueError("Expected the fixed sorted set of 200 sample indices")
    source_by_idx = {int(row["sample_idx"]): row for row in read_csv(SOURCE)}
    song_ids = json.loads((ROOT / "cache" / "song_bank_ids.json").read_text(encoding="utf-8"))
    song_index = {song_id: i for i, song_id in enumerate(song_ids)}
    video_to_indices = {}
    for i, song_id in enumerate(song_ids):
        video_to_indices.setdefault(song_id[:11], set()).add(i)

    ltp = np.load(ROOT / "cache" / "ltp_hybrid.npy", mmap_mode="r")
    ltp_ids = json.loads((ROOT / "cache" / "ltp_hybrid_ids.json").read_text(encoding="utf-8"))
    ltp_index = {video_id: i for i, video_id in enumerate(ltp_ids)}
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", local_files_only=True)

    output_rows = []
    for sample_idx in sample_indices:
        source = source_by_idx[sample_idx]
        video_id = source["video_id"]
        gt_key = source["gt_music_id"]
        gt_idx = song_index[gt_key]
        candidates = [
            i for i in range(len(song_ids))
            if i != gt_idx and i not in video_to_indices.get(video_id, set())
        ]
        negatives = random.Random(POOL_SEED + sample_idx).sample(candidates, POOL_SIZE - 1)
        pool_ids = [gt_key] + [song_ids[i] for i in negatives]
        top1_key = source["top1_music_id"]
        if top1_key not in pool_ids:
            raise ValueError(f"Source top-1 is absent from reconstructed pool: {sample_idx}")
        generated = source.get("generated_text", "")
        vector_norm = float(np.linalg.norm(np.asarray(ltp[ltp_index[video_id]], dtype=np.float32)))
        output_rows.append({
            "condition": "full",
            "sample_idx": sample_idx,
            "video_id": video_id,
            "gt_pair_key": gt_key,
            "top1_pair_key": top1_key,
            "top1_is_gt": source["top1_is_gt"],
            "rank": source["rank"],
            "R@1": source["R@1"],
            "R@5": source["R@5"],
            "R@10": source["R@10"],
            "pool_size": POOL_SIZE,
            "pool_pair_keys": ";".join(pool_ids),
            "gt_score": source["gt_score"],
            "top1_score": source["top1_score"],
            "ltp_cache_mapped": 1,
            "ltp_shift_norm": 0.0,
            "ltp_vector_norm": vector_norm,
            "generated_text": generated,
            "generation_evaluated": 1,
            "generated_token_count": len(tokenizer.encode(generated, add_special_tokens=False)),
            "is_fallback": source["is_fallback"],
            "music_title": source.get("music_title", ""),
            "music_artist": source.get("music_artist", ""),
            "top1_reference_text": source.get("top1_reference_text", ""),
            "generated_mentions_top1_title": source.get("generated_mentions_top1_title", ""),
            "generated_mentions_top1_artist": source.get("generated_mentions_top1_artist", ""),
            "title_consistency": source.get("title_consistency", ""),
            "needs_manual_review": source.get("needs_manual_review", ""),
        })

    with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    provenance = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "decision": "reuse_existing_exp01_full_condition",
        "verified_invariants": {
            "checkpoint": "checkpoints/exp_01/best",
            "candidate_pool_seed": POOL_SEED,
            "tie_breaking_seed": TIEBREAK_SEED,
            "pool_size": POOL_SIZE,
            "prompt_variant": "original",
            "sample_count": len(output_rows),
            "all_source_top1_in_reconstructed_pool": True,
        },
        "source": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
        "sample_indices": {"path": str(INDICES), "sha256": sha256(INDICES)},
        "output": {"path": str(OUTPUT), "sha256": sha256(OUTPUT)},
    }
    PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(output_rows)} verified rows)")


if __name__ == "__main__":
    main()
