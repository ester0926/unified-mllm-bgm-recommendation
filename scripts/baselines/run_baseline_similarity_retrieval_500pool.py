# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
VSCode-run baseline: non-parametric 500-pool similarity retrieval.

This script evaluates lightweight baselines on the same test split and the same
candidate-pool construction used by run_eval_500pool_detailed.py.

Baselines:
  1. random_500pool
  2. audio_ast_similarity
     Query: the candidate music audio feature in the query pair.
     Bank: target_music_all_cls features for all songs.
  3. video_audio_embedding_similarity
     Query: video feature pooled from the query video.
     Bank: target_music_all_cls features for all songs.

The third baseline is intentionally weak because the raw video and audio
embeddings are not trained into a shared space. It is still useful as a
traditional multimodal retrieval baseline that does not use the unified MLLM.
"""

import csv
import json
import os
import random
import statistics
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np

from scripts.eval_main import run_eval_500pool_detailed as core
from config import TrainConfig
from dataset import build_pair_index, build_song_bank, split_by_video_id


BASE_DIR = str(PROJECT_ROOT)
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints", "baseline_similarity", "detailed_eval")
POOL_SIZE = 500
CANDIDATE_POOL_SEED = 20260315
SPLIT_SEED = 42
EPS = 1e-8


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def mean_pool_dataset(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0)


def read_query_features(h5_path: str, pair_key: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        group = f[f"pairs/{pair_key}"]
        video_feat = mean_pool_dataset(group["video_features_all"][:])

        if "candidate_music_all_cls" in group:
            audio_feat = mean_pool_dataset(group["candidate_music_all_cls"][:])
        elif "candidate_music_all_seq" in group:
            # AST-like sequence features: use the CLS token from each segment if
            # available, then average across segments.
            seq = np.asarray(group["candidate_music_all_seq"][:], dtype=np.float32)
            if seq.ndim >= 3:
                audio_feat = seq[:, 0, :].mean(axis=0)
            else:
                audio_feat = mean_pool_dataset(seq)
        else:
            raise KeyError(f"No candidate music feature in {h5_path}::{pair_key}")

    return video_feat.astype(np.float32), audio_feat.astype(np.float32)


def build_candidate_indices(
    sample_idx: int,
    gt_music_id: str,
    all_music_ids: List[str],
    id_to_index: Dict[str, int],
) -> List[int]:
    gt_idx = id_to_index[gt_music_id]
    gt_video = gt_music_id[:11]
    negative_pool = [i for i, sid in enumerate(all_music_ids) if sid[:11] != gt_video and sid != gt_music_id]

    rng = random.Random(CANDIDATE_POOL_SEED + sample_idx)
    n_neg = POOL_SIZE - 1
    if len(negative_pool) < n_neg:
        raise ValueError(f"Not enough negatives for sample {sample_idx}")
    neg_indices = rng.sample(negative_pool, n_neg)

    candidate_indices = neg_indices + [gt_idx]
    rng.shuffle(candidate_indices)
    return candidate_indices


def compute_rank(scores: np.ndarray, gt_pos: int) -> Tuple[int, int]:
    order = np.argsort(-scores, kind="mergesort")
    rank = int(np.where(order == gt_pos)[0][0]) + 1
    top1_pos = int(order[0])
    return rank, top1_pos


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    ranks = [int(r["rank"]) for r in rows]
    n = len(ranks)
    return {
        "n": n,
        "pool_size": POOL_SIZE,
        "recall_at_1": sum(r <= 1 for r in ranks) / n,
        "recall_at_5": sum(r <= 5 for r in ranks) / n,
        "recall_at_10": sum(r <= 10 for r in ranks) / n,
        "median_rank": statistics.median(ranks),
        "mean_rank": sum(ranks) / n,
    }


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, summary: dict) -> None:
    lines = [
        "# Similarity Retrieval Baselines",
        "",
        "| Baseline | R@1 | R@5 | R@10 | Median Rank | Mean Rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["baselines"].items():
        lines.append(
            f"| {name} | "
            f"{item['recall_at_1'] * 100:.2f}% | "
            f"{item['recall_at_5'] * 100:.2f}% | "
            f"{item['recall_at_10'] * 100:.2f}% | "
            f"{item['median_rank']:.0f} | "
            f"{item['mean_rank']:.2f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading pair index and song bank...")
    pair_index = build_pair_index(core.H5_DIR, cache_path=os.path.join(core.CACHE_DIR, "pair_index.json"))
    song_bank_np, song_ids = build_song_bank(pair_index, cache_path=os.path.join(core.CACHE_DIR, "song_bank"))
    _, _, test_pairs = split_by_video_id(
        pair_index,
        TrainConfig.train_ratio,
        TrainConfig.val_ratio,
        TrainConfig.test_ratio,
        seed=SPLIT_SEED,
    )

    song_bank_norm = l2_normalize(song_bank_np.astype(np.float32))
    id_to_index = {sid: i for i, sid in enumerate(song_ids)}

    rows_by_method: Dict[str, List[dict]] = {
        "random_500pool": [],
        "audio_ast_similarity": [],
        "video_audio_embedding_similarity": [],
    }

    for sample_idx, (h5_path, pair_key) in enumerate(test_pairs):
        if sample_idx % 100 == 0:
            print(f"[{sample_idx}/{len(test_pairs)}] {pair_key}")

        candidate_indices = build_candidate_indices(sample_idx, pair_key, song_ids, id_to_index)
        gt_pos = candidate_indices.index(id_to_index[pair_key])
        candidate_ids = [song_ids[i] for i in candidate_indices]
        candidate_feats = song_bank_norm[candidate_indices]

        video_feat, audio_feat = read_query_features(h5_path, pair_key)
        queries = {
            "audio_ast_similarity": l2_normalize(audio_feat.reshape(1, -1))[0],
            "video_audio_embedding_similarity": l2_normalize(video_feat.reshape(1, -1))[0],
        }

        random_rng = np.random.default_rng(CANDIDATE_POOL_SEED + sample_idx)
        score_map = {
            "random_500pool": random_rng.random(len(candidate_indices), dtype=np.float32),
            "audio_ast_similarity": candidate_feats @ queries["audio_ast_similarity"],
            "video_audio_embedding_similarity": candidate_feats @ queries["video_audio_embedding_similarity"],
        }

        for method, scores in score_map.items():
            rank, top1_pos = compute_rank(scores, gt_pos)
            rows_by_method[method].append(
                {
                    "method": method,
                    "sample_idx": sample_idx,
                    "video_id": pair_key[:11],
                    "gt_music_id": pair_key,
                    "top1_music_id": candidate_ids[top1_pos],
                    "rank": rank,
                    "R@1": int(rank <= 1),
                    "R@5": int(rank <= 5),
                    "R@10": int(rank <= 10),
                    "gt_score": float(scores[gt_pos]),
                    "top1_score": float(scores[top1_pos]),
                }
            )

    all_rows = [row for rows in rows_by_method.values() for row in rows]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pool_size": POOL_SIZE,
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "split_seed": SPLIT_SEED,
        "n_test": len(test_pairs),
        "baselines": {method: summarize(rows) for method, rows in rows_by_method.items()},
    }

    ranking_csv = os.path.join(OUTPUT_DIR, "baseline_similarity_500pool_ranking_samples.csv")
    summary_json = os.path.join(OUTPUT_DIR, "baseline_similarity_500pool_summary.json")
    summary_md = os.path.join(OUTPUT_DIR, "baseline_similarity_500pool_summary.md")
    write_csv(ranking_csv, all_rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_markdown(summary_md, summary)

    print(f"Saved: {ranking_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_md}")
    print(json.dumps(summary["baselines"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
