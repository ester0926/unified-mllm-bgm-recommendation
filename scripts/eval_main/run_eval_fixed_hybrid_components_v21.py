"""
用途：評估固定 hybrid component 條件下的推薦表現。
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
import datetime as dt
import json
import os
import random
import time

import numpy as np
import torch

from scripts.eval_main import run_eval_500pool_ltp_control as ctrl
from scripts.eval_main import run_eval_500pool_detailed as core
from scripts.eval_main.run_eval_500pool_top1_generation_from_ranking import (
    generate_top1,
    title_consistency_flags,
)


OUT_DIR = PROJECT_ROOT / "results" / "main_eval" / "exp_01" / "fixed_component_intervention_v21"
CACHE_PREFIX = PROJECT_ROOT / "cache" / "ltp"
CONDITIONS = (
    "full", "no_explicit", "no_implicit",
    "no_explicit_norm", "no_implicit_norm", "no_both",
)
GEN_CONDITIONS = {"full", "no_explicit_norm", "no_implicit_norm"}
N_SAMPLES = 200
SAMPLE_SEED = 20260729
POOL_SIZE = 500
POOL_SEED = 20260315
TIEBREAK_SEED = 42
MICRO_BATCH = 64
LTP_DIM = 256
PROMPT_VARIANT = "original"


def load_cache(mode: str):
    arr = np.load(f"{CACHE_PREFIX}_{mode}.npy").astype(np.float32)
    with open(f"{CACHE_PREFIX}_{mode}_ids.json", encoding="utf-8") as f:
        ids = json.load(f)
    return arr, ids


def fit_decomposition(logger):
    hybrid, ids_h = load_cache("hybrid")
    explicit, ids_e = load_cache("explicit_only")
    implicit, ids_i = load_cache("implicit_only")
    if ids_h != ids_e or ids_h != ids_i:
        raise RuntimeError("Hybrid/Explicit/Implicit cache IDs are not aligned.")
    if hybrid.shape != explicit.shape or hybrid.shape != implicit.shape:
        raise RuntimeError("The three LTP caches do not share the same shape.")

    rng = np.random.default_rng(SAMPLE_SEED)
    order = rng.permutation(len(ids_h))
    n_train = min(12000, int(len(order) * 0.6))
    n_holdout = min(12000, len(order) - n_train)
    train, holdout = order[:n_train], order[n_train:n_train + n_holdout]

    def design(indices):
        return np.concatenate([
            explicit[indices].astype(np.float64),
            implicit[indices].astype(np.float64),
            np.ones((len(indices), 1), dtype=np.float64),
        ], axis=1)

    w_val, _, design_rank, singular_values = np.linalg.lstsq(
        design(train), hybrid[train].astype(np.float64), rcond=None,
    )
    pred = design(holdout) @ w_val
    err = pred - hybrid[holdout].astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    relative_mae = float(mae / max(np.mean(np.abs(hybrid[holdout])), 1e-12))
    ss_res = float(np.sum(err ** 2))
    hybrid_holdout = hybrid[holdout].astype(np.float64)
    ss_tot = float(np.sum((hybrid_holdout - hybrid_holdout.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    if relative_mae > 1e-4 or r2 < 0.9999:
        raise RuntimeError(
            f"Hybrid decomposition validation failed: relative_mae={relative_mae:.3e}, R2={r2:.8f}"
        )

    # held-out reconstruction 已足夠精準，因此保留這份
    # 獨立驗證過的對照表，不再用全部資料重新擬合。
    w = w_val
    exp_part = explicit.astype(np.float64) @ w[:LTP_DIM]
    imp_part = implicit.astype(np.float64) @ w[LTP_DIM:2 * LTP_DIM]
    bias = w[-1]
    reconstructed = exp_part + imp_part + bias
    residual = hybrid.astype(np.float64) - reconstructed

    # random Stage-5 map 即使條件數較差，使用 float64 求解時仍可能穩定。
    # 因此需用不重疊資料擬合結果確認 component split。
    alt_train = order[n_train + n_holdout:n_train + n_holdout + n_train]
    if len(alt_train) < n_train:
        alt_train = order[-n_train:]
    w_alt, *_ = np.linalg.lstsq(
        design(alt_train), hybrid[alt_train].astype(np.float64), rcond=None,
    )
    stability_idx = holdout[: min(512, len(holdout))]
    exp_alt = explicit[stability_idx].astype(np.float64) @ w_alt[:LTP_DIM]
    imp_alt = implicit[stability_idx].astype(np.float64) @ w_alt[LTP_DIM:2 * LTP_DIM]
    exp_ref = exp_part[stability_idx]
    imp_ref = imp_part[stability_idx]
    exp_cos = np.sum(exp_ref * exp_alt, axis=1) / (
        np.linalg.norm(exp_ref, axis=1) * np.linalg.norm(exp_alt, axis=1) + 1e-12)
    imp_cos = np.sum(imp_ref * imp_alt, axis=1) / (
        np.linalg.norm(imp_ref, axis=1) * np.linalg.norm(imp_alt, axis=1) + 1e-12)
    id_to_idx = {video_id: i for i, video_id in enumerate(ids_h)}
    diagnostics = {
        "n_vectors": len(ids_h),
        "holdout_n": len(holdout),
        "holdout_mae": mae,
        "holdout_max_abs": max_abs,
        "holdout_relative_mae": relative_mae,
        "holdout_r2": r2,
        "design_rank": int(design_rank),
        "design_columns": int(2 * LTP_DIM + 1),
        "design_condition_number": float(singular_values[0] / singular_values[-1]),
        "split_stability_explicit_component_cosine_mean": float(np.mean(exp_cos)),
        "split_stability_implicit_component_cosine_mean": float(np.mean(imp_cos)),
        "split_stability_explicit_component_mae": float(np.mean(np.abs(exp_ref - exp_alt))),
        "split_stability_implicit_component_mae": float(np.mean(np.abs(imp_ref - imp_alt))),
        "full_fit_mae": float(np.mean(np.abs(residual))),
        "hybrid_norm_mean": float(np.linalg.norm(hybrid, axis=1).mean()),
        "explicit_component_norm_mean": float(np.linalg.norm(exp_part, axis=1).mean()),
        "implicit_component_norm_mean": float(np.linalg.norm(imp_part, axis=1).mean()),
        "bias_norm": float(np.linalg.norm(bias)),
    }
    logger.info(
        "Decomposition holdout: MAE=%.3e relative=%.3e R2=%.12f",
        mae, relative_mae, r2,
    )
    return hybrid, exp_part, imp_part, id_to_idx, diagnostics


def intervened_vector(sample, condition, exp_part, imp_part, id_to_idx):
    full = sample["ltp_feat"].detach().cpu().numpy().astype(np.float32)
    video_id = sample.get("video_id", "")
    cache_idx = id_to_idx.get(video_id)
    if cache_idx is None:
        return full, False, 0.0
    if condition == "full":
        out = full
    elif condition in ("no_explicit", "no_explicit_norm"):
        out = full - exp_part[cache_idx]
    elif condition in ("no_implicit", "no_implicit_norm"):
        out = full - imp_part[cache_idx]
    elif condition == "no_both":
        out = full - exp_part[cache_idx] - imp_part[cache_idx]
    else:
        raise ValueError(condition)
    if condition.endswith("_norm"):
        out_norm = float(np.linalg.norm(out))
        full_norm = float(np.linalg.norm(full))
        if out_norm > 1e-8:
            out = out * (full_norm / out_norm)
    return out.astype(np.float32), True, float(np.linalg.norm(out - full))


def write_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_condition(condition, stack, sample_indices, exp_part, imp_part, id_to_idx,
                       conv_t3, conv_t4, device, logger):
    from evaluate import pointwise_pool_scoring
    from tqdm import tqdm

    model = stack["model"]
    dataset = stack["dataset"]
    all_feats = stack["features"]
    all_ids = stack["ids"]
    tokenizer = stack["tokenizer"]
    model.eval()
    id_to_music_idx = {sid: i for i, sid in enumerate(all_ids)}
    video_to_indices = {}
    for i, sid in enumerate(all_ids):
        video_to_indices.setdefault(sid[:11], set()).add(i)

    noise_rng = np.random.default_rng(TIEBREAK_SEED)
    rows = []
    rank_seconds = 0.0
    generation_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    for idx in tqdm(sample_indices, desc=f"[{condition:11s}]"):
        sample = dataset[idx]
        video_id = sample.get("video_id", "")
        gt_pair_key = sample.get("gt_music_id", "")
        gt_global_idx = id_to_music_idx.get(gt_pair_key)
        if gt_global_idx is None:
            continue
        vec, mapped, shift_norm = intervened_vector(
            sample, condition, exp_part, imp_part, id_to_idx,
        )
        ltp_feat = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
        video_feat = sample["video_feat"].unsqueeze(0).to(device)
        text_feat = sample["text_feat"].unsqueeze(0).to(device)
        input_ids, attention_mask = ctrl.sample_prompt_tensors(sample, device)

        candidates = [i for i in range(len(all_ids))
                      if i != gt_global_idx and i not in video_to_indices.get(video_id, set())]
        pool_rng = random.Random(POOL_SEED + idx)
        negatives = pool_rng.sample(candidates, min(POOL_SIZE - 1, len(candidates)))
        pool_idx = [gt_global_idx] + negatives

        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = pointwise_pool_scoring(
            model=model,
            video_feat=video_feat,
            ltp_feat=ltp_feat,
            text_feat=text_feat,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_music_features=all_feats[pool_idx].to(device),
            micro_batch_size=MICRO_BATCH,
            device=device,
        )
        torch.cuda.synchronize()
        rank_seconds += time.perf_counter() - started
        rank, top1_pool_idx, scores_np = ctrl.rank_from_scores(scores, noise_rng, add_noise=True)
        top1_global_idx = pool_idx[top1_pool_idx]
        top1_key = all_ids[top1_global_idx]

        generated_text, is_fallback, music_title, music_artist = "", True, None, None
        do_generation = condition in GEN_CONDITIONS
        if do_generation:
            gen_sample = dict(sample)
            gen_sample["ltp_feat"] = torch.tensor(vec, dtype=torch.float32)
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                generated_text, is_fallback, music_title, music_artist = generate_top1(
                    model=model,
                    sample=gen_sample,
                    tokenizer=tokenizer,
                    device=device,
                    active_modalities=stack["modalities"],
                    prompt_variant=PROMPT_VARIANT,
                    top1_music_feat=all_feats[top1_global_idx],
                    t3_text=conv_t3.get(video_id, ""),
                    title_ref_text=conv_t4.get(top1_key[:11], ""),
                    inject_title=True,
                )
            except Exception as exc:
                logger.warning("Generation failed condition=%s idx=%d: %s", condition, idx, exc)
            torch.cuda.synchronize()
            generation_seconds += time.perf_counter() - started

        row = {
            "condition": condition,
            "sample_idx": idx,
            "video_id": video_id,
            "gt_pair_key": gt_pair_key,
            "top1_pair_key": top1_key,
            "top1_is_gt": int(top1_global_idx == gt_global_idx),
            "rank": rank,
            "R@1": int(rank <= 1),
            "R@5": int(rank <= 5),
            "R@10": int(rank <= 10),
            "pool_size": len(pool_idx),
            "pool_pair_keys": ";".join(all_ids[j] for j in pool_idx),
            "gt_score": float(scores_np[0]),
            "top1_score": float(scores_np[top1_pool_idx]),
            "ltp_cache_mapped": int(mapped),
            "ltp_shift_norm": shift_norm,
            "ltp_vector_norm": float(np.linalg.norm(vec)),
            "generated_text": generated_text,
            "generation_evaluated": int(do_generation),
            "generated_token_count": len(tokenizer.encode(
                generated_text, add_special_tokens=False,
            )) if generated_text else 0,
            "is_fallback": int(is_fallback),
            "music_title": music_title or "",
            "music_artist": music_artist or "",
            "top1_reference_text": conv_t4.get(top1_key[:11], ""),
        }
        row.update(title_consistency_flags(generated_text, music_title, music_artist))
        rows.append(row)

    ranks = np.array([r["rank"] for r in rows], dtype=float)
    summary = {
        "condition": condition,
        "n": len(rows),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "generation_evaluated": condition in GEN_CONDITIONS,
        "fallback_rate": (float(np.mean([r["is_fallback"] for r in rows]))
                          if condition in GEN_CONDITIONS else None),
        "ltp_mapping_rate": float(np.mean([r["ltp_cache_mapped"] for r in rows])),
        "ltp_shift_norm_mean": float(np.mean([r["ltp_shift_norm"] for r in rows])),
        "ltp_vector_norm_mean": float(np.mean([r["ltp_vector_norm"] for r in rows])),
        "ranking_seconds_total": rank_seconds,
        "generation_seconds_total": generation_seconds,
        "ranking_seconds_per_query": rank_seconds / max(len(rows), 1),
        "generation_seconds_per_query": (generation_seconds / max(len(rows), 1)
                                         if condition in GEN_CONDITIONS else None),
        "generated_tokens_mean": (float(np.mean([r["generated_token_count"] for r in rows]))
                                  if condition in GEN_CONDITIONS else None),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    }
    return rows, summary


@torch.no_grad()
def benchmark_pool_scaling(stack, sample_indices, device, logger):
    from evaluate import pointwise_pool_scoring

    model, dataset = stack["model"], stack["dataset"]
    all_feats, all_ids = stack["features"], stack["ids"]
    id_to_music_idx = {sid: i for i, sid in enumerate(all_ids)}
    video_to_indices = {}
    for i, sid in enumerate(all_ids):
        video_to_indices.setdefault(sid[:11], set()).add(i)
    rows = []
    for pool_size in (100, 500, 1000):
        for idx in sample_indices[:30]:
            sample = dataset[idx]
            video_id = sample.get("video_id", "")
            gt_idx = id_to_music_idx.get(sample.get("gt_music_id", ""))
            if gt_idx is None:
                continue
            candidates = [i for i in range(len(all_ids))
                          if i != gt_idx and i not in video_to_indices.get(video_id, set())]
            rng = random.Random(POOL_SEED + idx)
            negatives = rng.sample(candidates, min(pool_size - 1, len(candidates)))
            pool_idx = [gt_idx] + negatives
            input_ids, attention_mask = ctrl.sample_prompt_tensors(sample, device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            pointwise_pool_scoring(
                model=model,
                video_feat=sample["video_feat"].unsqueeze(0).to(device),
                ltp_feat=sample["ltp_feat"].unsqueeze(0).to(device),
                text_feat=sample["text_feat"].unsqueeze(0).to(device),
                input_ids=input_ids,
                attention_mask=attention_mask,
                pool_music_features=all_feats[pool_idx].to(device),
                micro_batch_size=MICRO_BATCH,
                device=device,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            rows.append({
                "sample_idx": idx,
                "pool_size": pool_size,
                "seconds": seconds,
                "candidates_per_second": len(pool_idx) / seconds,
            })
    summaries = []
    for pool_size in (100, 500, 1000):
        values = np.array([r["seconds"] for r in rows if r["pool_size"] == pool_size])
        summaries.append({
            "pool_size": pool_size,
            "n": len(values),
            "seconds_mean": float(values.mean()),
            "seconds_median": float(np.median(values)),
            "seconds_p95": float(np.percentile(values, 95)),
            "candidates_per_second_mean": float(pool_size / values.mean()),
        })
    write_rows(OUT_DIR / "pool_scaling_latency.csv", rows)
    logger.info("Pool scaling latency: %s", summaries)
    return summaries


def main():
    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = ctrl.setup_logger(str(OUT_DIR / "fixed_component_intervention_v21.log"))
    started_at = dt.datetime.now()
    _, exp_part, imp_part, id_to_idx, diagnostics = fit_decomposition(logger)

    model_cfg = ModelConfig(
        llama_model_name=ctrl.LLAMA_MODEL,
        video_dim=768,
        music_dim=768,
        text_dim=512,
        ltp_dim=LTP_DIM,
        num_candidates=1,
        active_modalities=ctrl.EXP_TO_MODALITIES["exp_01"],
        music_token_offset=3,
        rank_special_token="[RANK]",
    )
    train_cfg = TrainConfig(
        output_dir=os.path.join(ctrl.BASE_DIR, "checkpoints", "exp_01"),
        pointwise_eval_batch_size=MICRO_BATCH,
        music_pool_size=POOL_SIZE,
    )
    tokenizer = LlamaTokenizer.from_pretrained(ctrl.LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    ltp_dict = ctrl.load_ltp_dict(
        ctrl.LTP_H5["hybrid"], "hybrid",
        cache_path=os.path.join(ctrl.CACHE_DIR, "ltp"), logger=logger,
    )
    dataset, features, ids = ctrl.build_test_data(
        model_cfg, train_cfg, tokenizer, ltp_dict, logger,
    )[:3]
    model = ctrl.load_model(
        os.path.join(ctrl.BASE_DIR, "checkpoints", "exp_01", "best"),
        model_cfg, tokenizer, logger,
    )
    stack = {
        "model": model,
        "dataset": dataset,
        "features": features,
        "ids": ids,
        "tokenizer": tokenizer,
        "modalities": model_cfg.active_modalities,
    }
    rng = random.Random(SAMPLE_SEED)
    sample_indices = sorted(rng.sample(range(len(dataset)), min(N_SAMPLES, len(dataset))))
    with open(OUT_DIR / "sample_indices.json", "w", encoding="utf-8") as f:
        json.dump(sample_indices, f, indent=2)
    conv_t3, conv_t4, _ = core.load_reference_maps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    summaries = []
    for condition in CONDITIONS:
        path = OUT_DIR / f"fixed_component_{condition}.csv"
        if path.exists():
            logger.info("[%s] output exists; preserving completed condition", condition)
            saved_summary = None
            summary_path = OUT_DIR / "fixed_component_intervention_summary.json"
            if summary_path.exists():
                with open(summary_path, encoding="utf-8") as f:
                    previous = json.load(f)
                saved_summary = next(
                    (item for item in previous.get("conditions", [])
                     if item.get("condition") == condition),
                    None,
                )
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if saved_summary is not None:
                saved_summary["resumed"] = True
                summaries.append(saved_summary)
            else:
                ranks = np.array([int(r["rank"]) for r in rows], dtype=float)
                summaries.append({
                    "condition": condition,
                    "n": len(rows),
                    "recall@1": float(np.mean(ranks <= 1)),
                    "recall@5": float(np.mean(ranks <= 5)),
                    "recall@10": float(np.mean(ranks <= 10)),
                    "mrr": float(np.mean(1.0 / ranks)),
                    "median_rank": float(np.median(ranks)),
                    "generation_evaluated": condition in GEN_CONDITIONS,
                    "fallback_rate": (
                        float(np.mean([int(r["is_fallback"]) for r in rows]))
                        if condition in GEN_CONDITIONS else None
                    ),
                    "ltp_mapping_rate": float(np.mean([
                        int(r["ltp_cache_mapped"]) for r in rows
                    ])),
                    "ltp_shift_norm_mean": float(np.mean([
                        float(r["ltp_shift_norm"]) for r in rows
                    ])),
                    "ltp_vector_norm_mean": float(np.mean([
                        float(r["ltp_vector_norm"]) for r in rows
                    ])),
                    "ranking_seconds_total": None,
                    "generation_seconds_total": None,
                    "ranking_seconds_per_query": None,
                    "generation_seconds_per_query": None,
                    "generated_tokens_mean": (
                        float(np.mean([int(r["generated_token_count"]) for r in rows]))
                        if condition in GEN_CONDITIONS else None
                    ),
                    "peak_gpu_memory_gib": None,
                    "resumed": True,
                })
            continue
        rows, summary = evaluate_condition(
            condition, stack, sample_indices, exp_part, imp_part, id_to_idx,
            conv_t3, conv_t4, device, logger,
        )
        write_rows(path, rows)
        summaries.append(summary)
        with open(OUT_DIR / "fixed_component_intervention_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "created_at": dt.datetime.now().isoformat(),
                "sample_seed": SAMPLE_SEED,
                "candidate_pool_seed": POOL_SEED,
                "n_requested": N_SAMPLES,
                "decomposition": diagnostics,
                "conditions": summaries,
            }, f, ensure_ascii=False, indent=2)

    benchmark_path = OUT_DIR / "pool_scaling_latency.csv"
    if benchmark_path.exists():
        benchmark_rows = []
        with open(benchmark_path, encoding="utf-8-sig", newline="") as f:
            saved = list(csv.DictReader(f))
        for pool_size in (100, 500, 1000):
            values = np.array([float(r["seconds"]) for r in saved
                               if int(r["pool_size"]) == pool_size])
            benchmark_rows.append({
                "pool_size": pool_size, "n": len(values),
                "seconds_mean": float(values.mean()),
                "seconds_median": float(np.median(values)),
                "seconds_p95": float(np.percentile(values, 95)),
                "candidates_per_second_mean": float(pool_size / values.mean()),
            })
    else:
        benchmark_rows = benchmark_pool_scaling(stack, sample_indices, device, logger)

    final = {
        "created_at": dt.datetime.now().isoformat(),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": (dt.datetime.now() - started_at).total_seconds(),
        "sample_seed": SAMPLE_SEED,
        "candidate_pool_seed": POOL_SEED,
        "n_requested": N_SAMPLES,
        "decomposition": diagnostics,
        "conditions": summaries,
        "pool_scaling_latency": benchmark_rows,
        "interpretation_boundary": (
            "Exploratory representation-level intervention in a fixed exp_01 model; "
            "component subtraction may create off-manifold vectors and does not establish "
            "causal identifiability beyond this checkpoint."
        ),
    }
    with open(OUT_DIR / "fixed_component_intervention_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    logger.info("Completed in %.1f min", final["elapsed_seconds"] / 60)


if __name__ == "__main__":
    main()
