"""
用途：使用 persona 條件執行 500-pool 評估。
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
import json
import os
import random

import numpy as np
import torch

from scripts.eval_main import run_eval_500pool_ltp_control as ctrl


# =============================================================================
# 設定
# =============================================================================

EXP_NAME   = "exp_01"
CKPT_NAME  = "best"
LTP_DIM    = 256
POOL_SIZE  = 500
CANDIDATE_POOL_SEED = 20260315       # 與其他評估一致
TIEBREAK_SEED = 42
RANDOM_SEED   = 1234
SHUFFLE_SEED  = 20260726
QUERY_SEED    = 20260726
MICRO_BATCH   = 64

N_QUERIES = 20                        # 每個 Persona 配對幾支查詢影片

CONDITIONS = ["matched", "shuffled", "random",
              "cf_tempo", "cf_energy", "cf_vocal", "cf_popularity", "cf_consistency"]

PERSONA_DIR = PROJECT_ROOT / "results" / "analysis" / "b5_personas_v21"
PERSONA_NPZ = PERSONA_DIR / "persona_ltp.npz"
CLUSTER_CSV = (PROJECT_ROOT / "results" / "analysis" / "video_clusters"
               / "video_cluster_assignments_named.csv")
SPECS_JSON  = PERSONA_DIR / "persona_specs.json"
REAL_LTP_NPY = PROJECT_ROOT / "cache" / "ltp_hybrid.npy"

OUT_DIR = PROJECT_ROOT / "results" / "main_eval" / EXP_NAME / "persona_eval"


def load_personas():
    data = np.load(PERSONA_NPZ, allow_pickle=True)
    keys = [str(k) for k in data["keys"]]
    vecs = data["vectors"].astype(np.float32)
    table = {k: vecs[i] for i, k in enumerate(keys)}
    specs = json.loads(SPECS_JSON.read_text(encoding="utf-8"))
    personas = [p for p in specs["personas"] if p["persona_id"] in table]
    return table, personas


def load_cluster_map():
    with open(CLUSTER_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["sample_idx"]): int(r["cluster_k4"]) for r in rows}


def build_assignments(personas, cluster_map, logger):
    """每個 Persona 由其情境叢集中抽 N_QUERIES 支查詢影片（固定種子）。"""
    by_cluster = {}
    for sidx, c in cluster_map.items():
        by_cluster.setdefault(c, []).append(sidx)
    for c in by_cluster:
        by_cluster[c].sort()

    rng = random.Random(QUERY_SEED)
    assign = {}
    for p in personas:
        pool = by_cluster.get(p["context_cluster"], [])
        assign[p["persona_id"]] = sorted(rng.sample(pool, min(N_QUERIES, len(pool))))
        logger.info("  %-28s 情境=%-22s 查詢影片 %d 支",
                    p["persona_id"], p["context_label"], len(assign[p["persona_id"]]))
    return assign


def persona_vector(table, persona_id, condition, shuffled_partner, rng=None, random_map=None):
    if condition == "matched":
        return table[persona_id]
    if condition == "random":
        if random_map is None:
            raise ValueError("random 條件需要 distribution-matched random_map")
        return random_map[persona_id]
    if condition == "shuffled":
        return table[shuffled_partner]
    key = f"{persona_id}::{condition}"
    return table.get(key, table[persona_id])


def build_distribution_matched_random_map(table, persona_ids):
    """每個 Persona 固定配對一條低相似、但完全來自真實 LTP 分布的向量。"""
    real = np.load(REAL_LTP_NPY).astype(np.float32)
    real_unit = real / (np.linalg.norm(real, axis=1, keepdims=True) + 1e-8)
    pmat = np.stack([table[p] for p in persona_ids])
    punit = pmat / (np.linalg.norm(pmat, axis=1, keepdims=True) + 1e-8)
    sims = real_unit @ punit.T
    rng = np.random.default_rng(RANDOM_SEED)
    out, diagnostics, used = {}, {}, set()
    for col, pid in enumerate(persona_ids):
        cutoff = float(np.quantile(sims[:, col], 0.25))
        candidates = np.flatnonzero(sims[:, col] <= cutoff)
        candidates = np.array([i for i in candidates if int(i) not in used], dtype=int)
        idx = int(rng.choice(candidates))
        used.add(idx)
        out[pid] = real[idx].copy()
        diagnostics[pid] = {
            "source_index": idx,
            "cosine_to_matched_persona": float(sims[idx, col]),
            "norm": float(np.linalg.norm(real[idx])),
            "selection_cutoff_q25": cutoff,
        }
    return out, diagnostics


@torch.no_grad()
def run_condition(condition, model, test_dataset, all_music_features, all_music_ids,
                  device, table, personas, assign, shuffle_map, random_map, logger):
    from evaluate import pointwise_pool_scoring
    from tqdm import tqdm

    model.eval()
    total_music = all_music_features.size(0)
    noise_rng = np.random.default_rng(TIEBREAK_SEED)

    id_to_index = {sid: i for i, sid in enumerate(all_music_ids)}
    vid_to_indices = {}
    for i, sid in enumerate(all_music_ids):
        vid_to_indices.setdefault(sid[:11], set()).add(i)

    rows = []
    tasks = [(p["persona_id"], sidx) for p in personas for sidx in assign[p["persona_id"]]]
    pmeta = {p["persona_id"]: p for p in personas}

    for persona_id, idx in tqdm(tasks, desc=f"[{condition:14s}]"):
        sample = test_dataset[idx]
        video_id = sample.get("video_id", "")
        gt_pair_key = sample.get("gt_music_id", "")

        vec = persona_vector(table, persona_id, condition,
                             shuffle_map[persona_id], random_map=random_map)
        ltp_feat = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

        video_feat = sample["video_feat"].unsqueeze(0).to(device)
        text_feat = sample["text_feat"].unsqueeze(0).to(device)
        input_ids, attention_mask = ctrl.sample_prompt_tensors(sample, device)

        gt_global_idx = id_to_index.get(gt_pair_key)
        if gt_global_idx is None:
            continue

        excluded = vid_to_indices.get(video_id, set())
        candidates = [i for i in range(total_music)
                      if i != gt_global_idx and i not in excluded]
        pool_rng = random.Random(CANDIDATE_POOL_SEED + idx)
        negatives = pool_rng.sample(candidates, min(POOL_SIZE - 1, len(candidates)))
        pool_idx = [gt_global_idx] + negatives

        scores = pointwise_pool_scoring(
            model=model, video_feat=video_feat, ltp_feat=ltp_feat, text_feat=text_feat,
            input_ids=input_ids, attention_mask=attention_mask,
            pool_music_features=all_music_features[pool_idx].to(device),
            micro_batch_size=MICRO_BATCH, device=device)

        rank, top1_pool_idx, scores_np = ctrl.rank_from_scores(scores, noise_rng, add_noise=True)
        p = pmeta[persona_id]
        rows.append({
            "persona_id": persona_id,
            "prototype": p["prototype_label"], "context": p["context_label"],
            "context_cluster": p["context_cluster"],
            "condition": condition, "sample_idx": idx, "video_id": video_id,
            "gt_pair_key": gt_pair_key,
            "top1_pair_key": all_music_ids[pool_idx[top1_pool_idx]],
            "rank": rank, "R@1": int(rank <= 1), "R@5": int(rank <= 5),
            "R@10": int(rank <= 10), "pool_size": POOL_SIZE,
            "gt_score": float(scores_np[0]),
            "top1_score": float(scores_np[top1_pool_idx]),
            "score_range": float(scores_np.max() - scores_np.min()),
        })

    ranks = np.array([r["rank"] for r in rows], dtype=float)
    summary = {
        "condition": condition, "n": len(rows),
        "recall@1": float(np.mean(ranks <= 1)), "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)), "median_rank": float(np.median(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
    }
    logger.info("[%s] n=%d R@1=%.4f R@5=%.4f MRR=%.4f MR=%.1f", condition, summary["n"],
                summary["recall@1"], summary["recall@5"], summary["MRR"],
                summary["median_rank"])
    return rows, summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = ctrl.setup_logger(str(OUT_DIR / "persona_eval.log"))
    started = _dt.datetime.now()

    logger.info("=" * 78)
    logger.info("B5 Persona 條件 500-pool 評估 | exp=%s pool=%d N_QUERIES=%d",
                EXP_NAME, POOL_SIZE, N_QUERIES)
    logger.info("=" * 78)

    table, personas = load_personas()
    cluster_map = load_cluster_map()
    logger.info("Persona 向量 %d 條、Persona %d 個、影片叢集標籤 %d 筆",
                len(table), len(personas), len(cluster_map))

    assign = build_assignments(personas, cluster_map, logger)

    # shuffled：每個 Persona 指派另一個不同的 Persona（固定種子、無不動點）
    pids = [p["persona_id"] for p in personas]
    rng_s = random.Random(SHUFFLE_SEED)
    partners = pids[:]
    for _ in range(100):
        rng_s.shuffle(partners)
        if all(a != b for a, b in zip(pids, partners)):
            break
    shuffle_map = dict(zip(pids, partners))
    random_map, random_diagnostics = build_distribution_matched_random_map(table, pids)

    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("需要 CUDA 才能執行 LLaMA 評估。")

    ltp_mode = ctrl.EXP_TO_LTP_MODE[EXP_NAME]
    active_mods = ctrl.EXP_TO_MODALITIES[EXP_NAME]
    model_cfg = ModelConfig(
        llama_model_name=ctrl.LLAMA_MODEL, video_dim=768, music_dim=768, text_dim=512,
        ltp_dim=LTP_DIM, num_candidates=1, active_modalities=active_mods,
        music_token_offset=3, rank_special_token="[RANK]")
    train_cfg = TrainConfig(output_dir=os.path.join(ctrl.BASE_DIR, "checkpoints", EXP_NAME),
                            pointwise_eval_batch_size=MICRO_BATCH, music_pool_size=POOL_SIZE)

    tokenizer = LlamaTokenizer.from_pretrained(ctrl.LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    ltp_dict = ctrl.load_ltp_dict(ctrl.LTP_H5[ltp_mode], ltp_mode,
                                  cache_path=os.path.join(ctrl.CACHE_DIR, "ltp"), logger=logger)
    test_dataset, all_music_features, all_music_ids = ctrl.build_test_data(
        model_cfg, train_cfg, tokenizer, ltp_dict, logger)[:3]
    model = ctrl.load_model(os.path.join(ctrl.BASE_DIR, "checkpoints", EXP_NAME, CKPT_NAME),
                            model_cfg, tokenizer, logger)

    summaries = []
    for cond in CONDITIONS:
        csv_path = OUT_DIR / f"persona_ranking_{cond}.csv"
        if csv_path.exists():
            logger.info("[%s] 已存在，跳過（斷點續跑）", cond)
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            ranks = np.array([int(r["rank"]) for r in rows], dtype=float)
            summaries.append({"condition": cond, "n": len(rows),
                              "recall@1": float(np.mean(ranks <= 1)),
                              "recall@5": float(np.mean(ranks <= 5)),
                              "recall@10": float(np.mean(ranks <= 10)),
                              "median_rank": float(np.median(ranks)),
                              "MRR": float(np.mean(1.0 / ranks))})
            continue

        rows, summary = run_condition(cond, model, test_dataset, all_music_features,
                                      all_music_ids, device, table, personas, assign,
                                      shuffle_map, random_map, logger)
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary["csv_path"] = str(csv_path)
        summaries.append(summary)

    out = {
        "generated_at": started.isoformat(timespec="seconds"),
        "exp": EXP_NAME, "pool_size": POOL_SIZE, "n_queries_per_persona": N_QUERIES,
        "n_personas": len(personas), "conditions": CONDITIONS,
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "random_control": {
            "type": "fixed_distribution_matched_real_ltp_bottom_quartile_similarity",
            "diagnostics": random_diagnostics,
        },
        "summaries": summaries,
        "scope_limit": "Persona LTP 由既有 LTP 向量組合並經偏差尺度校正而得，"
                       "未走完整 Stage 3–5 管線；驗證的是模型對偏好向量的可控性，"
                       "而非合成畫像流程的有效性。",
    }
    (OUT_DIR / "persona_eval_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("完成，耗時 %.1f 分鐘", (_dt.datetime.now() - started).total_seconds() / 60)


if __name__ == "__main__":
    main()
