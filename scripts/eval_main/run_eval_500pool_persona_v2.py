# Auto-added: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = PROJECT_ROOT / "scripts" / "diagnostics"
for _p in [str(PROJECT_ROOT), str(DIAGNOSTICS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

"""
run_eval_500pool_persona_v2.py
==============================
B5 補跑：一次補齊第一版評估的三個缺口

  缺口 1：只存了 top-1 → 無法計算 Persona-fit@K 與 nDCG@5/@10
          本版保存 **top-10** 推薦清單（分數已算出，argsort 即可，不增加 GPU 時間）

  缺口 2：缺 No-LTP 條件（教授 §七 明列的五組比較之一）
          本版以 **exp_04**（重新訓練且無 LTP 的模型）在同一批查詢上執行，
          需另建其 dataset 與模型（active_modalities 不含 ltp，prompt 結構不同）

  缺口 3：缺兩個生成側指標（Persona 屬性說明支持率、Unsupported Persona Claim Rate）
          本版對 matched / shuffled / random / no_ltp 四組**同時產生說明文字**，
          在排序完成後沿用同一 sample 與 LTP 向量呼叫既有的 generate_top1()

其餘設定與第一版完全相同（候選池種子 20260315、同一批查詢影片、同樣 8 個 LTP 條件），
故兩版結果可直接比對。

規模與預估：
  排序 8 條件 × 480 = 3,840 次 ≈ 5.9 h
  No-LTP        480 次 ≈ 0.8 h（含 exp_04 資料與模型建置）
  生成 4 條件 × 480 = 1,920 次 ≈ 2.1 h
  合計 ≈ 8.8 小時（支援斷點續跑，中斷後重跑會跳過已完成條件）

輸出：results/main_eval/exp_01/persona_eval_v2/
  persona_v2_{condition}.csv        逐樣本結果（含 top10_pair_keys 與 generated_text）
  persona_eval_v2_summary.json
  persona_eval_v2.log

執行（需 CUDA）：
  <user_home>/anaconda3\\envs\\ollama\\python.exe scripts/eval_main/run_eval_500pool_persona_v2.py
"""

import csv
import datetime as _dt
import json
import os
import random

import numpy as np
import torch

from scripts.eval_main import run_eval_500pool_ltp_control as ctrl
from scripts.eval_main import run_eval_500pool_detailed as core
from scripts.eval_main.run_eval_500pool_top1_generation_from_ranking import (
    generate_top1, title_consistency_flags,
)
from scripts.eval_main.run_eval_500pool_persona import (
    load_personas, load_cluster_map, build_assignments, persona_vector,
    build_distribution_matched_random_map,
    CANDIDATE_POOL_SEED, TIEBREAK_SEED, RANDOM_SEED, SHUFFLE_SEED,
    POOL_SIZE, LTP_DIM, MICRO_BATCH, N_QUERIES,
)


EXP_MAIN = "exp_01"
EXP_NOLTP = "exp_04"
CKPT_NAME = "best"
PROMPT_VARIANT = "original"
INJECT_TITLE = True
TOP_K = 10

LTP_CONDITIONS = ["matched", "shuffled", "random",
                  "cf_tempo", "cf_energy", "cf_vocal", "cf_popularity", "cf_consistency"]
GEN_CONDITIONS = {"matched", "shuffled", "random", "no_ltp"}

OUT_DIR = PROJECT_ROOT / "results" / "main_eval" / EXP_MAIN / "persona_eval_v21"


def build_stack(exp_name, logger):
    """建立某個實驗的 dataset / 音樂特徵 / 模型（exp_04 的模態組合不同，需分開建）。"""
    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer

    ltp_mode = ctrl.EXP_TO_LTP_MODE[exp_name]
    active_mods = ctrl.EXP_TO_MODALITIES[exp_name]
    logger.info("[%s] 建立資料與模型 | modalities=%s", exp_name, active_mods)

    model_cfg = ModelConfig(
        llama_model_name=ctrl.LLAMA_MODEL, video_dim=768, music_dim=768, text_dim=512,
        ltp_dim=LTP_DIM, num_candidates=1, active_modalities=active_mods,
        music_token_offset=3, rank_special_token="[RANK]")
    train_cfg = TrainConfig(output_dir=os.path.join(ctrl.BASE_DIR, "checkpoints", exp_name),
                            pointwise_eval_batch_size=MICRO_BATCH, music_pool_size=POOL_SIZE)

    tokenizer = LlamaTokenizer.from_pretrained(ctrl.LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    ltp_dict = ctrl.load_ltp_dict(ctrl.LTP_H5[ltp_mode], ltp_mode,
                                  cache_path=os.path.join(ctrl.CACHE_DIR, "ltp"), logger=logger)
    ds, feats, ids = ctrl.build_test_data(model_cfg, train_cfg, tokenizer, ltp_dict, logger)[:3]
    model = ctrl.load_model(os.path.join(ctrl.BASE_DIR, "checkpoints", exp_name, CKPT_NAME),
                            model_cfg, tokenizer, logger)
    return dict(cfg=model_cfg, tokenizer=tokenizer, dataset=ds,
                features=feats, ids=ids, model=model, modalities=active_mods)


@torch.no_grad()
def run_condition(condition, stack, device, table, personas, assign, shuffle_map, random_map,
                  conv_t3, conv_t4, logger):
    from evaluate import pointwise_pool_scoring
    from tqdm import tqdm

    model, ds = stack["model"], stack["dataset"]
    all_feats, all_ids = stack["features"], stack["ids"]
    tokenizer = stack["tokenizer"]
    model.eval()

    total_music = all_feats.size(0)
    noise_rng = np.random.default_rng(TIEBREAK_SEED)
    id_to_index = {sid: i for i, sid in enumerate(all_ids)}
    vid_to_indices = {}
    for i, sid in enumerate(all_ids):
        vid_to_indices.setdefault(sid[:11], set()).add(i)

    do_gen = condition in GEN_CONDITIONS
    pmeta = {p["persona_id"]: p for p in personas}
    tasks = [(p["persona_id"], s) for p in personas for s in assign[p["persona_id"]]]
    rows = []

    for persona_id, idx in tqdm(tasks, desc=f"[{condition:14s}]"):
        sample = ds[idx]
        video_id = sample.get("video_id", "")
        gt_pair_key = sample.get("gt_music_id", "")
        gt_global_idx = id_to_index.get(gt_pair_key)
        if gt_global_idx is None:
            continue

        # ── LTP：no_ltp 使用 exp_04（模型本身無 ltp 模態），其餘抽換 Persona 向量 ──
        if condition == "no_ltp":
            # exp_04 的 active_modalities 不含 ltp，prompt 中不會用到此張量；
            # dataset 若未提供該鍵則補零，避免 KeyError。
            base_ltp = sample.get("ltp_feat")
            if base_ltp is None:
                base_ltp = torch.zeros(LTP_DIM, dtype=torch.float32)
            ltp_feat = base_ltp.unsqueeze(0).to(device)
            vec = None
        else:
            vec = persona_vector(table, persona_id, condition,
                                 shuffle_map[persona_id], random_map=random_map)
            ltp_feat = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

        video_feat = sample["video_feat"].unsqueeze(0).to(device)
        text_feat = sample["text_feat"].unsqueeze(0).to(device)
        input_ids, attention_mask = ctrl.sample_prompt_tensors(sample, device)

        excluded = vid_to_indices.get(video_id, set())
        candidates = [i for i in range(total_music)
                      if i != gt_global_idx and i not in excluded]
        pool_rng = random.Random(CANDIDATE_POOL_SEED + idx)
        negatives = pool_rng.sample(candidates, min(POOL_SIZE - 1, len(candidates)))
        pool_idx = [gt_global_idx] + negatives

        scores = pointwise_pool_scoring(
            model=model, video_feat=video_feat, ltp_feat=ltp_feat, text_feat=text_feat,
            input_ids=input_ids, attention_mask=attention_mask,
            pool_music_features=all_feats[pool_idx].to(device),
            micro_batch_size=MICRO_BATCH, device=device)

        rank, top1_pool_idx, scores_np = ctrl.rank_from_scores(scores, noise_rng, add_noise=True)

        # ── 缺口 1：保存 top-K（分數已算出，僅需 argsort）──────────────────
        order = np.argsort(-scores_np)[:TOP_K]
        topk_keys = [all_ids[pool_idx[int(j)]] for j in order]

        # ── 缺口 3：生成說明（沿用同一 sample 與 LTP 向量）──────────────────
        generated_text, is_fallback, music_title, music_artist = "", True, None, None
        if do_gen:
            top1_global = pool_idx[top1_pool_idx]
            top1_key = all_ids[top1_global]
            gen_sample = dict(sample)
            if vec is not None:
                gen_sample["ltp_feat"] = torch.tensor(vec, dtype=torch.float32)
            try:
                generated_text, is_fallback, music_title, music_artist = generate_top1(
                    model=model, sample=gen_sample, tokenizer=tokenizer, device=device,
                    active_modalities=stack["modalities"], prompt_variant=PROMPT_VARIANT,
                    top1_music_feat=all_feats[top1_global],
                    t3_text=conv_t3.get(video_id, ""),
                    title_ref_text=conv_t4.get(top1_key[:11], ""),
                    inject_title=INJECT_TITLE)
            except Exception as exc:                      # 單筆失敗不中斷整輪
                logger.warning("生成失敗 persona=%s idx=%s: %s", persona_id, idx, exc)

        p = pmeta[persona_id]
        row = {
            "persona_id": persona_id, "prototype": p["prototype_label"],
            "context": p["context_label"], "context_cluster": p["context_cluster"],
            "condition": condition, "sample_idx": idx, "video_id": video_id,
            "gt_pair_key": gt_pair_key,
            "top1_pair_key": all_ids[pool_idx[top1_pool_idx]],
            "top10_pair_keys": ";".join(topk_keys),
            "pool_pair_keys": ";".join(all_ids[j] for j in pool_idx),
            "rank": rank, "R@1": int(rank <= 1), "R@5": int(rank <= 5),
            "R@10": int(rank <= 10), "pool_size": POOL_SIZE,
            "gt_score": float(scores_np[0]),
            "top1_score": float(scores_np[top1_pool_idx]),
            "score_range": float(scores_np.max() - scores_np.min()),
            "generated_text": generated_text, "is_fallback": int(is_fallback),
            "music_title": music_title or "", "music_artist": music_artist or "",
        }
        if do_gen:
            row.update(title_consistency_flags(generated_text, music_title, music_artist))
        rows.append(row)

    ranks = np.array([r["rank"] for r in rows], dtype=float)
    summary = {
        "condition": condition, "n": len(rows),
        "recall@1": float(np.mean(ranks <= 1)), "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)), "MRR": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "generated": bool(do_gen),
        "fallback_rate": (float(np.mean([r["is_fallback"] for r in rows])) if do_gen else None),
    }
    logger.info("[%s] n=%d R@1=%.4f R@5=%.4f MRR=%.4f%s", condition, summary["n"],
                summary["recall@1"], summary["recall@5"], summary["MRR"],
                f" fallback={summary['fallback_rate']:.3f}" if do_gen else "")
    return rows, summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = ctrl.setup_logger(str(OUT_DIR / "persona_eval_v2.log"))
    started = _dt.datetime.now()

    logger.info("=" * 78)
    logger.info("B5 補跑 v2：top-%d 保存 + No-LTP 條件 + 生成側指標", TOP_K)
    logger.info("=" * 78)

    table, personas = load_personas()
    cluster_map = load_cluster_map()
    assign = build_assignments(personas, cluster_map, logger)
    logger.info("Persona %d 個、每個 %d 支查詢、共 %d 筆/條件",
                len(personas), N_QUERIES, sum(len(v) for v in assign.values()))

    pids = [p["persona_id"] for p in personas]
    rng_s = random.Random(SHUFFLE_SEED)
    partners = pids[:]
    for _ in range(100):
        rng_s.shuffle(partners)
        if all(a != b for a, b in zip(pids, partners)):
            break
    shuffle_map = dict(zip(pids, partners))
    random_map, random_diagnostics = build_distribution_matched_random_map(table, pids)
    logger.info("distribution-matched random：%d 條固定真實 LTP 向量", len(random_map))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("需要 CUDA。")
    conv_t3, conv_t4, _ = core.load_reference_maps()

    summaries = []

    def run_and_save(cond, stack):
        path = OUT_DIR / f"persona_v2_{cond}.csv"
        if path.exists():
            logger.info("[%s] 已存在，跳過（斷點續跑）", cond)
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            ranks = np.array([int(r["rank"]) for r in rows], dtype=float)
            summaries.append({"condition": cond, "n": len(rows),
                              "recall@1": float(np.mean(ranks <= 1)),
                              "recall@5": float(np.mean(ranks <= 5)),
                              "recall@10": float(np.mean(ranks <= 10)),
                              "MRR": float(np.mean(1.0 / ranks)),
                              "median_rank": float(np.median(ranks)),
                              "generated": cond in GEN_CONDITIONS})
            return
        rows, summary = run_condition(cond, stack, device, table, personas, assign,
                                      shuffle_map, random_map, conv_t3, conv_t4, logger)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary["csv_path"] = str(path)
        summaries.append(summary)

    # ---- exp_01 的八個 LTP 條件 --------------------------------------------
    todo_main = [c for c in LTP_CONDITIONS if not (OUT_DIR / f"persona_v2_{c}.csv").exists()]
    if todo_main:
        stack_main = build_stack(EXP_MAIN, logger)
        for cond in LTP_CONDITIONS:
            run_and_save(cond, stack_main)
        del stack_main
        torch.cuda.empty_cache()
    else:
        for cond in LTP_CONDITIONS:
            run_and_save(cond, None)

    # ---- No-LTP（exp_04，需另建 dataset 與模型）-----------------------------
    if not (OUT_DIR / "persona_v2_no_ltp.csv").exists():
        stack_noltp = build_stack(EXP_NOLTP, logger)
        run_and_save("no_ltp", stack_noltp)
        del stack_noltp
        torch.cuda.empty_cache()
    else:
        run_and_save("no_ltp", None)

    out = {
        "generated_at": started.isoformat(timespec="seconds"),
        "top_k_saved": TOP_K,
        "ltp_conditions": LTP_CONDITIONS,
        "no_ltp_model": EXP_NOLTP,
        "generation_conditions": sorted(GEN_CONDITIONS),
        "n_personas": len(personas), "n_queries_per_persona": N_QUERIES,
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "random_control": {
            "type": "fixed_distribution_matched_real_ltp_bottom_quartile_similarity",
            "diagnostics": random_diagnostics,
        },
        "summaries": summaries,
        "scope_limit": "Persona LTP 由既有 LTP 向量組合並經偏差尺度校正而得，"
                       "未走完整 Stage 3–5 管線；驗證的是模型對偏好向量的可控性。",
    }
    (OUT_DIR / "persona_eval_v2_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("全部完成，耗時 %.1f 小時",
                (_dt.datetime.now() - started).total_seconds() / 3600)


if __name__ == "__main__":
    main()
