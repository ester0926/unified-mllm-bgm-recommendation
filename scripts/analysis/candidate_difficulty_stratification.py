"""
用途：整理實驗輸出並產生論文分析用表格或圖表。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import csv
import datetime as _dt
import json
import logging
import random

import numpy as np


# =============================================================================
# 路徑設定 — 與 run_eval_500pool_detailed.py 一致
# =============================================================================

BASE_DIR   = PROJECT_ROOT
CACHE_DIR  = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"
OUT_DIR    = RESULTS_DIR / "analysis" / "candidate_difficulty"

SONG_BANK_NPY = CACHE_DIR / "song_bank.npy"
SONG_BANK_IDS = CACHE_DIR / "song_bank_ids.json"


# =============================================================================
# 使用者設定
# =============================================================================
#
# CANDIDATE_POOL_SEED / POOL_SIZE 必須與產生下列 CSV 的那次評估完全相同。
# 已核驗：exp_01_best_500pool_prompt_original_ranking_samples.csv 與
#         exp_01_best_500pool_seed20260315_ranking_samples.csv 為位元組完全相同的檔案
#         → run_eval_500pool_detailed.py 的預設 CANDIDATE_POOL_SEED = 20260315 即為所用。
# =============================================================================

CANDIDATE_POOL_SEED = 20260315
POOL_SIZE           = 500
N_BINS              = 3
BIN_LABELS          = ["Easy", "Medium", "Hard"]

# 以哪個模型的 CSV 作為「樣本順序基準」（決定 idx → video_id 的對應）
REFERENCE_EXP = "exp_01"

# 參與分層彙整的模型：exp_name → (標籤, 逐樣本排序 CSV 相對於 results/ 的路徑)
EXPERIMENTS = {
    "exp_01": ("Hybrid LTP (Matched)",
               "main_eval/exp_01/detailed_eval/exp_01_best_500pool_prompt_original_ranking_samples.csv"),
    "exp_02": ("Explicit-only LTP",
               "main_eval/exp_02/detailed_eval/exp_02_best_500pool_ranking_samples.csv"),
    "exp_03": ("Implicit-only LTP",
               "main_eval/exp_03/detailed_eval/exp_03_best_500pool_ranking_samples.csv"),
    "exp_04": ("No-LTP",
               "main_eval/exp_04/detailed_eval/exp_04_best_500pool_ranking_samples.csv"),
}

# 增益比較：(實驗組, 對照組) — 教授指定的 Matched − No-LTP 差值
GAIN_PAIRS = [("exp_01", "exp_04"), ("exp_03", "exp_04"), ("exp_02", "exp_04")]

# Hard-Negative 對照（已有逐樣本 cosim，直接接在難度軸最右端，不需重建池）
HARDNEG_CSV = "main_eval/exp_01/hardneg_eval/exp_01_best_500pool_hardneg_top499_ranking_samples.csv"

# bootstrap 設定
BOOTSTRAP_N    = 2000
BOOTSTRAP_SEED = 20260726
CI_ALPHA       = 0.05

# 正確性驗證門檻：既有 CSV 的 top1_music_id 落在重建池內的比例下限
POOL_COVERAGE_MIN = 0.999


# =============================================================================
# Logger
# =============================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("candidate_difficulty")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# =============================================================================
# 資料載入
# =============================================================================

def load_song_bank(logger):
    """載入 AST song bank 並做 L2 正規化（與 run_eval_500pool_hardneg.py 相同處理）。"""
    if not SONG_BANK_NPY.exists() or not SONG_BANK_IDS.exists():
        raise FileNotFoundError(
            f"找不到 song bank 快取：{SONG_BANK_NPY} / {SONG_BANK_IDS}\n"
            "請先執行過任一次 detailed eval（會自動建立 cache/song_bank*）。"
        )
    bank = np.load(SONG_BANK_NPY).astype(np.float32)
    with open(SONG_BANK_IDS, encoding="utf-8") as f:
        song_ids = json.load(f)
    if bank.shape[0] != len(song_ids):
        raise ValueError(f"song bank 長度不一致：features={bank.shape[0]} ids={len(song_ids)}")

    norms = np.linalg.norm(bank, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bank /= norms
    logger.info("[SongBank] %d pairs, dim=%d, L2-normalised", bank.shape[0], bank.shape[1])
    return bank, song_ids


def load_ranking_csv(rel_path: str, logger) -> list:
    """讀取逐樣本排序 CSV（優先 results/，回退 checkpoints/ 舊路徑）。"""
    path = RESULTS_DIR / rel_path
    if not path.exists():
        legacy = BASE_DIR / "checkpoints" / rel_path.split("main_eval/", 1)[-1]
        if legacy.exists():
            logger.warning("results/ 找不到 %s，改用舊路徑 %s", rel_path, legacy)
            path = legacy
        else:
            raise FileNotFoundError(f"找不到排序結果 CSV：{path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    logger.info("[CSV] %-58s rows=%d", path.name, len(rows))
    return rows


# =============================================================================
# 候選池重建 —— 與 run_eval_500pool_detailed.py::eval_ranking_detailed 完全一致
# =============================================================================

def rebuild_pool_similarity(ref_rows, bank, song_ids, logger):
    """
    重建每個測試樣本的 499 個負例，計算 GT 與負例間的 AST 餘弦相似度。

    回傳 per_sample: list of dict，欄位包含
        sample_idx, video_id, gt_music_id,
        avg_neg_cosim, max_neg_cosim, top10_neg_cosim,
        top1_in_pool（驗證用；GT 命中時為 None）
    """
    total_music = len(song_ids)
    id_to_index = {sid: i for i, sid in enumerate(song_ids)}

    video_to_indices = {}
    for i, sid in enumerate(song_ids):
        video_to_indices.setdefault(sid[:11], []).append(i)

    per_sample = []
    n_checked = 0
    n_covered = 0

    for row in ref_rows:
        idx = int(row["sample_idx"])
        video_id = row["video_id"]
        gt_music_id = row["gt_music_id"]

        gt_global_idx = id_to_index.get(gt_music_id)
        if gt_global_idx is None:
            raise KeyError(f"GT music id 不在 song bank 內：{gt_music_id}（sample_idx={idx}）")

        # --- 與原始評估腳本相同的候選集合 ---------------------------------
        # 原始：candidates = [i for i in range(total) if i != gt and i not in excluded]
        # 這裡用 boolean mask + flatnonzero 產生「內容與順序完全相同」的遞增索引串列，
        # 因此 random.Random(seed + idx).sample(...) 會抽到完全一樣的 499 筆。
        mask = np.ones(total_music, dtype=bool)
        mask[gt_global_idx] = False
        excluded = video_to_indices.get(video_id)
        if excluded:
            mask[excluded] = False
        candidates = np.flatnonzero(mask).tolist()

        rng_pool = random.Random(CANDIDATE_POOL_SEED + idx)
        negatives = rng_pool.sample(candidates, min(POOL_SIZE - 1, len(candidates)))
        # ------------------------------------------------------------------

        neg_arr = np.asarray(negatives, dtype=np.int64)
        sims = bank[neg_arr] @ bank[gt_global_idx]
        sims_sorted = np.sort(sims)[::-1]

        # 驗證：CSV 記錄的 top1 若不是 GT，必須出現在重建出來的池裡
        top1_in_pool = None
        top1_music_id = row.get("top1_music_id", "")
        if top1_music_id and top1_music_id != gt_music_id:
            n_checked += 1
            top1_gidx = id_to_index.get(top1_music_id)
            top1_in_pool = bool(top1_gidx is not None and top1_gidx in set(negatives))
            n_covered += int(top1_in_pool)

        per_sample.append({
            "sample_idx": idx,
            "video_id": video_id,
            "gt_music_id": gt_music_id,
            "n_negatives": len(negatives),
            "avg_neg_cosim": float(sims.mean()),
            "max_neg_cosim": float(sims_sorted[0]),
            "top10_neg_cosim": float(sims_sorted[:10].mean()),
            "top1_in_pool": top1_in_pool,
        })

        if (len(per_sample) % 500) == 0:
            logger.info("  重建進度 %d/%d", len(per_sample), len(ref_rows))

    coverage = (n_covered / n_checked) if n_checked else 1.0
    logger.info("[驗證] 非 GT 的 top1 共 %d 筆，落在重建池內 %d 筆（coverage=%.4f）",
                n_checked, n_covered, coverage)
    if coverage < POOL_COVERAGE_MIN:
        raise RuntimeError(
            f"候選池重建驗證失敗：coverage={coverage:.4f} < {POOL_COVERAGE_MIN}。\n"
            f"表示 CANDIDATE_POOL_SEED({CANDIDATE_POOL_SEED})、POOL_SIZE({POOL_SIZE}) "
            f"或 song bank 順序與當初評估時不一致，請勿採用本次分層結果。"
        )
    return per_sample, coverage


# =============================================================================
# 分層
# =============================================================================

def assign_bins(values: np.ndarray, n_bins: int, labels: list):
    """依百分位等分切層；回傳 (labels_per_sample, edges)。"""
    qs = [100.0 * i / n_bins for i in range(1, n_bins)]
    edges = np.percentile(values, qs)
    idx = np.digitize(values, edges, right=False)
    return [labels[i] for i in idx], [float(e) for e in edges]


# =============================================================================
# 指標
# =============================================================================

def metrics_from_ranks(ranks: np.ndarray) -> dict:
    """
    單一相關項目（GT 唯一）情境下的排序指標。
    nDCG@10：IDCG = 1/log2(1+1) = 1，故 nDCG@10 = 1/log2(rank+1) if rank<=10 else 0。
    """
    if ranks.size == 0:
        return {k: float("nan") for k in
                ["n", "R@1", "R@5", "R@10", "MRR", "nDCG@10", "median_rank", "mean_rank"]}
    return {
        "n": int(ranks.size),
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
        "nDCG@10": float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0))),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def _metric_value(r: np.ndarray, metric: str) -> float:
    if metric == "R@1":
        return float(np.mean(r <= 1))
    if metric == "R@5":
        return float(np.mean(r <= 5))
    if metric == "MRR":
        return float(np.mean(1.0 / r))
    if metric == "nDCG@10":
        return float(np.mean(np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)))
    raise ValueError(metric)


def paired_bootstrap_gain(ranks_a: np.ndarray, ranks_b: np.ndarray, metric: str,
                          n_boot: int, seed: int, alpha: float):
    """
    配對 bootstrap：同一批樣本上 metric(A) − metric(B) 的點估計與百分位 CI。
    A、B 必須逐樣本對齊（同一 sample_idx 順序）。
    """
    def _metric(r):
        return _metric_value(r, metric)

    point = float(_metric(ranks_a) - _metric(ranks_b))
    n = ranks_a.size
    if n == 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)
        draws[b] = _metric(ranks_a[sel]) - _metric(ranks_b[sel])
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_did(ranks_a_hard, ranks_b_hard, ranks_a_easy, ranks_b_easy,
                  metric: str, n_boot: int, seed: int, alpha: float):
    """
    差異中的差異（difference-in-differences）：

        DiD = [metric(A|Hard) − metric(B|Hard)] − [metric(A|Easy) − metric(B|Easy)]

    這才是「LTP 增益是否隨候選相似度提高而縮小」（H4）的正式檢定。
    只看 Easy / Hard 兩條 CI 是否重疊會低估顯著性，因此另外對 DiD 本身做 bootstrap：
    兩層各自獨立重抽（層內配對、層間獨立），CI 不含 0 才算增益確實隨難度改變。
    """
    point = ((_metric_value(ranks_a_hard, metric) - _metric_value(ranks_b_hard, metric))
             - (_metric_value(ranks_a_easy, metric) - _metric_value(ranks_b_easy, metric)))
    n_h, n_e = ranks_a_hard.size, ranks_a_easy.size
    if n_h == 0 or n_e == 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sh = rng.integers(0, n_h, size=n_h)
        se = rng.integers(0, n_e, size=n_e)
        draws[b] = ((_metric_value(ranks_a_hard[sh], metric) - _metric_value(ranks_b_hard[sh], metric))
                    - (_metric_value(ranks_a_easy[se], metric) - _metric_value(ranks_b_easy[se], metric)))
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# =============================================================================
# 主流程
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUT_DIR / "candidate_difficulty.log")
    started = _dt.datetime.now()

    logger.info("=" * 78)
    logger.info("B1 候選聲學相似度分層 | pool=%d seed=%d bins=%d",
                POOL_SIZE, CANDIDATE_POOL_SEED, N_BINS)
    logger.info("=" * 78)

    # ---- 1. 載入 -----------------------------------------------------------
    bank, song_ids = load_song_bank(logger)

    exp_rows = {}
    for exp, (_label, rel) in EXPERIMENTS.items():
        exp_rows[exp] = load_ranking_csv(rel, logger)

    ref_rows = exp_rows[REFERENCE_EXP]
    n_samples = len(ref_rows)

    # 逐樣本對齊檢查（sample_idx / video_id / gt_music_id 三者都要一致）
    for exp, rows in exp_rows.items():
        if len(rows) != n_samples:
            raise ValueError(f"{exp} 樣本數 {len(rows)} 與基準 {REFERENCE_EXP} 的 {n_samples} 不符")
        for a, b in zip(ref_rows, rows):
            if (a["sample_idx"] != b["sample_idx"]
                    or a["video_id"] != b["video_id"]
                    or a["gt_music_id"] != b["gt_music_id"]):
                raise ValueError(
                    f"{exp} 與 {REFERENCE_EXP} 於 sample_idx={a['sample_idx']} 未對齊，"
                    "無法做配對比較。"
                )
    logger.info("[對齊] %d 個模型 × %d 樣本，sample_idx/video_id/gt_music_id 全數對齊",
                len(exp_rows), n_samples)

    # ---- 2. 重建候選池並計算相似度 ------------------------------------------
    logger.info("開始重建候選池（純 CPU）...")
    per_sample, coverage = rebuild_pool_similarity(ref_rows, bank, song_ids, logger)

    max_cos = np.array([r["max_neg_cosim"] for r in per_sample], dtype=np.float64)
    avg_cos = np.array([r["avg_neg_cosim"] for r in per_sample], dtype=np.float64)

    # ---- 3. 分層 -----------------------------------------------------------
    bins_max, edges_max = assign_bins(max_cos, N_BINS, BIN_LABELS)
    bins_avg, edges_avg = assign_bins(avg_cos, N_BINS, BIN_LABELS)
    for r, bm, ba in zip(per_sample, bins_max, bins_avg):
        r["difficulty_bin"] = bm
        r["difficulty_bin_by_avg"] = ba

    logger.info("[分層] 主指標 max_neg_cosim 切點：%s", [round(e, 4) for e in edges_max])
    logger.info("[分層] 次指標 avg_neg_cosim 切點：%s", [round(e, 4) for e in edges_avg])
    for lab in BIN_LABELS:
        sel = np.array([b == lab for b in bins_max])
        logger.info("        %-6s n=%4d  max_cos %.4f–%.4f (mean %.4f)",
                    lab, int(sel.sum()), max_cos[sel].min(), max_cos[sel].max(), max_cos[sel].mean())

    # ---- 4. 逐層彙整各模型指標 ----------------------------------------------
    ranks = {exp: np.array([int(r["rank"]) for r in rows], dtype=np.float64)
             for exp, rows in exp_rows.items()}
    bin_mask = {lab: np.array([b == lab for b in bins_max]) for lab in BIN_LABELS}

    by_bin_records = []
    for exp, (label, _rel) in EXPERIMENTS.items():
        for lab in BIN_LABELS + ["ALL"]:
            sel = np.ones(n_samples, dtype=bool) if lab == "ALL" else bin_mask[lab]
            m = metrics_from_ranks(ranks[exp][sel])
            by_bin_records.append({
                "exp": exp, "model": label, "difficulty_bin": lab,
                "mean_max_neg_cosim": float(max_cos[sel].mean()),
                "mean_avg_neg_cosim": float(avg_cos[sel].mean()),
                **m,
            })

    # Hard-Negative：已有逐樣本 cosim，直接當作難度軸最右端
    hardneg_record = None
    try:
        hn_rows = load_ranking_csv(HARDNEG_CSV, logger)
        hn_ranks = np.array([int(r["rank"]) for r in hn_rows], dtype=np.float64)
        hn_max = np.array([float(r["max_neg_cosim"]) for r in hn_rows], dtype=np.float64)
        hn_avg = np.array([float(r["avg_neg_cosim"]) for r in hn_rows], dtype=np.float64)
        hardneg_record = {
            "exp": "exp_01", "model": "Hybrid LTP (Matched)", "difficulty_bin": "HardNeg",
            "mean_max_neg_cosim": float(hn_max.mean()),
            "mean_avg_neg_cosim": float(hn_avg.mean()),
            **metrics_from_ranks(hn_ranks),
        }
        by_bin_records.append(hardneg_record)
        logger.info("[HardNeg] 併入難度軸：n=%d mean_max_cos=%.4f R@1=%.4f",
                    hardneg_record["n"], hardneg_record["mean_max_neg_cosim"], hardneg_record["R@1"])
    except FileNotFoundError:
        logger.warning("找不到 Hard-Negative CSV，難度軸將只含隨機池三層")

    # ---- 5. 逐層增益 + bootstrap CI ----------------------------------------
    gain_records = []
    for exp_a, exp_b in GAIN_PAIRS:
        if exp_a not in ranks or exp_b not in ranks:
            continue
        for lab in BIN_LABELS + ["ALL"]:
            sel = np.ones(n_samples, dtype=bool) if lab == "ALL" else bin_mask[lab]
            ra, rb = ranks[exp_a][sel], ranks[exp_b][sel]
            rec = {
                "comparison": f"{exp_a} - {exp_b}",
                "label": f"{EXPERIMENTS[exp_a][0]} − {EXPERIMENTS[exp_b][0]}",
                "difficulty_bin": lab,
                "n": int(sel.sum()),
                "mean_max_neg_cosim": float(max_cos[sel].mean()),
            }
            for metric in ["R@1", "R@5", "MRR", "nDCG@10"]:
                pt, lo, hi = paired_bootstrap_gain(
                    ra, rb, metric, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)
                rec[f"gain_{metric}"] = pt
                rec[f"gain_{metric}_ci_low"] = lo
                rec[f"gain_{metric}_ci_high"] = hi
                rec[f"gain_{metric}_significant"] = bool(lo > 0 or hi < 0)
            gain_records.append(rec)
            logger.info("[增益] %-17s %-6s n=%4d ΔR@1=%+.4f [%+.4f, %+.4f]",
                        rec["comparison"], lab, rec["n"],
                        rec["gain_R@1"], rec["gain_R@1_ci_low"], rec["gain_R@1_ci_high"])

    # ---- 5b. H4 正式檢定：增益是否隨難度縮小（DiD） --------------------------
    did_records = []
    for exp_a, exp_b in GAIN_PAIRS:
        if exp_a not in ranks or exp_b not in ranks:
            continue
        sel_e, sel_h = bin_mask[BIN_LABELS[0]], bin_mask[BIN_LABELS[-1]]
        rec = {
            "comparison": f"{exp_a} - {exp_b}",
            "label": f"{EXPERIMENTS[exp_a][0]} − {EXPERIMENTS[exp_b][0]}",
            "contrast": f"{BIN_LABELS[-1]} − {BIN_LABELS[0]}",
            "n_easy": int(sel_e.sum()), "n_hard": int(sel_h.sum()),
        }
        for metric in ["R@1", "R@5", "MRR", "nDCG@10"]:
            pt, lo, hi = bootstrap_did(
                ranks[exp_a][sel_h], ranks[exp_b][sel_h],
                ranks[exp_a][sel_e], ranks[exp_b][sel_e],
                metric, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)
            rec[f"did_{metric}"] = pt
            rec[f"did_{metric}_ci_low"] = lo
            rec[f"did_{metric}_ci_high"] = hi
            rec[f"did_{metric}_significant"] = bool(lo > 0 or hi < 0)
        did_records.append(rec)
        logger.info("[H4-DiD] %-17s ΔΔR@1=%+.4f [%+.4f, %+.4f] %s",
                    rec["comparison"], rec["did_R@1"],
                    rec["did_R@1_ci_low"], rec["did_R@1_ci_high"],
                    "顯著" if rec["did_R@1_significant"] else "不顯著")

    # ---- 5c. 難度軸對照：max cos vs avg cos ---------------------------------
    # Hard 層的 max_neg_cosim 已與 Hard-Negative 池相當，但 R@1 差距極大，
    # 關鍵差別在「整池的平均相似度」而非「單一最相似的干擾項」。
    axis_records = []
    for lab in BIN_LABELS:
        sel = bin_mask[lab]
        axis_records.append({
            "pool_type": f"random-500 / {lab}",
            "n": int(sel.sum()),
            "mean_max_neg_cosim": float(max_cos[sel].mean()),
            "mean_avg_neg_cosim": float(avg_cos[sel].mean()),
            "R@1": float(np.mean(ranks[REFERENCE_EXP][sel] <= 1)),
        })
    if hardneg_record is not None:
        axis_records.append({
            "pool_type": "hard-negative-500",
            "n": hardneg_record["n"],
            "mean_max_neg_cosim": hardneg_record["mean_max_neg_cosim"],
            "mean_avg_neg_cosim": hardneg_record["mean_avg_neg_cosim"],
            "R@1": hardneg_record["R@1"],
        })

    # ---- 6. 敏感度：改用 avg_neg_cosim 分層 ---------------------------------
    sens = []
    bin_mask_avg = {lab: np.array([b == lab for b in bins_avg]) for lab in BIN_LABELS}
    for lab in BIN_LABELS:
        sel = bin_mask_avg[lab]
        rec = {"difficulty_bin": lab, "n": int(sel.sum())}
        for exp in EXPERIMENTS:
            rec[f"{exp}_R@1"] = float(np.mean(ranks[exp][sel] <= 1))
        rec["gain_R@1_exp01_minus_exp04"] = rec["exp_01_R@1"] - rec["exp_04_R@1"]
        sens.append(rec)

    # ---- 7. 輸出 -----------------------------------------------------------
    ps_path = OUT_DIR / "candidate_difficulty_per_sample.csv"
    with open(ps_path, "w", newline="", encoding="utf-8-sig") as f:
        cols = ["sample_idx", "video_id", "gt_music_id", "n_negatives",
                "avg_neg_cosim", "max_neg_cosim", "top10_neg_cosim",
                "difficulty_bin", "difficulty_bin_by_avg"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_sample)

    bb_path = OUT_DIR / "candidate_difficulty_by_bin.csv"
    with open(bb_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(by_bin_records[0].keys()))
        w.writeheader()
        w.writerows(by_bin_records)

    gn_path = OUT_DIR / "candidate_difficulty_gain.csv"
    with open(gn_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(gain_records[0].keys()))
        w.writeheader()
        w.writerows(gain_records)

    did_path = OUT_DIR / "candidate_difficulty_h4_did.csv"
    with open(did_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(did_records[0].keys()))
        w.writeheader()
        w.writerows(did_records)

    summary = {
        "generated_at": started.isoformat(timespec="seconds"),
        "reconstruction": {
            "candidate_pool_seed": CANDIDATE_POOL_SEED,
            "pool_size": POOL_SIZE,
            "song_bank_size": len(song_ids),
            "n_samples": n_samples,
            "reference_exp": REFERENCE_EXP,
            "top1_in_pool_coverage": coverage,
            "note": "候選池以 random.Random(seed+idx).sample(candidates, 499) 離線重建，"
                    "邏輯與 run_eval_500pool_detailed.py 相同；未重跑任何模型推論。",
        },
        "binning": {
            "primary_metric": "max_neg_cosim",
            "rule": f"全體測試樣本的 {'/'.join(str(round(100 * i / N_BINS, 1)) for i in range(1, N_BINS))} 百分位等分",
            "edges_max_neg_cosim": edges_max,
            "edges_avg_neg_cosim": edges_avg,
            "labels": BIN_LABELS,
            "model_independent": True,
        },
        "by_bin": by_bin_records,
        "gain": gain_records,
        "h4_did": did_records,
        "difficulty_axis": axis_records,
        "sensitivity_by_avg_cosim": sens,
        "hardneg": hardneg_record,
    }
    sm_path = OUT_DIR / "candidate_difficulty_summary.json"
    with open(sm_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = OUT_DIR / "candidate_difficulty_summary.md"
    write_markdown(md_path, summary, by_bin_records, gain_records,
                   did_records, axis_records, edges_max, sens)

    elapsed = (_dt.datetime.now() - started).total_seconds()
    for p in [ps_path, bb_path, gn_path, did_path, sm_path, md_path]:
        logger.info("已輸出：%s", p)
    logger.info("完成，耗時 %.1f 秒", elapsed)


def write_markdown(path, summary, by_bin, gain, did, axis, edges, sens):
    L = []
    L.append("# 候選聲學相似度分層分析（B1）\n")
    L.append(f"- 產生時間：{summary['generated_at']}")
    r = summary["reconstruction"]
    L.append(f"- 候選池重建：seed={r['candidate_pool_seed']}、pool={r['pool_size']}、"
             f"song bank={r['song_bank_size']}、樣本數={r['n_samples']}")
    L.append(f"- **重建驗證**：既有結果中非 GT 的 top-1 有 {r['top1_in_pool_coverage']*100:.2f}% "
             f"落在重建出來的候選池內（門檻 99.9%）")
    L.append(f"- 分層規則（事前定義）：以 max_neg_cosim 三等分，切點 = "
             f"{edges[0]:.4f} / {edges[1]:.4f}；分層只由候選池幾何決定，與模型表現無關\n")

    L.append("## 表 1　各模型 × 難度層的排序表現\n")
    L.append("| 模型 | 難度層 | n | 平均 max cos | R@1 | R@5 | R@10 | MRR | nDCG@10 | 中位排名 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rec in by_bin:
        L.append(f"| {rec['model']} | {rec['difficulty_bin']} | {rec['n']} | "
                 f"{rec['mean_max_neg_cosim']:.4f} | {rec['R@1']*100:.2f}% | {rec['R@5']*100:.2f}% | "
                 f"{rec['R@10']*100:.2f}% | {rec['MRR']:.4f} | {rec['nDCG@10']:.4f} | "
                 f"{rec['median_rank']:.1f} |")

    L.append("\n## 表 2　LTP 增益隨候選相似度的變化（配對 bootstrap 95% CI，2000 次）\n")
    L.append("| 比較 | 難度層 | n | ΔR@1 | 95% CI | ΔR@5 | ΔMRR | ΔnDCG@10 | 顯著 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for rec in gain:
        L.append(f"| {rec['comparison']} | {rec['difficulty_bin']} | {rec['n']} | "
                 f"{rec['gain_R@1']*100:+.2f}pp | "
                 f"[{rec['gain_R@1_ci_low']*100:+.2f}, {rec['gain_R@1_ci_high']*100:+.2f}] | "
                 f"{rec['gain_R@5']*100:+.2f}pp | {rec['gain_MRR']:+.4f} | "
                 f"{rec['gain_nDCG@10']:+.4f} | {'是' if rec['gain_R@1_significant'] else '否'} |")

    L.append("\n## 表 3　H4 正式檢定：增益是否隨難度縮小（difference-in-differences）\n")
    L.append("| 比較 | 對比 | ΔΔR@1 | 95% CI | ΔΔR@5 | ΔΔMRR | ΔΔnDCG@10 | R@1 顯著 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for rec in did:
        L.append(f"| {rec['comparison']} | {rec['contrast']} | {rec['did_R@1']*100:+.2f}pp | "
                 f"[{rec['did_R@1_ci_low']*100:+.2f}, {rec['did_R@1_ci_high']*100:+.2f}] | "
                 f"{rec['did_R@5']*100:+.2f}pp | {rec['did_MRR']:+.4f} | "
                 f"{rec['did_nDCG@10']:+.4f} | {'是' if rec['did_R@1_significant'] else '否'} |")
    L.append("\n> 負值代表增益在 Hard 層縮小。CI 不含 0 才可宣稱「LTP 增益隨候選相似度下降」。\n")

    L.append("## 表 4　難度軸對照：單一最相似干擾項 vs 整池平均相似度\n")
    L.append("| 候選池 | n | 平均 max cos | 平均 avg cos | R@1（Hybrid） |")
    L.append("|---|---|---|---|---|")
    for rec in axis:
        L.append(f"| {rec['pool_type']} | {rec['n']} | {rec['mean_max_neg_cosim']:.4f} | "
                 f"{rec['mean_avg_neg_cosim']:.4f} | {rec['R@1']*100:.2f}% |")

    L.append("\n## 表 5　敏感度檢查：改以 avg_neg_cosim 分層的 R@1\n")
    L.append("| 難度層 | n | Hybrid | Explicit-only | Implicit-only | No-LTP | Δ(Hybrid−NoLTP) |")
    L.append("|---|---|---|---|---|---|---|")
    for rec in sens:
        L.append(f"| {rec['difficulty_bin']} | {rec['n']} | {rec['exp_01_R@1']*100:.2f}% | "
                 f"{rec['exp_02_R@1']*100:.2f}% | {rec['exp_03_R@1']*100:.2f}% | "
                 f"{rec['exp_04_R@1']*100:.2f}% | "
                 f"{rec['gain_R@1_exp01_minus_exp04']*100:+.2f}pp |")

    L.append("\n## 判讀說明\n")
    L.append("- **nDCG@10**：本任務每個查詢僅有一個相關項目（GT），故 IDCG = 1，"
             "nDCG@10 = 1/log2(rank+1)（rank ≤ 10）或 0。")
    L.append("- **Hard-Negative 列**：候選池改由 AST 最相似的 499 首構成，"
             "代表難度軸的極端端點，與隨機池三層共同構成連續難度軸。")
    L.append("- **顯著性**：以配對 bootstrap 95% CI 是否跨越 0 判定；"
             "本分析為事後分層（post-hoc stratification），分層規則事前固定且與模型無關。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
