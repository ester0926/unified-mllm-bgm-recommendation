# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
video_cluster_stratification.py
===============================
B4：短影音內容分層（CLIP 語意叢集 k = 3–4）

對應指導教授 0723 建議 §八，並依 2026-07-26 定案調整方法：

  教授原本設想依「旅遊／美食／知識教學／產品商業／娛樂舞蹈…」等內容類型分層。
  但 MuseChat 的「影片」其實就是**音樂自身的 YouTube 影片**（video_id == target_music_id），
  並非創作者拍攝的短影音，內容分布高度集中於 MV／演出／歌詞影片，硬套七類會切不出來。
  因此改為：對影片 CLIP 影像嵌入做**無監督語意叢集（k = 3–4）**，事後人工檢視命名，
  並在限制章明確說明資料性質。

為什麼可以直接叢集：
  影片特徵是 CLIP ViT-L/14 影像嵌入（768 維，12 幀），與模型實際看到的輸入完全相同
  （評估模式為 12 幀平均，見 dataset.py 第 390 行）。因此叢集切出的是
  「模型眼中的影片語意分群」，而非另一套與模型無關的標籤。

流程：
  1. 依測試集 4,205 筆的 gt_music_id（= pair_key）自 H5 取出 video_features_all，
     12 幀平均 → 768 維，L2 正規化。特徵快取於 cache/test_video_features.npz，
     之後重跑不需再讀 F: 磁碟。
  2. k-means 叢集；k = 2…6 皆計算輪廓係數（silhouette）作為選 k 依據，
     主結果報 k = 3 與 k = 4 兩種切法。
  3. 以 sample_idx 併回既有逐樣本結果，分層報告：
       樣本數、R@1 / R@5 / MRR / nDCG@10（exp_01 與 exp_04）
       LTP 增益（exp_01 − exp_04）+ 配對拔靴 95% CI
       Hard-Negative 下降幅度
       UCR（子句層級 L1 與母句校正 L2，沿用 B2 定義）
       B1 難度層（Easy/Medium/Hard）分布 —— 用以檢查叢集是否只是換個方式表達聲學難度
  4. 產出人工命名工作表：每叢集抽樣 40 支影片，附 YouTube 連結與標題供檢視命名。

⚠ 判讀限制（必須寫進論文）：
  叢集名稱由人工事後檢視得出，屬探索性分群，不等同於獨立標註的內容類型標籤；
  且此資料集的影片並非創作者拍攝之短影音，分層結果不可外推到真實短影音情境。

輸出（results/analysis/video_clusters/）：
  video_cluster_assignments.csv     逐樣本叢集標籤（k=3 與 k=4）
  video_cluster_quality.csv         k = 2…6 的輪廓係數與各叢集大小
  video_cluster_metrics.csv         各叢集 × 各模型的排序與生成指標
  video_cluster_gain.csv            LTP 增益逐叢集 + 拔靴 CI
  video_cluster_naming_sheet.csv    人工命名用抽樣清單（含 YouTube 連結）
  video_cluster_summary.json / .md
  video_cluster.log

使用方式：
  第一次執行需讀取 F: 磁碟的 H5，請用有 h5py 的環境：
    <user_home>/anaconda3\\envs\\ollama\\python.exe scripts/analysis/video_cluster_stratification.py
  特徵快取建立後，之後可改用 base 環境重跑分析。
"""

import csv
import datetime as _dt
import json
import logging
import random
from collections import Counter, defaultdict

import numpy as np

from scripts.faithfulness import faithfulness_claim_judge_v2 as J
from scripts.faithfulness import analyze_ucr_error_sources as B2


# =============================================================================
# 路徑設定
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR   = PROJECT_ROOT / "cache"
OUT_DIR     = RESULTS_DIR / "analysis" / "video_clusters"

PAIR_INDEX_JSON = CACHE_DIR / "pair_index.json"
FEATURE_CACHE   = CACHE_DIR / "test_video_features.npz"

YOUTUBE_METADATA = Path(
    r"data/user_profiling/music_metadata_simple\youtube_metadata.jsonl"
)

RANKING_CSV = {
    "exp_01": "main_eval/exp_01/detailed_eval/exp_01_best_500pool_prompt_original_ranking_samples.csv",
    "exp_04": "main_eval/exp_04/detailed_eval/exp_04_best_500pool_ranking_samples.csv",
}
HARDNEG_CSV = "main_eval/exp_01/hardneg_eval/exp_01_best_500pool_hardneg_top499_ranking_samples.csv"
GENERATION_CSV = ("main_eval/exp_01/detailed_eval/"
                  "exp_01_best_500pool_top1_prompt_original_samples_merged.csv")
DIFFICULTY_CSV = RESULTS_DIR / "analysis" / "candidate_difficulty" / "candidate_difficulty_per_sample.csv"


# =============================================================================
# 使用者設定
# =============================================================================

K_VALUES      = [2, 3, 4, 5, 6]      # 用於輪廓係數比較
K_REPORT      = [3, 4]               # 主結果報告的切法
KMEANS_SEED   = 20260726
KMEANS_N_INIT = 20

NAMING_SAMPLE_PER_CLUSTER = 40
NAMING_SEED = 20260726

BOOTSTRAP_N    = 2000
BOOTSTRAP_SEED = 20260726
CI_ALPHA       = 0.05


# =============================================================================
# Logger
# =============================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("video_cluster")
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


def read_csv(path: Path) -> list:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# =============================================================================
# 影片特徵抽取
# =============================================================================

def load_video_features(ref_rows, logger):
    """
    取出測試集每筆樣本的影片特徵（12 幀平均、L2 正規化）。
    與 dataset.py 評估模式的處理一致（is_train=False → mean over frames）。
    """
    if FEATURE_CACHE.exists():
        data = np.load(FEATURE_CACHE, allow_pickle=True)
        feats, keys = data["features"], list(data["pair_keys"])
        if len(keys) == len(ref_rows) and keys == [r["gt_music_id"] for r in ref_rows]:
            logger.info("[特徵] 由快取載入：%s %s", FEATURE_CACHE.name, feats.shape)
            return feats
        logger.warning("[特徵] 快取與目前樣本順序不符，重新抽取")

    import h5py                                   # 僅在需要讀 H5 時才載入

    pair_index = json.loads(PAIR_INDEX_JSON.read_text(encoding="utf-8"))
    key_to_h5 = {pk: h5 for h5, pk in pair_index}

    by_file = defaultdict(list)
    for i, row in enumerate(ref_rows):
        pk = row["gt_music_id"]
        h5_path = key_to_h5.get(pk)
        if h5_path is None:
            raise KeyError(f"pair_key 不在 pair_index 中：{pk}（sample_idx={row['sample_idx']}）")
        by_file[h5_path].append((i, pk))

    feats = np.zeros((len(ref_rows), 768), dtype=np.float32)
    logger.info("[特徵] 需讀取 %d 個 H5 檔、%d 筆樣本", len(by_file), len(ref_rows))
    for n, (h5_path, items) in enumerate(sorted(by_file.items()), start=1):
        with h5py.File(h5_path, "r") as f:
            for i, pk in items:
                arr = f[f"pairs/{pk}/video_features_all"][:].astype(np.float32)
                feats[i] = arr.mean(axis=0)
        if n % 20 == 0 or n == len(by_file):
            logger.info("  已讀 %d/%d 檔", n, len(by_file))

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats /= norms

    np.savez_compressed(FEATURE_CACHE, features=feats,
                        pair_keys=np.array([r["gt_music_id"] for r in ref_rows], dtype=object))
    logger.info("[特徵] 已抽取並快取至 %s", FEATURE_CACHE)
    return feats


# =============================================================================
# 叢集
# =============================================================================

def cluster_videos(feats, logger):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    quality, labels_by_k = [], {}
    rng = np.random.default_rng(KMEANS_SEED)
    sub = rng.choice(feats.shape[0], size=min(2000, feats.shape[0]), replace=False)

    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=KMEANS_SEED, n_init=KMEANS_N_INIT)
        labels = km.fit_predict(feats)
        sil = float(silhouette_score(feats[sub], labels[sub], metric="euclidean"))
        sizes = Counter(labels.tolist())
        quality.append({
            "k": k, "silhouette": sil, "inertia": float(km.inertia_),
            "cluster_sizes": ";".join(f"{c}:{sizes[c]}" for c in sorted(sizes)),
            "min_cluster_share": min(sizes.values()) / feats.shape[0],
        })
        labels_by_k[k] = labels
        logger.info("[叢集] k=%d silhouette=%.4f 最小叢集占比=%.1f%% 分布=%s",
                    k, sil, 100 * quality[-1]["min_cluster_share"],
                    quality[-1]["cluster_sizes"])
    return labels_by_k, quality


# =============================================================================
# 指標
# =============================================================================

def rank_metrics(ranks: np.ndarray) -> dict:
    if ranks.size == 0:
        return {k: float("nan") for k in ["R@1", "R@5", "R@10", "MRR", "nDCG@10", "median_rank"]}
    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
        "nDCG@10": float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0))),
        "median_rank": float(np.median(ranks)),
    }


def bootstrap_gain(ranks_a, ranks_b, metric, n_boot, seed, alpha):
    def _m(r):
        if metric == "R@1":
            return np.mean(r <= 1)
        if metric == "R@5":
            return np.mean(r <= 5)
        if metric == "MRR":
            return np.mean(1.0 / r)
        raise ValueError(metric)

    point = float(_m(ranks_a) - _m(ranks_b))
    n = ranks_a.size
    if n == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)
        draws[b] = _m(ranks_a[sel]) - _m(ranks_b[sel])
    return (point,
            float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def per_sample_ucr(gen_rows, logger):
    """逐樣本的 claim 數與未支持數（L1 子句層級、L2 母句校正），沿用 B2 定義。"""
    n_claims = np.zeros(len(gen_rows))
    n_l1 = np.zeros(len(gen_rows))
    n_l2 = np.zeros(len(gen_rows))
    for i, row in enumerate(gen_rows):
        generated = row.get("generated_text", "") or ""
        for claim in J.split_claims(generated):
            n_claims[i] += 1
            source, _ = J.classify_claim(claim, row)
            if source == J.SOURCE_UNSUPPORTED:
                n_l1[i] += 1
                parent = B2.find_parent_sentence(claim, generated)
                p_source, _ = J.classify_claim(parent, row)
                n_l2[i] += int(p_source == J.SOURCE_UNSUPPORTED)
    logger.info("[UCR] 逐樣本統計完成：claim=%d L1未支持=%d L2未支持=%d",
                int(n_claims.sum()), int(n_l1.sum()), int(n_l2.sum()))
    return n_claims, n_l1, n_l2


# =============================================================================
# 主流程
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUT_DIR / "video_cluster.log")
    started = _dt.datetime.now()

    logger.info("=" * 78)
    logger.info("B4 短影音內容分層（CLIP 語意叢集）")
    logger.info("=" * 78)

    # ---- 1. 載入既有逐樣本結果 ---------------------------------------------
    exp_rows = {e: read_csv(RESULTS_DIR / p) for e, p in RANKING_CSV.items()}
    ref_rows = exp_rows["exp_01"]
    n = len(ref_rows)
    for e, rows in exp_rows.items():
        if [r["sample_idx"] for r in rows] != [r["sample_idx"] for r in ref_rows]:
            raise ValueError(f"{e} 與 exp_01 的樣本順序不一致")
    logger.info("[載入] 測試樣本 %d 筆", n)

    ranks = {e: np.array([int(r["rank"]) for r in rows], dtype=np.float64)
             for e, rows in exp_rows.items()}

    hn_rows = read_csv(RESULTS_DIR / HARDNEG_CSV)
    hn_map = {r["sample_idx"]: int(r["rank"]) for r in hn_rows}
    hn_ranks = np.array([hn_map.get(r["sample_idx"], np.nan) for r in ref_rows], dtype=np.float64)

    gen_rows = read_csv(RESULTS_DIR / GENERATION_CSV)
    gen_map = {r["sample_idx"]: r for r in gen_rows}
    gen_aligned = [gen_map[r["sample_idx"]] for r in ref_rows]

    diff_bin = {}
    if DIFFICULTY_CSV.exists():
        for r in read_csv(DIFFICULTY_CSV):
            diff_bin[r["sample_idx"]] = r["difficulty_bin"]
        logger.info("[載入] B1 難度層標籤 %d 筆", len(diff_bin))

    # ---- 2. 影片特徵與叢集 --------------------------------------------------
    feats = load_video_features(ref_rows, logger)
    labels_by_k, quality = cluster_videos(feats, logger)

    # ---- 3. 逐樣本 UCR ------------------------------------------------------
    n_claims, n_l1, n_l2 = per_sample_ucr(gen_aligned, logger)

    # ---- 4. 分層彙整 --------------------------------------------------------
    assignments, metrics_rows, gain_rows = [], [], []
    for i, row in enumerate(ref_rows):
        rec = {"sample_idx": row["sample_idx"], "video_id": row["video_id"],
               "gt_music_id": row["gt_music_id"],
               "difficulty_bin": diff_bin.get(row["sample_idx"], "")}
        for k in K_REPORT:
            rec[f"cluster_k{k}"] = int(labels_by_k[k][i])
        assignments.append(rec)

    for k in K_REPORT:
        labels = labels_by_k[k]
        for c in range(k):
            sel = labels == c
            n_c = int(sel.sum())
            claims_c, l1_c, l2_c = n_claims[sel].sum(), n_l1[sel].sum(), n_l2[sel].sum()
            hn_sel = hn_ranks[sel]
            hn_valid = hn_sel[~np.isnan(hn_sel)]
            row_out = {
                "k": k, "cluster": c, "n": n_c, "share": n_c / n,
                **{f"exp_01_{m}": v for m, v in rank_metrics(ranks["exp_01"][sel]).items()},
                **{f"exp_04_{m}": v for m, v in rank_metrics(ranks["exp_04"][sel]).items()},
                "hardneg_R@1": float(np.mean(hn_valid <= 1)) if hn_valid.size else float("nan"),
                "hardneg_drop_R@1": (float(np.mean(ranks["exp_01"][sel] <= 1)
                                           - np.mean(hn_valid <= 1)) if hn_valid.size else float("nan")),
                "UCR_L1": float(l1_c / claims_c) if claims_c else float("nan"),
                "UCR_L2": float(l2_c / claims_c) if claims_c else float("nan"),
                "claims_per_generation": float(claims_c / n_c) if n_c else float("nan"),
            }
            if diff_bin:
                bins = Counter(assignments[i]["difficulty_bin"]
                               for i in range(n) if sel[i])
                total_b = sum(bins.values()) or 1
                for lab in ("Easy", "Medium", "Hard"):
                    row_out[f"difficulty_{lab}_share"] = bins.get(lab, 0) / total_b
            metrics_rows.append(row_out)

            for metric in ["R@1", "R@5", "MRR"]:
                pt, lo, hi = bootstrap_gain(ranks["exp_01"][sel], ranks["exp_04"][sel],
                                            metric, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)
                gain_rows.append({
                    "k": k, "cluster": c, "n": n_c, "metric": metric,
                    "gain": pt, "ci_low": lo, "ci_high": hi,
                    "significant": bool(lo > 0 or hi < 0),
                })

        logger.info("-" * 78)
        for r in [x for x in metrics_rows if x["k"] == k]:
            g = [x for x in gain_rows if x["k"] == k and x["cluster"] == r["cluster"]
                 and x["metric"] == "R@1"][0]
            logger.info("[k=%d 叢集%d] n=%4d (%.1f%%) exp_01 R@1=%.4f exp_04 R@1=%.4f "
                        "增益=%+.4f%s HardNeg R@1=%.4f UCR_L1=%.4f",
                        k, r["cluster"], r["n"], 100 * r["share"], r["exp_01_R@1"],
                        r["exp_04_R@1"], g["gain"], "*" if g["significant"] else " ",
                        r["hardneg_R@1"], r["UCR_L1"])

    # ---- 4b. 增益是否真的隨叢集改變（教授問題的正式檢定）--------------------
    # 逐叢集各自顯著只代表「每個叢集內 LTP 都有用」，不等於「增益因叢集而異」。
    # 這裡取增益最高與最低的兩個叢集做差異中的差異，兩層獨立重抽；
    # CI 含 0 即無證據支持「合成偏好只在某些影片類型有效」。
    homogeneity = []
    for k in K_REPORT:
        labels = labels_by_k[k]
        gains = {c: [x for x in gain_rows
                     if x["k"] == k and x["cluster"] == c and x["metric"] == "R@1"][0]["gain"]
                 for c in range(k)}
        c_hi = max(gains, key=gains.get)
        c_lo = min(gains, key=gains.get)
        sel_hi, sel_lo = labels == c_hi, labels == c_lo

        def _r1(a, b, s):
            return np.mean(a[s] <= 1) - np.mean(b[s] <= 1)

        point = _r1(ranks["exp_01"], ranks["exp_04"], sel_hi) - \
            _r1(ranks["exp_01"], ranks["exp_04"], sel_lo)
        idx_hi, idx_lo = np.flatnonzero(sel_hi), np.flatnonzero(sel_lo)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = np.empty(BOOTSTRAP_N)
        for b in range(BOOTSTRAP_N):
            sh = rng.choice(idx_hi, size=idx_hi.size, replace=True)
            sl = rng.choice(idx_lo, size=idx_lo.size, replace=True)
            draws[b] = ((np.mean(ranks["exp_01"][sh] <= 1) - np.mean(ranks["exp_04"][sh] <= 1))
                        - (np.mean(ranks["exp_01"][sl] <= 1) - np.mean(ranks["exp_04"][sl] <= 1)))
        lo = float(np.percentile(draws, 100 * CI_ALPHA / 2))
        hi = float(np.percentile(draws, 100 * (1 - CI_ALPHA / 2)))
        # 置換檢定：上面的「最高減最低」是在看過資料後才挑出來的，
        # 這種事後選擇會膨脹型一錯誤（k 個叢集就等於做了 C(k,2) 次比較）。
        # 這裡打亂叢集標籤（保持各叢集大小），在虛無假設下重算「最大－最小增益全距」，
        # 以觀測全距的分位數作為 p 值，即為對選擇效應校正後的整體檢定。
        gain_vec = (ranks["exp_01"] <= 1).astype(np.float64) - (ranks["exp_04"] <= 1).astype(np.float64)
        obs_range = max(gains.values()) - min(gains.values())
        rng_p = np.random.default_rng(BOOTSTRAP_SEED)
        perm_ranges = np.empty(BOOTSTRAP_N)
        perm_labels = labels.copy()
        for b in range(BOOTSTRAP_N):
            rng_p.shuffle(perm_labels)
            g = [gain_vec[perm_labels == c].mean() for c in range(k)]
            perm_ranges[b] = max(g) - min(g)
        p_perm = float((np.sum(perm_ranges >= obs_range) + 1) / (BOOTSTRAP_N + 1))

        rec = {"k": k, "cluster_max_gain": c_hi, "cluster_min_gain": c_lo,
               "gain_max": gains[c_hi], "gain_min": gains[c_lo],
               "gain_range": float(point), "ci_low": lo, "ci_high": hi,
               "significant": bool(lo > 0 or hi < 0),
               "permutation_p": p_perm,
               "permutation_significant": bool(p_perm < 0.05),
               "hard_share_range": float(
                   max(r.get("difficulty_Hard_share", 0) for r in metrics_rows if r["k"] == k)
                   - min(r.get("difficulty_Hard_share", 0) for r in metrics_rows if r["k"] == k))}
        homogeneity.append(rec)
        logger.info("[同質性 k=%d] 最高增益叢集%d(%.4f) − 最低叢集%d(%.4f) = %+.4f "
                    "[%+.4f, %+.4f]（事後選擇）｜置換檢定 p=%.4f → %s",
                    k, c_hi, gains[c_hi], c_lo, gains[c_lo], point, lo, hi, p_perm,
                    "增益因叢集而異" if rec["permutation_significant"]
                    else "無證據支持增益隨叢集改變")

    # ---- 5. 人工命名工作表 --------------------------------------------------
    yt = {}
    if YOUTUBE_METADATA.exists():
        with open(YOUTUBE_METADATA, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                yt[d["music_id"]] = d
        logger.info("[命名表] 載入 YouTube metadata %d 筆", len(yt))

    rng = random.Random(NAMING_SEED)
    naming = []
    for k in K_REPORT:
        labels = labels_by_k[k]
        for c in range(k):
            idxs = [i for i in range(n) if labels[i] == c]
            pick = rng.sample(idxs, min(NAMING_SAMPLE_PER_CLUSTER, len(idxs)))
            for i in pick:
                vid = ref_rows[i]["video_id"]
                meta = yt.get(vid, {})
                naming.append({
                    "k": k, "cluster": c, "sample_idx": ref_rows[i]["sample_idx"],
                    "video_id": vid,
                    "youtube_url": meta.get("youtube_url", f"https://www.youtube.com/watch?v={vid}"),
                    "title": meta.get("title", ""), "artist": meta.get("artist", ""),
                    "duration_sec": meta.get("duration", ""),
                    "view_count": meta.get("view_count", ""),
                    "cluster_name": "",          # 人工填寫
                    "note": "",
                })

    # ---- 6. 輸出 ------------------------------------------------------------
    def dump(name, rows):
        if not rows:
            return
        with open(OUT_DIR / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("video_cluster_assignments.csv", assignments)
    dump("video_cluster_quality.csv", quality)
    dump("video_cluster_metrics.csv", metrics_rows)
    dump("video_cluster_gain.csv", gain_rows)
    dump("video_cluster_homogeneity.csv", homogeneity)
    dump("video_cluster_naming_sheet.csv", naming)

    summary = {
        "generated_at": started.isoformat(timespec="seconds"),
        "n_samples": n,
        "feature": "CLIP ViT-L/14 影像嵌入，12 幀平均後 L2 正規化（與模型評估時輸入一致）",
        "clustering": {"algorithm": "k-means", "seed": KMEANS_SEED, "n_init": KMEANS_N_INIT,
                       "k_evaluated": K_VALUES, "k_reported": K_REPORT},
        "quality": quality,
        "metrics": metrics_rows,
        "gain": gain_rows,
        "homogeneity": homogeneity,
        "naming_sheet_rows": len(naming),
        "limits": [
            "本資料集的「影片」即音樂自身的 YouTube 影片（video_id == target_music_id），"
            "並非創作者拍攝之短影音，分層結果不可外推至真實短影音創作情境。",
            "叢集名稱由人工事後檢視命名，屬探索性分群，不等同獨立標註的內容類型標籤。",
            "須同時檢視各叢集的 B1 難度層分布：若叢集與聲學難度高度共變，"
            "則分層差異可能只是難度差異的另一種表現，而非內容類型效果。",
        ],
    }
    with open(OUT_DIR / "video_cluster_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_markdown(OUT_DIR / "video_cluster_summary.md", summary)

    logger.info("已輸出至 %s", OUT_DIR)
    logger.info("完成，耗時 %.1f 秒", (_dt.datetime.now() - started).total_seconds())


def write_markdown(path, summary):
    L = ["# 短影音內容分層（B4：CLIP 語意叢集）\n",
         f"- 產生時間：{summary['generated_at']}",
         f"- 樣本數：{summary['n_samples']}",
         f"- 特徵：{summary['feature']}",
         f"- 叢集：{summary['clustering']['algorithm']}，"
         f"seed={summary['clustering']['seed']}，n_init={summary['clustering']['n_init']}\n",
         "## 一、選 k 的依據\n",
         "| k | 輪廓係數 | 最小叢集占比 | 各叢集大小 |", "|---|---|---|---|"]
    for q in summary["quality"]:
        L.append(f"| {q['k']} | {q['silhouette']:.4f} | {q['min_cluster_share']*100:.1f}% | "
                 f"{q['cluster_sizes']} |")

    for k in summary["clustering"]["k_reported"]:
        L.append(f"\n## 二、k = {k} 的分層結果\n")
        L.append("| 叢集 | n | 占比 | exp_01 R@1 | exp_04 R@1 | LTP 增益 | HardNeg R@1 | "
                 "HardNeg 降幅 | UCR(L1) | UCR(L2) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in [x for x in summary["metrics"] if x["k"] == k]:
            g = [x for x in summary["gain"] if x["k"] == k and x["cluster"] == r["cluster"]
                 and x["metric"] == "R@1"][0]
            L.append(f"| {r['cluster']} | {r['n']} | {r['share']*100:.1f}% | "
                     f"{r['exp_01_R@1']*100:.2f}% | {r['exp_04_R@1']*100:.2f}% | "
                     f"{g['gain']*100:+.2f}pp{'*' if g['significant'] else ''} | "
                     f"{r['hardneg_R@1']*100:.2f}% | {r['hardneg_drop_R@1']*100:.2f}pp | "
                     f"{r['UCR_L1']*100:.2f}% | {r['UCR_L2']*100:.2f}% |")
        if any("difficulty_Easy_share" in x for x in summary["metrics"]):
            L.append(f"\n**k = {k} 各叢集的 B1 難度層分布**（檢查叢集是否只是聲學難度的代理）\n")
            L.append("| 叢集 | Easy | Medium | Hard |")
            L.append("|---|---|---|---|")
            for r in [x for x in summary["metrics"] if x["k"] == k]:
                L.append(f"| {r['cluster']} | {r.get('difficulty_Easy_share', 0)*100:.1f}% | "
                         f"{r.get('difficulty_Medium_share', 0)*100:.1f}% | "
                         f"{r.get('difficulty_Hard_share', 0)*100:.1f}% |")

    L.append("\n> `*` 表示 LTP 增益的配對拔靴 95% CI 不跨 0。\n")

    L.append("## 二之二、增益是否隨叢集改變（教授問題的正式檢定）\n")
    L.append("逐叢集各自顯著只代表「每個叢集內 LTP 都有用」，不等於「增益因影片類型而異」。"
             "下表取增益最高與最低的叢集做差異中的差異，兩層獨立重抽。\n")
    L.append("| k | 最高增益叢集 | 最低增益叢集 | 差距 | 事後 95% CI | 置換檢定 p | 結論 | Hard 佔比全距 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for h in summary["homogeneity"]:
        L.append(f"| {h['k']} | 叢集{h['cluster_max_gain']}（{h['gain_max']*100:+.2f}pp）| "
                 f"叢集{h['cluster_min_gain']}（{h['gain_min']*100:+.2f}pp）| "
                 f"{h['gain_range']*100:+.2f}pp | "
                 f"[{h['ci_low']*100:+.2f}, {h['ci_high']*100:+.2f}] | "
                 f"{h['permutation_p']:.4f} | "
                 f"{'增益因叢集而異' if h['permutation_significant'] else '無證據'} | "
                 f"{h['hard_share_range']*100:.1f}pp |")
    L.append("\n> **為何要看置換檢定**：「最高減最低」是在看過結果後才挑出的兩個叢集，"
             "等同做了 C(k,2) 次比較，事後 CI 會膨脹型一錯誤。"
             "置換檢定打亂叢集標籤（保持各叢集大小）重算最大－最小全距，"
             "已將此選擇效應納入虛無分布，應以其 p 值為準。\n")
    L.append("## 三、下一步：人工命名\n")
    L.append(f"`video_cluster_naming_sheet.csv` 已抽出每叢集 40 支影片（共 "
             f"{summary['naming_sheet_rows']} 列），附 YouTube 連結、標題與演出者。"
             "請逐叢集瀏覽後在 `cluster_name` 欄填入叢集名稱，再回填至論文表格。\n")
    L.append("## 四、限制\n")
    for s in summary["limits"]:
        L.append(f"- {s}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
