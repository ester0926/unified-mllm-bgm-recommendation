"""
用途：分析不同資訊來源路徑對生成解釋的影響。
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
import os
import random
import re

import numpy as np

from scripts.faithfulness import faithfulness_claim_judge_v2 as J
from scripts.faithfulness import analyze_ucr_error_sources as B2


# =============================================================================
# 路徑設定
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR     = RESULTS_DIR / "analysis" / "path_level_generation_v21"

PROFILES_JSONL = Path(os.environ.get(
    "PROFILES_JSONL_PATH",
    r"data/user_profiling/long_term_preference\stage4_recLLM\profiles.jsonl",
))

GENERATION_CSV_TMPL = (
    "main_eval/{exp}/detailed_eval/{exp}_best_500pool_top1_prompt_original_samples_merged.csv"
)


# =============================================================================
# 使用者設定
# =============================================================================

EXPERIMENTS = {
    "exp_01": "Hybrid LTP",
    "exp_02": "Explicit-only LTP",
    "exp_03": "Implicit-only LTP",
    "exp_04": "No-LTP",
}

# 兩兩對照：(A, B) 報 A − B
CONTRASTS = [
    ("exp_01", "exp_04"),   # 有無 LTP
    ("exp_03", "exp_04"),
    ("exp_02", "exp_04"),
    ("exp_01", "exp_03"),   # 混合 vs 隱性（排序上兩者無顯著差異，生成側是否有？）
    ("exp_02", "exp_03"),   # 顯性 vs 隱性（功能分工的關鍵對照）
    ("exp_01", "exp_02"),
]

BOOTSTRAP_N    = 2000
BOOTSTRAP_SEED = 20260726
CI_ALPHA       = 0.05

CLAIM_SAMPLE_PER_EXP = 60      # 抽樣輸出供人工查核
CLAIM_SAMPLE_SEED    = 20260726

# 用於比對偏好畫像的「可辨識屬性詞」。
# 刻意排除 sound / tone / timbre / texture / instrument / music 等泛用詞——
# 它們幾乎必然出現在任何畫像文字中，會讓「正確偏好主張率」虛高。
DISCRIMINATIVE_AUDIO_TERMS = {
    "rhythm", "beat", "beats", "tempo", "bpm", "upbeat", "slow tempo",
    "fast tempo", "fast-paced", "mid-tempo", "energetic", "calm",
    "melody", "melodic", "harmony", "harmonic", "bass", "drum", "drums",
    "guitar", "piano", "synth", "synthesizer", "vocal", "vocals",
    "instrumental", "chorus", "groove", "acoustic", "loud", "soft",
    "danceable", "male vocals", "female vocals",
}


# =============================================================================
# Logger
# =============================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("path_level_generation")
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
# 載入
# =============================================================================

def read_csv(path: Path) -> list:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


NEGATIVE_CUES = re.compile(
    r"\b(dislike|dislikes|avoid|avoids|hate|hates|reject|rejects|not prefer|"
    r"does not prefer|doesn't prefer|less interested|averse|not a fan|"
    r"does not enjoy|doesn't enjoy|not enjoy|steer clear|not fond)\b", re.I)


def load_profiles(logger) -> dict:
    """video_id → 帶正負極性的偏好畫像證據。"""
    if not PROFILES_JSONL.exists():
        raise FileNotFoundError(
            f"找不到 Stage 4 偏好畫像：{PROFILES_JSONL}\n"
            "偏好主張驗證需要此檔；若暫時無法存取，請先確認 E: 磁碟掛載。"
        )
    out = {}
    with open(PROFILES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            positive, negative = [], []
            summary = str(d.get("summary_text", ""))
            for sentence in re.split(r"(?<=[.!?])\s+", summary):
                (negative if NEGATIVE_CUES.search(sentence) else positive).append(sentence)
            for item in d.get("salient_facts", []) or []:
                fact = str(item.get("fact", ""))
                tag = str(item.get("conflict_tag", "")).upper()
                is_negative = tag == "CONFIRM_DISLIKE" or bool(NEGATIVE_CUES.search(fact))
                (negative if is_negative else positive).append(fact)
            out[d["music_id"]] = {
                "positive": " ".join(positive).lower(),
                "negative": " ".join(negative).lower(),
                "all": (" ".join(positive + negative)).lower(),
            }
    logger.info("[Profiles] 載入 %d 筆偏好畫像", len(out))
    return out


# =============================================================================
# 屬性抽取與驗證
# =============================================================================

def extract_attribute_terms(text: str) -> list:
    """從主張中抽出可辨識的偏好屬性詞（曲風 + 具鑑別力的音訊屬性）。"""
    lower = text.lower()
    hits = set(B2.extract_genre_terms(text))
    for term in DISCRIMINATIVE_AUDIO_TERMS:
        if term in lower:
            hits.add(term)
    return sorted(hits)


def verify_against_evidence(terms: list, evidence: str) -> str:
    """回傳 supported / partially_supported / not_found / uncheckable。"""
    if not terms:
        return "uncheckable"
    hit = [t for t in terms if t in evidence]
    if len(hit) == len(terms):
        return "supported"
    return "partially_supported" if hit else "not_found"


def verify_preference_claim(claim: str, terms: list, evidence: dict) -> str:
    """極性敏感的參考畫像一致性：supported/partial/contradicted/not_found。"""
    if not terms:
        return "uncheckable"
    claim_negative = bool(NEGATIVE_CUES.search(claim))
    desired = evidence["negative" if claim_negative else "positive"]
    opposite = evidence["positive" if claim_negative else "negative"]
    hit = [t for t in terms if t in desired]
    opposite_hit = [t for t in terms if t in opposite]
    if len(hit) == len(terms):
        return "supported"
    if hit:
        return "partially_supported"
    if opposite_hit:
        return "contradicted"
    return "not_found"


def build_metadata_evidence(row: dict, music_meta) -> str:
    pipeline, external, _ = B2.build_evidence(row, music_meta)
    return pipeline + " " + " ".join(sorted(external))


# =============================================================================
# 逐模型標註
# =============================================================================

COUNT_FIELDS = [
    "n_claims", "n_unsup_L1", "n_unsup_L2",
    "n_pref", "n_pref_supported", "n_pref_partial", "n_pref_contradicted",
    "n_pref_notfound", "n_pref_uncheckable",
    "n_meta", "n_meta_supported", "n_meta_partial", "n_meta_notfound", "n_meta_uncheckable",
    "n_audio", "n_video", "n_prompt", "n_general",
    "has_pref_claim", "is_degenerate", "n_words",
]


def annotate_experiment(exp, rows, profiles, music_meta, logger):
    """回傳 (per_sample_counts: np.ndarray[n_samples, len(COUNT_FIELDS)], claim_records)."""
    counts = np.zeros((len(rows), len(COUNT_FIELDS)), dtype=np.float64)
    fi = {name: i for i, name in enumerate(COUNT_FIELDS)}
    claim_records = []

    for si, row in enumerate(rows):
        generated = row.get("generated_text", "") or ""
        degen, _reason = B2.is_degenerate_generation(generated)
        counts[si, fi["is_degenerate"]] = int(degen)
        counts[si, fi["n_words"]] = len(generated.split())

        pref_evidence = profiles.get(row.get("video_id", ""),
                                     {"positive": "", "negative": "", "all": ""})
        meta_evidence = build_metadata_evidence(row, music_meta)

        claims = J.split_claims(generated)
        for cid, claim in enumerate(claims, start=1):
            source, ctype = J.classify_claim(claim, row)
            counts[si, fi["n_claims"]] += 1

            # UCR：L1 子句層級；L2 以母句重判（與 B2 一致）
            unsup_l1 = int(source == J.SOURCE_UNSUPPORTED)
            counts[si, fi["n_unsup_L1"]] += unsup_l1
            parent_source = source
            if unsup_l1:
                parent = B2.find_parent_sentence(claim, generated)
                parent_source, _ = J.classify_claim(parent, row)
                counts[si, fi["n_unsup_L2"]] += int(parent_source == J.SOURCE_UNSUPPORTED)

            verification = ""
            terms = []
            if ctype == "preference":
                counts[si, fi["n_pref"]] += 1
                terms = extract_attribute_terms(claim)
                verification = verify_preference_claim(claim, terms, pref_evidence)
                counts[si, fi[{
                    "supported": "n_pref_supported", "partially_supported": "n_pref_partial",
                    "contradicted": "n_pref_contradicted",
                    "not_found": "n_pref_notfound", "uncheckable": "n_pref_uncheckable",
                }[verification]]] += 1
            elif ctype == "metadata":
                counts[si, fi["n_meta"]] += 1
                terms = extract_attribute_terms(claim)
                verification = verify_against_evidence(terms, meta_evidence)
                counts[si, fi[{
                    "supported": "n_meta_supported", "partially_supported": "n_meta_partial",
                    "not_found": "n_meta_notfound", "uncheckable": "n_meta_uncheckable",
                }[verification]]] += 1
            elif ctype in ("audio", "video", "prompt", "general"):
                counts[si, fi[f"n_{ctype}"]] += 1

            claim_records.append({
                "exp": exp, "sample_idx": row.get("sample_idx", ""),
                "video_id": row.get("video_id", ""), "claim_id": cid,
                "claim_text": claim, "claim_type": ctype, "support_source": source,
                "parent_source": parent_source,
                "attribute_terms": ";".join(terms), "verification": verification,
                "reference_positive_evidence": (
                    pref_evidence.get("positive", "") if ctype == "preference" else ""
                ),
                "reference_negative_evidence": (
                    pref_evidence.get("negative", "") if ctype == "preference" else ""
                ),
            })

        counts[si, fi["has_pref_claim"]] = int(counts[si, fi["n_pref"]] > 0)

    logger.info("[%s] 樣本=%d claim=%d 偏好主張=%d 元資料主張=%d 退化生成=%d",
                exp, len(rows), int(counts[:, fi["n_claims"]].sum()),
                int(counts[:, fi["n_pref"]].sum()), int(counts[:, fi["n_meta"]].sum()),
                int(counts[:, fi["is_degenerate"]].sum()))
    return counts, claim_records


# =============================================================================
# 指標
# =============================================================================

def compute_metrics(counts: np.ndarray) -> dict:
    return metrics_from_sums(counts.sum(axis=0), counts.shape[0])


def metrics_from_sums(s: np.ndarray, n_samples: int) -> dict:
    """由「各欄位總和」直接算出全部指標；拔靴時每次重抽只需算一次總和。"""
    fi = {name: i for i, name in enumerate(COUNT_FIELDS)}

    def div(a, b):
        return float(a / b) if b else float("nan")

    n_claims = s[fi["n_claims"]]
    n_pref = s[fi["n_pref"]]
    n_pref_checkable = n_pref - s[fi["n_pref_uncheckable"]]
    n_meta = s[fi["n_meta"]]
    n_meta_checkable = n_meta - s[fi["n_meta_uncheckable"]]

    return {
        "n_samples": int(n_samples),
        "n_claims": int(n_claims),
        "claims_per_generation": div(n_claims, n_samples),
        "mean_words": div(s[fi["n_words"]], n_samples),
        "degenerate_rate": div(s[fi["is_degenerate"]], n_samples),
        # 1. 偏好屬性引用率
        "preference_claim_ratio": div(n_pref, n_claims),
        "preference_generation_coverage": div(s[fi["has_pref_claim"]], n_samples),
        # 2/3. 與事後參考畫像的一致／矛盾／不存在率（分母為可驗證偏好主張）
        "preference_reference_alignment_rate": div(
            s[fi["n_pref_supported"]], n_pref_checkable),
        "preference_partial_rate": div(s[fi["n_pref_partial"]], n_pref_checkable),
        "preference_reference_contradiction_rate": div(
            s[fi["n_pref_contradicted"]], n_pref_checkable),
        "preference_nonexistent_rate": div(s[fi["n_pref_notfound"]], n_pref_checkable),
        "preference_uncheckable_ratio": div(s[fi["n_pref_uncheckable"]], n_pref),
        # 4. 音樂元資料支持率
        "metadata_claim_ratio": div(n_meta, n_claims),
        "metadata_support_rate": div(s[fi["n_meta_supported"]], n_meta_checkable),
        "metadata_notfound_rate": div(s[fi["n_meta_notfound"]], n_meta_checkable),
        # 5. UCR
        "UCR_L1_clause": div(s[fi["n_unsup_L1"]], n_claims),
        "UCR_L2_sentence": div(s[fi["n_unsup_L2"]], n_claims),
        # 其他 claim 組成
        "audio_claim_ratio": div(s[fi["n_audio"]], n_claims),
        "video_claim_ratio": div(s[fi["n_video"]], n_claims),
        "prompt_claim_ratio": div(s[fi["n_prompt"]], n_claims),
        "general_claim_ratio": div(s[fi["n_general"]], n_claims),
    }


CONTRAST_METRICS = [
    "preference_claim_ratio", "preference_generation_coverage",
    "preference_reference_alignment_rate", "preference_reference_contradiction_rate",
    "preference_nonexistent_rate",
    "metadata_support_rate", "UCR_L1_clause", "UCR_L2_sentence",
    "audio_claim_ratio", "claims_per_generation",
]


def clustered_bootstrap_contrast(counts_a, counts_b, metrics, n_boot, seed, alpha):
    """
    樣本層級叢集拔靴：claim 巢套於樣本內，故重抽的單位是樣本。
    兩個模型在同一批樣本上評估，因此使用**同一組重抽索引**（配對）。

    每次重抽只算一次欄位總和，再由總和導出所有指標；
    否則每個指標各跑一輪拔靴會重複約 10^5 次陣列縮減。
    回傳 {metric: (point, ci_low, ci_high)}。
    """
    base_a, base_b = compute_metrics(counts_a), compute_metrics(counts_b)
    points = {m: base_a[m] - base_b[m] for m in metrics}

    n = counts_a.shape[0]
    rng = np.random.default_rng(seed)
    draws = {m: np.empty(n_boot, dtype=np.float64) for m in metrics}
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)
        ma = metrics_from_sums(counts_a[sel].sum(axis=0), n)
        mb = metrics_from_sums(counts_b[sel].sum(axis=0), n)
        for m in metrics:
            draws[m][b] = ma[m] - mb[m]

    out = {}
    for m in metrics:
        d = draws[m][~np.isnan(draws[m])]
        if d.size == 0:
            out[m] = (float(points[m]), float("nan"), float("nan"))
        else:
            out[m] = (float(points[m]),
                      float(np.percentile(d, 100 * alpha / 2)),
                      float(np.percentile(d, 100 * (1 - alpha / 2))))
    return out


# =============================================================================
# 主流程
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUT_DIR / "path_level_generation.log")
    started = _dt.datetime.now()

    logger.info("=" * 78)
    logger.info("B3 路徑級生成分析 | 模型：%s", "、".join(EXPERIMENTS))
    logger.info("=" * 78)

    profiles = load_profiles(logger)
    music_meta = B2.load_music_metadata(logger)

    all_counts, all_metrics, all_claims = {}, {}, []
    ref_keys = None
    for exp, label in EXPERIMENTS.items():
        path = RESULTS_DIR / GENERATION_CSV_TMPL.format(exp=exp)
        if not path.exists():
            raise FileNotFoundError(f"找不到 {exp} 的 Top-1 生成結果：{path}")
        rows = read_csv(path)

        keys = [(r.get("sample_idx"), r.get("video_id")) for r in rows]
        if ref_keys is None:
            ref_keys = keys
        elif keys != ref_keys:
            raise ValueError(f"{exp} 與 {list(EXPERIMENTS)[0]} 的樣本順序不一致，無法配對比較")

        counts, claims = annotate_experiment(exp, rows, profiles, music_meta, logger)
        all_counts[exp] = counts
        all_metrics[exp] = compute_metrics(counts)
        all_claims.extend(claims)

    logger.info("-" * 78)
    for exp, label in EXPERIMENTS.items():
        m = all_metrics[exp]
        logger.info("[%s %-18s] 偏好引用=%.4f 參考畫像一致=%.4f 不存在偏好=%.4f "
                    "元資料支持=%.4f UCR(L1)=%.4f",
                    exp, label, m["preference_claim_ratio"],
                    m["preference_reference_alignment_rate"],
                    m["preference_nonexistent_rate"], m["metadata_support_rate"],
                    m["UCR_L1_clause"])

    # ---- 兩兩對照 -----------------------------------------------------------
    logger.info("-" * 78)
    contrasts = []
    for exp_a, exp_b in CONTRASTS:
        res = clustered_bootstrap_contrast(
            all_counts[exp_a], all_counts[exp_b], CONTRAST_METRICS,
            BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)
        for metric in CONTRAST_METRICS:
            pt, lo, hi = res[metric]
            contrasts.append({
                "comparison": f"{exp_a} - {exp_b}",
                "label": f"{EXPERIMENTS[exp_a]} − {EXPERIMENTS[exp_b]}",
                "metric": metric, "difference": pt,
                "ci_low": lo, "ci_high": hi,
                "significant": bool(lo > 0 or hi < 0),
            })
        pt, lo, hi = res["preference_reference_alignment_rate"]
        logger.info("[對照] %-17s 參考畫像一致率 Δ=%+.4f [%+.4f, %+.4f] %s",
                    f"{exp_a} - {exp_b}", pt, lo, hi,
                    "顯著" if (lo > 0 or hi < 0) else "不顯著")

    # ---- 輸出 ---------------------------------------------------------------
    metrics_rows = [{
        "exp": e,
        "model": EXPERIMENTS[e],
        "preference_input_support": (
            "not_applicable_no_ltp" if e == "exp_04" else "not_observable_from_vector_input"
        ),
        **all_metrics[e],
    } for e in EXPERIMENTS]
    with open(OUT_DIR / "path_level_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader()
        w.writerows(metrics_rows)

    with open(OUT_DIR / "path_level_contrasts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0].keys()))
        w.writeheader()
        w.writerows(contrasts)

    rng = random.Random(CLAIM_SAMPLE_SEED)
    sampled = []
    for exp in EXPERIMENTS:
        pool = [c for c in all_claims if c["exp"] == exp and c["claim_type"] in ("preference", "metadata")]
        sampled.extend(rng.sample(pool, min(CLAIM_SAMPLE_PER_EXP, len(pool))))
    with open(OUT_DIR / "path_level_claims_sample.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sampled[0].keys()))
        w.writeheader()
        w.writerows(sampled)

    preference_claims = [c for c in all_claims if c["claim_type"] == "preference"]
    with open(OUT_DIR / "preference_claims_full_audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preference_claims[0].keys()))
        w.writeheader()
        w.writerows(preference_claims)

    summary = {
        "generated_at": started.isoformat(timespec="seconds"),
        "experiments": EXPERIMENTS,
        "ranking_reference_R@1": {"exp_01": 0.3065, "exp_02": 0.2193,
                                  "exp_03": 0.3106, "exp_04": 0.1907},
        "metrics": all_metrics,
        "contrasts": contrasts,
        "method": {
            "claim_judge": "沿用 faithfulness_claim_judge_v2.classify_claim（未修改）",
            "UCR_L2": "沿用 B2 的母句還原校正",
            "preference_evidence": "Stage 4 profiles.jsonl，依 conflict_tag 與否定詞分為"
                                   "正向／排斥證據後做極性敏感比對；依 video_id 對應，"
                                   "測試集 100% 覆蓋",
            "metadata_evidence": "top1_reference_text + title/artist + musicnn 標籤",
            "statistics": f"樣本層級叢集拔靴 {BOOTSTRAP_N} 次、配對重抽索引、"
                          f"{int((1-CI_ALPHA)*100)}% 百分位 CI",
            "caveat": "四模型皆與同一份事後參考畫像比較。exp_04 未接收該畫像，"
                      "其一致率是偶然／先驗一致的負向對照，不是輸入支持率；"
                      "因此本表統一稱參考畫像一致率。LTP 是不透明向量，無法由文字規則"
                      "判定逐條主張是否受向量輸入支持；此欄對 exp_01–03 標為不可觀察，"
                      "對 exp_04 標為不適用。",
        },
    }
    with open(OUT_DIR / "path_level_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown(OUT_DIR / "path_level_summary.md", summary)

    logger.info("已輸出至 %s", OUT_DIR)
    logger.info("完成，耗時 %.1f 秒", (_dt.datetime.now() - started).total_seconds())


def write_markdown(path, summary):
    m = summary["metrics"]
    exps = list(summary["experiments"])
    L = ["# 路徑級生成分析（B3）\n",
         f"- 產生時間：{summary['generated_at']}",
         "- 四個模型的 4,205 筆 Top-1 說明文字皆取自既有結果，未重跑生成\n",
         "## 一、排序 vs 說明：教授指定的對照表\n",
         "| 指標 | " + " | ".join(summary["experiments"][e] for e in exps) + " |",
         "|---" * (len(exps) + 1) + "|",
         "| **排序 R@1**（既有） | " + " | ".join(
             f"{summary['ranking_reference_R@1'][e]*100:.2f}%" for e in exps) + " |"]

    rows = [
        ("偏好屬性引用率", "preference_claim_ratio", "pct"),
        ("　樣本涵蓋率（≥1 條偏好主張）", "preference_generation_coverage", "pct"),
        ("偏好主張與參考畫像一致率", "preference_reference_alignment_rate", "pct"),
        ("偏好主張與參考畫像矛盾率", "preference_reference_contradiction_rate", "pct"),
        ("不存在偏好主張率", "preference_nonexistent_rate", "pct"),
        ("音樂元資料支持率", "metadata_support_rate", "pct"),
        ("UCR（子句層級 L1）", "UCR_L1_clause", "pct"),
        ("UCR（母句校正 L2）", "UCR_L2_sentence", "pct"),
        ("音訊屬性主張比例", "audio_claim_ratio", "pct"),
        ("每則說明的 claim 數", "claims_per_generation", "num"),
        ("平均字數", "mean_words", "num"),
    ]
    for label, key, kind in rows:
        vals = []
        for e in exps:
            v = m[e][key]
            vals.append(f"{v*100:.2f}%" if kind == "pct" else f"{v:.2f}")
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    L.append("\n## 二、關鍵對照（樣本層級叢集拔靴 95% CI）\n")
    L.append("| 對照 | 指標 | 差異 | 95% CI | 顯著 |")
    L.append("|---|---|---|---|---|")
    for c in summary["contrasts"]:
        if c["metric"] not in ("preference_claim_ratio",
                               "preference_reference_alignment_rate",
                               "preference_reference_contradiction_rate",
                               "preference_nonexistent_rate", "metadata_support_rate",
                               "UCR_L1_clause"):
            continue
        L.append(f"| {c['comparison']} | {c['metric']} | {c['difference']*100:+.2f}pp | "
                 f"[{c['ci_low']*100:+.2f}, {c['ci_high']*100:+.2f}] | "
                 f"{'是' if c['significant'] else '否'} |")

    L.append("\n## 三、方法與限制\n")
    for k, v in summary["method"].items():
        L.append(f"- **{k}**：{v}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
