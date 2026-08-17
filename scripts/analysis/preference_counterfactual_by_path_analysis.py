# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
preference_counterfactual_by_path_analysis.py
=============================================
B3 第二部分：偏好反事實的**方向敏感度**逐路徑比較

對應教授建議 §四所列的最後一項生成側指標：
  「偏好反事實改變後，說明方向是否同步改變」

資料來源（四個模型各 200 樣本 × 3 變體 = 600 列，0 fallback）：
  exp_01  results/faithfulness/preference_counterfactual_generations_top1.csv（2026-06-04 既有）
  exp_02  ┐
  exp_03  ├ results/faithfulness/preference_counterfactual_by_path/{exp}_..._top1.csv（2026-07-26 補跑）
  exp_04  ┘

三個變體：
  original              使用原始對話文字 t3
  cf_upbeat_electronic  換成「輕快電子、明亮、節奏清楚、有活力、現代舞曲流行」
  cf_lyrical_piano      換成「抒情鋼琴、慢速、柔和原音、旋律溫柔、平靜感傷」

⚠ 兩個必須寫進論文的範圍限制（取自資料本身的 scope_note）：
  1. `top1_generation_prompt_text_changed_only_precomputed_text_feature_unchanged`
     —— 只換了 prompt 的文字，[TEXT_CLIP] 預算特徵與 [LTP] 向量都沒有重算。
     因此本分析測的是「**模型對提示詞中偏好敘述的反應**」，
     **不是**對 LTP 向量本身的反應；不可宣稱為 LTP 路徑的因果證據。
  2. top-1 曲目沿用既有排序結果，三個變體完全相同。
     這是設計上的優點：曲目沒變，說明卻改變的部分即可歸因於提示詞。

四項指標：
  1. 方向命中率 Direction Accuracy
     反事實目標詞彙數是否上升（相對 original）。
  2. 淨方向命中率 Net Direction Accuracy
     目標詞彙上升 **且** 相反方向詞彙未上升，較嚴格。
  3. 提示詞複誦率 Prompt Echo Rate
     生成文中出現的目標詞彙，有多少比例是提示詞裡原本就有的字詞。
     此值偏高代表「方向改變」只是把提示詞抄回來，而非真的改變對音樂的描述。
  4. 曲名漂移率 Title Drift Rate
     三個變體的 top-1 曲目**完全相同**，因此模型提到的曲名理應不變。
     以既有欄位 generated_mentions_top1_title 判定：original 提到正確曲名、
     但反事實條件下不再提到者，即為漂移（無關屬性受偏好提示詞污染）。

統計：樣本層級配對拔靴（同一批 200 個 sample_idx），2000 次。

輸出（results/analysis/preference_counterfactual_by_path/）：
  cf_direction_by_model.csv      四模型 × 兩個反事實方向的全部指標
  cf_direction_contrasts.csv     模型兩兩差異 + 95% CI
  cf_direction_samples.csv       逐樣本判定結果（供人工查核）
  cf_direction_summary.json / .md
  cf_direction.log

使用方式：
  python scripts/analysis/preference_counterfactual_by_path_analysis.py
"""

import csv
import datetime as _dt
import json
import logging
import re
from collections import defaultdict

import numpy as np


RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR     = RESULTS_DIR / "analysis" / "preference_counterfactual_by_path"

CSV_PATHS = {
    "exp_01": RESULTS_DIR / "faithfulness" / "preference_counterfactual_generations_top1.csv",
    "exp_02": RESULTS_DIR / "faithfulness" / "preference_counterfactual_by_path"
              / "exp_02_preference_counterfactual_generations_top1.csv",
    "exp_03": RESULTS_DIR / "faithfulness" / "preference_counterfactual_by_path"
              / "exp_03_preference_counterfactual_generations_top1.csv",
    "exp_04": RESULTS_DIR / "faithfulness" / "preference_counterfactual_by_path"
              / "exp_04_preference_counterfactual_generations_top1.csv",
}

MODEL_LABELS = {
    "exp_01": "Hybrid LTP", "exp_02": "Explicit-only LTP",
    "exp_03": "Implicit-only LTP", "exp_04": "No-LTP",
}

BASELINE_VARIANT = "original"
CF_VARIANTS = ["cf_upbeat_electronic", "cf_lyrical_piano"]
OPPOSITE = {"cf_upbeat_electronic": "cf_lyrical_piano",
            "cf_lyrical_piano": "cf_upbeat_electronic"}

# 各反事實方向的目標詞彙（取自該變體提示詞的語義內容並適度擴充同義詞）
CF_TERMS = {
    "cf_upbeat_electronic": {
        "upbeat", "electronic", "electro", "bright", "rhythm", "rhythmic",
        "energetic", "energy", "dance", "danceable", "modern", "beat", "beats",
        "fast", "fast-paced", "uptempo", "techno", "house", "edm", "club",
        "lively", "driving", "punchy", "groove", "pop",
    },
    "cf_lyrical_piano": {
        "piano", "lyrical", "slow", "slow-paced", "soft", "acoustic", "gentle",
        "melody", "melodic", "calm", "sentimental", "mellow", "tender",
        "ballad", "intimate", "delicate", "quiet", "reflective", "soothing",
        "emotional", "downtempo",
    },
}

BOOTSTRAP_N    = 2000
BOOTSTRAP_SEED = 20260726
CI_ALPHA       = 0.05

CONTRASTS = [("exp_02", "exp_04"), ("exp_01", "exp_04"), ("exp_03", "exp_04"),
             ("exp_02", "exp_03"), ("exp_01", "exp_03"), ("exp_01", "exp_02")]

METRICS = ["direction_accuracy", "net_direction_accuracy",
           "prompt_echo_rate", "title_drift_rate", "text_change_rate"]

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_by_path")
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


def tokens(text: str) -> list:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def term_count(text: str, vocab: set) -> int:
    return sum(1 for w in tokens(text) if w in vocab)


def analyse_model(exp: str, rows: list, logger) -> tuple:
    """回傳 (per_sample_flags: dict[cf_variant] -> np.ndarray[n, len(METRICS)], sample_records)."""
    by_sample = defaultdict(dict)
    for r in rows:
        by_sample[r["sample_idx"]][r["preference_variant"]] = r

    sample_ids = sorted(by_sample, key=lambda x: int(x))
    flags = {cf: np.zeros((len(sample_ids), len(METRICS)), dtype=np.float64) for cf in CF_VARIANTS}
    records = []

    for i, sid in enumerate(sample_ids):
        base = by_sample[sid].get(BASELINE_VARIANT)
        if base is None:
            continue
        base_text = base.get("generated_text", "")
        base_title_ok = str(base.get("generated_mentions_top1_title", "0")) in ("1", "True")

        for cf in CF_VARIANTS:
            row = by_sample[sid].get(cf)
            if row is None:
                continue
            cf_text = row.get("generated_text", "")
            prompt_words = set(tokens(row.get("user_text", "")))

            tgt_vocab, opp_vocab = CF_TERMS[cf], CF_TERMS[OPPOSITE[cf]]
            d_tgt = term_count(cf_text, tgt_vocab) - term_count(base_text, tgt_vocab)
            d_opp = term_count(cf_text, opp_vocab) - term_count(base_text, opp_vocab)

            direction = int(d_tgt > 0)
            net_direction = int(d_tgt > 0 and d_opp <= 0)

            # 提示詞複誦：生成文中的目標詞，有多少是提示詞裡本來就有的
            cf_tgt_words = [w for w in tokens(cf_text) if w in tgt_vocab]
            echoed = [w for w in cf_tgt_words if w in prompt_words]
            echo_rate = (len(echoed) / len(cf_tgt_words)) if cf_tgt_words else np.nan

            cf_title_ok = str(row.get("generated_mentions_top1_title", "0")) in ("1", "True")
            title_drift = int(base_title_ok and not cf_title_ok)

            text_changed = int(" ".join(tokens(cf_text)) != " ".join(tokens(base_text)))

            flags[cf][i] = [direction, net_direction,
                            0.0 if np.isnan(echo_rate) else echo_rate,
                            title_drift, text_changed]
            if np.isnan(echo_rate):
                flags[cf][i, METRICS.index("prompt_echo_rate")] = np.nan

            records.append({
                "exp": exp, "sample_idx": sid, "cf_variant": cf,
                "delta_target_terms": d_tgt, "delta_opposite_terms": d_opp,
                "direction_accuracy": direction, "net_direction_accuracy": net_direction,
                "prompt_echo_rate": "" if np.isnan(echo_rate) else round(echo_rate, 4),
                "base_mentions_top1_title": int(base_title_ok),
                "cf_mentions_top1_title": int(cf_title_ok),
                "title_drift": title_drift, "text_changed": text_changed,
                "top1_music_id": row.get("top1_music_id", ""),
            })

    logger.info("[%s] 樣本=%d 變體=%d", exp, len(sample_ids), len(CF_VARIANTS))
    return flags, records


def metrics_from_flags(arr: np.ndarray) -> dict:
    return {m: float(np.nanmean(arr[:, i])) for i, m in enumerate(METRICS)}


def paired_bootstrap(arr_a: np.ndarray, arr_b: np.ndarray, n_boot, seed, alpha) -> dict:
    base_a, base_b = metrics_from_flags(arr_a), metrics_from_flags(arr_b)
    points = {m: base_a[m] - base_b[m] for m in METRICS}
    n = arr_a.shape[0]
    rng = np.random.default_rng(seed)
    draws = {m: np.empty(n_boot) for m in METRICS}
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)
        ma, mb = metrics_from_flags(arr_a[sel]), metrics_from_flags(arr_b[sel])
        for m in METRICS:
            draws[m][b] = ma[m] - mb[m]
    out = {}
    for m in METRICS:
        d = draws[m][~np.isnan(draws[m])]
        out[m] = (float(points[m]),
                  float(np.percentile(d, 100 * alpha / 2)),
                  float(np.percentile(d, 100 * (1 - alpha / 2)))) if d.size else \
                 (float(points[m]), float("nan"), float("nan"))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUT_DIR / "cf_direction.log")
    started = _dt.datetime.now()

    logger.info("=" * 78)
    logger.info("B3-2 偏好反事實方向敏感度（逐路徑）")
    logger.info("=" * 78)

    all_flags, all_records = {}, []
    for exp, path in CSV_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到 {exp} 的偏好反事實結果：{path}")
        rows = read_csv(path)
        flags, records = analyse_model(exp, rows, logger)
        all_flags[exp] = flags
        all_records.extend(records)

    # ---- 逐模型指標 ---------------------------------------------------------
    rows_out = []
    logger.info("-" * 78)
    for exp in CSV_PATHS:
        pooled = np.vstack([all_flags[exp][cf] for cf in CF_VARIANTS])
        for scope, arr in [("pooled", pooled)] + [(cf, all_flags[exp][cf]) for cf in CF_VARIANTS]:
            m = metrics_from_flags(arr)
            rows_out.append({"exp": exp, "model": MODEL_LABELS[exp], "cf_variant": scope,
                             "n": int(arr.shape[0]), **m})
        mp = metrics_from_flags(pooled)
        logger.info("[%s %-18s] 方向命中=%.4f 淨方向=%.4f 複誦率=%.4f 曲名漂移=%.4f 文字改變=%.4f",
                    exp, MODEL_LABELS[exp], mp["direction_accuracy"],
                    mp["net_direction_accuracy"], mp["prompt_echo_rate"],
                    mp["title_drift_rate"], mp["text_change_rate"])

    # ---- 兩兩對照 -----------------------------------------------------------
    logger.info("-" * 78)
    contrasts = []
    for a, b in CONTRASTS:
        pa = np.vstack([all_flags[a][cf] for cf in CF_VARIANTS])
        pb = np.vstack([all_flags[b][cf] for cf in CF_VARIANTS])
        res = paired_bootstrap(pa, pb, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)
        for m in METRICS:
            pt, lo, hi = res[m]
            contrasts.append({"comparison": f"{a} - {b}",
                              "label": f"{MODEL_LABELS[a]} − {MODEL_LABELS[b]}",
                              "metric": m, "difference": pt, "ci_low": lo, "ci_high": hi,
                              "significant": bool(lo > 0 or hi < 0)})
        pt, lo, hi = res["direction_accuracy"]
        logger.info("[對照] %-17s 方向命中率 Δ=%+.4f [%+.4f, %+.4f] %s",
                    f"{a} - {b}", pt, lo, hi, "顯著" if (lo > 0 or hi < 0) else "不顯著")

    # ---- 輸出 ---------------------------------------------------------------
    with open(OUT_DIR / "cf_direction_by_model.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    with open(OUT_DIR / "cf_direction_contrasts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0].keys()))
        w.writeheader()
        w.writerows(contrasts)
    with open(OUT_DIR / "cf_direction_samples.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        w.writeheader()
        w.writerows(all_records)

    summary = {
        "generated_at": started.isoformat(timespec="seconds"),
        "models": MODEL_LABELS,
        "by_model": rows_out,
        "contrasts": contrasts,
        "scope_limits": [
            "反事實只更換 prompt 文字，[TEXT_CLIP] 預算特徵與 [LTP] 向量均未重算"
            "（資料 scope_note 已註明），故本分析測的是對提示詞偏好敘述的反應，"
            "不是對 LTP 向量的反應，不可作為 LTP 路徑的因果證據。",
            "三個變體的 top-1 曲目完全相同，因此曲名漂移可直接歸因於提示詞。",
        ],
        "method": {
            "direction": "反事實目標詞彙數相對 original 上升即命中",
            "net_direction": "目標詞彙上升且相反方向詞彙未上升",
            "prompt_echo": "生成文中的目標詞有多少比例本來就出現在提示詞裡",
            "title_drift": "original 提到正確 top-1 曲名、反事實條件下不再提到",
            "statistics": f"樣本層級配對拔靴 {BOOTSTRAP_N} 次、{int((1-CI_ALPHA)*100)}% 百分位 CI",
        },
    }
    with open(OUT_DIR / "cf_direction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown(OUT_DIR / "cf_direction_summary.md", summary)
    logger.info("已輸出至 %s", OUT_DIR)
    logger.info("完成，耗時 %.1f 秒", (_dt.datetime.now() - started).total_seconds())


def write_markdown(path, summary):
    L = ["# 偏好反事實方向敏感度（B3-2）\n",
         f"- 產生時間：{summary['generated_at']}",
         "- 四個模型各 200 樣本 × 3 變體（original / cf_upbeat_electronic / cf_lyrical_piano）\n",
         "## 一、逐模型結果（兩個反事實方向合併）\n",
         "| 模型 | n | 方向命中率 | 淨方向命中率 | 提示詞複誦率 | 曲名漂移率 | 文字改變率 |",
         "|---|---|---|---|---|---|---|"]
    for r in summary["by_model"]:
        if r["cf_variant"] != "pooled":
            continue
        L.append(f"| {r['model']} | {r['n']} | {r['direction_accuracy']*100:.1f}% | "
                 f"{r['net_direction_accuracy']*100:.1f}% | {r['prompt_echo_rate']*100:.1f}% | "
                 f"{r['title_drift_rate']*100:.1f}% | {r['text_change_rate']*100:.1f}% |")

    L.append("\n## 二、分方向結果\n")
    L.append("| 模型 | 反事實方向 | 方向命中率 | 淨方向命中率 | 曲名漂移率 |")
    L.append("|---|---|---|---|---|")
    for r in summary["by_model"]:
        if r["cf_variant"] == "pooled":
            continue
        L.append(f"| {r['model']} | {r['cf_variant']} | {r['direction_accuracy']*100:.1f}% | "
                 f"{r['net_direction_accuracy']*100:.1f}% | {r['title_drift_rate']*100:.1f}% |")

    L.append("\n## 三、模型兩兩對照（配對拔靴 95% CI）\n")
    L.append("| 對照 | 指標 | 差異 | 95% CI | 顯著 |")
    L.append("|---|---|---|---|---|")
    for c in summary["contrasts"]:
        if c["metric"] not in ("direction_accuracy", "net_direction_accuracy", "title_drift_rate"):
            continue
        L.append(f"| {c['comparison']} | {c['metric']} | {c['difference']*100:+.2f}pp | "
                 f"[{c['ci_low']*100:+.2f}, {c['ci_high']*100:+.2f}] | "
                 f"{'是' if c['significant'] else '否'} |")

    L.append("\n## 四、範圍限制\n")
    for s in summary["scope_limits"]:
        L.append(f"- {s}")
    L.append("\n## 五、指標定義\n")
    for k, v in summary["method"].items():
        L.append(f"- **{k}**：{v}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
