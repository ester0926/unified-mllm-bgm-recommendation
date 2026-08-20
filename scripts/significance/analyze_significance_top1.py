"""
用途：計算主要評估結果的顯著性檢定與統計摘要。
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
import json
import math
import os
from statistics import NormalDist

import numpy as np


# =============================================================================
# 使用前可調整的設定
# =============================================================================

BASE_DIR = str(PROJECT_ROOT)
CHECKPOINT_NAME = "best"
POOL_SIZE = 500
PROMPT_VARIANT = "original"

# improvement 為正表示 FOCAL_EXP 優於 COMPARE_EXPS。
FOCAL_EXP = "exp_01"
COMPARE_EXPS = ["exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07"]

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260602
ALPHA = 0.05

# 若 scipy 可用，使用 scipy.stats.wilcoxon。
# 若 scipy 不可用，改用常態近似的符號等級檢定。
USE_SCIPY_IF_AVAILABLE = True

# ranking 與 generation 放在同一份輸出檔，
# 方便論文表格引用同一份 Top-1 end-to-end 顯著性分析。
INCLUDE_RANKING_METRICS = True
INCLUDE_GENERATION_METRICS = True


METRICS = [
    # 指標名稱、是否越高越好、論文顯示名稱, metric group
    ("R@1", True, "R@1", "ranking"),
    ("R@5", True, "R@5", "ranking"),
    ("R@10", True, "R@10", "ranking"),
    ("rank", False, "Rank", "ranking"),
    ("bertscore_f1", True, "BERTScore F1", "generation"),
    ("infolm_ab_divergence", False, "InfoLM AB", "generation"),
    ("infolm_l2_distance", False, "InfoLM L2", "generation"),
    ("infolm_fisher_rao", False, "InfoLM Fisher-Rao", "generation"),
]


METRIC_ALIASES = {
    "R@1": ["R@1", "r_at_1", "recall_at_1"],
    "R@5": ["R@5", "r_at_5", "recall_at_5"],
    "R@10": ["R@10", "r_at_10", "recall_at_10"],
    "rank": ["rank"],
    "bertscore_f1": ["bertscore_f1", "bert_f1", "BERTScore F1"],
    "infolm_ab_divergence": ["infolm_ab_divergence", "infolm_ab", "ab_divergence"],
    "infolm_l2_distance": ["infolm_l2_distance", "infolm_l2", "l2_distance"],
    "infolm_fisher_rao": [
        "infolm_fisher_rao",
        "infolm_fisher_rao_distance",
        "fisher_rao_distance",
    ],
}


def input_path(exp_name):
    return os.path.join(
        BASE_DIR,
        "checkpoints",
        exp_name,
        "detailed_eval",
        (
            f"{exp_name}_{CHECKPOINT_NAME}_{POOL_SIZE}pool_"
            f"top1_prompt_{PROMPT_VARIANT}_samples_merged.csv"
        ),
    )


def output_dir():
    path = os.path.join(BASE_DIR, "checkpoints", "significance_analysis_top1")
    os.makedirs(path, exist_ok=True)
    return path


def read_rows(exp_name):
    path = input_path(exp_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing Top-1 merged sample file for {exp_name}: {path}"
        )

    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_idx = int(row["sample_idx"])
            rows[sample_idx] = row
    return rows


def as_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def metric_value(row, metric):
    for key in METRIC_ALIASES.get(metric, [metric]):
        if key in row:
            value = as_float(row.get(key))
            if value is not None:
                return value
    return None


def paired_arrays(rows_a, rows_b, metric):
    common = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    a_vals, b_vals, idxs = [], [], []
    for idx in common:
        a = metric_value(rows_a[idx], metric)
        b = metric_value(rows_b[idx], metric)
        if a is None or b is None:
            continue
        a_vals.append(a)
        b_vals.append(b)
        idxs.append(idx)
    return np.asarray(a_vals, dtype=np.float64), np.asarray(b_vals, dtype=np.float64), idxs


def improvement_values(a_vals, b_vals, higher_is_better):
    if higher_is_better:
        return a_vals - b_vals
    return b_vals - a_vals


def bootstrap_ci_mean(diffs, n_boot=10000, seed=0, alpha=0.05, chunk=1000):
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return None, None

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        sample_idx = rng.integers(0, n, size=(k, n))
        boot_means[done:done + k] = diffs[sample_idx].mean(axis=1)
        done += k

    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def wilcoxon_scipy(diffs):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None

    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "scipy_zero_diff"}

    try:
        res = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        return {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "method": "scipy_wilcoxon",
        }
    except Exception:
        return None


def average_ranks(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def wilcoxon_normal_approx(diffs):
    nonzero = diffs[np.abs(diffs) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "normal_approx_zero_diff"}

    abs_d = np.abs(nonzero)
    signs = np.sign(nonzero)
    ranks = average_ranks(abs_d)
    w_plus = float(np.sum(ranks[signs > 0]))
    w_minus = float(np.sum(ranks[signs < 0]))
    statistic = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0

    _, counts = np.unique(abs_d, return_counts=True)
    tie_term = float(np.sum(counts * (counts + 1) * (2 * counts + 1)))
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return {"statistic": statistic, "p_value": 1.0, "method": "normal_approx_degenerate"}

    z = (w_plus - mean_w) / math.sqrt(var_w)
    p = 2.0 * (1.0 - NormalDist().cdf(abs(z)))
    return {
        "statistic": statistic,
        "p_value": float(max(min(p, 1.0), 0.0)),
        "method": "normal_approx_wilcoxon",
    }


def wilcoxon_test(diffs):
    if USE_SCIPY_IF_AVAILABLE:
        out = wilcoxon_scipy(diffs)
        if out is not None:
            return out
    return wilcoxon_normal_approx(diffs)


def holm_bonferroni(rows, alpha=0.05):
    indexed = [
        (i, row["p_value_raw"])
        for i, row in enumerate(rows)
        if row["p_value_raw"] is not None and not math.isnan(row["p_value_raw"])
    ]
    sorted_items = sorted(indexed, key=lambda x: x[1])
    m = len(sorted_items)

    adjusted = [None] * len(rows)
    significant = [False] * len(rows)
    running_max = 0.0

    for rank0, (original_i, p) in enumerate(sorted_items):
        adj = min((m - rank0) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[original_i] = running_max

    still_rejecting = True
    for rank0, (original_i, p) in enumerate(sorted_items):
        threshold = alpha / (m - rank0)
        if still_rejecting and p <= threshold:
            significant[original_i] = True
        else:
            still_rejecting = False
            significant[original_i] = False

    for i, row in enumerate(rows):
        row["p_value_holm"] = adjusted[i]
        row["significant_holm_0.05"] = significant[i]


def selected_metrics():
    out = []
    for metric in METRICS:
        group = metric[3]
        if group == "ranking" and not INCLUDE_RANKING_METRICS:
            continue
        if group == "generation" and not INCLUDE_GENERATION_METRICS:
            continue
        out.append(metric)
    return out


def analyze():
    focal_rows = read_rows(FOCAL_EXP)
    all_results = []

    for exp_name in COMPARE_EXPS:
        compare_rows = read_rows(exp_name)
        for metric, higher_is_better, label, metric_group in selected_metrics():
            a_vals, b_vals, sample_idxs = paired_arrays(focal_rows, compare_rows, metric)
            if len(a_vals) == 0:
                print(f"Skipped {exp_name} {metric}: no paired values")
                continue

            diffs = improvement_values(a_vals, b_vals, higher_is_better)
            ci_lo, ci_hi = bootstrap_ci_mean(
                diffs,
                n_boot=BOOTSTRAP_N,
                seed=BOOTSTRAP_SEED + len(all_results),
                alpha=ALPHA,
            )
            w = wilcoxon_test(diffs)
            mean_a = float(np.mean(a_vals))
            mean_b = float(np.mean(b_vals))
            mean_improvement = float(np.mean(diffs))

            all_results.append({
                "analysis_setting": "top1_end_to_end_generation",
                "prompt_variant": PROMPT_VARIANT,
                "focal_exp": FOCAL_EXP,
                "compare_exp": exp_name,
                "metric_group": metric_group,
                "metric": metric,
                "metric_label": label,
                "higher_is_better": higher_is_better,
                "n_pairs": int(len(diffs)),
                "focal_mean": mean_a,
                "compare_mean": mean_b,
                "mean_improvement_positive_is_focal_better": mean_improvement,
                "bootstrap_ci95_low": ci_lo,
                "bootstrap_ci95_high": ci_hi,
                "wilcoxon_statistic": w["statistic"],
                "p_value_raw": w["p_value"],
                "wilcoxon_method": w["method"],
            })

    holm_bonferroni(all_results, alpha=ALPHA)
    return all_results


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_float(x, digits=6):
    if x is None:
        return ""
    return f"{x:.{digits}g}"


def summarize_counts(rows):
    total = len(rows)
    sig = sum(1 for r in rows if r["significant_holm_0.05"])
    generation = [r for r in rows if r["metric_group"] == "generation"]
    ranking = [r for r in rows if r["metric_group"] == "ranking"]
    return {
        "total": total,
        "significant": sig,
        "ranking_total": len(ranking),
        "ranking_significant": sum(1 for r in ranking if r["significant_holm_0.05"]),
        "generation_total": len(generation),
        "generation_significant": sum(1 for r in generation if r["significant_holm_0.05"]),
    }


def write_markdown(path, rows):
    counts = summarize_counts(rows)
    lines = []
    lines.append("# Top-1 End-to-End Significance Analysis")
    lines.append("")
    lines.append(f"Setting: `Top-1 end-to-end generation`")
    lines.append(f"Prompt variant: `{PROMPT_VARIANT}`")
    lines.append(f"Focal experiment: `{FOCAL_EXP}`")
    lines.append(f"Comparisons: `{', '.join(COMPARE_EXPS)}`")
    lines.append(f"Bootstrap: {BOOTSTRAP_N:,} paired resamples, 95% CI")
    lines.append("Wilcoxon: signed-rank test on paired per-sample improvements")
    lines.append("Correction: Holm-Bonferroni across all tests in this file")
    lines.append("")
    lines.append(
        "This analysis uses explanations generated from the model-selected Top-1 "
        "music item, not GT-conditioned generation."
    )
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        f"- Total paired tests: {counts['total']}; Holm-significant: {counts['significant']}."
    )
    lines.append(
        f"- Ranking tests: {counts['ranking_total']}; Holm-significant: {counts['ranking_significant']}."
    )
    lines.append(
        f"- Generation tests: {counts['generation_total']}; Holm-significant: {counts['generation_significant']}."
    )
    lines.append("")
    lines.append("Positive improvement means the focal experiment is better.")
    lines.append("")
    lines.append("| Group | Compare | Metric | n | Focal mean | Compare mean | Improvement | 95% CI | p raw | p Holm | Sig. |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|---:|---:|---|")

    for row in rows:
        sig = "yes" if row["significant_holm_0.05"] else "no"
        ci = f"[{fmt_float(row['bootstrap_ci95_low'])}, {fmt_float(row['bootstrap_ci95_high'])}]"
        lines.append(
            "| {group} | {compare} | {metric} | {n} | {fm} | {cm} | {imp} | {ci} | {p} | {ph} | {sig} |".format(
                group=row["metric_group"],
                compare=row["compare_exp"],
                metric=row["metric_label"],
                n=row["n_pairs"],
                fm=fmt_float(row["focal_mean"]),
                cm=fmt_float(row["compare_mean"]),
                imp=fmt_float(row["mean_improvement_positive_is_focal_better"]),
                ci=ci,
                p=fmt_float(row["p_value_raw"], 4),
                ph=fmt_float(row["p_value_holm"], 4),
                sig=sig,
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- For `rank` and InfoLM metrics, lower raw values are better; the reported improvement is therefore `compare - focal`.")
    lines.append("- For Recall and BERTScore metrics, higher raw values are better; the reported improvement is `focal - compare`.")
    lines.append("- Ranking metrics are unchanged by the Top-1 generation rewrite, but they are included here for a single consistent thesis table.")
    lines.append("- Generation metrics are recomputed from Top-1 end-to-end generated explanations.")
    lines.append("- MuseChat cannot be included in this paired test unless per-sample MuseChat predictions are available.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    out_dir = output_dir()
    rows = analyze()

    csv_path = os.path.join(out_dir, "significance_top1_results.csv")
    json_path = os.path.join(out_dir, "significance_top1_results.json")
    md_path = os.path.join(out_dir, "significance_top1_summary.md")

    write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    write_markdown(md_path, rows)

    counts = summarize_counts(rows)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    print(
        "Completed {total} paired tests; Holm-significant at alpha={alpha}: {sig}".format(
            total=counts["total"],
            alpha=ALPHA,
            sig=counts["significant"],
        )
    )
    print(
        "Ranking significant: {}/{} | Generation significant: {}/{}".format(
            counts["ranking_significant"],
            counts["ranking_total"],
            counts["generation_significant"],
            counts["generation_total"],
        )
    )


if __name__ == "__main__":
    main()
