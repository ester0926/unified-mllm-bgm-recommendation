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

# improvement 為正表示 FOCAL_EXP 優於 COMPARE_EXPS。
FOCAL_EXP = "exp_01"
COMPARE_EXPS = ["exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07"]

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260512
ALPHA = 0.05

# 若 scipy 可用，使用 scipy.stats.wilcoxon。
# 若 scipy 不可用，改用常態近似的符號等級檢定。
USE_SCIPY_IF_AVAILABLE = True


METRICS = [
    # 指標名稱、是否越高越好、論文顯示名稱
    ("R@1", True, "R@1"),
    ("R@5", True, "R@5"),
    ("R@10", True, "R@10"),
    ("rank", False, "Rank"),
    ("bertscore_f1", True, "BERTScore F1"),
    ("infolm_ab_divergence", False, "InfoLM AB"),
    ("infolm_l2_distance", False, "InfoLM L2"),
    ("infolm_fisher_rao", False, "InfoLM Fisher-Rao"),
]


def input_path(exp_name):
    return os.path.join(
        BASE_DIR,
        "checkpoints",
        exp_name,
        "detailed_eval",
        f"{exp_name}_{CHECKPOINT_NAME}_{POOL_SIZE}pool_samples_merged.csv",
    )


def output_dir():
    path = os.path.join(BASE_DIR, "checkpoints", "significance_analysis")
    os.makedirs(path, exist_ok=True)
    return path


def read_rows(exp_name):
    path = input_path(exp_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

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


def paired_arrays(rows_a, rows_b, metric):
    common = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    a_vals, b_vals, idxs = [], [], []
    for idx in common:
        a = as_float(rows_a[idx].get(metric))
        b = as_float(rows_b[idx].get(metric))
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

    # 變異數的 tie correction。
    _, counts = np.unique(abs_d, return_counts=True)
    tie_term = float(np.sum(counts * (counts + 1) * (2 * counts + 1)))
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return {"statistic": statistic, "p_value": 1.0, "method": "normal_approx_degenerate"}

    z = (w_plus - mean_w) / math.sqrt(var_w)
    p = 2.0 * (1.0 - NormalDist().cdf(abs(z)))
    return {"statistic": statistic, "p_value": float(max(min(p, 1.0), 0.0)), "method": "normal_approx_wilcoxon"}


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
    m = len(indexed)
    sorted_items = sorted(indexed, key=lambda x: x[1])

    adjusted = [None] * len(rows)
    significant = [False] * len(rows)
    running_max = 0.0

    for rank0, (original_i, p) in enumerate(sorted_items):
        adj = min((m - rank0) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[original_i] = running_max

    # step-down 拒絕規則。
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


def fmt_float(x, digits=6):
    if x is None:
        return ""
    return f"{x:.{digits}g}"


def analyze():
    focal_rows = read_rows(FOCAL_EXP)
    all_results = []

    for exp_name in COMPARE_EXPS:
        compare_rows = read_rows(exp_name)
        for metric, higher_is_better, label in METRICS:
            a_vals, b_vals, sample_idxs = paired_arrays(focal_rows, compare_rows, metric)
            if len(a_vals) == 0:
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
                "focal_exp": FOCAL_EXP,
                "compare_exp": exp_name,
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


def write_markdown(path, rows):
    lines = []
    lines.append("# Ground-Truth Conditioned 顯著性分析")
    lines.append("")
    lines.append(f"主要比較實驗：`{FOCAL_EXP}`")
    lines.append(f"比較對象：`{', '.join(COMPARE_EXPS)}`")
    lines.append(f"Bootstrap：{BOOTSTRAP_N:,} 次 paired resamples，95% CI")
    lines.append("Wilcoxon：針對 paired per-sample improvements 進行 signed-rank test")
    lines.append("校正方式：本檔所有測試使用 Holm-Bonferroni correction")
    lines.append("")
    lines.append("Improvement 為正值代表主要比較實驗表現較好。")
    lines.append("")
    lines.append("| Compare | Metric | n | Focal mean | Compare mean | Improvement | 95% CI | p raw | p Holm | Sig. |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---|")

    for row in rows:
        sig = "yes" if row["significant_holm_0.05"] else "no"
        ci = f"[{fmt_float(row['bootstrap_ci95_low'])}, {fmt_float(row['bootstrap_ci95_high'])}]"
        lines.append(
            "| {compare} | {metric} | {n} | {fm} | {cm} | {imp} | {ci} | {p} | {ph} | {sig} |".format(
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
    lines.append("## 注意事項")
    lines.append("")
    lines.append("- 對 `rank` 與 InfoLM 指標而言，原始值越低越好，因此 improvement 計算為 `compare - focal`。")
    lines.append("- 對 Recall 與 BERTScore 指標而言，原始值越高越好，因此 improvement 計算為 `focal - compare`。")
    lines.append("- 若沒有 per-sample MuseChat predictions，MuseChat 無法納入此 paired test。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    out_dir = output_dir()
    rows = analyze()

    csv_path = os.path.join(out_dir, "significance_results.csv")
    json_path = os.path.join(out_dir, "significance_results.json")
    md_path = os.path.join(out_dir, "significance_summary.md")

    write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    write_markdown(md_path, rows)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")

    sig_count = sum(1 for r in rows if r["significant_holm_0.05"])
    print(f"Completed {len(rows)} paired tests; Holm-significant at alpha={ALPHA}: {sig_count}")


if __name__ == "__main__":
    main()
