"""
Cliff's Delta Effect Size + Wilcoxon Signed-Rank Test
with Holm-Bonferroni correction

Compares exp_01 (full model) vs ablation variants on rank-based metrics.
Output: CSV + Markdown table for thesis §4.3 / §4.4.
"""

import os
import csv
import json
import math
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]  # unified_mllm_pointwise_final
CKPT = BASE / "checkpoints"
OUT_DIR = BASE / "checkpoints" / "significance_analysis"
OUT_DIR.mkdir(exist_ok=True)

# ── experiment metadata ───────────────────────────────────────────────────────
EXPERIMENTS = {
    "exp_01": {"label": "Full Model (hybrid LTP, all modalities)",    "ablation": None},
    "exp_02": {"label": "wo_implicit (explicit LTP only)",           "ablation": "wo_implicit"},
    "exp_03": {"label": "wo_explicit (implicit LTP only)",           "ablation": "wo_explicit"},
    "exp_04": {"label": "wo_LTP (no LTP vector)",                    "ablation": "wo_ltp"},
    "exp_05": {"label": "wo_video (no video modality)",               "ablation": "wo_video"},
    "exp_06": {"label": "wo_text (no text/dialogue)",                 "ablation": "wo_text"},
    "exp_07": {"label": "wo_music (no music feature)",                "ablation": "wo_music"},
}

# Comparisons: (exp_01 vs each ablation)
COMPARISONS = ["exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07"]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_ranks(exp_name: str) -> np.ndarray:
    """Load per-sample ranks from the 500-pool ranking CSV."""
    if exp_name == "musechat_light":
        csv_path = CKPT / "musechat_light" / "detailed_eval" / "musechat_light_500pool_ranking_samples.csv"
    else:
        csv_path = CKPT / exp_name / "detailed_eval" / f"{exp_name}_best_500pool_ranking_samples.csv"
    ranks = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks.append(int(row["rank"]))
    return np.array(ranks, dtype=float)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: proportion of (x_i < y_j) minus (x_i > y_j), for all pairs.
    Positive value → x tends to have smaller values than y (x is better for rank).
    Here x = exp_01 ranks, y = ablation ranks → positive δ means full model ranks lower (better).
    """
    n, m = len(x), len(y)
    # vectorised: compare all pairs
    dom = np.sum(x[:, None] < y[None, :]) - np.sum(x[:, None] > y[None, :])
    return float(dom) / (n * m)


def delta_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    elif a < 0.330:
        return "small"
    elif a < 0.474:
        return "medium"
    else:
        return "large"


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values (same length as input)."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    prev_adj = 0.0
    for rank_i, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank_i)
        adj = max(adj, prev_adj)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev_adj = adj
    return adjusted


def format_p(p: float) -> str:
    if p < 0.001:
        return "< .001"
    elif p < 0.01:
        return f"= {p:.3f}"
    elif p < 0.05:
        return f"= {p:.3f}"
    else:
        return f"= {p:.3f}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading exp_01 ranks...")
    ranks_01 = load_ranks("exp_01")
    n = len(ranks_01)
    print(f"  n = {n} samples")

    results = []
    raw_p_values = []

    for exp in COMPARISONS:
        print(f"Processing {exp}...")
        ranks_ab = load_ranks(exp)

        # align sample count (should be same, but be safe)
        min_n = min(len(ranks_01), len(ranks_ab))
        r1 = ranks_01[:min_n]
        ra = ranks_ab[:min_n]

        # Cliff's delta (exp_01 vs ablation)
        delta = cliffs_delta(r1, ra)
        mag = delta_magnitude(delta)

        # Wilcoxon signed-rank test (two-tailed)
        # For tied differences, use zero_method='zsplit'
        diffs = r1 - ra  # negative = exp_01 ranks lower (better)
        try:
            stat, p_raw = wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
        except ValueError as e:
            # all zeros case
            stat, p_raw = 0.0, 1.0

        raw_p_values.append(p_raw)
        results.append({
            "exp": exp,
            "label": EXPERIMENTS[exp]["label"],
            "ablation": EXPERIMENTS[exp]["ablation"],
            "n": min_n,
            "delta": delta,
            "magnitude": mag,
            "W_stat": stat,
            "p_raw": p_raw,
        })

    # Holm-Bonferroni correction
    adj_ps = holm_bonferroni(raw_p_values)
    for i, r in enumerate(results):
        r["p_adj"] = adj_ps[i]
        r["sig"] = "***" if r["p_adj"] < 0.001 else ("**" if r["p_adj"] < 0.01 else ("*" if r["p_adj"] < 0.05 else "ns"))

    # ── also compute R@k means ──────────────────────────────────────────────
    def load_recall_columns(exp_name):
        if exp_name == "musechat_light":
            csv_path = CKPT / "musechat_light" / "detailed_eval" / "musechat_light_500pool_ranking_samples.csv"
        else:
            csv_path = CKPT / exp_name / "detailed_eval" / f"{exp_name}_best_500pool_ranking_samples.csv"
        r1_vals, r5_vals, r10_vals = [], [], []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r1_vals.append(int(row["R@1"]))
                r5_vals.append(int(row["R@5"]))
                r10_vals.append(int(row["R@10"]))
        return np.array(r1_vals), np.array(r5_vals), np.array(r10_vals)

    r1_01, r5_01, r10_01 = load_recall_columns("exp_01")

    for r in results:
        r1_ab, r5_ab, r10_ab = load_recall_columns(r["exp"])
        min_n = min(len(r1_01), len(r1_ab))
        r["R@1_full"]  = r1_01[:min_n].mean() * 100
        r["R@5_full"]  = r5_01[:min_n].mean() * 100
        r["R@10_full"] = r10_01[:min_n].mean() * 100
        r["R@1_abl"]   = r1_ab[:min_n].mean() * 100
        r["R@5_abl"]   = r5_ab[:min_n].mean() * 100
        r["R@10_abl"]  = r10_ab[:min_n].mean() * 100
        r["delta_R@1"] = r["R@1_full"] - r["R@1_abl"]
        r["delta_R@5"] = r["R@5_full"] - r["R@5_abl"]
        r["delta_R@10"]= r["R@10_full"] - r["R@10_abl"]

    # ── save CSV ────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "cliffs_delta_results.csv"
    fields = ["exp", "ablation", "label", "n",
              "R@1_full", "R@1_abl", "delta_R@1",
              "R@5_full", "R@5_abl", "delta_R@5",
              "R@10_full", "R@10_abl", "delta_R@10",
              "delta", "magnitude", "W_stat", "p_raw", "p_adj", "sig"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV → {csv_path}")

    # ── print Markdown table ────────────────────────────────────────────────
    md_lines = []
    md_lines.append("## Cliff's Delta + Wilcoxon Significance Analysis (500-pool, n=4205)\n")
    md_lines.append("Baseline: **exp_01** (Full Model, hybrid LTP, all modalities)  ")
    md_lines.append(f"R@1={r1_01.mean()*100:.2f}%, R@5={r5_01.mean()*100:.2f}%, R@10={r10_01.mean()*100:.2f}%\n")
    md_lines.append("Holm-Bonferroni correction applied across all pairwise comparisons.\n")

    header = "| Comparison | Ablation | ΔR@1 (pp) | ΔR@5 (pp) | ΔR@10 (pp) | Cliff's δ | Magnitude | W | p_adj | Sig |"
    sep    = "|---|---|---:|---:|---:|---:|---|---:|---:|---|"
    md_lines.append(header)
    md_lines.append(sep)
    for r in results:
        row = (f"| {r['exp']} | {r['ablation']} "
               f"| +{r['delta_R@1']:.2f} | +{r['delta_R@5']:.2f} | +{r['delta_R@10']:.2f} "
               f"| {r['delta']:+.3f} | {r['magnitude']} "
               f"| {r['W_stat']:.0f} | p {format_p(r['p_adj'])} | {r['sig']} |")
        md_lines.append(row)

    md_text = "\n".join(md_lines)
    md_path = OUT_DIR / "cliffs_delta_results.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Saved Markdown → {md_path}")

    # ── print to stdout ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(md_text)
    print("="*80)

    # ── thesis-ready summary ────────────────────────────────────────────────
    print("\n\n=== THESIS TEXT FRAGMENT ===\n")
    for r in results:
        print(f"exp_01 vs {r['ablation']} ({r['exp']}): "
              f"ΔR@1=+{r['delta_R@1']:.2f}pp, "
              f"δ={r['delta']:+.3f} ({r['magnitude']}), "
              f"W={r['W_stat']:.0f}, p_adj {format_p(r['p_adj'])} {r['sig']}")

    # ── JSON summary ─────────────────────────────────────────────────────────
    json_path = OUT_DIR / "cliffs_delta_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved JSON → {json_path}")


if __name__ == "__main__":
    main()
