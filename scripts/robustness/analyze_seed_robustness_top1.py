# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Summarize random-seed robustness under the formal Top-1 end-to-end setting.

Usage:
  1. Make sure these ranking runs have completed:
       run_eval_500pool_seed42.py
       run_eval_500pool_seed12345.py
       run_eval_500pool_seed987654.py
  2. Make sure these Top-1 generation runs have completed:
       run_eval_500pool_top1_seed42.py
       run_eval_500pool_top1_seed12345.py
       run_eval_500pool_top1_seed987654.py
  3. Open this file in VSCode and click Run.

Outputs:
  checkpoints/robustness_analysis_top1/seed_robustness_top1.csv
  checkpoints/robustness_analysis_top1/seed_robustness_top1_summary.json
  checkpoints/robustness_analysis_top1/seed_robustness_top1_summary.md

This script intentionally uses the revised, well-separated seeds
42 / 12345 / 987654 rather than the earlier adjacent seeds
20260315 / 20260316 / 20260317.
"""

import csv
import json
import math
import os
from datetime import datetime


BASE_DIR = str(PROJECT_ROOT)
EXP_NAME = "exp_01"
CKPT_NAME = "best"
POOL_SIZE = 500
PROMPT_VARIANT = "original"

SEEDS = [42, 12345, 987654]


def detail_dir():
    return os.path.join(BASE_DIR, "checkpoints", EXP_NAME, "detailed_eval")


def output_dir():
    path = os.path.join(BASE_DIR, "checkpoints", "robustness_analysis_top1")
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(obj, *keys):
    cur = obj or {}
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def as_float(value):
    if value in ("", None):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def round_or_blank(value, digits=6):
    x = as_float(value)
    if x is None:
        return ""
    return round(x, digits)


def ranking_summary_path(seed):
    return os.path.join(
        detail_dir(),
        f"{EXP_NAME}_{CKPT_NAME}_{POOL_SIZE}pool_seed{seed}_summary.json",
    )


def top1_summary_path(seed):
    return os.path.join(
        detail_dir(),
        f"{EXP_NAME}_{CKPT_NAME}_{POOL_SIZE}pool_top1_seed{seed}_summary.json",
    )


def rows_from_summaries():
    rows = []
    for seed in SEEDS:
        ranking_path = ranking_summary_path(seed)
        top1_path = top1_summary_path(seed)
        ranking_summary = read_json(ranking_path)
        top1_summary = read_json(top1_path)

        ranking = get_nested(ranking_summary, "ranking") or {}
        generation_all = get_nested(top1_summary, "split_generation", "all") or {}
        top1_correct = get_nested(top1_summary, "split_generation", "top1_correct_only") or {}
        top1_incorrect = get_nested(top1_summary, "split_generation", "top1_incorrect_only") or {}

        rows.append({
            "exp_name": EXP_NAME,
            "candidate_pool_seed": seed,
            "pool_size": POOL_SIZE,
            "prompt_variant": PROMPT_VARIANT,
            "ranking_summary_exists": int(ranking_summary is not None),
            "top1_summary_exists": int(top1_summary is not None),
            "ranking_summary_path": ranking_path,
            "top1_summary_path": top1_path,
            "num_samples": ranking.get("num_samples") or generation_all.get("num_samples", ""),
            "recall@1": round_or_blank(ranking.get("recall@1")),
            "recall@5": round_or_blank(ranking.get("recall@5")),
            "recall@10": round_or_blank(ranking.get("recall@10")),
            "mean_rank": round_or_blank(ranking.get("mean_rank")),
            "median_rank": round_or_blank(ranking.get("median_rank")),
            "top1_correct_n": top1_correct.get("num_samples", ""),
            "top1_incorrect_n": top1_incorrect.get("num_samples", ""),
            "bertscore_f1_top1_all": round_or_blank(generation_all.get("bertscore_f1")),
            "infolm_ab_top1_all": round_or_blank(generation_all.get("infolm_ab_divergence")),
            "infolm_l2_top1_all": round_or_blank(generation_all.get("infolm_l2_distance")),
            "infolm_fisher_rao_top1_all": round_or_blank(generation_all.get("infolm_fisher_rao")),
            "title_consistency_rate": round_or_blank(generation_all.get("title_consistency_rate")),
            "needs_manual_review_rate": round_or_blank(generation_all.get("needs_manual_review_rate")),
            "top1_correct_bertscore_f1": round_or_blank(top1_correct.get("bertscore_f1")),
            "top1_incorrect_bertscore_f1": round_or_blank(top1_incorrect.get("bertscore_f1")),
            "top1_correct_infolm_l2": round_or_blank(top1_correct.get("infolm_l2_distance")),
            "top1_incorrect_infolm_l2": round_or_blank(top1_incorrect.get("infolm_l2_distance")),
        })
    return rows


def summarize_metric(rows, metric):
    values = [as_float(row.get(metric)) for row in rows]
    values = [v for v in values if v is not None]
    if not values:
        return {
            "metric": metric,
            "n": 0,
            "mean": "",
            "min": "",
            "max": "",
            "range": "",
            "relative_range": "",
        }
    mean = sum(values) / len(values)
    min_v = min(values)
    max_v = max(values)
    value_range = max_v - min_v
    rel_range = value_range / abs(mean) if mean else ""
    return {
        "metric": metric,
        "n": len(values),
        "mean": round(mean, 6),
        "min": round(min_v, 6),
        "max": round(max_v, 6),
        "range": round(value_range, 6),
        "relative_range": round(rel_range, 6) if rel_range != "" else "",
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    x = as_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def write_markdown(path, rows, metric_summary):
    lines = []
    lines.append("# Top-1 Random Seed Robustness")
    lines.append("")
    lines.append(f"Experiment: `{EXP_NAME}`")
    lines.append(f"Pool size: `{POOL_SIZE}`")
    lines.append(f"Prompt variant: `{PROMPT_VARIANT}`")
    lines.append(f"Seeds: `{', '.join(str(s) for s in SEEDS)}`")
    lines.append("Generation setting: `Top-1 end-to-end`")
    lines.append("")
    lines.append("## Per-Seed Results")
    lines.append("")
    lines.append("| Seed | R@1 | R@5 | R@10 | Mean Rank | BERT F1 | InfoLM L2 | Title Consistency | Manual Review |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {seed} | {r1} | {r5} | {r10} | {rank} | {bert} | {l2} | {title} | {review} |".format(
                seed=row["candidate_pool_seed"],
                r1=fmt(row["recall@1"]),
                r5=fmt(row["recall@5"]),
                r10=fmt(row["recall@10"]),
                rank=fmt(row["mean_rank"]),
                bert=fmt(row["bertscore_f1_top1_all"]),
                l2=fmt(row["infolm_l2_top1_all"]),
                title=fmt(row["title_consistency_rate"]),
                review=fmt(row["needs_manual_review_rate"]),
            )
        )

    lines.append("")
    lines.append("## Stability Summary")
    lines.append("")
    lines.append("| Metric | Mean | Min | Max | Range | Relative Range |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in metric_summary:
        lines.append(
            "| {metric} | {mean} | {min_v} | {max_v} | {range_v} | {rel} |".format(
                metric=item["metric"],
                mean=fmt(item["mean"]),
                min_v=fmt(item["min"]),
                max_v=fmt(item["max"]),
                range_v=fmt(item["range"]),
                rel=fmt(item["relative_range"]),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This analysis uses the revised non-adjacent seeds `42`, `12345`, and `987654`.")
    lines.append("- Ranking metrics are recomputed from each seed-specific candidate pool.")
    lines.append("- Generation metrics are computed from the Top-1 music selected under each seed-specific ranking result.")
    lines.append("- For mean rank and InfoLM metrics, lower is better.")
    lines.append("- The earlier adjacent seeds `20260315/20260316/20260317` should be treated as historical pilot runs, not the formal seed robustness result.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    out_dir = output_dir()
    rows = rows_from_summaries()
    metric_summary = [
        summarize_metric(rows, "recall@1"),
        summarize_metric(rows, "recall@5"),
        summarize_metric(rows, "recall@10"),
        summarize_metric(rows, "mean_rank"),
        summarize_metric(rows, "bertscore_f1_top1_all"),
        summarize_metric(rows, "infolm_l2_top1_all"),
        summarize_metric(rows, "title_consistency_rate"),
        summarize_metric(rows, "needs_manual_review_rate"),
    ]

    csv_path = os.path.join(out_dir, "seed_robustness_top1.csv")
    json_path = os.path.join(out_dir, "seed_robustness_top1_summary.json")
    md_path = os.path.join(out_dir, "seed_robustness_top1_summary.md")

    write_csv(csv_path, rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "setting": "top1_end_to_end_random_seed_robustness",
        "exp_name": EXP_NAME,
        "pool_size": POOL_SIZE,
        "prompt_variant": PROMPT_VARIANT,
        "formal_seeds": SEEDS,
        "historical_seed_note": "Earlier adjacent seeds 20260315/20260316/20260317 are excluded from this formal summary.",
        "rows": rows,
        "metric_summary": metric_summary,
        "outputs": {
            "csv": csv_path,
            "json": json_path,
            "markdown": md_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_markdown(md_path, rows, metric_summary)

    missing = [
        row for row in rows
        if not row["ranking_summary_exists"] or not row["top1_summary_exists"]
    ]
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    if missing:
        print("Warning: missing summaries for seeds:")
        for row in missing:
            print(
                f"  seed={row['candidate_pool_seed']} "
                f"ranking_exists={row['ranking_summary_exists']} "
                f"top1_exists={row['top1_summary_exists']}"
            )
    else:
        print("All formal seed summaries found.")


if __name__ == "__main__":
    main()
