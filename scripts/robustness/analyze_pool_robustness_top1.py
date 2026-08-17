# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Summarize pool-size robustness under the formal Top-1 end-to-end setting.

Usage:
  1. Run:
       run_eval_500pool_top1_pool100.py
       run_eval_500pool_top1_prompt_original.py
       run_eval_500pool_top1_pool1000.py
  2. Open this file in VSCode and click Run.

Outputs:
  checkpoints/robustness_analysis_top1/pool_robustness_top1.csv
  checkpoints/robustness_analysis_top1/pool_robustness_top1_summary.md
  checkpoints/robustness_analysis_top1/pool_robustness_top1_summary.json
"""

import csv
import json
import os
from datetime import datetime


BASE_DIR = str(PROJECT_ROOT)
EXP_NAME = "exp_01"
CKPT_NAME = "best"
PROMPT_VARIANT = "original"

POOL_CONFIGS = [
    {
        "pool_size": 100,
        "ranking_summary": f"{EXP_NAME}_{CKPT_NAME}_100pool_pool100_summary.json",
        "top1_summary": f"{EXP_NAME}_{CKPT_NAME}_100pool_top1_pool100_prompt_original_summary.json",
    },
    {
        "pool_size": 500,
        "ranking_summary": f"{EXP_NAME}_{CKPT_NAME}_500pool_summary.json",
        "top1_summary": f"{EXP_NAME}_{CKPT_NAME}_500pool_top1_prompt_original_summary.json",
    },
    {
        "pool_size": 1000,
        "ranking_summary": f"{EXP_NAME}_{CKPT_NAME}_1000pool_pool1000_summary.json",
        "top1_summary": f"{EXP_NAME}_{CKPT_NAME}_1000pool_top1_pool1000_prompt_original_summary.json",
    },
]


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


def pct(x):
    if x is None:
        return ""
    return round(float(x) * 100, 4)


def round_or_blank(x, digits=6):
    if x is None:
        return ""
    return round(float(x), digits)


def rows_from_summaries():
    rows = []
    base = detail_dir()
    for cfg in POOL_CONFIGS:
        ranking_path = os.path.join(base, cfg["ranking_summary"])
        top1_path = os.path.join(base, cfg["top1_summary"])
        ranking_summary = read_json(ranking_path)
        top1_summary = read_json(top1_path)

        ranking = get_nested(ranking_summary, "ranking") or {}
        generation = get_nested(top1_summary, "split_generation", "all") or {}
        top1_correct = get_nested(top1_summary, "split_generation", "top1_correct_only") or {}
        top1_incorrect = get_nested(top1_summary, "split_generation", "top1_incorrect_only") or {}

        rows.append({
            "exp_name": EXP_NAME,
            "pool_size": cfg["pool_size"],
            "prompt_variant": PROMPT_VARIANT,
            "ranking_summary_exists": int(ranking_summary is not None),
            "top1_summary_exists": int(top1_summary is not None),
            "ranking_summary_path": ranking_path,
            "top1_summary_path": top1_path,
            "num_samples": ranking.get("num_samples") or generation.get("num_samples"),
            "recall@1": round_or_blank(ranking.get("recall@1")),
            "recall@5": round_or_blank(ranking.get("recall@5")),
            "recall@10": round_or_blank(ranking.get("recall@10")),
            "mean_rank": round_or_blank(ranking.get("mean_rank")),
            "median_rank": round_or_blank(ranking.get("median_rank")),
            "bertscore_f1_top1_all": round_or_blank(generation.get("bertscore_f1")),
            "infolm_ab_top1_all": round_or_blank(generation.get("infolm_ab_divergence")),
            "infolm_l2_top1_all": round_or_blank(generation.get("infolm_l2_distance")),
            "infolm_fisher_rao_top1_all": round_or_blank(generation.get("infolm_fisher_rao")),
            "title_consistency_rate": round_or_blank(generation.get("title_consistency_rate")),
            "needs_manual_review_rate": round_or_blank(generation.get("needs_manual_review_rate")),
            "top1_correct_n": top1_correct.get("num_samples", ""),
            "top1_incorrect_n": top1_incorrect.get("num_samples", ""),
            "top1_correct_bertscore_f1": round_or_blank(top1_correct.get("bertscore_f1")),
            "top1_incorrect_bertscore_f1": round_or_blank(top1_incorrect.get("bertscore_f1")),
            "top1_correct_infolm_l2": round_or_blank(top1_correct.get("infolm_l2_distance")),
            "top1_incorrect_infolm_l2": round_or_blank(top1_incorrect.get("infolm_l2_distance")),
        })
    return rows


def add_delta_rows(rows):
    by_pool = {int(r["pool_size"]): r for r in rows}
    base = by_pool.get(500)
    if not base:
        return rows

    metric_names = [
        "recall@1",
        "recall@5",
        "recall@10",
        "mean_rank",
        "bertscore_f1_top1_all",
        "infolm_ab_top1_all",
        "infolm_l2_top1_all",
        "infolm_fisher_rao_top1_all",
        "title_consistency_rate",
        "needs_manual_review_rate",
    ]
    for row in rows:
        for metric in metric_names:
            try:
                value = float(row[metric])
                base_value = float(base[metric])
            except (TypeError, ValueError):
                row[f"{metric}_delta_vs_500"] = ""
                row[f"{metric}_relative_change_vs_500"] = ""
                continue
            row[f"{metric}_delta_vs_500"] = round(value - base_value, 6)
            row[f"{metric}_relative_change_vs_500"] = (
                round((value - base_value) / abs(base_value), 6)
                if base_value != 0
                else ""
            )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=4):
    if x == "" or x is None:
        return "NA"
    try:
        return f"{float(x):.{digits}f}"
    except ValueError:
        return str(x)


def write_markdown(path, rows):
    lines = []
    lines.append("# Top-1 Pool-Size Robustness")
    lines.append("")
    lines.append(f"Experiment: `{EXP_NAME}`")
    lines.append(f"Prompt variant: `{PROMPT_VARIANT}`")
    lines.append("Generation setting: `Top-1 end-to-end`")
    lines.append("")
    lines.append("| Pool | R@1 | R@5 | R@10 | Mean Rank | BERT F1 | InfoLM L2 | Title Consistency | Manual Review |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {pool} | {r1} | {r5} | {r10} | {rank} | {bert} | {l2} | {title} | {review} |".format(
                pool=row["pool_size"],
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
    lines.append("## Delta vs 500-Pool")
    lines.append("")
    lines.append("| Pool | ΔR@1 | ΔR@5 | ΔR@10 | ΔMean Rank | ΔBERT F1 | ΔInfoLM L2 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {pool} | {r1} | {r5} | {r10} | {rank} | {bert} | {l2} |".format(
                pool=row["pool_size"],
                r1=fmt(row.get("recall@1_delta_vs_500")),
                r5=fmt(row.get("recall@5_delta_vs_500")),
                r10=fmt(row.get("recall@10_delta_vs_500")),
                rank=fmt(row.get("mean_rank_delta_vs_500")),
                bert=fmt(row.get("bertscore_f1_top1_all_delta_vs_500")),
                l2=fmt(row.get("infolm_l2_top1_all_delta_vs_500")),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Ranking metrics are read from the original pool-size ranking summaries.")
    lines.append("- Generation metrics are read from Top-1 end-to-end summaries, not GT-conditioned generation.")
    lines.append("- For InfoLM and mean rank, lower is better; negative delta vs 500-pool therefore means improvement.")
    lines.append("- If any field is `NA`, run the corresponding Top-1 pool wrapper first.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    out_dir = output_dir()
    rows = add_delta_rows(rows_from_summaries())

    csv_path = os.path.join(out_dir, "pool_robustness_top1.csv")
    json_path = os.path.join(out_dir, "pool_robustness_top1_summary.json")
    md_path = os.path.join(out_dir, "pool_robustness_top1_summary.md")

    write_csv(csv_path, rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "setting": "top1_end_to_end_pool_size_robustness",
        "exp_name": EXP_NAME,
        "prompt_variant": PROMPT_VARIANT,
        "rows": rows,
        "outputs": {
            "csv": csv_path,
            "json": json_path,
            "markdown": md_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_markdown(md_path, rows)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
