# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Collect thesis baseline results into one compact table.

Run after the corresponding baseline/evaluation scripts finish. Missing files
are reported as "missing" so this can also be used as a progress checklist.
"""

import json
import os
from typing import Any, Dict, Optional


BASE_DIR = str(PROJECT_ROOT)
OUT_DIR = os.path.join(BASE_DIR, "checkpoints", "baseline_summary")


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return "" if x is None else f"{x * 100:.2f}%"


def num(x):
    return "" if x is None else f"{float(x):.3f}"


def get_metric(data, *keys):
    if not data:
        return None
    for key in keys:
        if key in data:
            return data[key]
    return None


def add_ranking_row(rows, name, data, note):
    if not data:
        rows.append([name, "missing", "", "", "", "", note])
        return
    rows.append([
        name,
        pct(get_metric(data, "recall_at_1", "recall@1")),
        pct(get_metric(data, "recall_at_5", "recall@5")),
        pct(get_metric(data, "recall_at_10", "recall@10")),
        str(data.get("median_rank", "")),
        num(data.get("mean_rank")),
        note,
    ])


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    similarity = load_json(os.path.join(
        BASE_DIR,
        "checkpoints",
        "baseline_similarity",
        "detailed_eval",
        "baseline_similarity_500pool_summary.json",
    ))
    musechat = load_json(os.path.join(
        BASE_DIR,
        "checkpoints",
        "musechat_light",
        "detailed_eval",
        "musechat_light_500pool_summary.json",
    ))
    exp01 = load_json(os.path.join(
        BASE_DIR,
        "checkpoints",
        "exp_01",
        "detailed_eval",
        "exp_01_best_500pool_summary.json",
    ))
    exp04 = load_json(os.path.join(
        BASE_DIR,
        "checkpoints",
        "exp_04",
        "detailed_eval",
        "exp_04_best_500pool_summary.json",
    ))
    llama = load_json(os.path.join(
        BASE_DIR,
        "checkpoints",
        "baseline_llama_prompt_only",
        "detailed_eval",
        "llama_prompt_only_top1_generation_summary.json",
    ))

    ranking_rows = []
    if similarity:
        for name, item in similarity.get("baselines", {}).items():
            add_ranking_row(ranking_rows, name, item, "non-parametric 500-pool baseline")
    else:
        add_ranking_row(ranking_rows, "similarity baselines", None, "run run_baseline_similarity_retrieval_500pool.py")

    add_ranking_row(ranking_rows, "MuseChat-light", musechat.get("ranking") if musechat else None, "re-implemented baseline")
    add_ranking_row(ranking_rows, "Unified MLLM w/o LTP (exp_04)", exp04.get("ranking") if exp04 else None, "already covered by ablation")
    add_ranking_row(ranking_rows, "Unified MLLM full (exp_01)", exp01.get("ranking") if exp01 else None, "proposed model")

    gen_rows = []
    for name, data, note in [
        ("MuseChat-light GT", musechat.get("generation", {}).get("gt_conditioned") if musechat else None, "GT-conditioned generation"),
        ("MuseChat-light Top-1", musechat.get("generation", {}).get("top1_end_to_end") if musechat else None, "end-to-end generation"),
        ("LLaMA prompting-only Top-1", llama, "text-only explanation baseline"),
    ]:
        if not data:
            gen_rows.append([name, "missing", "", "", "", note])
        else:
            gen_rows.append([
                name,
                num(data.get("bertscore_f1")),
                num(data.get("infolm_ab_divergence")),
                num(data.get("infolm_l2_distance")),
                num(data.get("infolm_fisher_rao")),
                note,
            ])

    md = []
    md.append("# Baseline Summary")
    md.append("")
    md.append("## Ranking")
    md.append("")
    md.append("| Model | R@1 | R@5 | R@10 | Median Rank | Mean Rank | Note |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for row in ranking_rows:
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("## Generation")
    md.append("")
    md.append("| Model | BERTScore F1 | AB Div. | L2 Dist. | Fisher-Rao Dist. | Note |")
    md.append("|---|---:|---:|---:|---:|---|")
    for row in gen_rows:
        md.append("| " + " | ".join(row) + " |")

    out_md = os.path.join(OUT_DIR, "baseline_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
