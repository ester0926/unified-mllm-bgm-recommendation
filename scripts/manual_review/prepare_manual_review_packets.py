# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Prepare thesis manual-review packets from existing experiment outputs.

This script does not rerun any model. It converts existing Top-1 end-to-end,
faithfulness, and metadata-consistency outputs into compact CSV/Markdown files
that are ready for human spot-checking and failure-case writing.

Outputs:
  checkpoints/manual_review/
    failure_case_candidates.csv
    failure_case_candidates.md
    human_faithfulness_spotcheck_template.csv
    human_metadata_spotcheck_template.csv
    human_llm_judge_consistency_template.csv
    manual_review_readme.md
"""

import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List


BASE_DIR = str(PROJECT_ROOT)
OUT_DIR = os.path.join(BASE_DIR, "checkpoints", "manual_review")

TOP1_CSV = os.path.join(
    BASE_DIR,
    "checkpoints",
    "exp_01",
    "detailed_eval",
    "exp_01_best_500pool_top1_prompt_original_samples_merged.csv",
)
CLAIM_CSV = os.path.join(
    BASE_DIR,
    "checkpoints",
    "faithfulness_analysis",
    "claim_annotations_top1_v2.csv",
)
METADATA_CSV = os.path.join(
    BASE_DIR,
    "checkpoints",
    "faithfulness_analysis",
    "metadata_consistency_claims.csv",
)
LLM_JUDGE_DIR = os.path.join(
    BASE_DIR,
    "checkpoints",
    "faithfulness_analysis",
    "llm_judge",
)

SAMPLE_SEED = 20260606
SPOTCHECK_N = 100


def read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        return default if value in ("", None) else int(float(value))
    except Exception:
        return default


def clean_text(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def pick_failure_case_candidates(top1_rows: List[dict], metadata_rows: List[dict]) -> List[dict]:
    metadata_unsupported = defaultdict(int)
    metadata_total = defaultdict(int)
    for row in metadata_rows:
        if row.get("exp_name") != "exp_01":
            continue
        sample_idx = row.get("sample_idx")
        metadata_total[sample_idx] += 1
        if as_int(row, "is_metadata_supported") == 0:
            metadata_unsupported[sample_idx] += 1

    enriched = []
    for row in top1_rows:
        sample_idx = row.get("sample_idx")
        total = metadata_total.get(sample_idx, 0)
        unsupported = metadata_unsupported.get(sample_idx, 0)
        unsupported_rate = unsupported / total if total else 0.0
        item = dict(row)
        item["_metadata_unsupported_rate"] = unsupported_rate
        item["_metadata_unsupported_count"] = unsupported
        item["_metadata_claim_count"] = total
        enriched.append(item)

    success = [
        r for r in enriched
        if as_int(r, "top1_is_gt") == 1
        and as_float(r, "bertscore_f1") >= 0.78
        and as_float(r, "infolm_l2_distance", 999.0) <= 0.16
        and r.get("needs_manual_review") in ("0", 0, "", None)
    ]
    success.sort(key=lambda r: (-as_float(r, "bertscore_f1"), as_float(r, "infolm_l2_distance", 999.0)))

    ranking_fail = [r for r in enriched if as_int(r, "rank", 999) > 10]
    ranking_fail.sort(key=lambda r: (-as_int(r, "rank"), -as_float(r, "score_gap_top1_minus_gt")))

    generation_fail = [r for r in enriched if as_float(r, "bertscore_f1") < 0.70]
    generation_fail.sort(key=lambda r: (as_float(r, "bertscore_f1"), -as_float(r, "infolm_l2_distance")))

    metadata_fail = [r for r in enriched if r["_metadata_claim_count"] >= 2]
    metadata_fail.sort(key=lambda r: (-r["_metadata_unsupported_rate"], -r["_metadata_claim_count"]))

    review_fail = [
        r for r in enriched
        if r.get("needs_manual_review") in ("1", 1)
        or (r.get("title_consistency") not in ("", "1", 1) and r.get("title_consistency") is not None)
    ]
    review_fail.sort(key=lambda r: (-as_float(r, "score_gap_top1_minus_gt"), as_float(r, "bertscore_f1")))

    groups = [
        ("success_high_alignment", success[:6]),
        ("failure_ranking_miss", ranking_fail[:6]),
        ("failure_low_generation_similarity", generation_fail[:6]),
        ("failure_metadata_inconsistency", metadata_fail[:6]),
        ("failure_title_or_manual_review", review_fail[:6]),
    ]

    output = []
    seen = set()
    for category, rows in groups:
        for row in rows:
            key = (category, row.get("sample_idx"))
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "case_category": category,
                "sample_idx": row.get("sample_idx"),
                "video_id": row.get("video_id"),
                "gt_music_id": row.get("gt_music_id"),
                "top1_music_id": row.get("top1_music_id"),
                "top1_is_gt": row.get("top1_is_gt"),
                "rank": row.get("rank"),
                "R@1": row.get("R@1"),
                "R@5": row.get("R@5"),
                "R@10": row.get("R@10"),
                "bertscore_f1": row.get("bertscore_f1"),
                "infolm_l2_distance": row.get("infolm_l2_distance"),
                "metadata_unsupported_rate": f"{row['_metadata_unsupported_rate']:.4f}",
                "metadata_unsupported_count": row["_metadata_unsupported_count"],
                "metadata_claim_count": row["_metadata_claim_count"],
                "music_title": row.get("music_title"),
                "music_artist": row.get("music_artist"),
                "user_text": clean_text(row.get("user_text")),
                "generated_text": clean_text(row.get("generated_text")),
                "reference_text": clean_text(row.get("reference_text")),
                "top1_reference_text": clean_text(row.get("top1_reference_text")),
                "manual_case_label": "",
                "manual_error_source": "",
                "manual_notes": "",
            })
    return output


def make_faithfulness_spotcheck(claim_rows: List[dict]) -> List[dict]:
    rng = random.Random(SAMPLE_SEED)
    pool = [
        r for r in claim_rows
        if r.get("condition") in ("full", "wo_video", "wo_audio_feature_only", "wo_audio_all", "wo_ltp", "wo_prompt")
    ]
    rng.shuffle(pool)
    selected = pool[:SPOTCHECK_N]
    return [
        {
            "sample_idx": r.get("sample_idx"),
            "video_id": r.get("video_id"),
            "condition": r.get("condition"),
            "claim_id": r.get("claim_id"),
            "claim_text": r.get("claim_text"),
            "rule_claim_type": r.get("claim_type"),
            "rule_support_source": r.get("support_source"),
            "rule_is_supported": r.get("is_supported"),
            "generated_text": clean_text(r.get("generated_text")),
            "human_claim_type": "",
            "human_support_source": "",
            "human_is_supported": "",
            "human_notes": "",
        }
        for r in selected
    ]


def make_metadata_spotcheck(metadata_rows: List[dict]) -> List[dict]:
    rng = random.Random(SAMPLE_SEED + 1)
    exp01 = [r for r in metadata_rows if r.get("exp_name") == "exp_01"]
    unsupported = [r for r in exp01 if as_int(r, "is_metadata_supported") == 0]
    supported = [r for r in exp01 if as_int(r, "is_metadata_supported") == 1]
    rng.shuffle(unsupported)
    rng.shuffle(supported)
    selected = unsupported[: SPOTCHECK_N // 2] + supported[: SPOTCHECK_N // 2]
    rng.shuffle(selected)
    return [
        {
            "sample_idx": r.get("sample_idx"),
            "video_id": r.get("video_id"),
            "top1_is_gt": r.get("top1_is_gt"),
            "claim_id": r.get("claim_id"),
            "claim_text": r.get("claim_text"),
            "claim_terms": r.get("claim_terms"),
            "metadata_terms": r.get("metadata_terms"),
            "rule_is_metadata_supported": r.get("is_metadata_supported"),
            "generated_text": clean_text(r.get("generated_text")),
            "top1_reference_text": clean_text(r.get("top1_reference_text")),
            "human_is_metadata_supported": "",
            "human_unsupported_terms": "",
            "human_notes": "",
        }
        for r in selected
    ]


def make_llm_judge_consistency_template() -> List[dict]:
    rows = []
    sources = [
        ("feature_erasure", os.path.join(LLM_JUDGE_DIR, "llm_judge_feature_erasure.csv")),
        ("preference_counterfactual", os.path.join(LLM_JUDGE_DIR, "llm_judge_preference_counterfactual.csv")),
        ("metadata_consistency", os.path.join(LLM_JUDGE_DIR, "llm_judge_metadata_consistency.csv")),
    ]
    for task, path in sources:
        if not os.path.exists(path):
            continue
        for row in read_csv(path)[:40]:
            out = {"judge_task": task}
            for key, value in row.items():
                if key.lower().endswith("text") or key in {"sample_idx", "video_id", "condition", "variant"}:
                    out[key] = clean_text(value)
                elif key.startswith("llm_") or key in {"agreement", "rule_label", "judge_label"}:
                    out[key] = value
            out["human_label"] = ""
            out["human_notes"] = ""
            rows.append(out)
    return rows


def write_failure_md(path: str, rows: List[dict]) -> None:
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["case_category"]].append(row)
    lines = [
        "# Failure and Success Case Candidates",
        "",
        "Use this file to choose 3 successful and 3 failed examples for the thesis.",
        "",
    ]
    for category, items in by_cat.items():
        lines.append(f"## {category}")
        lines.append("")
        for row in items:
            lines.append(f"### sample_idx={row['sample_idx']} | rank={row['rank']} | BERT-F1={row['bertscore_f1']} | L2={row['infolm_l2_distance']}")
            lines.append("")
            lines.append(f"- video_id: `{row['video_id']}`")
            lines.append(f"- gt_music_id: `{row['gt_music_id']}`")
            lines.append(f"- top1_music_id: `{row['top1_music_id']}`")
            lines.append(f"- top1_is_gt: `{row['top1_is_gt']}`")
            lines.append(f"- metadata unsupported rate: `{row['metadata_unsupported_rate']}`")
            lines.append(f"- user_text: {row['user_text']}")
            lines.append(f"- generated_text: {row['generated_text']}")
            lines.append(f"- reference_text: {row['reference_text']}")
            lines.append(f"- top1_reference_text: {row['top1_reference_text']}")
            lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_readme(path: str, outputs: Dict[str, str]) -> None:
    lines = [
        "# Manual Review Packet",
        "",
        f"Created at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "These files convert existing automatic results into human-checkable materials.",
        "They are intended to support thesis sections on failure cases, human spot-checking, and judge consistency.",
        "",
        "## Suggested use",
        "",
        "1. Pick 3 successful and 3 failed examples from `failure_case_candidates.md`.",
        "2. Ask one or two annotators to complete `human_faithfulness_spotcheck_template.csv` and `human_metadata_spotcheck_template.csv`.",
        "3. If comparing human and LLM judge, fill `human_llm_judge_consistency_template.csv` and compute agreement or Cohen's kappa.",
        "",
        "## Outputs",
        "",
    ]
    for name, path_value in outputs.items():
        lines.append(f"- {name}: `{path_value}`")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    top1_rows = read_csv(TOP1_CSV)
    claim_rows = read_csv(CLAIM_CSV)
    metadata_rows = read_csv(METADATA_CSV)

    failure_rows = pick_failure_case_candidates(top1_rows, metadata_rows)
    faithfulness_rows = make_faithfulness_spotcheck(claim_rows)
    metadata_spotcheck_rows = make_metadata_spotcheck(metadata_rows)
    judge_rows = make_llm_judge_consistency_template()

    outputs = {
        "failure_case_candidates_csv": os.path.join(OUT_DIR, "failure_case_candidates.csv"),
        "failure_case_candidates_md": os.path.join(OUT_DIR, "failure_case_candidates.md"),
        "human_faithfulness_spotcheck_template": os.path.join(OUT_DIR, "human_faithfulness_spotcheck_template.csv"),
        "human_metadata_spotcheck_template": os.path.join(OUT_DIR, "human_metadata_spotcheck_template.csv"),
        "human_llm_judge_consistency_template": os.path.join(OUT_DIR, "human_llm_judge_consistency_template.csv"),
        "readme": os.path.join(OUT_DIR, "manual_review_readme.md"),
    }

    write_csv(outputs["failure_case_candidates_csv"], failure_rows)
    write_failure_md(outputs["failure_case_candidates_md"], failure_rows)
    write_csv(outputs["human_faithfulness_spotcheck_template"], faithfulness_rows)
    write_csv(outputs["human_metadata_spotcheck_template"], metadata_spotcheck_rows)
    write_csv(outputs["human_llm_judge_consistency_template"], judge_rows)
    write_readme(outputs["readme"], outputs)

    summary = {
        "failure_case_candidates": len(failure_rows),
        "faithfulness_spotcheck_rows": len(faithfulness_rows),
        "metadata_spotcheck_rows": len(metadata_spotcheck_rows),
        "llm_judge_consistency_rows": len(judge_rows),
        "outputs": outputs,
    }
    with open(os.path.join(OUT_DIR, "manual_review_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
