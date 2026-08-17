# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Metadata consistency test for generated recommendation explanations.

Usage:
  1. Run run_eval_500pool_top1_generation_from_ranking.py
  2. Open this file in VSCode and click Run.

This script checks whether music-detail claims in generated explanations are
supported by the top-1 candidate's available metadata/reference text. It is a
rule-based first-pass test for the advisor's "metadata consistency test".
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime


BASE_DIR = str(PROJECT_ROOT)
EXP_NAMES = [f"exp_{i:02d}" for i in range(1, 8)]
INPUT_GENERATION_TAG = "top1"
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints", "faithfulness_analysis")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "metadata_consistency_claims.csv")
OUTPUT_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "metadata_consistency_summary.json")
OUTPUT_SUMMARY_MD = os.path.join(OUTPUT_DIR, "metadata_consistency_summary.md")


MUSIC_KEYWORDS = {
    "genre": {
        "rock", "pop", "hip-hop", "hip hop", "rap", "electronic", "dance",
        "indie", "folk", "jazz", "classical", "metal", "punk", "r&b",
        "soul", "blues", "country", "acoustic", "techno", "house", "ambient",
        "alternative", "chillout",
    },
    "instrument": {
        "piano", "guitar", "drum", "drums", "bass", "synth", "synthesizer",
        "violin", "strings", "vocal", "vocals", "male vocals", "female vocals",
        "instrumental", "beat", "beats",
    },
    "mood_tempo": {
        "upbeat", "energetic", "calm", "relaxed", "chill", "melancholy",
        "sentimental", "bright", "dark", "slow", "fast", "soft", "gentle",
        "emotional", "uplifting", "rhythm", "tempo", "melody", "melodic",
    },
}

ALL_MUSIC_TERMS = sorted({term for terms in MUSIC_KEYWORDS.values() for term in terms}, key=len, reverse=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def split_claims(text):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    claims = []
    for sent in parts:
        sent = sent.strip(" ;")
        if len(sent.split()) >= 4:
            claims.append(sent)
    return claims


def term_hits(text):
    lower = normalize(text)
    return [term for term in ALL_MUSIC_TERMS if term in lower]


def term_categories(terms):
    cats = []
    for cat, keywords in MUSIC_KEYWORDS.items():
        if any(t in keywords for t in terms):
            cats.append(cat)
    return cats


def safe_div(num, den):
    return num / den if den else 0.0


def pct(x):
    return round(x * 100, 2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    claim_rows = []

    for exp_name in EXP_NAMES:
        input_csv = os.path.join(
            BASE_DIR,
            "checkpoints",
            exp_name,
            "detailed_eval",
            f"{exp_name}_best_500pool_{INPUT_GENERATION_TAG}_generation_samples.csv",
        )
        if not os.path.exists(input_csv):
            print(f"Skip missing: {input_csv}")
            continue

        rows = read_csv(input_csv)
        for row in rows:
            metadata_text = " ".join([
                row.get("music_title", ""),
                row.get("music_artist", ""),
                row.get("top1_reference_text", ""),
            ])
            metadata_terms = set(term_hits(metadata_text))
            generated = row.get("generated_text", "")
            for claim_id, claim in enumerate(split_claims(generated), start=1):
                claim_terms = set(term_hits(claim))
                if not claim_terms:
                    continue
                supported_terms = sorted(claim_terms & metadata_terms)
                unsupported_terms = sorted(claim_terms - metadata_terms)
                is_supported = int(len(unsupported_terms) == 0)
                claim_rows.append({
                    "exp_name": exp_name,
                    "sample_idx": row.get("sample_idx", ""),
                    "video_id": row.get("video_id", ""),
                    "gt_music_id": row.get("gt_music_id", ""),
                    "top1_music_id": row.get("top1_music_id", ""),
                    "top1_is_gt": row.get("top1_is_gt", ""),
                    "claim_id": claim_id,
                    "claim_text": claim,
                    "claim_terms": ";".join(sorted(claim_terms)),
                    "claim_categories": ";".join(term_categories(claim_terms)),
                    "metadata_terms": ";".join(sorted(metadata_terms)),
                    "supported_terms": ";".join(supported_terms),
                    "unsupported_terms": ";".join(unsupported_terms),
                    "is_metadata_supported": is_supported,
                    "generated_text": generated,
                    "top1_reference_text": row.get("top1_reference_text", ""),
                })

    write_csv(OUTPUT_CSV, claim_rows)

    by_exp = defaultdict(list)
    for row in claim_rows:
        by_exp[row["exp_name"]].append(row)

    summary_rows = []
    for exp_name in EXP_NAMES:
        items = by_exp.get(exp_name, [])
        if not items:
            continue
        unsupported = [r for r in items if int(r["is_metadata_supported"]) == 0]
        correct = [r for r in items if r["top1_is_gt"] == "1"]
        incorrect = [r for r in items if r["top1_is_gt"] == "0"]
        category_counts = Counter()
        for r in unsupported:
            for cat in r["claim_categories"].split(";"):
                if cat:
                    category_counts[cat] += 1
        summary_rows.append({
            "exp_name": exp_name,
            "n_music_claims": len(items),
            "unsupported_music_claims": len(unsupported),
            "unsupported_music_claim_rate": safe_div(len(unsupported), len(items)),
            "top1_correct_music_claims": len(correct),
            "top1_correct_unsupported_rate": safe_div(
                sum(1 for r in correct if int(r["is_metadata_supported"]) == 0),
                len(correct),
            ),
            "top1_incorrect_music_claims": len(incorrect),
            "top1_incorrect_unsupported_rate": safe_div(
                sum(1 for r in incorrect if int(r["is_metadata_supported"]) == 0),
                len(incorrect),
            ),
            "unsupported_category_counts": dict(category_counts),
        })

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "rule_based_music_keyword_overlap_with_top1_reference_text",
        "scope_note": "Uses top1_reference_text as the available metadata proxy; should be manually audited before strong claims.",
        "by_exp": summary_rows,
        "outputs": {
            "claims_csv": OUTPUT_CSV,
            "summary_json": OUTPUT_SUMMARY_JSON,
            "summary_md": OUTPUT_SUMMARY_MD,
        },
    }
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# Metadata Consistency Test",
        "",
        "This rule-based analysis checks whether music-detail terms in generated explanations are supported by the top-1 candidate metadata/reference text.",
        "",
        "| Exp | Music claims | Unsupported claims | Unsupported rate | Top-1 correct unsupported | Top-1 incorrect unsupported |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['exp_name']} | {row['n_music_claims']} | "
            f"{row['unsupported_music_claims']} | {pct(row['unsupported_music_claim_rate'])}% | "
            f"{pct(row['top1_correct_unsupported_rate'])}% | "
            f"{pct(row['top1_incorrect_unsupported_rate'])}% |"
        )
    lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- Lower unsupported rate indicates better metadata consistency.",
        "- This is a conservative keyword-overlap proxy; it can over-flag paraphrases and under-flag unsupported generic claims.",
        "- Use this table as first-pass evidence, then manually audit representative cases.",
    ])
    with open(OUTPUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY_JSON}")
    print(f"Saved: {OUTPUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
