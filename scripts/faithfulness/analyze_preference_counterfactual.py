"""
用途：分析偏好改寫前後的推薦解釋差異。
輸入：主評估輸出的推薦解釋、metadata、counterfactual 或人工複查檔。
輸出：claim 標註、faithfulness 指標、UCR 摘要或人工檢查表。
執行：通常需先完成主評估或 Top-1 生成，再執行本檔。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime


BASE_DIR = str(PROJECT_ROOT)
ANALYSIS_DIR = os.path.join(BASE_DIR, "checkpoints", "faithfulness_analysis")
INPUT_CSV = os.path.join(ANALYSIS_DIR, "preference_counterfactual_generations.csv")
OUTPUT_CSV = os.path.join(ANALYSIS_DIR, "preference_counterfactual_analysis.csv")
OUTPUT_SUMMARY_JSON = os.path.join(ANALYSIS_DIR, "preference_counterfactual_summary.json")
OUTPUT_SUMMARY_MD = os.path.join(ANALYSIS_DIR, "preference_counterfactual_summary.md")


TARGET_KEYWORDS = {
    "cf_upbeat_electronic": {
        "positive": {
            "upbeat", "electronic", "bright", "energetic", "beat", "beats",
            "rhythm", "dance", "pop", "modern", "tempo", "synth", "techno",
        },
        "negative": {
            "piano", "acoustic", "slow", "soft", "gentle", "sentimental",
            "calm", "lyrical", "melody", "melancholy",
        },
    },
    "cf_lyrical_piano": {
        "positive": {
            "piano", "acoustic", "slow", "soft", "gentle", "sentimental",
            "calm", "lyrical", "melody", "melodic", "emotional", "quiet",
        },
        "negative": {
            "upbeat", "electronic", "energetic", "dance", "techno", "beat",
            "beats", "synth", "hip-hop", "rap",
        },
    },
}


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tokenize(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z\-']+", (text or "").lower()))


def keyword_hits(text, keywords):
    lower = (text or "").lower()
    return sorted(k for k in keywords if k in lower)


def safe_div(num, den):
    return num / den if den else 0.0


def jaccard_distance(a, b):
    sa = tokenize(a)
    sb = tokenize(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / max(len(sa | sb), 1)


def pct(x):
    return round(x * 100, 2)


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Run run_preference_counterfactual_generation.py first: {INPUT_CSV}")

    rows = read_csv(INPUT_CSV)
    by_sample = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_idx"]][row["preference_variant"]] = row

    analysis_rows = []
    for sample_idx, variants in by_sample.items():
        original = variants.get("original", {})
        original_text = original.get("generated_text", "")
        for variant, cfg in TARGET_KEYWORDS.items():
            row = variants.get(variant)
            if not row:
                continue
            generated = row.get("generated_text", "")
            pos_hits = keyword_hits(generated, cfg["positive"])
            neg_hits = keyword_hits(generated, cfg["negative"])
            alignment_score = len(pos_hits) - len(neg_hits)
            analysis_rows.append({
                "sample_idx": sample_idx,
                "video_id": row.get("video_id", ""),
                "gt_music_id": row.get("gt_music_id", ""),
                "preference_variant": variant,
                "positive_hits": ";".join(pos_hits),
                "negative_hits": ";".join(neg_hits),
                "n_positive_hits": len(pos_hits),
                "n_negative_hits": len(neg_hits),
                "alignment_score": alignment_score,
                "is_aligned": int(len(pos_hits) > 0 and alignment_score > 0),
                "has_conflict": int(len(neg_hits) > 0),
                "ESS_from_original": jaccard_distance(original_text, generated),
                "generated_text": generated,
            })

    write_csv(OUTPUT_CSV, analysis_rows)

    summary_by_variant = {}
    for variant in TARGET_KEYWORDS:
        items = [r for r in analysis_rows if r["preference_variant"] == variant]
        summary_by_variant[variant] = {
            "n": len(items),
            "aligned_rate": safe_div(sum(int(r["is_aligned"]) for r in items), len(items)),
            "conflict_rate": safe_div(sum(int(r["has_conflict"]) for r in items), len(items)),
            "avg_positive_hits": safe_div(sum(int(r["n_positive_hits"]) for r in items), len(items)),
            "avg_negative_hits": safe_div(sum(int(r["n_negative_hits"]) for r in items), len(items)),
            "avg_alignment_score": safe_div(sum(float(r["alignment_score"]) for r in items), len(items)),
            "avg_ESS_from_original": safe_div(sum(float(r["ESS_from_original"]) for r in items), len(items)),
        }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": INPUT_CSV,
        "scope_note": "Prompt-level preference sensitivity; precomputed text features are unchanged.",
        "by_variant": summary_by_variant,
        "outputs": {
            "csv": OUTPUT_CSV,
            "json": OUTPUT_SUMMARY_JSON,
            "markdown": OUTPUT_SUMMARY_MD,
        },
    }
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# 偏好反事實測試",
        "",
        "本分析固定模型 checkpoint 與候選音樂，只改變自然語言使用者偏好 prompt，觀察生成推薦理由是否會跟著偏好條件改變。",
        "",
        "| Variant | n | Aligned rate | Conflict rate | Avg. ESS from original | Avg. alignment score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, item in summary_by_variant.items():
        lines.append(
            f"| {variant} | {item['n']} | {pct(item['aligned_rate'])}% | "
            f"{pct(item['conflict_rate'])}% | {item['avg_ESS_from_original']:.4f} | "
            f"{item['avg_alignment_score']:.2f} |"
        )
    lines.extend([
        "",
        "## 解讀注意事項",
        "",
        "- Aligned rate 越高，表示生成推薦理由越能反映反事實偏好文字。",
        "- Conflict rate 越高，表示推薦理由仍提到與反事實偏好相反的概念。",
        "- ESS 是與原始 prompt 推薦理由的 Jaccard distance；數值越高代表文字改變越大。",
        "- 這不是完整 reranking 測試，因為目前 pipeline 中的文字 embedding 已預先計算。",
    ])
    with open(OUTPUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY_JSON}")
    print(f"Saved: {OUTPUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
