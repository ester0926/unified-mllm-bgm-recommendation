"""
用途：彙整 claim 標註結果，計算解釋 faithfulness 指標。
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
from collections import Counter, defaultdict
from datetime import datetime


BASE_DIR = str(PROJECT_ROOT)
ANALYSIS_DIR = os.path.join(BASE_DIR, "checkpoints", "faithfulness_analysis")
GENERATION_CSV = os.path.join(ANALYSIS_DIR, "counterfactual_generations.csv")
CLAIM_CSV = os.path.join(ANALYSIS_DIR, "claim_annotations.csv")

OUTPUT_CONDITION_CSV = os.path.join(ANALYSIS_DIR, "faithfulness_by_condition.csv")
OUTPUT_SENSITIVITY_CSV = os.path.join(ANALYSIS_DIR, "faithfulness_sensitivity.csv")
OUTPUT_SUMMARY_JSON = os.path.join(ANALYSIS_DIR, "faithfulness_summary.json")
OUTPUT_SUMMARY_MD = os.path.join(ANALYSIS_DIR, "faithfulness_summary.md")


MODALITY_SOURCES = {
    "video-supported",
    "audio-supported",
    "prompt-supported",
    "preference-supported",
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
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", (text or "").lower())
    stop = {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
        "with", "this", "that", "it", "is", "are", "as", "by", "be", "your",
        "you", "i", "me", "my", "its", "it's", "from", "can", "could",
        "would", "should", "track", "song", "music",
    }
    return {w for w in words if w not in stop and len(w) > 2}


def jaccard_distance(a, b):
    sa = tokenize(a)
    sb = tokenize(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / max(len(sa | sb), 1)


def safe_div(num, den):
    return num / den if den else 0.0


def pct(x):
    return round(x * 100, 2)


def summarize_claims(claim_rows):
    by_condition = defaultdict(list)
    for row in claim_rows:
        by_condition[row["condition"]].append(row)

    rows = []
    for condition, items in sorted(by_condition.items()):
        n_claims = len(items)
        n_unsupported = sum(1 for r in items if int(r["is_supported"]) == 0)
        modality_items = [r for r in items if r["support_source"] in MODALITY_SOURCES]
        modality_supported = sum(1 for r in modality_items if int(r["is_supported"]) == 1)
        source_counts = Counter(r["support_source"] for r in items)

        rows.append({
            "condition": condition,
            "n_claims": n_claims,
            "unsupported_claims": n_unsupported,
            "UCR": safe_div(n_unsupported, n_claims),
            "MAA": safe_div(modality_supported, len(modality_items)),
            "video_claim_ratio": safe_div(source_counts["video-supported"], n_claims),
            "audio_claim_ratio": safe_div(source_counts["audio-supported"], n_claims),
            "prompt_claim_ratio": safe_div(source_counts["prompt-supported"], n_claims),
            "preference_claim_ratio": safe_div(source_counts["preference-supported"], n_claims),
            "metadata_claim_ratio": safe_div(source_counts["metadata-supported"], n_claims),
            "general_claim_ratio": safe_div(source_counts["general-supported"], n_claims),
            "unknown_claim_ratio": safe_div(source_counts["unsupported"], n_claims),
        })
    return rows


def summarize_sensitivity(generation_rows):
    by_sample = defaultdict(dict)
    for row in generation_rows:
        by_sample[row["sample_idx"]][row["condition"]] = row["generated_text"]

    distances_by_condition = defaultdict(list)
    for _, cond_map in by_sample.items():
        full = cond_map.get("full", "")
        if not full:
            continue
        for condition, text in cond_map.items():
            if condition == "full":
                continue
            distances_by_condition[condition].append(jaccard_distance(full, text))

    rows = []
    for condition, values in sorted(distances_by_condition.items()):
        rows.append({
            "condition": condition,
            "n_pairs": len(values),
            "ESS_jaccard_distance_mean": sum(values) / max(len(values), 1),
            "ESS_jaccard_distance_min": min(values) if values else 0.0,
            "ESS_jaccard_distance_max": max(values) if values else 0.0,
        })
    return rows


def get_condition_row(rows, condition):
    for row in rows:
        if row["condition"] == condition:
            return row
    return None


def build_summary(condition_rows, sensitivity_rows):
    full = get_condition_row(condition_rows, "full")
    wo_ltp = get_condition_row(condition_rows, "wo_ltp")
    wo_video = get_condition_row(condition_rows, "wo_video")
    wo_audio = get_condition_row(condition_rows, "wo_audio")
    wo_audio_feature_only = get_condition_row(condition_rows, "wo_audio_feature_only")
    wo_audio_all = get_condition_row(condition_rows, "wo_audio_all")
    wo_prompt = get_condition_row(condition_rows, "wo_prompt")

    full_pref = full["preference_claim_ratio"] if full else 0.0
    wo_ltp_pref = wo_ltp["preference_claim_ratio"] if wo_ltp else 0.0
    pcr = 1.0 - safe_div(wo_ltp_pref, full_pref) if full_pref > 0 else 0.0

    full_video = full["video_claim_ratio"] if full else 0.0
    full_audio = full["audio_claim_ratio"] if full else 0.0
    full_prompt = full["prompt_claim_ratio"] if full else 0.0

    claim_ratio_reductions = {
        "video_claim_reduction_after_wo_video": 1.0 - safe_div(wo_video["video_claim_ratio"], full_video) if wo_video and full_video > 0 else 0.0,
        "audio_claim_reduction_after_wo_audio_feature_only": 1.0 - safe_div(wo_audio_feature_only["audio_claim_ratio"], full_audio) if wo_audio_feature_only and full_audio > 0 else 0.0,
        "audio_claim_reduction_after_wo_audio_all": 1.0 - safe_div(wo_audio_all["audio_claim_ratio"], full_audio) if wo_audio_all and full_audio > 0 else 0.0,
        "prompt_claim_reduction_after_wo_prompt": 1.0 - safe_div(wo_prompt["prompt_claim_ratio"], full_prompt) if wo_prompt and full_prompt > 0 else 0.0,
        "preference_claim_reduction_after_wo_ltp": pcr,
    }
    if wo_audio:
        claim_ratio_reductions["audio_claim_reduction_after_wo_audio"] = (
            1.0 - safe_div(wo_audio["audio_claim_ratio"], full_audio)
            if full_audio > 0 else 0.0
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "UCR_target": "<= 10-15%",
        "MAA_target": ">= 80%",
        "PCR_target": ">= 60%",
        "PCR": pcr,
        "claim_ratio_reductions": claim_ratio_reductions,
        "by_condition": condition_rows,
        "sensitivity": sensitivity_rows,
    }
    return summary


def write_markdown(path, summary):
    lines = []
    lines.append("# 推薦理由忠實度分析")
    lines.append("")
    lines.append("本分析固定 `exp_01` checkpoint，透過移除不同模態輸入，觀察推薦理由中的主張是否仍有輸入依據。")
    lines.append("主張標籤由設定的規則式判官產生。")
    lines.append("")

    lines.append("## 各條件指標")
    lines.append("")
    has_metadata = any("metadata_claim_ratio" in row for row in summary["by_condition"])
    if has_metadata:
        lines.append("| Condition | Claims | UCR | MAA | Video claims | Audio claims | Metadata claims | Prompt claims | Preference claims |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| Condition | Claims | UCR | MAA | Video claims | Audio claims | Prompt claims | Preference claims |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["by_condition"]:
        if has_metadata:
            lines.append(
                f"| {row['condition']} | {row['n_claims']} | {pct(row['UCR'])}% | "
                f"{pct(row['MAA'])}% | {pct(row['video_claim_ratio'])}% | "
                f"{pct(row['audio_claim_ratio'])}% | {pct(row.get('metadata_claim_ratio', 0.0))}% | "
                f"{pct(row['prompt_claim_ratio'])}% | {pct(row['preference_claim_ratio'])}% |"
            )
        else:
            lines.append(
                f"| {row['condition']} | {row['n_claims']} | {pct(row['UCR'])}% | "
                f"{pct(row['MAA'])}% | {pct(row['video_claim_ratio'])}% | "
                f"{pct(row['audio_claim_ratio'])}% | {pct(row['prompt_claim_ratio'])}% | "
                f"{pct(row['preference_claim_ratio'])}% |"
            )

    lines.append("")
    lines.append("## 敏感度")
    lines.append("")
    lines.append("| Condition | n | ESS mean (Jaccard distance from full) |")
    lines.append("|---|---:|---:|")
    for row in summary["sensitivity"]:
        lines.append(f"| {row['condition']} | {row['n_pairs']} | {row['ESS_jaccard_distance_mean']:.4f} |")

    reductions = summary["claim_ratio_reductions"]
    lines.append("")
    lines.append("## 主要下降指標")
    lines.append("")
    lines.append(f"- 移除 `z_ltp` 後的 PCR：{pct(summary['PCR'])}%")
    lines.append(f"- 移除 video 後的 video-claim reduction：{pct(reductions['video_claim_reduction_after_wo_video'])}%")
    if "audio_claim_reduction_after_wo_audio_feature_only" in reductions:
        lines.append(
            "- 僅移除 audio feature 後的 audio-claim reduction："
            f"{pct(reductions['audio_claim_reduction_after_wo_audio_feature_only'])}%"
        )
    if "audio_claim_reduction_after_wo_audio_all" in reductions:
        lines.append(
            "- 同時移除 audio feature 與音樂 metadata 後的 audio-claim reduction："
            f"{pct(reductions['audio_claim_reduction_after_wo_audio_all'])}%"
        )
    if "audio_claim_reduction_after_wo_audio" in reductions:
        lines.append(f"- 移除 audio 後的 audio-claim reduction：{pct(reductions['audio_claim_reduction_after_wo_audio'])}%")
    lines.append(f"- 移除 prompt 後的 prompt-claim reduction：{pct(reductions['prompt_claim_reduction_after_wo_prompt'])}%")
    lines.append("")
    lines.append("## 解讀注意事項")
    lines.append("")
    lines.append("- UCR 越低越好；本研究用來觀察不受輸入支持的主張比例。")
    lines.append("- MAA 越高越好；本研究用來觀察移除模態後，仍被保留的主張是否仍有可用依據。")
    lines.append("- PCR 越高越好；本研究用來觀察偏好相關主張是否會隨 `P_ltp` 移除而下降。")
    lines.append("- `wo_audio_feature_only` 只移除音樂特徵，仍保留 title/artist metadata。")
    lines.append("- `wo_audio_all` 是較嚴格的反事實條件，會同時移除音樂特徵與 title/artist metadata。")
    lines.append("- 本檔為規則式初步分析摘要；正式解讀仍需搭配論文中的人工查核或 LLM-as-a-Judge 驗證。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if not os.path.exists(GENERATION_CSV):
        raise FileNotFoundError(f"Missing: {GENERATION_CSV}")
    if not os.path.exists(CLAIM_CSV):
        raise FileNotFoundError(f"Missing: {CLAIM_CSV}")

    generation_rows = read_csv(GENERATION_CSV)
    claim_rows = read_csv(CLAIM_CSV)

    condition_rows = summarize_claims(claim_rows)
    sensitivity_rows = summarize_sensitivity(generation_rows)
    summary = build_summary(condition_rows, sensitivity_rows)

    write_csv(OUTPUT_CONDITION_CSV, condition_rows)
    write_csv(OUTPUT_SENSITIVITY_CSV, sensitivity_rows)
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_markdown(OUTPUT_SUMMARY_MD, summary)

    print(f"Saved: {OUTPUT_CONDITION_CSV}")
    print(f"Saved: {OUTPUT_SENSITIVITY_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY_JSON}")
    print(f"Saved: {OUTPUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
