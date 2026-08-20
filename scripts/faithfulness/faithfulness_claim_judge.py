"""
用途：將推薦解釋切成 claim，並依規則標註每個 claim 的支持來源。
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
from datetime import datetime


BASE_DIR = str(PROJECT_ROOT)
ANALYSIS_DIR = os.path.join(BASE_DIR, "checkpoints", "faithfulness_analysis")
INPUT_CSV = os.path.join(ANALYSIS_DIR, "counterfactual_generations.csv")
OUTPUT_CSV = os.path.join(ANALYSIS_DIR, "claim_annotations.csv")
OUTPUT_JSONL = os.path.join(ANALYSIS_DIR, "claim_annotations.jsonl")
OUTPUT_SUMMARY = os.path.join(ANALYSIS_DIR, "claim_judge_summary.json")


SOURCE_VIDEO = "video-supported"
SOURCE_AUDIO = "audio-supported"
SOURCE_PROMPT = "prompt-supported"
SOURCE_PREF = "preference-supported"
SOURCE_GENERAL = "general-supported"
SOURCE_UNSUPPORTED = "unsupported"


REMOVED_SOURCE_BY_CONDITION = {
    "full": None,
    "wo_video": SOURCE_VIDEO,
    "wo_audio": SOURCE_AUDIO,
    "wo_audio_feature_only": SOURCE_AUDIO,
    "wo_audio_all": SOURCE_AUDIO,
    "wo_prompt": SOURCE_PROMPT,
    "wo_ltp": SOURCE_PREF,
}


VIDEO_KEYWORDS = {
    "video", "visual", "scene", "footage", "clip", "camera", "shot", "frame",
    "screen", "gameplay", "gaming", "quest", "dance", "dancing", "walk",
    "walking", "run", "running", "city", "landscape", "nature", "bright",
    "dark", "cinematic", "action", "travel", "vlog", "sports", "outdoor",
    "indoor", "animation", "animated", "story", "visuals", "imagery",
}

AUDIO_KEYWORDS = {
    "rhythm", "beat", "tempo", "upbeat", "slow", "fast", "energetic",
    "calm", "melody", "melodic", "harmony", "harmonic", "bass", "drum",
    "drums", "guitar", "piano", "synth", "synthesizer", "vocal", "vocals",
    "instrument", "instrumental", "sound", "sonic", "tone", "timbre",
    "texture", "genre", "rock", "pop", "hip-hop", "hip hop", "electronic",
    "indie", "folk", "jazz", "classical", "rap", "soul", "metal", "acoustic",
    "track", "song", "music", "album", "artist", "chorus", "groove",
}

PROMPT_KEYWORDS = {
    "request", "asked", "looking for", "searching for", "you want", "you wanted",
    "you're looking", "your requested", "context", "fits your need",
    "as requested", "preference in this request", "current preference",
}

PREFERENCE_KEYWORDS = {
    "preference", "preferences", "prefer", "prefers", "taste", "long-term",
    "long term", "history", "historical", "usually", "typically", "often",
    "consistent with your", "matches your preference", "aligns with your",
    "your style", "your musical taste", "your past", "you enjoy", "you like",
    "your affinity", "your inclination",
}

GENERIC_RECOMMENDATION_KEYWORDS = {
    "recommend", "suitable", "fit", "fitting", "choice", "match", "matches",
    "complement", "enhance", "background", "mood", "atmosphere",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def split_claims(text):
    text = normalize_text(text)
    if not text:
        return []

    # 先依句子邊界切分，再將過長句子切成子句。
    sentence_parts = re.split(r"(?<=[.!?])\s+", text)
    claims = []
    for sent in sentence_parts:
        sent = sent.strip(" ;")
        if not sent:
            continue
        clause_parts = re.split(r"\s+(?:and|while|because|as well as|, which|;)\s+", sent)
        for clause in clause_parts:
            clause = clause.strip(" ,;")
            if len(clause.split()) < 4:
                continue
            claims.append(clause)
    return claims


def keyword_hit(text, keywords):
    lower = text.lower()
    return any(k in lower for k in keywords)


def classify_claim(claim):
    lower = claim.lower()

    # 偏好 claim 優先處理，因為這類句子常包含音訊詞，
    # 例如曲風名稱，同時又是在描述使用者偏好。
    if keyword_hit(lower, PREFERENCE_KEYWORDS):
        return SOURCE_PREF, "preference"
    if keyword_hit(lower, PROMPT_KEYWORDS):
        return SOURCE_PROMPT, "prompt"
    if keyword_hit(lower, VIDEO_KEYWORDS):
        return SOURCE_VIDEO, "video"
    if keyword_hit(lower, AUDIO_KEYWORDS):
        return SOURCE_AUDIO, "audio"
    if keyword_hit(lower, GENERIC_RECOMMENDATION_KEYWORDS):
        return SOURCE_GENERAL, "general"
    return SOURCE_UNSUPPORTED, "unknown"


def support_status(condition, support_source):
    removed = REMOVED_SOURCE_BY_CONDITION.get(condition)
    if support_source == SOURCE_UNSUPPORTED:
        return False, "no_detected_support_source"
    if removed is not None and support_source == removed:
        return False, f"source_removed_by_{condition}"
    return True, "source_available"


def main():
    ensure_dir(ANALYSIS_DIR)
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Run run_faithfulness_counterfactual.py first: {INPUT_CSV}")

    generation_rows = read_csv(INPUT_CSV)
    claim_rows = []

    for row in generation_rows:
        condition = row["condition"]
        generated_text = row.get("generated_text", "")
        claims = split_claims(generated_text)
        if not claims:
            claims = ["<EMPTY_GENERATION>"]

        for claim_id, claim in enumerate(claims, start=1):
            support_source, claim_type = classify_claim(claim)
            is_supported, unsupported_reason = support_status(condition, support_source)
            removed_source = REMOVED_SOURCE_BY_CONDITION.get(condition) or ""
            claim_rows.append({
                "sample_idx": row["sample_idx"],
                "video_id": row["video_id"],
                "gt_music_id": row["gt_music_id"],
                "condition": condition,
                "condition_description": row.get("condition_description", ""),
                "claim_id": claim_id,
                "claim_text": claim,
                "claim_type": claim_type,
                "support_source": support_source,
                "removed_source": removed_source,
                "is_supported": int(is_supported),
                "unsupported_reason": unsupported_reason,
                "generated_text": generated_text,
            })

    write_csv(OUTPUT_CSV, claim_rows)
    write_jsonl(OUTPUT_JSONL, claim_rows)

    total = len(claim_rows)
    unsupported = sum(1 for r in claim_rows if int(r["is_supported"]) == 0)
    summary = {
        "input_csv": INPUT_CSV,
        "n_generation_rows": len(generation_rows),
        "n_claims": total,
        "unsupported_claims": unsupported,
        "unsupported_claim_rate": unsupported / max(total, 1),
        "judge": "rule_based_keyword_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "csv": OUTPUT_CSV,
            "jsonl": OUTPUT_JSONL,
        },
    }
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY}")
    print(f"Claims: {total}, unsupported: {unsupported} ({summary['unsupported_claim_rate']:.2%})")


if __name__ == "__main__":
    main()
