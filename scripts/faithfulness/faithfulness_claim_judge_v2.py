# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Rule-based claim-level judge v2 for explanation faithfulness.

Why v2:
  - v1 over-counted generic phrases such as "fit your video" as video claims.
  - v1 mixed title/artist/album metadata with true audio-feature claims.
  - v2 separates metadata-supported, audio-feature-supported, video-supported,
    prompt-supported, preference-supported, general-supported, and unsupported.

This script is still deterministic and auditable. It is intended as a stronger
rule-based proxy before optional manual or LLM-as-a-Judge auditing.
"""

import csv
import json
import os
import re
from datetime import datetime


BASE_DIR = str(PROJECT_ROOT)
ANALYSIS_DIR = os.path.join(BASE_DIR, "checkpoints", "faithfulness_analysis")
INPUT_CSV = os.path.join(ANALYSIS_DIR, "counterfactual_generations.csv")
OUTPUT_CSV = os.path.join(ANALYSIS_DIR, "claim_annotations_v2.csv")
OUTPUT_JSONL = os.path.join(ANALYSIS_DIR, "claim_annotations_v2.jsonl")
OUTPUT_SUMMARY = os.path.join(ANALYSIS_DIR, "claim_judge_summary_v2.json")


SOURCE_VIDEO = "video-supported"
SOURCE_AUDIO = "audio-supported"
SOURCE_METADATA = "metadata-supported"
SOURCE_PROMPT = "prompt-supported"
SOURCE_PREF = "preference-supported"
SOURCE_GENERAL = "general-supported"
SOURCE_UNSUPPORTED = "unsupported"


REMOVED_SOURCES_BY_CONDITION = {
    "full": set(),
    "wo_video": {SOURCE_VIDEO},
    "wo_audio": {SOURCE_AUDIO},
    "wo_audio_feature_only": {SOURCE_AUDIO},
    "wo_audio_all": {SOURCE_AUDIO, SOURCE_METADATA},
    "wo_prompt": {SOURCE_PROMPT},
    "wo_ltp": {SOURCE_PREF},
}


STRONG_VIDEO_TERMS = {
    "visual", "visuals", "scene", "scenes", "footage", "camera", "shot",
    "shots", "frame", "frames", "screen", "imagery", "animation",
    "animated", "landscape", "cityscape", "outdoor", "indoor", "lighting",
    "bright visuals", "dark visuals", "cinematic", "travel", "vlog",
    "sports", "gameplay", "quest", "storyline", "dance moves",
    "dancing scene", "walking scene", "running scene",
}

VIDEO_CONTEXT_TERMS = {
    "video content", "visual content", "your scene", "your visuals",
    "the scene", "the visuals", "on-screen", "screen action",
}

STRONG_AUDIO_TERMS = {
    "rhythm", "beat", "beats", "tempo", "bpm", "upbeat", "slow tempo",
    "fast tempo", "fast-paced", "mid-tempo", "energetic", "calm",
    "melody", "melodic", "harmony", "harmonic", "bass", "drum", "drums",
    "guitar", "piano", "synth", "synthesizer", "vocal", "vocals",
    "instrument", "instrumental", "sound", "sonic", "tone", "timbre",
    "texture", "chorus", "groove", "acoustic", "loud", "soft",
    "danceable", "techno", "electronic beat", "male vocals",
    "female vocals", "guitar-driven", "guitar focused",
}

GENRE_TERMS = {
    "rock", "pop", "hip-hop", "hip hop", "r&b", "rnb", "electronic",
    "indie", "folk", "jazz", "classical", "rap", "soul", "metal",
    "ambient", "house", "electro", "dance", "punk", "alternative",
    "hard rock", "heavy metal", "chillout",
}

METADATA_TERMS = {
    "album", "artist", "singer", "band", "released", "release", "recorded",
    "live performance", "theme song", "directed by", "featuring", "feat.",
    "ft.", "by ",
}

PROMPT_KEYWORDS = {
    "request", "asked", "looking for", "searching for", "you want",
    "you wanted", "you're looking", "your requested", "as requested",
    "current preference", "preference in this request",
}

PREFERENCE_KEYWORDS = {
    "preference", "preferences", "prefer", "prefers", "taste", "long-term",
    "long term", "history", "historical", "usually", "typically", "often",
    "consistent with your", "matches your preference", "aligns with your",
    "your style", "your musical taste", "your past", "you enjoy", "you like",
    "your affinity", "your inclination",
}

GENERIC_RECOMMENDATION_PATTERNS = {
    "recommend", "suitable", "fit", "fitting", "choice", "match", "matches",
    "complement", "enhance", "background", "mood", "atmosphere",
    "perfect choice", "great choice", "good choice",
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


def norm_for_match(text):
    text = normalize_text(text).lower()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def split_claims(text):
    text = normalize_text(text)
    if not text:
        return []

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


def contains_value(claim, value):
    if not value:
        return False
    claim_norm = norm_for_match(claim)
    value_norm = norm_for_match(value)
    return bool(value_norm and value_norm in claim_norm)


def row_title_artist(row):
    title = row.get("music_title") or ""
    artist = row.get("music_artist") or ""
    return title, artist


def is_metadata_claim(claim, row):
    title, artist = row_title_artist(row)
    if contains_value(claim, title) or contains_value(claim, artist):
        return True
    lower = claim.lower()
    if keyword_hit(lower, METADATA_TERMS):
        return True
    # "X by Y" recommendation clauses usually rely on injected title/artist,
    # not audio features.
    if re.search(r"['\"].+?['\"]\s+by\s+", claim) or re.search(r"\bby\s+[A-Z\u00C0-\uFFFF]", claim):
        return True
    return False


def is_video_claim(claim):
    lower = claim.lower()
    if keyword_hit(lower, VIDEO_CONTEXT_TERMS):
        return True
    if keyword_hit(lower, STRONG_VIDEO_TERMS):
        return True
    # The word "video" alone is often generic recommendation framing. Count it
    # as visual only when paired with concrete content/visual words.
    if "video" in lower and any(term in lower for term in ["visual", "scene", "content", "footage", "image", "action", "pace"]):
        return True
    return False


def is_audio_claim(claim):
    lower = claim.lower()
    if keyword_hit(lower, STRONG_AUDIO_TERMS):
        return True
    # Genre-only statements are treated as audio claims only when they describe
    # the track's sound/style, not title/artist/album metadata.
    if keyword_hit(lower, GENRE_TERMS) and any(anchor in lower for anchor in [
        "genre", "genres", "style", "sound", "influence", "influences",
        "combines elements", "blend", "falls under",
    ]):
        return True
    return False


def classify_claim(claim, row):
    lower = claim.lower()

    if keyword_hit(lower, PREFERENCE_KEYWORDS):
        return SOURCE_PREF, "preference"
    if keyword_hit(lower, PROMPT_KEYWORDS):
        return SOURCE_PROMPT, "prompt"
    if is_metadata_claim(claim, row):
        return SOURCE_METADATA, "metadata"
    if is_video_claim(claim):
        return SOURCE_VIDEO, "video"
    if is_audio_claim(claim):
        return SOURCE_AUDIO, "audio"
    if keyword_hit(lower, GENERIC_RECOMMENDATION_PATTERNS):
        return SOURCE_GENERAL, "general"
    return SOURCE_UNSUPPORTED, "unknown"


def support_status(condition, support_source):
    removed = REMOVED_SOURCES_BY_CONDITION.get(condition, set())
    if support_source == SOURCE_UNSUPPORTED:
        return False, "no_detected_support_source"
    if support_source in removed:
        return False, f"source_removed_by_{condition}"
    return True, "source_available"


def main():
    ensure_dir(ANALYSIS_DIR)
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Run counterfactual generation first: {INPUT_CSV}")

    generation_rows = read_csv(INPUT_CSV)
    claim_rows = []

    for row in generation_rows:
        condition = row["condition"]
        generated_text = row.get("generated_text", "")
        claims = split_claims(generated_text)
        if not claims:
            claims = ["<EMPTY_GENERATION>"]

        for claim_id, claim in enumerate(claims, start=1):
            support_source, claim_type = classify_claim(claim, row)
            is_supported, unsupported_reason = support_status(condition, support_source)
            removed_sources = sorted(REMOVED_SOURCES_BY_CONDITION.get(condition, set()))
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
                "removed_source": ";".join(removed_sources),
                "is_supported": int(is_supported),
                "unsupported_reason": unsupported_reason,
                "judge_version": "rule_based_keyword_v2_metadata_aware",
                "generated_text": generated_text,
            })

    write_csv(OUTPUT_CSV, claim_rows)
    write_jsonl(OUTPUT_JSONL, claim_rows)

    total = len(claim_rows)
    unsupported = sum(1 for r in claim_rows if int(r["is_supported"]) == 0)
    by_source = {}
    for row in claim_rows:
        by_source[row["support_source"]] = by_source.get(row["support_source"], 0) + 1
    summary = {
        "input_csv": INPUT_CSV,
        "n_generation_rows": len(generation_rows),
        "n_claims": total,
        "unsupported_claims": unsupported,
        "unsupported_claim_rate": unsupported / max(total, 1),
        "judge": "rule_based_keyword_v2_metadata_aware",
        "source_counts": by_source,
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
