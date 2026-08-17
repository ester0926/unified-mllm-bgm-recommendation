# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
LLM-as-a-Judge helper for explanation faithfulness analysis.

Usage:
  1. Make sure Ollama is running locally.
  2. Edit USER SETTINGS below if needed.
  3. Open this file in VSCode and click Run.

This script samples a manageable subset from the rule-based faithfulness outputs
and asks a local LLM judge to verify three analyses:
  1. Feature-erasure claim support
  2. Counterfactual preference alignment
  3. Metadata consistency

The goal is not to replace the full deterministic analysis, but to provide an
auditable LLM-assisted validation subset for the thesis.
"""

import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = str(PROJECT_ROOT)
ANALYSIS_DIR = os.path.join(BASE_DIR, "checkpoints", "faithfulness_analysis")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3:8b"  # Change to an installed Ollama model if needed.
REQUEST_TIMEOUT_SEC = 180
MAX_RETRIES = 2
SLEEP_BETWEEN_CALLS_SEC = 0.2

SAMPLE_SEED = 20260523
N_FEATURE_ERASURE_CLAIMS = 120
N_PREFERENCE_ROWS = 80
N_METADATA_CLAIMS = 120

RUN_FEATURE_ERASURE_JUDGE = True
RUN_PREFERENCE_JUDGE = True
RUN_METADATA_JUDGE = True
RESUME_EXISTING_OUTPUTS = True

OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "llm_judge")
OUTPUT_FEATURE_CSV = os.path.join(OUTPUT_DIR, "llm_judge_feature_erasure.csv")
OUTPUT_PREFERENCE_CSV = os.path.join(OUTPUT_DIR, "llm_judge_preference_counterfactual.csv")
OUTPUT_METADATA_CSV = os.path.join(OUTPUT_DIR, "llm_judge_metadata_consistency.csv")
OUTPUT_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "llm_judge_summary.json")
OUTPUT_SUMMARY_MD = os.path.join(OUTPUT_DIR, "llm_judge_summary.md")

FEATURE_CLAIMS_CSV = os.path.join(ANALYSIS_DIR, "claim_annotations.csv")
PREFERENCE_ANALYSIS_CSV = os.path.join(ANALYSIS_DIR, "preference_counterfactual_analysis.csv")
METADATA_CLAIMS_CSV = os.path.join(ANALYSIS_DIR, "metadata_consistency_claims.csv")


SOURCE_LABELS = {
    "video-supported",
    "audio-supported",
    "prompt-supported",
    "preference-supported",
    "general-supported",
    "unsupported",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path, row):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def sample_rows(rows, n, seed, stratify_key=None):
    rng = random.Random(seed)
    if n is None or n >= len(rows):
        return list(rows)

    if not stratify_key:
        return rng.sample(rows, n)

    by_key = defaultdict(list)
    for row in rows:
        by_key[row.get(stratify_key, "")].append(row)

    selected = []
    keys = sorted(by_key)
    per_key = max(1, n // max(len(keys), 1))
    for key in keys:
        group = by_key[key]
        selected.extend(rng.sample(group, min(per_key, len(group))))

    remaining = [r for r in rows if r not in selected]
    if len(selected) < n and remaining:
        selected.extend(rng.sample(remaining, min(n - len(selected), len(remaining))))
    return selected[:n]


def extract_json_object(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:300]}")
    json_text = match.group(0)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Local LLMs occasionally return invalid JSON string escapes such as
        # "\&" or "\'". Keep valid JSON escapes intact and double other
        # backslashes so the object remains parseable.
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_text)
        return json.loads(fixed)


def fallback_feature_result(row, error):
    return {
        **row,
        "llm_support_source": "judge_error",
        "llm_is_supported": "",
        "llm_reason": f"judge_error: {error}"[:500],
        "llm_raw": "",
        "rule_support_source": row.get("support_source", ""),
        "rule_is_supported": row.get("is_supported", ""),
        "agree_support_source": "",
        "agree_is_supported": "",
    }


def fallback_preference_result(row, error):
    return {
        **row,
        "llm_is_aligned": "",
        "llm_has_conflict": "",
        "llm_alignment_strength": "judge_error",
        "llm_reason": f"judge_error: {error}"[:500],
        "llm_raw": "",
        "rule_is_aligned": row.get("is_aligned", ""),
        "rule_has_conflict": row.get("has_conflict", ""),
        "agree_is_aligned": "",
        "agree_has_conflict": "",
    }


def fallback_metadata_result(row, error):
    return {
        **row,
        "llm_is_metadata_supported": "",
        "llm_unsupported_aspects": "judge_error",
        "llm_reason": f"judge_error: {error}"[:500],
        "llm_raw": "",
        "rule_is_metadata_supported": row.get("is_metadata_supported", ""),
        "agree_is_metadata_supported": "",
    }


def row_identity(row, fields):
    return "||".join(str(row.get(field, "")) for field in fields)


def load_done_identities(path, fields):
    if not RESUME_EXISTING_OUTPUTS or not os.path.exists(path):
        return set()
    try:
        return {row_identity(row, fields) for row in read_csv(path)}
    except Exception:
        return set()


def ollama_chat(system_prompt, user_prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")
            return extract_json_object(content), content
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.0 + attempt)
            else:
                raise RuntimeError(f"Ollama judge failed after retries: {exc}") from exc
    raise RuntimeError(f"Ollama judge failed: {last_error}")


def judge_feature_erasure(row):
    condition = row.get("condition", "")
    removed_source = row.get("removed_source", "")
    claim_text = row.get("claim_text", "")
    generated_text = row.get("generated_text", "")

    system_prompt = (
        "You are a careful academic annotator for explanation faithfulness. "
        "Judge whether a claim in a generated music recommendation explanation "
        "is supported by the available input modalities. Return only JSON."
    )
    user_prompt = f"""
Condition: {condition}
Removed source: {removed_source or "none"}

Available source labels:
- video-supported: claim depends on visual/video content.
- audio-supported: claim depends on music sound, genre, rhythm, mood, tempo, instruments, vocals, title/artist/album.
- prompt-supported: claim depends on the user's current request text.
- preference-supported: claim depends on long-term user preference/history.
- general-supported: generic recommendation wording that does not require a specific modality.
- unsupported: specific claim that is not supported or cannot be verified.

Generated explanation:
{generated_text}

Claim to judge:
{claim_text}

Return this JSON exactly:
{{
  "support_source": "one of video-supported/audio-supported/prompt-supported/preference-supported/general-supported/unsupported",
  "is_supported": true or false,
  "reason": "short reason, under 30 words"
}}
"""
    result, raw = ollama_chat(system_prompt, user_prompt)
    support_source = result.get("support_source", "unsupported")
    if support_source not in SOURCE_LABELS:
        support_source = "unsupported"
    is_supported = bool(result.get("is_supported", False))
    return {
        **row,
        "llm_support_source": support_source,
        "llm_is_supported": int(is_supported),
        "llm_reason": str(result.get("reason", ""))[:500],
        "llm_raw": raw,
        "rule_support_source": row.get("support_source", ""),
        "rule_is_supported": row.get("is_supported", ""),
        "agree_support_source": int(support_source == row.get("support_source", "")),
        "agree_is_supported": int(int(is_supported) == safe_int(row.get("is_supported"))),
    }


def judge_preference(row):
    variant = row.get("preference_variant", "")
    generated_text = row.get("generated_text", "")
    positive_hits = row.get("positive_hits", "")
    negative_hits = row.get("negative_hits", "")

    if variant == "cf_upbeat_electronic":
        preference = "upbeat electronic music with bright mood, energetic rhythm, modern dance/pop feeling"
    elif variant == "cf_lyrical_piano":
        preference = "lyrical piano music with slow tempo, soft acoustic texture, gentle melody, calm sentimental mood"
    else:
        preference = variant

    system_prompt = (
        "You are a careful academic annotator. Judge whether the generated "
        "recommendation explanation aligns with the counterfactual user preference. "
        "Return only JSON."
    )
    user_prompt = f"""
Counterfactual preference:
{preference}

Generated explanation:
{generated_text}

Rule-based positive hits: {positive_hits}
Rule-based conflicting hits: {negative_hits}

Return this JSON exactly:
{{
  "is_aligned": true or false,
  "has_conflict": true or false,
  "alignment_strength": "strong/moderate/weak/none",
  "reason": "short reason, under 30 words"
}}
"""
    result, raw = ollama_chat(system_prompt, user_prompt)
    is_aligned = bool(result.get("is_aligned", False))
    has_conflict = bool(result.get("has_conflict", False))
    return {
        **row,
        "llm_is_aligned": int(is_aligned),
        "llm_has_conflict": int(has_conflict),
        "llm_alignment_strength": str(result.get("alignment_strength", ""))[:50],
        "llm_reason": str(result.get("reason", ""))[:500],
        "llm_raw": raw,
        "rule_is_aligned": row.get("is_aligned", ""),
        "rule_has_conflict": row.get("has_conflict", ""),
        "agree_is_aligned": int(int(is_aligned) == safe_int(row.get("is_aligned"))),
        "agree_has_conflict": int(int(has_conflict) == safe_int(row.get("has_conflict"))),
    }


def judge_metadata(row):
    claim_text = row.get("claim_text", "")
    top1_reference = row.get("top1_reference_text", "")
    music_terms = row.get("claim_terms", "")
    unsupported_terms = row.get("unsupported_terms", "")

    system_prompt = (
        "You are a careful academic annotator for metadata consistency. "
        "Judge whether a music-detail claim is supported by the candidate track metadata/reference. "
        "Return only JSON."
    )
    user_prompt = f"""
Candidate track metadata/reference text:
{top1_reference}

Generated explanation claim:
{claim_text}

Rule-based detected music terms: {music_terms}
Rule-based unsupported terms: {unsupported_terms}

Question: Is the music-detail claim supported by the candidate metadata/reference?

Return this JSON exactly:
{{
  "is_metadata_supported": true or false,
  "unsupported_aspects": ["short aspect 1", "short aspect 2"],
  "reason": "short reason, under 30 words"
}}
"""
    result, raw = ollama_chat(system_prompt, user_prompt)
    is_supported = bool(result.get("is_metadata_supported", False))
    unsupported_aspects = result.get("unsupported_aspects", [])
    if not isinstance(unsupported_aspects, list):
        unsupported_aspects = [str(unsupported_aspects)]
    return {
        **row,
        "llm_is_metadata_supported": int(is_supported),
        "llm_unsupported_aspects": ";".join(str(x) for x in unsupported_aspects)[:500],
        "llm_reason": str(result.get("reason", ""))[:500],
        "llm_raw": raw,
        "rule_is_metadata_supported": row.get("is_metadata_supported", ""),
        "agree_is_metadata_supported": int(int(is_supported) == safe_int(row.get("is_metadata_supported"))),
    }


def summarize_binary(rows, llm_key, rule_key=None):
    valid = [r for r in rows if str(r.get(llm_key, "")).strip() != ""]
    n = len(valid)
    positives = sum(safe_int(r.get(llm_key)) for r in valid)
    out = {
        "n": len(rows),
        "n_valid": n,
        "n_errors": len(rows) - n,
        f"{llm_key}_rate": positives / max(n, 1),
    }
    if rule_key:
        agreements = sum(1 for r in valid if safe_int(r.get(llm_key)) == safe_int(r.get(rule_key)))
        out["agreement_rate"] = agreements / max(n, 1)
    return out


def run_feature_erasure():
    rows = read_csv(FEATURE_CLAIMS_CSV)
    # Favor informative counterfactual claims while keeping condition diversity.
    rows = [r for r in rows if r.get("claim_text") != "<EMPTY_GENERATION>"]
    sampled = sample_rows(rows, N_FEATURE_ERASURE_CLAIMS, SAMPLE_SEED, stratify_key="condition")
    identity_fields = ["condition", "sample_idx", "video_id", "claim_id", "claim_text"]
    done = load_done_identities(OUTPUT_FEATURE_CSV, identity_fields)
    judged = read_csv(OUTPUT_FEATURE_CSV) if RESUME_EXISTING_OUTPUTS and os.path.exists(OUTPUT_FEATURE_CSV) else []
    for i, row in enumerate(sampled, start=1):
        if row_identity(row, identity_fields) in done:
            continue
        print(f"[feature {i}/{len(sampled)}] {row.get('condition')} claim_id={row.get('claim_id')}")
        try:
            judged_row = judge_feature_erasure(row)
        except Exception as exc:
            judged_row = fallback_feature_result(row, exc)
            print(f"  judge error, recorded and continuing: {exc}")
        append_csv(OUTPUT_FEATURE_CSV, judged_row)
        judged.append(judged_row)
        done.add(row_identity(row, identity_fields))
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
    return judged


def run_preference():
    rows = read_csv(PREFERENCE_ANALYSIS_CSV)
    sampled = sample_rows(rows, N_PREFERENCE_ROWS, SAMPLE_SEED + 1, stratify_key="preference_variant")
    identity_fields = ["preference_variant", "sample_idx", "video_id"]
    done = load_done_identities(OUTPUT_PREFERENCE_CSV, identity_fields)
    judged = read_csv(OUTPUT_PREFERENCE_CSV) if RESUME_EXISTING_OUTPUTS and os.path.exists(OUTPUT_PREFERENCE_CSV) else []
    for i, row in enumerate(sampled, start=1):
        if row_identity(row, identity_fields) in done:
            continue
        print(f"[preference {i}/{len(sampled)}] {row.get('preference_variant')}")
        try:
            judged_row = judge_preference(row)
        except Exception as exc:
            judged_row = fallback_preference_result(row, exc)
            print(f"  judge error, recorded and continuing: {exc}")
        append_csv(OUTPUT_PREFERENCE_CSV, judged_row)
        judged.append(judged_row)
        done.add(row_identity(row, identity_fields))
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
    return judged


def run_metadata():
    rows = read_csv(METADATA_CLAIMS_CSV)
    sampled = sample_rows(rows, N_METADATA_CLAIMS, SAMPLE_SEED + 2, stratify_key="exp_name")
    identity_fields = ["exp_name", "sample_idx", "video_id", "claim_id", "claim_text"]
    done = load_done_identities(OUTPUT_METADATA_CSV, identity_fields)
    judged = read_csv(OUTPUT_METADATA_CSV) if RESUME_EXISTING_OUTPUTS and os.path.exists(OUTPUT_METADATA_CSV) else []
    for i, row in enumerate(sampled, start=1):
        if row_identity(row, identity_fields) in done:
            continue
        print(f"[metadata {i}/{len(sampled)}] {row.get('exp_name')} sample={row.get('sample_idx')}")
        try:
            judged_row = judge_metadata(row)
        except Exception as exc:
            judged_row = fallback_metadata_result(row, exc)
            print(f"  judge error, recorded and continuing: {exc}")
        append_csv(OUTPUT_METADATA_CSV, judged_row)
        judged.append(judged_row)
        done.add(row_identity(row, identity_fields))
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
    return judged


def write_summary(feature_rows, preference_rows, metadata_rows):
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "sample_seed": SAMPLE_SEED,
        "feature_erasure": summarize_binary(feature_rows, "llm_is_supported", "rule_is_supported") if feature_rows else None,
        "preference_alignment": summarize_binary(preference_rows, "llm_is_aligned", "rule_is_aligned") if preference_rows else None,
        "preference_conflict": summarize_binary(preference_rows, "llm_has_conflict", "rule_has_conflict") if preference_rows else None,
        "metadata_consistency": summarize_binary(metadata_rows, "llm_is_metadata_supported", "rule_is_metadata_supported") if metadata_rows else None,
        "outputs": {
            "feature_csv": OUTPUT_FEATURE_CSV if feature_rows else None,
            "preference_csv": OUTPUT_PREFERENCE_CSV if preference_rows else None,
            "metadata_csv": OUTPUT_METADATA_CSV if metadata_rows else None,
            "summary_json": OUTPUT_SUMMARY_JSON,
            "summary_md": OUTPUT_SUMMARY_MD,
        },
    }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# LLM-as-a-Judge Faithfulness Validation",
        "",
        f"Model: `{OLLAMA_MODEL}`",
        "",
        "| Task | n | valid | errors | LLM positive rate | Agreement with rule-based judge |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    task_rows = [
        ("Feature-erasure supported claims", summary["feature_erasure"], "llm_is_supported_rate"),
        ("Preference aligned", summary["preference_alignment"], "llm_is_aligned_rate"),
        ("Preference conflict", summary["preference_conflict"], "llm_has_conflict_rate"),
        ("Metadata supported", summary["metadata_consistency"], "llm_is_metadata_supported_rate"),
    ]
    for label, item, rate_key in task_rows:
        if not item:
            continue
        lines.append(
            f"| {label} | {item['n']} | {item.get('n_valid', item['n'])} | "
            f"{item.get('n_errors', 0)} | {item.get(rate_key, 0.0) * 100:.2f}% | "
            f"{item.get('agreement_rate', 0.0) * 100:.2f}% |"
        )
    lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- This is an LLM-assisted validation subset, not the full deterministic analysis.",
        "- Use agreement rates to discuss whether rule-based labels are directionally reliable.",
        "- Low agreement indicates the corresponding rule-based metric should be treated cautiously or manually audited.",
    ])
    with open(OUTPUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary


def main():
    ensure_dir(OUTPUT_DIR)
    feature_rows = []
    preference_rows = []
    metadata_rows = []

    if RUN_FEATURE_ERASURE_JUDGE:
        if not os.path.exists(FEATURE_CLAIMS_CSV):
            raise FileNotFoundError(FEATURE_CLAIMS_CSV)
        feature_rows = run_feature_erasure()

    if RUN_PREFERENCE_JUDGE:
        if not os.path.exists(PREFERENCE_ANALYSIS_CSV):
            raise FileNotFoundError(PREFERENCE_ANALYSIS_CSV)
        preference_rows = run_preference()

    if RUN_METADATA_JUDGE:
        if not os.path.exists(METADATA_CLAIMS_CSV):
            raise FileNotFoundError(METADATA_CLAIMS_CSV)
        metadata_rows = run_metadata()

    summary = write_summary(feature_rows, preference_rows, metadata_rows)
    print(f"Saved: {OUTPUT_SUMMARY_JSON}")
    print(f"Saved: {OUTPUT_SUMMARY_MD}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
