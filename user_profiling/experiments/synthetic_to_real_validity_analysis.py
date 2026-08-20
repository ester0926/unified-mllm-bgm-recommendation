"""
用途：比較 synthetic proxy 與真實偏好資料的可用性指標。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


BASE_DIR = Path(r"data/user_profiling")
DEFAULT_PROFILE = BASE_DIR / "long_term_preference/stage4_recLLM/profiles.jsonl"
DEFAULT_HISTORY_DIR = BASE_DIR / "long_term_preference/stage2_history/personax"
DEFAULT_METADATA = BASE_DIR / "music_metadata_simple/music_metadata_enriched.json"
DEFAULT_DIALOGUES = BASE_DIR / "long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl"
DEFAULT_OUT_DIR = BASE_DIR / "experiments/synthetic_validity_outputs"


# 使用前可調整下列設定；也可以改用命令列參數覆蓋。
RUN_CONFIG = {
    "profile_files": [DEFAULT_PROFILE],
    "history_dir": DEFAULT_HISTORY_DIR,
    "metadata": DEFAULT_METADATA,
    "dialogues": DEFAULT_DIALOGUES,
    "out_dir": DEFAULT_OUT_DIR,
    # 設為 None 或 0 會分析全部 profile；500/1000 適合產生論文表格。
    # 若只想快速檢查，可設為 50。
    "sample_size": 1000,
    "seed": 20260523,
    "background_max_items": 10000,
    "human_packet_size": 30,
    # 可選：推薦理由檔，支援 .csv、.json、.jsonl。
    # 讀取時會尋找 music_id/target_music 與 reason/rationale/
    # explanation/generated_reason 等欄位。
    "recommendation_file": None,
    # 可加入既有產生的 profile 檔，用於 multi-generator 穩定性分析。
    # 這樣可以保留已完成的 Gemma 輸出，只重跑 llama3:8b。
    "existing_stability_profile_files": [
        DEFAULT_OUT_DIR / "stage4_profiles_gemma3_12b.jsonl",
    ],
    # 從既有 Gemma 穩定性 profile 檔取前面的有效 ID。
    # 這可確保 llama3:8b 產生的是已有 Gemma profile 的案例，
    # 讓穩定性比較維持預期樣本數。
    "stability_sample_from_existing": True,
    # 完整 multi-generator 穩定性模式會對每個案例、每個模型呼叫一次 Ollama，
    # 並透過輸出 JSONL 支援中斷續跑。
    "generate_stability_profiles": True,
    "stability_generation_sample_size": 300, #1000
    "stability_generators": [
        # Gemma 已經產生過，因此預設註解此區塊，
        # 讓重跑時直接從 llama3:8b 開始。
        # {
        #     "model": "gemma3:12b",
        #     "output_file": DEFAULT_OUT_DIR / "stage4_profiles_gemma3_12b.jsonl",
        #     "temperature": 0.1,
        #     "num_ctx": 8192,
        #     "max_retries": 3,
        #     "json_format": True,
        # },
        {
            # 若本機 Ollama 使用不同 LLaMA/Qwen 模型名稱，請改這裡。
            "model": "llama3:8b",
            "output_file": DEFAULT_OUT_DIR / "stage4_profiles_llama3_8b.jsonl",
            "temperature": 0.1,
            "num_ctx": 8192,
            "max_retries": 3,
            "json_format": True,
        },
    ],
    # 專家填完 human_annotation_template.csv 後，請將此處指向該檔。
    "human_ratings": None,
}


# 這是一份刻意保持透明的音樂屬性詞表，方便附錄與人工檢查。
# 若資料集中有反覆出現的領域詞，可在此補充。
ATTRIBUTE_LEXICON: Dict[str, List[str]] = {
    "pop": ["pop"],
    "rock": ["rock", "guitar", "riff", "headbang", "gritty"],
    "electronic": ["electronic", "electronica", "edm", "synth", "synths", "synthesizer"],
    "techno": ["techno"],
    "hip_hop": ["hip-hop", "hip hop", "rap", "rapper"],
    "rnb": ["rnb", "r&b", "soul"],
    "jazz": ["jazz", "swing", "saxophone", "sax"],
    "classical": ["classical", "orchestral", "orchestra", "piano", "violin"],
    "folk": ["folk", "acoustic"],
    "metal": ["metal", "heavy metal"],
    "dance": ["dance", "club"],
    "house": ["house"],
    "ambient": ["ambient", "soundscape", "soundscapes", "atmospheric"],
    "vocal": ["vocal", "vocals", "voice", "singing"],
    "male_vocal": ["male vocal", "male vocals", "male voice", "male singer"],
    "female_vocal": ["female vocal", "female vocals", "female voice", "female singer"],
    "fast": ["fast", "fast-paced", "uptempo", "upbeat", "high energy", "energetic"],
    "slow": ["slow", "downtempo", "relaxed tempo", "slower", "laid-back", "chill"],
    "beat": ["beat", "beats", "rhythm", "rhythmic", "groove"],
    "drums": ["drum", "drums", "percussion", "percussive"],
    "melodic": ["melody", "melodies", "melodic"],
    "intense": ["intense", "aggressive", "powerful", "driving"],
    "smooth": ["smooth", "mellow", "softer", "calm", "calmer"],
    "simple": ["simple", "minimal", "sparse"],
    "complex": ["complex", "layered", "busy", "dense"],
}

NEGATION_HINTS = [
    "dislike",
    "dislikes",
    "does not like",
    "not like",
    "avoid",
    "avoids",
    "not appealing",
    "overwhelming",
    "too intense",
    "not a fan",
]


@dataclass
class ProfileRecord:
    music_id: str
    summary_text: str
    salient_facts: List[dict]
    source_file: str


@dataclass
class CaseMetrics:
    music_id: str
    profile_attr_count: int
    supported_attr_count: int
    hallucinated_attr_count: int
    grounding_ratio: float
    positive_alignment: float
    exploratory_alignment: float
    negative_preference_overlap: float
    negative_dislike_alignment: float
    discrimination_margin: float
    pos_vs_neg_win_rate: float
    within_positive_coherence: float
    random_background_coherence: float
    coherence_lift: float
    profile_naturalness_proxy: float
    profile_consistency_proxy: float
    profile_attrs: str
    supported_attrs: str
    hallucinated_attrs: str


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_profiles(path: Path, sample_size: Optional[int], seed: int) -> Dict[str, ProfileRecord]:
    rows: List[ProfileRecord] = []
    for obj in read_jsonl(path):
        mid = obj.get("music_id") or obj.get("target_music")
        if not mid:
            continue
        summary_text = str(obj.get("summary_text", ""))
        salient_facts = obj.get("salient_facts", []) if isinstance(obj.get("salient_facts"), list) else []
        if not summary_text or not salient_facts:
            continue
        rows.append(
            ProfileRecord(
                music_id=str(mid),
                summary_text=summary_text,
                salient_facts=salient_facts,
                source_file=str(path),
            )
        )
    if sample_size and len(rows) > sample_size:
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)
    return {r.music_id: r for r in rows}


def clean_json_response(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    text = re.sub(r'\\(?!["\\/bfnrtu])', "", text)
    return extract_json_object(text.strip())


def extract_json_object(text: str) -> str:
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text


def is_valid_profile_obj(obj: dict) -> bool:
    return bool(obj.get("music_id") and obj.get("summary_text") and obj.get("salient_facts"))


def build_stage4_profile_prompt(dialogues: List[dict]) -> str:
    by_type = defaultdict(list)
    for d in dialogues:
        by_type[d.get("dialogue_type")].append(d)

    dialogue_sections = []
    for dtype in ["Positive", "Exploratory", "Negative"]:
        if dtype in by_type:
            dialogue = by_type[dtype][0]
            turns = dialogue.get("dialogue_turns", [])
            dialogue_text = f"\n## {dtype} Dialogue:\n"
            for turn in turns[:6]:
                role = turn.get("role", "User")
                content = turn.get("content", "")
                dialogue_text += f"{role}: {content}\n"
            dialogue_sections.append(dialogue_text)

    return f"""You are a user preference analyzer. Extract the user's long-term music preferences from these dialogue histories.

{"".join(dialogue_sections)}

TASK: Generate a natural language user profile with salient preference facts.

OUTPUT FORMAT (JSON only, no markdown):
{{
    "summary_text": "The user [comprehensive summary in 80-100 words using third-person]",
    "salient_facts": [
        {{"fact": "The user prefers calm piano music", "conflict_tag": "CONFIRM"}},
        {{"fact": "The user dislikes heavy metal", "conflict_tag": "CONFIRM_DISLIKE"}},
        {{"fact": "The user is exploring jazz styles", "conflict_tag": "NEW"}}
    ]
}}

CONFLICT TAGS:
- CONFIRM: Reinforces existing preference
- CONFIRM_DISLIKE: Confirms dislike/avoidance
- MODULATE: Refines/adjusts preference within taste boundaries
- NEW: Explores new style/element
- OVERRIDE: Contradicts long-term preference (rare in this scenario)

CRITICAL RULES:
1. Use THIRD-PERSON ("The user prefers..." NOT "I prefer...")
2. Summary must be 80-100 words
3. Focus on MUSIC preferences (genres, moods, instruments)
4. Extract 5-8 salient facts. The 'salient_facts' list MUST NOT be empty.
5. Output ONLY valid JSON (no markdown, no backticks)
"""


def load_stage3_dialogues_for_ids(dialogue_path: Path, target_ids: Set[str]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    if not dialogue_path.exists():
        raise FileNotFoundError(f"Dialogue file not found: {dialogue_path}")

    required_types = {"Positive", "Exploratory", "Negative"}
    completed: Set[str] = set()
    for obj in read_jsonl(dialogue_path):
        mid = str(obj.get("music_id", ""))
        if mid not in target_ids or mid in completed:
            continue
        grouped[mid].append(obj)
        found_types = {x.get("dialogue_type") for x in grouped[mid]}
        if found_types >= required_types:
            completed.add(mid)
            if len(completed) == len(target_ids):
                break
    return dict(grouped)


def processed_profile_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {str(obj.get("music_id") or obj.get("target_music")) for obj in read_jsonl(path) if is_valid_profile_obj(obj)}


def valid_profile_ids_in_order(path: Path) -> List[str]:
    if not path.exists():
        return []
    ids = []
    seen = set()
    for obj in read_jsonl(path):
        mid = str(obj.get("music_id") or obj.get("target_music") or "")
        if mid and mid not in seen and is_valid_profile_obj(obj):
            ids.append(mid)
            seen.add(mid)
    return ids


def compact_profile_file(path: Path) -> dict:
    if not path.exists():
        return {"status": "not_found", "valid": 0, "invalid": 0}

    valid_rows = []
    invalid_rows = []
    seen = set()
    for obj in read_jsonl(path):
        mid = str(obj.get("music_id") or obj.get("target_music") or "")
        if is_valid_profile_obj(obj) and mid not in seen:
            valid_rows.append(obj)
            seen.add(mid)
        else:
            invalid_rows.append(obj)

    if invalid_rows:
        backup = path.with_suffix(path.suffix + ".invalid_backup")
        with backup.open("a", encoding="utf-8", newline="\n") as f:
            for obj in invalid_rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for obj in valid_rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {"status": "ok", "valid": len(valid_rows), "invalid": len(invalid_rows)}


def ollama_model_available(model_name: str) -> Tuple[bool, str]:
    try:
        import ollama

        listing = ollama.list()
        raw_models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
        names = set()
        for item in raw_models:
            if isinstance(item, dict):
                names.add(str(item.get("name") or item.get("model") or ""))
            else:
                names.add(str(getattr(item, "name", "") or getattr(item, "model", "")))
        if model_name in names:
            return True, ""
        return False, f"Model {model_name!r} is not installed in Ollama. Available models: {sorted(names)}"
    except Exception as exc:
        return False, f"Could not query Ollama models: {exc}"


def generate_profile_with_ollama(music_id: str, dialogues: List[dict], generator: dict) -> Optional[dict]:
    import ollama

    prompt = build_stage4_profile_prompt(dialogues)
    model = str(generator["model"])
    max_retries = int(generator.get("max_retries", 3))
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = ollama.generate(
                model=model,
                prompt=prompt,
                format="json" if generator.get("json_format", True) else "",
                options={
                    "temperature": float(generator.get("temperature", 0.1)),
                    "num_ctx": int(generator.get("num_ctx", 8192)),
                },
            )
            raw_text = response["response"] if isinstance(response, dict) else response.response
            result = json.loads(clean_json_response(raw_text))
            if "summary_text" not in result or "salient_facts" not in result:
                raise ValueError("Missing required JSON fields")
            if not isinstance(result["salient_facts"], list) or not result["salient_facts"]:
                raise ValueError("Empty salient_facts detected")
            return {
                "music_id": music_id,
                "summary_text": str(result["summary_text"]),
                "salient_facts": result["salient_facts"],
                "meta": {
                    "model": model,
                    "processing_time": round(time.time() - start_time, 2),
                    "word_count": len(str(result["summary_text"]).split()),
                    "num_facts": len(result["salient_facts"]),
                    "attempt": attempt + 1,
                },
            }
        except Exception as exc:
            if attempt == max_retries - 1:
                return {
                    "music_id": music_id,
                    "summary_text": "",
                    "salient_facts": [],
                    "error": str(exc),
                    "meta": {"model": model, "attempts": max_retries, "failed": True},
                }
    return None


def generate_stability_profile_files(args: argparse.Namespace, target_ids: Sequence[str]) -> Tuple[List[Path], List[dict]]:
    if not getattr(args, "generate_stability_profiles", False):
        return [], []

    sample_n = getattr(args, "stability_generation_sample_size", None)
    selected_ids = list(target_ids)
    existing_files = [
        Path(p) for p in getattr(args, "existing_stability_profile_files", []) if Path(p).exists()
    ]
    if getattr(args, "stability_sample_from_existing", False) and existing_files:
        valid_existing_ids = valid_profile_ids_in_order(existing_files[0])
        target_set = set(target_ids)
        selected_ids = [mid for mid in valid_existing_ids if mid in target_set]
    if sample_n and len(selected_ids) > int(sample_n):
        selected_ids = selected_ids[: int(sample_n)]
    target_set = set(selected_ids)

    print(f"Preparing Stage 4 multi-generator profiles for {len(selected_ids)} cases...")
    grouped_dialogues = load_stage3_dialogues_for_ids(args.dialogues, target_set)
    generated_files: List[Path] = []
    generation_status: List[dict] = []

    for generator in getattr(args, "stability_generators", []):
        model = str(generator["model"])
        output_file = Path(generator["output_file"])
        output_file.parent.mkdir(parents=True, exist_ok=True)

        available, reason = ollama_model_available(model)
        if not available:
            print(f"Skipping {model}: {reason}")
            generation_status.append({"model": model, "output_file": str(output_file), "status": "skipped", "reason": reason})
            continue

        compact_status = compact_profile_file(output_file)
        processed = processed_profile_ids(output_file)
        remaining = [mid for mid in selected_ids if mid not in processed and mid in grouped_dialogues]
        print(
            f"Generating profiles with {model}: {len(remaining)} remaining, "
            f"{len(processed & target_set)} already done. "
            f"Compacted file: {compact_status}"
        )
        ok_count = 0
        error_count = 0
        with output_file.open("a", encoding="utf-8", newline="\n") as f:
            for idx, mid in enumerate(remaining, 1):
                profile = generate_profile_with_ollama(mid, grouped_dialogues[mid], generator)
                if not profile:
                    error_count += 1
                    continue
                if profile.get("summary_text") and profile.get("salient_facts"):
                    ok_count += 1
                else:
                    error_count += 1
                f.write(json.dumps(profile, ensure_ascii=False) + "\n")
                if idx % 25 == 0 or idx == len(remaining):
                    print(f"  {model}: {idx}/{len(remaining)} generated")

        generated_files.append(output_file)
        generation_status.append(
            {
                "model": model,
                "output_file": str(output_file),
                "status": "ok",
                "compaction": compact_status,
                "already_done": len(processed & target_set),
                "new_ok": ok_count,
                "new_errors": error_count,
                "target_cases": len(selected_ids),
            }
        )

    status_path = args.out_dir / "stability_generation_status.json"
    status_path.write_text(json.dumps(generation_status, ensure_ascii=False, indent=2), encoding="utf-8")
    return generated_files, generation_status


def load_metadata(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {str(x.get("music_id") or x.get("id")): x for x in data if isinstance(x, dict)}
    return {}


def history_path_for(history_dir: Path, music_id: str) -> Path:
    return history_dir / f"{music_id}__history.json"


def load_history(history_dir: Path, music_id: str) -> Optional[dict]:
    path = history_path_for(history_dir, music_id)
    if not path.exists():
        matches = list(history_dir.glob(f"*{music_id}*history.json"))
        if not matches:
            return None
        path = matches[0]
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def extract_attrs_from_text(text: str) -> Set[str]:
    text_l = normalize_text(text)
    attrs = set()
    for attr, patterns in ATTRIBUTE_LEXICON.items():
        for pat in patterns:
            if re.search(r"(?<![a-z0-9])" + re.escape(pat.lower()) + r"(?![a-z0-9])", text_l):
                attrs.add(attr)
                break
    return attrs


def fact_text(fact_obj: object) -> str:
    if isinstance(fact_obj, dict):
        return str(fact_obj.get("fact", ""))
    return str(fact_obj)


def split_profile_attrs(profile: ProfileRecord) -> Tuple[Set[str], Set[str], Set[str]]:
    positive_parts = [profile.summary_text]
    dislike_parts = []
    for f in profile.salient_facts:
        txt = fact_text(f)
        tag = str(f.get("conflict_tag", "")).upper() if isinstance(f, dict) else ""
        low = normalize_text(txt)
        if "DISLIKE" in tag or any(h in low for h in NEGATION_HINTS):
            dislike_parts.append(txt)
        else:
            positive_parts.append(txt)
    all_attrs = extract_attrs_from_text(" ".join([profile.summary_text] + [fact_text(f) for f in profile.salient_facts]))
    positive_attrs = extract_attrs_from_text(" ".join(positive_parts))
    dislike_attrs = extract_attrs_from_text(" ".join(dislike_parts))
    return all_attrs, positive_attrs, dislike_attrs


def attrs_from_music_item(item: dict) -> Set[str]:
    parts = [
        str(item.get("genre", "")),
        str(item.get("title", "")),
        str(item.get("artist", "")),
        str(item.get("semantic_seed", "")),
        " ".join(map(str, item.get("tags", []) if isinstance(item.get("tags"), list) else [])),
    ]
    return extract_attrs_from_text(" ".join(parts))


def attrs_from_metadata(metadata: Dict[str, dict], music_id: str) -> Set[str]:
    item = metadata.get(music_id, {})
    return attrs_from_music_item(item) if isinstance(item, dict) else set()


def items_from_history(history: dict, key: str) -> List[dict]:
    balanced = history.get("balanced_history", {}) if isinstance(history, dict) else {}
    rows = balanced.get(key, []) if isinstance(balanced, dict) else []
    return [x for x in rows if isinstance(x, dict)]


def union_attrs(items: Sequence[dict]) -> Set[str]:
    attrs = set()
    for item in items:
        attrs |= attrs_from_music_item(item)
    return attrs


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: Set[str], b: Set[str]) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def pairwise_coherence(attr_sets: Sequence[Set[str]]) -> float:
    pairs = []
    for i in range(len(attr_sets)):
        for j in range(i + 1, len(attr_sets)):
            pairs.append(jaccard(attr_sets[i], attr_sets[j]))
    return mean(pairs) if pairs else 0.0


def random_background_coherence(all_attr_sets: Sequence[Set[str]], n_items: int, rng: random.Random, repeats: int = 30) -> float:
    if not all_attr_sets or n_items < 2:
        return 0.0
    vals = []
    sample_n = min(n_items, len(all_attr_sets))
    for _ in range(repeats):
        vals.append(pairwise_coherence(rng.sample(list(all_attr_sets), sample_n)))
    return mean(vals) if vals else 0.0


def item_alignment_win_rate(profile_attrs: Set[str], pos_items: Sequence[dict], neg_items: Sequence[dict]) -> float:
    if not profile_attrs or not pos_items or not neg_items:
        return 0.0
    pos_scores = [jaccard(profile_attrs, attrs_from_music_item(x)) for x in pos_items]
    neg_scores = [jaccard(profile_attrs, attrs_from_music_item(x)) for x in neg_items]
    wins = 0.0
    total = 0
    for ps in pos_scores:
        for ns in neg_scores:
            total += 1
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total if total else 0.0


def naturalness_proxy(profile: ProfileRecord) -> float:
    text = normalize_text(profile.summary_text)
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]*", text)
    wc = len(words)
    length_score = max(0.0, 1.0 - abs(wc - 90) / 90)
    third_person = 1.0 if "the user" in text and not re.search(r"\b(i|my|me)\b", text) else 0.5
    fact_score = min(1.0, len(profile.salient_facts) / 5)
    return round((length_score + third_person + fact_score) / 3, 4)


def consistency_proxy(profile: ProfileRecord) -> float:
    _, pos_attrs, dislike_attrs = split_profile_attrs(profile)
    if not pos_attrs and not dislike_attrs:
        return 0.0
    contradiction = len(pos_attrs & dislike_attrs)
    denom = len(pos_attrs | dislike_attrs) or 1
    return round(1.0 - contradiction / denom, 4)


def build_metadata_background(metadata: Dict[str, dict], max_items: int, seed: int) -> List[Set[str]]:
    rows = [attrs_from_music_item(v) for v in metadata.values() if isinstance(v, dict)]
    rows = [x for x in rows if x]
    if len(rows) > max_items:
        rows = random.Random(seed).sample(rows, max_items)
    return rows


def compute_case_metrics(
    profile: ProfileRecord,
    history: dict,
    metadata: Dict[str, dict],
    background_sets: Sequence[Set[str]],
    rng: random.Random,
) -> CaseMetrics:
    all_attrs, positive_profile_attrs, dislike_attrs = split_profile_attrs(profile)
    target_attrs = attrs_from_metadata(metadata, profile.music_id)
    core_items = items_from_history(history, "core_sbs")
    exploratory_items = items_from_history(history, "exploratory_sbs")
    negative_items = items_from_history(history, "negative_sbs")
    core_attrs = union_attrs(core_items)
    exploratory_attrs = union_attrs(exploratory_items)
    negative_attrs = union_attrs(negative_items)
    supported_by_metadata = target_attrs | core_attrs | exploratory_attrs | negative_attrs
    supported = all_attrs & supported_by_metadata
    hallucinated = all_attrs - supported_by_metadata
    positive_alignment = containment(positive_profile_attrs, target_attrs | core_attrs)
    exploratory_alignment = containment(positive_profile_attrs, exploratory_attrs)
    negative_preference_overlap = containment(positive_profile_attrs, negative_attrs)
    negative_dislike_alignment = containment(dislike_attrs, negative_attrs)
    discrimination_margin = positive_alignment - negative_preference_overlap
    win_rate = item_alignment_win_rate(positive_profile_attrs, core_items, negative_items)
    pos_sets = [attrs_from_music_item(x) for x in core_items if attrs_from_music_item(x)]
    within = pairwise_coherence(pos_sets)
    random_bg = random_background_coherence(background_sets, len(pos_sets), rng)
    coherence_lift = within - random_bg
    return CaseMetrics(
        music_id=profile.music_id,
        profile_attr_count=len(all_attrs),
        supported_attr_count=len(supported),
        hallucinated_attr_count=len(hallucinated),
        grounding_ratio=round(containment(all_attrs, supported_by_metadata), 4),
        positive_alignment=round(positive_alignment, 4),
        exploratory_alignment=round(exploratory_alignment, 4),
        negative_preference_overlap=round(negative_preference_overlap, 4),
        negative_dislike_alignment=round(negative_dislike_alignment, 4),
        discrimination_margin=round(discrimination_margin, 4),
        pos_vs_neg_win_rate=round(win_rate, 4),
        within_positive_coherence=round(within, 4),
        random_background_coherence=round(random_bg, 4),
        coherence_lift=round(coherence_lift, 4),
        profile_naturalness_proxy=naturalness_proxy(profile),
        profile_consistency_proxy=consistency_proxy(profile),
        profile_attrs=";".join(sorted(all_attrs)),
        supported_attrs=";".join(sorted(supported)),
        hallucinated_attrs=";".join(sorted(hallucinated)),
    )


def summarize_metrics(metrics: Sequence[CaseMetrics]) -> dict:
    def col(name: str) -> List[float]:
        return [float(getattr(m, name)) for m in metrics]

    summary = {"n_cases": len(metrics)}
    for name in [
        "grounding_ratio",
        "positive_alignment",
        "exploratory_alignment",
        "negative_preference_overlap",
        "negative_dislike_alignment",
        "discrimination_margin",
        "pos_vs_neg_win_rate",
        "within_positive_coherence",
        "random_background_coherence",
        "coherence_lift",
        "profile_naturalness_proxy",
        "profile_consistency_proxy",
    ]:
        vals = col(name)
        summary[name] = {
            "mean": round(mean(vals), 4) if vals else 0.0,
            "median": round(median(vals), 4) if vals else 0.0,
            "p10": round(percentile(vals, 10), 4) if vals else 0.0,
            "p90": round(percentile(vals, 90), 4) if vals else 0.0,
        }
    return summary


def percentile(vals: Sequence[float], p: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    k = (len(xs) - 1) * (p / 100)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def write_csv(path: Path, rows: Sequence[CaseMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CaseMetrics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: getattr(row, k) for k in fields})


def distribution_jsd(history_rows: Sequence[dict], metadata: Dict[str, dict]) -> dict:
    synthetic = Counter()
    background = Counter()
    for h in history_rows:
        for key in ["core_sbs", "exploratory_sbs", "negative_sbs"]:
            for item in items_from_history(h, key):
                synthetic.update(attrs_from_music_item(item))
    for item in metadata.values():
        if isinstance(item, dict):
            background.update(attrs_from_music_item(item))
    return {
        "synthetic_attr_entropy": round(entropy(synthetic), 4),
        "metadata_attr_entropy": round(entropy(background), 4),
        "synthetic_to_metadata_jsd": round(js_divergence(synthetic, background), 4),
        "top_synthetic_attrs": synthetic.most_common(20),
        "top_metadata_attrs": background.most_common(20),
    }


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counter.values() if v)


def js_divergence(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    ta = sum(a.values()) or 1
    tb = sum(b.values()) or 1
    pa = {k: a[k] / ta for k in keys}
    pb = {k: b[k] / tb for k in keys}
    pm = {k: (pa[k] + pb[k]) / 2 for k in keys}
    return (kl(pa, pm) + kl(pb, pm)) / 2


def kl(p: Dict[str, float], q: Dict[str, float]) -> float:
    return sum(p[k] * math.log2(p[k] / q[k]) for k in p if p[k] > 0 and q[k] > 0)


def compute_generator_stability(profile_files: Sequence[Path], sample_size: Optional[int], seed: int) -> dict:
    if len(profile_files) < 2:
        return {
            "status": "not_run",
            "reason": "Provide two or more --profile-files generated by different LLMs.",
        }
    loaded = [load_profiles(p, sample_size, seed) for p in profile_files]
    common = set(loaded[0])
    for profs in loaded[1:]:
        common &= set(profs)
    pair_scores = []
    for mid in sorted(common):
        attr_sets = [split_profile_attrs(profs[mid])[0] for profs in loaded]
        for i in range(len(attr_sets)):
            for j in range(i + 1, len(attr_sets)):
                pair_scores.append(jaccard(attr_sets[i], attr_sets[j]))
    return {
        "status": "ok",
        "n_profile_files": len(profile_files),
        "n_common_cases": len(common),
        "mean_pairwise_attribute_jaccard": round(mean(pair_scores), 4) if pair_scores else 0.0,
        "median_pairwise_attribute_jaccard": round(median(pair_scores), 4) if pair_scores else 0.0,
        "profile_files": [str(p) for p in profile_files],
    }


def load_dialogue_index(path: Path, wanted_ids: Set[str]) -> Dict[str, Dict[str, dict]]:
    index: Dict[str, Dict[str, dict]] = defaultdict(dict)
    if not path.exists():
        return index
    for obj in read_jsonl(path):
        mid = str(obj.get("music_id", ""))
        if mid not in wanted_ids:
            continue
        dtype = str(obj.get("dialogue_type", "")).lower()
        if dtype:
            index[mid][dtype] = obj
    return index


def short_dialogue(obj: Optional[dict], max_chars: int = 900) -> str:
    if not obj:
        return ""
    turns = obj.get("dialogue_turns", [])
    parts = []
    for t in turns[:10]:
        if isinstance(t, dict):
            parts.append(f"{t.get('role', '')}: {t.get('content', '')}")
    return re.sub(r"\s+", " ", "\n".join(parts))[:max_chars]


def load_recommendation_index(path: Optional[Path]) -> Dict[str, dict]:
    if not path or not path.exists():
        return {}

    def row_key(row: dict) -> str:
        for key in ["music_id", "target_music", "target_music_id", "user_id", "sample_idx"]:
            value = row.get(key)
            if value not in [None, ""]:
                return str(value)
        return ""

    suffix = path.suffix.lower()
    rows: List[dict] = []
    if suffix == ".jsonl":
        rows = list(read_jsonl(path))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            rows = [data]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    index = {}
    for row in rows:
        key = row_key(row)
        if key:
            index[key] = row
    return index


def extract_recommendation_reason(row: Optional[dict]) -> str:
    if not row:
        return ""
    for key in [
        "recommendation_reason",
        "generated_reason",
        "reason",
        "rationale",
        "explanation",
        "justification",
        "response",
        "generated_text",
    ]:
        value = row.get(key)
        if value:
            return re.sub(r"\s+", " ", str(value)).strip()[:1200]
    return ""


def extract_recommended_item(row: Optional[dict]) -> str:
    if not row:
        return ""
    for key in ["recommended_item", "recommended_music", "prediction", "top1", "item_id", "song_id"]:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def write_human_annotation_packet(
    out_csv: Path,
    profiles: Dict[str, ProfileRecord],
    histories: Dict[str, dict],
    dialogues: Dict[str, Dict[str, dict]],
    recommendations: Dict[str, dict],
    metrics: Sequence[CaseMetrics],
    n_cases: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    candidates = sorted(metrics, key=lambda m: (m.grounding_ratio, m.discrimination_margin))
    low_risk = candidates[: max(1, n_cases // 3)]
    remaining = [m for m in metrics if m not in low_risk]
    chosen = low_risk + rng.sample(remaining, min(len(remaining), n_cases - len(low_risk)))
    chosen = chosen[:n_cases]
    fields = [
        "case_id",
        "music_id",
        "profile_summary",
        "salient_facts",
        "positive_core_samples",
        "negative_samples",
        "recommended_item",
        "recommendation_reason",
        "positive_dialogue_excerpt",
        "negative_dialogue_excerpt",
        "natural_profile_1_5",
        "long_term_consistency_1_5",
        "reason_grounded_in_preference_1_5",
        "hallucination_or_overinference_1_5",
        "notes",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, m in enumerate(chosen, 1):
            p = profiles[m.music_id]
            h = histories.get(m.music_id, {})
            pos_samples = sample_titles(items_from_history(h, "core_sbs"), 4)
            neg_samples = sample_titles(items_from_history(h, "negative_sbs"), 4)
            rec_row = recommendations.get(m.music_id)
            writer.writerow(
                {
                    "case_id": idx,
                    "music_id": m.music_id,
                    "profile_summary": p.summary_text,
                    "salient_facts": " | ".join(fact_text(x) for x in p.salient_facts),
                    "positive_core_samples": pos_samples,
                    "negative_samples": neg_samples,
                    "recommended_item": extract_recommended_item(rec_row),
                    "recommendation_reason": extract_recommendation_reason(rec_row),
                    "positive_dialogue_excerpt": short_dialogue(dialogues.get(m.music_id, {}).get("positive")),
                    "negative_dialogue_excerpt": short_dialogue(dialogues.get(m.music_id, {}).get("negative")),
                    "natural_profile_1_5": "",
                    "long_term_consistency_1_5": "",
                    "reason_grounded_in_preference_1_5": "",
                    "hallucination_or_overinference_1_5": "",
                    "notes": "",
                }
            )


def sample_titles(items: Sequence[dict], n: int) -> str:
    rows = []
    for item in items[:n]:
        title = str(item.get("title", ""))
        genre = str(item.get("genre", ""))
        tags = ",".join(map(str, item.get("tags", [])[:5])) if isinstance(item.get("tags"), list) else ""
        rows.append(f"{title} ({genre}; {tags})")
    return " || ".join(rows)


def aggregate_human_ratings(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {"status": "not_run", "reason": "No --human-ratings CSV supplied."}
    rating_cols = [
        "natural_profile_1_5",
        "long_term_consistency_1_5",
        "reason_grounded_in_preference_1_5",
        "hallucination_or_overinference_1_5",
    ]
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    result = {"status": "ok", "n_rows": len(rows)}
    for col in rating_cols:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, "")))
            except ValueError:
                pass
        result[col] = {
            "mean": round(mean(vals), 4) if vals else 0.0,
            "median": round(median(vals), 4) if vals else 0.0,
            "n": len(vals),
        }
    return result


def write_report(
    path: Path,
    summary: dict,
    distribution: dict,
    stability: dict,
    human: dict,
    args: argparse.Namespace,
    generation_status: Optional[List[dict]] = None,
) -> None:
    lines = []
    lines.append("# Synthetic-to-real Validity Analysis Report")
    lines.append("")
    lines.append("## Goal")
    lines.append("This report tests whether synthetic long-term preference profiles are grounded in original music metadata and whether they discriminate positive from negative samples.")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Profiles: `{', '.join(map(str, args.profile_files))}`")
    lines.append(f"- Stage 2 history dir: `{args.history_dir}`")
    lines.append(f"- Metadata: `{args.metadata}`")
    lines.append(f"- Recommendation file: `{args.recommendation_file or 'not provided'}`")
    lines.append(f"- Sample size: `{args.sample_size or 'all'}`")
    lines.append("")
    lines.append("## Main Quantitative Results")
    lines.append(f"- Cases analyzed: {summary['n_cases']}")
    for key, label in [
        ("grounding_ratio", "Profile attributes supported by metadata/history"),
        ("positive_alignment", "Preference alignment with positive/core samples"),
        ("negative_preference_overlap", "Positive-preference overlap with negative samples"),
        ("negative_dislike_alignment", "Dislike alignment with negative samples"),
        ("discrimination_margin", "Positive alignment minus negative overlap"),
        ("pos_vs_neg_win_rate", "Pairwise positive-vs-negative win rate"),
        ("coherence_lift", "Within-user positive coherence above random metadata background"),
        ("profile_naturalness_proxy", "Automatic naturalness proxy"),
        ("profile_consistency_proxy", "Automatic internal consistency proxy"),
    ]:
        s = summary[key]
        lines.append(f"- {label}: mean={s['mean']}, median={s['median']}, p10={s['p10']}, p90={s['p90']}")
    lines.append("")
    lines.append("## Synthetic-to-metadata Distribution")
    lines.append(f"- Synthetic attribute entropy: {distribution['synthetic_attr_entropy']}")
    lines.append(f"- Metadata attribute entropy: {distribution['metadata_attr_entropy']}")
    lines.append(f"- Jensen-Shannon divergence synthetic vs metadata: {distribution['synthetic_to_metadata_jsd']}")
    lines.append(f"- Top synthetic attrs: {distribution['top_synthetic_attrs'][:10]}")
    lines.append(f"- Top metadata attrs: {distribution['top_metadata_attrs'][:10]}")
    lines.append("")
    lines.append("## Multi-generator Stability")
    if generation_status:
        lines.append("Generation status:")
        lines.append(json.dumps(generation_status, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("Stability result:")
    lines.append(json.dumps(stability, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("## Human Validation")
    lines.append(json.dumps(human, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("## Suggested Thesis Interpretation")
    lines.append("- If grounding_ratio is high, the profile text is mostly supported by observable metadata/history rather than arbitrary LLM invention.")
    lines.append("- If positive_alignment and pos_vs_neg_win_rate are high while negative_preference_overlap is low, the synthetic preference captures discriminative taste structure.")
    lines.append("- If coherence_lift is positive, the positive histories have stronger within-user structure than random metadata samples.")
    lines.append("- Multi-generator stability should be reported only after regenerating profiles with at least two different LLMs under the same Stage 4 prompt.")
    lines.append("- Human validation should be reported as a small expert audit, not as a large-scale user study.")
    path.write_text("\n".join(lines), encoding="utf-8")


def config_to_namespace(config: dict) -> argparse.Namespace:
    return argparse.Namespace(
        profile_files=[Path(p) for p in config["profile_files"]],
        history_dir=Path(config["history_dir"]),
        metadata=Path(config["metadata"]),
        dialogues=Path(config["dialogues"]),
        out_dir=Path(config["out_dir"]),
        sample_size=config["sample_size"],
        seed=int(config["seed"]),
        background_max_items=int(config["background_max_items"]),
        human_packet_size=int(config["human_packet_size"]),
        recommendation_file=Path(config["recommendation_file"]) if config.get("recommendation_file") else None,
        existing_stability_profile_files=[
            Path(p) for p in config.get("existing_stability_profile_files", []) if p
        ],
        stability_sample_from_existing=bool(config.get("stability_sample_from_existing", False)),
        generate_stability_profiles=bool(config.get("generate_stability_profiles", False)),
        stability_generation_sample_size=config.get("stability_generation_sample_size"),
        stability_generators=config.get("stability_generators", []),
        human_ratings=Path(config["human_ratings"]) if config.get("human_ratings") else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic-to-real validity analysis")
    parser.add_argument("--profile-files", nargs="+", type=Path, default=RUN_CONFIG["profile_files"])
    parser.add_argument("--history-dir", type=Path, default=RUN_CONFIG["history_dir"])
    parser.add_argument("--metadata", type=Path, default=RUN_CONFIG["metadata"])
    parser.add_argument("--dialogues", type=Path, default=RUN_CONFIG["dialogues"])
    parser.add_argument("--out-dir", type=Path, default=RUN_CONFIG["out_dir"])
    parser.add_argument("--sample-size", type=int, default=RUN_CONFIG["sample_size"])
    parser.add_argument("--seed", type=int, default=RUN_CONFIG["seed"])
    parser.add_argument("--background-max-items", type=int, default=RUN_CONFIG["background_max_items"])
    parser.add_argument("--human-packet-size", type=int, default=RUN_CONFIG["human_packet_size"])
    parser.add_argument("--recommendation-file", type=Path, default=RUN_CONFIG["recommendation_file"])
    parser.add_argument("--generate-stability-profiles", action="store_true", default=RUN_CONFIG["generate_stability_profiles"])
    parser.add_argument("--no-generate-stability-profiles", action="store_false", dest="generate_stability_profiles")
    parser.add_argument("--stability-generation-sample-size", type=int, default=RUN_CONFIG["stability_generation_sample_size"])
    parser.add_argument("--human-ratings", type=Path, default=RUN_CONFIG["human_ratings"])
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or config_to_namespace(RUN_CONFIG)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    primary_profile_path = args.profile_files[0]
    profiles = load_profiles(primary_profile_path, args.sample_size, args.seed)
    generated_profile_files, generation_status = generate_stability_profile_files(args, list(profiles.keys()))
    existing_stability_files = [
        Path(p) for p in getattr(args, "existing_stability_profile_files", []) if Path(p).exists()
    ]
    stability_profile_files = existing_stability_files + generated_profile_files
    if len(generated_profile_files) == 0 and len(args.profile_files) >= 2:
        stability_profile_files = args.profile_files
    metadata = load_metadata(args.metadata)
    background_sets = build_metadata_background(metadata, args.background_max_items, args.seed)

    metrics: List[CaseMetrics] = []
    histories: Dict[str, dict] = {}
    missing_history = 0
    for mid, profile in profiles.items():
        h = load_history(args.history_dir, mid)
        if not h:
            missing_history += 1
            continue
        histories[mid] = h
        metrics.append(compute_case_metrics(profile, h, metadata, background_sets, rng))

    summary = summarize_metrics(metrics)
    distribution = distribution_jsd(list(histories.values()), metadata)
    stability = compute_generator_stability(stability_profile_files, args.sample_size, args.seed)
    human = aggregate_human_ratings(args.human_ratings)

    write_csv(args.out_dir / "synthetic_validity_case_metrics.csv", metrics)
    (args.out_dir / "synthetic_validity_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "distribution": distribution,
                "generator_stability": stability,
                "stability_generation": generation_status,
                "human_validation": human,
                "missing_history": missing_history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dialogue_index = load_dialogue_index(args.dialogues, set(profiles))
    recommendation_index = load_recommendation_index(args.recommendation_file)
    write_human_annotation_packet(
        args.out_dir / "human_annotation_template.csv",
        profiles,
        histories,
        dialogue_index,
        recommendation_index,
        metrics,
        args.human_packet_size,
        args.seed,
    )
    write_report(args.out_dir / "synthetic_validity_report.md", summary, distribution, stability, human, args, generation_status)

    print("Synthetic-to-real validity analysis completed.")
    print(f"Cases analyzed: {len(metrics)}; missing history: {missing_history}")
    print(f"Outputs written to: {args.out_dir}")
    print(f"Report: {args.out_dir / 'synthetic_validity_report.md'}")


if __name__ == "__main__":
    main(parse_args() if len(sys.argv) > 1 else config_to_namespace(RUN_CONFIG))
