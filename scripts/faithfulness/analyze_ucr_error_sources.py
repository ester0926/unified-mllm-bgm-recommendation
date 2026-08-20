"""
用途：整理人工複查用資料與 UCR 錯誤來源分析結果。
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
import datetime as _dt
import json
import logging
import random
import re
from collections import Counter, defaultdict

from scripts.faithfulness import faithfulness_claim_judge_v2 as J


# =============================================================================
# 路徑設定
# =============================================================================

BASE_DIR    = PROJECT_ROOT
RESULTS_DIR = BASE_DIR / "results" / "faithfulness"
OUT_DIR     = RESULTS_DIR / "ucr_error_sources"

CLAIM_CSV      = RESULTS_DIR / "claim_annotations_top1_v2.csv"
GENERATION_CSV = RESULTS_DIR / "counterfactual_generations_top1.csv"

# 外部獨立證據（Stage 1 音樂 metadata）。找不到時 L3 自動降級為只用管線內證據。
MUSIC_METADATA_JSON = Path(
    r"data/user_profiling/music_metadata_simple\music_metadata_enriched.json"
)


# =============================================================================
# 使用者設定
# =============================================================================

PRIMARY_CONDITION   = "full"    # 教授指的 UCR 13.1% 即此條件
UNSUPPORTED_REASON  = "no_detected_support_source"

# 人工複核抽樣：full 條件全數複核，其他條件每類最多抽這麼多條
REVIEW_SAMPLE_PER_BUCKET = 15
REVIEW_SEED = 20260726


# =============================================================================
# L2 分類用字典（只新增本腳本需要、既有字典未涵蓋的樣態）
# =============================================================================

# 一般性空泛描述：對影片的泛用效用宣稱
GENERIC_EFFECT_PATTERNS = {
    "addition to your video", "great addition", "could add", "would add",
    "adds a", "add a touch", "touch to your video", "perfect for",
    "get people moving", "get the energy up", "elevate", "brings a",
    "sets the tone", "works well with your", "goes well with your",
}

# 主觀／不可驗證的情感陳述
SUBJECTIVE_TERMS = {
    "beautiful", "captures the essence", "essence of", "nostalgic",
    "introspective", "heartfelt", "emotional depth", "unique blend",
    "captivating", "soothing", "atmospheric touch", "fresh",
}

# 具體影片內容主張（非泛用框架語）
VIDEO_SPECIFIC_TERMS = {
    "wedding", "birthday", "party video", "travel video", "workout",
    "gaming video", "cooking", "tutorial video", "sports video",
    "dance video", "vlog",
}

# 元資料主張（v2 的 METADATA_TERMS 未涵蓋的發行載體詞）
EXTRA_METADATA_TERMS = {
    "mixtape", "ep", "single", "compilation", "soundtrack", "b-side",
    "debut", "remix of",
}

# 年代／時期主張（風格宣稱，規則無法裁決，一律送人工複核）
ERA_PATTERN = re.compile(r"\b((?:19|20)?\d0s)\b", re.IGNORECASE)

# 引號內的專有名稱（專輯／mixtape／曲名）
# 開引號必須位於字串開頭或非文字字元之後，否則 "It's" 的所有格撇號會被
# 誤判為開引號，進而把「s a track from his mixtape 」整段當成專有名稱。
QUOTED_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=[\s(\[:,;-]))['\"\u2018\u201c]([^'\"\u2018\u2019\u201c\u201d]{2,60})['\"\u2019\u201d]"
)

# 子句被切在對等連接結構中（"A and B" / "A, B"）→ 屬斷句碎片
TRUNCATED_TAIL_PATTERN = re.compile(r"^\s*(?:and|or|,)\s+", re.IGNORECASE)

# 曲風詞正規化（比對 metadata 標籤時用）
GENRE_ALIASES = {
    "hip hop": "hip-hop", "hiphop": "hip-hop", "rnb": "r&b", "r and b": "r&b",
    "electro": "electronic", "edm": "electronic", "alt": "alternative",
    "chill out": "chillout", "hard rock": "rock", "heavy metal": "metal",
    "dance-pop": "dance", "pop rock": "rock",
}

# --- 生成崩潰偵測（在 generated_text 層級判定，非子句層級）-------------------
# 門檻以既有資料校準：full / wo_prompt 各 0/200 誤判，wo_ltp 200/200 命中。
DEGEN_MIN_TOKENS   = 10     # 有效英文詞數下限
DEGEN_TTR_MAX      = 0.35   # type-token ratio 低於此值視為重複退化
DEGEN_TOPREP_MAX   = 0.25   # 單一詞占比高於此值視為重複退化
DEGEN_MEANLEN_MAX  = 2.5    # 平均詞長過短（如 "th? th? th?"）

BUCKET_LABELS = {
    "E7_degenerate_generation": "生成崩潰（退化輸出）",
    "E0_clause_split_artifact": "斷句碎片（標註流程假影）",
    "E0b_rule_miss_verified":   "規則漏判但有證據支持",
    "E1_preference_unsupported": "偏好不支持",
    "E2_metadata_unsupported":   "音樂元資料不支持",
    "E3_video_unsupported":      "影片內容不支持",
    "E4_vague_generic":          "一般性空泛描述／主觀陳述",
    "E5_multi_source_ambiguous": "多來源無法區分",
    "E6_likely_hallucination":   "可能幻覺",
}

# 屬於「方法學假影／非模型錯誤」的桶，計算校正後 UCR 時扣除
ARTIFACT_BUCKETS = {"E0_clause_split_artifact", "E0b_rule_miss_verified"}
# 只由斷句造成的假影（句子層級界線用）
SPLIT_ARTIFACT_BUCKETS = {"E0_clause_split_artifact"}


# =============================================================================
# Logger
# =============================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ucr_error_sources")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# =============================================================================
# 讀檔
# =============================================================================

def read_csv(path: Path) -> list:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list, fieldnames=None):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_music_metadata(logger):
    if not MUSIC_METADATA_JSON.exists():
        logger.warning("找不到外部音樂 metadata（%s），L3 將只使用管線內證據",
                       MUSIC_METADATA_JSON)
        return None
    with open(MUSIC_METADATA_JSON, "r", encoding="utf-8") as f:
        md = json.load(f)
    logger.info("[Metadata] 載入外部證據 %d 筆", len(md))
    return md


# =============================================================================
# L1：母句還原與重新歸屬
# =============================================================================

def split_sentences(text: str) -> list:
    return re.split(r"(?<=[.!?])\s+", J.normalize_text(text))


def is_degenerate_generation(text: str) -> tuple:
    """
    在**整段生成文字**層級判定是否為退化輸出（重複、亂碼、空輸出）。
    退化與否是輸出整體的性質，不應以子句判斷，故在此層級處理。
    回傳 (is_degenerate, reason)。
    """
    tokens = re.findall(r"[A-Za-z']+", (text or "").lower())
    if len(tokens) < DEGEN_MIN_TOKENS:
        return True, f"too_few_tokens={len(tokens)}"
    counts = Counter(tokens)
    ttr = len(counts) / len(tokens)
    top_share = counts.most_common(1)[0][1] / len(tokens)
    mean_len = sum(len(t) for t in tokens) / len(tokens)
    if ttr < DEGEN_TTR_MAX:
        return True, f"type_token_ratio={ttr:.2f}"
    if top_share > DEGEN_TOPREP_MAX:
        return True, f"top_token_share={top_share:.2f}"
    if mean_len <= DEGEN_MEANLEN_MAX:
        return True, f"mean_token_len={mean_len:.2f}"
    return False, ""


def find_parent_sentence(claim: str, generated_text: str) -> str:
    """找出 claim 所屬的原始句子；找不到時回傳 claim 自身（保守處理）。"""
    for sent in split_sentences(generated_text):
        if claim in sent:
            return sent
    return claim


def is_truncated_fragment(claim: str, parent: str) -> bool:
    """
    判斷 claim 是否為「切在對等連接結構中」的殘句，例如
      parent = "It has a chill and ambient vibe with electronic elements."
      claim  = "It has a chill"            ← 後面接 " and ..." → 殘句
    這類 claim 本身不構成完整主張，應以母句為分析單位。
    """
    if claim == parent or claim not in parent:
        return False
    tail = parent[parent.index(claim) + len(claim):]
    return bool(TRUNCATED_TAIL_PATTERN.match(tail))


# =============================================================================
# L3：可查證性驗證
# =============================================================================

def normalize_genre(term: str) -> str:
    t = term.lower().strip()
    return GENRE_ALIASES.get(t, t)


def extract_genre_terms(claim: str) -> list:
    lower = claim.lower()
    hits = set()
    for g in J.GENRE_TERMS:
        if re.search(r"(?<![\w-])" + re.escape(g) + r"(?![\w-])", lower):
            hits.add(normalize_genre(g))
    return sorted(hits)


def build_evidence(row: dict, music_meta) -> tuple:
    """
    回傳 (pipeline_evidence_text, external_terms_set, has_external)
      pipeline evidence：與 analyze_metadata_consistency.py 相同的證據定義
      external terms   ：music_metadata_enriched.json 的 genre + musicnn 標籤
    """
    pipeline = " ".join([
        row.get("music_title", "") or "",
        row.get("music_artist", "") or "",
        row.get("top1_reference_text", "") or "",
    ]).lower()

    external = set()
    has_external = False
    if music_meta is not None:
        mid = (row.get("top1_video_id") or "")[:11]
        entry = music_meta.get(mid)
        if entry:
            has_external = True
            if entry.get("genre"):
                external.add(normalize_genre(str(entry["genre"])))
            for tag in entry.get("tags", []) or []:
                external.add(normalize_genre(str(tag)))
            for field in ("album", "title", "artist"):
                if entry.get(field):
                    external.add(str(entry[field]).lower())
    return pipeline, external, has_external


def verify_genre_claims(genres: list, pipeline_text: str, external: set) -> tuple:
    """回傳 (status, supported_list, missing_list)。"""
    if not genres:
        return "not_applicable", [], []
    supported, missing = [], []
    for g in genres:
        ok = (g in pipeline_text) or any(g in e for e in external)
        (supported if ok else missing).append(g)
    if not missing:
        return "supported", supported, missing
    if supported:
        return "partially_supported", supported, missing
    return "not_found", supported, missing


def verify_quoted_names(claim: str, pipeline_text: str, external: set) -> tuple:
    """驗證引號內的專輯／mixtape／曲名是否出現在任一證據來源。"""
    names = QUOTED_PATTERN.findall(claim)
    if not names:
        return "not_applicable", [], []
    supported, missing = [], []
    for n in names:
        key = n.lower().strip()
        ok = (key in pipeline_text) or any(key in e for e in external)
        (supported if ok else missing).append(n)
    if not missing:
        return "supported", supported, missing
    return ("partially_supported" if supported else "not_found"), supported, missing


# =============================================================================
# L2：錯誤來源分類
# =============================================================================

def classify_error_source(claim, parent, parent_source, row, music_meta, degen):
    """
    回傳 dict：bucket / subtype / evidence_note / verification_* / needs_human_review
    分類順序固定且互斥，每條 claim 只落一個桶。
    degen = (is_degenerate, reason)，於整段生成層級先行判定。

    ⚠ L2 的關鍵詞與證據比對一律以**母句**為輸入，而非可能被切斷的子句：
      子句「It has a chill」不成主張，其可驗證內容全在母句
      「It has a chill and ambient vibe with electronic elements.」裡。
    """
    claim = claim  # 保留原子句供輸出與人工複核比對
    analysis_text = parent if parent else claim
    lower = analysis_text.lower()
    pipeline_text, external, has_external = build_evidence(row, music_meta)

    genres = extract_genre_terms(analysis_text)
    g_status, g_ok, g_missing = verify_genre_claims(genres, pipeline_text, external)
    q_status, q_ok, q_missing = verify_quoted_names(analysis_text, pipeline_text, external)

    base = {
        "genre_terms": ";".join(genres),
        "genre_verification": g_status,
        "genre_supported": ";".join(g_ok),
        "genre_missing": ";".join(g_missing),
        "quoted_names": ";".join(q_ok + q_missing),
        "quoted_verification": q_status,
        "external_evidence_available": int(has_external),
        "needs_human_review": 0,
    }

    # --- E7：整段生成已崩潰 → 不屬於「主張無支持」的問題 --------------------
    if degen[0]:
        base.update({
            "bucket": "E7_degenerate_generation",
            "subtype": "degenerate_output",
            "evidence_note": f"整段生成為退化輸出（{degen[1]}），"
                             f"此時的 unsupported 反映生成崩潰而非主張缺乏證據",
        })
        return base

    # --- E0：母句可歸屬 → 切分造成的假影 ---------------------------------
    if parent_source != J.SOURCE_UNSUPPORTED:
        base.update({
            "bucket": "E0_clause_split_artifact",
            "subtype": f"parent_{parent_source}",
            "evidence_note": "母句可歸屬，unsupported 係 split_claims 於 and/, 處切斷所致",
        })
        return base

    # --- E1：偏好主張 ------------------------------------------------------
    if J.keyword_hit(lower, J.PREFERENCE_KEYWORDS):
        base.update({
            "bucket": "E1_preference_unsupported",
            "subtype": "preference_claim_without_evidence",
            "evidence_note": "含偏好詞但母句層級仍無法歸屬",
            "needs_human_review": 1,
        })
        return base

    # --- E2 / E6：元資料主張（含發行載體詞或引號專有名稱）-------------------
    is_metadata_claim = (J.keyword_hit(lower, EXTRA_METADATA_TERMS)
                         or J.keyword_hit(lower, J.METADATA_TERMS))
    if is_metadata_claim or q_status != "not_applicable":
        if q_status == "not_found":
            base.update({
                "bucket": "E6_likely_hallucination",
                "subtype": "named_entity_absent_from_all_evidence",
                "evidence_note": f"引號名稱 {q_missing} 未見於管線內證據與外部 metadata；"
                                 f"屬具體可查證卻查無支持者，列為可能幻覺並送人工複核",
                "needs_human_review": 1,
            })
        elif q_status in ("supported", "partially_supported"):
            base.update({
                "bucket": "E0b_rule_miss_verified",
                "subtype": "metadata_claim_verified",
                "evidence_note": "元資料主張可由證據支持，屬 v2 規則漏判而非錯誤",
            })
        else:
            base.update({
                "bucket": "E2_metadata_unsupported",
                "subtype": "metadata_claim_without_named_entity",
                "evidence_note": "含元資料詞但無可比對的具名實體，查無支持",
                "needs_human_review": 1,
            })
        return base

    # --- E3：具體影片內容主張 ---------------------------------------------
    if J.keyword_hit(lower, VIDEO_SPECIFIC_TERMS):
        base.update({
            "bucket": "E3_video_unsupported",
            "subtype": "specific_video_content_asserted",
            "evidence_note": "主張具體影片情境（如婚禮／旅遊），但模型輸入僅有 CLIP 影像特徵，"
                             "規則層無法驗證，送人工複核",
            "needs_human_review": 1,
        })
        return base

    # --- E5：單一子句同時混合音樂屬性與影片效用 -----------------------------
    has_effect = J.keyword_hit(lower, GENERIC_EFFECT_PATTERNS)
    if genres and has_effect:
        base.update({
            "bucket": "E5_multi_source_ambiguous",
            "subtype": "music_property_plus_video_effect",
            "evidence_note": "同一子句混合曲風敘述與影片效用宣稱，無法單一歸屬",
        })
        return base

    # --- E0b / E2：純曲風敘述（v2 因缺少 anchor 詞而漏判）-------------------
    if genres:
        if g_status == "supported":
            base.update({
                "bucket": "E0b_rule_miss_verified",
                "subtype": "genre_claim_verified_by_tags",
                "evidence_note": "曲風敘述可由 musicnn 標籤／參考文本支持；"
                                 "v2 因 GENRE_TERMS 需搭配 anchor 詞才計為 audio 而漏判",
            })
        elif g_status == "partially_supported":
            base.update({
                "bucket": "E2_metadata_unsupported",
                "subtype": "genre_partially_supported",
                "evidence_note": f"部分曲風查無支持：{g_missing}（musicnn 標籤有雜訊，"
                                 f"查無支持不等於事實錯誤）",
                "needs_human_review": 1,
            })
        else:
            base.update({
                "bucket": "E2_metadata_unsupported",
                "subtype": "genre_not_found",
                "evidence_note": f"曲風 {g_missing} 未見於任何證據來源（標籤有雜訊，"
                                 f"故列為不支持而非幻覺）",
                "needs_human_review": 1,
            })
        return base

    # --- E4：一般性空泛描述／主觀陳述 --------------------------------------
    if has_effect or J.keyword_hit(lower, SUBJECTIVE_TERMS) \
            or J.keyword_hit(lower, J.GENERIC_RECOMMENDATION_PATTERNS):
        subtype = "subjective_statement" if J.keyword_hit(lower, SUBJECTIVE_TERMS) \
            else "generic_video_benefit"
        base.update({
            "bucket": "E4_vague_generic",
            "subtype": subtype,
            "evidence_note": "泛用效用或主觀陳述，本質上不具可驗證性",
        })
        return base

    # --- E6：其餘具體但無跡可循的主張 --------------------------------------
    era = ERA_PATTERN.search(analysis_text)
    base.update({
        "bucket": "E6_likely_hallucination",
        "subtype": "era_style_claim" if era else "unclassified_specific_claim",
        "evidence_note": ("年代風格主張，規則無法裁決" if era
                          else "具體主張但未命中任何證據來源"),
        "needs_human_review": 1,
    })
    return base


# =============================================================================
# 主流程
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUT_DIR / "ucr_error_sources.log")
    started = _dt.datetime.now()

    logger.info("=" * 78)
    logger.info("B2 UCR 錯誤來源拆解 | 主條件=%s", PRIMARY_CONDITION)
    logger.info("=" * 78)

    for p in (CLAIM_CSV, GENERATION_CSV):
        if not p.exists():
            raise FileNotFoundError(f"找不到輸入檔：{p}")

    claim_rows = read_csv(CLAIM_CSV)
    gen_rows = read_csv(GENERATION_CSV)
    gen_index = {(r["sample_idx"], r["condition"]): r for r in gen_rows}
    music_meta = load_music_metadata(logger)
    logger.info("[輸入] claims=%d generations=%d", len(claim_rows), len(gen_rows))

    # ---- L1 + L2 + L3 -------------------------------------------------------
    annotated = []
    n_parent_missing = 0
    for r in claim_rows:
        if r["unsupported_reason"] != UNSUPPORTED_REASON:
            continue
        claim = r["claim_text"]
        gen_row = gen_index.get((r["sample_idx"], r["condition"]), {})
        parent = find_parent_sentence(claim, r.get("generated_text", ""))
        if parent == claim and claim not in J.normalize_text(r.get("generated_text", "")):
            n_parent_missing += 1
        parent_source, parent_type = J.classify_claim(parent, gen_row)
        degen = is_degenerate_generation(gen_row.get("generated_text", r.get("generated_text", "")))

        result = classify_error_source(claim, parent, parent_source, gen_row, music_meta, degen)
        truncated = is_truncated_fragment(claim, parent)
        annotated.append({
            "sample_idx": r["sample_idx"],
            "video_id": r["video_id"],
            "gt_music_id": r["gt_music_id"],
            "top1_music_id": gen_row.get("top1_music_id", ""),
            "top1_video_id": gen_row.get("top1_video_id", ""),
            "condition": r["condition"],
            "claim_id": r["claim_id"],
            "claim_text": claim,
            "parent_sentence": parent,
            "parent_source": parent_source,
            "parent_claim_type": parent_type,
            "generation_degenerate": int(degen[0]),
            "degenerate_reason": degen[1],
            "is_truncated_fragment": int(truncated),
            "bucket": result["bucket"],
            "bucket_label": BUCKET_LABELS[result["bucket"]],
            "subtype": result["subtype"],
            "evidence_note": result["evidence_note"],
            "genre_terms": result["genre_terms"],
            "genre_verification": result["genre_verification"],
            "genre_supported": result["genre_supported"],
            "genre_missing": result["genre_missing"],
            "quoted_names": result["quoted_names"],
            "quoted_verification": result["quoted_verification"],
            "external_evidence_available": result["external_evidence_available"],
            "needs_human_review": result["needs_human_review"],
            "generated_text": r.get("generated_text", ""),
        })

    if n_parent_missing:
        logger.warning("有 %d 條 claim 無法在 generated_text 中定位母句（已保守以 claim 自身處理）",
                       n_parent_missing)
    logger.info("[標註] 共處理 %d 條 unsupported claims（全條件）", len(annotated))

    # ---- 生成崩潰統計（以整段生成為單位，涵蓋全部 1200 段）------------------
    degen_by_cond = {}
    for cond in sorted(set(r["condition"] for r in gen_rows)):
        subset = [r for r in gen_rows if r["condition"] == cond]
        n_deg = sum(1 for r in subset if is_degenerate_generation(r.get("generated_text", ""))[0])
        degen_by_cond[cond] = (n_deg, len(subset))
        logger.info("[退化] %-22s %3d / %3d 段生成為退化輸出", cond, n_deg, len(subset))

    # ---- 各條件的 claim 總數（計算 UCR 用）----------------------------------
    n_claims_by_cond = Counter(r["condition"] for r in claim_rows)
    n_unsup_by_cond = Counter(r["condition"] for r in claim_rows
                              if int(r["is_supported"]) == 0)

    # ---- UCR 上下界 ---------------------------------------------------------
    bounds = []
    for cond in sorted(n_claims_by_cond):
        rows_c = [a for a in annotated if a["condition"] == cond]
        n_total = n_claims_by_cond[cond]
        n_unsup_all = n_unsup_by_cond[cond]          # 含 source_removed_by_*
        n_no_source = len(rows_c)                    # 僅 no_detected_support_source
        n_split = sum(1 for a in rows_c if a["bucket"] in SPLIT_ARTIFACT_BUCKETS)
        n_artifact = sum(1 for a in rows_c if a["bucket"] in ARTIFACT_BUCKETS)
        n_degen = sum(1 for a in rows_c if a["bucket"] == "E7_degenerate_generation")
        n_genuine = n_no_source - n_artifact - n_degen
        div = n_total if n_total else float("nan")
        bounds.append({
            "condition": cond,
            "n_claims": n_total,
            "unsupported_claims_reported": n_unsup_all,
            "UCR_reported": n_unsup_all / div,
            "no_source_claims": n_no_source,
            "degenerate_claims": n_degen,
            "split_artifact_claims": n_split,
            "verified_rule_miss_claims": n_artifact - n_split,
            "genuine_unattributable_claims": n_genuine,
            # 三級界線：越往下扣除越多「非模型錯誤」的成分
            "UCR_L1_reported_upper": n_unsup_all / div,
            "UCR_L2_sentence_level": (n_unsup_all - n_split) / div,
            "UCR_L3_verification_adjusted": (n_unsup_all - n_artifact) / div,
            "artifact_share_of_no_source": n_artifact / n_no_source if n_no_source else float("nan"),
            "degenerate_share_of_no_source": n_degen / n_no_source if n_no_source else float("nan"),
        })
        b = bounds[-1]
        logger.info("[UCR] %-22s n=%4d L1=%.4f L2=%.4f L3=%.4f | 斷句假影=%d 規則漏判=%d 崩潰=%d 真實=%d",
                    cond, n_total, b["UCR_L1_reported_upper"], b["UCR_L2_sentence_level"],
                    b["UCR_L3_verification_adjusted"], n_split, n_artifact - n_split,
                    n_degen, n_genuine)

    # ---- 組成表 -------------------------------------------------------------
    comp_counter = defaultdict(Counter)
    for a in annotated:
        comp_counter[a["condition"]][a["bucket"]] += 1

    composition = []
    for cond in sorted(comp_counter):
        total = sum(comp_counter[cond].values())
        for bucket in BUCKET_LABELS:
            cnt = comp_counter[cond][bucket]
            composition.append({
                "condition": cond,
                "bucket": bucket,
                "bucket_label": BUCKET_LABELS[bucket],
                "n": cnt,
                "share_of_no_source": cnt / total if total else 0.0,
                "share_of_all_claims": cnt / n_claims_by_cond[cond] if n_claims_by_cond[cond] else 0.0,
            })

    # ---- 句子層級去重組成（避免同一母句的多個殘句被重複計數）----------------
    sent_seen = {}
    for a in annotated:
        key = (a["condition"], a["sample_idx"], a["parent_sentence"])
        if key not in sent_seen:
            sent_seen[key] = a["bucket"]
    sent_comp_counter = defaultdict(Counter)
    for (cond, _sid, _sent), bucket in sent_seen.items():
        sent_comp_counter[cond][bucket] += 1
    sentence_composition = []
    for cond in sorted(sent_comp_counter):
        total = sum(sent_comp_counter[cond].values())
        for bucket in BUCKET_LABELS:
            cnt = sent_comp_counter[cond][bucket]
            sentence_composition.append({
                "condition": cond, "bucket": bucket,
                "bucket_label": BUCKET_LABELS[bucket],
                "n_sentences": cnt,
                "share": cnt / total if total else 0.0,
            })

    logger.info("-" * 78)
    logger.info("主條件（%s）錯誤來源組成：", PRIMARY_CONDITION)
    prim_total = sum(comp_counter[PRIMARY_CONDITION].values())
    for bucket in BUCKET_LABELS:
        cnt = comp_counter[PRIMARY_CONDITION][bucket]
        if cnt:
            logger.info("  %-28s %-22s n=%3d (%.1f%%)",
                        bucket, BUCKET_LABELS[bucket], cnt, 100 * cnt / prim_total)

    # ---- 人工複核樣板 -------------------------------------------------------
    rng = random.Random(REVIEW_SEED)
    review = [dict(a) for a in annotated if a["condition"] == PRIMARY_CONDITION]
    by_bucket = defaultdict(list)
    for a in annotated:
        if a["condition"] != PRIMARY_CONDITION:
            by_bucket[(a["condition"], a["bucket"])].append(a)
    for key, items in sorted(by_bucket.items()):
        k = min(REVIEW_SAMPLE_PER_BUCKET, len(items))
        review.extend(dict(x) for x in rng.sample(items, k))
    for r in review:
        r["human_bucket"] = ""            # 複核者填寫：同上 8 類代碼
        r["human_is_genuine_error"] = ""  # 複核者填寫：1 = 真實無支持主張，0 = 非錯誤
        r["human_note"] = ""

    # ---- 輸出 ---------------------------------------------------------------
    claims_path = OUT_DIR / "ucr_error_source_claims.csv"
    comp_path   = OUT_DIR / "ucr_error_source_composition.csv"
    bounds_path = OUT_DIR / "ucr_bounds.csv"
    review_path = OUT_DIR / "human_review_template.csv"
    json_path   = OUT_DIR / "ucr_error_source_summary.json"
    md_path     = OUT_DIR / "ucr_error_source_summary.md"

    write_csv(claims_path, annotated)
    write_csv(comp_path, composition)
    write_csv(OUT_DIR / "ucr_error_source_composition_sentence_level.csv", sentence_composition)
    write_csv(bounds_path, bounds)
    write_csv(review_path, review, fieldnames=[
        "condition", "sample_idx", "claim_id", "claim_text", "parent_sentence",
        "bucket", "bucket_label", "subtype", "evidence_note",
        "genre_terms", "genre_verification", "genre_missing",
        "quoted_names", "quoted_verification", "needs_human_review",
        "human_bucket", "human_is_genuine_error", "human_note", "generated_text",
    ])

    summary = {
        "generated_at": started.isoformat(timespec="seconds"),
        "inputs": {"claim_csv": str(CLAIM_CSV), "generation_csv": str(GENERATION_CSV),
                   "external_metadata": str(MUSIC_METADATA_JSON) if music_meta else None},
        "method": {
            "L1": "母句還原後以既有 classify_claim() 重判，區分斷句假影與真正不可歸屬",
            "L2": "教授指定六類 + 斷句假影 / 規則漏判兩類方法學桶，規則互斥且固定順序",
            "L3": "曲風與具名實體對照 (a) top1_reference_text+title+artist (b) musicnn 標籤",
            "caveat": "musicnn 標籤含雜訊，查無支持一律歸為『不支持』，"
                      "僅具名實體完全查無者升級為『可能幻覺』並送人工複核",
        },
        "bucket_labels": BUCKET_LABELS,
        "ucr_bounds": bounds,
        "composition": composition,
        "sentence_level_composition": sentence_composition,
        "primary_condition": PRIMARY_CONDITION,
        "n_review_rows": len(review),
        "n_needs_human_review": sum(1 for a in annotated if a["needs_human_review"]),
        "degenerate_by_condition": degen_by_cond,
        "degeneracy_thresholds": {
            "min_tokens": DEGEN_MIN_TOKENS, "max_type_token_ratio": DEGEN_TTR_MAX,
            "max_top_token_share": DEGEN_TOPREP_MAX, "max_mean_token_len": DEGEN_MEANLEN_MAX,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown(md_path, summary, annotated, comp_counter, n_claims_by_cond)

    for p in [claims_path, comp_path, bounds_path, review_path, json_path, md_path]:
        logger.info("已輸出：%s", p)
    logger.info("完成，耗時 %.1f 秒", (_dt.datetime.now() - started).total_seconds())


def write_markdown(path, summary, annotated, comp_counter, n_claims_by_cond):
    prim = summary["primary_condition"]
    bounds = {b["condition"]: b for b in summary["ucr_bounds"]}
    pb = bounds[prim]
    L = []
    L.append("# UCR 錯誤來源拆解（B2）\n")
    L.append(f"- 產生時間：{summary['generated_at']}")
    L.append(f"- 主條件：`{prim}`（論文報告的 UCR = {pb['UCR_reported']*100:.2f}% 即此條件）")
    L.append(f"- 外部證據：{'musicnn 標籤 + genre（可用）' if summary['inputs']['external_metadata'] else '不可用，僅用管線內證據'}\n")

    L.append("## 一、UCR 的三級界線（判讀基準）\n")
    L.append("現行 UCR 以**子句**為單位計算，而 `split_claims()` 會在 ` and ` / `,` 處切斷句子，"
             "使「The song has a soulful」這類殘句失去可歸屬的關鍵詞。把每條 claim 還原回**母句**"
             "後以同一支規則重判，再對曲風／具名實體做證據比對，即可逐層分離"
             "「標註假影」「規則漏判」與「真正無法歸屬的主張」。\n")
    L.append("| 條件 | claim 數 | L1 報告值（上界） | L2 句子層級 | L3 證據校正（下界） | 斷句假影 | 規則漏判 | 生成崩潰 | 真實無支持 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for b in summary["ucr_bounds"]:
        L.append(f"| {b['condition']} | {b['n_claims']} | {b['UCR_L1_reported_upper']*100:.2f}% | "
                 f"{b['UCR_L2_sentence_level']*100:.2f}% | {b['UCR_L3_verification_adjusted']*100:.2f}% | "
                 f"{b['split_artifact_claims']} | {b['verified_rule_miss_claims']} | "
                 f"{b['degenerate_claims']} | {b['genuine_unattributable_claims']} |")
    L.append(f"\n> **判讀基準**：真值介於 L1 與 L3 之間。L1（子句層級）偏嚴，因為殘句必然缺少關鍵詞；"
             f"L2 只扣除斷句假影；L3 進一步扣除「規則漏判但有證據支持」者，偏寬。"
             f"主條件 `{prim}` 的區間為 **{pb['UCR_L3_verification_adjusted']*100:.2f}% – "
             f"{pb['UCR_L1_reported_upper']*100:.2f}%**（L2 = {pb['UCR_L2_sentence_level']*100:.2f}%），"
             f"定點估計需人工複核（樣板：`human_review_template.csv`，共 {summary['n_review_rows']} 列）。\n")
    L.append("> ⚠ **`wo_ltp` 條件不可與其他條件並列解讀**：該條件下 200/200 段生成均為退化輸出"
             "（重複、亂碼、空輸出），其接近 100% 的 UCR 反映的是**生成崩潰**，而非"
             "「說明缺乏偏好證據」。詳見第四節。\n")

    L.append(f"## 二、主條件（{prim}）的錯誤來源組成\n")
    total = sum(comp_counter[prim].values())
    L.append(f"以下針對 {total} 條 `no_detected_support_source` 的 claim 分類。\n")
    L.append("| 代碼 | 錯誤來源 | n | 占無來源者 | 占全部 claim |")
    L.append("|---|---|---|---|---|")
    for bucket, label in summary["bucket_labels"].items():
        cnt = comp_counter[prim][bucket]
        if not cnt:
            continue
        L.append(f"| `{bucket.split('_')[0]}` | {label} | {cnt} | {cnt/total*100:.1f}% | "
                 f"{cnt/n_claims_by_cond[prim]*100:.2f}% |")

    L.append("\n> 上表以**子句**計數。同一母句被切成多個殘句時會重複計入，"
             "故另附句子層級去重結果（`ucr_error_source_composition_sentence_level.csv`）：")
    sl = [s for s in summary["sentence_level_composition"]
          if s["condition"] == prim and s["n_sentences"]]
    L.append("\n| 錯誤來源 | 母句數 | 占比 |")
    L.append("|---|---|---|")
    for s in sl:
        L.append(f"| {s['bucket_label']} | {s['n_sentences']} | {s['share']*100:.1f}% |")

    L.append("\n## 三、各錯誤來源的代表案例\n")
    seen = set()
    for a in annotated:
        if a["condition"] != prim or a["bucket"] in seen:
            continue
        seen.add(a["bucket"])
        L.append(f"**{summary['bucket_labels'][a['bucket']]}**（`{a['subtype']}`）")
        L.append(f"- claim：`{a['claim_text']}`")
        L.append(f"- 母句：`{a['parent_sentence']}`")
        L.append(f"- 判定依據：{a['evidence_note']}\n")

    L.append("## 四、`wo_ltp` 的生成崩潰（重要發現）\n")
    dg = summary.get("degenerate_by_condition", {})
    L.append("| 條件 | 退化生成段數 / 總段數 |")
    L.append("|---|---|")
    for cond, (n_deg, n_tot) in dg.items():
        L.append(f"| {cond} | {n_deg} / {n_tot} |")
    L.append("\n退化判定於**整段生成**層級進行（非子句），準則為：有效英文詞數 < "
             f"{DEGEN_MIN_TOKENS}、type-token ratio < {DEGEN_TTR_MAX}、單一詞占比 > "
             f"{DEGEN_TOPREP_MAX}、或平均詞長 ≤ {DEGEN_MEANLEN_MAX}。"
             "門檻以既有資料校準，`full` 與 `wo_prompt` 皆為 0 誤判。\n")
    L.append("**論述含意**：在此已訓練 checkpoint 與 OOD 歸零設定下，推論時移除 "
             "[LTP] 前綴會使生成整體崩潰，而非僅降低偏好接地程度。因此 `wo_ltp` 的 "
             "UCR 不可用來支持「說明內容依賴偏好證據」或「LTP 普遍為必要輸入」等主張；"
             "它只顯示此模型對既有前綴結構高度敏感。此結果並與排序側的 No-LTP（exp_04，"
             "重新訓練且無 LTP）明確區分，後者才是偏好資訊有效性的模型層級對照。\n")

    L.append("## 五、方法與限制\n")
    for k, v in summary["method"].items():
        L.append(f"- **{k}**：{v}")
    L.append(f"- 標記需人工複核的 claim 共 {summary['n_needs_human_review']} 條；"
             "規則層只做可稽核的初判，最終歸類以人工複核為準。")
    L.append("- 本分析未重跑任何生成，全部基於既有 `counterfactual_generations_top1.csv`。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
