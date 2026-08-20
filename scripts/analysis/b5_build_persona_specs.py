"""
用途：建立或分析 persona 條件下的 LTP 與評估結果。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import csv
import datetime as _dt
import io
import json
import os
from collections import Counter

OUT_DIR = Path(os.environ.get(
    "B5_PERSONA_OUT_DIR",
    PROJECT_ROOT / "results" / "analysis" / "b5_personas_v21",
))
CLUSTER_CSV = (PROJECT_ROOT / "results" / "analysis" / "video_clusters"
               / "video_cluster_assignments_named.csv")

MUSIC_METADATA = Path(os.environ.get(
    "MUSIC_METADATA_PATH",
    r"data/user_profiling/music_metadata_simple\music_metadata_enriched.json",
))
YOUTUBE_METADATA = Path(os.environ.get(
    "YOUTUBE_METADATA_PATH",
    r"data/user_profiling/music_metadata_simple\youtube_metadata.jsonl",
))

MIN_POOL = 200          # 核心條件至少要有這麼多首才算可行
TARGET_HISTORY = 20     # 14/4/2，精確對應 70/20/10

# 顯著人聲標籤：用於「人聲敘事型」的必要條件與「純音樂背景型」的排除條件
VOCAL_TAGS = {"vocal", "vocals", "singing", "male vocal", "female vocal",
              "female vocalists", "male voice", "female voice", "voice",
              "male", "female", "man", "woman", "choir", "opera"}

# ---------------------------------------------------------------------------
# 6 種偏好原型（欄位值皆已對照實際標籤頻率挑選，確保可行）
# ---------------------------------------------------------------------------
PROTOTYPES = {
    "P1_high_energy": {
        "label": "高能量節奏型",
        "preferred_genres": ["techno", "dance", "house", "electro"],
        "rejected_genres": ["ambient", "chillout", "folk", "blues"],
        "tempo": "fast",
        "energy": "high",
        "valence": "正向、振奮",
        "vocal": "any",
        "instruments": ["drums", "beat", "synth"],
        "popularity": "mainstream",
        "novelty": "low",
        "consistency": "high",
        "selection_rule": "優先挑選節奏明快、音量飽滿的電子舞曲，避免慢速或氛圍取向的曲目。",
    },
    "P2_soft_ambient": {
        "label": "柔和氛圍型",
        "preferred_genres": ["ambient", "chillout", "acoustic", "folk"],
        "rejected_genres": ["metal", "heavy metal", "punk", "hard rock"],
        "tempo": "slow",
        "energy": "low",
        "valence": "平靜、內省",
        "vocal": "any",
        "instruments": ["guitar", "piano", "strings"],
        "popularity": "any",
        "novelty": "low",
        "consistency": "high",
        "selection_rule": "偏好慢速、不吵雜的氛圍或原音曲目，排斥高失真與強烈打擊樂。",
    },
    "P3_vocal_narrative": {
        "label": "人聲敘事型",
        "preferred_genres": ["pop", "rnb", "soul"],
        "rejected_genres": ["techno", "experimental"],
        "tempo": "any",
        "energy": "any",
        "valence": "情感豐富、敘事性強",
        "vocal": "vocal_required",
        "instruments": ["guitar", "piano"],
        "popularity": "mainstream",
        "novelty": "low",
        "consistency": "medium",
        "selection_rule": "以人聲為主的曲目為核心，重視歌唱表現與敘事性。",
    },
    "P4_instrumental_bed": {
        "label": "純音樂背景型",
        "preferred_genres": ["electronic", "ambient", "chillout"],
        "rejected_genres": ["rnb", "soul"],
        "tempo": "any",
        "energy": "any",
        "valence": "中性、不搶戲",
        "vocal": "instrumental_leaning",
        "instruments": ["synth", "guitar", "strings"],
        "popularity": "niche",
        "novelty": "medium",
        "consistency": "medium",
        "selection_rule": "偏好無顯著人聲的背景性曲目，避免歌聲搶走影片旁白的注意力。",
    },
    "P5_genre_consistent": {
        "label": "曲風一致型",
        "preferred_genres": ["rock", "alternative rock", "indie rock"],
        "rejected_genres": ["techno", "house", "hip-hop"],
        "tempo": "any",
        "energy": "any",
        "valence": "一致的樂團感",
        "vocal": "any",
        "instruments": ["guitar", "drums"],
        "popularity": "any",
        "novelty": "very_low",
        "consistency": "very_high",
        "selection_rule": "跨影片維持單一曲風家族，幾乎不跨出搖滾範圍。",
    },
    "P6_exploratory": {
        "label": "探索多樣型",
        "preferred_genres": ["indie", "alternative", "funk", "jazz", "country", "hip-hop"],
        "rejected_genres": [],
        "tempo": "any",
        "energy": "any",
        "valence": "多變",
        "vocal": "any",
        "instruments": [],
        "popularity": "niche",
        "novelty": "high",
        "consistency": "low",
        "selection_rule": "刻意跨曲風選曲，偏好小眾與新鮮感，不追求跨影片一致性。",
    },
}

# 欄位是否可用於曲目篩選與評分
FIELD_OPERATIONALIZABLE = {
    "preferred_genres": True, "rejected_genres": True, "tempo": True,
    "energy": True, "vocal": True, "instruments": True, "popularity": True,
    "novelty": True, "consistency": True, "selection_rule": False,
    "valence": False,      # 標籤詞彙中完全無 valence 對應項
}

FIELD_NOTES = {
    "tempo": "以 musicnn 標籤 fast / slow 定義，非 BPM 物理量測",
    "energy": "以 loud 標籤之有無定義（high = 具 loud；low = 不具 loud）",
    "vocal": "vocal_required = 具顯著人聲標籤；instrumental_leaning = 不具任何顯著人聲標籤"
             "（instrumental 標籤僅 424 首，過於稀疏故不採為必要條件）",
    "popularity": "以 youtube view_count 分位數定義：mainstream ≥ p75、niche ≤ p25",
    "novelty": "控制合成歷史中 70/20/10 三層的取樣半徑，不直接篩選曲目",
    "consistency": "控制所選曲目的屬性離散度，不直接篩選曲目",
    "valence": "本資料集標籤詞彙（共 80 詞）無任何情緒價向對應項，故不可操作化，"
               "僅供自然語言 Persona 描述使用",
}


def load_metadata():
    md = json.loads(MUSIC_METADATA.read_text(encoding="utf-8"))
    views = {}
    with io.open(YOUTUBE_METADATA, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("view_count") is not None:
                views[d["music_id"]] = d["view_count"]
    return md, views


def load_contexts():
    """由 B4 的 k=4 叢集取得四種內容情境。"""
    if not CLUSTER_CSV.exists():
        raise FileNotFoundError(f"找不到 B4 叢集結果：{CLUSTER_CSV}\n請先完成 B4。")
    with open(CLUSTER_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ctx = {}
    for r in rows:
        c = int(r["cluster_k4"])
        ctx.setdefault(c, {"cluster": c, "name": r["cluster_k4_name"], "n_videos": 0})
        ctx[c]["n_videos"] += 1
    return [ctx[c] for c in sorted(ctx)]


def track_tags(entry):
    tags = {str(t).lower() for t in (entry.get("tags") or [])}
    if entry.get("genre"):
        tags.add(str(entry["genre"]).lower())
    return tags


def matches_core(tags, views, spec, p25, p75):
    """核心條件：偏好曲風命中 + 排斥曲風未命中 + 節奏/能量/人聲/熱門度符合。"""
    if spec["preferred_genres"] and not (tags & set(spec["preferred_genres"])):
        return False
    if spec["rejected_genres"] and (tags & set(spec["rejected_genres"])):
        return False
    if spec["tempo"] == "fast" and "fast" not in tags:
        return False
    if spec["tempo"] == "slow" and "slow" not in tags:
        return False
    if spec["energy"] == "high" and "loud" not in tags:
        return False
    if spec["energy"] == "low" and "loud" in tags:
        return False
    if spec["vocal"] == "vocal_required" and not (tags & VOCAL_TAGS):
        return False
    if spec["vocal"] == "instrumental_leaning" and (tags & VOCAL_TAGS):
        return False
    if spec["popularity"] == "mainstream" and (views is None or views < p75):
        return False
    if spec["popularity"] == "niche" and (views is None or views > p25):
        return False
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md, views = load_metadata()
    contexts = load_contexts()

    vc = sorted(views.values())
    p25 = vc[int(len(vc) * 0.25)]
    p75 = vc[int(len(vc) * 0.75)]
    print(f"view_count 分位數：p25={p25:,}　p75={p75:,}")
    print(f"內容情境（B4 k=4 叢集）：" +
          "、".join(f"{c['name']}({c['n_videos']})" for c in contexts))

    # ---- 逐原型計算候選池 ---------------------------------------------------
    pools = {}
    for pid, spec in PROTOTYPES.items():
        pool = [mid for mid, e in md.items()
                if matches_core(track_tags(e), views.get(mid), spec, p25, p75)]
        pools[pid] = pool
        print(f"  {pid:22s} {spec['label']:8s} 候選曲目 = {len(pool):6d}")

    # ---- 組出 24 個 Persona -------------------------------------------------
    personas, feasibility = [], []
    for pid, spec in PROTOTYPES.items():
        for ctx in contexts:
            persona_id = f"{pid}__c{ctx['cluster']}"
            pool_n = len(pools[pid])
            ok = pool_n >= MIN_POOL
            persona = {
                "persona_id": persona_id,
                "prototype_id": pid,
                "prototype_label": spec["label"],
                "context_cluster": ctx["cluster"],
                "context_label": ctx["name"],
                "context_n_query_videos": ctx["n_videos"],
                "history_length": TARGET_HISTORY,
                "history_mix": {"core": 0.70, "adjacent": 0.20, "off": 0.10},
                "attributes": {
                    field: {
                        "value": spec[field],
                        "operationalizable": FIELD_OPERATIONALIZABLE[field],
                        "note": FIELD_NOTES.get(field, ""),
                    }
                    for field in ["preferred_genres", "rejected_genres", "tempo", "energy",
                                  "valence", "vocal", "instruments", "popularity",
                                  "novelty", "consistency", "selection_rule"]
                },
                "candidate_pool_size": pool_n,
                "feasible": ok,
                "description": (
                    f"這位創作者主要產出「{ctx['name']}」類型的影片，配樂偏好為"
                    f"「{spec['label']}」：{spec['selection_rule']}"
                ),
            }
            personas.append(persona)
            feasibility.append({
                "persona_id": persona_id, "prototype": spec["label"],
                "context": ctx["name"], "candidate_pool_size": pool_n,
                "min_required": MIN_POOL, "history_length": TARGET_HISTORY,
                "feasible": ok,
            })

    n_ok = sum(1 for f in feasibility if f["feasible"])
    print(f"\n共 {len(personas)} 個 Persona，可行 {n_ok}／{len(personas)}")

    spec_doc = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_personas": len(personas),
        "structure": "6 偏好原型 × 4 內容情境（B4 k=4 影片語意叢集）",
        "design_notes": [
            "內容情境改用 B4 影片語意叢集，而非教授原案的旅遊／美食／知識教學／日常生活："
            "本資料集的影片即音樂自身 YouTube 影片，無法操作化為創作者拍攝的內容類型。",
            "情境只決定該 Persona 配對哪些查詢影片，不限制其歷史曲目來源；"
            "原型與情境不相符的組合即為 B6 偏好—影片衝突分析的天然素材。",
            "節奏維度以 musicnn 的 fast / slow 標籤定義，非 BPM 物理量測。",
            "情緒價向（valence）在本資料集 80 個標籤詞彙中無任何對應項，"
            "標為不可操作化，僅供自然語言描述，不參與曲目篩選與評分。",
            "「純音樂背景型」以『不具任何顯著人聲標籤』定義，"
            "而非要求 instrumental 標籤（該標籤僅 424 首，佔 0.49%，過於稀疏）。",
        ],
        "view_count_quantiles": {"p25": p25, "p75": p75},
        "vocal_tags_used": sorted(VOCAL_TAGS),
        "field_operationalizable": FIELD_OPERATIONALIZABLE,
        "field_notes": FIELD_NOTES,
        "prototypes": PROTOTYPES,
        "contexts": contexts,
        "personas": personas,
    }
    (OUT_DIR / "persona_specs.json").write_text(
        json.dumps(spec_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(OUT_DIR / "persona_feasibility.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(feasibility[0].keys()))
        w.writeheader()
        w.writerows(feasibility)

    write_markdown(OUT_DIR / "persona_specs.md", spec_doc, pools)
    print(f"已輸出：{OUT_DIR / 'persona_specs.json'}")
    print(f"已輸出：{OUT_DIR / 'persona_specs.md'}")
    print(f"已輸出：{OUT_DIR / 'persona_feasibility.csv'}")


def write_markdown(path, doc, pools):
    L = ["# B5 結構化 Persona 規格（24 個）\n",
         f"- 產生時間：{doc['generated_at']}",
         f"- 結構：{doc['structure']}",
         f"- 每個 Persona 的合成歷史長度：{TARGET_HISTORY} 首（14/4/2，精確對應核心 70% / 鄰近探索 20% / 偏離 10%）\n",
         "## 一、設計說明\n"]
    for n in doc["design_notes"]:
        L.append(f"- {n}")

    L.append("\n## 二、欄位可操作化狀態\n")
    L.append("| 欄位 | 可操作化 | 操作定義 |")
    L.append("|---|---|---|")
    for field, ok in doc["field_operationalizable"].items():
        L.append(f"| {field} | {'是' if ok else '**否**'} | {doc['field_notes'].get(field, '—')} |")

    L.append("\n## 三、六種偏好原型\n")
    L.append("| 原型 | 偏好曲風 | 排斥曲風 | 節奏 | 能量 | 人聲 | 熱門度 | 新穎性 | 一致性 | 候選曲目數 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for pid, spec in doc["prototypes"].items():
        L.append(f"| {spec['label']} | {'、'.join(spec['preferred_genres']) or '—'} | "
                 f"{'、'.join(spec['rejected_genres']) or '—'} | {spec['tempo']} | "
                 f"{spec['energy']} | {spec['vocal']} | {spec['popularity']} | "
                 f"{spec['novelty']} | {spec['consistency']} | {len(pools[pid]):,} |")

    L.append("\n## 四、四種內容情境（B4 k=4 影片語意叢集）\n")
    L.append("| 叢集 | 名稱 | 可用查詢影片數 |")
    L.append("|---|---|---|")
    for c in doc["contexts"]:
        L.append(f"| {c['cluster']} | {c['name']} | {c['n_videos']} |")

    L.append("\n## 五、24 個 Persona 一覽\n")
    L.append("| Persona ID | 偏好原型 | 內容情境 | 候選曲目 | 可行 |")
    L.append("|---|---|---|---|---|")
    for p in doc["personas"]:
        L.append(f"| `{p['persona_id']}` | {p['prototype_label']} | {p['context_label']} | "
                 f"{p['candidate_pool_size']:,} | {'✓' if p['feasible'] else '✗'} |")

    L.append("\n## 六、限制\n")
    L.append("- 情緒價向欄位不可操作化，任何以該欄位為依據的結論皆不得寫入論文。")
    L.append("- Persona 為研究者依標籤分布設計，非真實創作者；"
             "其合成歷史由曲庫依屬性條件抽樣而得，不代表真實聆聽行為。")
    L.append("- Persona LTP 由既有 LTP 向量組合而成，未走完整 Stage 3–5 管線"
             "（原因見 `results/analysis/b5_smoketest/wout_recovery_report.md`）。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
