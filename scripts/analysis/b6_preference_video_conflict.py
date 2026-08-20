"""
用途：整理實驗輸出並產生論文分析用表格或圖表。
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
import json
import re
from collections import defaultdict

import numpy as np

from scripts.analysis import b5_build_persona_specs as SPEC


EVAL_DIR = PROJECT_ROOT / "results" / "main_eval" / "exp_01" / "persona_eval_v2"
PERSONA_DIR = PROJECT_ROOT / "results" / "analysis" / "b5_personas"
CLUSTER_CSV = (PROJECT_ROOT / "results" / "analysis" / "video_clusters"
               / "video_cluster_assignments_named.csv")
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "b6_conflict"

GROUP_LABELS = ["明顯衝突", "部分一致", "高度一致"]      # 由低到高相容度
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260726
CI_ALPHA = 0.05

# 讓步／對比語：用以判定說明是否「承認」偏好與影片之間的張力
CONCESSION_MARKERS = [
    "although", "though", "while", "however", "but ", "despite", "even though",
    "on the other hand", "that said", "nevertheless", "instead of", "rather than",
    "may not", "might not", "doesn't quite", "does not quite", "a bit different",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    u = a | b
    return len(a & b) / len(u) if u else float("nan")


def cluster_bootstrap(values_by_persona, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED,
                      alpha=CI_ALPHA):
    pids = [p for p, v in values_by_persona.items() if v]
    if not pids:
        return float("nan"), float("nan"), float("nan")
    flat = [v for p in pids for v in values_by_persona[p]]
    point = float(np.mean(flat))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.integers(0, len(pids), size=len(pids))
        vals = [v for i in sel for v in values_by_persona[pids[i]]]
        draws[b] = np.mean(vals) if vals else np.nan
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    return (point, float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs_doc = json.loads((PERSONA_DIR / "persona_specs.json").read_text(encoding="utf-8"))
    prototypes = specs_doc["prototypes"]
    p25 = specs_doc["view_count_quantiles"]["p25"]
    p75 = specs_doc["view_count_quantiles"]["p75"]
    personas = {p["persona_id"]: p for p in specs_doc["personas"]}

    md, views = SPEC.load_metadata()

    # ---- 1. 各叢集影片的 GT 音樂 → 標籤分布 ---------------------------------
    cluster_rows = read_csv(CLUSTER_CSV)
    cluster_gt = defaultdict(list)      # cluster → [gt music id]
    cluster_name = {}
    for r in cluster_rows:
        c = int(r["cluster_k4"])
        cluster_name[c] = r["cluster_k4_name"]
        cluster_gt[c].append(r["gt_music_id"][:11])
    print("叢集：" + "、".join(f"{c}:{cluster_name[c]}({len(cluster_gt[c])})"
                            for c in sorted(cluster_gt)))

    # ---- 2. 相容度：叢集音樂中符合該原型核心規格的比例 ----------------------
    cells = []
    for pid, p in personas.items():
        spec = prototypes[p["prototype_id"]]
        c = p["context_cluster"]
        ids = cluster_gt[c]
        hits = sum(1 for m in ids
                   if m in md and SPEC.matches_core(SPEC.track_tags(md[m]),
                                                    views.get(m), spec, p25, p75))
        cells.append({"persona_id": pid, "prototype": p["prototype_label"],
                      "cluster": c, "cluster_name": cluster_name[c],
                      "n_cluster_videos": len(ids),
                      "compatibility": hits / len(ids) if ids else float("nan")})

    comp = np.array([c["compatibility"] for c in cells])
    edges = np.percentile(comp, [100 / 3, 200 / 3])
    for c in cells:
        idx = int(np.digitize(c["compatibility"], edges, right=False))
        c["conflict_group"] = GROUP_LABELS[idx]
    print(f"相容度三等分切點：{edges[0]:.4f} / {edges[1]:.4f}")
    for g in GROUP_LABELS:
        sub = [c for c in cells if c["conflict_group"] == g]
        print(f"  {g}：{len(sub)} 格，相容度 "
              f"{min(x['compatibility'] for x in sub):.4f}–"
              f"{max(x['compatibility'] for x in sub):.4f}")
    group_of = {c["persona_id"]: c["conflict_group"] for c in cells}

    # ---- 3. 逐筆推薦計算指標 ------------------------------------------------
    def analyse(condition):
        path = EVAL_DIR / f"persona_v2_{condition}.csv"
        if not path.exists():
            return None
        acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in read_csv(path):
            pid = r["persona_id"]
            p = personas.get(pid)
            if p is None:
                continue
            spec = prototypes[p["prototype_id"]]
            grp = group_of[pid]

            top = [x[:11] for x in (r.get("top10_pair_keys") or "").split(";") if x]
            if not top:
                top = [r["top1_pair_key"][:11]]
            gt_music = r["gt_pair_key"][:11]
            gt_tags = SPEC.track_tags(md[gt_music]) if gt_music in md else set()
            top1_tags = SPEC.track_tags(md[top[0]]) if top[0] in md else set()

            # Persona-fit（偏好側）
            vals = []
            for a in ["genre", "tempo", "energy", "vocal", "popularity"]:
                v = _attr(top1_tags, views.get(top[0]), spec, a, p25, p75)
                if v is not None:
                    vals.append(float(v))
            pf = float(np.mean(vals)) if vals else np.nan

            # Video-fit（影片側）：與該影片 GT 音樂的標籤 Jaccard
            vf = jaccard(top1_tags, gt_tags)

            # 清單多樣性：top-10 兩兩標籤 Jaccard 的補數
            tagsets = [SPEC.track_tags(md[m]) for m in top if m in md]
            if len(tagsets) >= 2:
                sims = [jaccard(tagsets[i], tagsets[j])
                        for i in range(len(tagsets)) for j in range(i + 1, len(tagsets))]
                sims = [s for s in sims if not np.isnan(s)]
                div = 1.0 - float(np.mean(sims)) if sims else np.nan
            else:
                div = np.nan

            text = (r.get("generated_text") or "").lower()
            concede = float(any(m in text for m in CONCESSION_MARKERS)) if text else np.nan

            rank = int(r["rank"])
            for key, val in [("persona_fit", pf), ("video_fit", vf), ("diversity", div),
                             ("concession", concede), ("R@1", float(rank <= 1)),
                             ("MRR", 1.0 / rank),
                             ("lean", (pf - vf) if not (np.isnan(pf) or np.isnan(vf)) else np.nan)]:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    acc[grp][key][pid].append(val)
        return acc

    def _attr(tags, v, spec, a, lo, hi):
        from scripts.analysis.b5_persona_metrics_v2 import attr_satisfied
        return attr_satisfied(tags, v, spec, a, lo, hi)

    acc_matched = analyse("matched")
    acc_noltp = analyse("no_ltp")
    if acc_matched is None:
        raise SystemExit("找不到 persona_v2_matched.csv，請先完成 B5 v2 評估。")

    METRICS = ["persona_fit", "video_fit", "lean", "diversity", "concession", "R@1", "MRR"]
    group_rows = []
    for grp in GROUP_LABELS:
        row = {"conflict_group": grp,
               "n_personas": sum(1 for c in cells if c["conflict_group"] == grp),
               "mean_compatibility": float(np.mean(
                   [c["compatibility"] for c in cells if c["conflict_group"] == grp]))}
        for m in METRICS:
            pt, lo, hi = cluster_bootstrap(acc_matched[grp][m])
            row[m] = pt
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi
        if acc_noltp:
            for m in ["persona_fit", "video_fit", "R@1"]:
                row[f"noltp_{m}"] = cluster_bootstrap(acc_noltp[grp][m])[0]
        group_rows.append(row)
        print(f"[{grp}] persona_fit={row['persona_fit']:.4f} video_fit={row['video_fit']:.4f} "
              f"lean={row['lean']:+.4f} diversity={row['diversity']:.4f} "
              f"concession={row['concession']:.4f} R@1={row['R@1']:.4f}")

    # ---- 4. 組間差異檢定（衝突 vs 一致）------------------------------------
    contrasts = []
    for m in METRICS:
        a, b = acc_matched["高度一致"][m], acc_matched["明顯衝突"][m]
        pids = sorted(set(a) | set(b))
        diffs = {}
        for pid in pids:
            if pid in a and pid in b:
                diffs[pid] = [float(np.mean(a[pid]) - np.mean(b[pid]))]
        # 兩組的 Persona 互斥，故改以獨立重抽比較
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        pa = [p for p in a if a[p]]
        pb = [p for p in b if b[p]]
        if not pa or not pb:
            continue
        point = float(np.mean([v for p in pa for v in a[p]])
                      - np.mean([v for p in pb for v in b[p]]))
        draws = np.empty(BOOTSTRAP_N)
        for i in range(BOOTSTRAP_N):
            sa = rng.choice(pa, size=len(pa), replace=True)
            sb = rng.choice(pb, size=len(pb), replace=True)
            draws[i] = (np.mean([v for p in sa for v in a[p]])
                        - np.mean([v for p in sb for v in b[p]]))
        lo = float(np.percentile(draws, 100 * CI_ALPHA / 2))
        hi = float(np.percentile(draws, 100 * (1 - CI_ALPHA / 2)))
        contrasts.append({"metric": m, "comparison": "高度一致 - 明顯衝突",
                          "difference": point, "ci_low": lo, "ci_high": hi,
                          "significant": bool(lo > 0 or hi < 0)})
        print(f"[對照] {m:12s} align vs conflict = {point:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"{' significant' if (lo > 0 or hi < 0) else ' n.s.'}")

    # ---- 5. 逐原型的偏向 ----------------------------------------------------
    proto_rows = []
    by_proto = defaultdict(lambda: defaultdict(list))
    for grp in GROUP_LABELS:
        for m in ["persona_fit", "video_fit", "lean"]:
            for pid, vals in acc_matched[grp][m].items():
                by_proto[personas[pid]["prototype_label"]][m].extend(vals)
    for proto, d in sorted(by_proto.items()):
        proto_rows.append({"prototype": proto,
                           "persona_fit": float(np.mean(d["persona_fit"])),
                           "video_fit": float(np.mean(d["video_fit"])),
                           "lean": float(np.mean(d["lean"]))})

    # ---- 6. 輸出 ------------------------------------------------------------
    def dump(name, rows):
        if not rows:
            return
        with open(OUT_DIR / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("conflict_cell_compatibility.csv", cells)
    dump("conflict_group_metrics.csv", group_rows)
    dump("conflict_contrasts.csv", contrasts)
    dump("conflict_by_prototype.csv", proto_rows)

    summary = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "compatibility_edges": [float(e) for e in edges],
        "cells": cells, "group_metrics": group_rows,
        "contrasts": contrasts, "by_prototype": proto_rows,
        "notes": [
            "衝突程度由資料決定：相容度＝該叢集影片的 GT 音樂中符合此原型核心規格的比例，"
            "再三等分；未由研究者主觀指派。",
            "Video-fit 以 top-1 與該影片 GT 音樂的標籤 Jaccard 操作化，"
            "為相似度代理而非人工判定的適配度。",
            "說明衝突承認率以讓步／對比語（although / while / however …）判定，"
            "屬規則式詞彙偵測，與 B2/B3 同層級限制。",
            "lean = Persona-fit − Video-fit：正值代表推薦偏向偏好、負值偏向影片。",
            "GT 的 R@1 僅作操作檢查，非主指標（見 B5）。",
        ],
    }
    (OUT_DIR / "conflict_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# B6 偏好與影片情境的一致／衝突分析\n",
         f"- 產生時間：{summary['generated_at']}",
         f"- 相容度三等分切點：{edges[0]:.4f} / {edges[1]:.4f}",
         "- 資料：B5 v2 的 matched 條件（24 Persona × 20 支查詢 = 480 筆）\n",
         "## 一、三組的取捨行為\n",
         "| 組別 | 格數 | 平均相容度 | Persona-fit | Video-fit | lean（偏好−影片）| 清單多樣性 | 說明承認衝突 | GT R@1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in group_rows:
        L.append(f"| {r['conflict_group']} | {r['n_personas']} | "
                 f"{r['mean_compatibility']*100:.2f}% | {r['persona_fit']*100:.2f}% | "
                 f"{r['video_fit']*100:.2f}% | {r['lean']*100:+.2f}pp | "
                 f"{r['diversity']*100:.2f}% | {r['concession']*100:.2f}% | "
                 f"{r['R@1']*100:.2f}% |")
    L.append("\n## 二、高度一致 vs 明顯衝突（獨立重抽 95% CI）\n")
    L.append("| 指標 | 差異 | 95% CI | 顯著 |")
    L.append("|---|---|---|---|")
    for c in contrasts:
        L.append(f"| {c['metric']} | {c['difference']*100:+.2f}pp | "
                 f"[{c['ci_low']*100:+.2f}, {c['ci_high']*100:+.2f}] | "
                 f"{'是' if c['significant'] else '否'} |")
    L.append("\n## 三、逐偏好原型的偏向\n")
    L.append("| 偏好原型 | Persona-fit | Video-fit | lean |")
    L.append("|---|---|---|---|")
    for r in proto_rows:
        L.append(f"| {r['prototype']} | {r['persona_fit']*100:.2f}% | "
                 f"{r['video_fit']*100:.2f}% | {r['lean']*100:+.2f}pp |")
    L.append("\n## 四、方法與限制\n")
    for n in summary["notes"]:
        L.append(f"- {n}")
    (OUT_DIR / "conflict_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
