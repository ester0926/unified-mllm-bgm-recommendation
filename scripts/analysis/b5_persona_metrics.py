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
import json
from collections import defaultdict

import numpy as np

from scripts.analysis import b5_build_persona_specs as SPEC


EVAL_DIR = PROJECT_ROOT / "results" / "main_eval" / "exp_01" / "persona_eval"
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "b5_personas"
SPECS_JSON = OUT_DIR / "persona_specs.json"

CONDITIONS = ["matched", "shuffled", "random",
              "cf_tempo", "cf_energy", "cf_vocal", "cf_popularity", "cf_consistency"]
CF_TO_ATTR = {"cf_tempo": "tempo", "cf_energy": "energy", "cf_vocal": "vocal",
              "cf_popularity": "popularity", "cf_consistency": None}   # consistency 非曲目屬性

ATTRS = ["genre", "tempo", "energy", "vocal", "popularity"]

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260726
CI_ALPHA = 0.05


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def attr_satisfied(tags, views, spec, attr, p25, p75):
    """單一屬性是否符合規格。回傳 True/False/None（None = 該屬性對此 Persona 無約束）。"""
    if attr == "genre":
        if not spec["preferred_genres"]:
            return None
        if spec["rejected_genres"] and (tags & set(spec["rejected_genres"])):
            return False
        return bool(tags & set(spec["preferred_genres"]))
    if attr == "tempo":
        if spec["tempo"] == "any":
            return None
        return spec["tempo"] in tags
    if attr == "energy":
        if spec["energy"] == "any":
            return None
        return ("loud" in tags) if spec["energy"] == "high" else ("loud" not in tags)
    if attr == "vocal":
        if spec["vocal"] == "any":
            return None
        has = bool(tags & SPEC.VOCAL_TAGS)
        return has if spec["vocal"] == "vocal_required" else (not has)
    if attr == "popularity":
        if spec["popularity"] == "any" or views is None:
            return None
        return views >= p75 if spec["popularity"] == "mainstream" else views <= p25
    return None


def cluster_bootstrap(values_by_persona, n_boot, seed, alpha):
    """以 Persona 為重抽單位（同一 Persona 的查詢彼此相關，不可直接對查詢重抽）。"""
    pids = list(values_by_persona)
    flat = [v for p in pids for v in values_by_persona[p]]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(flat))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.integers(0, len(pids), size=len(pids))
        vals = [v for i in sel for v in values_by_persona[pids[i]]]
        draws[b] = np.mean(vals) if vals else np.nan
    draws = draws[~np.isnan(draws)]
    return (point,
            float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs_doc = json.loads(SPECS_JSON.read_text(encoding="utf-8"))
    prototypes = specs_doc["prototypes"]
    p25 = specs_doc["view_count_quantiles"]["p25"]
    p75 = specs_doc["view_count_quantiles"]["p75"]
    persona_spec = {p["persona_id"]: prototypes[p["prototype_id"]]
                    for p in specs_doc["personas"]}

    md, views = SPEC.load_metadata()
    print(f"metadata={len(md)}　view_count={len(views)}")

    # ---- 逐條件計算每筆推薦的屬性符合情形 ----------------------------------
    # per_case[condition][persona_id] 存放每個屬性的 True/False/None 判斷
    per_case = {c: defaultdict(list) for c in CONDITIONS}
    gt_rank = {c: defaultdict(list) for c in CONDITIONS}

    for cond in CONDITIONS:
        path = EVAL_DIR / f"persona_ranking_{cond}.csv"
        if not path.exists():
            print(f"⚠ 缺少 {path.name}，跳過")
            continue
        for r in read_csv(path):
            pid = r["persona_id"]
            spec = persona_spec.get(pid)
            if spec is None:
                continue
            top1_music = r["top1_pair_key"][:11]      # pair_key → target music id
            entry = md.get(top1_music)
            if entry is None:
                continue
            tags = SPEC.track_tags(entry)
            v = views.get(top1_music)
            per_case[cond][pid].append(
                {a: attr_satisfied(tags, v, spec, a, p25, p75) for a in ATTRS})
            gt_rank[cond][pid].append(int(r["rank"]))
        print(f"  {cond:16s} 有效推薦 {sum(len(v) for v in per_case[cond].values())} 筆")

    # ---- ACR / Persona-fit@1 -----------------------------------------------
    rows = []
    for cond in CONDITIONS:
        if not per_case[cond]:
            continue
        acr_by_p, fit_by_p = defaultdict(list), defaultdict(list)
        attr_rate = {a: defaultdict(list) for a in ATTRS}
        for pid, cases in per_case[cond].items():
            for c in cases:
                vals = [v for v in c.values() if v is not None]
                acr_by_p[pid].append(float(np.mean(vals)) if vals else np.nan)
                fit_by_p[pid].append(float(all(vals)) if vals else np.nan)
                for a in ATTRS:
                    if c[a] is not None:
                        attr_rate[a][pid].append(float(c[a]))
        acr_by_p = {k: [x for x in v if not np.isnan(x)] for k, v in acr_by_p.items()}
        fit_by_p = {k: [x for x in v if not np.isnan(x)] for k, v in fit_by_p.items()}

        acr, acr_lo, acr_hi = cluster_bootstrap(acr_by_p, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)
        fit, fit_lo, fit_hi = cluster_bootstrap(fit_by_p, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)
        ranks = np.array([r for p in gt_rank[cond].values() for r in p], dtype=float)
        row = {"condition": cond, "n_cases": sum(len(v) for v in per_case[cond].values()),
               "ACR": acr, "ACR_ci_low": acr_lo, "ACR_ci_high": acr_hi,
               "persona_fit@1": fit, "fit_ci_low": fit_lo, "fit_ci_high": fit_hi,
               "gt_R@1": float(np.mean(ranks <= 1)), "gt_MRR": float(np.mean(1.0 / ranks))}
        for a in ATTRS:
            sub = {k: v for k, v in attr_rate[a].items() if v}
            row[f"rate_{a}"] = cluster_bootstrap(sub, 200, BOOTSTRAP_SEED, CI_ALPHA)[0] \
                if sub else float("nan")
        rows.append(row)
        print(f"[{cond:16s}] ACR={acr:.4f} [{acr_lo:.4f},{acr_hi:.4f}]  "
              f"fit@1={fit:.4f}  GT R@1={row['gt_R@1']:.4f}")

    # ---- Matched 與 Shuffled 的差距 -----------------------------------------
    def paired_gap(cond_a, cond_b, metric="ACR"):
        pids = sorted(set(per_case[cond_a]) & set(per_case[cond_b]))
        diffs = {}
        for pid in pids:
            def _m(cond):
                out = []
                for c in per_case[cond][pid]:
                    vals = [v for v in c.values() if v is not None]
                    if vals:
                        out.append(float(np.mean(vals)) if metric == "ACR" else float(all(vals)))
                return out
            a, b = _m(cond_a), _m(cond_b)
            if a and b:
                diffs[pid] = [np.mean(a) - np.mean(b)]
        return cluster_bootstrap(diffs, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)

    gaps = []
    for a, b in [("matched", "shuffled"), ("matched", "random"), ("shuffled", "random")]:
        if per_case[a] and per_case[b]:
            pt, lo, hi = paired_gap(a, b)
            gaps.append({"comparison": f"{a} - {b}", "metric": "ACR", "difference": pt,
                         "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
            print(f"[Gap] {a} vs {b}: dACR={pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"{' significant' if (lo > 0 or hi < 0) else ' n.s.'}")

    # ---- CDA / IAD ----------------------------------------------------------
    cda_rows = []
    for cond, attr in CF_TO_ATTR.items():
        if not per_case.get(cond) or not per_case.get("matched"):
            continue
        pids = sorted(set(per_case[cond]) & set(per_case["matched"]))
        cda_by_p, iad_by_p = defaultdict(list), defaultdict(list)
        for pid in pids:
            m_cases, c_cases = per_case["matched"][pid], per_case[cond][pid]
            n = min(len(m_cases), len(c_cases))
            for i in range(n):
                if attr is not None:
                    mv, cv = m_cases[i].get(attr), c_cases[i].get(attr)
                    # 方向命中：原本符合 → 翻轉後不符合（即朝相反屬性移動）
                    if mv is not None and cv is not None:
                        cda_by_p[pid].append(float(mv and not cv))
                others = [a for a in ATTRS if a != attr]
                changed = [float(m_cases[i][a] != c_cases[i][a])
                           for a in others
                           if m_cases[i][a] is not None and c_cases[i][a] is not None]
                if changed:
                    iad_by_p[pid].append(float(np.mean(changed)))
        cda = cluster_bootstrap(cda_by_p, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA) \
            if cda_by_p else (float("nan"),) * 3
        iad = cluster_bootstrap(iad_by_p, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA)
        cda_rows.append({"counterfactual": cond, "target_attribute": attr or "（非曲目屬性）",
                         "CDA": cda[0], "CDA_ci_low": cda[1], "CDA_ci_high": cda[2],
                         "IAD": iad[0], "IAD_ci_low": iad[1], "IAD_ci_high": iad[2]})
        print(f"[CF] {cond:16s} CDA={cda[0]:.4f} IAD={iad[0]:.4f}")

    # ---- 輸出 ---------------------------------------------------------------
    def dump(name, data):
        if not data:
            return
        with open(OUT_DIR / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    dump("persona_metrics_by_condition.csv", rows)
    dump("persona_metrics_gaps.csv", gaps)
    dump("persona_metrics_counterfactual.csv", cda_rows)

    summary = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "by_condition": rows, "gaps": gaps, "counterfactual": cda_rows,
        "notes": [
            "GT 的 R@1 僅作操作檢查：GT 是為影片而非為 Persona 選的曲目，"
            "Persona 向量本就不應幫助找回 GT。",
            "ACR 與 Persona-fit 僅能在 K=1 計算，因評估腳本未保存 top-K 清單；"
            "教授建議的 Persona-fit@K 與 nDCG@5/@10 需重跑並保存 top-K（約 6 小時 GPU）。",
            "情緒價向不可操作化，未納入任何指標。",
            "統計以 Persona 為叢集重抽單位，因同一 Persona 的 20 支查詢彼此相關。",
        ],
    }
    (OUT_DIR / "persona_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# B5 Persona 偏好可控性指標\n", f"- 產生時間：{summary['generated_at']}",
         "- 統計：Persona 層級叢集拔靴 2000 次、95% 百分位 CI\n",
         "## 一、各條件的屬性合規度\n",
         "| 條件 | n | ACR | 95% CI | Persona-fit@1 | GT R@1（操作檢查）|",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['condition']} | {r['n_cases']} | {r['ACR']*100:.2f}% | "
                 f"[{r['ACR_ci_low']*100:.2f}, {r['ACR_ci_high']*100:.2f}] | "
                 f"{r['persona_fit@1']*100:.2f}% | {r['gt_R@1']*100:.2f}% |")
    L.append("\n## 二、配對敏感度\n")
    L.append("| 比較 | ΔACR | 95% CI | 顯著 |")
    L.append("|---|---|---|---|")
    for g in gaps:
        L.append(f"| {g['comparison']} | {g['difference']*100:+.2f}pp | "
                 f"[{g['ci_low']*100:+.2f}, {g['ci_high']*100:+.2f}] | "
                 f"{'是' if g['significant'] else '否'} |")
    L.append("\n## 三、反事實方向性與無關屬性漂移\n")
    L.append("| 反事實 | 目標屬性 | CDA（方向命中）| IAD（無關屬性漂移）|")
    L.append("|---|---|---|---|")
    for c in cda_rows:
        L.append(f"| {c['counterfactual']} | {c['target_attribute']} | "
                 f"{c['CDA']*100:.2f}% | {c['IAD']*100:.2f}% |")
    L.append("\n## 四、限制\n")
    for n in summary["notes"]:
        L.append(f"- {n}")
    (OUT_DIR / "persona_metrics_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
