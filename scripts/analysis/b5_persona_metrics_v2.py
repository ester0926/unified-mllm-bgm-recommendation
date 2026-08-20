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
import re
from collections import defaultdict

import numpy as np

from scripts.analysis import b5_build_persona_specs as SPEC


EVAL_DIR = PROJECT_ROOT / "results" / "main_eval" / "exp_01" / "persona_eval_v2"
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "b5_personas"
SPECS_JSON = OUT_DIR / "persona_specs.json"

CONDITIONS = ["matched", "shuffled", "random", "no_ltp",
              "cf_tempo", "cf_energy", "cf_vocal", "cf_popularity", "cf_consistency"]
GEN_CONDITIONS = ["matched", "shuffled", "random", "no_ltp"]
CF_TO_ATTR = {"cf_tempo": "tempo", "cf_energy": "energy", "cf_vocal": "vocal",
              "cf_popularity": "popularity", "cf_consistency": None}

ATTRS = ["genre", "tempo", "energy", "vocal", "popularity"]
K_VALUES = [1, 5, 10]

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260726
CI_ALPHA = 0.05

# ⑦⑧ 用的屬性詞彙（說明文字為英文）
TEMPO_TERMS = {"fast": {"fast", "fast-paced", "upbeat", "uptempo", "energetic tempo", "quick"},
               "slow": {"slow", "slow-paced", "slow tempo", "downtempo", "laid-back"}}
ENERGY_TERMS = {"high": {"energetic", "powerful", "intense", "loud", "driving", "high-energy"},
                "low": {"calm", "gentle", "soft", "mellow", "relaxed", "quiet", "soothing"}}
VOCAL_TERMS = {"vocal_required": {"vocals", "vocal", "singer", "singing", "male vocals",
                                  "female vocals", "lyrics"},
               "instrumental_leaning": {"instrumental", "no vocals", "without vocals"}}
POP_TERMS = {"mainstream": {"popular", "well-known", "hit", "famous", "mainstream", "chart"},
             "niche": {"underground", "lesser-known", "obscure", "niche", "indie artist"}}


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def attr_satisfied(tags, views, spec, attr, p25, p75):
    """單一屬性是否符合規格；None 表示該屬性對此 Persona 無約束。"""
    if attr == "genre":
        if not spec["preferred_genres"]:
            return None
        if spec["rejected_genres"] and (tags & set(spec["rejected_genres"])):
            return False
        return bool(tags & set(spec["preferred_genres"]))
    if attr == "tempo":
        return None if spec["tempo"] == "any" else (spec["tempo"] in tags)
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


def relevance(music_id, spec, md, views, p25, p75):
    """分級相關性：滿足的可操作化屬性比例（0–1）。無資料時回傳 None。"""
    entry = md.get(music_id)
    if entry is None:
        return None
    tags = SPEC.track_tags(entry)
    vals = [attr_satisfied(tags, views.get(music_id), spec, a, p25, p75) for a in ATTRS]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def ndcg_at_k(rels, k):
    """以檢索清單內的理想排序為 IDCG（無法得知整池理想排序，見檔頭說明）。"""
    r = [x for x in rels[:k] if x is not None]
    if not r:
        return None
    disc = 1.0 / np.log2(np.arange(2, len(r) + 2))
    dcg = float(np.sum(np.array(r) * disc))
    idcg = float(np.sum(np.array(sorted(r, reverse=True)) * disc))
    return dcg / idcg if idcg > 0 else None


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


def analyse_explanation(text, spec):
    """
    ⑦⑧：計算說明文字中與 Persona 規格一致／矛盾的屬性詞數。
    回傳 (n_supported, n_contradicted)。
    """
    if not text:
        return 0, 0
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "

    def hit(terms):
        return sum(1 for t in terms if t in low)

    sup = con = 0
    # 曲風
    sup += hit(set(spec["preferred_genres"]))
    con += hit(set(spec["rejected_genres"]))
    # 節奏
    if spec["tempo"] in TEMPO_TERMS:
        opp = "slow" if spec["tempo"] == "fast" else "fast"
        sup += hit(TEMPO_TERMS[spec["tempo"]])
        con += hit(TEMPO_TERMS[opp])
    # 能量
    if spec["energy"] in ENERGY_TERMS:
        opp = "low" if spec["energy"] == "high" else "high"
        sup += hit(ENERGY_TERMS[spec["energy"]])
        con += hit(ENERGY_TERMS[opp])
    # 人聲
    if spec["vocal"] in VOCAL_TERMS:
        opp = ("instrumental_leaning" if spec["vocal"] == "vocal_required"
               else "vocal_required")
        sup += hit(VOCAL_TERMS[spec["vocal"]])
        con += hit(VOCAL_TERMS[opp])
    # 熱門度
    if spec["popularity"] in POP_TERMS:
        opp = "niche" if spec["popularity"] == "mainstream" else "mainstream"
        sup += hit(POP_TERMS[spec["popularity"]])
        con += hit(POP_TERMS[opp])
    return sup, con


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs_doc = json.loads(SPECS_JSON.read_text(encoding="utf-8"))
    prototypes = specs_doc["prototypes"]
    p25 = specs_doc["view_count_quantiles"]["p25"]
    p75 = specs_doc["view_count_quantiles"]["p75"]
    pspec = {p["persona_id"]: prototypes[p["prototype_id"]] for p in specs_doc["personas"]}

    md, views = SPEC.load_metadata()
    print(f"metadata={len(md)}")

    # ---- 讀取各條件 ---------------------------------------------------------
    data = {}
    for cond in CONDITIONS:
        path = EVAL_DIR / f"persona_v2_{cond}.csv"
        if not path.exists():
            print(f"⚠ 缺少 {path.name}，略過該條件")
            continue
        data[cond] = read_csv(path)
        print(f"  {cond:16s} {len(data[cond])} 筆")
    if "matched" not in data:
        raise SystemExit("缺少 matched 條件，無法計算。請先完成 v2 評估。")

    # ---- 逐條件計算 ---------------------------------------------------------
    by_cond, expl_rows = [], []
    expl_by_persona = {}          # cond -> pid -> [(support, upcr), ...]
    per_case_attr = {}          # cond -> pid -> list of attr dict（供 CDA/IAD 用）

    for cond, rows in data.items():
        acr, fit, ndcg = defaultdict(list), {k: defaultdict(list) for k in K_VALUES}, \
            {k: defaultdict(list) for k in (5, 10)}
        sup_by_p, con_by_p, claim_by_p = defaultdict(list), defaultdict(list), defaultdict(list)
        per_case_attr[cond] = defaultdict(list)
        ranks = []

        for r in rows:
            pid = r["persona_id"]
            spec = pspec.get(pid)
            if spec is None:
                continue
            ranks.append(int(r["rank"]))

            topk = [x[:11] for x in (r.get("top10_pair_keys") or "").split(";") if x]
            if not topk:
                topk = [r["top1_pair_key"][:11]]
            rels = [relevance(m, spec, md, views, p25, p75) for m in topk]

            if rels and rels[0] is not None:
                acr[pid].append(rels[0])
            for k in K_VALUES:
                sub = [x for x in rels[:k] if x is not None]
                if sub:
                    fit[k][pid].append(float(np.mean([x >= 1.0 - 1e-9 for x in sub])))
            for k in (5, 10):
                v = ndcg_at_k(rels, k)
                if v is not None:
                    ndcg[k][pid].append(v)

            entry = md.get(topk[0])
            if entry is not None:
                tags = SPEC.track_tags(entry)
                per_case_attr[cond][pid].append(
                    {a: attr_satisfied(tags, views.get(topk[0]), spec, a, p25, p75)
                     for a in ATTRS})

            if cond in GEN_CONDITIONS:
                s, c = analyse_explanation(r.get("generated_text", ""), spec)
                total = s + c
                if total > 0:
                    sup_by_p[pid].append(s / total)
                    con_by_p[pid].append(c / total)
                    expl_by_persona.setdefault(cond, {}).setdefault(pid, []).append(
                        (s / total, c / total))
                claim_by_p[pid].append(float(total > 0))

        rk = np.array(ranks, dtype=float)
        row = {"condition": cond, "n": len(rows)}
        for name, d in [("ACR", acr)]:
            pt, lo, hi = cluster_bootstrap(d)
            row.update({name: pt, f"{name}_ci_low": lo, f"{name}_ci_high": hi})
        for k in K_VALUES:
            pt, lo, hi = cluster_bootstrap(fit[k])
            row.update({f"persona_fit@{k}": pt, f"fit@{k}_ci_low": lo, f"fit@{k}_ci_high": hi})
        for k in (5, 10):
            pt, lo, hi = cluster_bootstrap(ndcg[k])
            row.update({f"nDCG@{k}": pt, f"nDCG@{k}_ci_low": lo, f"nDCG@{k}_ci_high": hi})
        row["gt_R@1"] = float(np.mean(rk <= 1))
        row["gt_MRR"] = float(np.mean(1.0 / rk))
        by_cond.append(row)
        print(f"[{cond:16s}] ACR={row['ACR']:.4f} fit@1={row['persona_fit@1']:.4f} "
              f"fit@10={row['persona_fit@10']:.4f} nDCG@10={row['nDCG@10']:.4f} "
              f"GT R@1={row['gt_R@1']:.4f}")

        if cond in GEN_CONDITIONS:
            s_pt, s_lo, s_hi = cluster_bootstrap(sup_by_p)
            c_pt, c_lo, c_hi = cluster_bootstrap(con_by_p)
            r_pt, _, _ = cluster_bootstrap(claim_by_p)
            expl_rows.append({
                "condition": cond,
                "attribute_claim_support_rate": s_pt,
                "support_ci_low": s_lo, "support_ci_high": s_hi,
                "UPCR": c_pt, "UPCR_ci_low": c_lo, "UPCR_ci_high": c_hi,
                "share_with_any_attribute_claim": r_pt,
            })
            print(f"    ⑦支持率={s_pt:.4f} [{s_lo:.4f},{s_hi:.4f}]  "
                  f"⑧UPCR={c_pt:.4f} [{c_lo:.4f},{c_hi:.4f}]  "
                  f"有屬性主張的比例={r_pt:.4f}")

    # ---- Gap ---------------------------------------------------------------
    def paired_gap(a, b, metric_fn):
        pids = sorted(set(per_case_attr.get(a, {})) & set(per_case_attr.get(b, {})))
        diffs = {}
        for pid in pids:
            va, vb = metric_fn(a, pid), metric_fn(b, pid)
            if va is not None and vb is not None:
                diffs[pid] = [va - vb]
        return cluster_bootstrap(diffs)

    def mean_acr(cond, pid):
        cases = per_case_attr[cond][pid]
        vals = []
        for c in cases:
            v = [x for x in c.values() if x is not None]
            if v:
                vals.append(float(np.mean(v)))
        return float(np.mean(vals)) if vals else None

    gaps = []
    for a, b in [("matched", "shuffled"), ("matched", "no_ltp"), ("matched", "random"),
                 ("shuffled", "random"), ("shuffled", "no_ltp")]:
        if a in per_case_attr and b in per_case_attr:
            pt, lo, hi = paired_gap(a, b, mean_acr)
            gaps.append({"comparison": f"{a} - {b}", "metric": "ACR", "difference": pt,
                         "ci_low": lo, "ci_high": hi,
                         "significant": bool(lo > 0 or hi < 0)})
            print(f"[Gap] {a} vs {b}: dACR={pt:+.4f} [{lo:+.4f},{hi:+.4f}]"
                  f"{' significant' if (lo > 0 or hi < 0) else ' n.s.'}")

    # ---- ⑦⑧ 的配對檢定 -----------------------------------------------------
    # 各條件各自的 CI 會重疊並不代表差異不顯著；須以同一批 Persona 配對比較。
    expl_gaps = []
    if expl_by_persona:
        for a, b in [("matched", "shuffled"), ("matched", "no_ltp"),
                     ("matched", "random"), ("shuffled", "no_ltp")]:
            if a not in expl_by_persona or b not in expl_by_persona:
                continue
            for metric, idx in [("support_rate", 0), ("UPCR", 1)]:
                diffs = {}
                for pid in sorted(set(expl_by_persona[a]) & set(expl_by_persona[b])):
                    va = [x[idx] for x in expl_by_persona[a][pid]]
                    vb = [x[idx] for x in expl_by_persona[b][pid]]
                    if va and vb:
                        diffs[pid] = [float(np.mean(va) - np.mean(vb))]
                pt, lo, hi = cluster_bootstrap(diffs)
                expl_gaps.append({"comparison": f"{a} - {b}", "metric": metric,
                                  "difference": pt, "ci_low": lo, "ci_high": hi,
                                  "significant": bool(lo > 0 or hi < 0)})
                print(f"[⑦⑧ Gap] {a} vs {b} {metric}: {pt:+.4f} [{lo:+.4f},{hi:+.4f}]"
                      f"{' significant' if (lo > 0 or hi < 0) else ' n.s.'}")

    # ---- CDA / IAD ---------------------------------------------------------
    cda_rows = []
    for cond, attr in CF_TO_ATTR.items():
        if cond not in per_case_attr or "matched" not in per_case_attr:
            continue
        cda_by_p, iad_by_p = defaultdict(list), defaultdict(list)
        for pid in sorted(set(per_case_attr[cond]) & set(per_case_attr["matched"])):
            mc, cc = per_case_attr["matched"][pid], per_case_attr[cond][pid]
            for i in range(min(len(mc), len(cc))):
                if attr is not None:
                    mv, cv = mc[i].get(attr), cc[i].get(attr)
                    if mv is not None and cv is not None:
                        cda_by_p[pid].append(float(mv and not cv))
                ch = [float(mc[i][a] != cc[i][a]) for a in ATTRS
                      if a != attr and mc[i][a] is not None and cc[i][a] is not None]
                if ch:
                    iad_by_p[pid].append(float(np.mean(ch)))
        cda = cluster_bootstrap(cda_by_p) if cda_by_p else (float("nan"),) * 3
        iad = cluster_bootstrap(iad_by_p)
        cda_rows.append({"counterfactual": cond, "target_attribute": attr or "（非曲目屬性）",
                         "CDA": cda[0], "CDA_ci_low": cda[1], "CDA_ci_high": cda[2],
                         "IAD": iad[0], "IAD_ci_low": iad[1], "IAD_ci_high": iad[2]})
        print(f"[CF] {cond:16s} CDA={cda[0]:.4f} IAD={iad[0]:.4f}")

    # ---- 輸出 ---------------------------------------------------------------
    def dump(name, data_rows):
        if not data_rows:
            return
        with open(OUT_DIR / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data_rows[0].keys()))
            w.writeheader()
            w.writerows(data_rows)

    dump("persona_metrics_v2_by_condition.csv", by_cond)
    dump("persona_metrics_v2_gaps.csv", gaps)
    dump("persona_metrics_v2_counterfactual.csv", cda_rows)
    dump("persona_metrics_v2_explanation.csv", expl_rows)
    dump("persona_metrics_v2_explanation_gaps.csv", expl_gaps)

    summary = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "by_condition": by_cond, "gaps": gaps, "counterfactual": cda_rows,
        "explanation": expl_rows, "explanation_gaps": expl_gaps,
        "notes": [
            "GT 的 R@1 僅作操作檢查：GT 為影片而非為 Persona 所選，"
            "Persona 向量本就不應幫助找回 GT。",
            "nDCG 的 IDCG 取自同一組 top-K 的理想排序（無法得知整池理想排序），"
            "故為檢索清單內的排序品質。",
            "⑦⑧ 為規則式詞彙比對，僅反映詞彙層級一致性；"
            "No-LTP 條件提供『無偏好輸入時仍會說出的屬性詞』機率基線。",
            "情緒價向不可操作化，未納入任何指標。",
            "統計以 Persona 為叢集重抽單位（同一 Persona 的 20 支查詢彼此相關）。",
        ],
    }
    (OUT_DIR / "persona_metrics_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# B5 Persona 偏好可控性完整指標（v2）\n",
         f"- 產生時間：{summary['generated_at']}",
         "- 統計：Persona 層級叢集拔靴 2000 次、95% 百分位 CI\n",
         "## 一、各條件的屬性合規度與排序品質\n",
         "| 條件 | n | ACR | Persona-fit@1 | @5 | @10 | nDCG@5 | nDCG@10 | GT R@1（操作檢查）|",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in by_cond:
        L.append(f"| {r['condition']} | {r['n']} | {r['ACR']*100:.2f}% | "
                 f"{r['persona_fit@1']*100:.2f}% | {r['persona_fit@5']*100:.2f}% | "
                 f"{r['persona_fit@10']*100:.2f}% | {r['nDCG@5']:.4f} | "
                 f"{r['nDCG@10']:.4f} | {r['gt_R@1']*100:.2f}% |")

    L.append("\n## 二、配對敏感度\n")
    L.append("| 比較 | ΔACR | 95% CI | 顯著 |")
    L.append("|---|---|---|---|")
    for g in gaps:
        L.append(f"| {g['comparison']} | {g['difference']*100:+.2f}pp | "
                 f"[{g['ci_low']*100:+.2f}, {g['ci_high']*100:+.2f}] | "
                 f"{'是' if g['significant'] else '否'} |")

    L.append("\n## 三、反事實方向性與無關屬性漂移\n")
    L.append("| 反事實 | 目標屬性 | CDA | IAD |")
    L.append("|---|---|---|---|")
    for c in cda_rows:
        L.append(f"| {c['counterfactual']} | {c['target_attribute']} | "
                 f"{c['CDA']*100:.2f}% | {c['IAD']*100:.2f}% |")

    if expl_rows:
        L.append("\n## 四、說明文字的屬性主張（指標⑦⑧）\n")
        L.append("| 條件 | 屬性說明支持率⑦ | 95% CI | UPCR⑧ | 95% CI | 有屬性主張的比例 |")
        L.append("|---|---|---|---|---|---|")
        for e in expl_rows:
            L.append(f"| {e['condition']} | {e['attribute_claim_support_rate']*100:.2f}% | "
                     f"[{e['support_ci_low']*100:.2f}, {e['support_ci_high']*100:.2f}] | "
                     f"{e['UPCR']*100:.2f}% | "
                     f"[{e['UPCR_ci_low']*100:.2f}, {e['UPCR_ci_high']*100:.2f}] | "
                     f"{e['share_with_any_attribute_claim']*100:.2f}% |")
        L.append("\n> **No-LTP 是機率基線**：模型在完全沒有偏好輸入時仍會說出的屬性詞比例。"
                 "matched 若未顯著高於 no_ltp，則不可主張說明具備偏好接地。\n")

    L.append("\n## 五、限制\n")
    for n in summary["notes"]:
        L.append(f"- {n}")
    (OUT_DIR / "persona_metrics_v2_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
