"""
用途：計算統計檢定與效果量。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# ── 路徑 ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]   # unified_mllm_pointwise_final
LTP_DIR = BASE / "results" / "main_eval" / "exp_01" / "ltp_control"
OUT_DIR = LTP_DIR                                # 輸出到同一資料夾

# ── 成對比較定義 ──────────────────────────────────────────────────────────────
COMPARISONS = [
    {
        "id":    "A",
        "name":  "matched vs shuffled",
        "cond_x": "matched",
        "cond_y": "shuffled",
        "interpretation": "LTP 使用者身份的邊際效益（排除統計雜訊）",
    },
    {
        "id":    "B",
        "name":  "matched vs random",
        "cond_x": "matched",
        "cond_y": "random",
        "interpretation": "有正確 LTP vs 完全無 LTP",
    },
    {
        "id":    "C",
        "name":  "shuffled vs random",
        "cond_x": "shuffled",
        "cond_y": "random",
        "interpretation": "錯誤身份 LTP vs 無 LTP（診斷：是否存在分佈偏移）",
    },
]


# ── 工具函數 ─────────────────────────────────────────────────────────────────

def load_condition(condition: str) -> dict:
    """載入 ltp_{condition}_ranking.csv，回傳各欄 np.ndarray。"""
    csv_path = LTP_DIR / f"ltp_{condition}_ranking.csv"
    ranks, r1, r5, r10 = [], [], [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ranks.append(int(row["rank"]))
            r1.append(int(row["R@1"]))
            r5.append(int(row["R@5"]))
            r10.append(int(row["R@10"]))
    return {
        "ranks": np.array(ranks, dtype=float),
        "R@1":   np.array(r1,    dtype=float),
        "R@5":   np.array(r5,    dtype=float),
        "R@10":  np.array(r10,   dtype=float),
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta：正值代表 x 傾向排名低於 y（x 比 y 優）。
    x = 前條件 ranks，y = 後條件 ranks。
    """
    dom = np.sum(x[:, None] < y[None, :]) - np.sum(x[:, None] > y[None, :])
    return float(dom) / (len(x) * len(y))


def delta_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.330: return "small"
    if a < 0.474: return "medium"
    return "large"


def holm_bonferroni(p_values: list) -> list:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [None] * n
    prev = 0.0
    for rank_i, (orig_idx, p) in enumerate(indexed):
        adj = min(max(p * (n - rank_i), prev), 1.0)
        adjusted[orig_idx] = adj
        prev = adj
    return adjusted


def fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


# ── 主程式入口 ───────────────────────────────────────────────────────────────

def main():
    # 載入三個條件
    print("載入各條件資料...")
    data = {}
    for cond in ("matched", "shuffled", "random"):
        data[cond] = load_condition(cond)
        n = len(data[cond]["ranks"])
        r1 = data[cond]["R@1"].mean() * 100
        mr = float(np.median(data[cond]["ranks"]))
        print(f"  {cond:10s}  n={n}  R@1={r1:.2f}%  MedR={mr:.1f}")

    results = []
    raw_ps  = []

    for comp in COMPARISONS:
        cx, cy = comp["cond_x"], comp["cond_y"]
        n = min(len(data[cx]["ranks"]), len(data[cy]["ranks"]))
        rx = data[cx]["ranks"][:n]
        ry = data[cy]["ranks"][:n]

        delta = cliffs_delta(rx, ry)
        mag   = delta_magnitude(delta)

        diffs = rx - ry
        try:
            stat, p_raw = wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
        except ValueError:
            stat, p_raw = 0.0, 1.0

        raw_ps.append(p_raw)

        entry = {
            "comparison_id":    comp["id"],
            "comparison":       comp["name"],
            "interpretation":   comp["interpretation"],
            "n":                n,
            "cond_x":           cx,
            "cond_y":           cy,
            # recall 均值
            "R@1_x":            data[cx]["R@1"][:n].mean() * 100,
            "R@5_x":            data[cx]["R@5"][:n].mean() * 100,
            "R@10_x":           data[cx]["R@10"][:n].mean() * 100,
            "R@1_y":            data[cy]["R@1"][:n].mean() * 100,
            "R@5_y":            data[cy]["R@5"][:n].mean() * 100,
            "R@10_y":           data[cy]["R@10"][:n].mean() * 100,
            "delta_R@1":        (data[cx]["R@1"][:n].mean() - data[cy]["R@1"][:n].mean()) * 100,
            "delta_R@5":        (data[cx]["R@5"][:n].mean() - data[cy]["R@5"][:n].mean()) * 100,
            "delta_R@10":       (data[cx]["R@10"][:n].mean() - data[cy]["R@10"][:n].mean()) * 100,
            "MedR_x":           float(np.median(data[cx]["ranks"][:n])),
            "MedR_y":           float(np.median(data[cy]["ranks"][:n])),
            # 統計量
            "cliffs_delta":     delta,
            "magnitude":        mag,
            "W_stat":           stat,
            "p_raw":            p_raw,
        }
        results.append(entry)
        print(f"\n[{comp['id']}] {comp['name']}")
        print(f"     ΔR@1={entry['delta_R@1']:+.2f}pp  δ={delta:+.4f} ({mag})  W={stat:.0f}  p_raw={p_raw:.4g}")

    # Holm-Bonferroni 多重比較校正
    adj_ps = holm_bonferroni(raw_ps)
    for i, r in enumerate(results):
        r["p_adj"] = adj_ps[i]
        r["sig"]   = ("***" if r["p_adj"] < 0.001 else
                      ("**"  if r["p_adj"] < 0.01  else
                       ("*"   if r["p_adj"] < 0.05  else "ns")))

    # ── CSV ──────────────────────────────────────────────────────────────────
    fields = [
        "comparison_id", "comparison", "interpretation", "n",
        "cond_x", "R@1_x", "R@5_x", "R@10_x", "MedR_x",
        "cond_y", "R@1_y", "R@5_y", "R@10_y", "MedR_y",
        "delta_R@1", "delta_R@5", "delta_R@10",
        "cliffs_delta", "magnitude", "W_stat", "p_raw", "p_adj", "sig",
    ]
    csv_out = OUT_DIR / "ltp_control_significance.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    print(f"\n✓ CSV  → {csv_out}")

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_out = OUT_DIR / "ltp_control_significance.json"
    json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ JSON → {json_out}")

    # ── Markdown ─────────────────────────────────────────────────────────────
    lines = []
    lines.append("## LTP Perturbation Control — Wilcoxon + Cliff's Delta\n")
    lines.append("Holm-Bonferroni correction applied across 3 pairwise comparisons.\n")
    lines.append("Cliff's δ > 0 → 前條件 (cond_x) 排名較低（較優）。\n")

    # 各條件摘要表
    lines.append("### 各條件指標摘要\n")
    lines.append("| Condition | R@1 (%) | R@5 (%) | R@10 (%) | MedR |")
    lines.append("|---|---:|---:|---:|---:|")
    for cond in ("matched", "shuffled", "random"):
        d = data[cond]
        lines.append(
            f"| {cond} "
            f"| {d['R@1'].mean()*100:.2f} "
            f"| {d['R@5'].mean()*100:.2f} "
            f"| {d['R@10'].mean()*100:.2f} "
            f"| {np.median(d['ranks']):.1f} |"
        )

    lines.append("\n### 成對比較（Wilcoxon + Cliff's δ）\n")
    lines.append("| ID | Comparison | ΔR@1 (pp) | ΔR@5 (pp) | ΔR@10 (pp) | Cliff's δ | Magnitude | W | p_adj | Sig |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---|")
    for r in results:
        lines.append(
            f"| {r['comparison_id']} | {r['comparison']} "
            f"| {r['delta_R@1']:+.2f} | {r['delta_R@5']:+.2f} | {r['delta_R@10']:+.2f} "
            f"| {r['cliffs_delta']:+.4f} | {r['magnitude']} "
            f"| {r['W_stat']:.0f} | p {fmt_p(r['p_adj'])} | {r['sig']} |"
        )

    lines.append("\n### 詮釋\n")
    for r in results:
        lines.append(
            f"- **[{r['comparison_id']}] {r['comparison']}**：{r['interpretation']}  \n"
            f"  ΔR@1={r['delta_R@1']:+.2f}pp，δ={r['cliffs_delta']:+.4f} ({r['magnitude']})，"
            f"p_adj {fmt_p(r['p_adj'])} {r['sig']}"
        )

    lines.append("\n### 論文文字片段（§4.7.4 草稿）\n")
    for r in results:
        lines.append(
            f"> **[{r['comparison_id']}]** exp_01 在 {r['comparison']} 比較中："
            f"ΔR@1={r['delta_R@1']:+.2f}pp，Cliff's δ={r['cliffs_delta']:+.4f} ({r['magnitude']})，"
            f"W={r['W_stat']:.0f}，p_Holm {fmt_p(r['p_adj'])} ({r['sig']})。"
        )

    md_text = "\n".join(lines)
    md_out  = OUT_DIR / "ltp_control_significance.md"
    md_out.write_text(md_text, encoding="utf-8")
    print(f"✓ MD   → {md_out}")

    # ── stdout 摘要 ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(md_text)
    print("="*70)

    # ── 論文用文字摘要 ───────────────────────────────────────────────────────
    print("\n\n=== THESIS TEXT FRAGMENT (§4.7.4) ===\n")
    for r in results:
        print(
            f"[{r['comparison_id']}] {r['comparison']}: "
            f"ΔR@1={r['delta_R@1']:+.2f}pp, "
            f"δ={r['cliffs_delta']:+.4f} ({r['magnitude']}), "
            f"W={r['W_stat']:.0f}, p_Holm {fmt_p(r['p_adj'])} {r['sig']}"
        )


if __name__ == "__main__":
    main()
