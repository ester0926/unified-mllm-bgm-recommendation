# -*- coding: utf-8 -*-
"""
用途：計算統計檢定與效果量。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT
EV = BASE / "results" / "main_eval"

LABEL = {
    "exp_02": "僅顯性文字偏好（wo_implicit）",
    "exp_03": "僅隱性音訊偏好（wo_explicit）",
    "exp_04": "移除長期偏好（wo_LTP）",
    "exp_05": "移除影片視覺（wo_video）",
    "exp_06": "移除對話文字（wo_text）",
    "exp_07": "移除候選音訊（wo_music）",
}
THESIS_DELTA = {"exp_02": "+0.1717", "exp_03": "−0.0055", "exp_04": "+0.2412",
                "exp_05": "+0.0579", "exp_06": "—", "exp_07": "—"}


def load(exp):
    p = EV / exp / "detailed_eval" / f"{exp}_best_500pool_ranking_samples.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    return {r["gt_music_id"]: int(r["rank"]) for r in rows}


def cliffs_delta(x, y):
    dom = np.sum(x[:, None] < y[None, :]) - np.sum(x[:, None] > y[None, :])
    return float(dom) / (len(x) * len(y))


def rank_biserial(d):
    """配對秩二系列相關 r = (W+ − W−)/(W+ + W−)；d>0 表示 exp_01 較佳。"""
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0, 0, 0
    ranks = np.empty(len(nz), float)
    order = np.argsort(np.abs(nz), kind="mergesort")
    a = np.abs(nz)[order]
    i = 0
    tmp = np.empty(len(nz), float)
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[j + 1] == a[i]:
            j += 1
        tmp[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks[order] = tmp
    wp = ranks[nz > 0].sum()
    wm = ranks[nz < 0].sum()
    return float((wp - wm) / (wp + wm)), wp, wm


base = load("exp_01")
print(f"exp_01 樣本數：{len(base)}\n")
print(f"{'對照組':<26}{'n':>6}{'勝':>7}{'負':>7}{'平':>7}"
      f"{'勝率(排平)':>11}{'勝率(平=½)':>11}{'配對 r':>9}{'Cliff δ':>10}{'論文δ':>10}{'Wilcoxon p':>12}")
print("-" * 118)
rows_out = []
for e in ["exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07"]:
    o = load(e)
    keys = sorted(set(base) & set(o))
    x = np.array([base[k] for k in keys], float)
    y = np.array([o[k] for k in keys], float)
    d = y - x                       # >0 → exp_01 排名較低（較佳）
    w = int((d > 0).sum()); l = int((d < 0).sum()); t = int((d == 0).sum())
    wr_ex = w / (w + l) if (w + l) else float("nan")
    wr_h = (w + 0.5 * t) / len(d)
    r, wp, wm = rank_biserial(d)
    dl = cliffs_delta(x, y)
    try:
        p = wilcoxon(x, y, zero_method="wilcox").pvalue
    except Exception:
        p = float("nan")
    print(f"{LABEL[e]:<26}{len(keys):>6}{w:>7}{l:>7}{t:>7}"
          f"{wr_ex * 100:>10.2f}%{wr_h * 100:>10.2f}%{r:>+9.4f}{dl:>+10.4f}"
          f"{THESIS_DELTA[e]:>10}{p:>12.3g}")
    rows_out.append((e, LABEL[e], len(keys), w, l, t, wr_ex, wr_h, r, dl, p))

print("\n" + "=" * 118)
print("量級判讀（Cohen r 慣例：|r|<.1 忽略、.1–.3 小、.3–.5 中、>.5 大；"
      "Romano δ：<.147 忽略、.147–.330 小、.330–.474 中、>.474 大）")
print("=" * 118)


def mag_r(v):
    a = abs(v)
    return "可忽略" if a < .1 else ("小" if a < .3 else ("中" if a < .5 else "大"))


def mag_d(v):
    a = abs(v)
    return "可忽略" if a < .147 else ("小" if a < .330 else ("中" if a < .474 else "大"))


print(f"{'對照組':<26}{'配對 r':>9} {'判讀':<8}{'Cliff δ':>10} {'判讀':<8}  變化")
for e, lab, n, w, l, t, wre, wrh, r, dl, p in rows_out:
    chg = "一致" if mag_r(r) == mag_d(dl) else f"**{mag_d(dl)} → {mag_r(r)}**"
    print(f"{lab:<26}{r:>+9.4f} {mag_r(r):<8}{dl:>+10.4f} {mag_d(dl):<8}  {chg}")
