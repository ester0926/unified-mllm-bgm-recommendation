"""
compute_median_rank_unseen_seen.py
==================================
重新計算「候選池記憶效應敏感度」分析中
Unseen / Seen 兩組的 Median Rank（原始分析只有 Mean Rank）。

結果用於更新論文表 4-33 的 Median Rank 欄位。

執行方式：
    python scripts/diagnostics/compute_median_rank_unseen_seen.py
"""

from __future__ import annotations
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAIR_INDEX_CACHE = PROJECT_ROOT / "cache" / "pair_index.json"
EVAL_CSV = (
    PROJECT_ROOT
    / "results" / "main_eval" / "exp_01"
    / "detailed_eval" / "exp_01_best_500pool_ranking_samples.csv"
)


# ── 1. 載入 pair_index，複製 dataset.py 的 split 邏輯 ─────────────────────────

def split_by_video_id(pair_index, train_ratio=0.90, val_ratio=0.05, seed=42):
    vid_to_pairs = defaultdict(list)
    for item in pair_index:
        vid_to_pairs[item[1][:11]].append(item)
    vids = sorted(vid_to_pairs.keys())
    random.Random(seed).shuffle(vids)
    n = len(vids)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    tr_p = [p for v in vids[:n_tr]           for p in vid_to_pairs[v]]
    va_p = [p for v in vids[n_tr:n_tr+n_va]  for p in vid_to_pairs[v]]
    te_p = [p for v in vids[n_tr+n_va:]      for p in vid_to_pairs[v]]
    return tr_p, va_p, te_p


print("載入 pair_index（可能需要數秒）…")
with open(PAIR_INDEX_CACHE, encoding="utf-8") as f:
    pair_index = json.load(f)

train_pairs, _, _ = split_by_video_id(pair_index)
train_cand_ids = {pair[1][12:] for pair in train_pairs}
print(f"  訓練集 unique candidate IDs: {len(train_cand_ids)}")


# ── 2. 載入 ranking CSV ────────────────────────────────────────────────────────

print(f"載入 ranking CSV…")
with open(EVAL_CSV, encoding="utf-8-sig") as f:
    eval_rows = list(csv.DictReader(f))
print(f"  測試樣本數: {len(eval_rows)}")


# ── 3. 分組並計算指標 ─────────────────────────────────────────────────────────

group_a = []   # Unseen（candidate 未出現在訓練集）
group_b = []   # Seen  （candidate 曾出現在訓練集）

for row in eval_rows:
    cand_id = row["gt_music_id"][12:]
    if cand_id in train_cand_ids:
        group_b.append(row)
    else:
        group_a.append(row)


def metrics(rows, label):
    ranks = [int(r["rank"]) for r in rows]
    r1  = [int(r["R@1"])  for r in rows]
    r5  = [int(r["R@5"])  for r in rows]
    r10 = [int(r["R@10"]) for r in rows]
    print(f"\n【{label}】  n = {len(rows)}")
    print(f"  R@1         = {sum(r1)  / len(r1)  * 100:.2f}%")
    print(f"  R@5         = {sum(r5)  / len(r5)  * 100:.2f}%")
    print(f"  R@10        = {sum(r10) / len(r10) * 100:.2f}%")
    print(f"  Mean Rank   = {sum(ranks) / len(ranks):.1f}")
    print(f"  Median Rank = {statistics.median(ranks):.1f}  ← 更新論文用")
    return statistics.median(ranks)


med_a = metrics(group_a, "Unseen（訓練集未出現之候選）")
med_b = metrics(group_b, "Seen  （訓練集曾出現之候選）")

gap = med_a - med_b
print(f"\n差距（Unseen − Seen）Median Rank = {gap:+.1f}")
print("  （負值 = Unseen 排名更靠前 = 更好）")

print("\n" + "=" * 55)
print("論文表 4-33 更新後的 Median Rank 欄位：")
print(f"  Unseen  Median Rank = {med_a}")
print(f"  Seen    Median Rank = {med_b}")
print(f"  差距                = {gap:+.1f}")
