"""
用途：整理實驗輸出並產生論文分析用表格或圖表。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]
CACHE   = BASE / "cache"
CKPT    = BASE / "checkpoints"
FIG_DIR = BASE / "docs" / "figures" / "ltp_viz"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 圖表樣式 ─────────────────────────────────────────────────────────────────
COLORS = {
    "hybrid":        "#2563EB",   # blue
    "explicit_only": "#DC2626",   # red
    "implicit_only": "#16A34A",   # green
}
CLUSTER_PALETTE = [
    "#2563EB", "#DC2626", "#16A34A",
    "#D97706", "#7C3AED", "#DB2777",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

N_SAMPLE = 5000   # points to draw per model in Fig A/B (random subsample)
K_CLUSTERS = 6    # preference prototype clusters for Fig B
TSNE_PERP = 40
TSNE_ITER = 1000
RANDOM_STATE = 42

# ── 讀取資料 ─────────────────────────────────────────────────────────────────
print("Loading LTP vectors...")
ltp_hybrid   = np.load(CACHE / "ltp_hybrid.npy")
ltp_explicit = np.load(CACHE / "ltp_explicit_only.npy")
ltp_implicit = np.load(CACHE / "ltp_implicit_only.npy")
print(f"  hybrid: {ltp_hybrid.shape}, explicit: {ltp_explicit.shape}, implicit: {ltp_implicit.shape}")

# L2 正規化；所有模型共用 256 維 LTP 空間
ltp_hybrid_n   = normalize(ltp_hybrid,   norm="l2")
ltp_explicit_n = normalize(ltp_explicit, norm="l2")
ltp_implicit_n = normalize(ltp_implicit, norm="l2")

rng = np.random.RandomState(RANDOM_STATE)

# ── 共用抽樣索引 ─────────────────────────────────────────────────────────────
total = ltp_hybrid.shape[0]
idx   = rng.choice(total, size=min(N_SAMPLE, total), replace=False)

sub_h = ltp_hybrid_n[idx]
sub_e = ltp_explicit_n[idx]
sub_i = ltp_implicit_n[idx]

# ─────────────────────────────────────────────────────────────────────────────
# 圖 A：以 t-SNE 比較三種模型
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig A] t-SNE: hybrid vs explicit vs implicit ...")
# 將三種向量一起降維，讓座標軸可比較
X_all   = np.vstack([sub_h, sub_e, sub_i])
labels  = (["hybrid"] * len(sub_h) +
           ["explicit_only"] * len(sub_e) +
           ["implicit_only"] * len(sub_i))

tsne_A = TSNE(n_components=2, perplexity=TSNE_PERP, max_iter=TSNE_ITER,
              random_state=RANDOM_STATE, n_jobs=-1)
emb_A  = tsne_A.fit_transform(X_all)

fig, ax = plt.subplots(figsize=(7, 6))
for model, label, zorder in [
    ("implicit_only", "Implicit LTP (exp_03)", 2),
    ("explicit_only", "Explicit LTP (exp_02)", 3),
    ("hybrid",        "Hybrid LTP (exp_01)",   4),
]:
    mask = np.array(labels) == model
    ax.scatter(emb_A[mask, 0], emb_A[mask, 1],
               c=COLORS[model], label=label,
               s=4, alpha=0.35, linewidths=0, rasterized=True, zorder=zorder)

ax.set_title("LTP Embedding Space — Model Comparison (t-SNE)", fontsize=12, pad=10)
ax.set_xlabel("t-SNE Dim 1")
ax.set_ylabel("t-SNE Dim 2")
ax.legend(markerscale=3, frameon=False, loc="upper right")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
out_A = FIG_DIR / "figA_tsne_model_comparison.pdf"
plt.savefig(out_A, bbox_inches="tight")
plt.savefig(str(out_A).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved → {out_A}")

# ─────────────────────────────────────────────────────────────────────────────
# 圖 B：hybrid LTP 空間中的 K-Means 群集結構（UMAP）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Fig B] UMAP + K-Means ({K_CLUSTERS} clusters) on hybrid LTP ...")

# K-Means 使用完整 hybrid set，讓 centroid 較穩定
print("  Fitting K-Means on full hybrid set ...")
km = KMeans(n_clusters=K_CLUSTERS, n_init=20, random_state=RANDOM_STATE)
km.fit(ltp_hybrid_n)
cluster_labels_full = km.labels_

# UMAP 使用抽樣資料，並依群集上色
cluster_labels_sub = cluster_labels_full[idx]

print("  Fitting UMAP on subsample ...")
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    metric="cosine", random_state=RANDOM_STATE)
emb_B   = reducer.fit_transform(sub_h)

# 群集名稱依 K-Means 中心特徵做描述
cluster_names = [f"Cluster {i+1}" for i in range(K_CLUSTERS)]

fig, ax = plt.subplots(figsize=(7, 6))
for k in range(K_CLUSTERS):
    mask = cluster_labels_sub == k
    ax.scatter(emb_B[mask, 0], emb_B[mask, 1],
               c=CLUSTER_PALETTE[k], label=cluster_names[k],
               s=5, alpha=0.40, linewidths=0, rasterized=True)

ax.set_title(f"Hybrid LTP Preference Clusters (K={K_CLUSTERS}, UMAP)", fontsize=12, pad=10)
ax.set_xlabel("UMAP Dim 1")
ax.set_ylabel("UMAP Dim 2")
ax.legend(markerscale=2.5, frameon=False, loc="upper right",
          fontsize=8, ncol=2)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
out_B = FIG_DIR / "figB_umap_clusters.pdf"
plt.savefig(out_B, bbox_inches="tight")
plt.savefig(str(out_B).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved → {out_B}")

# 各群集樣本數摘要
print("\n  Cluster sizes:")
unique, counts = np.unique(cluster_labels_full, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    Cluster {u+1}: {c} ({c/len(cluster_labels_full)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 圖 C：評估集 LTP 向量中 Top-1 正確與錯誤樣本的 t-SNE 分布
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig C] t-SNE: eval query LTP — correct vs incorrect retrieval ...")

import csv

# 讀取評估集樣本索引；val_subset_indices_500.json 對應到 LTP cache 索引
val_idx_path = CACHE / "val_subset_indices_500.json"
with open(val_idx_path) as f:
    val_indices = json.load(f)   # list of ints, length ~500

# 從 exp_01 ranking CSV 讀取逐筆 R@1
ranking_csv = CKPT / "exp_01" / "detailed_eval" / "exp_01_best_500pool_ranking_samples.csv"
top1_hits = []
with open(ranking_csv, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        top1_hits.append(int(row["R@1"]))  # 1=correct, 0=incorrect
top1_hits = np.array(top1_hits)  # shape (4205,)

# 使用 val_indices 選取 LTP 向量與正確性標籤
# val_indices 對應完整 84150 筆 LTP cache 的索引
eval_ltp = ltp_hybrid_n[val_indices]  # (500, 256)

# 對齊正確性標籤，順序需與 val_indices 相同
# eval CSV 有 4205 列；val_subset 是 500-sample pool 子集
# 依位置對應：val_subset_indices_500 提供 eval set 中的樣本位置
# 這些索引可直接對應 eval CSV 的列位置
n_eval_csv = len(top1_hits)  # 4205
valid_mask = np.array(val_indices) < n_eval_csv  # safety filter
eval_ltp_valid    = eval_ltp[valid_mask]
eval_correct_mask = top1_hits[np.array(val_indices)[valid_mask]]  # 0 or 1

n_correct   = eval_correct_mask.sum()
n_incorrect = (1 - eval_correct_mask).sum()
print(f"  Eval subset: {valid_mask.sum()} samples ({n_correct} correct, {n_incorrect} incorrect Top-1)")

if valid_mask.sum() > 10:
    tsne_C = TSNE(n_components=2, perplexity=min(30, valid_mask.sum()//4),
                  max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    emb_C  = tsne_C.fit_transform(eval_ltp_valid)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors_C = np.where(eval_correct_mask == 1, "#2563EB", "#DC2626")
    for label, val, color, marker in [
        ("Correct Top-1",   1, "#2563EB", "o"),
        ("Incorrect Top-1", 0, "#DC2626", "x"),
    ]:
        mask = eval_correct_mask == val
        ax.scatter(emb_C[mask, 0], emb_C[mask, 1],
                   c=color, marker=marker, label=f"{label} (n={mask.sum()})",
                   s=20, alpha=0.65, linewidths=0.6, rasterized=True)

    ax.set_title("Eval-Set LTP Vectors — Retrieval Correctness (t-SNE)", fontsize=12, pad=10)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(markerscale=1.5, frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out_C = FIG_DIR / "figC_tsne_correctness.pdf"
    plt.savefig(out_C, bbox_inches="tight")
    plt.savefig(str(out_C).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out_C}")
else:
    print("  Not enough valid eval samples — skipping Fig C.")

print("All figures saved to:", str(FIG_DIR))
