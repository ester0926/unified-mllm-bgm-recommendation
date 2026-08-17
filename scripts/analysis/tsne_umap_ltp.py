"""
t-SNE / UMAP visualizations of LTP (Latent Taste Profile) vectors.

Generates three figures:
  Fig A — hybrid vs explicit_only vs implicit_only LTP space (t-SNE, 2D)
  Fig B — within hybrid LTP: K-Means cluster structure (UMAP, 2D)
  Fig C — eval-set query embeddings: correct vs incorrect retrieval (t-SNE, 2D)

All figures saved to docs/figures/ltp_viz/
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

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2]
CACHE   = BASE / "cache"
CKPT    = BASE / "checkpoints"
FIG_DIR = BASE / "docs" / "figures" / "ltp_viz"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
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

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading LTP vectors...")
ltp_hybrid   = np.load(CACHE / "ltp_hybrid.npy")
ltp_explicit = np.load(CACHE / "ltp_explicit_only.npy")
ltp_implicit = np.load(CACHE / "ltp_implicit_only.npy")
print(f"  hybrid: {ltp_hybrid.shape}, explicit: {ltp_explicit.shape}, implicit: {ltp_implicit.shape}")

# L2-normalise (all models share the 256-dim LTP space)
ltp_hybrid_n   = normalize(ltp_hybrid,   norm="l2")
ltp_explicit_n = normalize(ltp_explicit, norm="l2")
ltp_implicit_n = normalize(ltp_implicit, norm="l2")

rng = np.random.RandomState(RANDOM_STATE)

# ── shared subsample indices ───────────────────────────────────────────────────
total = ltp_hybrid.shape[0]
idx   = rng.choice(total, size=min(N_SAMPLE, total), replace=False)

sub_h = ltp_hybrid_n[idx]
sub_e = ltp_explicit_n[idx]
sub_i = ltp_implicit_n[idx]

# ─────────────────────────────────────────────────────────────────────────────
# Fig A — three-model comparison via t-SNE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig A] t-SNE: hybrid vs explicit vs implicit ...")
# Stack all three and embed jointly so axes are comparable
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
# Fig B — K-Means cluster structure in hybrid LTP space (UMAP)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Fig B] UMAP + K-Means ({K_CLUSTERS} clusters) on hybrid LTP ...")

# K-Means on the full hybrid set (not just subsample) for stable centroids
print("  Fitting K-Means on full hybrid set ...")
km = KMeans(n_clusters=K_CLUSTERS, n_init=20, random_state=RANDOM_STATE)
km.fit(ltp_hybrid_n)
cluster_labels_full = km.labels_

# UMAP on subsample, coloured by cluster
cluster_labels_sub = cluster_labels_full[idx]

print("  Fitting UMAP on subsample ...")
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    metric="cosine", random_state=RANDOM_STATE)
emb_B   = reducer.fit_transform(sub_h)

# Cluster centroid names (descriptive, based on k-means centre analysis)
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

# Cluster size breakdown
print("\n  Cluster sizes:")
unique, counts = np.unique(cluster_labels_full, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    Cluster {u+1}: {c} ({c/len(cluster_labels_full)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Fig C — Eval-set LTP vectors: correct vs incorrect Top-1 retrieval (t-SNE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig C] t-SNE: eval query LTP — correct vs incorrect retrieval ...")

import csv

# Load eval-set sample indices (val_subset_indices_500.json → indices into ltp cache)
val_idx_path = CACHE / "val_subset_indices_500.json"
with open(val_idx_path) as f:
    val_indices = json.load(f)   # list of ints, length ~500

# Load per-sample R@1 from exp_01 ranking CSV
ranking_csv = CKPT / "exp_01" / "detailed_eval" / "exp_01_best_500pool_ranking_samples.csv"
top1_hits = []
with open(ranking_csv, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        top1_hits.append(int(row["R@1"]))  # 1=correct, 0=incorrect
top1_hits = np.array(top1_hits)  # shape (4205,)

# Use val_indices to select LTP vectors & correctness labels
# val_indices are indices into the full 84150-length LTP cache
eval_ltp = ltp_hybrid_n[val_indices]  # (500, 256)

# Match correctness: we need same ordering as val_indices
# eval CSVs have 4205 rows; val_subset is a 500-sample pool subset
# map by position: val_subset_indices_500 gives sample positions in eval set
# These indices might index directly into the eval CSV rows
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
