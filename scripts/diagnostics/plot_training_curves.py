# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
plot_training_curves.py — 訓練曲線視覺化（修正版）

修正清單：
  Fix A：train.py 的 logger 修正後，step/epoch/val log 才會進 train.log，
          本腳本的 parse_train_log() 才能正確解析。
          訓練前請確認 train.py 已套用 Fix A。

  Fix B：新增 val loss 解析正則（對應 train.py Fix B 的 log 格式）
          "[Val Epoch N] ... | val_loss=X.XXXX"

  Fix C：新增最重要的過擬合判斷圖：
          train loss vs val loss（同一張圖，雙曲線）

  Fix D：從 epoch checkpoint 的 metrics.json 讀取數據（更可靠的資料來源，
          不依賴 log 解析）。metrics.json 由 save_checkpoint() 自動儲存。

  Fix E：LOG_VAL_RE 格式修正：同時支援 val_loss 欄位有無的舊版 log

使用方式：
  python plot_training_curves.py
  → 輸出圖片至 exp_01/analysis_plots/
  → 輸出解析摘要至 exp_01/analysis_plots/summary.json

過擬合判斷準則：
  - 若 train_loss 持續下降，但 val_loss 先降後升（或平台後升）→ 過擬合
  - 若 train_loss 和 val_loss 均持續下降且趨勢一致 → 欠擬合或正常收斂
  - 若 val R@1 持續提升，即使 val_loss 略有波動，仍以排序指標為主要判斷依據
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # 無 GUI 環境（遠端）也能用
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
EXP_DIR = Path(str(PROJECT_ROOT / "checkpoints" / "exp_01"))

# 是否同時繪製多個 exp 的對比圖（消融實驗用）
COMPARE_EXPS = {
    "exp_01 (hybrid)":        EXP_DIR,
    # "exp_02 (explicit_only)": EXP_DIR.parent / "exp_02",
    # "exp_03 (implicit_only)": EXP_DIR.parent / "exp_03",
}


# ── 正則表達式 ────────────────────────────────────────────────────────────────

# Step log："Epoch 3 Step 150 | loss=1.2345 | rank=0.8765 | gen=0.4321"
LOG_STEP_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+Step\s+(?P<step>\d+)\s*\|\s*"
    r"loss=(?P<loss>[0-9.nan]+)\s*\|\s*"
    r"rank=(?P<rank>[0-9.nan]+)\s*\|\s*"
    r"gen=(?P<gen>[0-9.nan]+)"
)

# Val log（Fix B：支援有無 val_loss 欄位）
# 格式 1（舊，無 val_loss）："[Val Epoch 3] R@1=0.7260 | R@5=0.9720 | R@10=0.9980 | MR=1.0"
# 格式 2（新，Fix B）  ："[Val Epoch 3] R@1=0.7260 | R@5=0.9720 | R@10=0.9980 | MR=1.0 | val_loss=1.2345"
LOG_VAL_RE = re.compile(
    r"\[Val(?:\s+Epoch\s+(?P<epoch>\d+))?\]\s+"
    r"R@1=(?P<r1>[0-9.]+)\s*\|\s*"
    r"R@5=(?P<r5>[0-9.]+)\s*\|\s*"
    r"R@10=(?P<r10>[0-9.]+)\s*\|\s*"
    r"MR=(?P<mr>[0-9.]+)"
    r"(?:\s*\|\s*val_loss=(?P<val_loss>[0-9.nan]+))?"  # 可選
)

# Epoch 完成 log（Fix B：支援有無 val_loss）
# 格式 1（舊）："Epoch 3 完成 | train_loss=1.2345"
# 格式 2（新）："Epoch 3 完成 | train_loss=1.2345 | val_loss=0.9876"
LOG_EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+完成\s*\|\s*"
    r"train_loss=(?P<train_loss>[0-9.]+)"
    r"(?:\s*\|\s*val_loss=(?P<val_loss>[0-9.nan]+))?"
)


# ── 解析函式 ──────────────────────────────────────────────────────────────────

def parse_train_log(log_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """解析 train.log，回傳 steps / vals / epochs 三個 list。"""
    steps, vals, epochs = [], [], []
    if not log_path.exists():
        print(f"  ⚠️  log 不存在：{log_path}")
        return {"steps": steps, "vals": vals, "epochs": epochs}

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LOG_STEP_RE.search(line)
        if m:
            steps.append({
                "epoch": int(m.group("epoch")),
                "step":  int(m.group("step")),
                "loss":  float(m.group("loss")),
                "rank":  float(m.group("rank")),
                "gen":   float(m.group("gen")),
            })
            continue

        m = LOG_VAL_RE.search(line)
        if m:
            vals.append({
                "epoch":    int(m.group("epoch")) if m.group("epoch") else None,
                "r1":       float(m.group("r1")),
                "r5":       float(m.group("r5")),
                "r10":      float(m.group("r10")),
                "mr":       float(m.group("mr")),
                "val_loss": float(m.group("val_loss")) if m.group("val_loss") else None,
            })
            continue

        m = LOG_EPOCH_RE.search(line)
        if m:
            epochs.append({
                "epoch":      int(m.group("epoch")),
                "train_loss": float(m.group("train_loss")),
                "val_loss":   float(m.group("val_loss")) if m.group("val_loss") else None,
            })

    return {"steps": steps, "vals": vals, "epochs": epochs}


def load_epoch_metrics(exp_dir: Path) -> List[Dict[str, Any]]:
    """
    Fix D：從 epoch checkpoint 的 metrics.json 讀取每個 epoch 的完整指標。
    這比 log 解析更可靠——只要 save_checkpoint() 有執行，數據就一定存在。
    """
    rows = []
    for ep in range(1, 20):  # 最多找 20 個 epoch
        p = exp_dir / f"epoch_{ep}" / "metrics.json"
        if not p.exists():
            break
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data.setdefault("epoch", ep)
            rows.append(data)
        except Exception:
            pass
    if rows:
        rows.sort(key=lambda r: r.get("epoch", 0))
    return rows


def load_best_metrics(exp_dir: Path) -> Optional[Dict]:
    p = exp_dir / "best" / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── 繪圖函式 ──────────────────────────────────────────────────────────────────

STYLE = {
    "figure.facecolor":  "#0D1B2A",
    "axes.facecolor":    "#1C3144",
    "axes.edgecolor":    "#94A3B8",
    "axes.labelcolor":   "#EFF6FF",
    "text.color":        "#EFF6FF",
    "xtick.color":       "#94A3B8",
    "ytick.color":       "#94A3B8",
    "grid.color":        "#2A3A4A",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "lines.linewidth":   2,
    "lines.markersize":  6,
}
plt.rcParams.update(STYLE)
COLORS = ["#00B4D8", "#2DC653", "#F4A261", "#E63946", "#FFD166", "#A8DADC"]


def _ax_setup(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def plot_train_val_loss(epoch_rows: List[Dict], out_path: Path):
    """
    Fix C：最重要的過擬合判斷圖——train loss vs val loss 雙曲線。
    需要 train.py Fix B 的資料（val_loss 欄位）。
    """
    if not epoch_rows:
        return

    epochs     = [r["epoch"]      for r in epoch_rows]
    train_loss = [r.get("train_loss") for r in epoch_rows]
    val_loss   = [r.get("val_loss")   for r in epoch_rows]

    has_val = any(v is not None and not isinstance(v, float) or
                  (isinstance(v, float) and v == v)   # not nan
                  for v in val_loss)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, train_loss, "o-", color=COLORS[0], label="Train Loss")
    if has_val:
        vl_clean = [v if v is not None else float("nan") for v in val_loss]
        ax.plot(epochs, vl_clean, "s--", color=COLORS[1], label="Val Loss")

    _ax_setup(ax, "Epoch", "Loss", "Train Loss vs Val Loss（過擬合判斷）")
    ax.legend(facecolor="#1C3144", edgecolor="#94A3B8")

    # 注釋
    if not has_val:
        ax.text(0.5, 0.5,
                "Val Loss 尚無資料\n請套用 train.py Fix B 後重新訓練",
                transform=ax.transAxes, ha="center", va="center",
                color="#F4A261", fontsize=11, alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_loss_components(steps: List[Dict], out_path: Path):
    """Step-level 三個 loss 分量（total / rank / gen）。"""
    if not steps:
        return
    xs   = [r["step"] for r in steps]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, label, col in zip(
        axes,
        ["loss", "rank", "gen"],
        ["Total Loss", "Rank Loss (BPR)", "Gen Loss (LM)"],
        COLORS[:3]
    ):
        ax.plot(xs, [r[key] for r in steps], color=col, linewidth=1.2)
        _ax_setup(ax, "Global Step", label, label)
    fig.suptitle("訓練 Loss 分量（Step-level）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_val_metrics(epoch_rows: List[Dict], out_path: Path):
    """Validation 排序指標曲線（R@1 / R@5 / R@10 / MR）。"""
    if not epoch_rows:
        return
    epochs = [r["epoch"] for r in epoch_rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [
        (axes[0][0], "recall@1",    "R@1",           COLORS[0]),
        (axes[0][1], "recall@5",    "R@5",           COLORS[1]),
        (axes[1][0], "recall@10",   "R@10",          COLORS[2]),
        (axes[1][1], "median_rank", "Median Rank ↓", COLORS[3]),
    ]
    for ax, key, label, col in pairs:
        vals = [r.get(key) for r in epoch_rows]
        if any(v is not None for v in vals):
            ax.plot(epochs, [v if v is not None else float("nan") for v in vals],
                    "o-", color=col)
        _ax_setup(ax, "Epoch", label, f"Val {label}")
    fig.suptitle("Validation 排序指標（Val 50-pool）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_composite(epoch_rows: List[Dict], out_path: Path):
    """Composite metric 曲線（best checkpoint 選擇依據）。"""
    if not epoch_rows:
        return
    epochs    = [r["epoch"] for r in epoch_rows]
    composite = [r.get("composite") for r in epoch_rows]
    if not any(v is not None for v in composite):
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [v if v is not None else float("nan") for v in composite],
            "o-", color=COLORS[4])

    # 標記最佳
    best_ep, best_val = max(
        ((r["epoch"], r.get("composite", 0)) for r in epoch_rows if r.get("composite")),
        key=lambda x: x[1]
    )
    ax.axvline(x=best_ep, color=COLORS[2], linestyle="--", linewidth=1.5,
               label=f"Best Epoch {best_ep} ({best_val:.4f})")
    ax.legend(facecolor="#1C3144", edgecolor="#94A3B8")
    _ax_setup(ax, "Epoch", "Composite", "Composite Metric（best checkpoint 判斷依據）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_multi_exp_r1(exp_dict: Dict[str, Path], out_path: Path):
    """消融實驗對比：多個 exp 的 val R@1 曲線。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False
    for i, (label, exp_dir) in enumerate(exp_dict.items()):
        rows = load_epoch_metrics(exp_dir)
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        r1     = [r.get("recall@1") for r in rows]
        if any(v is not None for v in r1):
            ax.plot(epochs, [v if v is not None else float("nan") for v in r1],
                    "o-", color=COLORS[i % len(COLORS)], label=label)
            any_data = True

    if not any_data:
        plt.close(fig)
        return

    _ax_setup(ax, "Epoch", "Val R@1", "消融實驗比較：Val R@1（各 exp）")
    ax.legend(facecolor="#1C3144", edgecolor="#94A3B8")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path.name}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    exp_dir  = EXP_DIR
    log_path = exp_dir / "train.log"
    out_dir  = exp_dir / "analysis_plots"
    out_dir.mkdir(exist_ok=True)

    print(f"解析：{log_path}")
    parsed = parse_train_log(log_path)

    print(f"  step logs:  {len(parsed['steps'])} 筆")
    print(f"  val  logs:  {len(parsed['vals'])}  筆")
    print(f"  epoch logs: {len(parsed['epochs'])} 筆")

    if not parsed["steps"] and not parsed["vals"]:
        print("\n  ⚠️  train.log 幾乎空白。")
        print("     請確認 train.py 已套用 Fix A（logger.propagate = True）")
        print("     修正前的訓練結果可改從 metrics.json 讀取（見下方 Fix D）")

    # Fix D：從 metrics.json 讀取（不依賴 log）
    epoch_rows = load_epoch_metrics(exp_dir)
    print(f"\n  metrics.json 讀取：{len(epoch_rows)} 個 epoch")
    if not epoch_rows and not parsed["steps"]:
        print("  無可用訓練數據，請先完成至少 1 個 epoch 的訓練。")
        return

    # 若 log 有 epoch 資料，與 metrics.json 合併（log 優先提供 val_loss）
    log_epoch_map = {r["epoch"]: r for r in parsed["epochs"]}
    log_val_map   = {}
    for r in parsed["vals"]:
        ep = r.get("epoch")
        if ep is not None:
            log_val_map[ep] = r

    for row in epoch_rows:
        ep = row["epoch"]
        if ep in log_epoch_map and row.get("val_loss") is None:
            row["val_loss"] = log_epoch_map[ep].get("val_loss")
        if ep in log_val_map:
            row.setdefault("recall@1",    log_val_map[ep]["r1"])
            row.setdefault("recall@5",    log_val_map[ep]["r5"])
            row.setdefault("recall@10",   log_val_map[ep]["r10"])
            row.setdefault("median_rank", log_val_map[ep]["mr"])
            if row.get("val_loss") is None:
                row["val_loss"] = log_val_map[ep].get("val_loss")

    print("\n繪圖中...")

    # ① Train vs Val Loss（過擬合判斷，最重要）
    plot_train_val_loss(epoch_rows, out_dir / "01_train_vs_val_loss.png")

    # ② Step-level Loss 分量
    plot_loss_components(parsed["steps"], out_dir / "02_loss_components_step.png")

    # ③ Val 排序指標
    plot_val_metrics(epoch_rows, out_dir / "03_val_ranking_metrics.png")

    # ④ Composite Metric（best checkpoint 依據）
    plot_composite(epoch_rows, out_dir / "04_composite_metric.png")

    # ⑤ 消融對比（若有多個 exp）
    if len(COMPARE_EXPS) > 1:
        plot_multi_exp_r1(COMPARE_EXPS, out_dir / "05_ablation_r1_compare.png")

    # 儲存摘要
    best = load_best_metrics(exp_dir)
    summary = {
        "exp_dir":             str(exp_dir),
        "log_step_count":      len(parsed["steps"]),
        "log_val_count":       len(parsed["vals"]),
        "log_epoch_count":     len(parsed["epochs"]),
        "epoch_metrics_count": len(epoch_rows),
        "has_val_loss":        any(r.get("val_loss") is not None for r in epoch_rows),
        "best_checkpoint":     best,
        "epoch_summary":       epoch_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 終端機摘要
    print("\n── 各 Epoch 訓練摘要 ─────────────────────────────────────────")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val R@1':>8}  {'Val MR':>7}  {'Composite':>10}")
    for r in epoch_rows:
        vl  = f"{r['val_loss']:.4f}" if r.get("val_loss") is not None else "  N/A   "
        r1  = f"{r.get('recall@1',0):.4f}"  if r.get("recall@1")    is not None else "  N/A  "
        mr  = f"{r.get('median_rank',0):.1f}" if r.get("median_rank") is not None else " N/A "
        cmp = f"{r.get('composite',0):.4f}"   if r.get("composite")   is not None else "  N/A   "
        print(f"  {r['epoch']:>5}  {r.get('train_loss',0):>10.4f}  {vl:>9}  {r1:>8}  {mr:>7}  {cmp:>10}")

    if not any(r.get("val_loss") is not None for r in epoch_rows):
        print("\n  ⚠️  Val Loss 全部為 N/A：請在 train.py 套用 Fix B 後，")
        print("     對 exp_02/exp_03 重新訓練，屆時即可看到 train/val loss 對比曲線。")
        print("     exp_01 已訓練完成，其 val loss 無法補算，但可從 val R@1 趨勢判斷收斂情況。")

    print(f"\n完成，輸出目錄：{out_dir}")


if __name__ == "__main__":
    main()