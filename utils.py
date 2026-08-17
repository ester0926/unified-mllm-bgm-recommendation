"""
utils.py — 工具函數與輔助類別

包含：
  - 隨機種子設定
  - AverageMeter（損失追蹤）
  - 參數計數
  - Checkpoint 儲存/載入
  - Logger 設定
  - EarlyStopping
"""

import os
import json
import random
import logging
import numpy as np
from typing import Optional, Dict, Any

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Logger 設定
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """建立格式化 Logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # propagate=False：防止訊息向上傳至 root logger 重複輸出
        # （root logger 由 run_train.py 的 basicConfig 獨立管理）
        logger.propagate = False
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# 隨機種子
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """確保實驗可復現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 犧牲一點速度換取確定性（nondeterministic ops 關閉）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 損失追蹤
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """移動平均計算器（用於 loss、metric 的 running average）"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 參數統計
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """計算模型參數量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module):
    """列印模型各模組的參數量"""
    total = 0
    trainable = 0
    print(f"\n{'='*60}")
    print(f"{'Module':<40} {'Params':>10} {'Trainable':>10}")
    print(f"{'='*60}")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        train_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:<40} {params:>10,} {train_params:>10,}")
        total += params
        trainable += train_params
    print(f"{'='*60}")
    print(f"{'Total':<40} {total:>10,} {trainable:>10,}")
    print(f"Trainable ratio: {100 * trainable / total:.2f}%\n")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 管理
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    save_dir: str,
    global_step: int,
    metrics: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
):
    """
    儲存訓練 checkpoint。
    只儲存可訓練參數（projectors + LoRA），不儲存凍結的 LLaMA 主體，
    節省磁碟空間（約 100MB vs. 14GB）。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 儲存模型（透過 UnifiedMLLM 的 save_model 方法）
    model.save_model(save_dir)

    # 儲存訓練狀態
    train_state = {
        "global_step": global_step,
        "metrics": metrics,
    }
    if optimizer is not None:
        train_state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        train_state["scheduler_state"] = scheduler.state_dict()

    torch.save(train_state, os.path.join(save_dir, "train_state.pt"))

    # 儲存 metrics 為 JSON（方便查看）
    metrics_json = {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in metrics.items()}
    metrics_json["global_step"] = global_step
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)


def load_checkpoint(
    save_dir: str,
    model,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    """從 checkpoint 恢復訓練狀態"""
    state_path = os.path.join(save_dir, "train_state.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"找不到訓練狀態檔案: {state_path}")

    state = torch.load(state_path, map_location="cpu")

    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    基於驗證指標的 Early Stopping。
    預設監控 recall@10（越大越好）。
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            (score > self.best_score + self.min_delta) if self.mode == "max"
            else (score < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False
    
# %%
