"""
用途：執行主要訓練流程，包含資料載入、模型建立、訓練迴圈與 checkpoint 輸出。
輸入：data/、cache/ 與 checkpoints/ 中的特徵、LTP 向量和資料切分。
輸出：新的訓練 checkpoint、log 與必要的中間結果。
執行：建議在 repo 根目錄執行，並先確認 config.py 的資料路徑。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
import sys
import logging
import json
from typing import Dict, Optional

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, get_cosine_schedule_with_warmup

from config import ModelConfig, TrainConfig
from dataset import build_dataloaders
from models.unified_mllm import UnifiedMLLM
from utils import (
    set_seed, AverageMeter, count_parameters,
    get_logger, save_checkpoint, load_checkpoint
)
from evaluate import pointwise_pool_evaluate_loader

# ★ Fix A：改用標準 logging.getLogger，不使用 utils.get_logger
# utils.get_logger 設定了 propagate=False 且只有 StreamHandler，
# 導致所有 train.py 的 log 只輸出到螢幕，不寫進 train.log。
# 改為標準 getLogger：由 run_train.py 的 logging.basicConfig 統一管理 handler，
# propagate=True（預設）讓 log 正確流向 root logger 的 FileHandler。
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BPR Loss
# ─────────────────────────────────────────────────────────────────────────────

def bpr_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking Loss（Rendle et al., UAI 2009）

    Args:
        score_pos : (B,) — positive（GT）music 的 relevance score
        score_neg : (B,) — negative（random）music 的 relevance score

    Returns:
        scalar loss
    """
    return -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-8).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 訓練單步
# ─────────────────────────────────────────────────────────────────────────────

def train_one_step(model, batch, device, scaler, use_bf16, model_config, train_config):

    video_feat     = batch["video_feat"].to(device)
    ltp_feat       = batch["ltp_feat"].to(device)
    text_feat      = batch["text_feat"].to(device)
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)

    pos_music = batch["pos_music_feat"].to(device).unsqueeze(1)   # (B, 1, 768) GT
    neg_mc    = batch["neg_music_feat"].to(device).unsqueeze(1)   # (B, 1, 768) mc Hard Negative
    # Cross-video negative：位移取同 batch 其他樣本的 GT
    # torch.roll(x, 1, 0) 讓每個 query 拿到下一個樣本的 GT music
    # 你的資料集每個 video 只有一筆 pair，所以不會有同 video 污染
    neg_cross = torch.roll(batch["pos_music_feat"], 1, 0).to(device).unsqueeze(1)  # (B, 1, 768)

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    with autocast(dtype=amp_dtype):
        # 前向傳播 1：pos（算 ranking + gen loss）
        out_pos = model(
            video_feat=video_feat, music_candidates=pos_music,
            ltp_feat=ltp_feat, text_feat=text_feat,
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, compute_gen_loss=True,
        )
        score_pos = out_pos["ranking_score"]   # (B,)
        loss_gen  = out_pos["loss_gen"]

        # 前向傳播 2：mc Hard Negative（同影片，不算 gen loss）
        out_neg_mc = model(
            video_feat=video_feat, music_candidates=neg_mc,
            ltp_feat=ltp_feat, text_feat=text_feat,
            input_ids=input_ids, attention_mask=attention_mask,
            labels=None, compute_gen_loss=False,
        )
        score_neg_mc = out_neg_mc["ranking_score"]   # (B,)

        # 前向傳播 3：Cross-video Negative（跨影片，不算 gen loss）
        out_neg_cross = model(
            video_feat=video_feat, music_candidates=neg_cross,
            ltp_feat=ltp_feat, text_feat=text_feat,
            input_ids=input_ids, attention_mask=attention_mask,
            labels=None, compute_gen_loss=False,
        )
        score_neg_cross = out_neg_cross["ranking_score"]   # (B,)

        # 混合 BPR Loss：mc 負責細粒度音樂鑑別，cross-video 負責跨 query 全域鑑別
        loss_rank_mc    = bpr_loss(score_pos, score_neg_mc)
        loss_rank_cross = bpr_loss(score_pos, score_neg_cross)
        loss_rank = 0.5 * loss_rank_mc + 0.5 * loss_rank_cross

        total_loss = (
            model_config.lambda_rank * loss_rank
            + model_config.lambda_gen * loss_gen
        )

    return {
        "loss":      total_loss,
        "loss_rank": loss_rank,
        "loss_gen":  loss_gen,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 一個 Epoch 的訓練
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: UnifiedMLLM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    train_config: TrainConfig,
    model_config: ModelConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> Dict:
    model.train()
    meters = {
        "loss":      AverageMeter(),
        "loss_rank": AverageMeter(),
        "loss_gen":  AverageMeter(),
    }
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        step_outputs = train_one_step(
            model=model,
            batch=batch,
            device=device,
            scaler=scaler,
            use_bf16=train_config.use_bf16,
            model_config=model_config,
            train_config=train_config,
        )

        accumulated_loss = step_outputs["loss"] / train_config.accumulation_steps
        scaler.scale(accumulated_loss).backward()

        meters["loss"].update(step_outputs["loss"].item())
        meters["loss_rank"].update(step_outputs["loss_rank"].item())
        if step_outputs["loss_gen"] is not None:
            meters["loss_gen"].update(step_outputs["loss_gen"].item())

        if (step + 1) % train_config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                train_config.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % 50 == 0:
                logger.info(
                    f"Epoch {epoch} Step {global_step} | "
                    f"loss={meters['loss'].avg:.4f} | "
                    f"rank={meters['loss_rank'].avg:.4f} | "
                    f"gen={meters['loss_gen'].avg:.4f}"
                )

    return {
        "global_step": global_step,
        "train_loss":  meters["loss"].avg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主訓練函數
# ─────────────────────────────────────────────────────────────────────────────

def train(model_config: ModelConfig, train_config: TrainConfig, ltp_dict=None):
    set_seed(42)
    os.makedirs(train_config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用裝置: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = LlamaTokenizer.from_pretrained(model_config.llama_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": train_config.special_tokens})
    logger.info(f"Tokenizer vocab 大小: {len(tokenizer)} "
                f"（加入 {len(train_config.special_tokens)} 個特殊 token）")
    # Pointwise: vocab = 32000 + 5 = 32005

    # ── 資料集 ────────────────────────────────────────────────────────────────
    logger.info("建立資料集（Video-Level 防外洩分割）...")
    train_loader, val_loader,  _ = build_dataloaders(
        data_dir         = train_config.data_dir,
        json_dir         = train_config.json_dir,
        tokenizer        = tokenizer,
        train_config     = train_config,
        model_config     = model_config,
        ltp_dict         = ltp_dict,
        pair_index_cache = train_config.pair_index_cache,
        song_bank_cache  = train_config.song_bank_cache,
    )
    logger.info(
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} "
    )

    from dataset import build_val_subset_loader
    val_subset_loader = build_val_subset_loader(
        val_loader.dataset,
        subset_size=500,
        seed=20260315,
        batch_size=train_config.eval_batch_size,
        cache_path=os.path.join(
            os.path.dirname(train_config.pair_index_cache),
            "val_subset_indices_500.json"
        ),
    )

    # 建立 song_bank tensor 供 validation 使用
    from dataset import build_pair_index, build_song_bank
    pair_index = build_pair_index(train_config.data_dir,
                                  cache_path=train_config.pair_index_cache)
    song_bank_np, song_ids = build_song_bank(pair_index,
                                              cache_path=train_config.song_bank_cache)
    all_music_features = torch.tensor(song_bank_np, dtype=torch.float32)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    logger.info("初始化 Unified MLLM（Pointwise v2）...")
    model = UnifiedMLLM(model_config=model_config, tokenizer=tokenizer)
    model.llama.resize_token_embeddings(len(tokenizer))
    # Pointwise: resize 至 32005（不是 v1 的 32013）
    model = model.to(device)
    logger.info(f"可訓練參數: {count_parameters(model, trainable_only=True):,}")

    # ── 優化器 ────────────────────────────────────────────────────────────────
    param_groups = model.get_parameter_groups(
        lr_projectors=train_config.learning_rate,
        lr_lora=train_config.lora_learning_rate,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    num_update_steps = (
        len(train_loader) // train_config.accumulation_steps
    ) * train_config.num_epochs
    num_warmup_steps = int(num_update_steps * train_config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_update_steps,
    )
    logger.info(f"訓練步數: {num_update_steps} | Warmup: {num_warmup_steps}")

    scaler = GradScaler(enabled=train_config.use_fp16)

    # ── Resume（從 checkpoint 接續訓練）────────────────────────────────────────
    global_step     = 0
    best_val_metric = 0.0
    start_epoch     = 1

    resume_ckpt = getattr(train_config, "resume_checkpoint", None)
    if resume_ckpt and os.path.isdir(resume_ckpt):
        logger.info(f"從 checkpoint 接續訓練：{resume_ckpt}")

        # 1. 載入 projectors 和 ranking_head 權重
        model.projectors.load_state_dict(
            torch.load(os.path.join(resume_ckpt, "projectors.pt"), map_location=device)
        )
        model.ranking_head.load_state_dict(
            torch.load(os.path.join(resume_ckpt, "ranking_head.pt"), map_location=device)
        )
        logger.info("  projectors / ranking_head 權重載入完成")

        # 2. 載入 LoRA 權重
        from peft import PeftModel
        base_llama = model.llama.base_model.model   # 取出凍結的 LLaMA 主體
        model.llama = PeftModel.from_pretrained(
            base_llama,
            resume_ckpt,
            torch_dtype=torch.bfloat16,
            is_trainable=True,
        )
        model.llama.to(device)
        logger.info("  LoRA 權重載入完成")

        # 3. 載入 optimizer / scheduler 狀態 + global_step
        state = load_checkpoint(resume_ckpt, model, optimizer, scheduler)
        global_step     = state.get("global_step", 0)
        best_val_metric = state.get("metrics", {}).get("recall@1", 0.0)

        # 4. 計算從哪個 epoch 開始
        steps_per_epoch = len(train_loader) // train_config.accumulation_steps
        start_epoch = (global_step // steps_per_epoch) + 1

        logger.info(
            f"  接續至 global_step={global_step}，"
            f"best_val_metric={best_val_metric:.4f}，"
            f"從 Epoch {start_epoch} 開始"
        )
    else:
        logger.info("從頭開始訓練（resume_checkpoint = None）")

    # ── 訓練迴圈 ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, train_config.num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{train_config.num_epochs}")
        logger.info(f"{'='*60}")

        epoch_result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_config=train_config,
            model_config=model_config,
            device=device,
            epoch=epoch,
            global_step=global_step,
        )
        global_step = epoch_result["global_step"]

        # ── Epoch 末 Validation（每個 epoch 只跑一次）──────────────────────────
        model.eval()
        val_metrics = pointwise_pool_evaluate_loader(
            model=model,
            data_loader=val_subset_loader,
            all_music_features=all_music_features,
            all_music_ids=song_ids,
            device=device,
            train_config=train_config,
            model_config=model_config,
            val_pool_size=50,
            max_samples=500,
            val_seed=20260315,
        )

        # ── ★ Fix B：計算 val loss（抽 50 個 batch，約 200 筆，不需全量）──────
        # 目的：提供 train loss vs val loss 的對比，判斷是否過擬合
        val_loss_meter  = AverageMeter()
        val_rank_meter  = AverageMeter()
        val_gen_meter   = AverageMeter()
        _val_steps = 0
        with torch.no_grad():
            for val_batch in val_subset_loader:
                if _val_steps >= 50:   # 最多 50 個 batch，避免 validation 太慢
                    break
                try:
                    step_out = train_one_step(
                        model=model, batch=val_batch, device=device,
                        scaler=scaler, use_bf16=train_config.use_bf16,
                        model_config=model_config, train_config=train_config,
                    )
                    val_loss_meter.update(step_out["loss"].item())
                    val_rank_meter.update(step_out["loss_rank"].item())
                    if step_out["loss_gen"] is not None:
                        val_gen_meter.update(step_out["loss_gen"].item())
                    _val_steps += 1
                except Exception:
                    break

        val_loss = val_loss_meter.avg if val_loss_meter.count > 0 else float("nan")
        # ─────────────────────────────────────────────────────────────────────

        logger.info(
            f"[Val Epoch {epoch}] R@1={val_metrics['recall@1']:.4f} | "
            f"R@5={val_metrics['recall@5']:.4f} | "
            f"R@10={val_metrics['recall@10']:.4f} | "
            f"MR={val_metrics['median_rank']:.1f} | "
            f"val_loss={val_loss:.4f}"   # ★ Fix B：加入 val_loss
        )
        # Composite metric：R@1×0.6 + R@5×0.2 + 1/MR×0.2
        metric_val = (
            val_metrics["recall@1"] * 0.6
            + val_metrics["recall@5"] * 0.2
            + (1.0 / max(val_metrics.get("median_rank", 10), 1)) * 0.2
        )
        if metric_val > best_val_metric:
            best_val_metric = metric_val
            save_checkpoint(
                model=model,
                save_dir=os.path.join(train_config.output_dir, "best"),
                global_step=global_step,
                metrics=val_metrics,
            )
            logger.info(
                f"  [Best] 新最佳 composite={metric_val:.4f} "
                f"(R@1={val_metrics['recall@1']:.4f} "
                f"R@5={val_metrics['recall@5']:.4f} "
                f"MR={val_metrics['median_rank']:.1f})"
            )
        model.train()

        # Epoch 末儲存（含 val_metrics，方便事後查閱每個 epoch 的完整記錄）
        save_checkpoint(
            model=model,
            save_dir=os.path.join(train_config.output_dir, f"epoch_{epoch}"),
            global_step=global_step,
            metrics={
                "epoch":       epoch,
                "train_loss":  epoch_result["train_loss"],
                "val_loss":    val_loss,                   # ★ Fix B：新增
                "recall@1":    val_metrics["recall@1"],
                "recall@5":    val_metrics["recall@5"],
                "recall@10":   val_metrics["recall@10"],
                "median_rank": val_metrics["median_rank"],
                "composite":   metric_val,
                "pool_size":   val_metrics["pool_size"],
                "num_samples": val_metrics["num_samples"],
            },
        )
        logger.info(f"Epoch {epoch} 完成 | train_loss={epoch_result['train_loss']:.4f} | val_loss={val_loss:.4f}")

    logger.info(f"\n訓練完成。Best {train_config.best_model_metric}={best_val_metric:.4f}")
    return {"global_step": global_step, "best_val_metric": best_val_metric}