"""
用途：訓練 MuseChat light generator baseline。
輸入：已訓練 checkpoint、測試集特徵、候選 pool 與 LTP/cache 資料。
輸出：ranking、generation、指標摘要或逐筆評估檔。
執行：建議在 repo 根目錄執行，必要資料請先由 Zenodo 解壓到對應資料夾。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import datetime as _dt
import gc
import importlib.util
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from scripts.eval_main import run_eval_500pool_detailed as core
from config import TrainConfig
from dataset import build_pair_index, split_by_video_id


# =============================================================================
# 使用前可調整的設定
# =============================================================================

MUSECHAT_DIR = r"external/musechat"
OUTPUT_DIR = os.path.join(MUSECHAT_DIR, "checkpoints")

LLM_MODEL_NAME = "lmsys/vicuna-7b-v1.5"
NUM_EPOCHS = 3
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
MAX_SEQ_LEN = 128
MAX_TRAIN_SAMPLES = None  # 例如 200 可快速檢查；None 表示完整訓練集

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "o_proj"]

USE_BF16 = True
GRAD_CLIP = 1.0
SEED = 42
LOG_EVERY = 10
SAVE_EVERY_EPOCH = True

PROMPT = (
    "### Recommender: Music feature: <Music> [music_token] </Music>; "
    "Generate Recommendation:"
)


def setup_logger():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger("musechat_light_generator_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    log_path = os.path.join(OUTPUT_DIR, "train_generator_current_data.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    spec.loader.exec_module(module)
    return module


def load_musechat_sentence_module():
    old_config = sys.modules.get("config")
    old_sys_path = list(sys.path)
    muse_config = load_module_from_file(
        "musechat_light_config_for_generator_train",
        os.path.join(MUSECHAT_DIR, "config.py"),
    )
    try:
        sys.modules["config"] = muse_config
        if MUSECHAT_DIR not in sys.path:
            sys.path.insert(0, MUSECHAT_DIR)
        sentence_module = load_module_from_file(
            "musechat_light_sentence_generator_for_train",
            os.path.join(MUSECHAT_DIR, "models", "sentence_generator.py"),
        )
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config
        sys.path[:] = old_sys_path
    return muse_config, sentence_module


def build_main_train_pairs(logger):
    train_cfg = TrainConfig()
    pair_index = build_pair_index(core.H5_DIR, cache_path=os.path.join(core.CACHE_DIR, "pair_index.json"))
    train_pairs, val_pairs, test_pairs = split_by_video_id(
        pair_index,
        train_cfg.train_ratio,
        train_cfg.val_ratio,
        train_cfg.test_ratio,
        train_cfg.split_seed,
    )
    if MAX_TRAIN_SAMPLES is not None:
        train_pairs = train_pairs[:MAX_TRAIN_SAMPLES]
    logger.info(
        "Unified split loaded: train=%d val=%d test=%d max_train_samples=%s",
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
        MAX_TRAIN_SAMPLES,
    )
    return train_pairs


class UnifiedMuseChatGeneratorDataset(Dataset):
    def __init__(self, pairs, tokenizer, conv_t4, max_seq_len):
        self.pairs = [
            (h5_path, pair_key)
            for h5_path, pair_key in pairs
            if pair_key[:11] in conv_t4 and conv_t4[pair_key[:11]].strip()
        ]
        self.tokenizer = tokenizer
        self.conv_t4 = conv_t4
        self.max_seq_len = max_seq_len
        self.handles = {}

    def __len__(self):
        return len(self.pairs)

    def __del__(self):
        for handle in getattr(self, "handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def _handle(self, h5_path):
        if h5_path not in self.handles:
            self.handles[h5_path] = h5py.File(h5_path, "r")
        return self.handles[h5_path]

    def _encode_prompt_and_response(self, response):
        full_text = PROMPT + " " + response
        prompt_ids = self.tokenizer(
            PROMPT,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        )["input_ids"]
        enc = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), labels.size(0))
        labels[:prompt_len] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

    def __getitem__(self, idx):
        h5_path, pair_key = self.pairs[idx]
        video_id = pair_key[:11]
        grp = self._handle(h5_path)[f"pairs/{pair_key}"]
        music_avg = grp["target_music_all_cls"][:].astype(np.float32).mean(axis=0)
        response = self.conv_t4[video_id].strip()
        input_ids, labels = self._encode_prompt_and_response(response)
        return torch.from_numpy(music_avg), input_ids, labels


def build_linear_decay_scheduler(optimizer, total_steps):
    def lr_lambda(current_step):
        return max(0.0, 1.0 - current_step / float(max(1, total_steps)))
    return LambdaLR(optimizer, lr_lambda)


def resize_token_embeddings(model, tokenizer, logger):
    try:
        model.llm.resize_token_embeddings(len(tokenizer))
        logger.info("Resized LLM token embeddings to %d", len(tokenizer))
        return
    except Exception as exc:
        logger.warning("Direct resize_token_embeddings failed: %s", exc)

    base_model = getattr(model.llm, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        base_model.model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized base_model token embeddings to %d", len(tokenizer))
        return
    raise RuntimeError("Unable to resize LLM token embeddings for [music_token].")


def save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, global_step, train_config):
    ckpt_dir = os.path.join(OUTPUT_DIR, f"generator_epoch{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.llm.save_pretrained(os.path.join(ckpt_dir, "lora_weights"))
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "music_proj_state": model.music_proj.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "avg_loss": avg_loss,
            "train_config": train_config,
            "prompt": PROMPT,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        os.path.join(ckpt_dir, "training_state.pt"),
    )
    with open(os.path.join(ckpt_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)
    return ckpt_dir


def main():
    logger = setup_logger()
    set_seed(SEED)
    started_at = _dt.datetime.now()
    logger.info("MuseChat-light Sentence Generator training on unified data")
    logger.info("MuseChat dir: %s", MUSECHAT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training expects CUDA.")

    muse_config, sentence_module = load_musechat_sentence_module()
    cfg = muse_config.MuseChatConfig()
    cfg.output_dir = OUTPUT_DIR
    cfg.seed = SEED
    cfg.gen.llm_model_name = LLM_MODEL_NAME
    cfg.gen.num_epochs = NUM_EPOCHS
    cfg.gen.micro_batch_size = MICRO_BATCH_SIZE
    cfg.gen.learning_rate = LEARNING_RATE
    cfg.gen.max_seq_len = MAX_SEQ_LEN
    cfg.gen.lora_r = LORA_R
    cfg.gen.lora_alpha = LORA_ALPHA
    cfg.gen.lora_dropout = LORA_DROPOUT
    cfg.gen.lora_target_modules = LORA_TARGET_MODULES

    tokenizer = sentence_module.build_tokenizer(cfg.gen.llm_model_name)
    conv_t3, conv_t4, _ = core.load_reference_maps()
    train_pairs = build_main_train_pairs(logger)
    train_dataset = UnifiedMuseChatGeneratorDataset(train_pairs, tokenizer, conv_t4, MAX_SEQ_LEN)
    if len(train_dataset) == 0:
        raise RuntimeError("No train samples with valid t4 references were found.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    logger.info("Generator train dataset=%d batches/epoch=%d", len(train_dataset), len(train_loader))

    model = sentence_module.SentenceGenerator(cfg=cfg.gen, tokenizer=tokenizer)
    resize_token_embeddings(model, tokenizer, logger)
    model.music_proj = model.music_proj.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable parameters: %d", sum(p.numel() for p in trainable_params))

    optimizer = AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    total_update_steps = math.ceil(len(train_loader) / max(GRAD_ACCUM_STEPS, 1)) * NUM_EPOCHS
    scheduler = build_linear_decay_scheduler(optimizer, total_update_steps)
    scaler = GradScaler(enabled=not USE_BF16)

    train_config = {
        "data_h5_dir": core.H5_DIR,
        "json_dir": core.JSON_DIR,
        "split_seed": TrainConfig().split_seed,
        "num_epochs": NUM_EPOCHS,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_seq_len": MAX_SEQ_LEN,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "use_bf16": USE_BF16,
        "seed": SEED,
        "started_at": started_at.isoformat(timespec="seconds"),
    }

    global_step = 0
    update_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            music_avg, input_ids, labels = batch
            music_avg = music_avg.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            if USE_BF16:
                with autocast(dtype=torch.bfloat16):
                    loss = model(
                        music_avg=music_avg,
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                    loss = loss / max(GRAD_ACCUM_STEPS, 1)
                loss.backward()
            else:
                with autocast(dtype=torch.float16):
                    loss = model(
                        music_avg=music_avg,
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                    loss = loss / max(GRAD_ACCUM_STEPS, 1)
                scaler.scale(loss).backward()

            epoch_loss += float(loss.item()) * max(GRAD_ACCUM_STEPS, 1)
            global_step += 1

            should_update = step % max(GRAD_ACCUM_STEPS, 1) == 0 or step == len(train_loader)
            if should_update:
                if USE_BF16:
                    nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            if global_step % LOG_EVERY == 0:
                logger.info(
                    "epoch=%d step=%d/%d update=%d loss=%.4f lr=%.2e",
                    epoch,
                    step,
                    len(train_loader),
                    update_step,
                    float(loss.item()) * max(GRAD_ACCUM_STEPS, 1),
                    scheduler.get_last_lr()[0],
                )

        avg_loss = epoch_loss / max(len(train_loader), 1)
        logger.info("Epoch %d finished: avg_loss=%.4f", epoch, avg_loss)
        if SAVE_EVERY_EPOCH:
            ckpt_dir = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                avg_loss=avg_loss,
                global_step=global_step,
                train_config=train_config,
            )
            logger.info("Saved generator checkpoint: %s", ckpt_dir)

    train_config["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(OUTPUT_DIR, "train_generator_current_data_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("MuseChat-light Sentence Generator training complete.")


if __name__ == "__main__":
    main()
