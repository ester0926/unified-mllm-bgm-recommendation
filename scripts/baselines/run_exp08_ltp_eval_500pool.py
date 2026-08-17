# ⚠️ 此腳本已更名為 run_musechat_ltp_eval_500pool.py
# 請改用新腳本：scripts/baselines/run_musechat_ltp_eval_500pool.py
#
# 原因：實驗 exp_08 重命名為 musechat_ltp（MuseChat-light 架構 + LTP 輸入，
# 定位為 baseline 對照，有別於 exp_01~07 統一 MLLM 消融系列）。
# 此檔案保留僅為向後相容，請勿再直接使用。

raise SystemExit(
    "此腳本已更名。請改用：\n"
    "  python scripts/baselines/run_musechat_ltp_eval_500pool.py"
)

# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
exp_08（MVTFusionWithLTP）完整 500-pool 評估腳本。

這是 exp_08 論文最終數字的來源，與以下腳本使用完全相同的評估協定：
  - run_musechat_light_eval_500pool.py  （MuseChat-light 基準線）
  - run_eval_500pool_detailed.py        （exp_01~07 統一 MLLM）

相同之處：
  - CANDIDATE_POOL_SEED = 20260315（per-query 固定候選池）
  - TIEBREAK_NOISE = True / TIEBREAK_SEED = 42
  - split_by_video_id()（與所有實驗相同的測試集）
  - 輸出相同格式的 per-sample CSV（供 Wilcoxon 配對檢定使用）

唯一差異（exp_08 特有）：
  - 載入 MVTFusionWithLTP 而非 MVTFusionModule
  - encode_query 時額外傳入 ltp_feat

2×2 比較對照：
  ┌──────────────┬──────────────────┬──────────────────┐
  │              │   無 LTP         │   有 LTP         │
  ├──────────────┼──────────────────┼──────────────────┤
  │ Non-unified  │ MuseChat-light   │ exp_08（本腳本） │
  │ Unified      │ exp_04           │ exp_01           │
  └──────────────┴──────────────────┴──────────────────┘

執行方式：
    在 VSCode 開啟本檔案後直接 Run，或：
    python scripts/baselines/run_exp08_ltp_eval_500pool.py

輸出：
    checkpoints/exp_08_ltp/detailed_eval/
      exp_08_ltp_500pool_ranking_samples.csv   ← Wilcoxon 檢定輸入
      exp_08_ltp_500pool_summary.json          ← 指標摘要
"""

import csv
import datetime as _dt
import gc
import importlib.util
import json
import logging
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_main import run_eval_500pool_detailed as core
from config import TrainConfig
from dataset import build_pair_index, build_song_bank, extract_music_title, split_by_video_id


# =============================================================================
# USER SETTINGS（通常只需要改 MVT_LTP_CKPT_PATH）
# =============================================================================

MUSECHAT_DIR       = r"external/musechat"

# exp_08 checkpoint（訓練完成後 mvt_ltp_best.pt 的路徑）
MVT_LTP_CKPT_PATH  = os.path.join(MUSECHAT_DIR, "checkpoints", "exp_08_ltp", "mvt_ltp_best.pt")

# 若要評估特定 epoch，改為：
# MVT_LTP_CKPT_PATH = os.path.join(MUSECHAT_DIR, "checkpoints", "exp_08_ltp", "mvt_ltp_epoch30.pt")

LTP_H5_PATH        = r"data/user_profiling/stage5_output/preference_vectors.h5"
GENERATOR_CKPT_DIR = None      # None = 自動偵測 MuseChat-light generator

POOL_SIZE          = 500
MAX_SAMPLES        = None      # None = 全測試集（~4205 筆）
RUN_GT_GENERATION  = True      # GT-conditioned 文字生成（需要 generator checkpoint）
RUN_TOP1_GENERATION = True     # Top-1 end-to-end 文字生成
KEEP_PER_SAMPLE_INFOLM = True

# ── 固定 seed（與 musechat_light / exp_01~07 完全相同）──────────────────────
CANDIDATE_POOL_SEED = 20260315
TIEBREAK_NOISE      = True
TIEBREAK_SEED       = 42

TARGET_ENCODE_BATCH_SIZE = 4096
SEGMENT_BATCH_SIZE       = 1
GEN_MAX_NEW_TOKENS       = 128
INJECT_TITLE             = True

OUTPUT_DIR    = os.path.join(core.BASE_DIR, "checkpoints", "exp_08_ltp", "detailed_eval")
OUTPUT_PREFIX = f"exp_08_ltp_{POOL_SIZE}pool"


# =============================================================================
# Logger & IO
# =============================================================================

def setup_logger():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger("exp08_ltp_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}.log"), encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def write_csv(path, rows):
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# Model Loading
# =============================================================================

def load_musechat_modules():
    """載入 musechat repo 的模組，避免與主專案的 config.py / dataset.py 衝突。

    問題根源：主專案在頂層已 import dataset，sys.modules["dataset"] 指向
    unified_mllm_pointwise_final/dataset.py。musechat/models/__init__.py 也會
    `from dataset import MuseChatDataset`，Python 從快取拿錯模組 → ImportError。
    解法：進入 musechat 載入範圍前暫時移除快取，結束後還原。
    """
    old_config   = sys.modules.get("config")
    old_dataset  = sys.modules.get("dataset")   # ← 額外保存主專案 dataset
    old_sys_path = list(sys.path)

    muse_cfg_spec = importlib.util.spec_from_file_location(
        "musechat_config", os.path.join(MUSECHAT_DIR, "config.py")
    )
    assert muse_cfg_spec is not None and muse_cfg_spec.loader is not None, \
        f"找不到或無法載入 {MUSECHAT_DIR}/config.py"
    muse_cfg_mod = importlib.util.module_from_spec(muse_cfg_spec)
    muse_cfg_spec.loader.exec_module(muse_cfg_mod)

    try:
        sys.modules["config"] = muse_cfg_mod
        # 移除主專案 dataset 快取，讓 musechat/models/__init__.py 重新從
        # MUSECHAT_DIR（sys.path 第一位）載入正確的 musechat dataset
        sys.modules.pop("dataset", None)

        if MUSECHAT_DIR not in sys.path:
            sys.path.insert(0, MUSECHAT_DIR)

        # MVTFusionWithLTP
        mvt_ltp_spec = importlib.util.spec_from_file_location(
            "mvt_fusion_ltp",
            os.path.join(MUSECHAT_DIR, "models", "mvt_fusion_ltp.py"),
        )
        assert mvt_ltp_spec is not None and mvt_ltp_spec.loader is not None, \
            "找不到或無法載入 mvt_fusion_ltp.py"
        mvt_ltp_mod = importlib.util.module_from_spec(mvt_ltp_spec)
        mvt_ltp_spec.loader.exec_module(mvt_ltp_mod)

        # MuseChat-light SentenceGenerator（用於文字生成）
        gen_spec = importlib.util.spec_from_file_location(
            "sentence_generator",
            os.path.join(MUSECHAT_DIR, "models", "sentence_generator.py"),
        )
        assert gen_spec is not None and gen_spec.loader is not None, \
            "找不到或無法載入 sentence_generator.py"
        gen_mod = importlib.util.module_from_spec(gen_spec)
        gen_spec.loader.exec_module(gen_mod)

    finally:
        # 還原 config
        if old_config is not None:
            sys.modules["config"] = old_config
        else:
            sys.modules.pop("config", None)
        # 還原主專案 dataset
        if old_dataset is not None:
            sys.modules["dataset"] = old_dataset
        else:
            sys.modules.pop("dataset", None)
        sys.path[:] = old_sys_path

    return muse_cfg_mod, mvt_ltp_mod, gen_mod


def load_exp08_model(muse_cfg_mod, mvt_ltp_mod, device, logger):
    """載入 exp_08 MVTFusionWithLTP checkpoint。"""
    if not os.path.isfile(MVT_LTP_CKPT_PATH):
        raise FileNotFoundError(
            f"找不到 exp_08 checkpoint：{MVT_LTP_CKPT_PATH}\n"
            "請先執行 musechat/train_recommendation_ltp.py 完成訓練。"
        )
    cfg   = muse_cfg_mod.MuseChatConfig()
    model = mvt_ltp_mod.MVTFusionWithLTP(cfg=cfg.mvt, ltp_dim=256).to(device)
    ckpt  = torch.load(MVT_LTP_CKPT_PATH, map_location="cpu")
    # mvt_ltp_best.pt 直接存 state_dict；epoch checkpoint 有 "model_state" key
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("exp_08 model loaded: %s", MVT_LTP_CKPT_PATH)
    return model


def load_ltp_dict(logger):
    """載入 LTP 向量字典：{video_id: np.ndarray(256,)}"""
    import h5py
    logger.info("載入 LTP 向量：%s", LTP_H5_PATH)
    ltp_dict = {}
    with h5py.File(LTP_H5_PATH, "r") as f:
        grp = f["preference_vectors"]
        assert isinstance(grp, h5py.Group), "preference_vectors 應為 h5py.Group"
        for vid in grp.keys():
            ds = grp[vid]
            assert isinstance(ds, h5py.Dataset), f"vid={vid} 應為 h5py.Dataset"
            ltp_dict[vid] = np.array(ds[()], dtype=np.float32)
    logger.info("LTP 向量載入完成：%d 個 video_id", len(ltp_dict))
    return ltp_dict


# =============================================================================
# Shared Data（與 musechat_light eval 完全相同）
# =============================================================================

def build_shared_data(logger):
    train_cfg  = TrainConfig()
    pair_index = build_pair_index(core.H5_DIR, cache_path=os.path.join(core.CACHE_DIR, "pair_index.json"))
    song_bank_np, song_ids = build_song_bank(
        pair_index, cache_path=os.path.join(core.CACHE_DIR, "song_bank")
    )
    _, _, test_pairs = split_by_video_id(
        pair_index,
        train_cfg.train_ratio,
        train_cfg.val_ratio,
        train_cfg.test_ratio,
        train_cfg.split_seed,
    )
    conv_t3, conv_t4, _ = core.load_reference_maps()
    logger.info("Shared data: test_pairs=%d, song_bank=%d", len(test_pairs), len(song_ids))
    return test_pairs, torch.tensor(song_bank_np, dtype=torch.float32), song_ids, conv_t3, conv_t4


# =============================================================================
# Encoding
# =============================================================================

class H5FeatureReader:
    def __init__(self): self.handles = {}

    def get_handle(self, h5_path):
        if h5_path not in self.handles:
            self.handles[h5_path] = h5py.File(h5_path, "r")
        return self.handles[h5_path]

    def read_features(self, h5_path, pair_key):
        grp           = self.get_handle(h5_path)[f"pairs/{pair_key}"]
        video_all     = torch.from_numpy(grp["video_features_all"][:].astype(np.float32))       # (12, 768)
        candidate_all = torch.from_numpy(grp["candidate_music_all_seq"][:].astype(np.float32))  # (12, 1214, 768)
        text_seq      = torch.from_numpy(grp["text_features"][:].astype(np.float32))            # (77, 512)
        return video_all, candidate_all, text_seq

    def close(self):
        for h in self.handles.values():
            try: h.close()
            except: pass
        self.handles.clear()


@torch.no_grad()
def encode_target_bank(model, song_bank_tensor, device, logger):
    outputs = []
    for start in range(0, song_bank_tensor.size(0), TARGET_ENCODE_BATCH_SIZE):
        batch = song_bank_tensor[start:start + TARGET_ENCODE_BATCH_SIZE].to(device)
        outputs.append(model.encode_target(batch).detach().cpu())
    target = F.normalize(torch.cat(outputs, dim=0), dim=-1)
    logger.info("Target bank encoded: %s", tuple(target.shape))
    return target


@torch.no_grad()
def encode_query_ltp(model, reader, h5_path, pair_key, ltp_dict, device):
    """
    與 musechat_light 的 encode_query 完全相同，僅增加 ltp_feat 傳入。

    每個 segment 分別計算再取平均（12 segments → 1 個 256D query 向量），
    與 train_recommendation_ltp.py 的 evaluate_recommendation_ltp() 一致。
    """
    video_all, candidate_all, text_seq = reader.read_features(h5_path, pair_key)
    video_id = pair_key[:11]

    ltp_vec  = ltp_dict.get(video_id, np.zeros(256, dtype=np.float32))
    ltp_feat = torch.from_numpy(ltp_vec)   # (256,)

    outputs = []
    for start in range(0, video_all.size(0), SEGMENT_BATCH_SIZE):
        end      = min(start + SEGMENT_BATCH_SIZE, video_all.size(0))
        video    = video_all[start:end].to(device)      # (bs, 768)
        cand     = candidate_all[start:end].to(device)  # (bs, 1214, 768)
        text     = text_seq.unsqueeze(0).expand(end - start, -1, -1).to(device)  # (bs, 77, 512)
        ltp      = ltp_feat.unsqueeze(0).expand(end - start, -1).to(device)       # (bs, 256)
        outputs.append(model(video, cand, text, ltp).detach().cpu())

    query = torch.cat(outputs, dim=0).mean(dim=0)      # (256,)
    return F.normalize(query, dim=-1).to(device)


# =============================================================================
# Ranking（與 musechat_light eval 完全相同，僅 encode 函數不同）
# =============================================================================

def rank_from_scores(scores, tiebreak_rng):
    scores_np = np.asarray(scores, dtype=np.float64)
    scores_for_sort = scores_np + (tiebreak_rng.uniform(0, 1e-8, size=scores_np.shape)
                                   if TIEBREAK_NOISE else 0)
    sorted_indices = np.argsort(scores_for_sort)[::-1]
    rank           = int(np.where(sorted_indices == 0)[0][0]) + 1
    top1_pool_idx  = int(sorted_indices[0])
    return rank, top1_pool_idx, scores_np


def run_ranking(model, ltp_dict, test_pairs, target_bank, song_ids, logger):
    reader       = H5FeatureReader()
    id_to_index  = {sid: i for i, sid in enumerate(song_ids)}
    vid_to_idxs  = {}
    for i, sid in enumerate(song_ids):
        vid_to_idxs.setdefault(sid[:11], set()).add(i)

    n_eval       = len(test_pairs) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(test_pairs))
    tiebreak_rng = np.random.default_rng(TIEBREAK_SEED)
    rows         = []

    from tqdm import tqdm
    try:
        for idx in tqdm(range(n_eval), desc=f"exp_08 ranking ({POOL_SIZE}-pool)"):
            h5_path, pair_key = test_pairs[idx]
            video_id          = pair_key[:11]
            gt_global_idx     = id_to_index.get(pair_key)
            if gt_global_idx is None:
                raise KeyError(f"GT music id not found in song bank: {pair_key}")

            query    = encode_query_ltp(model, reader, h5_path, pair_key, ltp_dict, target_bank.device)
            excluded = vid_to_idxs.get(video_id, set())
            candidates = [i for i in range(target_bank.size(0))
                          if i != gt_global_idx and i not in excluded]

            # per-query 固定候選池（與所有實驗相同）
            rng_pool  = random.Random(CANDIDATE_POOL_SEED + idx)
            negatives = rng_pool.sample(candidates, min(POOL_SIZE - 1, len(candidates)))
            pool_idx  = [gt_global_idx] + negatives

            pool_feats = target_bank[pool_idx]
            scores     = (query.unsqueeze(0) @ pool_feats.T).squeeze(0).cpu().numpy()
            rank, top1_pool_idx, scores_np = rank_from_scores(scores, tiebreak_rng)
            top1_global_idx = pool_idx[top1_pool_index := top1_pool_idx]

            rows.append({
                "sample_idx":               idx,
                "video_id":                 video_id,
                "gt_music_id":              pair_key,
                "top1_music_id":            song_ids[top1_global_idx],
                "top1_is_gt":               int(rank == 1),
                "rank":                     rank,
                "R@1":                      int(rank <= 1),
                "R@5":                      int(rank <= 5),
                "R@10":                     int(rank <= 10),
                "pool_size":                POOL_SIZE,
                "baseline":                 "exp_08_ltp",
                "has_ltp":                  int(video_id in ltp_dict),
                "gt_score":                 float(scores_np[0]),
                "top1_score":               float(scores_np[top1_pool_index]),
                "score_gap_top1_minus_gt":  float(scores_np[top1_pool_index] - scores_np[0]),
                "score_range":              float(scores_np.max() - scores_np.min()),
                "score_std":                float(scores_np.std()),
                "n_equal_to_gt_score":      int(np.isclose(scores_np, scores_np[0], atol=1e-7).sum()),
            })
    finally:
        reader.close()

    ranks   = np.array([r["rank"] for r in rows], dtype=np.float64)
    summary = {
        "recall@1":          float(np.mean([r["R@1"]  for r in rows])),
        "recall@5":          float(np.mean([r["R@5"]  for r in rows])),
        "recall@10":         float(np.mean([r["R@10"] for r in rows])),
        "median_rank":       float(np.median(ranks)),
        "mean_rank":         float(np.mean(ranks)),
        "num_samples":       len(rows),
        "n_with_ltp":        int(sum(r["has_ltp"] for r in rows)),
        "pool_size":         POOL_SIZE,
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "tiebreak_noise":    bool(TIEBREAK_NOISE),
        "tiebreak_seed":     int(TIEBREAK_SEED),
    }
    logger.info(
        "exp_08 ranking: R@1=%.4f  R@5=%.4f  R@10=%.4f  MedR=%.1f  (n=%d)",
        summary["recall@1"], summary["recall@5"],
        summary["recall@10"], summary["median_rank"], summary["num_samples"],
    )
    return rows, summary


# =============================================================================
# Generation（使用 MuseChat-light sentence generator，與 musechat_light eval 相同）
# =============================================================================

def normalize_for_match(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"\s+", " ", text).strip()


def mentions_value(generated_text, value):
    value = normalize_for_match(value)
    if not value: return None
    generated = normalize_for_match(generated_text)
    if value in generated: return True
    cv = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    cg = re.sub(r"[\W_]+", "", generated, flags=re.UNICODE)
    return bool(cv and cv in cg)


def title_consistency_flags(generated_text, music_title, music_artist):
    th = mentions_value(generated_text, music_title)
    ah = mentions_value(generated_text, music_artist)
    checked = th is not None or ah is not None
    if not checked:             consistency = None
    elif th is False or ah is False: consistency = False
    else:                            consistency = True
    return {
        "generated_mentions_music_title":  "" if th is None else int(th),
        "generated_mentions_music_artist": "" if ah is None else int(ah),
        "title_consistency":               "" if consistency is None else int(consistency),
        "needs_manual_review":             1 if consistency is False else 0,
    }


def find_generator_checkpoint():
    if GENERATOR_CKPT_DIR and os.path.exists(
        os.path.join(GENERATOR_CKPT_DIR, "training_state.pt")
    ):
        return GENERATOR_CKPT_DIR
    ckpt_root = Path(MUSECHAT_DIR) / "checkpoints"
    candidates = sorted(
        [p for p in ckpt_root.glob("generator_epoch*")
         if (p / "training_state.pt").exists()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return str(candidates[0]) if candidates else None


def load_generator(gen_mod, muse_cfg_mod, device, logger):
    ckpt_dir = find_generator_checkpoint()
    if not ckpt_dir:
        logger.warning("未找到 generator checkpoint，跳過文字生成。")
        return None, None, None
    muse_cfg  = muse_cfg_mod.MuseChatConfig()
    tokenizer = gen_mod.build_tokenizer(muse_cfg.gen.llm_model_name)
    generator = gen_mod.SentenceGenerator(cfg=muse_cfg.gen, tokenizer=tokenizer)
    try:
        generator.llm.resize_token_embeddings(len(tokenizer))
    except Exception:
        bm = getattr(generator.llm, "base_model", None)
        if bm: bm.model.resize_token_embeddings(len(tokenizer))
    state = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location="cpu")
    generator.music_proj.load_state_dict(state["music_proj_state"])
    generator.music_proj = generator.music_proj.to(device)
    lora_dir = os.path.join(ckpt_dir, "lora_weights")
    if os.path.isdir(lora_dir) and hasattr(generator.llm, "load_adapter"):
        generator.llm.load_adapter(lora_dir, adapter_name="exp08_gen", is_trainable=False)
        if hasattr(generator.llm, "set_adapter"): generator.llm.set_adapter("exp08_gen")
    generator.eval()
    logger.info("Generator loaded: %s", ckpt_dir)
    return generator, tokenizer, ckpt_dir


@torch.no_grad()
def generate_text(generator, music_feat, title_string):
    device = next(generator.music_proj.parameters()).device
    return generator.generate(
        music_avg=music_feat.unsqueeze(0).to(device),
        music_title=title_string,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        temperature=0.1, do_sample=False,
    ).strip()


def _title_from_ref(t4_text):
    title, artist = extract_music_title(t4_text)
    if title and artist: return f"{title} by {artist}", title, artist
    if title:            return title, title, artist
    return None, title, artist


def run_generation(generator, ranking_rows, song_bank_tensor, song_ids,
                   conv_t3, conv_t4, mode, logger):
    id_to_index = {sid: i for i, sid in enumerate(song_ids)}
    rows        = []
    from tqdm import tqdm
    for rank_row in tqdm(ranking_rows, desc=f"exp_08 {mode} generation"):
        sample_idx  = int(rank_row["sample_idx"])
        video_id    = rank_row["video_id"]
        gt_music_id = rank_row["gt_music_id"]
        ref_text    = conv_t4.get(video_id, "")
        user_text   = conv_t3.get(video_id, "")

        if mode == "gt_conditioned":
            gen_music_id = gt_music_id
            title_ref    = ref_text
            title_source = "query_gt_reference"
        else:
            gen_music_id = rank_row["top1_music_id"]
            top1_vid     = str(gen_music_id)[:11]
            title_ref    = conv_t4.get(top1_vid, "") or ref_text
            title_source = "top1_reference" if conv_t4.get(top1_vid, "") else "query_reference_fallback"

        music_index          = id_to_index.get(gen_music_id)
        title, m_title, m_artist = _title_from_ref(title_ref)

        if music_index is None or not ref_text:
            generated_text = ""; is_fallback = True
        else:
            try:
                generated_text = generate_text(generator, song_bank_tensor[music_index], title)
                is_fallback    = not bool(generated_text)
            except Exception as exc:
                logger.warning("Generation failed (sample=%d): %s", sample_idx, exc)
                generated_text = ""; is_fallback = True

        rows.append({
            "sample_idx":              sample_idx,
            "video_id":                video_id,
            "gt_music_id":             gt_music_id,
            "generation_music_id":     gen_music_id,
            "top1_music_id":           rank_row.get("top1_music_id", ""),
            "top1_is_gt":              rank_row.get("top1_is_gt", ""),
            "rank":                    rank_row.get("rank", ""),
            "R@1":                     rank_row.get("R@1", ""),
            "R@5":                     rank_row.get("R@5", ""),
            "R@10":                    rank_row.get("R@10", ""),
            "pool_size":               POOL_SIZE,
            "baseline":                "exp_08_ltp",
            "generation_mode":         mode,
            "title_source":            title_source,
            "music_title":             m_title,
            "music_artist":            m_artist,
            "user_text":               user_text,
            "generated_text":          generated_text,
            "reference_text":          ref_text,
            "title_reference_text":    title_ref,
            **title_consistency_flags(generated_text, m_title, m_artist),
            "is_fallback":             int(is_fallback),
            "bertscore_precision":     None,
            "bertscore_recall":        None,
            "bertscore_f1":            None,
            "infolm_ab_divergence":    None,
            "infolm_l2_distance":      None,
            "infolm_fisher_rao":       None,
        })

    # 文字指標（BERTScore + InfoLM）
    metric_summary = {}
    metric_summary.update(core.add_bertscore_to_rows(rows, logger))
    metric_summary.update(core.add_infolm_to_rows(rows, per_sample=KEEP_PER_SAMPLE_INFOLM, logger=logger))

    csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{mode}_generation_samples.csv")
    write_csv(csv_path, rows)
    logger.info("Generation saved: %s", csv_path)
    return rows, metric_summary


# =============================================================================
# Main
# =============================================================================

def main():
    logger     = setup_logger()
    started_at = _dt.datetime.now()
    logger.info("=" * 65)
    logger.info("exp_08（MVTFusionWithLTP）500-pool 完整評估")
    logger.info("Checkpoint : %s", MVT_LTP_CKPT_PATH)
    logger.info("LTP 路徑   : %s", LTP_H5_PATH)
    logger.info("Pool Size  : %d  |  Seed: %d  |  Tiebreak: %s",
                POOL_SIZE, CANDIDATE_POOL_SEED, TIEBREAK_NOISE)
    logger.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for inference.")

    # ── 1. 共用資料（與所有實驗相同）──────────────────────────────────────────
    test_pairs, song_bank_tensor, song_ids, conv_t3, conv_t4 = build_shared_data(logger)

    # ── 2. 載入 exp_08 模型 + LTP ─────────────────────────────────────────────
    muse_cfg_mod, mvt_ltp_mod, gen_mod = load_musechat_modules()
    model    = load_exp08_model(muse_cfg_mod, mvt_ltp_mod, device, logger)
    ltp_dict = load_ltp_dict(logger)

    # ── 3. 建立 target bank ────────────────────────────────────────────────────
    target_bank = encode_target_bank(model, song_bank_tensor, device, logger).to(device)

    # ── 4. Ranking ────────────────────────────────────────────────────────────
    ranking_rows, ranking_summary = run_ranking(
        model, ltp_dict, test_pairs, target_bank, song_ids, logger
    )
    ranking_csv  = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ranking_samples.csv")
    ranking_jsonl = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ranking_samples.jsonl")
    write_csv(ranking_csv, ranking_rows)
    write_jsonl(ranking_jsonl, ranking_rows)
    logger.info("Ranking CSV saved: %s", ranking_csv)

    # ── 5. 釋放 ranker，載入 generator ──────────────────────────────────────
    del model, target_bank
    torch.cuda.empty_cache(); gc.collect()

    generation_summary = {"gt_conditioned": {"skipped": True}, "top1_end_to_end": {"skipped": True}}
    generator_ckpt     = None

    if RUN_GT_GENERATION or RUN_TOP1_GENERATION:
        generator, _, generator_ckpt = load_generator(gen_mod, muse_cfg_mod, device, logger)
        if generator is not None:
            if RUN_GT_GENERATION:
                _, gt_sum = run_generation(
                    generator, ranking_rows, song_bank_tensor, song_ids,
                    conv_t3, conv_t4, "gt_conditioned", logger,
                )
                generation_summary["gt_conditioned"] = gt_sum
            if RUN_TOP1_GENERATION:
                _, top1_sum = run_generation(
                    generator, ranking_rows, song_bank_tensor, song_ids,
                    conv_t3, conv_t4, "top1_end_to_end", logger,
                )
                generation_summary["top1_end_to_end"] = top1_sum

    # ── 6. Summary ────────────────────────────────────────────────────────────
    summary = {
        "experiment":         "exp_08_ltp",
        "description": (
            "Non-unified model (MuseChat-light MVT-Fusion) + LTP (Long-Term Preference). "
            "Trained from scratch with identical hyperparams as MuseChat-light (lr=4e-5, "
            "epochs=30, batch=8, accum=68). "
            "Enables 2x2 factorial: unified-architecture × LTP contribution."
        ),
        "checkpoint":         MVT_LTP_CKPT_PATH,
        "generator_checkpoint": generator_ckpt,
        "pool_size":          POOL_SIZE,
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "tiebreak_noise":     TIEBREAK_NOISE,
        "tiebreak_seed":      TIEBREAK_SEED,
        "max_samples":        MAX_SAMPLES,
        "started_at":         started_at.isoformat(timespec="seconds"),
        "finished_at":        _dt.datetime.now().isoformat(timespec="seconds"),
        "ranking":            ranking_summary,
        "generation":         generation_summary,
    }
    summary_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 7. 印出結果 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"exp_08（MVTFusionWithLTP）500-pool 評估結果")
    print(f"  R@1   = {ranking_summary['recall@1']*100:.2f}%")
    print(f"  R@5   = {ranking_summary['recall@5']*100:.2f}%")
    print(f"  R@10  = {ranking_summary['recall@10']*100:.2f}%")
    print(f"  MedR  = {ranking_summary['median_rank']:.1f}")
    print(f"  n     = {ranking_summary['num_samples']}")
    print()
    print("【2×2 比較對照（更新後）】")
    print("              │ 無 LTP          │ 有 LTP")
    print("  Non-unified │ MuseChat-light  │ exp_08（本結果）")
    print(f"              │  2.88% R@1      │  {ranking_summary['recall@1']*100:.2f}% R@1")
    print("  Unified     │ exp_04          │ exp_01")
    print("              │ 19.07% R@1      │ 30.65% R@1")
    print(f"\nSummary → {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
