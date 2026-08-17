"""
dataset.py — Unified MLLM 資料集（Pointwise v2 Plan B）

Plan B 主要改動（相比原 Pointwise v2）：
  1. build_prompt() 在 [/INST] 後加入 [RANK] token
     → [RANK] 作為 ranking readout token，可 attend 所有 prefix 模態 + t3 文字
  2. prompt_len_for_generate 修正為純文字長度（含 [RANK]）
     → 不加 MULTIMODAL_PREFIX_LEN（prefix 不在 input_ids 中）
     → 這同時修正了先前 generation loss offset bug
  3. labels mask 到 prompt_token_len（含 [RANK]），response 才計算 loss
  4. MULTIMODAL_PREFIX_LEN 注解更新：新 prefix 順序 [VIDEO, LTP, TEXT_CLIP, MUSIC]
"""

import json
import os
import glob
import random
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 模板（Pointwise Plan B 版）
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert music recommendation assistant for short videos. "
    "Analyze the video content, user preferences, and candidate track to "
    "recommend the most suitable background music."
)

def extract_music_title(t4_text):
    """
    從 t4 response 解析歌名與藝人名，供推論階段注入 prompt 使用。

    支援格式：
      1. 'Title' by Artist（最常見，有引號）
      2. Title by Artist（recommend/suggest 後接無引號歌名）

    回傳：
      (title_str, artist_str) 解析成功
      (None, None)            無法解析（fallback / 無真實歌名的樣本）
    """
    import re
    if not t4_text:
        return None, None

    # 格式 1：有引號的歌名（單引號、雙引號、Unicode 引號均支援）
    m = re.search(
        r"['\u2018\u2019\u201c\u201d](.+?)['\u2018\u2019\u201c\u201d]\s+by\s+([^.?,!]+)",
        t4_text
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # 格式 2：無引號，recommend/suggest 後接 Title by Artist
    m = re.search(
        r"(?:recommend|suggest)\s+([A-Z][^.?,!]+?)\s+by\s+([^.?,!]+)",
        t4_text
    )
    if m:
        title  = m.group(1).strip()
        artist = m.group(2).strip()
        if len(title) > 2 and title.lower() not in ("this music", "a song", "the song"):
            return title, artist

    return None, None


def build_prompt(active_modalities=None, music_title=None, music_artist=None) -> str:
    """
    Plan B prompt：在 [/INST] 後加入 [RANK] token。

    ★ 推論階段標題注入（對應 MuseChat 論文的 Inference 設計）：
      訓練時：music_title=None → 只有聲學特徵，無歌名
      推論時：music_title="Song Name" → 在 prompt 中注入歌名行
      原因：模型無法從聲學 embedding 得知歌名，需要外部注入

      對應 MuseChat 論文的描述（Fig. 2 / Section 4.2）：
        訓練 prompt：只有 Music feature token
        推論 prompt：Music title: [title]; Music feature: [token]

    ★ 消融實驗：active_modalities 控制哪些模態行出現
      None = 全模態（向後相容）。

    ⚠️ train.py 呼叫此函式時永遠不傳 music_title（None），
       generate_recommendation.py / run_eval_500pool_v3.py 推論時才傳入。
    """
    if active_modalities is None:
        active_modalities = ["video", "ltp", "text", "music"]

    parts = [f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"]

    if "video" in active_modalities:
        parts.append("Video: [VIDEO]\n")

    if "music" in active_modalities:
        # ★ 推論階段：若有歌名，先寫出來再放音樂特徵 token
        if music_title:
            if music_artist:
                parts.append(f"Candidate: {music_title} by {music_artist}; [MUSIC]\n")
            else:
                parts.append(f"Candidate: {music_title}; [MUSIC]\n")
        else:
            # 訓練階段：只有聲學 token，不含歌名
            parts.append("Candidate: [MUSIC]\n")

    if "ltp" in active_modalities:
        parts.append("User preference: [LTP]\n")

    if "text" in active_modalities:
        parts.append("Context: [TEXT_CLIP] {user_text}\n")

    parts.append("\nDoes this candidate best fit this video? [/INST] [RANK] ")
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1：建立 pair 索引（與 v2 完全相同）
# ─────────────────────────────────────────────────────────────────────────────

def build_pair_index(
    h5_dir: str,
    cache_path: Optional[str] = None,
) -> List[Tuple[str, str]]:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        logger.info(f"[PairIndex] 快取載入：{len(data)} pairs")
        return [tuple(x) for x in data]

    h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"在 {h5_dir} 中找不到 .h5 檔案")

    logger.info(f"[PairIndex] 掃描 {len(h5_files)} 個 HDF5 檔案...")
    pair_index = []

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                if "pairs" not in f:
                    continue
                for key in f["pairs"].keys():
                    pair_index.append((h5_path, key))
        except Exception as e:
            logger.warning(f"  跳過 {h5_path}: {e}")

    logger.info(f"[PairIndex] {len(pair_index)} pairs")
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(pair_index, f)
    return pair_index


# ─────────────────────────────────────────────────────────────────────────────
# Step 2：對話文字（與 v2 完全相同）
# ─────────────────────────────────────────────────────────────────────────────

def load_conversation_map(json_dir: str) -> Dict[str, Tuple[str, str]]:
    conv_map = {}
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
    logger.info(f"[ConvMap] 載入 {len(json_files)} 個 JSON...")
    n_ok = n_skip = 0
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            convs = data.get("conversations", [])
            if len(convs) < 4:
                n_skip += 1; continue
            p = convs[2].get("value", "").strip()   # t3：用戶第二輪文字偏好
            r = convs[3].get("value", "").strip()   # t4：助手第二輪回覆（generation target）
            if not p or not r:
                n_skip += 1; continue
            video_id = Path(jf).parent.name
            conv_map[video_id] = (p, r)
            n_ok += 1
        except Exception:
            n_skip += 1
    logger.info(f"[ConvMap] {n_ok} OK，{n_skip} 跳過")
    return conv_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3：歌曲特徵庫（與 v2 完全相同）
# ─────────────────────────────────────────────────────────────────────────────

def build_song_bank(
    pair_index: List[Tuple[str, str]],
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    if cache_path:
        npy = cache_path + ".npy"
        ids = cache_path + "_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids) as f:
                sid = json.load(f)
            logger.info(f"[SongBank] 快取載入：{len(sid)} 首 {arr.shape}")
            return arr, sid

    logger.info(f"[SongBank] 建立中（{len(pair_index)} pairs）...")

    h5_to_keys = defaultdict(list)
    for h5_path, pk in pair_index:
        h5_to_keys[h5_path].append(pk)

    feats = {}
    for h5_path, keys in h5_to_keys.items():
        try:
            with h5py.File(h5_path, "r") as f:
                for key in keys:
                    try:
                        arr = f[f"pairs/{key}/target_music_all_cls"][:].astype(np.float32)
                        feats[key] = arr.mean(axis=0)   # (768,)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"  跳過 {h5_path}: {e}")

    song_ids   = sorted(feats.keys())
    song_array = np.stack([feats[k] for k in song_ids])

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path + ".npy", song_array)
        with open(cache_path + "_ids.json", "w") as f:
            json.dump(song_ids, f)

    logger.info(f"[SongBank] {len(song_ids)} 首 {song_array.shape}")
    return song_array, song_ids


# ─────────────────────────────────────────────────────────────────────────────
# 資料分割（與 v2 完全相同）
# ─────────────────────────────────────────────────────────────────────────────

def split_by_video_id(
    pair_index: List[Tuple[str, str]],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List, List, List]:
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

    logger.info(
        f"分割 — Train: {len(tr_p)} ({n_tr} vids) | "
        f"Val: {len(va_p)} ({n_va} vids) | "
        f"Test: {len(te_p)} ({n-n_tr-n_va} vids)"
    )
    return tr_p, va_p, te_p


# ─────────────────────────────────────────────────────────────────────────────
# Dataset（Pointwise v2 Plan B）
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedMLLMDataset(Dataset):
    """
    Pointwise Plan B Dataset。

    每筆 __getitem__ 回傳：
      - 查詢側特徵：video_feat, ltp_feat, text_feat, input_ids, attention_mask, labels
      - pos_music_feat : (768,) — GT 音樂特徵
      - neg_music_feat : (768,) — mc Hard Negative
      - gt_music_id    : str   — GT 音樂 ID（評估時用）
      - prompt_len     : int   — tokenized prompt 長度（含 [RANK]，不含 prefix）

    Plan B 關鍵設計：
      - prompt 末尾有 [RANK] token
      - prompt_len 是純文字長度（含 [RANK]），不加 MULTIMODAL_PREFIX_LEN
      - labels 遮蓋整個 prompt（含 [RANK]），只有 response 計算 generation loss
    """

    _FALLBACK_PROMPT   = "Can you recommend a music for my video?"
    _FALLBACK_RESPONSE = (
        "Based on your video content and preferences, I recommend this music "
        "as it best matches the mood and style of your video."
    )

    # Plan B prefix 新順序：[VIDEO:0][LTP:1][TEXT_CLIP:2][MUSIC:3]
    MULTIMODAL_PREFIX_LEN = 4

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: LlamaTokenizer,
        conv_map: Dict[str, Tuple[str, str]],
        song_bank: np.ndarray,
        song_ids: List[str],
        ltp_dict: Optional[Dict[str, np.ndarray]] = None,
        max_seq_len: int = 512,
        is_train: bool = True,
        ltp_dim: int = 256,
        seed: int = 42,
        mc_neg_cache_dir: Optional[str] = None,
        active_modalities: Optional[List[str]] = None,  # ★ 消融實驗
    ):
        self.pairs          = pairs
        self.tokenizer      = tokenizer
        self.conv_map       = conv_map
        self.song_bank      = song_bank
        self.song_ids       = song_ids
        self.N_songs        = len(song_ids)
        self.ltp_dict       = ltp_dict or {}
        self.max_seq_len    = max_seq_len
        self.is_train       = is_train
        self.ltp_dim        = ltp_dim
        self._eval_rng      = random.Random(seed)

        # ★ active_modalities 控制消融；訓練時永遠不注入 music_title
        if active_modalities is None:
            active_modalities = ["video", "ltp", "text", "music"]
        self.active_modalities = active_modalities
        self.is_text_active    = "text" in active_modalities
        self.prompt_tmpl       = build_prompt(active_modalities, music_title=None)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ★ 載入 mc Hard Negative 快取
        self.mc_neg_dict: Dict[str, np.ndarray] = {}
        if mc_neg_cache_dir:
            mc_npy = os.path.join(mc_neg_cache_dir, "mc_neg_bank.npy")
            mc_ids = os.path.join(mc_neg_cache_dir, "mc_neg_bank_ids.json")
            if os.path.exists(mc_npy) and os.path.exists(mc_ids):
                mc_arr = np.load(mc_npy)
                with open(mc_ids) as f:
                    mc_id_list = json.load(f)
                self.mc_neg_dict = {k: mc_arr[i] for i, k in enumerate(mc_id_list)}
                logger.info(f"[mc_neg_bank] 快取載入：{len(self.mc_neg_dict)} 筆")
            else:
                logger.warning(
                    f"[mc_neg_bank] 快取不存在（{mc_npy}），"
                    "將即時讀取 HDF5（速度慢）。請先執行 mc_neg_bank.py。"
                )

        n_conv = sum(1 for _, pk in pairs if pk[:11] in conv_map)
        n_ltp  = sum(1 for _, pk in pairs if pk[:11] in self.ltp_dict)
        logger.info(
            f"Dataset ({len(pairs)} pairs, train={is_train}) | "
            f"conv {n_conv/len(pairs)*100:.1f}% | "
            f"ltp {n_ltp/len(pairs)*100:.1f}%"
        )
        if n_ltp == 0:
            logger.warning("[Dataset] ltp_dict 為空，使用零向量（無個人化模式）")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h5_path, pair_key = self.pairs[idx]
        video_id = pair_key[:11]

        # ── 讀取 HDF5 特徵 ────────────────────────────────────────────────────
        with h5py.File(h5_path, "r") as f:
            grp   = f[f"pairs/{pair_key}"]
            n_seg = grp["video_features_all"].shape[0]   # 12

            if self.is_train:
                t          = random.randint(0, n_seg - 1)
                video_feat = torch.from_numpy(grp["video_features_all"][t].astype(np.float32))
            else:
                video_feat = torch.from_numpy(grp["video_features_all"][:].astype(np.float32).mean(0))

            # GT 音樂 mean pool → (768,)
            gt_vec = grp["target_music_all_cls"][:].astype(np.float32).mean(0)

            # 文字 CLS → (512,)
            text_feat = torch.from_numpy(grp["text_features"][0].astype(np.float32))

        # ── mc Hard Negative ──────────────────────────────────────────────────
        neg_vec = self.mc_neg_dict.get(pair_key, np.zeros(768, dtype=np.float32))

        # ── LTP ───────────────────────────────────────────────────────────────
        if video_id in self.ltp_dict:
            ltp_feat = torch.from_numpy(self.ltp_dict[video_id].astype(np.float32))
        else:
            ltp_feat = torch.zeros(self.ltp_dim)

        # ── Positive / Negative music ─────────────────────────────────────────
        pos_music_feat = torch.from_numpy(gt_vec.astype(np.float32))
        neg_music_feat = torch.from_numpy(neg_vec.astype(np.float32))

        # ── 對話文字 ─────────────────────────────────────────────────────────
        if video_id in self.conv_map:
            prompt_text, response_text = self.conv_map[video_id]
        else:
            prompt_text, response_text = self._FALLBACK_PROMPT, self._FALLBACK_RESPONSE

        # ── Tokenize ──────────────────────────────────────────────────────────
        # 訓練時 prompt_tmpl 不含歌名（music_title=None），保持與訓練一致
        if self.is_text_active:
            full_prompt = self.prompt_tmpl.format(user_text=prompt_text)
        else:
            full_prompt = self.prompt_tmpl
        full_text = full_prompt + response_text + self.tokenizer.eos_token

        # prompt_token_len：純文字序列長度，含 [RANK]，不含 multimodal prefix
        # ★ Plan B / bug fix：不加 MULTIMODAL_PREFIX_LEN
        #   因為 prefix 不在 input_ids 中，input_ids 只是文字部分
        prompt_token_len = len(self.tokenizer.encode(full_prompt, add_special_tokens=False))

        enc = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels mask：
        #   - prompt（含 [RANK]）→ -100（不計算 generation loss）
        #   - response（t4）→ 計算 generation loss
        #   - padding → -100
        # ★ Plan B / bug fix：只用 prompt_token_len，不加 MULTIMODAL_PREFIX_LEN
        labels = input_ids.clone()
        labels[:prompt_token_len] = -100
        labels[attention_mask == 0] = -100

        # prompt_len_for_generate：evaluate.py 和 model_service.py 用此截斷 prompt-only input
        # = 純文字 prompt 長度（含 [RANK]），不含 prefix
        prompt_len_for_generate = prompt_token_len

        return {
            # 查詢側（不隨 pos/neg 改變）
            "video_feat":     video_feat,      # (768,)
            "ltp_feat":       ltp_feat,        # (256,)
            "text_feat":      text_feat,       # (512,)
            "input_ids":      input_ids,       # (max_seq_len,)
            "attention_mask": attention_mask,  # (max_seq_len,)
            "labels":         labels,          # (max_seq_len,)，generation supervision
            "prompt_len":     torch.tensor(prompt_len_for_generate, dtype=torch.long),

            # 候選側（BPR 訓練用）
            "pos_music_feat": pos_music_feat,  # (768,)  target_music CLS mean pool
            "neg_music_feat": neg_music_feat,  # (768,)  mc CLS mean pool（Hard Negative）

            # 評估用 metadata
            "gt_music_id":    pair_key,        # str，23碼，500-pool 評估用
            "video_id":       video_id,        # str，排除 collision 用
        }


# ─────────────────────────────────────────────────────────────────────────────
# Custom collate_fn（module-level，Windows multiprocessing 需要）
# ─────────────────────────────────────────────────────────────────────────────

def pointwise_collate_fn(batch):
    str_keys = {"gt_music_id", "video_id"}
    tensor_batch = {}
    for key in batch[0].keys():
        if key in str_keys:
            tensor_batch[key] = [b[key] for b in batch]
        else:
            tensor_batch[key] = torch.stack([b[key] for b in batch])
    return tensor_batch


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader 建立函數
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    json_dir: str,
    tokenizer,
    train_config,
    model_config,
    ltp_dict: Optional[Dict[str, np.ndarray]] = None,
    pair_index_cache: Optional[str] = None,
    song_bank_cache: Optional[str] = None,
):
    pair_index = build_pair_index(data_dir, cache_path=pair_index_cache)
    conv_map   = load_conversation_map(json_dir)
    song_bank, song_ids = build_song_bank(pair_index, cache_path=song_bank_cache)

    tr_p, va_p, te_p = split_by_video_id(
        pair_index,
        train_config.train_ratio,
        train_config.val_ratio,
        train_config.test_ratio,
        train_config.split_seed,
    )

    kw = dict(
        tokenizer         = tokenizer,
        conv_map          = conv_map,
        song_bank         = song_bank,
        song_ids          = song_ids,
        ltp_dict          = ltp_dict,
        max_seq_len       = model_config.max_seq_len,
        ltp_dim           = model_config.ltp_dim,
        mc_neg_cache_dir  = os.path.dirname(train_config.pair_index_cache),
        active_modalities = getattr(model_config, "active_modalities", None),  # ★
    )

    train_loader = DataLoader(
        UnifiedMLLMDataset(tr_p, is_train=True,  **kw),
        batch_size=train_config.micro_batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True, persistent_workers=True,
        collate_fn=pointwise_collate_fn,
    )
    val_loader = DataLoader(
        UnifiedMLLMDataset(va_p, is_train=False, **kw),
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=pointwise_collate_fn,
    )
    test_loader = DataLoader(
        UnifiedMLLMDataset(te_p, is_train=False, **kw),
        batch_size=train_config.eval_batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        persistent_workers=True,
        collate_fn=pointwise_collate_fn,
    )
    return train_loader, val_loader, test_loader


def build_val_subset_loader(
    val_dataset,
    subset_size: int = 500,
    seed: int = 20260315,
    batch_size: int = 8,
    cache_path: Optional[str] = None,
) -> DataLoader:
    import json
    from torch.utils.data import Subset

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            indices = json.load(f)
        logger.info(f"[ValSubset] 快取載入：{len(indices)} 筆")
    else:
        rng = random.Random(seed)
        all_indices = list(range(len(val_dataset)))
        rng.shuffle(all_indices)
        indices = all_indices[:subset_size]
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(indices, f)
        logger.info(f"[ValSubset] 建立 {len(indices)} 筆 subset（seed={seed}）")

    subset = Subset(val_dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=pointwise_collate_fn,
    )