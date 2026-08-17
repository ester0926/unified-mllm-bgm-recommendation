# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
generate_recommendation.py — 批次生成音樂推薦理由 + 500-pool 排名

用途：
  修改 TARGET_VIDEO_IDS 清單，一次處理一筆或多筆 video_id，
  對每筆同時計算 500-pool 排名指標（R@1/R@5/R@10/MR）
  與生成指標（BERTScore F1/InfoLM L2/FR/AB），
  結果分別存為個別 JSON 和彙整 summary.csv。

執行：
  python generate_recommendation.py

設定（直接修改程式底部的 TARGET_VIDEO_IDS 等常數）：
  TARGET_VIDEO_IDS = ["VRzzLEMF8LU", "CyHc9CDPfl4", ...]
  INJECT_TITLE     = True   # 開啟標題注入（對齊 MuseChat 推論）
  DO_RANKING       = True   # 計算 500-pool 排名
  AUTO_SAVE_JSON   = True   # 儲存個別 JSON
  SAVE_SUMMARY     = True   # 儲存彙整 summary.csv

輸出資料夾：
  SAVE_DIR/gen_{video_id}.json  — 每筆詳細結果
  SAVE_DIR/summary.csv          — 所有筆數的論文指標彙整
"""

import os
import sys
import json
import glob
import logging
import numpy as np
import torch
from pathlib import Path

# ── 路徑設定（與訓練/評估腳本一致）─────────────────────────────────────────
SCRIPT_DIR  = str(PROJECT_ROOT)
H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
CKPT_DIR    = str(PROJECT_ROOT / "checkpoints" / "exp_01" / "best")
CACHE_DIR   = str(PROJECT_ROOT / "cache")
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"
LTP_H5      = str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors.h5")

SPECIAL_TOKENS = ["[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"]

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(
    level=logging.WARNING,  # 只顯示警告，避免干擾輸出
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 資料載入
# ══════════════════════════════════════════════════════════════════════════════

def load_ltp_dict(h5_path, cache_path=None):
    import h5py
    if cache_path:
        npy = cache_path + "_hybrid.npy"
        ids = cache_path + "_hybrid_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids) as f:
                vid_list = json.load(f)
            return {v: arr[i] for i, v in enumerate(vid_list)}
    out = {}
    with h5py.File(h5_path, "r") as f:
        grp = f["preference_vectors"]
        for k in grp.keys():
            out[k] = grp[k][:].astype(np.float32)
    return out


def find_h5_for_video(video_id, h5_dir, cache_path=None):
    """
    在 HDF5 索引中找到 video_id 對應的 (h5_path, pair_key)。
    """
    import h5py
    # 先從快取讀取 pair_index
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            pair_index = [tuple(x) for x in json.load(f)]
    else:
        pair_index = []
        for h5_file in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
            try:
                with h5py.File(h5_file, "r") as f:
                    if "pairs" in f:
                        for key in f["pairs"].keys():
                            pair_index.append((h5_file, key))
            except Exception:
                pass

    # 找 video_id（pair_key 前 11 碼）
    matches = [(h5, pk) for h5, pk in pair_index if pk[:11] == video_id]
    return matches


def load_sample_features(h5_path, pair_key, ltp_dict):
    """
    從 HDF5 讀取一筆樣本的特徵。
    """
    import h5py
    video_id = pair_key[:11]

    with h5py.File(h5_path, "r") as f:
        grp = f[f"pairs/{pair_key}"]
        # 推論時使用 mean pool（與 is_train=False 一致）
        video_feat = torch.from_numpy(
            grp["video_features_all"][:].astype(np.float32).mean(0)
        )
        gt_vec = torch.from_numpy(
            grp["target_music_all_cls"][:].astype(np.float32).mean(0)
        )
        text_feat = torch.from_numpy(
            grp["text_features"][0].astype(np.float32)
        )

    if video_id in ltp_dict:
        ltp_feat = torch.from_numpy(ltp_dict[video_id].astype(np.float32))
    else:
        ltp_feat = torch.zeros(256)
        print(f"  ⚠️  video_id={video_id} 沒有 P_ltp 資料，使用零向量")

    return video_feat, ltp_feat, text_feat, gt_vec


def load_gt_reference(video_id, json_dir):
    """
    從 JSON 對話資料集讀取 GT reference（t3 prompt 和 t4 response）。
    """
    pattern = os.path.join(json_dir, "**", video_id, "*.json")
    matches = glob.glob(pattern, recursive=True)

    # 也試著直接找
    if not matches:
        pattern2 = os.path.join(json_dir, video_id, "*.json")
        matches = glob.glob(pattern2)

    for jf in matches:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            convs = data.get("conversations", [])
            if len(convs) >= 4:
                t3 = convs[2].get("value", "").strip()
                t4 = convs[3].get("value", "").strip()
                if t4:
                    return t3, t4
        except Exception:
            pass

    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# 模型載入
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_dir):
    """
    載入 best checkpoint，返回 (model, tokenizer)。
    """
    import torch.nn as nn
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from peft import PeftModel
    from models.projectors import MultimodalProjectors
    from models.unified_mllm import UnifiedMLLM
    from config import ModelConfig

    print("📦 載入模型中（約需 15 秒）...")

    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    base = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    base.resize_token_embeddings(len(tokenizer))
    peft_llama = PeftModel.from_pretrained(base, ckpt_dir, torch_dtype=torch.bfloat16)
    peft_llama.eval()
    if hasattr(peft_llama, "gradient_checkpointing_disable"):
        peft_llama.gradient_checkpointing_disable()

    model_cfg = ModelConfig(
        llama_model_name=LLAMA_MODEL,
        video_dim=768, music_dim=768, text_dim=512, ltp_dim=256,
        num_candidates=1, multimodal_prefix_len=4,
        music_token_offset=3, rank_special_token="[RANK]",
    )

    projectors = MultimodalProjectors(
        video_dim=768, music_dim=768, text_dim=512, ltp_dim=256,
        llama_hidden_dim=4096, projector_hidden_dim=2048, dropout=0.0,
    )
    projectors.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "projectors.pt"), map_location="cuda:0")
    )
    projectors = projectors.to(torch.bfloat16).cuda().eval()

    ranking_head = nn.Sequential(
        nn.LayerNorm(4096), nn.Linear(4096, 256),
        nn.GELU(), nn.Dropout(0.0), nn.Linear(256, 1),
    )
    ranking_head.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "ranking_head.pt"), map_location="cuda:0")
    )
    ranking_head = ranking_head.to(torch.bfloat16).cuda().eval()

    try:
        model = UnifiedMLLM(model_config=model_cfg, tokenizer=tokenizer, _load_llama=False)
    except TypeError:
        model = UnifiedMLLM(model_config=model_cfg, tokenizer=tokenizer)
        del model.llama
        torch.cuda.empty_cache()

    model.llama        = peft_llama
    model.projectors   = projectors
    model.ranking_head = ranking_head
    model.eval()

    print("✅ 模型載入完成")
    return model, tokenizer, model_cfg


# ══════════════════════════════════════════════════════════════════════════════
# Song Bank 載入（排序評估用）
# ══════════════════════════════════════════════════════════════════════════════

def load_song_bank(h5_dir, cache_path=None):
    """
    載入全局 song bank（所有 GT 音樂特徵）。
    cache_path 對應 train.py 的 song_bank_cache。
    回傳 (song_bank_np: ndarray shape=(N,768), song_ids: List[str])
    """
    import h5py
    if cache_path:
        npy = cache_path + ".npy"
        ids = cache_path + "_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids) as f:
                sid = json.load(f)
            print(f"📂 Song bank 快取載入：{len(sid):,} 首")
            return arr, sid

    print("📂 Song bank 從 HDF5 建立中（首次執行較慢）...")
    import h5py
    song_bank = {}
    for h5_file in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
        try:
            with h5py.File(h5_file, "r") as f:
                if "pairs" not in f:
                    continue
                for key in f["pairs"].keys():
                    grp = f[f"pairs/{key}"]
                    gt_vec = grp["target_music_all_cls"][:].astype(np.float32).mean(0)
                    song_bank[key] = gt_vec
        except Exception:
            pass

    song_ids = list(song_bank.keys())
    arr = np.stack([song_bank[k] for k in song_ids])

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path + ".npy", arr)
        with open(cache_path + "_ids.json", "w") as f:
            json.dump(song_ids, f)

    print(f"✅ Song bank 建立完成：{len(song_ids):,} 首")
    return arr, song_ids


# ══════════════════════════════════════════════════════════════════════════════
# 500-pool 排序評估（單筆）
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_rank_in_pool(
    video_feat, ltp_feat, text_feat, gt_vec, gt_pair_key,
    song_bank_np, song_ids,
    model, tokenizer,
    pool_size=500, batch_size=128, seed=42,
):
    """
    對單筆樣本建立 500-pool，計算 GT 音樂的排名。

    ★ 優化版（比舊版快 ~4-8x）：
      舊版問題：每個 batch 都重算 video/ltp/text embedding 和 token embedding，
              即使這些對 500 首候選完全相同，浪費了大量 GPU 計算。
      優化：
        (A) 在 loop 前預先計算固定 embedding（只算一次）
            - video_emb：video_proj(video_feat)
            - ltp_emb：  ltp_proj(ltp_feat)
            - text_emb： text_proj(text_feat)
            - token_embs：LLaMA embed_tokens(input_ids)
        (B) loop 中只計算 music_emb（每批候選才會不同）
        (C) 直接呼叫 model.llama（跳過 projectors 的重複調度開銷）
        (D) batch_size 32→128（RTX 5090 有餘裕，減少 loop 次數 4x）

    改動前：16 次完整 forward（含 projectors）
    改動後：1 次固定 embedding + 4 次 LLaMA（僅 music emb 不同）
    """
    device = torch.device("cuda")
    bf16   = torch.bfloat16

    # ── 1. 找 GT 在 song_bank 的索引 ────────────────────────────────────────
    gt_idx_in_bank = None
    for i, sid in enumerate(song_ids):
        if sid == gt_pair_key:
            gt_idx_in_bank = i
            break

    # ── 2. 建立 500-pool ─────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    all_indices = list(range(len(song_ids)))
    if gt_idx_in_bank is not None:
        all_indices = [i for i in all_indices if i != gt_idx_in_bank]

    n_neg = min(pool_size - 1, len(all_indices))
    neg_indices = rng.choice(all_indices, size=n_neg, replace=False).tolist()

    gt_insert_pos = int(rng.integers(0, len(neg_indices) + 1))
    if gt_idx_in_bank is not None:
        pool_indices = (neg_indices[:gt_insert_pos]
                        + [gt_idx_in_bank]
                        + neg_indices[gt_insert_pos:])
        pool_feats   = [song_bank_np[i] for i in pool_indices[:pool_size]]
        gt_pool_idx  = gt_insert_pos
    else:
        pool_feats   = [song_bank_np[i] for i in neg_indices[:pool_size - 1]]
        pool_feats.insert(gt_insert_pos, gt_vec.cpu().numpy())
        gt_pool_idx  = gt_insert_pos

    pool_tensor    = torch.tensor(np.stack(pool_feats), dtype=bf16).to(device)
    actual_pool_sz = pool_tensor.shape[0]

    # ── 3. ranking 用 prompt（無歌名，與訓練一致）────────────────────────────
    from dataset import build_prompt
    prompt_tmpl = build_prompt(active_modalities=None, music_title=None)
    full_prompt = prompt_tmpl.format(
        user_text="Can you recommend a music for my video?"
    )
    enc = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    base_input_ids = enc["input_ids"].to(device)    # (1, seq_len)
    base_attn_mask = enc["attention_mask"].to(device)

    # ── 4A. ★ 預先計算固定 embedding（只算一次）────────────────────────────
    with torch.no_grad():
        proj = model.projectors
        type_emb = proj.modality_type_emb

        # 各模態 projection（固定，與候選無關）
        # 順序：video(ID=0) → ltp(ID=2) → text(ID=3)
        v_emb = (proj.video_proj(video_feat.unsqueeze(0).to(device, dtype=bf16))
                 + type_emb(torch.tensor([0], device=device))).unsqueeze(1)  # (1,1,4096)
        l_emb = (proj.ltp_proj(ltp_feat.unsqueeze(0).to(device, dtype=bf16))
                 + type_emb(torch.tensor([2], device=device))).unsqueeze(1)  # (1,1,4096)
        t_emb = (proj.text_proj(text_feat.unsqueeze(0).to(device, dtype=bf16))
                 + type_emb(torch.tensor([3], device=device))).unsqueeze(1)  # (1,1,4096)
        # music type embedding（ID=1）供 loop 使用
        m_type = type_emb(torch.tensor([1], device=device))                  # (1,4096)

        # Token embedding（prompt 文字，固定）
        embed_fn   = model.llama.get_input_embeddings()
        token_embs = embed_fn(base_input_ids).to(dtype=bf16)                 # (1,seq_len,4096)

        # [RANK] 在 input_ids 中的位置 → 換算到完整序列中的位置
        prefix_len     = 4   # video, ltp, text, music
        rank_pos_text  = (base_input_ids[0] == model.rank_token_id).nonzero(as_tuple=True)[0][0].item()
        rank_pos_full  = prefix_len + rank_pos_text   # (1,) scalar

        # 固定 attention mask（prefix 4 個 + prompt seq_len）
        prefix_mask   = torch.ones(1, prefix_len, dtype=torch.long, device=device)
        full_attn_1   = torch.cat([prefix_mask, base_attn_mask], dim=1)   # (1, 4+seq_len)

    # ── 4B. ★ Loop：只計算 music embedding，LLaMA 每批只跑一次 ────────────
    scores = []
    with torch.no_grad():
        for start in range(0, actual_pool_sz, batch_size):
            end = min(start + batch_size, actual_pool_sz)
            B   = end - start

            # 只有 music embedding 是 batch 變化的
            m_emb = (proj.music_proj(pool_tensor[start:end])
                     + m_type).unsqueeze(1)              # (B, 1, 4096)

            # 拼接 prefix：[VIDEO, LTP, TEXT_CLIP, MUSIC]
            prefix = torch.cat([
                v_emb.expand(B, -1, -1),
                l_emb.expand(B, -1, -1),
                t_emb.expand(B, -1, -1),
                m_emb,
            ], dim=1)                                    # (B, 4, 4096)

            # 拼接 token embeddings
            inputs_embeds = torch.cat(
                [prefix, token_embs.expand(B, -1, -1)], dim=1
            )                                            # (B, 4+seq_len, 4096)

            # 直接呼叫 LLaMA（跳過 projectors 的重複調度）
            llama_out = model.llama(
                inputs_embeds   = inputs_embeds,
                attention_mask  = full_attn_1.expand(B, -1),
                output_hidden_states = True,
                return_dict     = True,
            )

            # 提取 [RANK] 位置的 hidden state
            last_hidden  = llama_out.hidden_states[-1]           # (B, 4+seq_len, 4096)
            rank_hidden  = last_hidden[:, rank_pos_full, :]      # (B, 4096)
            batch_scores = model.ranking_head(rank_hidden).squeeze(-1)  # (B,)
            scores.extend(batch_scores.float().cpu().tolist())

    # ── 5. 排序，找 GT 名次 ──────────────────────────────────────────────────
    scores_arr   = np.array(scores)
    sorted_idx   = np.argsort(-scores_arr)
    gt_rank      = int(np.where(sorted_idx == gt_pool_idx)[0][0]) + 1
    gt_score     = float(scores_arr[gt_pool_idx])
    top1_pool_idx = int(sorted_idx[0])
    top1_score   = float(scores_arr[top1_pool_idx])

    # ── 6. 回傳 top-1 歌曲資訊供生成使用 ────────────────────────────────────
    # top1_is_gt=True  → 模型確實選了 GT，用 GT 生成
    # top1_is_gt=False → 模型選了另一首歌，生成應使用那首歌的特徵
    top1_is_gt      = (top1_pool_idx == gt_pool_idx)
    top1_music_feat = torch.tensor(pool_feats[top1_pool_idx], dtype=torch.float32)

    # 找 top-1 的 pair_key（若在 song_bank 中可對應到）
    top1_pair_key = None
    if gt_idx_in_bank is not None:
        # 重建 pool_indices 來找 top-1 的 song_bank index
        pool_bank_indices = (neg_indices[:gt_insert_pos]
                             + [gt_idx_in_bank]
                             + neg_indices[gt_insert_pos:])[:pool_size]
        top1_bank_idx = pool_bank_indices[top1_pool_idx]
        top1_pair_key = song_ids[top1_bank_idx] if top1_bank_idx < len(song_ids) else None
    elif not top1_is_gt:
        # GT 不在 bank 時，非 GT 的 top1 對應 neg_indices
        adj_idx = top1_pool_idx if top1_pool_idx < gt_insert_pos else top1_pool_idx - 1
        if 0 <= adj_idx < len(neg_indices):
            top1_pair_key = song_ids[neg_indices[adj_idx]]

    return {
        # ── 論文表格指標（R@K、MR）────────────────────────────────────────
        "R@1":         1 if gt_rank == 1  else 0,
        "R@5":         1 if gt_rank <= 5  else 0,
        "R@10":        1 if gt_rank <= 10 else 0,
        "median_rank": gt_rank,   # 單樣本的 median rank 就是 rank 本身
        "pool_size":   actual_pool_sz,
        # ── 輔助診斷用（不進論文表格）─────────────────────────────────────
        "score_gap":   round(top1_score - gt_score, 4),  # GT vs Top-1 分數差
        # ── 供 generate_for_video 使用（不進 JSON 輸出）──────────────────
        "top1_is_gt":      top1_is_gt,
        "top1_music_feat": top1_music_feat,   # tensor，JSON 序列化時會被過濾
        "top1_pair_key":   top1_pair_key,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 生成推薦理由
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_for_video(video_id, model, tokenizer, model_cfg,
                        ltp_dict, h5_dir, json_dir, cache_path=None,
                        max_new_tokens=200, inject_title=True,
                        song_bank_np=None, song_ids=None,
                        pool_size=500, rank_batch_size=32):
    """
    給定 video_id，同時計算：
      (A) 500-pool 排名（GT 音樂在模型排序中的位置）
      (B) 生成推薦理由（含/不含歌名注入）

    這兩項分開計算，清楚展示：
      「模型排序正確（rank=1）但不知道歌名（生成幻覺）」的現象。

    inject_title：
      True  → 從 t4 解析歌名注入，生成可包含真實歌名（對齊 MuseChat 推論）
      False → 純聲學生成，歌名為幻覺（等同 Vicuna w/ Music baseline）

    song_bank_np / song_ids：
      傳入時才計算排名，None 則跳過排名評估（節省時間）。
    """
    device = torch.device("cuda")
    bf16   = torch.bfloat16

    # ── 1. 找 HDF5 pair ──────────────────────────────────────────────────────
    matches = find_h5_for_video(video_id, h5_dir, cache_path)
    if not matches:
        return {"error": f"找不到 video_id={video_id} 對應的 HDF5 資料"}
    h5_path, pair_key = matches[0]

    # ── 2. 載入特徵 ───────────────────────────────────────────────────────────
    video_feat, ltp_feat, text_feat, gt_vec = load_sample_features(
        h5_path, pair_key, ltp_dict
    )

    # ── 3. GT reference & 解析歌名 ───────────────────────────────────────────
    t3_prompt, t4_reference = load_gt_reference(video_id, json_dir)

    music_title = music_artist = None
    title_source = "none"
    if inject_title and t4_reference:
        from dataset import extract_music_title
        music_title, music_artist = extract_music_title(t4_reference)
        title_source = "parsed_from_t4" if music_title else "parse_failed"

    # ── 4A. 計算 500-pool 排名（ranking prompt：無歌名，與訓練一致）──────────
    ranking_info = None
    if song_bank_np is not None and song_ids is not None:
        print("  📊 計算 500-pool 排名中...")
        ranking_info = compute_rank_in_pool(
            video_feat=video_feat,
            ltp_feat=ltp_feat,
            text_feat=text_feat,
            gt_vec=gt_vec,
            gt_pair_key=pair_key,
            song_bank_np=song_bank_np,
            song_ids=song_ids,
            model=model,
            tokenizer=tokenizer,
            pool_size=pool_size,
            batch_size=rank_batch_size,
        )

    # ── 4B. 決定生成要用哪首歌的特徵和歌名 ────────────────────────────────────
    # ★ 修正：應用 top-1 的音樂特徵生成，而非永遠用 GT
    #
    # rank=1（top-1 = GT）：
    #   → 生成特徵 = GT features，歌名 = GT 歌名（從 t4 解析）
    #   → 行為與 MuseChat 推論一致
    #
    # rank≥2（top-1 ≠ GT）：
    #   → 生成特徵 = top-1 的 music features（才是模型真正推薦的那首）
    #   → 歌名 = 嘗試從 top-1 的 video JSON 解析（若找不到則不注入）
    #   → 這才是誠實的輸出：模型推薦了「錯誤」的歌，顯示它的 t4 描述

    gen_music_feat = gt_vec    # 預設用 GT
    gen_pair_key   = pair_key  # 預設 GT pair_key
    top1_is_gt     = True

    if ranking_info is not None and not ranking_info.get("top1_is_gt", True):
        # 模型選了非 GT 的歌 → 用 top-1 的特徵生成
        top1_is_gt     = False
        gen_music_feat = ranking_info["top1_music_feat"]    # (768,)
        gen_pair_key   = ranking_info.get("top1_pair_key")

        # 嘗試查 top-1 的歌名（從它的 video_id 對應 JSON）
        if inject_title and gen_pair_key:
            top1_video_id = str(gen_pair_key)[:11]
            top1_t3, top1_t4 = load_gt_reference(top1_video_id, json_dir)
            if top1_t4:
                from dataset import extract_music_title
                music_title, music_artist = extract_music_title(top1_t4)
                title_source = "parsed_from_top1_t4" if music_title else "top1_parse_failed"
            else:
                music_title = music_artist = None
                title_source = "top1_no_json"
        else:
            music_title = music_artist = None
            title_source = "top1_no_title"

    # ── 4C. 建立生成 prompt ────────────────────────────────────────────────────
    from dataset import build_prompt
    prompt_tmpl = build_prompt(
        active_modalities=None,
        music_title=music_title,
        music_artist=music_artist,
    )
    user_text   = t3_prompt if t3_prompt else "Can you recommend a music for my video?"
    full_prompt = prompt_tmpl.format(user_text=user_text)

    enc = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids  = enc["input_ids"].to(device)
    attn_mask  = enc["attention_mask"].to(device)

    rank_token_id = model.rank_token_id
    if (input_ids == rank_token_id).sum().item() == 0:
        return {"error": "[RANK] token not found in prompt"}

    # ── 5. 生成推薦理由（使用 top-1 的音樂特徵）─────────────────────────────
    print("  ✍️  生成推薦理由中...")
    gen_music = gen_music_feat.unsqueeze(0).unsqueeze(0).to(device, dtype=bf16)
    generated_ids, _ = model.generate(
        video_feat       = video_feat.unsqueeze(0).to(device, dtype=bf16),
        music_candidates = gen_music,   # ★ top-1 的特徵（rank=1 時與 GT 相同）
        ltp_feat         = ltp_feat.unsqueeze(0).to(device, dtype=bf16),
        text_feat        = text_feat.unsqueeze(0).to(device, dtype=bf16),
        input_ids        = input_ids,
        attention_mask   = attn_mask,
        max_new_tokens   = max_new_tokens,
        do_sample        = False,
    )

    new_tokens = generated_ids[0]
    decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    decoded = decoded.replace("[RANK]", "").strip()

    # ── 6. 計算生成指標（BERTScore + InfoLM）────────────────────────────────
    # 針對單筆計算：首次會載入 DeBERTa / BERT（約 15-25 秒），後續快取後 ~5 秒
    gen_metrics = {}
    if t4_reference and decoded:
        print("  📐 計算生成指標（BERTScore + InfoLM）...")
        try:
            from bert_score import score as _bs
            P, R, F1 = _bs(
                [decoded], [t4_reference],
                model_type="microsoft/deberta-xlarge-mnli",
                lang="en", verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            gen_metrics["bertscore_f1"]        = round(float(F1.mean()), 4)
            gen_metrics["bertscore_precision"]  = round(float(P.mean()),  4)
            gen_metrics["bertscore_recall"]     = round(float(R.mean()),  4)
        except Exception as e:
            gen_metrics["bertscore_f1"] = None
            gen_metrics["bertscore_error"] = str(e)

        try:
            from torchmetrics.functional.text.infolm import infolm as tm_infolm
            _kw = dict(model_name_or_path="bert-base-uncased", idf=False,
                       device="cuda" if torch.cuda.is_available() else "cpu",
                       verbose=False, batch_size=1)
            gen_metrics["infolm_l2_distance"]    = round(float(
                tm_infolm([decoded], [t4_reference], information_measure="l2_distance",        **_kw)), 4)
            gen_metrics["infolm_fisher_rao"]     = round(float(
                tm_infolm([decoded], [t4_reference], information_measure="fisher_rao_distance", **_kw)), 4)
            try:
                gen_metrics["infolm_ab_divergence"] = round(float(
                    tm_infolm([decoded], [t4_reference], information_measure="ab_divergence",
                              alpha=0.5, beta=0.5, **_kw)), 4)
            except Exception:
                gen_metrics["infolm_ab_divergence"] = None
        except Exception as e:
            gen_metrics["infolm_l2_distance"]    = None
            gen_metrics["infolm_fisher_rao"]     = None
            gen_metrics["infolm_ab_divergence"]  = None
            gen_metrics["infolm_error"]          = str(e)
    else:
        gen_metrics = {
            "bertscore_f1": None, "bertscore_precision": None,
            "bertscore_recall": None, "infolm_ab_divergence": None,
            "infolm_l2_distance": None, "infolm_fisher_rao": None,
        }

    # ── 整理 ranking_metrics（移除 tensor，只保留論文指標）──────────────────
    ranking_metrics = None
    if ranking_info is not None:
        ranking_metrics = {k: v for k, v in ranking_info.items()
                           if k not in ("top1_music_feat", "top1_is_gt", "top1_pair_key")}

    return {
        "video_id":          video_id,
        "pair_key":          pair_key,
        "user_prompt":       user_text,
        "hypothesis":        decoded,
        "reference":         t4_reference or "(找不到 GT reference)",
        "has_ltp":           video_id in ltp_dict,
        "inject_title":      inject_title,
        "music_title":       music_title,
        "music_artist":      music_artist,
        "title_source":      title_source,
        "top1_is_gt":        top1_is_gt,
        "gen_pair_key":      str(gen_pair_key) if gen_pair_key else None,
        # ── 論文表格指標（兩個 dict，清楚對應論文 Table）─────────────────
        "ranking_metrics":    ranking_metrics,   # R@1/R@5/R@10/MR/pool_size/score_gap
        "generation_metrics": gen_metrics,       # BERTScore F1/P/R + InfoLM L2/FR/AB
        # 向後相容（舊程式讀 result["ranking"] 仍可用）
        "ranking":            ranking_info,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 顯示對比結果
# ══════════════════════════════════════════════════════════════════════════════

def display_result(result):
    if "error" in result:
        print(f"\n❌ 錯誤：{result['error']}")
        return

    sep = "─" * 68
    print(f"\n{'='*68}")
    print(f"  VIDEO ID  : {result['video_id']}")
    print(f"  PAIR KEY  : {result['pair_key']}")
    print(f"  P_ltp     : {'✅ 有個人化偏好' if result['has_ltp'] else '⚠️  無 P_ltp（零向量）'}")

    # ── 排名資訊（從 ranking_metrics 讀取）──────────────────────────────────
    # ★ Fix 3：使用更新後的 key 名稱（R@1/R@5/R@10/median_rank/score_gap）
    rm = result.get("ranking_metrics")
    if rm:
        rank    = rm.get("median_rank", "?")
        pool_sz = rm.get("pool_size",   "?")
        gap     = rm.get("score_gap",   0.0)
        r1      = rm.get("R@1",  0)
        r5      = rm.get("R@5",  0)
        r10     = rm.get("R@10", 0)

        if rank == 1:
            rank_label = f"🥇 第 {rank} 名（R@1 命中）"
        elif isinstance(rank, int) and rank <= 5:
            rank_label = f"🥈 第 {rank} 名（R@5 命中）"
        elif isinstance(rank, int) and rank <= 10:
            rank_label = f"🥉 第 {rank} 名（R@10 命中）"
        else:
            rank_label = f"  第 {rank} 名"

        print(f"\n  ┌─── 500-pool 排名結果（{pool_sz} 首候選）─────────────────────┐")
        print(f"  │  GT 名次  : {rank_label:<40} │")
        print(f"  │  score差距: {gap:>+8.4f}（GT 與 Top-1 的差距）              │")
        print(f"  │  R@1 : {'✅' if r1 else '❌'}   R@5 : {'✅' if r5 else '❌'}   R@10 : {'✅' if r10 else '❌'}                   │")
        print(f"  └───────────────────────────────────────────────────────────┘")

        top1_is_gt = result.get("top1_is_gt", True)
        if rank == 1:
            print(f"\n  💡 模型正確選到 GT（rank=1）→ 下方生成是 GT 音樂的推薦理由。")
        elif isinstance(rank, int) and rank <= 5:
            print(f"\n  ⚠️  GT 排名第 {rank}（非 Top-1），模型實際推薦了另一首歌。")
            if not top1_is_gt:
                gen_pk = result.get("gen_pair_key", "")
                top1_vid = str(gen_pk)[:11] if gen_pk else "未知"
                print(f"     → 下方生成使用 Top-1 的音樂特徵（video_id={top1_vid}）")
                print(f"     → score 差距僅 {gap:+.4f}，模型排序能力仍接近正確。")
        else:
            print(f"\n  ⚠️  此樣本排名 {rank}/{pool_sz}，屬於排序也不準確的案例。")
            if not top1_is_gt:
                print(f"     → 下方生成使用 Top-1 的音樂特徵（錯誤推薦的歌）。")

    # ── 標題注入狀態 ──────────────────────────────────────────────────────────
    top1_is_gt = result.get("top1_is_gt", True)
    src_label  = "GT t4" if top1_is_gt else "Top-1 video JSON"
    if result.get("inject_title"):
        if result.get("music_title"):
            print(f"\n  標題注入  : ✅ '{result['music_title']}'"
                  + (f" by {result['music_artist']}" if result.get("music_artist") else "")
                  + f"（來源：{src_label}）")
        else:
            ts = result.get("title_source", "")
            reason = {
                "parse_failed":      "t4 格式不符，無法解析",
                "top1_parse_failed": "Top-1 t4 格式不符",
                "top1_no_json":      "Top-1 無對應 JSON",
                "top1_no_title":     "未啟用 Top-1 標題查找",
            }.get(ts, "解析失敗")
            print(f"\n  標題注入  : ⚠️  {reason}，以純聲學特徵生成")
    else:
        print(f"\n  標題注入  : ❌ 關閉（Vicuna w/ Music baseline）")

    print(f"\n{'='*68}")
    print(f"\n📝 用戶推薦請求（t3）：")
    print(f"  {result['user_prompt']}")

    def wrap_print(text, indent="  "):
        words = text.split()
        line = []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 64:
                print(indent + " ".join(line[:-1]))
                line = [w]
        if line:
            print(indent + " ".join(line))

    print(f"\n{sep}")
    print("  🤖 本研究模型生成的推薦理由")
    print(sep)
    wrap_print(result["hypothesis"])

    print(f"\n{sep}")
    print("  📚 GT Reference（t4，對話資料集標注）")
    print(sep)
    wrap_print(result["reference"])
    print(f"\n{'='*68}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

import os
import json
import numpy as np

# ==========================================
# ⚙️ 程式內變數設定 (手動修改這裡即可)
# ==========================================

# ★ 支援單筆或批次：填一個或多個 11 碼 video_id
TARGET_VIDEO_IDS = [
    # "tu-n5fUfjc4", # 危險案例（P_ltp Δ < -2.0 且 rank > 50）；rank= 209  Δ_ltp=-8.0928  Δ_music= 4.6562
    # "9VqBjf-GiLk", # 完美多模態案例（所有 Δ > 0）；rank=   1  Δ_ltp=4.2246  Δ_video=9.4062
    "RvYL2iai7ks"  # 危險案例（P_ltp Δ < -2.0 且 rank > 50）；rank= 121  Δ_ltp=-17.9375  Δ_music= 0.6250
]

INJECT_TITLE    = True          # 是否開啟標題注入 (False 則等同 MuseChat baseline)
DO_RANKING      = True         # 是否計算 500-pool 排名 (設為 False 可大幅加速)
POOL_SIZE       = 500           # 排名用的候選池大小
MAX_GEN_TOKENS  = 200           # 生成的最大 Token 數
AUTO_SAVE_JSON  = True          # 是否自動儲存生成結果到 JSON 檔案
SAVE_DIR        = "句子生成評估" # 儲存資料夾名稱（每筆存一個 JSON）
SAVE_SUMMARY    = True          # 是否額外儲存彙整的 summary CSV（批次模式有用）
# ==========================================


def _json_safe(obj):
    """遞迴清理不能 JSON 序列化的物件（tensor / numpy / python bool）"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()
                if k != "top1_music_feat"}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _print_metrics_summary(result):
    """印出單筆的論文指標摘要行"""
    rm = result.get("ranking_metrics") or {}
    gm = result.get("generation_metrics") or {}
    vid = result.get("video_id", "?")
    r1  = rm.get("R@1",  "?")
    r5  = rm.get("R@5",  "?")
    r10 = rm.get("R@10", "?")
    mr  = rm.get("median_rank", "?")
    f1  = gm.get("bertscore_f1")
    l2  = gm.get("infolm_l2_distance")
    fr  = gm.get("infolm_fisher_rao")
    ab  = gm.get("infolm_ab_divergence")
    print(f"\n{'─'*60}")
    print(f"  📋 {vid} 論文指標摘要")
    print(f"{'─'*60}")
    print(f"  排序  R@1={r1}  R@5={r5}  R@10={r10}  MR={mr}")
    print(f"  生成  BERTScore F1={f1}  L2={l2}  FR={fr}  AB={ab}")
    print(f"{'─'*60}")


def _save_summary_csv(all_results, save_dir):
    """把所有樣本的指標彙整成一個 summary CSV"""
    import csv
    csv_path = os.path.join(save_dir, "summary.csv")
    fields = [
        "video_id", "pair_key", "music_title", "music_artist",
        "top1_is_gt",
        "R@1", "R@5", "R@10", "median_rank", "pool_size", "score_gap",
        "bertscore_f1", "bertscore_precision", "bertscore_recall",
        "infolm_ab_divergence", "infolm_l2_distance", "infolm_fisher_rao",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = {
                "video_id":    r.get("video_id"),
                "pair_key":    r.get("pair_key"),
                "music_title": r.get("music_title"),
                "music_artist":r.get("music_artist"),
                "top1_is_gt":  r.get("top1_is_gt"),
            }
            rm = r.get("ranking_metrics") or {}
            gm = r.get("generation_metrics") or {}
            row.update({k: rm.get(k) for k in
                        ["R@1","R@5","R@10","median_rank","pool_size","score_gap"]})
            row.update({k: gm.get(k) for k in
                        ["bertscore_f1","bertscore_precision","bertscore_recall",
                         "infolm_ab_divergence","infolm_l2_distance","infolm_fisher_rao"]})
            writer.writerow(row)
    print(f"\n📊 彙整 CSV 已儲存至: {csv_path}")


def main():
    # ── 驗證 video_id 格式 ───────────────────────────────────────────────────
    video_ids = [v.strip() for v in TARGET_VIDEO_IDS if v.strip()]
    invalid   = [v for v in video_ids if len(v) != 11]
    if invalid:
        print(f"⚠️  以下 video_id 格式錯誤（應為 11 碼）：{invalid}")
        video_ids = [v for v in video_ids if len(v) == 11]
    if not video_ids:
        print("❌ 沒有有效的 video_id，請修改 TARGET_VIDEO_IDS。")
        return

    # ── 1. 模型載入（只做一次）──────────────────────────────────────────────
    model, tokenizer, model_cfg = load_model(CKPT_DIR)

    # ── 2. 資料載入 ─────────────────────────────────────────────────────────
    print("📂 載入 P_ltp 快取...")
    ltp_dict   = load_ltp_dict(LTP_H5, cache_path=os.path.join(CACHE_DIR, "ltp"))
    cache_path = os.path.join(CACHE_DIR, "pair_index.json")

    song_bank_np = song_ids = None
    if DO_RANKING:
        print("📂 載入 Song Bank（排名計算用）...")
        song_bank_np, song_ids = load_song_bank(
            H5_DIR, cache_path=os.path.join(CACHE_DIR, "song_bank")
        )

    if AUTO_SAVE_JSON:
        os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 3. 批次處理 ─────────────────────────────────────────────────────────
    n = len(video_ids)
    print("\n" + "="*68)
    print(f" 🚀 Unified MLLM 推薦理由生成器 + 500-pool 排名")
    print(f" 🎯 批次筆數   : {n} 筆")
    print(f" 🏷️  標題注入   : {'✅ 開啟' if INJECT_TITLE else '❌ 關閉'}")
    print(f" 📊 排名計算   : {'✅ 開啟' if DO_RANKING else '❌ 關閉'}")
    print(f" 💾 輸出資料夾 : {SAVE_DIR}")
    print("="*68)

    all_results  = []
    success_cnt  = 0
    failed_ids   = []

    for i, vid in enumerate(video_ids):
        print(f"\n[{i+1}/{n}] ⏳ 處理 {vid} ...")
        try:
            result = generate_for_video(
                vid, model, tokenizer, model_cfg,
                ltp_dict, H5_DIR, JSON_DIR,
                cache_path=cache_path,
                max_new_tokens=MAX_GEN_TOKENS,
                inject_title=INJECT_TITLE,
                song_bank_np=song_bank_np,
                song_ids=song_ids,
                pool_size=POOL_SIZE,
            )

            if "error" in result:
                print(f"  ❌ {result['error']}")
                failed_ids.append(vid)
                continue

            display_result(result)
            _print_metrics_summary(result)
            all_results.append(result)
            success_cnt += 1

            # 個別 JSON
            if AUTO_SAVE_JSON:
                out_path = os.path.join(SAVE_DIR, f"gen_{vid}.json")
                with open(out_path, "w", encoding="utf-8") as fp:
                    json.dump(_json_safe(result), fp, indent=2, ensure_ascii=False)
                print(f"  💾 已儲存：{out_path}")

        except Exception as e:
            import traceback
            print(f"  ❌ 未預期錯誤：{e}")
            print(traceback.format_exc())
            failed_ids.append(vid)

    # ── 4. 批次彙整 ─────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print(f"  ✅ 完成 {success_cnt}/{n} 筆，失敗 {len(failed_ids)} 筆")
    if failed_ids:
        print(f"  ❌ 失敗清單：{failed_ids}")

    if success_cnt > 1:
        # 多筆時印彙整表格
        print(f"\n{'─'*77}")
        # 🟢 修正 1：增加 'AB' 標題
        print(f"  {'video_id':<14}{'R@1':>5}{'R@5':>5}{'R@10':>6}{'MR':>5}{'BertF1':>9}{'L2':>8}{'FR':>8}{'AB':>8}")
        print(f"{'─'*77}")
        for r in all_results:
            rm = r.get("ranking_metrics") or {}
            gm = r.get("generation_metrics") or {}
            # 🟢 修正 2：增加 ab_divergence 的取值與格式化輸出
            print(f"  {r['video_id']:<14}"
                  f"{str(rm.get('R@1','?')):>5}"
                  f"{str(rm.get('R@5','?')):>5}"
                  f"{str(rm.get('R@10','?')):>6}"
                  f"{str(rm.get('median_rank','?')):>5}"
                  f"{str(gm.get('bertscore_f1','?')):>9}"
                  f"{str(gm.get('infolm_l2_distance','?')):>8}"
                  f"{str(gm.get('infolm_fisher_rao','?')):>8}"
                  f"{str(gm.get('infolm_ab_divergence','?')):>8}")
        print(f"{'─'*77}")

    # summary CSV（批次時特別有用）
    if SAVE_SUMMARY and all_results and AUTO_SAVE_JSON:
        _save_summary_csv(all_results, SAVE_DIR)


if __name__ == "__main__":
    main()