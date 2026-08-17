"""
model_service.py — FastAPI XAI inference backend（Plan B）

Plan B 主要變更（相比原版）：
  Bug 1 (Option B): _gate_value() 回傳 -1.0
  Bug 2: gt_pair_key 改為 23 碼
  Bug 3: user_text 優先於 conv_map
  Bug 4: dtype 統一 bfloat16
  Bug 5: O(1) song_id_to_idx
  Plan B: SPECIAL_TOKENS 加入 [RANK]，vocab = 32005
  Plan B: build_prompt() 末尾加 [RANK]

  用法：uvicorn model_service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import os, sys, json, random, logging, hashlib
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import h5py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR  = PROJECT_ROOT
H5_DIR      = PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3"
JSON_DIR    = PROJECT_ROOT / "data" / "musechat_json"
OUTPUT_DIR  = PROJECT_ROOT / "checkpoints" / "exp_01"
CACHE_DIR   = PROJECT_ROOT / "cache"
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"

LTP_H5 = PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors.h5"
LTP_MODE = "hybrid"

CKPT_DIR = OUTPUT_DIR / "best"

# ★ Plan B：5 個 special tokens（含 [RANK]），vocab = 32000 + 5 = 32005
SPECIAL_TOKENS = ["[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POOL_SIZE = 500
MICRO_BATCH = 32

# ── Prompt 模板（與 dataset.py build_prompt() 完全一致）──────────────────────
SYSTEM_PROMPT = (
    "You are an expert music recommendation assistant for short videos. "
    "Analyze the video content, user preferences, and candidate track to "
    "recommend the most suitable background music."
)

def build_prompt(user_text: str) -> str:
    """
    Plan B：在 [/INST] 後加入 [RANK] token，與 dataset.py 完全一致。
    """
    tmpl = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"Video: [VIDEO]\n"
        f"Candidate: [MUSIC]\n"
        f"User preference: [LTP]\n"
        f"Context: [TEXT_CLIP] {{user_text}}\n\n"
        f"Does this candidate best fit this video? [/INST] [RANK] "
    )
    return tmpl.format(user_text=user_text)

FALLBACK_PROMPT = "Please recommend suitable background music for this video."

# ── 全域物件 ──────────────────────────────────────────────────────────────────
model       = None
tokenizer   = None
ltp_dict:   dict[str, np.ndarray] = {}
song_bank_arr: np.ndarray = None
song_ids:   list[str] = []
song_id_to_idx: dict[str, int] = {}   # ★ Bug 5：O(1) 反查字典
mc_neg_dict: dict[str, np.ndarray] = {}
conv_map:   dict[str, tuple[str, str]] = {}
test_pair_keys: list[str] = []

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("model_service_planB")


# ── 載入函式 ──────────────────────────────────────────────────────────────────

def _load_model():
    global model, tokenizer
    from transformers import LlamaTokenizer, LlamaForCausalLM
    from peft import PeftModel

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    from models.projectors import MultimodalProjectors
    from models.unified_mllm import UnifiedMLLM
    from config import ModelConfig

    cfg = ModelConfig(
        llama_model_name      = LLAMA_MODEL,
        video_dim=768, music_dim=768, text_dim=512, ltp_dim=256,
        num_candidates=1, multimodal_prefix_len=4,
        music_token_offset=3,
        rank_special_token="[RANK]",
    )

    logger.info("[model] 載入 tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    # ★ Plan B：5 個 special tokens，vocab = 32005
    n = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    logger.info("[model] add_special_tokens: +%d → vocab=%d", n, len(tokenizer))

    logger.info("[model] 載入 LLaMA base...")
    base = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    base.resize_token_embeddings(len(tokenizer))
    logger.info("[model] resize_token_embeddings → %d", len(tokenizer))

    logger.info("[model] 載入 PEFT weights from %s...", CKPT_DIR)
    peft_llama = PeftModel.from_pretrained(
        base, str(CKPT_DIR), torch_dtype=torch.bfloat16
    )
    peft_llama.eval()

    logger.info("[model] 載入 projectors.pt...")
    projectors = MultimodalProjectors(
        video_dim=cfg.video_dim, music_dim=cfg.music_dim,
        text_dim=cfg.text_dim,  ltp_dim=cfg.ltp_dim,
        llama_hidden_dim=cfg.llama_hidden_dim,
        projector_hidden_dim=cfg.projector_hidden_dim,
        dropout=0.0,
    )
    projectors.load_state_dict(
        torch.load(str(CKPT_DIR / "projectors.pt"), map_location="cuda:0")
    )
    projectors = projectors.to(torch.bfloat16).cuda().eval()

    logger.info("[model] 載入 ranking_head.pt...")
    import torch.nn as nn
    ranking_head = nn.Sequential(
        nn.LayerNorm(cfg.llama_hidden_dim),
        nn.Linear(cfg.llama_hidden_dim, 256),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(256, 1),
    )
    ranking_head.load_state_dict(
        torch.load(str(CKPT_DIR / "ranking_head.pt"), map_location="cuda:0")
    )
    ranking_head = ranking_head.to(torch.bfloat16).cuda().eval()

    model = UnifiedMLLM(model_config=cfg, tokenizer=tokenizer)
    del model.llama
    torch.cuda.empty_cache()
    model.llama        = peft_llama
    model.projectors   = projectors
    model.ranking_head = ranking_head
    model.eval()
    logger.info("[model] 載入完成，device=%s", DEVICE)


def _load_ltp():
    global ltp_dict
    npy = CACHE_DIR / f"ltp_{LTP_MODE}.npy"
    ids = CACHE_DIR / f"ltp_{LTP_MODE}_ids.json"
    if npy.exists() and ids.exists():
        arr = np.load(str(npy))
        with open(str(ids)) as f:
            vid_list = json.load(f)
        ltp_dict = {v: arr[i] for i, v in enumerate(vid_list)}
        logger.info("[ltp] 快取載入：%d 筆", len(ltp_dict))
        return
    with h5py.File(str(LTP_H5), "r") as f:
        grp = f["preference_vectors"]
        for k in grp.keys():
            ltp_dict[k] = grp[k][:].astype(np.float32)
    logger.info("[ltp] 載入完成：%d 筆", len(ltp_dict))


def _load_song_bank():
    global song_bank_arr, song_ids, song_id_to_idx
    npy = CACHE_DIR / "song_bank.npy"
    ids = CACHE_DIR / "song_bank_ids.json"
    if npy.exists() and ids.exists():
        song_bank_arr = np.load(str(npy))
        with open(str(ids)) as f:
            song_ids = json.load(f)
        logger.info("[song_bank] 快取載入：%d 首", len(song_ids))
    else:
        feats = {}
        for h5_path in sorted(H5_DIR.glob("*.h5")):
            try:
                with h5py.File(str(h5_path), "r") as f:
                    if "pairs" not in f:
                        continue
                    for key in f["pairs"].keys():
                        try:
                            arr = f[f"pairs/{key}/target_music_all_cls"][:].astype(np.float32)
                            feats[key] = arr.mean(axis=0)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning("跳過 %s: %s", h5_path, e)
        song_ids      = sorted(feats.keys())
        song_bank_arr = np.stack([feats[k] for k in song_ids])
        np.save(str(npy), song_bank_arr)
        with open(str(ids), "w") as f:
            json.dump(song_ids, f)
        logger.info("[song_bank] 建立完成：%d 首", len(song_ids))
    # ★ Bug 5：O(1) 反查字典
    song_id_to_idx = {sid: i for i, sid in enumerate(song_ids)}


def _load_mc_neg():
    global mc_neg_dict
    npy = CACHE_DIR / "mc_neg_bank.npy"
    ids = CACHE_DIR / "mc_neg_bank_ids.json"
    if npy.exists() and ids.exists():
        arr = np.load(str(npy))
        with open(str(ids)) as f:
            pk_list = json.load(f)
        mc_neg_dict = {k: arr[i] for i, k in enumerate(pk_list)}
        logger.info("[mc_neg] 快取載入：%d 筆", len(mc_neg_dict))
    else:
        logger.warning("[mc_neg] 快取不存在，請先執行 mc_neg_bank.py")


def _load_conv_map():
    global conv_map
    import glob as _glob
    json_files = _glob.glob(str(JSON_DIR / "**" / "*.json"), recursive=True)
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            convs = data.get("conversations", [])
            if len(convs) < 4:
                continue
            t3 = convs[2].get("value", "").strip()
            t4 = convs[3].get("value", "").strip()
            if not t3 or not t4:
                continue
            video_id = Path(jf).parent.name
            conv_map[video_id] = (t3, t4)
        except Exception:
            pass
    logger.info("[conv_map] 載入：%d 筆", len(conv_map))


def _load_test_pairs():
    global test_pair_keys
    pair_index_cache = CACHE_DIR / "pair_index.json"
    if not pair_index_cache.exists():
        logger.warning("[test_pairs] pair_index.json 不存在")
        return
    with open(str(pair_index_cache)) as f:
        pair_index = json.load(f)
    from collections import defaultdict
    vid_to_pairs = defaultdict(list)
    for item in pair_index:
        vid_to_pairs[item[1][:11]].append(item[1])
    vids = sorted(vid_to_pairs.keys())
    random.Random(42).shuffle(vids)
    n, n_tr, n_va = len(vids), int(len(vids)*0.90), int(len(vids)*0.05)
    te_vids = vids[n_tr + n_va:]
    test_pair_keys = [pk for v in te_vids for pk in vid_to_pairs[v]]
    logger.info("[test_pairs] test pairs：%d 筆", len(test_pair_keys))


# ── 推論核心函式 ──────────────────────────────────────────────────────────────

def _read_features(pair_key: str) -> dict:
    for h5_path in sorted(H5_DIR.glob("*.h5")):
        try:
            with h5py.File(str(h5_path), "r") as f:
                if "pairs" not in f or pair_key not in f["pairs"]:
                    continue
                grp = f[f"pairs/{pair_key}"]
                video_feat    = grp["video_features_all"][:].astype(np.float32).mean(0)
                gt_music_feat = grp["target_music_all_cls"][:].astype(np.float32).mean(0)
                text_feat     = grp["text_features"][0].astype(np.float32)
                return {"video_feat": video_feat, "gt_music_feat": gt_music_feat, "text_feat": text_feat}
        except Exception as e:
            logger.warning("讀取 %s 發生錯誤: %s", h5_path, e)
    raise ValueError(f"pair_key={pair_key} 不存在")


def _build_query_tensors(pair_key: str, user_text: str) -> dict:
    """
    ★ Bug 3 修正：user_text 優先於 conv_map t3
    ★ Plan B：build_prompt() 末尾含 [RANK]
    """
    video_id = pair_key[:11]
    feats    = _read_features(pair_key)
    ltp_vec  = ltp_dict.get(video_id, np.zeros(256, dtype=np.float32))

    # ★ Bug 3：user_text 優先
    if user_text.strip():
        prompt_text = user_text.strip()
    elif video_id in conv_map:
        prompt_text, _ = conv_map[video_id]
    else:
        prompt_text = FALLBACK_PROMPT

    full_prompt = build_prompt(prompt_text)   # 末尾含 [RANK]

    enc = tokenizer(
        full_prompt, return_tensors="pt",
        truncation=True, max_length=256, padding=False,
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    return {
        "video_feat":     torch.from_numpy(feats["video_feat"]).unsqueeze(0).to(DEVICE),
        "ltp_feat":       torch.from_numpy(ltp_vec).unsqueeze(0).to(DEVICE),
        "text_feat":      torch.from_numpy(feats["text_feat"]).unsqueeze(0).to(DEVICE),
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "gt_music_feat":  feats["gt_music_feat"],
        "prompt_text":    prompt_text,
    }


@torch.no_grad()
def _score_music(query_t: dict, music_feat_np: np.ndarray) -> float:
    mf = torch.from_numpy(music_feat_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            video_feat       = query_t["video_feat"],
            music_candidates = mf,
            ltp_feat         = query_t["ltp_feat"],
            text_feat        = query_t["text_feat"],
            input_ids        = query_t["input_ids"],
            attention_mask   = query_t["attention_mask"],
            labels           = None,
            compute_gen_loss = False,
        )
    return float(out["ranking_score"].float().cpu().item())


@torch.no_grad()
def _pool_500(query_t: dict, pair_key: str, pool_seed_idx: int) -> dict:
    video_id    = pair_key[:11]
    gt_music_id = pair_key
    M = len(song_ids)
    excl = {i for i, sid in enumerate(song_ids) if sid[:11] == video_id}
    gt_global_idx = next((i for i, sid in enumerate(song_ids) if sid == gt_music_id), 0)
    candidates = [i for i in range(M) if i not in excl and i != gt_global_idx]
    rng = random.Random(20260315 + pool_seed_idx)
    negatives  = rng.sample(candidates, min(POOL_SIZE - 1, len(candidates)))
    pool_idx   = [gt_global_idx] + negatives
    pool_feats = torch.tensor(song_bank_arr[pool_idx], dtype=torch.float32, device=DEVICE)

    from evaluate import pointwise_pool_scoring
    scores = pointwise_pool_scoring(
        model=model,
        video_feat=query_t["video_feat"],
        ltp_feat=query_t["ltp_feat"],
        text_feat=query_t["text_feat"],
        input_ids=query_t["input_ids"],
        attention_mask=query_t["attention_mask"],
        pool_music_features=pool_feats,
        micro_batch_size=MICRO_BATCH,
        device=torch.device(DEVICE),
    )

    scores_np     = scores.float().numpy()
    sorted_idx    = np.argsort(scores_np)[::-1]
    gt_rank       = int(np.where(sorted_idx == 0)[0][0]) + 1
    sorted_scores = scores_np[sorted_idx]
    gap_to_2nd    = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    bpr_score     = float(scores_np[0])
    top5 = [(song_ids[pool_idx[sorted_idx[i]]], float(sorted_scores[i]))
            for i in range(min(5, len(sorted_idx)))]

    return {
        "bpr_score":       round(bpr_score, 4),
        "pool_rank":       gt_rank,
        "gap_to_2nd":      round(gap_to_2nd, 4),
        "all_pool_scores": [round(float(s), 4) for s in sorted_scores],
        "top5":            [(pk, round(sc, 4)) for pk, sc in top5],
    }


@torch.no_grad()
def _ablation(query_t: dict) -> dict:
    gt_feat = query_t["gt_music_feat"]
    base    = _score_music(query_t, gt_feat)

    def _zeroed(key: str, zero_shape: tuple) -> float:
        qt_mod = {k: v for k, v in query_t.items()}
        qt_mod[key] = torch.zeros(zero_shape, dtype=torch.bfloat16, device=DEVICE)
        return _score_music(qt_mod, gt_feat)

    return {
        "完整模型":   round(base, 4),
        "移除影片":   round(_zeroed("video_feat", (1, 768)), 4),
        "移除 P_ltp": round(_zeroed("ltp_feat",   (1, 256)), 4),
        "移除文字":   round(_zeroed("text_feat",  (1, 512)), 4),
        "移除音樂":   round(_score_music(query_t, np.zeros(768, dtype=np.float32)), 4),
    }


def _modality_contrib(ablation: dict) -> dict:
    base = ablation["完整模型"]
    drops = {
        "影片內容": max(0.0, base - ablation["移除影片"]),
        "長期偏好": max(0.0, base - ablation["移除 P_ltp"]),
        "文字描述": max(0.0, base - ablation["移除文字"]),
        "音樂特徵": max(0.0, base - ablation["移除音樂"]),
    }
    total = sum(drops.values()) or 1.0
    return {k: int(v / total * 100) for k, v in drops.items()}


def _gate_value() -> float:
    """★ Bug 1 Option B：架構尚未加入 gate_scalar，回傳 -1.0"""
    return -1.0


@torch.no_grad()
def _generate_explanation(query_t: dict) -> str:
    gt_feat_np = query_t["gt_music_feat"]
    mf = torch.from_numpy(gt_feat_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    try:
        generated_ids, _ = model.generate(
            video_feat=query_t["video_feat"], music_candidates=mf,
            ltp_feat=query_t["ltp_feat"], text_feat=query_t["text_feat"],
            input_ids=query_t["input_ids"], attention_mask=query_t["attention_mask"],
            max_new_tokens=120, do_sample=False,
        )
        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1].strip()
        return decoded if decoded else "(模型未生成說明)"
    except Exception as e:
        logger.error("[generate] 生成失敗：%s", e)
        video_id = query_t.get("video_id_str", "")
        if video_id and video_id in conv_map:
            _, t4 = conv_map[video_id]
            return f"[generate 發生例外，顯示標註 t4]\n{t4}"
        return f"[generate 失敗：{e}]"


def _ltp_branch(gate_val: float) -> dict:
    if gate_val < 0:
        return {"Hybrid（完整）": -1, "Explicit only": -1, "Implicit only": -1}
    pct = int(gate_val * 100)
    return {"Hybrid（完整）": pct, "Explicit only": int(pct*0.60), "Implicit only": int(pct*0.40)}


def _modality_signed(ablation: dict) -> tuple[list, dict]:
    mapping = [
        ("影片內容", "移除影片"), ("長期偏好", "移除 P_ltp"),
        ("文字描述", "移除文字"), ("音樂特徵", "移除音樂"),
    ]
    base = ablation["完整模型"]
    signed, pos = [], {}
    for name, key in mapping:
        delta = round(base - ablation[key], 4)
        signed.append({"name": name, "delta": delta,
                        "kind": "support" if delta > 0 else "interference" if delta < 0 else "neutral"})
        pos[name] = max(delta, 0.0)
    total = sum(pos.values()) or 1.0
    return signed, {k: int(v / total * 100) for k, v in pos.items()}


def _case_status(rank: int, gap: float) -> tuple[str, str]:
    if rank <= 10 and gap >= 0.08:
        return "success",    f"GT 排名第 {rank}（高信心）"
    if rank <= 50:
        return "borderline", f"GT 排名第 {rank}（邊界案例）"
    return   "failure",      f"GT 排名第 {rank}，屬失敗案例，應優先視為 failure analysis。"


@torch.no_grad()
def _contrastive(query_t: dict, top1_pair_key: str) -> dict:
    """
    ★ Bug 2：gt_pair_key 用 pair_key_full（23碼）
    ★ Bug 4：零向量 dtype 統一 bfloat16
    ★ Bug 5：O(1) 查找
    """
    gt_feat  = query_t["gt_music_feat"]
    top1_idx = song_id_to_idx.get(top1_pair_key, 0)
    top1_feat = song_bank_arr[top1_idx].astype(np.float32)

    def _abl_for(music_feat_np):
        base = _score_music(query_t, music_feat_np)
        def _z(key, shape):
            qt = {k: v for k, v in query_t.items()}
            qt[key] = torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)   # ★ Bug 4
            return _score_music(qt, music_feat_np)
        return {
            "完整模型":   round(base, 4),
            "移除影片":   round(_z("video_feat", (1, 768)), 4),
            "移除 P_ltp": round(_z("ltp_feat",   (1, 256)), 4),
            "移除文字":   round(_z("text_feat",  (1, 512)), 4),
            "移除音樂":   round(_score_music(query_t, np.zeros(768, dtype=np.float32)), 4),
        }

    gt_abl, top1_abl = _abl_for(gt_feat), _abl_for(top1_feat)
    _, gt_pct   = _modality_signed(gt_abl)
    _, top1_pct = _modality_signed(top1_abl)
    dom = lambda p: max(p.items(), key=lambda x: x[1])[0] if p else "未知"
    gt_dom, t1_dom = dom(gt_pct), dom(top1_pct)
    return {
        "gt_pair_key":         query_t.get("pair_key_full", query_t.get("video_id_str", "")),   # ★ Bug 2
        "model_top1_pair_key": top1_pair_key,
        "gt_contrib_pct":      gt_pct,
        "top1_contrib_pct":    top1_pct,
        "gt_ablation":         gt_abl,
        "top1_ablation":       top1_abl,
        "summary": (
            f"GT 主導模態：「{gt_dom}」，Top1 主導模態：「{t1_dom}」。"
            + ("兩者一致。" if gt_dom == t1_dom else "兩者不同，可能是長期偏好對齊不足。")
        ),
    }


@torch.no_grad()
def _counterfactuals(query_t: dict, base_score: float, base_rank: int) -> list:
    gt_feat = query_t["gt_music_feat"]
    rows = [{"label": "原始設定", "score": round(base_score, 4), "delta_score": 0.0, "note": "完整輸入"}]
    variants = [
        ("移除文字輸入",         {"text_feat":  torch.zeros((1,512), dtype=torch.float32, device=DEVICE)}),
        ("移除長期偏好（P_ltp）", {"ltp_feat":   torch.zeros((1,256), dtype=torch.float32, device=DEVICE)}),
        ("移除影片特徵",          {"video_feat": torch.zeros((1,768), dtype=torch.float32, device=DEVICE)}),
        ("僅保留影片＋音樂",
         {"ltp_feat": torch.zeros((1,256), dtype=torch.float32, device=DEVICE),
          "text_feat": torch.zeros((1,512), dtype=torch.float32, device=DEVICE)}),
    ]
    for label, mods in variants:
        qt = {k: v for k, v in query_t.items()}
        qt.update(mods)
        score = _score_music(qt, gt_feat)
        delta = round(score - base_score, 4)
        rows.append({"label": label, "score": round(score,4), "delta_score": delta,
                     "note": "↑ GT 分數上升" if delta > 0 else "↓ GT 分數下降" if delta < 0 else "→ 無變化"})
    return rows


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Music XAI Backend (Unified MLLM Pointwise v2 Plan B)")


@app.on_event("startup")
async def startup_event():
    _load_ltp()
    _load_song_bank()
    _load_mc_neg()
    _load_conv_map()
    _load_test_pairs()
    _load_model()


class InferRequest(BaseModel):
    pair_key:      str
    user_text:     str = ""
    pool_seed_idx: int = 0


class InferResponse(BaseModel):
    pair_key:         str
    video_id:         str
    gt_music_id:      str
    prompt_text:      str
    prompt_source:    str
    bpr_score:        float
    pool_rank:        int
    gap_to_2nd:       float
    gate_value:       float
    gate_display:     str
    case_status:      str
    case_summary:     str
    nl_explanation:   str
    modality_contrib: dict
    contrib_signed:   list
    ablation:         dict
    ltp_branch:       dict
    contrastive:      dict
    counterfactuals:  list
    top5:             list
    all_pool_scores:  list
    warnings:         list[str]
    todos:            list[str]


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if model is None:
        raise HTTPException(503, "模型尚未載入完成")
    if not song_ids:
        raise HTTPException(503, "song_bank 尚未載入")

    try:
        query_t = _build_query_tensors(req.pair_key, req.user_text)
    except ValueError as e:
        raise HTTPException(404, str(e))

    query_t["video_id_str"]  = req.pair_key[:11]
    query_t["pair_key_full"] = req.pair_key   # ★ Bug 2

    pool_result = _pool_500(query_t, req.pair_key, req.pool_seed_idx)
    ablation_res = _ablation(query_t)
    contrib_pct  = _modality_contrib(ablation_res)
    contrib_signed, _ = _modality_signed(ablation_res)
    gate_val     = _gate_value()
    gate_display = "N/A（架構尚未加入 gate_scalar）" if gate_val < 0 else f"{gate_val:.3f}"
    ltp_b        = _ltp_branch(gate_val)
    nl_exp       = _generate_explanation(query_t)
    case_status, case_summary = _case_status(pool_result["pool_rank"], pool_result["gap_to_2nd"])
    top1_pk      = pool_result["top5"][0][0] if pool_result["top5"] else req.pair_key
    contrastive  = _contrastive(query_t, top1_pk)
    cfs          = _counterfactuals(query_t, pool_result["bpr_score"], pool_result["pool_rank"])

    # ★ Bug 3：prompt_source 追蹤與 _build_query_tensors 邏輯一致
    if req.user_text.strip():
        prompt_source = "user_text"
    elif req.pair_key[:11] in conv_map:
        prompt_source = "dataset_t3"
    else:
        prompt_source = "fallback"

    warnings, todos = [], []
    todos.append(
        "ℹ️ ltp_gate_scalar 尚未加入 unified_mllm.py 架構，Gate 值顯示 N/A。"
    )
    todos.append(
        "ℹ️ P_ltp 分支（Explicit/Implicit）目前為近似值，精確值需 A_explicit/A_implicit checkpoint。"
    )
    if pool_result["gap_to_2nd"] < 0.02:
        warnings.append("⚠️ 第 1 名與第 2 名分差 < 0.02，推薦不穩定")
    if pool_result["pool_rank"] > 100:
        warnings.append(f"⚠️ Pool Rank={pool_result['pool_rank']}，建議視為 failure analysis")
    if case_status == "failure":
        warnings.append("⚠️ 此為失敗案例，XAI 解釋應側重分析推薦失敗原因。")

    return InferResponse(
        pair_key=req.pair_key, video_id=req.pair_key[:11], gt_music_id=req.pair_key,
        prompt_text=query_t["prompt_text"], prompt_source=prompt_source,
        bpr_score=pool_result["bpr_score"], pool_rank=pool_result["pool_rank"],
        gap_to_2nd=pool_result["gap_to_2nd"], gate_value=gate_val, gate_display=gate_display,
        case_status=case_status, case_summary=case_summary, nl_explanation=nl_exp,
        modality_contrib=contrib_pct, contrib_signed=contrib_signed, ablation=ablation_res,
        ltp_branch=ltp_b, contrastive=contrastive, counterfactuals=cfs,
        top5=pool_result["top5"], all_pool_scores=pool_result["all_pool_scores"],
        warnings=warnings, todos=todos,
    )


@app.get("/health")
async def health():
    return {
        "status":       "ok" if model is not None else "loading",
        "device":       DEVICE,
        "songs_loaded": len(song_ids),
        "users_loaded": len(ltp_dict),
        "test_pairs":   len(test_pair_keys),
        "conv_map":     len(conv_map),
        "mc_neg":       len(mc_neg_dict),
    }


@app.get("/sample_ids")
async def sample_ids(n: int = 50):
    keys = test_pair_keys[:n] if test_pair_keys else []
    return {"pair_keys": keys, "total": len(test_pair_keys)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model_service:app", host="0.0.0.0", port=8000, reload=False)
