"""
用途：執行或整理消融實驗診斷結果。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
import sys
import json
import time
import glob
import logging
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import h5py
from scipy import stats as scipy_stats

# ── 路徑設定（與 run_train.py 一致）──────────────────────────────────────────
SCRIPT_DIR  = str(PROJECT_ROOT)
H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
OUTPUT_DIR  = str(PROJECT_ROOT / "checkpoints" / "exp_01")
CACHE_DIR   = str(PROJECT_ROOT / "cache")
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"
LTP_H5      = str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors.h5")
LTP_MODE    = "hybrid"
CKPT_DIR    = os.path.join(OUTPUT_DIR, "best")

SPECIAL_TOKENS = ["[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POOL_SIZE = 500
MICRO_BATCH = 32
POOL_SEED_IDX = 0

# None = 跑完整 test split
N_SAMPLES = None

# 續跑與中途儲存
SAVE_EVERY = 20
RESUME = True

OUT_JSON         = os.path.join(OUTPUT_DIR, "ablation_all_results.json")
OUT_TXT          = os.path.join(OUTPUT_DIR, "ablation_all_summary.txt")
OUT_PARTIAL_JSON = os.path.join(OUTPUT_DIR, "ablation_all_partial.json")

PAIR_H5_INDEX_CACHE = os.path.join(CACHE_DIR, "pair_to_h5_index.json")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ablation_all_fast")


# ══════════════════════════════════════════════════════════════════════════════
# 載入模型
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    from transformers import LlamaTokenizer, LlamaForCausalLM
    from peft import PeftModel
    from models.projectors import MultimodalProjectors
    from models.unified_mllm import UnifiedMLLM
    from config import ModelConfig
    import torch.nn as nn

    cfg = ModelConfig(
        llama_model_name=LLAMA_MODEL,
        video_dim=768,
        music_dim=768,
        text_dim=512,
        ltp_dim=256,
        num_candidates=1,
        multimodal_prefix_len=4,
        music_token_offset=3,
        rank_special_token="[RANK]",
    )

    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    logger.info("vocab = %d", len(tokenizer))

    base = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    base.resize_token_embeddings(len(tokenizer))

    peft_llama = PeftModel.from_pretrained(base, CKPT_DIR, torch_dtype=torch.bfloat16)
    peft_llama.eval()

    projectors = MultimodalProjectors(
        video_dim=cfg.video_dim,
        music_dim=cfg.music_dim,
        text_dim=cfg.text_dim,
        ltp_dim=cfg.ltp_dim,
        llama_hidden_dim=cfg.llama_hidden_dim,
        projector_hidden_dim=cfg.projector_hidden_dim,
        dropout=0.0,
    )
    projectors.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "projectors.pt"), map_location="cuda:0")
    )
    projectors = projectors.to(torch.bfloat16).cuda().eval()

    ranking_head = nn.Sequential(
        nn.LayerNorm(cfg.llama_hidden_dim),
        nn.Linear(cfg.llama_hidden_dim, 256),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(256, 1),
    )
    ranking_head.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "ranking_head.pt"), map_location="cuda:0")
    )
    ranking_head = ranking_head.to(torch.bfloat16).cuda().eval()

    model = UnifiedMLLM(model_config=cfg, tokenizer=tokenizer)
    del model.llama
    torch.cuda.empty_cache()
    model.llama = peft_llama
    model.projectors = projectors
    model.ranking_head = ranking_head
    model.eval()
    return model, tokenizer, cfg


# ══════════════════════════════════════════════════════════════════════════════
# 資料載入
# ══════════════════════════════════════════════════════════════════════════════

def load_resources():
    """載入 LTP、song_bank、conv_map、test_pairs"""
    # LTP
    ltp_dict = {}
    npy = os.path.join(CACHE_DIR, f"ltp_{LTP_MODE}.npy")
    ids = os.path.join(CACHE_DIR, f"ltp_{LTP_MODE}_ids.json")
    if os.path.exists(npy) and os.path.exists(ids):
        arr = np.load(npy)
        with open(ids, "r", encoding="utf-8") as f:
            vid_list = json.load(f)
        ltp_dict = {v: arr[i] for i, v in enumerate(vid_list)}
        logger.info("[ltp] 快取載入：%d 筆", len(ltp_dict))
    else:
        with h5py.File(LTP_H5, "r") as f:
            grp = f["preference_vectors"]
            for k in grp.keys():
                ltp_dict[k] = grp[k][:].astype(np.float32)
        logger.info("[ltp] HDF5 載入：%d 筆", len(ltp_dict))

    # 音樂特徵庫
    sb_npy = os.path.join(CACHE_DIR, "song_bank.npy")
    sb_ids = os.path.join(CACHE_DIR, "song_bank_ids.json")
    song_bank_arr = np.load(sb_npy)
    with open(sb_ids, "r", encoding="utf-8") as f:
        song_ids = json.load(f)
    song_id_to_idx = {sid: i for i, sid in enumerate(song_ids)}
    logger.info("[song_bank] %d 首", len(song_ids))

    # conv_map
    conv_map = {}
    for jf in glob.glob(os.path.join(JSON_DIR, "**", "*.json"), recursive=True):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            convs = data.get("conversations", [])
            if len(convs) < 4:
                continue
            t3 = convs[2].get("value", "").strip()
            t4 = convs[3].get("value", "").strip()
            if t3 and t4:
                conv_map[Path(jf).parent.name] = (t3, t4)
        except Exception:
            pass
    logger.info("[conv_map] %d 筆", len(conv_map))

    # 測試 pair 清單
    pair_index_cache = os.path.join(CACHE_DIR, "pair_index.json")
    with open(pair_index_cache, "r", encoding="utf-8") as f:
        pair_index = json.load(f)

    vid_to_pairs = defaultdict(list)
    for item in pair_index:
        vid_to_pairs[item[1][:11]].append(item[1])

    vids = sorted(vid_to_pairs.keys())
    random.Random(42).shuffle(vids)

    n = len(vids)
    n_tr = int(n * 0.90)
    n_va = int(n * 0.05)
    te_vids = vids[n_tr + n_va:]
    test_pair_keys = [pk for v in te_vids for pk in vid_to_pairs[v]]
    logger.info("[test_pairs] %d 筆", len(test_pair_keys))

    return ltp_dict, song_bank_arr, song_ids, song_id_to_idx, conv_map, test_pair_keys


# ══════════════════════════════════════════════════════════════════════════════
# H5 加速：pair_key → h5_path 索引 + file handle cache
# ══════════════════════════════════════════════════════════════════════════════

def build_or_load_pair_h5_index():
    """
    建立 pair_key -> h5_path 映射。
    第一次會掃全部 H5 並寫入 cache，之後直接讀 cache。
    """
    if os.path.exists(PAIR_H5_INDEX_CACHE):
        with open(PAIR_H5_INDEX_CACHE, "r", encoding="utf-8") as f:
            pair_to_h5 = json.load(f)
        logger.info("[pair_h5_index] 快取載入：%d 筆", len(pair_to_h5))
        return pair_to_h5

    logger.info("[pair_h5_index] 開始建立索引，這一步只需第一次做...")
    pair_to_h5 = {}
    h5_paths = sorted(Path(H5_DIR).glob("*.h5"))

    for i, h5_path in enumerate(h5_paths, 1):
        logger.info("  [%d/%d] 掃描 %s", i, len(h5_paths), h5_path.name)
        try:
            with h5py.File(str(h5_path), "r") as f:
                if "pairs" not in f:
                    continue
                for pair_key in f["pairs"].keys():
                    pair_to_h5[pair_key] = str(h5_path)
        except Exception as e:
            logger.warning("  [略過] %s: %s", h5_path.name, e)

    with open(PAIR_H5_INDEX_CACHE, "w", encoding="utf-8") as f:
        json.dump(pair_to_h5, f, ensure_ascii=False)

    logger.info("[pair_h5_index] 建立完成：%d 筆", len(pair_to_h5))
    return pair_to_h5


class H5Manager:
    def __init__(self):
        self.handles = {}

    def get_handle(self, path: str):
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        return self.handles[path]

    def close_all(self):
        for f in self.handles.values():
            try:
                f.close()
            except Exception:
                pass
        self.handles.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 推論核心
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert music recommendation assistant for short videos. "
    "Analyze the video content, user preferences, and candidate track to "
    "recommend the most suitable background music."
)

def build_prompt(user_text: str) -> str:
    tmpl = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"Video: [VIDEO]\n"
        f"Candidate: [MUSIC]\n"
        f"User preference: [LTP]\n"
        f"Context: [TEXT_CLIP] {{user_text}}\n\n"
        f"Does this candidate best fit this video? [/INST] [RANK] "
    )
    return tmpl.format(user_text=user_text)


def read_features_fast(pair_key: str, pair_to_h5: dict, h5_manager: H5Manager) -> dict:
    """
    使用預建索引直接找到 pair_key 所在的 H5，再透過 handle cache 讀取。
    """
    h5_path = pair_to_h5.get(pair_key)
    if h5_path is None:
        raise ValueError(f"pair_key={pair_key} not found in pair_to_h5 index")

    f = h5_manager.get_handle(h5_path)
    if "pairs" not in f or pair_key not in f["pairs"]:
        raise ValueError(f"pair_key={pair_key} not found in indexed H5: {h5_path}")

    grp = f[f"pairs/{pair_key}"]
    return {
        "video_feat":    grp["video_features_all"][:].astype(np.float32).mean(0),
        "gt_music_feat": grp["target_music_all_cls"][:].astype(np.float32).mean(0),
        "text_feat":     grp["text_features"][0].astype(np.float32),
    }


def build_query(pair_key, ltp_dict, conv_map, tokenizer, pair_to_h5, h5_manager):
    video_id = pair_key[:11]
    feats = read_features_fast(pair_key, pair_to_h5, h5_manager)
    ltp_vec = ltp_dict.get(video_id, np.zeros(256, dtype=np.float32))

    prompt_text = conv_map[video_id][0] if video_id in conv_map else "Please recommend background music."
    full_prompt = build_prompt(prompt_text)

    enc = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False
    )

    return {
        "video_feat":     torch.from_numpy(feats["video_feat"]).unsqueeze(0).to(DEVICE),
        "ltp_feat":       torch.from_numpy(ltp_vec).unsqueeze(0).to(DEVICE),
        "text_feat":      torch.from_numpy(feats["text_feat"]).unsqueeze(0).to(DEVICE),
        "input_ids":      enc["input_ids"].to(DEVICE),
        "attention_mask": enc["attention_mask"].to(DEVICE),
        "gt_music_feat":  feats["gt_music_feat"],
    }


@torch.no_grad()
def score_music(model, query_t, music_feat_np):
    mf = torch.from_numpy(music_feat_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            video_feat=query_t["video_feat"],
            music_candidates=mf,
            ltp_feat=query_t["ltp_feat"],
            text_feat=query_t["text_feat"],
            input_ids=query_t["input_ids"],
            attention_mask=query_t["attention_mask"],
            labels=None,
            compute_gen_loss=False,
        )
    return float(out["ranking_score"].float().cpu().item())


@torch.no_grad()
def pool_500(model, query_t, pair_key, song_bank_arr, song_ids, song_id_to_idx):
    from evaluate import pointwise_pool_scoring

    video_id = pair_key[:11]
    M = len(song_ids)

    excl = {i for i, sid in enumerate(song_ids) if sid[:11] == video_id}
    gt_global_idx = song_id_to_idx.get(pair_key, None)
    if gt_global_idx is None:
        raise ValueError(f"pair_key={pair_key} not found in song_id_to_idx")

    candidates = [i for i in range(M) if i not in excl and i != gt_global_idx]
    rng = random.Random(20260315 + POOL_SEED_IDX)
    negatives = rng.sample(candidates, min(POOL_SIZE - 1, len(candidates)))

    pool_idx = [gt_global_idx] + negatives
    pool_feats = torch.tensor(song_bank_arr[pool_idx], dtype=torch.float32, device=DEVICE)

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

    scores_np = scores.float().cpu().numpy()
    sorted_idx = np.argsort(scores_np)[::-1]
    gt_rank = int(np.where(sorted_idx == 0)[0][0]) + 1
    bpr_score = float(scores_np[0])
    gap = float(scores_np[sorted_idx[0]] - scores_np[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    return gt_rank, bpr_score, gap


@torch.no_grad()
def ablation_one(model, query_t):
    gt_feat = query_t["gt_music_feat"]
    base = score_music(model, query_t, gt_feat)

    def zeroed(key, shape):
        qt = dict(query_t)
        ref = query_t[key]
        qt[key] = torch.zeros(shape, dtype=ref.dtype, device=ref.device)
        return score_music(model, qt, gt_feat)

    s_no_video = zeroed("video_feat", (1, 768))
    s_no_ltp   = zeroed("ltp_feat",   (1, 256))
    s_no_text  = zeroed("text_feat",  (1, 512))
    s_no_music = score_music(model, query_t, np.zeros(768, dtype=np.float32))

    return {
        "base":         round(base, 4),
        "delta_video":  round(base - s_no_video, 4),
        "delta_ltp":    round(base - s_no_ltp,   4),
        "delta_text":   round(base - s_no_text,  4),
        "delta_music":  round(base - s_no_music, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 統計分析
# ══════════════════════════════════════════════════════════════════════════════

def analyze(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {
            "n_samples": 0,
            "modality_stats": {},
            "ltp_group_rank": {},
            "ltp_rank_corr": {},
            "danger_cases": [],
            "n_perfect": 0,
            "perfect_pair_keys": [],
        }

    modalities = ["ltp", "video", "text", "music"]

    stats = {}
    for mod in modalities:
        deltas = np.array([r[f"delta_{mod}"] for r in records], dtype=np.float32)
        positive_rate = float((deltas > 0.01).mean())
        negative_rate = float((deltas < -0.01).mean())
        stats[mod] = {
            "mean":          round(float(deltas.mean()), 4),
            "std":           round(float(deltas.std()),  4),
            "median":        round(float(np.median(deltas)), 4),
            "min":           round(float(deltas.min()),  4),
            "max":           round(float(deltas.max()),  4),
            "positive_rate": round(positive_rate, 4),
            "negative_rate": round(negative_rate, 4),
            "neutral_rate":  round(1 - positive_rate - negative_rate, 4),
        }

    ltp_pos = [r for r in records if r["delta_ltp"] > 0.01]
    ltp_neg = [r for r in records if r["delta_ltp"] < -0.01]
    ltp_neu = [r for r in records if abs(r["delta_ltp"]) <= 0.01]

    def group_stat(group):
        if not group:
            return {"n": 0, "mean_rank": None, "mean_delta_ltp": None}
        ranks = [r["rank"] for r in group]
        deltas = [r["delta_ltp"] for r in group]
        return {
            "n": len(group),
            "mean_rank":      round(float(np.mean(ranks)), 2),
            "median_rank":    round(float(np.median(ranks)), 2),
            "mean_delta_ltp": round(float(np.mean(deltas)), 4),
            "std_delta_ltp":  round(float(np.std(deltas)), 4),
        }

    ltp_groups = {
        "positive": group_stat(ltp_pos),
        "negative": group_stat(ltp_neg),
        "neutral":  group_stat(ltp_neu),
    }

    delta_ltp_arr = np.array([r["delta_ltp"] for r in records], dtype=np.float32)
    rank_arr = np.array([r["rank"] for r in records], dtype=np.float32)
    spearman_r, spearman_p = scipy_stats.spearmanr(delta_ltp_arr, rank_arr)
    ltp_rank_corr = {
        "spearman_r": round(float(spearman_r), 4),
        "p_value":    round(float(spearman_p), 4),
        "note": "負值代表 P_ltp 支持 GT 時 rank 確實較好",
    }

    danger_cases = [
        {
            "pair_key":    r["pair_key"],
            "rank":        r["rank"],
            "delta_ltp":   r["delta_ltp"],
            "delta_video": r["delta_video"],
            "delta_music": r["delta_music"],
        }
        for r in records
        if r["delta_ltp"] < -2.0 and r["rank"] > 50
    ]

    perfect_cases = [
        r for r in records
        if r["delta_video"] > 0
        and r["delta_ltp"] > 0
        and r["delta_text"] > 0
        and r["delta_music"] > 0
    ]

    return {
        "n_samples": n,
        "modality_stats": stats,
        "ltp_group_rank": ltp_groups,
        "ltp_rank_corr": ltp_rank_corr,
        "danger_cases": danger_cases,
        "n_perfect": len(perfect_cases),
        "perfect_pair_keys": [r["pair_key"] for r in perfect_cases],
    }


def print_summary(analysis: dict, records: list) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  Full-Test Modality Ablation Summary — Unified MLLM Pointwise v3")
    lines.append("=" * 70)
    lines.append(f"  Total samples: {analysis['n_samples']}")
    lines.append("")

    if analysis["n_samples"] == 0:
        lines.append("  無可分析樣本。")
        lines.append("=" * 70)
        return "\n".join(lines)

    lines.append("── 各模態 Δ score 統計（正值=支持GT，負值=干擾GT）──")
    lines.append(
        f"  {'模態':<12} {'mean':>8} {'std':>8} {'median':>8} "
        f"{'正向%':>7} {'負向%':>7} {'中性%':>7}"
    )
    lines.append("  " + "-" * 65)
    for mod in ["music", "video", "text", "ltp"]:
        s = analysis["modality_stats"][mod]
        lines.append(
            f"  {mod:<12} {s['mean']:>8.4f} {s['std']:>8.4f} {s['median']:>8.4f} "
            f"  {s['positive_rate']*100:>5.1f}%  {s['negative_rate']*100:>5.1f}%  {s['neutral_rate']*100:>5.1f}%"
        )
    lines.append("")

    lines.append("── P_ltp 分組的 500-pool Rank（核心分析）──")
    g = analysis["ltp_group_rank"]
    for key in ["positive", "negative", "neutral"]:
        gs = g[key]
        if gs["n"] == 0:
            lines.append(f"  P_ltp {key:>8}: n=0")
            continue
        lines.append(
            f"  P_ltp {key:>8}: n={gs['n']:>4} | "
            f"mean_rank={gs['mean_rank']:>7.2f} | "
            f"median_rank={gs['median_rank']:>6.2f} | "
            f"mean Δ_ltp={gs['mean_delta_ltp']:>8.4f}"
        )
    lines.append("")

    c = analysis["ltp_rank_corr"]
    sig = "（顯著）" if c["p_value"] < 0.05 else "（不顯著）"
    lines.append("── P_ltp Δ 與 rank 的 Spearman 相關 ──")
    lines.append(f"  r = {c['spearman_r']:.4f},  p = {c['p_value']:.4f}  {sig}")
    lines.append(f"  {c['note']}")
    lines.append("")

    ranks = np.array([r["rank"] for r in records], dtype=np.float32)
    lines.append("── 整體 Rank 分佈 ──")
    lines.append(
        f"  R@1  = {(ranks <= 1).mean()*100:.1f}%  "
        f"R@5  = {(ranks <= 5).mean()*100:.1f}%  "
        f"R@10 = {(ranks <= 10).mean()*100:.1f}%  "
        f"MR   = {np.median(ranks):.1f}"
    )
    lines.append("")

    dc = analysis["danger_cases"]
    lines.append(f"── 危險案例（P_ltp Δ < -2.0 且 rank > 50）：{len(dc)} 筆 ──")
    if dc:
        for case in dc[:20]:
            lines.append(
                f"  {case['pair_key'][:23]}  "
                f"rank={case['rank']:>4}  "
                f"Δ_ltp={case['delta_ltp']:>7.4f}  "
                f"Δ_music={case['delta_music']:>7.4f}"
            )
        if len(dc) > 20:
            lines.append(f"  ... 其餘 {len(dc) - 20} 筆略")
    else:
        lines.append("  （無危險案例）")
    lines.append("")

    lines.append(f"── 完美多模態案例（所有 Δ > 0）：{analysis['n_perfect']} 筆 ──")
    if analysis["n_perfect"] > 0:
        for pk in analysis["perfect_pair_keys"][:20]:
            r_info = next(r for r in records if r["pair_key"] == pk)
            lines.append(
                f"  {pk[:23]}  rank={r_info['rank']:>4}  "
                f"Δ_ltp={r_info['delta_ltp']:>6.4f}  "
                f"Δ_video={r_info['delta_video']:>6.4f}"
            )
        if analysis["n_perfect"] > 20:
            lines.append(f"  ... 其餘 {analysis['n_perfect'] - 20} 筆略")
    else:
        lines.append("  （無完美案例）")
    lines.append("")

    ltp_pos_rate = analysis["modality_stats"]["ltp"]["positive_rate"]
    ltp_neg_rate = analysis["modality_stats"]["ltp"]["negative_rate"]

    lines.append("── 解讀建議 ──")
    if ltp_neg_rate > 0.6:
        lines.append("  ⚠️  P_ltp 負向比例 > 60%：建議論文以 modality trade-off 框架敘述，")
        lines.append("      並在 future work 提及 LTP gate 設計。")
    elif ltp_pos_rate > 0.5:
        lines.append("  ✅  P_ltp 正向為主：可直接以個人化貢獻作為論文亮點。")
    else:
        lines.append("  ℹ️  P_ltp 正負比例均衡：屬情境依賴型貢獻，宜以 trade-off 框架敘述。")

    if c["spearman_r"] < -0.1 and c["p_value"] < 0.05:
        lines.append("  ✅  Spearman 相關顯著負值：P_ltp 支持 GT 時 rank 確實改善，")
        lines.append("      代表 P_ltp 學到了有意義的信號，非隨機雜訊。")
    elif abs(c["spearman_r"]) < 0.1:
        lines.append("  ℹ️  Spearman 相關接近 0：P_ltp 對排名影響不顯著。")

    if len(dc) == 0:
        lines.append("  ✅  無危險案例：P_ltp 負向時不會嚴重拖累 rank。")
    elif len(dc) <= 3:
        lines.append(f"  ⚠️  {len(dc)} 筆危險案例：少量樣本 P_ltp 顯著拖累 rank，")
        lines.append("      可考慮加 L2 正則化（ltp_proj）再次訓練。")
    else:
        lines.append(f"  🔴  {len(dc)} 筆危險案例：P_ltp 干擾問題較嚴重，")
        lines.append("      建議加入 cross-modal gate 並重新訓練。")

    lines.append("=" * 70)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 續跑與中途儲存
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_partial():
    if RESUME and os.path.exists(OUT_PARTIAL_JSON):
        try:
            with open(OUT_PARTIAL_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            logger.info("[resume] partial 載入成功：%d 筆", len(records))
            return records
        except Exception as e:
            logger.warning("[resume] partial 載入失敗，將重新開始：%s", e)
    return []


def save_partial(records):
    analysis = analyze(records)
    with open(OUT_PARTIAL_JSON, "w", encoding="utf-8") as f:
        json.dump({"records": records, "analysis": analysis}, f, indent=2, ensure_ascii=False)
    logger.info("[partial] 已儲存：%d 筆 → %s", len(records), OUT_PARTIAL_JSON)


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=== full-test modality ablation (fast) 啟動 ===")

    model, tokenizer, cfg = load_model_and_tokenizer()
    ltp_dict, song_bank_arr, song_ids, song_id_to_idx, conv_map, test_pair_keys = load_resources()
    pair_to_h5 = build_or_load_pair_h5_index()

    if N_SAMPLES is None:
        target_keys = test_pair_keys
        logger.info("目標樣本：全部 test split，共 %d 筆", len(target_keys))
    else:
        target_keys = test_pair_keys[:N_SAMPLES]
        logger.info("目標樣本：前 %d 筆", len(target_keys))

    existing_records = load_existing_partial()
    done_pair_keys = {r["pair_key"] for r in existing_records}
    records = list(existing_records)

    remaining_keys = [pk for pk in target_keys if pk not in done_pair_keys]
    logger.info("已完成 %d 筆，待跑 %d 筆", len(records), len(remaining_keys))

    total = len(target_keys)
    t0 = time.time()
    h5_manager = H5Manager()

    try:
        for run_idx, pair_key in enumerate(remaining_keys, 1):
            global_idx = len(records) + 1
            logger.info("[%d/%d] %s", global_idx, total, pair_key)

            try:
                query_t = build_query(
                    pair_key=pair_key,
                    ltp_dict=ltp_dict,
                    conv_map=conv_map,
                    tokenizer=tokenizer,
                    pair_to_h5=pair_to_h5,
                    h5_manager=h5_manager,
                )

                abl = ablation_one(model, query_t)

                rank, bpr, gap = pool_500(
                    model=model,
                    query_t=query_t,
                    pair_key=pair_key,
                    song_bank_arr=song_bank_arr,
                    song_ids=song_ids,
                    song_id_to_idx=song_id_to_idx,
                )

                records.append({
                    "idx": len(records),
                    "pair_key": pair_key,
                    "video_id": pair_key[:11],
                    "rank": rank,
                    "bpr_score": bpr,
                    "gap_to_2nd": gap,
                    "base_score": abl["base"],
                    "delta_video": abl["delta_video"],
                    "delta_ltp": abl["delta_ltp"],
                    "delta_text": abl["delta_text"],
                    "delta_music": abl["delta_music"],
                })

                logger.info(
                    "  rank=%d | bpr=%.4f | Δltp=%.4f | Δvideo=%.4f | Δtext=%.4f | Δmusic=%.4f",
                    rank, bpr, abl["delta_ltp"], abl["delta_video"],
                    abl["delta_text"], abl["delta_music"],
                )

                if len(records) % SAVE_EVERY == 0:
                    elapsed = time.time() - t0
                    avg_sec = elapsed / max(len(records), 1)
                    logger.info(
                        "[progress] 完成 %d/%d，平均 %.2f sec/sample",
                        len(records), total, avg_sec
                    )
                    save_partial(records)

                torch.cuda.empty_cache()

            except Exception as e:
                logger.error("  [跳過] %s: %s", pair_key, e)

        logger.info("=== 全部完成，共 %d 筆 ===", len(records))

    finally:
        h5_manager.close_all()

    if len(records) == 0:
        logger.error("沒有成功完成任何樣本，請檢查資料路徑、checkpoint、evaluate.py 或 H5 內容。")
        return

    analysis = analyze(records)
    summary = print_summary(analysis, records)
    print(summary)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"records": records, "analysis": analysis}, f, indent=2, ensure_ascii=False)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info("原始數據 → %s", OUT_JSON)
    logger.info("統計摘要 → %s", OUT_TXT)

    # 最終完成後也同步覆蓋 partial
    save_partial(records)


if __name__ == "__main__":
    main()