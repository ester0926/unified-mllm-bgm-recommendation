# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
run_eval_500pool.py — 統一評估腳本（排序指標 + 生成指標）

換 exp 只需修改下方「實驗設定區」，其他邏輯自動對應。

執行：
  python run_eval_500pool.py

依賴安裝：
  pip install bert-score "torchmetrics[text]"
"""

import os, sys, json, logging, traceback, datetime
import torch
import h5py
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# ★ 實驗設定區（換 exp 只改這裡）
# ══════════════════════════════════════════════════════════════════════════════

EXP_NAME = "exp_07"   # ← 改這裡：exp_01 / exp_02 / exp_03 / exp_04 / exp_05 / exp_06 / exp_07

# LTP 模式對應表：
#   exp_01 hybrid（主實驗）→ "hybrid"
#   exp_02 explicit_only   → "explicit_only"
#   exp_03 implicit_only   → "implicit_only"
#   exp_04 w/o P_ltp       → "hybrid"（ltp_dict 仍載入，但 active_modalities 不含 "ltp"）
#   exp_05 w/o Video       → "hybrid"
#   exp_06 w/o Text        → "hybrid"
#   exp_07 w/o Music       → "hybrid"
LTP_MODE = {
    "exp_01": "hybrid",
    "exp_02": "explicit_only",
    "exp_03": "implicit_only",
    "exp_04": "hybrid",
    "exp_05": "hybrid",
    "exp_06": "hybrid",
    "exp_07": "hybrid",
}[EXP_NAME]

# active_modalities 對應表：
#   exp_01~03 全模態         → ["video", "ltp", "text", "music"]
#   exp_04 w/o P_ltp        → ["video", "text", "music"]
#   exp_05 w/o Video        → ["ltp", "text", "music"]
#   exp_06 w/o Text         → ["video", "ltp", "music"]
#   exp_07 w/o Music        → ["video", "ltp", "text"]
ACTIVE_MODALITIES = {
    "exp_01": ["video", "ltp", "text", "music"],
    "exp_02": ["video", "ltp", "text", "music"],
    "exp_03": ["video", "ltp", "text", "music"],
    "exp_04": ["video", "text", "music"],
    "exp_05": ["ltp",   "text", "music"],
    "exp_06": ["video", "ltp",  "music"],
    "exp_07": ["video", "ltp",  "text"],
}[EXP_NAME]

# ── 路徑（OUTPUT_DIR 由 EXP_NAME 自動決定）──────────────────────────────────
BASE_DIR    = str(PROJECT_ROOT)
H5_DIR      = str(PROJECT_ROOT / "data" / "optimized_musechat_features_float16_v3")
JSON_DIR    = str(PROJECT_ROOT / "data" / "musechat_json")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
LLAMA_MODEL = r"meta-llama/Llama-2-7b-hf"

OUTPUT_DIR  = os.path.join(BASE_DIR, "checkpoints", EXP_NAME)
BEST_CKPT   = os.path.join(OUTPUT_DIR, "best")
LAST_CKPT   = os.path.join(OUTPUT_DIR, "epoch_10")
LOG_PATH    = os.path.join(OUTPUT_DIR, f"eval_500pool_{EXP_NAME}.log")

LTP_H5 = {
    "hybrid":        str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors.h5"),
    "explicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_explicit_only.h5"),
    "implicit_only": str(PROJECT_ROOT / "data" / "user_profiling" / "stage5_output" / "preference_vectors_implicit_only.h5"),
}

SPECIAL_TOKENS = ["[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"]

# ── 評估開關 ──────────────────────────────────────────────────────────────────
EVAL_RANKING    = True
EVAL_GENERATION = True
MAX_GEN_SAMPLES = None   # None=全量；先測試可設 200
INJECT_TITLE    = True   # True=對齊 MuseChat 推論；False=Vicuna w/ Music baseline

CHECKPOINTS = [
    ("best",     BEST_CKPT),
    # ("epoch_10", LAST_CKPT),
]
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = BASE_DIR
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

_fmt = "[%(asctime)s][%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO, format=_fmt,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger("eval_v3")


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def log_gpu_mem(tag=""):
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved()  / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info("  [GPU MEM%s] allocated=%.2fGB  reserved=%.2fGB  total=%.2fGB",
                (" " + tag) if tag else "", a, r, t)


# ══════════════════════════════════════════════════════════════════════════════
# 資料載入
# ══════════════════════════════════════════════════════════════════════════════

def load_ltp_dict(h5_path, mode, cache_path=None):
    if cache_path:
        npy = cache_path + f"_{mode}.npy"
        ids = cache_path + f"_{mode}_ids.json"
        if os.path.exists(npy) and os.path.exists(ids):
            arr = np.load(npy)
            with open(ids) as f:
                vl = json.load(f)
            d = {v: arr[i] for i, v in enumerate(vl)}
            logger.info("[LTP] 快取載入：%d 筆，模式=%s", len(d), mode)
            return d
    logger.info("[LTP] 從 HDF5 載入 %s ...", mode)
    out = {}
    with h5py.File(h5_path, "r") as f:
        grp = f["preference_vectors"]
        for k in grp.keys():
            out[k] = grp[k][:].astype(np.float32)
    logger.info("[LTP] 載入完成：%d 筆", len(out))
    return out


def build_test_data(model_config, train_config, tokenizer, ltp_dict):
    from dataset import (
        UnifiedMLLMDataset, build_pair_index, load_conversation_map,
        build_song_bank, split_by_video_id,
    )
    pair_index   = build_pair_index(H5_DIR, cache_path=os.path.join(CACHE_DIR, "pair_index.json"))
    conv_map     = load_conversation_map(JSON_DIR)
    song_bank_np, song_ids = build_song_bank(pair_index, cache_path=os.path.join(CACHE_DIR, "song_bank"))
    _, _, te_pairs = split_by_video_id(
        pair_index,
        train_config.train_ratio, train_config.val_ratio,
        train_config.test_ratio, train_config.split_seed,
    )
    # ★ Fix 1：傳入 active_modalities，確保 dataset 的 prompt 模板與訓練時一致
    # exp_01~03：全模態 prompt；exp_04~07：移除對應模態行的 prompt
    test_dataset = UnifiedMLLMDataset(
        pairs=te_pairs, tokenizer=tokenizer, conv_map=conv_map,
        song_bank=song_bank_np, song_ids=song_ids, ltp_dict=ltp_dict,
        max_seq_len=model_config.max_seq_len, is_train=False,
        ltp_dim=model_config.ltp_dim,
        mc_neg_cache_dir=CACHE_DIR,
        active_modalities=model_config.active_modalities,  # ★
    )
    logger.info("  Test pairs: %d | Song bank: %d", len(te_pairs), len(song_ids))
    return test_dataset, torch.tensor(song_bank_np, dtype=torch.float32), song_ids


# ══════════════════════════════════════════════════════════════════════════════
# 模型載入
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_dir, model_config, tokenizer):
    """
    乾淨載入推論模型。
    ★ Fix P4：gradient_checkpointing_disable() 確保 generate() 正常運作。
    """
    import torch.nn as nn
    from transformers import LlamaForCausalLM
    from peft import PeftModel
    from models.projectors import MultimodalProjectors
    from models.unified_mllm import UnifiedMLLM

    logger.info("  [載入] %s", ckpt_dir)
    log_gpu_mem("before load")

    n = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    logger.info("  tokenizer +%d special tokens → vocab=%d", n, len(tokenizer))

    base = LlamaForCausalLM.from_pretrained(
        model_config.llama_model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    base.resize_token_embeddings(len(tokenizer))

    peft_llama = PeftModel.from_pretrained(base, ckpt_dir, torch_dtype=torch.bfloat16)
    peft_llama.eval()

    # ★ Fix P4：推論時必須關閉 gradient checkpointing
    # 否則 generate() 使用 inputs_embeds 時與 PEFT hook 衝突，回傳 0 個新 token
    if hasattr(peft_llama, "gradient_checkpointing_disable"):
        peft_llama.gradient_checkpointing_disable()
        logger.info("  gradient_checkpointing_disable() 完成（推論模式）")

    log_gpu_mem("after PEFT")

    projectors = MultimodalProjectors(
        video_dim=model_config.video_dim,
        music_dim=model_config.music_dim,
        text_dim=model_config.text_dim,
        ltp_dim=model_config.ltp_dim,
        llama_hidden_dim=model_config.llama_hidden_dim,
        projector_hidden_dim=model_config.projector_hidden_dim,
        dropout=0.0,
    )
    projectors.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "projectors.pt"), map_location="cuda:0")
    )
    projectors = projectors.to(torch.bfloat16).cuda().eval()

    ranking_head = nn.Sequential(
        nn.LayerNorm(model_config.llama_hidden_dim),
        nn.Linear(model_config.llama_hidden_dim, 256),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(256, 1),
    )
    ranking_head.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "ranking_head.pt"), map_location="cuda:0")
    )
    ranking_head = ranking_head.to(torch.bfloat16).cuda().eval()
    log_gpu_mem("after projectors+head")

    # _load_llama=False 避免重複載入 LLaMA（若 unified_mllm.py 已加此參數）
    try:
        model = UnifiedMLLM(model_config=model_config, tokenizer=tokenizer, _load_llama=False)
    except TypeError:
        # 若 unified_mllm.py 尚未加 _load_llama 參數，退回原方式
        model = UnifiedMLLM(model_config=model_config, tokenizer=tokenizer)
        del model.llama
        torch.cuda.empty_cache()

    model.llama        = peft_llama
    model.projectors   = projectors
    model.ranking_head = ranking_head
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 排序指標
# ══════════════════════════════════════════════════════════════════════════════

def eval_ranking(model, test_dataset, all_music_features, all_music_ids,
                 device, model_cfg, train_cfg):
    """500-pool Pointwise 排序評估（全量 test set）"""
    logger.info("\n[Step 1] 排序指標（500-pool Pointwise，%d 筆）", len(test_dataset))
    from evaluate import evaluate_with_music_pool
    metrics = evaluate_with_music_pool(
        model=model, test_dataset=test_dataset,
        all_music_features=all_music_features, all_music_ids=all_music_ids,
        device=device, train_config=train_cfg, model_config=model_cfg,
    )
    logger.info("  R@1  = %.2f%%", metrics["recall@1"]  * 100)
    logger.info("  R@5  = %.2f%%", metrics["recall@5"]  * 100)
    logger.info("  R@10 = %.2f%%", metrics["recall@10"] * 100)
    logger.info("  MR   = %.1f",   metrics["median_rank"])
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 生成指標
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_one(model, sample, tokenizer, device,
                  max_new_tokens=256, inject_title=True):
    """
    單筆樣本生成推薦說明文字。

    ★ inject_title=True（預設，對齊 MuseChat 推論設計）：
      從 sample["t4_reference"] 解析歌名後重建 prompt 再生成。
      → BERTScore 可與 MuseChat 公平比較（論文 F1=0.9676）

    ★ inject_title=False（MuseChat 論文的 Vicuna w/ Music baseline）：
      只靠聲學特徵生成，不含歌名。
      → 用於量化「加歌名 vs 不加歌名」的 BERTScore 差距
    """
    try:
        bf16 = torch.bfloat16
        video_feat = sample["video_feat"].unsqueeze(0).to(device, dtype=bf16)
        ltp_feat   = sample["ltp_feat"].unsqueeze(0).to(device, dtype=bf16)
        text_feat  = sample["text_feat"].unsqueeze(0).to(device, dtype=bf16)
        pos_music  = sample["pos_music_feat"].unsqueeze(0).unsqueeze(0).to(device, dtype=bf16)

        prompt_len = int(sample["prompt_len"].item())
        input_ids  = sample["input_ids"][:prompt_len].unsqueeze(0).to(device)
        attn_mask  = sample["attention_mask"][:prompt_len].unsqueeze(0).to(device)

        # ★ 標題注入：解析歌名後重建含歌名的 prompt 並重新 tokenize
        if inject_title:
            t4_ref   = sample.get("t4_reference", "")
            user_txt = sample.get("user_text",    "")
            if t4_ref:
                from dataset import extract_music_title, build_prompt
                music_title, music_artist = extract_music_title(t4_ref)
                if music_title:
                    # ★ 用 model 的實際 active_modalities，消融版才能對齊訓練設定
                    model_active = getattr(model, "active_modalities",
                                           ["video", "ltp", "text", "music"])
                    prompt_tmpl = build_prompt(
                        active_modalities=model_active,
                        music_title=music_title,
                        music_artist=music_artist,
                    )
                    full_prompt = (prompt_tmpl.format(user_text=user_txt)
                                   if "{user_text}" in prompt_tmpl else prompt_tmpl)
                    enc = tokenizer(full_prompt, return_tensors="pt",
                                    add_special_tokens=False)
                    input_ids = enc["input_ids"].to(device)
                    attn_mask = enc["attention_mask"].to(device)

        rank_token_id = getattr(model, "rank_token_id", None)
        if rank_token_id is None or (input_ids == rank_token_id).sum().item() == 0:
            logger.warning("  [generate] [RANK] not found in prompt, skipping")
            return "", True

        generated_ids, _ = model.generate(
            video_feat=video_feat, music_candidates=pos_music,
            ltp_feat=ltp_feat, text_feat=text_feat,
            input_ids=input_ids, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
        new_tokens = generated_ids[0]
        decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1].strip()
        decoded = decoded.replace("[RANK]", "").strip()
        return decoded if decoded else "", False

    except Exception as e:
        logger.warning("  [generate] 失敗：%s", e)
        return "", True


def eval_generation(model, test_dataset, tokenizer, device, model_cfg,
                    max_samples=None, inject_title=True):
    """
    生成評估：對 test set 逐筆 generate，計算 BERTScore + InfoLM。

    inject_title=True（預設）：從 t4 解析歌名後注入 prompt（對齊 MuseChat）
    inject_title=False：純聲學生成（Vicuna w/ Music baseline）
    """
    import glob as _glob

    logger.info("\n[Step 2] 生成指標（BERTScore + InfoLM）")
    logger.info("  標題注入模式：%s", "✅ 開啟（對齊 MuseChat 推論）" if inject_title
                else "❌ 關閉（Vicuna w/ Music baseline）")

    # 載入 t4 reference map：video_id → (t3, t4)
    conv_map_t3 = {}   # video_id → t3（用戶請求）
    conv_map    = {}   # video_id → t4（GT response，含歌名）
    for jf in _glob.glob(os.path.join(JSON_DIR, "**", "*.json"), recursive=True):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            convs = data.get("conversations", [])
            if len(convs) >= 4:
                t3 = convs[2].get("value", "").strip()
                t4 = convs[3].get("value", "").strip()
                if t4:
                    vid = Path(jf).parent.name
                    conv_map[vid]    = t4
                    conv_map_t3[vid] = t3
        except Exception:
            pass
    logger.info("  [conv_map] t4 reference：%d 筆", len(conv_map))

    n_total = len(test_dataset)
    n_gen   = n_total if max_samples is None else min(max_samples, n_total)
    logger.info("  生成樣本數：%d / %d", n_gen, n_total)

    hypotheses, references, fallback_count = [], [], 0
    n_title_injected = 0

    for idx in range(n_gen):
        sample = test_dataset[idx]

        # video_id 取法（pair_key 前 11 碼最穩定）
        pair_key = sample.get("pair_key", "") or sample.get("gt_music_id", "")
        if not pair_key:
            raw_vid = sample.get("video_id", "")
            pair_key = raw_vid if isinstance(raw_vid, str) else str(raw_vid)
        if isinstance(pair_key, bytes):
            pair_key = pair_key.decode("utf-8")
        video_id = str(pair_key)[:11]

        ref_text = conv_map.get(video_id, "")
        if not ref_text:
            continue   # 無 t4 reference，跳過不污染指標

        # ★ 將 t4 和 t3 暫存進 sample dict，供 _generate_one 解析歌名
        sample["t4_reference"] = ref_text
        sample["user_text"]    = conv_map_t3.get(video_id, "")

        hyp_text, is_fallback = _generate_one(
            model, sample, tokenizer, device,
            max_new_tokens=getattr(model_cfg, "max_new_tokens", 256),
            inject_title=inject_title,
        )
        if is_fallback:
            fallback_count += 1
        else:
            # 確認是否真的注入了歌名
            if inject_title and ref_text:
                from dataset import extract_music_title
                _t, _ = extract_music_title(ref_text)
                if _t and _t.lower() in hyp_text.lower():
                    n_title_injected += 1

        hypotheses.append(hyp_text)
        references.append(ref_text)

        if (idx + 1) % 500 == 0 or (idx + 1) == n_gen:
            logger.info("  [%d/%d] fallback 率：%.1f%%  歌名命中：%d 筆",
                        idx + 1, n_gen,
                        fallback_count / max(len(hypotheses), 1) * 100,
                        n_title_injected)

    n_valid = sum(1 for h in hypotheses if h.strip())
    logger.info("  完成：%d 筆，有效 %d，fallback %d（%.1f%%）  歌名命中：%d 筆",
                len(hypotheses), n_valid, fallback_count,
                fallback_count / max(len(hypotheses), 1) * 100,
                n_title_injected)

    result = {
        "n_generated":    len(hypotheses),
        "n_valid":        n_valid,
        "fallback_count": fallback_count,
        "fallback_rate":  round(fallback_count / max(len(hypotheses), 1), 4),
    }
    result.update(_bertscore(hypotheses, references))
    result.update(_infolm(hypotheses, references))
    return result


def _bertscore(hypotheses, references):
    valid = [(h, r) for h, r in zip(hypotheses, references) if h.strip()]
    if not valid:
        logger.warning("  [BERTScore] 無有效樣本")
        return {"bertscore_f1": None, "bertscore_precision": None, "bertscore_recall": None}
    try:
        from bert_score import score as _bs
        P, R, F1 = _bs([p[0] for p in valid], [p[1] for p in valid],
                       model_type="microsoft/deberta-xlarge-mnli",
                       lang="en", verbose=False,
                       device="cuda" if torch.cuda.is_available() else "cpu")
        r = {
            "bertscore_precision": round(float(P.mean()), 4),
            "bertscore_recall":    round(float(R.mean()), 4),
            "bertscore_f1":        round(float(F1.mean()), 4),
        }
        logger.info("  BERTScore  P=%.4f  R=%.4f  F1=%.4f  (n=%d)",
                    r["bertscore_precision"], r["bertscore_recall"],
                    r["bertscore_f1"], len(valid))
        return r
    except ImportError:
        logger.error("  [BERTScore] pip install bert-score")
        return {"bertscore_f1": None, "error_bertscore": "not installed"}
    except Exception as e:
        logger.error("  [BERTScore] %s", e)
        return {"bertscore_f1": None, "error_bertscore": str(e)}


def _infolm(hypotheses, references):
    valid = [(h, r) for h, r in zip(hypotheses, references) if h.strip()]
    if not valid:
        logger.warning("  [InfoLM] 無有效樣本")
        return {"infolm_ab_divergence": None, "infolm_l2_distance": None, "infolm_fisher_rao": None}
    try:
        from torchmetrics.functional.text.infolm import infolm as tm_infolm
        hyps = [p[0] for p in valid]
        refs = [p[1] for p in valid]
        kw = dict(model_name_or_path="bert-base-uncased", idf=False,
                  device="cuda" if torch.cuda.is_available() else "cpu",
                  verbose=False, batch_size=16)
        l2 = tm_infolm(hyps, refs, information_measure="l2_distance",        **kw)
        fr = tm_infolm(hyps, refs, information_measure="fisher_rao_distance", **kw)
        try:
            ab = tm_infolm(hyps, refs, information_measure="ab_divergence",
                           alpha=0.5, beta=0.5, **kw)
            ab_val = round(float(ab), 4)
        except Exception as e_ab:
            logger.warning("  [InfoLM] ab_divergence 略過：%s", e_ab)
            ab_val = None
        r = {"infolm_ab_divergence": ab_val,
             "infolm_l2_distance":   round(float(l2), 4),
             "infolm_fisher_rao":    round(float(fr), 4)}
        logger.info("  InfoLM  AB=%s  L2=%.4f  FR=%.4f  (n=%d)",
                    str(ab_val), r["infolm_l2_distance"], r["infolm_fisher_rao"], len(hyps))
        return r
    except ImportError:
        logger.error("  [InfoLM] pip install 'torchmetrics[text]'")
        return {"infolm_l2_distance": None, "error_infolm": "not installed"}
    except Exception as e:
        logger.error("  [InfoLM] %s", e)
        return {"infolm_ab_divergence": None, "infolm_l2_distance": None,
                "infolm_fisher_rao": None, "error_infolm": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from config import ModelConfig, TrainConfig
    from transformers import LlamaTokenizer

    start = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info("run_eval_500pool（排序 + 生成，統一版）")
    logger.info("  實驗: %s  |  LTP: %s  |  active_modalities: %s",
                EXP_NAME, LTP_MODE, ACTIVE_MODALITIES)
    logger.info("  時間: %s", start.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("  EVAL_RANKING: %s  |  EVAL_GENERATION: %s  |  INJECT_TITLE: %s",
                EVAL_RANKING, EVAL_GENERATION, INJECT_TITLE)
    logger.info("=" * 60)

    device    = torch.device("cuda")
    # ★ Fix 2：ModelConfig 傳入 active_modalities，讓 __post_init__ 自動推導
    # multimodal_prefix_len（全模態=4，消融=3）
    model_cfg = ModelConfig(
        llama_model_name      = LLAMA_MODEL,
        video_dim=768, music_dim=768, text_dim=512, ltp_dim=256,
        num_candidates=1,
        active_modalities     = ACTIVE_MODALITIES,   # ★ 消融時自動設 prefix_len=3
        music_token_offset=3,
        rank_special_token="[RANK]",
    )
    train_cfg = TrainConfig(output_dir=OUTPUT_DIR, pointwise_eval_batch_size=32)

    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    ltp_dict = load_ltp_dict(LTP_H5[LTP_MODE], LTP_MODE,
                             cache_path=os.path.join(CACHE_DIR, "ltp"))
    test_dataset, all_music_features, all_music_ids = build_test_data(
        model_cfg, train_cfg, tokenizer, ltp_dict
    )

    all_results = {}
    for ckpt_name, ckpt_dir in CHECKPOINTS:
        if not os.path.isdir(ckpt_dir):
            logger.warning("  Checkpoint 不存在，跳過: %s", ckpt_dir)
            continue

        logger.info("\n%s\n  評估 [%s]\n%s", "="*60, ckpt_name, "="*60)
        model   = load_model(ckpt_dir, model_cfg, tokenizer)
        metrics = {}

        if EVAL_RANKING:
            ranking_m = eval_ranking(model, test_dataset, all_music_features,
                                     all_music_ids, device, model_cfg, train_cfg)
            metrics.update(ranking_m)

        if EVAL_GENERATION:
            gen_m = eval_generation(model, test_dataset, tokenizer, device,
                                    model_cfg, max_samples=MAX_GEN_SAMPLES,
                                    inject_title=INJECT_TITLE)
            metrics.update(gen_m)

        del model
        torch.cuda.empty_cache()

        all_results[ckpt_name] = metrics
        # 儲存單一 checkpoint（排除大欄位）
        save_m = {k: v for k, v in metrics.items() if k != "sample_outputs"}
        out_path = os.path.join(OUTPUT_DIR, f"eval_v3_{ckpt_name}.json")
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(save_m, fp, indent=2, ensure_ascii=False)

    # 完整結果
    final_path = os.path.join(OUTPUT_DIR, "eval_500pool_v3_results.json")
    with open(final_path, "w", encoding="utf-8") as fp:
        json.dump(all_results, fp, indent=2, ensure_ascii=False)

    elapsed = datetime.datetime.now() - start
    logger.info("\n%s", "=" * 60)
    logger.info("【v3 評估結果】")
    logger.info("%s", "=" * 60)
    for name, m in all_results.items():
        logger.info("  [%s]", name)
        if "recall@1" in m:
            logger.info("    排序  R@1=%.2f%%  R@5=%.2f%%  R@10=%.2f%%  MR=%.1f",
                        m["recall@1"]*100, m["recall@5"]*100,
                        m["recall@10"]*100, m["median_rank"])
        if m.get("bertscore_f1") is not None:
            logger.info("    BERTScore  P=%.4f  R=%.4f  F1=%.4f",
                        m.get("bertscore_precision",0),
                        m.get("bertscore_recall",0),
                        m["bertscore_f1"])
        if m.get("infolm_l2_distance") is not None:
            logger.info("    InfoLM  AB=%s  L2=%.4f  FR=%.4f",
                        str(m.get("infolm_ab_divergence")),
                        m["infolm_l2_distance"],
                        m["infolm_fisher_rao"])
        logger.info("    fallback=%d/%d",
                    m.get("fallback_count",0), m.get("n_generated",0))
    logger.info("")
    logger.info("  [MuseChat 論文]  R@1=32.79%%  R@5=63.92%%  R@10=76.53%%  MR=2"
                "  BERTScore F1=0.97")
    logger.info("  總耗時: %s  |  結果→ %s", str(elapsed).split(".")[0], final_path)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:
        logger.error("OOM: %s\n%s", e, traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s %s\n%s", type(e).__name__, e, traceback.format_exc())
        sys.exit(1)