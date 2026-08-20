"""
用途：整理實驗輸出並產生論文分析用表格或圖表。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import datetime as _dt
import json
import random

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer


# =============================================================================
# 路徑（與 stage5_preference_representation_v4.py 一致）
# =============================================================================

UP_ROOT = Path(r"user_profiling")
PROFILES_JSONL = UP_ROOT / "dataset" / "long_term_preference" / "stage4_recLLM" / "profiles.jsonl"
HISTORY_DIR    = UP_ROOT / "dataset" / "long_term_preference" / "stage2_history" / "personax"
PROJ_WEIGHTS   = UP_ROOT / "dataset" / "stage5_output" / "projection_weights.pt"

CACHE_DIR = PROJECT_ROOT / "cache"
LTP_NPY   = CACHE_DIR / "ltp_hybrid.npy"
LTP_IDS   = CACHE_DIR / "ltp_hybrid_ids.json"
BANK_NPY  = CACHE_DIR / "song_bank.npy"
BANK_IDS  = CACHE_DIR / "song_bank_ids.json"

OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "b5_smoketest"

CLIP_MODEL = "openai/clip-vit-base-patch32"
BETA = 2.0                    # 與 ModelConfig.BETA 一致
N_SAMPLES = 1500              # 512+1 個未知數，留出評估用
TRAIN_RATIO = 0.8
SEED = 20260726
RESIDUAL_THRESHOLD = 1e-3     # 留出樣本上的最大絕對誤差門檻


def log(msg):
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_music_embeddings():
    """
    以 cache/song_bank 重建 stage5 的 music_embeddings 字典。

    song_bank.npy 的內容即 target_music_all_cls 的 12 幀平均（與 stage5 相同計算），
    song_bank_ids.json 的順序來自 pair_index（sorted(h5 files) → pairs.keys()），
    與 stage5 掃描 HDF5 的順序一致，故可重現其「先到先存」的雙索引語義。
    """
    bank = np.load(BANK_NPY).astype(np.float32)
    ids = json.loads(BANK_IDS.read_text(encoding="utf-8"))
    emb = {}
    for vec_i, pair_key in enumerate(ids):
        parts = pair_key.rsplit("_", 1)
        if len(parts) == 2:
            video_id, cand_id = parts
            emb.setdefault(video_id, bank[vec_i])
            emb.setdefault(cand_id, bank[vec_i])
        emb.setdefault(pair_key, bank[vec_i])
    return emb


@torch.no_grad()
def clip_encode(texts, tokenizer, model, device, batch_size=64):
    """CLIPTextModel 的 pooler_output（與 stage5 相同）。"""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(device)
        out.append(model(**inputs).pooler_output.cpu())
    return torch.cat(out, dim=0) if out else torch.zeros(0, 512)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 全程關閉梯度：原始 Stage 5 的兩個編碼函式都包在 torch.no_grad() 內，
    # 此處統一關閉以完全對齊，同時避免 .numpy() 在 requires_grad 張量上失敗。
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device = {device}")

    # ---- 載入既有 LTP（即 y）------------------------------------------------
    ltp = np.load(LTP_NPY).astype(np.float32)
    ltp_ids = json.loads(LTP_IDS.read_text(encoding="utf-8"))
    ltp_map = {mid: i for i, mid in enumerate(ltp_ids)}
    log(f"既有 LTP：{ltp.shape}，ids={len(ltp_ids)}")

    # ---- 載入投影權重 -------------------------------------------------------
    ckpt = torch.load(PROJ_WEIGHTS, map_location=device)
    W_explicit = torch.nn.Linear(512, 256).to(device)
    W_implicit = torch.nn.Linear(768, 256).to(device)
    W_explicit.load_state_dict(ckpt["W_explicit"])
    W_implicit.load_state_dict(ckpt["W_implicit"])
    W_explicit.eval()
    W_implicit.eval()
    log(f"已載入 W_explicit / W_implicit（訓練最終 loss = {ckpt.get('final_loss')}）")

    # ---- 載入 profiles ------------------------------------------------------
    profiles = {}
    with open(PROFILES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            profiles[d["music_id"]] = d
    log(f"profiles = {len(profiles)}")

    music_emb = build_music_embeddings()
    log(f"music_embeddings = {len(music_emb)}")

    # ---- 抽樣 ---------------------------------------------------------------
    rng = random.Random(SEED)
    # 一次列出目錄再比對，避免對 8.4 萬個檔案逐一 stat（實測會多花 14 分鐘）
    import os
    hist_ids = {fn[:-len("__history.json")] for fn in os.listdir(HISTORY_DIR)
                if fn.endswith("__history.json")}
    log(f"stage2 history 檔案 = {len(hist_ids)}")
    candidates = [m for m in profiles if m in ltp_map and m in hist_ids]
    log(f"可用樣本（有 profile + 有 LTP + 有 history）= {len(candidates)}")
    picked = rng.sample(candidates, min(N_SAMPLES, len(candidates)))

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    clip = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device).eval()

    X, Y, used = [], [], []
    skipped = {"no_history": 0, "no_core_sbs": 0, "no_emb": 0}

    for n, mid in enumerate(picked, start=1):
        prof = profiles[mid]
        hist_path = HISTORY_DIR / f"{mid}__history.json"
        try:
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            skipped["no_history"] += 1
            continue
        core_sbs = hist.get("balanced_history", {}).get("core_sbs", [])
        if not core_sbs:
            skipped["no_core_sbs"] += 1
            continue

        # ---- explicit（與 encode_explicit_preference 完全相同的取字邏輯）----
        facts = prof.get("salient_facts", [])
        if facts:
            fact_texts = ([f.get("fact", "") for f in facts] if isinstance(facts[0], dict)
                          else list(facts))
            text_content = ". ".join(t for t in fact_texts if t)
        else:
            text_content = prof.get("summary_text", "")

        if not text_content or not text_content.strip():
            explicit_proj = np.zeros(256, dtype=np.float32)
            clip_emb = np.zeros(512, dtype=np.float32)
        else:
            emb = clip_encode([text_content], tokenizer, clip, device).to(device)
            explicit_proj = W_explicit(emb).cpu().numpy().flatten()
            clip_emb = emb.cpu().numpy().flatten()

        # ---- implicit（語義相似度 softmax 加權後投影）----------------------
        embs, seeds = [], []
        for item in core_sbs:
            m_id = item.get("music_id", "")
            if m_id in music_emb:
                embs.append(music_emb[m_id])
                seeds.append(item.get("semantic_seed", ""))
        if not embs:
            skipped["no_emb"] += 1
            continue

        seeds_emb = clip_encode(seeds, tokenizer, clip, device).to(device)
        exp_t = torch.tensor(clip_emb, dtype=torch.float32).unsqueeze(0).to(device)
        exp_n = exp_t / exp_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
        seeds_n = seeds_emb / seeds_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sims = torch.matmul(seeds_n, exp_n.T).squeeze(-1)
        weights = torch.softmax(BETA * sims, dim=0).cpu().numpy()

        weighted = np.average(np.array(embs), axis=0, weights=weights)
        with torch.no_grad():
            implicit_proj = W_implicit(
                torch.tensor(weighted, dtype=torch.float32).unsqueeze(0).to(device)
            ).cpu().numpy().flatten()

        X.append(np.concatenate([explicit_proj, implicit_proj]))
        Y.append(ltp[ltp_map[mid]])
        used.append(mid)

        if n % 200 == 0:
            log(f"  已處理 {n}/{len(picked)}（有效 {len(X)}）")

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    log(f"有效配對 = {X.shape[0]}，x 維度 = {X.shape[1]}，y 維度 = {Y.shape[1]}｜略過 {skipped}")

    if X.shape[0] < 600:
        raise SystemExit(f"有效樣本僅 {X.shape[0]} 筆，不足以求解 513 個未知數，請提高 N_SAMPLES。")

    # ---- 最小平方法求解 W, b ------------------------------------------------
    n_train = int(X.shape[0] * TRAIN_RATIO)
    Xd = np.hstack([X, np.ones((X.shape[0], 1))])          # 併入 bias 項
    A, B = Xd[:n_train], Y[:n_train]
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)            # [513, 256]

    pred_tr = Xd[:n_train] @ sol
    pred_te = Xd[n_train:] @ sol
    res_tr = np.abs(pred_tr - Y[:n_train])
    res_te = np.abs(pred_te - Y[n_train:])

    y_scale = float(np.abs(Y).mean())
    report = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_pairs": int(X.shape[0]), "n_train": n_train, "n_test": int(X.shape[0] - n_train),
        "skipped": skipped,
        "y_mean_abs": y_scale,
        "train_max_abs_error": float(res_tr.max()),
        "train_mean_abs_error": float(res_tr.mean()),
        "test_max_abs_error": float(res_te.max()),
        "test_mean_abs_error": float(res_te.mean()),
        "test_relative_mean_error": float(res_te.mean() / y_scale) if y_scale else float("nan"),
        "threshold": RESIDUAL_THRESHOLD,
        "recovered": bool(res_te.max() < RESIDUAL_THRESHOLD),
    }

    log("-" * 70)
    log(f"y 平均絕對值        = {y_scale:.6f}")
    log(f"訓練集最大絕對誤差  = {report['train_max_abs_error']:.3e}")
    log(f"留出集最大絕對誤差  = {report['test_max_abs_error']:.3e}")
    log(f"留出集平均絕對誤差  = {report['test_mean_abs_error']:.3e} "
        f"（相對 {report['test_relative_mean_error']*100:.4f}%）")
    # 逐樣本殘差分布：用以分辨「全體都差一點」與「多數精確、少數壞掉」。
    # 後者代表 W_out 其實可還原，只是部分樣本的 x 重建有問題（可排除後重解）。
    per_sample = np.abs(pred_te - Y[n_train:]).max(axis=1)
    pct = {f"p{q}": float(np.percentile(per_sample, q)) for q in (50, 75, 90, 95, 99)}
    n_tight = int((per_sample < 1e-4).sum())
    report["test_per_sample_max_err_percentiles"] = pct
    report["test_n_samples_under_1e-4"] = n_tight
    report["test_share_under_1e-4"] = n_tight / per_sample.size

    log(f"逐樣本最大誤差分位數  = " +
        "、".join(f"{k}={v:.3e}" for k, v in pct.items()))
    log(f"留出樣本中誤差 < 1e-4 者 = {n_tight}/{per_sample.size} "
        f"（{100 * n_tight / per_sample.size:.1f}%）")
    log(f"還原判定（門檻 {RESIDUAL_THRESHOLD}）→ "
        f"{'成功，B5 可繼續' if report['recovered'] else '失敗，B5 需改設計'}")

    if report["recovered"]:
        W = torch.tensor(sol[:-1].T, dtype=torch.float32)   # [256, 512]
        b = torch.tensor(sol[-1], dtype=torch.float32)      # [256]
        torch.save({"weight": W, "bias": b,
                    "note": "由 cache/ltp_hybrid.npy 與重算之 x 以最小平方法還原的 "
                            "Stage5 W_out_hybrid；供 B5 Persona LTP 編碼使用",
                    "report": report},
                   OUT_DIR / "recovered_W_out_hybrid.pt")
        log(f"已輸出還原權重：{OUT_DIR / 'recovered_W_out_hybrid.pt'}")

    (OUT_DIR / "wout_recovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# B5 快速檢查：Stage 5 輸出投影層還原\n",
          f"- 產生時間：{report['generated_at']}",
          f"- 有效配對樣本：{report['n_pairs']}（訓練 {report['n_train']} / 留出 {report['n_test']}）\n",
          "## 問題\n",
          "`stage5_preference_representation_v4.py` 的 `W_out_hybrid` 每次執行都重新 Xavier "
          "隨機初始化，且未存入 `projection_weights.pt`、全檔無 `manual_seed`。"
          "直接重跑 Stage 5 會產生與訓練時不同的 LTP 空間，Persona 向量將無法與 exp_01 相容。\n",
          "## 還原結果\n",
          "| 指標 | 數值 |", "|---|---|",
          f"| y 平均絕對值 | {report['y_mean_abs']:.6f} |",
          f"| 訓練集最大絕對誤差 | {report['train_max_abs_error']:.3e} |",
          f"| 留出集最大絕對誤差 | {report['test_max_abs_error']:.3e} |",
          f"| 留出集平均絕對誤差 | {report['test_mean_abs_error']:.3e} |",
          f"| 相對誤差 | {report['test_relative_mean_error']*100:.4f}% |",
          f"| 留出樣本誤差 < 1e-4 之比例 | "
          f"{report.get('test_share_under_1e-4', 0)*100:.1f}% "
          f"（{report.get('test_n_samples_under_1e-4', 0)}/{report['n_test']}）|",
          f"\n**判定**：{'還原成功' if report['recovered'] else '還原失敗'}"
          f"（門檻：留出集最大絕對誤差 < {RESIDUAL_THRESHOLD}）\n",
          "逐樣本最大誤差分位數："
          + "、".join(f"{k}={v:.3e}"
                     for k, v in report.get("test_per_sample_max_err_percentiles", {}).items())
          + "\n"]
    if report["recovered"]:
        md.append("留出樣本殘差趨近 0，同時證明兩件事：\n")
        md.append("1. `W_out_hybrid` 已精確還原，Persona LTP 可用同一投影產生，與訓練分布相容。")
        md.append("2. 本腳本重算 x（explicit_256 + implicit_256）的流程與原始 Stage 5 完全一致，"
                  "後續 Persona 編碼可信。\n")
        md.append("→ **B5 可繼續**。Persona LTP 產生流程："
                  "Persona 規格 → 合成歷史 → Stage 3/4 畫像 → 本腳本的 x 重算流程 → "
                  "`recovered_W_out_hybrid.pt` → 256 維 LTP。")
    else:
        md.append("殘差未達門檻，代表 x 的重算仍與原始流程有出入（可能的原因："
                  "music_embeddings 的雙索引先到先存順序、semantic_seed 內容、"
                  "或 profiles 版本不一致）。\n")
        md.append("→ **B5 不可直接沿用 Stage 5**。替代設計："
                  "改以 Persona 歷史曲目的既有 LTP 向量加權平均產生 Persona LTP，"
                  "此法保證落在訓練分布內，但無法宣稱走完整條 Stage 3–5 管線。")
    (OUT_DIR / "wout_recovery_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"已輸出報告：{OUT_DIR / 'wout_recovery_report.md'}")


if __name__ == "__main__":
    main()
