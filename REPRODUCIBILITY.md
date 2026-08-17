# 可重現流程說明

這份文件說明如何從資料準備到重現論文主要實驗結果。由於大型資料與模型權重未放入 GitHub，請先依照 [DATA.md](DATA.md) 和 [CHECKPOINTS.md](CHECKPOINTS.md) 放好外部檔案。

## 1. 建立環境

原始實驗環境約為：

```text
Python: 3.11
PyTorch: 2.11.0 dev + CUDA 12.8
GPU: NVIDIA GeForce RTX 5090
Base LLM: meta-llama/Llama-2-7b-hf
```

建議安裝：

```bash
conda create -n bgm-recommender python=3.11
conda activate bgm-recommender
pip install -r requirements.txt
```

若需使用 `meta-llama/Llama-2-7b-hf`，請先在 Hugging Face 申請權限並登入。

## 2. 準備資料

依照 [DATA.md](DATA.md) 準備：

```text
data/optimized_musechat_features_float16_v3/
data/musechat_json/
data/user_profiling/
cache/ltp_hybrid.npy
cache/ltp_hybrid_ids.json
cache/ltp_explicit_only.npy
cache/ltp_explicit_only_ids.json
cache/ltp_implicit_only.npy
cache/ltp_implicit_only_ids.json
```

如果資料放在其他位置，請調整 `config.py`、`project_paths.py` 或各 script 中的資料路徑設定。

## 3. 建立或確認 P_ltp

若要從頭重建 synthetic long-term preference proxy，請依序執行：

```text
user_profiling/stage0_data_prep/
user_profiling/stage1_metadata.py
user_profiling/stage2_personax.py
user_profiling/stage3/
user_profiling/stage4/
user_profiling/stage5/
```

流程細節、輸入輸出與限制請看 [LTP_PIPELINE.md](LTP_PIPELINE.md)。若只重現主模型評估，可直接使用已產生的 LTP cache。

## 4. 訓練 exp_01 至 exp_07

訓練腳本位於：

```text
scripts/train/
```

主要條件：

```text
exp_01: full model with hybrid P_ltp
exp_02: explicit_only P_ltp
exp_03: implicit_only P_ltp
exp_04: w/o P_ltp
exp_05: w/o Video
exp_06: w/o Text
exp_07: w/o Music
```

訓練後請將輸出整理到：

```text
checkpoints/exp_*/best/
checkpoints/exp_*/detailed_eval/
```

## 5. 執行 ranking 與 generation 評估

主評估腳本位於：

```text
scripts/eval_main/
```

論文主要報告 4,205 筆 MuseChat test samples 的 random 500-candidate pool 結果。`exp_01` full model 的主要結果為：

```text
R@1  = 30.65%
R@5  = 66.40%
R@10 = 79.88%
```

`exp_04` w/o LTP 的 R@1 為 19.07%，兩者差距用來支持 target-conditioned synthetic proxy 在此設定中具有可利用性。

## 6. 執行延伸分析

論文後續分析腳本整理於：

```text
scripts/baselines/
scripts/robustness/
scripts/faithfulness/
scripts/significance/
scripts/analysis/
scripts/manual_review/
scripts/diagnostics/
```

本 repo 已保留小型摘要結果於 `results/`，可用來核對輸出格式、表格來源與分析流程。完整重跑仍需外部大型資料與 checkpoint。

## 7. 解讀結果時的注意事項

- `P_ltp` 是 target-track-conditioned synthetic preference proxy，不是真實使用者長期偏好。
- `exp_07` w/o Music 在候選音樂被移除後，候選間幾乎不可區分；論文中加入極小 random jitter 以避免固定索引造成假象。
- LTP zeroing 屬於 out-of-distribution intervention，解碼崩壞不能直接解讀為 LTP 的單一因果證據。
- MuseChat 原文結果只能作外部參照，因 backbone、compute 與評估條件不同，不應視為完全同條件比較。
