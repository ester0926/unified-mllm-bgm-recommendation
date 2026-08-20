# 執行順序總覽

這份文件用來回答「拿到程式後應該先跑哪一支、再跑哪一支」。若只是要重現論文表格與檢查流程，建議先下載 Zenodo 壓縮包並使用既有 checkpoint 與 cache；只有在要重新產生 synthetic long-term preference proxy 時，才需要從 User Profiling 的 Stage 0 跑起。

## 0. 先決定重現範圍

| 目標 | 建議做法 |
|---|---|
| 檢查程式結構、閱讀實驗流程 | 先看 `README.md`、`DATA.md`、`PROGRAM_INDEX.md`、`THESIS_ALIGNMENT.md` |
| 重現論文主要評估 | 下載 Zenodo 檔案，放好 `cache/`、`checkpoints/`、`data/user_profiling/` 後跑評估與分析 |
| 完整重建 LTP | 從 `user_profiling/stage0_data_prep/` 到 `user_profiling/stage5/` 依序執行 |
| 重新訓練模型 | 確認 MuseChat features、LTP cache 與 split cache 都存在後，執行 `scripts/train/` |

## 1. 建立環境

```powershell
conda create -n bgm-recommender python=3.11
conda activate bgm-recommender
pip install -r requirements.txt
```

若要載入 `meta-llama/Llama-2-7b-hf`，請先完成 Hugging Face 權限申請與登入。若 GPU、CUDA 或 PyTorch 版本不同，請以實際機器可安裝的 PyTorch CUDA 版本為準。

## 2. 放置外部資料

先依照 `ZENODO.md` 解壓三個 Zenodo 壓縮包。解壓後 repo 根目錄應至少有：

```text
checkpoints/exp_01/best/
checkpoints/exp_02/best/
...
checkpoints/exp_07/best/
cache/ltp_hybrid.npy
cache/ltp_hybrid_ids.json
cache/ltp_explicit_only.npy
cache/ltp_explicit_only_ids.json
cache/ltp_implicit_only.npy
cache/ltp_implicit_only_ids.json
data/user_profiling/
data/musechat_split_cache.json
data/video_ids_from_hdf5_*.json
```

完整 MuseChat HDF5 features 未放在 GitHub 或 Zenodo v1。若要重跑訓練或完整評估，請另依 `DATA.md` 放到：

```text
data/optimized_musechat_features_float16_v3/
data/musechat_json/
```

## 3. 建立或重建 LTP

如果已使用 Zenodo 的 LTP cache，可跳過本節。若要從頭重建，順序如下：

```powershell
python user_profiling/stage1_metadata.py
python user_profiling/stage2_personax.py
python user_profiling/stage3/stage3_dialogue_optimized.py
python user_profiling/stage3/stage3.4_diagnose_failures.py
python user_profiling/stage3/stage3.5_repair_missing.py
python user_profiling/stage3/stage3.6_improved_post_evaluation.py
python user_profiling/stage4/stage4.1_pilot_experiment.py
python user_profiling/stage4/stage4_recllm.py
python user_profiling/stage4/stage4.2_consistency_eval.py
python user_profiling/stage5/stage5_preference_representation_v4.py
python data_utils/build_ltp.py
```

`stage0_data_prep/` 是早期用來整理音樂標籤與 YouTube metadata 的前處理工具；若 Zenodo 已提供整理後 metadata，通常不需要重跑。每個 Stage 的輸入輸出細節請看 `LTP_PIPELINE.md`。

## 4. 訓練七個主要模型條件

若只要重現結果，可使用 Zenodo checkpoint 跳過本節。若要重新訓練，從 repo 根目錄執行：

```powershell
python scripts/train/run_train.py
python scripts/train/run_train_exp02.py
python scripts/train/run_train_exp03.py
python scripts/train/run_train_exp04.py
python scripts/train/run_train_exp05.py
python scripts/train/run_train_exp06.py
python scripts/train/run_train_exp07.py
```

實驗條件對照如下：

| 實驗 | 條件 |
|---|---|
| `exp_01` | full model with hybrid `P_ltp` |
| `exp_02` | explicit-only `P_ltp` |
| `exp_03` | implicit-only `P_ltp` |
| `exp_04` | without `P_ltp` |
| `exp_05` | without video input |
| `exp_06` | without text input |
| `exp_07` | without candidate music input |

訓練完成後，主要權重應整理在 `checkpoints/exp_*/best/`。

## 5. 主評估與論文表格

主評估使用 random 500-candidate pool。建議先跑 full model，確認資料與 checkpoint 都能正常讀取，再跑其他條件。

```powershell
python scripts/eval_main/run_eval_500pool.py
python scripts/eval_main/run_eval_500pool_detailed.py
python scripts/eval_main/run_eval_500pool_top1_generation_from_ranking.py
```

其他 ablation、control 與 persona 評估在 `scripts/eval_main/` 中。每支程式對應的論文表格請看 `THESIS_ALIGNMENT.md`。

## 6. Baseline、穩健性與統計檢定

依論文需要執行：

```powershell
python scripts/baselines/run_baseline_similarity_retrieval_500pool.py
python scripts/baselines/run_musechat_light_eval_500pool.py
python scripts/baselines/run_musechat_ltp_eval_500pool.py
python scripts/significance/analyze_significance.py
python scripts/significance/analyze_significance_top1.py
python scripts/robustness/analyze_pool_robustness_top1.py
python scripts/robustness/analyze_seed_robustness_top1.py
```

單一 seed、pool size 或 prompt variant 的評估腳本很多，建議先對照 `PROGRAM_INDEX.md` 與 `THESIS_ALIGNMENT.md`，只跑論文需要的條件。

## 7. Faithfulness 與補充分析

推薦理由、metadata consistency、preference counterfactual、UCR 人工查核與 persona controllability 相關程式分散在：

```text
scripts/faithfulness/
scripts/manual_review/
scripts/analysis/
```

其中 `b6_preference_video_conflict_v21.py` 是論文表 4-41 的偏好與影片語意衝突補充分析，不是問卷程式。`write_reproducibility_manifest.py` 用來產生 `results/analysis/REPRODUCIBILITY_MANIFEST.json`，記錄補充分析與封存檔案的 provenance。

## 8. 核對輸出

重跑後請依序檢查：

```text
results/main_eval/
results/baselines/
results/robustness/
results/faithfulness/
results/analysis/
```

如果只要核對論文數字，先看 `RESULTS_GUIDE.md`；如果要知道某個結果對應論文哪張表，先看 `THESIS_ALIGNMENT.md`。
