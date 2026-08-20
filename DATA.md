# 資料準備說明

本 repo 不包含完整資料集。原因是 MuseChat 特徵、影音資料、LTP cache 與 checkpoint 體積過大，不適合直接放入 GitHub。重現時請依照下列結構自行放置資料，或依 [LTP_PIPELINE.md](LTP_PIPELINE.md) 重新產生中間輸出。

部分外部資料已整理為 Zenodo 下載包。下載、解壓與 checksum 驗證方式請見 [ZENODO.md](ZENODO.md)。

## 需要另外準備的資料

### 1. MuseChat HDF5 features

原始實驗使用的 MuseChat feature 約 1.64 TB，包含大量 `.h5` 檔，不放入 GitHub。發布版預期放置位置如下：

```text
data/optimized_musechat_features_float16_v3/
```

### 2. MuseChat JSON 與 metadata

MuseChat JSON/metadata 不直接放入 GitHub。發布版預期放置位置如下：

```text
data/musechat_json/
```

### 3. User Profiling 輸出

LTP 建構流程會產生音樂 metadata、synthetic dialogues、RecLLM profiles 與 preference representation。若不重新跑完整流程，可直接放入已產生的中間結果，例如：

```text
data/user_profiling/music_metadata_enriched.json
data/user_profiling/youtube_metadata.jsonl
data/user_profiling/profiles.jsonl
```

完整流程請看 [LTP_PIPELINE.md](LTP_PIPELINE.md)。

### 4. LTP cache

主模型訓練與評估會讀取 LTP cache。因 `.npy` 檔可能很大，GitHub 只保留說明，不保留實際檔案。

```text
cache/ltp_hybrid.npy
cache/ltp_hybrid_ids.json
cache/ltp_explicit_only.npy
cache/ltp_explicit_only_ids.json
cache/ltp_implicit_only.npy
cache/ltp_implicit_only_ids.json
```

### 5. Checkpoints

每個實驗條件的模型權重需另外保存。GitHub 只保留 `checkpoints/README.md` 與 [CHECKPOINTS.md](CHECKPOINTS.md)。

```text
checkpoints/exp_01/best/
checkpoints/exp_02/best/
checkpoints/exp_03/best/
checkpoints/exp_04/best/
checkpoints/exp_05/best/
checkpoints/exp_06/best/
checkpoints/exp_07/best/
```

## 已保留於 repo 的小型資料

```text
data_preparation/feature_alignment/valid_training_ids_v2.4_verified.json
results/analysis/v21_reproducibility_manifest.json
results/main_eval/
results/baselines/
results/robustness/
results/significance/
results/faithfulness/
results/diagnostics/
```

這些檔案主要用於核對論文表格、檢查輸出格式與追蹤實驗來源。它們不是完整資料集，也不能取代大型 MuseChat feature、LTP cache 或 checkpoint。

## 不放入本 repo 的資料

- 原始影音檔與完整 HDF5 features
- 模型權重、LoRA adapter、projector、ranking head
- 大型 `.npy`/`.npz` cache
