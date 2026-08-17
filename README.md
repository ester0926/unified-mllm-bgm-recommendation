# 基於合成長期偏好建模之整合式多模態大型語言模型應用於背景音樂推薦

英文題名：A Unified Multimodal Large Language Model Based on Synthetic Long-Term Preference Modeling for Background Music Recommendation

這是碩士論文最新版實驗的 GitHub 發布版。本資料夾只保留重現主實驗所需的程式、流程文件、設定與小型摘要結果；大型 MuseChat 特徵、模型權重、LTP cache 與原始輸入資料需另外下載或由原始流程產生。


## 研究範圍

- 任務：在給定影片、候選音樂與合成長期偏好代理訊號的條件下，進行背景音樂候選排序與推薦理由生成。
- 資料基礎：MuseChat 測試樣本與特徵。MuseChat 影片主要來自 YouTube music video，論文中將其作為可重現的背景音樂推薦實驗環境。
- 偏好訊號：`P_ltp` 是 target-track-conditioned synthetic cross-context preference proxy，不是真實使用者長期歷史。
- 主模型：Frozen CLIP/AST/LLaMA 2-7B backbone，加入 multimodal projectors、LoRA 與 ranking head。
- 主要實驗：`exp_01` 至 `exp_07`、baseline、robustness、significance、faithfulness/source attribution、metadata consistency、persona controllability 與 conflict analysis。

## 資料夾結構

```text
config.py             模型、訓練與資料路徑設定
dataset.py            MuseChat pair dataset 與候選資料讀取
app.py                互動式推薦服務入口
model_service.py      推薦推論服務
models/               Unified MLLM 與 projector 模組
data_utils/           LTP cache 建構輔助程式
scripts/              訓練、主評估、baseline、穩健性、faithfulness、統計與診斷腳本
user_profiling/       合成長期偏好代理訊號 P_ltp 的五階段建構流程
data_preparation/     MuseChat feature alignment 與 verified sample whitelist
results/              小型摘要結果、表格來源、稽核樣本與可重現性 manifest
data/                 外部資料放置位置說明，不含大型原始資料
cache/                LTP cache 放置位置說明，不含大型 .npy cache
checkpoints/          模型權重放置位置說明，不含 .pt/.safetensors 權重
```

## 建議閱讀順序

1. [DATA.md](DATA.md)：說明哪些大型資料需要另外準備。
2. [LTP_PIPELINE.md](LTP_PIPELINE.md)：說明 `P_ltp` 的五階段建構流程、輸入與輸出。
3. [PROGRAM_INDEX.md](PROGRAM_INDEX.md)：逐一對照每個程式的用途。
4. [EXPERIMENTS.md](EXPERIMENTS.md)：對照每個實驗條件與腳本。
5. [RESULTS_GUIDE.md](RESULTS_GUIDE.md)：說明 `results/` 中各資料夾與結果檔的用途。
6. [REPRODUCIBILITY.md](REPRODUCIBILITY.md)：完整重現主結果的執行順序。
7. [CHECKPOINTS.md](CHECKPOINTS.md)：說明模型權重與 cache 應如何另外保存。

## 快速開始

```bash
conda create -n bgm-recommender python=3.11
conda activate bgm-recommender
pip install -r requirements.txt
```

接著依照 [DATA.md](DATA.md) 放置 MuseChat HDF5 特徵、MuseChat JSON、User Profiling 輸出、LTP cache 與 checkpoints。若只是閱讀程式或核對結果表格，不需要先下載完整大型資料。

## 重要提醒

- `.h5`、`.npy`、`.pt`、`.safetensors`、影音檔與完整 checkpoint 不應直接放入 GitHub。
- 若要公開大型權重或 cache，建議使用 Git LFS、GitHub Release、Zenodo、Google Drive 或實驗室 NAS，並另外附上 checksum。
- 論文中的 LTP 結果代表此 synthetic proxy 在此評估設定中的可利用性，不等同於已驗證真實創作者長期偏好。
