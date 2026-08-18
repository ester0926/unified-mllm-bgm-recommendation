# 基於合成長期偏好建模之整合式多模態大型語言模型應用於背景音樂推薦

英文題名：A Unified Multimodal Large Language Model Based on Synthetic Long-Term Preference Modeling for Background Music Recommendation

本 repository 為碩士論文實驗的 GitHub 發布版，整理主實驗所需的程式、流程文件、設定檔與小型摘要結果。大型 MuseChat 特徵、模型權重、LTP cache 與原始輸入資料未納入版本控制，需依文件另外放置或重新產生。

## 專案重點

- 任務：根據短影片內容、候選音樂與合成長期偏好訊號，進行背景音樂推薦與推薦理由生成。
- 資料基礎：以 MuseChat HDF5 features 與 MuseChat JSON metadata 作為主要輸入。
- 偏好訊號：`P_ltp` 為 target-track-conditioned synthetic cross-context preference proxy。
- 模型架構：Frozen CLIP/AST/LLaMA 2-7B backbone，搭配 multimodal projectors、LoRA 與 ranking head。
- 實驗範圍：`exp_01` 至 `exp_07`、baseline、robustness、significance、faithfulness/source attribution、metadata consistency、persona controllability 與 conflict analysis。

## 資料夾結構

```text
config.py             模型、訓練與資料路徑設定
dataset.py            MuseChat pair dataset 與特徵讀取
app.py                Streamlit demo 前端
model_service.py      FastAPI 推論服務
models/               Unified MLLM 與 projector 模組
data_utils/           LTP cache 建立工具
scripts/              訓練、評估、baseline、robustness、faithfulness 與分析程式
user_profiling/       建立 synthetic P_ltp 的多階段流程
data_preparation/     MuseChat feature alignment 與 verified sample whitelist
results/              小型摘要結果、分析輸出與 reproducibility manifest
data/                 外部資料放置位置，僅保留 README
cache/                cache 放置位置，僅保留 README
checkpoints/          checkpoint 放置位置，僅保留 README
```

## 閱讀順序

1. [DATA.md](DATA.md)：外部資料與預期目錄結構。
2. [LTP_PIPELINE.md](LTP_PIPELINE.md)：`P_ltp` 建構流程與中間輸出。
3. [PROGRAM_INDEX.md](PROGRAM_INDEX.md)：主要程式功能與對應實驗。
4. [EXPERIMENTS.md](EXPERIMENTS.md)：論文實驗條件與程式對照。
5. [RESULTS_GUIDE.md](RESULTS_GUIDE.md)：`results/` 中摘要結果與分析檔案的用途。
6. [REPRODUCIBILITY.md](REPRODUCIBILITY.md)：重現主實驗的建議流程。
7. [CHECKPOINTS.md](CHECKPOINTS.md)：大型權重、cache 與 checkpoint 的放置方式。
8. [ZENODO.md](ZENODO.md)：Zenodo 下載包、解壓方式與 checksum 驗證。

## 環境安裝

```bash
conda create -n bgm-recommender python=3.11
conda activate bgm-recommender
pip install -r requirements.txt
```

`requirements.txt` 依原始 conda 實驗環境整理。PyTorch CUDA 版本、OpenAI CLIP 與部分選用分析套件可能需要依實際 GPU、CUDA 與網路環境調整。

## 外部資料與大型檔案

本 repo 僅保存可版本控制的程式與小型結果摘要。模型權重、LTP cache、metadata 與 ID 清單以 Zenodo 下載包提供；下載與解壓方式見 [ZENODO.md](ZENODO.md)。完整 MuseChat HDF5 features 與原始影音檔未包含於 Zenodo v1 重現包。
