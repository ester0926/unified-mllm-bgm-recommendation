# 清理決策與交接範圍

本文件說明 `github_release` 的保留與移除原則。整理基準為最新論文檔案：

`E:\YUYING_final\unified_mllm_pointwise_final\docs\論文_口試後修訂_v81_附錄表加入表目次_追蹤修訂版.docx`

## GitHub 發布版保留原則

GitHub 版本只保留能協助學弟妹理解與重現論文主要實驗的內容：

- 論文第 4 章與附錄 C、D 對應的訓練、評估、baseline、robustness、significance、faithfulness、metadata consistency、persona controllability、conflict analysis 程式。
- `v21` 系列的附錄分析程式與輸出，包含 path-level generation、persona controllability、preference-video conflict、fixed component intervention。
- Stage 3 到 Stage 5 的 synthetic long-term preference 建構流程。
- 小型 metadata、ID 清單、實驗摘要、統計結果與 reproducibility manifest。
- 說明文件、程式索引與 Zenodo 資料包對應說明。

## 已從 GitHub 發布版移除的內容

以下內容不是論文最後版本，或屬於測試、舊版、問卷與開發過程資料，因此不放入 GitHub 發布版：

- 問卷與 IRB 相關資料或腳本。
- 舊版 path-level generation 程式與舊輸出：
  - `scripts/analysis/path_level_generation_analysis.py`
  - `results/analysis/path_level_generation/`
- 舊版 preference counterfactual by path 程式與舊輸出：
  - `scripts/analysis/preference_counterfactual_by_path_analysis.py`
  - `scripts/faithfulness/run_preference_counterfactual_by_path.py`
  - `results/analysis/preference_counterfactual_by_path/`
  - `results/faithfulness/preference_counterfactual_by_path/`
- 舊版 persona controllability 程式與 smoke test：
  - `scripts/analysis/b5_persona_metrics.py`
  - `scripts/analysis/b5_persona_metrics_v2.py`
  - `scripts/analysis/b5_smoketest_recover_wout.py`
  - `results/analysis/b5_personas/`
  - `results/analysis/b5_smoketest/`
- 舊版 conflict analysis：
  - `scripts/analysis/b6_preference_video_conflict.py`
  - `results/analysis/b6_conflict/`
- 舊版 faithfulness wrapper：
  - `scripts/faithfulness/faithfulness_claim_judge_faithful.py`
  - `scripts/faithfulness/faithfulness_claim_judge_top1.py`
  - `scripts/faithfulness/analyze_faithfulness_faithful.py`
  - `scripts/faithfulness/analyze_faithfulness_top1.py`
  - `scripts/faithfulness/run_faithfulness_counterfactual_faithful.py`
- 早期診斷或測試用程式：
  - `scripts/diagnostics/evaluate_leakage.py`
  - `scripts/diagnostics/infolm_test.py`
  - `scripts/diagnostics/run_ablation_50.py`
  - `scripts/diagnostics/run_ablation_all.py`
  - `scripts/baselines/run_exp08_ltp_eval_500pool.py`

## 版本判斷規則

遇到同名或近似命名腳本時，請優先採用下列規則：

- 若同時存在一般版與 `_v21`，以 `_v21` 為論文附錄最後版本。
- 若同時存在 `top1` 與 `top1_v2`，以 `top1_v2` 為最後整理版本。
- 若檔名含 `smoketest`、`recover`、`leakage`、`infolm_test`、`ablation_50`，通常代表測試或診斷用途，不作為論文主結果流程。
- 若程式或輸出只服務問卷、IRB、論文 Word 修訂或排版，不屬於 GitHub 發布版。

## Zenodo 與硬碟資料

大型模型權重、LTP cache、metadata 與 ID 清單已整理為 Zenodo 壓縮包，GitHub 只保留下載與驗證說明。原始硬碟資料夾則保留較完整的研究過程資料，但不建議直接整包上傳。

硬碟交接資料夾中若看到 `_archive_not_for_handoff_20260820`，表示該處內容保留作為研究歷史或備查，不是學弟妹重現論文時的第一入口。
