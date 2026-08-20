# 程式與論文內容對應表

本文件用來判斷 `github_release` 中的程式與資料是否對應最新版碩論內容。整理基準為 v81 論文：

`E:\YUYING_final\unified_mllm_pointwise_final\docs\論文_口試後修訂_v81_附錄表加入表目次_追蹤修訂版.docx`

## 判斷類別

- **主實驗**：直接產生或整理第 4 章表格中的結果。
- **附錄/補充分析**：第 4 章、附錄 C 或附錄 D 有提及，通常是 v21 補充實驗或可重現性資料。
- **資料建構流程**：第 3 章方法流程的一部分，負責產生 LTP、metadata 或 profile。
- **重現輔助**：不直接產生論文表格，但用於產生 manifest、人工抽查資料或封存來源。
- **已封存/不保留**：問卷、smoketest、舊版 v/v2 或自我檢查程式，不屬於 GitHub 發布版入口。

## 第 3 章資料建構流程

| 論文位置 | 程式/資料 | 類別 | 說明 |
|---|---|---|---|
| 圖 3-2、3.2 節 | `user_profiling/stage0_data_prep/` | 資料建構流程 | MusicNN tags、YouTube metadata 與音樂 metadata 前處理。 |
| 圖 3-2、圖 3-3、表 3-3 | `user_profiling/stage1_metadata.py`, `user_profiling/stage2_personax.py` | 資料建構流程 | 建立 enriched metadata 與 PersonaX-style sampling。 |
| 圖 3-2、表 4-1 | `user_profiling/stage3/` | 資料建構流程 | 產生與檢查 synthetic preference dialogues。 |
| 表 4-2 | `user_profiling/stage4/stage4.1_pilot_experiment.py` | 主實驗 | Stage 4 profile extraction 模型選型。 |
| 第 3.2.5、表 4-2 | `user_profiling/stage4/stage4_recllm.py`, `stage4.2_consistency_eval.py` | 資料建構流程 | 從 Stage 3 對話萃取 profiles，並做 consistency evaluation。 |
| 第 3.2.6、圖 4-1、附錄 C | `user_profiling/stage5/stage5_preference_representation_v4.py` | 資料建構流程 | 產生 `preference_vectors*.h5` 與 `projection_weights.pt`。 |

## 第 4 章主實驗與評估

| 論文表/圖 | 程式/資料 | 類別 | 說明 |
|---|---|---|---|
| 表 4-3 | `user_profiling/experiments/synthetic_to_real_validity_analysis.py` | 主實驗 | 合成長期偏好品質、metadata grounding、positive/negative alignment 與跨模型穩定性。 |
| 圖 4-1、圖 4-2、圖 4-3 | `scripts/analysis/tsne_umap_ltp.py` | 主實驗 | LTP embedding 的 UMAP/t-SNE 視覺化。 |
| 表 4-4、表 4-5 | `scripts/train/run_train*.py`, `scripts/eval_main/run_eval_500pool*.py` | 主實驗 | exp_01 到 exp_07 訓練與 500-pool 評估。 |
| 表 4-6、表 4-7、表 4-22 | `scripts/significance/analyze_significance.py` | 主實驗 | 主實驗、LTP 消融與訓練模態消融的顯著性檢定。 |
| 表 4-8、表 4-9 | `scripts/eval_main/run_eval_500pool_ltp_control.py`, `scripts/analysis/ltp_control_significance.py` | 主實驗 | LTP matched、shuffled、random control。 |
| 表 4-10 | `scripts/analysis/path_level_generation_analysis_v21.py` | 主實驗/補充分析 | 四模型生成說明之偏好主張與證據代理指標。這是 v21 最終版，不是問卷程式。 |
| 表 4-11 | `scripts/eval_main/run_eval_fixed_hybrid_components_v21.py`, `scripts/analysis/fixed_hybrid_component_analysis_v21.py` | 補充分析 | 固定 exp_01 模型後移除 explicit/implicit component 的介入分析。 |
| 表 4-12 | `scripts/faithfulness/run_preference_counterfactual_generation*.py`, `scripts/faithfulness/analyze_preference_counterfactual*.py` | 主實驗 | 提示詞層級偏好方向敏感度。 |
| 表 4-13、表 4-14 | `scripts/baselines/` | 主實驗 | random/similarity/MuseChat-light/LLaMA prompting baselines。 |
| 表 4-15、表 4-16 | `scripts/baselines/run_musechat_ltp_eval_500pool.py`, `run_musechat_light_eval_500pool.py` | 主實驗 | 與 MuseChat 外部報告與重現 baseline 對照。 |
| 表 4-19 | `scripts/eval_main/run_eval_500pool_detailed.py` | 主實驗 | 推論時模態消融分析。 |
| 表 4-20、表 4-21 | `scripts/train/run_train_exp04.py` 到 `run_train_exp07.py` | 主實驗 | 訓練模態消融。 |
| 表 4-23 | `scripts/eval_main/run_eval_500pool_hardneg.py` | 主實驗 | 困難負例候選池評估。 |
| 表 4-24 | `scripts/analysis/candidate_difficulty_stratification.py` | 主實驗 | 候選音訊難度分層與 DiD 檢定。 |
| 表 4-25、表 4-26 | `scripts/robustness/run_eval_500pool*_pool*.py`, `analyze_pool_robustness_top1.py` | 主實驗 | 候選池大小穩健性。 |
| 表 4-27 | `scripts/prompt_variants/` | 主實驗 | prompt variants generation robustness。 |
| 表 4-28、表 4-29 | `scripts/robustness/run_eval_500pool*_seed*.py`, `analyze_seed_robustness_top1.py` | 主實驗 | 評估候選池抽樣種子穩健性。 |
| 表 4-30、表 4-31 | `scripts/diagnostics/dataset_overlap_analysis.py`, `compute_median_rank_unseen_seen.py`, `diagnose_exp07_scores.py` | 主實驗 | train/test overlap 與 seen/unseen candidate analysis。 |
| 表 4-32、表 4-33 | `scripts/faithfulness/run_faithfulness_counterfactual*.py`, `faithfulness_claim_judge*.py`, `analyze_faithfulness*.py` | 主實驗 | feature-erasure faithfulness/source attribution。 |
| 表 4-34 | `scripts/faithfulness/analyze_preference_counterfactual*.py` | 主實驗 | prompt preference counterfactual direction consistency。 |
| 表 4-35 | `scripts/faithfulness/analyze_metadata_consistency*.py` | 主實驗 | metadata unsupported claim rate。 |
| 表 4-36 | `scripts/faithfulness/llm_as_judge_faithfulness*.py` | 主實驗 | LLM-as-Judge 與規則式評估一致率。 |
| 表 4-37 | `scripts/faithfulness/make_ucr_review_workbook.py`, `score_ucr_human_review.py`, `analyze_ucr_error_sources.py` | 主實驗 | 生成說明失敗型態與 UCR error source analysis。 |
| 表 4-38 | `scripts/analysis/video_cluster_profile.py`, `video_cluster_finalize.py`, `video_cluster_stratification.py` | 主實驗 | 影片語義叢集分層之 LTP 排序增益。 |
| 表 4-39、表 4-40、附錄 D | `scripts/analysis/b5_build_persona_specs.py`, `b5_build_persona_ltp.py`, `b5_persona_metrics_v21.py`, `reuse_noltp_v2_for_persona_v21.py` | 補充分析 | 結構化 Persona 偏好可控性與反事實方向可控性。 |
| 表 4-41 | `scripts/analysis/b6_preference_video_conflict_v21.py` | 補充分析 | 偏好與影片相容度分組之原始正確配對結果。這是論文 v21 補充分析，不是問卷程式。 |

## 附錄與重現輔助

| 論文位置 | 程式/資料 | 類別 | 說明 |
|---|---|---|---|
| 附錄 C | `scripts/analysis/write_v21_reproducibility_manifest.py` | 重現輔助 | 產生 `results/analysis/v21_reproducibility_manifest.json`。此程式不直接產生表格，但附錄 C 明確提到 manifest，因此保留。 |
| 附錄 C | `results/analysis/v21_reproducibility_manifest.json` | 重現輔助 | 記錄 v21 補充分析與封存檔案的來源、大小與 checksum。 |
| 附錄 D | `results/analysis/b5_personas_v21/persona_specs.*` | 補充分析 | 結構化偏好原型規格。 |
| 人工覆核/校準 | `scripts/analysis/prepare_v21_preference_claim_blind_audit.py`, `scripts/manual_review/prepare_manual_review_packets.py` | 重現輔助 | 用於建立人工抽查或盲審資料，不是問卷研究，也不是主要執行入口。 |

## 已移除或封存的項目

以下項目不是 v81 論文最後版本，已從 GitHub 發布版移除；硬碟完整資料夾中若仍需追查，可到 `_archive_not_for_handoff_20260820/` 查看。

| 項目 | 原因 |
|---|---|
| 問卷與 IRB 相關程式/資料 | 未納入 v81 論文主實驗與附錄資料可用性範圍。 |
| `path_level_generation_analysis.py` | 舊版；最終版為 `path_level_generation_analysis_v21.py`。 |
| `b5_persona_metrics.py`, `b5_persona_metrics_v2.py` | 舊版；最終版為 `b5_persona_metrics_v21.py`。 |
| `b6_preference_video_conflict.py` | 舊版；最終版為 `b6_preference_video_conflict_v21.py`。 |
| `b5_smoketest_recover_wout.py` 與 `results/analysis/b5_smoketest/` | smoketest/限制驗證用途；限制已寫入論文附錄 C 與 `REPRODUCIBILITY.md`。 |
| `review_v21_experiment_completion.py`, `v21_experiment_completion_review.json` | 整理過程自我檢查，不是論文表格、圖、附錄 C manifest 或重現必要輸入。 |
| `evaluate_leakage.py`, `infolm_test.py`, `run_ablation_50.py`, `run_ablation_all.py` | 早期診斷或測試腳本，不是 v81 最終實驗入口。 |
