# 結果資料說明

本 repo 保留部分結果摘要與人工稽核檔案，目的是讓讀者能追溯論文表格來源、理解輸出格式，並檢查主要分析流程。這裡不包含完整模型權重、完整 MuseChat features 或大型 LTP cache。

## 常見檔案類型

| 副檔名 | 意義 |
|---|---|
| `.json` | 機器可讀的摘要、設定、provenance 或指標輸出。 |
| `.csv` | 表格式指標、逐樣本 ranking/generation 輸出或人工標註表。 |
| `.jsonl` | line-delimited generated samples 或 counterfactual outputs。 |
| `.md` | 人類可讀的分析摘要。 |
| `.xlsx` | 小型人工檢查或 cluster naming workbook。這些檔案用來記錄人工檢查流程，不是人體研究回覆資料。 |

## Results 頂層資料夾

| 資料夾 | 功用 |
|---|---|
| `results/main_eval/` | `exp_01` 到 `exp_07` 的主實驗輸出，包含 detailed ranking samples 與 top-1 generation summary。 |
| `results/top1_end_to_end/` | ranking 後 top-1 candidate 的 explanation generation 輸出。 |
| `results/baselines/` | baseline retrieval/generation 輸出與 baseline comparison summary。 |
| `results/robustness/` | pool size、seed 與其他 robustness summary。 |
| `results/significance/` | ranking/generation 比較的統計檢定結果。 |
| `results/faithfulness/` | explanation faithfulness、source attribution、metadata consistency、counterfactual generation、LLM-as-judge 與人工檢查檔案。 |
| `results/analysis/` | 論文延伸分析，例如 persona controllability、preference-video conflict、video clusters、path-level generation 與 candidate difficulty。 |
| `results/diagnostics/` | dataset overlap 與 leakage diagnostics。 |

## 主實驗結果

| 資料夾 | 意義 |
|---|---|
| `results/main_eval/exp_01/` | 完整模型，使用 hybrid LTP、video、music、text。 |
| `results/main_eval/exp_02/` | explicit-only LTP condition。 |
| `results/main_eval/exp_03/` | implicit-only LTP condition。 |
| `results/main_eval/exp_04/` | no-LTP ablation。 |
| `results/main_eval/exp_05/` | no-video ablation。 |
| `results/main_eval/exp_06/` | no-text ablation。 |
| `results/main_eval/exp_07/` | no-music ablation。 |
| `results/main_eval/exp_01/ltp_control/` | mismatched/random/LTP-control 類型檢查。 |
| `results/main_eval/exp_01/persona_eval*` | persona controllability 輸出。 |
| `results/main_eval/exp_01/fixed_component_intervention_v21/` | fixed explicit/implicit LTP component intervention 輸出。 |
| `results/main_eval/exp_01/hardneg_eval/` | hard-negative candidate evaluation 輸出。 |

## Analysis 結果

| 資料夾 | 意義 |
|---|---|
| `results/analysis/candidate_difficulty/` | 依 candidate-pool difficulty 分層的效能分析。 |
| `results/analysis/path_level_generation/` | 較早版本的 path-level explanation analysis。 |
| `results/analysis/path_level_generation_v21/` | 最終 v21 論文版本使用的 path-level explanation analysis。 |
| `results/analysis/preference_counterfactual_by_path/` | 依 preference path 區分的 counterfactual effect 方向與強度。 |
| `results/analysis/fixed_hybrid_component_v21/` | fixed explicit/implicit component intervention 指標。 |
| `results/analysis/video_clusters/` | video cluster assignments、metrics、names 與 worksheet。 |
| `results/analysis/b5_personas/` | 較早版本的 persona controllability artifacts。 |
| `results/analysis/b5_personas_v21/` | 最終 v21 persona controllability artifacts。 |
| `results/analysis/b5_smoketest/` | persona/no-LTP recovery logic 的 smoke-test checks。 |
| `results/analysis/b6_conflict/` | 較早版本的 preference-video conflict analysis。 |
| `results/analysis/b6_conflict_v21/` | 最終 v21 preference-video conflict analysis。 |
| `results/analysis/v21_reproducibility_manifest.json` | 追蹤主要 v21 outputs 與 provenance 的 manifest。 |
| `results/analysis/v21_experiment_completion_review.json` | v21 experiments 完整性檢查。 |

## Faithfulness 結果

| 檔案或資料夾 | 意義 |
|---|---|
| `results/faithfulness/faithfulness_summary*.md` | claim support 與 unsupported claim rate 的人類可讀摘要。 |
| `results/faithfulness/counterfactual_generations*` | feature intervention/counterfactual settings 下產生的推薦解釋。 |
| `results/faithfulness/preference_counterfactual*` | preference counterfactual 檢查的輸出與摘要。 |
| `results/faithfulness/metadata_consistency_summary.md` | generated claims 的 metadata consistency analysis。 |
| `results/faithfulness/llm_judge/` | LLM-as-judge calibration 輸出。 |
| `results/faithfulness/manual_review/` | 人工檢查封包、抽查標註、agreement reports 與 failure-case candidates。 |
| `results/faithfulness/ucr_error_sources/` | unsupported claim rate 的錯誤來源分解與人工檢查 workbook。 |
| `results/faithfulness/preference_counterfactual_by_path/` | 依 preference representation path 區分的 counterfactual outputs。 |

## Baseline、Robustness、Significance 與 Diagnostics

| 資料夾 | 意義 |
|---|---|
| `results/baselines/similarity_retrieval/` | similarity retrieval baseline outputs。 |
| `results/baselines/llama_prompt_only/` | prompt-only generation baseline outputs。 |
| `results/baselines/musechat_light/` | MuseChat-light baseline outputs。 |
| `results/baselines/summary/` | baseline comparison summary。 |
| `results/robustness/` | seed 與 pool-size robustness summaries。 |
| `results/significance/gt_conditioned/` | ground-truth-conditioned comparisons 的 significance tests。 |
| `results/significance/top1_end_to_end/` | end-to-end top-1 generation comparisons 的 significance tests。 |
| `results/diagnostics/` | train/test overlap 與 leakage-related diagnostics。 |

## 建議閱讀方式

1. 先看 `results/main_eval/exp_01/detailed_eval/`，理解 full model 的輸出格式。
2. 比較 `exp_01` 與 `exp_04`，理解 LTP 在主要 ranking task 中的角色。
3. 看 `results/robustness/` 與 `results/significance/`，確認效果是否穩定且具統計支持。
4. 看 `results/faithfulness/`，檢查生成解釋是否有輸入證據支持。
5. 看 `results/analysis/b5_personas_v21/` 與 `results/analysis/b6_conflict_v21/`，理解最終論文版本中的 controllability 與 preference-video conflict 分析。

這些結果檔主要用於 traceability 與 documentation。完整重現仍需要 [DATA.md](DATA.md) 與 [CHECKPOINTS.md](CHECKPOINTS.md) 中說明的外部資料與 checkpoint。
