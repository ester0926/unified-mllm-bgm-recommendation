# 實驗對照表

本文件對照論文中的主要實驗條件、用途與程式位置。實際路徑若因本機資料夾不同而失效，請先依 [DATA.md](DATA.md) 調整資料路徑。

## 主實驗

| ID | 條件 | 用途 | 訓練程式 | 預期 checkpoint |
|---|---|---|---|---|
| exp_01 | hybrid P_ltp + video + music + text | 完整模型 | `scripts/train/run_train.py` | `checkpoints/exp_01/best/` |
| exp_02 | explicit_only P_ltp | 測試文字/profile 偏好路徑 | `scripts/train/run_train_exp02.py` | `checkpoints/exp_02/best/` |
| exp_03 | implicit_only P_ltp | 測試 audio-behavior 偏好路徑 | `scripts/train/run_train_exp03.py` | `checkpoints/exp_03/best/` |
| exp_04 | w/o P_ltp | 移除長期偏好代理訊號 | `scripts/train/run_train_exp04.py` | `checkpoints/exp_04/best/` |
| exp_05 | w/o Video | 移除影片情境 | `scripts/train/run_train_exp05.py` | `checkpoints/exp_05/best/` |
| exp_06 | w/o Text | 移除文字條件 | `scripts/train/run_train_exp06.py` | `checkpoints/exp_06/best/` |
| exp_07 | w/o Music | 移除候選音樂訊號 | `scripts/train/run_train_exp07.py` | `checkpoints/exp_07/best/` |

## 主評估

| 評估 | 程式 |
|---|---|
| 500-pool ranking/generation | `scripts/eval_main/run_eval_500pool.py` |
| detailed 500-pool evaluation | `scripts/eval_main/run_eval_500pool_detailed.py` |
| ranking 後 top-1 generation | `scripts/eval_main/run_eval_500pool_top1_generation_from_ranking.py` |
| LTP control | `scripts/eval_main/run_eval_500pool_ltp_control.py` |
| persona evaluation | `scripts/eval_main/run_eval_500pool_persona.py`, `scripts/eval_main/run_eval_500pool_persona_v2.py` |
| fixed hybrid component evaluation | `scripts/eval_main/run_eval_fixed_hybrid_components_v21.py` |

## Baseline

| Baseline | 程式 |
|---|---|
| similarity retrieval | `scripts/baselines/run_baseline_similarity_retrieval_500pool.py` |
| LLaMA prompt-only generation | `scripts/baselines/run_baseline_llama_prompt_only_top1_generation.py` |
| MuseChat Light | `scripts/baselines/run_musechat_light_eval_500pool.py` |
| MuseChat + LTP | `scripts/baselines/run_musechat_ltp_eval_500pool.py` |
| baseline summary | `scripts/baselines/summarize_baseline_results.py` |

## Robustness

| 分析 | 程式 |
|---|---|
| pool size robustness | `scripts/robustness/run_eval_500pool_top1_pool100.py`, `scripts/robustness/run_eval_500pool_top1_pool1000.py` |
| seed robustness | `scripts/robustness/run_eval_500pool_top1_seed42.py`, `scripts/robustness/run_eval_500pool_top1_seed12345.py`, `scripts/robustness/run_eval_500pool_top1_seed987654.py` |
| robustness summary | `scripts/robustness/analyze_pool_robustness_top1.py`, `scripts/robustness/analyze_seed_robustness_top1.py` |
| prompt robustness | `scripts/prompt_variants/` |

## Faithfulness 與解釋分析

| 分析 | 程式 |
|---|---|
| claim judge | `scripts/faithfulness/faithfulness_claim_judge*.py` |
| preference counterfactual | `scripts/faithfulness/run_preference_counterfactual_generation*.py` |
| LLM-as-judge calibration | `scripts/faithfulness/llm_as_judge_faithfulness*.py` |
| metadata consistency | `scripts/faithfulness/analyze_metadata_consistency*.py` |
| manual review packets | `scripts/manual_review/` |

## 統計與診斷分析

| 分析 | 程式 |
|---|---|
| significance tests | `scripts/significance/analyze_significance*.py` |
| Cliff's delta | `scripts/analysis/cliffs_delta_significance.py` |
| path-level generation | `scripts/analysis/path_level_generation_analysis*.py` |
| persona controllability | `scripts/analysis/b5_*persona*.py` |
| preference-video conflict | `scripts/analysis/b6_preference_video_conflict*.py` |
| train/test overlap diagnostics | `scripts/diagnostics/` |
