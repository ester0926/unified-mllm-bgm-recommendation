# 程式索引

這份文件用來快速說明本 repo 內每個主要程式的用途。建議後續研究者或實驗室成員先看 [README.md](README.md)、[DATA.md](DATA.md)、[LTP_PIPELINE.md](LTP_PIPELINE.md)，再回到本文件查找各程式負責的實驗步驟。

執行訓練或評估前，請先依照 [DATA.md](DATA.md) 放好外部資料，並依照 [CHECKPOINTS.md](CHECKPOINTS.md) 放好模型權重。完整重現順序請看 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 核心執行程式

| 檔案 | 功用 |
|---|---|
| `config.py` | 定義模型、訓練、模態、資料路徑、LoRA、ranking 與 generation 相關設定。 |
| `dataset.py` | 建立 MuseChat pair dataset、候選池、特徵讀取、LTP 查找與訓練/驗證/測試切分邏輯。 |
| `project_paths.py` | 管理 repo 相對路徑，例如 `scripts/`、`cache/`、`checkpoints/`、`results/`。 |
| `evaluate.py` | 共用評估函式入口，讓訓練與評估腳本中的 `from evaluate import ...` 可以在發布版目錄正常運作。 |
| `utils.py` | 共用工具函式，包含指標計算、檢索、文字處理與資料處理輔助。 |
| `mc_neg_bank.py` | 建立或讀取音樂候選負樣本 cache。 |
| `app.py` | 互動式推薦應用程式入口。 |
| `model_service.py` | 載入 checkpoint，提供推薦與推論服務邏輯。 |
| `data_utils/build_ltp.py` | 將 LTP 表示輸出轉成主模型可讀取的 cache。 |
| `models/projectors.py` | 將 video、music、text、LTP features 投影到 LLM hidden space 的 multimodal projector。 |
| `models/unified_mllm.py` | Unified MLLM 主架構，包含 multimodal prefix、ranking head 與 generation 介面。 |

## 訓練程式

| 檔案 | 功用 |
|---|---|
| `scripts/train/train.py` | 共用訓練迴圈、loss 計算、checkpoint 儲存與 epoch 邏輯。 |
| `scripts/train/run_train.py` | 執行 `exp_01`，完整 hybrid LTP 模型。 |
| `scripts/train/run_train_exp02.py` | 執行 `exp_02`，只使用 explicit LTP。 |
| `scripts/train/run_train_exp03.py` | 執行 `exp_03`，只使用 implicit LTP。 |
| `scripts/train/run_train_exp04.py` | 執行 `exp_04`，移除 LTP。 |
| `scripts/train/run_train_exp05.py` | 執行 `exp_05`，移除 video input。 |
| `scripts/train/run_train_exp06.py` | 執行 `exp_06`，移除 text input。 |
| `scripts/train/run_train_exp07.py` | 執行 `exp_07`，移除 candidate music input。 |

## 主評估程式

| 檔案 | 功用 |
|---|---|
| `scripts/eval_main/generate_recommendation.py` | 使用訓練好的 checkpoint 對指定輸入產生推薦與解釋。 |
| `scripts/eval_main/run_eval_500pool.py` | 執行主要 500-candidate ranking/generation 評估。 |
| `scripts/eval_main/run_eval_500pool_detailed.py` | 輸出詳細 500-pool ranking samples 與逐樣本結果。 |
| `scripts/eval_main/run_eval_500pool_hardneg.py` | 評估 hard-negative candidate pool。 |
| `scripts/eval_main/run_eval_500pool_ltp_control.py` | 執行 LTP control，例如 random 或 mismatched preference vector。 |
| `scripts/eval_main/run_eval_500pool_persona.py` | 執行 persona 條件評估。 |
| `scripts/eval_main/run_eval_500pool_persona_v2.py` | 執行 persona controllability v2 評估。 |
| `scripts/eval_main/run_eval_500pool_top1_generation_from_ranking.py` | 針對 ranking top-1 candidate 產生推薦解釋。 |
| `scripts/eval_main/run_eval_fixed_hybrid_components_v21.py` | 測試 explicit/implicit LTP component 固定介入。 |

## Baseline 程式

| 檔案 | 功用 |
|---|---|
| `scripts/baselines/run_baseline_similarity_retrieval_500pool.py` | 評估 feature similarity retrieval baseline。 |
| `scripts/baselines/run_baseline_llama_prompt_only_top1_generation.py` | 評估 LLaMA prompt-only 推薦解釋 baseline。 |
| `scripts/baselines/run_musechat_light_eval_500pool.py` | 評估輕量版 MuseChat-style baseline。 |
| `scripts/baselines/train_musechat_light_generator_current_data.py` | 使用目前資料格式訓練輕量 MuseChat-style generator。 |
| `scripts/baselines/run_musechat_ltp_eval_500pool.py` | 在 MuseChat-light baseline 設定下加入 LTP 訊號。 |
| `scripts/baselines/summarize_baseline_results.py` | 將 baseline 結果整理成摘要表。 |

## Robustness 與 Prompt Variant 程式

| 檔案 | 功用 |
|---|---|
| `scripts/robustness/run_eval_500pool_pool100.py` | 使用 100-candidate pool 執行 ranking 評估。 |
| `scripts/robustness/run_eval_500pool_pool1000.py` | 使用 1000-candidate pool 執行 ranking 評估。 |
| `scripts/robustness/run_eval_500pool_seed42.py` | 使用 seed 42 執行 500-pool 評估。 |
| `scripts/robustness/run_eval_500pool_seed12345.py` | 使用 seed 12345 執行 500-pool 評估。 |
| `scripts/robustness/run_eval_500pool_seed987654.py` | 使用 seed 987654 執行 500-pool 評估。 |
| `scripts/robustness/run_eval_500pool_top1_pool100.py` | 100-pool ranking 後執行 top-1 generation。 |
| `scripts/robustness/run_eval_500pool_top1_pool1000.py` | 1000-pool ranking 後執行 top-1 generation。 |
| `scripts/robustness/run_eval_500pool_top1_seed42.py` | 使用 seed 42 執行 top-1 generation。 |
| `scripts/robustness/run_eval_500pool_top1_seed12345.py` | 使用 seed 12345 執行 top-1 generation。 |
| `scripts/robustness/run_eval_500pool_top1_seed987654.py` | 使用 seed 987654 執行 top-1 generation。 |
| `scripts/robustness/analyze_pool_robustness_top1.py` | 整理 pool size robustness 結果。 |
| `scripts/robustness/analyze_seed_robustness_top1.py` | 整理 seed robustness 結果。 |
| `scripts/prompt_variants/run_eval_500pool_prompt_original.py` | 使用 original prompt 執行 ranking/generation。 |
| `scripts/prompt_variants/run_eval_500pool_prompt_simple.py` | 使用 simple prompt 執行 ranking/generation。 |
| `scripts/prompt_variants/run_eval_500pool_prompt_strict.py` | 使用 strict prompt 執行 ranking/generation。 |
| `scripts/prompt_variants/run_eval_500pool_prompt_simple_v2.py` | 使用 simple prompt v2 執行 ranking/generation。 |
| `scripts/prompt_variants/run_eval_500pool_prompt_strict_v2.py` | 使用 strict prompt v2 執行 ranking/generation。 |
| `scripts/prompt_variants/run_eval_500pool_prompt_faithful.py` | 使用 faithful prompt 執行 ranking/generation。 |
| `scripts/prompt_variants/run_eval_500pool_top1_prompt_original.py` | 使用 original prompt 執行 top-1 generation。 |
| `scripts/prompt_variants/run_eval_500pool_top1_prompt_simple.py` | 使用 simple prompt 執行 top-1 generation。 |
| `scripts/prompt_variants/run_eval_500pool_top1_prompt_strict.py` | 使用 strict prompt 執行 top-1 generation。 |
| `scripts/prompt_variants/run_eval_500pool_top1_prompt_simple_v2.py` | 使用 simple prompt v2 執行 top-1 generation。 |
| `scripts/prompt_variants/run_eval_500pool_top1_prompt_strict_v2.py` | 使用 strict prompt v2 執行 top-1 generation。 |
| `scripts/prompt_variants/run_eval_500pool_top1_prompt_faithful.py` | 使用 faithful prompt 執行 top-1 generation。 |

## Faithfulness 與人工檢查程式

| 檔案 | 功用 |
|---|---|
| `scripts/faithfulness/run_faithfulness_counterfactual.py` | 產生 faithfulness counterfactual outputs。 |
| `scripts/faithfulness/run_faithfulness_counterfactual_top1.py` | 對 top-1 recommendations 執行 counterfactual generation。 |
| `scripts/faithfulness/faithfulness_claim_judge.py` | 判斷生成解釋中的 claims 是否有輸入證據支持。 |
| `scripts/faithfulness/faithfulness_claim_judge_v2.py` | claim judge v2 設定。 |
| `scripts/faithfulness/faithfulness_claim_judge_top1_v2.py` | top-1 claim judge v2。 |
| `scripts/faithfulness/analyze_faithfulness.py` | 整理 faithfulness claim-judge outputs。 |
| `scripts/faithfulness/analyze_faithfulness_top1_v2.py` | 整理 top-1 faithfulness v2 outputs。 |
| `scripts/faithfulness/run_preference_counterfactual_generation.py` | 進行 preference counterfactual intervention 後產生輸出。 |
| `scripts/faithfulness/run_preference_counterfactual_generation_top1.py` | 針對 top-1 執行 preference counterfactual generation。 |
| `scripts/faithfulness/analyze_preference_counterfactual.py` | 整理 preference counterfactual outputs。 |
| `scripts/faithfulness/analyze_preference_counterfactual_top1.py` | 整理 top-1 preference counterfactual outputs。 |
| `scripts/faithfulness/analyze_metadata_consistency.py` | 檢查生成解釋是否與 metadata 一致。 |
| `scripts/faithfulness/analyze_metadata_consistency_top1.py` | top-1 outputs 的 metadata consistency 分析。 |
| `scripts/faithfulness/llm_as_judge_faithfulness.py` | LLM-as-judge calibration 與 faithfulness 分析。 |
| `scripts/faithfulness/llm_as_judge_faithfulness_top1.py` | top-1 outputs 的 LLM-as-judge 分析。 |
| `scripts/faithfulness/analyze_ucr_error_sources.py` | 分析 unsupported claim rate 的錯誤來源。 |
| `scripts/faithfulness/make_ucr_review_workbook.py` | 製作用於人工檢查 UCR 的 workbook。 |
| `scripts/faithfulness/score_ucr_human_review.py` | 計算人工檢查標註與一致性。 |
| `scripts/manual_review/prepare_manual_review_packets.py` | 製作人工抽查用的 review packets。 |

## 統計、診斷與延伸分析程式

| 檔案 | 功用 |
|---|---|
| `scripts/significance/analyze_significance.py` | 執行主要 ranking/generation 比較的統計檢定。 |
| `scripts/significance/analyze_significance_top1.py` | 執行 top-1 generation 結果的統計檢定。 |
| `scripts/analysis/cliffs_delta_significance.py` | 計算 Cliff's delta effect size。 |
| `scripts/analysis/paired_effect_size.py` | 計算 paired effect-size statistics。 |
| `scripts/analysis/ltp_control_significance.py` | 檢定 LTP control experiments 的顯著性。 |
| `scripts/analysis/candidate_difficulty_stratification.py` | 分析 candidate pool difficulty 對效能的影響。 |
| `scripts/analysis/path_level_generation_analysis_v21.py` | 最終論文版本使用的 path-level explanation analysis。 |
| `scripts/analysis/fixed_hybrid_component_analysis_v21.py` | 整理 explicit/implicit component 固定介入結果。 |
| `scripts/analysis/b5_build_persona_specs.py` | 建立 controllability tests 使用的 structured persona specs。 |
| `scripts/analysis/b5_build_persona_ltp.py` | 建立 structured persona tests 使用的 LTP vectors。 |
| `scripts/analysis/b5_persona_metrics_v21.py` | 最終論文版本使用的 persona controllability metrics v21。 |
| `scripts/analysis/reuse_noltp_v2_for_persona_v21.py` | 重用 no-LTP outputs 以進行 persona v21 比較。 |
| `scripts/analysis/b6_preference_video_conflict_v21.py` | 最終論文版本使用的 preference-video conflict analysis。 |
| `scripts/analysis/reuse_full_exp01_for_fixed_component_v21.py` | 重用 full-model outputs 以進行 fixed-component v21 分析。 |
| `scripts/analysis/prepare_v21_preference_claim_blind_audit.py` | 製作 v21 preference-claim blind audit packets。 |
| `scripts/analysis/write_v21_reproducibility_manifest.py` | 產生 v21 reproducibility/provenance manifest。 |
| `scripts/analysis/tsne_umap_ltp.py` | 使用降維方式視覺化 LTP representation structure。 |
| `scripts/analysis/video_cluster_stratification.py` | 依 video clusters 分層分析效能。 |
| `scripts/analysis/video_cluster_profile.py` | 建立 video clusters 的描述性 profile。 |
| `scripts/analysis/video_cluster_finalize.py` | 整理與命名最終 video cluster outputs。 |
| `scripts/diagnostics/dataset_overlap_analysis.py` | 檢查 train/test overlap 與 leakage risk。 |
| `scripts/diagnostics/compute_median_rank_unseen_seen.py` | 比較 seen/unseen cases 的 median rank。 |
| `scripts/diagnostics/diagnose_exp07_scores.py` | 診斷 no-music experiment 的分數行為。 |
| `scripts/diagnostics/evaluate.py` | 一般診斷評估輔助程式。 |
| `scripts/diagnostics/plot_training_curves.py` | 由 checkpoint logs 繪製 training curves。 |

## LTP / User Profiling 流程程式

| 檔案 | 功用 |
|---|---|
| `user_profiling/stage0_data_prep/Step 1_musicnn_tag_extractor_py36.py` | 在舊版 Python 3.6 環境中萃取 MusicNN tags。 |
| `user_profiling/stage0_data_prep/Step 2_youtube_metadata_extractor.py` | 萃取或整理 YouTube metadata。 |
| `user_profiling/stage0_data_prep/Step 3_merge_to_metadata.py` | 合併 audio tags 與 metadata。 |
| `user_profiling/stage0_data_prep/Step_4a_fix_unknown_artist_title.py` | 修補缺失或 unknown 的 artist/title 欄位。 |
| `user_profiling/stage0_data_prep/Step_4b_reextract_corrupted_audio.py` | 重新萃取或修補損壞的 audio-derived metadata。 |
| `user_profiling/stage1_metadata.py` | 建立 LTP 建構流程使用的 enriched semantic music metadata。 |
| `user_profiling/stage2_personax.py` | 執行 target-conditioned PersonaX sampling。 |
| `user_profiling/stage3/config.py` | synthetic dialogue generation 的設定。 |
| `user_profiling/stage3/llm_client.py` | dialogue synthesis 使用的 local LLM/Ollama client wrapper。 |
| `user_profiling/stage3/prompt_builder.py` | 建立 synthetic preference dialogue generation prompts。 |
| `user_profiling/stage3/qa_validator.py` | 檢查生成 dialogue 格式與基本品質。 |
| `user_profiling/stage3/quality_evaluator.py` | 評分 synthetic dialogue quality。 |
| `user_profiling/stage3/stage3_dialogue_optimized.py` | Stage 3 主要 optimized dialogue generation script。 |
| `user_profiling/stage3/fix_stage3_jsonl.py` | 修補 Stage 3 JSONL 格式問題。 |
| `user_profiling/stage3/stage3.1_experiment_model_selection.py` | Stage 3 generation 的 model-selection experiment。 |
| `user_profiling/stage3/stage3.2_experiment_ablation.py` | Stage 3 prompt/generation design 的 ablation experiment。 |
| `user_profiling/stage3/stage3.3_periodic_sampling_eval.py` | periodic sampling quality evaluation。 |
| `user_profiling/stage3/stage3.4_diagnose_failures.py` | 診斷 Stage 3 generation failures。 |
| `user_profiling/stage3/stage3.5_repair_missing.py` | 修補 missing Stage 3 outputs。 |
| `user_profiling/stage3/stage3.6_improved_post_evaluation.py` | 改良版 post-generation quality evaluation。 |
| `user_profiling/stage4/stage4_recllm.py` | Stage 4 主要 RecLLM-style profile extraction script。 |
| `user_profiling/stage4/stage4.1_pilot_experiment.py` | Stage 4 profile extraction pilot experiment。 |
| `user_profiling/stage4/stage4.2_consistency_eval.py` | extracted profiles 的 consistency evaluation。 |
| `user_profiling/stage5/stage5_preference_representation_v4.py` | 建立 explicit、implicit、hybrid 三種 256D preference representation。 |
| `user_profiling/experiments/Exp1_audio_gradient.py` | audio-gradient behavior 輔助實驗。 |
| `user_profiling/experiments/Exp2_consistency_table.py` | consistency-table 輔助實驗。 |
| `user_profiling/experiments/Exp3_semantic_weights.py` | semantic weights 輔助實驗。 |
| `user_profiling/experiments/Exp4_dual_tsne.py` | t-SNE visualization 輔助實驗。 |
| `user_profiling/experiments/synthetic_to_real_validity_analysis.py` | synthetic-to-real validity signals 輔助分析。 |

## 補充說明

- `__init__.py` 只用來標記 Python package。
- 有些腳本保留是為了 provenance，不一定屬於最短重現路徑。
- 多數訓練與評估腳本都需要 GitHub 之外的大型資料或 checkpoint。請看 [DATA.md](DATA.md) 與 [CHECKPOINTS.md](CHECKPOINTS.md)。
