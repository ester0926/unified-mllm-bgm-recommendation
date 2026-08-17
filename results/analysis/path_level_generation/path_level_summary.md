# 路徑級生成分析（B3）

- 產生時間：2026-07-26T15:10:10
- 四個模型的 4,205 筆 Top-1 說明文字皆取自既有結果，未重跑生成

## 一、排序 vs 說明：教授指定的對照表

| 指標 | Hybrid LTP | Explicit-only LTP | Implicit-only LTP | No-LTP |
|---|---|---|---|---|
| **排序 R@1**（既有） | 30.65% | 21.93% | 31.06% | 19.07% |
| 偏好屬性引用率 | 0.87% | 1.35% | 0.95% | 0.90% |
| 　樣本涵蓋率（≥1 條偏好主張） | 4.57% | 6.49% | 4.73% | 4.71% |
| 正確偏好主張率 | 40.00% | 48.04% | 43.97% | 43.09% |
| 不存在偏好主張率 | 40.77% | 36.87% | 39.72% | 38.21% |
| 音樂元資料支持率 | 82.44% | 80.97% | 77.57% | 79.09% |
| UCR（子句層級 L1） | 12.77% | 13.66% | 12.75% | 13.62% |
| UCR（母句校正 L2） | 2.84% | 3.73% | 2.87% | 3.18% |
| 音訊屬性主張比例 | 37.06% | 34.77% | 35.67% | 35.43% |
| 每則說明的 claim 數 | 5.42 | 4.99 | 5.33 | 5.33 |
| 平均字數 | 47.45 | 45.95 | 47.38 | 47.84 |

## 二、關鍵對照（樣本層級叢集拔靴 95% CI）

| 對照 | 指標 | 差異 | 95% CI | 顯著 |
|---|---|---|---|---|
| exp_01 - exp_04 | preference_claim_ratio | -0.03pp | [-0.18, +0.13] | 否 |
| exp_01 - exp_04 | preference_correct_rate | -3.09pp | [-13.99, +8.18] | 否 |
| exp_01 - exp_04 | preference_nonexistent_rate | +2.56pp | [-8.76, +13.36] | 否 |
| exp_01 - exp_04 | metadata_support_rate | +3.35pp | [-1.04, +7.67] | 否 |
| exp_01 - exp_04 | UCR_L1_clause | -0.86pp | [-1.34, -0.40] | 是 |
| exp_03 - exp_04 | preference_claim_ratio | +0.05pp | [-0.13, +0.21] | 否 |
| exp_03 - exp_04 | preference_correct_rate | +0.88pp | [-9.96, +11.91] | 否 |
| exp_03 - exp_04 | preference_nonexistent_rate | +1.50pp | [-10.05, +12.40] | 否 |
| exp_03 - exp_04 | metadata_support_rate | -1.52pp | [-5.78, +2.94] | 否 |
| exp_03 - exp_04 | UCR_L1_clause | -0.87pp | [-1.36, -0.43] | 是 |
| exp_02 - exp_04 | preference_claim_ratio | +0.45pp | [+0.26, +0.63] | 是 |
| exp_02 - exp_04 | preference_correct_rate | +4.96pp | [-5.44, +15.88] | 否 |
| exp_02 - exp_04 | preference_nonexistent_rate | -1.34pp | [-12.97, +8.79] | 否 |
| exp_02 - exp_04 | metadata_support_rate | +1.88pp | [-1.92, +5.78] | 否 |
| exp_02 - exp_04 | UCR_L1_clause | +0.04pp | [-0.48, +0.53] | 否 |
| exp_01 - exp_03 | preference_claim_ratio | -0.07pp | [-0.22, +0.08] | 否 |
| exp_01 - exp_03 | preference_correct_rate | -3.97pp | [-13.45, +6.76] | 否 |
| exp_01 - exp_03 | preference_nonexistent_rate | +1.05pp | [-9.27, +11.55] | 否 |
| exp_01 - exp_03 | metadata_support_rate | +4.87pp | [+0.43, +9.28] | 是 |
| exp_01 - exp_03 | UCR_L1_clause | +0.02pp | [-0.42, +0.46] | 否 |
| exp_02 - exp_03 | preference_claim_ratio | +0.40pp | [+0.21, +0.59] | 是 |
| exp_02 - exp_03 | preference_correct_rate | +4.07pp | [-6.20, +13.70] | 否 |
| exp_02 - exp_03 | preference_nonexistent_rate | -2.84pp | [-12.55, +6.99] | 否 |
| exp_02 - exp_03 | metadata_support_rate | +3.40pp | [-0.79, +7.41] | 否 |
| exp_02 - exp_03 | UCR_L1_clause | +0.91pp | [+0.41, +1.41] | 是 |
| exp_01 - exp_02 | preference_claim_ratio | -0.48pp | [-0.66, -0.29] | 是 |
| exp_01 - exp_02 | preference_correct_rate | -8.04pp | [-17.76, +2.40] | 否 |
| exp_01 - exp_02 | preference_nonexistent_rate | +3.90pp | [-6.56, +13.90] | 否 |
| exp_01 - exp_02 | metadata_support_rate | +1.47pp | [-2.77, +5.42] | 否 |
| exp_01 - exp_02 | UCR_L1_clause | -0.90pp | [-1.39, -0.40] | 是 |

## 三、方法與限制

- **claim_judge**：沿用 faithfulness_claim_judge_v2.classify_claim（未修改）
- **UCR_L2**：沿用 B2 的母句還原校正
- **preference_evidence**：Stage 4 profiles.jsonl（summary_text + salient_facts），依 video_id 對應，測試集 100% 覆蓋
- **metadata_evidence**：top1_reference_text + title/artist + musicnn 標籤
- **statistics**：樣本層級叢集拔靴 2000 次、配對重抽索引、95% 百分位 CI
- **caveat**：exp_04 無偏好輸入，其偏好主張依定義必然無證據支持；此為對照設計本意，不可解讀為該模型特別容易杜撰。
