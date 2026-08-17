# 路徑級生成分析（B3）

- 產生時間：2026-07-29T17:22:23
- 四個模型的 4,205 筆 Top-1 說明文字皆取自既有結果，未重跑生成

## 一、排序 vs 說明：教授指定的對照表

| 指標 | Hybrid LTP | Explicit-only LTP | Implicit-only LTP | No-LTP |
|---|---|---|---|---|
| **排序 R@1**（既有） | 30.65% | 21.93% | 31.06% | 19.07% |
| 偏好屬性引用率 | 0.87% | 1.35% | 0.95% | 0.90% |
| 　樣本涵蓋率（≥1 條偏好主張） | 4.57% | 6.49% | 4.73% | 4.71% |
| 偏好主張與參考畫像一致率 | 39.23% | 47.49% | 42.55% | 43.09% |
| 偏好主張與參考畫像矛盾率 | 0.00% | 0.56% | 2.13% | 0.00% |
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
| exp_01 - exp_04 | preference_reference_alignment_rate | -3.86pp | [-14.81, +7.45] | 否 |
| exp_01 - exp_04 | preference_reference_contradiction_rate | +0.00pp | [+0.00, +0.00] | 否 |
| exp_01 - exp_04 | preference_nonexistent_rate | +2.56pp | [-8.76, +13.36] | 否 |
| exp_01 - exp_04 | metadata_support_rate | +3.35pp | [-1.04, +7.67] | 否 |
| exp_01 - exp_04 | UCR_L1_clause | -0.86pp | [-1.34, -0.40] | 是 |
| exp_03 - exp_04 | preference_claim_ratio | +0.05pp | [-0.13, +0.21] | 否 |
| exp_03 - exp_04 | preference_reference_alignment_rate | -0.54pp | [-11.17, +10.57] | 否 |
| exp_03 - exp_04 | preference_reference_contradiction_rate | +2.13pp | [+0.00, +4.67] | 否 |
| exp_03 - exp_04 | preference_nonexistent_rate | +1.50pp | [-10.05, +12.40] | 否 |
| exp_03 - exp_04 | metadata_support_rate | -1.52pp | [-5.78, +2.94] | 否 |
| exp_03 - exp_04 | UCR_L1_clause | -0.87pp | [-1.36, -0.43] | 是 |
| exp_02 - exp_04 | preference_claim_ratio | +0.45pp | [+0.26, +0.63] | 是 |
| exp_02 - exp_04 | preference_reference_alignment_rate | +4.40pp | [-6.05, +15.36] | 否 |
| exp_02 - exp_04 | preference_reference_contradiction_rate | +0.56pp | [+0.00, +1.85] | 否 |
| exp_02 - exp_04 | preference_nonexistent_rate | -1.34pp | [-12.97, +8.79] | 否 |
| exp_02 - exp_04 | metadata_support_rate | +1.88pp | [-1.92, +5.78] | 否 |
| exp_02 - exp_04 | UCR_L1_clause | +0.04pp | [-0.48, +0.53] | 否 |
| exp_01 - exp_03 | preference_claim_ratio | -0.07pp | [-0.22, +0.08] | 否 |
| exp_01 - exp_03 | preference_reference_alignment_rate | -3.32pp | [-13.33, +7.30] | 否 |
| exp_01 - exp_03 | preference_reference_contradiction_rate | -2.13pp | [-4.67, +0.00] | 否 |
| exp_01 - exp_03 | preference_nonexistent_rate | +1.05pp | [-9.27, +11.55] | 否 |
| exp_01 - exp_03 | metadata_support_rate | +4.87pp | [+0.43, +9.28] | 是 |
| exp_01 - exp_03 | UCR_L1_clause | +0.02pp | [-0.42, +0.46] | 否 |
| exp_02 - exp_03 | preference_claim_ratio | +0.40pp | [+0.21, +0.59] | 是 |
| exp_02 - exp_03 | preference_reference_alignment_rate | +4.93pp | [-5.26, +15.00] | 否 |
| exp_02 - exp_03 | preference_reference_contradiction_rate | -1.57pp | [-4.35, +0.93] | 否 |
| exp_02 - exp_03 | preference_nonexistent_rate | -2.84pp | [-12.55, +6.99] | 否 |
| exp_02 - exp_03 | metadata_support_rate | +3.40pp | [-0.79, +7.41] | 否 |
| exp_02 - exp_03 | UCR_L1_clause | +0.91pp | [+0.41, +1.41] | 是 |
| exp_01 - exp_02 | preference_claim_ratio | -0.48pp | [-0.66, -0.29] | 是 |
| exp_01 - exp_02 | preference_reference_alignment_rate | -8.26pp | [-18.16, +2.26] | 否 |
| exp_01 - exp_02 | preference_reference_contradiction_rate | -0.56pp | [-1.85, +0.00] | 否 |
| exp_01 - exp_02 | preference_nonexistent_rate | +3.90pp | [-6.56, +13.90] | 否 |
| exp_01 - exp_02 | metadata_support_rate | +1.47pp | [-2.77, +5.42] | 否 |
| exp_01 - exp_02 | UCR_L1_clause | -0.90pp | [-1.39, -0.40] | 是 |

## 三、方法與限制

- **claim_judge**：沿用 faithfulness_claim_judge_v2.classify_claim（未修改）
- **UCR_L2**：沿用 B2 的母句還原校正
- **preference_evidence**：Stage 4 profiles.jsonl，依 conflict_tag 與否定詞分為正向／排斥證據後做極性敏感比對；依 video_id 對應，測試集 100% 覆蓋
- **metadata_evidence**：top1_reference_text + title/artist + musicnn 標籤
- **statistics**：樣本層級叢集拔靴 2000 次、配對重抽索引、95% 百分位 CI
- **caveat**：四模型皆與同一份事後參考畫像比較。exp_04 未接收該畫像，其一致率是偶然／先驗一致的負向對照，不是輸入支持率；因此本表統一稱參考畫像一致率。LTP 是不透明向量，無法由文字規則判定逐條主張是否受向量輸入支持；此欄對 exp_01–03 標為不可觀察，對 exp_04 標為不適用。
