# 固定 Hybrid 模型之偏好成分介入分析（v21）

- 產生時間：2026-07-30T01:09:10
- 比較方向：完整 Hybrid 減介入條件；正值表示完整 Hybrid 較高。

## 一、分解驗證

- 保留集 R²：0.999999999978
- 保留集相對 MAE：1.571e-06
- 完整資料重建 MAE：1.554e-06

## 二、排序與生成摘要

| 條件 | R@1 | R@5 | MRR | 偏好主張比例 | 畫像一致率 | 畫像矛盾率 | 元資料支持率 | UCR L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整 Hybrid | 29.00% | 67.00% | 0.4557 | 0.83% | 80.00% | 0.00% | 67.74% | 11.54% |
| 移除顯式成分（原始） | 15.50% | 47.50% | 0.3060 | NA | NA | NA | NA | NA |
| 移除隱式成分（原始） | 3.00% | 11.50% | 0.0897 | NA | NA | NA | NA | NA |
| 移除顯式成分（範數校正） | 16.00% | 47.00% | 0.3088 | 1.34% | 50.00% | 0.00% | 48.15% | 20.84% |
| 移除隱式成分（範數校正） | 3.00% | 11.50% | 0.0897 | 1.09% | 66.67% | 0.00% | 70.97% | 11.31% |
| 同時移除兩成分 | 1.50% | 3.50% | 0.0361 | NA | NA | NA | NA | NA |

## 三、配對對照

| 對照 | 指標 | 差異 | 95% CI | 顯著 |
|---|---|---:|---:|:---:|
| full - no_explicit | recall@1 | +0.1350 | [+0.0700, +0.2000] | 是 |
| full - no_explicit | recall@5 | +0.1950 | [+0.1300, +0.2600] | 是 |
| full - no_explicit | mrr | +0.1497 | [+0.1020, +0.1993] | 是 |
| full - no_implicit | recall@1 | +0.2600 | [+0.2000, +0.3250] | 是 |
| full - no_implicit | recall@5 | +0.5550 | [+0.4800, +0.6300] | 是 |
| full - no_implicit | mrr | +0.3660 | [+0.3149, +0.4170] | 是 |
| full - no_explicit_norm | recall@1 | +0.1300 | [+0.0700, +0.1950] | 是 |
| full - no_explicit_norm | recall@5 | +0.2000 | [+0.1350, +0.2650] | 是 |
| full - no_explicit_norm | mrr | +0.1469 | [+0.0993, +0.1967] | 是 |
| full - no_implicit_norm | recall@1 | +0.2600 | [+0.2000, +0.3250] | 是 |
| full - no_implicit_norm | recall@5 | +0.5550 | [+0.4800, +0.6300] | 是 |
| full - no_implicit_norm | mrr | +0.3660 | [+0.3149, +0.4170] | 是 |
| full - no_both | recall@1 | +0.2750 | [+0.2100, +0.3450] | 是 |
| full - no_both | recall@5 | +0.6350 | [+0.5650, +0.7001] | 是 |
| full - no_both | mrr | +0.4196 | [+0.3652, +0.4744] | 是 |
| full - no_explicit_norm | preference_claim_ratio | -0.0050 | [-0.0119, +0.0019] | 否 |
| full - no_explicit_norm | preference_reference_alignment_rate | +0.3000 | [-0.1667, +0.7143] | 否 |
| full - no_explicit_norm | preference_reference_contradiction_rate | +0.0000 | [+0.0000, +0.0000] | 否 |
| full - no_explicit_norm | metadata_support_rate | +0.1959 | [-0.2261, +0.5237] | 否 |
| full - no_explicit_norm | UCR_L1_clause | -0.0930 | [-0.1456, -0.0460] | 是 |
| full - no_implicit_norm | preference_claim_ratio | -0.0025 | [-0.0118, +0.0064] | 否 |
| full - no_implicit_norm | preference_reference_alignment_rate | +0.1333 | [-0.5000, +0.6157] | 否 |
| full - no_implicit_norm | preference_reference_contradiction_rate | +0.0000 | [+0.0000, +0.0000] | 否 |
| full - no_implicit_norm | metadata_support_rate | -0.0323 | [-0.2564, +0.1905] | 否 |
| full - no_implicit_norm | UCR_L1_clause | +0.0023 | [-0.0163, +0.0207] | 否 |

## 四、判讀界線

- **design**：固定 exp_01 checkpoint，在推論時自 Hybrid 向量扣除線性分解出的顯式／隱式成分
- **sample**：固定亂數抽取 200 筆測試查詢；四條件共享相同查詢、500 候選池與候選池種子
- **ranking_statistics**：樣本配對拔靴 5,000 次，報完整 Hybrid 減介入條件之差與 95% CI
- **generation_statistics**：claim 巢套於樣本，採樣本層級配對叢集拔靴 3,000 次
- **boundary**：此為單一 checkpoint 的表徵層探索性介入；原始扣除可能形成離開訓練分布的向量，因此另報範數校正敏感度分析，但兩者仍不能單獨建立一般性的因果歸因
