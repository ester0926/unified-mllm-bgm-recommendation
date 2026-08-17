## LTP Perturbation Control — Wilcoxon + Cliff's Delta

Holm-Bonferroni correction applied across 3 pairwise comparisons.

Cliff's δ > 0 → 前條件 (cond_x) 排名較低（較優）。

### 各條件指標摘要

| Condition | R@1 (%) | R@5 (%) | R@10 (%) | MedR |
|---|---:|---:|---:|---:|
| matched | 30.58 | 66.23 | 79.88 | 3.0 |
| shuffled | 4.85 | 16.22 | 25.42 | 34.0 |
| random | 3.26 | 12.27 | 19.29 | 50.0 |

### 成對比較（Wilcoxon + Cliff's δ）

| ID | Comparison | ΔR@1 (pp) | ΔR@5 (pp) | ΔR@10 (pp) | Cliff's δ | Magnitude | W | p_adj | Sig |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| A | matched vs shuffled | +25.73 | +50.01 | +54.46 | +0.6928 | large | 286606 | p < .001 | *** |
| B | matched vs random | +27.32 | +53.96 | +60.59 | +0.7624 | large | 217462 | p < .001 | *** |
| C | shuffled vs random | +1.59 | +3.95 | +6.14 | +0.1410 | negligible | 3430892 | p < .001 | *** |

### 詮釋

- **[A] matched vs shuffled**：LTP 使用者身份的邊際效益（排除統計雜訊）  
  ΔR@1=+25.73pp，δ=+0.6928 (large)，p_adj < .001 ***
- **[B] matched vs random**：有正確 LTP vs 完全無 LTP  
  ΔR@1=+27.32pp，δ=+0.7624 (large)，p_adj < .001 ***
- **[C] shuffled vs random**：錯誤身份 LTP vs 無 LTP（診斷：是否存在分佈偏移）  
  ΔR@1=+1.59pp，δ=+0.1410 (negligible)，p_adj < .001 ***

### 論文文字片段（§4.7.4 草稿）

> **[A]** exp_01 在 matched vs shuffled 比較中：ΔR@1=+25.73pp，Cliff's δ=+0.6928 (large)，W=286606，p_Holm < .001 (***)。
> **[B]** exp_01 在 matched vs random 比較中：ΔR@1=+27.32pp，Cliff's δ=+0.7624 (large)，W=217462，p_Holm < .001 (***)。
> **[C]** exp_01 在 shuffled vs random 比較中：ΔR@1=+1.59pp，Cliff's δ=+0.1410 (negligible)，W=3430892，p_Holm < .001 (***)。