# 偏好反事實方向敏感度（B3-2）

- 產生時間：2026-07-26T17:37:56
- 四個模型各 200 樣本 × 3 變體（original / cf_upbeat_electronic / cf_lyrical_piano）

## 一、逐模型結果（兩個反事實方向合併）

| 模型 | n | 方向命中率 | 淨方向命中率 | 提示詞複誦率 | 曲名漂移率 | 文字改變率 |
|---|---|---|---|---|---|---|
| Hybrid LTP | 400 | 97.2% | 91.8% | 80.0% | 1.0% | 100.0% |
| Explicit-only LTP | 400 | 77.0% | 73.0% | 89.2% | 6.8% | 100.0% |
| Implicit-only LTP | 400 | 83.2% | 80.8% | 84.0% | 12.2% | 100.0% |
| No-LTP | 400 | 88.0% | 83.0% | 82.6% | 6.8% | 100.0% |

## 二、分方向結果

| 模型 | 反事實方向 | 方向命中率 | 淨方向命中率 | 曲名漂移率 |
|---|---|---|---|---|
| Hybrid LTP | cf_upbeat_electronic | 95.5% | 93.5% | 0.0% |
| Hybrid LTP | cf_lyrical_piano | 99.0% | 90.0% | 2.0% |
| Explicit-only LTP | cf_upbeat_electronic | 84.0% | 82.0% | 1.5% |
| Explicit-only LTP | cf_lyrical_piano | 70.0% | 64.0% | 12.0% |
| Implicit-only LTP | cf_upbeat_electronic | 93.5% | 91.0% | 1.5% |
| Implicit-only LTP | cf_lyrical_piano | 73.0% | 70.5% | 23.0% |
| No-LTP | cf_upbeat_electronic | 89.5% | 87.0% | 5.5% |
| No-LTP | cf_lyrical_piano | 86.5% | 79.0% | 8.0% |

## 三、模型兩兩對照（配對拔靴 95% CI）

| 對照 | 指標 | 差異 | 95% CI | 顯著 |
|---|---|---|---|---|
| exp_02 - exp_04 | direction_accuracy | -11.00pp | [-15.75, -6.00] | 是 |
| exp_02 - exp_04 | net_direction_accuracy | -10.00pp | [-15.00, -4.50] | 是 |
| exp_02 - exp_04 | title_drift_rate | +0.00pp | [-3.25, +3.50] | 否 |
| exp_01 - exp_04 | direction_accuracy | +9.25pp | [+6.00, +13.00] | 是 |
| exp_01 - exp_04 | net_direction_accuracy | +8.75pp | [+4.25, +13.01] | 是 |
| exp_01 - exp_04 | title_drift_rate | -5.75pp | [-8.50, -3.00] | 是 |
| exp_03 - exp_04 | direction_accuracy | -4.75pp | [-9.00, -0.25] | 是 |
| exp_03 - exp_04 | net_direction_accuracy | -2.25pp | [-7.25, +2.50] | 否 |
| exp_03 - exp_04 | title_drift_rate | +5.50pp | [+1.75, +9.50] | 是 |
| exp_02 - exp_03 | direction_accuracy | -6.25pp | [-11.50, -1.00] | 是 |
| exp_02 - exp_03 | net_direction_accuracy | -7.75pp | [-13.00, -2.25] | 是 |
| exp_02 - exp_03 | title_drift_rate | -5.50pp | [-9.25, -1.75] | 是 |
| exp_01 - exp_03 | direction_accuracy | +14.00pp | [+10.00, +18.00] | 是 |
| exp_01 - exp_03 | net_direction_accuracy | +11.00pp | [+6.75, +15.25] | 是 |
| exp_01 - exp_03 | title_drift_rate | -11.25pp | [-14.75, -8.25] | 是 |
| exp_01 - exp_02 | direction_accuracy | +20.25pp | [+15.75, +24.76] | 是 |
| exp_01 - exp_02 | net_direction_accuracy | +18.75pp | [+13.75, +23.50] | 是 |
| exp_01 - exp_02 | title_drift_rate | -5.75pp | [-8.50, -3.25] | 是 |

## 四、範圍限制

- 反事實只更換 prompt 文字，[TEXT_CLIP] 預算特徵與 [LTP] 向量均未重算（資料 scope_note 已註明），故本分析測的是對提示詞偏好敘述的反應，不是對 LTP 向量的反應，不可作為 LTP 路徑的因果證據。
- 三個變體的 top-1 曲目完全相同，因此曲名漂移可直接歸因於提示詞。

## 五、指標定義

- **direction**：反事實目標詞彙數相對 original 上升即命中
- **net_direction**：目標詞彙上升且相反方向詞彙未上升
- **prompt_echo**：生成文中的目標詞有多少比例本來就出現在提示詞裡
- **title_drift**：original 提到正確 top-1 曲名、反事實條件下不再提到
- **statistics**：樣本層級配對拔靴 2000 次、95% 百分位 CI
