# Metadata 一致性測試

本規則式分析檢查生成推薦理由中的音樂細節主張，是否能被 Top-1 候選音樂的 metadata 或 reference text 支持。

| Exp | Music claims | Unsupported claims | Unsupported rate | Top-1 correct unsupported | Top-1 incorrect unsupported |
|---|---:|---:|---:|---:|---:|
| exp_01 | 7600 | 4834 | 63.61% | 44.24% | 72.15% |
| exp_02 | 7240 | 4591 | 63.41% | 43.0% | 69.01% |
| exp_03 | 7661 | 4843 | 63.22% | 41.36% | 73.02% |
| exp_04 | 7797 | 5159 | 66.17% | 46.19% | 70.71% |
| exp_05 | 7376 | 4637 | 62.87% | 43.39% | 70.44% |
| exp_06 | 7814 | 6118 | 78.3% | 77.32% | 78.5% |
| exp_07 | 8010 | 7375 | 92.07% | 66.67% | 92.08% |

## 解讀注意事項

- Unsupported rate 越低，表示 metadata consistency 越好。
- 本分析使用保守的 keyword-overlap proxy，可能高估改寫語句的 unsupported 情況，也可能低估較籠統但缺乏依據的主張。
- 本表適合作為初步證據；正式解讀時應搭配代表案例的人工查核。
