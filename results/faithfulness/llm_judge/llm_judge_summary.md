# LLM-as-a-Judge 忠實度驗證

評審模型：`llama3:8b`

| Task | n | valid | errors | LLM positive rate | Agreement with rule-based judge |
|---|---:|---:|---:|---:|---:|
| Feature-erasure supported claims | 120 | 120 | 0 | 95.00% | 60.83% |
| Preference aligned | 80 | 80 | 0 | 91.25% | 92.50% |
| Preference conflict | 80 | 80 | 0 | 16.25% | 75.00% |
| Metadata supported | 120 | 119 | 1 | 84.03% | 44.54% |

## 解讀注意事項

- 這是 LLM 輔助驗證子集，不是完整 deterministic analysis。
- Agreement rate 可用來討論規則式標籤是否具有方向性可靠度。
- 若 agreement rate 偏低，對應的規則式指標應保守解讀，並搭配人工查核。
