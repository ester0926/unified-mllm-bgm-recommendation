# Top-1 偏好反事實測試

本分析固定模型 checkpoint 與候選音樂，只改變自然語言使用者偏好 prompt，觀察 Top-1 推薦理由是否會跟著偏好條件改變。

| Variant | n | Aligned rate | Conflict rate | Avg. ESS from original | Avg. alignment score |
|---|---:|---:|---:|---:|---:|
| cf_upbeat_electronic | 200 | 99.5% | 2.5% | 0.6546 | 5.89 |
| cf_lyrical_piano | 200 | 94.0% | 59.0% | 0.6437 | 4.08 |

## 解讀注意事項

- Aligned rate 越高，表示生成推薦理由越能反映反事實偏好文字。
- Conflict rate 越高，表示推薦理由仍提到與反事實偏好相反的概念。
- ESS 是與原始 prompt 推薦理由的 Jaccard distance；數值越高代表文字改變越大。
- 這不是完整 reranking 測試，因為目前 pipeline 中的文字 embedding 已預先計算。
