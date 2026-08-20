# Top-1 推薦理由忠實度分析

本分析固定 `exp_01` checkpoint，使用模型選出的 Top-1 音樂生成推薦理由，並透過移除不同模態輸入檢查主張來源。
主張標籤由 `faithfulness_claim_judge.py` 的規則式判官 v1 產生。

## 各條件指標

| Condition | Claims | UCR | MAA | Video claims | Audio claims | Prompt claims | Preference claims |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 1090 | 11.93% | 100.0% | 19.63% | 51.65% | 3.85% | 0.83% |
| wo_audio_all | 1015 | 60.39% | 35.6% | 20.59% | 53.1% | 2.76% | 6.01% |
| wo_audio_feature_only | 953 | 59.5% | 33.38% | 16.79% | 54.88% | 4.62% | 6.09% |
| wo_ltp | 329 | 97.26% | 100.0% | 2.74% | 0.0% | 0.0% | 0.0% |
| wo_prompt | 1061 | 10.65% | 98.54% | 22.24% | 53.82% | 1.13% | 0.09% |
| wo_video | 939 | 27.05% | 74.73% | 19.6% | 49.73% | 5.43% | 2.77% |

## 敏感度

| Condition | n | ESS mean (Jaccard distance from full) |
|---|---:|---:|
| wo_audio_all | 200 | 0.6765 |
| wo_audio_feature_only | 200 | 0.7286 |
| wo_ltp | 200 | 0.9998 |
| wo_prompt | 200 | 0.6526 |
| wo_video | 200 | 0.5647 |

## 主要下降指標

- 移除 `z_ltp` 後的 PCR：100.0%
- 移除 video 後的 video-claim reduction：0.19%
- 僅移除 audio feature 後的 audio-claim reduction：-6.25%
- 同時移除 audio feature 與音樂 metadata 後的 audio-claim reduction：-2.81%
- 移除 prompt 後的 prompt-claim reduction：70.65%

## 解讀注意事項

- UCR 越低越好；本研究用來觀察不受輸入支持的主張比例。
- MAA 越高越好；本研究用來觀察移除模態後，仍被保留的主張是否仍有可用依據。
- PCR 越高越好；本研究用來觀察偏好相關主張是否會隨 `P_ltp` 移除而下降。
- `wo_audio_feature_only` 只移除音樂特徵，仍保留 title/artist metadata。
- `wo_audio_all` 是較嚴格的反事實條件，會同時移除音樂特徵與 title/artist metadata。
- 本檔為規則式初步分析摘要；正式解讀仍需搭配論文中的人工查核或 LLM-as-a-Judge 驗證。
