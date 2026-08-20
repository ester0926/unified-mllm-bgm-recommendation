# 推薦理由忠實度分析

本分析固定 `exp_01` checkpoint，透過移除不同模態輸入，觀察推薦理由中的主張是否仍有輸入依據。
主張標籤由 `faithfulness_claim_judge.py` 的規則式判官 v1 產生。

## 各條件指標

| Condition | Claims | UCR | MAA | Video claims | Audio claims | Prompt claims | Preference claims |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 1079 | 10.75% | 100.0% | 20.48% | 52.36% | 3.61% | 0.56% |
| wo_audio_all | 1015 | 60.39% | 35.6% | 20.59% | 53.1% | 2.76% | 6.01% |
| wo_audio_feature_only | 955 | 58.74% | 33.63% | 15.5% | 54.14% | 4.82% | 7.12% |
| wo_ltp | 342 | 98.25% | 100.0% | 1.75% | 0.0% | 0.0% | 0.0% |
| wo_prompt | 1048 | 8.97% | 98.56% | 22.14% | 55.92% | 1.15% | 0.29% |
| wo_video | 923 | 28.6% | 73.07% | 20.8% | 49.19% | 4.55% | 2.71% |

## 敏感度

| Condition | n | ESS mean (Jaccard distance from full) |
|---|---:|---:|
| wo_audio_all | 200 | 0.6728 |
| wo_audio_feature_only | 200 | 0.7209 |
| wo_ltp | 200 | 0.9998 |
| wo_prompt | 200 | 0.6323 |
| wo_video | 200 | 0.5424 |

## 主要下降指標

- 移除 `z_ltp` 後的 PCR：100.0%
- 移除 video 後的 video-claim reduction：-1.56%
- 僅移除 audio feature 後的 audio-claim reduction：-3.39%
- 同時移除 audio feature 與音樂 metadata 後的 audio-claim reduction：-1.41%
- 移除 prompt 後的 prompt-claim reduction：68.32%

## 解讀注意事項

- UCR 越低越好；本研究用來觀察不受輸入支持的主張比例。
- MAA 越高越好；本研究用來觀察移除模態後，仍被保留的主張是否仍有可用依據。
- PCR 越高越好；本研究用來觀察偏好相關主張是否會隨 `P_ltp` 移除而下降。
- `wo_audio_feature_only` 只移除音樂特徵，仍保留 title/artist metadata。
- `wo_audio_all` 是較嚴格的反事實條件，會同時移除音樂特徵與 title/artist metadata。
- 本檔為規則式初步分析摘要；正式解讀仍需搭配論文中的人工查核或 LLM-as-a-Judge 驗證。
