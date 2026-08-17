# LTP 建構流程

`P_ltp` 是論文中的 synthetic long-term preference proxy。它不是由真實使用者長期歷史產生，而是以目標音樂為條件，合成跨情境偏好代理訊號，用來檢驗偏好表示是否能改善背景音樂推薦與解釋生成。

## 流程總覽

論文使用五階段流程建立 `P_ltp`：

| 階段 | 目標 | 主要程式 | 主要輸入 | 主要輸出 |
|---|---|---|---|---|
| 1. Semantic integration | 整合 music tags 與 metadata，形成 semantic seed text | `user_profiling/stage1_metadata.py` | MuseChat metadata、MusicNN tags、YouTube metadata | `music_metadata_enriched.json`, `youtube_metadata.jsonl` |
| 2. PersonaX sampling | 以 target track 為條件，建立多樣化偏好樣本 | `user_profiling/stage2_personax.py` | enriched metadata, target track ids | persona sampling records |
| 3. Dialogue synthesis | 使用 local LLM 產生 synthetic preference dialogues | `user_profiling/stage3/` | persona records, prompt templates | synthetic dialogues and QA logs |
| 4. RecLLM profile extraction | 從 dialogues 萃取 profile summary 與 salient facts | `user_profiling/stage4/` | synthetic dialogues | `profiles.jsonl` |
| 5. Preference representation | 建立 explicit/implicit/hybrid 256D preference vectors | `user_profiling/stage5/` | profiles, metadata, AST/CLIP features | LTP cache `.npy` and id maps |

## Stage 1：Semantic integration

此階段將音樂 metadata 與 tag-based features 轉成文字語義種子。論文使用的 template 概念為：

```text
Features {genre} style {popularity_desc}, titled '{title}' by {artist}. Characterized by {top_5_tags}.
```

輸出會作為後續 persona sampling 與 profile extraction 的基礎。

## Stage 2：PersonaX sampling

此階段以目標音樂為條件建立合成偏好樣本。論文設定包含：

```text
Candidate pool P = 2000
KMeans K = 6
alpha = 0.6
Core = 6
Exploratory = 3
Negative = 1
```

這一步的重點不是模擬真實使用者，而是建立可稽核、可控制、與目標音樂條件相關的 synthetic preference proxy。

## Stage 3：Dialogue synthesis

論文使用 local Gemma 3 4B 合成偏好對話。最終規模為：

```text
84,082 synthetic preference individuals
3 dialogues per individual
252,246 deduplicated dialogues
```

此階段包含品質檢查與 judge-based validation。輸出可供 Stage 4 萃取 profile。

## Stage 4：RecLLM profile extraction

論文使用 local Gemma 3 12B 萃取：

```text
80-100 word profile summary
5-8 salient facts
conflict labels: CONFIRM, CONFIRM_DISLIKE, MODULATE, NEW, OVERRIDE
```

輸出為 `profiles.jsonl`。論文抽樣檢查中，profile hallucination 約 3.0%，omission 約 0.0%，平均 accuracy 約 4.26。

## Stage 5：Preference representation

此階段建立三種 LTP 表示：

```text
hybrid: explicit text path + implicit audio path
explicit_only: profile/text-derived preference representation
implicit_only: audio-derived preference representation
```

模型訓練與評估主要讀取：

```text
cache/ltp_hybrid.npy
cache/ltp_hybrid_ids.json
cache/ltp_explicit_only.npy
cache/ltp_explicit_only_ids.json
cache/ltp_implicit_only.npy
cache/ltp_implicit_only_ids.json
```

## 重要限制

- `P_ltp` 是 target-track-conditioned synthetic proxy，不是真實使用者長期偏好。
- Stage 5 的 representation pre-alignment 使用完整 84,082 pairs；論文結果應理解為 full-data representation pretraining condition。
- LTP 對推薦結果的改善代表此 proxy 在目前資料與評估設定中可被模型利用，不代表已完成真實創作者使用情境驗證。
