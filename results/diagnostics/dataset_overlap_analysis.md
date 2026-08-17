# MuseChat Dataset Overlap Analysis

> **pair_key 結構**：`pair_key[:11]` = `target_music_id`（GT 音樂 = video_id）；
> `pair_key[12:]` = `candidate_music_id`（訓練負例）。
> Group split 以 `target_music_id` 為單位，故 GT 在任何層級的重疊率均為 **0%**。

---

## Task 1 — GT（Target Music）重疊率

| 層級 | Train 唯一數 | Test 唯一數 | 重疊數 | 重疊率 |
|------|------------|------------|------|------|
| Item   | 75,673 | 4,205 | 0 | **0.0%** |
| Title  | 75,172 | 4,181 | 23 | **0.6%** |
| Artist | 62,835 | 4,066 | 955 | **23.5%** |
| Album  | 1 | 0 | 0 | **0.0%** |

> Item-level = 0% 直接確認 group split 正確隔離 GT 音樂。
> Title/Artist 若非 0% 代表同名藝術家有不同 YouTube 影片（屬合理現象）。

---

## Task 2 — Candidate Music 重疊率

| 層級 | Train 唯一數 | Test 唯一數 | 重疊數 | 重疊率 |
|------|------------|------------|------|------|
| Item   | 23,301 | 3,461 | 2,846 | **82.2%** |
| Title  | 23,131 | 3,445 | 2,832 | **82.2%** |
| Artist | 20,941 | 3,332 | 2,808 | **84.3%** |
| Album  | 0 | 0 | 0 | **0.0%** |

> Candidate overlap 高（~82%）是合理的：candidate_music 的種數（~24k）
> 遠少於 pair 數（84k），同一首候選音樂被重複用於多個不同的 pair。
> **候選音樂不出現在 500-pool 評估的排序候選中**，故不影響 retrieval 效能評估的公平性。

---

## Task 3 — Candidate-disjoint 敏感度分析

訓練集唯一 candidate_music_id 數：23,301

| 子集 | N | R@1 (%) | R@5 (%) | R@10 (%) | MR |
|------|---|--------|--------|---------|-----|
| (A) Candidate-disjoint（unseen） | 627 | 33.01 | 68.26 | 83.09 | 7.8 |
| (B) Candidate-overlap（seen）    | 3578 | 30.24 | 66.07 | 79.32 | 8.0 |

Gap（A − B）：ΔR@1 = **+2.77 pp**，ΔR@10 = **+3.77 pp**

> **結論（無候選記憶效應）**：|ΔR@1| = 2.77 pp ≤ 5 pp，|ΔR@10| = 3.77 pp ≤ 5 pp。模型對從未在訓練中作為負例出現的候選音樂，仍能有效進行語義對齊與排序，顯示學習到的是跨曲目的通用特徵，而非記憶特定候選的分數偏好。

---

## 結語

1. **GT 隔離**：Group split by `target_music_id` 確保 GT 音樂 item-level 零重疊，
   train/test 切分設計正確，不存在 GT 資訊洩漏。

2. **Candidate 重疊**：~82% 的候選音樂重疊是資料特性的自然結果，
   且 candidate 不出現在評估排序池中，對 retrieval 效能評估無直接影響。

3. **敏感度分析**：模型對 unseen/seen candidate 兩組的表現差距在可接受範圍內，
   支持模型學到的是音樂語義匹配能力，而非候選偏好記憶。