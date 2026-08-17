# Stage 5：Core Set Preference Representation (v4)
## 長期偏好向量生成模組

**版本：** v4  
**對應論文章節：** 第 3.5 節（Hybrid Preference Representation）  
**輸入：** Stage 2 PersonaX 歷史、Stage 4 RecLLM 偏好畫像、MuseChat HDF5 音訊特徵  
**輸出：** `preference_vectors.h5`，每個 music_id 對應一個 **256D** 混合偏好向量 P_ltp

---

## 1. 版本變更摘要（v3 → v4）

| 項目 | v3 | v4 |
|---|---|---|
| 消融模式輸出維度 | hybrid=512D, others=512D（後半補零）| **三種模式統一 256D** |
| explicit_only 輸出 | `[explicit_256, zeros_256]` | `W_out_single(explicit_256)` → 256D |
| implicit_only 輸出 | `[zeros_256, implicit_256]` | `W_out_single(implicit_256)` → 256D |
| MVT-Fusion 接口 | 需依模式調整輸入維度 | **固定接收 256D，架構不動** |
| 新增組件 | — | `W_out_hybrid(512→256)`, `W_out_single(256→256)` |

**為什麼不能補零：**
補零對推薦模型造成兩個干擾：(1) 零向量是假信號，模型可能學到「後半全零 = explicit_only」的人工規律；(2) 補零側梯度永遠為零，三個消融實驗的 W_out 學到的東西不可比，消融結果失去意義。

---

## 2. 系統架構

```
Stage 4 profiles.jsonl              Stage 2 history/*.json
    │  salient_facts / summary_text      │  core_sbs: [music_id, semantic_seed]
    ▼                                    ▼
CLIP Text Encoder（凍結）          AST 嵌入索引
    │  pooler_output [512D]              │  target_music_all_cls mean pool [768D]
    │                                    │
    ▼                                    │  ┌─ Softmax 語義加權 ─────────────┐
  W_explicit                            │  │ sim(CLIP(seed_i), CLIP(text))  │
  Linear(512→256)                       │  │ w_i = softmax(β × sim_i)       │
    │                                   │  └────────────────────────────────┘
    │  explicit_vec [256D]              ▼
    │                           weighted_avg [768D]
    │                                    │
    │                               W_implicit
    │                               Linear(768→256)
    │                                    │
    │                            implicit_vec [256D]
    │                                    │
    └──────────────┬─────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   消融模式切換      │
         │                    │
         │ hybrid:            │
         │  concat [512D]     │
         │  → W_out_hybrid    │
         │    (512→256)       │
         │                    │
         │ explicit_only:     │
         │  explicit_vec      │
         │  → W_out_single    │
         │    (256→256)       │
         │                    │
         │ implicit_only:     │
         │  implicit_vec      │
         │  → W_out_single    │
         │    (256→256)       │
         └─────────┬──────────┘
                   │
              P_ltp [256D]
                   │
        ┌──────────▼───────────┐
        │ preference_vectors.h5 │
        │  {music_id}: [256D]  │
        └──────────────────────┘
                   │
                   ▼
           MVT-Fusion 推薦模組
           （固定接收 256D，
             架構無需修改）
```

---

## 3. 兩階段對齊策略

### 3.1 Stage 5 跨模態預對齊（InfoNCE）

W_explicit 和 W_implicit 在生成 P_ltp 之前，先透過對比學習建立跨模態語義對應關係。

**監督信號：**  
MuseChat HDF5 的天然配對——每個 pair 的 `text_features` 和 `target_music_all_cls` 是描述同一首推薦場景的文字與音訊，天然構成正例。

**Loss 設計（雙向對稱 InfoNCE）：**

```
L = 0.5 × (L_text→audio + L_audio→text)

其中：
sim[i,j] = cosine(W_explicit · text_i, W_implicit · audio_j) / τ
L_text→audio = CrossEntropy(sim,   diagonal_labels)
L_audio→text = CrossEntropy(sim.T, diagonal_labels)
```

同一 pair 的 (text, audio) 在 256D 空間靠近，不同 pair 的遠離。

**設定方式：**
```python
ModelConfig.PRETRAIN_ALIGNMENT   = True    # 開啟（推薦）
ModelConfig.PRETRAIN_EPOCHS      = 10
ModelConfig.PRETRAIN_TEMPERATURE = 0.07    # InfoNCE temperature（CLIP 預設值）
ModelConfig.PRETRAIN_BATCH_SIZE  = 64      # 越大負例越多，對比效果越好
```

**重要：** 第一次執行訓練完後自動儲存 `projection_weights.pt`，後續執行直接載入，不重複訓練。

### 3.2 MVT-Fusion 任務導向微調（端對端）

MVT-Fusion 訓練時，梯度透過 `W_out → W_explicit / W_implicit` 反向傳播，針對推薦任務進一步優化偏好表示。這是架構天然提供的，無需額外設計。

```
Stage 5 InfoNCE 預對齊   →   MVT-Fusion 推薦任務微調
（通用跨模態語義對齊）         （任務特定語義對齊）
類比：BERT 預訓練          →   BERT fine-tune
```

---

## 4. 消融實驗設計

修改 `ModelConfig.REPRESENTATION_MODE` 後分別執行三次：

```python
# 第一次（同時完成預對齊訓練，約 15~30 分鐘）
ModelConfig.REPRESENTATION_MODE = "hybrid"
# → preference_vectors.h5          [N, 256D]

# 第二次（載入已訓練權重，約 10~20 分鐘）
ModelConfig.REPRESENTATION_MODE = "explicit_only"
# → preference_vectors_explicit_only.h5  [N, 256D]

# 第三次（載入已訓練權重，約 10~20 分鐘）
ModelConfig.REPRESENTATION_MODE = "implicit_only"
# → preference_vectors_implicit_only.h5  [N, 256D]
```

**三次執行共用同一份 `projection_weights.pt`**——W_explicit / W_implicit 只訓練一次，差異只在 W_out 的接收對象。

| 消融實驗 | 模式 | 輸入 W_out | 預期作用 |
|---|---|---|---|
| Baseline | 無 P_ltp | — | MVT-Fusion 原始效能 |
| Ablation-A | `explicit_only` | explicit_256 | 量化顯性語義偏好的貢獻 |
| Ablation-B | `implicit_only` | implicit_256 | 量化隱性行為偏好的貢獻 |
| 主方法 | `hybrid` | concat(explicit, implicit) | 完整混合偏好效能 |
| Ablation-C | `hybrid` + PRETRAIN=False | Xavier concat | 量化預對齊的貢獻 |

若 hybrid > explicit_only 且 hybrid > implicit_only，直接證明「顯性 + 隱性融合優於單一模態」，即論文核心貢獻。

---

## 5. 語義相似度加權（Semantic Grounding）

隱性向量在加權前，用 CLIP 計算顯性文字與每首 core music 的 semantic_seed 語義相似度：

```
w_i = softmax(β × cosine(CLIP(semantic_seed_i), CLIP(explicit_text)))
implicit_input = Σ w_i × AST(music_i)
```

**作用：** core_sbs 中語義偏離的音樂自動獲得低權重，避免污染 implicit_vec。  
**實作優化：** `encode_explicit_preference()` 同時回傳 `(proj_256d, clip_512d)`，後者直接傳入加權函數，避免對相同文字做兩次 CLIP forward。  
**參數：** `ModelConfig.BETA = 2.0`（越大，高相似度音樂越 dominant）

---

## 6. 輸入輸出規格

### 輸入路徑

| 路徑 | 格式 | 說明 |
|---|---|---|
| `dataset/long_term_preference/stage4_recLLM/profiles.jsonl` | JSONL | salient_facts + summary_text |
| `dataset/long_term_preference/stage2_history/personax/{id}__history.json` | JSON | balanced_history.core_sbs |
| `data/optimized_musechat_features_float16_v3/musechat_features_*.h5` | HDF5 v3 | AST 音訊特徵 |

### HDF5 v3 結構

```
pairs/
  {video_id}_{candidate_music_id}/
    target_music_all_cls    [12, 768]   float16  ← Stage 5 使用（mean pool → 768D）
    text_features           [77, 512]   float16  ← 預對齊訓練使用（masked mean pool → 512D）
    candidate_music_all_seq [12, 1214, 768]      ← MVT-Fusion Cross-Attention 使用
    video_features_all      [12, 768]   float16  ← MVT-Fusion 使用
```

> **注意：** v3 結構不含 `metadata` group。music_id 從 pair_key 用 `rsplit('_', 1)` 分離，取 video_id（前段）和 candidate_music_id（後段）雙索引。

### 輸出

| 路徑 | 格式 | 說明 |
|---|---|---|
| `dataset/stage5_output/preference_vectors.h5` | HDF5 | `preference_vectors/{music_id}` → float32 [256D] |
| `dataset/stage5_output/preference_vectors_explicit_only.h5` | HDF5 | 消融 Ablation-A |
| `dataset/stage5_output/preference_vectors_implicit_only.h5` | HDF5 | 消融 Ablation-B |
| `dataset/stage5_output/projection_weights.pt` | PyTorch | W_explicit / W_implicit 預對齊權重 |
| `dataset/stage5_output/generation_log.jsonl` | JSONL | 每筆的生成狀態與 core_sbs_matched 統計 |

---

## 7. 執行方式

```bash
cd "<repo_root>"

# 第一次執行（hybrid，同時訓練投影層）
python stage5_preference_representation_v4.py

# 修改 ModelConfig.REPRESENTATION_MODE = "explicit_only" 後
python stage5_preference_representation_v4.py

# 修改 ModelConfig.REPRESENTATION_MODE = "implicit_only" 後
python stage5_preference_representation_v4.py
```

**第一次執行完整流程：**
1. 載入並凍結 CLIP Text Encoder
2. Xavier 初始化 W_explicit / W_implicit / W_out_hybrid / W_out_single
3. 掃描 18 個 HDF5，收集 (text, audio) 訓練配對
4. InfoNCE 預對齊訓練（10 epochs）
5. 儲存 `projection_weights.pt`
6. 逐一處理 84,150 個 profiles，生成並儲存 P_ltp [256D]

**第二、三次執行：**
- 自動偵測 `projection_weights.pt` 存在 → 跳過訓練，直接載入
- 只執行步驟 6，速度明顯較快

---

## 8. 投影層組件說明

| 組件 | 維度 | 訓練方式 | 使用時機 |
|---|---|---|---|
| `W_explicit` | Linear(512→256) | Stage 5 InfoNCE 預對齊 + MVT-Fusion 微調 | 所有模式 |
| `W_implicit` | Linear(768→256) | Stage 5 InfoNCE 預對齊 + MVT-Fusion 微調 | hybrid / implicit_only |
| `W_out_hybrid` | Linear(512→256) | Xavier 初始化 + MVT-Fusion 微調 | hybrid 模式 |
| `W_out_single` | Linear(256→256) | Xavier 初始化 + MVT-Fusion 微調 | explicit_only / implicit_only |

> W_out 在 Stage 5 生成階段使用 Xavier 初始化的固定值（`torch.no_grad()`）。  
> W_out 的真正優化發生在 MVT-Fusion 端對端訓練時（梯度從推薦 Loss 反向傳播）。

---

## 9. 關鍵設計決策

### 為什麼消融實驗不補零，而是用 W_out 統一輸出維度？

補零存在兩個根本問題：

**梯度問題：** explicit_only 模式下後半 256D 永遠為零，對應的梯度也永遠為零，W_out 的後半權重無法學習。三個消融實驗的 W_out 實際學到的是不同的東西，比較的不是「偏好模態的貢獻」而是「不同架構的效能」，消融失去意義。

**信號污染：** 零向量是非語義的人工信號。MVT-Fusion 可能學到「向量後半為零 = explicit_only 場景」的捷徑規律，而不是真正學習顯性偏好的語義內容。

W_out 方案確保：三個實驗輸入維度相同、架構相同、W_out 在相同條件下學習，唯一變數是偏好信號來源，消融結果才具可比性。

### 為什麼 salient_facts 優先於 summary_text？

`salient_facts` 是 Gemma 從三種對話（Positive / Exploratory / Negative）中抽取的結構化事實，每條 fact 對應一個 `conflict_tag`，語義精確且互不冗餘。`summary_text` 是散文描述，語義重疊度高。CLIP 對短句（10~20 字）的編碼品質優於長段落（80~100 字），facts 拼接後通常比整段 summary 的 CLIP 向量更有判別力。

### 為什麼缺失嵌入直接跳過而非補隨機值？

隨機向量進入語義加權的分母，導致 softmax 分布失真；進入加權平均後，隨機方向的干擾無法在後續投影中消除。`generation_log.jsonl` 的 `core_sbs_matched` 欄位記錄每個用戶實際命中的 core music 數量，可追蹤資料完整性，等特徵提取完成後重跑即可提升命中率。

---

## 10. 常見問題排查

| 現象 | 原因 | 解法 |
|---|---|---|
| `core_sbs_matched = 0` | HDF5 特徵提取未完成 | 等待特徵提取完成後重跑 |
| `missing_history` 大量出現 | Stage 2 history 路徑不符 | 確認 `PathConfig.STAGE2_HISTORY_DIR` |
| 預對齊 loss 不下降 | batch_size 太小（InfoNCE 需足夠負例）| 調大 `PRETRAIN_BATCH_SIZE`（建議 ≥ 64）|
| CUDA OOM | batch_size 過大 | 降低 `PRETRAIN_BATCH_SIZE` |
| 第二次執行仍在重訓 | `projection_weights.pt` 路徑不符 | 確認 `ModelConfig.PRETRAIN_SAVE_PATH` |
| P_ltp 維度不是 256D | 使用舊版 v3 程式碼 | 確認使用 v4，`ModelConfig.OUTPUT_DIM = 256` |