# UCR 錯誤來源拆解（B2）

- 產生時間：2026-07-29T15:52:11
- 主條件：`full`（論文報告的 UCR = 13.12% 即此條件）
- 外部證據：musicnn 標籤 + genre（可用）

## 一、UCR 的三級界線（判讀基準）

現行 UCR 以**子句**為單位計算，而 `split_claims()` 會在 ` and ` / `,` 處切斷句子，使「The song has a soulful」這類殘句失去可歸屬的關鍵詞。把每條 claim 還原回**母句**後以同一支規則重判，再對曲風／具名實體做證據比對，即可逐層分離「標註假影」「規則漏判」與「真正無法歸屬的主張」。

| 條件 | claim 數 | L1 報告值（上界） | L2 句子層級 | L3 證據校正（下界） | 斷句假影 | 規則漏判 | 生成崩潰 | 真實無支持 |
|---|---|---|---|---|---|---|---|---|
| full | 1090 | 13.12% | 3.12% | 1.65% | 109 | 16 | 0 | 18 |
| wo_audio_all | 1015 | 78.62% | 66.90% | 65.52% | 119 | 14 | 0 | 37 |
| wo_audio_feature_only | 953 | 54.14% | 41.45% | 39.77% | 121 | 16 | 0 | 26 |
| wo_ltp | 329 | 99.70% | 99.70% | 99.70% | 0 | 0 | 328 | 0 |
| wo_prompt | 1061 | 17.72% | 3.02% | 2.36% | 156 | 7 | 0 | 13 |
| wo_video | 939 | 17.68% | 4.37% | 2.77% | 125 | 15 | 0 | 22 |

> **判讀基準**：真值介於 L1 與 L3 之間。L1（子句層級）偏嚴，因為殘句必然缺少關鍵詞；L2 只扣除斷句假影；L3 進一步扣除「規則漏判但有證據支持」者，偏寬。主條件 `full` 的區間為 **1.65% – 13.12%**（L2 = 3.12%），定點估計需人工複核（樣板：`human_review_template.csv`，共 363 列）。

> ⚠ **`wo_ltp` 條件不可與其他條件並列解讀**：該條件下 200/200 段生成均為退化輸出（重複、亂碼、空輸出），其接近 100% 的 UCR 反映的是**生成崩潰**，而非「說明缺乏偏好證據」。詳見第四節。

## 二、主條件（full）的錯誤來源組成

以下針對 143 條 `no_detected_support_source` 的 claim 分類。

| 代碼 | 錯誤來源 | n | 占無來源者 | 占全部 claim |
|---|---|---|---|---|
| `E0` | 斷句碎片（標註流程假影） | 109 | 76.2% | 10.00% |
| `E0b` | 規則漏判但有證據支持 | 16 | 11.2% | 1.47% |
| `E2` | 音樂元資料不支持 | 1 | 0.7% | 0.09% |
| `E3` | 影片內容不支持 | 1 | 0.7% | 0.09% |
| `E4` | 一般性空泛描述／主觀陳述 | 14 | 9.8% | 1.28% |
| `E6` | 可能幻覺 | 2 | 1.4% | 0.18% |

> 上表以**子句**計數。同一母句被切成多個殘句時會重複計入，故另附句子層級去重結果（`ucr_error_source_composition_sentence_level.csv`）：

| 錯誤來源 | 母句數 | 占比 |
|---|---|---|
| 斷句碎片（標註流程假影） | 105 | 78.9% |
| 規則漏判但有證據支持 | 13 | 9.8% |
| 音樂元資料不支持 | 1 | 0.8% |
| 影片內容不支持 | 1 | 0.8% |
| 一般性空泛描述／主觀陳述 | 11 | 8.3% |
| 可能幻覺 | 2 | 1.5% |

## 三、各錯誤來源的代表案例

**斷句碎片（標註流程假影）**（`parent_audio-supported`）
- claim：`This track has a beautiful indie`
- 母句：`This track has a beautiful indie and folk sound, with a touch of melancholy.`
- 判定依據：母句可歸屬，unsupported 係 split_claims 於 and/, 處切斷所致

**規則漏判但有證據支持**（`genre_claim_verified_by_tags`）
- claim：`It's a track that captures the essence of summer with its catchy pop`
- 母句：`It's a track that captures the essence of summer with its catchy pop and indie rock elements.`
- 判定依據：曲風敘述可由 musicnn 標籤／參考文本支持；v2 因 GENRE_TERMS 需搭配 anchor 詞才計為 audio 而漏判

**一般性空泛描述／主觀陳述**（`generic_video_benefit`）
- claim：`It's a high-energy song that will definitely get the energy up in your video.`
- 母句：`It's a high-energy song that will definitely get the energy up in your video.`
- 判定依據：泛用效用或主觀陳述，本質上不具可驗證性

**可能幻覺**（`named_entity_absent_from_all_evidence`）
- claim：`It's a track from his mixtape 'The Tay Tape'.`
- 母句：`It's a track from his mixtape 'The Tay Tape'.`
- 判定依據：引號名稱 ['The Tay Tape'] 未見於管線內證據與外部 metadata；屬具體可查證卻查無支持者，列為可能幻覺並送人工複核

**影片內容不支持**（`specific_video_content_asserted`）
- claim：`It's perfect for adding a touch of romance to your wedding video.`
- 母句：`It's perfect for adding a touch of romance to your wedding video.`
- 判定依據：主張具體影片情境（如婚禮／旅遊），但模型輸入僅有 CLIP 影像特徵，規則層無法驗證，送人工複核

**音樂元資料不支持**（`genre_partially_supported`）
- claim：`It's an electronic track with elements of pop, indie`
- 母句：`It's an electronic track with elements of pop, indie, and rock.`
- 判定依據：部分曲風查無支持：['rock']（musicnn 標籤有雜訊，查無支持不等於事實錯誤）

## 四、`wo_ltp` 的生成崩潰（重要發現）

| 條件 | 退化生成段數 / 總段數 |
|---|---|
| full | 0 / 200 |
| wo_audio_all | 3 / 200 |
| wo_audio_feature_only | 4 / 200 |
| wo_ltp | 200 / 200 |
| wo_prompt | 0 / 200 |
| wo_video | 2 / 200 |

退化判定於**整段生成**層級進行（非子句），準則為：有效英文詞數 < 10、type-token ratio < 0.35、單一詞占比 > 0.25、或平均詞長 ≤ 2.5。門檻以既有資料校準，`full` 與 `wo_prompt` 皆為 0 誤判。

**論述含意**：在此已訓練 checkpoint 與 OOD 歸零設定下，推論時移除 [LTP] 前綴會使生成整體崩潰，而非僅降低偏好接地程度。因此 `wo_ltp` 的 UCR 不可用來支持「說明內容依賴偏好證據」或「LTP 普遍為必要輸入」等主張；它只顯示此模型對既有前綴結構高度敏感。此結果並與排序側的 No-LTP（exp_04，重新訓練且無 LTP）明確區分，後者才是偏好資訊有效性的模型層級對照。

## 五、方法與限制

- **L1**：母句還原後以既有 classify_claim() 重判，區分斷句假影與真正不可歸屬
- **L2**：教授指定六類 + 斷句假影 / 規則漏判兩類方法學桶，規則互斥且固定順序
- **L3**：曲風與具名實體對照 (a) top1_reference_text+title+artist (b) musicnn 標籤
- **caveat**：musicnn 標籤含雜訊，查無支持一律歸為『不支持』，僅具名實體完全查無者升級為『可能幻覺』並送人工複核
- 標記需人工複核的 claim 共 83 條；規則層只做可稽核的初判，最終歸類以人工複核為準。
- 本分析未重跑任何生成，全部基於既有 `counterfactual_generations_top1.csv`。
