# B5 結構化 Persona 規格（24 個）

- 產生時間：2026-07-29T15:22:36
- 結構：6 偏好原型 × 4 內容情境（B4 k=4 影片語意叢集）
- 每個 Persona 的合成歷史長度：20 首（14/4/2，精確對應核心 70% / 鄰近探索 20% / 偏離 10%）

## 一、設計說明

- 內容情境改用 B4 影片語意叢集，而非教授原案的旅遊／美食／知識教學／日常生活：本資料集的影片即音樂自身 YouTube 影片，無法操作化為創作者拍攝的內容類型。
- 情境只決定該 Persona 配對哪些查詢影片，不限制其歷史曲目來源；原型與情境不相符的組合即為 B6 偏好—影片衝突分析的天然素材。
- 節奏維度以 musicnn 的 fast / slow 標籤定義，非 BPM 物理量測。
- 情緒價向（valence）在本資料集 80 個標籤詞彙中無任何對應項，標為不可操作化，僅供自然語言描述，不參與曲目篩選與評分。
- 「純音樂背景型」以『不具任何顯著人聲標籤』定義，而非要求 instrumental 標籤（該標籤僅 424 首，佔 0.49%，過於稀疏）。

## 二、欄位可操作化狀態

| 欄位 | 可操作化 | 操作定義 |
|---|---|---|
| preferred_genres | 是 | — |
| rejected_genres | 是 | — |
| tempo | 是 | 以 musicnn 標籤 fast / slow 定義，非 BPM 物理量測 |
| energy | 是 | 以 loud 標籤之有無定義（high = 具 loud；low = 不具 loud） |
| vocal | 是 | vocal_required = 具顯著人聲標籤；instrumental_leaning = 不具任何顯著人聲標籤（instrumental 標籤僅 424 首，過於稀疏故不採為必要條件） |
| instruments | 是 | — |
| popularity | 是 | 以 youtube view_count 分位數定義：mainstream ≥ p75、niche ≤ p25 |
| novelty | 是 | 控制合成歷史中 70/20/10 三層的取樣半徑，不直接篩選曲目 |
| consistency | 是 | 控制所選曲目的屬性離散度，不直接篩選曲目 |
| selection_rule | **否** | — |
| valence | **否** | 本資料集標籤詞彙（共 80 詞）無任何情緒價向對應項，故不可操作化，僅供自然語言 Persona 描述使用 |

## 三、六種偏好原型

| 原型 | 偏好曲風 | 排斥曲風 | 節奏 | 能量 | 人聲 | 熱門度 | 新穎性 | 一致性 | 候選曲目數 |
|---|---|---|---|---|---|---|---|---|---|
| 高能量節奏型 | techno、dance、house、electro | ambient、chillout、folk、blues | fast | high | any | mainstream | low | high | 4,794 |
| 柔和氛圍型 | ambient、chillout、acoustic、folk | metal、heavy metal、punk、hard rock | slow | low | any | any | low | high | 2,453 |
| 人聲敘事型 | pop、rnb、soul | techno、experimental | any | any | vocal_required | mainstream | low | medium | 6,051 |
| 純音樂背景型 | electronic、ambient、chillout | rnb、soul | any | any | instrumental_leaning | niche | medium | medium | 3,482 |
| 曲風一致型 | rock、alternative rock、indie rock | techno、house、hip-hop | any | any | any | any | very_low | very_high | 22,943 |
| 探索多樣型 | indie、alternative、funk、jazz、country、hip-hop | — | any | any | any | niche | high | low | 16,149 |

## 四、四種內容情境（B4 k=4 影片語意叢集）

| 叢集 | 名稱 | 可用查詢影片數 |
|---|---|---|
| 0 | 主流流行舞曲音樂影片 | 1094 |
| 1 | 粉絲二創／歌詞與玩偶動畫影片 | 1309 |
| 2 | 樂團演出／搖滾金屬音樂影片 | 618 |
| 3 | 嘻哈饒舌／街頭拍攝音樂影片 | 1184 |

## 五、24 個 Persona 一覽

| Persona ID | 偏好原型 | 內容情境 | 候選曲目 | 可行 |
|---|---|---|---|---|
| `P1_high_energy__c0` | 高能量節奏型 | 主流流行舞曲音樂影片 | 4,794 | ✓ |
| `P1_high_energy__c1` | 高能量節奏型 | 粉絲二創／歌詞與玩偶動畫影片 | 4,794 | ✓ |
| `P1_high_energy__c2` | 高能量節奏型 | 樂團演出／搖滾金屬音樂影片 | 4,794 | ✓ |
| `P1_high_energy__c3` | 高能量節奏型 | 嘻哈饒舌／街頭拍攝音樂影片 | 4,794 | ✓ |
| `P2_soft_ambient__c0` | 柔和氛圍型 | 主流流行舞曲音樂影片 | 2,453 | ✓ |
| `P2_soft_ambient__c1` | 柔和氛圍型 | 粉絲二創／歌詞與玩偶動畫影片 | 2,453 | ✓ |
| `P2_soft_ambient__c2` | 柔和氛圍型 | 樂團演出／搖滾金屬音樂影片 | 2,453 | ✓ |
| `P2_soft_ambient__c3` | 柔和氛圍型 | 嘻哈饒舌／街頭拍攝音樂影片 | 2,453 | ✓ |
| `P3_vocal_narrative__c0` | 人聲敘事型 | 主流流行舞曲音樂影片 | 6,051 | ✓ |
| `P3_vocal_narrative__c1` | 人聲敘事型 | 粉絲二創／歌詞與玩偶動畫影片 | 6,051 | ✓ |
| `P3_vocal_narrative__c2` | 人聲敘事型 | 樂團演出／搖滾金屬音樂影片 | 6,051 | ✓ |
| `P3_vocal_narrative__c3` | 人聲敘事型 | 嘻哈饒舌／街頭拍攝音樂影片 | 6,051 | ✓ |
| `P4_instrumental_bed__c0` | 純音樂背景型 | 主流流行舞曲音樂影片 | 3,482 | ✓ |
| `P4_instrumental_bed__c1` | 純音樂背景型 | 粉絲二創／歌詞與玩偶動畫影片 | 3,482 | ✓ |
| `P4_instrumental_bed__c2` | 純音樂背景型 | 樂團演出／搖滾金屬音樂影片 | 3,482 | ✓ |
| `P4_instrumental_bed__c3` | 純音樂背景型 | 嘻哈饒舌／街頭拍攝音樂影片 | 3,482 | ✓ |
| `P5_genre_consistent__c0` | 曲風一致型 | 主流流行舞曲音樂影片 | 22,943 | ✓ |
| `P5_genre_consistent__c1` | 曲風一致型 | 粉絲二創／歌詞與玩偶動畫影片 | 22,943 | ✓ |
| `P5_genre_consistent__c2` | 曲風一致型 | 樂團演出／搖滾金屬音樂影片 | 22,943 | ✓ |
| `P5_genre_consistent__c3` | 曲風一致型 | 嘻哈饒舌／街頭拍攝音樂影片 | 22,943 | ✓ |
| `P6_exploratory__c0` | 探索多樣型 | 主流流行舞曲音樂影片 | 16,149 | ✓ |
| `P6_exploratory__c1` | 探索多樣型 | 粉絲二創／歌詞與玩偶動畫影片 | 16,149 | ✓ |
| `P6_exploratory__c2` | 探索多樣型 | 樂團演出／搖滾金屬音樂影片 | 16,149 | ✓ |
| `P6_exploratory__c3` | 探索多樣型 | 嘻哈饒舌／街頭拍攝音樂影片 | 16,149 | ✓ |

## 六、限制

- 情緒價向欄位不可操作化，任何以該欄位為依據的結論皆不得寫入論文。
- Persona 為研究者依標籤分布設計，非真實創作者；其合成歷史由曲庫依屬性條件抽樣而得，不代表真實聆聽行為。
- Persona LTP 由既有 LTP 向量組合而成，未走完整 Stage 3–5 管線（原因見 `results/analysis/b5_smoketest/wout_recovery_report.md`）。
