"""
Stage 1: Metadata Construction & Semantic Integration
功能：
1. 載入原始 music_metadata.json (來自 Stage 0)
2. 執行「語義整合」：將結構化標籤轉為自然語言描述 (Semantic Seed)
   Spec 3.2.3: Semantic_Seed(m) = Template(Tags(m), Metadata(m))
3. 輸出 enriched_metadata.json 供 Stage 2 使用
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
from stage3.config import PathConfig  # 依賴 config.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetadataIntegrator:
    def __init__(self):
        self.input_path = PathConfig.MUSIC_METADATA_RAW
        self.output_path = PathConfig.MUSIC_METADATA_ENRICHED

    def load_data(self):
        if not self.input_path.exists():
            raise FileNotFoundError(f"找不到輸入檔案: {self.input_path}")
        
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"載入 {len(data)} 筆原始 metadata")
        return data

    def create_semantic_seed(self, music_id, info):
        """
        實作 Spec 3.2.3: 使用模板將元數據轉換為語義種子
        Semantic_Seed(m) = Template(Tags(m), Metadata(m))
        """
        # 1. 提取並清理欄位
        title = info.get('title', 'Unknown Title')
        artist = info.get('artist', 'Unknown Artist')
        genre = info.get('genre', 'unknown')
        
        # Tags 處理：取前 5 個高信心度標籤
        tags = info.get('tags', [])
        # 相容性處理：如果 tags 是物件列表 [{"tag": "jazz", "confidence": 0.9}, ...]
        if tags and isinstance(tags[0], dict):
            tags = [t.get('tag') for t in tags]
        
        # 過濾掉 genre 本身重複出現在 tags 的情況，增加描述豐富度
        filtered_tags = [t for t in tags if t.lower() != genre.lower()]
        top_tags = filtered_tags[:5]
        tags_str = ", ".join(top_tags) if top_tags else "various musical elements"

        # 2. 外部元數據 (Social Metrics)
        # Spec 3.2.2 提到使用 views 來反映 popularity
        views = info.get('view_count', None)
        popularity_desc = ""
        if views:
            try:
                v_int = int(views)
                if v_int > 5000000: popularity_desc = " with massive popularity"
                elif v_int > 1000000: popularity_desc = " with high popularity"
                elif v_int > 100000: popularity_desc = " with moderate popularity"
            except: pass

        # 3. 模板化整合 (Template Approach)
        # 參考文獻範例：「Features smooth jazz piano melodies with high popularity... created by professional jazz artist」
        
        # 建構種子句子
        seed = (
            f"Features {genre} style{popularity_desc}, titled '{title}' by {artist}. "
            f"Characterized by {tags_str}."
        )

        return seed

    def validate_and_enrich(self):
        data = self.load_data()
        enriched_count = 0
        missing_fields_count = 0

        # 用來檢查格式的 Report
        sample_checked = False

        for mid, info in tqdm(data.items(), desc="Generating Semantic Seeds"):
            # 格式檢查與補缺
            required = ['title', 'artist', 'genre', 'tags']
            for k in required:
                if k not in info:
                    info[k] = 'unknown' if k != 'tags' else []
                    missing_fields_count += 1
            
            # 產生 Semantic Seed
            seed = self.create_semantic_seed(mid, info)
            info['semantic_seed'] = seed
            enriched_count += 1

            # 顯示第一筆結果給用戶檢查 (Debug 用)
            if not sample_checked:
                logger.info(f"\n[Sample Check] ID: {mid}")
                logger.info(f"  Genre: {info.get('genre')}")
                logger.info(f"  Tags: {info.get('tags')[:3]}")
                logger.info(f"  -> Generated Seed: {seed}\n")
                sample_checked = True

        logger.info(f"處理完成。成功生成: {enriched_count} 筆, 自動修補缺失欄位: {missing_fields_count} 處")
        
        # 儲存
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"已儲存至: {self.output_path}")

if __name__ == "__main__":
    integrator = MetadataIntegrator()
    integrator.validate_and_enrich()