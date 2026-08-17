"""
config.py
全局配置檔：包含路徑、模型參數、演算法參數、消融實驗開關以及全域模型載入。
"""
import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ================================
# 1. 路徑配置 (PathConfig)
# ================================
class PathConfig:
    # 專案根目錄 
    BASE_DIR = Path(r"data/user_profiling")
    
    # Stage 0: 原始資料
    MUSIC_METADATA_RAW = BASE_DIR / "music_metadata_simple/music_metadata.json"
    HDF5_DIR = Path("data/optimized_musechat_features_float16_v3")
    JSON_DIR = Path("data/musechat_json")
    
    # Stage 1: 語義整合輸出
    MUSIC_METADATA_ENRICHED = BASE_DIR / "music_metadata_simple/music_metadata_enriched.json"
    
    # Stage 2: 歷史選擇輸出
    STAGE2_OUTPUT_DIR = BASE_DIR / "long_term_preference/stage2_history"
    
    # Stage 3: 對話生成輸出
    STAGE3_OUTPUT_DIR = BASE_DIR / "long_term_preference/stage3_dialogues"
    
    # Stage 4: 偏好提取輸出
    STAGE4_OUTPUT_FILE = BASE_DIR / "long_term_preference/stage4_profiles.jsonl"
    
    # Stage 5: 向量融合輸出
    STAGE5_OUTPUT_FILE = BASE_DIR / "long_term_preference/final_user_vectors.npy"

    @classmethod
    def ensure_dirs(cls):
        """確保所有輸出目錄存在"""
        for path in [cls.STAGE2_OUTPUT_DIR, cls.STAGE3_OUTPUT_DIR, cls.STAGE4_OUTPUT_FILE.parent]:
            path.mkdir(parents=True, exist_ok=True)

# ================================
# 2. 模型配置 (ModelConfig)
# ================================
class ModelConfig:
    # LLM (Ollama)
    OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_MODEL_NAME = "gemma3:4b"
    LLM_TEMP = 0.7
    
    # Embedding Models
    TEXT_ENCODER = "all-mpnet-base-v2"
    AUDIO_DIM = 768

# ================================
# 3. 演算法參數 (AlgoConfig)
# ================================
class AlgoConfig:
    # Stage 2: PersonaX
    POOL_SIZE = 2000
    N_CLUSTERS = 6
    ALPHA = 0.6
    
    # Stage 2: 歷史長度
    CORE_SIZE = 6
    EXPLOR_SIZE = 3
    NEG_SIZE = 1

# ================================
# 4. 消融實驗配置 (AblationConfig)
# ================================
class AblationConfig:
    # 實驗 1: 採樣策略 (personax, top_n, random, full)
    SAMPLING_STRATEGY = 'personax' 

    # 實驗 2: 合成策略 (diverse_template, single_template, free_form)
    SYNTHESIS_MODE = 'diverse_template'

    # 實驗 3: 偏好表示 (hybrid, implicit_only, explicit_only)
    REPRESENTATION_MODE = 'hybrid'
    
    @classmethod
    def get_experiment_suffix(cls):
        suffix = []
        if cls.SAMPLING_STRATEGY != 'personax':
            suffix.append(cls.SAMPLING_STRATEGY)
        if cls.REPRESENTATION_MODE != 'hybrid':
            suffix.append(cls.REPRESENTATION_MODE)
        if cls.SYNTHESIS_MODE != 'diverse_template':
            suffix.append(cls.SYNTHESIS_MODE)
        
        return "_" + "_".join(suffix) if suffix else ""

# ================================
# 5. 全域模型單例 (GlobalModels)
# ================================
class GlobalModels:
    _sbert = None
    
    @classmethod
    def get_sbert(cls):
        if cls._sbert is None:
            # 這裡使用 ModelConfig 中定義的模型名稱
            cls._sbert = SentenceTransformer(ModelConfig.TEXT_ENCODER)
            if torch.cuda.is_available():
                cls._sbert = cls._sbert.to('cuda')
        return cls._sbert

# 自動創建目錄
PathConfig.ensure_dirs()
