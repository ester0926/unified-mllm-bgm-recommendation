"""
用途：產生或評估 Stage 3 的合成使用者偏好對話。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

import os
import requests
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama 本地模型客戶端（V3：支援動態參數）"""
    
    def __init__(
        self, 
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        top_p: float = 0.9,  # V3 新增
        num_ctx: int = 4096
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p  # V3 新增
        self.num_ctx = num_ctx
        
        logger.info(f"Initialized Ollama client: {model_name} (temp={temperature}, top_p={top_p})")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成回應"""
        try:
            url = f"{self.base_url}/api/generate"
            
            # V3：支援 top_p
            options = {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),  # V3 新增
                "num_ctx": kwargs.get("num_ctx", self.num_ctx),
            }
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options
            }
            
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
        
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timeout for model {self.model_name}")
            raise Exception(f"Request timeout for {self.model_name}")
        
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise
    
    def is_available(self) -> bool:
        """檢查 Ollama 服務和模型是否可用"""
        try:
            # 檢查服務
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # 檢查模型是否已下載
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # 支援完整名稱或簡短名稱匹配
            return any(
                self.model_name in name or name.startswith(self.model_name.split(":")[0])
                for name in model_names
            )
        
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False
    
    def list_models(self) -> list:
        """列出所有可用模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            return [m.get("name", "") for m in models]
        
        except Exception as e:
            logger.error(f"Failed to get model list: {e}")
            return []
    
    def get_info(self) -> Dict:
        """獲取模型資訊"""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,  # V3 新增
            "num_ctx": self.num_ctx
        }


# ================================
# 推薦的英文任務模型（2025 最新）
# ================================
RECOMMENDED_MODELS_FOR_ENGLISH = {
    # 🔥 大型模型（70B+）- 最佳品質
    "llama3.3:70b": {
        "description": "Meta Llama 3.3 70B (Dec 2024) - NEW! Near Llama 3.1 405B performance ⭐",
        "size": "~43GB",
        "min_vram": "48GB",
        "release": "December 2024",
        "context": "128K tokens",
        "pros": ["State-of-the-art quality", "405B-level performance", "Latest release"],
        "cons": ["Requires high-end GPU", "Slower inference"],
        "recommended_for": "Best quality English dialogue (2025)",
        "speed": "⭐⭐ (slow)",
        "quality": "⭐⭐⭐⭐⭐ (excellent)",
        "priority": 1
    },
    
    "mixtral:8x22b": {
        "description": "Mixtral 8x22B - Large MoE model",
        "size": "~80GB",
        "min_vram": "80GB",
        "release": "2024",
        "context": "64K tokens",
        "pros": ["Top-tier quality", "Multilingual", "Strong reasoning"],
        "cons": ["Very large", "Slow inference"],
        "recommended_for": "Research, high-quality generation",
        "speed": "⭐ (very slow)",
        "quality": "⭐⭐⭐⭐⭐ (excellent)",
        "priority": 2
    },
    
    # 🎯 中型模型（12B-30B）- 平衡品質與效能（推薦）
    "gemma3:12b": {
        "description": "Google Gemma 3 12B (Nov 2024) - NEW! Multimodal, 128K context ⭐⭐⭐",
        "size": "~8.1GB",
        "min_vram": "16GB",
        "release": "November 2024",
        "context": "128K tokens",
        "pros": ["Latest Gemma", "Multimodal", "Long context", "Fast"],
        "cons": ["Newer, less battle-tested"],
        "recommended_for": "Modern dialogue with long context (2025 recommended)",
        "speed": "⭐⭐⭐⭐ (fast)",
        "quality": "⭐⭐⭐⭐ (very good)",
        "priority": 1
    },
    
    "mixtral:8x7b": {
        "description": "Mixtral 8x7B - Proven MoE model",
        "size": "~26GB",
        "min_vram": "32GB",
        "release": "2023",
        "context": "32K tokens",
        "pros": ["Excellent performance", "Good quality", "Efficient"],
        "cons": ["Medium VRAM requirement"],
        "recommended_for": "Balanced quality and speed",
        "speed": "⭐⭐⭐ (moderate)",
        "quality": "⭐⭐⭐⭐ (very good)",
        "priority": 2
    },
    
    # ⚡ 小型模型（3B-7B）- 最佳日常選擇（強烈推薦）
    "gemma3:4b": {
        "description": "Google Gemma 3 4B (Nov 2024) - BEST CHOICE for most users ⭐⭐⭐⭐⭐",
        "size": "~3.3GB",
        "min_vram": "6GB",
        "release": "November 2024",
        "context": "128K tokens",
        "pros": ["Latest release", "Multimodal", "Very long context", "Fast", "Efficient"],
        "cons": ["Smaller capacity than 12B"],
        "recommended_for": "BEST OVERALL - Perfect balance for dialogue generation (2025)",
        "speed": "⭐⭐⭐⭐⭐ (very fast)",
        "quality": "⭐⭐⭐⭐ (very good)",
        "priority": 1
    },
    
    "llama3.2:3b": {
        "description": "Meta Llama 3.2 3B (Sep 2024) - Optimized for dialogue ⭐⭐⭐",
        "size": "~2.0GB",
        "min_vram": "4GB",
        "release": "September 2024",
        "context": "128K tokens",
        "pros": ["Very fast", "Low VRAM", "Chat-optimized", "Long context"],
        "cons": ["Less capable than larger models"],
        "recommended_for": "Fast dialogue, low VRAM systems",
        "speed": "⭐⭐⭐⭐⭐ (very fast)",
        "quality": "⭐⭐⭐⭐ (very good)",
        "priority": 2
    },
    
    "mistral:7b": {
        "description": "Mistral 7B v0.3 - Proven reliable workhorse",
        "size": "~4.1GB",
        "min_vram": "8GB",
        "release": "2024 (v0.3)",
        "context": "32K tokens",
        "pros": ["Very fast", "Stable", "Community support", "Battle-tested"],
        "cons": ["Older architecture"],
        "recommended_for": "Reliable fast generation, proven quality",
        "speed": "⭐⭐⭐⭐⭐ (very fast)",
        "quality": "⭐⭐⭐⭐ (very good)",
        "priority": 3
    },
    
    # 🪶 超小型模型（1B-2B）- 資源受限環境
    "gemma3:1b": {
        "description": "Google Gemma 3 1B (Nov 2024) - NEW! Ultra-light",
        "size": "~815MB",
        "min_vram": "2GB",
        "release": "November 2024",
        "context": "32K tokens (text-only)",
        "pros": ["Extremely small", "Fast", "Latest architecture"],
        "cons": ["Limited capability", "Text-only"],
        "recommended_for": "Ultra-lightweight, edge devices",
        "speed": "⭐⭐⭐⭐⭐ (extremely fast)",
        "quality": "⭐⭐⭐ (good for size)",
        "priority": 1
    },
    
    "llama3.2:1b": {
        "description": "Meta Llama 3.2 1B (Sep 2024) - Mobile-optimized",
        "size": "~1.3GB",
        "min_vram": "2GB",
        "release": "September 2024",
        "context": "128K tokens",
        "pros": ["Tiny footprint", "Multilingual", "Mobile-ready"],
        "cons": ["Very limited capability"],
        "recommended_for": "Mobile and edge devices",
        "speed": "⭐⭐⭐⭐⭐ (extremely fast)",
        "quality": "⭐⭐⭐ (good for size)",
        "priority": 2
    },
}


def get_model_recommendation(vram_gb: int) -> str:
    """根據 VRAM 推薦 2025 最佳模型"""
    if vram_gb >= 80:
        return "mixtral:8x22b"  # 最強
    elif vram_gb >= 48:
        return "llama3.3:70b"  # 2025 新王者
    elif vram_gb >= 32:
        return "mixtral:8x7b"
    elif vram_gb >= 16:
        return "gemma3:12b"  # 2025 新推薦
    elif vram_gb >= 6:
        return "gemma3:4b"  # ⭐ 2025 最佳選擇
    elif vram_gb >= 4:
        return "llama3.2:3b"
    elif vram_gb >= 2:
        return "gemma3:1b"
    else:
        return "gemma3:1b"


def print_model_recommendations():
    """打印所有推薦模型的資訊（2025 版本）"""
    print("\n" + "="*80)
    print("RECOMMENDED OLLAMA MODELS FOR ENGLISH TASKS (2025 Latest)")
    print("="*80 + "\n")
    
    categories = {
        "🔥 Large Models (70B+) - Best Quality": [
            "llama3.3:70b", "mixtral:8x22b"
        ],
        "🎯 Medium Models (12B-30B) - Balanced (Recommended)": [
            "gemma3:12b", "mixtral:8x7b"
        ],
        "⚡ Small Models (3B-7B) - Best Daily Choice (HIGHLY RECOMMENDED)": [
            "gemma3:4b", "llama3.2:3b", "mistral:7b"
        ],
        "🪶 Tiny Models (1B-2B) - Resource Constrained": [
            "gemma3:1b", "llama3.2:1b"
        ]
    }
    
    for category, models in categories.items():
        print(f"\n{'─'*80}")
        print(f"{category}")
        print(f"{'─'*80}")
        
        for model_name in models:
            if model_name in RECOMMENDED_MODELS_FOR_ENGLISH:
                info = RECOMMENDED_MODELS_FOR_ENGLISH[model_name]
                print(f"\n📦 Model: {model_name}")
                print(f"   Description: {info['description']}")
                print(f"   Release: {info['release']}")
                print(f"   Size: {info['size']} (Min VRAM: {info['min_vram']})")
                print(f"   Context: {info['context']}")
                print(f"   Speed: {info['speed']}")
                print(f"   Quality: {info['quality']}")
                print(f"   Pros: {', '.join(info['pros'])}")
                print(f"   Use case: {info['recommended_for']}")
    
    print("\n" + "="*80)
    print("🌟 2025 RECOMMENDATIONS:")
    print("="*80)
    print("✅ BEST OVERALL: gemma3:4b (6GB VRAM) - Latest, fast, multimodal ⭐⭐⭐⭐⭐")
    print("✅ BEST QUALITY: llama3.3:70b (48GB VRAM) - Near 405B performance")
    print("✅ BEST BALANCED: gemma3:12b (16GB VRAM) - Modern, 128K context")
    print("✅ FASTEST: mistral:7b or llama3.2:3b")
    print("✅ LOWEST VRAM: gemma3:1b (2GB VRAM)")
    print("="*80 + "\n")
    
    print("📊 WHY THESE 2025 MODELS?")
    print("─"*80)
    print("• Gemma 3 (Nov 2024): Multimodal, 128K context, 140+ languages")
    print("• Llama 3.3 (Dec 2024): Near 405B quality in 70B size")
    print("• Llama 3.2 (Sep 2024): Optimized for dialogue and mobile")
    print("─"*80 + "\n")


if __name__ == "__main__":
    # 打印模型推薦
    print_model_recommendations()
    
    # 根據 VRAM 推薦
    print("\n🎯 RECOMMENDATIONS BY VRAM (2025):")
    print("="*80)
    for vram in [2, 4, 6, 8, 16, 24, 32, 48, 80]:
        model = get_model_recommendation(vram)
        info = RECOMMENDED_MODELS_FOR_ENGLISH.get(model, {})
        release = info.get("release", "")
        print(f"{vram:2d}GB VRAM → {model:20s} ({release})")
    print("="*80 + "\n")