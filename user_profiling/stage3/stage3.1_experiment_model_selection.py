"""
用途：產生或評估 Stage 3 的合成使用者偏好對話。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

import time
import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging

# 引用現有模組
from config import PathConfig, ModelConfig, GlobalModels
from llm_client import OllamaClient
from stage3_dialogue_optimized import TwoPhaseGenerator
from stage3.quality_evaluator import QualityEvaluatorLLM
from prompt_builder import build_prompt

# 設定 Log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Experiment")

# ===========================
# 實驗配置 (User Config)
# ===========================
EXP_CONFIG = {
    # 要比較的候選模型
    "CANDIDATE_MODELS": [
        "gemma3:4b", 
        "llama3.2:3b", 
        "mistral:7b"
    ],
    
    # 擔任評審的模型 (建議使用參數量較大的模型，如 gemma3:12b, mixtral, llama3.3:70b)
    "JUDGE_MODEL": "gemma3:12b", 
    
    # 測試樣本數 (從小規模開始，例如 20 首歌)
    "NUM_SAMPLES": 20, 
    
    # 輸出的實驗結果檔案至dataset路徑
    "OUTPUT_CSV": "data/user_profiling/stage3.1/experiment_results_v1.csv"
}

class ExperimentRunner(TwoPhaseGenerator):
    """繼承 TwoPhaseGenerator 但支援動態切換模型"""
    
    def __init__(self, judge_model_name):
        # 初始化父類別 (載入 Metadata)
        super().__init__()
        # 初始化評分器 (使用指定的 Judge Model)
        self.evaluator = QualityEvaluatorLLM()
        self.evaluator.client = OllamaClient(
            model_name=judge_model_name,
            temperature=0.1
        )
        logger.info(f"👨‍⚖️ Judge Model Initialized: {judge_model_name}")

    def run_comparison(self):
        # 1. 準備數據 (隨機採樣)
        samples = self._prepare_samples(EXP_CONFIG["NUM_SAMPLES"])
        logger.info(f"🧪 Loaded {len(samples)} samples for experiment.")
        
        results = []

        # 2. 針對每個模型進行測試
        for model_name in EXP_CONFIG["CANDIDATE_MODELS"]:
            logger.info(f"\n🤖 Testing Model: {model_name}...")
            
            # 動態切換生成模型
            self.client = OllamaClient(model_name=model_name, temperature=0.7)
            
            for sample in tqdm(samples, desc=f"Generating with {model_name}"):
                music_id = sample['target_music']
                balanced_hist = sample['balanced_history']
                
                # 為了實驗簡化，我們只測 "Positive" 對話類型
                dialogue_type = "Positive"
                
                # 準備 Context
                core_sbs = balanced_hist.get('core_sbs', [])
                if not core_sbs: continue
                
                # 生成 Persona (所有模型共用同一組 Persona 比較公平，或是讓每個模型自己生成)
                # 這裡讓每個模型自己生成，測試其指令遵循能力
                pos_snippet, neg_snippet = self.generate_persona_snippets(
                    core_sbs, balanced_hist.get('negative_sbs', [])
                )
                
                target_meta = self.metadata.get(music_id, {"title": "Unknown", "genre": "Unknown"})
                
                # 計時開始
                start_time = time.time()
                
                # 執行生成
                raw_res, turns, _, retry_count = self.generate_dialogue(
                    target_meta, pos_snippet, neg_snippet, core_sbs[:3], dialogue_type
                )
                
                elapsed_time = time.time() - start_time
                
                # 如果生成失敗，記錄失敗
                if not turns:
                    results.append({
                        "model": model_name,
                        "music_id": music_id,
                        "success": False,
                        "time": elapsed_time,
                        "retry_count": retry_count,
                        "overall_score": 0
                    })
                    continue

                # 3. 執行評分 (LLM-as-a-Judge)
                full_persona = f"{pos_snippet} {neg_snippet}"
                scores = self.evaluator.evaluate(turns, full_persona, dialogue_type)
                
                results.append({
                    "model": model_name,
                    "music_id": music_id,
                    "success": True,
                    "time": round(elapsed_time, 2),
                    "retry_count": retry_count,
                    # 新增這裡：詳細記錄 4 個學術指標
                    "coherence": scores.get("coherence", 0),
                    "consistency": scores.get("consistency", 0),
                    "naturalness": scores.get("naturalness", 0),
                    "instruction_following": scores.get("instruction_following", 0),
                    "overall_score": scores.get("overall", 0),
                    "reason": scores.get("reason", "")
                })

        # 4. 輸出結果與統計 (這裡要更新顯示欄位)
        df = pd.DataFrame(results)
        df.to_csv(EXP_CONFIG["OUTPUT_CSV"], index=False)
        
        if not df.empty:
            summary = df.groupby("model").agg({
                "overall_score": "mean",
                "coherence": "mean",      # 新增
                "consistency": "mean",    # 新增
                "naturalness": "mean",    # 新增
                "instruction_following": "mean", # 新增
                "time": "mean",
                "success": "mean"
            }).sort_values("overall_score", ascending=False)
            
            print("\n" + "="*80)
            print("🏆 Model Comparison Report (Literature-Aligned Metrics)")
            print("="*80)
            print(summary)
            print("="*80)
            summary.to_csv("data/user_profiling/stage3.1/experiment_summary_report.csv")

    def _prepare_samples(self, n):
        """從 Stage 2 輸出中隨機讀取 N 筆資料"""
        input_dir = PathConfig.STAGE2_OUTPUT_DIR
        # 簡單搜尋所有 history json
        files = list(input_dir.rglob("*_history.json"))
        
        if len(files) < n:
            selected_files = files
        else:
            selected_files = random.sample(files, n)
            
        data = []
        for p in selected_files:
            with open(p, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        return data

    def _save_and_report(self, results):
        df = pd.DataFrame(results)
        
        # 存檔
        df.to_csv(EXP_CONFIG["OUTPUT_CSV"], index=False)
        logger.info(f"📄 Raw results saved to {EXP_CONFIG['OUTPUT_CSV']}")
        
        # 計算平均統計
        if not df.empty:
            summary = df.groupby("model").agg({
                "overall_score": "mean",
                "coherence": "mean",
                "consistency": "mean",
                "naturalness": "mean",
                "time": "mean",
                "success": "mean" # 成功率
            }).sort_values("overall_score", ascending=False)
            
            print("\n" + "="*50)
            print("🏆 Model Comparison Report (Avg Scores)")
            print("="*50)
            print(summary)
            print("="*50)
            
            # 儲存統計表
            summary.to_csv("experiment_summary_report.csv")

if __name__ == "__main__":
    runner = ExperimentRunner(judge_model_name=EXP_CONFIG["JUDGE_MODEL"])
    runner.run_comparison()