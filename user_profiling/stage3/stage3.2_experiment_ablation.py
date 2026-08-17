"""
Ablation Experiment: Prompt Components Verification
論文實驗專用：驗證 CoT (Chain-of-Thought) 與 Reflection 機制的個別貢獻
"""
import time
import json
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
import random

# 引用現有模組
from config import PathConfig, ModelConfig
from llm_client import OllamaClient
from stage3_dialogue_optimized import TwoPhaseGenerator
from prompt_builder import build_prompt
from qa_validator import validate_dialogue
from quality_evaluator import evaluate_dialogue_quality

# 設定 Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AblationExp")

# ===========================
# 實驗配置
# ===========================
ABLATION_CONFIG = {
    "NUM_SAMPLES": 20,  # 樣本數 (建議 15-20 筆即可)
    "OUTPUT_FILE": "data/user_profiling/stage3.2/ablation_results_v1.csv",
    "SETTINGS": {
        "Full_Method":   {"use_cot": True,  "use_reflection": True},
        "No_CoT":        {"use_cot": False, "use_reflection": True},
        "No_Reflection": {"use_cot": True,  "use_reflection": False},
    }
}

class AblationRunner(TwoPhaseGenerator):
    """
    繼承 Stage 3 生成器，但允許動態調整 Prompt 參數
    """
    def __init__(self):
        super().__init__()
        # 確保使用選定的基座模型
        logger.info(f"🧬 Base Model: {ModelConfig.LLM_MODEL_NAME}")

    def generate_dialogue_variant(
        self, 
        target_meta, pos_snippet, neg_snippet, history, 
        dialogue_type, 
        setting_name, 
        params
    ):
        """
        執行單次實驗生成 (針對特定設定)
        """
        max_retries = 3
        last_failure_reason = None
        start_time = time.time()
        
        use_cot = params["use_cot"]
        use_reflection = params["use_reflection"]

        for attempt in range(max_retries):
            # 1. 構建 Prompt (根據消融參數)
            # 如果禁用 Reflection，即使是 Retry 也不傳入 last_failure_reason
            reason_to_pass = last_failure_reason if use_reflection else None
            
            prompt = build_prompt(
                target_music=target_meta,
                ltp_text=pos_snippet,
                search_history=history,
                dialogue_type=dialogue_type,
                turn_number=10,
                retry_attempt=attempt,
                last_failure_reason=reason_to_pass, # 關鍵：控制是否啟用反思
                user_dislikes=neg_snippet,
                use_cot=use_cot,               # 關鍵：控制是否啟用 CoT
                use_reflection=use_reflection 
            )

            try:
                # 溫度設定：重試時若無 Reflection，保持高溫或略降；有 Reflection 則降溫
                temp = 0.7 if attempt == 0 else 0.5
                raw_res = self.client.generate(prompt, temperature=temp)
                
                # 解析
                plan, turns = self._parse_response_flexible(raw_res)
                
                if not turns:
                    last_failure_reason = "Format Error: No dialogue turns parsed."
                    continue

                # 2. 驗證 (Validator)
                full_persona = f"{pos_snippet} {neg_snippet}"
                is_valid, f_type, f_reason = validate_dialogue(
                    turns, dialogue_type, full_persona, retry_attempt=attempt
                )
                
                if is_valid:
                    elapsed = time.time() - start_time
                    
                    # 3. 評分 (LLM-as-a-Judge)
                    # 注意：這裡會呼叫 quality_evaluator.py 的新版 evaluate_dialogue_quality
                    scores = evaluate_dialogue_quality(turns, full_persona, dialogue_type)
                    
                    return {
                        "setting": setting_name,
                        "success": True,
                        "turns": len(turns),
                        "retry_count": attempt,
                        "time": round(elapsed, 2),
                        "scores": scores
                    }
                
                last_failure_reason = f"{f_type}: {f_reason}"
                
            except Exception as e:
                logger.error(f"Error: {e}")
                last_failure_reason = "System Error"

        # 失敗回傳
        return {
            "setting": setting_name,
            "success": False,
            "retry_count": max_retries,
            "time": round(time.time() - start_time, 2),
            "scores": {} # Empty scores
        }

    def run_experiment(self):
        # 1. 準備數據 (隨機採樣)
        samples = self._prepare_samples(ABLATION_CONFIG["NUM_SAMPLES"])
        logger.info(f"🧪 Loaded {len(samples)} samples for ablation study.")
        
        results = []
        
        for sample in tqdm(samples, desc="Running Samples"):
            music_id = sample.get('target_music', 'Unknown')
            balanced_hist = sample.get('balanced_history', {})
            core_sbs = balanced_hist.get('core_sbs', [])
            if not core_sbs: continue
            
            # 生成 Persona
            pos_snippet, neg_snippet = self.generate_persona_snippets(
                core_sbs, balanced_hist.get('negative_sbs', [])
            )
            target_meta = self.metadata.get(music_id, {"title": "Unknown", "genre": "Unknown"})
            
            # 針對每一種實驗設定跑一次
            for setting_name, params in ABLATION_CONFIG["SETTINGS"].items():
                
                # 統一測試 "Positive" 類型以控制變因
                res = self.generate_dialogue_variant(
                    target_meta, pos_snippet, neg_snippet, core_sbs[:3], 
                    "Positive", setting_name, params
                )
                
                # 展開分數以便存成 CSV
                row = {
                    "music_id": music_id,
                    "setting": setting_name,
                    "success": res["success"],
                    "retry_count": res["retry_count"],
                    "time": res["time"],
                    # 展開 Score Dict
                    "overall": res["scores"].get("overall", 0),
                    "coherence": res["scores"].get("coherence", 0),
                    "consistency": res["scores"].get("consistency", 0),
                    "naturalness": res["scores"].get("naturalness", 0),
                    "instruction_following": res["scores"].get("instruction_following", 0)
                }
                results.append(row)

        # 2. 存檔與分析
        self._save_report(results)

    def _prepare_samples(self, n):
        """讀取並隨機採樣"""
        input_dir = PathConfig.STAGE2_OUTPUT_DIR
        files = list(input_dir.rglob("*_history.json"))
        if len(files) > n:
            files = random.sample(files, n)
        data = []
        for p in files:
            with open(p, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        return data

    def _save_report(self, results):
        df = pd.DataFrame(results)
        df.to_csv(ABLATION_CONFIG["OUTPUT_FILE"], index=False)
        logger.info(f"📄 Saved raw results to {ABLATION_CONFIG['OUTPUT_FILE']}")

        if not df.empty:
            # 計算平均值
            summary = df.groupby("setting").agg({
                "overall": "mean",
                "consistency": "mean",        # 預期 CoT 影響最大
                "instruction_following": "mean", # 預期 Reflection 影響最大
                "coherence": "mean",
                "success": "mean",
                "retry_count": "mean",
                "time": "mean"
            }).sort_values("overall", ascending=False)

            print("\n" + "="*60)
            print("🔬 Ablation Study Results (Avg Scores)")
            print("="*60)
            print(summary)
            print("="*60)
            summary.to_csv("data/user_profiling/stage3.2/ablation_summary.csv")

if __name__ == "__main__":
    runner = AblationRunner()
    runner.run_experiment()