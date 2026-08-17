"""
Quality Evaluator V2 (Literature-Aligned LLM-as-a-Judge)
功能：
1. 使用較強的 LLM (Judge Model) 對生成的對話進行多維度評分 (1-5分)。
2. 評估維度依據：
   - Coherence (G-Eval)
   - Consistency (RoleLLM)
   - Naturalness (Human-likeness)
   - Instruction Following (MT-Bench)
3. 保留 SBERT 做為輔助的客觀語意指標 (LTP Coverage)。
"""

import json
import logging
import re
import numpy as np
from typing import Dict, List
from config import GlobalModels, ModelConfig
from llm_client import OllamaClient

logger = logging.getLogger(__name__)

# ================================
# 獨立函式介面 (保持向後兼容)
# ================================
def evaluate_dialogue_quality(
    dialogue_turns: List[Dict],
    ltp_text: str,
    dialogue_type: str
) -> Dict:
    """
    外部呼叫接口
    """
    evaluator = QualityEvaluatorLLM()
    return evaluator.evaluate(dialogue_turns, ltp_text, dialogue_type)


class QualityEvaluatorLLM:
    def __init__(self, judge_model_name: str = "gemma3:12b"):
        """
        初始化評分器
        :param judge_model_name: 擔任裁判的模型名稱 (建議比生成模型大，如 gemma3:12b, llama3:70b)
        """
        # 1. 初始化 SBERT (用於計算客觀的 LTP Coverage)
        self.sbert = GlobalModels.get_sbert()
        
        # 2. 初始化 Judge LLM Client
        # 注意：這裡使用 temperature=0.0 以確保評分的穩定性 (Deterministic)
        self.judge_model = judge_model_name
        
        # 嘗試使用指定的 Judge Model，如果 Ollama 沒跑起來或沒下載，client 內部會報錯，
        # 但我們這裡先假設使用者已經 pull 了 gemma3:12b
        self.client = OllamaClient(
            model_name=self.judge_model,
            base_url=ModelConfig.OLLAMA_BASE_URL,
            temperature=0.0 
        )
        logger.info(f"⚖️ Quality Evaluator initialized with Judge: {self.judge_model}")

    def evaluate(self, dialogue_turns: List[Dict], ltp_text: str, dialogue_type: str) -> Dict:
        """
        執行混合評估：
        1. SBERT Similarity (客觀語意距離)
        2. LLM Judge Score (主觀品質評分 1-5)
        """
        if not dialogue_turns:
            return self._get_fallback_scores("Empty dialogue")

        # 1. 準備對話文本
        dialogue_text = "\n".join([f"{t['role']}: {t['content']}" for t in dialogue_turns])
        user_turns = [t['content'] for t in dialogue_turns if t['role'] == 'User']
        last_turn = user_turns[-1] if user_turns else ""

        # 2. 計算 SBERT LTP Coverage (0.0 - 1.0)
        # 這是檢查 "最後一句話" 是否有提到 "長期偏好" 的客觀指標
        ltp_coverage = self._compute_embedding_similarity(last_turn, ltp_text)

        # 3. 執行 LLM 評分 (1-5 Scale)
        llm_scores = self._ask_llm_judge(dialogue_text, ltp_text, dialogue_type)

        # 4. 整合分數
        final_scores = {
            "ltp_coverage": round(ltp_coverage, 3), # SBERT score
            "coherence": llm_scores.get("coherence", 1),
            "consistency": llm_scores.get("consistency", 1),
            "naturalness": llm_scores.get("naturalness", 1),
            "instruction_following": llm_scores.get("instruction_following", 1),
            "overall": llm_scores.get("overall", 1.0),
            "reason": llm_scores.get("reason", "N/A")
        }

        return final_scores

    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """計算兩段文字的 Cosine Similarity"""
        if not text1 or not text2: return 0.0
        try:
            e1 = self.sbert.encode(text1, show_progress_bar=False)
            e2 = self.sbert.encode(text2, show_progress_bar=False)
            sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            return float(sim)
        except Exception:
            return 0.0

    def _ask_llm_judge(self, dialogue_text: str, ltp_text: str, dtype: str) -> Dict:
        """
        G-Eval & MT-Bench Style Prompting
        """
        prompt = f"""
        You are an expert evaluator for a Dialogue System.
        Your task is to evaluate a music recommendation dialogue based on the User Persona.
        
        **Input Data:**
        - User Persona (LTP): "{ltp_text}"
        - Dialogue Type: {dtype}
        - Dialogue Content:
        {dialogue_text}
        
        **Evaluation Criteria (Score 1-5):**
        
        1. **Coherence (1-5)**: (Logic & Flow)
           - Is the conversation logically connected? 
           - Does the Recommender directly address the User's inputs?
           
        2. **Consistency (1-5)**: (Persona Adherence - Ref: RoleLLM)
           - Does the User's behavior strictly align with the 'User Persona'? 
           - If Persona says 'hates rock', the User MUST NOT ask for rock.
           - If Dialogue Type is 'Negative', does the user reject the music?
           
        3. **Naturalness (1-5)**: (Human-likeness - Ref: G-Eval)
           - Does the dialogue sound like a real human conversation? 
           - Is the tone appropriate (e.g., hesitant for 'Exploratory' type)?
           - Avoids robotic or repetitive phrases.
           
        4. **Instruction Following (1-5)**: (Constraints - Ref: MT-Bench)
           - Does the dialogue end at Turn 10?
           - Does the User clearly summarize their final preference in the last turn?
           - Is the format correct (no "User says:" prefixes, clean dialogue)?
        
        **Output Format:**
        Return ONLY a JSON object with integer scores (1-5).
        {{
            "coherence": <int>,
            "consistency": <int>,
            "naturalness": <int>,
            "instruction_following": <int>,
            "reason": "<short explanation>"
        }}
        """
        
        try:
            response = self.client.generate(prompt, temperature=0.0)
            
            # 清理並解析 JSON (增強魯棒性)
            clean_json = response.replace("```json", "").replace("```", "").strip()
            # 嘗試提取 JSON 區塊
            json_match = re.search(r'\{.*\}', clean_json, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group(0))
            else:
                scores = json.loads(clean_json)

            # 計算平均分 (Overall Quality)
            scores["overall"] = (
                scores.get("coherence", 3) + 
                scores.get("consistency", 3) + 
                scores.get("naturalness", 3) + 
                scores.get("instruction_following", 3)
            ) / 4.0
            
            return scores
            
        except Exception as e:
            logger.warning(f"LLM Judge failed: {e}")
            return self._get_fallback_scores(f"Judge Error: {str(e)}")

    def _get_fallback_scores(self, reason: str) -> Dict:
        """當評分失敗時的回傳值"""
        return {
            "coherence": 1, 
            "consistency": 1, 
            "naturalness": 1, 
            "instruction_following": 1, 
            "overall": 1.0, 
            "reason": reason
        }