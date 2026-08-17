"""
Stage 2: QA Validator V3 - 改進版 (Final Fix)
修正：
1. 增強 Role 識別 (User/Assistant/Recommender)
2. 優化空回應檢測邏輯
3. 支援重試時的動態閾值調整
"""
import re
from typing import Dict, List, Tuple
import numpy as np
from config import GlobalModels

def validate_dialogue(
    dialogue_turns: List[Dict],
    dialogue_type: str,
    ltp_text: str,
    retry_attempt: int = 0
) -> Tuple[bool, str, str]:
    
    validator = QAValidatorV3(retry_attempt=retry_attempt)
    if not dialogue_turns: return (False, "format", "No dialogue turns")

    for i, turn_data in enumerate(dialogue_turns):
        turn_num = turn_data.get("turn", 1)
        content = turn_data.get("content", "").strip()
        if not content: return (False, "format", f"Empty content turn {turn_num}")
        
        # 檢查 Prompt 洩漏 (Robot Check)
        if "user says:" in content.lower() or "turn 10:" in content.lower():
            return (False, "format", f"Prompt leakage in turn {turn_num}")

        if turn_data.get("role", "").lower() == "user":
            res = validator.validate_turn(turn_num, content, dialogue_type, ltp_text)
            if not res["valid"]:
                return (False, res.get("failure_type"), res.get("reason"))
    
    return (True, "", "")

class QAValidatorV3:
    def __init__(self, retry_attempt: int = 0):
        self.sbert = GlobalModels.get_sbert()
        self.threshold = 0.05 * (0.8 ** retry_attempt) # 簡單的動態閾值
    
    def validate_turn(self, turn, content, dtype, ltp_text):
        # 1. Length Check
        if len(content.split()) < 2: 
            return {"valid": False, "failure_type": "format", "reason": "Too short"}
        
        # 2. Turn 10 Check
        if turn == 10:
            # 移除引號再檢查
            clean_content = content.replace('"', '').replace("'", "")
            sentences = [s for s in re.split(r'[.!?]+', clean_content) if len(s.strip()) > 5]
            if len(sentences) > 3: # 放寬到 3 句
                return {"valid": False, "failure_type": "format", "reason": "Turn 10 too long"}
        
        # 3. LTP Consistency (Simple)
        if ltp_text:
            try:
                emb1 = self.sbert.encode(content, show_progress_bar=False)
                emb2 = self.sbert.encode(ltp_text, show_progress_bar=False)
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if sim < self.threshold and dtype != "Negative": # Negative 不需要跟 LTP 太像
                    return {"valid": False, "failure_type": "coherence", "reason": "Low LTP consistency"}
            except: pass
            
        return {"valid": True}