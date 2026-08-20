"""
用途：建立 Stage 3 產生合成對話時使用的 prompt。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

from typing import Dict, List, Optional

def build_prompt(
    target_music: Dict,
    ltp_text: str,
    search_history: List[Dict],
    dialogue_type: str,
    turn_number: int = 10,
    retry_attempt: int = 0,
    last_failure_reason: str = None,
    user_dislikes: str = None, 
    use_cot: bool = True,
    use_reflection: bool = True
) -> str:
    builder = PromptBuilderV3CoT()
    return builder.build_full_dialogue_prompt(
        target_music, ltp_text, search_history, dialogue_type, 
        turn_number, retry_attempt, last_failure_reason, 
        user_dislikes, use_cot, use_reflection
    )

class PromptBuilderV3CoT:
    def __init__(self):
        self.system_prompts = self._load_system_prompts()
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_system_prompts(self) -> Dict[str, str]:
        base_instruction = "IMPORTANT: You are roleplaying. Output mostly dialogue."
        return {
            "Positive": f"""{base_instruction}
You are simulating a user with POSITIVE preferences.
Role: User is looking for background music and is enthusiastic about the target track.
Key trait: Uses positive words ('love', 'perfect') and clearly explains why they like the music attributes.""",

            "Exploratory": f"""{base_instruction}
You are simulating a user with EXPLORATORY preferences.
Role: User is curious but hesitant. They compare the track to their usual taste.
Key trait: Uses conditional language ('maybe', 'interesting', 'depends').""",

            "Negative": f"""{base_instruction}
You are simulating a user with NEGATIVE preferences.
Role: User REJECTS the target track.
Key trait: Politely but firmly explains why the track fails (e.g., 'too loud', 'wrong vibe')."""
        }
    
    def _load_few_shot_examples(self) -> Dict[str, List[str]]:
        # 簡化範例，確保 Turn 10 格式正確
        return {
            "Positive": [
"""[PLAN]
...
[DIALOGUE]
Turn 1:
User: This is great!
Recommender: Glad you like it. Why?
...
Turn 10:
User: I definitely want high-energy electronic music with a fast tempo.
Recommender: Understood, focusing on high-energy tracks."""
            ]
        }

    def build_full_dialogue_prompt(
        self,
        target_music: Dict,
        ltp_text: str,
        search_history: List[Dict],
        dialogue_type: str,
        num_turns: int,
        retry_attempt: int,
        last_failure_reason: str = None,
        user_dislikes: str = None,
        use_cot: bool = True,
        use_reflection: bool = True
    ) -> str:
        
        system_prompt = self.system_prompts.get(dialogue_type, self.system_prompts["Positive"])
        
        # 建立 prompt 上下文
        history_text = "None"
        if search_history:
            history_text = "\n".join([f"- {i.get('title')} ({i.get('genre')})" for i in search_history[:3]])

        dislikes_block = ""
        if user_dislikes and "no specific" not in user_dislikes.lower():
            dislikes_block = f"\n**User's Dislikes (Avoid these):**\n{user_dislikes}\n"

        context = f"""## Context
**Target Song:** {target_music.get('title')} ({target_music.get('genre')})
**User Likes:** {ltp_text}
{dislikes_block}
**History:**
{history_text}
"""

        retry_instr = ""
        if retry_attempt > 0 and last_failure_reason:
            retry_instr = f"\n⚠️ **Fix Previous Error:** {last_failure_reason}\n"

        task = f"""
════════════════════════════════════
TASK: Generate a {num_turns}-Turn Dialogue ({dialogue_type})
════════════════════════════════════

{retry_instr}

**Rules:**
1. **Turn 10 Requirement:** The User must naturally summarize their final preference in one sentence.
   - ❌ BAD: User: "I am looking for rock." (Do not use quotes)
   - ✅ GOOD: User: I am looking for a rock track with heavy drums.
2. **Format:**
   [PLAN]
   ...
   [DIALOGUE]
   Turn 1:
   User: ...
   Recommender: ...
"""
        
        return f"{system_prompt}\n\n{context}\n\n{task}"