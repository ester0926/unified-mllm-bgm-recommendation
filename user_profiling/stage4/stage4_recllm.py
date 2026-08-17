"""
Stage 4: Full Scale User Profile Generation (RecLLM)
目的：基於 Stage 3 的對話數據，大規模生成用戶長期偏好畫像 (User Profiles)。

核心邏輯：
1. 讀取 Stage 3 的對話數據 (dialogues.jsonl)。
2. 按 music_id 聚合三種對話類型 (Positive, Exploratory, Negative)。
3. 使用 Gemma 3 12B 進行偏好萃取 (Extraction)。
4. 執行嚴格的品質檢查 (拒絕空內容)。
5. 寫入 profiles.jsonl。

輸入：data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl
輸出：data/user_profiling/long_term_preference/stage4_recLLM/profiles.jsonl
"""

import json
import jsonlines
import time
import re
import logging
import threading
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ollama

# ============================================================================
# 配置區 (基於 Stage 4.1 Pilot 結論)
# ============================================================================

STAGE4_CONFIG = {
    # 核心模型 (Pilot 勝出者)
    "MODEL_NAME": "gemma3:12b",
    
    # 參數設定
    "TEMPERATURE": 0.1,    # 低溫保證事實提取的穩定性
    "NUM_CTX": 8192,       # 足夠容納三段對話歷史
    "MAX_RETRIES": 3,      # 遇到 JSON 錯誤或空內容時的重試次數
    
    # 執行緒設定 (本地推理建議 1-2，視顯存而定；若 Ollama 有 Queue 機制可設高一點)
    "MAX_WORKERS": 2,      
    
    # 路徑配置
    "INPUT_FILE": Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl"),
    "OUTPUT_DIR": Path("data/user_profiling/long_term_preference/stage4_recLLM"),
    "OUTPUT_FILE": Path("data/user_profiling/long_term_preference/stage4_recLLM/profiles.jsonl"),
    "LOG_FILE": Path("logs/stage4_recLLM.log")
}

# 設定 Logging
STAGE4_CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(STAGE4_CONFIG["LOG_FILE"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 工具函數
# ============================================================================

def clean_json_response(text: str) -> str:
    """清理 JSON 響應 (三層防禦)"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
    text = re.sub(r'\\(?!["\\/bfnrtu])', '', text)
    return text.strip()

def build_extraction_prompt(dialogues: List[Dict]) -> str:
    """構建 RecLLM 風格的萃取 Prompt"""
    by_type = defaultdict(list)
    for d in dialogues:
        by_type[d.get('dialogue_type')].append(d)
    
    dialogue_sections = []
    # 確保順序：正向 -> 探索 -> 負向 (符合人類認知邏輯)
    for dtype in ['Positive', 'Exploratory', 'Negative']:
        if dtype in by_type:
            dialogue = by_type[dtype][0]
            turns = dialogue.get('dialogue_turns', [])
            
            dialogue_text = f"\n## {dtype} Dialogue:\n"
            # 取前 6 輪，避免 Context 過長且通常偏好在前幾輪就出現
            for turn in turns[:6]:  
                role = turn.get('role', 'User')
                content = turn.get('content', '')
                dialogue_text += f"{role}: {content}\n"
            
            dialogue_sections.append(dialogue_text)
    
    prompt = f"""You are a user preference analyzer. Extract the user's long-term music preferences from these dialogue histories.

{"".join(dialogue_sections)}

TASK: Generate a natural language user profile with salient preference facts.

OUTPUT FORMAT (JSON only, no markdown):
{{
    "summary_text": "The user [comprehensive summary in 80-100 words using third-person]",
    "salient_facts": [
        {{"fact": "The user prefers calm piano music", "conflict_tag": "CONFIRM"}},
        {{"fact": "The user dislikes heavy metal", "conflict_tag": "CONFIRM_DISLIKE"}},
        {{"fact": "The user is exploring jazz styles", "conflict_tag": "NEW"}}
    ]
}}

CONFLICT TAGS:
- CONFIRM: Reinforces existing preference
- CONFIRM_DISLIKE: Confirms dislike/avoidance
- MODULATE: Refines/adjusts preference within taste boundaries
- NEW: Explores new style/element
- OVERRIDE: Contradicts long-term preference (rare in this scenario)

CRITICAL RULES:
1. Use THIRD-PERSON ("The user prefers..." NOT "I prefer...")
2. Summary must be 80-100 words
3. Focus on MUSIC preferences (genres, moods, instruments)
4. Extract 5-8 salient facts. The 'salient_facts' list MUST NOT be empty.
5. Output ONLY valid JSON (no markdown, no backticks)
"""
    return prompt

# ============================================================================
# 資料準備
# ============================================================================

def load_and_group_dialogues() -> Dict[str, List[Dict]]:
    """讀取並聚合對話數據"""
    logger.info("步驟 1: 讀取並聚合對話數據...")
    
    all_dialogues = defaultdict(list)
    count = 0
    
    if not STAGE4_CONFIG["INPUT_FILE"].exists():
        raise FileNotFoundError(f"找不到輸入檔案: {STAGE4_CONFIG['INPUT_FILE']}")

    with jsonlines.open(STAGE4_CONFIG["INPUT_FILE"], 'r') as reader:
        for dialogue in reader:
            music_id = dialogue.get('music_id')
            if music_id:
                all_dialogues[music_id].append(dialogue)
                count += 1
    
    # 過濾掉不完整的歌曲 (必須要有三種對話)
    complete_dialogues = {}
    for mid, dlgs in all_dialogues.items():
        types = {d.get('dialogue_type') for d in dlgs}
        if types >= {'Positive', 'Exploratory', 'Negative'}:
            complete_dialogues[mid] = dlgs
            
    logger.info(f"讀取完成: {count} 筆對話")
    logger.info(f"聚合完成: {len(complete_dialogues)} 首完整歌曲 (準備處理)")
    
    return complete_dialogues

def get_processed_ids() -> Set[str]:
    """讀取已處理的 ID 以支援斷點續傳"""
    processed = set()
    if STAGE4_CONFIG["OUTPUT_FILE"].exists():
        with jsonlines.open(STAGE4_CONFIG["OUTPUT_FILE"], 'r') as reader:
            for obj in reader:
                if 'music_id' in obj:
                    processed.add(obj['music_id'])
    logger.info(f"斷點續傳: 已發現 {len(processed)} 筆處理完成的資料")
    return processed

# ============================================================================
# 核心處理邏輯
# ============================================================================

def process_single_song(music_id: str, dialogues: List[Dict]) -> Optional[Dict]:
    """處理單一歌曲：生成 Profile"""
    
    prompt = build_extraction_prompt(dialogues)
    
    for attempt in range(STAGE4_CONFIG["MAX_RETRIES"]):
        try:
            start_time = time.time()
            
            response = ollama.generate(
                model=STAGE4_CONFIG["MODEL_NAME"],
                prompt=prompt,
                options={
                    'temperature': STAGE4_CONFIG["TEMPERATURE"],
                    'num_ctx': STAGE4_CONFIG["NUM_CTX"]
                }
            )
            
            raw_text = response['response']
            clean_text = clean_json_response(raw_text)
            result = json.loads(clean_text)
            
            # === 關鍵檢查邏輯 (來自 Stage 4.1 Pilot) ===
            # 1. 檢查欄位是否存在
            if 'summary_text' not in result or 'salient_facts' not in result:
                raise ValueError("Missing required JSON fields")
            
            # 2. 檢查是否為空內容 (LLaMA 3 8B 失敗的主因)
            if not result['salient_facts'] or len(result['salient_facts']) == 0:
                raise ValueError("Empty salient_facts detected (Model extraction failed)")
            
            # 3. 檢查長度 (選擇性警告，不強制報錯)
            word_count = len(result['summary_text'].split())
            if word_count < 40:
                logger.warning(f"{music_id}: Summary too short ({word_count} words)")
            
            return {
                "music_id": music_id,
                "summary_text": result['summary_text'],
                "salient_facts": result['salient_facts'],
                "meta": {
                    "model": STAGE4_CONFIG["MODEL_NAME"],
                    "processing_time": round(time.time() - start_time, 2),
                    "word_count": word_count,
                    "num_facts": len(result['salient_facts'])
                }
            }
            
        except json.JSONDecodeError:
            # 最後一次嘗試失敗才記錄錯誤
            if attempt == STAGE4_CONFIG["MAX_RETRIES"] - 1:
                logger.error(f"{music_id}: JSON Decode Error after {attempt+1} attempts.")
        except ValueError as ve:
            # 捕獲空內容錯誤
            if attempt == STAGE4_CONFIG["MAX_RETRIES"] - 1:
                logger.error(f"{music_id}: Validation Error - {ve}")
        except Exception as e:
            logger.error(f"{music_id}: Unknown Error - {e}")
            time.sleep(1) # 發生未知錯誤稍微冷卻
    
    return None

# ============================================================================
# 主流程
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("Stage 4: Full Scale User Profile Generation Start")
    logger.info(f"Model: {STAGE4_CONFIG['MODEL_NAME']}")
    logger.info("="*80)
    
    # 1. 準備數據
    dialogues_map = load_and_group_dialogues()
    processed_ids = get_processed_ids()
    
    # 過濾待處理任務
    tasks = [
        (mid, dlgs) 
        for mid, dlgs in dialogues_map.items() 
        if mid not in processed_ids
    ]
    
    logger.info(f"剩餘待處理任務: {len(tasks)}")
    if not tasks:
        logger.info("所有任務已完成！")
        return

    # 2. 執行生成 (使用 ThreadPoolExecutor 但須注意 Ollama 並發能力)
    # 若 Ollama 在本地跑，建議 workers=1 或 2，避免顯存頻繁切換導致變慢
    
    file_lock = threading.Lock()
    success_count = 0
    fail_count = 0
    
    with jsonlines.open(STAGE4_CONFIG["OUTPUT_FILE"], mode='a', flush=True) as writer:
        with ThreadPoolExecutor(max_workers=STAGE4_CONFIG["MAX_WORKERS"]) as executor:
            
            # 提交任務
            future_to_mid = {
                executor.submit(process_single_song, mid, dlgs): mid 
                for mid, dlgs in tasks
            }
            
            # 進度條
            pbar = tqdm(total=len(tasks), desc="Generating Profiles")
            
            for future in as_completed(future_to_mid):
                mid = future_to_mid[future]
                try:
                    result = future.result()
                    if result:
                        with file_lock:
                            writer.write(result)
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"Critical error processing {mid}: {e}")
                    fail_count += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"Success": success_count, "Fail": fail_count})
    
    logger.info("="*80)
    logger.info("Stage 4 完成")
    logger.info(f"成功: {success_count}")
    logger.info(f"失敗: {fail_count}")
    logger.info(f"輸出檔案: {STAGE4_CONFIG['OUTPUT_FILE']}")
    logger.info("="*80)

if __name__ == "__main__":
    main()