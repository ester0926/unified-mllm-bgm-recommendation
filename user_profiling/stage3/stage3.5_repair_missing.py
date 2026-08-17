"""
Stage 3 失敗案例補齊程式（修復版）
功能：針對 missing_tasks.jsonl 中的失敗案例重新生成對話
策略：
1. 更高溫度（0.85）提升創意
2. 更多重試次數（5次）
3. 放寬驗證條件（允許更短對話）
4. 支援斷點續傳

修復：
- 修正導入錯誤（使用正確的模組和類）
- 加入強健的錯誤處理
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

# ==================== 配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stage3_repair.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 路徑配置
MISSING_TASKS_FILE = Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/stage3_diagnostics/missing_tasks.jsonl")
HISTORY_DIR = Path("data/user_profiling/long_term_preference/stage2_history/personax")
OUTPUT_FILE = Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl")
METADATA_FILE = Path("data/user_profiling/music_metadata_simple/music_metadata_enriched.json")

# ==================== 正確的導入 ====================
# 從原始模組導入
from llm_client import OllamaClient
from prompt_builder import build_prompt
from qa_validator import validate_dialogue

# 導入 TwoPhaseGeneratorOptimized 類
from stage3_dialogue_optimized import TwoPhaseGeneratorOptimized

# ==================== 強健的 JSONL 讀取 ====================
def read_jsonl_robust(file_path):
    """強健地讀取 JSONL，跳過損壞的行"""
    objects = []
    errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                objects.append(obj)
            except json.JSONDecodeError as e:
                errors.append({'line': line_num, 'error': str(e)})
    
    if errors:
        logger.warning(f"⚠️  {len(errors)} 行解析失敗")
    
    return objects

# ==================== 載入 Metadata ====================
def load_metadata(metadata_file):
    """載入音樂元數據"""
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# ==================== 初始化 ====================
# 創建客戶端（用於補齊，溫度較高）
client = OllamaClient(
    model_name="gemma3:4b",
    temperature=0.85,  # ✨ 提高溫度增加創意
    top_p=0.95
)

# 載入 metadata
metadata = load_metadata(METADATA_FILE)

# 創建生成器實例（用於 Persona 生成）
# 關閉品質評估以加速補齊
generator = TwoPhaseGeneratorOptimized(
    enable_quality_eval=False,
    use_cache=False,  # 補齊時不使用快取，確保每次都重新生成
    num_workers=1
)

# ==================== 補齊函數 ====================
def repair_single_dialogue(music_id: str, dialogue_type: str, max_retries: int = 5) -> Tuple[bool, dict]:
    """
    補齊單個對話
    
    Args:
        music_id: 音樂ID
        dialogue_type: 對話類型
        max_retries: 最大重試次數（比主程式多）
    
    Returns:
        (成功與否, 對話數據)
    """
    # 讀取歷史數據
    history_file = HISTORY_DIR / f"{music_id}__history.json"
    if not history_file.exists():
        logger.error(f"History not found: {music_id}")
        return False, {}
    
    with open(history_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    balanced_hist = data.get('balanced_history', {})
    core_sbs = balanced_hist.get('core_sbs', [])
    explor_sbs = balanced_hist.get('exploratory_sbs', [])
    negative_sbs = balanced_hist.get('negative_sbs', [])
    
    if not core_sbs:
        logger.error(f"Empty core_sbs: {music_id}")
        return False, {}
    
    # ✅ 使用生成器的方法生成 Persona
    pos_snippet, neg_snippet = generator.generate_persona_snippets(core_sbs, negative_sbs)
    
    # 準備目標音樂
    target_meta = metadata.get(music_id, {"title": "Unknown", "genre": "Unknown", "tags": []})
    hist_context = core_sbs[:3]
    
    if dialogue_type == "Exploratory" and explor_sbs:
        hist_context = explor_sbs[:3]
    elif dialogue_type == "Negative" and negative_sbs:
        neg_item = negative_sbs[0]
        target_meta = {
            "title": neg_item.get('title', 'Unknown'),
            "artist": neg_item.get('artist', 'Unknown'),
            "genre": neg_item.get('genre', 'Unknown'),
            "tags": neg_item.get('tags', [])
        }
    
    # 多次重試生成
    last_failure_reason = None
    for attempt in range(max_retries):
        try:
            # ✅ 使用 prompt_builder 的函數構建 Prompt
            prompt = build_prompt(
                target_music=target_meta,
                ltp_text=pos_snippet,
                search_history=hist_context,
                dialogue_type=dialogue_type,
                turn_number=10,
                retry_attempt=attempt,
                last_failure_reason=last_failure_reason,
                user_dislikes=neg_snippet,
                use_cot=True
            )
            
            # 生成
            temp = 0.85 if attempt == 0 else 0.7  # 首次高溫，重試降溫
            raw_res = client.generate(prompt, temperature=temp)
            
            # ✅ 使用生成器的解析方法
            plan, turns = generator._parse_response_flexible(raw_res)
            
            if not turns:
                last_failure_reason = "Format Error: No dialogue turns parsed."
                continue
            
            # ✅ 使用 qa_validator 的函數驗證
            full_persona = f"{pos_snippet} {neg_snippet}"
            is_valid, f_type, f_reason = validate_dialogue(
                turns, 
                dialogue_type, 
                full_persona, 
                retry_attempt=attempt
            )
            
            if is_valid:
                result = {
                    "music_id": music_id,
                    "dialogue_target_id": target_meta.get('title'),
                    "dialogue_type": dialogue_type,
                    "persona_snippet": pos_snippet,
                    "negative_snippet": neg_snippet,
                    "dialogue_raw": raw_res,
                    "dialogue_turns": turns,
                    "quality_scores": {},  # 補齊時跳過評分
                    "retry_count": attempt,
                    "repair_flag": True  # 標記為補齊數據
                }
                return True, result
            
            last_failure_reason = f"{f_type}: {f_reason}"
            
        except Exception as e:
            logger.error(f"Error in attempt {attempt}: {e}")
            last_failure_reason = f"System Error: {e}"
    
    return False, {}

# ==================== 主流程 ====================
def main():
    logger.info("="*80)
    logger.info("Stage 3.6: 失敗案例補齊程式（修復版）")
    logger.info("="*80)
    
    # 讀取待補齊任務
    if not MISSING_TASKS_FILE.exists():
        logger.error(f"❌ Missing tasks file not found: {MISSING_TASKS_FILE}")
        logger.info("   請先執行 stage3_5_diagnose_failures_fixed.py 生成任務清單")
        return
    
    logger.info(f"讀取待補齊任務：{MISSING_TASKS_FILE}")
    tasks = read_jsonl_robust(MISSING_TASKS_FILE)
    logger.info(f"📋 待補齊任務：{len(tasks):,} 個")
    
    if not tasks:
        logger.error("❌ 無待補齊任務")
        return
    
    # 檢查已完成（斷點續傳）
    logger.info(f"\n檢查已完成的對話...")
    existing_pairs = set()
    if OUTPUT_FILE.exists():
        existing_dialogues = read_jsonl_robust(OUTPUT_FILE)
        for obj in existing_dialogues:
            mid = obj.get('music_id')
            dtype = obj.get('dialogue_type')
            if mid and dtype:
                existing_pairs.add(f"{mid}_{dtype}")
        logger.info(f"   已有對話數：{len(existing_dialogues):,}")
    
    # 過濾待補齊
    pending_tasks = [
        t for t in tasks 
        if f"{t['music_id']}_{t['dialogue_type']}" not in existing_pairs
    ]
    
    logger.info(f"\n" + "="*80)
    logger.info(f"任務統計")
    logger.info(f"="*80)
    logger.info(f"總任務數：{len(tasks):,}")
    logger.info(f"已完成：{len(tasks) - len(pending_tasks):,}")
    logger.info(f"待補齊：{len(pending_tasks):,}")
    
    if not pending_tasks:
        logger.info("\n🎉 所有任務已完成！")
        return
    
    # 執行補齊
    logger.info(f"\n" + "="*80)
    logger.info(f"開始補齊（共 {len(pending_tasks):,} 個任務）")
    logger.info(f"="*80)
    logger.info(f"預估時間：{len(pending_tasks) * 40 / 3600:.1f} 小時\n")
    
    success_count = 0
    fail_count = 0
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for task in tqdm(pending_tasks, desc="補齊進度"):
            music_id = task['music_id']
            dialogue_type = task['dialogue_type']
            
            success, result = repair_single_dialogue(music_id, dialogue_type)
            
            if success:
                # 寫入 JSONL
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # 立即寫入，避免意外丟失
                success_count += 1
                if success_count % 10 == 0:
                    logger.info(f"✅ 已補齊 {success_count}/{len(pending_tasks)}")
            else:
                fail_count += 1
                logger.warning(f"❌ 補齊失敗：{dialogue_type} for {music_id}")
    
    # 報告結果
    logger.info(f"\n" + "="*80)
    logger.info(f"補齊完成")
    logger.info(f"="*80)
    logger.info(f"成功：{success_count:,}")
    logger.info(f"失敗：{fail_count:,}")
    
    if success_count + fail_count > 0:
        success_rate = success_count / (success_count + fail_count) * 100
        logger.info(f"成功率：{success_rate:.1f}%")
    
    logger.info(f"="*80)
    
    if fail_count > 0:
        logger.warning(f"\n⚠️  仍有 {fail_count} 個任務失敗")
        logger.warning("建議：")
        logger.warning("   1. 檢查失敗案例的 history 數據是否完整")
        logger.warning("   2. 考慮進一步放寬驗證條件")
        logger.warning("   3. 或接受少量缺失（<1%可忽略）")
        logger.warning("\n   可重新執行 stage3_5 診斷剩餘問題")
    else:
        logger.info("\n🎉 所有任務補齊成功！")
        logger.info("   建議：重新執行 stage3_5 確認完整性")

if __name__ == "__main__":
    main()