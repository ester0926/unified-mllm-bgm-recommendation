"""
Stage 4.1: Multi-Model Pilot Experiment (Fixed Evaluation Logic)
目的：比較不同 LLM 在用戶畫像萃取任務上的表現
修正重點：
1. 修正評分公式 Bug
2. 加入「空事實檢查」(Empty Facts Check)，確保模型真的有在萃取
3. 調整權重，優先考慮任務完成度與文本品質，而非單純速度

輸出：
- data/user_profiling/long_term_preference/stage4.1_pilot_results.json
- data/user_profiling/long_term_preference/stage4.1_comparison_report.md
"""

import json
import jsonlines
import random
import time
import re
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import ollama

# ============================================================================
# 配置區
# ============================================================================

PILOT_CONFIG = {
    "MODELS": [
        {
            "name": "llama3:8b",
            "display_name": "LLaMA 3 8B",
            "params": "8B",
            "source": "Meta"
        },
        {
            "name": "mistral-nemo:12b",
            "display_name": "Mistral Nemo 12B",
            "params": "12B",
            "source": "Mistral AI"
        },
        {
            "name": "gemma3:12b",
            "display_name": "Gemma 3 12B",
            "params": "12B",
            "source": "Google"
        }
    ],
    "NUM_SAMPLES": 50,
    "RANDOM_SEED": 42,
    "TEMPERATURE": 0.1,
    "NUM_CTX": 8192,
    "MAX_RETRIES": 2,
    "STAGE3_OUTPUT": Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl"),
    "STAGE4_1_OUTPUT_DIR": Path("data/user_profiling/long_term_preference/stage4_recLLM/stage4.1_pilot"),
    "SAMPLE_LIST_FILE": Path("data/user_profiling/long_term_preference/stage4_recLLM/stage4.1_pilot/stage4.1_sample_list.json")
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stage4.1_pilot_fixed.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 步驟 1: 準備固定測試樣本 (保持不變)
# ============================================================================

def prepare_fixed_samples():
    logger.info("="*80)
    logger.info("步驟 1: 準備固定測試樣本")
    logger.info("="*80)
    
    all_dialogues = defaultdict(list)
    if not PILOT_CONFIG["STAGE3_OUTPUT"].exists():
        logger.error(f"找不到輸入檔案: {PILOT_CONFIG['STAGE3_OUTPUT']}")
        return [], {}

    with jsonlines.open(PILOT_CONFIG["STAGE3_OUTPUT"], 'r') as reader:
        for dialogue in reader:
            music_id = dialogue.get('music_id')
            if music_id:
                all_dialogues[music_id].append(dialogue)
    
    complete_songs = []
    for music_id, dialogues in all_dialogues.items():
        types = {d.get('dialogue_type') for d in dialogues}
        if types >= {'Positive', 'Exploratory', 'Negative'}:
            complete_songs.append(music_id)
    
    logger.info(f"找到 {len(complete_songs)} 首有完整三種對話的歌曲")
    
    random.seed(PILOT_CONFIG["RANDOM_SEED"])
    sample_ids = random.sample(complete_songs, min(PILOT_CONFIG["NUM_SAMPLES"], len(complete_songs)))
    
    PILOT_CONFIG["STAGE4_1_OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    with open(PILOT_CONFIG["SAMPLE_LIST_FILE"], 'w', encoding='utf-8') as f:
        json.dump({
            "sample_music_ids": sample_ids,
            "total_samples": len(sample_ids),
            "random_seed": PILOT_CONFIG["RANDOM_SEED"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)
    
    return sample_ids, all_dialogues

# ============================================================================
# 步驟 2: RecLLM 畫像萃取 (Prompt 保持不變)
# ============================================================================

def build_extraction_prompt(dialogues: List[Dict]) -> str:
    by_type = defaultdict(list)
    for d in dialogues:
        by_type[d.get('dialogue_type')].append(d)
    
    dialogue_sections = []
    for dtype in ['Positive', 'Exploratory', 'Negative']:
        if dtype in by_type:
            dialogue = by_type[dtype][0]
            turns = dialogue.get('dialogue_turns', [])
            dialogue_text = f"\n## {dtype} Dialogue:\n"
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
4. Extract 5-8 salient facts
5. Output ONLY valid JSON (no markdown, no backticks)
"""
    return prompt

def clean_json_response(text: str) -> str:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
    text = re.sub(r'\\(?!["\\/bfnrtu])', '', text)
    return text.strip()

def extract_profile_with_model(music_id: str, dialogues: List[Dict], model_name: str) -> Dict:
    prompt = build_extraction_prompt(dialogues)
    
    for attempt in range(PILOT_CONFIG["MAX_RETRIES"]):
        try:
            start_time = time.time()
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': PILOT_CONFIG["TEMPERATURE"],
                    'num_ctx': PILOT_CONFIG["NUM_CTX"]
                }
            )
            elapsed = time.time() - start_time
            raw_text = response['response']
            
            clean_text = clean_json_response(raw_text)
            result = json.loads(clean_text)
            
            if 'summary_text' not in result or 'salient_facts' not in result:
                raise ValueError("Missing required fields")
            
            summary = result['summary_text']
            preferences = result['salient_facts']
            
            return {
                "music_id": music_id,
                "model": model_name,
                "success": True,
                "summary_text": summary,
                "preferences": preferences,
                "word_count": len(summary.split()),
                "num_facts": len(preferences),
                "has_facts": len(preferences) > 0, # 新增：明確標記是否有 facts
                "processing_time": round(elapsed, 2),
                "attempts": attempt + 1
            }
            
        except json.JSONDecodeError:
            if attempt == PILOT_CONFIG["MAX_RETRIES"] - 1:
                # Fallback: Check if we can at least find the summary
                summary_match = re.search(r'"summary_text"\s*:\s*"([^"]+)"', raw_text)
                if summary_match:
                     return {
                        "music_id": music_id,
                        "model": model_name,
                        "success": True, # Still count as format success for summary
                        "summary_text": summary_match.group(1),
                        "preferences": [],
                        "word_count": len(summary_match.group(1).split()),
                        "num_facts": 0,
                        "has_facts": False,
                        "processing_time": round(time.time() - start_time, 2),
                        "attempts": attempt + 1,
                        "fallback_extraction": True
                    }
        except Exception as e:
            logger.error(f"{music_id}: Error - {e}")
    
    return {
        "music_id": music_id,
        "model": model_name,
        "success": False,
        "error": "Max retries exceeded",
        "has_facts": False
    }

# ============================================================================
# 步驟 3: 執行與分析 (修正評分邏輯)
# ============================================================================

def run_pilot_experiment(sample_ids: List[str], all_dialogues: Dict):
    results = {model['name']: [] for model in PILOT_CONFIG["MODELS"]}
    
    for model_info in PILOT_CONFIG["MODELS"]:
        model_name = model_info['name']
        logger.info(f"測試模型: {model_info['display_name']}")
        
        for idx, music_id in enumerate(sample_ids, 1):
            dialogues = all_dialogues[music_id]
            result = extract_profile_with_model(music_id, dialogues, model_name)
            results[model_name].append(result)
            if idx % 10 == 0:
                logger.info(f"  進度: {idx}/{len(sample_ids)}")
                
    return results

def analyze_results(results: Dict[str, List[Dict]]) -> Dict:
    analysis = {}
    
    for model_name, model_results in results.items():
        total = len(model_results)
        successful = [r for r in model_results if r.get('success')]
        
        # 真正有效的成功：格式正確且有提取出 facts
        valid_extraction = [r for r in successful if r.get('has_facts')]
        
        word_counts = [r['word_count'] for r in successful if 'word_count' in r]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        
        times = [r['processing_time'] for r in successful if 'processing_time' in r]
        avg_time = sum(times) / len(times) if times else 0
        
        # Tag 分布
        all_tags = []
        for r in successful:
            for pref in r.get('preferences', []):
                all_tags.append(pref.get('conflict_tag', 'UNKNOWN'))
        tag_dist = Counter(all_tags)
        
        third_person_count = sum(1 for r in successful if r.get('summary_text', '').lower().startswith('the user'))
        
        analysis[model_name] = {
            'total_samples': total,
            'success_rate': (len(successful) / total * 100) if total else 0,
            'valid_extraction_rate': (len(valid_extraction) / total * 100) if total else 0, # 新增指標
            'empty_facts_rate': ((len(successful) - len(valid_extraction)) / total * 100) if total else 0, # 新增指標
            'avg_words': avg_words,
            'avg_time': avg_time,
            'tag_counts': dict(tag_dist),
            'third_person_rate': (third_person_count / len(successful) * 100) if successful else 0
        }
    
    return analysis

def calculate_score(data: Dict, fastest_time: float) -> float:
    """
    修正後的評分公式 (總分 100)
    1. 任務完成度 (50%): 必須有效提取 Facts
    2. 文本品質 (30%): 長度是否在合理範圍 (60-120字)
    3. 格式規範 (10%): 第三人稱
    4. 速度效率 (10%): 相對速度
    """
    
    # 1. 任務完成度 (權重 50)
    # 直接使用有效萃取率 (valid_extraction_rate)
    score_task = data['valid_extraction_rate'] * 0.5
    
    # 2. 文本品質 (權重 30)
    # 目標 90 字，容忍範圍 60-120，之外線性扣分
    avg_words = data['avg_words']
    if avg_words < 40: # 太短，品質極差
        score_quality = 0
    else:
        dist = abs(avg_words - 90)
        # 距離越遠分越低，最大扣完 30 分
        quality_ratio = max(0, 1 - (dist / 60)) 
        score_quality = 30 * quality_ratio
        
    # 3. 格式規範 (權重 10)
    score_format = data['third_person_rate'] * 0.1
    
    # 4. 速度 (權重 10)
    # 使用相對速度，最快為滿分，其餘按比例
    if data['avg_time'] > 0:
        speed_ratio = min(1.0, fastest_time / data['avg_time'])
    else:
        speed_ratio = 0
    score_speed = 10 * speed_ratio
    
    total_score = score_task + score_quality + score_format + score_speed
    return round(total_score, 1)

def generate_comparison_report(analysis: Dict, results: Dict):
    model_display = {m['name']: m['display_name'] for m in PILOT_CONFIG["MODELS"]}
    
    # 找出最快時間作為基準
    valid_times = [d['avg_time'] for d in analysis.values() if d['avg_time'] > 0]
    fastest_time = min(valid_times) if valid_times else 1.0
    
    print("\n" + "="*80)
    print("Stage 4.1 先導實驗結果對比 (評分修正版)")
    print("="*80)
    
    # 計算分數
    scores = {}
    for model_name, data in analysis.items():
        scores[model_name] = calculate_score(data, fastest_time)
        
    # 表格輸出
    headers = ["Model", "Valid Extract%", "Empty Facts%", "Avg Words", "Time(s)", "Score"]
    row_fmt = "{:<20} {:<15} {:<15} {:<12} {:<10} {:<10}"
    
    print(row_fmt.format(*headers))
    print("-" * 85)
    
    for model_name, data in analysis.items():
        print(row_fmt.format(
            model_display[model_name],
            f"{data['valid_extraction_rate']:.1f}%",
            f"{data['empty_facts_rate']:.1f}%",  # 關鍵指標
            f"{data['avg_words']:.1f}",
            f"{data['avg_time']:.1f}",
            f"{scores[model_name]}"
        ))

    # 推薦邏輯
    best_model_name = max(scores.items(), key=lambda x: x[1])[0]
    
    print("\n" + "="*80)
    print("模型推薦")
    print("="*80)
    print(f"🏆 推薦模型: {model_display[best_model_name]}")
    print(f"   綜合得分: {scores[best_model_name]}/100")
    
    # 寫入 Markdown 報告
    report_path = PILOT_CONFIG["STAGE4_1_OUTPUT_DIR"] / "comparison_report_fixed.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Stage 4.1 對比報告 (修正版)\n\n")
        f.write("## 評分標準\n")
        f.write("- **有效萃取率 (50%)**: 是否成功提取出偏好事實 (Empty Facts = 0 分)\n")
        f.write("- **文本品質 (30%)**: 摘要長度是否符合 80-100 字要求\n")
        f.write("- **格式與速度 (20%)**: 第三人稱格式與生成效率\n\n")
        
        f.write("## 結果數據\n")
        f.write("| Model | Valid Extraction | Empty Facts | Avg Words | Time | Score |\n")
        f.write("|---|---|---|---|---|---|\n")
        for model_name, data in analysis.items():
            f.write(f"| {model_display[model_name]} | "
                    f"{data['valid_extraction_rate']:.1f}% | "
                    f"**{data['empty_facts_rate']:.1f}%** | "
                    f"{data['avg_words']:.1f} | "
                    f"{data['avg_time']:.1f}s | "
                    f"**{scores[model_name]}** |\n")
        
        f.write(f"\n## 結論\n")
        f.write(f"推薦使用 **{model_display[best_model_name]}**，因為它在有效萃取率和文本品質上取得了最佳平衡。")

    return best_model_name

# ============================================================================
# 主程式
# ============================================================================

def main():
    try:
        sample_ids, all_dialogues = prepare_fixed_samples()
        results = run_pilot_experiment(sample_ids, all_dialogues)
        analysis = analyze_results(results)
        best_model = generate_comparison_report(analysis, results)
        
        # 儲存詳細結果
        with open(PILOT_CONFIG["STAGE4_1_OUTPUT_DIR"] / "results_fixed.json", 'w', encoding='utf-8') as f:
            json.dump({'analysis': analysis, 'scores': best_model}, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()