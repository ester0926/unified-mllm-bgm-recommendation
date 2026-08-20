"""
用途：建立或評估 Stage 4 使用者 profile 表示。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

import json
import jsonlines
import random
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import ollama
from scipy import stats

# ============================================================================
# 配置區
# ============================================================================

EVAL_CONFIG = {
    # 評審模型 (建議與生成模型相同或更強，Gemma 3 12B 足以勝任此任務)
    "JUDGE_MODEL": "gemma3:12b",
    
    # 評估樣本數 (論文建議 100-200 筆即可證明品質)
    "SAMPLE_SIZE": 100,
    "RANDOM_SEED": 999,
    
    # 參數設定
    "TEMPERATURE": 0.1, # 評估需要嚴謹，溫度調低
    "NUM_CTX": 8192,
    
    # 路徑配置
    "STAGE3_DIALOGUES": Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl"),
    # 注意：這裡假設你已經跑了一部分 Stage 4 的全量生成，或者先用 Pilot 的結果來測試
    # 如果要評估 Pilot 結果，請指向 stage4.1_pilot/results.json (需稍微調整讀取邏輯)
    # 這裡預設指向 Stage 4 的標準輸出目錄 (請根據實際情況修改檔案名稱)
    "STAGE4_PROFILES": Path("data/user_profiling/long_term_preference/stage4_recLLM/profiles_20525_84151.jsonl"), # 預設範例；正式重跑時可改為完整 profiles.jsonl
    "OUTPUT_DIR": Path("data/user_profiling/long_term_preference/stage4_recLLM/stage4.2_eval")
}

# 設定 Logging
EVAL_CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(EVAL_CONFIG["OUTPUT_DIR"] / 'eval.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 工具函數
# ============================================================================

def clean_json_response(text: str) -> str:
    """清理 JSON 響應"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def load_and_pair_data(sample_size: int) -> List[Dict]:
    """讀取並配對 Stage 3 (Source) 與 Stage 4 (Target) 數據"""
    logger.info("正在讀取數據並進行配對...")
    
    # 1. 讀取所有對話 (Source)
    dialogues_map = {} # music_id -> list of dialogue turns
    if EVAL_CONFIG["STAGE3_DIALOGUES"].exists():
        with jsonlines.open(EVAL_CONFIG["STAGE3_DIALOGUES"]) as reader:
            for obj in reader:
                mid = obj.get('music_id')
                dtype = obj.get('dialogue_type')
                turns = obj.get('dialogue_turns', [])
                
                if mid not in dialogues_map:
                    dialogues_map[mid] = []
                
                # 格式化對話文本
                dialogue_text = f"\n=== {dtype} Dialogue ===\n"
                for turn in turns:
                    dialogue_text += f"{turn['role']}: {turn['content']}\n"
                dialogues_map[mid].append(dialogue_text)
    else:
        logger.error(f"找不到 Stage 3 對話檔: {EVAL_CONFIG['STAGE3_DIALOGUES']}")
        return []

    # 2. 讀取生成的 Profiles (Target)
    profiles = []
    if EVAL_CONFIG["STAGE4_PROFILES"].exists():
        with jsonlines.open(EVAL_CONFIG["STAGE4_PROFILES"]) as reader:
            for obj in reader:
                profiles.append(obj)
    else:
        logger.error(f"找不到 Stage 4 Profile 檔: {EVAL_CONFIG['STAGE4_PROFILES']}")
        logger.warning("如果是測試階段，請確保您已生成了一些 Profile 數據")
        return []

    # 3. 配對與抽樣
    paired_data = []
    for prof in profiles:
        mid = prof.get('music_id')
        if mid in dialogues_map:
            paired_data.append({
                'music_id': mid,
                'source_dialogues': "\n".join(dialogues_map[mid]),
                'generated_profile': prof
            })
    
    logger.info(f"配對成功: {len(paired_data)} 筆")
    
    # 隨機抽樣
    random.seed(EVAL_CONFIG["RANDOM_SEED"])
    if len(paired_data) > sample_size:
        sampled = random.sample(paired_data, sample_size)
        logger.info(f"已隨機抽樣: {len(sampled)} 筆")
        return sampled
    
    return paired_data

# ============================================================================
# 核心評估邏輯 (LLM-as-a-Judge)
# ============================================================================

def build_judge_prompt(dialogue_text: str, profile_json: Dict) -> str:
    """構建裁判 Prompt"""
    
    # 簡化 Profile 顯示，只取關鍵部分
    profile_summary = {
        "summary_text": profile_json.get("summary_text", ""),
        "salient_facts": profile_json.get("salient_facts", [])
    }
    
    prompt = f"""
You are an expert data quality evaluator. 
Your task is to verify if the GENERATED USER PROFILE is factually consistent with the SOURCE DIALOGUES.

### 來源對話（ground truth）：
{dialogue_text}

### 產生的使用者 profile（待評估）：
{json.dumps(profile_summary, indent=2, ensure_ascii=False)}

### 評估標準：
1. **Hallucination Check**: Does the profile invent preferences not mentioned or implied in the dialogues? (e.g., Profile says "Likes Jazz" but user never mentioned Jazz).
2. **Consistency**: Does the profile contradict the dialogues? (e.g., User said "I hate rock", Profile says "Loves rock").
3. **Completeness**: Does the profile capture the main musical preferences expresssed by the user?

### 輸出格式（僅 JSON）：
{{
    "accuracy_score": <int, 1-5>,  // 5 = Perfect, 1 = Major Hallucinations/Errors
    "hallucination_detected": <bool>, // true if generated facts are NOT in dialogue
    "missed_key_info": <bool>, // true if major user preferences were ignored
    "reasoning": "<string, explain your score in 1 sentence>"
}}
"""
    return prompt

def evaluate_single_case(case: Dict) -> Dict:
    """評估單個樣本"""
    mid = case['music_id']
    prompt = build_judge_prompt(case['source_dialogues'], case['generated_profile'])
    
    try:
        response = ollama.generate(
            model=EVAL_CONFIG["JUDGE_MODEL"],
            prompt=prompt,
            options={
                "temperature": EVAL_CONFIG["TEMPERATURE"],
                "num_ctx": EVAL_CONFIG["NUM_CTX"]
            }
        )
        
        result_text = clean_json_response(response['response'])
        eval_result = json.loads(result_text)
        
        # 合併原始資訊以便查閱
        return {
            "music_id": mid,
            "eval_result": eval_result,
            "profile_summary": case['generated_profile'].get('summary_text')[:100] + "..."
        }
        
    except Exception as e:
        logger.error(f"Eval failed for {mid}: {e}")
        return {
            "music_id": mid,
            "error": str(e)
        }

# ============================================================================
# 報告生成
# ============================================================================

def generate_report(results: List[Dict]):
    """生成 Markdown 與 JSON 報告"""
    
    valid_results = [r for r in results if 'eval_result' in r]
    total = len(valid_results)
    
    if total == 0:
        logger.warning("沒有有效的評估結果")
        return

    # 統計指標
    avg_score = sum(r['eval_result']['accuracy_score'] for r in valid_results) / total
    hallucination_count = sum(1 for r in valid_results if r['eval_result']['hallucination_detected'])
    omission_count = sum(1 for r in valid_results if r['eval_result']['missed_key_info'])
    
    hallucination_rate = (hallucination_count / total) * 100
    omission_rate = (omission_count / total) * 100

    # Clopper-Pearson 精確二項式信賴區間 (95%)
    # 幻覺率：雙尾 CI
    k_h, n = hallucination_count, total
    hall_ci_lo = stats.beta.ppf(0.025, k_h,     n - k_h + 1) * 100 if k_h > 0 else 0.0
    hall_ci_hi = stats.beta.ppf(0.975, k_h + 1, n - k_h)     * 100
    # 遺漏率：雙尾 95% CI 上界（k=0 時下界固定為 0）
    k_o = omission_count
    omit_ci_hi = stats.beta.ppf(0.975, k_o + 1, n - k_o)     * 100

    # 輸出 Markdown
    report_path = EVAL_CONFIG["OUTPUT_DIR"] / "evaluation_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Stage 4.2: Profile Consistency Evaluation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Judge Model:** {EVAL_CONFIG['JUDGE_MODEL']}\n")
        f.write(f"**Sample Size:** {total}\n\n")
        
        f.write("## 📊 Executive Summary\n\n")
        f.write(f"- **Average Accuracy Score:** {avg_score:.2f} / 5.0\n")
        f.write(f"- **Hallucination Rate:** {hallucination_rate:.1f}%"
                f"  (95% Clopper-Pearson CI: [{hall_ci_lo:.1f}%, {hall_ci_hi:.1f}%],"
                f" n={hallucination_count}/{total}，越低越好)\n")
        f.write(f"- **Omission Rate:** {omission_rate:.1f}%"
                f"  (95% Clopper-Pearson CI upper bound: {omit_ci_hi:.1f}%,"
                f" n={omission_count}/{total}，越低越好)\n\n")
        
        f.write("## 📝 Detailed Analysis\n\n")
        f.write("| Music ID | Score | Hallucination? | Omission? | Reasoning |\n")
        f.write("|---|---|---|---|---|\n")
        
        for res in valid_results:
            eval_data = res['eval_result']
            f.write(f"| {res['music_id']} | "
                    f"**{eval_data['accuracy_score']}** | "
                    f"{'🔴 YES' if eval_data['hallucination_detected'] else '🟢 NO'} | "
                    f"{'🟠 YES' if eval_data['missed_key_info'] else '🟢 NO'} | "
                    f"{eval_data.get('reasoning', '')} |\n")
            
    # 輸出詳細 JSON（含信賴區間）
    summary = {
        "sample_size": total,
        "avg_accuracy_score": round(avg_score, 4),
        "hallucination": {
            "count": hallucination_count,
            "rate_pct": round(hallucination_rate, 4),
            "ci_method": "Clopper-Pearson exact binomial, 95%",
            "ci_lo_pct": round(hall_ci_lo, 4),
            "ci_hi_pct": round(hall_ci_hi, 4),
        },
        "omission": {
            "count": omission_count,
            "rate_pct": round(omission_rate, 4),
            "ci_method": "Clopper-Pearson exact binomial, 95% two-sided upper bound",
            "ci_hi_pct": round(omit_ci_hi, 4),
        },
    }
    json_path = EVAL_CONFIG["OUTPUT_DIR"] / "eval_results_full.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    logger.info("="*60)
    logger.info("評估完成！")
    logger.info(f"平均分數: {avg_score:.2f}")
    logger.info(f"幻覺率: {hallucination_rate:.1f}%  95% CP CI [{hall_ci_lo:.1f}%, {hall_ci_hi:.1f}%]")
    logger.info(f"遺漏率: {omission_rate:.1f}%  95% CP CI 上界 {omit_ci_hi:.1f}%")
    logger.info(f"報告位置: {report_path}")
    logger.info("="*60)

# ============================================================================
# 主程式
# ============================================================================

def main():
    logger.info("啟動 Stage 4.2: 一致性檢查 (LLM-as-a-Judge)")
    
    # 1. 準備數據
    cases = load_and_pair_data(EVAL_CONFIG["SAMPLE_SIZE"])
    if not cases:
        logger.error("沒有數據可供評估，終止程序。")
        return
        
    # 2. 執行評估
    results = []
    logger.info(f"開始評估 {len(cases)} 筆樣本...")
    
    for case in tqdm(cases, desc="Evaluating"):
        res = evaluate_single_case(case)
        results.append(res)
        
    # 3. 生成報告
    generate_report(results)

if __name__ == "__main__":
    main()
