"""
定期抽樣評估 - 在生成過程中監控品質

功能：
1. 每完成 N 筆對話，自動抽樣評估
2. 使用相同的 LLM-as-a-Judge (gemma3:12b)
3. 監控品質趨勢，及早發現問題

用法：
- 在另一個終端執行
- 會持續監控 dialogues.jsonl
- 每 10,000 筆抽樣 100 筆評估
"""

import time
import jsonlines
import random
import json
from pathlib import Path
from datetime import datetime
from stage3.quality_evaluator import QualityEvaluatorLLM
from stage3.llm_client import OllamaClient

print("=" * 70)
print("定期抽樣品質評估")
print("=" * 70)
print()

# 配置
DIALOGUE_FILE = Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl")
SAMPLE_INTERVAL = 10000  # 每 10,000 筆抽樣一次
SAMPLE_SIZE = 100        # 每次抽樣 100 筆
OUTPUT_FILE = Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/periodic_quality_report.jsonl")

# 初始化評估器
print("初始化 LLM-as-a-Judge (gemma3:12b)...")
evaluator = QualityEvaluatorLLM()
evaluator.client = OllamaClient(
    model_name="gemma3:12b",
    temperature=0.0  # 確保一致性
)
print("✅ 評估器就緒\n")

# 確保輸出目錄存在
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 記錄已評估的檢查點
evaluated_checkpoints = set()
if OUTPUT_FILE.exists():
    with jsonlines.open(OUTPUT_FILE) as reader:
        for obj in reader:
            evaluated_checkpoints.add(obj['checkpoint'])

last_count = 0

try:
    while True:
        # 檢查檔案是否存在
        if not DIALOGUE_FILE.exists():
            time.sleep(30)
            continue
        
        # 統計當前對話數
        with jsonlines.open(DIALOGUE_FILE) as reader:
            dialogues = list(reader)
        
        current_count = len(dialogues)
        
        # 檢查是否達到新的檢查點
        checkpoint = (current_count // SAMPLE_INTERVAL) * SAMPLE_INTERVAL
        
        if checkpoint > 0 and checkpoint not in evaluated_checkpoints and current_count >= checkpoint + 100:
            print(f"\n{'='*70}")
            print(f"檢查點 {checkpoint:,} 已達成！開始抽樣評估...")
            print(f"{'='*70}\n")
            
            # 抽樣
            # 確保抽樣範圍在有效區間內
            sample_start = max(0, checkpoint - SAMPLE_INTERVAL)
            sample_end = min(checkpoint, len(dialogues))
            sample_pool = dialogues[sample_start:sample_end]
            
            if len(sample_pool) < SAMPLE_SIZE:
                sample = sample_pool
            else:
                sample = random.sample(sample_pool, SAMPLE_SIZE)
            
            print(f"抽樣範圍：{sample_start:,} - {sample_end:,}")
            print(f"抽樣數量：{len(sample)} 筆\n")
            
            # 評估
            scores_list = []
            print("開始評估...")
            
            for i, dialogue in enumerate(sample, 1):
                try:
                    scores = evaluator.evaluate(
                        dialogue['dialogue_turns'],
                        dialogue['persona_snippet'],
                        dialogue['dialogue_type']
                    )
                    scores_list.append(scores)
                    
                    if i % 10 == 0:
                        print(f"  進度：{i}/{len(sample)} ({i/len(sample)*100:.0f}%)")
                    
                except Exception as e:
                    print(f"  ⚠️ 評估失敗：{e}")
                    continue
            
            # 統計結果
            if scores_list:
                avg_overall = sum(s.get('overall', 0) for s in scores_list) / len(scores_list)
                avg_coherence = sum(s.get('coherence', 0) for s in scores_list) / len(scores_list)
                avg_consistency = sum(s.get('consistency', 0) for s in scores_list) / len(scores_list)
                avg_naturalness = sum(s.get('naturalness', 0) for s in scores_list) / len(scores_list)
                avg_instruction = sum(s.get('instruction_following', 0) for s in scores_list) / len(scores_list)
                
                high_quality = sum(1 for s in scores_list if s.get('overall', 0) >= 4.0)
                low_quality = sum(1 for s in scores_list if s.get('overall', 0) < 3.5)
                
                # 顯示結果
                print(f"\n{'='*70}")
                print(f"評估結果 (檢查點 {checkpoint:,})")
                print(f"{'='*70}")
                print(f"樣本數：{len(scores_list)}")
                print(f"\n平均分數：")
                print(f"  Overall:              {avg_overall:.2f}")
                print(f"  Coherence:            {avg_coherence:.2f}")
                print(f"  Consistency:          {avg_consistency:.2f}")
                print(f"  Naturalness:          {avg_naturalness:.2f}")
                print(f"  Instruction Following: {avg_instruction:.2f}")
                print(f"\n品質分佈：")
                print(f"  優秀 (>= 4.0): {high_quality} ({high_quality/len(scores_list)*100:.1f}%)")
                print(f"  不佳 (< 3.5):  {low_quality} ({low_quality/len(scores_list)*100:.1f}%)")
                
                # 警告檢查
                if avg_overall < 4.0:
                    print(f"\n⚠️  警告：平均分數低於 4.0")
                    print(f"   建議：檢查生成配置")
                elif avg_overall >= 4.2:
                    print(f"\n✅ 品質優良")
                else:
                    print(f"\n✅ 品質正常")
                
                print(f"{'='*70}\n")
                
                # 儲存結果
                report = {
                    "checkpoint": checkpoint,
                    "timestamp": datetime.now().isoformat(),
                    "sample_size": len(scores_list),
                    "avg_scores": {
                        "overall": avg_overall,
                        "coherence": avg_coherence,
                        "consistency": avg_consistency,
                        "naturalness": avg_naturalness,
                        "instruction_following": avg_instruction
                    },
                    "quality_distribution": {
                        "high_quality_count": high_quality,
                        "high_quality_pct": high_quality/len(scores_list)*100,
                        "low_quality_count": low_quality,
                        "low_quality_pct": low_quality/len(scores_list)*100
                    }
                }
                
                with jsonlines.open(OUTPUT_FILE, 'a') as writer:
                    writer.write(report)
                
                evaluated_checkpoints.add(checkpoint)
            
            else:
                print("⚠️  所有評估都失敗了")
        
        # 顯示當前進度
        if current_count != last_count:
            next_checkpoint = ((current_count // SAMPLE_INTERVAL) + 1) * SAMPLE_INTERVAL
            remaining = next_checkpoint - current_count
            now = datetime.now().strftime("%H:%M:%S")
            print(f"\r[{now}] 已生成: {current_count:,} | 距下次評估: {remaining:,}", end="", flush=True)
            last_count = current_count
        
        time.sleep(60)  # 每分鐘檢查一次
        
except KeyboardInterrupt:
    print("\n\n監控結束")
    print(f"已評估檢查點：{sorted(evaluated_checkpoints)}")
