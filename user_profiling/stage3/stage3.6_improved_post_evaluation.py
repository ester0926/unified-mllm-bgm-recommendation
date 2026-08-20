"""
用途：產生或評估 Stage 3 的合成使用者偏好對話。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

import jsonlines
import random
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from quality_evaluator import QualityEvaluatorLLM
from llm_client import OllamaClient
from tqdm import tqdm

# ==================== 配置區（可修改） ====================
CONFIG = {
    # 抽樣參數
    'NUM_SONGS': 1000,             # 抽樣歌曲數（會產生 1000×3 = 3000 筆對話）
    'RANDOM_SEED': 42,             # 隨機種子（確保可重複）
    'FORCE_REEVALUATE': False,     # 是否強制重新評估
    
    # 檔案路徑
    'INPUT_FILE': Path('data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl'),
    'OUTPUT_DIR': Path('data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/stage3.4_post_evaluation'),
    
    # 評估器配置
    'JUDGE_MODEL': 'gemma3:12b',
    'JUDGE_TEMPERATURE': 0.0       # 確保評分一致性
}

# ==================== 歌曲級抽樣函數 ====================
def song_based_sampling(all_dialogues, num_songs, random_seed=42):
    """
    歌曲級抽樣：先抽歌曲，再取該歌的全部 3 種對話
    
    Args:
        all_dialogues: 所有對話列表
        num_songs: 要抽樣的歌曲數量
        random_seed: 隨機種子
    
    Returns:
        抽樣的對話列表（num_songs × 3 筆）
    """
    # 1. 按 music_id 分組
    dialogues_by_song = defaultdict(list)
    for d in all_dialogues:
        music_id = d.get('music_id')
        if music_id:
            dialogues_by_song[music_id].append(d)
    
    # 2. 檢查數據完整性
    complete_songs = []
    incomplete_songs = []
    
    for music_id, dialogues in dialogues_by_song.items():
        # 每首歌應該有 3 種對話
        types = {d.get('dialogue_type') for d in dialogues}
        expected_types = {'Positive', 'Exploratory', 'Negative'}
        
        if types == expected_types:
            complete_songs.append(music_id)
        else:
            incomplete_songs.append({
                'music_id': music_id,
                'existing': list(types),
                'missing': list(expected_types - types)
            })
    
    print(f"\n數據完整性檢查：")
    print(f"  完整歌曲：{len(complete_songs):,} 首 ({len(complete_songs)/len(dialogues_by_song)*100:.2f}%)")
    print(f"  不完整歌曲：{len(incomplete_songs):,} 首")
    
    if incomplete_songs:
        print(f"  ⚠️  警告：{len(incomplete_songs)} 首歌的對話不完整，將被排除")
        print(f"     （前 5 個範例：{[s['music_id'] for s in incomplete_songs[:5]]}）")
    
    # 3. 從完整歌曲中抽樣
    if len(complete_songs) < num_songs:
        print(f"\n  ⚠️  可用完整歌曲數 ({len(complete_songs)}) 少於需求 ({num_songs})")
        print(f"      將使用全部 {len(complete_songs)} 首歌")
        sampled_song_ids = complete_songs
    else:
        random.seed(random_seed)
        sampled_song_ids = random.sample(complete_songs, num_songs)
    
    # 4. 收集抽樣歌曲的全部對話
    sampled_dialogues = []
    for music_id in sampled_song_ids:
        sampled_dialogues.extend(dialogues_by_song[music_id])
    
    # 5. 驗證抽樣結果
    type_dist = defaultdict(int)
    for d in sampled_dialogues:
        type_dist[d.get('dialogue_type')] += 1
    
    print(f"\n抽樣結果：")
    print(f"  歌曲數：{len(sampled_song_ids):,} 首")
    print(f"  對話數：{len(sampled_dialogues):,} 筆")
    print(f"\n  對話類型分布：")
    for dtype in ['Positive', 'Exploratory', 'Negative']:
        count = type_dist.get(dtype, 0)
        pct = count / len(sampled_dialogues) * 100 if sampled_dialogues else 0
        print(f"    {dtype:12s}: {count:5d} ({pct:5.1f}%)")
    
    expected_per_type = len(sampled_song_ids)
    if all(type_dist[t] == expected_per_type for t in ['Positive', 'Exploratory', 'Negative']):
        print(f"  ✅ 每種類型都是 {expected_per_type} 筆（完美 1:1:1）")
    else:
        print(f"  ⚠️  分布不均勻，可能有數據問題")
    
    return sampled_dialogues

# ==================== 主程式 ====================
def main():
    # 設定隨機種子
    random.seed(CONFIG['RANDOM_SEED'])
    
    # 建立輸出目錄
    CONFIG['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Stage 3.4: 事後品質評估（歌曲級抽樣）")
    print("=" * 80)
    print()
    print(f"配置資訊：")
    print(f"  抽樣歌曲數: {CONFIG['NUM_SONGS']:,} 首")
    print(f"  預期對話數: {CONFIG['NUM_SONGS'] * 3:,} 筆（每首歌 3 種對話）")
    print(f"  隨機種子: {CONFIG['RANDOM_SEED']}")
    print(f"  評審模型: {CONFIG['JUDGE_MODEL']}")
    print(f"  預期分布: 1:1:1（Positive : Exploratory : Negative）")
    print()
    
    # ========== 載入對話 ==========
    dialogue_file = CONFIG['INPUT_FILE']
    if not dialogue_file.exists():
        print(f"❌ 錯誤：找不到檔案 {dialogue_file}")
        return
    
    print(f"載入對話檔案：{dialogue_file}")
    with jsonlines.open(dialogue_file) as reader:
        all_dialogues = list(reader)
    
    print(f"總對話數：{len(all_dialogues):,}")
    
    # ========== 檢查已評估 ==========
    evaluated_file = CONFIG['OUTPUT_DIR'] / "evaluated_dialogues.jsonl"
    evaluated_ids = set()
    
    if evaluated_file.exists() and not CONFIG['FORCE_REEVALUATE']:
        print("\n發現已評估的對話，載入中...")
        with jsonlines.open(evaluated_file) as reader:
            for obj in reader:
                evaluated_ids.add(f"{obj['music_id']}_{obj['dialogue_type']}")
        print(f"已評估：{len(evaluated_ids):,} 筆")
    
    # ========== 過濾未評估 ==========
    unevaluated = [d for d in all_dialogues 
                   if f"{d['music_id']}_{d['dialogue_type']}" not in evaluated_ids]
    
    print(f"待評估對話：{len(unevaluated):,}")
    
    # ========== 歌曲級抽樣 ==========
    print("\n" + "=" * 80)
    print("執行歌曲級抽樣（每首歌取全部 3 種對話）")
    print("=" * 80)
    
    sample = song_based_sampling(
        unevaluated, 
        CONFIG['NUM_SONGS'],
        CONFIG['RANDOM_SEED']
    )
    
    if not sample:
        print("\n❌ 無可用樣本，程式結束")
        return
    
    # ========== 初始化評估器 ==========
    print("\n" + "=" * 80)
    print("初始化評估器")
    print("=" * 80)
    print(f"評審模型: {CONFIG['JUDGE_MODEL']}")
    
    evaluator = QualityEvaluatorLLM()
    evaluator.client = OllamaClient(
        model_name=CONFIG['JUDGE_MODEL'],
        temperature=CONFIG['JUDGE_TEMPERATURE']
    )
    print("✅ 評估器就緒\n")
    
    # ========== 執行評估 ==========
    print("=" * 80)
    print("開始評估")
    print("=" * 80)
    
    est_time_hours = len(sample) * 8 / 3600
    print(f"對話數：{len(sample):,}")
    print(f"預估時間：{est_time_hours:.1f} 小時 ({est_time_hours/24:.2f} 天)\n")
    
    results = []
    failed = []
    
    with jsonlines.open(evaluated_file, 'a') as writer:
        for dialogue in tqdm(sample, desc="評估進度"):
            try:
                scores = evaluator.evaluate(
                    dialogue['dialogue_turns'],
                    dialogue['persona_snippet'],
                    dialogue['dialogue_type']
                )
                
                result = {
                    'music_id': dialogue['music_id'],
                    'dialogue_type': dialogue['dialogue_type'],
                    'retry_count': dialogue.get('retry_count', 0),
                    'turn_count': len(dialogue['dialogue_turns']),
                    'scores': scores,
                    'evaluated_at': datetime.now().isoformat()
                }
                
                results.append(result)
                writer.write(result)
                
            except Exception as e:
                failed.append({
                    'music_id': dialogue['music_id'],
                    'dialogue_type': dialogue['dialogue_type'],
                    'error': str(e)
                })
                tqdm.write(f"⚠️  評估失敗：{dialogue['music_id']} ({dialogue['dialogue_type']}) - {e}")
    
    print(f"\n評估完成！")
    print(f"  成功：{len(results):,}")
    print(f"  失敗：{len(failed):,}")
    
    if len(failed) > 0:
        fail_rate = len(failed) / (len(results) + len(failed)) * 100
        print(f"  失敗率：{fail_rate:.2f}%")
    print()
    
    # ========== 生成統計報告 ==========
    if not results:
        print("❌ 無有效評估結果，無法生成報告")
        return
    
    print("=" * 80)
    print("生成統計報告")
    print("=" * 80)
    
    # 提取各維度分數
    overall_scores = [r['scores'].get('overall', 0) for r in results]
    coherence_scores = [r['scores'].get('coherence', 0) for r in results]
    consistency_scores = [r['scores'].get('consistency', 0) for r in results]
    naturalness_scores = [r['scores'].get('naturalness', 0) for r in results]
    instruction_scores = [r['scores'].get('instruction_following', 0) for r in results]
    
    # 計算評估的歌曲數
    evaluated_songs = len(set(r['music_id'] for r in results))
    
    # 構建報告
    report = {
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'total_dialogues_in_file': len(all_dialogues),
            'evaluated_songs': evaluated_songs,
            'evaluated_dialogues': len(results),
            'target_songs': CONFIG['NUM_SONGS'],
            'random_seed': CONFIG['RANDOM_SEED'],
            'judge_model': CONFIG['JUDGE_MODEL'],
            'sampling_method': 'song-based (每首歌取全部 3 種對話)'
        },
        'overall_statistics': {
            'overall': {
                'mean': float(np.mean(overall_scores)),
                'std': float(np.std(overall_scores)),
                'min': float(np.min(overall_scores)),
                'max': float(np.max(overall_scores)),
                'median': float(np.median(overall_scores)),
                'q1': float(np.percentile(overall_scores, 25)),
                'q3': float(np.percentile(overall_scores, 75))
            },
            'coherence': {
                'mean': float(np.mean(coherence_scores)),
                'std': float(np.std(coherence_scores))
            },
            'consistency': {
                'mean': float(np.mean(consistency_scores)),
                'std': float(np.std(consistency_scores))
            },
            'naturalness': {
                'mean': float(np.mean(naturalness_scores)),
                'std': float(np.std(naturalness_scores))
            },
            'instruction_following': {
                'mean': float(np.mean(instruction_scores)),
                'std': float(np.std(instruction_scores))
            }
        },
        'quality_distribution': {
            'excellent_4.5+': sum(1 for s in overall_scores if s >= 4.5),
            'good_4.0-4.5': sum(1 for s in overall_scores if 4.0 <= s < 4.5),
            'acceptable_3.5-4.0': sum(1 for s in overall_scores if 3.5 <= s < 4.0),
            'poor_below_3.5': sum(1 for s in overall_scores if s < 3.5)
        },
        'by_dialogue_type': {}
    }
    
    # 按類型統計
    for dtype in ['Positive', 'Exploratory', 'Negative']:
        type_results = [r for r in results if r['dialogue_type'] == dtype]
        if not type_results:
            continue
        
        type_scores = [r['scores'].get('overall', 0) for r in type_results]
        type_coherence = [r['scores'].get('coherence', 0) for r in type_results]
        type_consistency = [r['scores'].get('consistency', 0) for r in type_results]
        type_naturalness = [r['scores'].get('naturalness', 0) for r in type_results]
        type_instruction = [r['scores'].get('instruction_following', 0) for r in type_results]
        
        report['by_dialogue_type'][dtype] = {
            'count': len(type_results),
            'overall': {
                'mean': float(np.mean(type_scores)),
                'std': float(np.std(type_scores)),
                'min': float(np.min(type_scores)),
                'max': float(np.max(type_scores))
            },
            'coherence': {'mean': float(np.mean(type_coherence))},
            'consistency': {'mean': float(np.mean(type_consistency))},
            'naturalness': {'mean': float(np.mean(type_naturalness))},
            'instruction_following': {'mean': float(np.mean(type_instruction))}
        }
    
    # 歌曲級分析（同一首歌的 3 種對話比較）
    songs_analysis = defaultdict(lambda: defaultdict(dict))
    for r in results:
        music_id = r['music_id']
        dtype = r['dialogue_type']
        songs_analysis[music_id][dtype] = r['scores'].get('overall', 0)
    
    # 找出同一首歌 3 種對話分數差異最大的
    score_variances = []
    for music_id, type_scores in songs_analysis.items():
        if len(type_scores) == 3:  # 確保有全部 3 種對話
            scores = list(type_scores.values())
            variance = np.std(scores)
            score_variances.append({
                'music_id': music_id,
                'variance': variance,
                'scores': dict(type_scores)
            })
    
    score_variances.sort(key=lambda x: x['variance'], reverse=True)
    
    report['song_level_analysis'] = {
        'songs_with_all_types': len(score_variances),
        'avg_variance_across_types': float(np.mean([s['variance'] for s in score_variances])) if score_variances else 0,
        'top_10_most_variant_songs': score_variances[:10]
    }
    
    # 儲存報告
    report_file = CONFIG['OUTPUT_DIR'] / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # ========== 顯示摘要 ==========
    print("\n" + "=" * 80)
    print("評估摘要")
    print("=" * 80)
    
    stats = report['overall_statistics']
    dist = report['quality_distribution']
    
    print(f"\n評估規模：")
    print(f"  歌曲數：{evaluated_songs:,} 首")
    print(f"  對話數：{len(results):,} 筆")
    print()
    
    print("平均分數 (±標準差)：")
    print(f"  Overall:              {stats['overall']['mean']:.3f} ± {stats['overall']['std']:.3f}")
    print(f"  Coherence:            {stats['coherence']['mean']:.3f} ± {stats['coherence']['std']:.3f}")
    print(f"  Consistency:          {stats['consistency']['mean']:.3f} ± {stats['consistency']['std']:.3f}")
    print(f"  Naturalness:          {stats['naturalness']['mean']:.3f} ± {stats['naturalness']['std']:.3f}")
    print(f"  Instruction Following: {stats['instruction_following']['mean']:.3f} ± {stats['instruction_following']['std']:.3f}")
    
    print(f"\n品質分佈：")
    total = len(results)
    print(f"  優秀 (≥ 4.5):      {dist['excellent_4.5+']:5d} ({dist['excellent_4.5+']/total*100:5.1f}%)")
    print(f"  良好 (4.0-4.5):    {dist['good_4.0-4.5']:5d} ({dist['good_4.0-4.5']/total*100:5.1f}%)")
    print(f"  可接受 (3.5-4.0):  {dist['acceptable_3.5-4.0']:5d} ({dist['acceptable_3.5-4.0']/total*100:5.1f}%)")
    print(f"  不佳 (< 3.5):      {dist['poor_below_3.5']:5d} ({dist['poor_below_3.5']/total*100:5.1f}%)")
    
    print(f"\n各類型平均分數：")
    for dtype in ['Positive', 'Exploratory', 'Negative']:
        if dtype not in report['by_dialogue_type']:
            continue
        type_stats = report['by_dialogue_type'][dtype]
        print(f"\n  {dtype}:")
        print(f"    樣本數: {type_stats['count']:,}")
        print(f"    Overall:              {type_stats['overall']['mean']:.3f} ± {type_stats['overall']['std']:.3f}")
        print(f"    Coherence:            {type_stats['coherence']['mean']:.3f}")
        print(f"    Consistency:          {type_stats['consistency']['mean']:.3f}")
        print(f"    Naturalness:          {type_stats['naturalness']['mean']:.3f}")
        print(f"    Instruction Following: {type_stats['instruction_following']['mean']:.3f}")
    
    # 歌曲級分析
    song_analysis = report['song_level_analysis']
    print(f"\n歌曲級分析（同一首歌的 3 種對話比較）：")
    print(f"  完整評估歌曲數：{song_analysis['songs_with_all_types']:,} 首")
    print(f"  平均分數差異（標準差）：{song_analysis['avg_variance_across_types']:.3f}")
    print(f"\n  分數差異最大的前 3 首歌：")
    for i, item in enumerate(song_analysis['top_10_most_variant_songs'][:3], 1):
        print(f"    {i}. {item['music_id']}: 差異 {item['variance']:.3f}")
        print(f"       Positive: {item['scores'].get('Positive', 0):.2f} | "
              f"Exploratory: {item['scores'].get('Exploratory', 0):.2f} | "
              f"Negative: {item['scores'].get('Negative', 0):.2f}")
    
    print(f"\n{'='*80}")
    print(f"✅ 報告已儲存")
    print(f"{'='*80}")
    print(f"報告檔案：{report_file}")
    print(f"詳細結果：{evaluated_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()