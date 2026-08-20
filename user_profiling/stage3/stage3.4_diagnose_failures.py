"""
用途：診斷 Stage 3 產生失敗或品質不足的原因。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

import jsonlines
import json
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

# ==================== 配置 ====================
DIALOGUES_FILE = Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl")
OUTPUT_DIR = Path("data/user_profiling/long_term_preference/stage3_dialogues/diverse_template/stage3.5_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TYPES = ["Positive", "Exploratory", "Negative"]

# ==================== 讀取數據 ====================
print("🔍 Reading dialogues...")
music_dialogues = defaultdict(dict)  # {music_id: {dialogue_type: dialogue_data}}

with jsonlines.open(DIALOGUES_FILE) as reader:
    for obj in tqdm(reader, desc="Loading"):
        music_id = obj.get('music_id')
        dtype = obj.get('dialogue_type')
        if music_id and dtype:
            music_dialogues[music_id][dtype] = obj

total_songs = len(music_dialogues)
total_dialogues = sum(len(v) for v in music_dialogues.values())

print(f"\n📊 統計結果：")
print(f"   總歌曲數：{total_songs}")
print(f"   總對話數：{total_dialogues}")
print(f"   預期對話數：{total_songs * 3}")

# ==================== 完整性檢查 ====================
complete_songs = []
incomplete_songs = []
missing_combinations = []

for music_id, dialogues in tqdm(music_dialogues.items(), desc="Checking completeness"):
    existing_types = set(dialogues.keys())
    expected_set = set(EXPECTED_TYPES)
    
    if existing_types == expected_set:
        complete_songs.append(music_id)
    else:
        missing_types = expected_set - existing_types
        incomplete_songs.append({
            'music_id': music_id,
            'existing': list(existing_types),
            'missing': list(missing_types)
        })
        for dtype in missing_types:
            missing_combinations.append({
                'music_id': music_id,
                'dialogue_type': dtype
            })

print(f"\n✅ 完整歌曲：{len(complete_songs)} ({len(complete_songs)/total_songs*100:.2f}%)")
print(f"⚠️  不完整歌曲：{len(incomplete_songs)} ({len(incomplete_songs)/total_songs*100:.2f}%)")
print(f"📝 缺失對話數：{len(missing_combinations)}")

# ==================== 分析缺失模式 ====================
missing_type_counter = Counter()
for item in incomplete_songs:
    for dtype in item['missing']:
        missing_type_counter[dtype] += 1

print(f"\n📈 缺失對話類型分布：")
for dtype, count in missing_type_counter.most_common():
    print(f"   {dtype}: {count} ({count/len(incomplete_songs)*100:.1f}%)")

# ==================== 保存診斷報告 ====================
report = {
    'summary': {
        'total_songs': total_songs,
        'total_dialogues': total_dialogues,
        'expected_dialogues': total_songs * 3,
        'complete_songs': len(complete_songs),
        'incomplete_songs': len(incomplete_songs),
        'missing_dialogues': len(missing_combinations),
        'completion_rate': f"{len(complete_songs)/total_songs*100:.2f}%"
    },
    'missing_type_distribution': dict(missing_type_counter),
    'incomplete_songs': incomplete_songs[:100],  # 只保存前100個範例
    'sample_missing': missing_combinations[:20]  # 範例
}

report_file = OUTPUT_DIR / "diagnosis_report.json"
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n💾 診斷報告已保存：{report_file}")

# ==================== 生成補齊任務清單 ====================
補齊_file = OUTPUT_DIR / "missing_tasks.jsonl"
with jsonlines.open(補齊_file, 'w') as writer:
    for task in missing_combinations:
        writer.write(task)

print(f"📋 補齊任務清單已保存：{補齊_file}")
print(f"   共 {len(missing_combinations)} 個待補齊任務")

# ==================== 嚴重性評估 ====================
print("\n" + "="*60)
print("🎯 評估與建議：")
print("="*60)

missing_rate = len(missing_combinations) / (total_songs * 3) * 100

if missing_rate < 1:
    print(f"✅ 缺失率 {missing_rate:.2f}% < 1%，屬於正常範圍")
    print("   建議：執行補齊程式即可，不需修改主程式")
elif missing_rate < 5:
    print(f"⚠️  缺失率 {missing_rate:.2f}% 在 1-5%，略高但可接受")
    print("   建議：執行補齊程式，並檢查是否有特定模式")
else:
    print(f"❌ 缺失率 {missing_rate:.2f}% > 5%，需要調查原因")
    print("   建議：檢查驗證邏輯或提示詞設計")

print("\n📌 下一步操作：")
print("   1. 執行補齊程式：python stage3_補齊_missing.py")
print("   2. 若補齊後仍有失敗，可調整溫度參數或放寬驗證條件")
print("="*60)