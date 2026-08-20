"""
用途：修補 Stage 3 合成對話輸出中的缺漏或格式問題。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

import json
from pathlib import Path

# 設定你的輸出檔案路徑 (根據你的 LOG 修改)
file_path = Path(r"data/user_profiling/long_term_preference\stage3_dialogues\diverse_template\dialogues.jsonl")

def fix_jsonl(path):
    if not path.exists():
        print("檔案不存在！")
        return

    valid_lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError as e:
                print(f"發現損壞資料，位於第 {i+1} 行，將被移除。錯誤: {e}")

    # 寫回檔案 (只保留有效行)
    with open(path, 'w', encoding='utf-8') as f:
        for line in valid_lines:
            f.write(line + '\n')
    
    print(f"修復完成！保留了 {len(valid_lines)} 行有效資料。")

if __name__ == "__main__":
    fix_jsonl(file_path)