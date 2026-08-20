"""
用途：檢查推薦解釋中的音樂細節是否與 metadata 一致。
輸入：主評估輸出的推薦解釋、metadata、counterfactual 或人工複查檔。
輸出：claim 標註、faithfulness 指標、UCR 摘要或人工檢查表。
執行：通常需先完成主評估或 Top-1 生成，再執行本檔。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os

from scripts.faithfulness import analyze_metadata_consistency as analyzer


analyzer.EXP_NAMES = [f"exp_{i:02d}" for i in range(1, 8)]
analyzer.OUTPUT_CSV = os.path.join(
    analyzer.OUTPUT_DIR,
    "metadata_consistency_claims_top1_prompt_original.csv",
)
analyzer.OUTPUT_SUMMARY_JSON = os.path.join(
    analyzer.OUTPUT_DIR,
    "metadata_consistency_summary_top1_prompt_original.json",
)
analyzer.OUTPUT_SUMMARY_MD = os.path.join(
    analyzer.OUTPUT_DIR,
    "metadata_consistency_summary_top1_prompt_original.md",
)
analyzer.INPUT_GENERATION_TAG = "top1_prompt_original"


if __name__ == "__main__":
    analyzer.main()
