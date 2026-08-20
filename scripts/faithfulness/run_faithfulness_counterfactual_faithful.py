"""
用途：產生 counterfactual 條件下的推薦解釋，用於後續 faithfulness 分析。
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

import run_faithfulness_counterfactual as core


core.PROMPT_VARIANT = "faithful"
core.OUTPUT_CSV = os.path.join(core.OUTPUT_DIR, "counterfactual_generations_faithful.csv")
core.OUTPUT_JSONL = os.path.join(core.OUTPUT_DIR, "counterfactual_generations_faithful.jsonl")
core.OUTPUT_SUMMARY = os.path.join(core.OUTPUT_DIR, "counterfactual_generation_summary_faithful.json")


if __name__ == "__main__":
    core.main()
