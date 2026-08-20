"""
用途：將推薦解釋切成 claim，並依規則標註每個 claim 的支持來源。
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

from scripts.faithfulness import faithfulness_claim_judge_v2 as judge


judge.INPUT_CSV = os.path.join(judge.ANALYSIS_DIR, "counterfactual_generations_top1.csv")
judge.OUTPUT_CSV = os.path.join(judge.ANALYSIS_DIR, "claim_annotations_top1_v2.csv")
judge.OUTPUT_JSONL = os.path.join(judge.ANALYSIS_DIR, "claim_annotations_top1_v2.jsonl")
judge.OUTPUT_SUMMARY = os.path.join(judge.ANALYSIS_DIR, "claim_judge_top1_summary_v2.json")


if __name__ == "__main__":
    judge.main()
