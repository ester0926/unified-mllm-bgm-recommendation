"""
用途：分析偏好改寫前後的推薦解釋差異。
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

from scripts.faithfulness import analyze_preference_counterfactual as analyzer


analyzer.INPUT_CSV = os.path.join(analyzer.ANALYSIS_DIR, "preference_counterfactual_generations_top1.csv")
analyzer.OUTPUT_CSV = os.path.join(analyzer.ANALYSIS_DIR, "preference_counterfactual_analysis_top1.csv")
analyzer.OUTPUT_SUMMARY_JSON = os.path.join(analyzer.ANALYSIS_DIR, "preference_counterfactual_summary_top1.json")
analyzer.OUTPUT_SUMMARY_MD = os.path.join(analyzer.ANALYSIS_DIR, "preference_counterfactual_summary_top1.md")


if __name__ == "__main__":
    analyzer.main()
