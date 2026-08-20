"""
用途：準備或執行 LLM-as-a-Judge 的解釋品質輔助評估。
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

from scripts.faithfulness import llm_as_judge_faithfulness as judge


judge.OUTPUT_DIR = os.path.join(judge.ANALYSIS_DIR, "llm_judge_top1")
judge.OUTPUT_FEATURE_CSV = os.path.join(judge.OUTPUT_DIR, "llm_judge_feature_erasure_top1.csv")
judge.OUTPUT_PREFERENCE_CSV = os.path.join(judge.OUTPUT_DIR, "llm_judge_preference_counterfactual_top1.csv")
judge.OUTPUT_METADATA_CSV = os.path.join(judge.OUTPUT_DIR, "llm_judge_metadata_consistency_top1.csv")
judge.OUTPUT_SUMMARY_JSON = os.path.join(judge.OUTPUT_DIR, "llm_judge_summary_top1.json")
judge.OUTPUT_SUMMARY_MD = os.path.join(judge.OUTPUT_DIR, "llm_judge_summary_top1.md")

judge.FEATURE_CLAIMS_CSV = os.path.join(judge.ANALYSIS_DIR, "claim_annotations_top1.csv")
judge.PREFERENCE_ANALYSIS_CSV = os.path.join(
    judge.ANALYSIS_DIR,
    "preference_counterfactual_analysis_top1.csv",
)
judge.METADATA_CLAIMS_CSV = os.path.join(
    judge.ANALYSIS_DIR,
    "metadata_consistency_claims_top1_prompt_original.csv",
)


if __name__ == "__main__":
    judge.main()
