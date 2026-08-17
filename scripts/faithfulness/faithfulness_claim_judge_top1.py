# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode-run wrapper: claim judge for Top-1 counterfactual generations."""

import os

from scripts.faithfulness import faithfulness_claim_judge as judge


judge.INPUT_CSV = os.path.join(judge.ANALYSIS_DIR, "counterfactual_generations_top1.csv")
judge.OUTPUT_CSV = os.path.join(judge.ANALYSIS_DIR, "claim_annotations_top1.csv")
judge.OUTPUT_JSONL = os.path.join(judge.ANALYSIS_DIR, "claim_annotations_top1.jsonl")
judge.OUTPUT_SUMMARY = os.path.join(judge.ANALYSIS_DIR, "claim_judge_top1_summary.json")


if __name__ == "__main__":
    judge.main()
