# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode Run wrapper: claim judge for faithful-prompt counterfactual output."""

import os

from scripts.faithfulness import faithfulness_claim_judge as core


core.INPUT_CSV = os.path.join(core.ANALYSIS_DIR, "counterfactual_generations_faithful.csv")
core.OUTPUT_CSV = os.path.join(core.ANALYSIS_DIR, "claim_annotations_faithful.csv")
core.OUTPUT_JSONL = os.path.join(core.ANALYSIS_DIR, "claim_annotations_faithful.jsonl")
core.OUTPUT_SUMMARY = os.path.join(core.ANALYSIS_DIR, "claim_judge_summary_faithful.json")


if __name__ == "__main__":
    core.main()
