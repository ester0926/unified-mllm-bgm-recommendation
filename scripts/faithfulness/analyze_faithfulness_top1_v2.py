# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode-run wrapper: faithfulness metrics for Top-1 v2 claim annotations."""

import os

from scripts.faithfulness import analyze_faithfulness as analyzer


analyzer.GENERATION_CSV = os.path.join(analyzer.ANALYSIS_DIR, "counterfactual_generations_top1.csv")
analyzer.CLAIM_CSV = os.path.join(analyzer.ANALYSIS_DIR, "claim_annotations_top1_v2.csv")
analyzer.OUTPUT_CONDITION_CSV = os.path.join(analyzer.ANALYSIS_DIR, "faithfulness_by_condition_top1_v2.csv")
analyzer.OUTPUT_SENSITIVITY_CSV = os.path.join(analyzer.ANALYSIS_DIR, "faithfulness_sensitivity_top1_v2.csv")
analyzer.OUTPUT_SUMMARY_JSON = os.path.join(analyzer.ANALYSIS_DIR, "faithfulness_summary_top1_v2.json")
analyzer.OUTPUT_SUMMARY_MD = os.path.join(analyzer.ANALYSIS_DIR, "faithfulness_summary_top1_v2.md")


if __name__ == "__main__":
    analyzer.main()
