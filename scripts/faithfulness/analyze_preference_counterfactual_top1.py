# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode-run wrapper: analyze Top-1 preference counterfactual generations."""

import os

from scripts.faithfulness import analyze_preference_counterfactual as analyzer


analyzer.INPUT_CSV = os.path.join(analyzer.ANALYSIS_DIR, "preference_counterfactual_generations_top1.csv")
analyzer.OUTPUT_CSV = os.path.join(analyzer.ANALYSIS_DIR, "preference_counterfactual_analysis_top1.csv")
analyzer.OUTPUT_SUMMARY_JSON = os.path.join(analyzer.ANALYSIS_DIR, "preference_counterfactual_summary_top1.json")
analyzer.OUTPUT_SUMMARY_MD = os.path.join(analyzer.ANALYSIS_DIR, "preference_counterfactual_summary_top1.md")


if __name__ == "__main__":
    analyzer.main()
