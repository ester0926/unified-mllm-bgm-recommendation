# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode Run wrapper: analyze faithful-prompt counterfactual faithfulness."""

import os

from scripts.faithfulness import analyze_faithfulness as core


core.GENERATION_CSV = os.path.join(core.ANALYSIS_DIR, "counterfactual_generations_faithful.csv")
core.CLAIM_CSV = os.path.join(core.ANALYSIS_DIR, "claim_annotations_faithful.csv")
core.OUTPUT_CONDITION_CSV = os.path.join(core.ANALYSIS_DIR, "faithfulness_by_condition_faithful.csv")
core.OUTPUT_SENSITIVITY_CSV = os.path.join(core.ANALYSIS_DIR, "faithfulness_sensitivity_faithful.csv")
core.OUTPUT_SUMMARY_JSON = os.path.join(core.ANALYSIS_DIR, "faithfulness_summary_faithful.json")
core.OUTPUT_SUMMARY_MD = os.path.join(core.ANALYSIS_DIR, "faithfulness_summary_faithful.md")


if __name__ == "__main__":
    core.main()
