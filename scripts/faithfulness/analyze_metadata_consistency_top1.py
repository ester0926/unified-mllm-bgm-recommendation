# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode-run wrapper: metadata consistency using formal Top-1 prompt output."""

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
