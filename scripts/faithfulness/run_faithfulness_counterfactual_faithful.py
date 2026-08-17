# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode Run wrapper: feature-erasure faithfulness with faithful prompt."""

import os

import run_faithfulness_counterfactual as core


core.PROMPT_VARIANT = "faithful"
core.OUTPUT_CSV = os.path.join(core.OUTPUT_DIR, "counterfactual_generations_faithful.csv")
core.OUTPUT_JSONL = os.path.join(core.OUTPUT_DIR, "counterfactual_generations_faithful.jsonl")
core.OUTPUT_SUMMARY = os.path.join(core.OUTPUT_DIR, "counterfactual_generation_summary_faithful.json")


if __name__ == "__main__":
    core.main()
