# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode-run wrapper: Top-1 end-to-end generation with the strict_v2 prompt."""

from scripts.eval_main import run_eval_500pool_top1_generation_from_ranking as runner


runner.EXP_NAME = "exp_01"
runner.CKPT_NAME = "best"
runner.POOL_SIZE = 500
runner.PROMPT_VARIANT = "strict_v2"
runner.RANKING_RESULT_TAG = ""
runner.OUTPUT_TAG = "top1_prompt_strict_v2"
runner.MAX_SAMPLES = None
runner.INJECT_TITLE = True
runner.KEEP_PER_SAMPLE_INFOLM = True


if __name__ == "__main__":
    runner.main()
