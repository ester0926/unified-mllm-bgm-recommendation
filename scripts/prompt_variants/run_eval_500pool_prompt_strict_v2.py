# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""VSCode Run wrapper: revised strict prompt robustness test."""

from scripts.eval_main import run_eval_500pool_detailed as core


core.EXP_NAME = "exp_01"
core.CKPT_NAME = "best"
core.POOL_SIZE = 500
core.PROMPT_VARIANT = "strict_v2"
core.RESULT_TAG = "prompt_strict_v2"
core.MAX_SAMPLES = None
core.MAX_GEN_SAMPLES = None
core.INJECT_TITLE = True
core.KEEP_PER_SAMPLE_INFOLM = True


if __name__ == "__main__":
    core.main()
