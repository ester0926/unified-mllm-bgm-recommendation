# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Seed robustness eval: candidate-pool sampling seed 987654.

This changes only the random seed used to sample negative candidates for each
500-pool test instance. Model weights, test split, prompt, and checkpoint stay
fixed.
"""

from scripts.eval_main import run_eval_500pool_detailed as core


core.EXP_NAME = "exp_01"
core.CKPT_NAME = "best"
core.POOL_SIZE = 500
core.PROMPT_VARIANT = "original"
core.RESULT_TAG = "seed987654"
core.CANDIDATE_POOL_SEED = 987654
core.MAX_SAMPLES = None
core.MAX_GEN_SAMPLES = None
core.POINTWISE_BATCH_SIZE = 32
core.INJECT_TITLE = True
core.TIEBREAK_NOISE = True
core.TIEBREAK_SEED = 42
core.KEEP_PER_SAMPLE_INFOLM = True


if __name__ == "__main__":
    core.main()
