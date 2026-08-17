from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"


def add_root_to_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
