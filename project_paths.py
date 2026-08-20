"""
用途：統一管理專案根目錄、資料夾與常用輸出路徑。
輸入：依程式內路徑設定讀取本專案資料或前一階段輸出。
輸出：依程式內 OUTPUT_DIR、results 或 checkpoints 設定寫出結果。
執行：建議在 repo 根目錄執行，避免相對路徑錯誤。
"""

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
