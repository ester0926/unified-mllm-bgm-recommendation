"""
用途：提供舊實驗腳本相容的評估函式入口，實作位於 scripts/diagnostics/evaluate.py。
輸入：依程式內路徑設定讀取本專案資料或前一階段輸出。
輸出：依程式內 OUTPUT_DIR、results 或 checkpoints 設定寫出結果。
執行：建議在 repo 根目錄執行，避免相對路徑錯誤。
"""

from scripts.diagnostics.evaluate import *  # noqa: F401,F403
