"""
用途：在 Top-1 生成設定下測試不同 prompt 版本對解釋結果的影響。
輸入：已訓練 checkpoint、測試集特徵、候選 pool 與 LTP/cache 資料。
輸出：ranking、generation、指標摘要或逐筆評估檔。
執行：建議在 repo 根目錄執行，必要資料請先由 Zenodo 解壓到對應資料夾。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.eval_main import run_eval_500pool_top1_generation_from_ranking as runner


runner.EXP_NAME = "exp_01"
runner.CKPT_NAME = "best"
runner.POOL_SIZE = 500
runner.PROMPT_VARIANT = "simple_v2"
runner.RANKING_RESULT_TAG = ""
runner.OUTPUT_TAG = "top1_prompt_simple_v2"
runner.MAX_SAMPLES = None
runner.INJECT_TITLE = True
runner.KEEP_PER_SAMPLE_INFOLM = True


if __name__ == "__main__":
    runner.main()
