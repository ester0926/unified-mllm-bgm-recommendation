# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
run_preference_counterfactual_by_path.py
========================================
B3 的 GPU 部分：對 exp_02 / exp_03 / exp_04 補做偏好反事實生成
（exp_01 已於 2026-06-04 完成，輸出即 results/faithfulness/preference_counterfactual_generations_top1.csv）

目的：
  claim 層級的偏好主張驗證只能看「說明有沒有提到偏好、提得對不對」，
  無法回答教授問的「**偏好反事實改變後，說明方向是否同步改變**」。
  本腳本對每個路徑變體各跑一次偏好反事實生成，之後即可逐路徑比較方向敏感度。

做法：
  完全沿用既有的 run_preference_counterfactual_generation_top1.py，
  只在匯入後覆寫 EXP_NAME 與輸出路徑（與 faithfulness_claim_judge_top1_v2.py 相同的包裝寫法），
  不修改原始腳本，確保與 exp_01 既有結果的產生條件完全一致：
    N_SAMPLES=200、SAMPLE_SEED=20260516、PROMPT_VARIANT="original"、
    三個偏好變體 original / cf_upbeat_electronic / cf_lyrical_piano

輸出（results/faithfulness/preference_counterfactual_by_path/）：
  {exp}_preference_counterfactual_generations_top1.csv / .jsonl / _summary.json / .log

使用方式（每個模型一個獨立行程，避免連續載入三次 LLaMA 造成顯存碎片）：
    python scripts/faithfulness/run_preference_counterfactual_by_path.py exp_02
    python scripts/faithfulness/run_preference_counterfactual_by_path.py exp_03
    python scripts/faithfulness/run_preference_counterfactual_by_path.py exp_04

  未帶參數時使用下方 DEFAULT_EXP。VSCode Run 亦可（改 DEFAULT_EXP 即可）。

預估耗時：每個模型約 30–40 分鐘（載入與建測試集約 15 分鐘 + 200×3 生成約 16 分鐘），
三個合計約 1.5–2 小時；需要 CUDA。
"""

from scripts.faithfulness import run_preference_counterfactual_generation_top1 as runner


DEFAULT_EXP = "exp_02"
VALID_EXPS = ("exp_01", "exp_02", "exp_03", "exp_04")

OUT_DIR = PROJECT_ROOT / "results" / "faithfulness" / "preference_counterfactual_by_path"


def configure(exp_name: str):
    if exp_name not in VALID_EXPS:
        raise ValueError(f"EXP_NAME 必須是 {VALID_EXPS} 之一，收到 {exp_name!r}")

    ranking_csv = (PROJECT_ROOT / "checkpoints" / exp_name / "detailed_eval"
                   / f"{exp_name}_best_500pool_ranking_samples.csv")
    if not ranking_csv.exists():
        raise FileNotFoundError(
            f"找不到 {exp_name} 的排序結果：{ranking_csv}\n"
            "偏好反事實生成需沿用既有排序的 top-1，請先確認該檔存在。"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runner.EXP_NAME = exp_name
    runner.OUTPUT_DIR = str(OUT_DIR)
    runner.OUTPUT_CSV = str(OUT_DIR / f"{exp_name}_preference_counterfactual_generations_top1.csv")
    runner.OUTPUT_JSONL = str(OUT_DIR / f"{exp_name}_preference_counterfactual_generations_top1.jsonl")
    runner.OUTPUT_SUMMARY = str(
        OUT_DIR / f"{exp_name}_preference_counterfactual_generation_top1_summary.json")

    # 原腳本的 log 檔名固定，改成逐模型分開以免互相覆蓋
    _orig_setup = runner.core.setup_logger

    def _setup_logger(_path):
        # 忽略呼叫端傳入的固定路徑，改用逐模型分開的 log 檔
        return _orig_setup(str(OUT_DIR / f"{exp_name}_preference_counterfactual.log"))

    runner.core.setup_logger = _setup_logger

    print(f"[設定] exp={exp_name}")
    print(f"[設定] ranking_csv={ranking_csv}")
    print(f"[設定] 輸出={runner.OUTPUT_CSV}")
    print(f"[設定] n_samples={runner.N_SAMPLES} seed={runner.SAMPLE_SEED} "
          f"variants={list(runner.PREFERENCE_VARIANTS)}")


def main():
    """依命令列參數（或 DEFAULT_EXP）設定模型別，再呼叫既有的偏好反事實生成流程。"""
    exp_name = "exp_04"  # 預設值
    configure(exp_name)
    runner.main()


if __name__ == "__main__":
    main()
