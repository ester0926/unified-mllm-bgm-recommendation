"""
用途：產生 counterfactual 條件下的推薦解釋，用於後續 faithfulness 分析。
輸入：主評估輸出的推薦解釋、metadata、counterfactual 或人工複查檔。
輸出：claim 標註、faithfulness 指標、UCR 摘要或人工檢查表。
執行：通常需先完成主評估或 Top-1 生成，再執行本檔。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
