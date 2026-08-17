# Auto-added after project reorganization: allow VSCode Run from subfolders.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
check_infolm_measures.py

用途：
確認目前環境 torchmetrics 的 InfoLM 支援哪些 measure 名稱，
避免在 run_eval_500pool 時因名稱錯誤而 crash。
"""

import torch

def main():
    try:
        from torchmetrics.functional.text.infolm import infolm
    except ImportError:
        print("❌ 未安裝 torchmetrics[text]")
        print("請先執行：pip install \"torchmetrics[text]\"")
        return

    hyps = ["this is a test"]
    refs = ["this is a test"]

    print("=" * 50)
    print("InfoLM measure 檢測開始")
    print("=" * 50)

    measures = [
        "ab_divergence",
        "l2_distance",
        "fisher_rao_distance",
        "fisher_rao",
    ]

    results = {}

    for m in measures:
        try:
            val = infolm(
                hyps,
                refs,
                information_measure=m,
                model_name_or_path="bert-base-uncased",
                idf=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False,
            )
            print(f"✅ {m:25s} -> OK | value = {float(val):.6f}")
            results[m] = True

        except Exception as e:
            print(f"❌ {m:25s} -> FAIL | {str(e)}")
            results[m] = False

    print("=" * 50)

    # 🔍 推薦設定
    print("建議使用：")
    if results.get("fisher_rao_distance"):
        print("→ fisher_rao_distance")
    elif results.get("fisher_rao"):
        print("→ fisher_rao")
    else:
        print("⚠ 沒有可用的 Fisher-Rao measure")

    print("=" * 50)


if __name__ == "__main__":
    main()