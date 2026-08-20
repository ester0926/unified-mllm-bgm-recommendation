"""
用途：執行模型訓練與評估結果的診斷檢查。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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