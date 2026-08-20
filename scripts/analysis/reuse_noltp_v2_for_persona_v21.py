"""
用途：建立或分析 persona 條件下的 LTP 與評估結果。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import csv
import datetime as dt
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results" / "main_eval" / "exp_01" / "persona_eval_v2" / "persona_v2_no_ltp.csv"
MATCHED = ROOT / "results" / "main_eval" / "exp_01" / "persona_eval_v21" / "persona_v2_matched.csv"
OUT_DIR = ROOT / "results" / "main_eval" / "exp_01" / "persona_eval_v21"
OUTPUT = OUT_DIR / "persona_v2_no_ltp.csv"
PROVENANCE = OUT_DIR / "persona_v2_no_ltp_reuse_provenance.json"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    old = read_csv(OLD)
    matched = read_csv(MATCHED)
    if len(old) != 480 or len(matched) != 480:
        raise ValueError(f"Expected 480 rows, found old={len(old)}, matched={len(matched)}")
    joined = []
    for i, (source, reference) in enumerate(zip(old, matched)):
        identity = ("sample_idx", "video_id", "gt_pair_key", "persona_id")
        mismatched = [key for key in identity if source[key] != reference[key]]
        if mismatched:
            raise ValueError(f"Row {i} identity mismatch: {mismatched}")
        pool = reference["pool_pair_keys"].split(";")
        if len(pool) != 500 or pool[0] != source["gt_pair_key"]:
            raise ValueError(f"Row {i} does not retain the expected GT-first 500-pool")
        copied = dict(source)
        copied["pool_pair_keys"] = reference["pool_pair_keys"]
        joined.append(copied)

    fieldnames = list(joined[0])
    with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(joined)
    provenance = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "decision": "reuse_existing_exp04_noltp_and_attach_v21_candidate_pool_ids",
        "justification": (
            "exp_04 does not consume Persona vectors; v2 and v21 use identical sample "
            "indices/order and candidate-pool construction. Rebuilt histories cannot "
            "change No-LTP ranking or generation."
        ),
        "verified_identity_fields": ["sample_idx", "video_id", "gt_pair_key", "persona_id"],
        "rows": len(joined),
        "pool_size": 500,
        "source": {"path": str(OLD), "sha256": sha256(OLD)},
        "pool_reference": {"path": str(MATCHED), "sha256": sha256(MATCHED)},
        "output": {"path": str(OUTPUT), "sha256": sha256(OUTPUT)},
    }
    PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(joined)} verified rows)")


if __name__ == "__main__":
    main()
