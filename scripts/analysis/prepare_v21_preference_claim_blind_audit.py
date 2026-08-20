"""
用途：準備偏好 claim 人工盲審資料。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import csv
import hashlib
import random


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "analysis" / "path_level_generation_v21" / "preference_claims_full_audit.csv"
OUT_DIR = ROOT / "results" / "analysis" / "path_level_generation_v21"
SEED = 20260729
PER_EXPERIMENT = 25


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_rows(INPUT)
    rng = random.Random(SEED)
    selected = []
    for exp in sorted({row["exp"] for row in rows}):
        pool = [row for row in rows if row["exp"] == exp]
        by_label = {}
        for row in pool:
            by_label.setdefault(row["verification"], []).append(row)
        # 先覆蓋所有已觀察到的規則分層，再隨機補足樣本。
        # 釋出的審查包不會揭露分層或實驗條件。
        chosen = []
        labels = sorted(by_label)
        base = max(1, PER_EXPERIMENT // max(len(labels), 1))
        for label in labels:
            candidates = by_label[label][:]
            rng.shuffle(candidates)
            chosen.extend(candidates[:base])
        chosen_ids = {id(row) for row in chosen}
        remaining = [row for row in pool if id(row) not in chosen_ids]
        rng.shuffle(remaining)
        chosen.extend(remaining[: max(0, PER_EXPERIMENT - len(chosen))])
        selected.extend(chosen[:PER_EXPERIMENT])

    rng.shuffle(selected)
    packet, key = [], []
    for ordinal, row in enumerate(selected, start=1):
        digest = hashlib.sha256(
            f"{SEED}|{row['exp']}|{row['sample_idx']}|{row['claim_id']}".encode("utf-8")
        ).hexdigest()[:12]
        audit_id = f"PC-{ordinal:03d}-{digest}"
        packet.append({
            "audit_id": audit_id,
            "claim_text": row["claim_text"],
            "positive_reference": row["reference_positive_evidence"],
            "negative_reference": row["reference_negative_evidence"],
            "adjudicated_label": "",
            "adjudicator_note": "",
        })
        key.append({
            "audit_id": audit_id,
            "exp": row["exp"],
            "sample_idx": row["sample_idx"],
            "video_id": row["video_id"],
            "claim_id": row["claim_id"],
            "rule_label": row["verification"],
        })

    for path, data in [
        (OUT_DIR / "preference_claim_blind_audit_packet.csv", packet),
        (OUT_DIR / "preference_claim_blind_audit_key.csv", key),
    ]:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f"Wrote {len(packet)} blinded claims ({PER_EXPERIMENT} per experiment).")


if __name__ == "__main__":
    main()
