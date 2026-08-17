from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Analyse the fixed exp_01 Hybrid component intervention outputs."""

import csv
import datetime as dt
import json

import numpy as np

from scripts.analysis import path_level_generation_analysis_v21 as path_analysis
from scripts.faithfulness import analyze_ucr_error_sources as B2


INPUT_DIR = PROJECT_ROOT / "results" / "main_eval" / "exp_01" / "fixed_component_intervention_v21"
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "fixed_hybrid_component_v21"
CONDITIONS = (
    "full", "no_explicit", "no_implicit",
    "no_explicit_norm", "no_implicit_norm", "no_both",
)
GEN_CONDITIONS = {"full", "no_explicit_norm", "no_implicit_norm"}
LABELS = {
    "full": "完整 Hybrid",
    "no_explicit": "移除顯式成分（原始）",
    "no_implicit": "移除隱式成分（原始）",
    "no_explicit_norm": "移除顯式成分（範數校正）",
    "no_implicit_norm": "移除隱式成分（範數校正）",
    "no_both": "同時移除兩成分",
}
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260729


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranking_values(rows):
    ranks = np.array([int(r["rank"]) for r in rows], dtype=float)
    return {
        "recall@1": (ranks <= 1).astype(float),
        "recall@5": (ranks <= 5).astype(float),
        "recall@10": (ranks <= 10).astype(float),
        "mrr": 1.0 / ranks,
        "rank": ranks,
    }


def paired_bootstrap(a, b, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    point = float(np.mean(a - b))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sel = rng.integers(0, len(a), size=len(a))
        draws[i] = np.mean(a[sel] - b[sel])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = path_analysis.setup_logger(OUT_DIR / "fixed_hybrid_component_analysis.log")
    rows_by_condition = {
        c: read_csv(INPUT_DIR / f"fixed_component_{c}.csv") for c in CONDITIONS
    }
    keys = {
        c: [(r.get("sample_idx"), r.get("video_id")) for r in rows]
        for c, rows in rows_by_condition.items()
    }
    if any(keys[c] != keys["full"] for c in CONDITIONS[1:]):
        raise ValueError("Condition outputs are not paired in the same sample order.")

    profiles = path_analysis.load_profiles(logger)
    music_meta = B2.load_music_metadata(logger)
    counts = {}
    generation_metrics = {}
    claim_records = []
    ranking_metrics = {}
    for condition, rows in rows_by_condition.items():
        if condition in GEN_CONDITIONS:
            count_matrix, claims = path_analysis.annotate_experiment(
                condition, rows, profiles, music_meta, logger,
            )
            counts[condition] = count_matrix
            generation_metrics[condition] = path_analysis.compute_metrics(count_matrix)
            claim_records.extend(claims)
        else:
            generation_metrics[condition] = {
                key: float("nan") for key in path_analysis.compute_metrics(
                    np.zeros((1, len(path_analysis.COUNT_FIELDS)), dtype=float)
                )
            }
        vals = ranking_values(rows)
        ranking_metrics[condition] = {
            "n": len(rows),
            "recall@1": float(vals["recall@1"].mean()),
            "recall@5": float(vals["recall@5"].mean()),
            "recall@10": float(vals["recall@10"].mean()),
            "mrr": float(vals["mrr"].mean()),
            "median_rank": float(np.median(vals["rank"])),
            "mean_rank": float(vals["rank"].mean()),
            "ltp_vector_norm_mean": float(np.mean([float(r["ltp_vector_norm"]) for r in rows])),
            "ltp_shift_norm_mean": float(np.mean([float(r["ltp_shift_norm"]) for r in rows])),
        }

    rank_contrasts = []
    gen_contrasts = []
    full_rank = ranking_values(rows_by_condition["full"])
    for condition in CONDITIONS[1:]:
        other_rank = ranking_values(rows_by_condition[condition])
        for metric in ("recall@1", "recall@5", "recall@10", "mrr"):
            point, lo, hi = paired_bootstrap(full_rank[metric], other_rank[metric])
            rank_contrasts.append({
                "comparison": f"full - {condition}",
                "metric": metric,
                "difference": point,
                "ci_low": lo,
                "ci_high": hi,
                "significant": bool(lo > 0 or hi < 0),
            })
        if condition in GEN_CONDITIONS:
            result = path_analysis.clustered_bootstrap_contrast(
                counts["full"], counts[condition], path_analysis.CONTRAST_METRICS,
                3000, BOOTSTRAP_SEED, 0.05,
            )
            for metric, (point, lo, hi) in result.items():
                gen_contrasts.append({
                    "comparison": f"full - {condition}",
                    "metric": metric,
                    "difference": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": bool(lo > 0 or hi < 0),
                })

    change_metrics = []
    full_rows = rows_by_condition["full"]
    for condition in CONDITIONS[1:]:
        other = rows_by_condition[condition]
        top1_changed = np.mean([
            a["top1_pair_key"] != b["top1_pair_key"] for a, b in zip(full_rows, other)
        ])
        text_changed = (np.mean([
            a["generated_text"].strip() != b["generated_text"].strip()
            for a, b in zip(full_rows, other)
        ]) if condition in GEN_CONDITIONS else None)
        change_metrics.append({
            "condition": condition,
            "top1_changed_rate": float(top1_changed),
            "generated_text_changed_rate": (float(text_changed) if text_changed is not None else None),
        })

    metrics_rows = []
    for condition in CONDITIONS:
        metrics_rows.append({
            "condition": condition,
            "label": LABELS[condition],
            "generation_evaluated": condition in GEN_CONDITIONS,
            **ranking_metrics[condition],
            **generation_metrics[condition],
        })
    write_csv(OUT_DIR / "fixed_component_metrics.csv", metrics_rows)
    write_csv(OUT_DIR / "fixed_component_ranking_contrasts.csv", rank_contrasts)
    write_csv(OUT_DIR / "fixed_component_generation_contrasts.csv", gen_contrasts)
    write_csv(OUT_DIR / "fixed_component_change_rates.csv", change_metrics)
    write_csv(OUT_DIR / "fixed_component_claims.csv", claim_records)

    with open(INPUT_DIR / "fixed_component_intervention_summary.json", encoding="utf-8") as f:
        experiment_summary = json.load(f)
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "labels": LABELS,
        "experiment": experiment_summary,
        "metrics": {r["condition"]: r for r in metrics_rows},
        "ranking_contrasts": rank_contrasts,
        "generation_contrasts": gen_contrasts,
        "change_rates": change_metrics,
        "method": {
            "design": "固定 exp_01 checkpoint，在推論時自 Hybrid 向量扣除線性分解出的顯式／隱式成分",
            "sample": "固定亂數抽取 200 筆測試查詢；四條件共享相同查詢、500 候選池與候選池種子",
            "ranking_statistics": "樣本配對拔靴 5,000 次，報完整 Hybrid 減介入條件之差與 95% CI",
            "generation_statistics": "claim 巢套於樣本，採樣本層級配對叢集拔靴 3,000 次",
            "boundary": "此為單一 checkpoint 的表徵層探索性介入；原始扣除可能形成離開訓練分布的向量，因此另報範數校正敏感度分析，但兩者仍不能單獨建立一般性的因果歸因",
        },
    }
    with open(OUT_DIR / "fixed_component_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_markdown(OUT_DIR / "fixed_component_summary.md", summary)
    logger.info("Analysis written to %s", OUT_DIR)


def pct(value):
    return "NA" if value is None or np.isnan(value) else f"{value * 100:.2f}%"


def write_markdown(path, summary):
    lines = [
        "# 固定 Hybrid 模型之偏好成分介入分析（v21）",
        "",
        f"- 產生時間：{summary['generated_at']}",
        "- 比較方向：完整 Hybrid 減介入條件；正值表示完整 Hybrid 較高。",
        "",
        "## 一、分解驗證",
        "",
    ]
    d = summary["experiment"]["decomposition"]
    lines.extend([
        f"- 保留集 R²：{d['holdout_r2']:.12f}",
        f"- 保留集相對 MAE：{d['holdout_relative_mae']:.3e}",
        f"- 完整資料重建 MAE：{d['full_fit_mae']:.3e}",
        "",
        "## 二、排序與生成摘要",
        "",
        "| 條件 | R@1 | R@5 | MRR | 偏好主張比例 | 畫像一致率 | 畫像矛盾率 | 元資料支持率 | UCR L1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        m = summary["metrics"][condition]
        lines.append(
            f"| {LABELS[condition]} | {pct(m['recall@1'])} | {pct(m['recall@5'])} | "
            f"{m['mrr']:.4f} | {pct(m['preference_claim_ratio'])} | "
            f"{pct(m['preference_reference_alignment_rate'])} | "
            f"{pct(m['preference_reference_contradiction_rate'])} | "
            f"{pct(m['metadata_support_rate'])} | {pct(m['UCR_L1_clause'])} |"
        )

    lines.extend([
        "",
        "## 三、配對對照",
        "",
        "| 對照 | 指標 | 差異 | 95% CI | 顯著 |",
        "|---|---|---:|---:|:---:|",
    ])
    show_rank = {"recall@1", "recall@5", "mrr"}
    show_gen = {
        "preference_claim_ratio", "preference_reference_alignment_rate",
        "preference_reference_contradiction_rate", "metadata_support_rate", "UCR_L1_clause",
    }
    for row in summary["ranking_contrasts"] + summary["generation_contrasts"]:
        if row["metric"] not in show_rank | show_gen:
            continue
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['difference']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{'是' if row['significant'] else '否'} |"
        )
    lines.extend(["", "## 四、判讀界線", ""])
    for key, value in summary["method"].items():
        lines.append(f"- **{key}**：{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
