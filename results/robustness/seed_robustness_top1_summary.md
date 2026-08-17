# Top-1 Random Seed Robustness

Experiment: `exp_01`
Pool size: `500`
Prompt variant: `original`
Seeds: `42, 12345, 987654`
Generation setting: `Top-1 end-to-end`

## Per-Seed Results

| Seed | R@1 | R@5 | R@10 | Mean Rank | BERT F1 | InfoLM L2 | Title Consistency | Manual Review |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.3199 | 0.6647 | 0.8005 | 7.9658 | 0.7538 | 0.2041 | 0.6133 | 0.3351 |
| 12345 | 0.3132 | 0.6647 | 0.8059 | 7.9584 | 0.7532 | 0.2048 | 0.6233 | 0.3241 |
| 987654 | 0.3082 | 0.6604 | 0.7986 | 7.9753 | 0.7538 | 0.2047 | 0.6200 | 0.3303 |

## Stability Summary

| Metric | Mean | Min | Max | Range | Relative Range |
|---|---:|---:|---:|---:|---:|
| recall@1 | 0.3138 | 0.3082 | 0.3199 | 0.0117 | 0.0371 |
| recall@5 | 0.6633 | 0.6604 | 0.6647 | 0.0043 | 0.0065 |
| recall@10 | 0.8017 | 0.7986 | 0.8059 | 0.0074 | 0.0092 |
| mean_rank | 7.9665 | 7.9584 | 7.9753 | 0.0169 | 0.0021 |
| bertscore_f1_top1_all | 0.7536 | 0.7532 | 0.7538 | 0.0006 | 0.0008 |
| infolm_l2_top1_all | 0.2046 | 0.2041 | 0.2048 | 0.0007 | 0.0034 |
| title_consistency_rate | 0.6189 | 0.6133 | 0.6233 | 0.0099 | 0.0161 |
| needs_manual_review_rate | 0.3298 | 0.3241 | 0.3351 | 0.0109 | 0.0332 |

## Notes

- This analysis uses the revised non-adjacent seeds `42`, `12345`, and `987654`.
- Ranking metrics are recomputed from each seed-specific candidate pool.
- Generation metrics are computed from the Top-1 music selected under each seed-specific ranking result.
- For mean rank and InfoLM metrics, lower is better.
- The earlier adjacent seeds `20260315/20260316/20260317` should be treated as historical pilot runs, not the formal seed robustness result.
