# Top-1 Pool-Size Robustness

Experiment: `exp_01`
Prompt variant: `original`
Generation setting: `Top-1 end-to-end`

| Pool | R@1 | R@5 | R@10 | Mean Rank | BERT F1 | InfoLM L2 | Title Consistency | Manual Review |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.5838 | 0.9132 | 0.9736 | 2.3750 | 0.7615 | 0.1777 | 0.7025 | 0.2571 |
| 500 | 0.3065 | 0.6640 | 0.7988 | 7.9491 | 0.7526 | 0.2063 | 0.6071 | 0.3403 |
| 1000 | 0.2109 | 0.5241 | 0.6754 | 14.9496 | 0.7507 | 0.2160 | 0.5788 | 0.3643 |

## Delta vs 500-Pool

| Pool | ΔR@1 | ΔR@5 | ΔR@10 | ΔMean Rank | ΔBERT F1 | ΔInfoLM L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.2773 | 0.2492 | 0.1748 | -5.5741 | 0.0089 | -0.0286 |
| 500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1000 | -0.0956 | -0.1398 | -0.1234 | 7.0005 | -0.0019 | 0.0096 |

## Notes

- Ranking metrics are read from the original pool-size ranking summaries.
- Generation metrics are read from Top-1 end-to-end summaries, not GT-conditioned generation.
- For InfoLM and mean rank, lower is better; negative delta vs 500-pool therefore means improvement.
- If any field is `NA`, run the corresponding Top-1 pool wrapper first.
