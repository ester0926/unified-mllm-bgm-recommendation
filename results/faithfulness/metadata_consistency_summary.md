# Metadata Consistency Test

This rule-based analysis checks whether music-detail terms in generated explanations are supported by the top-1 candidate metadata/reference text.

| Exp | Music claims | Unsupported claims | Unsupported rate | Top-1 correct unsupported | Top-1 incorrect unsupported |
|---|---:|---:|---:|---:|---:|
| exp_01 | 7600 | 4834 | 63.61% | 44.24% | 72.15% |
| exp_02 | 7240 | 4591 | 63.41% | 43.0% | 69.01% |
| exp_03 | 7661 | 4843 | 63.22% | 41.36% | 73.02% |
| exp_04 | 7797 | 5159 | 66.17% | 46.19% | 70.71% |
| exp_05 | 7376 | 4637 | 62.87% | 43.39% | 70.44% |
| exp_06 | 7814 | 6118 | 78.3% | 77.32% | 78.5% |
| exp_07 | 8010 | 7375 | 92.07% | 66.67% | 92.08% |

## Interpretation Notes

- Lower unsupported rate indicates better metadata consistency.
- This is a conservative keyword-overlap proxy; it can over-flag paraphrases and under-flag unsupported generic claims.
- Use this table as first-pass evidence, then manually audit representative cases.
