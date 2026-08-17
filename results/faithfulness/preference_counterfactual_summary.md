# Counterfactual Preference Test

This analysis changes the natural-language user preference prompt while keeping the model checkpoint and candidate music fixed.

| Variant | n | Aligned rate | Conflict rate | Avg. ESS from original | Avg. alignment score |
|---|---:|---:|---:|---:|---:|
| cf_upbeat_electronic | 200 | 99.5% | 5.5% | 0.6380 | 6.00 |
| cf_lyrical_piano | 200 | 92.0% | 60.0% | 0.6369 | 3.94 |

## Interpretation Notes

- Higher aligned rate suggests the generated explanation reflects the counterfactual preference text.
- Higher conflict rate suggests the explanation still mentions concepts from the opposite preference.
- ESS is Jaccard distance from the original-prompt explanation; higher values indicate larger textual change.
- This is not a full reranking test because text embeddings are precomputed in the current pipeline.
