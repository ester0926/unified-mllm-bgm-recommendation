# LLM-as-a-Judge Faithfulness Validation

Model: `llama3:8b`

| Task | n | valid | errors | LLM positive rate | Agreement with rule-based judge |
|---|---:|---:|---:|---:|---:|
| Feature-erasure supported claims | 120 | 120 | 0 | 95.00% | 60.83% |
| Preference aligned | 80 | 80 | 0 | 91.25% | 92.50% |
| Preference conflict | 80 | 80 | 0 | 16.25% | 75.00% |
| Metadata supported | 120 | 119 | 1 | 84.03% | 44.54% |

## Interpretation Notes

- This is an LLM-assisted validation subset, not the full deterministic analysis.
- Use agreement rates to discuss whether rule-based labels are directionally reliable.
- Low agreement indicates the corresponding rule-based metric should be treated cautiously or manually audited.
