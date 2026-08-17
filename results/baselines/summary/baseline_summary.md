# Baseline Summary

## Ranking

| Model | R@1 | R@5 | R@10 | Median Rank | Mean Rank | Note |
|---|---:|---:|---:|---:|---:|---|
| random_500pool | 0.29% | 1.14% | 1.95% | 249 | 251.444 | non-parametric 500-pool baseline |
| audio_ast_similarity | 0.83% | 4.26% | 7.11% | 110 | 151.522 | non-parametric 500-pool baseline |
| video_audio_embedding_similarity | 0.17% | 0.83% | 1.81% | 264 | 258.522 | non-parametric 500-pool baseline |
| MuseChat-light | 2.88% | 11.68% | 20.33% | 37.0 | 58.308 | re-implemented baseline |
| Unified MLLM w/o LTP (exp_04) | 19.07% | 47.94% | 63.61% | 6.0 | 16.422 | already covered by ablation |
| Unified MLLM full (exp_01) | 30.65% | 66.40% | 79.88% | 3.0 | 7.949 | proposed model |

## Generation

| Model | BERTScore F1 | AB Div. | L2 Dist. | Fisher-Rao Dist. | Note |
|---|---:|---:|---:|---:|---|
| MuseChat-light GT | 0.706 | 3.158 | 0.248 | 2.085 | GT-conditioned generation |
| MuseChat-light Top-1 | 0.686 | 3.891 | 0.270 | 2.286 | end-to-end generation |
| LLaMA prompting-only Top-1 | 0.560 | 6.299 | 0.352 | 2.680 | text-only explanation baseline |
