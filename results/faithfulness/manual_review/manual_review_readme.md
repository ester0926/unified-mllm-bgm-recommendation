# Manual Review Packet

Created at: 2026-06-07T00:01:42

## Purpose

These files convert existing automatic results into human-checkable materials.
They are intended to support thesis sections on failure cases, human spot-checking, and judge consistency.

## Suggested use

1. Pick 3 successful and 3 failed examples from `failure_case_candidates.md`.
2. Ask one or two annotators to complete `human_faithfulness_spotcheck_template.csv` and `human_metadata_spotcheck_template.csv`.
3. If comparing human and LLM judge, fill `human_llm_judge_consistency_template.csv` and compute agreement or Cohen's kappa.

## Outputs

- failure_case_candidates_csv: `<project_root>/checkpoints\manual_review\failure_case_candidates.csv`
- failure_case_candidates_md: `<project_root>/checkpoints\manual_review\failure_case_candidates.md`
- human_faithfulness_spotcheck_template: `<project_root>/checkpoints\manual_review\human_faithfulness_spotcheck_template.csv`
- human_metadata_spotcheck_template: `<project_root>/checkpoints\manual_review\human_metadata_spotcheck_template.csv`
- human_llm_judge_consistency_template: `<project_root>/checkpoints\manual_review\human_llm_judge_consistency_template.csv`
- readme: `<project_root>/checkpoints\manual_review\manual_review_readme.md`
