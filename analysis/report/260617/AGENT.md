# TEH/PICS Psych-101 scalability audit (2026-06-17)

## Goal (step 1)

Inspect whether the current `teh.py` implementation uses mostly general dataset loading / program-input logic or many dataset-specific engineered branches.

## Goal (step 2)

Inspect **remaining Psych-101 train-set datasets** not in the top-10 implementation. Infer program input/output and implementation work. **Analysis only.**

## Goal (step 3)

Evaluate generalizing to binary + multi-action datasets with:

```python
choose(problem, history) -> dict[int, float]
```

Offline adapter normalizes actions to `0..K-1`. **Feasibility/design only.**

## Goal (step 4)

Evaluate **Version B — LLM-suggested parser-plan adapter** for scaling without manual per-dataset parsers:

1. LLM reads 3–10 representative raw HF rows/transcripts.
2. LLM outputs a **parser plan JSON** (extraction rules, not parsed data).
3. Fixed Python engine executes the plan → structured categorical trials.
4. Validator checks output; failures quarantined.
5. PICS runs on validated trials (population-level programs initially).
6. **No LLM at program evaluation time.** **No manual parser per dataset.** **Analysis only.**

## Scope

- Steps 1–4: analysis and design under `analysis/report/260617/`.
- **No code changes** to `teh.py`, eval, or parsers.

## Outputs

| File | Step |
|------|------|
| `AGENT.md` | — |
| `current_10_dataset_code_audit.csv` / `.md` | 1 |
| `psych101_remaining_dataset_inventory.csv` | 2 |
| `psych101_remaining_dataset_examples.json` | 2 |
| `psych101_remaining_dataset_summary.md` | 2 |
| `generalized_categorical_api_feasibility.md` | 3 |
| `generalized_categorical_api_dataset_fit.csv` | 3 |
| `generalized_categorical_adapter_schema.json` | 3 |
| `generalized_categorical_eval_plan.md` | 3 |
| `llm_parser_plan_feasibility.md` | 4 |
| `llm_parser_plan_dataset_fit.csv` | 4 |
| `llm_parser_plan_schema.json` | 4 |
| `llm_parser_plan_execution_options.md` | 4 |
| `psych101_76_reconciled_coverage.csv` | 4 |
| `psych101_76_reconciled_coverage_summary.md` | 4 |

## Method

1. Enumerate all 76 Psych-101 train `experiment_id`s from HF.
2. Reconcile coverage counts across steps 2–4.
3. Classify parser-plan feasibility (`easy_plan` / `medium_plan` / `hard_plan` / `not_suitable_for_plan`).
4. Design parser plan schema, execution engine options, population-level checks.
