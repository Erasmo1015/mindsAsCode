# PICS v3 documentation completeness audit (supporting)

Companion to `docs/Documentation_pics_v3.md`. Documentation-only; no runtime
claims beyond what that file states.

## Audit date

2026-09-23 (updated: active gated paths → reminder-v1 for guan/Schulz/Kool +
centaur-gap-v1 for Speekenbrink/Badham/Steyvers; CPC18 remains `pics_v3/271238`)

## Authority

Current code under `utils/teh/pics_v3.py`, `t_pics_gated_transfer.py`,
`limited_data_*.py`, `g2_paired_packing.py`, `teh_runtime.py`, G.1 helpers in
`cluster/v2/ours/Qwen/`, gated submitter `cluster/v3/ours/main/submit_gated.sh`,
plus historical `docs/Documentation.md` and `docs/Documentation_v2.md` for
disposition only.

## Present on disk (final ICLR Method)

| Path | Present? |
| --- | --- |
| `Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml` | **yes** (active runtime) |
| `Transfer_source/pics_v3/schema5_occurrence_eb_freeze/` | **yes** |
| `analysis_2026Sep/mem/pics_v3_g1_schema_v5/` | **yes** (1414 programs) |
| `pics_v3/g1_job_paths_g5e50p30.tsv` | **yes** (jobs 265753–265767) |
| `cluster/record/2026Sep21_PICS_v3_gated_g5e50p30.tsv` | **yes** (historical submit ledger; superseded actives kept) |
| `cluster/record/2026Sep22_PICS_v3_gated_reminder_v1.tsv` | **yes** (historical; six keyed-reminder submits `287666`–`287671`) |
| `pics_v3/gated_job_paths_g5e50p30.tsv` | **yes** (15 active: 9×`pics_v3` + 3×`pics_v3_reminder_v1` + 3×`pics_v3_centaur_gap_v1`) |
| `baseline_methods/config_baselines.yaml` Ours paths | **yes** (aligned to `gated_job_paths_g5e50p30.tsv`) |
| `analysis_2026Sep/Sep20_V3/others/gated_g5e50p30/STATUS.md` | **yes** (arm / reason / map source / mean LL) |
| `pics_v3` / `pics_v3_reminder_v1` / `pics_v3_centaur_gap_v1` job outputs | **yes** (g5e50p30 gated complete) |

## Historical (kept; not active runtime)

| Path | Present? |
| --- | --- |
| `occurrence_eb_schema4_iter10_pics_v3.yaml` | yes (50-person schema-v4) |
| `occurrence_eb_schema4_5construct_iter10_pics_v3.yaml` | yes |
| `analysis_2026Sep/mem/pics_v3_g1_schema_v4/` | yes |
| `pics_v3/g1_job_paths.tsv` | yes (259052–259197) |

## Ambiguities resolved for the main doc

| Ambiguity | Resolution |
| --- | --- |
| Is `--require_auto_llm_prompt` a CLI flag? | No; fail-closed via `g1_require_auto` scoped to `structure_aware_v3` |
| Active runtime source map | schema5 / g5e50p30 (`occurrence_eb_schema5_iter10_pics_v3.yaml`) |
| Schema-v4 freezes | historical only; do not overwrite |
