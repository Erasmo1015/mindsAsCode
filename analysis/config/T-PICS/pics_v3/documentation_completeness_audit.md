# PICS v3 documentation completeness audit (supporting)

Companion to `docs/Documentation_pics_v3.md`. Documentation-only; no runtime
claims beyond what that file states.

## Audit date

2026-09-20

## Authority

Current code under `utils/teh/pics_v3.py`, `t_pics_gated_transfer.py`,
`limited_data_*.py`, `g2_paired_packing.py`, `teh_runtime.py`, G.1 helpers in
`cluster/v2/ours/Qwen/`, plus historical `docs/Documentation.md` and
`docs/Documentation_v2.md` for disposition only.

## Planned vs present on disk

| Path | Present? |
| --- | --- |
| `Transfer_source/pics_v3/occurrence_eb_schema4_iter10_pics_v3.yaml` | no (planned) |
| `analysis_2026Sep/mem/pics_v3_g1_schema_v4/` | no (planned) |
| `pics_v3_g1` job outputs | no (not submitted) |
| Final 30k G.2 packing CSV | no (regenerate after map) |

## Ambiguities resolved for the main doc

| Ambiguity | Resolution |
| --- | --- |
| Is `--require_auto_llm_prompt` a CLI flag? | No; fail-closed via `g1_require_auto` scoped to `structure_aware_v3` |
| Is “350” total LLM calls? | Convention \(100+100+50+100\); G.3/person are per-participant schedules |
| Final source identities? | Not documented as facts; map planned only |
| v1 one-line vs snapshot prompts | v3 uses snapshots (`prompt_snapshots.py`) |
| G.3 vs person | Explore = G.3; person = separate `--n_iterations 10` |

## Files touched in this audit pass

- `analysis/config/T-PICS/docs/Documentation_pics_v3.md` (rewritten)
- `analysis/config/T-PICS/pics_v3/documentation_completeness_audit.md` (this file)
