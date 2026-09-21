# PICS v3 transfer source map

**Active runtime freeze (final ICLR Method / g5e50p30):**

`occurrence_eb_schema5_iter10_pics_v3.yaml`

Freeze companions: `schema5_occurrence_eb_freeze/`.

Reports:

- Concise: `analysis_2026Sep/Sep20_V3/others/source_selection_schema_v5/report/ICLR_METHOD_SOURCE_SELECTION_REPORT.md`
- Diffs vs prior: `…/SOURCE_MAP_DIFFERENCES_SCHEMA_V5_G5E50P30.md`

G.1 path ledger: `analysis/config/T-PICS/pics_v3/g1_job_paths_g5e50p30.tsv`  
Annotation package: `analysis_2026Sep/mem/pics_v3_g1_schema_v5/`  
Gated job ledger: `cluster/record/2026Sep21_PICS_v3_gated_g5e50p30.tsv`

## Historical (do not overwrite; not the active runtime default)

- `occurrence_eb_schema4_iter10_pics_v3.yaml` + `schema4_occurrence_eb_freeze/` (50-person G.1, schema-v4)
- `occurrence_eb_schema4_5construct_iter10_pics_v3.yaml` (five-construct subset of schema-v4; same 15 winners as six-construct freeze)
- `../occurrence_eb_schema4_iter10.yaml` (v1)
- `../v2/occurrence_eb_schema4_iter10_sa40_v2.yaml` (preliminary v2)

`peeked_transfer_at_freeze: false`. Selector = Occurrence-EB cosine on the
six-source allowlist (schema-v5 five constructs).
