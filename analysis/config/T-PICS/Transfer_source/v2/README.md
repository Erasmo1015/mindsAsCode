# T-PICS v2 source-selection artifacts

This directory is the **v2** Occurrence-EB freeze. It does not overwrite
`analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml`
(v1 jobs 257174–257188 / 257756).

| File | Role |
| --- | --- |
| `occurrence_eb_schema4_iter10_sa40_v2.yaml` | **Runtime** frozen source map (Occurrence-EB on SA40 v2 G.1) |
| `schema4_occurrence_eb_freeze/` | Optional machine-readable freeze package (if present) |

G.1 index: `analysis/config/misc/Sep20_T-PICS_v2/config_T-PICS_schema4.yaml`  
Fit/report: `analysis_2026Sep/Sep20_V2/others/source_selection/`  
Annotations: `analysis_2026Sep/mem/t_pics_g1_sa40_v2_schema_v4/annotations/`

Motifs, annotation prompts, EB formulas, and the six-source allowlist are
reused unchanged from v1. Only the source programs (v2 SA40 data/prompts) change.

Replay v1:

```
--t_pics_source_config analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml
--limited_data_protocol structure_aware
```
