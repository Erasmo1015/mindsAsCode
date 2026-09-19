# T-PICS v2 source-selection artifacts

This directory is the **v2** Occurrence-EB freeze. It does not overwrite
`analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml`
(v1 jobs 257174–257188 / 257756).

| File | When it is written |
| --- | --- |
| `analysis/config/misc/Sep20_T-PICS_v2/config_T-PICS_schema4.yaml` | v2 G.1 path index (same role as Sep17 `config_T-PICS_schema4.yaml`). Fill as each source pop completes. |
| `occurrence_eb_schema4_iter10_sa40_v2.yaml` | After v2 G.1 (15 source pops) + schema-v4 annotation of **new** G.1 programs + Occurrence-EB refit (same formulas, six-source allowlist, no-transfer-peek) |
| `schema4_occurrence_eb_freeze/` | Same freeze package layout as v1, new paths only |

Gated T-PICS now **defaults** `--t_pics_source_config` to
`occurrence_eb_schema4_iter10_sa40_v2.yaml`. That file is intentionally
absent until G.1+EB finish. G.1 jobs do not need it.

Replay v1:

```
--t_pics_source_config analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml
--limited_data_protocol structure_aware
```

Motifs, annotation prompts, EB formulas, and the six-source allowlist are
reused unchanged. Only the source programs (v2 SA40 data/prompts) change.
