# PICS v3 transfer source map

Runtime YAML (written after G.1 + schema-v4 annotation + Occurrence-EB refit):

`occurrence_eb_schema4_iter10_pics_v3.yaml`

This directory must not overwrite v1 or preliminary-v2 maps:

- `../occurrence_eb_schema4_iter10.yaml` (v1)
- `../v2/occurrence_eb_schema4_iter10_sa40_v2.yaml` (preliminary v2)

Until the PICS v3 G.1 → annotate → Occurrence-EB sequence completes, the
runtime YAML is intentionally absent. Gated dry-runs that need a map should
wait for that freeze; G.1 source-population jobs do not require this file.
