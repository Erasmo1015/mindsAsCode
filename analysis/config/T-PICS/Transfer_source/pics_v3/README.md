# PICS v3 transfer source map

**Active runtime freeze (final ICLR Method / g5e50p30):**

`occurrence_eb_schema5_iter10_pics_v3.yaml`

Freeze companions: `schema5_occurrence_eb_freeze/`.

Reports:

- Concise: `analysis_2026Sep/Sep20_V3/others/source_selection_schema_v5/report/ICLR_METHOD_SOURCE_SELECTION_REPORT.md`
- Diffs vs prior: `…/SOURCE_MAP_DIFFERENCES_SCHEMA_V5_G5E50P30.md`

G.1 path ledger: `analysis/config/T-PICS/pics_v3/g1_job_paths_g5e50p30.tsv`  
Annotation package: `analysis_2026Sep/mem/pics_v3_g1_schema_v5/`  
Gated **active** paths (15 canonical jobs): `analysis/config/T-PICS/pics_v3/gated_job_paths_g5e50p30.tsv`  
Baseline Ours dirs: `analysis/config/T-PICS/baseline_methods/config_baselines.yaml` (must match active path ledger)  
Gated status (mean LL + **gate arm/reason** + map source): `analysis_2026Sep/Sep20_V3/others/gated_g5e50p30/STATUS.md`

Historical submit ledgers (superseded IDs kept):  
`cluster/record/2026Sep21_PICS_v3_gated_g5e50p30.tsv`,  
`cluster/record/2026Sep22_PICS_v3_gated_reminder_v1.tsv`.

## Official map: target → selected source

From `occurrence_eb_schema5_iter10_pics_v3.yaml` (Occurrence-EB cosine; six-source allowlist; self excluded). Source G.1 = `selected_source_job_id`.

| Target | Selected source | Source G.1 | Cosine |
| --- | --- | ---: | ---: |
| `1peterson2021using` | `mixed_gambles` | 265762 | 0.996 |
| `2plonsky2018when` | `1peterson2021using` | 265753 | 0.939 |
| `3frey2017cct` | `1peterson2021using` | 265753 | 0.936 |
| `4wulff2018description` | `11enkavi2019recentprobes` | 265760 | 0.981 |
| `5speekenbrink2008learning` | `3frey2017cct` | 265755 | 0.795 |
| `7hilbig2014generalized` | `11enkavi2019recentprobes` | 265760 | 0.979 |
| `10frey2017risk` | `3frey2017cct` | 265755 | 0.939 |
| `11enkavi2019recentprobes` | `4wulff2018description` | 265756 | 0.981 |
| `12badham2017deficits` | `3frey2017cct` | 265755 | 0.845 |
| `mixed_gambles` | `1peterson2021using` | 265753 | 0.996 |
| `bergert_nosofsky_2007` | `7hilbig2014generalized` | 265758 | 0.929 |
| `guan_2020_stopping` | `3frey2017cct` | 265755 | 0.953 |
| `steyvers_2009_bandit` | `3frey2017cct` | 265755 | 0.663 |
| `13schulz2020finding` | `3frey2017cct` | 265755 | 0.671 |
| `14kool2016when` | `3frey2017cct` | 265755 | 0.758 |

The map selects the transfer-arm seed only. Gate outcomes (which arm was kept) are in `STATUS.md` / each job’s `gate/gate_record.json` — transfer won on CPC18, CCT, and bergert only.

## Historical (do not overwrite; not the active runtime default)

- `occurrence_eb_schema4_iter10_pics_v3.yaml` + `schema4_occurrence_eb_freeze/` (50-person G.1, schema-v4)
- `occurrence_eb_schema4_5construct_iter10_pics_v3.yaml` (five-construct subset of schema-v4; same 15 winners as six-construct freeze)
- `../occurrence_eb_schema4_iter10.yaml` (v1)
- `../v2/occurrence_eb_schema4_iter10_sa40_v2.yaml` (preliminary v2)

`peeked_transfer_at_freeze: false`. Selector = Occurrence-EB cosine on the
six-source allowlist (schema-v5 five constructs).
