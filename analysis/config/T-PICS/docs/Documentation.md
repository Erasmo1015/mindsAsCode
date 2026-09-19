# ICLR T-PICS (gated dual-arm transfer) — technical record

This file is the authoritative description of the **final ICLR T-PICS method** as implemented. Every statement below is taken from current code, cluster defaults, or the frozen source-selection YAML. It does not retune Occurrence-EB, recompute source ranking, or describe leftover `--t_pics` / 6×6 KIND jobs.

**Method name in this repository:** gated T-PICS / T-PICS / PICS / tpics / “our method” = `teh.py --t_pics_gated_transfer` unless an older pipeline is named explicitly.

| Role | Path |
| --- | --- |
| Production submitter | `cluster/2026Sep18_T_PICS_gated/submit_gated.sh` |
| Worker | `cluster/2026Sep18_T_PICS_gated/job_gated_l40s.sh` |
| Shared gated helpers | `cluster/2026Sep18_T_PICS_gated/_gated_common.sh` |
| Shared T-PICS helpers | `cluster/2026Sep17_T_PICS/_t_pics_common.sh` |
| Pipeline constants / gate / argv | `utils/teh/t_pics_gated_transfer.py` |
| Runtime (G.1 live helper, G.2 arms, G.3/person) | `teh.py` |
| Frozen source map | `analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml` |
| Schema-v4 / Occurrence-EB freeze manifest | `analysis/config/T-PICS/Transfer_source/schema4_occurrence_eb_freeze/` |
| Reporting-only W&B | `utils/teh/t_pics_gated_wandb.py` |
| Dataset registry / EMNLP ranges | `analysis/config/teh_datasets.yaml`, `utils/teh/teh_datasets.py` |
| SA40 protocol | `utils/teh/limited_data_protocol.py` |

---

## 1. Phase-by-phase overview

One Slurm job per **target** dataset runs a single `python teh.py --t_pics_gated_transfer` process:

1. **G.1 (source population, 10 iterations).** Default mode **reuses** the frozen 10-iteration source-population rank-1 already recorded in the YAML (jobs **257174–257188**). It does not rerun those source pops. Optional `--t_pics_gated_independent` trains a **new** 10-iteration source pop on the YAML-selected source dataset only.
2. **G.2 (two matched 5-iteration target populations).** Control and transfer start from the **same** target automatic prompt, target seed, target people, and target SA40 splits. They run **sequentially**. Control has **no** source program and **no** source examples. Transfer injects the selected **source rank-1** plus **one** injected source train+val trial into the instruction suffix.
3. **Observed-data gate.** Each arm’s target-generated rank-1 is scored as the **equal-person mean** of per-person `train_val` log-likelihood on that person’s observed train+val. Transfer is kept only if that mean is **strictly greater** than control (tolerance \(10^{-12}\); all ties/failures keep control). Target **test is never used**.
4. **G.3 (50 explore candidates).** All 50 candidates are generated from the **retained target rank-1 only** (`--explore_population_top_k 1`). There is **no** source suffix. The winner arm’s **full elite pool** is loaded as the person initial pool via `--initial_pool_dir` / in-process handoff.
5. **Participant evolution (10 iterations).** Each target person starts from that full retained pool plus decaying seed-parented fresh candidates. Gated person prompts include that person’s train **and** val examples (capped). Test never enters prompts.

Default production does **not** set `INDEPENDENT=1`.

---

## 2. Terminology

| Term | Meaning |
| --- | --- |
| **Target** | The dataset this job is fitting (`--dataset`). |
| **Source** | The other-task dataset nominated by the frozen Occurrence-EB map for this target. Never equal to the target (self is excluded at freeze time). |
| **Source rank-1** | `global_phase/best_program.py` of the source G.1 population. In default mode this is the frozen YAML path. In independent mode it is the live `run_root/source_population/global_phase/best_program.py`. Used **only** as extra G.2 **transfer-arm context**, never as a G.3 parent and never as a person-evolution parent. |
| **Injected source trial** | The **one** source train+val trial serialized into the G.2 transfer suffix (`one_example_trial_text`). Sampled from the first ≤8 TV trials of the **first person in the source EMNLP range** (`emnlp_ordinal_range` → `pids[start]`). Not eight trials. For Wulff that person is ordinal 1290, not `valid_participant_ids[0]`. |
| **Target rank-1** | `global_phase/best_program.py` of a G.2 arm (control or transfer), ranked by pooled `train_val` on the target. After the gate, the **winner** arm’s rank-1 is the **sole G.3 explore parent**. |
| **Retained elite pool** | Winner arm’s `global_phase/global_elite_pool/` (up to 50 programs). Loaded in full into participant evolution. G.3 does **not** sample explore parents from the rest of this pool. |
| **G.1** | Source-dataset population evolution (10 global iterations, no person evolution, no cross-task suffix). |
| **G.2** | Dual target-dataset population evolution (5 global iterations each arm). |
| **G.3** | Per-person explore phase: 50 candidates from retained target rank-1. |
| **Participant / person evolution** | Per-person 10-iteration TEH evolution after G.3, using the full retained pool. |
| **Observed** | Train ∪ val after SA40. Test is held out. |
| **`train_val`** | Trial-count-weighted mean of train and val average log-likelihood = mean loglik on the observed union. |
| **Leftover `--t_pics`** | A different live-source CLI. **Not** this method. Combining it with `--t_pics_gated_transfer` is a hard error. |

Three programs that must not be confused:

1. **Source rank-1** → extra **context** in the G.2 **transfer** instruction suffix.
2. **Retained target rank-1** → the **actual sole parent** of all 50 G.3 explore requests.
3. **Full retained target pool** → person-evolution **initial elite**, not G.3 parents.

---

## 3. Default gated mode vs optional independent mode

| | Default `--t_pics_gated_transfer` | Optional `--t_pics_gated_independent` |
| --- | --- | --- |
| Cluster | `INDEPENDENT` unset or `0` | `INDEPENDENT=1` |
| KIND / output folder | `t_pics_gated` | `t_pics_gated_independent` |
| YAML use | Selected source **identity** + frozen G.1 **rank-1 file** | Selected source **identity only** (`require_files=False`) |
| G.1 | Reuse jobs 257174–257188 rank-1 | Live 10-iter source pop under this job |
| G.1 people | Frozen G.1 used source EMNLP ranges | Live G.1 uses source EMNLP ranges |
| G.2 transfer suffix | Frozen rank-1 **program** + **one** trial from the first source EMNLP person (`pids[start]`) | Live rank-1 **program** + the **same** one-trial EMNLP-first-person rule |
| G.2 / gate / G.3 / person knobs | Identical | Identical |
| When to use independent | **Not** the main ICLR experiments | Sensitivity: retrain G.1 instead of reusing frozen programs |

Independent **requires** `--t_pics_gated_transfer`. It cannot be combined with leftover `--t_pics` / `--t_pics_source`.

---

## 4. Frozen source-selection YAML (Occurrence-EB)

**Path:** `analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml`  
**Loader:** `load_frozen_transfer_config` / `selected_source_for_target` in `utils/teh/t_pics_gated_transfer.py`  
**Constant:** `FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG`  
**Full freeze provenance (motifs, annotation, EB equations, artifacts, policy):** [Frozen schema-v4 annotation and Occurrence-EB source selection](#frozen-schema-v4-annotation-and-occurrence-eb-source-selection)  
**Machine-readable freeze:** `analysis/config/T-PICS/Transfer_source/schema4_occurrence_eb_freeze/`

The map is **frozen**. The gated pipeline does not recompute cosine ranks, retune the selector, or peek at transfer outcomes (`peeked_transfer_at_freeze: false`). Fit/eval job recorded in the YAML: **257756**.

Selector (copied from the YAML; not re-derived here):

- `name`: `occurrence_eb_schema4_iter10`
- `method`: hierarchical motif occurrence-EB cosine
- equation: \(\mathrm{logit}(p_{d,j,e})=\alpha_e+b_{d,e}\), \(b\sim N(0,\tau_e^2)\)
- selection: argmax cosine on the **six-source allowlist**, self excluded, lexicographic source-id tie-break
- iteration prefix: source-population iterations **1–10** (schema v4)
- motifs: history, value, probability_used, feedback, learning, explicit_risk_mechanism

**Six-source allowlist:** `1peterson2021using`, `3frey2017cct`, `4wulff2018description`, `7hilbig2014generalized`, `11enkavi2019recentprobes`, `mixed_gambles`.

### 4.1 Target → selected source (identity used by both modes)

| Target | Selected source | Cosine | Frozen G.1 job | Frozen rank-1 |
| --- | --- | --- | --- | --- |
| `10frey2017risk` | `3frey2017cct` | 0.8846 | 257176 | `.../3frey2017cct/t_pics_source_pop10/job_257176/global_phase/best_program.py` |
| `11enkavi2019recentprobes` | `1peterson2021using` | 0.9650 | 257174 | `.../1peterson2021using/t_pics_source_pop10/job_257174/global_phase/best_program.py` |
| `12badham2017deficits` | `11enkavi2019recentprobes` | 0.8369 | 257181 | `.../11enkavi2019recentprobes/t_pics_source_pop10/job_257181/global_phase/best_program.py` |
| `13schulz2020finding` | `11enkavi2019recentprobes` | 0.8330 | 257181 | same Enkavi rank-1 |
| `14kool2016when` | `11enkavi2019recentprobes` | 0.8937 | 257181 | same Enkavi rank-1 |
| `1peterson2021using` | `11enkavi2019recentprobes` | 0.9650 | 257181 | same Enkavi rank-1 |
| `2plonsky2018when` | `1peterson2021using` | 0.9997 | 257174 | Choice13k rank-1 |
| `3frey2017cct` | `11enkavi2019recentprobes` | 0.9560 | 257181 | Enkavi rank-1 |
| `4wulff2018description` | `mixed_gambles` | 0.9968 | 257183 | `.../mixed_gambles/t_pics_source_pop10/job_257183/global_phase/best_program.py` |
| `5speekenbrink2008learning` | `1peterson2021using` | 0.9159 | 257174 | Choice13k rank-1 |
| `7hilbig2014generalized` | `4wulff2018description` | 0.9181 | 257177 | `.../4wulff2018description/t_pics_source_pop10/job_257177/global_phase/best_program.py` |
| `bergert_nosofsky_2007` | `7hilbig2014generalized` | 0.9664 | 257179 | Hilbig rank-1 |
| `guan_2020_stopping` | `3frey2017cct` | 0.9799 | 257176 | Frey CCT rank-1 |
| `mixed_gambles` | `4wulff2018description` | 0.9968 | 257177 | Wulff rank-1 |
| `steyvers_2009_bandit` | `11enkavi2019recentprobes` | 0.8146 | 257181 | Enkavi rank-1 |

Default mode **opens** those rank-1 files (`validate_frozen_transfer_config(..., require_files=True)` plus `program_has_valid_choose`). Independent mode parses the YAML for `selected_source` only and does **not** require those files to exist.

Supporting science artifacts (not re-read by `teh.py` at runtime):  
`analysis_2026Sep/mem/t_pics_source_pop10_schema_v4/source_selection_v4/` (`FROZEN_SPEC.json`, `occurrence_signatures.csv`, `frozen_ranks_15targets_occurrence.csv`, `MEETING_REPORT_OCCURRENCE_EB.md`).

---

## Frozen schema-v4 annotation and Occurrence-EB source selection

This section is the **authoritative freeze record** for how `occurrence_eb_schema4_iter10.yaml` was produced. It does not retune annotations, source populations, or the selector. Machine-readable companion: `analysis/config/T-PICS/Transfer_source/schema4_occurrence_eb_freeze/freeze_manifest.json`.

### Future-development policy

1. The frozen schema-v4 source-selection files and artifacts **must remain unchanged** because they define the main ICLR method.
2. Future participant-program annotation or MEM changes **must** use a **new versioned directory/module** and **new output paths**.
3. If existing logic is reused, **copy or import** it into that new analysis version **without overwriting** frozen source-selection artifacts.
4. Any future taxonomy or model change must receive a **new schema/model name** and must **not** silently regenerate `occurrence_eb_schema4_iter10.yaml`.

### A. Finalized six schema-v4 motifs (exact definitions)

Authoritative module: `utils/mem/schema_population_motif_v4.py`  
`SCHEMA_VERSION = 4`, `PROMPT_VERSION = "population_transition_v4_2"`, `ANNOTATION_KIND = "population_program_motif_transition"`.

| Motif | Definition (verbatim from `BEHAVIORAL_MOTIF_DEFINITIONS_V4`) |
| --- | --- |
| `history` | Program explicitly uses earlier choices, outcomes, trials, streaks, recency, counts, or a supplied history/trial buffer to change the current decision. Unused history parameters do not count. |
| `value` | Program computes or compares option attractiveness, utility, expected payoff, benefits/costs, or a general option score that drives choice. Constant/random choice without option scoring does not count. |
| `probability_used` | Program explicitly reads or uses probability/likelihood/odds/uncertainty fields from the problem, including ordinary linear EV terms such as p*x. Empirical success rates built only from feedback history are feedback/learning, not probability_used. Unused probability fields do not count. |
| `feedback` | Program directly uses observed reward, correctness, success/failure, or outcome feedback to influence a later choice. Static current-trial payoffs are value, not feedback. Feedback without an update rule is NOT learning. |
| `learning` | Program updates or reconstructs an internal belief, preference, option estimate, or decision rule across trials from experience (running means, Bayesian update, Q-like counts, parameter adaptation). Merely re-reading the last choice/reward without an update rule is history/feedback, not learning. |
| `explicit_risk_mechanism` | Beyond merely reading probability, the program explicitly models risk/uncertainty (nonlinear probability weighting, variance/downside, loss aversion multipliers, risk penalty/bonus, or other uncertainty-sensitive transforms). Linear EV alone is probability_used only. A variable named risk_aversion that only scales a non-probability state (e.g. pump count) does not count. Dataset subject matter alone does not count. |

Hard distinctions enforced in the annotator system prompt: linear \(p\cdot x\) is `probability_used` only; learning requires an across-trial update rule.

### B. Annotation outputs, transitions, applicability, validation, resume, lineage

**LLM-required fields** (`REQUIRED_LLM_FIELDS`): `candidate_id`, `reference_motif_state`, `modified_motifs`, `motif_details`, `confidence`.

**Per-motif `motif_details` row:** `motif`, `presence` (bool), `applicability`, `confidence` \(\in[0,1]\), `rationale` (non-empty string), `code_evidence` (list of strings). Exactly one detail row per allowed motif.

**Applicability values:** `applicable` | `structural_na` | `not_applicable`  
- `structural_na`: schema lacks fields needed for the motif (e.g. no probability fields; no outcome feedback channel).  
- Presence + `structural_na` is rejected.

**Derived (not trusted from a parallel LLM list):** `candidate_motif_state` / `program_motif_state` from `motif_details.presence`; then `added_motifs` / `removed_motifs` / `modified_motifs` via `derive_directional_motifs` (modified ⊆ intersection of reference and candidate states).

**Validation rules** (`validate_program_motif_response`): JSON array; expected `candidate_id`s present exactly once; motif vocabulary closed; complete motif_details; confidence ranges; reject missing/malformed/truncated JSON. Guided JSON schema avoids `minimum`/`maximum` for xgrammar compatibility.

**Resume key:** `dataset|run_id|iteration|candidate|parent` (`schema_population_motif_v4.resume_key`). Append-only skip of successful keys.

**Lineage / reference rules** (`utils/mem/reconstruct_global_pop_lineage.py`; these G.1 runs had **no** `global_phase/mem_trace.jsonl`):

- `source=fresh` → parent `global_baseline` (**exact** `seed_baseline`).
- `source=normal` → iteration-start pool-best: **exact** `best_prompted_parent` only when generation guarantees elite index 0 is prompted (`best_k >= 1` under `--sample_parents --sampled_parents_decay`); else **`pool_best_proxy`** (`reference_is_exact=false`).
- Primary fitness-effect MEM uses **exact** rows only; Occurrence-EB uses motif **presence** on all 1385 annotated programs (does not require ΔF).

Synthetic lineage trees: `analysis_2026Sep/mem/t_pics_source_pop10_schema_v4/lineage/`.

### C. Annotation prompt / schema / modules / scripts

| Piece | Path |
| --- | --- |
| Motif schema + defs + guided schema + validators | `utils/mem/schema_population_motif_v4.py` |
| Annotator CLI (system prompt `_SYSTEM_V4`, schema_version=4) | `analysis/mem/annotate_population_programs.py` |
| Lineage reconstruction | `utils/mem/reconstruct_global_pop_lineage.py` |
| Program / shard manifests | `analysis_2026Sep/mem/t_pics_source_pop10_schema_v4/build_manifest.py` → `program_manifest_all.json`, `manifest_shard_XX.json`, `shard_plan.json` |
| G.1 run index | `analysis/config/misc/Sep17_T-PICS/config_T-PICS_schema4.yaml` |
| Cluster README / submit / workers | `cluster/2026Sep18_T_PICS_PopAnnot_v4/` (`submit_pop_annot_v4_l40s.sh`, `job_pop_annot_v4_l40s.sh`, `job_pop_annot_v4_h100nvl.sh`, `job_pop_annot_v4_dataset.sh`, `submit_pop_annot_v4_rest.sh`, `submit_pop_annot_v4_xfermap.sh`) |

Prompt identity: **`population_transition_v4_2`**. Default v4 `max_tokens=8192`; batch failure → singleton retry; exit **75** → bounded vLLM restart.

### D. Fifteen G.1 source populations and 1,385 annotated programs

All runs: KIND `t_pics_source_pop10`, iterations **1–10**, jobs **257174–257188**, under  
`generated_outputs/psych101_train/teh/<dataset>/t_pics_source_pop10/job_<id>/`.

| Dataset | Job | `n_annotated_programs` | Run dir |
| --- | ---: | ---: | --- |
| `1peterson2021using` | 257174 | 86 | `.../1peterson2021using/t_pics_source_pop10/job_257174` |
| `2plonsky2018when` | 257175 | 81 | `.../2plonsky2018when/t_pics_source_pop10/job_257175` |
| `3frey2017cct` | 257176 | 84 | `.../3frey2017cct/t_pics_source_pop10/job_257176` |
| `4wulff2018description` | 257177 | 98 | `.../4wulff2018description/t_pics_source_pop10/job_257177` |
| `5speekenbrink2008learning` | 257178 | 90 | `.../5speekenbrink2008learning/t_pics_source_pop10/job_257178` |
| `7hilbig2014generalized` | 257179 | 100 | `.../7hilbig2014generalized/t_pics_source_pop10/job_257179` |
| `10frey2017risk` | 257180 | 94 | `.../10frey2017risk/t_pics_source_pop10/job_257180` |
| `11enkavi2019recentprobes` | 257181 | 100 | `.../11enkavi2019recentprobes/t_pics_source_pop10/job_257181` |
| `12badham2017deficits` | 257182 | 83 | `.../12badham2017deficits/t_pics_source_pop10/job_257182` |
| `mixed_gambles` | 257183 | 100 | `.../mixed_gambles/t_pics_source_pop10/job_257183` |
| `bergert_nosofsky_2007` | 257184 | 100 | `.../bergert_nosofsky_2007/t_pics_source_pop10/job_257184` |
| `guan_2020_stopping` | 257185 | 98 | `.../guan_2020_stopping/t_pics_source_pop10/job_257185` |
| `steyvers_2009_bandit` | 257186 | 93 | `.../steyvers_2009_bandit/t_pics_source_pop10/job_257186` |
| `13schulz2020finding` | 257187 | 96 | `.../13schulz2020finding/t_pics_source_pop10/job_257187` |
| `14kool2016when` | 257188 | 82 | `.../14kool2016when/t_pics_source_pop10/job_257188` |

**Total annotated programs: 1,385** (`program_manifest_all.json` / `panel_stats.json`). Rank-1 paths are each run’s `global_phase/best_program.py` (also listed in the YAML and freeze manifest).

### E. Annotation shards, submits, merged locations

Workload: 1385 programs → **8 shards** (largest-with-smallest pairing; max 2 datasets/shard). See `cluster/2026Sep18_T_PICS_PopAnnot_v4/README.md` and `shard_plan.json`.

| Location | Role |
| --- | --- |
| `.../annotations/shard_XX/annotations_population_v4.jsonl` | Per-shard append-only outputs (897 lines on disk across shards) |
| `.../annotations/by_dataset/<ds>/annotations_population_v4.jsonl` | Preferred merge layer for 14 datasets (1285 keys) |
| Union in `load_annotations_union(..., prefer_by_dataset=True)` | **1385** keys: by_dataset preferred; **100 Hilbig keys only in shards** (`7hilbig2014generalized` has no `by_dataset/` tree) |

Primary submit: `bash cluster/2026Sep18_T_PICS_PopAnnot_v4/submit_pop_annot_v4_l40s.sh --submit` (shards 01–04 L40S TP=2; 05–08 H100 NVL). Remaining/xfermap fills: `submit_pop_annot_v4_rest.sh`, `submit_pop_annot_v4_xfermap.sh` + `job_pop_annot_v4_dataset.sh`.

### F. Occurrence-EB model (ICLR selector)

Implemented in `analysis/mem/pop_v4_source_selection/run_source_selection_v4.py` (`fit_occurrence_eb`, `select_sources`). Frozen settings also recorded in `FROZEN_SPEC.json` / YAML `selector:`.

For each motif \(e\) and dataset \(d\), with \(n_d\) annotated programs and \(k_{d,e}\) present:

1. **Jeffreys smoothing:** \(\tilde p_{d,e}=(k_{d,e}+0.5)/(n_d+1)\); overall \(\alpha_e=\mathrm{logit}((k_{\cdot,e}+0.5)/(n_{\cdot}+1))\).
2. **Sampling variance on logit scale:** \(1/(n_d \tilde p_{d,e}(1-\tilde p_{d,e}))\).
3. **MoM \(\tau_e^2\):** \(\max(0, \mathrm{Var}_d(\mathrm{logit}\,\tilde p_{d,e}) - \overline{\mathrm{samp\_var}})\).
4. **EB shrink:** \(b_{d,e}\) / posterior mean toward \(\alpha_e\) with weight \(\tau^2/(\tau^2+v_{d,e})\) (boundary \(\tau^2=0\) → full shrink to \(\alpha_e\)).
5. **Signature coordinate:** \(\mathrm{invlogit}(\alpha_e + \hat b_{d,e})\) → 6-vector per dataset (`occurrence_signatures.csv`).
6. **Ranking:** among **six-source allowlist** only; **exclude self**; **argmax cosine**; ties → **lexicographic source id** (`max_cosine_then_lexicographic_source_id`).

**SIX allowlist:** `1peterson2021using`, `3frey2017cct`, `4wulff2018description`, `7hilbig2014generalized`, `11enkavi2019recentprobes`, `mixed_gambles`.

Job **257756** also fit fitness-effect MixedLM and combined signatures for comparison; **only Occurrence-EB** was frozen into the runtime YAML.

### G. Fit/eval scripts, job 257756, artifacts, no-transfer freeze

| Item | Path / value |
| --- | --- |
| Fit/eval script | `analysis/mem/pop_v4_source_selection/run_source_selection_v4.py` |
| CPU submit | `cluster/2026Sep18_T_PICS_SourceSelect_v4/submit_source_select_v4_cpu.sh` → `job_source_select_v4_cpu.sh` |
| Slurm job | **257756** |
| Output root | `analysis_2026Sep/mem/t_pics_source_pop10_schema_v4/source_selection_v4/` |
| Freeze flag | `peeked_transfer_at_freeze: false` in `FROZEN_SPEC.json` and YAML |
| Transfer GT | Loaded **only** for post-freeze 6×6 eval: `generated_outputs_transfer_old/teh_transfer/run_260616_235649/summary_csv/single_transfer/matrix_improve_test_loglik.csv` |

**Authoritative Occurrence-EB inputs:** annotation union (1385) + G.1 run dirs from `config_T-PICS_schema4.yaml` + manifests.  
**Authoritative Occurrence-EB outputs:** `analysis_panel.csv`, `occurrence_*.csv/json`, `frozen_ranks_15targets_occurrence.csv`, `frozen_ranks_6x6_occurrence.csv`, `FROZEN_SPEC.json`, `SOURCE_SELECTION_V4_REPORT.md`, `meeting_report/MEETING_REPORT_OCCURRENCE_EB.md`, eval tables.  
**Runtime map:** `occurrence_eb_schema4_iter10.yaml` was assembled from `frozen_ranks_15targets_occurrence.csv` + G.1 rank-1 paths (commit `9f066f14`); cosine/source identities match that CSV exactly.

Confirmation: transfer outcomes were **not** used when freezing signatures or selecting sources (`peeked_transfer_at_freeze: false`; transfer matrix path used only inside `evaluate_6x6` after freeze).

### H. Relationship to later participant-level MEM analysis

- This freeze is **population-program** annotation on G.1 global candidates (schema **v4** / `population_transition_v4_2`) used solely to build the **source map**.
- Participant-level / edit-level MEM elsewhere in the repo (schema v2/v3 modules, person `mem_trace`, etc.) is a **separate analysis track**.
- Shared motif *names* must not be silently redefined: any participant MEM taxonomy change needs a **new versioned schema** and **new output paths**, leaving this population Occurrence-EB stack untouched.
- Fitness-effect MEM on population transitions was computed in job 257756 for method comparison; it is **not** the ICLR source selector and must not overwrite Occurrence-EB artifacts.

### I. Freeze manifest

`analysis/config/T-PICS/Transfer_source/schema4_occurrence_eb_freeze/freeze_manifest.json` lists each authoritative file with repository path, SHA256, role, Git commit (when tracked), and `runtime_required`. README in that directory points back here.

---

## 5. G.1 source population

Constants: `SOURCE_POPULATION_ITERS = 10` in `utils/teh/t_pics_gated_transfer.py`.

### 5.1 What default mode reuses

Default G.2 transfer reads `entry.rank1_program` from the YAML. It does **not** call `_run_t_pics_source_population`. Completed source pops live under:

`generated_outputs/psych101_train/teh/<source>/t_pics_source_pop10/job_<257174–257188>/`

Those runs were 10-iteration source populations with `--explore_candidates 0` (no G.3/person on the source). Default gated jobs must not rewrite them.

### 5.2 What independent mode trains

`_ensure_gated_independent_source_population` → `_run_t_pics_source_population`:

- Dataset = YAML `selected_source`
- Participant range = **source** EMNLP ordinals from `emnlp_ordinal_range(source_dataset)` (`analysis/config/teh_datasets.yaml`), **not** the target job’s range
- Automatic prompt = `setup_teh_run_prompts` on the source, `require_auto_llm_prompt=True`, `llm_decoding_seed=split_seed+90_000`
- Seed = `default_seed_path(source)` (`categorical_uniform.py` for Steyvers/Schulz; else `choices13k.py`)
- SA40 knobs copied from the target command (`structure_aware`, `limited_train_val=40`, `split_ratio=0.6`, `split_seed=0`) but applied to **source** people/trials
- `n_iterations=10` (not G.2’s 5)
- `prompt_suffix=None` (no cross-task block)
- Mixed-gambles source forces `filter_mixed_gambles=True`
- Output: `run_root/source_population/`
- Resume: skip only if `population_arm_is_complete(..., expected_global_iters=10)` in **that** live directory

---

## 6. G.2 target arms

Implemented in `_run_t_pics_gated_population_arms`. Logical argvs: `build_control_argv` / `build_transfer_argv` (documentation / `INTENDED_ARGV.txt`; production runs both arms in one process).

### 6.1 Shared settings

| Knob | Production value | Where |
| --- | --- | --- |
| Target people | EMNLP range for **this target** | `t_pics_resolve_emnlp_range "${DATASET}"` |
| Target seed | `t_pics_seed_path(DATASET)` | worker; copied into `prompts/seed_program.py` |
| Automatic prompt | One target `setup_teh_run_prompts` under `run_root/prompts/` | `require_auto_llm_prompt=True`; `prefer_auto_llm_prompt` |
| Prompt sharing | Same `run_prompts_dir` for both arms | `_gated_global_phase_kwargs` |
| Global iters | 5 | `--global_iters 5` / `DEFAULT_GLOBAL_ITERS` |
| Candidates / iter | 10 | `--n_candidates 10` |
| Fresh max | 10, decayed | `--fresh_n_candidates 10 --sampled_parents_decay` |
| Parent sample size | 8 | `--sample_size 8 --sample_parents` |
| Elite cap | 50 | `--elite_pool_size 50` |
| Selection | `train_val` | `--evolution_selection_score train_val` |
| SA40 | structure-aware 40 | `--limited_data_protocol structure_aware --limited_train_val 40` |
| Split | within-person, ratio 0.6, seed 0 | `--split_mode within_participant --split_ratio 0.6 --split_seed 0` |
| LLM decoding seed | `_phase_llm_decoding_seed_base(split_seed, iteration, …)` **without arm id** | matched formula across arms; prompts still differ on transfer |
| Refinement | off | `--no-refinement_phase` |
| Error feedback | `--max_error_prompt_chars 0 --error-feedback-mode legacy` | no error section in prompts |
| MDL | 0 | `--mdl_lambda 0` |
| Order | control **then** transfer | sequential in one process |

G.2 fitness/ranking is computed on **pooled** target train and val trials across all people in the job (`run_global_evolution_phase`). Pool rank-1 is `best_program.py`.

### 6.2 Control vs transfer (the only scientific difference)

| | Control | Transfer |
| --- | --- | --- |
| `prompt_suffix` | `None` | `_cross_task_source_suffix(...)` |
| Source program | none | source rank-1 `choose()` code |
| Source examples | none | **one** injected source train+val trial (see below) |
| Source flags in logical argv | none (runtime error if leaked) | `--global_prompt_source_program` + `--global_prompt_source_dataset` |
| Starts from | target seed | target seed (source is suffix context, not the seed parent) |

Transfer suffix builder: `_cross_task_source_suffix` → `build_rank1_explore_prompt_suffix` (`utils/teh/explore_source_prompt.py`) → `make_source_context` → `one_example_trial_text`. Mixed-gambles source forces mixed-gambles filtering even if the target is not mixed_gambles.

How the one trial is chosen (gated transfer, `require_source_examples=True`):

1. `_load_source_example_trials` takes the **first person in `emnlp_ordinal_range(source_dataset)`** (`pid = pids[start]`), the same source G.1 cohort. For Wulff that is ordinal 1290, not `valid_participant_ids[0]`. For sources whose EMNLP range starts at 0, `pids[start]` equals `pids[0]`.
2. It loads that person’s SA40 train+val (test discarded) and keeps at most the first eight of those trials (`trials[:8]`). That eight is a **candidate pool**, not eight injected examples.
3. `one_example_trial_text(..., seed=split_seed)` draws **exactly one** trial from that pool (`np.random.default_rng(split_seed)`) and serializes it. The suffix heading is singular: `SOURCE example from source train+validation only`.

The suffix therefore always contains **one** source trial, not “up to 8.” Independent live G.1 uses the same EMNLP-first-person example rule.

### 6.3 Candidate schedule (implemented `floor` decay)

`fresh_n = min(n_candidates, max(1, floor(fresh_n_candidates * (1 - iter_idx / total_iters))))`  
(`_decayed_fresh_n_for_iteration`). Binary float makes `10*(1-4/5)` and `10*(1-8/10)` equal `1.999…` → **floor 1**.

**G.2 (5 iters):**

| Iter (0-based) | Fresh (seed only) | Elite/sample-parented |
| --- | --- | --- |
| 0 | 10 | 0 |
| 1 | 8 | 2 |
| 2 | 6 | 4 |
| 3 | 4 | 6 |
| 4 | **1** | **9** |

**G.1 / person (10 iters):** fresh `[10,9,8,7,6,5,4,3,1,1]` and parented `[0,1,2,3,4,5,6,7,9,9]`. Iterations 8 and 9 both have `fresh_n=1` for the same float reason.

Sampled parent count also decays (`_decayed_sampled_parents_k_for_iteration`) from `--sample_size 8` toward 0. Rank-1 is always eligible; remaining parents are sampled without replacement from the elite pool when `--sample_parents` is on.

---

## 7. Observed-data gate

**Functions:** `decide_gate`, `evaluate_mean_train_val_loglik`, `participant_train_val_loglik`  
**Record:** `gate/gate_record.json` schema `t_pics_gated_transfer_gate_v2`  
**Field:** `mean_train_val_loglik` (`GATE_SCORE_FIELD`)  
**Name:** `train_val` / “observed-data performance gate”  
**Tie tolerance:** `GATE_TIE_TOLERANCE = 1e-12`

### 7.1 Per-person score (same formula as evolution `train_val`)

Let \(n_\mathrm{tr}, n_\mathrm{vl}\) be that person’s SA40 train/val trial counts, and \(\ell_\mathrm{tr}, \ell_\mathrm{vl}\) the average log-likelihoods (`n_eval_seeds=3`). Test is discarded.

\[
s_i =
\begin{cases}
\ell_\mathrm{tr} & n_\mathrm{vl}=0 \\
\ell_\mathrm{vl} & n_\mathrm{tr}=0 \\
\dfrac{n_\mathrm{tr}\,\ell_\mathrm{tr}+n_\mathrm{vl}\,\ell_\mathrm{vl}}{n_\mathrm{tr}+n_\mathrm{vl}} & \text{otherwise}
\end{cases}
\]

(`_evolution_selection_score` with `evolution_selection_score="train_val"`). Non-finite scores on a non-empty split → person invalid → whole arm score `None`.

### 7.2 Aggregation across people

\[
S = \frac{1}{M}\sum_{i=1}^{M} s_i
\]

Equal person weight (`participant_weighting: "equal"` in the gate record). **Not** pooled-trial weight. People with no observed trials are skipped; if nobody scores, \(S\) is `None`.

This is **deliberately different** from G.2 pool ranking, which uses **pooled** train/val across people. The gate re-evaluates each arm’s saved rank-1 on every target person independently, then averages people.

### 7.3 Decision and fallbacks (`decide_gate`)

Transfer is selected **only** if both arms are complete, transfer `choose()` is valid, both scores are finite, and \(S_\mathrm{transfer} > S_\mathrm{control}\) by more than \(10^{-12}\).

| Condition | `reason` | Selected arm |
| --- | --- | --- |
| Transfer strictly better | `transfer_strictly_better` | transfer |
| Control strictly better | `control_better` | control |
| Exact equality | `exact_tie` | control |
| \(\lvert\Delta S\rvert \le 10^{-12}\) | `tolerance_tie` | control |
| Missing/non-finite score | `invalid_score` | control |
| Transfer arm exception / incomplete pool | `failed_transfer_arm` | control |
| Transfer rank-1 has no valid `choose()` | `failed_transfer_program` | control |
| Control missing **or** both arms bad | `missing_arm` | control (then control failure **raises**) |

Control failure is fatal (`RuntimeError`). Transfer exceptions are caught, written to `target_population/transfer/FAILED.json`, and the gate keeps control. Raw source programs are **never** scored on the target (`never_evaluated_raw_source_on_target: true`).

Winner pool is copied/symlinked to `selected/retained_global_elite_pool`. `selected/SELECTED_ARM.txt` stores `"control"` or `"transfer"`.

---

## 8. G.3 explore (50 from retained target rank-1)

After the gate, `teh.py` rewrites the downstream `base_run_dir` to `run_root/selected/` and hands `global_elite_for_handoff` = **full winner pool**.

G.3 settings (enforced):

- `--explore_candidates 50` (`DEFAULT_EXPLORE_CANDIDATES`)
- `--explore_from_population_parents` (required)
- `--explore_population_top_k 1` (required; `select_explore_handoff_parents` keeps only elite `[0]`)
- `--explore_seed_candidates` default **0** → all 50 from that one parent
- `plan_g3_explore_parents`: `source_suffix_in_explore=False`, `source_examples_in_explore=False`
- CLI rejects `--explore_prompt_source_program` under gated transfer

Each explore LLM request has **one** parent (the retained target rank-1). Explore prompts include that **person’s** train+val (`extra_prompt_trials=val_trials` in `_run_pre_evolution_explore_phase`), capped by `_cap_prompt_train_and_val_trials` (`max_prompt_train_trials=60`, `max_prompt_trials_per_problem=5`). **No source suffix.**

Valid explore candidates merge into the person elite. Diagnostic test is scored **only on the selected explore best**, not on all 50.

---

## 9. Participant evolution

- `--n_iterations 10` (`DEFAULT_N_ITERATIONS`); gated `apply_gated_cli_defaults` sets this unless the flag was passed; cluster always passes `10`
- Same candidate/fresh/parent/elite knobs as G.1 (`n_candidates=10`, `fresh_n_candidates=10`, `sample_size=8`, `elite_pool_size=50`, decay on)
- Initial elite = **full retained G.2 pool** (not rank-1 only)
- Fresh candidates are generated from the **target seed program only** (`person_fresh_from_seed_only`)
- Gated-only: `_person_evolution_extra_prompt_trials` injects that person’s **val** trials into candidate prompts in addition to train. Generic TEH stays train-only. Test is never passed in.
- Shared union cap `_cap_prompt_train_and_val_trials` (60 trials / 5 per problem)
- No source rank-1 and no source examples in person prompts
- `--no-refinement_phase`; `--max_error_prompt_chars 0`

---

## 10. SA40 splits: train, val, observed, test

Protocol: `--limited_data_protocol structure_aware --limited_train_val 40`  
Module: `utils/teh/limited_data_protocol.py`

- **Test is reserved first** and never counted in the 40.
- At most **40 combined train+validation** observations per person (complete resetting units, not a random 40-trial slice of the full series).
- Remaining observed trials are split with `--split_ratio 0.6` and `--split_seed 0` (`within_participant`).
- **Observed** = those retained train ∪ val trials (typically ≤40). This is the set used for fitness, ranking, gating, and gated prompt examples.
- **Test** = held-out trials after the protocol. Diagnostic reporting only.
- Speekenbrink uses chronological session split by default (`--speekenbrink_split chronological`).
- Psych-101 rows come from `--psych_dataset_split train` (HF split name; not the within-person train/val/test).

### Who is in the job (two layers)

There is **no** global “drop people with fewer than N trials” cutoff. Person selection is:

1. **Valid list** (`valid_participant_ids.json`, built by `utils/teh/participant_ids.py`): keep anyone who still has **non-empty train and non-empty test** after the within-person split. Same rule for Psych-101, mixed gambles, and the three externals (Bergert, Guan, Steyvers). That is “can be split,” not “has ≥N trials.”
2. **EMNLP table** (`analysis/config/teh_datasets.yaml` → `emnlp_ordinal_range` → cluster `t_pics_resolve_emnlp_range`): inclusive ordinal slice into that valid list. G.2 uses the **target** row. Frozen G.1 and independent live G.1 use the **source** row.

Worker argv is always `--participant_scope range --range_start_ordinal … --range_end_ordinal …`. Do not hardcode 0–49 on Wulff, Speekenbrink, or Badham.

| Dataset | Group | Valid *n* | Inclusive ordinals | People in the job |
| --- | --- | ---: | --- | ---: |
| `1peterson2021using` | EMNLP-10 | 751 | 0–49 | 50 |
| `2plonsky2018when` | EMNLP-10 | 216 | 0–49 | 50 |
| `3frey2017cct` | EMNLP-10 | 1368 | 0–49 | 50 |
| `4wulff2018description` | EMNLP-10 | 1721 | **1290–1339** | 50 |
| `5speekenbrink2008learning` | EMNLP-10 | 23 | **0–22** | 23 (all valid) |
| `7hilbig2014generalized` | EMNLP-10 | 73 | 0–49 | 50 |
| `10frey2017risk` | EMNLP-10 | 1331 | 0–49 | 50 |
| `11enkavi2019recentprobes` | EMNLP-10 | 471 | 0–49 | 50 |
| `12badham2017deficits` | EMNLP-10 | 85 | **0–9** | 10 (EMNLP table size; valid list is longer) |
| `mixed_gambles` | EMNLP-10 | 578 | 0–49 | 50 |
| `13schulz2020finding` | five-new | 99 | 0–49 | 50 |
| `14kool2016when` | five-new | 188 | 0–49 | 50 |
| `bergert_nosofsky_2007` | five-new | 61 | 0–49 | 50 |
| `guan_2020_stopping` | five-new | 56 | 0–49 | 50 |
| `steyvers_2009_bandit` | five-new | 451 | 0–49 | 50 |

The five-new datasets all have ≥50 valid people, so **0–49 is the 50-person table**. They do **not** get a second high-trial filter.

**Wulff is the only “enough trials” exception.** The valid list still includes sparse people (ordinal 0 has 3 total trials). The EMNLP slice **1290–1339** is the higher-trial cohort used by June TEH and by G.1 job `257177` (ordinal 1290 has 66 total trials). That slice is **not** applied to any other dataset.

### Mixed gambles trial types

CSV `datasets/mixed_gambles/data_all_2021-01-08.csv` has `gamble_type` values **`gain_loss`** and **`gain_only`** (no `loss_only` in this file).

**Production default is all trial types**, not gain_loss-only and not gain_only-only. Gated `PICS_ARGS` and frozen G.1 job `257183` do **not** pass `--filter_mixed_gambles` (`teh.py` default False). Target mixed-gambles jobs therefore load both `gain_loss` and `gain_only` trials for the 50 people at ordinals 0–49.

`--filter_mixed_gambles` would keep only `gamble_type==gain_loss` and would read `valid_participant_ids_gain_loss.json` instead of `valid_participant_ids.json`. Those two files currently list the **same 578 people**; the flag changes **which trials** are loaded, not the EMNLP person table.

When mixed gambles is the **source** (Wulff as target), `_cross_task_source_suffix` and independent live G.1 still set `filter_mixed_gambles=True` (`source_dataset == "mixed_gambles"`). That affects the suffix example / live G.1 trials only. Frozen G.1 `257183` itself was trained on all trial types.

---

## 11. Which trials enter which computation

`T` = train, `V` = val, `TV` = observed union, `Test` = held-out. “Capped” = `_cap_prompt_train_and_val_trials` (60 / 5 per problem).

| Stage | Automatic / candidate prompts | Fitness & ranking | Gate | Diagnostic Test |
| --- | --- | --- | --- | --- |
| G.1 source pop (live or historical) | Source auto-prompt from source TV (capped). Candidates: pooled source T+V | Pooled source `train_val` | n/a | Pool-best after each iter (passive) |
| G.2 control | Target auto-prompt (TV, shared). Candidates: pooled target T+V. **No source** | Pooled target `train_val` | Rank-1 re-scored per person on TV; equal-person mean | Pool-best after each of 5 iters (pooled target test) |
| G.2 transfer | Same target prompt **plus** source rank-1 code + **one** injected source TV trial (from source EMNLP `pids[start]`, sampled from that person’s first ≤8 TV trials) | Same pooled target `train_val` | Same gate formula | Same as control |
| G.3 explore | Person T+V (capped). **No source**. Parent = target rank-1 | Person `train_val` | n/a | **Selected explore best only** |
| Person evolution | Person T+V (gated). **No source** | Person `train_val` | n/a | Pool-best after each iter + final selected |
| Source selection YAML | n/a | n/a | n/a | Frozen; test unused |

Test never enters prompts, fitness, ranking, elite membership, gating, stopping, resume completeness, or G.3/person handoff (`diagnostic_reporting` in `gated_run_metadata`).

---

## 12. Frozen production knobs

Cluster `PICS_ARGS` in `job_gated_l40s.sh` plus `apply_gated_cli_defaults` and argparse values **not** overridden by the worker:

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-Coder-32B-Instruct`, `--mode local`, `--llm_max_tokens 1024`, `--max_workers 100` |
| `--n_candidates` / `--fresh_n_candidates` | 10 / 10 |
| `--sample_size` / `--sample_parents` / `--sampled_parents_decay` | 8 / on / on |
| `--elite_pool_size` | 50 |
| `--global_iters` / `--n_iterations` / `--explore_candidates` | 5 / 10 / 50 |
| `--explore_from_population_parents` / `--explore_population_top_k` | on / **1** |
| `--explore_seed_candidates` | 0 (argparse default; not passed) |
| `--evolution_selection_score` | `train_val` |
| `--fitness_metric` | `loglik` |
| `--split_ratio` / `--split_seed` / `--split_mode` | 0.6 / 0 / `within_participant` |
| `--max_prompt_train_trials` / `--max_prompt_trials_per_problem` | 60 / 5 |
| `--max_parent_chars` | 3500 |
| `--hard_prompt_token_cap` / `--strict_prompt_budget` | 14000 / True (argparse defaults) |
| `--n_eval_seeds` | 3 (argparse default) |
| `--max_error_prompt_chars` / `--error-feedback-mode` | 0 / `legacy` |
| `--mdl_lambda` | 0 |
| `--no-refinement_phase` | required |
| `--prefer_auto_llm_prompt` | on; gated also sets `require_auto_llm_prompt=True` |
| `--mem_trace` | on |
| `--parallel_participants` | on |
| `--early_stop_iters` | -1 (disabled) |
| `--phase` | `all` |
| Seed programs | `persona_code_example/te_vanilla/choices13k.py` except Steyvers/Schulz → `persona_code_example/teh/categorical_uniform.py` |
| Auto-prompt LLM seed | `split_seed + 90_000` |
| Candidate decoding seed | `split_seed + 80_000 + iteration*1_000_003 + pid*17_179 + batch_offset` |
| Prompt subsample seeds | global `split_seed+60_000+iter`; explore `split_seed+70_000+parent_i*1009` |
| KIND | `t_pics_gated` (independent: `t_pics_gated_independent`) |
| `RUN_TAG` | `g5e50p10_occurrence_eb` |
| W&B project | `teh_t_pics_gated` |

Generic argparse defaults that **must not** be confused with gated production: `--global_iters` default 10, `--n_iterations` default 5, `--sample_size` default 10, `--explore_population_top_k` default 0. Gated production **overrides** these via cluster argv and/or `apply_gated_cli_defaults`.

---

## 13. Diagnostic test-evaluation policy

`DIAGNOSTIC_TEST_EVAL_POLICY` in `utils/teh/t_pics_gated_transfer.py`:

| Phase | What is test-scored |
| --- | --- |
| G.1 / G.2 population | Pool-best after **each** iteration (G.2: pooled target test, once per iter per arm) |
| G.3 | Selected explore-best only (per person) |
| Person | Pool-best after each iteration **plus** the final `train_val`-selected program |
| Passive | `True` — never a decision input |

Completed default G.1 jobs are not rerun (`excludes_completed_g1: true`). Helpers: `_passive_diagnostic_test_eval` / `_passive_diagnostic_test_of_program` in `teh.py`.

---

## 14. Outputs, provenance, W&B, resume, failure

### 14.1 Directory layout (`run_layout`)

```
generated_outputs/psych101_train/teh/<target>/<KIND>/job_<SLURM_JOB_ID>/
  prompts/                          # shared target automatic prompt + seed copy
  INTENDED_ARGV.txt                 # worker argv + logical control/transfer/selected argvs
  log/run_metadata.json
  source_population/                # independent G.1 only
  target_population/control/global_phase/{best_program.py,global_elite_pool,results.json}
  target_population/transfer/global_phase/{...}  # or FAILED.json
  gate/gate_record.json
  selected/
    SELECTED_ARM.txt
    retained_global_elite_pool/     # symlink/copy of winner pool
    STAGE_COMPLETE.json             # after all people finish
    participant_<id>/               # G.3 + person artifacts
```

KIND is `t_pics_gated` or `t_pics_gated_independent`. Independent **cannot** overwrite default gated job dirs unless `KIND` is forced to a colliding value after the override (the worker/submitter rewrite `KIND=t_pics_gated` → `t_pics_gated_independent` whenever `INDEPENDENT=1`).

### 14.2 Provenance

- `gated_run_metadata` → `log/run_metadata.json` (config sha256, selected source, rank-1 path, gate knobs, diagnostic policy, `independent_source_population`, `reused_frozen_g1_rank1`)
- Per-arm `INTENDED_ARGV.txt`
- Gate record fields listed in `gate_record_payload`
- Prompt `prompt_meta.json` (`prompt_mode=auto_llm`, sha256)

### 14.3 Completeness / resume

| Check | Function / helper | Skip when |
| --- | --- | --- |
| Wrapper job | `t_pics_gated_job_complete` | `selected/STAGE_COMPLETE.json` **and** gate record **and** control best/pool/results exist (`SKIP_COMPLETED=1`) |
| Independent G.1 | `population_arm_is_complete(..., 10)` | live 10-iter source pop complete |
| G.2 arm | `population_arm_is_complete(..., 5)` | `best_program.py` + pool manifest + `results.json` with `n_iterations==5` + valid `choose()` |
| Person | `participant_run_is_complete` | best + results + explore metrics requesting 50 + iteration artifacts |
| Selected stage | `selected_stage_is_complete` | `STAGE_COMPLETE.json` participant set, or every person complete |

Wrapper skip is a **file-existence** check; Python skip is stricter (iteration counts, valid `choose()`). Test scores are not part of completeness.

### 14.4 W&B

`utils/teh/t_pics_gated_wandb.py` is **reporting-only**. It is not imported by fitness, ranking, gating, resume, or argv builders. Failures must not change local artifacts. Group `t_pics_gated_main`, job type `t_pics_gated`, tags `ICLR`, `SA40`, `gated_t_pics`. Independent still logs `source_g1_job_id="live"` and the intended live rank-1 path (sha256 may be null until G.1 writes the file).

### 14.5 Failure behavior

- Control G.2 incomplete → abort
- Transfer G.2 exception → `FAILED.json`, gate selects control
- Invalid transfer `choose()` → control
- Over-cap prompts after truncation → `strict_prompt_budget` raises (`PromptBudgetExceeded`)
- Independent G.1 incomplete 5-iter leftovers are **not** treated as done (expects 10)

---

## 15. CLI arguments (gated / independent)

### 15.1 New / gated-specific

| Argument | Default | Effect |
| --- | --- | --- |
| `--t_pics_gated_transfer` | off | Enable this method; apply gated defaults; run dual-arm G.2 + gate + G.3/person |
| `--t_pics_gated_independent` | off | Live 10-iter G.1 from YAML source identity; requires gated flag |
| `--t_pics_source_config` | leftover map unless gated | Gated forces `analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml` if empty |

`apply_gated_cli_defaults`: `global_phase=True`, `refinement_phase=False`, `explore_from_population_parents=True`, `explore_population_top_k=1`; if not passed on argv, `n_iterations=10`, `global_iters=5`, `explore_candidates=50`.

### 15.2 Hard conflicts under `--t_pics_gated_transfer`

- `--t_pics` or `--t_pics_source` (leftover live-source pipeline)
- `--t_pics_gated_independent` without `--t_pics_gated_transfer`
- `--global_prompt_source_program` / `--global_prompt_source_dataset` (transfer suffix is set internally)
- `--explore_prompt_source_program`
- `--explore_population_top_k != 1`
- missing `--explore_from_population_parents`
- `--refinement_phase`
- missing `--global_phase`
- `--explore_candidates <= 0`

### 15.3 Leftover flags that are **not** this method

`--t_pics`, `--t_pics_source` (optional auto source), and the default `--t_pics_source_config` of `analysis/config/transfer_source/t_pics_score_weighted_temp_fix.yaml` belong to the old live-source T-PICS CLI.

---

## 16. Exact commands

Default is dry-run (`DRY_RUN=1`). **Default main experiments do not set `INDEPENDENT=1`.**

Dry-run (prints 15-job plan, submits nothing):

```bash
bash cluster/2026Sep18_T_PICS_gated/submit_gated.sh
```

Production submit (frozen G.1 reuse):

```bash
DRY_RUN=0 CONFIRM_SUBMIT=1 bash cluster/2026Sep18_T_PICS_gated/submit_gated.sh
```

Optional independent (live G.1; **not** the default main run):

```bash
INDEPENDENT=1 DRY_RUN=1 bash cluster/2026Sep18_T_PICS_gated/submit_gated.sh
INDEPENDENT=1 DRY_RUN=0 CONFIRM_SUBMIT=1 bash cluster/2026Sep18_T_PICS_gated/submit_gated.sh
```

Subset retry: `TARGETS="1peterson2021using guan_2020_stopping"`.

Submit ledger: stdout table plus `cluster/record/2026Sep18_T_PICS_gated.tsv`. The worker is always `cluster/2026Sep18_T_PICS_gated/job_gated_l40s.sh` with `DATASET` exported (no per-dataset temp sbatch file).

After all 15 complete, analysis paths for `analysis/code/utils/compare.py` belong in `analysis/config/T-PICS/ICLR/` (person artifacts under `job_<ID>/selected/`). That YAML is **not** written by sbatch.

---

## 17. Optional independent mode (detail)

Use only when you want a **new** source population instead of jobs 257174–257188.

- YAML contributes **selected source dataset identity**
- Live G.1: source SA40, source automatic prompt, source seed, **source EMNLP range**, 10 iterations
- Transfer suffix uses `run_root/source_population/global_phase/best_program.py` as the source rank-1 **program**. The **one** injected source trial is taken from the first person in the source EMNLP range (`pids[start]`), matching live G.1 people.
- Control still has no source program/examples
- Outputs isolated under KIND `t_pics_gated_independent`
- Resume skips a genuinely complete **live** 10-iter G.1; it does not mix frozen `t_pics_source_pop10` artifacts
- G.2/G.3/person/gate/test policy unchanged (same one-trial suffix rule)

---

## 18. Known minor warnings (do not change the scientific design)

- **Decay floor:** G.2 last iteration has `fresh_n=1` (not 2) and G.1/person iterations 8–9 both have `fresh_n=1` because `floor(10 * (1 - k/n))` hits `1.999…`. Same function for G.1; not retuned.
- **G.2 vs gate aggregation:** G.2 ranks on **pooled** TV; the gate averages **per-person** TV with equal person weight.
- **Wrapper `SKIP_COMPLETED`:** existence of `STAGE_COMPLETE.json` + control artifacts; Python completeness also checks iteration counts and `choose()`.
- **Matched decoding:** G.2 arms share the decoding-seed **formula**, but transfer prompts include the source suffix, so paired identical tokens are impossible. vLLM may still be non-deterministic under load.
- **Prompt cap:** instruction+suffix are whitespace-compressed; extra parents then trials drop; leftover over-cap raises. Instruction+suffix for the 15 pairs is far under 14000 tokens.
- **Transfer failure → control:** recorded in `FAILED.json` / gate `reason`.
- **W&B `kind` string:** `gated_run_metadata` always stores `"kind": "t_pics_gated"` even in independent mode; the **directory** KIND is still `t_pics_gated_independent`. Reporting-only.
- **`--sample_size` help text** in generic `teh.py` says “default: 3” while argparse default is 10. Gated production passes **8**.

---

## 19. Paper-method checklist

| Scientific claim | Implemented by |
| --- | --- |
| Final method is gated dual-arm T-PICS, not leftover `--t_pics` | `--t_pics_gated_transfer`; conflict errors in `teh.py` `main()` |
| Source nominated by frozen Occurrence-EB, not retuned | `occurrence_eb_schema4_iter10.yaml`; `load_frozen_transfer_config`; `peeked_transfer_at_freeze: false` |
| Default reuses G.1 jobs 257174–257188 rank-1 | `entry.rank1_program`; cluster `INDEPENDENT=0`; KIND `t_pics_gated` |
| G.1 is 10-iter source pop | `SOURCE_POPULATION_ITERS`; YAML `iteration_prefix: 1-10` |
| Two matched 5-iter target pops | `_run_t_pics_gated_population_arms`; `DEFAULT_GLOBAL_ITERS=5` |
| Control has no source program/examples | `prompt_suffix=None`; `argv_source_conditioning_flags` raise |
| Transfer uses source rank-1 as **context only** | `_cross_task_source_suffix(program_path=source_rank1)` |
| Transfer injects **one** source TV trial, not eight | `_load_source_example_trials` (`trials[:8]` pool) + `one_example_trial_text`; person = first source EMNLP ordinal (`pids[start]`) |
| Shared target automatic prompt | one `setup_teh_run_prompts` on `args.dataset`; both arms get `run_prompts_dir` |
| Gate = equal-person mean observed `train_val`; test unused | `evaluate_mean_train_val_loglik` + `decide_gate`; `never_used_target_test_for_gate` |
| Transfer kept iff strictly better | `REASON_TRANSFER_STRICTLY_BETTER`; ties/failures → control |
| G.3: 50 cands from **retained target rank-1 only** | `--explore_population_top_k 1`; `select_explore_handoff_parents`; `plan_g3_explore_parents` |
| G.3 has **no** source suffix | `source_suffix_in_explore: False`; explore `--explore_prompt_source_program` forbidden |
| Full winner pool enters **person** elite | `--initial_pool_dir` / in-process `global_elite_for_handoff`; `g3_retains_full_winner_pool_in_person_elite` |
| Person: 10 iters, seed-fresh + elite parents | `--n_iterations 10`; `_decayed_fresh_n_for_iteration` |
| Gated person prompts include val | `_person_evolution_extra_prompt_trials` |
| SA40: ≤40 TV, test reserved first | `--limited_data_protocol structure_aware --limited_train_val 40` |
| People = valid-list train∧test, then EMNLP ordinal slice (five-new all 0–49) | `teh_datasets.yaml`; `t_pics_resolve_emnlp_range`; Wulff **1290–1339** is the only high-trial slice |
| Production mixed gambles uses **all** trial types (`gain_loss` + `gain_only`) | no `--filter_mixed_gambles` on gated/G.1 argv; G.1 job 257183 |
| Test never decides anything | `DIAGNOSTIC_TEST_EVAL_POLICY`; `_passive_diagnostic_test_*`; metadata `diagnostic_reporting` |
| Default main run is frozen G.1, not independent | `INDEPENDENT` default 0; no `--t_pics_gated_independent` in default `PICS_ARGS` |
| Independent is optional live G.1 on YAML source identity | `--t_pics_gated_independent`; `_ensure_gated_independent_source_population`; KIND `t_pics_gated_independent` |

---

## 20. Short code anchors

Gate rule (`utils/teh/t_pics_gated_transfer.py` `decide_gate`): transfer only if finite scores and `transfer_ll > control_ll` beyond `1e-12`; otherwise control.

G.3 parent slice (`utils/teh/explore_handoff.py`): `top_k == 1` keeps `elite_parents[:1]` (retained target rank-1).

Independent G.1 range (`teh.py` `_ensure_gated_independent_source_population`): `emnlp_ordinal_range(source_dataset)` with `participant_scope="range"`, then 10 live iterations.

G.2 transfer example person (`utils/teh/explore_source_prompt.py` `_load_source_example_trials`): `pid = pids[emnlp_ordinal_range(source)[0]]`; SA40 train+val; `trials[:8]` pool; `one_example_trial_text` injects one trial.

Valid-list builder (`utils/teh/participant_ids.py`): keep people with non-empty train **and** test after the within-person split. Mixed-gambles production omits `--filter_mixed_gambles` (all trial types).

---

## 21. Data splits, SA40, histories, and prompt examples

Verified from the implementation (`utils/teh/limited_data_protocol.py`, `utils/teh/limited_data_registry.py`, `teh.py`) and from production W&B configuration. Production gated T-PICS uses:

| Knob | Production value |
| --- | --- |
| Limited-data protocol | `--limited_data_protocol structure_aware` |
| Observed train+validation cap | `--limited_train_val 40` |
| Prompt example cap | `--max_prompt_train_trials 60` |
| Split | `--split_ratio 0.6 --split_seed 0` (`within_participant`) |

These knobs do not retune the method. They record how observations, runtime histories, and LLM examples are constructed.

### Split then SA40

The initial within-person train/validation/test split is **approximately 60/20/20** and respects dataset structure (whole problems/games/rounds where that is the unit; contiguous sessions for Speekenbrink and Kool). **Speekenbrink uses the chronological session split by default** (`--speekenbrink_split chronological`). Its difference from the legacy shuffled pseudo-block split is a **split-policy choice**, not an SA40 modification.

SA40 then retains **at most 40 combined train+validation** observations per participant (complete resetting units, or a contiguous pre-test segment on continuous sessions—not a random 40-trial slice of the full series). **Test membership, problems, and actions are not capped or replaced.** Test is reserved first and is never counted toward the 40.

In the final gated method, train+validation are conceptually the **observed** set. They remain **separate arrays** internally. Program fitness, ranking, and the observed-data gate use their **complete union** (`train_val`: trial-count-weighted mean of the two split averages).

### Prompt examples vs evaluation data

`--max_prompt_train_trials` is a **legacy argument name**. In gated T-PICS it does **not** mean “train only.” `_cap_prompt_train_and_val_trials` samples examples from the **combined observed train+validation union**. The production cap is **60** and affects **only LLM prompt examples**, never the fitness or evaluation set.

- **Participant SA40 prompts** have at most **40** observed trials (the person already has ≤40 TV, so the 60 cap does not drop further).
- **Pooled population prompts** (G.1 / G.2) may be subsampled to **60** from the pooled observed union.

Gated person and G.3 prompts include that person’s val as well as train. Test never enters prompts.

### Runtime histories (`choose(problem, history)`)

Evaluation and fitness call `choose(problem, history)` on the **stored runtime trial lists**, not on prompt-formatted text.

| Family | Datasets | Runtime history |
| --- | --- | --- |
| Independent | Wulff, Hilbig, Enkavi, mixed gambles, Bergert | Empty. |
| Resetting | Choice13k, CPC18, Frey CCT, Frey balloon, Badham, Schulz, Guan, Steyvers | Rebuilt **within** each game/problem/round/balloon; **reset** across units. |
| Continuous | Speekenbrink, Kool | Rebuilt over **retained observed trials followed by test trials** (one continuing sequence). |

Reconstruction **removes references to observations omitted by SA40**. When predicting any test action, **only information preceding that action** may appear in history. Earlier observed actions—and, for a continuing sequence, **earlier test actions**—may be legitimate history. The **current and future test actions must never** be included.

### Prompt-only formatting (not runtime inputs)

These limits never rewrite the `history` list passed to `choose()`:

- Candidate/person example **text** is a compact display (`history_len` / last feedback), not a dump of the full runtime history array.
- Automatic dataset-prompt generation caps nested history entries and example count (`dataset_prompt_history_max_entries` / example char budget). That path writes the shared instruction; it is not per-iteration state.
- Hard prompt token cap (`--hard_prompt_token_cap 14000`) may drop prompt examples after structured truncation; leftover over-cap raises under `--strict_prompt_budget`.

G.2 transfer still injects **one** source train+validation trial (sampled from a first-eight pool of the source EMNLP-first person). That example is prompt context only; source test is never used.
