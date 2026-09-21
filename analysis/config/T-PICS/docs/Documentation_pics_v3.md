# PICS v3 — final ICLR method

**PICS v3** is the final ICLR method for structure-aware program induction with
cross-task transfer under a shared training-only SA40 data protocol. It
supersedes EMNLP PICS, v1 gated T-PICS, and preliminary T-PICS v2. This file is
the single authoritative, self-contained description of the method as
implemented; older docs are historical records only.

Tracked constants: `utils/teh/pics_v3.py`. Gated pipeline:
`utils/teh/t_pics_gated_transfer.py`. Data protocol:
`utils/teh/limited_data_protocol.py` / `limited_data_registry.py`.

## Phase-by-phase overview

| # | Stage | Purpose | Input data | Sees | Init / parents | Iters × cands | Selection | Principal output | Transfer? | Test influence? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Data prep / `structure_aware_v3` | Build per-person observed TV ≤40; reserve test | Psych-101 train / local / external valid lists + EMNLP ordinals | train+val capped; test reserved | n/a | n/a | n/a | SA40 splits + histories | no | no (test membership/order/actions fixed; not used for fitting) |
| 2 | Automatic dataset prompt | Write fail-closed LLM instruction for this dataset | sample person’s retained TV + task metadata | train+val only | n/a (one LLM prompt gen) | 1 generation | n/a | `prompts/infer_single_choice.txt`, `prompt_meta.json`, seed copy | no | no |
| 3 | G.1 source population (`pics_v3_g1`) | Evolve each dataset’s own pooled population | that dataset’s EMNLP people, SA40 TV | train+val fitness | dataset seed; sample parents from elite | **10 × 10** (`n_iterations=0`, no people) | `train_val` (pooled) | `global_phase/best_program.py` + elite ≤50 | **no** | no |
| 4 | **Population** Schema-v5 annotation | Construct presence on G.1 population programs | G.1 programs + lineage | G.1 code only | n/a | n/a (annotator LLM) | n/a | `pics_v3_g1_schema_v5` annotations | no | no |
| 5 | Occurrence-EB + map freeze | Nominate one source per target | schema-v5 presence counts | annotations only | n/a | n/a | cosine on 6-source allowlist | `Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml` | no | no |
| 6 | G.2 control arm | Target population **without** source context | target people, shared auto prompt, seed | train+val | target seed; elite parents | **5 × 10** | pooled `train_val` | control `best_program.py` + pool | no | no |
| 7 | G.2 transfer arm | Matched target population **with** source suffix | same target data + selected source rank-1 + 1 source TV example | train+val (+ source TV in prompt only) | target seed; elite parents | **5 × 10** | pooled `train_val` | transfer `best_program.py` + pool | **yes** (suffix only) | no |
| 8 | G.2 gate | Pick control vs transfer on observed data | both arm rank-1s | train+val (**count-pooled**, same as G.2 ranking) | n/a | n/a | strict `S_tr > S_ctl` beyond \(10^{-12}\) | `gate/gate_record.json`, `selected/` | decision only | **no** |
| 9 | G.3 exploration | 50 candidates from retained **target** rank-1 | that person’s TV; winner pool loaded for later | train+val prompts | **rank-1 only** (`explore_population_top_k=1`); no source suffix | **50** explore cands / person | explore best by `train_val` | explore elites merged into person pool | no | no |
| 10 | Participant evolution | Per-person TEH after G.3 | person’s SA40 TV; full winner elite as initial pool | train+val | full retained pool + seed-fresh | **10 × 10** / person | `train_val` | `participant_*/` best + traces | no | no |
| 11 | Held-out eval / reporting | Passive test scores + W&B finals | frozen programs | test (passive) | n/a | n/a | n/a | `final/mean_*`, CSVs, completeness | n/a | **reporting only** |

**Naming note.** In code, “G.3” is the pre-person explore phase
(`--explore_candidates 50` with `--explore_from_population_parents` and
`--explore_population_top_k 1`). Participant evolution is a separate stage
(`--n_iterations 10`) that consumes the full winner elite via
`--initial_pool_dir` / in-process handoff. A single gated target job runs
stages 6–11 (plus target automatic prompt); stages 3–5 are separate jobs/CPU
steps. Stage 3 is **not** inside the default gated job (default gated **reuses**
frozen G.1 rank-1 after the map exists).

**Candidate-accounting convention (LLM-generated candidates).** Verified from
`SOURCE_POPULATION_ITERS=10`, `DEFAULT_GLOBAL_ITERS=5`,
`DEFAULT_EXPLORE_CANDIDATES=50`, `DEFAULT_N_ITERATIONS=10`,
`DEFAULT_N_CANDIDATES_PER_ITER=10`:

| Block | Count | Scope |
| --- | ---: | --- |
| G.1 | \(10\times10=100\) | one source-population job |
| G.2 | \(2\times5\times10=100\) | one gated target job (both arms) |
| G.3 | 50 | **per participant** |
| Person | \(10\times10=100\) | **per participant** |
| Paper “350” schedule | \(100+100+50+100=350\) | counts G.3 and person **once** (the per-person schedule), plus one G.1 and one dual-arm G.2 |

Total LLM candidates across \(M\) people in a gated job ≈ \(100 + 50M + 100M\)
(plus a separate 100 per G.1 dataset job). Do not confuse this with OpenEvolve’s
350 **iterations**.

---

## A. Final identity and scope

| Concept | Name | Meaning |
| --- | --- | --- |
| Method / gated target kind | `pics_v3` | Output folder KIND for gated dual-arm target jobs |
| G.1 kind | `pics_v3_g1` | Output folder KIND for global-only source populations |
| Independent-source kind | `pics_v3_independent` | Optional live source-pop under gated independent mode |
| Explicit independent source | `--t_pics_gated_source DATASET` | With `--t_pics_gated_independent`: name the live G.1 source dataset **without** a transfer map / `--t_pics_source_config`. Default map lookup unchanged when this flag is omitted. |
| Data protocol | `structure_aware_v3` | Training-only SA40 revision (same data path as v2; new flag) |
| Annotation taxonomy | `schema_v5` | Final five-construct vocabulary (population: `population_transition_v5`; offline person MEM: `participant_transition_v5`) |
| Runtime source YAML | `Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml` | Frozen after g5e50p30 G.1 → **population** schema-v5 annotate → Occurrence-EB |
| Annotation package (method) | `pics_v3_g1_schema_v5` | Schema-v5 annotations of g5e50p30 G.1 programs (feeds Occurrence-EB) |
| Annotation package (offline MEM) | `pics_v3_participant_schema_v5` | Person-trace transitions; **not** a method stage; currently NONFINAL/cancelled |
| G.1 path ledger | `pics_v3/g1_job_paths_g5e50p30.tsv` | Jobs `265753`–`265767` |
| W&B project (G.1) | `teh_pics_v3` | G.1 submitter default |
| `RUN_TAG` | `g5e50p30_occurrence_eb_pics_v3` | Gated run tag constant |

These axes are separate: method KIND ≠ data protocol ≠ annotation taxonomy.

**Fifteen datasets** (production EMNLP ordinals into `valid_participant_ids.json`).
Where the historical table was 50 people, PICS v3 uses the **first 30**:

| Dataset | Ordinals |
| --- | --- |
| `1peterson2021using` | 0–29 |
| `2plonsky2018when` | 0–29 |
| `3frey2017cct` | 0–29 |
| `4wulff2018description` | **1290–1319** |
| `5speekenbrink2008learning` | **0–22** |
| `7hilbig2014generalized` | 0–29 |
| `10frey2017risk` | 0–29 |
| `11enkavi2019recentprobes` | 0–29 |
| `12badham2017deficits` | **0–9** |
| `mixed_gambles` | 0–29 |
| `bergert_nosofsky_2007` | 0–29 |
| `guan_2020_stopping` | 0–29 |
| `steyvers_2009_bandit` | 0–29 |
| `13schulz2020finding` | 0–29 |
| `14kool2016when` | 0–29 |

Ordinals are list indices, not necessarily raw HF subject ids (e.g. Wulff
1290–1319 maps into the higher-trial cohort).

**Artifact status**

| Artifact | Status |
| --- | --- |
| Code / protocol / G.1 orchestration / docs | **final method definition** |
| 15× `pics_v3_g1` G.1 jobs (g5e50p30) | **complete** (`265753`–`265767`) |
| `pics_v3_g1_schema_v5` annotations | **complete** (1414 / 1414) |
| `occurrence_eb_schema5_iter10_pics_v3.yaml` | **frozen** (active runtime) |
| Participant Schema-v5 annotations (`pics_v3_participant_schema_v5`) | **NONFINAL / cancelled mid-run (2026-09-22)** — offline MEM/interpretability only; not a method stage; do not resume until final gated paths are approved |
| 15× `pics_v3` gated targets (schema5) | **in flight / resubmits** (ledger `2026Sep21_PICS_v3_gated_g5e50p30.tsv`; active paths `pics_v3/gated_job_paths_g5e50p30.tsv`; active IDs: `271238`–`271240`, `271242`–`271247`; resubmits `277676` Peterson, `277944` Speekenbrink, `277677` guan, `277678` steyvers, `277679` Schulz, `277697` Kool — replace `271237`/`271241`/`271248`–`271251`) |
| schema-v4 50-person G.1 / YAML freezes | **historical** (not active runtime) |
| v1 jobs 257174–257188 / 257756, v1 YAML | **frozen historical** |
| preliminary-v2 G.1 (incl. 258518), v2 YAML | **frozen non-final** |

**v1 and preliminary v2 are not the final ICLR method.** Do not treat their
kinds or YAML maps as normative for PICS v3. Production v3 ceilings match the
proven preliminary-v2 16k-class pair (16384 / 14000 / 1024) with first-30
ordinals and trial-first overflow truncation.

---

## B. Data protocol (`structure_aware_v3`)

**Flag:** `--limited_data_protocol structure_aware_v3 --limited_train_val 40`  
**Split:** `--split_mode within_participant --split_ratio 0.6 --split_seed 0`  
**Psych corpus:** `--psych_dataset_split train` (HF split name; not within-person).  
**Speekenbrink:** `--speekenbrink_split chronological` (default).

### Observed / training union (authoritative)

PICS v3 has **no validation role**. After the loader’s legacy 60/20/20 fields are
built, implementation `train` and `val` are one indivisible **observed/training
set**:

| Paper term | Implementation |
| --- | --- |
| Observed / training trials \(D^{\mathrm{train}}\) | chronologically ordered `train ∪ val` |
| Held-out test trials | `test` only — evaluation / reporting |

Helpers: `utils/teh/pics_v3_observed.py` (`merge_observed_trials`,
`count_pooled_observed_loglik`, `uses_pics_v3_observed_union`). Under
`structure_aware_v3` every ranking / elite / gate / parent / stopping decision
uses **count-pooled mean log-likelihood on that union**. A runtime error or
non-finite score on **any** observed trial invalidates the candidate (no
train-only fallback when legacy `val` is non-empty but crashing). Separate
`train_loglik` / `val_loglik` fields may remain as diagnostics only.

Artificial re-partitions of the same observed trials must not change merged
identities, prompt-example selection under the shared cap, the count-pooled
score, elite order, G.2 winners, or the gate decision.

### Training-only SA40

“Training-only SA40” means the 40-trial budget applies **only** to scored
observations used to build or select programs:

| Used for | Split |
| --- | --- |
| Automatic dataset prompt | observed union (`train∪val`) only |
| Candidate-generation examples | observed union under **one** shared cap |
| Evolution fitness / ranking / elite / gate / stopping | observed union only |
| Held-out evaluation | test, **after** selection |

Implementation: `uses_training_only_sa40` is true for `structure_aware_v2` and
`structure_aware_v3` (shared data path; `limited_data_protocol_revision` → `v3`).
v1 `structure_aware` remains a separate frozen path. **Strict observed-union
selection (no train-only fallback; interface preflight) applies only to
`structure_aware_v3`.**

### Interface robustness (separate from observed-union scoring)

History outcome fields (`feedback`, `reward`, and related keys) **may be
absent** depending on trial/stage. Runtime contracts document this; programs
must use `.get()` / membership checks.

Every PICS v3 candidate-generation prompt inserts this **immutable** block
immediately after the dataset-adaptive task description (not into trial JSON;
genuine field absence is preserved):

> `history` may be empty, and different history entries may contain different
> fields. Never assume optional fields such as `feedback`, `reward`, or outcome
> fields exist or are non-null. Check for a key or use `.get(...)` before reading
> it. Only fields explicitly required by the current task/API contract may be
> accessed directly.

Before elite admission, PICS v3 runs a **test-independent** interface preflight
(`utils/teh/pics_v3_contract_preflight.py`) on synthetic admissible
`problem`/`history` variants: empty history, action-only entries, heterogeneous
entries, missing/null optional outcomes, and valid dataset/stage required problem
fields. A candidate that raises on any required variant cannot enter the elite
pool. Test scores never drive selection or replacement.

### Deployment-time test fallback (not test-set model selection)

After the **single** selected program is frozen from **observed-union** scores
only, held-out test execution under `structure_aware_v3` uses an
OpenEvolve-matched **uniform failure fallback**
(`utils/teh/pics_v3_elite_failover.py`):

1. Evaluate **only rank-1** (the selected / best frozen program) on each test trial.
2. On exception or invalid/non-finite returned probability only, emit the uniform
   prediction for that trial: `0.5` (binary) or `1/K` over valid K-way actions.
3. **Include** every uniform-fallback trial in the participant (and cohort) mean
   log-likelihood; never omit the participant because of a test-time choose failure.
4. Never switch programs because another program’s prediction or likelihood looks
   better; never use the test label to choose, reorder, or validate programs.
5. Artifacts record `test_time_fallback` / `elite_failover` diagnostics
   (`mode=uniform_rank1`, `uniform_fallback_trials`, `uniform_fallback_rate`,
   `rank1_failures`). Cohort summary still reports `final/n_inf_test_loglik` for
   any remaining non-finite person-level test LL (should be rare under uniform
   fallback).

**Optional only (disabled by default):** `--pics_v3_elite_failover` retries frozen
ranks 2 then 3 (`MAX_ELITE_FAILOVER_RANKS=3`) before uniform. This is **not** the
official default method. Failover remains local to the trial.

Strict observed-union invalidation (selection) and the test-independent interface
preflight are unchanged and independent of this test-time policy.

### Construction

1. Within-person split ≈60/20/20 respecting structure (units / chronological
   sessions).
2. **Test reserved first**; never counted in the 40.
3. Retain **at most 40 combined train+validation** observations (complete
   resetting units, or contiguous pre-test suffix on continuous sessions—not a
   random slice of the full series).
4. Some people legitimately have **fewer than 40** TV trials (short series);
   there is no global “drop if <N trials” cutoff. People need non-empty train
   **and** test after the within-person split to be on the valid list, then the
   EMNLP ordinal slice selects the job cohort.

### Task families (`limited_data_registry`)

| Family | Datasets | Observed-set / history |
| --- | --- | --- |
| Independent | Wulff, Hilbig, Enkavi, mixed gambles, Bergert | Deterministic TV sample, seed 0, cap 40; **`history=[]` on train, val, and test** |
| Resetting | Choice13k, CPC18, CCT, Frey Risk, Badham, Guan, Steyvers, Schulz | Complete units then chronological prefix; stop at 40; within-unit histories |
| Continuous | Speekenbrink, Kool | Contiguous pre-test TV suffix, cap 40; original meaningful test histories |

**Kool exact-40:** under v2/v3, ordinals that would have been 41 under v1’s
stage-1 backup retain **exactly 40** (no `kool_include_matching_stage1`
overshoot).

**Test:** membership, order, problems, and actions unchanged. Test never enters
generation, fitness, ranking, gating, stopping, annotation, or source selection.

### `problem` vs `history` vs labels

- `choose(problem, history)` receives a **sanitized** `problem` and the runtime
  `history` list (possibly empty).
- Observed action/label is **never** inside `problem`; prompt snapshots put it
  in `observed_action_label` outside `choose()` input.
- Sanitizer (`sanitize_problem_for_choose`) strips current/future outcomes,
  correctness, and oracles—including Enkavi **`probe_in_set`**—from schema
  inference, automatic prompts, examples, contracts, and evaluator problems.
  Indexing a removed field → invalid candidate (`runtime_valid=False`,
  `avg_loglik=-inf`), not chance repair.

### Enkavi

Independent-task empty history on all splits; `probe_in_set` never reaches
programs. Preliminary contaminated G.1 job **258518** is non-final; PICS v3
reruns **all 15** G.1 datasets (no Enkavi-only shortcut).

---

## C. Automatic dataset prompts

**Mandatory for PICS v3 G.1 and fail-closed.**

- CLI: `--prefer_auto_llm_prompt` (there is **no** `--require_auto_llm_prompt`
  flag).
- `teh.py` main wires internal `require_auto_llm_prompt=True` when  
  `prefer_auto ∧ global_phase ∧ n_iterations==0 ∧ limited_data_protocol=="structure_aware_v3"`.
- Skips registered/reference prompts; on LLM failure **raises** (no merge
  fallback). Gated targets also set `require_auto_llm_prompt=True`.
- Historical `structure_aware` / `structure_aware_v2` G.1-shaped runs are
  **unchanged** (soft fallback remains possible).

**Inputs to generation:** task instruction/metadata + sample participant’s
retained TV examples (diverse selection, `DEFAULT_MAX_EXAMPLES=8`,
`DEFAULT_HISTORY_MAX_ENTRIES=8`, example char budget 10_000). Recursive schema /
conditional fields come from sanitized problems. Nested history truncated with
head/tail when longer than 8.

**Outputs** under the run’s `prompts/`: `infer_single_choice.txt`,
`prompt_meta.json` (`prompt_mode=auto_llm`), `llm_input_prompt.txt`,
`runtime_contract`, `seed_program.py`, `refine.txt`.

**When:** once at run start via `setup_teh_run_prompts`; reused for all
iterations/arms that share that `run_prompts_dir`. Distinct from
**per-generation** candidate packing (parents + displayed TV examples under the
hard token cap).

Seeds: G.1 uses ordinary prefer_auto path; gated auto-prompt decoding seed
`split_seed + 90_000`.

---

## D. Candidate-generation snapshots and prompt packing

**Snapshot contract** (`utils/teh/prompt_snapshots.py` / `prompt_context.py`):

- Full sanitized `problem`.
- History ≤ **8** entries; if longer, keep head \(\lfloor 8/2\rfloor\) + tail
  remainder with truncation metadata (not a dump of runtime history into
  `choose()`).
- Observed label outside `choose()` input.
- Participant-balanced pooled sampling for population prompts; participant /
  chronological ordering preserved in selection helpers.
- Display cap **`--max_prompt_train_trials 60`** (legacy name; gated path samples
  from **train∪val** via `_cap_prompt_train_and_val_trials`).
- **`--max_prompt_trials_per_problem 5`**; whole-example dropping under budget.
- Exact-token budgeting with Qwen chat template.

**Qwen ceilings** (`pics_v3.py` / `prompt_units.py`):

| Knob | Value |
| --- | ---: |
| Model | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| vLLM `--max-model-len` | **16384** |
| Hard input (templated) | **14000** |
| Output | 1024 |
| `--max_parent_chars` | 5000 |
| `--n_eval_seeds` (G.1) | **1** (deterministic `choose`; matches TEH CLI default) |
| `--sample_size` | 8 |

**Parent copies** (`_truncate_parent_program_for_prompt` in `teh.py`): if over
cap, keep ≈**70% head** + `# truncated; keep concise` + ≈**30% tail**. Complete
on-disk programs are always used for evaluation. Under token pressure on
**non-frozen** prompts, drop whole observed trials first, then extra parent
**copies** (keep ≥1). Frozen G.2 keeps the freeze-chosen example set and
parent-trims only. “Up to 8 parents” and “up to 60 examples” are caps, not
guarantees.

**Fitness vs display:** fitness always uses the full SA40 observed union;
prompt truncation never rewrites the evaluation dataset.

**G.2 pack version:** `g2_paired_pack_pics_v3` (freeze under transfer budget;
record `paired_parent_count`).

**Prompt-budget audit:** regenerator
`analysis/config/T-PICS/pics_v3/audit_prompt_budget.py`. Final per-dataset
retained-example/parent table is **not yet frozen** for 14k/5000 (requires
completed v3 G.1 + map for transfer audits). Known structural extremes under
the snapshot contract: **Badham** (tiny cohort / rich feature problems),
**Frey Risk** (balloon state in problem), **Kool** (long parents often still
under 5000 chars), **Speekenbrink** (continuous history).

---

## E. G.1 (`pics_v3_g1`)

**Global-only source-population evolution on that dataset’s own pooled
participants.** Separate Slurm jobs via
`cluster/v2/ours/Qwen/submit_g1_pics_v3.sh` → `job_g1_*.sh` →
`t_pics_v3_fill_g1_args` (reuses `t_pics_v2_fill_g1_args`, then patches
parent/hard-cap).

### Must not appear

`--t_pics_gated_independent`, `--t_pics_gated_transfer`, `--t_pics`,
`--t_pics_source_config`, preliminary-v2 YAML, selected source identity, source
rank-1, source trial, transfer suffix/gate, participant evolution after G.1,
or `--output_kind` (not a `teh.py` flag; KIND is path-only).

### Scientific knobs (from `t_pics_v2_fill_g1_args` + v3 patch)

| Knob | Value |
| --- | --- |
| `--global_phase` / `--global_iters` | on / **10** |
| `--n_iterations` | **0** (no person stage) |
| `--explore_candidates` | **0** |
| `--n_candidates` / `--fresh_n_candidates` | 10 / 10 |
| `--sample_size` / `--sample_parents` / `--sampled_parents_decay` | 8 / on / on |
| `--elite_pool_size` | 50 |
| `--evolution_selection_score` | `train_val` |
| `--fitness_metric` | `loglik` |
| `--mdl_lambda` | 0 |
| Refinement | `--no-refinement_phase` |
| Error feedback | `--max_error_prompt_chars 0 --error-feedback-mode legacy` |
| Protocol | `structure_aware_v3`, `limited_train_val=40` |
| Context | hard 14000, parent 5000, out 1024 |
| `--n_eval_seeds` | **1** (v3 G.1 override; deterministic programs) |
| Auto prompt | `--prefer_auto_llm_prompt` + fail-closed wiring |
| Ordinals | explicit `--participant_scope range` + EMNLP start/end |
| Seeds | `choices13k.py` except Steyvers/Schulz → `categorical_uniform.py` |

Fresh/parent decay over 10 iters: fresh
`[10,9,8,7,6,5,4,3,1,1]`, parented
`[0,1,2,3,4,5,6,7,9,9]` (`floor` decay; iters 8–9 both fresh=1).

### Outputs

```
generated_outputs/psych101_train/teh/<dataset>/pics_v3_g1/job_<id>/
  prompts/
  global_phase/best_program.py          # rank-1
  global_phase/global_elite_pool/       # ≤50 + pool_manifest.json
  INTENDED_ARGV.txt, mem_trace, logs
```

W&B: project `teh_pics_v3`. Resume/skip: worker may skip if
`best_program.py` and pool manifest already present (`SKIP_COMPLETED=1`).

---

## F. Schema-v4 annotation

**Taxonomy unchanged** from the v1 freeze; PICS v3 **re-annotates** new G.1
programs into a **new package path** (do not overwrite v1 annotations).

| Item | Value |
| --- | --- |
| Module | `utils/mem/schema_population_motif_v4.py` |
| `SCHEMA_VERSION` | 4 |
| Prompt id | `population_transition_v4_2` |
| Kind | `population_program_motif_transition` |
| Annotator | `analysis/mem/annotate_population_programs.py` |
| Model | `Qwen/Qwen2.5-Coder-32B-Instruct` (same as G.1) |
| vLLM `--max-model-len` | **16384** (aligned with PICS v3 production; Schema-v5 pop+person share this ceiling via `utils/mem/annotation_context.py`) |
| Planned outputs | `analysis_2026Sep/mem/pics_v3_g1_schema_v4/annotations/` |
| Cluster template | `cluster/v2/ours/Qwen/job_pop_annot_dataset.sh` (`VLLM_MAX_MODEL_LEN`, default 16384) |

### Six motifs (verbatim definitions)

| Motif | Definition |
| --- | --- |
| `history` | Explicit use of earlier choices/outcomes/trials/streaks/recency/counts or a history buffer to change the current decision. Unused history parameters do not count. |
| `value` | Computes/compares attractiveness, utility, expected payoff, benefits/costs, or option score. Constant/random choice without scoring does not count. |
| `probability_used` | Explicitly reads probability/likelihood/odds/uncertainty from the problem, including linear EV \(p\cdot x\). Empirical rates from feedback alone are feedback/learning. |
| `feedback` | Uses observed reward/correctness/success/failure/outcome to influence a later choice. Static current-trial payoffs are value. Feedback without an update rule is not learning. |
| `learning` | Updates/reconstructs an internal belief/preference/estimate/rule across trials from experience. Re-reading last choice/reward without update is not learning. |
| `explicit_risk_mechanism` | Beyond reading probability: nonlinear weighting, variance/downside, loss-aversion multipliers, risk penalty/bonus, etc. Linear EV alone is `probability_used` only. |

**Unit:** one G.1 population program (candidate) vs reference parent under the
population-transition contract. Resume key:
`dataset|run_id|iteration|candidate|parent`. Annotations never use transfer
outcomes or held-out test performance.

Required LLM fields: `candidate_id`, `reference_motif_state`, `modified_motifs`,
`motif_details`, `confidence`. Applicability:
`applicable` | `structural_na` | `not_applicable`.

---

## F2. Offline participant Schema-v5 annotation (MEM / interpretability)

**Not a main ICLR Method stage.** Stages 1–11 above induce and select programs.
This section documents the **post-hoc** person-program motif-transition pipeline
used for MEM / construct-effect interpretability on gated `pics_v3` person
traces. It does **not** feed Occurrence-EB, the frozen source map, G.2/G.3, or
program selection. Population annotations (`pics_v3_g1_schema_v5`), Occurrence-EB
artifacts, and completed main results are unchanged by this pipeline.

**Fleet status (2026-09-22):** participant annot shards `pannot5_*`
(`283950–283953`, `283955–283957`; earlier `283954`) were **cancelled**. Partial
outputs under `analysis_2026Sep/mem/pics_v3_participant_schema_v5/` are preserved
and marked `NONFINAL_CANCELLED_2026Sep22.txt`. Do **not** resume, resubmit, or
attach `afterok` annotation dependencies until final main-run gated paths are
explicitly approved.

### Five constructs; lean code-based labels

Shared closed vocabulary with population Schema-v5 (no `risk` /
`explicit_risk_mechanism` on the person side):

`history`, `value`, `probability_used`, `feedback`, `learning`

Annotation is **lean and code-based**: the LLM sees reference + candidate
program text (and dataset-gated field glossaries), not task narrative, trial
outcomes, fitness, or transfer context. Presence is about what the code
implements; transitions are relative to the resolved reference parent.

| Module | Role |
| --- | --- |
| `utils/mem/schema_participant_transition_v5.py` | Schema, prompt rules, validation, eligibility |
| `analysis/mem/annotate_edits.py` | Person-trace annotator (schema_version=5) |
| `utils/mem/participant_semantic_postprocess_v5.py` | Deterministic semantic postprocess |
| `analysis/mem/build_dataset.py` | MEM CSV from **corrected** fields |
| `analysis/mem/fit_mem_random_slopes.py` / `fit_mem_joint_random_slopes.py` | Construct-effect fits (exclude unresolved NMC) |
| `analysis/mem/coverage_eligibility_v5.py` | Coverage / risk-set audit (retains unresolved) |

Prompt stamp: `participant_transition_v5_calibrated_2026Sep22` (frozen calibration
rules + optional dataset glossary).

### Exploration vs evolution references

| Phase | Reference | Notes |
| --- | --- | --- |
| `evolution` | Prior person parent (`reference_id` / parent in pool) | Normal / fresh person edits vs that parent |
| `explore` | Gate-winning **population** program (shared explore reference) | Not the seed baseline unless unresolved; stamped `reference_type=population_program` when resolved via gate-winning / explore helpers |

Resume identity is global:
`dataset|run_id|participant|phase|iteration|candidate|reference_id|reference_type`
so explore and evolution never collide.

### Eligible baseline-fresh transitions

`source=fresh` candidates are annotated only when an **explicit** reference
resolves (including official seed-baseline artifact resolution when the parent
is the constant seed program) and ΔF is finite and consistent with
candidate/reference scores. Unresolved strict references and ΔF inconsistencies
are recorded in `annotation_exclusions.jsonl` — not silently coerced.

Exact seed-baseline constant programs (`return 0.5` body) force **all five**
reference constructs absent under deterministic postprocess
(`SEED_BASELINE_REF_ABSENT`).

### Raw LLM labels vs corrected production labels

Automatic production write path (`annotate_edits._write_annotation_rows` for
schema v5):

1. Preserve raw LLM response text under `raw_responses/`.
2. Schema-validate the JSON batch.
3. Snapshot validated fields into `raw_llm_annotation` on the row.
4. Run deterministic semantic postprocess + transition rederivation.
5. Append **corrected** row to `annotations_v5.jsonl`.
6. Append rule hits to `semantic_corrections.jsonl`; queue unresolved NMC to
   `nmc_adjudication_queue.jsonl`.

`build_dataset.py --schema_version 5` reads **corrected** top-level fields and
also propagates `raw_llm_annotation`, `semantic_resolution_status`,
`nmc_adjudication_status`, and `exclude_from_construct_effect_fitting` into the
MEM CSV for audit.

### Deterministic semantic validation / postprocessing

Generic (all datasets): unused-history absence; seed-baseline ref absence; clear
false `no_meaningful_change` when normalized ASTs differ in parameters /
operators / control / update shape (**without** auto-mapping every numeric change
to `value_modified`); modified∩intersection; rederive added/removed/modified/
unchanged + `transition_by_construct`; correction logging.

**Bergert-only** (strict `dataset == bergert_nosofsky_2007` gate):
`action_means_option_A_when_1` sole-evidence strips; unsupported P/F/L absence;
cue-weight/scoring param|op → `value_modified`. These rules cannot fire on other
datasets.

### Dataset-gated semantic glossaries

Optional field glossaries are injected into the person prompt only for datasets
that need disambiguation of code field names (currently Bergert:
`action_means_option_A_when_1` is a Boolean action-coding flag — not Value,
Probability, or Feedback). No task narrative or outcomes.

### Transition and eligibility rederivation

After presence/NMC fixes, directions are re-derived from reference/candidate
states and `modified_motifs`. Eligibility columns follow Schema-v5 risk sets
(addition / removal / retained-construct modification). MEM builders never
coerce missing state to empty sets.

### Unresolved NMC adjudication

When NMC is cleared by AST evidence but construct attribution remains ambiguous
(generic path): **no automatic focused LLM adjudication pass**. Rows stay
explicitly unresolved:

- `semantic_resolution_status = nmc_needs_adjudication`
- `nmc_adjudication_status = unresolved`
- `exclude_from_construct_effect_fitting = true`
- retained constructs without attributed modification → transition `unresolved`
  (not silently NMC, modified, or unchanged)

Focal/joint construct-effect fitters **drop** these rows. Coverage / audit
reports **retain** them and count status. Offline queue:
`nmc_adjudication_queue.jsonl`.

### Exact 16k token-budget enforcement; no program-code truncation

Person annot shares the production ceiling via
`utils/mem/annotation_context.py`:

| Knob | Value |
| --- | ---: |
| vLLM / `max_model_len` | **16384** |
| Reserved output tokens | **2048** (person default) |
| Safety margin | **512** |
| Max candidates / batch | **5** |

Packing uses the Qwen tokenizer when available: chat input + reserved output +
margin ≤ `max_model_len`. Batches split before overflow. **Program code is never
truncated** to fit the budget — only batching / splitting.

### Output files and reproducibility paths

| Path | Contents |
| --- | --- |
| `analysis_2026Sep/mem/pics_v3_participant_schema_v5/annotations/by_dataset/<ds>/<gated_job>/` | Per main-run package |
| `annotations_v5.jsonl` | Corrected rows (+ `raw_llm_annotation`, resolution stamps) |
| `raw_responses/` | Per-attempt LLM text |
| `semantic_corrections.jsonl` | Deterministic rule log |
| `nmc_adjudication_queue.jsonl` | Unresolved NMC payloads |
| `annotation_failures.jsonl` / `annotation_exclusions.jsonl` / `annotation_normalizations.jsonl` | Soft-fail / eligibility / modified∉∩ |
| `annotation_summary.json` (+ `_by_phase`) | Coverage / budget / resolution tallies |
| Cluster submit | `cluster/v3/ours/main/submit_participant_annot_schema_v5.sh` |
| Shard table | `cluster/v3/ours/main/participant_annot_schema_v5_shards.tsv` |
| Path ledger | `analysis/config/T-PICS/pics_v3/gated_job_paths_g5e50p30.tsv` only |
| Calibration notes | `analysis_2026Sep/Sep20_V3/mem/annnotation/` |

Downstream MEM CSV: `analysis/mem/build_dataset.py --schema_version 5
--annotations <annotations_v5.jsonl> --run_dir <gated run> --output_csv …`.

---

## G. Occurrence-EB source selection

**Precommitted selector family** (do not switch to combined/fitness-effect
selectors from retrospective transfer diagnostics).

| Piece | Spec |
| --- | --- |
| Fit script | `analysis/mem/pop_v4_source_selection/run_source_selection_v4.py` |
| Six-source allowlist | `1peterson2021using`, `3frey2017cct`, `4wulff2018description`, `7hilbig2014generalized`, `11enkavi2019recentprobes`, `mixed_gambles` |
| Self-exclusion | target never selects itself |
| Jeffreys | \(\tilde p=(k+0.5)/(n+1)\) |
| EB | MoM \(\tau^2\); shrink \(b\) toward \(\alpha\) |
| Rank | argmax cosine; ties → lexicographic source id |
| Selected artifact | source identity + path to that source’s G.1 **rank-1** |
| Peek policy | no transfer / test / G.2 peek at freeze |

**Runtime YAML:**  
`analysis/config/T-PICS/Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml`  
(frozen; `peeked_transfer_at_freeze: false`). Freeze companions:
`Transfer_source/pics_v3/schema5_occurrence_eb_freeze/`.

Historical schema-v4 freezes (50-person G.1; not active):
`occurrence_eb_schema4_iter10_pics_v3.yaml`,
`occurrence_eb_schema4_5construct_iter10_pics_v3.yaml`.

Fitting the map ≠ evaluating it retrospectively. Do **not** retune source
identities from transfer outcomes.

Gated defaults and submitters **require** a pics_v3 path (default schema5). There is **no**
preliminary-v2/v1 fallback for production PICS v3 targets.

---

## H. G.2 control and transfer arms

Implemented in `_run_t_pics_gated_population_arms` after the frozen map exists.
Both arms share: target EMNLP people, one target automatic prompt, target seed,
SA40 `structure_aware_v3`, split 0.6/0, `global_iters=5`, `n_candidates=10`,
`fresh_n_candidates=10`, `sample_size=8`, `elite_pool_size=50`,
`evolution_selection_score=train_val`, `mdl_lambda=0`, no refinement, error
feedback off, sequential control **then** transfer.

**Paired packing (`g2_paired_pack_pics_v3`):**

1. Freeze one deterministic target-example ID/count/order **before** either arm
   under the **transfer** condition (reserve transfer-suffix cost + up to
   `sample_size` parents at `max_parent_chars=5000`).
2. Both arms reuse that set for all five iterations.
3. Record `paired_parent_count`; both arms use that parent **count** (identities
   may differ naturally). No control padding with fake source text.
4. Drop whole target examples before dropping extra parent copies (runtime
   non-frozen path); freeze-time packing still parent-trims under the transfer
   budget so frozen G.2 examples are not dropped at runtime.
5. Do not re-append runtime contracts or transfer suffixes after packing.
6. Both fully templated prompts ≤14000 tokens.

**Transfer suffix only** (`_cross_task_source_suffix` /
`build_rank1_explore_prompt_suffix`):

- Selected source **rank-1** program text;
- **One** source train+val example (pool = first ≤8 TV of first source EMNLP
  person `pids[start]`; `one_example_trial_text` draws one with `split_seed`);
- Source schema / transfer instructions;
- **Never** source test.

Control: `prompt_suffix=None`. Source is context, not the seed parent.

**G.2 fresh schedule (5 iters):** fresh `[10,8,6,4,1]`, parented `[0,2,4,6,9]`.

Fitness/ranking: **pooled** target TV across people. Rank-1 =
`global_phase/best_program.py`.

---

## I. Gate

| Item | Spec |
| --- | --- |
| Inputs | each arm’s **pooled-TV** rank-1 (`global_phase/best_program.py`) + complete pools |
| Score | **count-pooled** `train_val` on the target population — identical objective to G.2 arm ranking (`evaluate_pooled_train_val_loglik` / `GATE_SCORE_FIELD=pooled_train_val_loglik`) |
| Aggregation | trial-count-weighted over the pooled train+val union (**not** equal-person mean) |
| Decision | transfer only if finite scores and \(S_\mathrm{tr}>S_\mathrm{ctl}\) by more than \(10^{-12}\) |
| Ties / failures | keep **control** (`exact_tie`, `tolerance_tie`, invalid/missing/failed transfer) |
| Record | `gate/gate_record.json` (`t_pics_gated_transfer_gate_v3`) with `control_pooled_train_val_loglik` / `transfer_pooled_train_val_loglik` |
| Pointers | `selected/SELECTED_ARM.txt`; winner pool → `selected/retained_global_elite_pool` |
| Test | never used (`never_used_target_test_for_gate`) |
| Equal-person | **not** used for the gate (`never_used_equal_person_mean_for_gate`) |

Retained for G.3 / person: winner arm’s **full elite pool**; G.3 explore parent =
that arm’s pooled-TV rank-1 only (`explore_population_top_k=1`). No re-rank by
participant-mean; personalization starts in G.3 (per-person TV). Raw source
programs are never scored on the target.

---

## J. G.3 and participant evolution

### G.3 (explore)

- Purpose: generate 50 target-conditioned candidates from the gated winner
  before person search.
- `--explore_candidates 50`, `--explore_from_population_parents`,
  `--explore_population_top_k 1`, `--explore_seed_candidates` default 0.
- **All** explore parents = retained **target** rank-1 only; full winner pool is
  **not** sampled for explore parents but is retained for person elite.
- **No** source suffix / source examples (`plan_g3_explore_parents`).
- Prompts: that person’s train+val, capped (60 / 5-per-problem).
- Diagnostic test: selected explore-best only (passive).

### Participant evolution

- `--n_iterations 10`; same 10/10/8/50 candidate/fresh/parent/elite knobs as G.1.
- Initial elite = **full** retained G.2 winner pool.
- Fresh candidates from **target seed only** (`person_fresh_from_seed_only`).
- Gated person prompts include that person’s **val** as well as train.
- No source rank-1 / examples; refinement off; error feedback off; `mdl_lambda=0`.
- `--parallel_participants` on; `--early_stop_iters -1`.
- Outputs under `selected/participant_<id>/` (iteration dirs, best, mem_trace).
- Completion: expected people finished → `STAGE_COMPLETE.json`.

---

## K. Evaluation and reporting

| Item | Spec |
| --- | --- |
| Test timing | only after programs selected; never a decision input |
| Diagnostic policy | G.2 pool-best each iter; G.3 selected-best; person pool-best each iter + final selected (`DIAGNOSTIC_TEST_EVAL_POLICY`) |
| Paper-safe | equal-person means on complete runs only |
| W&B finals | `final/mean_train_val_loglik`, `final/mean_test_loglik`, `final/is_complete` (`t_pics_gated_wandb.py`) |
| Incomplete runs | final means **absent** / not authoritative when `final/is_complete` is false |
| Resume | skip complete arms/people; persist `{output_root}/wandb_run.json`; stable id `tpg_{alias}_{sha256[:20]}` from `method|run_tag|dataset|output_root`; init `resume="allow"` (never overwrite a conflicting persisted id) |
| Gated W&B group/tags | group `t_pics_gated_main`; tags include `ICLR`, `SA40`, `gated_t_pics` (reporter defaults) |
| Layout | `generated_outputs/psych101_train/teh/<target>/pics_v3/job_<id>/` with `target_population/{control,transfer}`, `gate/`, `selected/` |
| G.1 W&B | project `teh_pics_v3`; `RUN_TAG=g5e50p30_occurrence_eb_pics_v3` |
| Per-person W&B | **local CSV/JSON + dirs remain authoritative**; W&B keeps one `final/participant_table` and fixed keys (`dataset`, `status/*`, `progress/*`, `gate/*`, `final/*`, `global/*`, `g2/*`, `participant/*`). **No** dynamic Runs-table scalars `p{pid}/*` / `p{pid}_*` (reporting-only filter in `t_pics_gated_wandb.py`; G.1 skips those uploads). Historical W&B runs unchanged. |

Local CSVs / `INTENDED_ARGV.txt` / `log/run_metadata.json` are provenance;
authoritative gated results require complete participant set.

---

## L. Reproducibility and execution order

1. **15 G.1** (`KIND=pics_v3_g1`, GPU, g5e50p30) — submitter
   `cluster/v2/ours/Qwen/submit_g1_pics_v3.sh` (default `DRY_RUN=1`).
2. **Population Schema-v5 annotate** → `pics_v3_g1_schema_v5`
   (`cluster/v3/ours/main/submit_pop_annot_schema_v5.sh`).
3. **Occurrence-EB fit** (CPU) →
   `Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml`.
4. **Validate** YAML paths/hashes/identities; `peeked_transfer_at_freeze: false`.
5. **Final G.2 paired-packing audit** with selected v3 sources (CPU).
6. **15 gated targets** (`KIND=pics_v3`, `--t_pics_gated_transfer`, GPU) —
   `cluster/v3/ours/main/submit_gated.sh` (default `DRY_RUN=1`;
   default `--t_pics_source_config` → schema5 YAML).
7. **Completeness / result checks** (CPU + W&B).

**Offline (not in the method critical path):** after approved final gated
person traces exist, optional participant Schema-v5 MEM annotation
(`submit_participant_annot_schema_v5.sh` → `pics_v3_participant_schema_v5`)
followed by `build_dataset.py` / coverage / focal–joint fits. See **§F2**.
Does not regenerate Occurrence-EB or gated programs.

| If G.1 changes | Must regenerate |
| --- | --- |
| Any G.1 program set | annotations, Occurrence-EB YAML, G.2 audits, gated targets |
| Protocol/prompt ceilings | G.1 and all downstream |

| May stay frozen | v1/v2 artifacts, schema-v4 **definitions**, EB formulas/allowlist/tie policy |

**Hardware templates** (docs / dry-run): H100 NVL TP=1; 2×L40S TP=2; 4×RTX3090
TP=4; dtype BF16 (L40S/H100). Seeds: `split_seed=0`; phase decoding seeds as in
gated helpers (`split_seed+80_000+…`).

**Baselines:** Centaur / LM / PT / OpenEvolve may **share** `structure_aware_v3`
but keep their own ceilings (OE frozen **32768/30000/1024**) and are **not**
PICS. Do not rename OE outputs `pics_v3`. OE search (350 iterations, official
packing) is a separate baseline.

---

## M. Explicit differences from older implementations

| Surface | EMNLP PICS | v1 gated T-PICS | Preliminary v2 | **PICS v3 (final)** |
| --- | --- | --- | --- | --- |
| Datasets / ordinals | EMNLP table | same 15 + ordinals | same | **first 30** where table was 50 |
| Protocol | full / early sparse | `structure_aware` | `structure_aware_v2` | **`structure_aware_v3`** |
| Snapshots | compact one-liners (v1) | one-liners | JSON snapshots | JSON snapshots |
| vLLM / hard / out | smaller | 16k-class / 14k / 1024 | 16384 / 14000 / 1024 | **16384 / 14000 / 1024** (30 people) |
| `max_parent_chars` | varies | 3500 | 3500 | **5000** |
| Enkavi `probe_in_set` | risk of leak | sanitized in later code | sanitized | sanitized + full 15 G.1 rerun |
| G.2 packing | n/a / older | unmatched risk | `g2_paired_pack_v1` @14k | **`g2_paired_pack_pics_v3` @14k**, `paired_parent_count` |
| G.1 kind / path | older | `t_pics_source_pop10` | `t_pics_g1_sa40_v2` | **`pics_v3_g1`** |
| Gated kind | n/a | `t_pics_gated` | `t_pics_gated_sa40_v2` | **`pics_v3`** |
| Source YAML | n/a | `occurrence_eb_schema4_iter10.yaml` | `Transfer_source/v2/..._sa40_v2.yaml` | **`Transfer_source/pics_v3/..._pics_v3.yaml` (frozen)** |
| Normative doc | older notes | `docs/Documentation.md` (historical) | `Documentation_v2.md` (historical) | **this file** |

---

## Explicit exclusions (not normative PICS v3)

- Leftover `--t_pics` / 6×6 KIND jobs.
- Default gated **reuse** of v1 jobs 257174–257188 as the final source programs
  (v3 must use v3 G.1 + v3 map).
- Treating preliminary-v2 YAML or job 258518 as final.
- OpenEvolve / Centaur / LM / PT as part of the PICS method body.
- Claiming completed v3 annotations or G.2 source identities from preliminary-v2
  maps (use the frozen pics_v3 YAML only).

---

## Documentation audit (v1/v2 → v3 disposition)

Sources compared: `docs/Documentation.md` (v1 gated record),
`docs/Documentation_v2.md` (preliminary v2), current `pics_v3` / gated / G.1 /
protocol / packing / W&B code.

| Older section | Disposition in PICS v3 doc |
| --- | --- |
| v1 §1 phase overview | **Updated** — table rewritten for v3 kinds, ceilings, separate G.1 jobs |
| v1 §2 terminology | **Updated** — retained meanings; paths/kinds v3 |
| v1 §3 default vs independent | **Updated** — independent remains optional; default after v3 map reuses **v3** G.1 rank-1 |
| v1 §4 / freeze Occurrence-EB + motifs | **Retained unchanged** scientifically; **updated** paths to planned `pics_v3` package; v1 numbers labeled historical |
| v1 §5 G.1 | **Updated** — final G.1 is `pics_v3_g1` global-only; not YAML-bootstrap |
| v1 §6 G.2 arms | **Updated** — protocol v3, 14k/5k, `g2_paired_pack_pics_v3` |
| v1 §7 gate | **Changed in v3** — count-pooled TV (same as G.2 ranking); equal-person mean removed from the gate; tie rule unchanged |
| v1 §8 G.3 | **Retained unchanged** (50 from rank-1, no source suffix) |
| v1 §9 person evolution | **Retained unchanged** knobs; protocol/context v3 |
| v1 §10–11 SA40 / which trials | **Updated** → `structure_aware_v3` / training-only SA40 |
| v1 §12 production knobs | **Updated** — 14000/5000; KIND `pics_v3` |
| v1 §13 diagnostic test policy | **Retained unchanged** |
| v1 §14 outputs / W&B / resume | **Updated** — kinds/paths/W&B project; metrics names retained |
| v1 §15–18 CLI / commands / warnings | **Superseded** by v3 submitters; not copied as normative commands |
| v1 §19 checklist | **Updated** into sections A–M |
| v1 §21 splits/histories | **Updated** — snapshots replace one-line prompt notes; independent empty history |
| v2 §1–4 SA40 / histories / Kool | **Updated** into §B (v3 flag; same data path as v2) |
| v2 §7–9 prompt / packing / auto prompt | **Updated** into §C–D (14k/5k; trial-first truncate; fail-closed G.1) |
| v2 §11–12 OpenEvolve / versioning | **Baselines only** / frozen non-final list |
| v2 G.2 14k pack | **Superseded** by `g2_paired_pack_pics_v3` |
| v1 target→source identity table | **Superseded** — use `Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml` |
| v1 annotated program counts (1385) | **Historical only** — g5e50p30 schema-v5 count is 1414 |
| *(new)* Offline participant Schema-v5 MEM | **Added §F2** — lean person transitions, postprocess, unresolved NMC, 16k budget; distinct from method stages 1–11 |
| Phase table stage 4 | **Clarified** — population Schema-v5 only (feeds Occurrence-EB) |
| §L execution order | **Clarified** — step 2 is population annotate; person MEM is offline after approved gated traces |
| Artifact status / §A packages | **Updated** — person package NONFINAL/cancelled; dual pop vs person packages |

---

## Code anchors (normative)

| Topic | Location |
| --- | --- |
| v3 constants | `utils/teh/pics_v3.py` |
| G.1 argv | `cluster/v2/ours/Qwen/_common.sh` (`t_pics_v3_fill_g1_args`) |
| Fail-closed auto prompt | `teh.py` `g1_require_auto` + `utils/teh/teh_runtime.py` `setup_teh_run_prompts` |
| Gated defaults / gate / G.3 plan | `utils/teh/t_pics_gated_transfer.py` |
| Paired packing | `utils/teh/g2_paired_packing.py` |
| Snapshots / sanitize | `utils/teh/prompt_snapshots.py` |
| Parent 70/30 truncate | `teh.py` `_truncate_parent_program_for_prompt` |
| Motifs (population / method) | `utils/mem/schema_population_motif_v5.py` (final); `…_v4.py` historical |
| Motifs (offline person MEM) | `utils/mem/schema_participant_transition_v5.py` |
| Person annotator + postprocess | `analysis/mem/annotate_edits.py`; `utils/mem/participant_semantic_postprocess_v5.py` |
| Annot context / 16k ceiling | `utils/mem/annotation_context.py` |
| MEM CSV / fits | `analysis/mem/build_dataset.py`; `fit_mem_random_slopes.py`; `fit_mem_joint_random_slopes.py` |
| Occurrence-EB | `analysis/mem/pop_v4_source_selection/run_source_selection_v4.py` |
| W&B finals | `utils/teh/t_pics_gated_wandb.py` |
| Ordinals | `utils/teh/teh_datasets.py` / `analysis/config/teh_datasets.yaml` |
