# ICLR OpenEvolve baseline (frozen for PICS v3 comparison)

**How OpenEvolve is used in this paper.** We run the **pinned official
OpenEvolve** library (one mutable parent per generation, island / MAP-Elites
search) as a participant-level Python `choose(problem, history)` baseline.
Each person is an independent 350-iteration evolution job under the shared
PICS v3 data contract `structure_aware_v3` (training-only SA40). Fitness is
count-pooled mean log-likelihood on the retained train+validation union; test
is scored once after program selection. Generation uses Qwen2.5-Coder-32B with
**32768 / 30000 / 1024** context ceilings. OpenEvolve does **not** use PICS
transfer, G.1/G.2/G.3, Occurrence-EB, or PICS automatic prompt evolution.

This document is the authoritative ICLR OpenEvolve description for
`baseline_methods/Psych101/run_openevolve.py` against official commit
`411fb59c886c18704caaffb611e17cf9e7d824d2`. Older 14k / `structure_aware` /
600-iteration notes are **obsolete** for final ICLR launches.

| Label | Meaning |
|---|---|
| **Frozen scientific setting** | ICLR comparison knob; do not change in place |
| **Implementation detail** | Necessary adapter/runtime behavior |
| **Diagnostic/reporting only** | Logged; not used for evolution/ranking/paper means unless stated |
| **Historical / non-final** | Prior configs; do not launch as final |

## Overview

Official OpenEvolve samples one mutable parent per generation, proposes one
child, and scores it on that person’s observed data. Search uses
`structure_aware_v3` (≤40 train+val; Kool exact-40; independent histories empty
on all splits). Nominal generation budget: **350 iterations per participant**.
Prompts show a registry task description, neutral `choose()` API, the current
parent, observed TV examples (display-capped), and official optional contextual
programs when they fit the **30000**-token Qwen chat-template ceiling.
Contextual programs are **not** extra formal parents.

Evolution maximizes observed-union `combined_score`. Held-out test log-likelihood
is computed only after the best-by-observed program is frozen.

---

## 1. Purpose and scope

**Why OpenEvolve is included.** Official OpenEvolve is an island/MAP-Elites program-evolution library. We include it as an official OpenEvolve search core with a task-specific adapter for evolving Python `choose()` predictors of human choices, against which template/PICS methods can be compared under a shared data protocol and a matched nominal full-pipeline generation budget. Prompt construction, full-rewrite configuration, failure-score compatibility, and data/evaluator interfaces are adapted; islands/MAP-Elites and parent selection remain official.

**What constitutes one run.** One OS process of `run_openevolve.py` for one `--dataset` alias, over a frozen ordinal range of participants, writing one timestamped `run_*` directory (or `--output_dir`). **Frozen scientific setting:** 350 iterations per participant, `structure_aware_v3` SA40, split 0.6 / seed 0, Qwen2.5-Coder-32B-Instruct, **max_model_len 32768**, hard input **30000**, output **1024**.

**What constitutes one participant.** One `run_participant()` call: load that person’s trials, write train+val evolution JSON and a post-hoc test JSON, copy the seed program, generate `evaluator.py`, construct official `OpenEvolve(...)`, run `oe.run(iterations=n_iterations)`, select the best checkpoint program by observed `combined_score`, then score train, val, and test once.

**Entry point.** `python baseline_methods/Psych101/run_openevolve.py`. Helpers: `iclr_frozen_argv()`, `require_openevolve_checkout()`, `trials_for_participant()`, `_render_evaluator_py()`, `_patched_build_prompt()`. Shared data: `utils.teh.limited_data_protocol.load_participant_limited_splits`. Official library: `openevolve.OpenEvolve`, `openevolve.config.Config`, `openevolve.process_parallel.ProcessParallelController`.

**Adapter vs library.** The official library still owns parent sampling, islands, MAP-Elites, child `parent_id`, retries, and checkpoints. Our adapter (1) supplies dataset-specific seeds, evaluators, and vanilla prompts; (2) patches `PromptSampler.build_prompt` so the LLM sees the ICLR prompt, not the stock rewrite template; (3) maps official failed-eval `{error: 0.0}` onto a log-likelihood failure floor; (4) never clones or installs the library.

---

## 2. Pinned dependency and reproducibility

| Item | Value | Label |
|---|---|---|
| Expected checkout | `reference_repos/openevolve` (gitignored) | Frozen scientific setting |
| Required SHA | `411fb59c886c18704caaffb611e17cf9e7d824d2` (`EXPECTED_OPENEVOLVE_GIT_SHA`) | Frozen scientific setting |
| Fail-fast | `require_openevolve_checkout()` in `main()` | Implementation detail |
| Provenance | `run_config.json` fields `openevolve_git_sha`, `openevolve_expected_git_sha`, `openevolve_root` | Diagnostic/reporting only |
| Auto clone/fetch/install/checkout | Never | Frozen scientific setting |

**Manual install (example; do not run from this file):**

```bash
mkdir -p reference_repos
git clone https://github.com/codelion/openevolve.git reference_repos/openevolve
git -C reference_repos/openevolve checkout 411fb59c886c18704caaffb611e17cf9e7d824d2
```

Verify:

```bash
git -C reference_repos/openevolve rev-parse HEAD
# must equal 411fb59c886c18704caaffb611e17cf9e7d824d2
```

**Runtime sources vs documentation-only / ignored**

| Path | Role |
|---|---|
| `baseline_methods/Psych101/run_openevolve.py` | Sole production runner; in-runner `CHOOSE_API_*` strings are the interface source of truth |
| `utils/psych101_openevolve_pool.py` | Process-worker patch loader |
| `utils/teh/limited_data_protocol.py`, `limited_data_registry.py` | SA40 |
| `utils/teh/teh_datasets.py`, `analysis/config/teh_datasets.yaml` | Aliases and ordinal ranges |
| `data_modules/psych101_binary.py`, `data_modules/psych101_extensions/*`, `data_modules/external/*`, `data_modules/mixed_gambles.py` | Loaders |
| `persona_code_example/openevolve_vanilla/choices13k.py` | Bernoulli seed |
| `persona_code_example/teh/categorical_uniform.py` | Categorical seed |
| `data_modules/psych101_binary.py` `task_description`, `data_modules/mixed_gambles.py` `TASK_DESCRIPTION`, `data_modules/external/*/TASK_DESCRIPTION` | All 15 OE `# Task` strings (`dataset_task_description`) |
| `reference_repos/openevolve` | Imported library (gitignored, unpinned until SHA check) |
| `reference_repos/openevolve_official_audit` | Audit clone only; not imported by production |
| `analysis_2026Sep/**`, Cursor canvases | Ignored / not imported |
| This file | Documentation only |

---

## 3. Frozen ICLR configuration

All knobs live as `ICLR_FROZEN_*` in the runner and as argparse defaults / `iclr_frozen_argv()`.

| Knob | Value | Label |
|---|---|---|
| `--n_iterations` | 350 | Frozen scientific setting |
| `--model` | `Qwen/Qwen2.5-Coder-32B-Instruct` | Frozen scientific setting |
| `--llm_max_tokens` (output) | 1024 | Frozen scientific setting |
| `--hard_prompt_token_cap` (input) | **30000** chat-templated tokens | Frozen scientific setting |
| `--max_model_len` | **32768** | Frozen scientific setting |
| `--parallel_participants` | 1 | Frozen scientific setting |
| `--parallel_evaluations` | 4 | Frozen scientific setting |
| `--split_ratio` / `--split_seed` | 0.6 / 0 | Frozen scientific setting |
| `--limited_data_protocol` / `--limited_train_val` | **`structure_aware_v3`** / 40 (SA40) | Frozen scientific setting |
| `--max_prompt_train_trials` | 60 (display only) | Frozen scientific setting |
| `--num_top_programs` / `--num_diverse_programs` | 3 / 2 | Frozen scientific setting |
| `--include_artifacts` | `True` (official PromptConfig default) | Frozen scientific setting |
| `--psych_dataset_split` | `train` | Frozen scientific setting |
| `--random_seed` | 0 | Frozen scientific setting |
| `--temperature` | 0.7 | Implementation detail (matches official LLMConfig) |
| `--llm_timeout` / `--llm_retries` | 300 s / 3 | Implementation detail |
| `--evaluator_timeout` / `--evaluator_max_retries` | 120 s / 2 | Implementation detail |
| `--cascade_evaluation` | `False` | Implementation detail |
| `--checkpoint_interval` | 50 | Implementation detail |
| `--max_prompt_trials_per_problem` | 5 (only if observed union \(> 60\)) | Implementation detail |
| `--early_stopping_patience` | `None` (disabled) | Frozen scientific setting |

**Population / island / database (CLI defaults = official `DatabaseConfig` at 411fb59):**

| Field | Value |
|---|---|
| `population_size` | 1000 |
| `archive_size` | 100 |
| `num_islands` | 5 |
| `exploration_ratio` | 0.2 |
| `exploitation_ratio` | 0.7 |
| `elite_selection_ratio` | 0.1 |
| `migration_interval` | 50 |
| `migration_rate` | 0.1 |
| `feature_dimensions` | `complexity`, `diversity` |
| `feature_bins` | 10 |

**Intentional deviations from official 411fb59 defaults**

| Official default | Ours | Reason |
|---|---|---|
| `diff_based_evolution=True` | `False` | Full-program rewrite so `choose()` remains a complete file |
| `use_template_stochasticity=True` | `False` | Reproducible prompt text |
| `cascade_evaluation=True` | `False` | Single-stage loglik evaluator |
| `max_iterations=10000` | 350 | Nominal ICLR budget |
| `checkpoint_interval=100` | 50 | Finer checkpoints |
| `random_seed=42` (Config/Database) | 0 | Align with split seed |
| `evaluator.parallel_evaluations=1` | 4 | Concurrent island workers vs local vLLM |
| `evaluator.timeout=300`, `max_retries=3` | 120, 2 | Shorter hung-eval wait |
| `llm.max_tokens=4096`, `timeout=60` | 1024, 300 | Output ceiling + slower local vLLM |
| Stock full-rewrite user template | Patched vanilla prompt | Task interface + observed examples + 30000 packer |

Official `max_code_length=10000` is left at the library default (`ICLR_FROZEN_MAX_PROGRAM_CHARS` documents the audited bound; packing tests use it).

---

## 4. Candidate accounting and comparison

**One official iteration produces one proposed child.** At 411fb59, `_run_iteration_worker` samples one parent, builds one prompt, calls the LLM once (`generate_with_context`), parses one program, evaluates it, and returns at most one `Program` with `parent_id=parent.id`. Parse/LLM failures return `SerializableResult.error` and insert no child, but `completed_iterations` still increments. There is no extra regeneration loop in our adapter.

**Seed evaluation.** Official `OpenEvolve.run` evaluates the initial seed once when the database is empty. That is **in addition to** the 350 generation attempts.

**Why 350 vs gated T-PICS.** Gated T-PICS constants in `utils/teh/t_pics_gated_transfer.py`. OE’s 350 candidates per participant match T-PICS’s **nominal sum across pipeline stages**, not equal per-participant compute:

| Stage | Formula | Count | Shared? |
|---|---|---|---|
| G.1 source population | `SOURCE_POPULATION_ITERS=10` × `10` candidates | 100 | Yes (shared source-population candidates) |
| G.2 dual-arm transfer | `DEFAULT_GLOBAL_ITERS=5` × `2` arms × `10` | 100 | Yes (shared target-population candidates) |
| G.3 explore | `DEFAULT_EXPLORE_CANDIDATES=50` | 50 | No (participant-specific exploration per person) |
| Person evolution | `DEFAULT_N_ITERATIONS=10` × `10` | 100 | No (participant-specific per person) |
| **T-PICS nominal full-pipeline total** | | **350** | 200 shared + 150 participant-specific |
| **OpenEvolve** | 350 iterations / person | **350** | all 350 participant-specific |

This is a matched **nominal full-pipeline generation budget**, not wall-clock, GPU-seconds, prompt tokens, or equal per-participant compute. T-PICS has 200 shared candidates (G.1 + G.2) plus 150 participant-specific candidates (G.3 + person evolution). All 350 OE candidates are participant-specific; therefore this budget is favorable/generous to OE.

Failed OE iterations still consume budget. LLM HTTP retries (`--llm_retries 3` → up to 4 attempts per call) are extra requests, not extra official iterations.

---

## 5. Data protocol

**Symbols.** Let \(D\) be one participant. Split construction (before SA40) yields disjoint trial lists \(T_{\mathrm{tr}}\), \(T_{\mathrm{va}}\), \(T_{\mathrm{te}}\). **Observed** data is \(T_{\mathrm{obs}} = T_{\mathrm{tr}} \cup T_{\mathrm{va}}\). Test \(T_{\mathrm{te}}\) is held out.

**Split ratio 0.6 / seed 0.** `split_psych_experiment`: shuffle problem/game/round blocks with `numpy.random.default_rng(0)`, take \(n_{\mathrm{tr}} = \lfloor 0.6 \cdot n_{\mathrm{blocks}} \rfloor\) (clamped so val and test are nonempty), split the remainder about 50/50 val/test. History does not cross blocks except Kool (continuous session, contiguous day split). Speekenbrink default SA40 split is chronological (`--speekenbrink_split chronological`). mixed-gambles / external datasets use their loaders’ unit splits with the same ratio/seed.

**SA40.** After the split, `apply_structure_aware_protocol` keeps at most \(N=40\) observed trials, preserving task structure (`INDEPENDENT_TRIAL` / `RESETTING_UNIT` / `CONTINUOUS_SESSION` in `LIMITED_DATA_REGISTRY`). Test is never truncated.

**Kool cardinality.** Registry notes: history carries across days; the split is a contiguous usable-day cut (**split seed unused**). Under final ICLR **`structure_aware_v3`**, Kool retains **exact-40** observed train+val (stage-1 pairing keeps the continuous session without the v1 +1). The older v1 `structure_aware` path that could yield \(|T_{\mathrm{obs}}|=41\) via `kool_include_matching_stage1` is **historical / non-final** only.

**Prompt-example cap 60 vs observed cap 40.** Fitness always uses the complete retained \(T_{\mathrm{obs}}\). Prompts display a subsample of \(T_{\mathrm{obs}}\) of size \(\le 60\) (`cap_and_subsample_prompt_trials`; T-PICS `--max_prompt_train_trials`). Under `structure_aware_v3`, \(|T_{\mathrm{obs}}|\le 40\) (Kool exact-40 via stage-1 pairing), so the display cap 60 does not further truncate the retained union.

**Test isolation.** Evolution JSON (`trials_evolution_split.json`) stores only train+val. Test is `trials_test_posthoc.json` outside the evaluator directory. The generated evaluator never reads `data["test"]`. Test log-likelihood is computed in `run_participant` after selection.

**Participant ranges** (`emnlp_ordinal_range` / `teh_datasets.yaml`; inclusive ordinals into `valid_participant_ids`):

| alias | start | end | \(n\) people |
|---|---:|---:|---:|
| `1peterson2021using` | 0 | 49 | 50 |
| `2plonsky2018when` | 0 | 49 | 50 |
| `3frey2017cct` | 0 | 49 | 50 |
| `4wulff2018description` | 1290 | 1339 | 50 |
| `5speekenbrink2008learning` | 0 | 22 | 23 |
| `7hilbig2014generalized` | 0 | 49 | 50 |
| `10frey2017risk` | 0 | 49 | 50 |
| `11enkavi2019recentprobes` | 0 | 49 | 50 |
| `12badham2017deficits` | 0 | 9 | 10 |
| `mixed_gambles` | 0 | 49 | 50 |
| `bergert_nosofsky_2007` | 0 | 49 | 50 |
| `guan_2020_stopping` | 0 | 49 | 50 |
| `steyvers_2009_bandit` | 0 | 49 | 50 |
| `13schulz2020finding` | 0 | 49 | 50 |
| `14kool2016when` | 0 | 49 | 50 |

`apply_iclr_frozen_range_ordinals()` replaces leftover CLI defaults `(0, 49)` with the YAML range when `--participant_scope range` and `--ordinals` is unset.

---

## 6. Program interface

Generated programs must define:

```python
def choose(problem, history):
    ...
```

**Bernoulli** (all aliases except Steyvers and Schulz): return a Python `float` in \([0,1]\) equal to \(P(y=1)\). Action 0 is `problem["option_keys"][0]`; action 1 is `problem["option_keys"][1]`. Do not return a dict.

**Categorical:** return `dict[int, float]` over every valid id, nonnegative, finite, summing to 1. Valid ids = `[opt["action"] for opt in problem["options"]]` if present, else `problem["option_keys"]`. Steyvers: \(K=4\) (ids \(0..3\)). Schulz: \(K=8\) (ids \(0..7\)). A Bernoulli float is accepted only when \(K=2\).

**Neutral seeds.** Bernoulli: `persona_code_example/openevolve_vanilla/choices13k.py` returns `0.5`. Categorical: `persona_code_example/teh/categorical_uniform.py` returns uniform over `problem["options"]`.

**Meaning of arguments.** `problem` is current-trial task features. `history` is previously realized trials (actions and observed outcomes). Current-trial `trial["action"]` is **not** passed into `choose`; the evaluator uses it only as \(y\) after the call.

**Sanitization** (loaders + `sanitize_trial_for_program`; history of prior trials is not stripped of realized outcomes):

| Dataset | Removed from **current** `problem` | Notes |
|---|---|---|
| Speekenbrink | `weather_outcome`, `was_correct` (`_CURRENT_PROBLEM_EXCLUDED_FIELDS`) | Prior correctness may appear in history |
| Frey Risk | `outcome_marker`, `exploded` | Pump counts before the choice remain |
| Badham | `response_key`, `correct_category` | Feedback of **prior** trials may include category |
| Enkavi | `probe_in_set` (oracle) | Memory set + probe letter remain |
| Kool stage 1 | `planet`, `alien_options`, `stage1_action`, `spaceship`, `reward`, `treasure` | Unobserved until after stage 1 |
| Kool stage 2 | `reward`, `treasure` | Planet / stage-1 action are observed |
| Guan | `_stop_position`, `_full_values_len` popped before split return | `values_observed` is the prefix up to the current position |
| Schulz | current reward not on problem | History entries `{action, reward}` |
| Others | chosen `action` lives on the trial, not on `problem` | |

`compile_program` / generated evaluator `exec` use a restricted builtin set (no `open`, `import`, `print`). Programs cannot read split JSON or the unsanitized trial dict.

---

## 7. All 15 dataset adapters

| alias | Family | Return | \(K\) / ids | Seed | Compact examples (`format_trial_compact`) | SA40 class | Special handling |
|---|---|---|---|---|---|---|---|
| `1peterson2021using` | Peterson 2021 / 5-trial description problems | Bernoulli | 2 | choices13k.py | `gamble_A`/`B` | resetting problem | History accumulates within a problem; registry `task_description` |
| `2plonsky2018when` | CPC18 / 25-trial feedback problems | Bernoulli | 2 | choices13k.py | gambles | resetting problem | History resets between problems |
| `3frey2017cct` | Columbia Card Task | Bernoulli | 2 | choices13k.py | CCT round/flips | resetting round | Sequential flips |
| `4wulff2018description` | Description lotteries | Bernoulli | 2 | choices13k.py | cards/keys | independent | Empty history |
| `5speekenbrink2008learning` | Weather cards | Bernoulli | 2 | choices13k.py | keys | continuous session | Chronological split; outcome fields excluded from current problem |
| `7hilbig2014generalized` | Product ratings | Bernoulli | 2 | choices13k.py | ratings | independent | Empty history |
| `10frey2017risk` | BART pumps | Bernoulli | 2 | choices13k.py | balloon/pumps | resetting balloon | Explosion fields excluded from current problem |
| `11enkavi2019recentprobes` | Recent probes | Bernoulli | 2 | choices13k.py | memory_set/probe | independent | `probe_in_set` stripped |
| `12badham2017deficits` | Category learning | Bernoulli | 2 | choices13k.py | stimulus features | resetting rule block | `response_key` / `correct_category` stripped from current problem |
| `mixed_gambles` | Mixed gambles | Bernoulli | 2 | choices13k.py | gambles | independent | Always-empty history; registry `TASK_DESCRIPTION` |
| `bergert_nosofsky_2007` | Pairwise cues | Bernoulli | 2 | choices13k.py | bergert cues | independent | Empty history |
| `guan_2020_stopping` | Optimal stopping | Bernoulli | 2 | choices13k.py | env/pos/values | resetting problem | Prefix `values_observed`; analysis keys popped |
| `steyvers_2009_bandit` | 4-arm bandit | Categorical | 4, \(0..3\) | categorical_uniform.py | game/trial/n_arms | resetting game | History resets each game |
| `13schulz2020finding` | 8-arm bandit | Categorical | 8, \(0..7\) | categorical_uniform.py | round/trial/n_arms | resetting round | History `{action, reward}`; options reset each round |
| `14kool2016when` | Daw two-step | Bernoulli (both stages) | 2 | choices13k.py | stage 1/2 fields | continuous session | Same `choose()` for stage 1 and 2; history carries across days; `structure_aware_v3` exact-40 (stage-1 pairing) |

`vanilla_dataset_description()` uses registered `task_description` / `TASK_DESCRIPTION` for **all 15** aliases via `dataset_task_description()` (Psych-101 `PSYCH101_BINARY_DATASETS`, mixed-gambles `data_modules.mixed_gambles.TASK_DESCRIPTION`, external `EXTERNAL_DATASET_META`). The universal `# API` block is supplied separately by in-runner `choose_api_text()`. It never loads `prompts/openevolve_vanilla/*/infer_single_choice.txt`, `prompts/teh/`, or `prompts/external/` PICS strategy files (`reference_prompt` paths exist on dataset specs but are not used by this runner). Legacy vanilla infer files may still exist for `deprecated_run_openevolve.py --prompt_mode vanilla`; ICLR OE ignores them even if `--base_prompt` points at those paths.

---

## 8. Prompt construction

**No PICS automatic dataset-prompt generator and no PICS behavioral-strategy prompt.** No leakage-specific coaching beyond the neutral interface (no “do not read a current-trial label” sentence).

**Final user-message block order** (after the system message), exact headers from `_patched_build_prompt` / `truncate_vanilla_messages`:

1. `# Task (vanilla — no TEH prompt engineering)` — registered dataset `task_description` for all 15  
2. `# API` — tracked `choose_api_text()` (Bernoulli or categorical; not part of the task blurb)  
3. `# Output format (required for parser)` — one fenced full program  
4. `# Current program` — the mutable parent  
5. `# Current metrics` — parent metrics (omitted entirely if required+examples already overflow)  
6. `# Example trials (train+val only; compact)`  
7. `# Evaluation artifacts (official OpenEvolve; parent artifacts, not co-parents)` then `## Last Execution Output` (if nonempty and `include_artifacts`)  
8. `# Previous attempts (official OpenEvolve; contextual examples, not co-parents)` then `## Previous Attempts` (compact changes/performance/outcome; worker list = island top-`num_top`)  
9. `# Top performing programs (official OpenEvolve; contextual examples, not co-parents)` then `## Top Performing Programs` (3)  
10. `# Diverse programs (official OpenEvolve; contextual examples, not co-parents)` then `## Diverse Programs` (`random.sample` leftover of the island slice)  
11. `# Inspiration programs (official OpenEvolve; contextual examples, not co-parents)` then `## Inspiration Programs` (2, id-deduped)

**System messages (implementation detail).** The packer / patched LLM call uses: *You improve Python programs for human choice prediction. Return one fenced python code block containing the full revised program.* `_build_config` also sets `cfg.prompt.system_message` to *Evolve Python code for human choice prediction. Return a complete program file.* That Config string is **not** the chat system text once `PromptSampler.build_prompt` is patched.

Compact examples include `y=<action>` and `split=train|val`. That is **observed-example** supervision, not current-trial leakage into `choose()`.

---

## 9. Parent and contextual-program semantics

**One formal parent.** Official `_submit_iteration` → `database.sample_from_island(island_id, num_inspirations=num_diverse_programs)` returns one parent. The child is constructed with `parent_id=parent.id`. Our `VanillaProcessParallelController._submit_iteration` only stamps iteration into thread context and calls `super()`.

**Contextual examples, not co-parents.** Top / diverse / inspirations / previous-attempt text / artifacts are concatenated into the prompt. They are not crossover inputs and do not set additional `parent_id`s. Islands and MAP-Elites remain active even if the packer drops every optional block. The complete archive is never serialized into the prompt (worker snapshots exist for sampling only).

**Official sampling (411fb59).** Island programs sorted by `combined_score` descending. `programs_for_prompt = island[:num_top+num_diverse]`. Sampler: top = first 3; diverse = `random.sample(remaining, num_diverse)` (seeded global `random`; we call the same `random.sample`, not a score prefix). Inspirations from `sample_from_island`, then drop ids already shown as top/diverse.

---

## 10. Token-budget behavior

Chat-templated input (Qwen `apply_chat_template`, `add_generation_prompt=True`) must be \(\le 30000\). Output `max_tokens=1024`. Required: \(30000+1024 \le 32768\).

**Required (never trimmed):** system, task, API contract, current parent (plus the parser output-format block).

**Preserve examples ahead of optional context.** Fit required + all displayed examples first. Then greedily add optional whole blocks in official template order: artifacts → previous attempts → top programs (per program) → diverse → inspirations.

**Drop order (reverse):** inspirations → diverse → top → previous attempts → artifacts. The generate-time safety guard uses the same order, then drops example lines from the end only.

**Example reduction** (only if required+examples still exceed 30000 with **all** optional already omitted): deterministic prefix caps `(60, 40, 30, 20, 10, 5)` on the already-capped/subsampled display list (`ICLR_PROMPT_EXAMPLE_REDUCTION_CAPS`). Fitness is unchanged.

**Fail-fast.** `RequiredPromptOverflowError` if system/task/interface/parent alone exceeds the ceiling (or required + 5 examples still cannot fit). The worker iteration then errors; no LLM call.

**Diagnostics** (`prompt_truncation_diagnostics.jsonl`, **diagnostic/reporting only**): examples available/included; each optional block requested/included/dropped; estimated chat-template tokens; `trim_reason`; truncation steps.

**Audited bounds.** CPU tests cover all 15 interfaces with \(\le 60\) examples and program code \(\le 10000\) characters: required prompts fit without trimming examples; optional blocks may drop. Tokenizer: local HuggingFace cache of `Qwen/Qwen2.5-Coder-32B-Instruct`.

---

## 11. Evaluator and objective

The generated `evaluator.py` loads `trials_evolution_split.json` (full retained train+val). Let \(n_{\mathrm{tr}}=|T_{\mathrm{tr}}|\), \(n_{\mathrm{va}}=|T_{\mathrm{va}}|\), and \(\ell_{\mathrm{tr}}\), \(\ell_{\mathrm{va}}\) be mean per-trial log-likelihoods on those lists.

\[
\texttt{combined\_score}
= \frac{n_{\mathrm{tr}}\,\ell_{\mathrm{tr}} + n_{\mathrm{va}}\,\ell_{\mathrm{va}}}{n_{\mathrm{tr}}+n_{\mathrm{va}}}
\quad (n_{\mathrm{va}}>0;\ \text{else }\ell_{\mathrm{tr}}).
\]

If \(\ell_{\mathrm{va}}\) is non-finite, the pool falls back to \(\ell_{\mathrm{tr}}\). Non-finite combined raises (official failed-eval path).

**Bernoulli trial \(i\)** with label \(y_i\in\{0,1\}\) and clipped probability \(\tilde p_i=\mathrm{clip}(p_i,\varepsilon,1-\varepsilon)\), \(\varepsilon=10^{-9}\):

\[
\log p(y_i)=\; y_i\log\tilde p_i + (1-y_i)\log(1-\tilde p_i).
\]

**Categorical:** \(p_i=\tilde p_i(y_i)\) after coercing/renormalizing nonnegative mass on valid ids, then \(\log\tilde p_i\). Malformed outputs **raise** (not uniform). An explicit uniform dict is valid.

**Selection.** Maximize `combined_score` (official `get_fitness_score` prefers that key). `_find_best_program_by_observed_loglik` scans `checkpoint_*/programs/*.json` the same way after the run. Test is scored only on that frozen file.

**Local files per participant:** `openevolve_experiment/{initial_program.py,evaluator.py,config.yaml,trials_evolution_split.json,vanilla_task_prompt.txt,vanilla_interface_contract.txt}`, `openevolve_output/` (official checkpoints, `best/`), `best_program.py`, `results.json`, `trials_test_posthoc.json`, `prompt_truncation_diagnostics.jsonl`.

---

## 12. Invalid programs and failures

Official `Evaluator.evaluate_program` (411fb59) returns `{"error": 0.0}` or `{"error": 0.0, "timeout": True}` on timeout/exception/bad return, with **no** `combined_score`. `get_fitness_score` then uses `0.0`, which **beats** every valid negative log-likelihood.

**Adapter only:** `_adapt_official_failure_metrics_for_loglik` attaches

\[
\texttt{FAILED\_COMBINED\_SCORE}=\log(10^{-9})-1 \approx -21.723
\]

when official failure metrics lack `combined_score`. Any clipped valid mean is \(\ge \log\varepsilon \approx -20.723\), so the floor is strictly worse. Successful returns are unchanged. No repair, regeneration, or validity optimizer is added.

Checkpoint JSON: IEEE \(-\infty\) is not JSON-safe; the finite floor also avoids `json.dump` `-Infinity` → visualizer `None` dropping `combined_score`.

---

## 13. Parallelism and runtime

`--parallel_participants=1`: one person at a time in a thread pool of size 1. `--parallel_evaluations=4`: official `ProcessPoolExecutor(max_workers=4)` for **evolution iterations** (parent sample → LLM → eval). Approximate simultaneous vLLM generate requests \(\approx 1\times 4=4\). This is unrelated to PICS `--max_workers 100`.

OE is slow relative to PICS-at-width-10 because each iteration is a full-program rewrite, evaluation walks the full observed union in-process, islands keep a large population, and vLLM sees long prompts (up to 30000 input + 1024 output). Retries (`llm_retries=3`) multiply HTTP attempts on failures, not iteration count.

---

## 14. Outputs, checkpoints, resume, and W&B

**Run directory** (`openevolve_output_base_dir`):

- Psych-101: `generated_outputs/psych101_train/openevolve/{alias}/run_{YYMMDD_HHMMSS}/`
- mixed-gambles: `generated_outputs/mixed_gambles/openevolve/run_*`
- external: `generated_outputs/external/{alias}/openevolve/run_*`

Important files: `run_config.json`, `log/` SA40 manifests, `participant_details_loglik.csv`, `summary_loglik.csv`, `final_participant_summary.csv`, per-person trees above.

**Resume.** No first-class `--resume`. A new timestamped directory starts fresh. Official `OpenEvolve.run` can continue from `database.last_iteration` if the **same** `openevolve_output` tree is reused via `--output_dir`; that is library behavior, not an ICLR workflow.

**W&B.** Project `openevolve` (`WANDB_PROJECT`). Run name `{dataset}_{timestamp}`. Disable with `--no_log` (`WANDB_DISABLED=true`). **Diagnostic/reporting only** while running: `avg_test_loglik` and other `avg_*` over completed rows. **Paper-safe** `final/mean_test_loglik` is written only when `wandb_completion_fields` sees every expected person with a finite test loglik and zero failures (`final/is_complete=true`). Dataset mean is the equal-person mean of person-level test log-likelihoods.

---

## 15. Official behavior versus our adapter

| Topic | Official 411fb59 default | Ours | Class | Scientific effect |
|---|---|---|---|---|
| Search | 1 parent, islands, MAP-Elites | Unchanged | Necessary (use the library) | Same parent semantics |
| Child `parent_id` | `parent.id` | Unchanged | Necessary | One formal parent |
| Top / diverse / inspirations | 3 / 2 / 2 | Same counts + `random.sample` | Frozen scientific setting | Contextual examples |
| Previous attempts | Always filled from island top-3 | Packed if they fit | Frozen scientific setting | Compact history, not extra parents |
| Artifacts | `include_artifacts=True` | Same; skip empty | Frozen scientific setting | Optional execution text |
| Evolution mode | Diff | Full rewrite | Discretionary / task-fit | Whole `choose()` files |
| Template stochasticity | On | Off | Discretionary reproducibility | Fixed prompt wording |
| Prompt layout | Stock rewrite template | Vanilla task+API+examples then optional | Necessary task interface | Observed-data few-shot |
| Observed examples | None | Cap 60 from train+val | Necessary fairness vs T-PICS | Prompt supervision ≠ fitness cap |
| Failure metrics | `{error: 0.0}` | Floor \(\log\varepsilon-1\) | Necessary for NLL | Failures cannot win |
| Data | User evaluator | SA40 + split 0.6/0 | Frozen scientific setting | Same protocol as main method |
| Token packing | None | 30000 drop-optional-first | Necessary vs 32768 server | No silent mid-string truncation |
| Model | Caller | Qwen2.5-Coder-32B-Instruct, vLLM | Frozen scientific setting | Shared model family with other ICLR runs |
| Concurrency | `parallel_evaluations=1` | 4; people=1 | Discretionary runtime | Throughput, not extra candidates |
| Candidate budget | `max_iterations=10000` | 350 | Frozen scientific setting | Matched nominal full-pipeline generation budget (generous to OE; see §4) |
| Cascade / LLM feedback | cascade on; LLM feedback off | cascade off; feedback off | Implementation detail | Single loglik stage |

---

## 16. Fairness and leakage guarantees

- Same SA40 + split 0.6/seed 0 + YAML ordinals as the main method.  
- Fitness is **participant-specific**, never pooled across people.  
- Test is absent from prompts, evaluator JSON, `combined_score`, early stopping (`patience=None`), parent selection, and checkpoint best-program scan.  
- Interface text is Bernoulli vs categorical mechanics only; no PICS strategies.  
- Valid comparisons: same people, same observed/test splits, same SA40 rules, matched nominal full-pipeline generation budget of 350 (not equal wall-clock or equal per-participant compute). T-PICS: 200 shared (G.1+G.2) + 150 participant-specific (G.3+person). OE: all 350 participant-specific (favorable/generous to OE).

---

## 17. Commands and preflight

**Examples only. Do not launch from this document.**

Environment: `VLLM_LOCAL_URL` (default `http://localhost:8000/v1`), `VLLM_LOCAL_API_KEY` (default `EMPTY`). Optional W&B standard keys if logging.

Dependency check:

```bash
conda activate evo310   # or the project’s 3.12 env
bash scripts/check_environment.sh
git -C reference_repos/openevolve rev-parse HEAD
# expect 411fb59c886c18704caaffb611e17cf9e7d824d2
```

Dry-run / preflight (no GPU): `python baseline_methods/Psych101/run_openevolve.py --help` (argparse exits before the SHA check). There is no `--dry-run` flag. A real `main()` without the checkout `SystemExit`s in `require_openevolve_checkout()`.

Expected vLLM (example host script `scripts/utils/vllm_3090_gpu.sh`; **do not run here**): `vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000 --tensor-parallel-size 4 --max-model-len 32768`.

Canonical production template (substitute `ALIAS`, `START`, `END` from §5):

```bash
python baseline_methods/Psych101/run_openevolve.py \
  --dataset ALIAS \
  --psych_dataset_split train \
  --participant_scope range \
  --range_start_ordinal START \
  --range_end_ordinal END \
  --split_ratio 0.6 \
  --split_seed 0 \
  --n_iterations 350 \
  --parallel_participants 1 \
  --parallel_evaluations 4 \
  --limited_data_protocol structure_aware_v3 \
  --limited_train_val 40 \
  --max_prompt_train_trials 60 \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --llm_max_tokens 1024 \
  --hard_prompt_token_cap 30000 \
  --max_model_len 32768 \
  --num_diverse_programs 2 \
  --num_top_programs 3 \
  --api_base http://localhost:8000/v1
```

Do not pass `--max_workers`. Do not use `10×10` participant/eval concurrency, `--n_iterations 600`, `--hard_prompt_token_cap 14000`, `--max_model_len 16384`, or `--limited_data_protocol structure_aware` for final ICLR results.

### Remaining launch requirements (final ICLR)

1. Pin checkout `reference_repos/openevolve` at `411fb59c886c18704caaffb611e17cf9e7d824d2` (`require_openevolve_checkout` SystemExits on miss/mismatch; this host currently may show a different HEAD until checked out).
2. Serve Qwen2.5-Coder-32B-Instruct with `--max-model-len 32768`.
3. Launch **all 15** datasets with the template above and EMNLP ordinals from §5.
4. Treat any prior 14k / 16384 / `structure_aware` / 600-iteration / incomplete W&B run as **non-final**.
5. Quote paper means only when `final/is_complete=true` for the full expected cohort.
---

## 18. Tests and audits

| File | Protects |
|---|---|
| `utils/teh_psych/test_iclr_baseline_guards.py` | 15 aliases, SA40 `structure_aware_v3` (Kool exact-40), test isolation, failure floor, interface source of truth, SHA miss/mismatch, official `random.sample` diverse split, one `parent_id`, frozen CLI |
| `utils/teh_psych/test_openevolve_prompt_budget.py` | 30000 packing, early/mixed/max-10k programs, no example trim under audited bounds, coaching-sentence absence, required overflow |
| `utils/teh_psych/test_openevolve_failure_checkpoint_roundtrip.py` | CPU evaluator→database→checkpoint using the **audit** clone (skips if missing) |

These two modules currently define **43 tests** (34 + 9). `test_all_15_use_registered_task_descriptions_not_vanilla_files` asserts every ICLR alias uses `dataset_task_description` and that Choice13k / mixed-gambles vanilla infer files are not loaded as `# Task` text. `test_openevolve_failure_checkpoint_roundtrip.py` skips unless `reference_repos/openevolve_official_audit` exists.

`analysis_2026Sep/Sep19_3090_server/openevolve/*` audits are **supplementary and stale in places** (e.g. 600 iterations). They are not runtime dependencies.

---

## 19. Known limitations and future-change policy

- Full rewrite and disabled template stochasticity are frozen ICLR choices, not official OE defaults.  
- 350 is a matched **nominal full-pipeline generation budget** versus gated T-PICS, not equal per-participant compute; the accounting is generous to OE (all 350 OE candidates are participant-specific).  
- Production depends on a gitignored pinned checkout the runner will not create.  
- Final ICLR launches must use `--limited_data_protocol structure_aware_v3` (Kool exact-40). Historical `structure_aware` / 14k / 16384 / 600-iteration runs are **non-final** and must not be reported as paper results.  

- No first-class resume CLI.  
- Optional previous-attempt section is official island-best programs, not a true parent-lineage trace.

**Future OpenEvolve experiments must use new files/configs** (new runner, new freeze constants, or a new docs revision) rather than silently editing this frozen ICLR implementation.

---

## Documentation vs code discrepancies

These are not silent fixes; the code behavior is what this document describes.

1. Ignored `analysis_2026Sep/.../AUDIT_runtime_correctness_fairness.md` still discusses `--n_iterations 600` and pre-freeze defaults. Runtime freeze is 350 and the table in §3.  
2. Ignored `ICLR_FROZEN_COMMANDS.md` omits `--include_artifacts` (runtime default `True`).  
3. `FOCUS_DATASETS` in the runner is unused (commented as documentation only).  
4. Official PromptSampler `previous_programs` is the island top-3 by score, not chronological children of the current parent; we pack that list as official does.  
5. `scripts/check_environment.sh` expects Python 3.12; OE CPU tests on this host used conda env `evo310`.  
6. `_build_config` writes `cfg.prompt.system_message` (“Evolve Python code…”) but the patched sampler replaces the LLM system text with the packer string (“You improve Python programs…”). The Config field is unused for the actual chat once the patch is installed.
