# Current 10 Psych-101 dataset code audit

**Date:** 2026-06-17  
**Scope:** `1peterson2021using` … `10frey2017risk` (numbered Psych-101 binary aliases).  
**Inspected:** `teh.py`, `data_modules/psych101_binary.py`, `data_modules/psych101_parsers.py`, `utils/teh/teh_datasets.py`, `utils/teh/teh_runtime.py`, `utils/teh/participant_ids.py`, `data_modules/choice13k.py` (peterson/plonsky gamble helpers).

Per-dataset detail: [`current_10_dataset_code_audit.csv`](current_10_dataset_code_audit.csv).

---

## Executive summary

The **TEH evolution/evaluation path in `teh.py` is almost entirely dataset-agnostic** for these 10 datasets. Every Psych-101 alias follows the same chain:

1. Filter HF rows by `experiment_id` (`get_filtered_psych101_split`)
2. Parse NL transcript → `PsychExperiment` via registry `parser` id (`parse_psych101_binary_row` / `PARSER_DISPATCH`)
3. Block-level train/val/test split (`split_psych_experiment`, with pseudo-block expansion when needed)
4. Trial dicts with `problem`, `history`, `options`, `action` (`experiment_to_trial_dicts` / `trials_from_blocks`)
5. Program API `choose(problem, history) -> float` and log-lik eval (`evaluate_choice13k_program`)

**Specialization concentrates in NL parsing and prompt/schema documentation**, not in `teh.py` per-dataset branches.

---

## How many use the same shared loading / input pipeline?

**10 / 10** use the same shared Psych-101 loading and trial-dict pipeline:

| Layer | Shared entry points |
|-------|---------------------|
| HF load + filter | `get_filtered_psych101_split`, `_load_hf_split` |
| Per-participant load | `get_psych101_binary_experiment` |
| Parse dispatch | `_parse_row` → `PARSER_DISPATCH[spec["parser"]]` |
| Split | `split_psych_experiment` (+ `_expand_single_block_to_pseudo_blocks`) |
| Trial API | `experiment_to_trial_dicts` / `trials_from_blocks` |
| TEH load in `teh.py` | `_psych101_trials_for_participant` → `_trials_for_loglik_participant` |
| Evaluation | `_evaluate_loglik_for_dataset` → `evaluate_choice13k_program` |
| Prompts | `utils/teh/teh_runtime.setup_teh_run_prompts` + `summarize_runtime_schema_for_prompt` / `format_trials_for_prompt` |

`teh.py` branches on `is_psych101_dataset(dataset)` and `is_binary_loglik_dataset(dataset)` — **not** on individual dataset names (except legacy/unused `cpc18`/`gridworld` paths and peterson-only training modes below).

---

## How many require specialized engineered code?

**10 / 10** require a **dataset-specific NL parser** registered in `PSYCH101_BINARY_DATASETS` and implemented in `psych101_parsers.py`.

However, specialization is **layered**:

| Count | Category | Datasets |
|-------|----------|----------|
| **10** | Registry row (`experiment_id`, `schema_type`, `parser`, `task_description`) | all |
| **10** | Dedicated parser function | all |
| **3** | Reuse gamble helpers from `choice13k.py` | peterson, plonsky, wulff (schema A) |
| **2** | Share bandit regex / game structure | sadeghiyeh, wilson (schema C) |
| **2** | Schema D sequential state machines | frey CCT, frey balloon risk |
| **3** | Single-block transcripts → pseudo-block split | speekenbrink, hilbig, flesch |
| **1** | Non-target trial filtering (instructed → context only) | wilson |
| **1** | HF filter uses `startswith` not equality | wilson (`experiment_id` ends with `/`) |

**Shared post-parse code covers all 10** — no dataset-specific branches in split, history construction, or log-lik evaluation.

---

## Main kinds of specialization

### 1. NL transcript parsers (required — format differs)

Each dataset’s HuggingFace `text` field has a distinct layout. Parsers use regex/block splitting to extract:

- **Gamble / lottery tasks (schema A):** outcome probabilities, option keys, feedback lines (peterson, plonsky, wulff)
- **Feature choice (schema B):** cards, ratings, tree features, training phase (speekenbrink, hilbig, flesch)
- **Bandits (schema C):** game headers, machine labels, instructed vs free trials (sadeghiyeh, wilson)
- **Sequential risk (schema D):** round/balloon state, flip/stop or pump/stop keys (frey CCT, frey balloon)

### 2. Schema-type prompt / semantics branches (partially generalizable)

`psych101_binary.py` centralizes schema-aware behavior:

- `summarize_runtime_schema_for_prompt` — documents observed `problem` keys
- `_action_semantics_for_schema` — action=0/1 meaning by schema (and schema-B subtype)
- `format_trial_for_prompt` — one-line trial examples for LLM prompts

These are **schema-driven**, not dataset-name-driven, but **schema D currently assumes CCT fields** in prompt formatting (balloon risk is under-served — accidental hard-coding).

### 3. Trial-target and history semantics (mixed: required vs engineered)

- **Shared:** history resets per block; entries carry `action`, optional `feedback`, and trial `problem_fields`
- **Wilson-specific (task-required):** instructed trials excluded from supervised set but injected as `instructed_context` / `within_game_action_reward_history` in `problem`
- **Pseudo-blocks (engineered workaround):** single long blocks chunked for split when `len(blocks)==1`

### 4. Accidental hard-coding (could be abstracted)

| Location | Issue |
|----------|-------|
| `teh.py` | `PETERSON2021USING_ALIAS` — default `--dataset`, only `across_participants` target |
| `participant_ids.py` | `3frey2017cct` writes legacy compat path under `datasets/psych101/frey2017cct/` |
| `get_filtered_psych101_split` | Trailing `/` on `experiment_id` triggers `startswith` filter (wilson only today) |
| `format_trial_for_prompt` | Schema D branch is CCT-specific; balloon risk lacks dedicated subtype |
| `teh_runtime.py` | Default seed program path points at `choices13k.py` (naming legacy, works for all via schema-neutral prompts) |

None of these are needed for correctness of the shared pipeline; they are coupling / convenience.

---

## Reusable for broader Psych-101

**Ready to reuse without per-dataset `teh.py` changes:**

1. **HF ingestion** — `load_dataset` / `load_from_disk`, train/test split selector, row filter by `experiment_id`
2. **Registry + dispatch** — add row to `PSYCH101_BINARY_DATASETS`, wire parser id in `PARSER_DISPATCH`
3. **`PsychExperiment` → trial dict contract** — uniform `choose(problem, history)` input
4. **Block-level split** — `split_psych_experiment` with pseudo-block fallback
5. **Participant validation** — `collect_psych101_valid_participants` (parse + split check per row)
6. **Evaluation** — Bernoulli log-lik on `action` ∈ {0,1}
7. **Prompt scaffolding** — schema summary from parsed trials; gamble vs non-gamble base prompt selection in `teh_runtime`
8. **Parser helper library** — press-key extraction, gamble parsing, game-header splitting (already shared across subsets)

**Adding dataset 11+ is “register + implement parser + validate participants”**, not “fork `teh.py`”.

---

## Likely blockers for scaling to all Psych-101 train datasets

1. **Parser authoring bottleneck** — every new experiment needs hand-written regex/state logic; no generic NL→structured-trial layer. This is the dominant cost.

2. **Binary press-key assumption** — pipeline assumes two `option_keys` and `action` ∈ {0,1}. Multi-option, continuous, or non-press responses need API extension.

3. **Supervision target ambiguity** — wilson shows that some transcripts mix instructed demonstration and free choice; scaling requires a general rule (or metadata) for which lines are prediction targets.

4. **Block / problem definition** — “block” semantics vary (gamble pair, game, round, balloon, single long sequence). Pseudo-block chunking is a heuristic, not ground truth.

5. **Schema taxonomy gap** — only four schema types (A–D) in prompts; new task families need new schema branches or subtype detection (as with schema B: weather / tree / product).

6. **Prompt formatting drift** — `format_trial_for_prompt` must stay aligned with actual `problem` keys; missing subtypes (balloon vs CCT) can mislead the LLM.

7. **Validation cost** — `collect_psych101_valid_participants` re-parses every HF row sequentially; scales poorly as corpus grows.

8. **Metadata / ID edge cases** — wilson’s directory-style `experiment_id` needed a special filter; other HF rows may have similar quirks.

9. **Unimplemented registry entries** — `PSYCH101_BINARY_DATASETS` already lists datasets 11–12 with parsers, but scope here is 1–10; remaining Psych-101 train experiments are not yet in the registry at all.

---

## Required vs accidental specialization

| Required (task format differs) | Accidental (could abstract) |
|--------------------------------|-----------------------------|
| Per-dataset NL parser | Peterson-only `across_participants` guard |
| Gamble vs bandit vs CCT vs balloon field schemas | Default seed filename `choices13k.py` |
| Wilson instructed-trial handling | Frey CCT compat valid-id JSON path |
| Stateful sequential parsing (CCT, balloon) | Schema D prompt lines assuming CCT only |
| Participant-specific option key order (speekenbrink, hilbig) | Numbered CLI aliases vs HF `experiment_id` string (already normalized via registry) |

---

## Bottom line

| Question | Answer |
|----------|--------|
| Is `teh.py` mostly general? | **Yes** — one Psych-101 code path for load, split, evolve, evaluate. |
| Where is complexity? | **`psych101_parsers.py`** (+ registry metadata and schema-aware prompt helpers). |
| Can it scale beyond 10? | **Architecturally yes** via registry + parser plug-ins; **operationally** scaling is blocked by manual parser engineering, binary-trial assumptions, and prompt/schema coverage — not by `teh.py` dataset branches. |

**Suggested next steps (analysis follow-up, not implemented here):** inventory unimplemented Psych-101 train `experiment_id`s; cluster by transcript similarity to existing parsers; flag non-binary or multi-action experiments early.
