# Psych-101 remaining train datasets — compatibility summary

**Date:** 2026-06-17 (step 2)  
**Source:** `marcelbinz/Psych-101` train split (60,092 participant rows; 76 unique `experiment` ids)  
**Excluded:** 10 implemented datasets from [`current_10_dataset_code_audit.csv`](current_10_dataset_code_audit.csv)  
**Artifacts:** [`psych101_remaining_dataset_inventory.csv`](psych101_remaining_dataset_inventory.csv), [`psych101_remaining_dataset_examples.json`](psych101_remaining_dataset_examples.json)

---

## Headline counts

| Metric | Count |
|--------|------:|
| Remaining train experiments (unique `experiment` ids) | **62** |
| Remaining participant rows | **40,746** |
| Compatible with current binary `choose(problem,history)->float` API (**yes**) | **23** |
| Uncertain binary fit | **4** |
| Clearly **not** binary (`no`) | **35** |

### Compatibility buckets (experiments)

| Bucket | Count | Meaning |
|--------|------:|---------|
| **direct_after_parser** | 22 | Binary TEH API + new NL parser (or wire existing parser) |
| **small_schema_extension** | 3 | Binary API but sequential/hierarchical state (sampling, kool MDP) |
| **multi_action_api** | 21 | Needs `K`-way categorical choice API |
| **scalar_api_extension** | 3 | Needs scalar probability/rating output |
| **api_extension** | 4 | Mixed phases (e.g. garcia: choice + probability reports) |
| **out_of_scope** | 7 | Sequences, grid search, go/no-go omission, verbal 9-way labels, etc. |
| **uncertain** | 2 | wu2023chunking (4-key instruction following) |

### Effort type (conservative)

| Work type | Experiments | Notes |
|-----------|------------:|-------|
| **Parser + registry only** (current binary API) | **25** | `direct_after_parser` + `small_schema_extension` |
| **Parser + multi-action API** | 21 | Same trial dict shell possible; eval must change |
| **Parser + scalar / mixed API** | 7 | `scalar_api_extension` + `api_extension` |
| **Unlikely worth TEH/PICS without redesign** | 7 | `out_of_scope` |
| **Needs manual review** | 2 | `uncertain` |

---

## Answers to key questions

### How many remaining Psych-101 train datasets are there?

**62** unique experiment ids on the train split (after excluding the implemented 10).  
Total train corpus: **76** experiments; **14** are the implemented 10 plus **2** additional registry entries with parsers already in code (`enkavi2019recentprobes`, `badham2017deficits`) that were outside the step-1 “top 10” scope.

### How many look directly compatible with the current binary TEH API?

**23 experiments** are classified **`current_binary_api_compatible=yes`** — each has a clear two-option (or two-option-per-stage) press target suitable for `P(action=1)` log-likelihood, assuming a parser can extract `problem`, `history`, and `action`.

**Not counted as direct:** 4 `uncertain` (hierarchical kool tasks, wu2023chunking) and 35 `no`.

### How many need only parser/schema work?

**25** experiments fit **`direct_after_parser`** (22) or **`small_schema_extension`** (3) — pipeline/eval can stay binary; effort is dominated by NL parsing and prompt/schema fields.

Two of these (**enkavi recent probes**, **badham category learning**) already have parsers in `psych101_parsers.py` and only need registry wiring + validation.

### How many require changing the synthesized program output type?

**31** experiments need more than binary Bernoulli log-lik:

| Output change | Count |
|---------------|------:|
| **K-way categorical** (`multi_action_api`) | 21 |
| **Scalar probability/rating** (`scalar_api_extension`) | 3 |
| **Mixed phases** (`api_extension`, garcia) | 4 |
| **Out of scope** (sequences, grid, go/no-go, 9-way verbal) | 7 |

### How many seem unsuitable or ambiguous?

| Class | Count | Examples |
|-------|------:|---------|
| **Unsuitable / out-of-scope** | 7 | digit span, go/no-go withhold, kumar 49-cell grid, collsiöö 9-way verbal, jansen global self-reports |
| **Ambiguous** | 4 | kool hierarchical MDP (timeouts), wu2023chunking (4-key + RT) |
| **Scalar-only** | 3 | zhu Bayesian estimates, wise shock ratings |

---

## Main repeated task families (remaining 62)

| Task family | ~Experiments | Typical response | Binary TEH? |
|-------------|-------------:|------------------|-------------|
| **Two-arm bandit / slot machines** | 10+ | 2 keys | Yes |
| **Multi-arm bandit (3–8 arms)** | 12+ | 3–8 keys | No (K-way) |
| **Graph / spatial MDP (tomov)** | 6 | 3–5 direction/door keys | No |
| **Hierarchical spaceship MDP (kool)** | 4 | staged binary/multi | Uncertain / multi |
| **Category / rule learning** | 4 | 2 category keys | Yes (levering, badham) |
| **Memory / n-back / probes** | 4 | 2 keys or sequence | Mixed (probes yes; digit span no) |
| **IGT / multi-deck (steingroever)** | 3 | 4 decks | No |
| **Probability elicitation** | 5 | typed % | Scalar API |
| **Episodic judgment (popov)** | 3 | 2 judgment keys | Yes |
| **Experiential learning (garcia)** | 4 | multi-key + % reports | Mixed API |
| **Instruction / RT (wu chunking)** | 2 | 4 keys + RT | Uncertain |

---

## Parser/schema reuse (most leverage)

| Existing parser / schema | Best remaining targets |
|--------------------------|------------------------|
| **`slot_machine_bandit` / wilson** | gershman2018, somerville, waltz, xiong, feng, lefebvre, zorowitz |
| **`badham_category_learning`** | levering2020revisiting (both exps) |
| **`enkavi_recent_probes`** | enkavi2019recentprobes (already implemented) |
| **`lottery_offers` / wulff** | ruggeri2022, wulff2018sampling (with sampling phase) |
| **`product_ratings` / hilbig** | ludwig2023human (vitamin vectors) |
| **`generic_press_parser` helpers** | cox2017, popov2023, many bandits |
| **New shared multi-arm parser** | schulz2020 (8 arms), bahrami (4), steingroever (4), tomov (3–5) |

---

## Top blockers for scaling 10 → all compatible Psych-101 train data

1. **Manual NL parser authoring** — still one parser (or family) per transcript format; 62 remaining formats vs 10 done.

2. **Multi-action prevalence** — **21/62** remaining experiments are **3+ way** choices; current `evaluate_choice13k_program` cannot score them without a **K-way log-lik** extension.

3. **Non-choice outputs** — scalar probability reports (zhu, wise) and verbal categories (collsiöö) do not fit Bernoulli action log-lik.

4. **Trial segmentation ambiguity** — hierarchical tasks (kool, garcia multi-part, wulff sampling) require conventions for what counts as one supervised trial.

5. **Omission / sequence responses** — go/no-go withhold, digit span, kumar grid — not single discrete choices.

6. **Validation cost** — participant scanning re-parses every row; 40k+ remaining rows magnifies this.

7. **Prompt/schema coverage** — schema types A–D and subtypes do not cover IGT, 8-arm bandits, navigation MDPs without new prompt branches.

---

## Required vs accidental gaps

| Required (task-driven) | Accidental (engineering) |
|------------------------|---------------------------|
| K-way APIs for 3–30 option tasks | No generic multi-arm eval helper yet |
| Scalar API for probability ratings | `format_trial_for_prompt` schema gaps |
| Sequence/grid tasks out of scope | Registry only lists 12 datasets though 76 exist |
| Hierarchical trial definitions | Parsers for enkavi/badham exist but not in “10” rollout |

---

## Recommended next implementation order

Prioritize **high row-count**, **binary-compatible**, **parser reuse**:

| Priority | Experiment(s) | Why |
|----------|---------------|-----|
| **P0 — wire existing parsers** | `enkavi2019recentprobes/exp1.csv`, `badham2017deficits/exp1.csv` | Parsers already in `psych101_parsers.py`; low risk |
| **P1 — easy binary, large N** | `ruggeri2022globalizability/exp1.csv` (11.9k rows) | Binary intertemporal; gamble-like |
| **P1** | `gershman2018deconstructing/exp1-2`, `somerville2017charting`, `xiong2023neural`, `feng2021dynamics` | Standard 2-arm bandit pattern |
| **P2 — binary + medium parser** | `cox2017information`, `popov2023intent` (×3), `levering2020revisiting` (×2), `lefebvre2017behavioural` (×2), `ludwig2023human` (×3), `zorowitz2023data` | Clear 2-option trials; reuse bandit/category parsers |
| **P3 — binary sequential** | `wulff2018sampling`, `waltz2020differential`, `enkavi2019adaptivenback` | Need state/history handling |
| **Platform — before broad rollout** | Multi-action eval (`K`-way log-lik) | Unlocks schulz (×5), steingroever (×3), tomov (×6), bahrami, wu2018, hebart, etc. |
| **Defer** | scalar (zhu, wise), out-of-scope (kumar, digitspan, gonogo, collsiöö, jansen), garcia mixed API | Need API redesign or different objective |

---

## Bottom line

The **TEH loading/eval shell generalizes**, but the **remaining Psych-101 train corpus is heterogenous**:

- Only **~37%** (23/62) of remaining experiments are clearly **binary-compatible** today.
- Another **~34%** (21/62) need a **multi-action program API** before log-lik training makes sense.
- **~11%** (7/62) are **poor fits** for trial-level choice prediction without a major task reformulation.

Scaling to “all of Psych-101 train” is therefore **two parallel tracks**: (1) parser factory for binary-compatible families, and (2) **eval/API extension** for multi-option and scalar tasks — not more `teh.py` branching.
