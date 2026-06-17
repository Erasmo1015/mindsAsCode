# Psych-101 train corpus — reconciled coverage (76 experiments)

**Date:** 2026-06-17 (step 4)  
**Source:** `marcelbinz/Psych-101` train split  
**Detail:** [`psych101_76_reconciled_coverage.csv`](psych101_76_reconciled_coverage.csv)

---

## Reconciling the step 2 / step 3 count mismatch

| Counting unit | Categorical-choice addressable | Notes |
|---------------|-----------------------------:|-------|
| **Logical datasets** (step 3) | **54** | 10 implemented aliases + 44 remaining targets in categorical fit CSV |
| **HF `experiment_id` rows** (this table) | **59** | Wilson is **5** separate HF files under one alias; `wu2023chunking/exp2` added as categorical |
| **Non-categorical** | **17** | Sum with 59 = **76** |

Step 3’s “54/76” used **dataset aliases** (~74% of experiments). At the **HF experiment_id** level, categorical coverage is **59/76 (78%)**.

---

## Summary buckets (sum = 76)

| Bucket | Count | Experiment IDs (representative) |
|--------|------:|--------------------------------|
| **categorical_choice_addressable** | **59** | peterson, plonsky, ruggeri, hebart, gershman2018, schulz, tomov, wilson×5, … |
| **scalar_rating** | **3** | `zhu2020bayesian/exp1-2`, `wise2019acomputational/exp1` |
| **mixed_choice_and_rating** | **4** | `garcia2023experiential/exp1-4` |
| **sequence_reconstruction** | **1** | `enkavi2019digitspan/exp1` |
| **omission_gonogo_rt_special** | **1** | `enkavi2019gonogo/exp1` |
| **grid_search_highK_impractical** | **1** | `kumar2023disentangling/exp1` |
| **verbal_free_text_category** | **3** | `collsiöö2023MCPL/exp1-3` |
| **global_self_report** | **1** | `jansen2021dunningkruger/exp1` |
| **compound_action_special** | **1** | `krueger2022identifying/exp1` |
| **uncertain** | **2** | `kool2016when/exp1`, `kool2017cost/exp1` |
| **Total** | **76** | |

---

## Categorical API + parser plan (59 experiments)

Among **59** categorical-addressable `experiment_id`s:

| Parser plan feasibility | Count | % of categorical |
|-------------------------|------:|-----------------:|
| **easy_plan** | 25 | 42% |
| **medium_plan** | 27 | 46% |
| **hard_plan** | 7 | 12% |

**Version B target population (easy + medium):** **52 / 76** experiments (68% of full corpus), **88%** of categorical-addressable experiments.

**Hard_plan categorical (7):** `frey2017risk`, `tomov2020discovery` ×4, `kool2016when/exp2`, `kool2017cost/exp2` — need state machines or variable-K segmentation; still categorical but higher adapter risk.

---

## Needed output API (non-categorical 17)

| API | Count |
|-----|------:|
| `dict[int,float]` (categorical) | 59 |
| `scalar` | 3 |
| `mixed` (choice + scalar phases) | 4 |
| `sequence` | 1 |
| `withhold_or_press` (go/no-go) | 1 |
| `categorical_K49` / impractical high-K grid | 1 |
| `categorical_verbal` (free-text labels) | 3 |
| `compound_action` | 1 |
| `dict_or_omit` (uncertain) | 2 |

---

## Implications for scaling

1. **Version B parser-plan adapter** can target **59** categorical experiments without changing the PICS output API.
2. **Realistic first-wave automation:** **52** experiments (easy + medium plans) before tackling hard_plan MDP/sequential tasks.
3. **17** experiments need different task types, human rules, or explicit deferral — not fixable by categorical dict API alone.
