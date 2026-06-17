# Generalized categorical API — feasibility analysis

**Date:** 2026-06-17 (step 3)  
**Proposed program API:**

```python
def choose(problem, history) -> dict[int, float]:
    # Maps normalized action IDs 0..K-1 to probabilities; sums to 1
```

**Related artifacts:** [`generalized_categorical_api_dataset_fit.csv`](generalized_categorical_api_dataset_fit.csv), [`generalized_categorical_adapter_schema.json`](generalized_categorical_adapter_schema.json), [`generalized_categorical_eval_plan.md`](generalized_categorical_eval_plan.md)

---

## Executive answer

**Yes — the unified categorical dict API is a plausible single output design for both current binary datasets and multi-action Psych-101 choice datasets**, provided that:

1. **The adapter** normalizes each trial’s valid actions into `problem["options"]` with consecutive integer IDs `0..K-1`.
2. **The evaluator** reads `K` **per trial** from `len(problem["options"])`, not from a dataset-global constant.
3. **Offline adapter** (LLM-assisted or rule-based) produces structured trials once; **no LLM at program runtime**.

**Recommendation: prototype first** — implement evaluator + schema validation + adapter spec on a small binary/multi mix before committing to repo-wide migration. Do **not** defer the API idea; defer **full corpus adapter coverage**.

---

## 1. Does `dict[int, float]` cover binary + multi-action?

| Case | Mapping | Log-lik |
|------|---------|---------|
| **Binary (current 10 + 23 remaining)** | `{0: 1-p, 1: p}` where `p = P(action=1)` | `log p[y]` same as current Bernoulli on `y∈{0,1}` |
| **K-way categorical (21 multi experiments)** | `{0: p0, …, K-1: p_{K-1}}` | `log p[y]` for observed `target_action` |

**Implemented 10 datasets** already use integer `action` 0/1 with `option_keys` order — they lift cleanly to `K=2` dict output with no semantic change.

**Coverage within analyzed inventory (44 experiments):**

| `generalized_categorical_api_fit` | Count | Notes |
|-----------------------------------|------:|-------|
| **full** | 39 | Standard categorical choice after adapter |
| **partial** | 5 | Needs `is_prediction_target=false`, stage segmentation, or large/variable K handling |
| **insufficient** | 0 | Among the 44 binary+multi targets |

**Plus 10 already-implemented binary datasets** → **54 / 76** train experiments are addressable with this API **if adapters/parsers exist**. Step-2’s remaining **18** experiments still need other output types or are out of scope (scalar, sequences, omission, grid 49-way, verbal labels, compound actions).

---

## 2. Trial representation checklist (44 datasets)

For each analyzed dataset, the adapter **can** emit:

| Field | Supported? |
|-------|------------|
| `target_action: int` | Yes for all 44 (null only when `is_prediction_target=false`) |
| Action space `0..K-1` | Yes; K fixed per trial via `problem["options"]` |
| `problem["options"]` with `action`, `raw_key`, `description` | Yes — required design choice |
| Optional `history` | Yes — built deterministically from prior trials in block |

**Partial-fit nuances (5 datasets):**

| Dataset | Issue | Adapter rule |
|---------|-------|--------------|
| `waltz2020differential`, `feng2021dynamics` | Instructed bandit trials | `is_prediction_target=false`; include in `history` or `problem.context` |
| `kool2016when/exp2`, `kool2017cost/exp2` | Multi-stage MDP | One trial per **decision stage**; each trial has its own `options` (K∈{2,…,6}) |
| `wu2018generalisation` | K up to 30 per environment | `options` lists only keys offered that block |

---

## 3. Where the API is still insufficient

Not among the 44-target set, but **in the broader Psych-101 train corpus (76 experiments)**:

| Failure mode | Examples | Why dict API fails |
|--------------|----------|------------------|
| **Scalar probability / rating** | `zhu2020bayesian`, `wise2019acomputational` | Output is continuous % not a categorical distribution |
| **Mixed choice + rating phases** | `garcia2023experiential` (×4) | Part 3 needs `predict() -> scalar` or separate task_type |
| **Sequence output** | `enkavi2019digitspan` | Variable-length digit string per trial |
| **High-K spatial coordinate** | `kumar2023disentangling` | 49 cells; dict possible but impractical for evolved programs |
| **Omission / withhold** | `enkavi2019gonogo` | Valid action set includes “no press”; not standard K-way choice without absorbing state |
| **Verbal categorical (unbounded labels)** | `collsiöö2023MCPL` (×3) | `You say <<high>>`; ordered categories could be K=9 dict but labels are text not keys |
| **Compound / joint actions** | `krueger2022identifying` | Gamble key × typed color/stop — product action space |
| **Global self-report** | `jansen2021dunningkruger` | Few numeric reports, not repeated trial choices |
| **Timeout / no response** | `kool2016when/exp1` (uncertain) | Missing choice events unless modeled as extra action |

**Conclusion:** One categorical dict API covers **discrete choice** tasks; it does **not** replace scalar, sequence, or omission-focused objectives without extensions (`task_type` enum or separate evaluators).

---

## 4. Can K vary safely?

| Level | Varies? | Evaluator implication |
|-------|---------|----------------------|
| **Across datasets** | Always | Never hard-code global K |
| **Across participants** | Rarely for K; often for **raw key labels** | Integer IDs are per-trial; `raw_key` preserved in options |
| **Across blocks** | Yes (games, environments, rounds) | `block_id` + `problem.context` |
| **Across trials** | **Yes** (tomov neighbors, kool stages, wulff phases) | **Must** use per-trial `problem["options"]` |

**Design rule:** Evaluator valid actions = `[o["action"] for o in problem["options"]]` sorted; `K = len(options)` with `options` length ≥ 2.

**Do not assume** one fixed K per dataset or participant.

---

## 5. Adapter output schema

Canonical schema: [`generalized_categorical_adapter_schema.json`](generalized_categorical_adapter_schema.json)

**Improvements over the suggested starting point:**

- `schema_version` + `action_id_convention` for reproducible ID assignment
- Split `stimulus` vs `context` inside `problem`
- `is_prediction_target` + nullable `target_action` for context-only trials (wilson-style, instructed bandits)
- Optional precomputed `history` but **deterministic rebuild rule** documented for eval
- `adapter_metadata` block (warnings, parse_coverage) separated from trial payload
- JSON Schema constraints on consecutive action IDs and min 2 options

Runtime trial dict (fed to `choose`) can be a thin projection:

```python
{
  "problem": {**block_problem_static, **trial.problem},
  "history": [...],  # list of {action, feedback?, ...}
  "target_action": 1,  # eval only
}
```

---

## 6. Validation rules (adapter output)

| Rule | Enforcement |
|------|-------------|
| Action IDs consecutive `0..K-1` | JSON Schema + validator script |
| Each prediction trial has ≥ 2 options | Schema `minItems: 2` on `options` |
| `target_action ∈ {0,…,K-1}` when `is_prediction_target=true` | Validator |
| `raw_key` present for each option | Required field; enables audit trail |
| History deterministic | Rebuild: prior trials in same `block_id` with `is_prediction_target=true`, in order |
| Context-only trials | `is_prediction_target=false`, `target_action=null` |
| Probabilities from program | Finite, ≥0, sum≈1 (evaluator renormalizes) |
| Program must cover all valid action IDs | Evaluator: missing keys → assign ε floor then renormalize |
| No duplicate `action` in `options` | Validator |
| Parse coverage | Adapter warns if `< threshold` (e.g. 0.95 press lines parsed) |

---

## 7. Evaluator changes (summary)

See [`generalized_categorical_eval_plan.md`](generalized_categorical_eval_plan.md) for detail.

**Categorical log-likelihood** for trial with observed `y`:

\[
\ell = \log \max(\epsilon, \tilde{p}_y)
\]

where \(\tilde{p}\) is normalized from program dict over valid action IDs.

**Backward compatibility:** If program returns `float` and `K==2`, treat as `P(action=1)`: `{0: 1-p, 1: p}`.

**Cross-dataset metrics:** Report **mean test log-lik per trial** (and per participant); do not compare raw sums across different K. Optional secondary: **per-trial normalized entropy baseline** or **pseudo-Bayesian information** — but primary metric stays average log p(observed action).

---

## 8. Prompt difficulty for LLM-generated programs

| Question | Assessment |
|----------|------------|
| Is `dict[int, float]` simpler than raw string labels? | **Yes** — avoids key aliasing bugs; matches evaluator |
| Should code iterate `range(len(problem["options"]))`? | **Yes — require in prompt template** |
| Main invalid outputs | Wrong keys, non-normalized probs, returning only `p1`, empty dict, using raw_key as dict key |
| Mitigations | Prompt: “return `{o['action']: prob}` for every option”; post-process wrapper in eval sandbox; few-shot binary K=2 and K=3 examples |
| Large K (8, 30) | **Higher prompt/evolution difficulty** — not an API problem but a search problem |

Returning ints as keys (not strings `"0"`) matches Python and reduces JSON serialization confusion.

---

## Expected coverage gain

| Corpus slice | Experiments | Row share (train) |
|--------------|------------:|------------------:|
| Current implemented binary | 10 | (subset of corpus) |
| + Remaining binary-compatible | 23 | ~high for ruggeri alone |
| + Multi-action with dict API | 21 | moderate |
| **Total categorical API (est.)** | **54** | **~71% of 76 experiments** |
| Still need other APIs / skip | 22 | scalar, mixed, out-of-scope, uncertain |

**Row-weighted gain is larger** if big-N datasets (ruggeri, hebart, wulff sampling) are prioritized.

---

## Risks and failure modes

1. **Adapter errors** — wrong option order permutes action IDs → correct program looks wrong.
2. **Variable K trials** — program assumes fixed K from first trial → runtime KeyError or silent mis-eval without per-trial check.
3. **Large K** — uniform baselines dominate; evolution may struggle (schulz K=8, wu2018 K≤30).
4. **Context-only trial policy** — inconsistent `is_prediction_target` rules across adapters breaks history.
5. **Backward compat drift** — mixed float/dict programs during migration need eval shim.
6. **LLM adapter hallucination** — offline adapter must be validated; never trusted without schema + spot checks.

---

## Recommendation

| Option | Verdict |
|--------|---------|
| **Implement now (full rollout)** | No — adapter coverage and eval shim need proof |
| **Prototype first** | **Yes (recommended)** |
| **Defer categorical API** | No — binary-only eval is the main blocker for 21 multi experiments |

**Prototype scope:** categorical evaluator + schema validator + adapter spec; test on **K=2** (gershman2018), **K=3** (hebart or gershman2020reward), **variable K** (tomov2020discovery), plus one current implemented dataset for backward compat.

---

## Relation to prior steps

| Step | Finding | Step 3 implication |
|------|---------|---------------------|
| Step 1 | `teh.py` is dataset-agnostic post-parse | Only eval + trial dict contract need generalization |
| Step 2 | 21 multi-action, 7 out-of-scope | Dict API unlocks the 21; does not fix the 7 |
| Step 3 | Dict API + per-trial options | Unifies binary as K=2 special case |
