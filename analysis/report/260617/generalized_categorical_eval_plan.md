# Generalized categorical evaluator — implementation plan

**Date:** 2026-06-17  
**Status:** Design only (no code changes in this step)  
**API:** `choose(problem, history) -> dict[int, float]`  
**Schema:** [`generalized_categorical_adapter_schema.json`](generalized_categorical_adapter_schema.json)

---

## Goals

1. Score synthesized programs with **categorical log-likelihood** on trials where `target_action` is a discrete choice.
2. Support **K=2 through large K** with **per-trial** valid action sets.
3. **Backward compatible** with existing binary programs returning `float` = `P(action=1)`.
4. Consume **offline adapter output** without runtime LLM calls.

---

## Minimal evaluator changes

### New core function (conceptual)

Replace / wrap `evaluate_choice13k_program` with:

```text
evaluate_categorical_program(choose_fn, trials, *, eps=1e-9) -> metrics
```

**Per trial:**

1. Read `valid_actions = [opt["action"] for opt in trial["problem"]["options"]]` (sorted).
2. Assert `len(valid_actions) >= 2` and consecutive from 0.
3. Call `probs_raw = choose_fn(trial["problem"], trial["history"])`.
4. **Coerce** output (see backward compatibility).
5. **Normalize** to `probs` over `valid_actions`.
6. `ll += log(probs[target_action])`.

**Aggregate:** `avg_loglik = sum(ll) / n_trials` (same interface shape as today: `avg_loglik`, `accuracy` optional).

### Categorical log-likelihood

For observed action `y`:

```python
p = normalized_probs[y]
ll = log(max(eps, p))
```

Optional **accuracy**: `argmax(probs) == y` (secondary metric; evolution still uses log-lik).

### Clamping and normalization

```text
1. Coerce choose() output -> dict[int, float]
2. For each a in valid_actions:
     p[a] = max(0, finite_float(probs_raw.get(a, 0.0)))
3. If sum(p) == 0: set uniform p[a] = 1/K
4. Else: p[a] /= sum(p)
5. Clip each p[a] to [eps, 1-eps] optional, then renormalize (match current binary epsilon handling)
```

**Missing keys:** assign 0 before step 3 (not uniform per missing key alone — only if entire dict empty or sum zero).

**Extra keys:** ignore keys not in `valid_actions` (optionally log warning in verbose mode).

### Per-trial K (not per-dataset)

```python
K = len(trial["problem"]["options"])
valid_actions = list(range(K))  # after validator enforces consecutive IDs
```

Never read K from dataset config.

---

## Backward compatibility strategy

| Program return type | `K` | Coercion |
|-------------------|-----|----------|
| `float` or `int` 0/1 | 2 | `{0: 1-p, 1: p}` with `p = clamp(float)` |
| `dict` with keys `{0,1}` | 2 | normalize |
| `dict` with keys `{1}` only (legacy mistake) | 2 | treat as `p1`, build `{0:1-p1,1:p1}` if `K==2` |
| `dict` full range | K | normalize |

**Detection:** If return is not dict and `len(options)==2`, use binary shim. If return is not dict and `K>2`, count as eval error (or uniform fallback + error flag).

**Migration path:**

1. Phase A: eval accepts **both** float and dict; prompts still ask for float on legacy datasets.
2. Phase B: prompts ask for dict; shim keeps old checkpoint programs runnable.
3. Phase C (optional): remove float acceptance after re-evolution.

**Compile-time:** No change to sandbox imports; optional wrapper injected after `compile_program`:

```python
def choose(problem, history):
    ...
# eval wraps to coerce output
```

---

## Trial dict bridge from adapter schema

Adapter JSON → runtime list (compatible with existing split utilities):

```python
def adapter_to_eval_trials(participant_doc) -> list[dict]:
    trials = []
    for block in participant_doc["blocks"]:
        hist = []
        for t in block["trials"]:
            problem = {**block.get("problem_static", {}), **t["problem"]}
            if t["is_prediction_target"]:
                trials.append({
                    "problem": problem,
                    "history": list(hist),
                    "target_action": t["target_action"],
                    "action": t["target_action"],  # alias for legacy code paths
                    "feedback": t.get("feedback"),
                })
            # update history regardless (include context-only if configured)
            if t.get("target_action") is not None:
                hist.append({"action": t["target_action"], "feedback": t.get("feedback")})
    return trials
```

**Policy knob:** `history_includes_context_only: bool` (default False for prediction history; True for rich MDPs).

---

## Validation / normalization pipeline (offline)

```
HF row -> offline adapter -> JSON document -> validate_adapter_json()
  -> write cache datasets/psych101_categorical/{split}/{experiment}/{pid}.json
  -> collect valid participant ids (non-empty train/test split)
```

**Validator checks:**

- Schema version match
- Consecutive action IDs per trial
- `target_action` validity
- ≥1 prediction target per participant
- ≥3 blocks for TEH split (or pseudo-block chunker on adapter output)
- Optional: compare press-line coverage vs transcript

**On failure:** quarantine participant with warning in `adapter_metadata.warnings`; exclude from `valid_participant_ids.json`.

---

## Prompt constraints (evolution, not adapter)

Add to evolution/refine prompts:

```text
- choose(problem, history) must return dict[int, float]
- Keys must be exactly: o["action"] for o in problem["options"]
- Values strictly positive and sum to 1 (evaluator may renormalize small errors)
- Use integers for keys, not strings
- Example K=2: {0: 0.4, 1: 0.6}
- Example K=3: {0: 0.2, 1: 0.5, 2: 0.3}
- Build probs with: for i, o in enumerate(problem["options"]): ...
```

**Easier than raw labels** because programs index `problem["options"]` instead of matching `<<G>>` vs gamble_A strings.

---

## Cross-dataset metric summarization

| Metric | Definition |
|--------|------------|
| `avg_loglik` | Mean per-trial log prob (primary) |
| `accuracy` | Mean argmax hit rate (optional) |
| `n_trials` | Count |
| `eval_errors` | Exceptions + invalid return type |

**Do not** sum log-lik across trials of different K for model selection inside one dataset (mean is fine).

**Cross-dataset comparison:** use mean log-lik per trial per participant, then aggregate participants — same as current TEH practice.

---

## First datasets to test

| Phase | Dataset | K | Why |
|-------|---------|---|-----|
| P0 | `gershman2018deconstructing/exp1` | 2 | Simple 2-arm; validates dict + binary shim |
| P0 | `1peterson2021using` (implemented) | 2 | Backward compat with existing parser |
| P1 | `gershman2020reward/exp1` | 3 | Small multi-action |
| P1 | `hebart2023things/exp1` | 3 | Large N; stress test |
| P2 | `steingroever2015data/exp1` | 4 | IGT |
| P2 | `schulz2020finding/exp1` | 8 | Large K prompt difficulty |
| P3 | `tomov2020discovery/exp2` | 2–5 variable | Per-trial K |
| P3 | `waltz2020differential` | 2 | `is_prediction_target` filtering |

---

## Success criteria (prototype)

| Criterion | Target |
|-----------|--------|
| Binary shim | Float program on K=2 trials matches current `evaluate_choice13k_program` log-lik within `1e-6` |
| Dict K=2 | `{0:1-p,1:p}` matches float program |
| Dict K=3+ | Uniform baseline log-lik = `-log(K)` per trial |
| Variable K | No crash when `len(options)` changes trial-to-trial |
| Invalid output | Missing keys → stable normalized eval + error counted, no crash |
| Adapter validator | Catches permuted/non-consecutive action IDs |
| End-to-end | One offline adapter JSON → split → eval seed program on train/test |

---

## Out of scope for this evaluator plan

- Scalar `predict() -> float` for zhu/wise (separate `task_type`)
- Sequence transducers for digit span
- Go/no-go omission as explicit action without schema extension
- Runtime LLM adapter calls

---

## Suggested implementation order (when coding)

1. `validate_adapter_json()` + JSON Schema tests
2. `coerce_choose_output()` + `evaluate_categorical_program()`
3. Binary shim tests against existing choice13k trials
4. `adapter_to_eval_trials()` projection
5. Wire behind feature flag `fitness_metric=loglik_categorical` or auto-detect from trial schema
6. Offline adapter prototype on 2 datasets
7. Prompt template update for dict output

No changes to `teh.py` until steps 1–4 are unit-tested in isolation.
