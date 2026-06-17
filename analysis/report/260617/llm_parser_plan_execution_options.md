# Parser plan execution options — comparison

**Date:** 2026-06-17  
**Context:** Version B — LLM produces parser plan; fixed code parses full corpus offline.

---

## Options compared

| Option | Description |
|--------|-------------|
| **A** | Pure regex/template plan |
| **B** | Line-type classification plan |
| **C** | Table/CSV column mapping plan |
| **D** | Hybrid: plan + LLM-generated regexes + state machine |
| **E** | LLM-generated Python parser + validator |

---

## A. Regex / template-based plan

**Mechanism:** Plan specifies regex per field; scan full transcript sequentially.

| Criterion | Assessment |
|-----------|------------|
| **Robustness** | High for **easy_plan** (~25 exps): press lines, lottery pairs, numeric keys |
| **Implementation difficulty** | Low for engine; medium to get schema right |
| **Hallucination risk** | Medium — LLM may emit regex that never matches |
| **Avoids manual per-dataset code** | Yes for stereotyped formats |
| **Stateful tasks** | Poor alone (CCT, balloon, sampling, MDP) |

**Verdict:** Necessary **building block**, insufficient as sole engine.

**Covers ~40%** of categorical corpus alone.

---

## B. Line-type classification plan

**Mechanism:** Classify each line/chunk → assemble trials from typed sequences (e.g. `stimulus → action_press → feedback`).

| Criterion | Assessment |
|-----------|------------|
| **Robustness** | Good for study-test (cox), study+judge (popov), game headers |
| **Implementation difficulty** | Medium — need line tokenizer + state for "current block" |
| **Hallucination risk** | Lower than freeform regex — types are enumerable |
| **Avoids manual labor** | Yes when patterns are consistent |
| **Stateful tasks** | Moderate — pairs with state machine extension |

**Verdict:** **Core architecture** for Version B engine.

**Extends coverage to ~70%** of categorical corpus combined with A.

---

## C. Table / CSV column mapping plan

**Mechanism:** Map HF `text` column through fixed delimiters as if tabular.

| Criterion | Assessment |
|-----------|------------|
| **Robustness** | **Low** — Psych-101 rows are NL transcripts, not CSV tables |
| **Implementation difficulty** | Low technically, wrong tool |
| **Hallucination risk** | N/A |
| **Avoids manual labor** | No |
| **Stateful tasks** | No |

**Verdict:** **Not applicable** to Psych-101 HF format. Skip.

---

## D. Hybrid plan + LLM regexes + state machine (recommended)

**Mechanism:**

1. Line classifier (B)
2. Regex captures from plan (A)
3. Optional `state_machine` for sequential tasks
4. LLM may propose regexes but **validator** checks match counts on sample rows before full parse

| Criterion | Assessment |
|-----------|------------|
| **Robustness** | Good for **easy + medium** (52 exps); partial for **hard** (7) |
| **Implementation difficulty** | Medium-high — one engine, many plans |
| **Hallucination risk** | Mitigated by coverage validation before accept |
| **Avoids manual labor** | Yes — new dataset = new plan JSON, not new `.py` file |
| **Stateful tasks** | Yes via explicit state variables (cards_flipped, pump_count, station graph) |

### State machine examples

| Task | State vars | Transitions on action |
|------|------------|----------------------|
| Frey CCT | `cards_flipped`, `current_score` | flip increments; stop ends round |
| Frey balloon | `pump_count`, `accumulated_points` | pump vs stop |
| Wulff sampling | `samples_seen`, `phase` | sample vs stop-and-choose |
| Tomov subway | `current_station`, `neighbors` | move along edge |

**Verdict:** **Recommended prototype architecture.**

---

## E. LLM-generated Python parser function + validator

**Mechanism:** LLM outputs restricted Python `parse_row(text) -> adapter JSON`; run in sandbox; validator compares coverage vs transcript.

| Criterion | Assessment |
|-----------|------------|
| **Robustness** | Highest ceiling for **hard_plan** and edge cases |
| **Implementation difficulty** | Medium — reuse eval sandbox; security sandbox needed |
| **Hallucination risk** | High without validator; **must quarantine** failures |
| **Avoids manual labor** | Yes but **reintroduces opaque code** per dataset |
| **Stateful tasks** | Excellent |

### Risks vs Version B spirit

- Blurs line between "plan" and "manual parser"
- Harder to audit than declarative JSON
- Maintenance burden if LLM code diverges

**Verdict:** Use as **Tier 3 fallback** only when D fails validation twice, with `human_review_required=true`.

---

## Side-by-side summary

| Option | Robustness | Impl effort | Hallucination risk | Manual labor | Stateful | Recommend |
|--------|------------|-------------|-------------------|--------------|----------|-----------|
| A regex | Low–Med | Low | Med | Low | Poor | Building block |
| B line-type | Med | Med | Low–Med | Low | Med | **Core** |
| C column | N/A | Low | — | — | — | **Skip** |
| D hybrid | Med–High | Med–High | Med (mitigated) | Low | Good | **Prototype** |
| E Py parser | High | Med | High | Low | Excellent | **Fallback** |

---

## Prototype recommendation

### Build **Option D** as the primary engine

```text
LLM (3–10 rows) → parser plan JSON (schema v1)
                 → line classifier
                 → trial extractor
                 → option normalizer
                 → [optional] state machine
                 → adapter JSON
                 → validator
```

### Acceptance gates (per experiment)

1. Plan JSON schema-valid
2. On 3 holdout rows not shown to LLM: press coverage ≥ 0.90
3. ≥95% trials have valid `target_action` and ≥2 options
4. Pooled trial count ≥ minimum threshold

### Fallback path

```text
If gates fail → LLM revises plan (max 2 iterations)
             → still fails → Tier E Python parser (sandboxed)
             → still fails → quarantine + manual review queue
```

### Do not prototype first

- **C** column mapping
- Runtime LLM parsing per trial
- Full 76-experiment batch before quartet validation

---

## Handling specific task families with Option D

| Family | Plan features |
|--------|---------------|
| Binary gamble / bandit | `action_press` regex; `fixed_keys_from_instruction` |
| Multi-arm (4–8) | `per_trial_available_keys` or numeric key list in stimulus |
| Variable K navigation | `per_trial_available_keys` from neighbor line regex |
| Study → test | line types `trial_stimulus` + `action_press`; block `study_then_test_pair` |
| Instructed bandits | `context_only_trial_rule` on "instructed to press" |
| Sequential risk | `state_machine` + `round_state_machine` boundary |

---

## Conclusion

**Prototype Option D (hybrid declarative plan engine)** with **Option E as quarantined fallback**.

This best matches Version B goals:

- No manual parser per dataset (new JSON plan instead)
- No LLM at program evaluation time
- Validator-gated quality
- Covers **~88%** of categorical experiments at easy/medium plan tiers

See implementation sequencing in [`llm_parser_plan_feasibility.md`](llm_parser_plan_feasibility.md) and eval integration in [`generalized_categorical_eval_plan.md`](generalized_categorical_eval_plan.md).
