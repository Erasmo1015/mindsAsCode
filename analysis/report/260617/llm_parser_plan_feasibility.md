# LLM parser-plan adapter (Version B) — feasibility analysis

**Date:** 2026-06-17 (step 4)  
**Proposal:** LLM reads 3–10 representative HF rows → outputs **parser plan JSON** → fixed Python engine executes plan → validator → PICS on structured categorical trials.

**Related:** [`llm_parser_plan_schema.json`](llm_parser_plan_schema.json), [`llm_parser_plan_dataset_fit.csv`](llm_parser_plan_dataset_fit.csv), [`llm_parser_plan_execution_options.md`](llm_parser_plan_execution_options.md), [`psych101_76_reconciled_coverage.csv`](psych101_76_reconciled_coverage.csv)

---

## Executive answer

**Yes — Version B is feasible as the primary scaling path for categorical-choice Psych-101 experiments**, with important caveats:

| Verdict | Detail |
|---------|--------|
| **Feasible for** | **59/76** HF experiments (categorical `dict[int,float]` API) |
| **Strong automation expected** | **52/76** (`easy_plan` + `medium_plan`) |
| **Needs hybrid / review** | **7** categorical `hard_plan` + **2** uncertain kool exps |
| **Not suitable for plan-only** | **17** non-categorical experiments |

**Recommendation: prototype before full implementation** — build plan schema + regex/state engine + validator on 4 datasets (2 easy, 1 medium, 1 hard), then expand.

---

## 1. Parser-plan feasibility (59 categorical experiments)

Classification is **LLM plan difficulty** (can an LLM infer extraction rules from few rows?), separate from PICS difficulty (evolving good programs).

| Class | Count | Description | Examples |
|-------|------:|-------------|----------|
| **easy_plan** | 25 | Repeated `You press <<KEY>>` / numeric key templates; trial = one press line | ruggeri, gershman2018, schulz, hebart, steingroever, popov |
| **medium_plan** | 27 | Block/game headers, study-test, casino context, instructed→context-only, variable options per block | plonsky, wilson×5, waltz, zorowitz, wulff sampling, ludwig |
| **hard_plan** | 7 | State machines, variable K neighbors, multi-stage MDP segmentation | frey2017risk, tomov×4, kool exp2 variants |

**Non-categorical (17):** `not_suitable_for_plan` — scalar, mixed, sequence, omission, verbal labels, grid 49-way, compound actions, global self-report.

### Why Psych-101 is plan-friendly

Representative transcripts share **stereotyped NL patterns**:

- `You press <<X>>` / `You press <<X>> and get N points`
- `Game N` / `Round N` / `Balloon N` headers
- `Lottery A offers …` gamble blocks
- `You are instructed to press …` (context-only candidates)

An LLM seeing 3–10 rows can plausibly propose:

- line-type regexes (`action_press`, `block_header`, `feedback_points`)
- trial boundary = one press event or stimulus+press pair
- option keys from instruction or per-trial neighbor list

This matches how **manual parsers in `psych101_parsers.py` were written** — but as **declarative plans** instead of hand-coded functions.

### Plan difficulty vs PICS difficulty

| Dataset | Plan | PICS |
|---------|------|------|
| hebart (3-way, huge N) | easy_plan | medium (K=3 programs) |
| tomov navigation | hard_plan | hard (variable K) |
| ruggeri (binary, huge N) | easy_plan | easy |

---

## 2. What should the LLM output?

**Not parsed trials.** A **parser plan** executable by fixed code.

Canonical schema: [`llm_parser_plan_schema.json`](llm_parser_plan_schema.json)

### Improvements over the suggested starter schema

| Addition | Why |
|----------|-----|
| `line_classifier.line_types[]` | Separates instruction / press / feedback / context-only before trial assembly |
| `trial_extraction.boundary_strategy` enum | Explicit trial segmentation modes |
| `option_normalization.strategy` | Handles fixed vs per-trial vs per-block key sets |
| `state_machine` block | CCT, balloon, sampling without bespoke Python per dataset |
| `context_only_trial_rule` | Wilson / waltz instructed trials |
| `source_assessment` + `human_review_required` | Audit trail and quarantine flag |
| `validation_expectations.expected_press_coverage` | Detect hallucinated rules |

### Plan output flow

```text
HF row text
  → line classifier (plan-driven regex)
  → trial candidates
  → option normalization → action IDs 0..K-1
  → problem.stimulus / problem.context / target_action
  → adapter JSON (psych101_categorical_v1)
  → validator
```

---

## 3. Can fixed code execute the plan?

See detailed comparison: [`llm_parser_plan_execution_options.md`](llm_parser_plan_execution_options.md)

**Short answer:** A **hybrid plan engine (D)** — validated regex + line types + optional state machine — covers **most easy/medium** datasets. **Pure regex template (A)** alone is insufficient for ~30% of categorical corpus. **LLM-generated Python (E)** is the fallback for `hard_plan` quarantine retries, still **offline only**.

---

## 4. Recommended design (first prototype)

### Pipeline (Version B refined)

```text
1. Sample 3–10 representative participant rows (stratified by length/parse quirks)
2. LLM → parser plan JSON (schema-validated)
3. Optional: human/auto review if plan.human_review_required
4. Generic plan engine executes on all rows (first 50 participants for population-level)
5. Validator on adapter output (schema + coverage + trial counts)
6. Quarantine failed participants/experiments
7. PICS population-level evolution on pooled validated trials
```

### Closest robust alternative if pure plan engine fails

**Tiered fallback (still no per-trial LLM):**

1. **Tier 1:** Plan engine (regex + line types + state machine)
2. **Tier 2:** LLM generates **supplemental regex list** only (approach D), re-execute
3. **Tier 3:** LLM generates **sandboxed Python parser function** (approach E), validator gates output; human review required before production

**Do not** fall back to LLM-per-trial parsing at evaluation time.

### Prototype dataset quartet

| Dataset | Plan tier | Why |
|---------|-----------|-----|
| `gershman2018deconstructing/exp1` | easy | Smoke test |
| `ruggeri2022globalizability/exp1` | easy | Scale test (11k rows) |
| `waltz2020differential/exp1` | medium | context-only trials |
| `tomov2020discovery/exp2` OR `frey2017risk` | hard | State / variable K |

---

## 5. Population-level setting

Initial scope: **one program per dataset (population-level)**, pooling trials across participants.

| Simplification | Implication |
|----------------|-------------|
| Use raw HF rows directly | `participant_id` = row index; no legacy valid-id JSON required initially |
| Pool first **50** participants (if available) | Sufficient trial mass for evolution |
| No participant-specific programs | Drops valid_participant_ids orchestration from critical path |

### Checks still required

| Check | Why |
|-------|-----|
| **Min pooled prediction trials** | e.g. ≥500 after parse (tune per dataset size) |
| **Per-participant parse success rate** | Quarantine if <80% participants pass validator |
| **Press-line coverage** | `parsed_trials / expected_press_lines` ≥ threshold |
| **Action ID validity** | consecutive 0..K-1, target in range |
| **Min options ≥ 2** | per prediction trial |
| **Block count for split** | ≥3 blocks or pseudo-block chunking for train/val/test |
| **Holdout** | Still use Psych-101-test rows for eval when available |
| **Plan stability** | Re-run plan on 2 disjoint row samples; compare trial counts |

Population pooling **increases trial count** but **mixes participants** — acceptable for population-level programs; document if later moving to participant-level again.

---

## 6. Reconciled coverage (76 experiments)

Full table: [`psych101_76_reconciled_coverage.csv`](psych101_76_reconciled_coverage.csv)  
Summary: [`psych101_76_reconciled_coverage_summary.md`](psych101_76_reconciled_coverage_summary.md)

**Buckets sum to exactly 76.**

Key fix vs step 3: **54 logical datasets → 59 HF experiment_ids** (Wilson ×5, wu chunking exp2).

---

## Limitations

1. **17 experiments** will not be saved by parser plans alone (scalar, mixed, sequence, omission, verbal, grid).
2. **LLM plan hallucination** — regexes that match nothing or over-match; requires validator + coverage metrics.
3. **hard_plan** tasks may need Tier 3 Python fallback or manual review.
4. **Option order convention** must be stable across participants or explicitly per-participant in plan.
5. **Population pooling** hides participant heterogeneity that TEH currently models.
6. **Wilson / instructed trials** — incorrect `context_only` rules silently corrupt history.

---

## Failure modes

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Wrong trial boundaries | Low press coverage | Tighten plan; human review |
| Permuted action IDs | Validator + spot-check raw_key recovery | Fix `option_normalization` |
| Missing context-only filter | Instructed trials as targets | Add `context_only_trial_rule` |
| State machine drift (CCT/balloon) | Score/flip count mismatch | Tier 3 fallback |
| Over-broad regex | Spurious trials | Line classifier + negative examples in plan |
| Plan non-portable across exps | Same paper, different exp file | One plan per `experiment_id` |

---

## Should we prototype before implementation?

**Yes.**

| Phase | Deliverable |
|-------|-------------|
| **P0** | `llm_parser_plan_schema.json` + validator + plan engine core |
| **P1** | 4-dataset prototype + quarantine report |
| **P2** | Batch plan generation for 52 easy/medium experiments |
| **P3** | Integrate with categorical evaluator + population-level PICS |

Defer **full repo migration** until P1 shows ≥90% press coverage on prototype quartet.

---

## Relation to manual parsers

Existing `psych101_parsers.py` functions are **hand-written special cases** of what the plan engine should express declaratively. Migration path:

1. Reverse-engineer **plan JSON** from 2–3 existing parsers (peterson, gershman2018, wilson) as gold plans.
2. Use diff between plan-engine output and manual parser output as regression test.

This validates Version B without abandoning current 10-dataset investment.
