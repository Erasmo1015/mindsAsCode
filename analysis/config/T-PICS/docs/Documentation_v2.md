# ICLR T-PICS v2 — training-only SA40

This file describes **T-PICS v2** as implemented. v1 remains the frozen method in [Documentation.md](../Documentation.md) and in jobs **257174–257188**, schema-v4 annotations, Occurrence-EB job **257756**, and `occurrence_eb_schema4_iter10.yaml`. v2 is a new method version. It does not overwrite those artifacts.

| Role | v1 (frozen) | v2 (this document) |
| --- | --- | --- |
| Protocol flag | `--limited_data_protocol structure_aware` | `--limited_data_protocol structure_aware_v2` |
| Kind / output folder | `t_pics_gated` | `t_pics_gated_sa40_v2` |
| Independent kind | `t_pics_gated_independent` | `t_pics_gated_independent_sa40_v2` |
| Source YAML | `Transfer_source/occurrence_eb_schema4_iter10.yaml` | `Transfer_source/v2/occurrence_eb_schema4_iter10_sa40_v2.yaml` (to be frozen after v2 G.1) |
| Shared loader | `utils/teh/limited_data_protocol.py` revision v1 | same file, `revision="v2"` |
| Candidate examples | one-line `format_trial_for_prompt` | sanitized JSON snapshots (`utils/teh/prompt_snapshots.py`) |

---

## 1. Final v2 pipeline (concise)

For each **target** dataset a v2 gated job still runs G.1 reuse-or-rebuild → dual G.2 → observed-data `train_val` gate → G.3 50 explore candidates → person evolution. Budgets are unchanged: G.1 100 / G.2 100 / G.3 50 / person 100 at 350 total T-PICS iterations when counted that way; production gated knobs remain G.1 10, G.2 5+5, G.3 50, person 10.

What v2 changes is the **data contract**:

1. SA40 caps only train+val used for prompts, generation, fitting, ranking, elite pools, G.2 gating, parent selection, and stopping.
2. Held-out test keeps original membership, order, problems, labels, and legitimate pre-choice histories.
3. Kool observed count is a hard 40 (no stage-1 backup to 41).
4. Candidate examples are complete sanitized snapshots with history cap 8.
5. Prompt units are selected structure-aware and packed as a balanced prefix under 14k Qwen tokens.
6. Source population and Occurrence-EB selection are **re-run under v2 data** into new paths. Motifs, EB formulas, six-source allowlist, and no-transfer-peek are reused, not retuned.

$$
|T_{\mathrm{train}}^{40}|+|T_{\mathrm{val}}^{40}|\le 40,
\qquad
T_{\mathrm{test}}=T_{\mathrm{test}}^{\mathrm{raw}}.
$$

Observed union for fitness:

$$
T_{\mathrm{obs}}^{40}=T_{\mathrm{train}}^{40}\cup T_{\mathrm{val}}^{40}.
$$

The displayed prompt subset \(T_{\mathrm{prompt}}\subseteq T_{\mathrm{obs}}^{40}\) may have \(|T_{\mathrm{prompt}}|\le 60\). Fitness always uses all of \(T_{\mathrm{obs}}^{40}\).

---

## 2. Training-only SA40

**Scientific protocol.** The 40-trial budget applies only to scored observations used to build or pick a program.

| Used for | Split |
| --- | --- |
| Automatic dataset prompt | train+val only |
| Candidate-generation examples | train+val only |
| Evolution / LM / PT fitting | train+val only |
| `train_val` ranking, elite pools, G.2 gate, stopping | train+val only |
| Held-out evaluation | test, after the program is frozen |

**Implementation.** `load_participant_limited_splits(..., limited_data_protocol="structure_aware_v2")`. v1 `structure_aware` is unchanged.

**Diagnostics only.** Person-level and pool-best test log-likelihood remain passive records (`diagnostic_only`, `not_used_for_selection`).

---

## 3. Original test evaluation

**Scientific protocol.** Test trial \(i\) may condition on all behavior realized **before** that choice, including earlier test trials and omitted pre-SA40 session trials. It must not see the current action, the current outcome, or any future test trial.

**v1 vs v2**

| Dataset class | v1 | v2 |
| --- | --- | --- |
| Resetting units | Test-unit histories rebuilt from the held-out unit (already matched raw) | Keep original test trials (fingerprints unchanged) |
| Independent, empty history (Wulff, mixed gambles, Bergert) | Force `history=[]` on test | Leave raw (still empty) |
| Independent with accumulated test history (Hilbig, Enkavi) | Force `history=[]` on **test** | Restore raw accumulated test histories |
| Continuous (Speekenbrink, Kool) | Rebuild test history from the retained 40/41 suffix | Restore full original pre-choice history |

---

## 4. Structure-aware observed-set selection

Unchanged except Kool overshoot.

| Category | Datasets | Observed-set rule |
| --- | --- | --- |
| Independent | Wulff, Hilbig, Enkavi, mixed gambles, Bergert | Deterministic sample of train+val trials, seed 0, cap 40; empty history on retained train+val |
| Resetting | Choice13k, CPC18, CCT, Frey Risk, Badham, Guan, Steyvers, Schulz | Complete units then a chronological prefix; stop at 40; never exceed 40 |
| Continuous | Speekenbrink, Kool | Contiguous pre-test suffix of train+val, cap 40; rebuild **train+val** histories from retained earlier observations only |

### 4.1 Kool exact-40 (v2 only)

v1: if the 40-suffix would start on stage-2, prepend matching stage-1 → 41 (`kool_include_matching_stage1`), and the hard-cap assertion was exempt.

v2: keep the nominal 40-suffix even if it starts at stage-2. That problem already contains planet, aliens, spaceship, stage-1 action, and mappings.

| Person | v1 \(|T_{\mathrm{obs}}|\) | v2 | First retained row |
| --- | --- | --- | --- |
| ordinal/pid 41 | 41 | **40** | stage-2, day 79 |
| ordinal/pid 45 | 41 | **40** | stage-2, day 81 |

Latest retained trial is unchanged. Test membership is unchanged.

---

## 5. Shared method parity

All of gated T-PICS v2, independent gated T-PICS v2, OpenEvolve, Centaur, LM, and PT load splits through `load_participant_limited_splits`. Passing `structure_aware_v2` selects the corrected contract.

Guards:

- T-PICS `evolution_selection_score=train_val`; diagnostic test never enters prompts/fitness/ranking/gate.
- OpenEvolve evaluator `combined_score` is train+val only; test JSON is written post-selection.
- LM/PT `fit_trials = train + val`.
- Centaur scores test indices only; v2 prefixes use unscored original pre-test rows.

OE keeps its own prompt sampler. It only consumes the corrected splits.

---

## 6. Centaur original-history support

v1 mapped `history` onto the concatenated **SA40** train+val+test list, so restored long test histories would not fit.

v2 timeline: `[raw train]+[raw val]+[original test]`. Score only test indices. Omitted pre-test rows are context, not SA40 observations, and are never scored.

Kool stage-2: if history does not already reveal today’s planet, the prefix states `problem["planet"]` once.

---

## 7. Candidate-generation prompt contract

**Authoritative v2 representation:** `format_snapshot_example` / `format_snapshot_examples`.

Each example contains participant/session boundary, unit identity, sanitized `problem`, at most 8 history entries (head+tail, same policy as the automatic dataset prompt), truncation metadata, and the observed action **after** the snapshot as `observed_action_label` (not a `choose()` input).

Sanitizer (`sanitize_problem_for_choose`) is shared with automatic dataset-prompt JSON (`_trial_to_example_dict`). It drops current action/outcome/oracle fields (including Enkavi `probe_in_set` and Kool stage-1 unobserved keys). History may contain past rewards.

Compact fallback: `format_trial_for_prompt(..., contract="v2")` fixes Badham (`stimulus_features`, `rule_block_id`; no `ratings_A=None`) and Frey Risk (balloon fields; action 0=pump, 1=stop; no CCT `round_id=None`). v1 one-liners are unchanged.

---

## 8. Prompt-example selection

Selection ≠ fitness.

| Setting | Rule |
| --- | --- |
| Independent | Sample snapshots; G.1/G.2 round-robin across participants |
| Resetting | Complete units; chronological prefix only if the 60-trial or 14k cap requires it; never shuffle inside a kept unit |
| Continuous | Contiguous windows, chronological inside the window, participant boundaries marked |
| G.3 / person | Full observed chronology if it fits; otherwise ordered units/prefixes, not an independent reshuffle |

v1 `_prompt_block_key` gamble-signature collapse is not used for v2.

---

## 9. Token packing

Frozen limits: display 60 trials; Qwen chat-templated input 14,000; output reserve 1,024; vLLM 16,384.

v2 packs a **balanced prefix** of the selection, then renders grouped by participant and chronological order. It does not sort the pool and tail-drop (that removed later people). Required task/API/parent content is preserved; parent-drop order is unchanged. If required content cannot fit with zero examples, fail fast. Packing uses the real Qwen tokenizer when available and refuses to silently substitute a different counter.

---

## 10. Automatic dataset prompt

Unchanged mechanism: ≤8 real sanitized train+val JSON examples, 10k-character pack, history cap 8 with truncation note, no test. Sanitizer now matches candidate snapshots.

---

## 11. OpenEvolve

Frozen OE machinery is unchanged (official SHA `411fb59`, one parent, official optional blocks, 350 iterations, 1×4 concurrency, failure floor, test post-selection). v2 OE jobs pass `--limited_data_protocol structure_aware_v2`. Defaults remain v1 so existing CPU guards stay valid.

---

## 12. Versioning and frozen artifacts

**Do not edit:** jobs 257174–257188; existing schema-v4 annotation outputs; Occurrence-EB freeze package; `occurrence_eb_schema4_iter10.yaml`; existing `t_pics_gated` outputs.

**Reuse unchanged:** schema-v4 motif definitions and annotation prompts; Occurrence-EB formulas, smoothing, hyperparameters, cosine selection, no-transfer-peek; six-source allowlist; G.1/G.2/G.3/person budgets; `train_val` gate; diagnostic-only test policy.

**New v2 artifacts (proposed; not written until G.1/EB run):**

```
analysis/config/T-PICS/Transfer_source/v2/occurrence_eb_schema4_iter10_sa40_v2.yaml
analysis/config/T-PICS/Transfer_source/v2/schema4_occurrence_eb_freeze/
generated_outputs/psych101_train/teh/<dataset>/t_pics_g1_sa40_v2/job_<tag>/
generated_outputs/psych101_train/teh/<dataset>/t_pics_gated_sa40_v2/job_<tag>/
generated_outputs/psych101_train/teh/<dataset>/t_pics_gated_independent_sa40_v2/job_<tag>/
generated_outputs/psych101_train/openevolve/<dataset>/..._sa40_v2/
generated_outputs/.../Centaur/..._sa40_v2/
generated_outputs/.../MLE|prospect_theory/..._sa40_v2/
```

Constants: `utils/teh/t_pics_v2.py`.

---

## 13. Data-flow (v2)

| Phase | Reads | Must not read |
| --- | --- | --- |
| Split | Full person sequence | — |
| SA40 cap | train+val only | test |
| Auto prompt / candidates / fitness / gate | \(T_{\mathrm{obs}}^{40}\) | test |
| Test eval | Frozen program + original test | — (program already chosen) |
| Centaur prefix | Original pre-test rows (unscored) + original test | current/future test labels as inputs |

---

## 14. Baseline rerun scope (from fingerprints)

| Method | Rerun | Skip |
| --- | --- | --- |
| T-PICS | Entire v2 stack (prompts + source map + all 15 targets) | Do not rerun v1 jobs in place |
| OpenEvolve | No existing ICLR jobs; all **future** jobs use v2 splits | — |
| Centaur | Speekenbrink, Kool (test history + unscored context); Hilbig, Enkavi (restored test history). Mixed gambles only if the A/B prompt correction was not already rerun | Resetting datasets whose test fingerprints match v1; Wulff, Bergert, mixed gambles if already empty and prompt-correct |
| LM / PT | Speekenbrink, Kool, Hilbig, Enkavi if features use `history`; Kool also for the 41→40 observed set | Resetting datasets with unchanged observed+test fingerprints; empty-history independents except if TV subsample identity is the only change (still refit if you need exact v2 observed sets) |

Observed-set identity also changes for every dataset whose train+val subsample is used in fitting. Conservatively, **LM/PT should be refit on all 15 under v2** if paper tables mix SA40 observed likelihoods. Test-history-driven features change only on Speekenbrink, Kool, Hilbig, Enkavi.

---

## 15. Reproduction (dry-run by default)

See `analysis/config/T-PICS/v2/dry_run_commands.sh`. Submitters must pass `--confirm` before any `sbatch`. Order:

1. CPU preflight (`pytest utils/teh_psych/test_tpics_v2_protocol.py` plus optional full fingerprint script)
2. 15 v2 G.1 source-population jobs (`structure_aware_v2`, new output kind)
3. Schema-v4 annotation of **new** G.1 programs (same motif code/prompts)
4. Occurrence-EB refit (same formulas) → freeze `..._sa40_v2.yaml`
5. Validate source map; no transfer peek
6. 15 v2 gated target jobs
7. Affected Centaur / LM / PT / OE reruns into v2 output dirs

---

## 16. Tests

`utils/teh_psych/test_tpics_v2_protocol.py` covers all 15 categories (one person), Kool 41/45 exact-40, original test histories, Badham/Frey fields, sanitizer parity, balanced packing, Centaur unscored context, gate/OE/LM test isolation. v1 Kool-41 behavior remains in `test_iclr_baseline_guards.py`.

---

## 17. Known limitations

- v2 source YAML does not exist until G.1+EB complete. Default gated T-PICS still points at the v1 freeze.
- Cluster submitters under `cluster/` are gitignored; v2 dry-run lives in `analysis/config/T-PICS/v2/`.
- Full production-table fingerprints (all ordinals × 15) are CPU-heavy; the checked-in test uses one person per dataset plus Kool 41/45.
- Qwen tokenizer must be locally available for production packing counts; tests that need exact 14k numbers fail clearly rather than substituting char/4.
- Independent G.1/G.2 snapshot sampling shuffles **units within person** with the frozen seed, then round-robins people; it does not shuffle chronology inside a resetting unit.
