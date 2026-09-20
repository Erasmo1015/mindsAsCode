# ICLR Centaur baseline (PICS-fair adapter)

**How Centaur is used in this paper.** We evaluate the official
Llama-3.1–Centaur-70B adapter as an **in-context behavioral predictor** under
the same training-only SA40 data contract as PICS v3
(`structure_aware_v3`). For each participant we build a Psych-101-style
transcript from observed train+validation (and, under v2/v3 timelines, unscored
pre-test context rows that were omitted by SA40), then score **only held-out
test trials** with a PICS-fair per-trial normalized action likelihood: softmax
over legal `<<key>>.` suffix log-probabilities, then log p(observed action).
Centaur is **not** a program-induction method: it has no G.1/G.2/G.3,
Occurrence-EB, transfer, or PICS prompt evolution.

This file is the authoritative ICLR Centaur description. It does not redesign
the model; it documents our evaluation adapter and shared data contract.

| Label | Meaning |
| --- | --- |
| **Official Centaur** | Marcel Binz et al. Psych-101 / Centaur model & transcript conventions |
| **Our adapter** | `baseline_methods/Psych101/Centaur.py` + `centaur_prompts.py` |
| **Paper-safe** | Requires `final/is_complete=true` over the full expected cohort |

---

## 1. Entry point

```bash
python baseline_methods/Psych101/Centaur.py \
  --dataset <alias> \
  --psych_dataset_split train \
  --participant_scope range \
  --range_start_ordinal <start> --range_end_ordinal <end> \
  --split_ratio 0.6 --split_seed 0 \
  --limited_data_protocol structure_aware_v3 \
  --limited_train_val 40
```

- Model default: `marcelbinz/Llama-3.1-Centaur-70B-adapter`
- `--max_seq_length` default **32768** (Centaur context; not required to match
  PICS Qwen packing)
- CLI data protocol default comes from `add_limited_data_cli_arguments` →
  **`structure_aware_v3`**
- **Ordinal ranges are not auto-filled.** Production must pass EMNLP ranges
  explicitly (unlike OpenEvolve’s `apply_iclr_frozen_range_ordinals`).
- Smoke (CPU): `--smoke_prompt_only`, `--smoke_all_datasets`, `--check_deps`

Helpers: `load_participant_limited_splits`, `_centaur_prompt_timeline_v2`,
`CentaurChooser.action_probs_from_suffixes`, `wandb_completion_fields`.

---

## 2. Official Centaur vs our evaluation adapter

| Surface | Official Centaur | Our ICLR adapter |
| --- | --- | --- |
| Model | Centaur 70B adapter | same default HF id |
| Transcript | Psych-101-style presses | same family; dataset-specific prefixes in `centaur_prompts.py` / gamble formatter |
| Metric | Official papers often report session token NLL | **PICS-fair**: suffix logprobs → softmax over legal keys → log p(action) |
| Mixed Gambles | Letter A/B gambles | Forced A/B remapping + “Option A/B delivers …” text; rejects numeric `<<0>>`/`<<1>>` empty prompts |
| Data | Full Psych-101 | Shared `structure_aware_v3` SA40 + EMNLP ordinals |
| Selection | n/a (no evolution) | Test never selects anything; scored after transcript build |

---

## 3. Shared fairness contract (`structure_aware_v3`)

| Knob | Value |
| --- | --- |
| Protocol | `structure_aware_v3` |
| Cap | `--limited_train_val 40` |
| Split | `--split_ratio 0.6 --split_seed 0` |
| Speekenbrink | chronological session split (default) |
| Independent history | empty on train, val, **and** test |
| Continuous/resetting test histories | meaningful originals under v3 |
| Test membership/order/actions | unchanged |
| Enkavi `probe_in_set` | stripped from program/prompt inputs (oracle) |
| Fitness / selection | **none** on Centaur; only test scoring for reporting |

v2/v3 timeline (`_centaur_prompt_timeline_v2`): unscored raw pre-test TV context
may appear in the transcript for sequential continuity, but **only test indices
are scored**. `run_smoke_prompt_check` uses the same v2 raw timeline under
`structure_aware_v3` (required for continuous Speekenbrink histories).

---

## 4. Fifteen datasets, ordinals, and action mappings

Production ordinals (list indices into `valid_participant_ids.json`):

| Dataset | Ordinals | Actions / display keys | Notes |
| --- | --- | --- | --- |
| `1peterson2021using` | 0–49 | binary `option_keys` | gamble schema |
| `2plonsky2018when` | 0–49 | binary | |
| `3frey2017cct` | 0–49 | binary | |
| `4wulff2018description` | **1290–1339** | binary | independent, empty history |
| `5speekenbrink2008learning` | **0–22** | binary | continuous |
| `7hilbig2014generalized` | 0–49 | binary | independent |
| `10frey2017risk` | 0–49 | binary | balloon state in problem |
| `11enkavi2019recentprobes` | 0–49 | binary | no `probe_in_set` in inputs |
| `12badham2017deficits` | **0–9** | binary | features in problem |
| `mixed_gambles` | 0–49 | **`A`,`B`** | remapped; Option A/B text required |
| `bergert_nosofsky_2007` | 0–49 | `0`,`1` | independent |
| `guan_2020_stopping` | 0–49 | `0`,`1` | continue/stop |
| `steyvers_2009_bandit` | 0–49 | **`1`–`4`** (K=4) | 1-indexed presses |
| `13schulz2020finding` | 0–49 | **`1`–`8`** (K=8) | categorical bandit |
| `14kool2016when` | 0–49 | binary | stage/state under SA40 exact-40 |

`centaur_display_keys()` is the authority for press tokens. Internal action
index `a` maps to `keys[a]`.

---

## 5. Prompt / transcript construction

1. Load SA40 splits via shared limited-data loaders.
2. Mixed Gambles: `_prepare_mixed_gambles_centaur_trials` forces
   `option_keys=["A","B"]`, `schema_type="A"`.
3. Build prefix with current-trial observables only (no future rewards,
   correctness, oracles). Dataset-specific builders in `centaur_prompts.py`
   (Bergert, Guan, Steyvers, Schulz, Kool) plus gamble formatters in
   `Centaur.py`.
4. Score each legal suffix `<<{key}>>.` after the prefix; softmax; log p of
   observed action’s key.
5. Repeated K-way passes are slower but do **not** change the metric definition.

**Mixed Gambles validity gate:** missing Option A/B text or non-`['A','B']`
keys raise. The historical empty `<<0>>`/`<<1>>` prompt that produced
approximately **−2.004** mean test loglik is **invalid** and must not enter
paper tables.

---

## 6. Likelihood (PICS-fair)

`CentaurChooser.action_probs_from_suffixes`:

1. Build prefix for trial `i`.
2. For each display key, compute suffix logprob of `<<key>>.`.
3. Softmax over the K logprobs.
4. Bernoulli: `P(action=1)=probs[1]`; categorical: `log probs[action]`.

This is **not** the official Centaur whole-session token-NLL metric.

---

## 7. Outputs, W&B, resume, completeness

- Outputs under `generated_outputs/.../centaur/run_*` (mixed_gambles has its
  own root).
- W&B project `centaur`.
- Completion via `utils/teh/baseline_wandb_completion.py`:
  - `final/is_complete` true only when attempted == expected and every person
    has a finite `test_loglik` with no `failed` status;
  - `final/mean_test_loglik` written **only** when complete (equal-person mean).
- Partial/interrupted runs must not be treated as paper results.
- Resume: re-run incomplete people; do not invent means for missing rows.

---

## 8. Runtime characteristics

- Large adapter LM; GPU required for production scoring.
- K-way categorical (Schulz K=8, Steyvers K=4) needs K forward suffix scores
  per trial → slower but same math.
- `--smoke_*` / `--check_deps` are CPU-safe.

---

## 9. Known invalid / non-final history

| Item | Status |
| --- | --- |
| Mixed Gambles run ≈ **−2.004** from empty `<<0>>`/`<<1>>` prompts | **INVALID — do not use** |
| Runs without `structure_aware_v3` / wrong ordinals | non-comparable to PICS v3 |
| Incomplete W&B (`final/is_complete=false`) | reporting only, not paper means |

---

## 10. Final ICLR Centaur launch / rerun requirements

For **every** of the 15 datasets under production EMNLP ordinals:

1. Pass `--limited_data_protocol structure_aware_v3 --limited_train_val 40`.
2. Pass explicit `--participant_scope range` with EMNLP ordinals above.
3. Confirm Mixed Gambles logs show A/B Option text (not numeric empty presses).
4. Require `final/is_complete=true` before quoting `final/mean_test_loglik`.
5. **Rerun Mixed Gambles** if any historical −2.004 / empty-prompt artifact was
   used; treat all prior MG Centaur numbers as suspect until A/B-contract runs
   complete.
6. Prefer fresh SA40-v3 runs for all 15 if prior Centaur jobs used `structure_aware`
   (v1) or `off` / wrong ranges.

No PICS transfer, G.1/G.2/G.3, Occurrence-EB, or PICS prompt-evolution code paths
are invoked.
