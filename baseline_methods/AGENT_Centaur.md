# Centaur baseline (`Centaur.py`)

This document summarizes how the **Centaur** agent baseline is implemented and how to run it. The code lives in `baseline_methods/Centaur.py`.

## Role

`Centaur.py` scores **Choice13k** using a **fixed Hugging Face model** (default: [Llama 3.1 Centaur](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter) via Unsloth), not an evolved program. It reuses **Template Evo (non-strict)–compatible** flags for participant selection and splits so cluster scripts and comparisons stay aligned with `Template_evo_non_strict.py`.

There is **no code evolution** and **no JAX** dependency here: evaluation uses **PyTorch + Unsloth** and local trial dictionaries built the same way as in TE.

## Dependencies and environment

- **Python packages:** `numpy`, `tqdm`, `torch`, `unsloth`, `transformers` (as required by Unsloth for the chosen checkpoint).
- **Conda:** A dedicated env (e.g. `centaur`) is typical on the cluster; see `cluster/sbatch_centaur.sh` for module and CUDA loading patterns.
- **GPU:** Model load goes through Unsloth’s CUDA path; **large checkpoints (e.g. 70B in 4-bit) need a suitable GPU and host RAM**. If the model fails to run forward passes, evaluation falls back to **p = 0.5** per trial (see below), which shows up as **mean log-likelihood ≈ ln(0.5)**.
- **Data:** Choice13k is loaded through `data_modules.choice13k` (Hugging Face / project layout as in that module). For gated assets, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the environment.

## How to run

From the **repository root**:

```bash
python baseline_methods/Centaur.py --dataset choice13k --fitness_metric loglik \
  --participant_scope range --range_start_ordinal 0 --range_end_ordinal 2
```

**Cluster:** Submit a job that `cd`s to the repo and invokes the same command (see `cluster/sbatch_centaur.sh` as a template; adjust partitions, memory, and log paths for your site).

### Model and generation

| Flag | Meaning |
|------|--------|
| `--centaur_model` | Hugging Face model id (default `marcelbinz/Llama-3.1-Centaur-70B-adapter`). |
| `--max_seq_length` | Passed to Unsloth `from_pretrained` (default `32768`). |

The script does **not** call an external LLM API for evolution; `--model_name`, `--mode`, `--n_iterations`, `--n_candidates`, and `--seed_path` exist for **CLI compatibility** with TE and are **ignored** for the Centaur forward pass.

### Participants (same semantics as TE)

Participant ids are resolved from `datasets/choice13k/valid_participant_ids.json` (generate with `utils/tools/collect_participant_ids.py` if missing).

| Flag | Role |
|------|------|
| `--participant_scope` | `single` \| `range` \| `all` |
| `--single_participant_id` | Raw id when `single` |
| `--range_start_ordinal`, `--range_end_ordinal` | Inclusive slice into the valid-id list when `range` |
| `--all_max_participants` | Optional cap when `all` |
| `--filter_mixed_gambles` | Only relevant if the dataset machinery were extended; for `choice13k` the list path is the standard Choice13k JSON |

### Train / test splits

| Flag | Role |
|------|------|
| `--split_mode` | `within_participant` (default): block-level train/test split **per participant**, matching TE. `across_participants`: shuffle **participants** by `split_seed`, assign a **train** and **test** set of people, then **pool all trials** from each side (again aligned with TE). |
| `--split_ratio` | Fraction of blocks (within) or participants (across) for training; must be in `(0, 1)`. |
| `--split_seed` | RNG seed for the split. |
| `--n_eval_seeds` | Number of evaluation passes over the trial list (metrics averaged); default `1`. |

**Important:** With `across_participants` and `participant_scope` other than `single`, CSV outputs use a **single aggregate row** with `participant_id=0` (pooled train/test trials). For **one row per human participant**, use **`within_participant`** with `range` or `all`.

### Output directory

- Default: `generated_outputs/choice13k/centaur/run_YYMMDD_HHMMSS/`
- Override: `--output_dir /path/to/run`

Early in the run (after argument and participant validation, **before** loading the model), the script writes:

- `log/command.txt` — timestamp, cwd, hostname, and a shell-safe replay of `python …` via `shlex.join([sys.executable, *sys.argv])`.

Stdout prints `Wrote full command line to …` so batch logs point at that file.

## Implementation outline

1. **Trials:** `experiment_to_trials`, `trials_from_blocks_chronological`, and `split_trials` mirror TE so block structure, history, and option keys stay consistent.
2. **Prompting:** A fixed **Peterson-style** intro (`PETERSON_INTRO`) plus formatted current gamble lines and prior choices with **`<<letter>>`** markers; history lines use each **prior trial’s** `option_keys` so cross-block key labels stay correct (`build_centaur_prompt_prefix_indexed`).
3. **`CentaurChooser`:** Lazy-loads the model with Unsloth (`FastLanguageModel.from_pretrained`, 4-bit by default), then for each trial computes the **conditional probability of choosing the second option** (action index 1) by comparing **token log-probabilities** of suffixes `<<key0>>.` vs `<<key1>>.` after the shared prefix (`prob_choose_second_option`).
4. **Metrics:** Per trial, **Bernoulli log-likelihood** with label \(y \in \{0,1\}\) = human action (0 = first key, 1 = second key). **Accuracy** uses threshold 0.5 on \(P(\text{second})\). On exceptions during scoring, the code uses **p = 0.5** and still increments error counts (first seed only when `n_eval_seeds == 1`).
5. **Outputs:** Three behaviors:
   - **`across_participants`:** `participants_details.csv`, `summary.csv`, `participant_details_loglik.csv`, `summary_loglik.csv` via `_write_all_mode_csvs` (one pooled “participant”).
   - **`participant_scope=all`:** Same four files, updated incrementally per participant.
   - **`single` / `range`:** `participants_summary.csv`, `participant_details_loglik.csv`, `summary_loglik.csv` (running aggregates as participants complete).

## Relation to Template Evo (non-strict)

| Aspect | TE non-strict | Centaur baseline |
|--------|----------------|------------------|
| Program | Evolved Python `choose` | Fixed HF LM |
| Stack | JAX + LLM client | PyTorch + Unsloth |
| Choice13k splits / participants | `Template_evo_non_strict.py` | Same rules, duplicated helpers to avoid importing TE |
| Primary metric for selection here | N/A (no evolution) | Log-likelihood only (`--fitness_metric` is `loglik`) |

For reproducibility of **TE** runs, see `Template_evo_non_strict.py` and its `log/command.txt` behavior.

## Files

| Path | Purpose |
|------|--------|
| `baseline_methods/Centaur.py` | Entry point and evaluation logic |
| `cluster/sbatch_centaur.sh` | Example Slurm job (env modules, CUDA, `python baseline_methods/Centaur.py …`) |
| `data_modules/choice13k.py` | Experiment loading / HF details |
