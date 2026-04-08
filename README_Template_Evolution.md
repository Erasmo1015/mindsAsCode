# Template Evolution (Non-Strict) Quick Guide

This guide documents participant-selection arguments for `Template_evo_non_strict.py` only.

## Run Command (skeleton)

```bash
python Template_evo_non_strict.py \
  --dataset <choice13k|cpc18|mixed_gambles|gridworld|gridworld_ensemble> \
  --split_mode <within_participant|across_participants> \
  --participant_scope <single|range|all> \
  [scope-specific args] \
  --n_iterations 5 --n_candidates 10 --mode local
```

## Choice13k split mode (top-level switch)

For Choice13k, decide `--split_mode` first, then set the remaining args.

- `--split_mode within_participant` (default)
  - Uses per-participant trial split.
  - Train/test trial ratio is controlled by `--split_ratio` (default `0.9`).
  - Split randomness is controlled by `--split_seed` (default `0`).
  - Works with existing `--participant_scope` behavior.

- `--split_mode across_participants`
  - Uses `--participant_scope` to select participants first.
  - Selected participants are shuffled with `--split_seed`, then split by `--split_ratio` into train-participants and test-participants.
  - Training uses **all trials** from train participants; testing uses **all trials** from test participants.
  - Output is simplified to: `seed_program.py`, `iterations/`, `iterations.csv`, `summary.csv`.

Notes:
- `--split_ratio` is always interpreted as **train ratio**.
- `across_participants` is supported for `--dataset choice13k` only.

## LLM prompt size (`--max_prompt_train_trials`)

Generation prompts include serialized **train** trials. If that list is huge (e.g. `across_participants`), the model may hit a **context limit**.

- **`--max_prompt_train_trials N`** (default `1000000`): if `len(train_trials) > N`, the code **randomly subsamples** `N` train trials **only for each LLM generation call**. **Fitness evaluation still uses all train/test trials.**
- Sampling uses the same RNG seed as **`--split_seed`** (reproducible).
- **`--max_prompt_train_trials 0`**: no cap — every prompt includes **all** train trials (can exceed context on large runs).

Typical use for large train sets:

```bash
python Template_evo_non_strict.py --dataset choice13k --split_mode across_participants \
  --max_prompt_train_trials 200 --split_seed 0 ...
```

## Participant Scope (choice13k / cpc18 / mixed_gambles)

`--participant_scope` controls how participants are selected:

- `single` (default)
  - Use `--single_participant_id <raw_id>`
  - `raw_id` must exist in `datasets/*/valid_participant_ids.json`

- `range`
  - Use `--range_start_ordinal <int>` and `--range_end_ordinal <int>`
  - Ordinals are 0-based indices into `valid_participant_ids.json`
  - End is inclusive: selected raw ids are `valid_ids[start:end+1]`

- `all`
  - Uses all raw ids from `valid_participant_ids.json`
  - Optional cap: `--all_max_participants N` (first `N` valid ids)

### Mixed gambles filter

- Default: `--filter_mixed_gambles` is off (uses all trial types; larger valid list)
- If enabled, ordinals resolve against:
  - `datasets/mixed_gambles/valid_participant_ids_gain_loss.json`
- Otherwise:
  - `datasets/mixed_gambles/valid_participant_ids.json`

## Gridworld note

For `gridworld` / `gridworld_ensemble`, participant scope is ignored.
Use:

- `--num_agents_to_sample`
- `--agent_id`

## Examples

```bash
# Single raw participant id (default scope=single)
python Template_evo_non_strict.py --dataset choice13k --single_participant_id 42 --n_iterations 5 --n_candidates 10

# Choice13k within-participant split (90/10, deterministic)
python Template_evo_non_strict.py --dataset choice13k --split_mode within_participant --split_ratio 0.9 --split_seed 0 --single_participant_id 42

# Choice13k across-participants split on ordinal range
python Template_evo_non_strict.py --dataset choice13k --split_mode across_participants --participant_scope range --range_start_ordinal 0 --range_end_ordinal 99 --split_ratio 0.9 --split_seed 0

# Same, but cap train trials in the LLM prompt (evaluation still uses full train set)
python Template_evo_non_strict.py --dataset choice13k --split_mode across_participants --participant_scope range --range_start_ordinal 0 --range_end_ordinal 9 --split_ratio 0.9 --split_seed 0 --max_prompt_train_trials 200

# Ordinal range (inclusive) from precomputed valid list
python Template_evo_non_strict.py --dataset cpc18 --participant_scope range --range_start_ordinal 0 --range_end_ordinal 9 --n_iterations 5 --n_candidates 10

# All valid participants, capped to first 100
python Template_evo_non_strict.py --dataset mixed_gambles --participant_scope all --all_max_participants 100 --n_iterations 5 --n_candidates 10
```

## Required precomputed files

Generate valid-participant JSON files with:

```bash
python utils/tools/collect_participant_ids.py --dataset choice13k
python utils/tools/collect_participant_ids.py --dataset cpc18
python utils/tools/collect_participant_ids.py --dataset mixed_gambles
```
