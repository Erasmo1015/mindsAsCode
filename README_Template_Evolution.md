# Template Evolution (Non-Strict) Quick Guide

This guide documents participant-selection arguments for `Template_evo_non_strict.py` only.

## Run Command (skeleton)

```bash
python Template_evo_non_strict.py \
  --dataset <choice13k|cpc18|mixed_gambles|gridworld|gridworld_ensemble> \
  --participant_scope <single|range|all> \
  [scope-specific args] \
  --n_iterations 5 --n_candidates 10 --mode local
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
