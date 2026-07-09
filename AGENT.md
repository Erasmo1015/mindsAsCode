Local vLLM mode

- Recent updates (Apr 2026):
  - `te_aggregate.py` (two-phase TE profile+adapt) summary:
    - Implements Phase 1 **text-profile warmup** (default on): one LLM call per participant using `prompts/Template_evo/<dataset>/text_profile/text_profile.txt` plus a token-budgeted prefix of that participant’s **train** trials; writes `participant_*/profile.txt` and `participant_*/text_profile_meta.json`. Disable with `--profile_warmup False`.
    - Implements Phase 2 participant adaptation (`--n_iterations`, default 10) starting from the run **seed** program (`--seed_path`), with the saved text profile injected into adaptation prompts when phase 1 ran.
    - There is **no** global aggregate program evolution or `run_dir/aggregate/` pool anymore.
    - Participant adaptation reuses per-participant logging/output structure and writes final run-level summaries (`participant_details_loglik.csv`, `summary.csv`).
    - Optional diagnostic prompt trials are supported via `--num_diagnostic_trials` (total count, split into bad/good by log-likelihood; recommend 10 = 5 hard + 5 easy).
    - Optional adaptation regularization is supported via `--lambda_complexity` and `--lambda_change` (both default 0.0).
    - Runtime-valid filtering is strict for loglik paths: candidates with runtime/evaluation errors are marked `runtime_valid=False` and penalized (invalid candidates can yield `-inf` loglik / low fitness).
    - Parent fallback robustness (new): when the parent pool is too small (or effectively only baseline/seed), `te_aggregate.py` now injects up to 3 extra "fallback parents" from a runtime-error bank into prompt context; each fallback parent includes a one-line runtime error summary (captured from first failing trial, last line of exception text). If no finite-quality fallback exists, it samples fallback parents randomly from the error bank.
    - Test-leakage safeguards in evolution: test trials are not used in prompts or candidate selection logic (test eval is reporting-only where applicable).
    - Output naming/location for this method uses `te_aggregate` (generated outputs and run naming), distinct from `non_strict`.
  - Fixed OpenEvolve local Gemma-2 compatibility: avoid sending `system` role in local vLLM mode to prevent `System role not supported` errors.
  - Added explicit TE generation cap `--llm_max_tokens` (default 800) and propagated it through non-strict participant flows.
  - Standardized H100 TE/OpenEvolve scripts to use tighter prompt budgets (`--max_prompt_train_trials 60`) and parent count 3.
  - Mixed-gambles split is now problem-disjoint (by `(gain, loss, cert)` signature) in TE non-strict, OpenEvolve, and Centaur paths.
  - Added prompt-side per-problem cap: `--max_prompt_trials_per_problem` (0 disables, scripts use 10) for TE non-strict and OpenEvolve.
  - OpenEvolve prompt artifacts now omit serialized per-trial history by default (`--max_history_items_per_trial 0` in H100 scripts) to reduce token pressure.
  - CPC18 valid participants were checked: all have 5 problems (no single-problem participants in current valid-id list).
  - Loglik fixes (TE + OpenEvolve):
    - Unified loglik return semantics to `P(action=1)` and aligned wording/prompts accordingly (choice13k/cpc18 held-out/mixed_gambles).
    - TE non-strict now uses dataset-specific loglik prompt folders for `cpc18` and `mixed_gambles` (instead of reusing choice13k loglik prompts).
    - OpenEvolve now injects dataset-specific loglik prompt guidance from `prompts/Template_evo/<dataset>/non_strict/loglik/` into generation-time system guidance.
    - Updated TE and OpenEvolve cluster scripts to use renewed loglik seeds:
      - `persona_code_example/cpc18/prospect_theory.py`
      - `persona_code_example/mixed_gambles/prospect_theory.py`
  - Best-program reporting fix (TE + OpenEvolve, Apr 28, 2026):
    - Final reporting now uses a single pool-best program selected by train fitness after the last iteration/pool update.
    - `train_loglik` and `test_loglik` in participant/loglik CSVs are paired values from that same best program (no mixed-program pairing).
    - Per-participant best-program artifact is saved with origin in filename (e.g., `best_program_fr_iterX_candY.py`).
    - Result naming is metric-agnostic (`overall_best_train` / `overall_best_test`).
    - In `fitness_metric=loglik` mode, per-iteration test loglik is evaluated only for the updated-pool best program, and W&B curves use those pool-best paired metrics.
  - `utils/adhoc_fix_report.py` is for legacy runs before 2026-04-28 (use only for pre-fix experiments).

- Recent updates (May 2026, `te_aggregate.py`):
  - **Text-profile warmup vs phase 2:** `--profile_warmup True|False` (default **True**). Phase 1 only builds per-participant text profiles; phase 2 always adapts from the **seed** program. Removed: global aggregate evolution, elite pool over pooled trials, `aggregate/best_aggregate_program.py`, and `--aggregate_iterations`.
  - **Hard-participant early stop vs reporting:** **`--early_stop True|False`** (default **False**). When **True**, phase-2 adaptation can **break** the loop when best train loglik stays below **`--hard_participant_train_loglik_threshold`** after **`--hard_participant_warmup_iters`**. When **False**, the run finishes all **`--n_iterations`**, and final CSV / `results.json` metrics come from the **best program after the last iteration** (`elite_parents[0]`). To **continue** iterations after early stop fires but **report** metrics from the first stop, use **`--debug_continue_after_early_stop`** (freezes final reporting to the early-stop snapshot while the loop keeps going for logs). Same flags exist in **`te_dr.py`**.
  - **`debug_continue_after_early_stop` wiring:** `run_evolution` now accepts this keyword (fixes `TypeError` when `main()` passed it). On first early stop, optional snapshot fields support the frozen-reporting path above.
  - **Parent selection:** `--sample_parents` is **on by default** (`BooleanOptionalAction`; disable with `--no-sample_parents`). When on, each iteration draws `min(sample_size, len(elite))` parents **uniformly at random without replacement** from the trimmed elite pool (not “top‑k by fitness”). When off, behavior matches the old rule: first `sample_size` programs after sorting by fitness. RNG uses `--split_seed` + iteration index (and participant id where applicable). **Not applied** to `gridworld_ensemble` member pools.
  - **Elite pool cap:** `--elite_pool_size N` (optional). If omitted, cap remains `max(2 * sample_size, 20)` as before. If set, keep the top `max(1, N)` programs after sorting. Smaller pools concentrate sampling mass; larger pools increase diversity under `--sample_parents`.

- Recent updates (May 2026, analysis + baseline tooling):
  - **Choices13k proposal plotting (`analysis/code/choices13k/proposal_graph.py`):** streamlined to proposal-focused outputs with clearer labels and cleaner defaults; includes delta-from-Centaur and grouped score views plus optional PDF export via `--save_pdf`.
  - **Participant reward-space map (`analysis/code/choices13k/participant_ev_risk_map.py`):** current plot uses `ΔPositiveReward` (x) and `ΔNegativeReward` (y), overlays both train+test trials, and renders a single-panel red/green `Predicted P(B)` map with participant-specific switching via `--participant_id`.
  - **Participant split/data utilities:** train/test trial JSONs for participants are maintained under `analysis/data/choices13k/` (e.g., `participant_2_train_trials.json`, `participant_2_test_trials.json`), with helper extraction script `analysis/code/choices13k/extract_participant_split.py`.
  - **Evidence artifacts:** added `analysis/code/choices13k/participant_evidence_table.py` (figure/table summary) and `analysis/code/choices13k/participant_evidence_csv.py` (raw CSV export) for proposal-ready behavioral evidence.
  - **Centaur baseline logging (`baseline_methods/Centaur.py`):** evaluation now supports test-only reporting for choice13k/cpc18/mixed_gambles workflows and writes per-trial predictions vs actual actions to `run_dir/log/predictions_vs_actual.csv`.

- Start your local server (example): `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`.
- Run with the new mode switch to route LLM calls externally:  
  `python plot_and_eval.py --baseline_model ROTE --mode local --model_name Qwen/Qwen2.5-7B-Instruct --llm_server_url http://localhost:8000/v1 --llm_api_key EMPTY`
- When `--mode local` is set, no models are loaded in-process; all prompts go to the OpenAI-compatible vLLM server. Default mode preserves the original in-process behavior.

OpenEvolve (Choice13k, TE/Centaur-compatible participants)

- Main runner: `baseline_methods/openevolve.py`.
- Uses the same participant selection source and semantics as TE/Centaur (`datasets/choice13k/valid_participant_ids.json` with `--participant_scope single|range|all`, ordinals for `range`).
- Supports the same split controls (`--split_mode within_participant|across_participants`, `--split_ratio`, `--split_seed`).
- Default output dir: `generated_outputs/choice13k/openevolve/run_YYMMDD_HHMMSS/` with CSVs (`participants_summary.csv`, `participant_details_loglik.csv`, `summary_loglik.csv`) and per-participant OpenEvolve artifacts.
- Command logging: writes `run_dir/log/command.txt`.
- Failure policy is explicit: evaluator prints `[FATAL]` + traceback and returns `fatal_failure=1.0`; runner raises by default (unless `--allow_failure`).
- Example run (matches Centaur/TE participant setting for ordinals 0..9):
  `python baseline_methods/openevolve.py --dataset choice13k --seed_path persona_code_example/choice13k/prospect_theory.py --n_iterations 100 --n_candidates 10 --model_name Qwen/Qwen2.5-7B-Instruct --mode local --llm_server_url http://localhost:8000/v1 --llm_api_key EMPTY --fitness_metric loglik --participant_scope range --range_start_ordinal 0 --range_end_ordinal 9 --split_mode within_participant --split_ratio 0.9 --split_seed 0`

Wandb logging

- Enabled by default; disable with `--no_log True`.
- Logs the same metrics written to CSV (accuracy, first_step_accuracy, accuracy_after_flip, timing, program_length, num_hypothesis, epoch, model/llm_model, mode).

Choice13k (MindAsCode)

- Run: `python plot_and_eval.py --dataset choice13k --baseline_model ROTE --bootstrap --n_hypothesis 10 --mode local`
- Behavior mirrors gridworld MindAsCode: programs are generated, saved to `generated_outputs/choice13k/participant_x/`, executed on train/test splits, and metrics are logged per participant. Gridworld flow is unchanged.
- Prompt modes: `--prompt_mode strict` uses parametrized program prompts; `--prompt_mode non_strict` (default) uses standard prompts.
- Iteration structure: Outer loop processes participants sequentially (0 to num_agents_to_sample-1). Inner loop runs `num_epochs` independent bootstrap iterations per participant (each epoch generates fresh programs and evaluates independently, like different random seeds).

Gridworld evaluation modes

- Loop mode: `--loop_mode random` (default) randomly samples problem configs and agent types per epoch. `--loop_mode sequential` systematically evaluates all (num_blocks, num_walls) combinations (3-7 blocks × 1-4 walls = 20 configs) sequentially.
- Sequential mode: Each epoch uses a different problem config. The number of agent types evaluated per epoch is controlled by `--num_agents_to_sample` (default: 1). With `--num_agents_to_sample=1`, each epoch evaluates agent type 0 for that problem config. With `--num_agents_to_sample=10`, each epoch evaluates all 10 agent types for that problem config. CSV saved to experiment folder (`generated_outputs/gridworld/run_XXX/results.csv`) with `num_blocks` and `num_walls` columns. Best programs tracked per agent type in `epoch_X/epoch_X_agent_types.json` (saved inside each epoch directory). When multiple agents are evaluated, the JSON includes information for all agent types.
- Random mode: Each epoch randomly samples a problem config and agent type(s). The number of agent types per epoch is controlled by `--num_agents_to_sample` (default: 1). CSV saved to fixed location `results/ROTE/...`.

Template Evolution

- Iterative evolution loop over executable Choice13k, Gridworld, and CPC18 Track II programs, combining ROTE's program-based modeling with evo's evolutionary control flow.
- Files: `Template_evo.py` (strict mode - parameter-only evolution), `Template_evo_non_strict.py` (non-strict mode - full program evolution), and `Template_evo_exp_para.py` (except-parameters mode - full program evolution with parameter preservation).
- Recent non-strict Choice13k updates:
  - `--fitness_metric {accuracy,loglik}` enables train log-likelihood as an optional fitness (higher is better), while accuracy is still computed and logged for debugging/plots.
  - Choice model output is now strict probability semantics for Choice13k: `choose(problem, history)` must return a Python `float` in `[0,1]` (invalid outputs fail explicitly).
  - New split controls: `--split_mode {within_participant,across_participants}`, `--split_ratio` (train ratio), `--split_seed`.
  - `across_participants` mode (Choice13k only) splits selected participants into train/test groups and uses all trials per group; artifacts are simplified to `seed_program.py`, `iterations/`, `iterations.csv`, and `summary.csv`.
  - `--max_prompt_train_trials` (default very large): if there are more train trials than this cap, `Template_evo_non_strict.py` **randomly subsamples** that many trials **only for the LLM generation prompt** (seed aligned with `--split_seed`). **Evaluation still uses the full train/test trial lists.** Use `--max_prompt_train_trials 0` to disable capping (full train set in every prompt; large `across_participants` runs can exceed model context). Subsampling changes which trials appear and their order in the prompt; each trial’s own `history` field is unchanged.

Choice13k:
- Run: `python Template_evo.py --dataset choice13k --participant_id 0 --n_iterations 5 --n_candidates 10 --mode local --model_name Qwen/Qwen2.5-7B-Instruct`
- Multiple participants: Use `--num_agents_to_sample N` (without `--participant_id`) to process participants 0 to N-1 sequentially. Wandb metrics use participant-specific keys (e.g., `p0_train_accuracy`, `p1_test_accuracy`).
- Process: Starts from seed program (`persona_code_example/vanilla.py`), generates 10 candidate variants per iteration using LLM, evaluates each on Choice13k train/test splits (fixed 80:20), reports performance, and uses best performers as parents for next generation. Results saved to `generated_outputs/choice13k_ROTE_evo/run_TIMESTAMP/participant_X/`.
- Output structure: Seed program saved once in experiment folder (`run_TIMESTAMP/seed_program.py`). Each participant folder contains `results.json` (baseline + overall best train/test accuracies with program IDs) and `iteration_X/metrics.json` (per-iteration results with program IDs). `participants_summary.csv` in experiment folder tracks all participants' best train/test accuracies (updated after each participant).
- Wandb logging: Enabled by default (project "ROTE_evo"), logs iteration metrics with participant prefixes (e.g., `p0_train_accuracy`, `p0_test_accuracy`, `p0_n_valid`). Disable with `--no_log`. Metrics are also saved locally to `wandb_metrics.jsonl` in each participant's output directory.
- Standalone implementation: Reimplements ROTE-style program generation and evaluation logic without direct imports from ROTE/evo modules.
- Train/test split: Fixed 80:20 split (first 80% train, last 20% test), preserving temporal order.
- Elite parent selection (`--sample_size`, default: 10, Template_evo_non_strict and Template_evo_exp_para only): Controls how many parent programs are used to generate each child. See Gridworld section for details.
- Except-parameters mode (`Template_evo_exp_para.py`): Full program code evolution while preserving all parameter values from the seed program. Parameters are extracted from the seed program's "# Parameters" section and injected back into all generated variants. This allows exploring different program structures and logic while keeping parameter values fixed. Example: `python Template_evo_exp_para.py --dataset choice13k --participant_id 0 --n_iterations 5 --n_candidates 10 --mode local --model_name Qwen/Qwen2.5-7B-Instruct`. Results saved to `generated_outputs/choice13k_ROTE_evo_exp_para/run_TIMESTAMP/participant_X/`.

Gridworld:
- Run: `python Template_evo.py --dataset gridworld --seed_path persona_code_example/gridworld/block3wall1.py --data_path data --loop_mode sequential --num_agents_to_sample 1 --num_epochs 1 --num_blocks 3 --num_walls 1 --agent_id 0 --n_iterations 100 --n_candidates 10`
- Auto-detection: If `--seed_path` is not provided, Template_evo automatically finds template programs in `persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/` based on the agent_id. This works seamlessly with programs collected from ROTE output using `utils/collect_template_program.py`.
- Evaluation: Uses the same evaluation logic as ROTE (`eval_fsm_bootstrap` from `plot_and_eval.py`). Uses `make_dataloader` for consistent data loading, ground truth observations from data (not simulated), and exact action extraction/comparison matching ROTE's logic.
- Strict mode (`Template_evo.py`): Parameter-only evolution (even if template has no parameters, keeps definition consistent). For Gridworld, this means the LLM is prompted to generate parameter values, but the program structure remains fixed.
- Non-strict mode (`Template_evo_non_strict.py`): Full program code evolution. Generates entirely new program implementations, not just parameter tuning.
- Non-strict participant scope (choice13k / cpc18 / mixed_gambles):
  - `--participant_scope single` (default): use raw id via `--single_participant_id`.
  - `--participant_scope range`: use `--range_start_ordinal` / `--range_end_ordinal` (inclusive), where ordinals index into `datasets/*/valid_participant_ids.json`.
  - `--participant_scope all`: run all valid ids from JSON; optional cap via `--all_max_participants` (first N).
  - `--filter_mixed_gambles` is optional (default off). When enabled, mixed-gambles ordinals resolve against `datasets/mixed_gambles/valid_participant_ids_gain_loss.json`; otherwise they use `datasets/mixed_gambles/valid_participant_ids.json`.
  - Legacy participant-selection flags (`--participant_id`, `--all_data`) were replaced in non-strict mode by the scope-based API above.
- Except-parameters mode (`Template_evo_exp_para.py`): Full program code evolution while preserving all parameter values from the seed program. Parameters are extracted from the seed program and injected back into all generated variants. This allows exploring different program structures and logic while keeping parameter values fixed. Example: `python Template_evo_exp_para.py --dataset gridworld --num_blocks 3 --num_walls 1 --agent_id 0 --n_iterations 100 --n_candidates 10 --mode local --model_name Qwen/Qwen2.5-7B-Instruct`. Results saved to `generated_outputs/gridworld_ROTE_evo_exp_para/run_TIMESTAMP/agent_{agent_id}/`.
- Single problem config (multiple agent types): When `--num_blocks` and `--num_walls` are provided without `--loop_mode sequential`, you can process multiple agent types for that single problem config. Use `--num_agents_to_sample N` to process agent types 0 to N-1. Example: `python Template_evo_non_strict.py --dataset gridworld --num_blocks 3 --num_walls 1 --num_agents_to_sample 10 --n_iterations 100` will process all 10 agent types for the (3 blocks, 1 wall) problem config. Wandb metrics use agent-specific keys (e.g., `a0_train_accuracy`, `a1_train_accuracy`, etc.). Results saved to `generated_outputs/gridworld_ROTE_evo_non_strict/run_TIMESTAMP/agent_{agent_id}/`.
- Sequential mode (`--loop_mode sequential`): Evaluates multiple problem configs sequentially. Each epoch uses a different (num_blocks, num_walls) combination. With `--num_agents_to_sample 10`, each epoch processes all 10 agent types for that problem config. Results saved to `generated_outputs/gridworld_ROTE_evo/run_TIMESTAMP/epoch_X/agent_{agent_id}/` where X is the epoch (problem config) number.
- Wandb logging: Uses agent-specific keys (`a0_train_accuracy`, `a1_train_accuracy`, etc.) when processing multiple agent types, or Gridworld-specific keys (`gw_train_accuracy`, `gw_test_accuracy`) for single agent runs. Metrics are also saved locally to `wandb_metrics.jsonl` in each agent's output directory. Run names include dataset prefix (e.g., `gridworld_non_strict_TIMESTAMP` or `choice13k_non_strict_TIMESTAMP`).
- Elite parent selection (`--sample_size`, default: 10, Template_evo_non_strict and Template_evo_exp_para only): Controls how many parent programs are used to generate each child. These modes maintain an elite set of top-performing programs across all iterations (sorted by train accuracy). Each iteration selects `sample_size` parents from this elite set (always including the best parent first) and passes them to the LLM, which generates variants combining ideas from multiple parents. This helps prevent regression by maintaining diversity and ensuring the best programs are always available as parents. The elite set keeps the top `max(sample_size * 2, 20)` programs to maintain diversity. All valid candidates from each iteration are added to the elite set, which is then sorted and trimmed to the top performers. When `sample_size=1`, only the best parent is used (backward compatible behavior).
- **Gridworld ROTE code setting** (non-strict, non-sequential only): When `--dataset gridworld` and `--loop_mode` is not `sequential`, evaluation follows the official ROTE code protocol (reference: `plot_and_eval.py`, `baselines/gridROTE.py`). **Per episode**: (1) One trajectory is sampled from the test split (`make_dataloader(..., training=False)`). (2) Prefix = exactly first 20 steps (fixed; `gridworld_prefix_to_text` serializes this). (3) K initial candidates are generated **inside the episode loop**, conditioned on this episode’s prefix; the prompt includes the serialized prefix trajectory. (4) Each candidate is evolved for N iterations; **evolution fitness uses only prefix accuracy** (train_acc on first 20 steps). Parent selection and candidate ranking use train_acc only; **test_acc is never** in LLM prompts, parent selection, or ranking (test_acc may be computed and logged only). (5) **All evolution prompts** include the serialized prefix trajectory, current parent code, prefix accuracy (X / 20), and optional mismatch summary. (6) **Ensemble weights** use **raw prefix correct counts** only: `prefix_score_i` = number of correct predicted actions on the first 20 steps (integer); `weights = softmax(prefix_scores)` with `score = np.array(prefix_scores)`, `weights = np.exp(score - score.max()) / sum(...)`. Do not use train_acc or any normalized metric for weighting. Weights are frozen before predicting future steps. (7) Ensemble is evaluated on future steps with teacher-forced GT states. **Output**: `run_TIMESTAMP/episodes_summary.csv`, `episode_i/meta.json`, `episode_i/candidate_j/iteration_t/parents/parent_<t>.py`, `episode_i/ensemble/weights.json`. **CLI**: `--num_episodes` (default 10), `--num_blocks`, `--num_walls`, `--agent_id` (default 0). Overall metric: mean `episode_test_acc` (ensemble) over episodes.

Gridworld ensemble (ablation, ROTE-aligned):
- `--dataset gridworld_ensemble`: Same data and evolution logic as gridworld. K = `--sample_size` programs evolved in parallel (separate elite pools, no cross-breeding). **Hypothesis selection**: first K programs by fitness (order preserved); optional `--top_k` (0 = use all): if `top_k > 0` and `< K`, keep top_k by weight and renormalize. **Weights**: score_h = number of correct predictions on first 20 steps (init 1e-6); `scores = scores - max(scores)`, `weights = exp(scores) / sum(exp(scores))` (no epsilon in normalization). **Test aggregation**: weighted one-hot — for each future step, `pi[action] += weight` per program; **tie-aware accuracy**: if `pi[gt] == max(pi)`, add `1/num_max`. **Teacher-forced**: future-step observations from dataset states (no env.step). Per-iteration logging: `a{agent_id}_test_accuracy`, `a{agent_id}_best_program_id`. Same CLI as gridworld. Example: `python Template_evo_non_strict.py --dataset gridworld_ensemble --num_blocks 3 --num_walls 1 --agent_id 0 --sample_size 3 --n_iterations 5 --n_candidates 10`. Output: `generated_outputs/gridworld_ensemble/non_strict/run_TIMESTAMP/agent_{id}/`.

CPC18 Track II (Individual Behavior):
- Run: `python Template_evo.py --dataset cpc18 --participant_id 0 --seed_path persona_code_example/cpc18/hard.py --data_path datasets/cpc18 --n_iterations 100 --n_candidates 10 --mode local --model_name Qwen/Qwen2.5-7B-Instruct`
- Dataset: CPC18 Track II focuses on individual-level modeling. All predictions are conditioned on a single participant. Problems used in testing are familiar (already observed during training). The task is prediction/completion, not generalization to new problems.
- Data sources: Training uses `datasets/cpc18/raw-comp-set-data-Track-2.csv` (trial-level with real actions/feedback). Testing uses `datasets/cpc18/Data-to-predict-Track-2.csv` (block-level B-choice rates for MSE computation). Default data path is `datasets/cpc18` if `--data_path` is not specified.
- Template program: Uses `persona_code_example/cpc18/hard.py` by default. The template implements `choose(problem, history)` where `problem` contains CPC18 parameters (Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr) and `history` is a list of previous actions and feedback.
- Evaluation metrics: Reports both trial-level accuracy (auxiliary metric, same as Choice13k) and block-level MSE (official CPC18 metric). MSE is computed as `100 * mean((predicted_block_rate - observed_block_rate)^2)` averaged over all 5 blocks and all problems, matching the baseline implementation exactly.
- Train/test split: **NO artificial split** - uses ALL trials from `raw-comp-set-data-Track-2.csv` for both training (parameter evolution) and testing (predictions). This matches the official CPC18 Track II protocol. Block-level MSE is computed against observed rates from `Data-to-predict-Track-2.csv`. History is built sequentially as in Choice13k (accumulating actions and feedback).
- LLM guidance: For CPC18, the LLM receives both training accuracy and training MSE metrics to guide parameter/program evolution. This exposes the accuracy-MSE misalignment, allowing the LLM to search for parameters that reduce MSE while maintaining reasonable accuracy. Parent selection still uses training accuracy (not MSE) to maintain evolution stability.
- Modes: All three Template Evolution modes are supported: strict (`Template_evo.py` - parameter-only), non-strict (`Template_evo_non_strict.py` - full code evolution), and except-parameters (`Template_evo_exp_para.py` - full code evolution with parameter preservation). Default seed path is `persona_code_example/cpc18/hard.py` for all modes.
- Output structure: Results saved to `generated_outputs/cpc18_ROTE_evo/run_TIMESTAMP/participant_X/` (or `cpc18_ROTE_evo_non_strict`/`cpc18_ROTE_evo_exp_para` for other modes). Wandb metrics use participant-specific keys (e.g., `p0_train_accuracy`, `p0_train_mse`, `p0_test_accuracy`, `p0_test_mse`). Run names include dataset prefix (e.g., `cpc18_strict_TIMESTAMP` or `cpc18_non_strict_TIMESTAMP`). Metrics are also saved locally to `wandb_metrics.jsonl`.
- Individual-level modeling: Start with `participant_id=0`. Do NOT aggregate across participants. Each participant's data is processed independently.

Mixed Gambles:
- Dataset: `--dataset mixed_gambles`. CSV at `datasets/mixed_gambles/data_all_2021-01-08.csv`. Rows filtered by `subject == participant_id`. Each row is one independent trial (no temporal dependence; history always empty).
- Trial format (choice13k-compatible): Option A = gamble `rewards [gain, loss]`, `probs [0.5, 0.5]`; Option B = certain `rewards [cert]`, `probs [1.0]`.
- Raw CSV encoding: `took_gamble` with `1` = chose the gamble, `0` = chose the certain option.
- TE mixed-gambles trial `action` after conversion: `action = 1 - took_gamble`, so `0` = `gamble_A` (accept the gamble), `1` = `gamble_B` (certain / reject the gamble). This matches the general TE option index (`0` = Option A, `1` = Option B).
- Fixed 80/20 train/test split (reproducible shuffle, RNG seed 42), not raw row order.
- Strict mode (`Template_evo.py`): Parameter-only evolution. Prompts from `prompts/Template_evo/mixed_gambles/strict/`. Default seed: `persona_code_example/hard_Qwen.py`. Run: `python Template_evo.py --dataset mixed_gambles --participant_id 101`.
- Non-strict mode (`Template_evo_non_strict.py`): Full program evolution. Prompts from `prompts/Template_evo/mixed_gambles/non_strict/`. Default seed: `persona_code_example/hard_Qwen.py`. Output: `generated_outputs/mixed_gambles/non_strict/run_TIMESTAMP/participant_{id}/`.

`--participant_scope all` mode (Template_evo_non_strict.py):
- Supported datasets: `choice13k`, `cpc18`, `mixed_gambles`.
- Startup prints one line indicating scope=all and total participants from precomputed valid-id JSON.
- Output is compact and updated after each participant: only `seed_program.py`, `participants_details.csv`, and `summary.csv` are kept in the run folder (no per-participant `iteration_*`, candidate code, or `results.json` artifacts).
- `participants_details.csv` schema: `participant_id, train_fitness, test_fitness, total_runtime, seed_program_train_fitness, seed_program_test_fitness`.
- `summary.csv` schema: `num_of_participants, avg_train_fitness, avg_test_fitness` (single row).
- Wandb in scope=all logs only two participant metrics: `pX_train_fitness` and `pX_test_fitness`.
- Optional cap for scope=all: `--all_max_participants N` (first N valid ids in JSON order).

Collecting ROTE programs for Template_evo:
- After running ROTE on gridworld, use `python utils/collect_template_program.py --exp_folder generated_outputs/gridworld/run_XXX --epoch 0` to extract best programs from ROTE output.
- This organizes programs by problem config (num_blocks, num_walls) in `persona_code_example/gridworld/num_blocksX_num_wallsY/` with names like `block_cycle_agent0.py`.
- Agent IDs map to hand-designed programs alphabetically: agent_id 0 = block_cycle.txt, agent_id 1 = clockwise_patrol.txt, etc.
- Once collected, Template_evo can auto-detect these programs when `--seed_path` is not provided.

Extracting best accuracies from old runs: For experiments without `participants_summary.csv`, use `python utils/extract_best_accuracies.py <run_dir>` to extract best train/test accuracies per participant from iteration metrics. Outputs CSV to `run_dir/participants_summary.csv`. Use `--include_program_ids` to see which iteration/candidate had best results.

---

## Recent updates (append-only, May 2026 — `te_aggregate.py` and related)

- **Phase 1 is text-profile warmup, not aggregate code evolution.** Global aggregate iterations, pooled elite selection, `run_dir/aggregate/`, and `--aggregate_iterations` are removed. Phase 1 runs **one LLM call per participant** using `prompts/Template_evo/<dataset>/text_profile/text_profile.txt`, appends a token-budgeted prefix of that participant’s **train** trials, and writes `participant_<id>/profile.txt` plus `participant_<id>/text_profile_meta.json`. Phase 2 always adapts from the run **seed** program (`--seed_path`); when a profile exists, it is injected into adaptation prompts (optional `participant_text_profile` block plus the usual “modify seed minimally” instructions).

- **`--profile_warmup` takes `True` or `False` explicitly** (e.g. `--profile_warmup False` to skip phase 1). Default is `True`. There is no `--no-profile_warmup` flag.

- **CPC18 text-profile trial budget.** For `dataset=cpc18`, serialized **trials only** are capped at approximately **6000 tokens** (`_TEXT_PROFILE_CPC18_MAX_TRIAL_BLOCK_TOKENS`); the profile template and trials header stay full. Other datasets still cap the **entire** user message at the previous ~10k approximate-token budget. Logs and `text_profile_meta.json` can record `max_trial_block_tokens_approx` for CPC18. Token counts remain a **heuristic** (~4 characters per token); real server token counts can be higher, so very long templates plus `max_tokens` for the profile reply can still hit provider context limits.

- **RBU selection score (default).** With **`--use_rbu True`** (default), the run first writes **`{run_dir}/instruction.txt`** from `prompts/Template_evo/{dataset}/text_profile/prepare_instruction.txt` (one LLM call). Before phase 2, a **single combined** structure-scoring LLM call builds a prompt from **`use_instruction.txt` + `instruction.txt` + training trials for all selected participants** (symmetric trial cap per participant if the estimated prompt size would exceed **`--structure_prompt_max_tokens`**; token estimate uses tiktoken `cl100k_base` when available, otherwise `ceil(len(text)/3.5)`). Raw model output is saved to **`analysis/Structure_score_all.txt`**; **`utils.rbu.parse_all_participant_structure_scores`** reads one JSON object keyed by **`participant_<id>`** with an **`evidence`** map of numeric scores; **S** is the **mean** of those values (each clipped to **[0,1]**); a JSON **`structure_score`** field is ignored. **RBU = clip01(BIR − w·S)** with **`--structure_weight`** as **w** (default **0.5**; alias **`--rbu_structure_weight`**). `_compute_selection_score` uses `selection_score = train_loglik - λ * (RBU ** 2) * confidence_penalty` with **`--uncertainty_lambda`** default **`30.0`** (alias **`--rbu_lambda`**). Run-level diagnostics are written once to **`analysis/behavioral_inconsistency_rate.csv`** (includes **`BIR`**, **`behavioral_inconsistency_rate`**, **`rbu`**, **`structure_score`**, etc.; numeric rounding on write). Parsed evidence components remain in **`results.json`** / participant summaries as **`structure_components`** when RBU is enabled. Scalar RBU/BIR/structure fields are **not** sent to Weights & Biases as separate series (only `selection_score`, `confidence_penalty`, and related fitness metrics are logged there).

- **`--use_rbu False` (ablation).** Skips structure-score LLM steps; phase 2 uses **BIR** as the regularization rate in the same squared formula with **`--uncertainty_lambda`** (alias **`--rbu_lambda`**).

- **Adaptation prompt wording vs RBU.** If the **regularization rate** (RBU when `--use_rbu True`, else BIR) is strictly below **`--uncertainty_threshold`** (default **0.6**; alias **`--rbu_threshold`**), phase-2 **extra** instructions omit score/confidence wording; otherwise the full regularization explanation (RBU- or BIR-worded depending on `--use_rbu`) and the note about avoiding unjustified extreme probabilities are kept.

- **Optional overrides:** `--prepare_instruction_path` and `--use_instruction_path` override the default `Template_evo/{dataset}/text_profile/{prepare_instruction,use_instruction}.txt` paths. Parsing helpers live in **`utils/rbu.py`** (`extract_first_json_object`, `count_tokens_approx`, `parse_all_participant_structure_scores`, `parse_structure_score` for single-object payloads, `compute_rbu`); parse failures surface as run-stopping errors (no silent fallbacks for **S**).

---

## Summary (end of `AGENT.md`)

This section is a **short orientation** for anyone (human or agent) touching MindAsCode; the sections above remain the detailed reference.

- **`te_aggregate.py` (choice13k / cpc18 / mixed_gambles):** Two-phase runs: optional **text-profile warmup** (per participant, `participant_<id>/profile.txt`), then **per-participant adaptation** from the run seed program with held-out loglik, CSV summaries under the run root, and optional W&B. There is **no** cross-participant aggregate evolution directory.

- **`teh.py` (Psych-101 TEH):** Per-participant loglik evolution on numbered Psych-101 aliases + optional **`--global_phase`**; see **`### Latest updates (teh.py — TEH, May 2026)`** above.

- **`teh_transfer.py` (cross-task transfer):** Multi-dataset population global evolution + leave-one-out transfer from other datasets’ best programs; config **`configs/teh_transfer.yaml`**; outputs under **`generated_outputs_transfer/teh_transfer/`**. See **`### teh_transfer.py`** above.

- **`te_dr.py` (Choice13k focus):** Single-phase **data-driven** evolution with train/val/test by **problem block**, default loglik prompts from **`prompts/te_data_driven/evolution/choices13k.txt`**, default run root **`generated_outputs/choice13k/te_dr/run_*`**. See the **`### te_dr.py`** subsection below for CLI constraints and behavior.

- **RBU vs BIR:** With **`--use_rbu True`** (default), the run uses **one** dataset instruction file, **one** combined structure-scoring response in **`analysis/Structure_score_all.txt`**, and per-participant **RBU** in selection and adaptation prompts. With **`--use_rbu False`**, structure scoring is skipped and **BIR** substitutes in the same regularization slot. Primary flags: **`--uncertainty_lambda`**, **`--uncertainty_threshold`**, **`--structure_weight`**, **`--structure_prompt_max_tokens`**, **`--prepare_instruction_path`**, **`--use_instruction_path`** (aliases **`--rbu_lambda`**, **`--rbu_threshold`**, **`--rbu_structure_weight`** still work).

- **Template evolution family:** `Template_evo.py` (strict), `Template_evo_non_strict.py`, `Template_evo_exp_para.py` cover choice13k, gridworld, CPC18, and mixed gambles with dataset-specific prompts and split controls (`--split_mode`, `--split_ratio`, `--split_seed`, participant scope via **`datasets/*/valid_participant_ids.json`**).

- **Baselines and analysis:** OpenEvolve (`baseline_methods/openevolve.py`), Centaur (`baseline_methods/Centaur.py`), ROTE-style evaluation via **`plot_and_eval.py`**, and choice13k analysis utilities under **`analysis/code/choices13k/`** (see bullets above for paths and behaviors).

- **Local LLMs:** **`--mode local`** routes generation to an OpenAI-compatible server (e.g. vLLM); no in-process model load for that path.

### Latest updates (te_aggregate / choice13k)

- **`--phase_option {all,score,evolution}`** (choice13k, `within_participant` only; not with `participant_scope=all`). **`all`** — unchanged full pipeline (BIR, structure-score LLM, evolution). **`score`** — scoring only; writes **`analysis/behavioral_inconsistency_rate.csv`** then exits. **`evolution`** — skips scoring LLMs; **recomputes BIR** on the current train split; with **`--use_rbu True`** reads structure JSON from **`--structure_path`** (typically a prior run’s **`analysis/Structure_score_all.txt`**), copies it into the new run’s **`analysis/`**, **recomputes RBU** with this run’s **`--structure_weight`** (and related args), writes **`behavioral_inconsistency_rate.csv`**, then runs phase-2 evolution.

- **CLI renames (backward-compatible aliases kept):** **`--uncertainty_lambda`** / **`--uncertainty_threshold`** / **`--structure_weight`** replace the old `rbu_*` names for the regularization rate and structure multiplier **w** in **RBU = clip(BIR − w·S)**.

- **Structure-score JSON** (`utils/rbu.py`): **`S`** is the **mean** of numeric values under each participant’s **`evidence`** object (values clipped to **[0,1]**); a top-level **`structure_score`** field in model JSON is **ignored** if present.

- **`analysis/behavioral_inconsistency_rate.csv`:** includes a **`BIR`** column; numeric cells are **rounded to two decimals** on write.

- **Adaptation prompts:** when **`num_diagnostic_trials`** builds diagnostic trial text, a short **Note** clarifies that those trials are illustrative only and must not be treated as behavioral labels.

### `te_dr.py` (data-driven Choice13k evolution)

- **Role:** Companion to `te_aggregate.py`, but **not** two-phase profile + adaptation. For Choice13k it runs **single-phase** full-program evolution with **train / validation / test** defined by **problem blocks**: `--split_ratio` is the **train fraction** of blocks; the remaining blocks are split evenly into validation and test. Per-trial `history` stays **within the same block** only. There is **no** `--data_driven_mode` CLI flag; the main Choice13k path turns on data-driven behavior inside `main()` when running the standard evolution loop.

- **Requirements (Choice13k main path):** `--dataset choice13k`, **`--fitness_metric loglik`**, and **`--split_mode within_participant`**. The script prints an error and exits if these are wrong. **`--split_mode across_participants`** is only supported for a separate pooled-trial experiment branch (not the per-participant data-driven loop above).

- **Participant selection:** Same JSON ordering as other TE tools: **`--participant_scope single|range|ordinals|all`** with **`--single_participant_id`**, **`--range_start_ordinal` / `--range_end_ordinal`**, **`--ordinals`** (space-separated 0-based indices into `datasets/choice13k/valid_participant_ids.json`, not raw ids), or **`--all_max_participants`** with `all`. **`--filter_mixed_gambles`** applies to mixed_gambles only.

- **Evolution objective:** Parents are ranked by **train** log-likelihood (selection). Candidates are still evaluated on **train and validation** log-likelihood where wired; generation uses **`prompts/te_data_driven/evolution/choices13k.txt`** and can include up to **`--max_prompt_val`** serialized **validation trials** in the prompt, plus **parent `train_loglik` / `val_loglik` on the same line** when validation scores exist.

- **Outputs:** If **`--output_dir`** is omitted, auto runs go under **`generated_outputs/choice13k/te_dr/run_YYMMDD_HHMMSS/`** (per-participant folders `participant_<id>/` when processing multiple participants). Range/ordinals multi-participant runs can also write aggregate **`participants_summary.csv`**, **`participant_details_loglik.csv`**, and **`summary_loglik.csv`** under that run root. **`participant_scope=all`** uses the same compact CSV-only layout as Template non-strict (no heavy per-participant artifact tree).

- **Early stop flags:** **`--early_stop True|False`** (default **`False`**) and **`--debug_continue_after_early_stop`** are passed into `run_evolution`. The hard-participant early-stop **break** only runs when **`adaptation_mode=True`** inside `run_evolution` (e.g. the embedded **`_te_aggregate_run_evolution_stage`** / phase-2 adaptation path). The main Choice13k **data-driven** loops call `run_evolution` with **`adaptation_mode=False`**, so **`--early_stop` does not change behavior** on those paths. Thresholds: **`--hard_participant_train_loglik_threshold`**, **`--hard_participant_warmup_iters`**.

- **Other datasets:** `te_dr.py` still supports **gridworld**, **gridworld_ensemble**, **cpc18**, and **mixed_gambles** entry points similar to other TE runners; the **data-driven block-split Choice13k** behavior above is the distinguishing feature.

### Latest updates (`Template_evo_non_strict.py`, May 2026 — prior commits summary)

Short changelog of committed non-strict work (choice13k / cpc18 / mixed_gambles).

- **Parallel runs:** **`--parallel_participants`** (default **True**; **`--no-parallel_participants`** to disable) — nested pool: `participant_workers = max(1, max_workers // n_candidates)`; each participant still runs **`n_candidates`** candidate LLM workers. Experiment-level CSVs are written on the main thread only (lock); per-participant trees stay under **`participant_<id>/`**. Applies to **`--phase all|evolution|refine`** and **`--participant_scope all`** (not gridworld / **`across_participants`**).

- **Phased CLI:** **`--phase {all,evolution,refine}`** (default **`all`**). **`evolution`** = evolution only; **`refine`** = refinement-only from **`--prev_exp_path`** (requires each **`participant_<id>/best_program.py`**). Refine path copies prior loglik CSV, clears **`gated_test_loglik`**, then fills it from refinement test eval.

- **Loglik refinement:** **`--refinement_phase`** (on for **`all`**), **`--refinement_iters`**, **`--refinement_val_threshold`** (default **-1.0**). Runs when **`val_loglik < threshold`**. Refinement LLM prompts use **validation trials only** (not train+val). **`gated_test_loglik`** in run CSVs = refinement test loglik when refinement ran; else evolution test loglik. Artifacts under **`participant_<id>/refinement/`**.

- **Parent / elite pool:** **`--sample_parents`** default **True** (uniform random **`sample_size`** parents from elite, without replacement). **`--no-sample_parents`** = top **`sample_size`** programs by fitness after sort — **not** a single parent; once elite has ≥10 programs, up to **10 full `choose()` sources** are pasted into each candidate-generation prompt. Optional **`--elite_pool_size`** caps retained elites (default **`max(2 * sample_size, 20)`**).

- **Prompt token controls:** **`--max_prompt_train_trials`**, **`--max_prompt_trials_per_problem`** (block-aware subsample for LLM only; evaluation still uses full splits). Dataset-specific loglik prompts under **`prompts/Template_evo/<dataset>/non_strict/loglik/`** (choice13k, cpc18, mixed_gambles).

- **Reporting:** Pool-best program after last elite update; paired train/test loglik in CSVs; per-participant artifact **`best_program.py`** (replaces **`best_program_fr_iter*_cand*.py`**).

- **LLM context (operational):** With **`--sample_size 10`**, **`--no-sample_parents`**, and vLLM **`--max-model-len 10240`**, candidate generation often fails from ~iteration 4 onward (~9217 input tokens + 1024 output). Production choice13k non-strict runs that completed reliably used **`--max-model-len 12288`**; same prompt assembly, higher ceiling (~11265 input at limit). Mitigations: raise context, **`--sample_size 1`**, or **`--sample_parents`** (still up to 10 parents unless **`sample_size`** is reduced).

### Latest updates (`teh.py` — TEH, May 2026)

**TEH** (Template Evolution HuggingFace) is a Psych-101–focused fork of `Template_evo_non_strict.py`. WandB project: **`teh`**. Main script: **`teh.py`**.

- **Datasets (CLI `--dataset`):** Eight implemented Psych-101 binary aliases use a **numeric prefix** for output folders and WandB run names (HF `experiment_id` strings are unchanged):

  | ID | CLI alias | HF `experiment_id` |
  |----|-----------|-------------------|
  | 1 | `1peterson2021using` | `peterson2021using/exp1.csv` |
  | 2 | `2plonsky2018when` | `plonsky2018when/exp1.csv` |
  | 3 | `3frey2017cct` | `frey2017cct/exp1.csv` |
  | 4 | `4wulff2018description` | `wulff2018description/exp1.csv` |
  | 5 | `5speekenbrink2008learning` | `speekenbrink2008learning/exp1.csv` |
  | 6 | `6sadeghiyeh2020temporal` | `sadeghiyeh2020temporal/exp1.csv` |
  | 7 | `7hilbig2014generalized` | `hilbig2014generalized/exp1.csv` |
  | 8 | `8flesch2018comparing` | `flesch2018comparing/exp1.csv` |

  Unprefixed legacy names (e.g. `peterson2021using`) are accepted and normalized via **`normalize_psych101_dataset_alias()`**. Loaders: **`data_modules/psych101_binary.py`**, parsers **`data_modules/psych101_parsers.py`**. **`--psych_dataset_split {train,test}`** → `marcelbinz/Psych-101` or `Psych-101-test`. **`mixed_gambles`** stays a local CSV dataset (no numeric prefix).

- **Trial API:** Evolved code implements **`choose(problem, history) -> float`** = **P(action=1)** where action=1 is the **second** option in **`option_keys`**. NL transcripts are parsed once per participant into structured trial dicts; evolution/eval use those dicts (not raw text). Prompt trial lines use **`format_trials_for_prompt()`** (schema-aware one-liners).

- **Metadata paths:** Valid participant lists live under **`datasets/psych101_{train|test}/<numbered-alias>/valid_participant_ids.json`** (e.g. `datasets/psych101_train/1peterson2021using/`). On startup, **`utils/teh/participant_ids.py`** auto-scans, writes when missing, and **renames** legacy unprefixed metadata folders when found. Manual refresh: **`python utils/tools/collect_teh_participant_ids.py --dataset 1peterson2021using --psych_dataset_split train|test`**.

- **Run layout & prompts:** Outputs under **`generated_outputs/psych101_{train|test}/teh/<numbered-alias>/run_TIMESTAMP/`** (e.g. `.../teh/1peterson2021using/run_YYMMDD_HHMMSS/`); WandB names follow the same prefix (`1peterson2021using_teh_train_...`). **`utils/teh/teh_runtime.setup_teh_run_prompts()`** creates **`run_*/prompts/`** (`infer_single_choice.txt`, **`refine.txt`**, templates, seed copy). Refinement and candidate generation read **`run_prompts_dir`** when set (not repo-default choice13k paths). **`--no_llm_prompt`** merges base loglik text + dataset description without an extra LLM call.

- **Evolution CLI (inherited from non_strict):** **`--parallel_participants`**, **`--phase {all,evolution,refine}`**, loglik refinement (**`--refinement_iters`**, **`--refinement_val_threshold`**), **`--sample_parents` / `--no-sample_parents`**, **`--elite_pool_size`**, **`--max_prompt_train_trials`**, **`--max_prompt_trials_per_problem`**, optional **`--global_phase`** (pooled evolution; **`peterson2021using`** + **`within_participant`** only for **`across_participants`**). Default **`--fitness_metric loglik`**; accuracy mode is rejected for TEH participant datasets.

- **Removed / legacy:** CLI no longer exposes **`choice13k`**, **`cpc18`**, or **gridworld** as `--dataset` values (use Psych-101 aliases, e.g. **`peterson2021using`** for Choice13k-style gambles). Gridworld/JAX imports are **lazy** so **`python teh.py --help`** does not require JAX.

- **Package layout:** **`utils/teh/teh_datasets.py`** (registry, path helpers), **`utils/teh/teh_runtime.py`**, **`utils/teh/participant_ids.py`**, **`utils/teh/__init__.py`**.

- **Validation & cluster:** **`analysis/code/psych-101/validate_teh_all_datasets.py`**, **`validate_teh_peterson.py`**. Slurm examples: **`cluster/teh/1choices13k.sh`** (peterson train), **`2cpc18.sh`** (plonsky), **`3mixed_gambles.sh`**.

- **LLM context (same as non_strict):** **`--no-sample_parents`** with **`--sample_size 10`** embeds up to **10 elite programs** per candidate prompt once the pool is full; with vLLM **`--max-model-len 10240`** this typically overflows (~9217 input tokens) from iteration 4+. Prefer **`--max-model-len 12288`** (as in completed choice13k non-strict runs) or **`--sample_size 1`** for prompt parents. This is not psych-101–specific parser bloat; it is shared multi-parent prompt assembly.

- **Seed exploration (`--explore_candidates`, default `0`):** One-shot **before** the evolution loop (phases **`all`** / **`evolution`** only). Generates **`N`** LLM variants from the **seed program only** (`--seed_path`), evaluates them on train (and val/test when applicable), and merges **runtime-valid** programs into the initial elite pool (sorted/capped like normal elites). Artifacts: **`participant_<id>/explore_phase/`**. Disabled when **`0`**. Skipped on **global-phase handoff** (`global_elite_parents` set). Not the same as per-iteration fresh candidates.

- **Per-iteration fresh candidates (`--fresh_n_candidates`, default `0`):** Each evolution/refinement/global iteration, up to **`fresh_n`** children are generated from the **seed/baseline parent only** (not sampled elite parents); the rest use normal parent sampling. **`0` disables** fresh candidates. When **`> 0`**, **`fresh_n` decays automatically** within each phase (no extra flag): `fresh_n = max(1, floor(fresh_n_candidates * (1 - iter_idx / total_iters)))`, clamped to **`n_candidates`**. Example with **`--n_iterations 10`** and **`--fresh_n_candidates 10`**: iteration 0 → 10 fresh / 0 parent-sampled; iteration 9 → 1 fresh / 9 parent-sampled. Refinement uses the same schedule over **`--refinement_iters`**. Requires seed/baseline as fresh parent (evolution passes **`seed_code`**; refinement passes **`fresh_parent_code`**).

- **Combined score (`train_val_loglik` / OpenEvolve `combined_score`):** Refinement pool ranking and Psych-101 OpenEvolve evolution use split-proportional train+val loglik — `(split_ratio·train + val_ratio·val) / (split_ratio + val_ratio)` with `val_ratio = (1 - split_ratio)/2` (e.g. `--split_ratio 0.6` → `(0.6·train + 0.2·val)/0.8`); per-split **`train_loglik`**, **`val_loglik`**, and **`test_loglik`** logging are unchanged.

### `teh_transfer.py` (population-level cross-task transfer)

- **Role:** Extends TEH with **population-level** program evolution and **leave-one-dataset-out** transfer. There are **no shared participants** across datasets; fitness pools **train+val trials across all selected participants** per dataset (not per-participant adaptation). Main script: **`teh_transfer.py`**. Helpers: **`utils/teh_transfer/`** (`config.py`, `evolution.py`, `prompts.py`, `participants.py`).

- **Pipeline:** (1) Load datasets from **`--transfer_config`** (default **`configs/teh_transfer.yaml`**; keys like **`1peterson2021using_test`** → test split). (2) **Global phase** per dataset: same machinery as **`teh.run_global_evolution_phase`** on pooled train+val (**`--global_iters`**, **`--max_prompt_train_trials`**, **`--n_candidates`**, etc.). (3) **Transfer phase** per target: inject each of the **N−1** source datasets’ task description, **one example trial**, and **best global program** into the prompt; evolve on the target for **`--transfer_iters`** with **`--transfer_max_prompt_trials`** target trials (default **1**). Target-side **elite parents** (seed → evolved programs) use the same **`--sample_size`**, **`--fresh_n_candidates`** decay, and parent sampling as TEH global evolution.

- **Outputs:** **`generated_outputs_transfer/teh_transfer/run_TIMESTAMP/<config_key>/`** with **`global/`** (renamed from `global_phase`), **`transfer/`**, per-dataset **`prompts/`**, and run-level **`summary_csv/transfer.csv`** (columns: **`dataset`**, **`global_best`**, **`transfer`**, **`transfer_1st`**). WandB project: **`teh_transfer`**. Optional **`--debug_prompt`** writes **`run_*/debug/prompt_<dataset>.txt`** (one full global-iter-1 and transfer-iter-1 prompt).

- **Participants:** Same scope flags as **`teh.py`** (**`--participant_scope`**, **`range`**, **`all`**, etc.), applied **per dataset**. With **`range`**, **`range_end_ordinal`** is **auto-clamped** per dataset when it exceeds that dataset’s valid-id list (warning logged).

- **Parallelism:** **`--parallel_datasets`** (default **True**): `dataset_workers = max_workers // n_candidates`; each dataset uses **`n_candidates`** LLM workers for child generation.

- **Prompt budget note:** Transfer prompts are heavier than global (fixed **N−1 source program** block). **`--transfer_max_prompt_trials`** can be set high (e.g. 60); TEH truncates trials under **`--hard_prompt_token_cap`** if needed. Source trials stay **1 each** regardless of CLI.

- **Cluster example:** **`cluster/teh_transfer/1.sh`** (vLLM local mode; ensure line continuations after every arg — a missing `\` drops **`--mode local`**).

### `prototype/teh_psych.py` (parse-plan retry, Jul 2026)

- **Role:** Population-level categorical TEH over Psych-101 **train** experiments: LLM parser plan → fixed parser engine → categorical trials → population program evolution. Main script: **`prototype/teh_psych.py`**. Cluster example: **`cluster/teh_psych/1.sh`**.

- **Parse retry (on by default):** Before evolution, each experiment runs a bounded retry loop (**`--max_parse_plan_attempts`**, default **3**) over parse-plan generation, validation, execution, normalization, prediction-trial extraction, categorical validation, minimum prediction-trial count, and action-space checks. A attempt succeeds only when **all** of those pass.

- **Retryable failures** (compact feedback → next LLM prompt): parse-plan prompt/generation/validation/execution errors, no or too few trials, normalization/partitioning errors, categorical validation errors, invalid action-space summary. **Non-retryable** (stop immediately): **`unsupported_current_pipeline`**, **`state_machine_not_implemented`**, and clearly unsupported scalar/verbal task types detected by the parse-plan pipeline.

- **Feedback:** Cumulative structured summary per failed attempt (stage, one-line error, trial counts, first few validation errors, action-space summary, categorical-trial format reminder), hard-capped by **`--parse_plan_feedback_max_chars`** (default **4000**). Full tracebacks stay in per-attempt debug files, not in the LLM prompt.

- **Cache:** **`--reuse_parse_plan_cache`** applies only on **attempt 1**; later attempts always bypass cache so a bad cached plan is not reused.

- **Debug outputs:** Per attempt **`parse_plan_attempt_N/`** under the dataset debug dir; run-level **`parse_plan_retry_summary.json`** with all attempts and final outcome. Existing **`failure.json`**, summary CSV, and JSONL behavior unchanged.

- **Opt out / tuning:** Use **`--max_parse_plan_attempts 1`** for old single-shot parsing. No changes required to existing cluster scripts unless you want to tune retries or disable them.
