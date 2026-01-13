Local vLLM mode

- Start your local server (example): `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`.
- Run with the new mode switch to route LLM calls externally:  
  `python plot_and_eval.py --baseline_model ROTE --mode local --model_name Qwen/Qwen2.5-7B-Instruct --llm_server_url http://localhost:8000/v1 --llm_api_key EMPTY`
- When `--mode local` is set, no models are loaded in-process; all prompts go to the OpenAI-compatible vLLM server. Default mode preserves the original in-process behavior.

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

- Iterative evolution loop over executable Choice13k and Gridworld programs, combining ROTE's program-based modeling with evo's evolutionary control flow.
- Files: `Template_evo.py` (strict mode - parameter-only evolution) and `Template_evo_non_strict.py` (non-strict mode - full program evolution).

Choice13k:
- Run: `python Template_evo.py --dataset choice13k --participant_id 0 --n_iterations 5 --n_candidates 10 --mode local --model_name Qwen/Qwen2.5-7B-Instruct`
- Multiple participants: Use `--num_agents_to_sample N` (without `--participant_id`) to process participants 0 to N-1 sequentially. Wandb metrics use participant-specific keys (e.g., `p0_train_accuracy`, `p1_test_accuracy`).
- Process: Starts from seed program (`persona_code_example/vanilla.py`), generates 10 candidate variants per iteration using LLM, evaluates each on Choice13k train/test splits (fixed 80:20), reports performance, and uses best performers as parents for next generation. Results saved to `generated_outputs/choice13k_ROTE_evo/run_TIMESTAMP/participant_X/`.
- Output structure: Seed program saved once in experiment folder (`run_TIMESTAMP/seed_program.py`). Each participant folder contains `results.json` (baseline + overall best train/test accuracies with program IDs) and `iteration_X/metrics.json` (per-iteration results with program IDs). `participants_summary.csv` in experiment folder tracks all participants' best train/test accuracies (updated after each participant).
- Wandb logging: Enabled by default (project "ROTE_evo"), logs iteration metrics with participant prefixes (e.g., `p0_train_accuracy`, `p0_test_accuracy`, `p0_n_valid`). Disable with `--no_log`.
- Standalone implementation: Reimplements ROTE-style program generation and evaluation logic without direct imports from ROTE/evo modules.
- Train/test split: Fixed 80:20 split (first 80% train, last 20% test), preserving temporal order.

Gridworld:
- Run: `python Template_evo.py --dataset gridworld --seed_path persona_code_example/gridworld/block3wall1.py --data_path data --loop_mode sequential --num_agents_to_sample 1 --num_epochs 1 --num_blocks 3 --num_walls 1 --agent_id 0 --n_iterations 100 --n_candidates 10`
- Auto-detection: If `--seed_path` is not provided, Template_evo automatically finds template programs in `persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/` based on the agent_id. This works seamlessly with programs collected from ROTE output using `utils/collect_template_program.py`.
- Evaluation: Uses the same evaluation logic as ROTE (`eval_fsm_bootstrap` from `plot_and_eval.py`). Uses `make_dataloader` for consistent data loading, ground truth observations from data (not simulated), and exact action extraction/comparison matching ROTE's logic.
- Strict mode (`Template_evo.py`): Parameter-only evolution (even if template has no parameters, keeps definition consistent). For Gridworld, this means the LLM is prompted to generate parameter values, but the program structure remains fixed.
- Non-strict mode (`Template_evo_non_strict.py`): Full program code evolution. Generates entirely new program implementations, not just parameter tuning.
- Single problem config (multiple agent types): When `--num_blocks` and `--num_walls` are provided without `--loop_mode sequential`, you can process multiple agent types for that single problem config. Use `--num_agents_to_sample N` to process agent types 0 to N-1. Example: `python Template_evo_non_strict.py --dataset gridworld --num_blocks 3 --num_walls 1 --num_agents_to_sample 10 --n_iterations 100` will process all 10 agent types for the (3 blocks, 1 wall) problem config. Wandb metrics use agent-specific keys (e.g., `a0_train_accuracy`, `a1_train_accuracy`, etc.). Results saved to `generated_outputs/gridworld_ROTE_evo_non_strict/run_TIMESTAMP/agent_{agent_id}/`.
- Sequential mode (`--loop_mode sequential`): Evaluates multiple problem configs sequentially. Each epoch uses a different (num_blocks, num_walls) combination. With `--num_agents_to_sample 10`, each epoch processes all 10 agent types for that problem config. Results saved to `generated_outputs/gridworld_ROTE_evo/run_TIMESTAMP/epoch_X/agent_{agent_id}/` where X is the epoch (problem config) number.
- Wandb logging: Uses agent-specific keys (`a0_train_accuracy`, `a1_train_accuracy`, etc.) when processing multiple agent types, or Gridworld-specific keys (`gw_train_accuracy`, `gw_test_accuracy`) for single agent runs.

Collecting ROTE programs for Template_evo:
- After running ROTE on gridworld, use `python utils/collect_template_program.py --exp_folder generated_outputs/gridworld/run_XXX --epoch 0` to extract best programs from ROTE output.
- This organizes programs by problem config (num_blocks, num_walls) in `persona_code_example/gridworld/num_blocksX_num_wallsY/` with names like `block_cycle_agent0.py`.
- Agent IDs map to hand-designed programs alphabetically: agent_id 0 = block_cycle.txt, agent_id 1 = clockwise_patrol.txt, etc.
- Once collected, Template_evo can auto-detect these programs when `--seed_path` is not provided.

Extracting best accuracies from old runs: For experiments without `participants_summary.csv`, use `python utils/extract_best_accuracies.py <run_dir>` to extract best train/test accuracies per participant from iteration metrics. Outputs CSV to `run_dir/participants_summary.csv`. Use `--include_program_ids` to see which iteration/candidate had best results.

