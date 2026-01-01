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

Template Evolution (ROTE_evo)

- Iterative evolution loop over executable Choice13k programs, combining ROTE's program-based modeling with evo's evolutionary control flow.
- Run: `python ROTE_evo.py --participant_id 0 --n_iterations 5 --n_candidates 10 --mode local --model_name Qwen/Qwen2.5-7B-Instruct`
- Multiple participants: Use `--num_agents_to_sample N` (without `--participant_id`) to process participants 0 to N-1 sequentially. Wandb metrics use participant-specific keys (e.g., `p0_train_accuracy`, `p1_test_accuracy`).
- Process: Starts from seed program (`persona_code_example/vanilla.py`), generates 10 candidate variants per iteration using LLM, evaluates each on Choice13k train/test splits (fixed 80:20), reports performance, and uses best performers as parents for next generation. Results saved to `generated_outputs/choice13k_ROTE_evo/run_TIMESTAMP/participant_X/`.
- Output structure: Seed program saved once in experiment folder (`run_TIMESTAMP/seed_program.py`). Each participant folder contains `results.json` (baseline + overall best train/test accuracies with program IDs) and `iteration_X/metrics.json` (per-iteration results with program IDs). `participants_summary.csv` in experiment folder tracks all participants' best train/test accuracies (updated after each participant).
- Wandb logging: Enabled by default (project "ROTE_evo"), logs iteration metrics with participant prefixes (e.g., `p0_train_accuracy`, `p0_test_accuracy`, `p0_n_valid`). Disable with `--no_log`.
- Standalone implementation: Reimplements ROTE-style program generation and evaluation logic without direct imports from ROTE/evo modules.
- Train/test split: Fixed 80:20 split (first 80% train, last 20% test), preserving temporal order.
- ROTE_evo_non_strict: `ROTE_evo_non_strict.py` provides a variant that generates full program code without parameter restrictions (allows entirely new implementations, not just parameter tuning). Uses same `--num_agents_to_sample` argument, participant-specific wandb logging, and output structure as ROTE_evo.
- Extracting best accuracies from old runs: For experiments without `participants_summary.csv`, use `python utils/extract_best_accuracies.py <run_dir>` to extract best train/test accuracies per participant from iteration metrics. Outputs CSV to `run_dir/participants_summary.csv`. Use `--include_program_ids` to see which iteration/candidate had best results.

