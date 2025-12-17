Local vLLM mode

- Start your local server (example): `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`.
- Run with the new mode switch to route LLM calls externally:  
  `python plot_and_eval.py --baseline_model ROTE --mode local --model_name Qwen/Qwen2.5-7B-Instruct --llm_server_url http://localhost:8000/v1 --llm_api_key EMPTY`
- When `--mode local` is set, no models are loaded in-process; all prompts go to the OpenAI-compatible vLLM server. Default mode preserves the original in-process behavior.

Wandb logging

- Enabled by default; disable with `--no_log True`.
- Logs the same metrics written to CSV (accuracy, first_step_accuracy, accuracy_after_flip, timing, program_length, num_hypothesis, epoch, model/llm_model, mode).

