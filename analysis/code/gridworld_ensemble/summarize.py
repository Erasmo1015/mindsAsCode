#!/usr/bin/env python3
"""
analysis/code/gridworld_ensemble/summarize.py

Read an existing gridworld_ensemble log folder, find the best program per iteration
from wandb_metrics.jsonl, load the corresponding candidate_*.py files, and generate
very short natural-language summaries using an OpenAI-compatible local vLLM server.

Default setup matches your case:
- local vLLM server
- model: Qwen/Qwen2.5-7B-Instruct
- server URL: http://localhost:8000/v1
- only summarize best programs from the log
- default max_iteration = 5
- deduplicate repeated best_program_id entries
- save outputs into the same agent folder

Example:
python analysis/code/gridworld_ensemble/summarize.py \
  --agent_dir generated_outputs/gridworld_ensemble/non_strict/run_260306_010902/agent_0 \
  --agent_id 0

Optional:
python analysis/code/gridworld_ensemble/summarize.py \
  --agent_dir generated_outputs/gridworld_ensemble/non_strict/run_260306_010902/agent_0 \
  --agent_id 0 \
  --max_iteration 10 \
  --no_deduplicate

Output files:
- <agent_dir>/best_program_summaries.json
- <agent_dir>/best_program_summaries.csv
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize best gridworld ensemble programs from an existing log folder."
    )
    parser.add_argument(
        "--agent_dir",
        type=str,
        required=True,
        help="Path to agent folder, e.g. generated_outputs/gridworld_ensemble/non_strict/run_xxx/agent_0",
    )
    parser.add_argument(
        "--agent_id",
        type=int,
        default=0,
        help="Agent id used in wandb_metrics.jsonl keys, default: 0",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local"],
        help="Only local OpenAI-compatible vLLM mode is supported here.",
    )
    parser.add_argument(
        "--llm_server_url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible vLLM server URL, default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=DEFAULT_API_KEY,
        help=f"API key for local vLLM server, default: {DEFAULT_API_KEY}",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name served by vLLM, default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max_iteration",
        type=int,
        default=5,
        help="Only process log entries with iteration <= max_iteration. Default: 5",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Deduplicate repeated best_program_id values. Default: True",
    )
    parser.add_argument(
        "--no_deduplicate",
        action="store_true",
        help="Disable deduplication and summarize repeated best_program_id entries too.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for summarization. Default: 0.7",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="Max output tokens for summary. Default: 10",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        default="best_program_summaries.json",
        help="Output JSON filename inside agent_dir",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="best_program_summaries.csv",
        help="Output CSV filename inside agent_dir",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSONL line {line_num}: {e}")
    return rows


def extract_best_program_entries(
    rows: List[Dict[str, Any]],
    agent_id: int,
    max_iteration: Optional[int],
    deduplicate: bool,
) -> List[Dict[str, Any]]:
    best_key = f"a{agent_id}_best_program_id"
    train_key = f"a{agent_id}_train_accuracy"
    test_key = f"a{agent_id}_test_accuracy"

    entries = []
    seen_program_ids = set()

    for row in rows:
        iteration = row.get("iteration", None)
        if iteration is None:
            continue
        if iteration < 0:
            continue
        if max_iteration is not None and iteration > max_iteration:
            continue
        if best_key not in row:
            continue

        program_id = row[best_key]
        if not isinstance(program_id, str) or not program_id.strip():
            continue

        if deduplicate and program_id in seen_program_ids:
            continue

        seen_program_ids.add(program_id)

        entries.append(
            {
                "step": row.get("step"),
                "iteration": iteration,
                "program_id": program_id,
                "train_accuracy": row.get(train_key),
                "test_accuracy": row.get(test_key),
            }
        )

    return entries


def program_id_to_path(agent_dir: Path, program_id: str) -> Path:
    """
    Convert:
      iteration_4_member_7_candidate_0
    to:
      <agent_dir>/iteration_4/member_7/candidate_0.py
    """
    match = re.fullmatch(r"(iteration_\d+)_(member_\d+)_(candidate_\d+)", program_id)
    if not match:
        raise ValueError(f"Unrecognized program_id format: {program_id}")
    iteration_part, member_part, candidate_part = match.groups()
    return agent_dir / iteration_part / member_part / f"{candidate_part}.py"


def summarize_agent_code(
    client: OpenAI,
    model_name: str,
    agent_code: str,
    temperature: float = 0.7,
    max_tokens: int = 10,
) -> str:
    prompt = f"""Below is an agent code that implements a finite state machine (FSM) for a grid world environment.
Please provide a high-level summary of the agent's behavior pattern, focusing on:
1. The agent's overall goal or strategy
2. How the agent responds to different environmental features (blocks, walls)
3. Any patterns in movement or interaction

Provide a very short summary (5 words or less) with as few words as possible.
For instance:
- If the code describes an agent which moves right constantly, your summary should be "Move right".
- If they choose a random action, your summary should be "Random".
- If they alternate between moving up and down until they hit a wall, your summary should be "Up/down until wall".

Agent code:
{agent_code}

Your high-level 5 word summary:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content.strip()
    return text


def main():
    args = parse_args()

    if args.no_deduplicate:
        deduplicate = False
    else:
        deduplicate = True

    agent_dir = Path(args.agent_dir)
    if not agent_dir.exists():
        raise FileNotFoundError(f"agent_dir does not exist: {agent_dir}")

    metrics_path = agent_dir / "wandb_metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find wandb_metrics.jsonl at: {metrics_path}")

    client = OpenAI(
        api_key=args.llm_api_key,
        base_url=args.llm_server_url,
    )

    rows = load_jsonl(metrics_path)
    best_entries = extract_best_program_entries(
        rows=rows,
        agent_id=args.agent_id,
        max_iteration=args.max_iteration,
        deduplicate=deduplicate,
    )

    print(f"Loaded {len(rows)} log rows from: {metrics_path}")
    print(f"Found {len(best_entries)} best-program entries to summarize")
    if args.max_iteration is not None:
        print(f"Iteration filter: <= {args.max_iteration}")
    print(f"Deduplicate: {deduplicate}")
    print()

    results = []

    for i, entry in enumerate(best_entries, start=1):
        program_id = entry["program_id"]
        try:
            program_path = program_id_to_path(agent_dir, program_id)
        except Exception as e:
            print(f"[{i}/{len(best_entries)}] Failed to parse program_id {program_id}: {e}")
            results.append(
                {
                    **entry,
                    "program_path": None,
                    "exists": False,
                    "summary": None,
                    "error": f"program_id_parse_error: {e}",
                }
            )
            continue

        print(f"[{i}/{len(best_entries)}] Iteration {entry['iteration']} | {program_id}")
        print(f"  Path: {program_path}")

        if not program_path.exists():
            print("  Missing file")
            results.append(
                {
                    **entry,
                    "program_path": str(program_path),
                    "exists": False,
                    "summary": None,
                    "error": "file_not_found",
                }
            )
            continue

        try:
            code = program_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  Failed to read file: {e}")
            results.append(
                {
                    **entry,
                    "program_path": str(program_path),
                    "exists": True,
                    "summary": None,
                    "error": f"read_error: {e}",
                }
            )
            continue

        try:
            summary = summarize_agent_code(
                client=client,
                model_name=args.model_name,
                agent_code=code,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            print(f"  Summary: {summary}")
            results.append(
                {
                    **entry,
                    "program_path": str(program_path),
                    "exists": True,
                    "summary": summary,
                    "error": None,
                }
            )
        except Exception as e:
            print(f"  Summarization failed: {e}")
            results.append(
                {
                    **entry,
                    "program_path": str(program_path),
                    "exists": True,
                    "summary": None,
                    "error": f"summarization_error: {e}",
                }
            )

    json_path = agent_dir / args.json_name
    csv_path = agent_dir / args.csv_name

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "agent_dir": str(agent_dir),
                "agent_id": args.agent_id,
                "model_name": args.model_name,
                "llm_server_url": args.llm_server_url,
                "max_iteration": args.max_iteration,
                "deduplicate": deduplicate,
                "n_entries": len(results),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = [
        "step",
        "iteration",
        "program_id",
        "program_path",
        "exists",
        "train_accuracy",
        "test_accuracy",
        "summary",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()