"""
ROTE_evo_non_strict.py: Iterative evolution loop over executable Choice13k programs.

Non-strict version: Generates full program code without parameter restrictions.
This version allows the LLM to generate entirely new choose(problem, history) implementations,
not restricted to parameter-only changes.

The evolution process:
1. Starts with seed program from persona_code_example/vanilla.py (configurable via --seed_path)
2. Generates 10 candidate program variants per iteration (full code, not just parameters)
3. Evaluates each program on Choice13k dataset
4. Reports performance and selects best performers
5. Uses best programs as parents for next generation
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# Import data loading (this is acceptable as it's a data module, not ROTE/evo code)
from data_modules.choice13k import get_choice13k_experiments, Experiment, Block


def load_seed_program(seed_path: str) -> str:
    """Load the seed program from the specified path."""
    with open(seed_path, 'r') as f:
        return f.read()


def compile_program(code_str: str) -> Optional[Callable]:
    """Safely compile program code and return choose callable if present."""
    # Provide minimal safe builtins needed for the program to run
    # Only include what's necessary for pure Python computation
    import builtins
    safe_builtins = {
        'zip': zip,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'sum': sum,
        'abs': abs,
        'min': min,
        'max': max,
        'float': float,
        'int': int,
        'str': str,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'bool': bool,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,
    }
    global_ns = {"__builtins__": safe_builtins}
    local_ns = {}
    try:
        exec(code_str, global_ns, local_ns)
    except Exception as e:
        # For debugging: uncomment to see what went wrong
        # print(f"Compilation error: {e}")
        return None
    choose_fn = local_ns.get("choose") or global_ns.get("choose")
    if callable(choose_fn):
        return choose_fn
    return None


def format_trials_to_text(trials: List[Dict[str, Any]]) -> str:
    """Convert Choice13k trials to numbered text for prompt."""
    lines = []
    for idx, t in enumerate(trials):
        prob_a = t["problem"]["gamble_A"]["probs"]
        rew_a = t["problem"]["gamble_A"]["rewards"]
        prob_b = t["problem"]["gamble_B"]["probs"]
        rew_b = t["problem"]["gamble_B"]["rewards"]
        has_fb = t["problem"].get("has_feedback", False)
        action = t["action"]
        lines.append(
            f"{idx+1}. Problem: Option A probs {prob_a} rewards {rew_a}; "
            f"Option B probs {prob_b} rewards {rew_b}; has_feedback={has_fb}; "
            f"Observed action: {action}"
        )
    return "\n".join(lines)


def split_trials(exp: Experiment) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], list]:
    """Split trials into train/test 80/20 (fixed split, matching ROTE); return (train, test, options)."""
    options = exp.blocks[0].option_keys
    all_trials = []
    history_accum = []
    for block in exp.blocks:
        for trial in block.trials:
            history_entry = {"action": trial.action, "feedback": trial.feedback}
            all_trials.append(
                {
                    "problem": {
                        "gamble_A": {
                            "probs": block.gamble_A.probs,
                            "rewards": block.gamble_A.rewards,
                        },
                        "gamble_B": {
                            "probs": block.gamble_B.probs,
                            "rewards": block.gamble_B.rewards,
                        },
                        "option_keys": options,
                        "has_feedback": block.has_feedback,
                    },
                    "history": list(history_accum),
                    "options": options,
                    "action": trial.action,
                }
            )
            history_accum.append(history_entry)
    # Fixed 80:20 split (matching ROTE's approach in plot_and_eval.py)
    split_point = int(len(all_trials) * 0.8)
    train_trials = all_trials[:split_point]
    test_trials = all_trials[split_point:]
    return train_trials, test_trials, options


def evaluate_program(choose_fn: Callable, trials: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, float]:
    """Evaluate a program on trials and return accuracy metrics."""
    correct = 0
    total = 0
    errors = 0
    for t in trials:
        try:
            pred = choose_fn(t["problem"], t["history"])
            total += 1
            if pred is not None and pred == t["action"]:
                correct += 1
        except Exception as e:
            total += 1
            errors += 1
            if verbose and errors <= 3:  # Only print first 3 errors to avoid spam
                print(f"  Evaluation error: {e}")
    acc = correct / total if total > 0 else 0.0
    result = {"accuracy": acc, "total": total, "correct": correct, "errors": errors}
    if verbose and errors > 0:
        print(f"  Total evaluation errors: {errors}/{total}")
    return result


def generate_program_variants(
    client: OpenAI,
    model_name: str,
    parent_program: str,
    train_trials: List[Dict[str, Any]],
    n_variants: int = 10,
    max_tokens: int = 800,
) -> List[str]:
    """
    Generate full program variants based on parent program and training trials.
    
    This generates complete choose(problem, history) implementations without
    restrictions on structure or logic - only the function signature is fixed.
    """
    base_prompt = """You are given observations of human choices in risky-gamble problems.
Each problem presents two gambles: Option A and Option B. A gamble has outcomes and their probabilities (percent).
You will see a short history of previous trials for the same participant and problem, including chosen option and feedback if available.

Write Python code that reproduces the observed behavior. You must generate a program implementing:

def choose(problem, history):
    \"\"\"
    problem: dict with keys
        - gamble_A: {"probs": List[float], "rewards": List[float]}
        - gamble_B: {"probs": List[float], "rewards": List[float]}
        - option_keys: e.g., ["A","B"]
        - has_feedback: bool
    history: list of dicts with keys
        - action: int (0 for A, 1 for B)
        - feedback: float or None
    return: int, 0 for Option A or 1 for Option B
    \"\"\"

Constraints:
- Pure Python, no imports, deterministic.
- Use only the provided problem and history.
- Do not call external APIs.

Provide only the code for choose(...) as a complete function body.
"""
    
    code_template = """```python
def choose(problem, history):
    \"\"\"
    problem: dict with gamble_A/gamble_B (probs, rewards), option_keys, has_feedback
    history: list of dicts with keys action (int) and feedback (float or None)
    return: int index (0 for Option A, 1 for Option B)
    \"\"\"
    # Write your decision logic here.
    # You can use probabilities, rewards, and history.
    # Must return 0 or 1.
    return 0
```
"""
    
    # Format training trials for context
    state_text = format_trials_to_text(train_trials)
    
    # Include parent program as reference
    parent_context = f"\n\nReference program (parent):\n```python\n{parent_program}\n```\n\n"
    parent_context += "Generate a variant that improves upon or explores alternatives to the parent program.\n"
    
    prompt_text = f"{base_prompt}\n{state_text}\n{parent_context}\n{code_template}"
    
    programs = []
    for _ in tqdm(range(n_variants), desc="Generating candidate programs"):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            match = re.search(r"```python(.*?)```", content, re.DOTALL | re.IGNORECASE)
            code = match.group(1).strip() if match else content.strip()
            programs.append(code)
        except Exception as e:
            print(f"Warning: Failed to generate program variant: {e}")
            programs.append("")
    return programs


def run_evolution(
    seed_program_path: str,
    participant_id: int = 0,
    n_iterations: int = 5,
    n_candidates_per_iteration: int = 10,
    model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    client_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    wandb=None,
):
    """
    Run iterative evolution loop over Choice13k programs.
    
    Args:
        seed_program_path: Path to seed program (default: vanilla.py)
        participant_id: Which participant's data to use (0-indexed)
        n_iterations: Number of evolution iterations
        n_candidates_per_iteration: Number of candidate programs per iteration
        model_name: LLM model name for generation
        client_kwargs: Optional OpenAI client kwargs (for local vLLM server)
        output_dir: Optional output directory for saving results
    """
    # Initialize client
    if client_kwargs is None:
        client_kwargs = {}
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    
    # Load seed program
    print(f"Loading seed program from {seed_program_path}...")
    seed_code = load_seed_program(seed_program_path)
    parent_program = seed_code
    
    # Load Choice13k data
    print(f"Loading Choice13k data for participant {participant_id}...")
    experiments = get_choice13k_experiments(n_participants=participant_id + 1)
    exp = experiments[participant_id]
    
    # Split trials
    train_trials, test_trials, options = split_trials(exp)
    print(f"Split trials: {len(train_trials)} train, {len(test_trials)} test")
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        output_dir = f"generated_outputs/choice13k_ROTE_evo_non_strict/run_{timestamp}/participant_{participant_id}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ===== BASELINE EVALUATION =====
    print(f"\n{'='*80}")
    print(f"BASELINE EVALUATION: Evaluating seed program ({seed_program_path})")
    print(f"{'='*80}")
    
    baseline_fn = compile_program(seed_code)
    
    if baseline_fn is None:
        print("ERROR: Failed to compile baseline program!")
        return None
    
    baseline_train_eval = evaluate_program(baseline_fn, train_trials, verbose=True)
    baseline_test_eval = evaluate_program(baseline_fn, test_trials, verbose=True)
    
    print(f"\nBaseline Performance:")
    print(f"  Train accuracy: {baseline_train_eval['accuracy']:.4f} ({baseline_train_eval['correct']}/{baseline_train_eval['total']})")
    print(f"  Test accuracy: {baseline_test_eval['accuracy']:.4f} ({baseline_test_eval['correct']}/{baseline_test_eval['total']})")
    
    # Store baseline results (will be included in final results.json)
    baseline_results = {
        "train_accuracy": baseline_train_eval['accuracy'],
        "test_accuracy": baseline_test_eval['accuracy'],
        "train_correct": baseline_train_eval['correct'],
        "train_total": baseline_train_eval['total'],
        "test_correct": baseline_test_eval['correct'],
        "test_total": baseline_test_eval['total'],
    }
    
    # Track all candidate results across iterations for finding overall best
    all_candidate_results = []  # List of dicts with iteration, candidate_idx, train_acc, test_acc
    
    # Log baseline to wandb at step=0
    if wandb is not None:
        wandb.log({
            f"p{participant_id}_train_accuracy": baseline_train_eval["accuracy"],
            f"p{participant_id}_test_accuracy": baseline_test_eval["accuracy"],
            f"p{participant_id}_is_baseline": 1,
        }, step=0)
    
    # Initialize best program tracking with baseline
    best_fitness = baseline_train_eval['accuracy']
    
    # Track overall best across all iterations
    overall_best_train = {
        "train_accuracy": baseline_train_eval['accuracy'],
        "test_accuracy": baseline_test_eval['accuracy'],
        "program_id": "baseline"
    }
    overall_best_test = {
        "train_accuracy": baseline_train_eval['accuracy'],
        "test_accuracy": baseline_test_eval['accuracy'],
        "program_id": "baseline"
    }
    
    # Evolution loop
    for iteration in range(n_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*80}")
        
        iter_dir = output_path / f"iteration_{iteration}"
        iter_dir.mkdir(exist_ok=True)
        candidates_dir = iter_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)
        
        # Generate candidate programs (full code, not just parameters)
        print(f"\nGenerating {n_candidates_per_iteration} candidate programs...")
        candidate_codes = generate_program_variants(
            client=client,
            model_name=model_name,
            parent_program=parent_program,
            train_trials=train_trials,
            n_variants=n_candidates_per_iteration,
        )
        
        # Evaluate candidates
        print(f"\nEvaluating candidates...")
        candidate_results = []
        for idx, code in enumerate(tqdm(candidate_codes, desc="Evaluating")):
            # Save candidate code
            (candidates_dir / f"candidate_{idx}.py").write_text(code or "")
            
            if not code:
                candidate_results.append({
                    "idx": idx,
                    "code": "",
                    "train_acc": 0.0,
                    "test_acc": 0.0,
                    "valid": False,
                })
                continue
            
            # Compile program
            choose_fn = compile_program(code)
            if choose_fn is None:
                candidate_results.append({
                    "idx": idx,
                    "code": code,
                    "train_acc": 0.0,
                    "test_acc": 0.0,
                    "valid": False,
                })
                continue
            
            # Evaluate on train and test
            train_eval = evaluate_program(choose_fn, train_trials)
            test_eval = evaluate_program(choose_fn, test_trials)
            
            train_acc = train_eval["accuracy"]
            test_acc = test_eval["accuracy"]
            
            candidate_results.append({
                "idx": idx,
                "code": code,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_correct": train_eval["correct"],
                "test_correct": test_eval["correct"],
                "train_total": train_eval["total"],
                "test_total": test_eval["total"],
                "valid": True,
            })
            
            # Track for overall best (only valid candidates)
            all_candidate_results.append({
                "iteration": iteration,
                "candidate_idx": idx,
                "train_acc": train_acc,
                "test_acc": test_acc,
            })
        
        # Report results
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1} Results:")
        print(f"{'='*80}")
        
        valid_results = [r for r in candidate_results if r["valid"]]
        if valid_results:
            # Sort by train accuracy
            valid_results.sort(key=lambda x: x["train_acc"], reverse=True)
            
            print(f"\nTop performers (by train accuracy):")
            for i, result in enumerate(valid_results[:5]):
                print(
                    f"  {i+1}. Candidate {result['idx']}: "
                    f"train_acc={result['train_acc']:.4f}, "
                    f"test_acc={result['test_acc']:.4f}"
                )
            
            # Select best program as parent for next iteration
            # Use top performer as parent
            best_result = valid_results[0]
            parent_program = best_result["code"]
            best_fitness = best_result["train_acc"]
            
            print(f"\nBest program selected: Candidate {best_result['idx']}")
            print(f"  Train accuracy: {best_result['train_acc']:.4f}")
            print(f"  Test accuracy: {best_result['test_acc']:.4f}")
            
            # Update overall best tracking
            if best_result['train_acc'] > overall_best_train["train_accuracy"]:
                overall_best_train = {
                    "train_accuracy": best_result['train_acc'],
                    "test_accuracy": best_result['test_acc'],
                    "program_id": f"iteration_{iteration}_candidate_{best_result['idx']}"
                }
            if best_result['test_acc'] > overall_best_test["test_accuracy"]:
                overall_best_test = {
                    "train_accuracy": best_result['train_acc'],
                    "test_accuracy": best_result['test_acc'],
                    "program_id": f"iteration_{iteration}_candidate_{best_result['idx']}"
                }
        else:
            print("\nWarning: No valid programs generated in this iteration!")
            print("Continuing with previous parent program...")
        
        # Save iteration results
        best_program_id = None
        if valid_results:
            best_program_id = f"iteration_{iteration}_candidate_{valid_results[0]['idx']}"
        
        metrics = {
            "iteration": iteration,
            "n_candidates": n_candidates_per_iteration,
            "n_valid": len(valid_results),
            "best_program_id": best_program_id,
            "candidate_results": [
                {
                    "idx": r["idx"],
                    "train_acc": r["train_acc"],
                    "test_acc": r["test_acc"],
                    "valid": r["valid"],
                }
                for r in candidate_results
            ],
            "best_train_acc": best_fitness if valid_results else None,
            "best_test_acc": valid_results[0]["test_acc"] if valid_results else None,
        }
        (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        
        # Save summary
        summary = {
            "iteration": iteration,
            "best_train_acc": best_fitness if valid_results else None,
            "best_test_acc": valid_results[0]["test_acc"] if valid_results else None,
            "n_valid": len(valid_results),
        }
        print(f"\nSummary: {json.dumps(summary, indent=2)}")
        
        # Log to wandb (use participant-specific metric names)
        if wandb is not None:
            log_dict = {
                f"p{participant_id}_n_valid": len(valid_results),
            }
            if valid_results:
                # Use participant-specific metric names (e.g., p0_train_accuracy, p1_train_accuracy)
                log_dict[f"p{participant_id}_train_accuracy"] = best_fitness  # Best train accuracy
                log_dict[f"p{participant_id}_test_accuracy"] = valid_results[0]["test_acc"]  # Best test accuracy
                log_dict[f"p{participant_id}_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                log_dict[f"p{participant_id}_avg_test_accuracy"] = np.mean([r["test_acc"] for r in valid_results])
            wandb.log(log_dict, step=iteration + 1)  # Step starts at 1 (baseline is step=0)
    
    # Final summary and save comprehensive results.json
    print(f"\n{'='*80}")
    print("Evolution Complete")
    print(f"{'='*80}")
    
    # Check all candidates to find true overall best (in case best wasn't selected as parent)
    for candidate in all_candidate_results:
        if candidate['train_acc'] > overall_best_train["train_accuracy"]:
            overall_best_train = {
                "train_accuracy": candidate['train_acc'],
                "test_accuracy": candidate['test_acc'],
                "program_id": f"iteration_{candidate['iteration']}_candidate_{candidate['candidate_idx']}"
            }
        if candidate['test_acc'] > overall_best_test["test_accuracy"]:
            overall_best_test = {
                "train_accuracy": candidate['train_acc'],
                "test_accuracy": candidate['test_acc'],
                "program_id": f"iteration_{candidate['iteration']}_candidate_{candidate['candidate_idx']}"
            }
    
    # Create comprehensive results.json
    results = {
        "baseline": baseline_results,
        "overall_best_train_accuracy": overall_best_train,
        "overall_best_test_accuracy": overall_best_test,
    }
    (output_path / "results.json").write_text(json.dumps(results, indent=2))
    
    if n_iterations > 0:
        print(f"Final best train accuracy: {overall_best_train['train_accuracy']:.4f} (from {overall_best_train['program_id']})")
        print(f"Final best test accuracy: {overall_best_test['test_accuracy']:.4f} (from {overall_best_test['program_id']})")
        print(f"Baseline train accuracy: {baseline_train_eval['accuracy']:.4f}")
        print(f"Train accuracy improvement: {overall_best_train['train_accuracy'] - baseline_train_eval['accuracy']:.4f}")
        print(f"Test accuracy improvement: {overall_best_test['test_accuracy'] - baseline_test_eval['accuracy']:.4f}")
    print(f"\nResults saved to: {output_path / 'results.json'}")
    
    # Return summary for participants summary file (just the essentials)
    return {
        "participant_id": participant_id,
        "train_acc": overall_best_train['train_accuracy'],
        "test_acc": overall_best_test['test_accuracy'],
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ROTE Evolution (Non-Strict): Iterative evolution of Choice13k programs")
    parser.add_argument(
        "--seed_path",
        type=str,
        default="persona_code_example/vanilla.py",
        help="Path to seed program (starting persona)",
    )
    parser.add_argument(
        "--participant_id",
        type=int,
        default=None,
        help="Specific participant ID to evaluate (0-indexed). If None, evaluates all participants from 0 to num_agents_to_sample-1.",
    )
    parser.add_argument(
        "--num_agents_to_sample",
        type=int,
        default=1,
        help="Number of participants to process (0-indexed, from 0 to num_agents_to_sample-1). Used when participant_id is None.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=5,
        help="Number of evolution iterations",
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=10,
        help="Number of candidate programs per iteration",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        help="LLM model name for generation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "local"],
        help="LLM mode: default uses OpenAI API; local routes to vLLM server",
    )
    parser.add_argument(
        "--llm_server_url",
        type=str,
        default=os.getenv("VLLM_LOCAL_URL", "http://localhost:8000/v1"),
        help="Base URL for local vLLM server when --mode local",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=os.getenv("VLLM_LOCAL_API_KEY", "EMPTY"),
        help="API key for local vLLM server when --mode local",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable wandb logging. Default is enabled.",
    )
    
    args = parser.parse_args()
    
    # Optional wandb setup
    wandb_enabled = False
    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = f"non_strict_{datetime.now():%y%m%d_%H%M%S}"
            if args.participant_id is not None:
                run_name = f"{run_name}_participant_{args.participant_id}"
            else:
                run_name = f"{run_name}_participants_0to{args.num_agents_to_sample-1}"
            wandb.init(
                project="ROTE_evo",
                name=run_name,
                config=vars(args),
                reinit=False,
            )
            wandb_enabled = True
        except Exception as e:
            print(f"wandb logging disabled: {e}")
            wandb_enabled = False
    
    # Setup client kwargs
    client_kwargs = {}
    if args.mode == "local":
        client_kwargs = {
            "api_key": args.llm_api_key,
            "base_url": args.llm_server_url,
        }
    
    # Determine which participants to process
    if args.participant_id is not None:
        # Single participant mode (backward compatible)
        participants_to_process = [args.participant_id]
    else:
        # Multiple participants mode
        participants_to_process = list(range(args.num_agents_to_sample))
    
    # Create base run directory and save seed program once
    base_run_dir = None
    if args.output_dir is None:
        # Auto-generated output: create base run directory
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        base_run_dir = f"generated_outputs/choice13k_ROTE_evo_non_strict/run_{timestamp}"
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)
    elif len(participants_to_process) > 1:
        # Multiple participants with custom output_dir: use that as base directory
        base_run_dir = args.output_dir
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Single participant with custom output_dir: use parent directory if it looks like a participant dir
        # Otherwise use the directory itself
        output_path = Path(args.output_dir)
        if output_path.name.startswith("participant_"):
            # It's a participant directory, use parent as base
            base_run_dir = str(output_path.parent)
        else:
            # It's already a base directory
            base_run_dir = args.output_dir
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and save seed program once in the experiment folder
    seed_code = load_seed_program(args.seed_path)
    (Path(base_run_dir) / "seed_program.py").write_text(seed_code)
    print(f"Seed program saved to: {Path(base_run_dir) / 'seed_program.py'}")
    
    # Initialize participants summary (list for CSV)
    participants_summary = []
    # Determine summary file location (use base_run_dir if available, otherwise use output_dir or its parent)
    if base_run_dir is not None:
        summary_file = Path(base_run_dir) / "participants_summary.csv"
    elif args.output_dir is not None:
        output_path = Path(args.output_dir)
        if output_path.name.startswith("participant_"):
            # It's a participant directory, use parent
            summary_file = output_path.parent / "participants_summary.csv"
        else:
            # Use the directory itself
            summary_file = output_path / "participants_summary.csv"
    else:
        # Auto-generated single participant - will be determined after first run
        summary_file = None
    
    # Run evolution for each participant
    try:
        for participant_id in tqdm(participants_to_process, desc="Participants"):
            print(f"\n{'='*80}")
            print(f"Processing participant {participant_id}")
            print(f"{'='*80}")
            # If base_run_dir is set, construct participant-specific output_dir
            if base_run_dir is not None:
                participant_output_dir = os.path.join(base_run_dir, f"participant_{participant_id}")
            else:
                participant_output_dir = args.output_dir
            
            # If summary_file is None (auto-generated single participant), determine it now
            if summary_file is None and participant_output_dir is not None:
                output_path = Path(participant_output_dir)
                if output_path.name.startswith("participant_"):
                    summary_file = output_path.parent / "participants_summary.csv"
                else:
                    summary_file = output_path / "participants_summary.csv"
            
            participant_summary = run_evolution(
                seed_program_path=args.seed_path,
                participant_id=participant_id,
                n_iterations=args.n_iterations,
                n_candidates_per_iteration=args.n_candidates,
                model_name=args.model_name,
                client_kwargs=client_kwargs if client_kwargs else None,
                output_dir=participant_output_dir,
                wandb=wandb,
            )
            
            # Update participants summary after each participant completes
            if participant_summary is not None and summary_file is not None:
                participants_summary.append(participant_summary)
                # Write CSV file
                with open(summary_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['participant_id', 'train_acc', 'test_acc'])
                    writer.writeheader()
                    writer.writerows(participants_summary)
                print(f"\nParticipants summary updated: {summary_file}")
    finally:
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
