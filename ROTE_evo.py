"""
ROTE_evo.py: Iterative evolution loop over executable Choice13k programs.

This module implements an evolutionary approach to improving Choice13k decision-making
programs, starting from a seed program and iteratively generating and evaluating variants.

The evolution process:
1. Starts with seed program from persona_code_example/vanilla.py (configurable via --seed_path)
2. Generates 10 candidate program variants per iteration
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


def extract_parameters_from_template(template_code: str) -> Dict[str, float]:
    """Extract parameter values from the template code.
    
    Only extracts parameters that appear in the "# Parameters" section
    (the first few lines after the function definition).
    """
    params = {}
    lines = template_code.split('\n')
    in_parameters_section = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if we're entering the parameters section
        if 'parameters' in stripped.lower() and stripped.startswith('#'):
            in_parameters_section = True
            continue
        
        # Extract parameters until we hit another comment section or non-assignment code
        if in_parameters_section:
            # Stop if we hit another comment section (but not the Parameters comment itself)
            # Ignore separator lines (lines with only dashes, equals, or whitespace)
            if stripped.startswith('#'):
                # Check if it's a separator line (only dashes/equals/whitespace after #)
                separator_chars = set(stripped[1:].strip())  # Characters after #
                if separator_chars.issubset({'-', '=', ' '}) or len(separator_chars) == 0:
                    # It's a separator line, skip it and continue
                    continue
                elif 'parameters' not in stripped.lower():
                    # Hit another comment section, stop extracting
                    break
            
            # Extract parameter assignments
            if '=' in stripped and not stripped.startswith('#'):
                # Look for parameter assignments like "alpha = 0.85"
                parts = stripped.split('=')
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    try:
                        param_value = float(parts[1].strip())
                        params[param_name] = param_value
                    except ValueError:
                        pass
            # Stop if we hit code that's not a comment or assignment (like "if", "def", etc.)
            elif stripped and not stripped.startswith('#') and '=' not in stripped:
                # Check if this looks like control flow or function definition
                if any(stripped.startswith(keyword) for keyword in ['if', 'def', 'for', 'while', 'return', 'class']):
                    if params:  # If we've already found parameters, stop
                        break
    
    return params


def inject_parameters_into_template(template_code: str, params: Dict[str, float]) -> str:
    """Inject parameter values into the template code, replacing existing values."""
    lines = template_code.split('\n')
    result_lines = []
    param_names = set(params.keys())
    
    for line in lines:
        original_line = line
        stripped = line.strip()
        
        # Check if this line assigns a parameter we want to replace
        if '=' in stripped and not stripped.startswith('#'):
            parts = stripped.split('=')
            if len(parts) == 2:
                param_name = parts[0].strip()
                if param_name in param_names:
                    # Replace the parameter value
                    indent = len(line) - len(line.lstrip())
                    new_line = ' ' * indent + f"{param_name} = {params[param_name]}"
                    result_lines.append(new_line)
                    continue
        
        result_lines.append(original_line)
    
    return '\n'.join(result_lines)


def compile_program(code_str: str) -> Optional[Callable]:
    """Safely compile program code and return choose callable if present."""
    # Provide minimal safe builtins needed for the program to run
    # Only include what's necessary for pure Python computation
    import builtins
    import math
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
        '__import__': __import__,  # Needed for dynamic imports like __import__("math")
    }
    global_ns = {
        "__builtins__": safe_builtins,
        "math": math,  # Pre-import math module for convenience
    }
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


def generate_parameter_variants(
    client: OpenAI,
    model_name: str,
    template_code: str,
    parent_params: Dict[str, float],
    train_trials: List[Dict[str, Any]],
    n_variants: int = 10,
    max_tokens: int = 200,
    exploration_factor: float = 1.0,
    parent_train_accuracy: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Generate parameter value variants only. The program structure stays fixed.
    
    Args:
        train_trials: Training dataset - list of trials containing problems, history, and observed actions
        parent_train_accuracy: Training accuracy of parent parameters (used to guide exploration)
    
    Returns a list of parameter dictionaries, each containing the parameter values.
    """
    # Extract parameter names from template
    param_names = sorted(parent_params.keys())
    num_params = len(param_names)
    
    # Build parameter info for prompt
    param_info = ""
    for param_name in param_names:
        param_info += f"- {param_name}: currently {parent_params[param_name]}\n"
    
    # Build performance feedback info for prompt (only train accuracy, NOT test)
    performance_info = ""
    if parent_train_accuracy is not None:
        performance_info += f"\nCurrent parent performance on training data:\n"
        performance_info += f"- Train accuracy: {parent_train_accuracy:.4f}\n"
        # Add guidance based on performance
        if parent_train_accuracy < 0.5:
            performance_info += f"\nNOTE: Current performance is LOW. Consider trying different parameter combinations.\n"
        elif parent_train_accuracy > 0.8:
            performance_info += f"\nNOTE: Current performance is HIGH. Make refined adjustments to improve further.\n"
        else:
            performance_info += f"\nNOTE: Current performance is MODERATE. Explore various parameter combinations.\n"
    
    # Determine exploration ranges based on exploration_factor
    # For high exploration (factor > 0.7): very wide ranges
    # For medium (0.3-0.7): moderate ranges  
    # For low (< 0.3): narrow ranges around current values
    if exploration_factor >= 0.7:
        exploration_mode = "EXTREMELY AGGRESSIVE"
        range_desc = "Explore EXTREMELY WIDE ranges - be BOLD and DRAMATIC!"
        param_ranges = {
            "alpha": (0.05, 0.995),
            "lambda_loss": (0.1, 10.0),
            "gamma": (0.1, 0.99),
            "kappa": (0.01, 0.5),
            "delta": (0.1, 1.0),
            "phi": (0.01, 0.5),
        }
        boldness_instruction = "Make DRAMATIC changes to ALL parameters! Don't keep any parameter close to its current value. Try extreme combinations!"
    elif exploration_factor >= 0.4:
        exploration_mode = "VERY AGGRESSIVE"
        range_desc = "Explore WIDE ranges - be BOLD!"
        param_ranges = {
            "alpha": (0.2, 0.99),
            "lambda_loss": (0.5, 8.0),
            "gamma": (0.3, 0.9),
            "kappa": (0.05, 0.4),
            "delta": (0.3, 0.8),
            "phi": (0.1, 0.4),
        }
        boldness_instruction = "Make SIGNIFICANT changes to ALL parameters! Explore different regions of parameter space."
    elif exploration_factor >= 0.2:
        exploration_mode = "MODERATELY AGGRESSIVE"
        range_desc = "Explore MODERATE ranges"
        param_ranges = {
            "alpha": (0.5, 0.95),
            "lambda_loss": (1.0, 6.0),
            "gamma": (0.4, 0.8),
            "kappa": (0.1, 0.3),
            "delta": (0.4, 0.6),
            "phi": (0.15, 0.25),
        }
        boldness_instruction = "Make noticeable changes to ALL parameters, but stay within reasonable bounds."
    else:
        exploration_mode = "CONSERVATIVE"
        range_desc = "Make REFINED adjustments"
        param_ranges = {
            "alpha": (0.7, 0.95),
            "lambda_loss": (1.5, 4.0),
            "gamma": (0.5, 0.75),
            "kappa": (0.15, 0.25),
            "delta": (0.45, 0.55),
            "phi": (0.18, 0.22),
        }
        boldness_instruction = "Make fine-tuned adjustments to ALL parameters for better performance."
    
    # Build range instruction string
    range_instruction = f"\nParameter ranges to explore ({exploration_mode}):\n"
    for param_name in param_names:
        if param_name in param_ranges:
            min_val, max_val = param_ranges[param_name]
            range_instruction += f"- {param_name}: {min_val} to {max_val}\n"
        else:
            # Fallback for unknown parameters
            range_instruction += f"- {param_name}: explore widely\n"
    
    # Create diverse exploration strategies that emphasize ALL parameters
    exploration_strategies = [
        f"ALL HIGH: Set ALL {num_params} parameters to HIGH values (aggressive risk-taking)",
        f"ALL LOW: Set ALL {num_params} parameters to LOW values (conservative)",
        f"EXTREME MIX: Vary ALL {num_params} parameters dramatically - some very high, some very low",
        f"OPPOSITE POLES: Push ALL {num_params} parameters to opposite extremes from current values",
        f"BOUNDARY EXPLORATION: Set ALL {num_params} parameters near their range boundaries",
        f"RANDOM WALK: Change ALL {num_params} parameters randomly across full ranges",
        f"BALANCED SHIFT: Modify ALL {num_params} parameters in a coordinated way",
        f"CHAOTIC MIX: Combine very different values for ALL {num_params} parameters",
        f"SYMMETRIC EXTREMES: Mirror high/low patterns across ALL {num_params} parameters",
        f"UNIFORM EXPLORATION: Explore ALL {num_params} parameters uniformly across their ranges",
    ]
    
    base_prompt_template = """You are optimizing a decision-making model for risky-gamble problems.
The model structure is FIXED - you can only change parameter values.

Current parameters:
{param_info}
{performance_info}

Training observations (first 10):
{training_sample}

EXPLORATION MODE: {exploration_mode} (factor: {exploration_factor:.3f})
{range_desc}

CRITICAL REQUIREMENT: You MUST change ALL {num_params} parameters! Do NOT keep any parameter unchanged or close to its current value!

Your task: Generate {exploration_mode} parameter values (as JSON) to explore the optimization space.

MANDATORY INSTRUCTIONS:
{boldness_instruction}
- You MUST modify ALL {num_params} parameters: {param_list}
- Each parameter must be changed from its current value
- Try DIFFERENT combinations: high/low mixes, extreme values, unexpected ranges
- Each variant should explore a DIFFERENT region of parameter space
{range_instruction}
- Don't be conservative - the goal is to discover diverse behavioral patterns
- If exploration_factor is high (>0.7), make DRAMATIC changes to ALL parameters
- If exploration_factor is low (<0.3), make refined adjustments to ALL parameters

Exploration strategy for this variant: {strategy}

Output format (JSON only, no code blocks, no markdown):
{output_format}

REMEMBER: Change ALL {num_params} parameters! Do not leave any parameter unchanged!
"""
    
    # Sample training trials for prompt (don't overwhelm with all trials)
    training_sample = format_trials_to_text(train_trials[:min(10, len(train_trials))])
    
    # Build output format string with all parameters
    output_format_dict = {name: "X.XX" for name in param_names}
    output_format = json.dumps(output_format_dict, indent=2)
    param_list = ", ".join(param_names)
    
    param_variants = []
    for variant_idx in range(n_variants):
        try:
            # Select exploration strategy (cycle through strategies)
            strategy = exploration_strategies[variant_idx % len(exploration_strategies)]
            
            # Build prompt with strategy and exploration details
            variant_prompt = base_prompt_template.format(
                param_info=param_info,
                performance_info=performance_info,
                training_sample=training_sample,
                strategy=strategy,
                exploration_mode=exploration_mode,
                exploration_factor=exploration_factor,
                range_desc=range_desc,
                boldness_instruction=boldness_instruction,
                range_instruction=range_instruction,
                num_params=num_params,
                param_list=param_list,
                output_format=output_format,
            )
            
            # Adjust LLM temperature based on exploration factor
            # Higher exploration_factor -> higher temperature for more diversity
            # Range: 0.8 (conservative) to 1.5 (very aggressive)
            llm_temperature = 0.8 + (exploration_factor * 0.7)
            
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": variant_prompt}],
                temperature=llm_temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            # First, try to find JSON object (handles multi-line JSON)
            json_str = None
            
            # Try to find JSON object with balanced braces
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        json_str = content[start_idx:i+1]
                        break
            
            # If that didn't work, try regex as fallback
            if json_str is None:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
            
            if json_str:
                try:
                    params = json.loads(json_str)
                    # Validate that all required parameters are present
                    if all(name in params for name in param_names):
                        # Ensure all values are floats and filter to only param_names
                        params = {k: float(v) for k, v in params.items() if k in param_names}
                        # Ensure all parameters are positive (clamp negative values to small positive)
                        for name in param_names:
                            if name not in params:
                                params[name] = parent_params[name]
                            elif params[name] <= 0:
                                # If negative, use a small positive value instead
                                params[name] = 0.01
                        param_variants.append(params)
                    else:
                        # Missing some parameters, fill in from parent
                        filled_params = parent_params.copy()
                        for k, v in params.items():
                            if k in param_names:
                                filled_params[k] = float(v)
                        param_variants.append(filled_params)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    # If JSON parsing fails, try to extract individual parameter values
                    # Look for patterns like "alpha": 0.85 or alpha: 0.85
                    extracted_params = parent_params.copy()
                    for param_name in param_names:
                        # Try to find param_name: value or "param_name": value
                        pattern = rf'["\']?{re.escape(param_name)}["\']?\s*[:=]\s*([0-9]+\.?[0-9]*)'
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            try:
                                val = float(match.group(1))
                                extracted_params[param_name] = max(0.01, val) if val <= 0 else val  # Ensure positive
                            except (ValueError, IndexError):
                                pass
                    param_variants.append(extracted_params)
            else:
                # No JSON found, try to extract individual parameter values
                extracted_params = parent_params.copy()
                for param_name in param_names:
                    pattern = rf'["\']?{re.escape(param_name)}["\']?\s*[:=]\s*([0-9]+\.?[0-9]*)'
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        try:
                            val = float(match.group(1))
                            extracted_params[param_name] = max(0.01, val) if val <= 0 else val  # Ensure positive
                        except (ValueError, IndexError):
                            pass
                param_variants.append(extracted_params)
        except Exception as e:
            print(f"Warning: Failed to generate parameter variant: {e}")
            # Fallback to parent parameters
            param_variants.append(parent_params.copy())
    
    return param_variants


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
    
    # Load seed program template
    print(f"Loading seed program template from {seed_program_path}...")
    template_code = load_seed_program(seed_program_path)
    
    # Extract baseline parameters from template
    baseline_params = extract_parameters_from_template(template_code)
    
    # Fallback: if extraction failed, try to extract manually from common parameter names
    if not baseline_params:
        print("Warning: Parameter extraction returned empty dict. Trying fallback extraction...")
        import re
        # Look for common parameter patterns: param_name = value
        param_pattern = r'(\w+)\s*=\s*([0-9]+\.?[0-9]*)'
        matches = re.findall(param_pattern, template_code)
        for param_name, param_value in matches:
            # Only take parameters that appear before the first function definition or control flow
            param_pos = template_code.find(f"{param_name} = {param_value}")
            first_def_pos = template_code.find("def ", template_code.find("def choose"))
            if param_pos < first_def_pos or first_def_pos == -1:
                try:
                    baseline_params[param_name] = float(param_value)
                except ValueError:
                    pass
    
    if not baseline_params:
        print("ERROR: Could not extract any parameters from template!")
        print("Template preview (first 10 lines):")
        print("\n".join(template_code.split("\n")[:10]))
        return
    
    print(f"Baseline parameters: {baseline_params}")
    
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
        output_dir = f"generated_outputs/choice13k_ROTE_evo/run_{timestamp}/participant_{participant_id}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save baseline parameters (only JSON, no .py files)
    (output_path / "baseline_parameters.json").write_text(json.dumps(baseline_params, indent=2))
    
    # ===== BASELINE EVALUATION =====
    print(f"\n{'='*80}")
    print(f"BASELINE EVALUATION: Evaluating seed program ({seed_program_path})")
    print(f"{'='*80}")
    
    baseline_code = inject_parameters_into_template(template_code, baseline_params)
    baseline_fn = compile_program(baseline_code)
    
    if baseline_fn is None:
        print("ERROR: Failed to compile baseline program!")
        return None
    
    baseline_train_eval = evaluate_program(baseline_fn, train_trials, verbose=True)
    baseline_test_eval = evaluate_program(baseline_fn, test_trials, verbose=True)
    
    print(f"\nBaseline Performance:")
    print(f"  Train accuracy: {baseline_train_eval['accuracy']:.4f} ({baseline_train_eval['correct']}/{baseline_train_eval['total']})")
    print(f"  Test accuracy: {baseline_test_eval['accuracy']:.4f} ({baseline_test_eval['correct']}/{baseline_test_eval['total']})")
    print(f"  Parameters: {baseline_params}")
    
    # Store baseline results (will be included in final results.json)
    baseline_results = {
        "parameters": baseline_params,
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
    parent_params = baseline_params.copy()
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
    
    # Evolution loop with cooling schedule
    for iteration in range(n_iterations):
        iter_dir = output_path / f"iteration_{iteration}"
        iter_dir.mkdir(exist_ok=True)
        candidates_dir = iter_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)
        
        # Calculate exploration factor using exponential cooling schedule
        # For 100 iterations: starts at 1.0, cools to 0.1
        # Using exponential decay: exploration = start * (end/start)^(progress)
        # This gives aggressive exploration early, gentle refinement later
        start_exploration = 1.0
        end_exploration = 0.1
        if n_iterations > 1:
            # Progress from 0.0 to 1.0
            progress = iteration / (n_iterations - 1)
            # Exponential cooling: more aggressive decay early, slower later
            exploration_factor = start_exploration * ((end_exploration / start_exploration) ** progress)
        else:
            exploration_factor = start_exploration
        
        # Log exploration factor for monitoring
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{n_iterations} - Exploration factor: {exploration_factor:.3f}")
        if exploration_factor >= 0.7:
            print(f"Mode: EXTREMELY AGGRESSIVE - exploring wide parameter ranges")
        elif exploration_factor >= 0.4:
            print(f"Mode: VERY AGGRESSIVE - exploring wide parameter ranges")
        elif exploration_factor >= 0.2:
            print(f"Mode: MODERATELY AGGRESSIVE - exploring moderate ranges")
        else:
            print(f"Mode: CONSERVATIVE - fine-tuning parameters")
        print(f"{'='*80}")
        
        # Generate candidate parameter sets
        # Pass parent train accuracy (NOT test accuracy) to guide LLM exploration
        candidate_param_sets = generate_parameter_variants(
            client=client,
            model_name=model_name,
            template_code=template_code,
            parent_params=parent_params,
            train_trials=train_trials,
            n_variants=n_candidates_per_iteration,
            exploration_factor=exploration_factor,
            parent_train_accuracy=best_fitness,
        )
        
        # Evaluate candidates
        candidate_results = []
        for idx, params in enumerate(candidate_param_sets):
            # Save candidate parameters as JSON (only JSON, no .py files)
            (candidates_dir / f"candidate_{idx}_params.json").write_text(json.dumps(params, indent=2))
            
            # Generate full program code from template with these parameters (for evaluation only, not saved)
            candidate_code = inject_parameters_into_template(template_code, params)
            
            # Compile program
            choose_fn = compile_program(candidate_code)
            if choose_fn is None:
                candidate_results.append({
                    "idx": idx,
                    "parameters": params,
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
                "parameters": params,
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
        
        # Process results
        valid_results = [r for r in candidate_results if r["valid"]]
        if valid_results:
            # Sort by train accuracy
            valid_results.sort(key=lambda x: x["train_acc"], reverse=True)
            
            # Select best parameter set as parent for next iteration
            best_result = valid_results[0]
            parent_params = best_result["parameters"].copy()
            best_fitness = best_result["train_acc"]
            
            # Save best parameters (only JSON, no .py files)
            (iter_dir / "best_parameters.json").write_text(json.dumps(parent_params, indent=2))
            
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
            
            # Print compressed one-line summary
            print(f"Iter {iteration + 1}: best_candidate={best_result['idx']}, train_acc={best_fitness:.4f}, test_acc={best_result['test_acc']:.4f}")
        else:
            print(f"Iter {iteration + 1}: WARNING - No valid programs generated, continuing with previous parent")
        
        # Save iteration results (convert parameters to JSON-serializable format)
        best_program_id = None
        if valid_results:
            best_program_id = f"iteration_{iteration}_candidate_{valid_results[0]['idx']}"
        
        metrics = {
            "iteration": iteration,
            "n_candidates": n_candidates_per_iteration,
            "n_valid": len(valid_results),
            "exploration_factor": exploration_factor,
            "best_program_id": best_program_id,
            "candidate_results": [
                {
                    **r,
                    "parameters": r["parameters"] if "parameters" in r else {}
                }
                for r in candidate_results
            ],
            "best_train_acc": best_fitness if valid_results else None,
            "best_test_acc": valid_results[0]["test_acc"] if valid_results else None,
            "best_parameters": parent_params.copy() if valid_results else parent_params.copy(),
        }
        (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        
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
                # Log best parameters with participant prefix
                for param_name, param_value in parent_params.items():
                    log_dict[f"p{participant_id}_best_{param_name}"] = param_value
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
    
    parser = argparse.ArgumentParser(description="ROTE Evolution: Iterative evolution of Choice13k programs")
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
            run_name = f"{datetime.now():%y%m%d_%H%M%S}"
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
        base_run_dir = f"generated_outputs/choice13k_ROTE_evo/run_{timestamp}"
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

