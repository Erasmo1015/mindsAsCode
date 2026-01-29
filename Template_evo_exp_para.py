"""
ROTE_evo_exp_para.py: Iterative evolution loop over executable Choice13k and Gridworld programs.

Except-parameters version: Generates full program code but preserves original parameter values.
This version allows the LLM to generate entirely new program implementations for Choice13k or Gridworld,
but all parameter names and values from the seed program are preserved exactly.

The evolution process:
1. Starts with seed program (configurable via --seed_path)
2. Extracts parameters from seed program
3. Generates candidate program variants per iteration (full code, but parameters are preserved)
4. Injects original parameters back into generated code
5. Evaluates each program on dataset (Choice13k or Gridworld)
6. Reports performance and selects best performers
7. Uses best programs as parents for next generation
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
import jax
import jax.numpy as jnp
import flax

# Import data loading (this is acceptable as it's a data module, not ROTE/evo code)
from data_modules.choice13k import get_choice13k_experiments, Experiment, Block
from agent import AgentExecutionFramework
from plot_and_eval import get_all_problem_configs, make_dataloader
from environment import AutomaticityEnv, State


def load_seed_program(seed_path: str) -> str:
    """Load the seed program from the specified path.
    Handles markdown code blocks by extracting Python code."""
    with open(seed_path, 'r') as f:
        content = f.read()
    
    # Extract code from markdown code blocks if present
    code_pattern = r'```(?:python)?(.*?)```'
    matches = re.findall(code_pattern, content, re.DOTALL)
    if matches:
        # Return the first code block found
        return matches[0].strip()
    
    # Return as-is if no code blocks found
    return content


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


def find_template_program_for_gridworld(num_blocks: int, num_walls: int, agent_id: int) -> Optional[str]:
    """
    Auto-detect template program for gridworld based on problem config and agent_id.
    
    Looks in persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/
    for a program matching the agent_id.
    
    Args:
        num_blocks: Number of blocks in the problem
        num_walls: Number of walls in the problem
        agent_id: Agent type ID
        
    Returns:
        Path to template program if found, None otherwise
    """
    # Get hand-designed program name mapping
    hand_designed_dir = Path("generated_outputs/hand_designed")
    if not hand_designed_dir.exists():
        return None
    
    files = sorted([f for f in os.listdir(hand_designed_dir) if f.endswith('.txt')])
    if agent_id >= len(files):
        return None
    
    hand_designed_name = files[agent_id].replace('.txt', '')
    
    # Try to find program in the problem-specific folder
    problem_dir = Path(f"persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}")
    
    # Try patterns: hand_designed_name_agent{agent_id}.py or agent_{agent_id}.py
    possible_names = [
        f"{hand_designed_name}_agent{agent_id}.py",
        f"agent_{agent_id}.py",
    ]
    
    for name in possible_names:
        candidate_path = problem_dir / name
        if candidate_path.exists():
            return str(candidate_path)
    
    # If not found, return None
    return None


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
        "__import__": __import__,  # Make __import__ directly available in global namespace
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


def evaluate_program(choose_fn: Callable, trials: List[Dict[str, Any]], verbose: bool = False, n_seeds: int = 1) -> Dict[str, float]:
    """Evaluate a program on trials and return accuracy metrics.
    
    Args:
        choose_fn: The program function to evaluate
        trials: List of trial dictionaries
        verbose: Whether to print verbose output
        n_seeds: Number of evaluation runs to average (default: 1)
    
    Returns:
        Dictionary with averaged accuracy metrics across n_seeds runs
    """
    accuracies = []
    total = len(trials)
    
    for seed in range(n_seeds):
        correct = 0
        errors = 0
        for t in trials:
            try:
                pred = choose_fn(t["problem"], t["history"])
                if pred is not None and pred == t["action"]:
                    correct += 1
            except Exception as e:
                errors += 1
                if verbose and errors <= 3 and seed == 0:  # Only print first 3 errors from first seed
                    print(f"  Evaluation error: {e}")
        acc = correct / total if total > 0 else 0.0
        accuracies.append(acc)
    
    # Average across seeds
    avg_acc = np.mean(accuracies) if accuracies else 0.0
    # Use first seed's error count for reporting
    correct = int(avg_acc * total)
    errors = total - correct if n_seeds == 1 else 0  # Error count only meaningful for single seed
    
    result = {"accuracy": avg_acc, "total": total, "correct": correct, "errors": errors}
    if verbose and errors > 0 and n_seeds == 1:
        print(f"  Total evaluation errors: {errors}/{total}")
    return result


def load_gridworld_data(data_path: str, num_blocks: int, num_walls: int, agent_id: int, 
                        num_datapoints: int = 100, start_idx: int = 0):
    """Load gridworld trajectory data for a specific problem config and agent type.
    
    Args:
        data_path: Path to data directory
        num_blocks: Number of blocks in the problem
        num_walls: Number of walls in the problem
        agent_id: Agent type ID (0-indexed)
        num_datapoints: Number of datapoints to load
        start_idx: Starting index for datapoints (for train/test split)
    
    Returns:
        Dictionary with 'states' and 'actions' for evaluation
    """
    data_folder = f"{data_path}/num_blocks{num_blocks}/num_walls{num_walls}"
    data_file = f"{data_folder}/gt_fsm_traj_data_1agents.msgpack"
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data structure (similar to plot_and_eval.py)
    with open(data_file, "rb") as f:
        serialized_data = f.read()
    
    # Create target structure
    num_steps = 100
    action_target = jnp.zeros((20, num_datapoints, num_steps, 1))
    agent_id_target = jnp.zeros((20, num_datapoints, num_steps))
    state_target = {
        'agent_id': agent_id_target,
        'agent_locations': jnp.zeros((20, num_datapoints, num_steps, 1, 2)),
        'agent_inventory': jnp.zeros((20, num_datapoints, num_steps, 1)),
        'agent_inventory_colors': jnp.zeros((20, num_datapoints, num_steps, 1, 3)),
        'block_colors': jnp.zeros((20, num_datapoints, num_steps, num_blocks, 3)),
        'block_locations': jnp.zeros((20, num_datapoints, num_steps, num_blocks, 2)),
        'time': jnp.zeros((20, num_datapoints, num_steps)),
        'terminal': jnp.zeros((20, num_datapoints, num_steps)),
        'wall_locations': jnp.zeros((20, num_datapoints, num_steps, num_walls + 2 * (10 * 2 - 1) + 2, 2)),
    }
    target = {
        'states': state_target,
        'actions': action_target,
        'agent_ids': agent_id_target,
    }
    
    loaded_data = flax.serialization.from_bytes(target, serialized_data)
    
    # Extract data for the specific agent type, with start_idx for train/test split
    end_idx = min(start_idx + num_datapoints, loaded_data['states']['agent_locations'].shape[1])
    actual_num = end_idx - start_idx
    
    agent_data = {
        'states': jax.tree.map(lambda x: x[agent_id, start_idx:end_idx, :, ...], loaded_data['states']),
        'actions': loaded_data['actions'][agent_id, start_idx:end_idx, :, :],
    }
    
    return agent_data


def evaluate_gridworld_program(agent_code: str, data_path: str, num_blocks: int, num_walls: int, 
                                agent_id: int, num_datapoints: int = 100, num_steps: int = 20,
                                verbose: bool = False, n_seeds: int = 1, 
                                evaluate_on_observed: bool = False) -> Dict[str, float]:
    """Evaluate a gridworld program on trajectory data using the same logic as ROTE.
    
    Args:
        agent_code: The program code to evaluate
        data_path: Path to gridworld data directory
        num_blocks: Number of blocks in the problem
        num_walls: Number of walls in the problem
        agent_id: Agent type ID (0-indexed)
        num_datapoints: Number of datapoints to evaluate on
        num_steps: Number of steps to evaluate (default: 20)
        verbose: Whether to print verbose output
        n_seeds: Number of evaluation runs to average (default: 1)
        evaluate_on_observed: If True, evaluate on first 20 steps (matching ROTE's training/weighting).
                             If False, evaluate on future steps (matching ROTE's evaluation).
    
    Returns:
        Dictionary with accuracy metrics
    """
    framework = AgentExecutionFramework()
    
    # Compile the agent
    try:
        agent = framework.compile_agent(agent_code, num_agents=1, num_blocks=num_blocks)
    except Exception as e:
        if verbose:
            print(f"  Compilation error: {e}")
        return {"accuracy": 0.0, "total": 0, "correct": 0, "errors": 1}
    
    # Create a dummy args object for make_dataloader
    class DummyArgs:
        def __init__(self):
            self.data_path = data_path
            self.num_agents = 1
            self.num_datapoints_per_agent = num_datapoints
            self.num_steps = num_steps
            self.group = False
            self.flip_quarter = True  # Data files use _flip_quarter extension
            self.env_size = 10
            self.as_images = False
    
    dummy_args = DummyArgs()
    
    accuracies = []
    total_steps = 0
    correct_steps = 0
    
    for seed in range(n_seeds):
        try:
            # Use make_dataloader to load data (same as ROTE)
            # For train: use first 80 datapoints, for test: use datapoints 80-100
            start_idx = 0 if num_datapoints >= 80 else 80
            num_datapoints_to_load = num_datapoints if num_datapoints >= 80 else 20
            
            # Create dataloader using make_dataloader (same as plot_and_eval.py)
            dataloader = make_dataloader(
                dummy_args,
                num_agents_to_sample=1,
                num_datapoints_per_agent_to_sample=num_datapoints_to_load,
                training=False,
                epoch=0,
                num_blocks=num_blocks,
                num_walls=num_walls,
                agent_indices=[agent_id]
            )
            
            datapoint = next(dataloader)
            
            # Evaluate on datapoints (same structure as eval_fsm_bootstrap)
            seed_correct = 0
            seed_total = 0
            
            # ROTE evaluates on the last datapoint per agent (line 1234: x[a_idx, -1, :20+num_future_steps])
            # Match ROTE exactly: use -1 for datapoint index, iterate through agents
            # But we only have 1 agent, so we'll evaluate on multiple datapoints for better statistics
            for dp_idx in range(num_datapoints_to_load):
                try:
                    # Extract data sample exactly like ROTE (line 1234)
                    # ROTE: data_sample = jax.tree.map(lambda x: x[a_idx, -1, :20+num_future_steps], datapoint)
                    # We use dp_idx instead of -1 to iterate through datapoints
                    # ROTE uses :20+num_future_steps where num_future_steps=20, so :40 steps total
                    data_sample = jax.tree.map(lambda x: x[0, dp_idx, :20+num_steps], datapoint)
                    
                    # Extract initial trajectory (first 20 steps) from data_sample
                    initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
                    initial_actions_traj = jax.tree.map(lambda x: x[:20], data_sample['actions'])
                    
                    # Convert JAX arrays to numpy arrays (same as ROTE does implicitly)
                    def to_numpy(x):
                        if isinstance(x, (jnp.ndarray, jax.Array)):
                            return np.array(x)
                        return x
                    
                    if evaluate_on_observed:
                        # TRAIN MODE: Evaluate on first 20 steps (matching ROTE's training/weighting phase)
                        # This matches how ROTE calculates log_prob_hypothesis (baselines/gridROTE.py line 469-496)
                        for timestep in range(initial_actions_traj.shape[0] - 1):  # 0 to 18 (19 steps)
                            try:
                                state = jax.tree.map(lambda x: x[timestep], initial_states_traj)
                                state = jax.tree.map(to_numpy, state)
                                if len(state['agent_locations']) == 1:
                                    state['agent_id'] = 0
                                
                                # Get ground truth action for this timestep
                                gt_action = int(initial_actions_traj[timestep][0])
                                
                                # Get prediction from agent
                                predicted_action = framework.execute_agent(agent, state)
                                
                                # Convert action to int (same as ROTE)
                                action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
                                action_space_2 = ["stay", "right", "left", "down", "up", "interact"]
                                
                                if isinstance(predicted_action, tuple):
                                    predicted_action = list(predicted_action)
                                elif isinstance(predicted_action, str):
                                    predicted_action = predicted_action.lower()
                                    predicted_action = action_space_2.index(predicted_action)
                                else:
                                    predicted_action = int(predicted_action)
                                
                                if predicted_action in action_space:
                                    predicted_action = action_space.index(predicted_action)
                                elif predicted_action in action_space_2:
                                    predicted_action = action_space_2.index(predicted_action)
                                
                                # Compare with ground truth (same as ROTE line 484)
                                if predicted_action == gt_action:
                                    seed_correct += 1
                                seed_total += 1
                                
                            except Exception as e:
                                seed_total += 1  # Count as incorrect
                    else:
                        # TEST MODE: Evaluate on future steps (matching ROTE's evaluation phase)
                        # Get ground truth future actions (from step 19 onwards) - exactly like ROTE (line 1244)
                        gt_future_actions = data_sample['actions'][19:]  # Shape (num_steps, num_env_agents) or (num_steps, 1)
                        
                        # If we don't have enough future actions, use what we have
                        if gt_future_actions.shape[0] < num_steps:
                            actual_num_steps = min(gt_future_actions.shape[0], num_steps)
                        else:
                            actual_num_steps = num_steps
                        
                        # Initialize environment for simulation (same as ROTE)
                        env = AutomaticityEnv(num_agents=1, size=10, max_steps=num_steps, 
                                              num_blocks=num_blocks, num_walls=num_walls)
                        
                        # Extract state at step 19 (end of initial trajectory) exactly like ROTE
                        state_at_t19 = jax.tree.map(lambda x: x[19], initial_states_traj)
                        
                        # Verify state_at_t19 is a dict (should be preserved by jax.tree.map)
                        if not isinstance(state_at_t19, dict):
                            # Try to convert if it's a list or tuple
                            if isinstance(state_at_t19, (list, tuple)) and len(state_at_t19) > 0:
                                # Maybe it's a list of dicts? Take the first one
                                if isinstance(state_at_t19[0], dict):
                                    state_at_t19 = state_at_t19[0]
                                else:
                                    seed_total += actual_num_steps
                                    continue
                            else:
                                seed_total += actual_num_steps
                                continue
                        
                        state_at_t19_np = jax.tree.map(to_numpy, state_at_t19)
                        
                        # Start from state at step 19 (end of initial trajectory) - exactly like ROTE
                        current_sim_state_pytree = state_at_t19_np
                        current_sim_state_pytree = State(
                            wall_locations=current_sim_state_pytree['wall_locations'],
                            agent_locations=current_sim_state_pytree['agent_locations'],
                            block_locations=current_sim_state_pytree['block_locations'],
                            agent_inventory=current_sim_state_pytree['agent_inventory'],
                            agent_inventory_colors=current_sim_state_pytree['agent_inventory_colors'],
                            block_colors=current_sim_state_pytree['block_colors'],
                            time=current_sim_state_pytree['time'],
                            terminal=False,
                            agent_id=0
                        )
                        current_obs = env.get_observation(current_sim_state_pytree)[0]
                        
                        # Simulate future steps (same as ROTE - use ground truth observations from data)
                        for step_idx in range(actual_num_steps):
                            if step_idx >= gt_future_actions.shape[0]:
                                break
                            
                            try:
                                # Get observation from ground truth data (same as ROTE line 1530)
                                # ROTE uses: current_obs = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])
                                current_obs_raw = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])
                                # Convert to numpy and ensure it's a dict
                                current_obs = jax.tree.map(to_numpy, current_obs_raw)
                                current_obs['agent_id'] = 0
                                
                                # Get prediction from agent
                                predicted_action = framework.execute_agent(agent, current_obs)
                                
                                # Extract ground truth action exactly like ROTE (line 1502, 1506)
                                gt_action_this_step = gt_future_actions[step_idx]  # (num_env_agents,) or (1,)
                                # For single agent, use index 0 (same as ROTE line 1506 with aid=0)
                                if hasattr(gt_action_this_step, '__len__') and len(gt_action_this_step) > 0:
                                    gt_action = int(gt_action_this_step[0])
                                else:
                                    gt_action = int(gt_action_this_step)
                                
                                # Convert action to int (same as ROTE)
                                action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
                                action_space_2 = ["stay", "right", "left", "down", "up", "interact"]
                                
                                if isinstance(predicted_action, tuple):
                                    predicted_action = list(predicted_action)
                                elif isinstance(predicted_action, str):
                                    predicted_action = predicted_action.lower()
                                    predicted_action = action_space_2.index(predicted_action)
                                else:
                                    predicted_action = int(predicted_action)
                                
                                if predicted_action in action_space:
                                    predicted_action = action_space.index(predicted_action)
                                elif predicted_action in action_space_2:
                                    predicted_action = action_space_2.index(predicted_action)
                                
                                # Compare with ground truth
                                if predicted_action == gt_action:
                                    seed_correct += 1
                                seed_total += 1
                                
                            except Exception as e:
                                seed_total += 1  # Count as incorrect
                        
                except Exception as e:
                    if verbose:
                        print(f"  Error processing datapoint {dp_idx}: {e}")
                    # Count as incorrect based on mode
                    if evaluate_on_observed:
                        seed_total += 19  # First 20 steps minus 1 (timestep 0 to 18)
                    else:
                        seed_total += num_steps
                    continue
                        
        except Exception as e:
            if verbose:
                print(f"  Data loading error: {e}")
            seed_correct = 0
            seed_total = 1  # Avoid division by zero
        
        acc = seed_correct / seed_total if seed_total > 0 else 0.0
        accuracies.append(acc)
        total_steps = seed_total
        correct_steps = seed_correct
    
    # Average across seeds
    avg_acc = np.mean(accuracies) if accuracies else 0.0
    correct = int(avg_acc * total_steps) if total_steps > 0 else 0
    
    result = {"accuracy": avg_acc, "total": total_steps, "correct": correct, "errors": 0}
    return result


def generate_gridworld_program_variants(
    client: OpenAI,
    model_name: str,
    template_code: str,
    parent_codes: List[str],
    n_variants: int = 10,
    max_tokens: int = 2000,
    parent_train_accuracies: Optional[List[float]] = None,
    fixed_parameters: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Generate full program code variants for gridworld (except-parameters mode).
    The LLM modifies the entire program code, but parameters are preserved.
    
    Args:
        template_code: Original template code
        parent_codes: List of parent program codes (elite programs from previous iterations)
        n_variants: Number of variants to generate
        max_tokens: Maximum tokens for generation
        parent_train_accuracies: List of training accuracies for each parent (for guidance)
        fixed_parameters: Dictionary of parameter names and values to preserve (if None, extracted from template)
    
    Returns a list of program code strings with original parameters injected.
    """
    # Load prompts from file
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    prompt_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "gridworld", "non_strict", "infer_single_fsm.txt")
    code_template_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "gridworld", "non_strict", "single_code_template.txt")
    
    try:
        base_prompt_template = open(prompt_path).read()
        code_template = open(code_template_path).read()
    except FileNotFoundError as e:
        print(f"Warning: Could not load prompt files: {e}")
        print("Falling back to hardcoded prompts.")
        # Fallback to hardcoded prompt
        base_prompt_template = """You are a robot viewing agents acting in an object-centric environment. Your goal is to model the behavior of the agents as a finite state machine (FSM) code in python. You will be provided experiences in the format of (state, action) tuples.

This environment simulates potentially multiple agents interacting in a grid world filled with colored blocks and walls. The world is a square grid (default 7x7) with walls on the perimeter. There are also walls scattered across the interior 6x6 region. Multiple agents, each represented by a distinct color (red, blue, green, etc.), navigate this space alongside colored blocks that can be picked up and transported.

Agents can perform six basic actions: staying in place, moving in any of the four cardinal directions (up, down, left, right), or interacting with blocks. If agent is on a grid cell that a colored block is on and they don't have an item in their inventory, they have to press the 'interact' action to add that block to their inventory. If they press the interact button but have an item in their inventory, they stay in place, but remove the item they had from their inventory.  Importantly, agents can't occupy the same space or swap positions, and they're limited to carrying one block at a time. If they both try to move into the same cell, they will both stay in place. If you don't have an item in your inventory, this is represented by your inventory being equal to -1. If you are holding a block and try walking onto a cell where another block is, you will remain in the same place with the same block in your inventory (equivalent of a stay action).

Each agent receives detailed information about the environment's state, including the positions of all walls, agents, and blocks, as well as information about what blocks are being carried by which agents.

You need to implement the python code to model the logic of the agent's behavior, as seen in the provided experiences. Please follow the template to implement the code. The code needs to be directly runnable on the state and return the action in python as provided in the experiences. Try to keep your code as concise as possible.

You need to implement python code to model the logic of the world as seen in the following experiences:"""
        code_template = """Please implement code to model the logic of the agent's behavior as demonstrated by the experiences. Here is the template for the agent's FSM class. Please implement the FSM code for an agent following the template. The code needs to be directly runnable on the inputs of state and return an action based on an observation. Make sure the agent always returns an action in the list [0, 1, 2, 3, 4, 5] corresponding to "stay", "right", "left", "down", "up", "interact"."""
    
    # Format multiple parent programs
    num_parents = len(parent_codes)
    parent_programs_text = ""
    if num_parents == 1:
        parent_programs_text = f"Current parent program:\n```python\n{parent_codes[0]}\n```"
    else:
        parent_programs_text = f"Reference parent programs ({num_parents} elite programs):\n"
        for i, (parent_code, acc) in enumerate(zip(parent_codes, parent_train_accuracies or [None] * num_parents)):
            acc_str = f" (train_acc: {acc:.4f})" if acc is not None else ""
            parent_programs_text += f"\nParent {i+1}{acc_str}:\n```python\n{parent_code}\n```\n"
    
    performance_info = ""
    if parent_train_accuracies:
        avg_acc = sum(parent_train_accuracies) / len(parent_train_accuracies)
        max_acc = max(parent_train_accuracies)
        performance_info = f"\nParent performance: Average train accuracy = {avg_acc:.4f}, Best = {max_acc:.4f}\n"
        if avg_acc < 0.5:
            performance_info += "NOTE: Performance is LOW. Consider significant changes to the program logic.\n"
        elif avg_acc > 0.8:
            performance_info += "NOTE: Performance is HIGH. Make refined improvements.\n"
        else:
            performance_info += "NOTE: Performance is MODERATE. Explore different approaches.\n"
        if num_parents > 1:
            performance_info += f"NOTE: You have {num_parents} parent programs to learn from. Combine the best ideas from each.\n"
    
    # Extract parameters if not provided
    if fixed_parameters is None:
        fixed_parameters = extract_parameters_from_template(template_code)
    
    # Format parameter preservation instruction
    parameter_constraint = ""
    if fixed_parameters:
        param_list = ", ".join([f"{k} = {v}" for k, v in fixed_parameters.items()])
        parameter_constraint = f"""
CRITICAL CONSTRAINT: You MUST preserve these exact parameter assignments in your generated code:
{param_list}

These parameters and their values must appear exactly as shown above. You can change everything else
(program logic, functions, control flow, etc.), but these parameter assignments must remain unchanged.
"""
    
    base_prompt = f"""{base_prompt_template}

{parent_programs_text}

{performance_info}
{parameter_constraint}

Your task: Generate an improved program variant. The variant should:
- Maintain the same class structure (FSMAgent with __init__ and act methods)
- Preserve the exact parameter assignments specified above
- Improve the decision-making logic (everything except parameters)
- Handle edge cases better
- Be more efficient or accurate

{code_template}

Output format: Provide the variant as a code block marked with ```python and ```.
The variant should be a complete, runnable program with the exact parameter values preserved.

Generate the variant now:"""

    # Generate variants one at a time to avoid huge prompts (especially with multiple parents)
    variants = []
    best_parent = parent_codes[0] if parent_codes else ""
    if fixed_parameters and best_parent:
        best_parent = inject_parameters_into_template(best_parent, fixed_parameters)
    
    for _ in tqdm(range(n_variants), desc="Generating gridworld variants"):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": base_prompt}],
                temperature=0.7,
                max_tokens=max_tokens,  # Per variant - keep original max_tokens
            )
            content = response.choices[0].message.content
            
            # Extract code block
            code_pattern = r'```(?:python)?(.*?)```'
            matches = re.findall(code_pattern, content, re.DOTALL)
            
            if matches:
                code = matches[0].strip()
                if 'class FSMAgent' in code or 'def act' in code:
                    # Inject fixed parameters into generated code
                    if fixed_parameters:
                        code = inject_parameters_into_template(code, fixed_parameters)
                    variants.append(code)
                else:
                    # If no valid code found, use parent
                    variants.append(best_parent)
            else:
                # No code block found, use parent
                variants.append(best_parent)
        except Exception as e:
            print(f"Warning: Failed to generate program variant: {e}")
            # Fallback: use best parent code
            variants.append(best_parent)
    
    return variants[:n_variants]


def generate_program_variants(
    client: OpenAI,
    model_name: str,
    parent_programs: List[str],
    train_trials: List[Dict[str, Any]],
    n_variants: int = 10,
    max_tokens: int = 800,
    parent_train_accuracies: Optional[List[float]] = None,
    fixed_parameters: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Generate full program variants based on parent program and training trials.
    
    This generates complete choose(problem, history) implementations without
    restrictions on structure or logic - only the function signature is fixed.
    However, all parameters from the seed program are preserved exactly.
    """
    # Load prompts from file
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    prompt_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "infer_single_choice.txt")
    code_template_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "single_code_template.txt")
    
    try:
        base_prompt = open(prompt_path).read()
        code_template = open(code_template_path).read()
    except FileNotFoundError as e:
        print(f"Warning: Could not load prompt files: {e}")
        print("Falling back to hardcoded prompts.")
        # Fallback to hardcoded prompts
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
    
    # Include parent programs as reference
    num_parents = len(parent_programs)
    if num_parents == 1:
        parent_context = f"\n\nReference program (parent):\n```python\n{parent_programs[0]}\n```\n\n"
        parent_context += "Generate a variant that improves upon or explores alternatives to the parent program.\n"
    else:
        parent_context = f"\n\nReference parent programs ({num_parents} elite programs):\n"
        for i, (parent_program, acc) in enumerate(zip(parent_programs, parent_train_accuracies or [None] * num_parents)):
            acc_str = f" (train_acc: {acc:.4f})" if acc is not None else ""
            parent_context += f"\nParent {i+1}{acc_str}:\n```python\n{parent_program}\n```\n"
        parent_context += "\nGenerate a variant that combines the best ideas from these parent programs.\n"
    
    # Format parameter preservation instruction
    parameter_constraint = ""
    if fixed_parameters:
        param_list = ", ".join([f"{k} = {v}" for k, v in fixed_parameters.items()])
        parameter_constraint = f"""
CRITICAL CONSTRAINT: You MUST preserve these exact parameter assignments in your generated code:
{param_list}

These parameters and their values must appear exactly as shown above. You can change everything else
(program logic, functions, control flow, etc.), but these parameter assignments must remain unchanged.
"""
    
    prompt_text = f"{base_prompt}\n{state_text}\n{parent_context}\n{parameter_constraint}\n{code_template}"
    
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
            # Inject fixed parameters into generated code
            if fixed_parameters:
                code = inject_parameters_into_template(code, fixed_parameters)
            programs.append(code)
        except Exception as e:
            print(f"Warning: Failed to generate program variant: {e}")
            programs.append("")
    return programs


def run_evolution(
    seed_program_path: str,
    dataset: str = "choice13k",
    participant_id: int = 0,
    data_path: str = "data",
    num_blocks: Optional[int] = None,
    num_walls: Optional[int] = None,
    agent_id: Optional[int] = None,
    n_iterations: int = 5,
    n_candidates_per_iteration: int = 10,
    model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    client_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    wandb=None,
    n_eval_seeds: int = 3,
    sample_size: int = 10,
):
    """
    Run iterative evolution loop over programs (Choice13k or Gridworld, non-strict mode).
    
    Args:
        seed_program_path: Path to seed program
        dataset: "choice13k" or "gridworld"
        participant_id: Which participant's data to use (0-indexed, for choice13k)
        data_path: Path to data directory (for gridworld)
        num_blocks: Number of blocks (for gridworld)
        num_walls: Number of walls (for gridworld)
        agent_id: Agent type ID (for gridworld)
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
    
    # Extract parameters from seed program (to preserve them)
    fixed_parameters = extract_parameters_from_template(seed_code)
    if fixed_parameters:
        print(f"Extracted {len(fixed_parameters)} parameters to preserve: {list(fixed_parameters.keys())}")
    else:
        print("No parameters found in seed program - will allow full code generation")
    
    # Branch based on dataset
    if dataset == "gridworld":
        if num_blocks is None or num_walls is None or agent_id is None:
            raise ValueError("For gridworld, num_blocks, num_walls, and agent_id must be provided")
        print(f"Gridworld mode (non-strict): num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")
        train_trials = None  # Not used for gridworld
        test_trials = None
        options = None
    else:
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
        mode = "exp_para"  # Template_evo_exp_para.py uses exp_para mode
        if dataset == "gridworld":
            output_dir = f"generated_outputs/gridworld/{mode}/run_{timestamp}/epoch_0/agent_{agent_id}"
        else:
            output_dir = f"generated_outputs/choice13k/{mode}/run_{timestamp}/participant_{participant_id}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up local log file for wandb metrics (if wandb is enabled)
    log_file_path = None
    if wandb is not None:
        log_file_path = output_path / "wandb_metrics.jsonl"
    
    # ===== BASELINE EVALUATION =====
    print(f"\n{'='*80}")
    print(f"BASELINE EVALUATION: Evaluating seed program ({seed_program_path})")
    print(f"{'='*80}")
    
    if dataset == "gridworld":
        # Train: Evaluate on first 20 steps (matching ROTE's training/weighting phase)
        baseline_train_eval = evaluate_gridworld_program(
            seed_code, data_path, num_blocks, num_walls, agent_id,
            num_datapoints=80, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
            evaluate_on_observed=True  # Match ROTE's training: evaluate on first 20 steps
        )
        # Test: Evaluate on future steps (matching ROTE's evaluation phase)
        baseline_test_eval = evaluate_gridworld_program(
            seed_code, data_path, num_blocks, num_walls, agent_id,
            num_datapoints=20, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
            evaluate_on_observed=False  # Match ROTE's evaluation: evaluate on future steps
        )
    else:
        baseline_fn = compile_program(seed_code)
        if baseline_fn is None:
            print("ERROR: Failed to compile baseline program!")
            return None
        baseline_train_eval = evaluate_program(baseline_fn, train_trials, verbose=True, n_seeds=n_eval_seeds)
        baseline_test_eval = evaluate_program(baseline_fn, test_trials, verbose=True, n_seeds=n_eval_seeds)
    
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
        baseline_log_dict = {}
        if dataset == "gridworld":
            # Use agent-specific keys if agent_id is provided
            if agent_id is not None:
                baseline_log_dict = {
                    f"a{agent_id}_train_accuracy": baseline_train_eval["accuracy"],
                    f"a{agent_id}_test_accuracy": baseline_test_eval["accuracy"],
                    f"a{agent_id}_is_baseline": 1,
                }
            else:
                baseline_log_dict = {
                    f"gw_train_accuracy": baseline_train_eval["accuracy"],
                    f"gw_test_accuracy": baseline_test_eval["accuracy"],
                    f"gw_is_baseline": 1,
                }
        else:
            baseline_log_dict = {
                f"p{participant_id}_train_accuracy": baseline_train_eval["accuracy"],
                f"p{participant_id}_test_accuracy": baseline_test_eval["accuracy"],
                f"p{participant_id}_is_baseline": 1,
            }
        wandb.log(baseline_log_dict, step=0)
        
        # Also save baseline to local JSONL file
        if log_file_path is not None:
            baseline_entry = {
                "step": 0,
                "iteration": -1,  # Baseline is before iteration 0
                **baseline_log_dict
            }
            with open(log_file_path, "a") as f:
                f.write(json.dumps(baseline_entry) + "\n")
    
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
    
    # Track elite parents (top programs across all iterations)
    # Format: list of (code, train_acc, test_acc, program_id) tuples, sorted by train_acc descending
    elite_parents = [(
        seed_code,
        baseline_train_eval['accuracy'],
        baseline_test_eval['accuracy'],
        "baseline"
    )]
    
    # Initialize parent_program for first iteration
    parent_program = seed_code
    
    # Evolution loop
    for iteration in range(n_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*80}")
        
        iter_dir = output_path / f"iteration_{iteration}"
        iter_dir.mkdir(exist_ok=True)
        candidates_dir = iter_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)
        
        # Select sample_size parents from elite set (sorted by train_acc descending)
        # Always include the best parent first
        num_parents_to_use = min(sample_size, len(elite_parents))
        selected_parents = elite_parents[:num_parents_to_use]
        parent_codes = [p[0] for p in selected_parents]
        parent_train_accs = [p[1] for p in selected_parents]
        
        print(f"\nUsing {num_parents_to_use} parent(s) from elite set (sample_size={sample_size}):")
        for i, (code, train_acc, test_acc, prog_id) in enumerate(selected_parents):
            print(f"  Parent {i+1}: {prog_id} (train_acc={train_acc:.4f}, test_acc={test_acc:.4f})")
        
        # Generate candidate programs (full code, not just parameters)
        print(f"\nGenerating {n_candidates_per_iteration} candidate programs...")
        if dataset == "gridworld":
            candidate_codes = generate_gridworld_program_variants(
                client=client,
                model_name=model_name,
                template_code=seed_code,
                parent_codes=parent_codes,
                n_variants=n_candidates_per_iteration,
                parent_train_accuracies=parent_train_accs,
                fixed_parameters=fixed_parameters,
            )
        else:
            candidate_codes = generate_program_variants(
                client=client,
                model_name=model_name,
                parent_programs=parent_codes,
                train_trials=train_trials,
                n_variants=n_candidates_per_iteration,
                parent_train_accuracies=parent_train_accs,
                fixed_parameters=fixed_parameters,
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
            
            if dataset == "gridworld":
                # Gridworld: evaluate using gridworld evaluation function
                # Train: Evaluate on first 20 steps (matching ROTE's training/weighting phase)
                train_eval = evaluate_gridworld_program(
                    code, data_path, num_blocks, num_walls, agent_id,
                    num_datapoints=80, num_steps=20, n_seeds=n_eval_seeds,
                    evaluate_on_observed=True  # Match ROTE's training: evaluate on first 20 steps
                )
                # Test: Evaluate on future steps (matching ROTE's evaluation phase)
                test_eval = evaluate_gridworld_program(
                    code, data_path, num_blocks, num_walls, agent_id,
                    num_datapoints=20, num_steps=20, n_seeds=n_eval_seeds,
                    evaluate_on_observed=False  # Match ROTE's evaluation: evaluate on future steps
                )
                
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
                    "valid": train_eval["errors"] == 0,
                })
            else:
                # Choice13k: compile and evaluate
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
                
                # Evaluate on train and test (with multiple seeds)
                train_eval = evaluate_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                test_eval = evaluate_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                
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
            if candidate_results[-1]["valid"]:
                all_candidate_results.append({
                    "iteration": iteration,
                    "candidate_idx": idx,
                    "train_acc": candidate_results[-1]["train_acc"],
                    "test_acc": candidate_results[-1]["test_acc"],
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
            
            # Add all valid candidates to elite set
            for result in valid_results:
                program_id = f"iteration_{iteration}_candidate_{result['idx']}"
                elite_parents.append((
                    result["code"],
                    result["train_acc"],
                    result["test_acc"],
                    program_id
                ))
            
            # Sort elite set by train accuracy (descending) and keep top programs
            # Keep at least sample_size * 2 programs to have diversity
            elite_parents.sort(key=lambda x: x[1], reverse=True)
            max_elite_size = max(sample_size * 2, 20)  # Keep at least 20 or 2x sample_size
            elite_parents = elite_parents[:max_elite_size]
            
            print(f"\nElite set updated: {len(elite_parents)} programs (top {max_elite_size} kept)")
            
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
        
        # Log to wandb (use dataset-specific metric names)
        if wandb is not None:
            if dataset == "gridworld":
                # Use agent-specific keys if agent_id is provided
                if agent_id is not None:
                    log_dict = {
                        f"a{agent_id}_n_valid": len(valid_results),
                    }
                    if valid_results:
                        log_dict[f"a{agent_id}_train_accuracy"] = best_fitness
                        log_dict[f"a{agent_id}_test_accuracy"] = valid_results[0]["test_acc"]
                        log_dict[f"a{agent_id}_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                        log_dict[f"a{agent_id}_avg_test_accuracy"] = np.mean([r["test_acc"] for r in valid_results])
                else:
                    log_dict = {
                        f"gw_n_valid": len(valid_results),
                    }
                    if valid_results:
                        log_dict[f"gw_train_accuracy"] = best_fitness
                        log_dict[f"gw_test_accuracy"] = valid_results[0]["test_acc"]
                        log_dict[f"gw_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                        log_dict[f"gw_avg_test_accuracy"] = np.mean([r["test_acc"] for r in valid_results])
            else:
                log_dict = {
                    f"p{participant_id}_n_valid": len(valid_results),
                }
                if valid_results:
                    log_dict[f"p{participant_id}_train_accuracy"] = best_fitness
                    log_dict[f"p{participant_id}_test_accuracy"] = valid_results[0]["test_acc"]
                    log_dict[f"p{participant_id}_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                    log_dict[f"p{participant_id}_avg_test_accuracy"] = np.mean([r["test_acc"] for r in valid_results])
            wandb.log(log_dict, step=iteration + 1)  # Step starts at 1 (baseline is step=0)
            
            # Also save to local JSONL file
            if log_file_path is not None:
                log_entry = {
                    "step": iteration + 1,
                    "iteration": iteration,
                    **log_dict
                }
                with open(log_file_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
    
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
        "participant_id": participant_id if dataset == "choice13k" else agent_id,
        "train_acc": overall_best_train['train_accuracy'],
        "test_acc": overall_best_test['test_accuracy'],
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ROTE Evolution (Except-Parameters): Iterative evolution of Choice13k and Gridworld programs with parameter preservation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="choice13k",
        choices=["choice13k", "gridworld"],
        help="Dataset to use: choice13k or gridworld",
    )
    parser.add_argument(
        "--seed_path",
        type=str,
        default=None,
        help="Path to seed program (starting persona). If not set, auto-detects from persona_code_example/gridworld/ for gridworld. Default for choice13k: persona_code_example/hard.py",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to data directory (for gridworld)",
    )
    parser.add_argument(
        "--loop_mode",
        type=str,
        default="random",
        choices=["random", "sequential"],
        help="Loop mode for gridworld: sequential evaluates problem configs systematically",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs (problem configs) to evaluate in sequential mode",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="Number of blocks in the problem (for gridworld)",
    )
    parser.add_argument(
        "--num_walls",
        type=int,
        default=None,
        help="Number of walls in the problem (for gridworld)",
    )
    parser.add_argument(
        "--agent_id",
        type=int,
        default=None,
        help="Agent type ID to evaluate (0-indexed, for gridworld). If None and num_agents_to_sample > 1, processes all agent types.",
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
        help="Number of participants/agent types to process (0-indexed, from 0 to num_agents_to_sample-1). Used when participant_id/agent_id is None.",
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
        "--sample_size",
        type=int,
        default=3,
        help="Number of parent programs to use when generating each child (default: 3)",
    )
    parser.add_argument(
        "--n_eval_seeds",
        type=int,
        default=3,
        help="Number of evaluation runs per program (averaged for final accuracy). Default: 3",
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
    
    # Create timestamp once at the beginning to ensure consistency between wandb name and folder name
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    
    # Optional wandb setup
    wandb_enabled = False
    wandb = None
    log_file_path = None
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb
            # Include dataset in run name
            dataset_prefix = "gridworld" if args.dataset == "gridworld" else "choice13k"
            run_name = f"{dataset_prefix}_exp_para_{timestamp}"
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
        # Auto-generated output: create base run directory (use same timestamp)
        mode = "exp_para"  # Template_evo_exp_para.py uses exp_para mode
        if args.dataset == "gridworld":
            base_run_dir = f"generated_outputs/gridworld/{mode}/run_{timestamp}"
        else:
            base_run_dir = f"generated_outputs/choice13k/{mode}/run_{timestamp}"
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
    
    # Determine seed program path
    if args.seed_path is None:
        if args.dataset == "gridworld":
            # Auto-detect template program for gridworld
            # For sequential mode, we'll determine this per epoch
            # For now, we'll handle it in the epoch loop
            seed_program_path = None
        else:
            # Default for choice13k
            seed_program_path = "persona_code_example/hard.py"
    else:
        seed_program_path = args.seed_path
    
    # Load and save seed program once in the experiment folder (if we have a single seed)
    if seed_program_path is not None:
        seed_code = load_seed_program(seed_program_path)
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
    
    # Handle gridworld: check if we have a single problem config with multiple agent types
    # This happens when num_blocks and num_walls are provided but not in sequential mode
    # AND num_agents_to_sample > 1 (or agent_id is not set, meaning process all)
    if args.dataset == "gridworld":
        num_blocks_arg = getattr(args, 'num_blocks', None)
        num_walls_arg = getattr(args, 'num_walls', None)
        agent_id_arg = getattr(args, 'agent_id', None)
        
        # Check if we're processing multiple agent types for a single problem config
        # Condition: not sequential mode, have problem config, and either:
        #   - num_agents_to_sample > 1, OR
        #   - agent_id is not set (process all agent types)
        if (num_blocks_arg is not None and num_walls_arg is not None and 
            args.loop_mode != "sequential" and 
            (args.num_agents_to_sample > 1 or agent_id_arg is None)):
            # Process all agent types for this single problem config
            print(f"\n{'='*80}")
            print(f"Processing {args.num_agents_to_sample} agent types for problem: num_blocks={num_blocks_arg}, num_walls={num_walls_arg}")
            print(f"{'='*80}")
            
            # Determine which agent types to process
            if agent_id_arg is not None and args.num_agents_to_sample == 1:
                # Single agent type specified
                agent_types_to_process = [agent_id_arg]
            else:
                # Process all agent types up to num_agents_to_sample
                agent_types_to_process = list(range(args.num_agents_to_sample))
            
            for agent_id in tqdm(agent_types_to_process, desc="Agent types"):
                print(f"\n{'='*80}")
                print(f"Processing agent type {agent_id}/{args.num_agents_to_sample-1}")
                print(f"{'='*80}")
                
                # Auto-detect template program for this agent type
                if args.seed_path is None:
                    detected_seed_path = find_template_program_for_gridworld(num_blocks_arg, num_walls_arg, agent_id)
                    if detected_seed_path is None:
                        print(f"Warning: Could not auto-detect template program for num_blocks={num_blocks_arg}, num_walls={num_walls_arg}, agent_id={agent_id}")
                        print(f"Expected location: persona_code_example/gridworld/num_blocks{num_blocks_arg}_num_walls{num_walls_arg}/")
                        print("Skipping this agent type...")
                        continue
                    agent_seed_path = detected_seed_path
                    print(f"Auto-detected seed program: {agent_seed_path}")
                else:
                    agent_seed_path = args.seed_path
                
                # Construct output directory
                if base_run_dir is not None:
                    agent_output_dir = os.path.join(base_run_dir, f"agent_{agent_id}")
                else:
                    mode = "exp_para"  # Template_evo_exp_para.py uses exp_para mode
                    agent_output_dir = f"generated_outputs/gridworld/{mode}/run_{timestamp}/agent_{agent_id}"
                
                agent_summary = run_evolution(
                    seed_program_path=agent_seed_path,
                    dataset=args.dataset,
                    participant_id=agent_id,  # Use agent_id as participant_id for tracking
                    data_path=args.data_path,
                    num_blocks=num_blocks_arg,
                    num_walls=num_walls_arg,
                    agent_id=agent_id,
                    n_iterations=args.n_iterations,
                    n_candidates_per_iteration=args.n_candidates,
                    model_name=args.model_name,
                    client_kwargs=client_kwargs if client_kwargs else None,
                    output_dir=agent_output_dir,
                    wandb=wandb,
                    n_eval_seeds=args.n_eval_seeds,
                    sample_size=args.sample_size,
                )
                
                # Update summary
                if agent_summary is not None and summary_file is not None:
                    participants_summary.append({
                        **agent_summary,
                        'agent_id': agent_id,
                        'num_blocks': num_blocks_arg,
                        'num_walls': num_walls_arg,
                    })
                    # Write CSV file
                    with open(summary_file, 'w', newline='') as f:
                        fieldnames = ['agent_id', 'num_blocks', 'num_walls', 'train_acc', 'test_acc']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(participants_summary)
                    print(f"\nSummary updated: {summary_file}")
            
            # Finished processing all agent types
            if wandb is not None:
                wandb.finish()
            return
    
    # Handle sequential mode for gridworld
    if args.dataset == "gridworld" and args.loop_mode == "sequential":
        all_problem_configs = get_all_problem_configs()
        num_agent_types = 10  # Total number of agent types
        total_configs = len(all_problem_configs)
        
        # Calculate which config and agent to use for each epoch
        def get_config_and_agents_for_epoch(epoch_idx):
            """Get (num_blocks, num_walls, agent_indices_list) for a given epoch index."""
            if epoch_idx >= total_configs:
                return None, None, None
            num_blocks, num_walls = all_problem_configs[epoch_idx]
            # Use first num_agents_to_sample agent types
            agent_indices = list(range(min(args.num_agents_to_sample, num_agent_types)))
            return num_blocks, num_walls, agent_indices
        
        # Process each epoch
        epochs_to_process = min(args.num_epochs, total_configs)
        for epoch in range(epochs_to_process):
            num_blocks, num_walls, agent_indices = get_config_and_agents_for_epoch(epoch)
            if num_blocks is None:
                break
            
            # Process all agent types for this epoch
            for agent_id in agent_indices:
                print(f"\n{'='*80}")
                print(f"Processing epoch {epoch+1}/{epochs_to_process} - Problem: num_blocks={num_blocks}, num_walls={num_walls}, Agent: {agent_id}")
                print(f"{'='*80}")
                
                # Determine seed program path for this agent type
                if args.seed_path is None:
                    # Auto-detect template program
                    detected_seed_path = find_template_program_for_gridworld(num_blocks, num_walls, agent_id)
                    if detected_seed_path is None:
                        print(f"Warning: Could not auto-detect template program for num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")
                        print(f"Expected location: persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/")
                        print("Skipping this agent type...")
                        continue
                    epoch_seed_path = detected_seed_path
                    print(f"Auto-detected seed program: {epoch_seed_path}")
                else:
                    epoch_seed_path = args.seed_path
                
                # Construct output directory
                if base_run_dir is not None:
                    participant_output_dir = os.path.join(base_run_dir, f"epoch_{epoch}", f"agent_{agent_id}")
                else:
                    mode = "exp_para"  # Template_evo_exp_para.py uses exp_para mode
                    participant_output_dir = f"generated_outputs/gridworld/{mode}/run_{timestamp}/epoch_{epoch}/agent_{agent_id}"
                
                participant_summary = run_evolution(
                    seed_program_path=epoch_seed_path,
                    dataset=args.dataset,
                    participant_id=agent_id,  # Use agent_id for tracking
                    data_path=args.data_path,
                    num_blocks=num_blocks,
                    num_walls=num_walls,
                    agent_id=agent_id,
                    n_iterations=args.n_iterations,
                    n_candidates_per_iteration=args.n_candidates,
                    model_name=args.model_name,
                    client_kwargs=client_kwargs if client_kwargs else None,
                    output_dir=participant_output_dir,
                    wandb=wandb,
                    n_eval_seeds=args.n_eval_seeds,
                    sample_size=args.sample_size,
                )
                
                # Update summary (for gridworld, we might want a different summary structure)
                if participant_summary is not None and summary_file is not None:
                    participants_summary.append({
                        **participant_summary,
                        'epoch': epoch,
                        'num_blocks': num_blocks,
                        'num_walls': num_walls,
                        'agent_id': agent_id,
                    })
                    # Write CSV file
                    with open(summary_file, 'w', newline='') as f:
                        fieldnames = ['epoch', 'num_blocks', 'num_walls', 'agent_id', 'train_acc', 'test_acc']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(participants_summary)
                    print(f"\nSummary updated: {summary_file}")
    else:
        # Original logic for choice13k or random mode
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
                
                # Determine seed program path
                if args.seed_path is None:
                    if args.dataset == "gridworld":
                        num_blocks = getattr(args, 'num_blocks', None)
                        num_walls = getattr(args, 'num_walls', None)
                        agent_id = getattr(args, 'agent_id', None)
                        if num_blocks is not None and num_walls is not None and agent_id is not None:
                            detected_seed_path = find_template_program_for_gridworld(num_blocks, num_walls, agent_id)
                            if detected_seed_path is None:
                                print(f"Error: Could not auto-detect template program for num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")
                                print(f"Expected location: persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/")
                                continue
                            seed_program_path = detected_seed_path
                            print(f"Auto-detected seed program: {seed_program_path}")
                        else:
                            print("Error: For gridworld without --seed_path, must provide --num_blocks, --num_walls, and --agent_id")
                            continue
                    else:
                        seed_program_path = "persona_code_example/hard.py"
                else:
                    seed_program_path = args.seed_path
                
                participant_summary = run_evolution(
                    seed_program_path=seed_program_path,
                    dataset=args.dataset,
                    participant_id=participant_id,
                    data_path=args.data_path,
                    num_blocks=getattr(args, 'num_blocks', None),
                    num_walls=getattr(args, 'num_walls', None),
                    agent_id=getattr(args, 'agent_id', None),
                    n_iterations=args.n_iterations,
                    n_candidates_per_iteration=args.n_candidates,
                    model_name=args.model_name,
                    client_kwargs=client_kwargs if client_kwargs else None,
                    output_dir=participant_output_dir,
                    wandb=wandb,
                    n_eval_seeds=args.n_eval_seeds,
                    sample_size=args.sample_size,
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
