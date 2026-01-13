"""
ROTE_evo.py: Iterative evolution loop over executable Choice13k and Gridworld programs.

This module implements an evolutionary approach to improving decision-making programs,
starting from a seed program and iteratively generating and evaluating variants.

The evolution process:
1. Starts with seed program (configurable via --seed_path)
2. Generates candidate program variants per iteration
3. Evaluates each program on dataset (Choice13k or Gridworld)
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
import jax
import jax.numpy as jnp
import flax

# Import data loading (this is acceptable as it's a data module, not ROTE/evo code)
from data_modules.choice13k import get_choice13k_experiments, Experiment, Block
from agent import AgentExecutionFramework
from plot_and_eval import make_dataloader
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
    
    # Load base prompt template from file
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    prompt_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "strict", "infer_single_choice.txt")
    code_template_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "strict", "single_code_template.txt")
    
    try:
        base_prompt_file = open(prompt_path).read()
        code_template_file = open(code_template_path).read()
        # Use the file content as base, but we'll still format it with parameter-specific info
        base_prompt_template = base_prompt_file + """

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
    except FileNotFoundError as e:
        print(f"Warning: Could not load prompt files: {e}")
        print("Falling back to hardcoded prompts.")
        # Fallback to hardcoded prompt
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
    n_eval_seeds: int = 3,
    dataset: str = "choice13k",
    data_path: str = "data",
    num_blocks: Optional[int] = None,
    num_walls: Optional[int] = None,
    agent_id: Optional[int] = None,
):
    """
    Run iterative evolution loop over Choice13k or Gridworld programs.
    
    Args:
        seed_program_path: Path to seed program
        participant_id: Which participant's data to use (0-indexed) - for choice13k
        n_iterations: Number of evolution iterations
        n_candidates_per_iteration: Number of candidate programs per iteration
        model_name: LLM model name for generation
        client_kwargs: Optional OpenAI client kwargs (for local vLLM server)
        output_dir: Optional output directory for saving results
        dataset: "choice13k" or "gridworld"
        data_path: Path to data directory (for gridworld)
        num_blocks: Number of blocks (for gridworld)
        num_walls: Number of walls (for gridworld)
        agent_id: Agent type ID (for gridworld)
    """
    # Initialize client
    if client_kwargs is None:
        client_kwargs = {}
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    
    # Load seed program template
    print(f"Loading seed program template from {seed_program_path}...")
    template_code = load_seed_program(seed_program_path)
    
    # Branch based on dataset
    if dataset == "gridworld":
        # Gridworld: strict mode - try to extract parameters (even if none exist, keep definition consistent)
        if num_blocks is None or num_walls is None or agent_id is None:
            raise ValueError("For gridworld, num_blocks, num_walls, and agent_id must be provided")
        
        print(f"Gridworld mode (strict): num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")
        # Try to extract parameters (may be empty for gridworld templates)
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
                first_def_pos = template_code.find("def ", template_code.find("def act"))
                if param_pos < first_def_pos or first_def_pos == -1:
                    try:
                        baseline_params[param_name] = float(param_value)
                    except ValueError:
                        pass
        
        # For gridworld, if no parameters found, use empty dict (strict mode definition)
        if not baseline_params:
            print("Note: No parameters found in gridworld template. Using template as-is (strict mode with no parameters).")
            baseline_params = {}
        
        print(f"Baseline parameters: {baseline_params if baseline_params else '(none)'}")
        baseline_code = inject_parameters_into_template(template_code, baseline_params)
        parent_params = baseline_params.copy()
        
        # For gridworld, we'll use a simple train/test split on datapoints
        # Load data to get number of datapoints
        try:
            test_data = load_gridworld_data(data_path, num_blocks, num_walls, agent_id, num_datapoints=100)
            num_total = test_data['states']['agent_locations'].shape[0]
            train_split = int(num_total * 0.8)
            print(f"Gridworld data: {num_total} total datapoints, using {train_split} for train, {num_total - train_split} for test")
        except Exception as e:
            print(f"Warning: Could not load gridworld data: {e}")
            return None
    else:
        # Choice13k: extract parameters
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
            return None
        
        print(f"Baseline parameters: {baseline_params}")
        baseline_code = inject_parameters_into_template(template_code, baseline_params)
        parent_params = baseline_params.copy()
        
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
        if dataset == "gridworld":
            output_dir = f"generated_outputs/gridworld_ROTE_evo/run_{timestamp}/epoch_{participant_id}"
        else:
            output_dir = f"generated_outputs/choice13k_ROTE_evo/run_{timestamp}/participant_{participant_id}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ===== BASELINE EVALUATION =====
    print(f"\n{'='*80}")
    print(f"BASELINE EVALUATION: Evaluating seed program ({seed_program_path})")
    print(f"{'='*80}")
    
    if dataset == "gridworld":
        # Gridworld: strict mode - evaluate using parameters (even if empty)
        # Train: Evaluate on first 20 steps (matching ROTE's training/weighting phase)
        baseline_train_eval = evaluate_gridworld_program(
            baseline_code, data_path, num_blocks, num_walls, agent_id,
            num_datapoints=80, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
            evaluate_on_observed=True  # Match ROTE's training: evaluate on first 20 steps
        )
        # Test: Evaluate on future steps (matching ROTE's evaluation phase)
        baseline_test_eval = evaluate_gridworld_program(
            baseline_code, data_path, num_blocks, num_walls, agent_id,
            num_datapoints=20, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
            evaluate_on_observed=False  # Match ROTE's evaluation: evaluate on future steps
        )
        
        # Save baseline parameters (even if empty, for consistency)
        (output_path / "baseline_parameters.json").write_text(json.dumps(baseline_params, indent=2))
        
        baseline_results = {
            "parameters": baseline_params,
            "train_accuracy": baseline_train_eval['accuracy'],
            "test_accuracy": baseline_test_eval['accuracy'],
            "train_correct": baseline_train_eval['correct'],
            "train_total": baseline_train_eval['total'],
            "test_correct": baseline_test_eval['correct'],
            "test_total": baseline_test_eval['total'],
        }
    else:
        # Choice13k: use parameter-based evaluation
        # Save baseline parameters (only JSON, no .py files)
        (output_path / "baseline_parameters.json").write_text(json.dumps(baseline_params, indent=2))
        
        baseline_code = inject_parameters_into_template(template_code, baseline_params)
        baseline_fn = compile_program(baseline_code)
        
        if baseline_fn is None:
            print("ERROR: Failed to compile baseline program!")
            return None
        
        baseline_train_eval = evaluate_program(baseline_fn, train_trials, verbose=True, n_seeds=n_eval_seeds)
        baseline_test_eval = evaluate_program(baseline_fn, test_trials, verbose=True, n_seeds=n_eval_seeds)
        
        baseline_results = {
            "parameters": baseline_params,
            "train_accuracy": baseline_train_eval['accuracy'],
            "test_accuracy": baseline_test_eval['accuracy'],
            "train_correct": baseline_train_eval['correct'],
            "train_total": baseline_train_eval['total'],
            "test_correct": baseline_test_eval['correct'],
            "test_total": baseline_test_eval['total'],
        }
    
    print(f"\nBaseline Performance:")
    print(f"  Train accuracy: {baseline_results['train_accuracy']:.4f} ({baseline_results['train_correct']}/{baseline_results['train_total']})")
    print(f"  Test accuracy: {baseline_results['test_accuracy']:.4f} ({baseline_results['test_correct']}/{baseline_results['test_total']})")
    
    # Track all candidate results across iterations for finding overall best
    all_candidate_results = []  # List of dicts with iteration, candidate_idx, train_acc, test_acc
    
    # Log baseline to wandb at step=0
    if wandb is not None:
        wandb.log({
            f"p{participant_id}_train_accuracy": baseline_results['train_accuracy'],
            f"p{participant_id}_test_accuracy": baseline_results['test_accuracy'],
            f"p{participant_id}_is_baseline": 1,
        }, step=0)
    
    # Initialize best program tracking with baseline
    parent_params = baseline_params.copy()
    best_fitness = baseline_results['train_accuracy']
    
    # Track overall best across all iterations
    overall_best_train = {
        "train_accuracy": baseline_results['train_accuracy'],
        "test_accuracy": baseline_results['test_accuracy'],
        "program_id": "baseline"
    }
    overall_best_test = {
        "train_accuracy": baseline_results['train_accuracy'],
        "test_accuracy": baseline_results['test_accuracy'],
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
        
        # Generate candidates (parameter variants for both choice13k and gridworld - strict mode)
        if dataset == "gridworld":
            # Gridworld: strict mode - generate parameter variants (even if params are empty)
            if not parent_params:
                # No parameters: just use template as-is for all candidates (no evolution possible)
                print("Warning: No parameters in template. All candidates will be identical to baseline.")
                candidate_param_sets = [{}] * n_candidates_per_iteration
            else:
                # Generate parameter variants (strict mode)
                candidate_param_sets = generate_parameter_variants(
                    client=client,
                    model_name=model_name,
                    template_code=template_code,
                    parent_params=parent_params,
                    train_trials=[],  # Empty for gridworld (not used in parameter generation)
                    n_variants=n_candidates_per_iteration,
                    exploration_factor=exploration_factor,
                    parent_train_accuracy=best_fitness,
                )
        else:
            # Choice13k: generate parameter variants
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
        
        # Evaluate candidates (strict mode: parameter-based for both datasets)
        candidate_results = []
        for idx in range(n_candidates_per_iteration):
            params = candidate_param_sets[idx]
            # Save candidate parameters as JSON (only JSON, no .py files)
            (candidates_dir / f"candidate_{idx}_params.json").write_text(json.dumps(params, indent=2))
            
            # Generate full program code from template with these parameters
            candidate_code = inject_parameters_into_template(template_code, params)
            
            if dataset == "gridworld":
                # Gridworld: evaluate using gridworld evaluation function
                # Train: Evaluate on first 20 steps (matching ROTE's training/weighting phase)
                train_eval = evaluate_gridworld_program(
                    candidate_code, data_path, num_blocks, num_walls, agent_id,
                    num_datapoints=80, num_steps=20, n_seeds=n_eval_seeds,
                    evaluate_on_observed=True  # Match ROTE's training: evaluate on first 20 steps
                )
                # Test: Evaluate on future steps (matching ROTE's evaluation phase)
                test_eval = evaluate_gridworld_program(
                    candidate_code, data_path, num_blocks, num_walls, agent_id,
                    num_datapoints=20, num_steps=20, n_seeds=n_eval_seeds,
                    evaluate_on_observed=False  # Match ROTE's evaluation: evaluate on future steps
                )
                
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
                    "valid": train_eval["errors"] == 0,
                })
            else:
                # Choice13k: use parameter-based evaluation
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
                
                # Evaluate on train and test (with multiple seeds)
                train_eval = evaluate_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                test_eval = evaluate_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                
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
            if candidate_results[-1]["valid"]:
                all_candidate_results.append({
                    "iteration": iteration,
                    "candidate_idx": idx,
                    "train_acc": candidate_results[-1]["train_acc"],
                    "test_acc": candidate_results[-1]["test_acc"],
                })
        
        # Process results
        valid_results = [r for r in candidate_results if r["valid"]]
        if valid_results:
            # Sort by train accuracy
            valid_results.sort(key=lambda x: x["train_acc"], reverse=True)
            
            # Select best as parent for next iteration (strict mode: always use parameters)
            best_result = valid_results[0]
            parent_params = best_result["parameters"].copy()
            best_fitness = best_result["train_acc"]
            
            # Save best parameters (only JSON, no .py files) - for both datasets
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
                    "parameters": r.get("parameters", {}),
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
                # Log best parameters with participant prefix (for both datasets, strict mode)
                if parent_params:
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
        baseline_train_acc = baseline_results['train_accuracy']
        baseline_test_acc = baseline_results['test_accuracy']
        print(f"Baseline train accuracy: {baseline_train_acc:.4f}")
        print(f"Train accuracy improvement: {overall_best_train['train_accuracy'] - baseline_train_acc:.4f}")
        print(f"Test accuracy improvement: {overall_best_test['test_accuracy'] - baseline_test_acc:.4f}")
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
    
    parser = argparse.ArgumentParser(description="ROTE Evolution: Iterative evolution of Choice13k and Gridworld programs")
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
        help="Path to seed program (starting persona). If not set, auto-detects from persona_code_example/gridworld/ for gridworld. Default for choice13k: persona_code_example/vanilla.py",
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
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = timestamp
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
        if args.dataset == "gridworld":
            base_run_dir = f"generated_outputs/gridworld_ROTE_evo/run_{timestamp}"
        else:
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
    
    # Determine seed program path
    if args.seed_path is None:
        if args.dataset == "gridworld":
            # Auto-detect template program for gridworld
            # For sequential mode, we'll determine this per epoch
            # For now, we'll handle it in the epoch loop
            seed_program_path = None
        else:
            # Default for choice13k
            seed_program_path = "persona_code_example/vanilla.py"
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
    
    # Handle sequential mode for gridworld
    if args.dataset == "gridworld" and args.loop_mode == "sequential":
        # Import get_all_problem_configs from plot_and_eval
        import sys
        import importlib.util
        plot_eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_and_eval.py")
        spec = importlib.util.spec_from_file_location("plot_and_eval", plot_eval_path)
        plot_eval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot_eval)
        get_all_problem_configs = plot_eval.get_all_problem_configs
        
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
            
            # For simplicity, use first agent type
            agent_id = agent_indices[0] if agent_indices else 0
            
            print(f"\n{'='*80}")
            print(f"Processing epoch {epoch+1}/{epochs_to_process} - Problem: num_blocks={num_blocks}, num_walls={num_walls}, Agent: {agent_id}")
            print(f"{'='*80}")
            
            # Determine seed program path for this epoch
            if args.seed_path is None:
                # Auto-detect template program
                detected_seed_path = find_template_program_for_gridworld(num_blocks, num_walls, agent_id)
                if detected_seed_path is None:
                    print(f"Warning: Could not auto-detect template program for num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")
                    print(f"Expected location: persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/")
                    print("Skipping this epoch...")
                    continue
                epoch_seed_path = detected_seed_path
                print(f"Auto-detected seed program: {epoch_seed_path}")
            else:
                epoch_seed_path = args.seed_path
            
            # Construct output directory
            if base_run_dir is not None:
                participant_output_dir = os.path.join(base_run_dir, f"epoch_{epoch}")
            else:
                timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
                participant_output_dir = f"generated_outputs/gridworld_ROTE_evo/run_{timestamp}/epoch_{epoch}"
            
            participant_summary = run_evolution(
                seed_program_path=epoch_seed_path,
                participant_id=epoch,  # Use epoch as participant_id for gridworld
                n_iterations=args.n_iterations,
                n_candidates_per_iteration=args.n_candidates,
                model_name=args.model_name,
                client_kwargs=client_kwargs if client_kwargs else None,
                output_dir=participant_output_dir,
                wandb=wandb,
                n_eval_seeds=args.n_eval_seeds,
                dataset=args.dataset,
                data_path=args.data_path,
                num_blocks=num_blocks,
                num_walls=num_walls,
                agent_id=agent_id,
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
                        seed_program_path = "persona_code_example/vanilla.py"
                else:
                    seed_program_path = args.seed_path
                
                participant_summary = run_evolution(
                    seed_program_path=seed_program_path,
                    participant_id=participant_id,
                    n_iterations=args.n_iterations,
                    n_candidates_per_iteration=args.n_candidates,
                    model_name=args.model_name,
                    client_kwargs=client_kwargs if client_kwargs else None,
                    output_dir=participant_output_dir,
                    wandb=wandb,
                    n_eval_seeds=args.n_eval_seeds,
                    dataset=args.dataset,
                    data_path=args.data_path,
                    num_blocks=getattr(args, 'num_blocks', None),
                    num_walls=getattr(args, 'num_walls', None),
                    agent_id=getattr(args, 'agent_id', None),
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

