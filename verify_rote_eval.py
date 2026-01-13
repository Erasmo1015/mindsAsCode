#!/usr/bin/env python3
"""Script to verify ROTE evaluation by re-evaluating all 'good' programs from an epoch.
This uses the exact same evaluation logic as ROTE to ensure consistency."""

import json
import os
import sys
from pathlib import Path
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import plot_and_eval
sys.path.insert(0, str(Path(__file__).parent))

from plot_and_eval import make_dataloader
from agent import AgentExecutionFramework
from environment import AutomaticityEnv, State

def load_program_code(program_path):
    """Load program code from a file."""
    with open(program_path, 'r') as f:
        return f.read()

def evaluate_single_program(program_code, data_path, num_blocks, num_walls, agent_id, 
                            num_steps_to_predict=20, verbose=False):
    """Evaluate a single program using ROTE's exact evaluation logic.
    
    This replicates the exact logic from eval_fsm_bootstrap in plot_and_eval.py.
    
    Args:
        program_code: The program code to evaluate
        data_path: Path to data directory
        num_blocks: Number of blocks
        num_walls: Number of walls
        agent_id: Agent ID
        num_steps_to_predict: Number of future steps to predict (default: 20)
        verbose: Whether to print verbose output
    
    Returns:
        Dictionary with accuracy results
    """
    # Create a dummy args object matching ROTE's evaluation setup
    class EvalArgs:
        def __init__(self):
            self.data_path = data_path
            self.num_agents = 1
            self.num_agents_to_sample = 1  # Number of agents to sample
            self.num_datapoints_per_agent = 100
            self.num_steps = 20  # This is the data loading num_steps
            self.num_steps_to_predict = num_steps_to_predict
            self.group = False
            self.flip_quarter = True
            self.env_size = 10
            self.as_images = False
            self.multi_step_eval = True
            self.n_hypothesis = 1  # We're evaluating one program at a time
            self.plot_gifs = False
    
    args = EvalArgs()
    framework = AgentExecutionFramework()
    
    # Compile the agent
    try:
        agent = framework.compile_agent(program_code, num_agents=1, num_blocks=num_blocks)
    except Exception as e:
        if verbose:
            print(f"  Compilation error: {e}")
        return {"accuracy": 0.0, "total": 0, "correct": 0, "errors": 1}
    
    # Create dataloader exactly like ROTE does
    dataloader = make_dataloader(
        args,
        num_agents_to_sample=1,
        num_datapoints_per_agent_to_sample=100,  # Use all datapoints
        training=False,
        epoch=0,
        num_blocks=num_blocks,
        num_walls=num_walls,
        agent_indices=[agent_id]
    )
    
    # Replicate ROTE's exact evaluation logic from eval_fsm_bootstrap
    num_future_steps = args.num_steps_to_predict
    results = {'correct': 0, 'total': 0, 'first_step_correct': 0, 'first_step_total': 0}
    
    datapoint = next(dataloader)
    
    # ROTE evaluates on the last datapoint per agent (line 1234)
    for a_idx in range(args.num_agents_to_sample):
        data_sample = jax.tree.map(lambda x: x[a_idx, -1, :20+num_future_steps], datapoint)
        
        initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
        initial_actions_traj = jax.tree.map(lambda x: x[:20], data_sample['actions'])
        
        gt_future_actions = data_sample['actions'][19:]  # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)
        
        num_env_agents = 4 if args.group else 1
        
        # Infer num_blocks and num_walls from state (same as ROTE)
        state_at_t14 = jax.tree.map(lambda x: x[19], initial_states_traj)
        inferred_num_blocks = state_at_t14['block_locations'].shape[0] if len(state_at_t14['block_locations'].shape) > 1 else state_at_t14['block_locations'].shape[0]
        if isinstance(state_at_t14['block_locations'], (np.ndarray, jnp.ndarray)) and state_at_t14['block_locations'].ndim == 1 and inferred_num_blocks > 0:
            inferred_num_blocks = 1
        elif isinstance(state_at_t14['block_locations'], (np.ndarray, jnp.ndarray)) and state_at_t14['block_locations'].ndim == 2:
            inferred_num_blocks = state_at_t14['block_locations'].shape[0]
        else:
            inferred_num_blocks = 0
        
        inferred_num_walls = state_at_t14['wall_locations'].shape[0]
        
        env = AutomaticityEnv(num_agents=num_env_agents, size=args.env_size, max_steps=num_future_steps + 5,
                            num_blocks=inferred_num_blocks, num_walls=inferred_num_walls)
        
        # Initialize simulation state (same as ROTE)
        current_sim_state_pytree = state_at_t14
        current_sim_state_pytree = State(
            wall_locations=current_sim_state_pytree['wall_locations'],
            agent_locations=current_sim_state_pytree['agent_locations'],
            block_locations=current_sim_state_pytree['block_locations'],
            agent_inventory=current_sim_state_pytree['agent_inventory'],
            agent_inventory_colors=current_sim_state_pytree['agent_inventory_colors'],
            block_colors=current_sim_state_pytree['block_colors'],
            time=current_sim_state_pytree['time'],
            terminal=False,
            agent_id=-1
        )
        current_obs = env.get_observation(current_sim_state_pytree)[0]
        
        # Simulate future steps (exact ROTE logic)
        step_correct = 0
        step_total = 0
        
        for step_idx in range(num_future_steps):
            if step_idx >= gt_future_actions.shape[0]:
                break
            
            if step_idx < 10:
                step_total += num_env_agents
            else:
                step_total += 1
            
            # Track first step separately
            if step_idx == 0:
                results['first_step_total'] += num_env_agents
            
            # Get observation from ground truth data (same as ROTE line 1530)
            current_obs = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])
            
            # Get prediction from agent (same as ROTE line 1449)
            try:
                current_obs['agent_id'] = 0
                predicted_action = framework.execute_agent(agent, current_obs)
                
                # Convert action (same as ROTE lines 1450-1475)
                action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
                action_space_2 = ["stay", "right", "left", "down", "up", "interact"]
                
                if type(predicted_action) == tuple:
                    predicted_action = list(predicted_action)
                elif type(predicted_action) == str:
                    predicted_action = predicted_action.lower()
                    predicted_action = action_space_2.index(predicted_action)
                else:
                    predicted_action = int(predicted_action)
                
                if predicted_action in action_space:
                    predicted_action = action_space.index(predicted_action)
                elif predicted_action in action_space_2:
                    predicted_action = action_space_2.index(predicted_action)
                
                # Compare with ground truth (same as ROTE line 1502, 1506)
                gt_action_this_step = gt_future_actions[step_idx]  # (num_env_agents,) or (1,)
                gt_action = gt_action_this_step[0] if len(gt_action_this_step) > 0 else int(gt_action_this_step)
                gt_action = int(gt_action)
                
                # Check if correct (same as ROTE line 1511)
                if predicted_action == gt_action:
                    step_correct += 1
                    if step_idx == 0:
                        results['first_step_correct'] += 1
                
            except Exception as e:
                if verbose and step_idx == 0:
                    print(f"  Step {step_idx} error: {e}")
                continue
        
        results['correct'] += step_correct
        results['total'] += step_total
    
    # Calculate accuracy
    accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0.0
    first_step_accuracy = results['first_step_correct'] / results['first_step_total'] if results['first_step_total'] > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "total": results['total'],
        "correct": results['correct'],
        "first_step_accuracy": first_step_accuracy,
        "first_step_total": results['first_step_total'],
        "first_step_correct": results['first_step_correct']
    }

def main():
    parser = argparse.ArgumentParser(description="Verify ROTE evaluation by re-evaluating programs")
    parser.add_argument("--json_file", type=str, required=True,
                       help="Path to epoch_X_agent_types.json file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Base output directory (e.g., generated_outputs/gridworld/run_260107_010024)")
    parser.add_argument("--data_path", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
    args = parser.parse_args()
    
    # Load JSON file
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)
    
    # Extract epoch and agent info
    epoch = json_data['epoch']
    agent_types = json_data['agent_types']
    
    # Get problem config from the output directory structure or JSON
    # For now, we'll need to infer from the directory or pass as args
    # Let's check if we can infer from the JSON structure
    
    print("=" * 80)
    print(f"Verifying ROTE Evaluation for Epoch {epoch}")
    print("=" * 80)
    
    results = {}
    
    for agent_id_str, agent_data in agent_types.items():
        agent_id = agent_data['agent_id']
        program_paths = agent_data['program_paths']
        hypothesis_ids = agent_data['hypotheses']
        
        print(f"\nAgent ID: {agent_id}")
        print(f"Total hypotheses: {len(hypothesis_ids)}")
        
        # Extract problem config - we need to infer this
        # For gridworld, we can check the output directory or use defaults
        # Let's use the same defaults as the script: num_blocks=3, num_walls=1
        num_blocks = 3
        num_walls = 1
        
        # Evaluate each "good" program
        good_programs = []
        for hyp_id, prog_path in zip(hypothesis_ids, program_paths):
            if "good" in prog_path:
                full_path = os.path.join(args.output_dir, prog_path)
                if os.path.exists(full_path):
                    good_programs.append((hyp_id, full_path, prog_path))
        
        print(f"\nFound {len(good_programs)} 'good' programs to evaluate")
        print("-" * 80)
        
        for hyp_id, full_path, rel_path in good_programs:
            print(f"\nEvaluating Hypothesis {hyp_id}: {rel_path}")
            
            # Load program code
            try:
                program_code = load_program_code(full_path)
            except Exception as e:
                print(f"  Error loading program: {e}")
                results[hyp_id] = {"accuracy": 0.0, "error": str(e)}
                continue
            
            # Evaluate using ROTE's exact logic
            eval_result = evaluate_single_program(
                program_code,
                args.data_path,
                num_blocks,
                num_walls,
                agent_id,
                num_steps_to_predict=20,
                verbose=args.verbose
            )
            
            results[hyp_id] = eval_result
            print(f"  Accuracy: {eval_result['accuracy']:.4f} ({eval_result['correct']}/{eval_result['total']})")
            if 'first_step_accuracy' in eval_result:
                print(f"  First step accuracy: {eval_result['first_step_accuracy']:.4f}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Hypothesis':<12} {'Accuracy':<12} {'Correct/Total':<15} {'First Step Acc':<15}")
        print("-" * 80)
        
        for hyp_id in sorted(results.keys()):
            r = results[hyp_id]
            acc_str = f"{r['accuracy']:.4f}" if 'accuracy' in r else "N/A"
            correct_total = f"{r.get('correct', 0)}/{r.get('total', 0)}"
            first_step = f"{r.get('first_step_accuracy', 0):.4f}" if 'first_step_accuracy' in r else "N/A"
            print(f"{hyp_id:<12} {acc_str:<12} {correct_total:<15} {first_step:<15}")
        
        # Compare with JSON weights (weights are normalized log probs, not accuracies)
        print("\n" + "=" * 80)
        print("COMPARISON WITH JSON WEIGHTS")
        print("=" * 80)
        print("Note: JSON weights are model confidence (normalized log probabilities), not accuracies")
        print("Best program in JSON: Hypothesis", agent_data['best_program']['hypothesis_id'])
        print(f"  Weight: {agent_data['best_program']['weight']:.4f}")
        print(f"  Ensemble accuracy: {agent_data.get('ensemble_accuracy', 'N/A')}")

if __name__ == "__main__":
    main()

