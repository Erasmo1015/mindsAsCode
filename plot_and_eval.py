# Set JAX memory allocation to grow as needed
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Set PyTorch to use expandable segments to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import torch
# torch.multiprocessing.set_start_method('spawn')
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse

import msgpack
import flax
import jax.numpy as jnp
import jax
import numpy as np
import random
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from environment import state_to_image_jit, AutomaticityEnv, State
import optax
from flax.training import train_state
from flax import struct
import pandas as pd
import imageio.v2 as imageio
from baselines.BC import BCNet
from agent import AgentExecutionFramework
from data_modules.choice13k import get_choice13k_experiments
from openai import OpenAI

import time
from io import BytesIO
import json
from pathlib import Path

def run_eval_with_seeds(eval_fn, args, n_seeds, *eval_args, **eval_kwargs):
    """Run evaluation function multiple times and average numeric results.
    
    Args:
        eval_fn: The evaluation function to call
        args: Arguments object (needed for dataloader recreation)
        n_seeds: Number of times to run evaluation
        *eval_args: Positional arguments to pass to eval_fn
        **eval_kwargs: Keyword arguments to pass to eval_fn
    
    Returns:
        Averaged results (same structure as eval_fn return, but numeric values averaged)
    """
    import numpy as np
    
    if n_seeds == 1:
        return eval_fn(*eval_args, **eval_kwargs)
    
    # Collect results from all seeds
    all_results = []
    for seed in range(n_seeds):
        result = eval_fn(*eval_args, **eval_kwargs)
        all_results.append(result)
    
    # Average numeric results
    if isinstance(all_results[0], dict):
        # Handle dictionary results (e.g., accuracies_dict)
        averaged = {}
        for key in all_results[0].keys():
            if isinstance(all_results[0][key], dict):
                # Nested dictionary (e.g., accuracies_dict[n_hyp])
                averaged[key] = {}
                for sub_key in all_results[0][key].keys():
                    values = [r[key][sub_key] for r in all_results if key in r and sub_key in r[key]]
                    if values and isinstance(values[0], (int, float)):
                        averaged[key][sub_key] = np.mean(values)
                    else:
                        averaged[key][sub_key] = all_results[-1][key][sub_key]  # Use last result for non-numeric
            elif isinstance(all_results[0][key], (int, float)):
                values = [r[key] for r in all_results if key in r]
                averaged[key] = np.mean(values) if values else all_results[-1][key]
            else:
                # Non-numeric: use last result
                averaged[key] = all_results[-1][key]
        return averaged
    elif isinstance(all_results[0], tuple):
        # Handle tuple results
        num_elements = len(all_results[0])
        averaged = []
        for i in range(num_elements):
            element = all_results[0][i]
            if isinstance(element, dict):
                # Dictionary element (e.g., accuracies_dict)
                avg_dict = {}
                for key in element.keys():
                    if isinstance(element[key], dict):
                        avg_dict[key] = {}
                        for sub_key in element[key].keys():
                            values = [r[i][key][sub_key] for r in all_results if i < len(r) and key in r[i] and sub_key in r[i][key]]
                            if values and isinstance(values[0], (int, float)):
                                avg_dict[key][sub_key] = np.mean(values)
                            else:
                                avg_dict[key][sub_key] = all_results[-1][i][key][sub_key]
                    elif isinstance(element[key], (int, float)):
                        values = [r[i][key] for r in all_results if i < len(r) and key in r[i]]
                        avg_dict[key] = np.mean(values) if values else all_results[-1][i][key]
                    else:
                        avg_dict[key] = all_results[-1][i][key]
                averaged.append(avg_dict)
            elif isinstance(element, (int, float)):
                values = [r[i] for r in all_results if i < len(r)]
                averaged.append(np.mean(values) if values else all_results[-1][i])
            else:
                # Non-numeric: use last result (e.g., agent_id, env)
                averaged.append(all_results[-1][i])
        return tuple(averaged)
    else:
        # Single value result
        if isinstance(all_results[0], (int, float)):
            return np.mean(all_results)
        else:
            return all_results[-1]  # Use last result for non-numeric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline models for automaticity.")
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="ROTE",
        help="Baseline model to train."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the dataset folders."
    )
    parser.add_argument(
        "--group",
        type=bool,
        default=False,
        help="Whether to use joint-planner data."
    )
    parser.add_argument('--num_agents_to_sample', type=int, default=1, help='Number of agents to sample from the dataset.')
    parser.add_argument('--num_datapoints_per_agent_to_sample', type=int, default=3, help='Number of datapoints per agent to sample from the dataset.')
    parser.add_argument('--num_agents', type=int, default=1, help='Number of agents in the dataset.')
    parser.add_argument('--num_datapoints_per_agent', type=int, default=5, help='Number of datapoints per agent in the dataset.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of steps in the dataset.')
    parser.add_argument('--env_size', type=int, default=7, help='Size of the environment.')
    # parser.add_argument('--num_blocks', type=int, default=10, help='Number of blocks in the dataset.')
    # parser.add_argument('--num_walls', type=int, default=10, help='Number of walls in the dataset.')

    parser.add_argument('--as_images', type=bool, default=False, help='Whether to load the data as images.')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save the model.')
    parser.add_argument('--seed', type=int, default=12, help='Random seed.')
    parser.add_argument('--n_hypothesis', type=int, default=30, help='Number of hypothesis for thought trace.')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", help='Name of the model to use.')  # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct or meta-llama/Llama-3.1-8B-Instruct or deepseek-ai/DeepSeek-V2-Lite
    parser.add_argument('--mode', type=str, default="default", choices=["default", "local"], help='LLM mode: default uses in-process models; local routes to an external vLLM server.')
    parser.add_argument('--llm_server_url', type=str, default=os.getenv("VLLM_LOCAL_URL", "http://localhost:8000/v1"), help='Base URL for local vLLM server when --mode local.')
    parser.add_argument('--llm_api_key', type=str, default=os.getenv("VLLM_LOCAL_API_KEY", "EMPTY"), help='API key for local vLLM server when --mode local.')
    parser.add_argument('--dataset', type=str, default="gridworld", choices=["gridworld", "choice13k"], help='Dataset to use. Default gridworld; choice13k mirrors llm_evo_cog.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of tensor parallel size.')
    parser.add_argument('--dtype', type=str, default="float16", help='Data type.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization.')
    parser.add_argument('--no_log', action='store_true', help='Disable wandb logging. Default is enabled.')
    parser.add_argument('--overfit', type=bool, default=False, help='Whether to overfit on a single environment.')
    parser.add_argument('--bootstrap', action='store_true', help='Whether to use bootstrapping for hypothesis evaluation')
    parser.add_argument('--two_stage', action='store_true', help='Whether to use two-stage approach for ROTE reasoning')
    parser.add_argument('--structured', type=str, default="False", choices=["False", "p1", "p2"], 
                        help='Structured prompting type for ROTE reasoning: False, p1, or p2')
    parser.add_argument('--rejuvenation', action='store_true', help='Use rejuvenation for ROTE model')
    parser.add_argument('--plot_gifs', action='store_true', help='Plot gifs for ROTE model')
    parser.add_argument('--rejuvenation_threshold', type=float, default=1, help='Threshold for rejuvenation')
    parser.add_argument('--max_rejuvenation_attempts', type=int, default=2, help='Maximum number of rejuvenation attempts')
    parser.add_argument('--top_k', type=int, default=0, help='If > 0, only average over the top k most likely hypotheses')
    parser.add_argument('--multi_step_eval', type=bool, default=True, help='Perform multi-step evaluation for ROTE')
    parser.add_argument('--num_steps_to_predict', type=int, default=20, help='Number of future steps to predict in multi-step eval')
    parser.add_argument('--flip_quarter', type=bool, default=True, help='reset the environment after 30 steps')
    parser.add_argument('--human_data', type=bool, default=False, help='Use human data')
    parser.add_argument('--participant_id', type=int, default=None, help='Specific participant ID to evaluate (0-indexed). If None, evaluates all participants from 0 to num_agents_to_sample-1.')
    parser.add_argument('--n_eval_seeds', type=int, default=1, help='Number of evaluation runs per epoch (averaged for final accuracy). Default: 3')
    parser.add_argument('--prompt_mode', type=str, default="non_strict", choices=["strict", "non_strict"], help='Prompt mode for choice13k: "non_strict" (default) uses standard prompts, "strict" uses parametrized program prompts.')
    parser.add_argument('--loop_mode', type=str, default="random", choices=["random", "sequential"], help='Evaluation loop mode: "random" (default) uses random problem and agent sampling per epoch; "sequential" evaluates all problem configs and agent types systematically.')
    args = parser.parse_args()
    
    # Check if the selected baseline model is implemented
    if args.baseline_model not in ['BC', 'AutoToM', 'ROTE', 'NLLM', 'Oracle']:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    if args.baseline_model == 'AutoToM':
        os.environ['CURRENT_MODEL_NAME'] = args.model_name
    
    return args

def initialize_environment(num_agents: int, num_blocks: int, num_walls: int, size: int = 10, max_steps: int = 100):
    env = AutomaticityEnv(num_agents=num_agents, size=size, max_steps=max_steps, num_blocks=num_blocks, num_walls=num_walls)
    return env

def get_all_problem_configs():
    """Generate all (num_blocks, num_walls) combinations in sequential order."""
    configs = []
    for num_blocks in range(3, 8):  # 3 to 7
        for num_walls in range(1, 5):  # 1 to 4
            configs.append((num_blocks, num_walls))
    return configs


def make_dataloader_human(args, num_agents_to_sample: int = 2, num_datapoints_per_agent_to_sample: int = 20, overfit: bool = False, training: bool = False, epoch: int = 0):
    """Load data from the dataset folders."""
    data_path = args.data_path
    as_images = args.as_images

    data_folder = f"{data_path}/human_data_fix"
    from human_dataloader import load_and_stack_human_gameplay_data
    human_dataloader = load_and_stack_human_gameplay_data(data_folder)
    human_data = None
    try:
        for j in range(epoch+1):
            human_data, human_actions, agent_id, filename, task = next(human_dataloader)
        current_data_loaded = epoch
    except StopIteration:
        print("Reached end of human data, exiting successfully")
        exit(0)

    i = epoch
    while True:
        '''
        actions is (num_agents to sample, num_datapoints per agent, traj_length, 1)
        agent ids is (num_agents_to_sample, num_datapoints per_agent, traj_length) where each value for agent is the task number
        states is (num_agents_to_sample, num_datapoints per agent, traj_length, *state_shape)
        '''
        human_states = human_data
        # Add two dimensions of length 1 to each leaf in human_states
        human_states = jax.tree.map(
            lambda x: x[None, None, :, ...] if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
            human_states
        )
        human_actions = human_actions[None, None, :, None]
        human_agent_id = jnp.array([agent_id])[None, None, :] # (1, 1, 1)
        # Repeat agent_id to match length of human_actions
        human_agent_id = jnp.repeat(human_agent_id, human_actions.shape[2], axis=2)  # (1, 1, traj_length)
        sampled_states = human_states
        sampled_data = {
            'states': sampled_states,
            'actions': human_actions,
            'agent_ids': human_agent_id,
            # 'filename': filename,
            # 'task': task
        }

        # Reshape all arrays to (-1, *original_shape[3:])
        reshaped_state = jax.tree.map(
            lambda x: jnp.array(x).reshape(-1, *x.shape[3:]) if (isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray)) else x,
            sampled_states
        )
        # Process images in smaller batches to reduce memory usage
        batch_size = 100  # Process images in smaller batches
        num_samples = reshaped_state['agent_locations'].shape[0]
        all_images = []


        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = jnp.arange(start_idx, end_idx)
            
            batch_state = jax.tree.map(lambda x: x[batch_indices] if isinstance(x, (jnp.ndarray, np.ndarray)) else x, reshaped_state)
            
            def convert_to_image(index, stacked_state):
                indexed_state = jax.tree.map(lambda x: x[index], stacked_state)
                img_size = args.env_size * 7
                tile_size = 7
                grid_size = args.env_size
                img_gen_fn = jax.jit(state_to_image_jit, static_argnums=(1, 2, 3))
                render_fn = lambda x: img_gen_fn(x, img_size, grid_size, tile_size)
                return render_fn(indexed_state)
            
            batch_images = jax.vmap(convert_to_image, in_axes=(0, None))(jnp.arange(len(batch_indices)), batch_state)
            all_images.append(batch_images)
        
        # Concatenate all batches
        stacked_images = jnp.concatenate(all_images, axis=0)

        del reshaped_state, all_images

        if as_images:
            sampled_data['states'] = stacked_images.reshape(1, 1, -1, *stacked_images.shape[1:])
            sampled_data['images'] = stacked_images.reshape(1, 1, -1, *stacked_images.shape[1:])
        else:
            sampled_data['states'] = sampled_states
            sampled_data['images'] = stacked_images.reshape(1, 1, -1, *stacked_images.shape[1:])

        yield sampled_data
        del sampled_data



        
        try:
            human_data, human_actions, agent_id, filename, task = next(human_dataloader)
            current_data_loaded += 1
        except StopIteration:
            print("Reached end of human data, exiting successfully")
            exit(0)

def make_dataloader(args, num_agents_to_sample: int = 2, num_datapoints_per_agent_to_sample: int = 20, overfit: bool = False, training: bool = False, epoch: int = 0, num_blocks: int = None, num_walls: int = None, agent_indices: list = None):
    """Load data from the dataset folders.
    
    Args:
        num_blocks: Optional. If provided, use this specific num_blocks (for sequential mode).
        num_walls: Optional. If provided, use this specific num_walls (for sequential mode).
        agent_indices: Optional. If provided, use these specific agent indices (for sequential mode).
    """
    data_path = args.data_path
    as_images = args.as_images

    i = epoch
    while True:
        if not overfit:
            if num_blocks is not None and num_walls is not None:
                # Sequential mode: use provided values
                pass  # num_blocks and num_walls already set
            else:
                # Random mode: choose randomly
                i += 1
                num_blocks = random.choice(list(range(3,8,1)))
                num_walls = random.choice(list(range(1, 5, 1)))
        else:
            num_blocks = 4
            num_walls = 1

        data_folder = f"{data_path}/num_blocks{num_blocks}/num_walls{num_walls}"
        extension = "_group" if args.group else ""
        if args.flip_quarter:
            extension += "_flip_quarter"
        data_file = f"{data_folder}/gt_fsm{extension}_traj_data_{args.num_agents}agents.msgpack"
        # print(f"Loading data from {data_file}")

        # Load data in chunks to reduce memory usage
        with open(data_file, "rb") as f:
            serialized_data = f.read()
            
        # Create target structure with correct shapes
        action_target = jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, 1))
        agent_id_target = jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps))
        state_target = {
            'agent_id': agent_id_target,
            'agent_locations': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, 1, 2)),
            'agent_inventory': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, 1)),
            'agent_inventory_colors': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, 1, 3)),
            'block_colors': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, num_blocks, 3)),
            'block_locations': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, num_blocks, 2)),
            'time': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps)),
            'terminal': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps)),
            'wall_locations': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, num_walls + 2 * (args.env_size * 2 - 1) + 2, 2)),
        }
        target = {
            'states': state_target,
            'actions': action_target,
            'agent_ids': agent_id_target,
        }
            
        loaded_data = flax.serialization.from_bytes(target, serialized_data)
        del serialized_data  # Free memory immediately

        # Validate shapes match target
        for key in target['states'].keys():
            if loaded_data['states'][key].shape != target['states'][key].shape:
                continue
        # print(f"Loaded data from {data_file}")
                
        # Sample only what we need
        if agent_indices is None:
            # Not provided, use random sampling
            if overfit:
                agent_indices = jax.random.randint(jax.random.PRNGKey(i), (num_agents_to_sample,), minval=1, maxval=2)
            else:
                agent_indices = jax.random.randint(jax.random.PRNGKey(i), (num_agents_to_sample,), minval=0, maxval=loaded_data['states']['agent_locations'].shape[0])
        else:
            # Sequential mode: use provided agent_indices, convert to jax array if needed
            if isinstance(agent_indices, list):
                agent_indices = jnp.array(agent_indices)
        
        i += 1
        if training:  # for creating held out set
            batch_floor = 0
            batch_limit = int(loaded_data['states']['agent_locations'].shape[1] * 0.8)
        else:
            batch_floor = int(loaded_data['states']['agent_locations'].shape[1] * 0.8)
            batch_limit = loaded_data['states']['agent_locations'].shape[1]


        batch_indices = jax.random.randint(jax.random.PRNGKey(i), (num_datapoints_per_agent_to_sample,), minval=batch_floor, maxval=batch_limit)

        i += 1
        
        # Process data in smaller chunks to reduce memory usage
        sampled_data = {}
        sampled_data['actions'] = loaded_data['actions'][agent_indices][:, batch_indices][:, :, :]
        sampled_data['agent_ids'] = loaded_data['agent_ids'][agent_indices][:, batch_indices][:, :, :]
        
        # Process states separately to reduce peak memory usage
        sampled_states = {}
        for key in loaded_data['states']:
            sampled_states[key] = loaded_data['states'][key][agent_indices][:, batch_indices][:, :, :]

        # Free memory
        del loaded_data
        
        # Reshape all arrays to (-1, *original_shape[3:])
        reshaped_state = jax.tree.map(
            lambda x: jnp.array(x).reshape(-1, *x.shape[3:]) if (isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray)) else x,
            sampled_states
        )
        
        # Process images in smaller batches to reduce memory usage
        batch_size = 100  # Process images in smaller batches
        num_samples = reshaped_state['agent_locations'].shape[0]
        all_images = []
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = jnp.arange(start_idx, end_idx)
            
            batch_state = jax.tree.map(lambda x: x[batch_indices] if isinstance(x, (jnp.ndarray, np.ndarray)) else x, reshaped_state)
            
            def convert_to_image(index, stacked_state):
                indexed_state = jax.tree.map(lambda x: x[index], stacked_state)
                img_size = args.env_size * 7
                tile_size = 7
                grid_size = args.env_size
                img_gen_fn = jax.jit(state_to_image_jit, static_argnums=(1, 2, 3))
                render_fn = lambda x: img_gen_fn(x, img_size, grid_size, tile_size)
                return render_fn(indexed_state)
            
            batch_images = jax.vmap(convert_to_image, in_axes=(0, None))(jnp.arange(len(batch_indices)), batch_state)
            all_images.append(batch_images)
        
        # Concatenate all batches
        stacked_images = jnp.concatenate(all_images, axis=0)
        
        # Free memory
        del reshaped_state, all_images
        
        if as_images:
            sampled_data['states'] = stacked_images.reshape(num_agents_to_sample, num_datapoints_per_agent_to_sample, args.num_steps, *stacked_images.shape[1:])
            sampled_data['images'] = stacked_images.reshape(num_agents_to_sample, num_datapoints_per_agent_to_sample, -1, *stacked_images.shape[1:])
        else:
            sampled_data['states'] = sampled_states
            sampled_data['images'] = stacked_images.reshape(num_agents_to_sample, num_datapoints_per_agent_to_sample, -1, *stacked_images.shape[1:])

        yield sampled_data
        del sampled_data

def eval_autoToM(args, dataloader, model, episode_id: int = 0):
    if args.multi_step_eval:
        # --- Multi-Step Evaluation Logic ---
        num_future_steps = args.num_steps_to_predict
        num_correct = 0
        num_total = 0
        total_prediction_time = 0
        num_predictions = 0
        first_step_correct = 0
        first_step_total = 0
        correct_after_flip = 0
        total_after_flip = 0
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)
        
        # Initialize environment (parameters will be set per datapoint)
        env_size = 10
        env_max_steps = num_future_steps + 5  # Sufficiently large

        for a_idx in tqdm(range(args.num_agents_to_sample), desc="Multi-step Eval Samples"):
            data_sample = jax.tree.map(lambda x: x[a_idx, -1, :20+num_future_steps], datapoint)
            
            initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
            initial_actions_traj = jax.tree.map(lambda x: x[:20], data_sample['actions'])
            gt_agent_script_id = int(initial_states_traj['agent_id'][0])

            avg_matching_states, mean_equal_actions = get_matching_states_and_actions(initial_states_traj, initial_actions_traj)
            
            gt_future_actions = data_sample['actions'][19:]  # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)

            num_env_agents = 4 if args.group else 1
            # Infer num_blocks and num_walls from the state
            state_at_t19 = jax.tree.map(lambda x: x[19], initial_states_traj)
            
            # Handle different possible shapes for block_locations
            if isinstance(state_at_t19['block_locations'], (np.ndarray, jnp.ndarray)):
                if state_at_t19['block_locations'].ndim == 1:
                    num_blocks = 1
                elif state_at_t19['block_locations'].ndim == 2:
                    num_blocks = state_at_t19['block_locations'].shape[0]
                else:
                    num_blocks = 0
            else:
                num_blocks = 0
                
            num_walls = state_at_t19['wall_locations'].shape[0]
            
            env = AutomaticityEnv(num_agents=num_env_agents, size=env_size, max_steps=env_max_steps, 
                                  num_blocks=num_blocks, num_walls=num_walls)
            
            # Initialize simulation state from the last observed state
            current_sim_state_pytree = state_at_t19
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
            
            # Simulate future steps
            for step_idx in range(num_future_steps):
                if step_idx >= gt_future_actions.shape[0]:
                    break
                
                step_prediction_start = time.time()
                # For each agent in the environment
                for agent_id in range(num_env_agents):
                    if step_idx < 10:
                        num_total += 1
                        if step_idx == 0:
                            first_step_total += 1
                    else:
                        total_after_flip += 1
                    
                    # Create a trajectory for this specific prediction
                    if step_idx == 0:
                        # First prediction uses the initial trajectory
                        pred_states = initial_states_traj
                        pred_actions = initial_actions_traj
                    else:
                        # Subsequent predictions use updated trajectory with previous predictions
                        # Process each updated state individually to get observations
                        updated_obs = []
                        for t in range(step_idx):
                            # obs_t = env.get_observation(updated_states[t])[0]
                            obs_t = updated_states[t]
                            updated_obs.append(obs_t)
                        
                        # Stack observations into a sequence
                        stacked_obs = jax.tree.map(lambda *xs: jnp.stack(xs), *updated_obs)
                        
                        # Concatenate with initial trajectory
                        pred_states = jax.tree.map(lambda x, y: jnp.concatenate([x[:20], y], axis=0), 
                                                  initial_states_traj, stacked_obs)
                        pred_actions = jnp.concatenate([initial_actions_traj, 
                                                       jnp.array(action_history[:step_idx])], axis=0)
                    
                    # Get model prediction for this step and agent
                    try:
                        predicted_action, predicted_probs = model.predict_action(pred_states, pred_actions, 
                                                                               agent_id=agent_id, timestep=19+step_idx)
                        
                        # Compare with ground truth
                        gt_action = gt_future_actions[step_idx][agent_id] if args.group else gt_future_actions[step_idx][0]
                        if predicted_action == gt_action:
                            if step_idx < 10:
                                num_correct += 1
                                if step_idx == 0:
                                    first_step_correct += 1
                            else:
                                correct_after_flip += 1
                    except Exception as e:
                        print(f"Error predicting action for agent {agent_id}, step {step_idx}: {e}")
                        continue
                
                step_prediction_time = time.time() - step_prediction_start
                total_prediction_time += step_prediction_time
                num_predictions += 1
                
                # Collect actions for all agents to update environment
                action_to_take_in_env_list = []
                for agent_id in range(num_env_agents):
                    try:
                        predicted_action, _ = model.predict_action(pred_states, pred_actions, 
                                                                 agent_id=agent_id, timestep=19+step_idx)
                        action_to_take_in_env_list.append(predicted_action)
                    except:
                        # If prediction fails, use ground truth (to keep simulation going)
                        gt_action = gt_future_actions[step_idx][agent_id] if args.group else gt_future_actions[step_idx][0]
                        action_to_take_in_env_list.append(gt_action)
                
                # Store actions for next iteration
                if step_idx == 0:
                    action_history = [action_to_take_in_env_list]
                    updated_states = []
                else:
                    action_history.append(action_to_take_in_env_list)
                
                # Simulate environment step
                current_sim_state_pytree = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])

                updated_states.append(jax.tree.map(lambda x: x, current_sim_state_pytree))
        
        accuracy = num_correct / num_total if num_total > 0 else 0
        avg_prediction_time = total_prediction_time / num_predictions if num_predictions > 0 else 0
        first_step_accuracy = first_step_correct / first_step_total if first_step_total > 0 else 0
        accuracy_after_flip = correct_after_flip / total_after_flip if total_after_flip > 0 else 0
        print(f"AutoToM Multi-Step Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
        print(f"First Step Accuracy: {first_step_accuracy:.4f} ({first_step_correct}/{first_step_total})")
        print(f"Accuracy After Flip: {accuracy_after_flip:.4f} ({correct_after_flip}/{total_after_flip})")
        print(f"Average prediction time per step: {avg_prediction_time:.4f} seconds")
        return accuracy, avg_prediction_time, first_step_accuracy, gt_agent_script_id, accuracy_after_flip, avg_matching_states, mean_equal_actions
    else:
        # Original single-step evaluation
        num_correct = 0
        num_total = 0
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)

        for a in tqdm(range(args.num_agents_to_sample)):
            data = jax.tree.map(lambda x: x[a, -1, :15], datapoint)
            states = data['states']
            actions = data['actions']  # (20, n)  # 20 timesteps, n agent actions
            agent_ids = data['agent_ids']
            for agent_id in tqdm(range(actions.shape[1])):
                tries = 0
                gt_action = actions[-1, agent_id]
                num_total += 1
                try:
                    predicted_final_action, predicted_probs = model.predict_action(states, actions, agent_id=agent_id, episode_id=episode_id)
                    final_action_prob = predicted_probs[gt_action]
                    if final_action_prob >= np.max(predicted_probs):
                        num_correct += 1
                except Exception as e:
                    continue
        
        print(f"Accuracy: {num_correct / num_total}")
        return num_correct / num_total, 0.0  # Return 0.0 as placeholder for avg_prediction_time


def get_matching_states_and_actions(states, actions):
    # Get all state indices
    state_indices = list(np.arange(states['agent_locations'].shape[0]))
    
    # Track groups of matching states
    matching_groups = []
    used_indices = set()
    
    # Compare each state with every other state
    for i in state_indices:
        if i in used_indices:
            continue
            
        current_group = {i}
        used_indices.add(i)
        
        # Compare with remaining states
        for j in state_indices:
            if j in used_indices:
                continue
                
            # Check if all state components match using jnp.all
            states_match = True
            for key in states.keys():
                if key in ['time', 'terminal']:
                    continue
                    
                if not jnp.all(states[key][i] == states[key][j]):
                    states_match = False
                    break
                    
            if states_match:
                current_group.add(j)
                used_indices.add(j)
                
        matching_groups.append(len(current_group))
        
    # Calculate average size of matching groups
    avg_matching_states = sum(matching_groups) / len(matching_groups) if matching_groups else 1
    print(f"Average size of matching state groups: {avg_matching_states:.2f}")

    next_time_actions = actions[1:]
    old_time_actions = actions[:-1]
    action_equal = jnp.all(next_time_actions == old_time_actions, axis=1)
    mean_equal_actions = jnp.mean(action_equal)
    return avg_matching_states, mean_equal_actions

def eval_naive_llm(args, dataloader, model, episode_id: int = 0):
    if args.multi_step_eval:
        # --- Multi-Step Evaluation Logic ---
        num_future_steps = args.num_steps_to_predict
        num_correct = 0
        num_total = 0
        total_prediction_time = 0
        num_predictions = 0
        first_step_correct = 0
        first_step_total = 0
        correct_after_flip = 0
        total_after_flip = 0
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)
        
        # Initialize environment (parameters will be set per datapoint)
        env_size = 7
        env_max_steps = num_future_steps + 5  # Sufficiently large

        for a_idx in tqdm(range(args.num_agents_to_sample), desc="Multi-step Eval Samples"):
            data_sample = jax.tree.map(lambda x: x[a_idx, -1, :20+num_future_steps], datapoint)
            
            initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
            initial_actions_traj = jax.tree.map(lambda x: x[:20], data_sample['actions'])
            gt_agent_script_id = int(initial_states_traj['agent_id'][0])

            avg_matching_states, mean_equal_actions = get_matching_states_and_actions(initial_states_traj, initial_actions_traj)
            
            gt_future_actions = data_sample['actions'][19:]  # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)

            num_env_agents = 4 if args.group else 1
            # Infer num_blocks and num_walls from the state
            state_at_t19 = jax.tree.map(lambda x: x[19], initial_states_traj)
            
            # Handle different possible shapes for block_locations
            if isinstance(state_at_t19['block_locations'], (np.ndarray, jnp.ndarray)):
                if state_at_t19['block_locations'].ndim == 1:
                    num_blocks = 1
                elif state_at_t19['block_locations'].ndim == 2:
                    num_blocks = state_at_t19['block_locations'].shape[0]
                else:
                    num_blocks = 0
            else:
                num_blocks = 0
                
            num_walls = state_at_t19['wall_locations'].shape[0]
            
            env = AutomaticityEnv(num_agents=num_env_agents, size=env_size, max_steps=env_max_steps, 
                                  num_blocks=num_blocks, num_walls=num_walls)
            
            # Initialize simulation state from the last observed state
            current_sim_state_pytree = state_at_t19
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
            
            # Simulate future steps
            for step_idx in range(num_future_steps):
                if step_idx >= gt_future_actions.shape[0]:
                    break
                
                step_prediction_start = time.time()
                # For each agent in the environment
                for agent_id in range(num_env_agents):
                    if step_idx < 10:
                        num_total += 1
                    else:
                        total_after_flip += 1
                            
                    # Create a trajectory for this specific prediction
                    if step_idx == 0:
                        # First prediction uses the initial trajectory
                        pred_states = initial_states_traj
                        pred_actions = initial_actions_traj
                        first_step_total += 1
                    else:
                        # Subsequent predictions use updated trajectory with previous predictions
                        # Process each updated state individually to get observations
                        updated_obs = []
                        for t in range(step_idx):
                            # obs_t = env.get_observation(updated_states[t])[0]
                            obs_t = updated_states[t]
                            updated_obs.append(obs_t)
                        
                        # Stack observations into a sequence
                        stacked_obs = jax.tree.map(lambda *xs: jnp.stack(xs), *updated_obs)
                        
                        # Concatenate with initial trajectory
                        pred_states = jax.tree.map(lambda x, y: jnp.concatenate([x[:20], y], axis=0), 
                                                  initial_states_traj, stacked_obs)
                        pred_actions = jnp.concatenate([initial_actions_traj, 
                                                       jnp.array(action_history[:step_idx])], axis=0)
                    
                    # Get model prediction for this step and agent
                    try:
                        predicted_probs = model.predict_action(pred_states, pred_actions, 
                                                              agent_id=agent_id, timestep=14+step_idx)
                        
                        # Get the predicted action (highest probability)
                        predicted_action = np.argmax(predicted_probs)
                        
                        # Compare with ground truth
                        gt_action = gt_future_actions[step_idx][agent_id] if args.group else gt_future_actions[step_idx][0]
                        if predicted_action == gt_action:
                            if step_idx < 10:
                                num_correct += 1
                                if step_idx == 0:
                                    first_step_correct += 1
                            else:
                                correct_after_flip += 1
                    except Exception as e:
                        print(f"Error predicting action for agent {agent_id}, step {step_idx}: {e}")
                        continue
                
                step_prediction_time = time.time() - step_prediction_start
                total_prediction_time += step_prediction_time
                num_predictions += 1
                
                # Collect actions for all agents to update environment
                action_to_take_in_env_list = []
                for agent_id in range(num_env_agents):
                    try:
                        predicted_probs = model.predict_action(pred_states, pred_actions, 
                                                              agent_id=agent_id, timestep=19+step_idx)
                        action_to_take_in_env_list.append(np.argmax(predicted_probs))
                    except:
                        # If prediction fails, use ground truth (to keep simulation going)
                        gt_action = gt_future_actions[step_idx][agent_id] if args.group else gt_future_actions[step_idx][0]
                        action_to_take_in_env_list.append(gt_action)
                
                # Store actions for next iteration
                if step_idx == 0:
                    action_history = [action_to_take_in_env_list]
                    updated_states = []
                else:
                    action_history.append(action_to_take_in_env_list)
                
                current_sim_state_pytree = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])

                updated_states.append(jax.tree.map(lambda x: x, current_sim_state_pytree))

        
        accuracy = num_correct / num_total if num_total > 0 else 0
        accuracy_after_flip = correct_after_flip / total_after_flip if total_after_flip > 0 else 0
        avg_prediction_time = total_prediction_time / num_predictions if num_predictions > 0 else 0
        first_step_accuracy = first_step_correct / first_step_total if first_step_total > 0 else 0
        print(f"NLLM Multi-Step Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
        print(f"Average prediction time per step: {avg_prediction_time:.4f} seconds")
        return accuracy, avg_prediction_time, first_step_accuracy, gt_agent_script_id, accuracy_after_flip, avg_matching_states, mean_equal_actions
    else:
        # Original single-step evaluation
        num_correct = 0
        num_total = 0
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)

        for a in tqdm(range(args.num_agents_to_sample)):
            data = jax.tree.map(lambda x: x[a, -1, :15], datapoint)
            states = data['states']
            actions = data['actions']  # (20, 1)  # 20 timesteps, 1 action
            agent_ids = data['agent_ids']
            for agent_id in tqdm(range(actions.shape[1])):
                num_total += 1
                tries = 0
                gt_action = actions[-1, agent_id]
                try:
                    predicted_probs = model.predict_action(states, actions, agent_id=agent_id)
                    final_action_prob = predicted_probs[gt_action]
                    if final_action_prob >= np.max(predicted_probs):
                        num_correct += 1
                except Exception as e:
                    tries += 1
            
        print(f"Accuracy: {num_correct / num_total}")
        return num_correct / num_total


def load_bc_models(args):
    """Load BC models from multiple seeds."""
    states = []
    
    # Define the learning rate to use
    lr = args.learning_rate

    # Initialize model
    if args.group:
        num_to_predict = 4
    else:
        num_to_predict = 1
    model = BCNet(output_size=6, hidden_size=32, num_to_predict=num_to_predict)
    
    for seed in range(6):
        # Construct the path pattern similar to what's used in train_baselines.py
        model_path = f"baselines/BC/{args.save_path}/nagents{args.num_agents_to_sample}_ndatapoints{args.num_datapoints_per_agent_to_sample}_seed{seed}_lr{lr}_group{args.group}"
        
        # Look for the latest checkpoint
        checkpoint_dir = os.path.dirname(model_path)

        if not os.path.exists(checkpoint_dir):
            print(f"Directory {checkpoint_dir} does not exist, skipping seed {seed}")
            continue
            
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                            if os.path.basename(model_path) in f and "checkpoint" in f]
        
        if not checkpoint_files:
            # Try looking for final model
            final_model = f"{model_path}_final.msgpack"
            if os.path.exists(final_model):
                checkpoint_path = final_model
            else:
                print(f"No checkpoint found for seed {seed}, skipping")
                continue
        else:
            # Find the most recent checkpoint
            checkpoint_epochs = [int(f.split("epoch")[-1].split(".")[0]) for f in checkpoint_files]
            if 2500 in checkpoint_epochs:
                most_recent_epoch = 2500
            else:
                most_recent_epoch = max(checkpoint_epochs)
            
            most_recent_checkpoint = [f for f in checkpoint_files if f"epoch{most_recent_epoch}" in f][0]
            checkpoint_path = os.path.join(checkpoint_dir, most_recent_checkpoint)
        
        print(f"Loading BC model from seed {seed}: {checkpoint_path}")
        
        
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_bytes = f.read()
        
        # Create a dummy state to get the structure right
        rng_key = jax.random.PRNGKey(0)
        dummy_states = jnp.zeros((args.num_datapoints_per_agent_to_sample, 20, args.env_size*7, args.env_size*7, 3))
        dummy_actions = jnp.zeros((args.num_datapoints_per_agent_to_sample, 20, 1))
        variables = model.init(rng_key, dummy_states, dummy_actions)
        
        # Create target structure for deserialization
        target = {
            'params': variables['params'],
            'batch_stats': variables.get('batch_stats', flax.core.FrozenDict()),
            'epoch': 0,
            'loss': 0.0
        }
        
        # Deserialize checkpoint
        checkpoint = flax.serialization.from_bytes(target, checkpoint_bytes)
        
        states.append({
            'params': checkpoint['params'],
            'batch_stats': checkpoint['batch_stats']
        })
    # Stack all parameters from different seeds into a single state
    stacked_state = {
        'params': jax.tree.map(lambda *xs: jnp.stack(xs), *[s['params'] for s in states]),
        'batch_stats': jax.tree.map(lambda *xs: jnp.stack(xs), *[s['batch_stats'] for s in states])
    }
    return model, stacked_state


img_gen_fn = jax.jit(state_to_image_jit, static_argnums=(1, 2, 3))
def eval_bc(args, dataloader, model, states, episode_id: int = 0, env=None):
    """Evaluate BC models."""
    
    if args.multi_step_eval:
        # --- Multi-Step Evaluation Logic ---
        num_future_steps = args.num_steps_to_predict
        num_correct = 0
        num_total = 0
        total_prediction_time = 0
        num_predictions = 0
        first_step_correct = 0
        first_step_total = 0
        correct_after_flip = 0
        total_after_flip = 0
        
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)

        env_size = 7
        env_max_steps = num_future_steps + 5 # Sufficiently large
        
        for a_idx in tqdm(range(args.num_agents_to_sample), desc="Multi-step Eval Samples"):
            data_sample = jax.tree.map(lambda x: x[a_idx, -1, :20+num_future_steps], datapoint)
            
            initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
            initial_actions_traj = jax.tree.map(lambda x: x[:20], data_sample['actions'])
            gt_agent_script_id = int(initial_states_traj['agent_id'][0])

            avg_matching_states, mean_equal_actions = get_matching_states_and_actions(initial_states_traj, initial_actions_traj)
            
            gt_future_actions = data_sample['actions'][19:]  # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)

            num_env_agents = 4 if args.group else 1
            # Infer num_blocks and num_walls from the state
            state_at_t19 = jax.tree.map(lambda x: x[19], initial_states_traj)
            
            # Handle different possible shapes for block_locations
            if isinstance(state_at_t19['block_locations'], (np.ndarray, jnp.ndarray)):
                if state_at_t19['block_locations'].ndim == 1:
                    num_blocks = 1
                elif state_at_t19['block_locations'].ndim == 2:
                    num_blocks = state_at_t19['block_locations'].shape[0]
                else:
                    num_blocks = 0
            else:
                num_blocks = 0
                
            num_walls = state_at_t19['wall_locations'].shape[0]
            
            if env is None:
                env = AutomaticityEnv(num_agents=num_env_agents, size=env_size, max_steps=env_max_steps, 
                                    num_blocks=num_blocks, num_walls=num_walls)
            
            # Initialize simulation state from the last observed state
            current_sim_state_pytree = state_at_t19
            def convert_to_state(state_t):
                return State(
                    wall_locations=state_t['wall_locations'],
                    agent_locations=state_t['agent_locations'],
                    block_locations=state_t['block_locations'],
                    agent_inventory=state_t['agent_inventory'],
                    agent_inventory_colors=state_t['agent_inventory_colors'],
                    block_colors=state_t['block_colors'],
                    time=state_t['time'],
                    terminal=False,
                    agent_id=-1
                )
            current_sim_state_pytree = convert_to_state(current_sim_state_pytree)
            
            # Convert initial states to images for BC model
            initial_states_images = []
            for t in range(20):
                state_t = jax.tree.map(lambda x: x[t], initial_states_traj)
                img_size = args.env_size * 7
                tile_size = 7
                grid_size = args.env_size

                try:
                    image_t = img_gen_fn(state_t, img_size, grid_size, tile_size)
                    initial_states_images.append(image_t)
                except Exception as e:
                    print(f"Error generating image for state {state_t} at time {t}: {e}")
            
            initial_states_images = jnp.stack(initial_states_images)
            
            # Simulate future steps
            for step_idx in range(num_future_steps):
                try:
                    current_sim_state_pytree = convert_to_state(current_sim_state_pytree)
                except Exception as e:
                    pass

                if step_idx >= gt_future_actions.shape[0]:
                    break
                # For each agent in the environment
                for agent_id in range(num_env_agents):
                    if step_idx < 10:
                        num_total += 1
                        if step_idx == 0:
                            first_step_total += 1
                    else:
                        total_after_flip += 1
                    
                    # Create a trajectory for this specific prediction
                    if step_idx == 0:
                        # First prediction uses the initial trajectory
                        pred_states_images = initial_states_images
                        pred_actions = initial_actions_traj
                    else:
                        # Convert updated states to images
                        updated_states_images = []
                        for t in range(step_idx):
                            state_t = updated_states[t]
                            try:
                                state_t = convert_to_state(state_t)
                            except Exception as e:
                                pass
                            img_size = args.env_size * 7
                            tile_size = 7
                            grid_size = args.env_size
                            
                            try:
                                obs = jax.tree.map(lambda x: jnp.array(x), env.get_observation(state_t)[0])
                            except Exception as e:
                                breakpoint()
                            image_t = img_gen_fn(obs, img_size, grid_size, tile_size)
                            updated_states_images.append(image_t)
                        
                        # Concatenate with initial images
                        pred_states_images = jnp.concatenate([initial_states_images, jnp.stack(updated_states_images)], axis=0)
                        pred_actions = jnp.concatenate([initial_actions_traj, jnp.array(action_history[:step_idx])], axis=0)
                    
                    # Get model prediction for this step and agent
                    try:
                        # Reshape for batch prediction
                        pred_states_batch = jnp.expand_dims(pred_states_images, axis=0)  # Add batch dimension
                        pred_actions_batch = jnp.expand_dims(pred_actions, axis=0)  # Add batch dimension
                        
                        # Apply model to get predictions
                        def single_agent_pred(param_state, data_states, data_actions):
                            variables = {'params': param_state['params'], 'batch_stats': param_state['batch_stats']}
                            res = model.apply(variables, data_states, data_actions, training=False)
                            return res
                        
                        step_prediction_start = time.time()
                        
                        # Get predictions from all ensemble models
                        action_preds = jax.vmap(single_agent_pred, in_axes=(0, None, None))(
                            states, pred_states_batch, pred_actions_batch)

                        step_prediction_time = time.time() - step_prediction_start
                        total_prediction_time += step_prediction_time
                        # action_preds shape: [num_models, num_to_predict, batch*timesteps, 6]
                        # We want the prediction for the last timestep
                        action_preds = action_preds[:, agent_id, -1, :]  # [num_models, 6]
                        
                        # Average predictions across ensemble

                        def single_network_acc(action_pred, gt_action):
                            max_action_prob = jnp.max(action_pred)
                            num_max_actions = jnp.sum(action_pred == max_action_prob)
                            gt_action_prob = action_pred[gt_action]
                            acc = jnp.where(max_action_prob == gt_action_prob, 1 / num_max_actions, 0)
                            return acc
                        gt_action = gt_future_actions[step_idx][agent_id] if args.group else gt_future_actions[step_idx][0]
                        accs = jax.vmap(single_network_acc, in_axes=(0, None))(action_preds, gt_action)
                        avg_accs = jnp.mean(accs)
                        if step_idx < 10:
                            num_correct += avg_accs
                            if step_idx == 0:
                                first_step_correct += avg_accs
                        else:
                            correct_after_flip += avg_accs

                    except Exception as e:
                        print(f"Error predicting action for agent {agent_id}, step {step_idx}: {e}")
                        continue
                

                num_predictions += 1
                
                

                # Store actions for next iteration
                if step_idx == 0:
                    action_history = [gt_future_actions[step_idx+1]]
                    updated_states = []
                else:
                    action_history.append(gt_future_actions[step_idx+1])
                
                # Use ground truth next timestep instead of simulating environment step
                current_sim_state_pytree = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])
                updated_states.append(jax.tree.map(lambda x: x, current_sim_state_pytree))
        
        accuracy = num_correct / num_total if num_total > 0 else 0
        avg_prediction_time = total_prediction_time / num_predictions if num_predictions > 0 else 0
        first_step_accuracy = first_step_correct / first_step_total if first_step_total > 0 else 0
        accuracy_after_flip = correct_after_flip / total_after_flip if total_after_flip > 0 else 0
        print(f"BC Multi-Step Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
        print(f"First Step Accuracy: {first_step_accuracy:.4f} ({first_step_correct}/{first_step_total})")
        print(f"Accuracy After Flip: {accuracy_after_flip:.4f} ({correct_after_flip}/{total_after_flip})")
        print(f"Average prediction time per step: {avg_prediction_time:.4f} seconds")
        return accuracy, avg_prediction_time, first_step_accuracy, avg_matching_states, mean_equal_actions, accuracy_after_flip, env, gt_agent_script_id
    else:
        # Original single-step evaluation
        num_correct = 0
        num_total = 0
        
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)
        
        for a in range(args.num_agents_to_sample):
            data = jax.tree.map(lambda x: x[a, -1:, :15], datapoint)  # num_datapoints, 15, *
            gt_final_actions = data['actions'][-1, -1]  # num_agents,

            def single_agent_pred(param_state, data):
                # Include batch_stats in the variables dictionary
                variables = {'params': param_state['params'], 'batch_stats': param_state['batch_stats']}
                res = model.apply(variables, data['states'], data['actions'], training=False)
                return res
            
            action_preds = jax.vmap(single_agent_pred, in_axes=(0, None))(states, data)  # num loaded params, num_agents,num_timepoints, 6
            action_preds = action_preds[:, :, -1, :]  # get final timepoint prediction for each agent, (num loaded params, num_agents, 6)

            def single_agent_acc(action_ps, gt_final_action):
                def single_action_acc(action_p, gt_final_action):
                    action_pred = jnp.argmax(action_p, axis=-1)
                    return action_pred == gt_final_action
                action_accs = jax.vmap(single_action_acc, in_axes=(0, 0))(action_ps, gt_final_action)
                return jnp.sum(action_accs)

            action_accs = jax.vmap(single_agent_acc, in_axes=(0, None))(action_preds, gt_final_actions)

            num_correct += jnp.mean(action_accs)  # average over num_parameters
            num_total += action_preds.shape[1]  # count number of agents predictions

        accuracy = num_correct / num_total if num_total > 0 else 0
        # print(f"BC Ensemble Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
        return accuracy

def eval_fsm_bootstrap(args, dataloader, model, episode_id: int = 0):
    """Evaluate FSM with bootstrapping for different numbers of hypotheses."""
    
    if args.multi_step_eval:
        # --- Multi-Step Evaluation Logic ---
        max_hypotheses = args.n_hypothesis  # Maximum number of hypotheses to consider
        num_future_steps = args.num_steps_to_predict
        
        # Track results for each hypothesis count
        results = {n: {'correct': 0, 'total': 0, 'program_length': 0, 
                       'first_step_correct': 0, 'first_step_total': 0, 'correct_after_flip': 0, 'total_after_flip': 0} for n in range(1, max_hypotheses + 1)}
        
        # Add timing measurements
        total_prediction_time = 0  # <-- This will now accumulate action prediction time
        num_predictions = 0
        
        # Initialize environment (parameters will be set per datapoint)
        # Env params that are usually fixed or can be default
        env_size = 10 
        env_max_steps = num_future_steps + 5 # Sufficiently large

        datapoint = next(dataloader) # Process one batch of data
        generation_times = 0
        num_generations = 0
        
        # Track agent type information for JSON output
        agent_type_info = {}

        for a_idx in tqdm(range(args.num_agents_to_sample), desc="Multi-step Eval Samples"):
            data_sample = jax.tree.map(lambda x: x[a_idx, -1, :20+num_future_steps], datapoint) 
            
            initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
            initial_actions_traj = jax.tree.map(lambda x: x[:20], data_sample['actions'])
            initial_images_traj = jax.tree.map(lambda x: x[:20], data_sample['images'])

            avg_matching_states, mean_equal_actions = get_matching_states_and_actions(initial_states_traj, initial_actions_traj)

            agent_id = int(initial_states_traj['agent_id'][0])
            
            gt_future_actions = data_sample['actions'][19:] # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)

            num_env_agents = 4 if args.group else 1
            # Infer num_blocks and num_walls from the state at the beginning of the prediction horizon
            # Use state at t=14 (end of initial trajectory) to setup env for t=15 prediction
            state_at_t14 = jax.tree.map(lambda x: x[19], initial_states_traj)
            num_blocks = state_at_t14['block_locations'].shape[0] if len(state_at_t14['block_locations'].shape) > 1 else state_at_t14['block_locations'].shape[0] # handle single block case
            if isinstance(state_at_t14['block_locations'], (np.ndarray, jnp.ndarray)) and state_at_t14['block_locations'].ndim == 1 and num_blocks > 0 : # if it's a flat array for a single block
                 num_blocks = 1
            elif isinstance(state_at_t14['block_locations'], (np.ndarray, jnp.ndarray)) and state_at_t14['block_locations'].ndim == 2:
                 num_blocks = state_at_t14['block_locations'].shape[0]
            else: # No blocks or unexpected shape
                 num_blocks = 0

            num_walls = state_at_t14['wall_locations'].shape[0]
            
            env = AutomaticityEnv(num_agents=num_env_agents, size=env_size, max_steps=env_max_steps, 
                                  num_blocks=num_blocks, num_walls=num_walls)

            # Get all available hypotheses
            compiled_agents = None
            agent_probs = None
            agent_codes = None
            generation_state = time.time()
            framework = AgentExecutionFramework()
            # Hypothesis generation time is NOT used for avg_prediction_time anymore
            # if args.rejuvenation:
            compiled_agents, agent_probs, agent_codes, all_time_all_hyp_log_prob_list = model.predict_action_with_bootstrap(
                initial_states_traj, initial_actions_traj, episode_id=episode_id,
                max_hypotheses=max_hypotheses,
                rejuvenation_threshold=args.rejuvenation_threshold,
                max_rejuvenation_attempts=args.max_rejuvenation_attempts,
                top_k=0,  # Don't apply top_k here, we'll apply it per n_hyp
                return_compiled_agents=True,
                return_all_time_log_prob_list=True,
                doing_rejuvenation=args.rejuvenation
            )

            generation_time = time.time() - generation_state
            # print(f"Generation time: {generation_time:.4f} seconds")
            generation_times += generation_time
            num_generations += 1

            if args.plot_gifs:
                # summarize the agent code
                agent_summaries = [model.summarize_agent_code(code) for code in agent_codes]
                agent_summaries = [summary.replace(".", "") for summary in agent_summaries] # remove periods
                agent_summaries = [summary.replace("'", "") for summary in agent_summaries] # remove quotes
                agent_summaries = [summary.replace("\n", "") for summary in agent_summaries] # remove quotes
                agent_summaries = [summary.replace("`", "") for summary in agent_summaries] # remove quotes
                agent_summaries = [summary.replace('"', "") for summary in agent_summaries] # remove quotes

            num_predictions += 1

            if not compiled_agents or not agent_probs:
                print(f"Sample {a_idx}: No hypotheses generated, skipping.")
                continue
            
            # Track hypotheses for this agent type
            # Get program save root if available
            program_save_root = getattr(model, 'program_save_root', None)
            hypotheses_info = []
            hypothesis_weights = []
            program_paths = []
            
            # Get the number of available hypotheses
            num_available_hyp = len(compiled_agents)
            
            # Track all hypotheses that were successfully compiled
            # Note: The hypothesis IDs in the file paths correspond to the order they were generated
            # We need to track which ones actually compiled successfully
            for hyp_idx in range(num_available_hyp):
                # The hypothesis ID used in file paths is the original generation index
                # We need to check what the actual hyp_id is - it should match the index in agent_probs
                hyp_id = hyp_idx  # This should match the hyp_X folder name
                hyp_weight = float(agent_probs[hyp_idx]) if hyp_idx < len(agent_probs) else 0.0
                hypotheses_info.append(hyp_id)
                hypothesis_weights.append(hyp_weight)
                
                # Construct program path (relative to program_save_root)
                if program_save_root:
                    # Try good first, then raw
                    good_path = program_save_root / f"epoch_{episode_id}/hyp_{hyp_id}/good/program.py"
                    raw_path = program_save_root / f"epoch_{episode_id}/hyp_{hyp_id}/raw/program.py"
                    if good_path.exists():
                        program_path = f"epoch_{episode_id}/hyp_{hyp_id}/good/program.py"
                    elif raw_path.exists():
                        program_path = f"epoch_{episode_id}/hyp_{hyp_id}/raw/program.py"
                    else:
                        # File might not exist yet, but we'll record the expected path
                        program_path = f"epoch_{episode_id}/hyp_{hyp_id}/good/program.py"
                    program_paths.append(program_path)
                else:
                    program_paths.append(None)
            
            # Find the highest-weighted hypothesis
            if hypothesis_weights:
                best_hyp_idx = np.argmax(hypothesis_weights)
                best_hyp_id = hypotheses_info[best_hyp_idx]
                best_hyp_weight = hypothesis_weights[best_hyp_idx]
                best_program_path = program_paths[best_hyp_idx] if best_hyp_idx < len(program_paths) else None
            else:
                best_hyp_id = None
                best_hyp_weight = 0.0
                best_program_path = None
            
            # Store agent type information (will update ensemble accuracy and best accuracy program later)
            agent_type_info[f"agent_id_{agent_id}"] = {
                "agent_id": int(agent_id),
                "hypotheses": [int(h) for h in hypotheses_info],
                "hypothesis_weights": [float(w) for w in hypothesis_weights],
                "program_paths": program_paths,
                "best_program": {
                    "hypothesis_id": int(best_hyp_id) if best_hyp_id is not None else None,
                    "weight": float(best_hyp_weight),
                    "program_path": best_program_path
                }
            }
            
            # Initialize individual hypothesis accuracy tracking for this agent type
            # Track accuracy for each hypothesis individually (not ensemble)
            individual_hyp_accuracies = {}  # hyp_id -> {'correct': int, 'total': int}
            for hyp_id in hypotheses_info:
                individual_hyp_accuracies[hyp_id] = {'correct': 0, 'total': 0}
            
            # Helper function to convert JAX arrays to numpy (for individual evaluation)
            def to_numpy(x):
                if isinstance(x, (jnp.ndarray, jax.Array)):
                    return np.array(x)
                return x
            
            # Evaluate each hypothesis individually to track individual accuracy
            # (Do this once per agent type, using all available hypotheses)
            for hyp_idx, hyp_agent in enumerate(compiled_agents):
                hyp_id = hypotheses_info[hyp_idx]
                hyp_correct = 0
                hyp_total = 0
                
                # Reset simulation state for this hypothesis
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
                
                # Simulate future steps for this individual hypothesis
                for step_idx in range(num_future_steps):
                    if step_idx >= gt_future_actions.shape[0]:
                        break
                    
                    try:
                        # Get observation from ground truth data
                        current_obs_raw = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])
                        current_obs = jax.tree.map(to_numpy, current_obs_raw)
                        current_obs['agent_id'] = 0
                        
                        # Get prediction from this individual hypothesis
                        predicted_action = framework.execute_agent(hyp_agent, current_obs)
                        
                        # Extract ground truth action
                        gt_action_this_step = gt_future_actions[step_idx]
                        if hasattr(gt_action_this_step, '__len__') and len(gt_action_this_step) > 0:
                            gt_action = int(gt_action_this_step[0])
                        else:
                            gt_action = int(gt_action_this_step)
                        
                        # Convert action to int
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
                        for aid in range(num_env_agents):
                            hyp_total += 1
                            if predicted_action == gt_action:
                                hyp_correct += 1
                            
                    except Exception as e:
                        hyp_total += 1  # Count as incorrect
                        continue
                
                # Update individual hypothesis accuracy tracking
                if hyp_id in individual_hyp_accuracies:
                    individual_hyp_accuracies[hyp_id]['correct'] += hyp_correct
                    individual_hyp_accuracies[hyp_id]['total'] += hyp_total
            
            # Find the hypothesis with highest individual accuracy for this agent type
            best_acc_hyp_id = None
            best_acc_value = -1.0
            best_acc_program_path = None
            
            for hyp_id, acc_data in individual_hyp_accuracies.items():
                if acc_data['total'] > 0:
                    acc_value = acc_data['correct'] / acc_data['total']
                    if acc_value > best_acc_value:
                        best_acc_value = acc_value
                        best_acc_hyp_id = hyp_id
                        # Find the program path for this hypothesis
                        if best_acc_hyp_id is not None:
                            hyp_idx_in_list = hypotheses_info.index(best_acc_hyp_id) if best_acc_hyp_id in hypotheses_info else None
                            if hyp_idx_in_list is not None and hyp_idx_in_list < len(program_paths):
                                best_acc_program_path = program_paths[hyp_idx_in_list]
            
            # Add best accuracy program to agent_type_info for this agent
            agent_type_info[f"agent_id_{agent_id}"]["best_accuracy_program"] = {
                "hypothesis_id": int(best_acc_hyp_id) if best_acc_hyp_id is not None else None,
                "accuracy": float(best_acc_value) if best_acc_hyp_id is not None else 0.0,
                "program_path": best_acc_program_path
            }
            
            # For each hypothesis count we want to evaluate
            for n_hyp in range(1, max_hypotheses + 1):
                # If we don't have enough hypotheses, use what we have
                actual_n_hyp = min(n_hyp, num_available_hyp)
                
                # Take only the first actual_n_hyp hypotheses
                curr_agents = compiled_agents[:actual_n_hyp]
                curr_probs = np.array(agent_probs[:actual_n_hyp])
                curr_codes = agent_codes[:actual_n_hyp]
                curr_all_time_log_prob_list = all_time_all_hyp_log_prob_list
                
                
                # Normalize probabilities for this subset
                curr_probs = curr_probs / np.sum(curr_probs)
                
                # Apply top_k filtering if specified and valid
                if args.top_k > 0 and args.top_k < actual_n_hyp:
                    # Get indices of top k hypotheses by probability
                    top_k_indices = np.argsort(curr_probs)[-args.top_k:]
                    
                    # Filter to only include top k
                    filtered_agents = [curr_agents[i] for i in top_k_indices]
                    filtered_probs = curr_probs[top_k_indices]
                    filtered_codes = [curr_codes[i] for i in top_k_indices]

                    # Renormalize the filtered probabilities
                    filtered_probs = filtered_probs / np.sum(filtered_probs)
                    
                    # Use the filtered lists
                    curr_agents = filtered_agents
                    curr_probs = filtered_probs
                    curr_codes = filtered_codes
                
                # Calculate weighted program length for this hypothesis count
                program_lengths = np.array([len(code) for code in curr_codes])
                weighted_prog_len = np.sum(curr_probs * program_lengths)
                results[n_hyp]['program_length'] += weighted_prog_len
                
                # Initialize simulation state from the last observed state
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
                
                # Track correct predictions for this hypothesis count
                step_correct = 0
                step_total = 0
                correct_after_flip = 0
                total_after_flip = 0
                action_time_per_step = time.time()

                predicted_distribution_actions = []
                ground_truth_actions = []

                # Simulate future steps
                for step_idx in range(num_future_steps):
                    if step_idx >= gt_future_actions.shape[0]:
                        break
                    
                    if step_idx < 10:
                        step_total += num_env_agents
                    else:
                        total_after_flip += 1
                    
                    # Prepare agent_input_state for agent.act()
                    agent_input_state_for_act = current_sim_state_pytree
                    
                    # Aggregate predictions from the filtered hypotheses
                    all_agent_pis = np.zeros((num_env_agents, 6))

                    # --- Start timing action prediction ---
                    step_prediction_start = time.time()
                    
                    for hyp_idx, hyp_agent in enumerate(curr_agents):
                        hyp_prob = curr_probs[hyp_idx]
                        try:
                            current_obs['agent_id'] = 0
                            predicted_action = framework.execute_agent(hyp_agent, current_obs)
                            action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
                            action_space_2 = ["stay", "right", "left", "down", "up", "interact"]
                            if args.group:
                                # proposed_pi_for_hyp is a list of np.arrays. Stack them.
                                for pred_aid, pred_action in enumerate(predicted_action):
                                    if type(pred_action) == tuple:
                                        pred_action = list(pred_action)
                                    elif type(pred_action) == str:
                                        pred_action = pred_action.lower()
                                        pred_action = action_space_2.index(pred_action)
                                    else:
                                        pred_action = int(pred_action)
                                    if pred_action in action_space:
                                        pred_action = action_space.index(pred_action)
                                    elif pred_action in action_space_2:
                                        pred_action = action_space_2.index(pred_action)
                                    all_agent_pis[pred_aid, pred_action] += hyp_prob # (num_env_agents, num_actions)
                            else:
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
                                if predicted_action < all_agent_pis.shape[1]:
                                    all_agent_pis[0, predicted_action] += hyp_prob # (num_actions,)
                                else:
                                    continue
                        except Exception as e:
                            continue


                    # Sum weighted pis
                    final_predicted_pi_for_step = all_agent_pis
                    
                    action_to_take_in_env_list = []
                    if args.group:
                        action_indices = np.argmax(final_predicted_pi_for_step, axis=1) # (num_env_agents,)
                        action_to_take_in_env_list = list(action_indices)
                    else:
                        action_index = np.argmax(final_predicted_pi_for_step) # scalar
                        action_to_take_in_env_list = [action_index]
                    # --- End timing action prediction ---
                    step_prediction_time = time.time() - step_prediction_start
                    total_prediction_time += step_prediction_time
                    num_predictions += len(action_to_take_in_env_list)
                    # Compare with ground truth for this step
                    gt_action_this_step = gt_future_actions[step_idx] # (num_env_agents,) or (1,)
                    
                    for aid in range(num_env_agents):
                        predicted_distribution_actions.append(final_predicted_pi_for_step[aid])
                        ground_truth_actions.append(gt_action_this_step[aid])
                        max_action_prob = np.max(final_predicted_pi_for_step[aid])
                        # count number of actions with max probability  
                        num_max_actions = np.sum(final_predicted_pi_for_step[aid] == max_action_prob)
                        gt_action_prob = final_predicted_pi_for_step[aid, gt_action_this_step[aid]]
                        if max_action_prob == gt_action_prob:
                            if step_idx < 10:
                                step_correct += 1 / num_max_actions
                                if step_idx == 0:
                                    # breakpoint()
                                    # sample_code = curr_codes[0]
                                    # # save sample code as a python file but remove everything before ```python and after ```
                                    # sample_code = sample_code.split("```python")[1].split("```")[0]
                                    # with open(f"sample_code.py", "w") as f:
                                    #     f.write(sample_code)
                                    # exit()
                                    results[n_hyp]['first_step_correct'] += (1 / num_max_actions)
                            else:
                                correct_after_flip += 1 / num_max_actions
                    
                    # Track first step total separately
                    if step_idx == 0:
                        results[n_hyp]['first_step_total'] += num_env_agents
                    
                    current_obs = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])

                
                if args.plot_gifs and n_hyp == max_hypotheses:
                    curr_summaries = agent_summaries
                    # here is where we'll plot
                    images = data_sample['images']  # should
                    predicted_distributions = [np.zeros(6)] * (initial_images_traj.shape[0] - 1)
                    predicted_distributions.extend(predicted_distribution_actions)
                    predicted_distributions.append(np.zeros(6))

                    # Create gif frames
                    frames = []
                    action_to_name = {0: "stay", 1: "right", 2: "left", 3: "down", 4: "up", 5: "interact"}
                    
                    for time_idx in range(len(predicted_distributions)):
                        if time_idx < len(curr_all_time_log_prob_list):
                            curr_log_prob_time = curr_all_time_log_prob_list[time_idx]
                            # find 3 most likely log probs
                            most_likely_log_probs_idx = np.argsort(curr_log_prob_time)[-3:]
                            most_likely_log_probs = curr_log_prob_time[most_likely_log_probs_idx]
                            most_likely_log_probs_summaries = [curr_summaries[i] for i in most_likely_log_probs_idx]

                        # Create a new figure for each frame
                        fig = plt.figure(figsize=(25, 10))
                        
                        # Plot image on left subplot of top row
                        plt.subplot(2, 2, 1)
                        plt.imshow(images[time_idx])
                        plt.axis('off')
                        
                        # Plot distribution in right subplot of top row
                        plt.subplot(2, 2, 2)
                        predicted_distribution = predicted_distributions[time_idx]
                        ground_truth_action = data_sample['actions'][time_idx][0]
                        
                        # Create bar colors - green for correct prediction
                        colors = ['lightblue'] * len(predicted_distribution)
                        max_prob = np.max(predicted_distribution)
                        if max_prob == predicted_distribution[ground_truth_action]:
                            colors[ground_truth_action] = 'lightgreen'
                            
                        plt.bar(range(len(predicted_distribution)), predicted_distribution, color=colors)
                        plt.xticks(range(len(predicted_distribution)), [action_to_name[i] for i in range(len(predicted_distribution))])
                        plt.ylim(0, 1)
                        plt.title('Action Distribution')

                        # Plot ground truth in bottom left
                        plt.subplot(2, 2, 3)
                        script_list = sorted(os.listdir("generated_outputs/hand_designed"))
                        script_list = [f.replace('.txt', '') for f in script_list]
                        plt.text(0.5, 0.5, f'Ground Truth Action: {action_to_name[ground_truth_action]}\n Ground Truth FSM: {script_list[agent_id]}',
                               ha='center', va='center')
                        plt.axis('off')

                        # Plot log probs in bottom right 
                        plt.subplot(2, 2, 4)
                        plt.barh(range(len(most_likely_log_probs)), most_likely_log_probs)
                        plt.yticks(range(len(most_likely_log_probs)), most_likely_log_probs_summaries)
                        plt.title('Top 3 Most Likely Programs')
                        plt.xlim(0, 1)
                        
                        # Save plot to temporary file and read it back
                        temp_file = f'temp_frame_{time_idx}.png'
                        fig.savefig(temp_file, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
                        frame = imageio.imread(temp_file)
                        frames.append(frame)
                        
                        # Clean up temporary file and close figure
                        os.remove(temp_file)
                        plt.close(fig)
                    
                    # Save as gif
                    gif_file_name = f'results/{args.baseline_model}/fixed5_results_fsm_bootstrap_multistep_topk{args.top_k}_steps{args.num_steps_to_predict}_actionTime/episode_{episode_id}/prediction_visualization_{n_hyp}.gif'
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(gif_file_name), exist_ok=True)
                    
                    # Save the GIF
                    imageio.mimsave(gif_file_name, frames, fps=2)
                    print(f"Saved GIF to: {gif_file_name}")
                
                # Update results for this hypothesis count
                results[n_hyp]['correct'] += step_correct
                results[n_hyp]['total'] += step_total
                results[n_hyp]['correct_after_flip'] += correct_after_flip
                results[n_hyp]['total_after_flip'] += total_after_flip

        # Calculate accuracies and average program lengths for each number of hypotheses
        accuracies = {}
        first_step_accuracies = {}
        accuracies_after_flip = {}
        program_lengths = {}
        action_times = {}
        for n_hyp in results:
            if results[n_hyp]['total'] > 0:
                accuracies[n_hyp] = results[n_hyp]['correct'] / results[n_hyp]['total']
                program_lengths[n_hyp] = results[n_hyp]['program_length'] / args.num_agents_to_sample
            else:
                accuracies[n_hyp] = 0.0
                program_lengths[n_hyp] = 0.0
            
            if results[n_hyp]['total_after_flip'] > 0:
                accuracies_after_flip[n_hyp] = results[n_hyp]['correct_after_flip'] / results[n_hyp]['total_after_flip']
            else:
                accuracies_after_flip[n_hyp] = 0.0
                
            # Calculate first step accuracy
            if results[n_hyp]['first_step_total'] > 0:
                first_step_accuracies[n_hyp] = results[n_hyp]['first_step_correct'] / results[n_hyp]['first_step_total']
            else:
                first_step_accuracies[n_hyp] = 0.0
        
        # Calculate average prediction time
        avg_prediction_time = total_prediction_time / num_predictions if num_predictions > 0 else 0
        avg_generation_time = generation_times / num_generations if num_generations > 0 else 0
        # Print results
        for n_hyp, acc in accuracies.items():
            print(f"Hypotheses: {n_hyp}, Multi-Step Accuracy: {acc:.4f} ({results[n_hyp]['correct']}/{results[n_hyp]['total']}), First Step Accuracy: {first_step_accuracies[n_hyp]:.4f}, Avg Program Length: {program_lengths[n_hyp]:.1f}")
        print(f"Average prediction time per step: {avg_prediction_time:.4f} seconds")
        
        # Update ensemble accuracy for each agent type (using max_hypotheses accuracy)
        max_hyp_acc = accuracies.get(max_hypotheses, 0.0)
        for agent_key in agent_type_info:
            agent_type_info[agent_key]["ensemble_accuracy"] = float(max_hyp_acc)
        
        # Return the full dictionary of accuracies and program lengths, plus timing info and first step accuracies
        return accuracies, program_lengths, action_times, avg_prediction_time, first_step_accuracies, avg_generation_time, agent_id, accuracies_after_flip, avg_matching_states, mean_equal_actions, agent_type_info

def main():
    """Main function to eval baseline models."""
    args = parse_args()

    # Quiet chat HTTP request spam from the OpenAI/httpx client while retaining warnings/errors.
    for _logger in ("httpx", "openai", "openai._base_client", "openai._client"):
        logging.getLogger(_logger).setLevel(logging.WARNING)
    
    # Set JAX memory allocation to grow as needed
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    # Set PyTorch to use expandable segments to avoid memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        

    print(f"Evaluating baseline model: {args.baseline_model}; n_hypothesis: {args.n_hypothesis}; num_epochs: {args.num_epochs}; model_arch: {args.model_name}; dataset: {args.dataset}")
    if args.baseline_model == "ROTE" and args.bootstrap and args.multi_step_eval:
        print(f"Multi-step evaluation enabled: predicting {args.num_steps_to_predict} future steps.")


    save_path_dir = f"results/{args.baseline_model}/{args.save_path}/nagents{args.num_agents_to_sample}_ndatapoints{args.num_datapoints_per_agent_to_sample}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path_dir), exist_ok=True)
    
    # Optional wandb setup
    wandb_enabled = False
    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb
            dataset_name = args.dataset
            run_name = f"{datetime.now():%y%m%d_%H%M%S}_{dataset_name}"
            # Append prompt_mode for choice13k
            if args.dataset == "choice13k":
                run_name = f"{run_name}_{args.prompt_mode}"
            # Append participant_id to run name if specified
            if args.participant_id is not None:
                run_name = f"{run_name}_participant{args.participant_id}"
            wandb.init(
                project="mindAsCode",
                name=run_name,
                config=vars(args),
                reinit=False,
            )
            wandb_enabled = True
        except Exception as e:
            print(f"wandb logging disabled: {e}")
            wandb_enabled = False

    def _log_to_wandb(data_dict, step):
        if not wandb_enabled or wandb is None:
            return
        try:
            cleaned = {}
            for k, v in data_dict.items():
                if isinstance(v, np.generic):
                    v = v.item()
                cleaned[k] = v
            cleaned.setdefault('mode', args.mode)
            cleaned.setdefault('dataset', args.dataset)
            wandb.log(cleaned, step=step)
        except Exception as e:
            print(f"wandb log failed: {e}")

    # Early dataset dispatch for Choice13k
    if args.dataset == "choice13k":
        run_choice13k_mindascode(args, log_fn=_log_to_wandb if not args.no_log else None, participant_id=args.participant_id)
        print("Finished Choice13k evaluation.")
        return

    # === Gridworld flow ===
    rng_key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    group_extension = "_group" if args.group else ""
    two_stage_extension = "_two_stage" if args.two_stage else ""
    structured_extension = f"_structured_{args.structured}" if args.structured != "False" else ""
    rejuvenation_extension = "_rejuvenation" if args.rejuvenation else ""
    human_data_extension = "_human_data" if args.human_data else ""
    
    # Determine CSV path based on loop mode
    if args.loop_mode == "sequential":
        # For sequential mode, CSV will be saved in the experiment folder (set later when model is created)
        csv_path = None  # Will be set after model.program_save_root is created
    else:
        # Random mode: use fixed location
        if args.baseline_model == "ROTE" and args.bootstrap:
            if args.multi_step_eval:
                csv_path = f"results/{args.baseline_model}/results_rote_bootstrap_multistep{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}{human_data_extension}_topk{args.top_k}_steps{args.num_steps_to_predict}_actionTime_Dec17.csv"
            else: # Single-step ROTE bootstrap
                csv_path = f"results/{args.baseline_model}/results_rote_bootstrap_singlestep{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}{human_data_extension}_topk{args.top_k}.csv"
        else: # Non-bootstrap ROTE or other models
            csv_path = f"results/{args.baseline_model}/results_grid_{args.baseline_model}_{args.n_hypothesis}hyp{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}{human_data_extension}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    start_epoch = 0
    # For sequential mode, csv_path will be set after model creation, so skip resume check here
    if csv_path is not None and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                filter_conditions = [
                    (existing_df['model'].astype(str) == str(args.baseline_model)),
                    (existing_df['group'].astype(str) == str(args.group)),
                    (existing_df['num_agents_evaluated'].astype(str) == str(args.num_agents_to_sample)),
                    (existing_df['datapoints_per_agent'].astype(str) == str(args.num_datapoints_per_agent_to_sample)),
                    (existing_df['llm_model'].astype(str) == str(args.model_name if args.baseline_model in ['TT', 'AutoToM', 'ROTE', 'NLLM'] else 'N/A'))
                ]
                
                if 'two_stage' in existing_df.columns:
                    filter_conditions.append(existing_df['two_stage'].astype(str) == str(args.two_stage))
                if 'structured' in existing_df.columns:
                    filter_conditions.append(existing_df['structured'].astype(str) == str(args.structured))
                if 'rejuvenation' in existing_df.columns:
                    filter_conditions.append(existing_df['rejuvenation'].astype(str) == str(args.rejuvenation))
                if 'top_k' in existing_df.columns: # Relevant for ROTE bootstrap
                     filter_conditions.append(existing_df['top_k'].astype(str) == str(args.top_k))

                if args.baseline_model == "ROTE" and args.bootstrap:
                    if 'multi_step_eval' in existing_df.columns:
                        filter_conditions.append(existing_df['multi_step_eval'].astype(str) == str(args.multi_step_eval))
                    if args.multi_step_eval and 'num_steps_predicted' in existing_df.columns:
                        filter_conditions.append(existing_df['num_steps_predicted'].astype(str) == str(args.num_steps_to_predict))
                    if args.multi_step_eval and 'num_hypothesis' in existing_df.columns:
                         filter_conditions.append(existing_df['num_hypothesis'].astype(str) == str(args.n_hypothesis))  

                elif 'num_hypothesis' in existing_df.columns : # For non-ROTE bootstrap models like TT, NLLM
                    filter_conditions.append(existing_df['num_hypothesis'].astype(str) == str(args.n_hypothesis))
                
                matching_rows = existing_df[np.logical_and.reduce(filter_conditions)]
                
                if len(matching_rows) > 0:
                    start_epoch = matching_rows['epoch'].astype(int).max() + 1
                    print(f"Resuming from epoch {start_epoch}")
                else:
                    print("No matching rows found. Starting from epoch 0.")
        except pd.errors.EmptyDataError:
            print(f"CSV file {csv_path} is empty. Starting from epoch 0.")
        except Exception as e:
            print(f"Error reading CSV for resume: {e}. Starting from epoch 0.")

    model = None
    states = None

    print("Loading grid data")
    if args.human_data:
        print("Loading human data")
        dataloader_fn = make_dataloader_human
    else:
        dataloader_fn = make_dataloader
    

    if args.baseline_model == "AutoToM":
        from baselines.AutoToM.autoToM import AutoToM
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = AutoToM(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
        eval_fn = eval_autoToM
    elif args.baseline_model == "ROTE":
        from baselines.gridROTE import ROTEReasoner
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = ROTEReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, 
                           dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, 
                           num_hypothesis=args.n_hypothesis, group=args.group, two_stage=args.two_stage,
                           structured=args.structured, mode=args.mode, api_base=args.llm_server_url, api_key=args.llm_api_key)
        if not args.no_log:
            model.save_programs = True
            model.program_save_root = Path(f"generated_outputs/gridworld/run_{datetime.now():%y%m%d_%H%M%S}")
        if args.bootstrap:
            eval_fn = eval_fsm_bootstrap
        else:
            eval_fn = eval_fsm 
    elif args.baseline_model == "Oracle":
        from baselines.gridROTE import ROTEReasoner
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = ROTEReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, 
                           dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, 
                           num_hypothesis=args.n_hypothesis, group=args.group, two_stage=args.two_stage,
                           structured=args.structured, oracle=True, mode=args.mode, api_base=args.llm_server_url, api_key=args.llm_api_key)
        if not args.no_log:
            model.save_programs = True
            model.program_save_root = Path(f"generated_outputs/gridworld/run_{datetime.now():%y%m%d_%H%M%S}")
        if args.bootstrap:
            eval_fn = eval_fsm_bootstrap
        else:
            eval_fn = eval_fsm 
    elif args.baseline_model == 'NLLM':
        from baselines.basic_LLM import NaiveLLMReasoner
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = NaiveLLMReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, num_hypothesis=args.n_hypothesis, group=args.group, partnr=False, mode=args.mode, api_base=args.llm_server_url, api_key=args.llm_api_key)
        eval_fn = eval_naive_llm
    elif args.baseline_model == "BC":
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model, states = load_bc_models(args)
        if not states['params']:
            print("No BC models found or loaded. Please train models first or check paths.")
            return
        eval_fn = lambda a, d, m, s, ep_id, env: eval_bc(a, d, m, s, ep_id, env)
    else:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")

    fsm_names = os.listdir(f'generated_outputs/hand_designed')
    fsm_names = [f.replace('.txt', '') for f in fsm_names]
    num_agent_types = len(fsm_names)  # Total number of agent types (10)
    env = None
    assert args.dataset == "gridworld", "Gridworld epoch loop requires dataset=gridworld."
    
    # Setup for sequential mode
    get_config_and_agent_for_epoch = None  # Initialize function variable
    if args.loop_mode == "sequential":
        # Set CSV path in experiment folder
        if hasattr(model, 'program_save_root') and model.program_save_root:
            csv_path = model.program_save_root / "results.csv"
        else:
            # Fallback: create a run directory
            run_dir = Path(f"generated_outputs/gridworld/run_{datetime.now():%y%m%d_%H%M%S}")
            run_dir.mkdir(parents=True, exist_ok=True)
            csv_path = run_dir / "results.csv"
        os.makedirs(os.path.dirname(str(csv_path)), exist_ok=True)
        
        # Check for resume in sequential mode (after csv_path is set)
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                if not existing_df.empty:
                    # Find max epoch from CSV
                    if 'epoch' in existing_df.columns:
                        start_epoch = existing_df['epoch'].astype(int).max() + 1
                        print(f"Sequential mode: Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"Error reading CSV for resume in sequential mode: {e}. Starting from epoch 0.")
        
        # Generate all problem configs
        all_problem_configs = get_all_problem_configs()
        total_configs = len(all_problem_configs)  # 20 configs
        actual_evaluations = min(args.num_epochs, total_configs)
        print(f"Sequential mode: Will evaluate {actual_evaluations} problem config(s) (--num_epochs={args.num_epochs}, --num_agents_to_sample={args.num_agents_to_sample})")
        
        # Calculate which config and agent types to use for each epoch
        def get_config_and_agents_for_epoch(epoch_idx):
            """Get (num_blocks, num_walls, agent_indices_list) for a given epoch index.
            Each epoch uses a different problem config, and evaluates num_agents_to_sample agent types."""
            if epoch_idx >= total_configs:
                return None, None, None  # Out of bounds
            num_blocks, num_walls = all_problem_configs[epoch_idx]
            # Select agent types: always use the first num_agents_to_sample agent types
            agent_indices = list(range(min(args.num_agents_to_sample, num_agent_types)))
            return num_blocks, num_walls, agent_indices
    else:
        # Random mode: CSV path already set above
        pass
    
    for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Epochs"):
        # Determine problem config and agent type(s) for this epoch
        if args.loop_mode == "sequential":
            num_blocks, num_walls, agent_indices_list = get_config_and_agents_for_epoch(epoch)
            if num_blocks is None:
                print(f"Epoch {epoch}: Out of problem configs, stopping.")
                break
            agent_str = f"Agents: {agent_indices_list}" if len(agent_indices_list) > 1 else f"Agent: {agent_indices_list[0]}"
            print(f"\nRunning epoch {epoch+1}/{args.num_epochs} - Problem: num_blocks={num_blocks}, num_walls={num_walls}, {agent_str}")
            # In sequential mode, evaluate num_agents_to_sample agent types per epoch
            agent_indices_to_use = agent_indices_list
        else:
            # Random mode: use default behavior
            num_blocks = None
            num_walls = None
            agent_indices_to_use = None
            print(f"\nRunning epoch {epoch+1}/{args.num_epochs}")
        
        results_to_save = []

        if args.baseline_model in ["ROTE", "Oracle"] and args.bootstrap:
            # Recreate dataloader for each seed to ensure fresh data
            if args.n_eval_seeds > 1:
                results_list = []
                for seed in range(args.n_eval_seeds):
                    if args.loop_mode == "sequential":
                        dataloader = dataloader_fn(args, num_agents_to_sample=len(agent_indices_to_use), num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=epoch, num_blocks=num_blocks, num_walls=num_walls, agent_indices=agent_indices_to_use)
                    else:
                        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=epoch)
                    result = eval_fn(args, dataloader, model, episode_id=epoch)
                    results_list.append(result)
                # Average results
                accuracies_dict = {}
                program_lengths_dict = {}
                action_times_dict = {}
                first_step_accuracies_dict = {}
                accuracies_after_flip_dict = {}
                avg_prediction_times = []
                avg_generation_times = []
                avg_matching_states_list = []
                mean_equal_actions_list = []
                agent_id = results_list[-1][6]  # Use last agent_id
                agent_type_info = results_list[-1][10] if len(results_list[-1]) > 10 else {}  # Use last agent_type_info
                
                for result in results_list:
                    accuracies_dict_seed, program_lengths_dict_seed, action_times_dict_seed, avg_prediction_time_seed, first_step_accuracies_dict_seed, avg_generation_time_seed, _, accuracies_after_flip_dict_seed, avg_matching_states_seed, mean_equal_actions_seed, _ = result
                    avg_prediction_times.append(avg_prediction_time_seed)
                    avg_generation_times.append(avg_generation_time_seed)
                    avg_matching_states_list.append(avg_matching_states_seed)
                    mean_equal_actions_list.append(mean_equal_actions_seed)
                    
                    for n_hyp in accuracies_dict_seed.keys():
                        if n_hyp not in accuracies_dict:
                            accuracies_dict[n_hyp] = []
                            program_lengths_dict[n_hyp] = []
                            first_step_accuracies_dict[n_hyp] = []
                            accuracies_after_flip_dict[n_hyp] = []
                        accuracies_dict[n_hyp].append(accuracies_dict_seed[n_hyp])
                        program_lengths_dict[n_hyp].append(program_lengths_dict_seed.get(n_hyp, 0.0))
                        first_step_accuracies_dict[n_hyp].append(first_step_accuracies_dict_seed.get(n_hyp, 0.0))
                        accuracies_after_flip_dict[n_hyp].append(accuracies_after_flip_dict_seed.get(n_hyp, 0.0))
                
                # Average dictionaries
                accuracies_dict = {k: np.mean(v) for k, v in accuracies_dict.items()}
                program_lengths_dict = {k: np.mean(v) for k, v in program_lengths_dict.items()}
                first_step_accuracies_dict = {k: np.mean(v) for k, v in first_step_accuracies_dict.items()}
                accuracies_after_flip_dict = {k: np.mean(v) for k, v in accuracies_after_flip_dict.items()}
                avg_prediction_time = np.mean(avg_prediction_times)
                avg_generation_time = np.mean(avg_generation_times)
                avg_matching_states = np.mean(avg_matching_states_list)
                mean_equal_actions = np.mean(mean_equal_actions_list)
            else:
                if args.loop_mode == "sequential":
                    dataloader = dataloader_fn(args, num_agents_to_sample=len(agent_indices_to_use), num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=epoch, num_blocks=num_blocks, num_walls=num_walls, agent_indices=agent_indices_to_use)
                else:
                    dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=epoch)
                accuracies_dict, program_lengths_dict, action_times_dict, avg_prediction_time, first_step_accuracies_dict, avg_generation_time, agent_id, accuracies_after_flip_dict, avg_matching_states, mean_equal_actions, agent_type_info = eval_fn(args, dataloader, model, episode_id=epoch)
            
            # Save agent type information to JSON file (inside epoch directory)
            if agent_type_info and hasattr(model, 'program_save_root') and model.program_save_root:
                json_output = {
                    "epoch": epoch,
                    "agent_types": agent_type_info
                }
                # Create epoch directory if it doesn't exist
                epoch_dir = model.program_save_root / f"epoch_{epoch}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                # Save JSON inside the epoch directory
                json_file = epoch_dir / f"epoch_{epoch}_agent_types.json"
                json_file.write_text(json.dumps(json_output, indent=2))
                print(f"Saved agent type information to {json_file}")
            
            if args.multi_step_eval:
                for n_hyp, accuracy_val in accuracies_dict.items():
                    include_prediction_time = True
                    if os.path.exists(csv_path):
                        try:
                            existing_cols = pd.read_csv(csv_path, nrows=0).columns
                            include_prediction_time = 'avg_prediction_time' in existing_cols
                        except:
                            pass

                    result = {
                        'gt_fsm_id': agent_id,
                        'model': args.baseline_model,
                        'accuracy': accuracy_val,
                        'group': args.group,
                        'num_agents_evaluated': args.num_agents_to_sample,
                        'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                        'epoch': epoch,
                        'llm_model': args.model_name,
                        'num_hypothesis': n_hyp,
                        'two_stage': args.two_stage,
                        'structured': args.structured,
                        'rejuvenation': args.rejuvenation,
                        'top_k': args.top_k,
                        'program_length': program_lengths_dict.get(n_hyp, 0.0),
                        'multi_step_eval': True,
                        'num_steps_predicted': args.num_steps_to_predict,
                        'first_step_accuracy': first_step_accuracies_dict.get(n_hyp, 0.0),
                        'accuracy_after_flip': accuracies_after_flip_dict.get(n_hyp, 0.0),
                        'avg_matching_states': avg_matching_states,
                        'mean_equal_actions': mean_equal_actions,
                    }
                    
                    # Add num_blocks and num_walls if in sequential mode or if they're missing from CSV
                    if args.loop_mode == "sequential":
                        result['num_blocks'] = num_blocks
                        result['num_walls'] = num_walls
                    elif csv_path and os.path.exists(csv_path):
                        # Check if columns exist, if not add None
                        try:
                            existing_cols = pd.read_csv(csv_path, nrows=0).columns
                            if 'num_blocks' not in existing_cols:
                                result['num_blocks'] = None
                            if 'num_walls' not in existing_cols:
                                result['num_walls'] = None
                        except:
                            pass
                    
                    if include_prediction_time:
                        result['avg_prediction_time'] = avg_prediction_time
                        result['avg_generation_time'] = avg_generation_time
                    results_to_save.append(result)
            else:
                for n_hyp, accuracy_val in accuracies_dict.items():
                    include_prediction_time = True
                            
                    result = {
                        'gt_fsm_id': agent_id,
                        'model': args.baseline_model,
                        'accuracy': accuracy_val,
                        'group': args.group,
                        'num_agents_evaluated': args.num_agents_to_sample,
                        'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                        'epoch': epoch,
                        'llm_model': args.model_name,
                        'num_hypothesis': n_hyp,
                        'two_stage': args.two_stage,
                        'structured': args.structured,
                        'rejuvenation': args.rejuvenation,
                        'top_k': args.top_k,
                        'program_length': program_lengths_dict.get(n_hyp, 0.0),
                        'multi_step_eval': False,
                        'num_steps_predicted': 1,
                        'avg_matching_states': avg_matching_states,
                        'mean_equal_actions': mean_equal_actions,
                    }
                    
                    # Add num_blocks and num_walls if in sequential mode or if they're missing from CSV
                    if args.loop_mode == "sequential":
                        result['num_blocks'] = num_blocks
                        result['num_walls'] = num_walls
                    elif csv_path and os.path.exists(csv_path):
                        # Check if columns exist, if not add None
                        try:
                            existing_cols = pd.read_csv(csv_path, nrows=0).columns
                            if 'num_blocks' not in existing_cols:
                                result['num_blocks'] = None
                            if 'num_walls' not in existing_cols:
                                result['num_walls'] = None
                        except:
                            pass
                    
                    if include_prediction_time:
                        result['avg_prediction_time'] = avg_prediction_time
                        
                    results_to_save.append(result)
        else:
            if args.baseline_model in ["BC"]:
                if args.n_eval_seeds > 1:
                    # Run multiple seeds and average
                    accuracies_list = []
                    avg_prediction_times_list = []
                    first_step_accuracies_list = []
                    avg_matching_states_list = []
                    mean_equal_actions_list = []
                    accuracy_after_flip_list = []
                    for seed in range(args.n_eval_seeds):
                        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=epoch)
                        if args.multi_step_eval and args.baseline_model == "BC":
                            result = eval_fn(args, dataloader, model, states, epoch, env)
                            acc, pred_time, first_step, match_states, mean_actions, acc_after_flip, env, gt_id = result
                            accuracies_list.append(acc)
                            avg_prediction_times_list.append(pred_time)
                            first_step_accuracies_list.append(first_step)
                            avg_matching_states_list.append(match_states)
                            mean_equal_actions_list.append(mean_actions)
                            accuracy_after_flip_list.append(acc_after_flip)
                            gt_agent_script_id = gt_id  # Use last
                        else:
                            acc, pred_time = eval_fn(args, dataloader, model, states, epoch)
                            accuracies_list.append(acc)
                            avg_prediction_times_list.append(pred_time)
                    current_accuracy = np.mean(accuracies_list)
                    avg_prediction_time = np.mean(avg_prediction_times_list)
                    if args.multi_step_eval and args.baseline_model == "BC":
                        first_step_accuracy = np.mean(first_step_accuracies_list)
                        avg_matching_states = np.mean(avg_matching_states_list)
                        mean_equal_actions = np.mean(mean_equal_actions_list)
                        accuracy_after_flip = np.mean(accuracy_after_flip_list)
                    else:
                        first_step_accuracy = None
                        gt_agent_script_id = None
                        accuracy_after_flip = None
                        avg_matching_states = None
                        mean_equal_actions = None
                else:
                    if args.multi_step_eval and args.baseline_model == "BC":
                        current_accuracy, avg_prediction_time, first_step_accuracy, avg_matching_states, mean_equal_actions, accuracy_after_flip, env, gt_agent_script_id = eval_fn(args, dataloader, model, states, epoch, env)
                    else:
                        current_accuracy, avg_prediction_time = eval_fn(args, dataloader, model, states, epoch)
                        first_step_accuracy = None
                        gt_agent_script_id = None
                        accuracy_after_flip = None
                        avg_matching_states = None
                        mean_equal_actions = None
            else:
                if args.n_eval_seeds > 1:
                    # Run multiple seeds and average
                    accuracies_list = []
                    avg_prediction_times_list = []
                    first_step_accuracies_list = []
                    accuracy_after_flip_list = []
                    avg_matching_states_list = []
                    mean_equal_actions_list = []
                    for seed in range(args.n_eval_seeds):
                        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=epoch)
                        if args.multi_step_eval and args.baseline_model in ["AutoToM", "NLLM"]:
                            result = eval_fn(args, dataloader, model, episode_id=epoch, env=env)
                            acc, pred_time, first_step, gt_id, acc_after_flip, match_states, mean_actions, env = result
                            accuracies_list.append(acc)
                            avg_prediction_times_list.append(pred_time)
                            first_step_accuracies_list.append(first_step)
                            accuracy_after_flip_list.append(acc_after_flip)
                            avg_matching_states_list.append(match_states)
                            mean_equal_actions_list.append(mean_actions)
                            gt_agent_script_id = gt_id  # Use last
                        else:
                            acc, pred_time = eval_fn(args, dataloader, model, episode_id=epoch)
                            accuracies_list.append(acc)
                            avg_prediction_times_list.append(pred_time)
                    current_accuracy = np.mean(accuracies_list)
                    avg_prediction_time = np.mean(avg_prediction_times_list)
                    if args.multi_step_eval and args.baseline_model in ["AutoToM", "NLLM"]:
                        first_step_accuracy = np.mean(first_step_accuracies_list)
                        accuracy_after_flip = np.mean(accuracy_after_flip_list)
                        avg_matching_states = np.mean(avg_matching_states_list)
                        mean_equal_actions = np.mean(mean_equal_actions_list)
                    else:
                        first_step_accuracy = None
                        gt_agent_script_id = None
                        accuracy_after_flip = None
                        avg_matching_states = None
                        mean_equal_actions = None
                else:
                    if args.multi_step_eval and args.baseline_model in ["AutoToM", "NLLM"]:
                        current_accuracy, avg_prediction_time, first_step_accuracy, gt_agent_script_id, accuracy_after_flip, avg_matching_states, mean_equal_actions, env = eval_fn(args, dataloader, model, episode_id=epoch, env=env)
                    else:
                        current_accuracy, avg_prediction_time = eval_fn(args, dataloader, model, episode_id=epoch)
                        first_step_accuracy = None
                        gt_agent_script_id = None
                        accuracy_after_flip = None
                        avg_matching_states = None
                        mean_equal_actions = None
            include_prediction_time = True
            if csv_path and os.path.exists(csv_path):
                try:
                    existing_cols = pd.read_csv(csv_path, nrows=0).columns
                    include_prediction_time = 'avg_prediction_time' in existing_cols
                except:
                    pass
            result = {
                'gt_fsm_id': gt_agent_script_id,
                'model': args.baseline_model,
                'accuracy': current_accuracy,
                'group': args.group,
                'num_agents_evaluated': args.num_agents_to_sample if args.loop_mode != "sequential" else 1,
                'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                'epoch': epoch,
                'llm_model': args.model_name if args.baseline_model in ['AutoToM', 'ROTE', 'NLLM'] else 'N/A',
                'num_hypothesis': args.n_hypothesis if args.baseline_model in ['ROTE', 'NLLM'] else ('ensemble' if args.baseline_model in ["BC"] else 'N/A'),
                'two_stage': args.two_stage if args.baseline_model == 'ROTE' else False,
                'structured': args.structured if args.baseline_model == 'ROTE' else "False",
                'rejuvenation': args.rejuvenation if args.baseline_model == 'ROTE' else False,
                'top_k': args.top_k if args.baseline_model == 'ROTE' else 0,
                'program_length': getattr(model, 'weighted_program_length', 0) if args.baseline_model == 'ROTE' and not args.bootstrap else 0,
                'multi_step_eval': args.multi_step_eval,
                'num_steps_predicted': args.num_steps_to_predict,
                'avg_matching_states': avg_matching_states,
                'mean_equal_actions': mean_equal_actions,
            }
            
            # Add num_blocks and num_walls if in sequential mode or if they're missing from CSV
            if args.loop_mode == "sequential":
                result['num_blocks'] = num_blocks
                result['num_walls'] = num_walls
            elif csv_path and os.path.exists(csv_path):
                # Check if columns exist, if not add None
                try:
                    existing_cols = pd.read_csv(csv_path, nrows=0).columns
                    if 'num_blocks' not in existing_cols:
                        result['num_blocks'] = None
                    if 'num_walls' not in existing_cols:
                        result['num_walls'] = None
                except:
                    pass
            
            if first_step_accuracy is not None:
                result['first_step_accuracy'] = first_step_accuracy
            if accuracy_after_flip is not None:
                result['accuracy_after_flip'] = accuracy_after_flip
            if include_prediction_time:
                result['avg_prediction_time'] = avg_prediction_time
            if accuracy_after_flip is not None:
                result['accuracy_after_flip'] = accuracy_after_flip
            results_to_save.append(result)

        if results_to_save:
            df = pd.DataFrame(results_to_save)
            
            # Ensure csv_path is set (for sequential mode, it's set above)
            if csv_path is None and args.loop_mode == "sequential":
                if hasattr(model, 'program_save_root') and model.program_save_root:
                    csv_path = model.program_save_root / "results.csv"
                else:
                    csv_path = Path(f"generated_outputs/gridworld/run_{datetime.now():%y%m%d_%H%M%S}") / "results.csv"
                os.makedirs(os.path.dirname(str(csv_path)), exist_ok=True)
            
            if csv_path and os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path)
                    if not existing_df.empty:
                        existing_cols = existing_df.columns.tolist()
                        # Add num_blocks and num_walls if they don't exist in existing CSV
                        for col in ['num_blocks', 'num_walls']:
                            if col not in existing_cols and col in df.columns:
                                existing_cols.append(col)
                        for col in existing_cols:
                            if col not in df.columns:
                                df[col] = None
                        df = df[existing_cols]
                except Exception as e:
                    print(f"Error matching columns with existing CSV: {e}")
            
            if csv_path:
                if os.path.exists(csv_path) and start_epoch <= epoch:
                    try:
                        header_needed = pd.read_csv(csv_path, nrows=0).empty
                    except pd.errors.EmptyDataError:
                        header_needed = True
                    except FileNotFoundError:
                        header_needed = True

                    df.to_csv(csv_path, mode='a', header=header_needed, index=False)
                else:
                    df.to_csv(csv_path, index=False)
                print(f"Saved results for epoch {epoch+1} to {csv_path}")
        else:
            print(f"No results to save for epoch {epoch+1}")

    if args.baseline_model == 'AutoToM':
        os.environ['CURRENT_MODEL_NAME'] = ''


def run_choice13k(args, log_fn=None):
    """Evaluate on Choice13k dataset using a simple prompt-driven LLM choice predictor."""
    experiments = get_choice13k_experiments(n_participants=args.num_agents_to_sample)
    client_kwargs = {}
    if args.mode == "local":
        client_kwargs = {"api_key": args.llm_api_key, "base_url": args.llm_server_url}
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()

    def predict_choice(prompt: str):
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        return response.choices[0].message.content.strip()

    overall_total = 0
    overall_correct = 0
    per_participant = []

    for participant_idx, exp in enumerate(experiments):
        participant_total = 0
        participant_correct = 0
        for block in exp.blocks:
            instruction = exp.instruction.strip()
            for trial in block.trials:
                participant_total += 1
                option_keys = block.option_keys
                gt_key = option_keys[trial.action]
                prompt = f"{instruction}\n\n{trial.history}\n\nWhich option do you choose? Reply with only the option letter."
                pred_text = predict_choice(prompt)
                pred_key = None
                for k in option_keys:
                    if k in pred_text:
                        pred_key = k
                        break
                if pred_key == gt_key:
                    participant_correct += 1
        overall_total += participant_total
        overall_correct += participant_correct
        acc = participant_correct / participant_total if participant_total > 0 else 0
        per_participant.append({"participant": participant_idx, "accuracy": acc})
        print(f"Participant {participant_idx}: accuracy {acc:.4f} ({participant_correct}/{participant_total})")

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    summary = {
        "dataset": "choice13k",
        "model": args.model_name,
        "mode": args.mode,
        "accuracy": overall_acc,
        "total_trials": overall_total,
    }
    print(f"Choice13k overall accuracy: {overall_acc:.4f} ({overall_correct}/{overall_total})")
    if log_fn:
        log_fn(summary, step=0)

    # Initialize random key for parameter initialization
    rng_key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create CSV path for results
    group_extension = "_group" if args.group else ""
    two_stage_extension = "_two_stage" if args.two_stage else ""
    structured_extension = f"_structured_{args.structured}" if args.structured != "False" else ""
    rejuvenation_extension = "_rejuvenation" if args.rejuvenation else ""
    human_data_extension = "_human_data" if args.human_data else ""
    
    if args.baseline_model == "ROTE" and args.bootstrap:
        if args.multi_step_eval:
            csv_path = f"results/{args.baseline_model}/results_rote_bootstrap_multistep{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}{human_data_extension}_topk{args.top_k}_steps{args.num_steps_to_predict}_actionTime.csv"
        else: # Single-step ROTE bootstrap
            csv_path = f"results/{args.baseline_model}/results_rote_bootstrap_singlestep{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}{human_data_extension}_topk{args.top_k}.csv"
    else: # Non-bootstrap ROTE or other models
        csv_path = f"results/{args.baseline_model}/results_grid_{args.baseline_model}_{args.n_hypothesis}hyp{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}{human_data_extension}.csv"
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if CSV exists and determine starting epoch
    start_epoch = 0
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                filter_conditions = [
                    (existing_df['model'].astype(str) == str(args.baseline_model)),
                    (existing_df['group'].astype(str) == str(args.group)),
                    (existing_df['num_agents_evaluated'].astype(str) == str(args.num_agents_to_sample)),
                    (existing_df['datapoints_per_agent'].astype(str) == str(args.num_datapoints_per_agent_to_sample)),
                    (existing_df['llm_model'].astype(str) == str(args.model_name if args.baseline_model in ['TT', 'AutoToM', 'ROTE', 'NLLM'] else 'N/A'))
                ]
                
                if 'two_stage' in existing_df.columns:
                    filter_conditions.append(existing_df['two_stage'].astype(str) == str(args.two_stage))
                if 'structured' in existing_df.columns:
                    filter_conditions.append(existing_df['structured'].astype(str) == str(args.structured))
                if 'rejuvenation' in existing_df.columns:
                    filter_conditions.append(existing_df['rejuvenation'].astype(str) == str(args.rejuvenation))
                if 'top_k' in existing_df.columns: # Relevant for ROTE bootstrap
                     filter_conditions.append(existing_df['top_k'].astype(str) == str(args.top_k))

                if args.baseline_model == "ROTE" and args.bootstrap:
                    if 'multi_step_eval' in existing_df.columns:
                        filter_conditions.append(existing_df['multi_step_eval'].astype(str) == str(args.multi_step_eval))
                    if args.multi_step_eval and 'num_steps_predicted' in existing_df.columns:
                        filter_conditions.append(existing_df['num_steps_predicted'].astype(str) == str(args.num_steps_to_predict))
                    # For single-step bootstrap, num_hypothesis varies, so we don't filter by it here for resuming.
                    # For multi-step, n_hypothesis is fixed by args.n_hypothesis for the rollout.
                    if args.multi_step_eval and 'num_hypothesis' in existing_df.columns:
                         filter_conditions.append(existing_df['num_hypothesis'].astype(str) == str(args.n_hypothesis))  

                elif 'num_hypothesis' in existing_df.columns : # For non-ROTE bootstrap models like TT, NLLM
                    filter_conditions.append(existing_df['num_hypothesis'].astype(str) == str(args.n_hypothesis))
                
                matching_rows = existing_df[np.logical_and.reduce(filter_conditions)]
                
                if len(matching_rows) > 0:
                    start_epoch = matching_rows['epoch'].astype(int).max() + 1
                    print(f"Resuming from epoch {start_epoch}")
                else:
                    print("No matching rows found. Starting from epoch 0.")
        except pd.errors.EmptyDataError:
            print(f"CSV file {csv_path} is empty. Starting from epoch 0.")
        except Exception as e:
            print(f"Error reading CSV for resume: {e}. Starting from epoch 0.")

    # Initialize model, dataloader, and evaluation function
    model = None
    states = None # For BC

    print("Loading grid data")
    if args.human_data:
        print("Loading human data")
        dataloader_fn = make_dataloader_human
    else:
        dataloader_fn = make_dataloader
    

    if args.baseline_model == "AutoToM":
        from baselines.AutoToM.autoToM import AutoToM
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = AutoToM(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
        eval_fn = eval_autoToM
    elif args.baseline_model == "ROTE":
        from baselines.gridROTE import ROTEReasoner
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = ROTEReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, 
                           dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, 
                           num_hypothesis=args.n_hypothesis, group=args.group, two_stage=args.two_stage,
                           structured=args.structured, mode=args.mode, api_base=args.llm_server_url, api_key=args.llm_api_key)
        if args.bootstrap:
            eval_fn = eval_fsm_bootstrap # This function handles multi_step_eval internally
        else:
            eval_fn = eval_fsm 
    elif args.baseline_model == "Oracle":
        from baselines.gridROTE import ROTEReasoner
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = ROTEReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, 
                           dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, 
                           num_hypothesis=args.n_hypothesis, group=args.group, two_stage=args.two_stage,
                           structured=args.structured, oracle=True, mode=args.mode, api_base=args.llm_server_url, api_key=args.llm_api_key)
        if args.bootstrap:
            eval_fn = eval_fsm_bootstrap # This function handles multi_step_eval internally
        else:
            eval_fn = eval_fsm 
    elif args.baseline_model == 'NLLM':
        from baselines.basic_LLM import NaiveLLMReasoner
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = NaiveLLMReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, num_hypothesis=args.n_hypothesis, group=args.group, partnr=False, mode=args.mode, api_base=args.llm_server_url, api_key=args.llm_api_key) # Assuming partnr=False for this context
        eval_fn = eval_naive_llm
    elif args.baseline_model == "BC":
        # args.as_images = True
        dataloader = dataloader_fn(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model, states = load_bc_models(args)
        if not states['params']: # Check if any models were loaded
            print("No BC models found or loaded. Please train models first or check paths.")
            return
        eval_fn = lambda a, d, m, s, ep_id, env: eval_bc(a, d, m, s, ep_id, env) # eval_bc uses 'states'
    else:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")

    # Run evaluation for each epoch and save results
    fsm_names = os.listdir(f'generated_outputs/hand_designed')
    fsm_names = [f.replace('.txt', '') for f in fsm_names]
    env = None
    assert args.dataset == "gridworld", "Gridworld epoch loop requires dataset=gridworld."
    for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Epochs"):
        print(f"\nRunning epoch {epoch+1}/{args.num_epochs}")
        
        results_to_save = []

        if args.baseline_model in ["ROTE", "Oracle"] and args.bootstrap:
            # eval_rote_bootstrap returns (accuracies_dict, program_lengths_dict)
            accuracies_dict, program_lengths_dict, action_times_dict, avg_prediction_time, first_step_accuracies_dict, avg_generation_time, agent_id, accuracies_after_flip_dict, avg_matching_states, mean_equal_actions = eval_fn(args, dataloader, model, episode_id=epoch)
            
            if args.multi_step_eval:
                for n_hyp, accuracy_val in accuracies_dict.items():
                    # Check if existing CSV has avg_prediction_time column
                    include_prediction_time = True
                    if os.path.exists(csv_path):
                        try:
                            existing_cols = pd.read_csv(csv_path, nrows=0).columns
                            include_prediction_time = 'avg_prediction_time' in existing_cols
                        except:
                            pass

                    result = {
                        'gt_fsm_id': agent_id,
                        'model': args.baseline_model,
                        'accuracy': accuracy_val,
                        'group': args.group,
                        'num_agents_evaluated': args.num_agents_to_sample,
                        'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                        'epoch': epoch,
                        'llm_model': args.model_name,
                        'num_hypothesis': n_hyp, # This is k_hypotheses_to_use for rollout
                        'two_stage': args.two_stage,
                        'structured': args.structured,
                        'rejuvenation': args.rejuvenation,
                        'top_k': args.top_k,
                        'program_length': program_lengths_dict.get(n_hyp, 0.0),
                        'multi_step_eval': True,
                        'num_steps_predicted': args.num_steps_to_predict,
                        'first_step_accuracy': first_step_accuracies_dict.get(n_hyp, 0.0),
                        'accuracy_after_flip': accuracies_after_flip_dict.get(n_hyp, 0.0),
                        'avg_matching_states': avg_matching_states,
                        'mean_equal_actions': mean_equal_actions,
                    }
                    
                    # Only include avg_prediction_time if it's in the existing CSV
                    if include_prediction_time:
                        result['avg_prediction_time'] = avg_prediction_time
                        result['avg_generation_time'] = avg_generation_time
                    results_to_save.append(result)
            else: # Single-step ROTE bootstrap
                for n_hyp, accuracy_val in accuracies_dict.items():
                    # Check if existing CSV has avg_prediction_time column
                    include_prediction_time = True
                            
                    result = {
                        'gt_fsm_id': agent_id,
                        'model': args.baseline_model,
                        'accuracy': accuracy_val,
                        'group': args.group,
                        'num_agents_evaluated': args.num_agents_to_sample,
                        'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                        'epoch': epoch,
                        'llm_model': args.model_name,
                        'num_hypothesis': n_hyp,
                        'two_stage': args.two_stage,
                        'structured': args.structured,
                        'rejuvenation': args.rejuvenation,
                        'top_k': args.top_k,
                        'program_length': program_lengths_dict.get(n_hyp, 0.0),
                        'multi_step_eval': False,
                        'num_steps_predicted': 1,
                        'avg_matching_states': avg_matching_states,
                        'mean_equal_actions': mean_equal_actions,
                    }
                    
                    # Only include avg_prediction_time if it's in the existing CSV
                    if include_prediction_time:
                        result['avg_prediction_time'] = avg_prediction_time
                        
                    results_to_save.append(result)
            # wandb summary logging (bootstrap)
            accuracy_final = accuracies_dict.get(args.n_hypothesis) if hasattr(args, "n_hypothesis") else None
            first_step_final = first_step_accuracies_dict.get(args.n_hypothesis) if hasattr(args, "n_hypothesis") else None
            program_len_final = program_lengths_dict.get(args.n_hypothesis) if hasattr(args, "n_hypothesis") else None
            num_valid_programs = len(program_lengths_dict)
            expected = getattr(args, "n_hypothesis", num_valid_programs or 1)
            compile_failure_rate = max(0.0, (expected - num_valid_programs) / expected) if expected else None
            log_metrics = {
                "accuracy_final": accuracy_final,
                "first_step_accuracy_final": first_step_final,
                "avg_program_length": program_len_final,
                "num_valid_programs": num_valid_programs,
                "compile_failure_rate": compile_failure_rate,
                "avg_generation_time": avg_generation_time,
                "epoch": epoch,
            }
            _log_to_wandb(log_metrics, step=epoch)
        else: # Other models or non-bootstrap ROTE
            if args.baseline_model in ["BC"]:
                if args.multi_step_eval and args.baseline_model == "BC":
                    current_accuracy, avg_prediction_time, first_step_accuracy, avg_matching_states, mean_equal_actions, accuracy_after_flip, env, gt_agent_script_id = eval_fn(args, dataloader, model, states, epoch, env)
                else:
                    current_accuracy, avg_prediction_time = eval_fn(args, dataloader, model, states, epoch)
                    first_step_accuracy = None
                    gt_agent_script_id = None
                    accuracy_after_flip = None
                    avg_matching_states = None
                    mean_equal_actions = None
            else:
                if args.multi_step_eval and args.baseline_model in ["AutoToM", "NLLM"]:
                    current_accuracy, avg_prediction_time, first_step_accuracy, gt_agent_script_id, accuracy_after_flip, avg_matching_states, mean_equal_actions, env = eval_fn(args, dataloader, model, episode_id=epoch, env=env)
                else:
                    current_accuracy, avg_prediction_time = eval_fn(args, dataloader, model, episode_id=epoch)
                    first_step_accuracy = None
                    gt_agent_script_id = None
                    accuracy_after_flip = None
                    avg_matching_states = None
                    mean_equal_actions = None
            # Check if existing CSV has avg_prediction_time column
            include_prediction_time = True
            if os.path.exists(csv_path):
                try:
                    existing_cols = pd.read_csv(csv_path, nrows=0).columns
                    include_prediction_time = 'avg_prediction_time' in existing_cols
                except:
                    pass
            result = {
                'gt_fsm_id': gt_agent_script_id,
                'model': args.baseline_model,
                'accuracy': current_accuracy,
                'group': args.group,
                'num_agents_evaluated': args.num_agents_to_sample,
                'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                'epoch': epoch,
                'llm_model': args.model_name if args.baseline_model in ['AutoToM', 'ROTE', 'NLLM'] else 'N/A',
                'num_hypothesis': args.n_hypothesis if args.baseline_model in ['ROTE', 'NLLM'] else ('ensemble' if args.baseline_model in ["BC"] else 'N/A'),
                'two_stage': args.two_stage if args.baseline_model == 'ROTE' else False,
                'structured': args.structured if args.baseline_model == 'ROTE' else "False",
                'rejuvenation': args.rejuvenation if args.baseline_model == 'ROTE' else False,
                'top_k': args.top_k if args.baseline_model == 'ROTE' else 0, # or N/A
                'program_length': getattr(model, 'weighted_program_length', 0) if args.baseline_model == 'ROTE' and not args.bootstrap else 0, # Placeholder
                'multi_step_eval': args.multi_step_eval,
                'num_steps_predicted': args.num_steps_to_predict,
                'avg_matching_states': avg_matching_states,
                'mean_equal_actions': mean_equal_actions,
            }
            if first_step_accuracy is not None:
                result['first_step_accuracy'] = first_step_accuracy
            if accuracy_after_flip is not None:
                result['accuracy_after_flip'] = accuracy_after_flip
            # Only include avg_prediction_time if it's in the existing CSV
            if include_prediction_time:
                result['avg_prediction_time'] = avg_prediction_time
            if accuracy_after_flip is not None:
                result['accuracy_after_flip'] = accuracy_after_flip
            results_to_save.append(result)
            # wandb summary logging (non-bootstrap)
            log_metrics = {
                "accuracy_final": current_accuracy,
                "first_step_accuracy_final": first_step_accuracy,
                "avg_program_length": result.get('program_length'),
                "num_valid_programs": None,
                "compile_failure_rate": None,
                "avg_generation_time": avg_generation_time if 'avg_generation_time' in locals() else None,
                "epoch": epoch,
            }
            _log_to_wandb(log_metrics, step=epoch)

        if results_to_save:
            if wandb_enabled:
                for entry in results_to_save:
                    entry_with_mode = dict(entry)
                    entry_with_mode.setdefault('mode', args.mode)
                    _log_to_wandb(entry_with_mode, step=epoch)

            df = pd.DataFrame(results_to_save)
            
            # Check if we need to match columns with existing CSV
            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path)
                    if not existing_df.empty:
                        # Get columns from existing CSV
                        existing_cols = existing_df.columns.tolist()
                        
                        # Ensure our new dataframe has the same columns in the same order
                        for col in existing_cols:
                            if col not in df.columns:
                                df[col] = None  # Add missing columns with None values
                        
                        # Reorder columns to match existing CSV
                        df = df[existing_cols]
                except Exception as e:
                    print(f"Error matching columns with existing CSV: {e}")
            
            if os.path.exists(csv_path) and start_epoch <= epoch: # Append if file exists and we are not re-writing old epochs
                 # Check if header is needed
                try:
                    header_needed = pd.read_csv(csv_path, nrows=0).empty
                except pd.errors.EmptyDataError:
                    header_needed = True # File exists but is empty
                except FileNotFoundError: # Should not happen due to os.path.exists
                    header_needed = True

                df.to_csv(csv_path, mode='a', header=header_needed, index=False)
            else: # File does not exist, or we are writing the first epoch(s) after resuming from an empty/new file
                df.to_csv(csv_path, index=False)
            print(f"Saved results for epoch {epoch+1} to {csv_path}")
        else:
            print(f"No results to save for epoch {epoch+1}")

    if args.baseline_model == 'AutoToM': # Clear environment variable if set
        os.environ['CURRENT_MODEL_NAME'] = ''


def run_choice13k_mindascode(args, log_fn=None, participant_id=None):
    """
    MindAsCode evaluation on Choice13k:
    - generate programs from train trials
    - execute on train/test split
    - aggregate via bootstrap weighting
    
    Args:
        args: Command line arguments
        log_fn: Optional logging function for wandb
        participant_id: Optional specific participant ID to evaluate (0-indexed). 
                       If None, evaluates all participants.
    """
    from baselines.choice13k import (
    Choice13kProgramGenerator,
    compile_program,
    load_choice13k,
    split_trials,
    evaluate_program,
    aggregate_predictions,
)
    from baselines.choice13k.state_formatter import format_trials_to_text

    # If participant_id is specified, ensure we load enough participants to include it
    if participant_id is not None:
        # Ensure num_agents_to_sample is at least participant_id + 1 to include the requested participant
        if args.num_agents_to_sample <= participant_id:
            args.num_agents_to_sample = participant_id + 1
            print(f"Loading {args.num_agents_to_sample} participants to include participant_id {participant_id}")
    
    experiments = load_choice13k(args)
    
    # Filter to specific participant if requested
    if participant_id is not None:
        if participant_id >= len(experiments):
            raise ValueError(f"participant_id {participant_id} is out of range. Available participants: 0-{len(experiments)-1}")
        experiments = [experiments[participant_id]]
        print(f"Evaluating only participant {participant_id}")

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
    
    # Select prompt files based on prompt_mode
    if args.prompt_mode == "strict":
        prompt_subdir = os.path.join("prompts", "choice13k", "strict")
    else:  # non_strict (default)
        prompt_subdir = os.path.join("prompts", "choice13k")
    
    prompt_path = os.path.join(PROJECT_ROOT, prompt_subdir, "infer_single_choice.txt")
    code_template_path = os.path.join(PROJECT_ROOT, prompt_subdir, "single_code_template.txt")
    refinement_1_path = os.path.join(PROJECT_ROOT, "prompts", "refinement_1.txt")
    refinement_2_path = os.path.join(PROJECT_ROOT, "prompts", "refinement_2.txt")
    refinement_3_path = os.path.join(PROJECT_ROOT, "prompts", "refinement_3.txt")
    base_prompt = open(prompt_path).read()
    code_template = open(code_template_path).read()
    refinement_1 = open(refinement_1_path).read()
    refinement_2 = open(refinement_2_path).read()
    refinement_3 = open(refinement_3_path).read()

    client_kwargs = {}
    if args.mode == "local":
        client_kwargs = {"api_key": args.llm_api_key, "base_url": args.llm_server_url}
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    generator = Choice13kProgramGenerator(client, args.model_name, max_tokens=800)

    out_root = Path("generated_outputs/choice13k")
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = Path(f"generated_outputs/choice13k/run_{datetime.now():%y%m%d_%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)

    for idx, exp in enumerate(tqdm(experiments, desc="Participants")):
        # Use actual participant_id: if filtering was done, participant_id is set; otherwise use index
        if participant_id is not None:
            actual_participant_id = participant_id
        else:
            actual_participant_id = idx
        train_trials, test_trials, options = split_trials(exp)

        participant_dir = run_dir / f"participant_{actual_participant_id}"
        participant_dir.mkdir(parents=True, exist_ok=True)

        for epoch in tqdm(range(args.num_epochs), desc=f"Participant {actual_participant_id} epochs", leave=False):
            pass_dir = participant_dir / f"epoch_{epoch}"
            raw_dir = pass_dir / "raw"
            good_dir = pass_dir / "good"
            bad_dir = pass_dir / "bad_compile"
            prompt_dir = pass_dir / "prompt"
            pass_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(exist_ok=True)
            good_dir.mkdir(exist_ok=True)
            bad_dir.mkdir(exist_ok=True)
            prompt_dir.mkdir(exist_ok=True)

            n_programs = args.n_hypothesis
            state_text = format_trials_to_text(train_trials)
            prompt_text = f"{base_prompt}\n{state_text}\n{code_template}"
            (prompt_dir / "prompt.txt").write_text(prompt_text)
            program_codes = generator.generate_programs(prompt_text, n_programs)

            compiled = []
            train_scores = []
            for idx, code in enumerate(program_codes):
                (raw_dir / f"program_{idx}.txt").write_text(code or "")
                if not code:
                    (bad_dir / f"program_{idx}.reason.txt").write_text("empty_code")
                    continue
                choose_fn = compile_program(code)
                if choose_fn is None:
                    (bad_dir / f"program_{idx}.reason.txt").write_text("compile_failed")
                    continue
                train_eval = evaluate_program(choose_fn, train_trials)
                train_scores.append(train_eval["accuracy"])
                compiled.append((idx, code, choose_fn))
                (good_dir / f"program_{idx}.py").write_text(code)

            if not compiled:
                print(f"Participant {actual_participant_id} epoch {epoch}: no valid programs generated.")
                metrics = {
                    "participant": actual_participant_id,
                    "epoch": epoch,
                    "train_accuracy": 0.0,
                    "test_accuracy": 0.0,
                    "num_programs": 0,
                    "avg_program_length": 0.0,
                }
                (pass_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
                if log_fn:
                    log_fn({**metrics, "dataset": "choice13k"}, step=epoch)
                continue

            weights = np.array(train_scores, dtype=float)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            weights = weights / weights.sum()

            choose_fns = [c[2] for c in compiled]
            # Use the same ensemble aggregation method for both train and test
            train_accuracy = aggregate_predictions(choose_fns, weights, train_trials)
            test_accuracy = aggregate_predictions(choose_fns, weights, test_trials)
            avg_program_length = np.mean([len(code.splitlines()) for _, code, _ in compiled])

            metrics = {
                "participant": actual_participant_id,
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "num_programs": len(compiled),
                "avg_program_length": avg_program_length,
            }
            (pass_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            print(f"Participant {actual_participant_id} epoch {epoch}: train_acc={train_accuracy:.4f}, test_acc={test_accuracy:.4f}, programs={len(compiled)}")
            if log_fn:
                log_fn({**metrics, "dataset": "choice13k"}, step=epoch)
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()