# Set JAX memory allocation to grow as needed
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Set PyTorch to use expandable segments to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
torch.multiprocessing.set_start_method('spawn')
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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from environment import state_to_image_jit, AutomaticityEnv, State
import optax
from flax.training import train_state
from flax import struct
import pandas as pd

from baselines.ToMnet import ToMNet
from baselines.BC import BCNet

import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline models for automaticity.")
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="FSM",
        help="Baseline model to train. Currently only 'ToMnet' and 'BC' are implemented."
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
    parser.add_argument('--num_datapoints_per_agent_to_sample', type=int, default=1, help='Number of datapoints per agent to sample from the dataset.')
    parser.add_argument('--num_agents', type=int, default=20, help='Number of agents in the dataset.')
    parser.add_argument('--num_datapoints_per_agent', type=int, default=100, help='Number of datapoints per agent in the dataset.')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps in the dataset.')
    parser.add_argument('--env_size', type=int, default=10, help='Size of the environment.')
    # parser.add_argument('--num_blocks', type=int, default=10, help='Number of blocks in the dataset.')
    # parser.add_argument('--num_walls', type=int, default=10, help='Number of walls in the dataset.')

    parser.add_argument('--as_images', type=bool, default=False, help='Whether to load the data as images.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save the model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--n_hypothesis', type=int, default=25, help='Number of hypothesis for thought trace.')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", help='Name of the model to use.')  # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct or meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of tensor parallel size.')
    parser.add_argument('--dtype', type=str, default="float16", help='Data type.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization.')
    parser.add_argument('--overfit', type=bool, default=False, help='Whether to overfit on a single environment.')
    parser.add_argument('--bootstrap', action='store_true', help='Whether to use bootstrapping for hypothesis evaluation')
    parser.add_argument('--two_stage', action='store_true', help='Whether to use two-stage approach for FSM reasoning')
    parser.add_argument('--structured', type=str, default="False", choices=["False", "p1", "p2"], 
                        help='Structured prompting type for FSM reasoning: False, p1, or p2')
    parser.add_argument('--rejuvenation', action='store_true', help='Use rejuvenation for FSM model')
    parser.add_argument('--rejuvenation_threshold', type=float, default=-10, help='Threshold for rejuvenation')
    parser.add_argument('--max_rejuvenation_attempts', type=int, default=5, help='Maximum number of rejuvenation attempts')
    parser.add_argument('--top_k', type=int, default=0, help='If > 0, only average over the top k most likely hypotheses')
    parser.add_argument('--multi_step_eval', type=bool, default=True, help='Perform multi-step evaluation for FSM')
    parser.add_argument('--num_steps_to_predict', type=int, default=5, help='Number of future steps to predict in multi-step eval')
    args = parser.parse_args()
    
    # Check if the selected baseline model is implemented
    if args.baseline_model not in ["ToMnet", 'BC', 'TT', 'AutoToM', 'FSM', 'NLLM']:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    if args.baseline_model == 'AutoToM':
        os.environ['CURRENT_MODEL_NAME'] = args.model_name
    
    return args


def make_dataloader(args, num_agents_to_sample: int = 2, num_datapoints_per_agent_to_sample: int = 20, overfit: bool = False, training: bool = False, epoch: int = 0):
    """Load data from the dataset folders."""
    data_path = args.data_path
    as_images = args.as_images

    i = epoch
    j = -1
    # Create all possible combinations of num_blocks and num_walls
    block_combo = []
    for num_blocks in range(2, 10, 2):  # From 2 to 8 in steps of 2
        for num_walls in range(2, 22, 2):  # From 2 to 20 in steps of 2
            block_combo.append((num_blocks, num_walls))


    
    while True:
        j += 1

        if j >= len(block_combo):
            yield None
            break

        num_blocks, num_walls = block_combo[j]

        # if not overfit:
        #     i += 1
        #     num_blocks = random.choice(list(range(2, 10, 2)))
        #     num_walls = random.choice(list(range(2, 22, 2)))
        # else:
        #     num_blocks = 2
        #     num_walls = 2

        data_folder = f"{data_path}/num_blocks{num_blocks}/num_walls{num_walls}"
        extension = "_group" if args.group else ""
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

        
        # Process data in smaller chunks to reduce memory usage
        sampled_data = {}
        sampled_data['actions'] = loaded_data['actions'][:, :, :args.num_steps//2]
        sampled_data['agent_ids'] = loaded_data['agent_ids'][:, :, :args.num_steps//2]
        
        # Process states separately to reduce peak memory usage
        sampled_states = {}
        for key in loaded_data['states']:
            sampled_states[key] = loaded_data['states'][key][:, :, :args.num_steps//2]

        # Free memory
        del loaded_data
        
        sampled_data['states'] = sampled_states

        yield sampled_data
        del sampled_data

task_list = [
    'Always move right',
    'Wander randomly without any specific direction',
    'Always pick up the nearest block',
    'Move in a vertical line (up and down)',
    'Bounce off walls without moving beyond them',
    'Stay in place',
    'Always pick up purple blocks',
    'Only pick up the first block encountered',
    'Move towards the farthest block each time',
    'Follow a clockwise square pattern',
    'Snake through the grid (right, up, left, down)',
    'Collect blocks of a specific color',
    'Move left if possible, otherwise right',
    'Move in an L-shape pattern',
    'Oscillate between two points',
    'Follow a path to collect all blocks of a specific color',
    'Create a spiral movement pattern',
    'Move diagonally towards blocks',
    'Return to a specific location when possible',
    'Maximize the number of blocks collected frontally',
]

def main():
    args = parse_args()
    dataloader = make_dataloader(args)
    data = next(dataloader)

    per_script_data = {}
    per_script_data_var = {}
    while data is not None:
        print("running")

        actions = data['actions'][:, :, :, 0]
        # Calculate variance across the final (time) dimension
        action_vars = jnp.var(actions, axis=-1)  # Shape: (num_agents, num_datapoints)
        # mean variance in a trajectory
        action_vars = jnp.mean(action_vars, axis=-1)

        
        old_actions = actions[:, :, :-1]
        new_actions = actions[:, :, 1:]

        same_actions = (old_actions == new_actions)
        same_actions = same_actions.sum(axis=-1)
        same_actions = same_actions / old_actions.shape[-1]
        same_actions = same_actions.mean(axis=1)

        
        for script_id in range(actions.shape[0]):
            if script_id not in per_script_data:
                per_script_data[script_id] = []
                per_script_data_var[script_id] = []
            per_script_data[script_id].append(same_actions[script_id])
            per_script_data_var[script_id].append(action_vars[script_id])

        
        data = next(dataloader)

    # Calculate mean and standard error for each script
    means = []
    std_errs = []
    var_means = []
    var_std_errs = []
    for script_id in range(len(per_script_data)):
        script_data = np.array(per_script_data[script_id])
        var_data = np.array(per_script_data_var[script_id])
        means.append(np.mean(script_data))
        std_errs.append(np.std(script_data) / np.sqrt(len(script_data)))
        var_means.append(np.mean(var_data))
        var_std_errs.append(np.std(var_data) / np.sqrt(len(var_data)))

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 8))
    x = np.arange(len(means))

    # Plot consistency on left subplot
    ax1.bar(x, means, yerr=std_errs, capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_list, rotation=45, ha='right')
    ax1.set_ylabel('Mean Action Consistency')
    ax1.set_title('Action Consistency by Task Type')

    # Plot variance on right subplot 
    ax2.bar(x, var_means, yerr=var_std_errs, capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_list, rotation=45, ha='right')
    ax2.set_ylabel('Mean Action Variance')
    ax2.set_title('Action Variance by Task Type')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save and show plot
    plt.savefig('action_analysis.png')
    plt.show()

main()