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

def initialize_environment(num_agents: int, num_blocks: int, num_walls: int, size: int = 10, max_steps: int = 100):
    env = AutomaticityEnv(num_agents=num_agents, size=size, max_steps=max_steps, num_blocks=num_blocks, num_walls=num_walls)
    return env

def make_dataloader(args, num_agents_to_sample: int = 2, num_datapoints_per_agent_to_sample: int = 20, overfit: bool = False, training: bool = False, epoch: int = 0):
    """Load data from the dataset folders."""
    data_path = args.data_path
    as_images = args.as_images

    i = epoch
    while True:
        if not overfit:
            i += 1
            num_blocks = random.choice(list(range(2, 10, 2)))
            num_walls = random.choice(list(range(2, 22, 2)))
        else:
            num_blocks = 2
            num_walls = 2

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
        # print(f"Loaded data from {data_file}")
                
        # Sample only what we need
        if overfit:
            agent_indices = jax.random.randint(jax.random.PRNGKey(i), (num_agents_to_sample,), minval=14, maxval=15)
        else:
            agent_indices = jax.random.randint(jax.random.PRNGKey(i), (num_agents_to_sample,), minval=0, maxval=loaded_data['states']['agent_locations'].shape[0])
        
        # print("Agent indices:", agent_indices.tolist())  # Convert to Python list for printing
            
        # agent_indices = jnp.array([8])
        # if not overfit:
        i += 1
        if training:  # for creating held out set
            batch_floor = 0
            batch_limit = int(loaded_data['states']['agent_locations'].shape[1] * 0.8)
        else:
            batch_floor = int(loaded_data['states']['agent_locations'].shape[1] * 0.8)
            batch_limit = loaded_data['states']['agent_locations'].shape[1]


        batch_indices = jax.random.randint(jax.random.PRNGKey(i), (num_datapoints_per_agent_to_sample,), minval=batch_floor, maxval=batch_limit)

        # if not overfit:
        i += 1
        # breakpoint()
        
        # Process data in smaller chunks to reduce memory usage
        sampled_data = {}
        sampled_data['actions'] = loaded_data['actions'][agent_indices][:, batch_indices][:, :, :args.num_steps//2]
        sampled_data['agent_ids'] = loaded_data['agent_ids'][agent_indices][:, batch_indices][:, :, :args.num_steps//2]
        
        # Process states separately to reduce peak memory usage
        sampled_states = {}
        for key in loaded_data['states']:
            sampled_states[key] = loaded_data['states'][key][agent_indices][:, batch_indices][:, :, :args.num_steps//2]

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
                img_size = args.env_size * 8
                tile_size = 8
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
            sampled_data['states'] = stacked_images.reshape(num_agents_to_sample, num_datapoints_per_agent_to_sample, args.num_steps//2, *stacked_images.shape[1:])
        else:
            sampled_data['states'] = sampled_states

        yield sampled_data
        del sampled_data

def eval_thoughtTrace(args, dataloader, model):
    for i in range(args.num_epochs):
        datapoint = next(dataloader)
        state = datapoint['states']
        actions = datapoint['actions']
        agent_ids = datapoint['agent_ids']

        predicted_final_action = model.predict_action(state, actions, agent_ids)

        breakpoint()

def eval_autoToM(args, dataloader, model, episode_id: int = 0):
    if args.multi_step_eval:
        # --- Multi-Step Evaluation Logic ---
        num_future_steps = args.num_steps_to_predict
        num_correct = 0
        num_total = 0
        total_prediction_time = 0
        num_predictions = 0
        
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)
        
        # Initialize environment (parameters will be set per datapoint)
        env_size = 10
        env_max_steps = num_future_steps + 5  # Sufficiently large

        for a_idx in tqdm(range(args.num_agents_to_sample), desc="Multi-step Eval Samples"):
            data_sample = jax.tree.map(lambda x: x[a_idx, -1, :15+num_future_steps], datapoint)
            
            initial_states_traj = jax.tree.map(lambda x: x[:15], data_sample['states'])
            initial_actions_traj = jax.tree.map(lambda x: x[:15], data_sample['actions'])
            
            gt_future_actions = data_sample['actions'][14:]  # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)

            num_env_agents = 4 if args.group else 1
            # Infer num_blocks and num_walls from the state
            state_at_t14 = jax.tree.map(lambda x: x[14], initial_states_traj)
            
            # Handle different possible shapes for block_locations
            if isinstance(state_at_t14['block_locations'], (np.ndarray, jnp.ndarray)):
                if state_at_t14['block_locations'].ndim == 1:
                    num_blocks = 1
                elif state_at_t14['block_locations'].ndim == 2:
                    num_blocks = state_at_t14['block_locations'].shape[0]
                else:
                    num_blocks = 0
            else:
                num_blocks = 0
                
            num_walls = state_at_t14['wall_locations'].shape[0]
            
            env = AutomaticityEnv(num_agents=num_env_agents, size=env_size, max_steps=env_max_steps, 
                                  num_blocks=num_blocks, num_walls=num_walls)
            
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
            
            # Simulate future steps
            for step_idx in range(num_future_steps):
                if step_idx >= gt_future_actions.shape[0]:
                    break
                
                step_prediction_start = time.time()
                # For each agent in the environment
                for agent_id in range(num_env_agents):
                    num_total += 1
                    
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
                            obs_t = env.get_observation(updated_states[t])[0]
                            updated_obs.append(obs_t)
                        
                        # Stack observations into a sequence
                        stacked_obs = jax.tree.map(lambda *xs: jnp.stack(xs), *updated_obs)
                        
                        # Concatenate with initial trajectory
                        pred_states = jax.tree.map(lambda x, y: jnp.concatenate([x[:15], y], axis=0), 
                                                  initial_states_traj, stacked_obs)
                        pred_actions = jnp.concatenate([initial_actions_traj, 
                                                       jnp.array(action_history[:step_idx])], axis=0)
                    
                    # Get model prediction for this step and agent
                    try:
                        predicted_action, predicted_probs = model.predict_action(pred_states, pred_actions, 
                                                                               agent_id=agent_id, timestep=14+step_idx)
                        
                        # Compare with ground truth
                        gt_action = gt_future_actions[step_idx][agent_id] if args.group else gt_future_actions[step_idx][0]
                        if predicted_action == gt_action:
                            num_correct += 1
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
                                                                 agent_id=agent_id, timestep=14+step_idx)
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
                try:
                    _, next_env_state_pytree = env.step(current_sim_state_pytree, action_to_take_in_env_list)
                    current_sim_state_pytree = next_env_state_pytree
                    updated_states.append(jax.tree.map(lambda x: x, current_sim_state_pytree))
                except Exception as e:
                    print(f"Error during env.step for sample {a_idx}, step {step_idx}: {e}")
                    break
        
        accuracy = num_correct / num_total if num_total > 0 else 0
        avg_prediction_time = total_prediction_time / num_predictions if num_predictions > 0 else 0
        print(f"AutoToM Multi-Step Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
        print(f"Average prediction time per step: {avg_prediction_time:.4f} seconds")
        return accuracy, avg_prediction_time
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

def eval_naive_llm(args, dataloader, model, episode_id: int = 0):
    if args.multi_step_eval:
        # --- Multi-Step Evaluation Logic ---
        num_future_steps = args.num_steps_to_predict
        num_correct = 0
        num_total = 0
        total_prediction_time = 0
        num_predictions = 0
        
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)
        
        # Initialize environment (parameters will be set per datapoint)
        env_size = 10
        env_max_steps = num_future_steps + 5  # Sufficiently large

        for a_idx in tqdm(range(args.num_agents_to_sample), desc="Multi-step Eval Samples"):
            data_sample = jax.tree.map(lambda x: x[a_idx, -1, :15+num_future_steps], datapoint)
            
            initial_states_traj = jax.tree.map(lambda x: x[:15], data_sample['states'])
            initial_actions_traj = jax.tree.map(lambda x: x[:15], data_sample['actions'])
            
            gt_future_actions = data_sample['actions'][14:]  # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)

            num_env_agents = 4 if args.group else 1
            # Infer num_blocks and num_walls from the state
            state_at_t14 = jax.tree.map(lambda x: x[14], initial_states_traj)
            
            # Handle different possible shapes for block_locations
            if isinstance(state_at_t14['block_locations'], (np.ndarray, jnp.ndarray)):
                if state_at_t14['block_locations'].ndim == 1:
                    num_blocks = 1
                elif state_at_t14['block_locations'].ndim == 2:
                    num_blocks = state_at_t14['block_locations'].shape[0]
                else:
                    num_blocks = 0
            else:
                num_blocks = 0
                
            num_walls = state_at_t14['wall_locations'].shape[0]
            
            env = AutomaticityEnv(num_agents=num_env_agents, size=env_size, max_steps=env_max_steps, 
                                  num_blocks=num_blocks, num_walls=num_walls)
            
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
            
            # Simulate future steps
            for step_idx in range(num_future_steps):
                if step_idx >= gt_future_actions.shape[0]:
                    break
                
                step_prediction_start = time.time()
                # For each agent in the environment
                for agent_id in range(num_env_agents):
                    num_total += 1
                    
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
                            obs_t = env.get_observation(updated_states[t])[0]
                            updated_obs.append(obs_t)
                        
                        # Stack observations into a sequence
                        stacked_obs = jax.tree.map(lambda *xs: jnp.stack(xs), *updated_obs)
                        
                        # Concatenate with initial trajectory
                        pred_states = jax.tree.map(lambda x, y: jnp.concatenate([x[:15], y], axis=0), 
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
                            num_correct += 1
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
                                                              agent_id=agent_id, timestep=14+step_idx)
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
                
                # Simulate environment step
                try:
                    _, next_env_state_pytree = env.step(current_sim_state_pytree, action_to_take_in_env_list)
                    current_sim_state_pytree = next_env_state_pytree
                    updated_states.append(jax.tree.map(lambda x: x, current_sim_state_pytree))
                except Exception as e:
                    print(f"Error during env.step for sample {a_idx}, step {step_idx}: {e}")
                    break
        
        accuracy = num_correct / num_total if num_total > 0 else 0
        avg_prediction_time = total_prediction_time / num_predictions if num_predictions > 0 else 0
        print(f"NLLM Multi-Step Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
        print(f"Average prediction time per step: {avg_prediction_time:.4f} seconds")
        return accuracy, avg_prediction_time
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

def eval_fsm(args, dataloader, model, episode_id: int = 0):
    num_correct = 0
    num_total = 0
    # Process just one datapoint (one epoch) at a time
    datapoint = next(dataloader)
    
    for a in tqdm(range(args.num_agents_to_sample)):
        if args.group:
            num_total += 4
        else:
            num_total += 1
        data = jax.tree.map(lambda x: x[a, -1, :15], datapoint)  # last datapoint, 20 timesteps for each agent
        states = data['states']
        actions = data['actions']  # (20, 1)  # 20 timesteps, 1 action
        agent_ids = data['agent_ids']
        try:
            tries = 0
            succeeded = False
            predicted_final_action = model.predict_action(states, actions, agent_ids)
            if predicted_final_action is None:
                continue
            elif len(predicted_final_action.shape) == 1:
                predicted_final_action = predicted_final_action.reshape(1, -1)

            gt_final_action = actions[-1]
            for aid in tqdm(range(len(gt_final_action))):
                final_action_prob = predicted_final_action[aid, gt_final_action[aid]]
                if final_action_prob >= np.max(predicted_final_action[aid]):  # only count a success if the model succeesfully made a prediction
                    num_correct += 1
        except:
            # full_traceback = traceback.format_exc()
            # print(full_traceback)
            continue

    del datapoint
        
    if num_total > 0:
        # print(f"Successful Prediction rate: {num_successes / num_total}")
        # if num_successes > 0:
        #     print(f"Successful Prediction Accuracy: {num_correct / num_successes}")
        print(f"Total Accuracy: {num_correct / num_total}")
        return num_correct / num_total
    else:
        print("No valid predictions were made")
        return 0.0
            

def load_tomnet_models(args):
    """Load ToMnet models from multiple seeds."""
    states = []
    
    # Define the learning rate to use
    lr = args.learning_rate

    # Initialize model
    if args.group:
        num_to_predict = 4
    else:
        num_to_predict = 1
    model = ToMNet(character_net_features=64, mental_net_features=128, output_size=6, num_to_predict=num_to_predict)
    
    for seed in range(6):
        # Construct the path pattern similar to what's used in train_baselines.py
        model_path = f"baselines/ToMnet/{args.save_path}/nagents{args.num_agents_to_sample}_ndatapoints{args.num_datapoints_per_agent_to_sample}_seed{seed}_lr{lr}_group{args.group}"
        
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
            most_recent_epoch = max(checkpoint_epochs)
            most_recent_checkpoint = [f for f in checkpoint_files if f"epoch{most_recent_epoch}" in f][0]
            checkpoint_path = os.path.join(checkpoint_dir, most_recent_checkpoint)
        
        print(f"Loading ToMnet model from seed {seed}: {checkpoint_path}")
        
        
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_bytes = f.read()
        
        # Create a dummy state to get the structure right
        rng_key = jax.random.PRNGKey(0)
        dummy_states = jnp.zeros((args.num_datapoints_per_agent_to_sample, 15, args.env_size*5, args.env_size*5, 3))
        dummy_actions = jnp.zeros((args.num_datapoints_per_agent_to_sample, 15, 1))
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

def eval_mtom(args, dataloader, model, states):
    """Evaluate ToMnet models."""
    num_correct = 0
    num_total = 0
    
    for i in tqdm(range(args.num_epochs)):
        datapoint = next(dataloader)
        
        for a in range(args.num_agents_to_sample):
            data = jax.tree.map(lambda x: x[a, :, :15], datapoint)  # num_datapoints, 15, *
            gt_final_actions = data['actions'][-1, -1]  # num_agents,

            def single_agent_pred(param_state, data):
                # Include batch_stats in the variables dictionary
                variables = {'params': param_state['params'], 'batch_stats': param_state['batch_stats']}
                res = model.apply(variables, data['states'], data['actions'], training=False)
                return res
            
            action_preds = jax.vmap(single_agent_pred, in_axes=(0, None))(states, data)  # num loaded params, num_agents,num_timepoints - 1, 6

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
    print(f"ToMnet Ensemble Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
    return accuracy

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
            most_recent_epoch = max(checkpoint_epochs)
            most_recent_checkpoint = [f for f in checkpoint_files if f"epoch{most_recent_epoch}" in f][0]
            checkpoint_path = os.path.join(checkpoint_dir, most_recent_checkpoint)
        
        print(f"Loading BC model from seed {seed}: {checkpoint_path}")
        
        
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_bytes = f.read()
        
        # Create a dummy state to get the structure right
        rng_key = jax.random.PRNGKey(0)
        dummy_states = jnp.zeros((args.num_datapoints_per_agent_to_sample, 15, args.env_size*5, args.env_size*5, 3))
        dummy_actions = jnp.zeros((args.num_datapoints_per_agent_to_sample, 15, 1))
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
def eval_bc(args, dataloader, model, states, episode_id: int = 0):
    """Evaluate BC models."""
    
    if args.multi_step_eval:
        # --- Multi-Step Evaluation Logic ---
        num_future_steps = args.num_steps_to_predict
        num_correct = 0
        num_total = 0
        total_prediction_time = 0
        num_predictions = 0
        
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)
        
        # Initialize environment (parameters will be set per datapoint)
        env_size = 10
        env_max_steps = num_future_steps + 5  # Sufficiently large

        for a_idx in tqdm(range(args.num_agents_to_sample), desc="Multi-step Eval Samples"):
            data_sample = jax.tree.map(lambda x: x[a_idx, -1, :15+num_future_steps], datapoint)
            
            initial_states_traj = jax.tree.map(lambda x: x[:15], data_sample['states'])
            initial_actions_traj = jax.tree.map(lambda x: x[:15], data_sample['actions'])
            
            gt_future_actions = data_sample['actions'][14:]  # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)

            num_env_agents = 4 if args.group else 1
            # Infer num_blocks and num_walls from the state
            state_at_t14 = jax.tree.map(lambda x: x[14], initial_states_traj)
            
            # Handle different possible shapes for block_locations
            if isinstance(state_at_t14['block_locations'], (np.ndarray, jnp.ndarray)):
                if state_at_t14['block_locations'].ndim == 1:
                    num_blocks = 1
                elif state_at_t14['block_locations'].ndim == 2:
                    num_blocks = state_at_t14['block_locations'].shape[0]
                else:
                    num_blocks = 0
            else:
                num_blocks = 0
                
            num_walls = state_at_t14['wall_locations'].shape[0]
            
            env = AutomaticityEnv(num_agents=num_env_agents, size=env_size, max_steps=env_max_steps, 
                                  num_blocks=num_blocks, num_walls=num_walls)
            
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
            
            # Convert initial states to images for BC model
            initial_states_images = []
            for t in range(15):
                state_t = jax.tree.map(lambda x: x[t], initial_states_traj)
                img_size = args.env_size * 8
                tile_size = 8
                grid_size = args.env_size
                
                # img_gen_fn = jax.jit(state_to_image_jit, static_argnums=(1, 2, 3))
                try:
                    image_t = img_gen_fn(state_t, img_size, grid_size, tile_size)
                    initial_states_images.append(image_t)
                except Exception as e:
                    print(f"Error generating image for state {state_t} at time {t}: {e}")
            
            initial_states_images = jnp.stack(initial_states_images)
            
            # Simulate future steps
            for step_idx in range(num_future_steps):
                if step_idx >= gt_future_actions.shape[0]:
                    break
                # For each agent in the environment
                for agent_id in range(num_env_agents):
                    num_total += 1
                    
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
                            img_size = args.env_size * 8
                            tile_size = 8
                            grid_size = args.env_size
                            
                            obs = jax.tree.map(lambda x: jnp.array(x), env.get_observation(state_t)[0])
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
                        
                        # Get predictions from all ensemble models
                        action_preds = jax.vmap(single_agent_pred, in_axes=(0, None, None))(
                            states, pred_states_batch, pred_actions_batch)
                        
                        # action_preds shape: [num_models, num_to_predict, batch*timesteps, 6]
                        # We want the prediction for the last timestep
                        action_preds = action_preds[:, agent_id, -1, :]  # [num_models, 6]
                        
                        # Average predictions across ensemble
                        avg_action_pred = jnp.mean(action_preds, axis=0)  # [6]
                        predicted_action = jnp.argmax(avg_action_pred)
                        
                        # Compare with ground truth
                        gt_action = gt_future_actions[step_idx][agent_id] if args.group else gt_future_actions[step_idx][0]
                        if predicted_action == gt_action:
                            num_correct += 1
                    except Exception as e:
                        print(f"Error predicting action for agent {agent_id}, step {step_idx}: {e}")
                        continue
                

                num_predictions += 1
                
                # Collect actions for all agents to update environment
                action_to_take_in_env_list = []
                step_prediction_start = time.time()
                for agent_id in range(num_env_agents):
                    try:
                        # Reshape for batch prediction
                        pred_states_batch = jnp.expand_dims(pred_states_images, axis=0)
                        pred_actions_batch = jnp.expand_dims(pred_actions, axis=0)
                        
                        # Apply model to get predictions
                        def single_agent_pred(param_state, data_states, data_actions):
                            variables = {'params': param_state['params'], 'batch_stats': param_state['batch_stats']}
                            res = model.apply(variables, data_states, data_actions, training=False)
                            return res
                        
                        # Get predictions from all ensemble models
                        action_preds = jax.vmap(single_agent_pred, in_axes=(0, None, None))(
                            states, pred_states_batch, pred_actions_batch)
                        
                        # Get prediction for this agent
                        action_preds = action_preds[:, agent_id, -1, :]  # [num_models, 6]
                        
                        # Average predictions across ensemble
                        avg_action_pred = jnp.mean(action_preds, axis=0)
                        predicted_action = jnp.argmax(avg_action_pred)
                        
                        action_to_take_in_env_list.append(predicted_action)
                    except Exception as e:
                        # If prediction fails, use ground truth (to keep simulation going)
                        gt_action = gt_future_actions[step_idx][agent_id] if args.group else gt_future_actions[step_idx][0]
                        action_to_take_in_env_list.append(gt_action)
                        print(f"Using ground truth for agent {agent_id}, step {step_idx} due to error: {e}")
                
                
                step_prediction_time = time.time() - step_prediction_start
                total_prediction_time += step_prediction_time
                # Store actions for next iteration
                if step_idx == 0:
                    action_history = [action_to_take_in_env_list]
                    updated_states = []
                else:
                    action_history.append(action_to_take_in_env_list)
                
                # Simulate environment step
                try:
                    _, next_env_state_pytree = env.step(current_sim_state_pytree, action_to_take_in_env_list)
                    current_sim_state_pytree = next_env_state_pytree
                    updated_states.append(jax.tree.map(lambda x: x, current_sim_state_pytree))
                except Exception as e:
                    print(f"Error during env.step for sample {a_idx}, step {step_idx}: {e}")
                    break
        
        accuracy = num_correct / num_total if num_total > 0 else 0
        avg_prediction_time = total_prediction_time / num_predictions if num_predictions > 0 else 0
        print(f"BC Multi-Step Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
        print(f"Average prediction time per step: {avg_prediction_time:.4f} seconds")
        return accuracy, avg_prediction_time
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
                       'first_step_correct': 0, 'first_step_total': 0} for n in range(1, max_hypotheses + 1)}
        
        # Add timing measurements
        total_prediction_time = 0  # <-- This will now accumulate action prediction time
        num_predictions = 0
        
        # Initialize environment (parameters will be set per datapoint)
        # Env params that are usually fixed or can be default
        env_size = 10 
        env_max_steps = num_future_steps + 5 # Sufficiently large

        datapoint = next(dataloader) # Process one batch of data

        for a_idx in tqdm(range(args.num_agents_to_sample), desc="Multi-step Eval Samples"):
            data_sample = jax.tree.map(lambda x: x[a_idx, -1, :15+num_future_steps], datapoint) 
            
            initial_states_traj = jax.tree.map(lambda x: x[:15], data_sample['states'])
            initial_actions_traj = jax.tree.map(lambda x: x[:15], data_sample['actions'])
            
            gt_future_actions = data_sample['actions'][14:] # Shape (num_future_steps, num_env_agents) or (num_future_steps, 1)

            num_env_agents = 4 if args.group else 1
            # Infer num_blocks and num_walls from the state at the beginning of the prediction horizon
            # Use state at t=14 (end of initial trajectory) to setup env for t=15 prediction
            state_at_t14 = jax.tree.map(lambda x: x[14], initial_states_traj)
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
            
            # Hypothesis generation time is NOT used for avg_prediction_time anymore
            if args.rejuvenation:
                compiled_agents, agent_probs, agent_codes = model.predict_action_with_rejuvenation(
                    initial_states_traj, initial_actions_traj, episode_id=episode_id,
                    max_hypotheses=max_hypotheses,
                    rejuvenation_threshold=args.rejuvenation_threshold,
                    max_rejuvenation_attempts=args.max_rejuvenation_attempts,
                    top_k=0,  # Don't apply top_k here, we'll apply it per n_hyp
                    return_compiled_agents=True
                )
            else:
                compiled_agents, agent_probs, agent_codes = model.predict_action_with_bootstrap(
                    initial_states_traj, initial_actions_traj, episode_id=episode_id,
                    max_hypotheses=max_hypotheses,
                    top_k=0,  # Don't apply top_k here, we'll apply it per n_hyp
                    return_compiled_agents=True
                )
                
            num_predictions += 1

            if not compiled_agents or not agent_probs:
                print(f"Sample {a_idx}: No hypotheses generated, skipping.")
                continue
            
            # Get the number of available hypotheses
            num_available_hyp = len(compiled_agents)
            
            # For each hypothesis count we want to evaluate
            for n_hyp in range(1, max_hypotheses + 1):
                # If we don't have enough hypotheses, use what we have
                actual_n_hyp = min(n_hyp, num_available_hyp)
                
                # Take only the first actual_n_hyp hypotheses
                curr_agents = compiled_agents[:actual_n_hyp]
                curr_probs = np.array(agent_probs[:actual_n_hyp])
                curr_codes = agent_codes[:actual_n_hyp]
                
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
                action_time_per_step = time.time()
                # Simulate future steps
                for step_idx in range(num_future_steps):
                    if step_idx >= gt_future_actions.shape[0]:
                        break
                    
                    step_total += num_env_agents
                    
                    # Prepare agent_input_state for agent.act()
                    agent_input_state_for_act = current_sim_state_pytree
                    
                    # Aggregate predictions from the filtered hypotheses
                    all_hyp_pis = []

                    # --- Start timing action prediction ---
                    step_prediction_start = time.time()
                    
                    for hyp_idx, hyp_agent in enumerate(curr_agents):
                        hyp_prob = curr_probs[hyp_idx]
                        try:
                            _, proposed_pi_for_hyp = hyp_agent.act(current_obs)
                            
                            if args.group:
                                # proposed_pi_for_hyp is a list of np.arrays. Stack them.
                                all_hyp_pis.append(np.array(proposed_pi_for_hyp) * hyp_prob) # (num_env_agents, num_actions)
                            else:
                                all_hyp_pis.append(np.array(proposed_pi_for_hyp) * hyp_prob) # (num_actions,)
                        except Exception as e:
                            # print(f"Error in hypothesis {hyp_idx} agent.act(): {e}. Using uniform distribution.")
                            # Create uniform distribution as fallback
                            if args.group:
                                # Create uniform distribution for each agent (shape: num_env_agents, 6)
                                uniform_pi = np.ones((num_env_agents, 6)) / 6
                                all_hyp_pis.append(uniform_pi * hyp_prob)
                            else:
                                # Create uniform distribution for single agent (shape: 6)
                                uniform_pi = np.ones(6) / 6
                                all_hyp_pis.append(uniform_pi * hyp_prob)


                    # Sum weighted pis
                    final_predicted_pi_for_step = np.sum(np.array(all_hyp_pis), axis=0)
                    
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
                    
                    # for aid in range(num_env_agents):
                    #     predicted_action_for_agent = action_to_take_in_env_list[aid]
                    #     gt_action_for_agent = gt_action_this_step[aid] if args.group else gt_action_this_step[0]
                    #     breakpoint()
                    #     if predicted_action_for_agent == gt_action_for_agent:
                    #         step_correct += 1
                    #         # Track first step accuracy separately
                    #         if step_idx == 0:
                    #             results[n_hyp]['first_step_correct'] += 1
                    if not args.group:
                        max_prob = np.max(final_predicted_pi_for_step)
                        # count number of actions with max probability
                        num_max_actions = np.sum(final_predicted_pi_for_step == max_prob)
                        gt_prob = final_predicted_pi_for_step[gt_action_this_step[0]]  
                        if max_prob == gt_prob:
                            step_correct += 1 / num_max_actions
                            if step_idx == 0:
                                results[n_hyp]['first_step_correct'] += (1 / num_max_actions)
                    else:
                        for aid in range(num_env_agents):
                            max_action_prob = np.max(final_predicted_pi_for_step[aid])
                            # count number of actions with max probability  
                            num_max_actions = np.sum(final_predicted_pi_for_step[aid] == max_action_prob)
                            gt_action_prob = final_predicted_pi_for_step[aid, gt_action_this_step[aid]]
                            if max_action_prob == gt_action_prob:
                                step_correct += 1 / num_max_actions
                                if step_idx == 0:
                                    results[n_hyp]['first_step_correct'] += (1 / num_max_actions)
                    
                    
                    # Track first step total separately
                    if step_idx == 0:
                        results[n_hyp]['first_step_total'] += num_env_agents
                    
                    current_obs = jax.tree.map(lambda x: x[14+step_idx+1], data_sample['states'])
                    # # Simulate environment step
                    # try:
                    #     # env.step expects a list of actions, one per agent in the env
                    #     next_obs, next_env_state_pytree = env.step(current_sim_state_pytree, action_to_take_in_env_list)
                    #     current_sim_state_pytree = next_env_state_pytree
                    #     current_obs = next_obs[0]
                    # except Exception as e:
                    #     print(f"Error during env.step for sample {a_idx}, step {step_idx}: {e}")
                    #     break # Stop simulation for this sample if env step fails
                
                # Update results for this hypothesis count
                results[n_hyp]['correct'] += step_correct
                results[n_hyp]['total'] += step_total

        # Calculate accuracies and average program lengths for each number of hypotheses
        accuracies = {}
        first_step_accuracies = {}
        program_lengths = {}
        action_times = {}
        for n_hyp in results:
            if results[n_hyp]['total'] > 0:
                accuracies[n_hyp] = results[n_hyp]['correct'] / results[n_hyp]['total']
                program_lengths[n_hyp] = results[n_hyp]['program_length'] / args.num_agents_to_sample
            else:
                accuracies[n_hyp] = 0.0
                program_lengths[n_hyp] = 0.0
                
            # Calculate first step accuracy
            if results[n_hyp]['first_step_total'] > 0:
                first_step_accuracies[n_hyp] = results[n_hyp]['first_step_correct'] / results[n_hyp]['first_step_total']
            else:
                first_step_accuracies[n_hyp] = 0.0
        
        # Calculate average prediction time
        avg_prediction_time = total_prediction_time / num_predictions if num_predictions > 0 else 0
        
        # Print results
        for n_hyp, acc in accuracies.items():
            print(f"Hypotheses: {n_hyp}, Multi-Step Accuracy: {acc:.4f} ({results[n_hyp]['correct']}/{results[n_hyp]['total']}), First Step Accuracy: {first_step_accuracies[n_hyp]:.4f}, Avg Program Length: {program_lengths[n_hyp]:.1f}")
        print(f"Average prediction time per step: {avg_prediction_time:.4f} seconds")
        
        # Return the full dictionary of accuracies and program lengths, plus timing info and first step accuracies
        return accuracies, program_lengths, action_times, avg_prediction_time, first_step_accuracies
        
    else:
        # --- Original Single-Step Bootstrap Evaluation Logic ---
        # This part remains unchanged as it's already measuring single-step accuracy
        max_hypotheses = args.n_hypothesis  # Maximum number of hypotheses to consider
        results = {n: {'correct': 0, 'total': 0, 'program_length': 0} for n in range(1, max_hypotheses + 1)}
        
        # Process just one datapoint (one epoch) at a time
        datapoint = next(dataloader)
        
        for a in tqdm(range(args.num_agents_to_sample)):
            if args.group:
                num_agents_per_sample = 4
            else:
                num_agents_per_sample = 1
            
            data = jax.tree.map(lambda x: x[a, -1, :15], datapoint)  # last datapoint, 15 timesteps for each agent
            states = data['states']
            actions = data['actions']  # (15, num_agents_per_sample)
            agent_ids = data['agent_ids']
            
            try:
                if args.rejuvenation:
                    bootstrap_results = model.predict_action_with_rejuvenation(
                        states, actions, episode_id=episode_id, 
                        max_hypotheses=args.n_hypothesis,
                        rejuvenation_threshold=args.rejuvenation_threshold,
                        max_rejuvenation_attempts=args.max_rejuvenation_attempts,
                        top_k=args.top_k
                    )
                else:
                    bootstrap_results = model.predict_action_with_bootstrap(
                        states, actions, episode_id=episode_id, 
                        max_hypotheses=args.n_hypothesis,
                        top_k=args.top_k
                    )
                
                if bootstrap_results is None:
                    continue
                    
                gt_final_action = actions[-1]
                
                # Get the number of available hypotheses
                num_available_hyp = len(bootstrap_results)
                
                # Evaluate each bootstrap prediction (for all hypothesis counts up to max_hypotheses)
                for n_hyp in range(1, max_hypotheses + 1):
                    results[n_hyp]['total'] += num_agents_per_sample
                    
                    # If we have this hypothesis result available, use it
                    # Otherwise, use the last available hypothesis result
                    hyp_idx = min(n_hyp, num_available_hyp) - 1  # Convert to 0-indexed
                    prediction, program_length = bootstrap_results[hyp_idx]
                    results[n_hyp]['program_length'] += program_length
                    
                    if len(prediction.shape) == 1:
                        prediction = prediction.reshape(1, -1)
                    
                    for aid in range(len(gt_final_action)):
                        final_action_prob = prediction[aid, gt_final_action[aid]]
                        # Count number of actions with max probability
                        max_prob = np.max(prediction[aid])
                        num_max_actions = np.sum(prediction[aid] == max_prob)
                        
                        # If final action has max probability, add fractional correct count
                        if final_action_prob == max_prob:
                            results[n_hyp]['correct'] += 1.0 / num_max_actions
                            
            except Exception as e:
                continue
        
        # Calculate accuracies and average program lengths for each number of hypotheses
        accuracies = {}
        program_lengths = {}
        for n_hyp in results:
            if results[n_hyp]['total'] > 0:
                accuracies[n_hyp] = results[n_hyp]['correct'] / results[n_hyp]['total']
                program_lengths[n_hyp] = results[n_hyp]['program_length'] / args.num_agents_to_sample
            else:
                accuracies[n_hyp] = 0.0
                program_lengths[n_hyp] = 0.0
        
        # Print results
        for n_hyp, acc in accuracies.items():
            print(f"Hypotheses: {n_hyp}, Accuracy: {acc:.4f} ({results[n_hyp]['correct']}/{results[n_hyp]['total']}), Avg Program Length: {program_lengths[n_hyp]:.1f}")
        
        # Return the full dictionary of accuracies and program lengths
        return accuracies, program_lengths

def main():
    """Main function to eval baseline models."""
    args = parse_args()
    
    # Set JAX memory allocation to grow as needed
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    # Set PyTorch to use expandable segments to avoid memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
    print(f"Evaluating baseline model: {args.baseline_model}; n_hypothesis: {args.n_hypothesis}; num_epochs: {args.num_epochs}; model_arch: {args.model_name}")
    if args.baseline_model == "FSM" and args.bootstrap and args.multi_step_eval:
        print(f"Multi-step evaluation enabled: predicting {args.num_steps_to_predict} future steps.")

    save_path_dir = f"baselines/{args.baseline_model}/{args.save_path}/nagents{args.num_agents_to_sample}_ndatapoints{args.num_datapoints_per_agent_to_sample}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path_dir), exist_ok=True)
    
    # Initialize random key for parameter initialization
    rng_key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create CSV path for results
    group_extension = "_group" if args.group else ""
    two_stage_extension = "_two_stage" if args.two_stage else ""
    structured_extension = f"_structured_{args.structured}" if args.structured != "False" else ""
    rejuvenation_extension = "_rejuvenation" if args.rejuvenation else ""
    
    if args.baseline_model == "FSM" and args.bootstrap:
        if args.multi_step_eval:
            csv_path = f"baselines/{args.baseline_model}/fixed2_results_fsm_bootstrap_multistep{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}_topk{args.top_k}_steps{args.num_steps_to_predict}_actionTime.csv"
        else: # Single-step FSM bootstrap
            csv_path = f"baselines/{args.baseline_model}/fixed2_results_fsm_bootstrap_singlestep{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}_topk{args.top_k}.csv"
    else: # Non-bootstrap FSM or other models
        csv_path = f"baselines/{args.baseline_model}/fixed2_results_grid_{args.baseline_model}_{args.n_hypothesis}hyp{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}.csv"
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if CSV exists and determine starting epoch
    start_epoch = 0
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                filter_conditions = [
                    (existing_df['model'] == args.baseline_model),
                    (existing_df['group'] == args.group),
                    (existing_df['num_agents_evaluated'] == args.num_agents_to_sample),
                    (existing_df['datapoints_per_agent'] == args.num_datapoints_per_agent_to_sample),
                    (existing_df['llm_model'] == (args.model_name if args.baseline_model in ['TT', 'AutoToM', 'FSM', 'NLLM'] else 'N/A'))
                ]
                
                if 'two_stage' in existing_df.columns:
                    filter_conditions.append(existing_df['two_stage'] == args.two_stage)
                if 'structured' in existing_df.columns:
                    filter_conditions.append(existing_df['structured'].astype(str) == str(args.structured))
                if 'rejuvenation' in existing_df.columns:
                    filter_conditions.append(existing_df['rejuvenation'] == args.rejuvenation)
                if 'top_k' in existing_df.columns: # Relevant for FSM bootstrap
                     filter_conditions.append(existing_df['top_k'] == args.top_k)

                if args.baseline_model == "FSM" and args.bootstrap:
                    if 'multi_step_eval' in existing_df.columns:
                        filter_conditions.append(existing_df['multi_step_eval'] == args.multi_step_eval)
                    if args.multi_step_eval and 'num_steps_predicted' in existing_df.columns:
                        filter_conditions.append(existing_df['num_steps_predicted'] == args.num_steps_to_predict)
                    # For single-step bootstrap, num_hypothesis varies, so we don't filter by it here for resuming.
                    # For multi-step, n_hypothesis is fixed by args.n_hypothesis for the rollout.
                    if args.multi_step_eval and 'num_hypothesis' in existing_df.columns:
                         filter_conditions.append(existing_df['num_hypothesis'] == args.n_hypothesis)

                elif 'num_hypothesis' in existing_df.columns : # For non-FSM bootstrap models like TT, NLLM
                    filter_conditions.append(existing_df['num_hypothesis'].astype(str) == str(args.n_hypothesis))
                
                matching_rows = existing_df[np.logical_and.reduce(filter_conditions)]
                
                if len(matching_rows) > 0:
                    start_epoch = matching_rows['epoch'].max() + 1
                    print(f"Resuming from epoch {start_epoch}")
        except pd.errors.EmptyDataError:
            print(f"CSV file {csv_path} is empty. Starting from epoch 0.")
        except Exception as e:
            print(f"Error reading CSV for resume: {e}. Starting from epoch 0.")

    # Initialize model, dataloader, and evaluation function
    model = None
    states = None # For ToMnet/BC
    
    if args.baseline_model == "TT":
        from baselines.thoughtTrace import ThoughtTrace
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = ThoughtTrace(n_hypothesis=args.n_hypothesis, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
        eval_fn = eval_thoughtTrace
    elif args.baseline_model == "AutoToM":
        from baselines.AutoToM.autoToM import AutoToM
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = AutoToM(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
        eval_fn = eval_autoToM
    elif args.baseline_model == "FSM":
        from baselines.inferFSM import FSMReasoner
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = FSMReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, 
                           dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, 
                           num_hypothesis=args.n_hypothesis, group=args.group, two_stage=args.two_stage,
                           structured=args.structured)
        if args.bootstrap:
            eval_fn = eval_fsm_bootstrap # This function handles multi_step_eval internally
        else:
            eval_fn = eval_fsm 
    elif args.baseline_model == 'NLLM':
        from baselines.basic_LLM import NaiveLLMReasoner
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = NaiveLLMReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, num_hypothesis=args.n_hypothesis, group=args.group, partnr=False) # Assuming partnr=False for this context
        eval_fn = eval_naive_llm
    elif args.baseline_model == "ToMnet":
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model, states = load_tomnet_models(args)
        if not states['params']: # Check if any models were loaded
            print("No ToMnet models found or loaded. Please train models first or check paths.")
            return
        eval_fn = lambda a, d, m, s, ep_id: eval_mtom(a, d, m, s) # eval_mtom uses 'states'
    elif args.baseline_model == "BC":
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model, states = load_bc_models(args)
        if not states['params']: # Check if any models were loaded
            print("No BC models found or loaded. Please train models first or check paths.")
            return
        eval_fn = lambda a, d, m, s, ep_id: eval_bc(a, d, m, s, ep_id) # eval_bc uses 'states'
    else:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")

    # Run evaluation for each epoch and save results
    for epoch in tqdm(range(start_epoch, args.num_epochs), desc="Epochs"):
        print(f"\nRunning epoch {epoch+1}/{args.num_epochs}")
        
        results_to_save = []

        if args.baseline_model == "FSM" and args.bootstrap:
            # eval_fsm_bootstrap returns (accuracies_dict, program_lengths_dict)
            accuracies_dict, program_lengths_dict, action_times_dict, avg_prediction_time, first_step_accuracies_dict = eval_fn(args, dataloader, model, episode_id=epoch)
            
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
                        'model': args.baseline_model,
                        'accuracy': accuracy_val,
                        'group': args.group,
                        'num_agents_evaluated': args.num_agents_to_sample,
                        'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                        'epoch': epoch,
                        'llm_model': args.model_name,
                        'num_hypothesis': args.n_hypothesis, # This is k_hypotheses_to_use for rollout
                        'two_stage': args.two_stage,
                        'structured': args.structured,
                        'rejuvenation': args.rejuvenation,
                        'top_k': args.top_k,
                        'program_length': program_lengths_dict.get(n_hyp, 0.0),
                        'multi_step_eval': True,
                        'num_steps_predicted': args.num_steps_to_predict,
                        'first_step_accuracy': first_step_accuracies_dict.get(n_hyp, 0.0)
                    }
                    
                    # Only include avg_prediction_time if it's in the existing CSV
                    if include_prediction_time:
                        result['avg_prediction_time'] = avg_prediction_time
                        
                    results_to_save.append(result)
            else: # Single-step FSM bootstrap
                for n_hyp, accuracy_val in accuracies_dict.items():
                    # Check if existing CSV has avg_prediction_time column
                    include_prediction_time = True
                    # if os.path.exists(csv_path):
                    #     try:
                    #         existing_cols = pd.read_csv(csv_path, nrows=0).columns
                    #         include_prediction_time = 'avg_prediction_time' in existing_cols
                    #     except:
                    #         pass
                            
                    result = {
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
                        'num_steps_predicted': 1
                    }
                    
                    # Only include avg_prediction_time if it's in the existing CSV
                    if include_prediction_time:
                        result['avg_prediction_time'] = avg_prediction_time
                        
                    results_to_save.append(result)
        else: # Other models or non-bootstrap FSM
            if args.baseline_model in ["ToMnet", "BC"]:
                current_accuracy, avg_prediction_time = eval_fn(args, dataloader, model, states, epoch)
            else:
                current_accuracy, avg_prediction_time = eval_fn(args, dataloader, model, episode_id=epoch)

            # Check if existing CSV has avg_prediction_time column
            include_prediction_time = True
            if os.path.exists(csv_path):
                try:
                    existing_cols = pd.read_csv(csv_path, nrows=0).columns
                    include_prediction_time = 'avg_prediction_time' in existing_cols
                except:
                    pass
                    
            result = {
                'model': args.baseline_model,
                'accuracy': current_accuracy,
                'group': args.group,
                'num_agents_evaluated': args.num_agents_to_sample,
                'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                'epoch': epoch,
                'llm_model': args.model_name if args.baseline_model in ['TT', 'AutoToM', 'FSM', 'NLLM'] else 'N/A',
                'num_hypothesis': args.n_hypothesis if args.baseline_model in ['TT', 'FSM', 'NLLM'] else ('ensemble' if args.baseline_model in ["ToMnet", "BC"] else 'N/A'),
                'two_stage': args.two_stage if args.baseline_model == 'FSM' else False,
                'structured': args.structured if args.baseline_model == 'FSM' else "False",
                'rejuvenation': args.rejuvenation if args.baseline_model == 'FSM' else False,
                'top_k': args.top_k if args.baseline_model == 'FSM' else 0, # or N/A
                'program_length': getattr(model, 'weighted_program_length', 0) if args.baseline_model == 'FSM' and not args.bootstrap else 0, # Placeholder
                'multi_step_eval': False, # Assuming these are single-step
                'num_steps_predicted': 1  # Assuming these are single-step
            }
            
            # Only include avg_prediction_time if it's in the existing CSV
            if include_prediction_time:
                result['avg_prediction_time'] = avg_prediction_time
                
            results_to_save.append(result)

        if results_to_save:
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
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()