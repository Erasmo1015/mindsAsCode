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
from environment import state_to_image_jit
import optax
from flax.training import train_state
from flax import struct
import pandas as pd
from human_dataloader import load_and_stack_data
import traceback

from baselines.ToMnet import ToMNet
from baselines.BC import BCNet

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
    parser.add_argument('--rejuvenation', type=bool, default=False, help='Whether to use rejuvenation for FSM.')
    parser.add_argument('--rejuvenation_threshold', type=float, default=-10.0, help='Threshold for rejuvenation.')
    parser.add_argument('--max_rejuvenation_attempts', type=int, default=5, help='Maximum number of rejuvenation attempts.')
    parser.add_argument('--top_k', type=int, default=0, help='If > 0, only average over the top k most likely hypotheses.')
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

    data_folder = f"{data_path}/human_data_fix"
    human_dataloader = load_and_stack_data(data_folder)
    human_data = None
    for j in range(epoch+1):
        human_data, human_actions, agent_id, filename, task = next(human_dataloader)
    current_data_loaded = epoch

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
            'filename': filename,
            'task': task
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

        if as_images:
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

            del reshaped_state, all_images

            sampled_data['states'] = stacked_images[None, None]
        else:
            sampled_data['states'] = sampled_states

        yield sampled_data
        del sampled_data
        human_data, human_actions, agent_id, filename, task = next(human_dataloader)
        current_data_loaded += 1

def eval_thoughtTrace(args, dataloader, model):
    for i in range(args.num_epochs):
        datapoint = next(dataloader)
        state = datapoint['states']
        actions = datapoint['actions']
        agent_ids = datapoint['agent_ids']

        predicted_final_action = model.predict_action(state, actions, agent_ids)

        breakpoint()

def eval_autoToM(args, dataloader, model, episode_id: int = 0):
    num_correct = 0
    num_total = 0
    # Process just one datapoint (one epoch) at a time
    datapoint = next(dataloader)

    for a in tqdm(range(args.num_agents_to_sample)):
        # Extract filename and task before tree_map
        data = {}
        # Apply tree_map only to array data, not to filename and task
        for key, value in datapoint.items():
            if key in ['filename', 'task']:
                data[key] = value  # Keep these as is
            else:
                data[key] = jax.tree.map(lambda x: x[a, -1, :15], value)
                
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
                print(f"Error: {e}")
                # full_traceback = traceback.format_exc()
                # print(full_traceback)
                # tries += 1
                continue
            # break  # end loop if prediction is successful
            # try:

            # except Exception as e:
            #     print(f"Error: {e}")
            #     tries += 1
           
    
    print(f"Accuracy: {num_correct / num_total}")
    return num_correct / num_total

def eval_naive_llm(args, dataloader, model, episode_id: int = 0):
    num_correct = 0
    num_total = 0
    # Process just one datapoint (one epoch) at a time
    datapoint = next(dataloader)

    for a in tqdm(range(args.num_agents_to_sample)):
        # Extract filename and task before tree_map
        data = {}
        # Apply tree_map only to array data, not to filename and task
        for key, value in datapoint.items():
            if key in ['filename', 'task']:
                data[key] = value  # Keep these as is
            else:
                data[key] = jax.tree.map(lambda x: x[a, -1, :15], value)
                
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
                # break   # end loop if prediction is successful
            except Exception as e:
                # print(f"Error: {e}")
                full_traceback = traceback.format_exc()
                print(full_traceback)
                tries += 1
            
    
    print(f"Accuracy: {num_correct / num_total}")
    return num_correct / num_total

def eval_fsm(args, dataloader, model, episode_id: int = 0):
    num_correct = 0
    num_total = 0
    # Process just one datapoint (one epoch) at a time
    datapoint = next(dataloader)
    
    # Create a list to store per-file, per-task results
    detailed_results = []
    filename = datapoint.get('filename', 'unknown')
    task = datapoint.get('task', 'unknown')
    
    for a in tqdm(range(args.num_agents_to_sample)):
        if args.group:
            num_total += 4
        else:
            num_total += 1
        # Extract filename and task before tree_map
        data = {}
        # Apply tree_map only to array data, not to filename and task
        for key, value in datapoint.items():
            if key in ['filename', 'task']:
                data[key] = value  # Keep these as is
            else:
                data[key] = jax.tree.map(lambda x: x[a, -1, :15], value)
        
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
                is_correct = final_action_prob >= np.max(predicted_final_action[aid])
                if is_correct:  # only count a success if the model successfully made a prediction
                    num_correct += 1
                
                # Store detailed result
                detailed_results.append({
                    'filename': filename,
                    'task': task,
                    'agent_id': int(agent_ids[-1, aid]) if len(agent_ids.shape) > 1 else int(agent_ids[-1]),
                    'correct': int(is_correct),
                    'gt_action': int(gt_final_action[aid]),
                    'pred_action': int(np.argmax(predicted_final_action[aid])),
                    'confidence': float(np.max(predicted_final_action[aid]))
                })
                
        except Exception as e:
            # print(f"Error: {e}")
            full_traceback = traceback.format_exc()
            print(full_traceback)
            exit(1)

    del datapoint
        
    if num_total > 0:
        # print(f"Successful Prediction rate: {num_successes / num_total}")
        # if num_successes > 0:
        #     print(f"Successful Prediction Accuracy: {num_correct / num_successes}")
        print(f"Total Accuracy: {num_correct / num_total}")
        return num_correct / num_total, detailed_results
    else:
        print("No valid predictions were made")
        return 0.0, detailed_results
            
            

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

def eval_bc(args, dataloader, model, states, episode_id: int = 0):
    """Evaluate BC models."""
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
                # action_pred = jnp.argmax(action_p, axis=-1)
                max_prob = jnp.max(action_p, axis=-1)
                return action_p[gt_final_action] == max_prob
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
    max_hypotheses = args.n_hypothesis  # Maximum number of hypotheses to consider
    results = {n: {'correct': 0, 'total': 0, 'program_length': 0} for n in range(1, max_hypotheses + 1)}
    
    # Process just one datapoint (one epoch) at a time
    datapoint = next(dataloader)
    
    # Create a list to store per-file, per-task results for each hypothesis count
    detailed_results = {n: [] for n in range(1, max_hypotheses + 1)}
    filename = datapoint.get('filename', 'unknown')
    task = datapoint.get('task', 'unknown')
    
    for a in tqdm(range(args.num_agents_to_sample)):
        if args.group:
            num_agents_per_sample = 4
        else:
            num_agents_per_sample = 1
        
        # Extract filename and task before tree_map
        data = {}
        # Apply tree_map only to array data, not to filename and task
        for key, value in datapoint.items():
            if key in ['filename', 'task']:
                data[key] = value  # Keep these as is
            else:
                data[key] = jax.tree.map(lambda x: x[a, -1, :15], value)
        
        states = data['states']
        actions = data['actions']  # (15, num_agents_per_sample)
        agent_ids = data['agent_ids']
        
        try:
            # Enable bootstrapping
            model.bootstrap = True
            if args.rejuvenation:
                bootstrap_predictions = model.predict_action_with_rejuvenation(
                    states, actions, agent_ids, episode_id=episode_id, 
                    max_hypotheses=max_hypotheses,
                    rejuvenation_threshold=args.rejuvenation_threshold,
                    max_rejuvenation_attempts=args.max_rejuvenation_attempts,
                    top_k=args.top_k
                )
            else:
                bootstrap_predictions = model.predict_action_with_bootstrap(
                    states, actions, agent_ids, episode_id=episode_id, 
                    max_hypotheses=max_hypotheses,
                    top_k=args.top_k
                )
                
            if bootstrap_predictions is None:
                continue
                
            gt_final_action = actions[-1]
            
            # Get the number of available hypotheses
            num_available_hyp = len(bootstrap_predictions)
            
            # Evaluate each bootstrap prediction (for each number of hypotheses)
            for n_hyp in range(1, max_hypotheses + 1):
                results[n_hyp]['total'] += num_agents_per_sample
                
                # If we have this hypothesis result available, use it
                # Otherwise, use the last available hypothesis result
                hyp_idx = min(n_hyp, num_available_hyp) - 1  # Convert to 0-indexed
                prediction, program_length = bootstrap_predictions[hyp_idx]
                results[n_hyp]['program_length'] += program_length
                
                if len(prediction.shape) == 1:
                    prediction = prediction.reshape(1, -1)
                
                for aid in range(len(gt_final_action)):
                    
                    final_action_prob = prediction[aid, gt_final_action[aid]]

                    # Count number of actions with max probability
                    max_prob = np.max(prediction[aid])
                    num_max_actions = np.sum(prediction[aid] == max_prob)
                    
                    # If final action has max probability, add fractional correct count
                    is_correct = final_action_prob == max_prob
                    if is_correct:
                        results[n_hyp]['correct'] += 1.0 / num_max_actions
                    
                    # Store detailed result for this hypothesis count
                    detailed_results[n_hyp].append({
                        'filename': filename,
                        'task': task,
                        'agent_id': int(agent_ids[-1, aid]) if len(agent_ids.shape) > 1 else int(agent_ids[-1]),
                        'correct': int(is_correct),
                        'gt_action': int(gt_final_action[aid]),
                        'pred_action': int(np.argmax(prediction[aid])),
                        'confidence': float(np.max(prediction[aid])),
                        'program_length': program_length
                    })
                        
        except Exception as e:
            full_traceback = traceback.format_exc()
            print(full_traceback)
            breakpoint()
            continue

    # Calculate accuracies and average program lengths for each number of hypotheses
    accuracies = {}
    program_lengths = {}
    for n_hyp in results:
        if results[n_hyp]['total'] > 0:
            accuracies[n_hyp] = results[n_hyp]['correct'] / results[n_hyp]['total']
            program_lengths[n_hyp] = results[n_hyp]['program_length'] / results[n_hyp]['total']
        else:
            accuracies[n_hyp] = 0.0
            program_lengths[n_hyp] = 0.0
    
    # Print results
    for n_hyp, acc in accuracies.items():
        print(f"Hypotheses: {n_hyp}, Accuracy: {acc:.4f} ({results[n_hyp]['correct']}/{results[n_hyp]['total']}), Avg Program Length: {program_lengths[n_hyp]:.1f}")
    
    # Return the full dictionary of accuracies, program lengths, and detailed results
    return accuracies, program_lengths, detailed_results

def main():
    """Main function to eval baseline models."""
    args = parse_args()
    
    # Set JAX memory allocation to grow as needed
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    print(f"Evaluating baseline model: {args.baseline_model}; n_hypothesis: {args.n_hypothesis}; num_epochs: {args.num_epochs}; model_arch: {args.model_name}")
    save_path = f"baselines/{args.baseline_model}/{args.save_path}/nagents{args.num_agents_to_sample}_ndatapoints{args.num_datapoints_per_agent_to_sample}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize random key for parameter initialization
    rng_key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create CSV path for bootstrap results
    if args.group:
        group_extension = "_group"
    else:
        group_extension = ""
        
    two_stage_extension = "_two_stage" if args.two_stage else ""
    structured_extension = f"_structured_{args.structured}" if args.structured != "False" else ""
    rejuvenation_extension = "_rejuvenation" if args.rejuvenation else ""

    if args.bootstrap and args.baseline_model == "FSM":
        if args.top_k > 0:
            csv_path = f"baselines/{args.baseline_model}/fixed_human_bootstrap_accuracy_{args.baseline_model}{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}_topk{args.top_k}.csv"
            detailed_csv_path = f"baselines/{args.baseline_model}/fixed_human_bootstrap_detailed_{args.baseline_model}{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}_topk{args.top_k}.csv"
        else:
            csv_path = f"baselines/{args.baseline_model}/fixed_human_bootstrap_accuracy_{args.baseline_model}{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}.csv"
            detailed_csv_path = f"baselines/{args.baseline_model}/fixed_human_bootstrap_detailed_{args.baseline_model}{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}.csv"
    else:
        csv_path = f"baselines/{args.baseline_model}/fixed_human_accuracy_{args.baseline_model}_{args.n_hypothesis}hyp{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}.csv"
        detailed_csv_path = f"baselines/{args.baseline_model}/fixed_human_detailed_{args.baseline_model}_{args.n_hypothesis}hyp{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}.csv"
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if CSV exists and determine starting epoch
    start_epoch = 0
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        # Filter for the current configuration
        filter_conditions = [
            (existing_df['model'] == args.baseline_model),
            (existing_df['group'] == args.group),
            (existing_df['num_agents_evaluated'] == args.num_agents_to_sample),
            (existing_df['datapoints_per_agent'] == args.num_datapoints_per_agent_to_sample),
            (existing_df['llm_model'] == args.model_name)
        ]
        
        # Add optional filter conditions if columns exist
        if 'top_k' in existing_df.columns:
            filter_conditions.append(existing_df['top_k'] == args.top_k)
        if 'two_stage' in existing_df.columns:
            filter_conditions.append(existing_df['two_stage'] == args.two_stage)
        if 'structured' in existing_df.columns:
            filter_conditions.append(existing_df['structured'].astype(str) == args.structured)
        if 'rejuvenation' in existing_df.columns:
            filter_conditions.append(existing_df['rejuvenation'] == args.rejuvenation)
        if 'num_hypothesis' in existing_df.columns:
            filter_conditions.append(existing_df['num_hypothesis'] == args.n_hypothesis)
        
        # Apply all filter conditions
        matching_rows = existing_df[np.logical_and.reduce(filter_conditions)]
        
        if len(matching_rows) > 0:
            # Get the highest epoch completed
            start_epoch = matching_rows['epoch'].max() + 1
            print(f"Resuming from epoch {start_epoch}")
    
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
                           num_hypothesis=20, group=args.group, two_stage=args.two_stage,
                           structured=args.structured)
        if args.bootstrap:
            eval_fn = eval_fsm_bootstrap
        else:
            eval_fn = eval_fsm
    elif args.baseline_model == 'NLLM':
        from baselines.basic_LLM import NaiveLLMReasoner
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model = NaiveLLMReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, num_hypothesis=args.n_hypothesis, group=args.group, partnr=False)
        eval_fn = eval_naive_llm
    elif args.baseline_model == "ToMnet":
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model, states = load_tomnet_models(args)
        if not model:
            print("No ToMnet models found. Please train models first.")
            return
        def eval_fn(a, d, m, episode_id):
            return eval_mtom(a, d, m, states)
            
    elif args.baseline_model == "BC":
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False, epoch=start_epoch)
        model, states = load_bc_models(args)
        if not model:
            print("No BC models found. Please train models first.")
            return
        def eval_fn(a, d, m, episode_id):
            return eval_bc(a, d, m, states, episode_id)
    else:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    # Run evaluation for each epoch and save results after each epoch
    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        print(f"Running epoch {epoch}/{args.num_epochs}")
        
        if args.bootstrap and args.baseline_model == "FSM":
            accuracies, program_lengths, detailed_results = eval_fn(args, dataloader, model, episode_id=epoch)
            
            # Create results for each hypothesis count
            for n_hyp, accuracy in accuracies.items():
                result = {
                    'model': args.baseline_model,
                    'accuracy': accuracy,
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
                    'program_length': program_lengths[n_hyp]
                }
                
                # Create single-row dataframe for this result
                df = pd.DataFrame([result])
                
                if os.path.exists(csv_path) and epoch >= start_epoch:
                    # Read existing data and append new row
                    df.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    # Create new CSV file or overwrite for first epoch
                    df.to_csv(csv_path, index=False)
                
                # Save detailed results for this hypothesis count
                if detailed_results and n_hyp in detailed_results:
                    detailed_df = pd.DataFrame(detailed_results[n_hyp])
                    detailed_df['epoch'] = epoch
                    detailed_df['num_hypothesis'] = n_hyp
                    detailed_df['model'] = args.baseline_model
                    detailed_df['llm_model'] = args.model_name
                    
                    if os.path.exists(detailed_csv_path) and epoch >= start_epoch:
                        # Read existing data and append new rows
                        detailed_df.to_csv(detailed_csv_path, mode='a', header=False, index=False)
                    else:
                        # Create new CSV file
                        detailed_df.to_csv(detailed_csv_path, index=False)
        else:
            res, detailed_results = eval_fn(args, dataloader, model, episode_id=epoch) if args.baseline_model == "FSM" else (eval_fn(args, dataloader, model, episode_id=epoch), [])
            
            # Create results dictionary for this epoch
            result = {
                'model': args.baseline_model,
                'accuracy': res,
                'group': args.group,
                'num_agents_evaluated': args.num_agents_to_sample,
                'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                'epoch': epoch,
                'llm_model': args.model_name if args.baseline_model in ['TT', 'AutoToM', 'FSM', 'NLLM'] else 'N/A',
                'num_hypothesis': args.n_hypothesis if args.baseline_model in ['TT', 'FSM', 'NLLM'] else 'N/A',
                'two_stage': args.two_stage if args.baseline_model == 'FSM' else False,
                'structured': args.structured if args.baseline_model == 'FSM' else "False",
                'top_k': args.top_k if args.baseline_model == 'FSM' else 0,
                'program_length': getattr(model, 'weighted_program_length', 0) if args.baseline_model == 'FSM' else 0
            }
            
            # Create single-row dataframe for this epoch's result
            df = pd.DataFrame([result])
            
            if os.path.exists(csv_path):
                # Read existing data and append new row
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                # Create new CSV file
                df.to_csv(csv_path, index=False)
            
            # Save detailed results if available
            if detailed_results and args.baseline_model == "FSM":
                detailed_df = pd.DataFrame(detailed_results)
                detailed_df['epoch'] = epoch
                detailed_df['num_hypothesis'] = args.n_hypothesis
                detailed_df['model'] = args.baseline_model
                detailed_df['llm_model'] = args.model_name
                
                if os.path.exists(detailed_csv_path):
                    # Read existing data and append new rows
                    detailed_df.to_csv(detailed_csv_path, mode='a', header=False, index=False)
                else:
                    # Create new CSV file
                    detailed_df.to_csv(detailed_csv_path, index=False)
        
        print(f"Saved results for epoch {epoch}")
    os.environ['CURRENT_MODEL_NAME'] = ''

if __name__ == "__main__":
    main()