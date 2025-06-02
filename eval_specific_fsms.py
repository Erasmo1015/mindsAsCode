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
    parser.add_argument('--n_hypothesis', type=int, default=4, help='Number of hypothesis for thought trace.')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", help='Name of the model to use.')  # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct or meta-llama/Llama-3.1-8B-Instruct or deepseek-ai/DeepSeek-V2-Lite
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
                # print(f"Error: {e}")
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
                # break   # end loop if prediction is successful
            except Exception as e:
                # print(f"Error: {e}")
                # full_traceback = traceback.format_exc()
                # print(full_traceback)
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
    max_hypotheses = args.n_hypothesis  # Maximum number of hypotheses to consider
    # results = {n: {'correct': 0, 'total': 0, 'program_length': 0} for n in range(1, max_hypotheses + 1)}
    
    # Process just one datapoint (one epoch) at a time
    datapoint = next(dataloader)

    results = []
    
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
                    top_k=args.top_k,
                    return_compiled_agents=True
                )
            else:
                bootstrap_results = model.predict_action_with_bootstrap(
                    states, actions, episode_id=episode_id, 
                    max_hypotheses=args.n_hypothesis,
                    top_k=args.top_k,
                    return_compiled_agents=True
                )
            
            if bootstrap_results is None:
                continue

            final_state = jax.tree.map(lambda x: x[-1], states)
            if len(final_state['agent_locations']) == 1:
                final_state['agent_id'] = 0

            gt_final_action = actions[-1]

            agent_accuracies = []
            agent_lengths = []
            agent_predictions = []
            agent_programs, agent_weights, agent_codes = bootstrap_results
            for agent_idx in range(len(agent_programs)):
                agent_program = agent_programs[agent_idx]
                predicted_action = agent_program.act(final_state)
                
                acc = 0
                for aid in range(len(gt_final_action)):
                    acc += predicted_action[aid] == gt_final_action[aid]
                agent_accuracies.append(acc / len(gt_final_action))
                agent_lengths.append(len(agent_codes[agent_idx]))
                agent_predictions.append(predicted_action[0])
            agent_accuracies = np.array(agent_accuracies)
            agent_lengths = np.array(agent_lengths)
            task_ids = np.ones(len(agent_accuracies)) * agent_ids[-1]

            action_std = np.std(actions)
            for k in range(len(agent_accuracies)):
                results.append({
                    'task_id': task_ids[k],
                    'accuracy': agent_accuracies[k],
                    'length': agent_lengths[k],
                    'agent_weight': agent_weights[k],
                    'agent_prediction': agent_predictions[k],
                    'gt_action': gt_final_action[0],
                    'action_std': action_std,
                    'num_actions': len(actions)
                })
        except Exception as e:
            continue

    
    return results

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
        csv_path = f"baselines/{args.baseline_model}/all_fsm_bootstrap_accuracy_{args.baseline_model}{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}_topk{args.top_k}.csv"
    else:
        csv_path = f"baselines/{args.baseline_model}/grid_accuracy_{args.baseline_model}_{args.n_hypothesis}hyp{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}.csv"
    
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
                           num_hypothesis=args.n_hypothesis, group=args.group, two_stage=args.two_stage,
                           structured=args.structured)
        # model.bootstrap = True
        eval_fn = eval_fsm_bootstrap
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
            return eval_bc(a, d, m, states, episode_id)
            
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
            results = eval_fn(args, dataloader, model, episode_id=epoch)
            
            # Create results for each hypothesis count
            for result in results:
                temp_result = {
                    'model': args.baseline_model,
                    'group': args.group,
                    'num_agents_evaluated': args.num_agents_to_sample,
                    'datapoints_per_agent': args.num_datapoints_per_agent_to_sample, 
                    'epoch': epoch,
                    'llm_model': args.model_name,
                    'num_hypothesis': args.n_hypothesis,
                    'two_stage': args.two_stage,
                    'structured': args.structured,
                    'rejuvenation': args.rejuvenation,
                    'top_k': args.top_k,
                }
                for key in temp_result.keys():
                    result[key] = temp_result[key]
                # Create single-row dataframe for this result
                df = pd.DataFrame([result])
                
                if os.path.exists(csv_path) and epoch >= start_epoch:
                    # Read existing data and append new row
                    df.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    # Create new CSV file or overwrite for first epoch
                    df.to_csv(csv_path, index=False)
        else:
            res = eval_fn(args, dataloader, model, episode_id=epoch)
            
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
                'rejuvenation': args.rejuvenation if args.baseline_model == 'FSM' else False,
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
        
        print(f"Saved results for epoch {epoch}")
    os.environ['CURRENT_MODEL_NAME'] = ''

if __name__ == "__main__":
    main()