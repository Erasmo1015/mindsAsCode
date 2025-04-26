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

import numpy as np
import random
import jax
import jax.numpy as jnp
import flax
import flax.core
import flax.serialization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import gzip
import json
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
    parser.add_argument('--num_agents_to_sample', type=int, default=3, help='Number of agents to sample from the dataset.')
    parser.add_argument('--num_datapoints_per_agent_to_sample', type=int, default=3, help='Number of datapoints per agent to sample from the dataset.')
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
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", help='Name of the model to use.')  # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct or meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of tensor parallel size.')
    parser.add_argument('--dtype', type=str, default="float16", help='Data type.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization.')
    parser.add_argument('--overfit', type=bool, default=False, help='Whether to overfit on a single environment.')
    args = parser.parse_args()
    
    # Check if the selected baseline model is implemented
    if args.baseline_model not in ["ToMnet", 'BC', 'TT', 'AutoToM', 'FSM']:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    return args

def make_dataloader(args, num_agents_to_sample: int = 2, num_datapoints_per_agent_to_sample: int = 20, overfit: bool = False, training: bool = False):
    """Load data from the dataset folders."""
    data_path = args.data_path
    as_images = args.as_images

    i = 0
    while True:    
        partnr_data_folder = "/mmfs1/gscratch/socialrl/kjha/habitat/partnr-planner/outputs/habitat_llm/2025-04-22_17-27-57-val_mini.json/results/val_mini.json.gz/state_action_traj_data"
        episode_dirs = os.listdir(partnr_data_folder)
        episode_dir = random.choice(episode_dirs)
        episode_path = os.path.join(partnr_data_folder, episode_dir, 'agent_0.json.gz')
        with gzip.open(episode_path, "rt") as f:
            episode = json.load(f)
        

        states = []
        actions = []
        for timepoint in range(len(episode)):
            curr_graph = episode[timepoint]['curr_graph']['0']
            action = episode[timepoint]['agent_action']
            split_graph = curr_graph.split('\n')
            # Initialize dictionaries for furniture and objects
            scene_graph = {
                'furniture': {},  # room -> list of furniture
                'objects': {}     # room -> list of objects
            }
            # Track current section being parsed
            current_section = None
            current_room = None
            for line in split_graph:
                '''data may be 
                    an empty line, 
                    a header which says something like "Furniture:" or "Objects:" 
                    or it could be in the format "room_name: item_in_room, item_in_room, ..."
                '''

                # Parse each line
                if line.strip():  # Skip empty lines
                    if line.endswith(':'):  # Section header
                        current_section = line.strip(':').lower()
                        if current_section not in scene_graph:
                            scene_graph[current_section] = {}
                    elif ':' in line:  # Room contents
                        room, items = line.split(':')
                        room = room.strip()
                        items = items.split(',')
                        temp = []
                        for item in items:
                            item = item.strip()
                            temp.append(item)
                        items = temp

                        if current_section == 'furniture':
                            scene_graph['furniture'][room] = items
                        elif current_section == 'objects':
                            scene_graph['objects'][room] = items
            try:
                state = {
                    'scene_graph': scene_graph,
                    'agent_state': episode[timepoint]['agent_state'],
                    'tool_list': episode[timepoint]['tool_list'],
                    'tool_descriptions': episode[timepoint]['tool_descriptions'],
                }
            except Exception as e:
                print(f"Error: {e}")
                breakpoint()
            states.append(state)
            actions.append(action)
        
        yield states, actions
        del states, actions

def eval_thoughtTrace(args, dataloader, model):
    for i in range(args.num_epochs):
        datapoint = next(dataloader)
        state = datapoint['states']
        actions = datapoint['actions']
        agent_ids = datapoint['agent_ids']

        predicted_final_action = model.predict_action(state, actions, agent_ids)

        breakpoint()

def eval_autoToM(args, dataloader, model):
    num_correct = 0
    num_total = 0
    for i in range(args.num_epochs):
        datapoint = next(dataloader)

        for a in range(args.num_agents_to_sample):
            data = jax.tree.map(lambda x: x[a, 0, :15], datapoint)
            states = data['states']
            actions = data['actions']  # (20, 1)  # 20 timesteps, 1 action
            agent_ids = data['agent_ids']
            tries = 0
            while tries < 6:
                try:
                    predicted_final_action, predicted_probs = model.predict_action(states, actions, agent_ids)
                    final_action_prob = predicted_probs[actions[-1, 0]]
                    if final_action_prob >= np.max(predicted_probs):
                        num_correct += 1
                    break  # end loop if prediction is successful
                except Exception as e:
                    tries += 1
            num_total += 1
            if tries == 6:
                print(f"Failed to make a prediction for agent {a}")



    print(f"Accuracy: {num_correct / num_total}")
    return num_correct / num_total

def eval_fsm(args, dataloader, model):
    num_correct = 0
    num_total = 0
    num_successes = 0
    for i in tqdm(range(args.num_epochs)):
        datapoint = next(dataloader)
        states, actions = datapoint
        agent_ids = [0] * len(states)
        tries = 0
        succeeded = False
        while tries < 6:
            # try:
            predicted_final_action = model.predict_action(states, actions, agent_ids)
            if predicted_final_action is not None:
                succeeded = True
                num_successes += 1
                break
            else:
                tries += 1
        if predicted_final_action is None:
            final_action_prob = 0
            print(f"Failed to make a prediction for agent {a}")
            if args.group:
                num_total += 4
            else:
                num_total += 1
        else:
            if not args.group:
                gt_action = actions[-2]
                correct_action = True
                for aid in range(len(gt_action)):
                    ga = gt_action[aid]
                    pa = predicted_final_action[aid]
                    if ga != pa:
                        correct_action = False
                        break
                if correct_action:
                    num_correct += 1
                num_total += 1
            else:
                gt_final_action = actions[-1]
                for aid in range(len(gt_final_action)):
                    final_action_prob = predicted_final_action[aid, gt_final_action[aid]]
                    if final_action_prob >= np.max(predicted_final_action[aid]):  # only count a success if the model succeesfully made a prediction
                        num_correct += 1
                    num_total += 1
                # else:
                #     num_correct += 0
                # breakpoint()

            
            
    print(f"Successful Prediction rate: {num_successes / num_total}")
    print(f"Successful Prediction Accuracy: {num_correct / num_successes}")
    print(f"Total Accuracy: {num_correct / num_total}")
    return num_correct / num_total
            

def load_tomnet_models(args):
    """Load ToMnet models from multiple seeds."""
    states = []
    
    # Define the learning rate to use
    lr = args.learning_rate

    # Initialize model
    model = ToMNet(character_net_features=16, mental_net_features=32, output_size=6)
    
    for seed in range(2):
        # Construct the path pattern similar to what's used in train_baselines.py
        model_path = f"baselines/ToMnet/{args.save_path}/nagents{args.num_agents_to_sample}_ndatapoints{args.num_datapoints_per_agent_to_sample}_seed{seed}_lr{lr}"
        
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
            gt_final_actions = data['actions'][-1, -1, 0]

            def single_agent_pred(param_state, data):
                # Include batch_stats in the variables dictionary
                variables = {'params': param_state['params'], 'batch_stats': param_state['batch_stats']}
                res = model.apply(variables, data['states'], data['actions'], training=False)
                return res
            
            action_preds = jax.vmap(single_agent_pred, in_axes=(0, None))(states, data)  # num loaded params, num_timepoints - 1, 6

            action_preds = action_preds[:, -1, :]  # get final timepoint prediction for each agent

            def single_action_acc(action_preds, gt_final_action):
                action_pred = jnp.argmax(action_preds, axis=-1)
                return action_pred == gt_final_action

            action_accs = jax.vmap(single_action_acc, in_axes=(0, None))(action_preds, gt_final_actions)

            num_correct += jnp.sum(action_accs)
            num_total += action_accs.shape[0]

    accuracy = num_correct / num_total if num_total > 0 else 0
    print(f"ToMnet Ensemble Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
    return accuracy

def main():
    """Main function to eval baseline models."""
    args = parse_args()
    
    # Set JAX memory allocation to grow as needed
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    print(f"Evaluating baseline model: {args.baseline_model}; n_hypothesis: {args.n_hypothesis}")
    save_path = f"baselines/{args.baseline_model}/{args.save_path}/nagents{args.num_agents_to_sample}_ndatapoints{args.num_datapoints_per_agent_to_sample}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize random key for parameter initialization
    rng_key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.baseline_model == "TT":
        from baselines.thoughtTrace import ThoughtTrace
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model = ThoughtTrace(n_hypothesis=args.n_hypothesis, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
        eval_fn = eval_thoughtTrace
    elif args.baseline_model == "AutoToM":
        from baselines.AutoToM.autoToM import AutoToM
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model = AutoToM(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
        eval_fn = eval_autoToM
    elif args.baseline_model == "FSM":
        from baselines.llmFSM import FSMReasoner
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model = FSMReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, num_hypothesis=args.n_hypothesis, group=args.group)
        eval_fn = eval_fsm
    elif args.baseline_model == "ToMnet":
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model, states = load_tomnet_models(args)
        if not model:
            print("No ToMnet models found. Please train models first.")
            return
        eval_fn = lambda a, d, m: eval_mtom(a, d, m, states)
    else:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    res = eval_fn(args, dataloader, model)
    # Create results dictionary
    results = {
        'model': [args.baseline_model],
        'accuracy': [res],
        'group': [args.group],
        'num_agents_evaluated': [args.num_agents_to_sample],
        'datapoints_per_agent': [args.num_datapoints_per_agent_to_sample], 
        'num_epochs': [args.num_epochs],
        'llm_model': [args.model_name] if args.baseline_model in ['TT', 'AutoToM', 'FSM'] else ['N/A'],
        'num_hypothesis': [args.n_hypothesis] if args.baseline_model in ['TT', 'FSM'] else ['N/A']
    }

    # Convert to dataframe
    df = pd.DataFrame(results)

    # Create directory if it doesn't exist
    csv_path = f"baselines/{args.baseline_model}/accuracy_{args.baseline_model}_{args.n_hypothesis}hyp.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Append or create CSV file
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()