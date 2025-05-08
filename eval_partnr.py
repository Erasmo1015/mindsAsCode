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
import traceback

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
    parser.add_argument('--n_hypothesis', type=int, default=2, help='Number of hypothesis for thought trace.')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", help='Name of the model to use.')  # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct or meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of tensor parallel size.')
    parser.add_argument('--dtype', type=str, default="float16", help='Data type.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization.')
    parser.add_argument('--overfit', type=bool, default=False, help='Whether to overfit on a single environment.')
    parser.add_argument('--bootstrap', action='store_true', help='Whether to evaluate with bootstrapping for different numbers of hypotheses.')
    parser.add_argument('--rejuvenation', action='store_true', help='Whether to use rejuvenation for FSM evaluation.')
    parser.add_argument('--rejuvenation_threshold', type=float, default=-10, help='Threshold for rejuvenation in FSM.')
    parser.add_argument('--max_rejuvenation_attempts', type=int, default=5, help='Maximum number of rejuvenation attempts in FSM.')
    parser.add_argument('--top_k', type=int, default=0, help='Number of top hypotheses to consider (0 means use all).')
    parser.add_argument('--two_stage', action='store_true', help='Whether to use two-stage reasoning for FSM.')
    parser.add_argument('--structured', type=str, default="False", choices=["False", "p1", "p2"], 
                        help='Type of structured reasoning to use (False, "p1", or "p2").')
    args = parser.parse_args()
    
    # Check if the selected baseline model is implemented
    if args.baseline_model not in ["ToMnet", 'BC', 'TT', 'AutoToM', 'FSM', 'NLLM']:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    if args.baseline_model == 'AutoToM':
        os.environ['CURRENT_MODEL_NAME'] = args.model_name
    
    return args

def make_dataloader(args, num_agents_to_sample: int = 2, num_datapoints_per_agent_to_sample: int = 20, overfit: bool = False, training: bool = False):
    """Load data from the dataset folders."""
    data_path = args.data_path
    as_images = args.as_images

    i = 0
    while True:    
        if not args.group:
            partnr_data_folder = "/mmfs1/gscratch/socialrl/kjha/habitat/partnr-planner/outputs/habitat_llm/single_agent_traj_data/results/single_agent_traj_data/state_action_traj_data"
        else:
            partnr_data_folder = "/mmfs1/gscratch/socialrl/kjha/habitat/partnr-planner/outputs/habitat_llm/group_traj_data/results/group_traj_data/state_action_traj_data"
        episode_dirs = os.listdir(partnr_data_folder)
        episode_dir = random.choice(episode_dirs)
        if args.group:
            agent_id_list = [0, 1]
        else:
            agent_id_list = [0]
        for agent_id in agent_id_list:
            episode_path = os.path.join(partnr_data_folder, episode_dir, f'agent_{agent_id}.json.gz')
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

def eval_naive_LLM(args, dataloader, model, episode_id: int = 0):
    num_correct = 0
    num_total = 1
    datapoint = next(dataloader)
    states, actions = datapoint
    agent_ids = [0] * len(states)
    tries = 0
    succeeded = False

    gt_action = actions[-2][0]
    
    # while tries < 6:
    # try:
    try:
        predicted_final_action = model.predict_action(states, actions, agent_id=0, episode_id=episode_id)
        if predicted_final_action.lower() == gt_action.lower():
            num_correct += 1
        # break  # end loop if prediction is successful
    except Exception as e:
        print(f"Error: {e}")
        tries += 1
    # num_total += 1
    accuracy = num_correct / num_total
    print(f"Accuracy: {accuracy}")
    return accuracy

def eval_autoToM(args, dataloader, model, episode_id: int = 0):
    num_correct = 0
    num_total = 1
    datapoint = next(dataloader)
    states, actions = datapoint
    agent_ids = [0] * len(states)
    tries = 0
    succeeded = False

    gt_action = actions[-2][0]
    
    # while tries < 6:
    try:
        predicted_final_action, predicted_probs, choices = model.predict_action(states, actions, agent_id=0, episode_id=episode_id)
        gt_id = choices.index(gt_action)
        max_prob = np.max(predicted_probs)
        if predicted_probs[gt_id] == max_prob:
            num_correct += 1
        # break  # end loop if prediction is successful
    except Exception as e:
        print(f"Error: {e}")
        tries += 1
    # num_total += 1
    accuracy = num_correct / num_total
    print(f"Accuracy: {accuracy}")
    return accuracy
            

def eval_fsm(args, dataloader, model, episode_id: int = 0):
    num_correct = 0
    num_total = 0
    num_successes = 0

    datapoint = next(dataloader)
    states, actions = datapoint
    agent_ids = [0] * len(states)
    tries = 0
    succeeded = False
    if args.group:
        num_total += 2
    else:
        num_total += 1

    try:
        predicted_final_action = model.predict_action(states, actions)
        if predicted_final_action is not None:
            succeeded = True
            num_successes += 1
        else:
            tries += 1
        if predicted_final_action is None:
            final_action_prob = 0
            print(f"Failed to make a prediction for agent {a}")
        else:
            if not args.group:
                gt_action = actions[-2][0]
                predicted_final_action = predicted_final_action[0]
                print(f"gt_action: {gt_action}, predicted_final_action: {predicted_final_action}")
                if gt_action == predicted_final_action:
                    num_correct += 1
            else:
                gt_final_action = actions[-1]
                for aid in range(len(gt_final_action)):
                    final_action_prob = predicted_final_action[aid, gt_final_action[aid]]
                    if final_action_prob >= np.max(predicted_final_action[aid]):  # only count a success if the model succeesfully made 
                        num_correct += 1
    except Exception as e:
        # print(f"Error: {e}")
        pass

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

def eval_fsm_bootstrap(args, dataloader, model, episode_id: int = 0):
    """Evaluate FSM with bootstrapping for different numbers of hypotheses."""
    max_hypotheses = args.n_hypothesis  # Maximum number of hypotheses to consider
    results = {n: {'correct': 0, 'total': 0, 'program_length': 0} for n in range(1, max_hypotheses + 1)}
    
    datapoint = next(dataloader)
    states, actions = datapoint
    agent_ids = [0] * len(states)
    
    if args.group:
        num_agents_per_sample = 2
    else:
        num_agents_per_sample = 1
        
    try:
        # Enable bootstrapping
        model.bootstrap = True
        
        # Choose between rejuvenation and regular bootstrap based on args
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
            return {n: 0.0 for n in range(1, max_hypotheses + 1)}, {n: 0.0 for n in range(1, max_hypotheses + 1)}
            
        gt_final_action = actions[-2][0] if not args.group else actions[-1]
        
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
            
            if not args.group:
                if prediction == gt_final_action:
                    results[n_hyp]['correct'] += 1
            else:
                if len(prediction.shape) == 1:
                    prediction = prediction.reshape(1, -1)
                
                for aid in range(len(gt_final_action)):
                    final_action_prob = prediction[aid, gt_final_action[aid]]
                    if final_action_prob >= np.max(prediction[aid]):
                        results[n_hyp]['correct'] += 1
                    
    except Exception as e:
        print(f"Error: {e}")
        full_traceback = traceback.format_exc()
        print(full_traceback)
    
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
        print(f"Hypotheses: {n_hyp}, Accuracy: {acc:.4f} ({results[n_hyp]['correct']}/{results[n_hyp]['total']}), Avg Program Length: {program_lengths[n_hyp]:.2f}")
    
    return accuracies, program_lengths

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

    # Create CSV path
    if args.group:
        group_extension = "_group"
    else:
        group_extension = ""
    
    two_stage_extension = "_two_stage" if args.two_stage else ""
    structured_extension = f"_structured_{args.structured}" if args.structured != "False" else ""
    rejuvenation_extension = "_rejuvenation" if args.rejuvenation else ""
        
    if args.bootstrap and args.baseline_model == "FSM":
        csv_path = f"baselines/{args.baseline_model}/partnr_bootstrap_accuracy_{args.baseline_model}{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}_topk{args.top_k}.csv"
    else:
        csv_path = f"baselines/{args.baseline_model}/partnr_accuracy_{args.baseline_model}_{args.n_hypothesis}hyp{group_extension}{two_stage_extension}{structured_extension}{rejuvenation_extension}.csv"
    
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
        
        if len(matching_rows) > 0 and not args.bootstrap:
            # Get the highest epoch completed
            start_epoch = matching_rows['epoch'].max() + 1
            print(f"Resuming from epoch {start_epoch}")
    
    if args.baseline_model == "TT":
        from baselines.thoughtTrace import ThoughtTrace
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model = ThoughtTrace(n_hypothesis=args.n_hypothesis, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
        eval_fn = eval_thoughtTrace
    elif args.baseline_model == "AutoToM":
        from baselines.AutoToM.partnr_autoToM import AutoToM
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model = AutoToM(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, group=args.group)
        eval_fn = eval_autoToM
    elif args.baseline_model == "FSM":
        from baselines.llmFSM import FSMReasoner
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model = FSMReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, 
                           dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, 
                           num_hypothesis=args.n_hypothesis, group=args.group, 
                           two_stage=args.two_stage, structured=args.structured)
        if args.bootstrap:
            eval_fn = eval_fsm_bootstrap
        else:
            eval_fn = eval_fsm
    elif args.baseline_model == "ToMnet":
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model, states = load_tomnet_models(args)
        if not model:
            print("No ToMnet models found. Please train models first.")
            return
        eval_fn = lambda a, d, m: eval_mtom(a, d, m, states)
    elif args.baseline_model == "NLLM":
        from baselines.basic_LLM import NaiveLLMReasoner
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, training=False)
        model = NaiveLLMReasoner(model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, num_hypothesis=args.n_hypothesis, group=args.group, partnr=True)
        eval_fn = eval_naive_LLM
    else:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    # Run evaluation for each epoch and save results after each epoch
    all_results = []
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Running epoch {epoch}/{args.num_epochs}")
        
        if args.bootstrap and args.baseline_model == "FSM":
            accuracies, program_lengths = eval_fn(args, dataloader, model, episode_id=epoch)
            
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
        else:
            if args.baseline_model == "FSM" or args.baseline_model == "AutoToM" or args.baseline_model == "NLLM":
                res = eval_fn(args, dataloader, model, episode_id=epoch)
            else:
                res = eval_fn(args, dataloader, model)
            
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