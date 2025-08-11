import argparse
import msgpack
import flax
import os
import jax.numpy as jnp
import jax
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from environment import state_to_image_jit
import optax
from flax.training import train_state
from flax import struct

from baselines.ToMnet import ToMNet
from baselines.BC import BCNet

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline models for automaticity.")
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="ToMnet",
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
    parser.add_argument('--num_datapoints_per_agent_to_sample', type=int, default=3, help='Number of datapoints per agent to sample from the dataset.')
    parser.add_argument('--num_agents', type=int, default=1, help='Number of agents in the dataset.')
    parser.add_argument('--num_datapoints_per_agent', type=int, default=100, help='Number of datapoints per agent in the dataset.')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps in the dataset.')
    parser.add_argument('--env_size', type=int, default=10, help='Size of the environment.')
    # parser.add_argument('--num_blocks', type=int, default=10, help='Number of blocks in the dataset.')
    # parser.add_argument('--num_walls', type=int, default=10, help='Number of walls in the dataset.')

    parser.add_argument('--as_images', type=bool, default=True, help='Whether to load the data as images.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save the model.')
    parser.add_argument('--seed', type=int, default=30, help='Random seed.')
    parser.add_argument('--overfit', type=bool, default=False, help='Whether to overfit on a single environment.')
    
    args = parser.parse_args()
    
    # Check if the selected baseline model is implemented
    if args.baseline_model not in ["ToMnet", 'BC']:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    return args

def make_dataloader(args, num_agents_to_sample: int = 2, num_datapoints_per_agent_to_sample: int = 20, overfit: bool = False, training: bool = True):
    """Load data from the dataset folders."""
    data_path = args.data_path
    as_images = args.as_images

    i = 0
    while True:
        if not overfit:
            i += 1
            num_blocks = random.choice(list(range(3,8,1)))
            num_walls = random.choice(list(range(1, 5, 1)))
        else:
            num_blocks = 4
            num_walls = 1
            
        data_folder = f"{data_path}/num_blocks{num_blocks}/num_walls{num_walls}"
        if args.group:
            extension = "_group"
            num_agents_acting = 4
        else:
            extension = "_flip_quarter"
            num_agents_acting = 1
            
        data_file = f"{data_folder}/gt_fsm{extension}_traj_data_{args.num_agents}agents.msgpack"

        # Load data in chunks to reduce memory usage
        with open(data_file, "rb") as f:
            serialized_data = f.read()
            
        # Create target structure with correct shapes
        action_target = jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, 4))
        agent_id_target = jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps))
        state_target = {
            'agent_id': agent_id_target,
            'agent_locations': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, num_agents_acting, 2)),
            'agent_inventory': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, num_agents_acting, 1)),
            'agent_inventory_colors': jnp.zeros((20, args.num_datapoints_per_agent, args.num_steps, num_agents_acting, 3)),
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
        # for key in target['states'].keys():
        #     if loaded_data['states'][key].shape != target['states'][key].shape:
        #         # print(f"Shape mismatch for {key}:")
        #         # print(f"Target shape: {target['states'][key].shape}")
        #         # print(f"Loaded shape: {loaded_data['states'][key].shape}")
        #         continue
                
        if not as_images:
            yield loaded_data
            del loaded_data
        else:
            # Sample only what we need
            if overfit:
                agent_indices = jax.random.randint(jax.random.PRNGKey(i), (num_agents_to_sample,), minval=0, maxval=1)
            else:
                agent_indices = jax.random.randint(jax.random.PRNGKey(i), (num_agents_to_sample,), minval=0, maxval=loaded_data['states']['agent_locations'].shape[0])
            # print(f"Agent indices: {agent_indices}")
            # agent_indices = jnp.array([8])
            # if not overfit:
            i += 1
            if training:  # for creating held out set
                batch_floor = 0
                batch_limit = int(loaded_data['states']['agent_locations'].shape[1] * 0.8)
                # if overfit:
                #     batch_limit = 1
            else:
                batch_floor = int(loaded_data['states']['agent_locations'].shape[1] * 0.8)
                batch_limit = loaded_data['states']['agent_locations'].shape[1]

            batch_indices = jax.random.randint(jax.random.PRNGKey(i), (num_datapoints_per_agent_to_sample,), minval=batch_floor, maxval=batch_limit)

            # if not overfit:
            i += 1
            
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

                img_size = args.env_size * 7
                tile_size = 7
                grid_size = args.env_size
                img_gen_fn = jax.jit(state_to_image_jit, static_argnums=(1, 2, 3))
                render_fn = lambda x: img_gen_fn(x, img_size, grid_size, tile_size)
                
                def convert_to_image(index, stacked_state):
                    indexed_state = jax.tree.map(lambda x: x[index], stacked_state)
                    return render_fn(indexed_state)
                
                batch_images = jax.vmap(convert_to_image, in_axes=(0, None))(jnp.arange(len(batch_indices)), batch_state)
                all_images.append(batch_images)
            
            # Concatenate all batches
            stacked_images = jnp.concatenate(all_images, axis=0)
            
            # Free memory
            del reshaped_state, all_images
            
            sampled_data['states'] = stacked_images.reshape(num_agents_to_sample, num_datapoints_per_agent_to_sample, args.num_steps//2, *stacked_images.shape[1:])

            yield sampled_data
            del sampled_data

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict

def create_train_state(rng_key, model, learning_rate, example_states, example_actions):
    """Create initial training state."""
    # Create learning rate schedule
    lr_schedule = optax.linear_schedule(
        init_value=learning_rate,
        end_value=learning_rate / 100.0,
        transition_steps=10000,
    )
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=lr_schedule)
    )
    
    # Initialize model
    variables = model.init(rng_key, example_states, example_actions)
    
    # Create train state
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables.get('batch_stats', flax.core.FrozenDict())
    )

def tomnet_train_step(state, batch, rng_key):
    """Train for a single step."""
    # Extract states and actions from batch
    states = batch['states']  # Shape: (num_agents, num_datapoints_per_agent, num_steps, height, width, 3)
    actions = batch['actions']  # Shape: (num_agents, num_datapoints_per_agent, num_steps, 1)
    
    # Use scan instead of vmap to reduce memory usage
    def loss_fn(params):
        def single_agent_loss(agent_idx):
            agent_states = states[agent_idx]
            agent_actions = actions[agent_idx]
            
            # Forward pass through the model
            variables = {'params': params, 'batch_stats': state.batch_stats}
            action_prediction, updates = state.apply_fn(  # states are num traj, num_steps, height, width, 3
                variables, 
                agent_states, 
                agent_actions,
                mutable=['batch_stats']  # Make batch_stats mutable
            )
            action_prediction = jnp.transpose(action_prediction, (1, 0, 2))  # (num_steps, num_agents, 6)
           
            action_prediction = action_prediction.reshape(-1, 6)

            gt_action = agent_actions[-1:, 1:, :]  # Get actions from the last batch
            # Convert ground truth action to one-hot
            gt_action_one_hot = jax.nn.one_hot(gt_action, num_classes=6)  
            gt_action_one_hot = gt_action_one_hot.reshape(-1, 6) # (batch, 6)
            
            # Compute negative log likelihood loss
            epsilon = 1e-8
            nll_loss = -jnp.sum(gt_action_one_hot * jnp.log(action_prediction + epsilon), axis=-1)
            return nll_loss.mean(), updates
        
        # Process agents one by one to reduce memory usage
        total_loss = 0
        all_updates = None
        
        for agent_idx in range(states.shape[0]):
            agent_loss, agent_updates = single_agent_loss(agent_idx)
            total_loss += agent_loss
            
            # Accumulate updates
            if all_updates is None:
                all_updates = agent_updates
            else:
                all_updates = jax.tree.map(
                    lambda x, y: x + y, 
                    all_updates, 
                    agent_updates
                )
        
        # Average the updates
        all_updates = jax.tree.map(
            lambda x: x / states.shape[0], 
            all_updates
        )
        
        return total_loss / states.shape[0], all_updates
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    
    # Update state
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=updates['batch_stats']
    )
    
    return new_state, loss

def bc_train_step(state, batch, rng_key):
    """Train for a single step."""
    # Extract states and actions from batch
    states = batch['states']  # Shape: (num_agents, num_datapoints_per_agent, num_steps, height, width, 3)
    actions = batch['actions']  # Shape: (num_agents, num_datapoints_per_agent, num_steps, 1)

    def loss_fn(params):
        def single_agent_loss(agent_idx):
            agent_states = states[agent_idx]
            agent_actions = actions[agent_idx]
            
            # Forward pass through the model
            variables = {'params': params, 'batch_stats': state.batch_stats}
            action_prediction, updates = state.apply_fn(
                variables, 
                agent_states, 
                agent_actions,
                mutable=['batch_stats']  # Make batch_stats mutable
            )
            
            action_prediction = jnp.transpose(action_prediction, (1, 0, 2))  # (num_steps * batch_size, num_agents, 6)
            action_prediction = action_prediction.reshape(-1, 6)

            gt_action = agent_actions
            # Convert ground truth action to one-hot
            gt_action_one_hot = jax.nn.one_hot(gt_action, num_classes=6)  
            gt_action_one_hot = gt_action_one_hot.reshape(-1, 6) # (batch, 6)
            
            # Compute negative log likelihood loss
            epsilon = 1e-8
            nll_loss = -jnp.sum(gt_action_one_hot * jnp.log(action_prediction + epsilon), axis=-1)
            return nll_loss.mean(), updates
        
        # Process agents one by one to reduce memory usage
        total_loss = 0
        all_updates = None
        
        for agent_idx in range(states.shape[0]):
            agent_loss, agent_updates = single_agent_loss(agent_idx)
            total_loss += agent_loss
            
            # Accumulate updates
            if all_updates is None:
                all_updates = agent_updates
            else:
                all_updates = jax.tree.map(
                    lambda x, y: x + y, 
                    all_updates, 
                    agent_updates
                )
        
        # Average the updates
        all_updates = jax.tree.map(
            lambda x: x / states.shape[0], 
            all_updates
        )
        
        return total_loss / states.shape[0], all_updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    
    # Update state
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=updates['batch_stats']
    )
    
    return new_state, loss
    

def main():
    """Main function to train baseline models."""
    args = parse_args()
    
    group_decide = args.group
    print(f"Predicting group: {group_decide}")

    # Set JAX memory allocation to grow as needed
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    print(f"Training baseline model: {args.baseline_model}")
    save_path = f"baselines/{args.baseline_model}/{args.save_path}/nagents{args.num_agents_to_sample}_ndatapoints{args.num_datapoints_per_agent_to_sample}_seed{args.seed}_lr{args.learning_rate}_group{args.group}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize random key for parameter initialization
    rng_key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.baseline_model == "ToMnet":
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, overfit=args.overfit)

        # Get a batch of data for initialization
        example_batch = next(dataloader)
        
        # Initialize model
        if args.group:
            num_to_predict = 4
        else:
            num_to_predict = 1
        model = ToMNet(character_net_features=64, mental_net_features=128, output_size=6, num_to_predict=num_to_predict)
        train_step = tomnet_train_step
    elif args.baseline_model == "BC":
        dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, overfit=args.overfit)

        # Get a batch of data for initialization
        example_batch = next(dataloader)
        
        # Initialize model
        if args.group:
            num_to_predict = 4
        else:
            num_to_predict = 1
        model = BCNet(output_size=6, hidden_size=32, num_to_predict=num_to_predict)
        train_step = bc_train_step

    # Check for existing checkpoints
    checkpoint_files = [f for f in os.listdir(os.path.dirname(save_path)) 
                        if os.path.basename(save_path) in f and "checkpoint" in f]
    
    start_epoch = 0
    if checkpoint_files and not args.overfit:
        # Find the most recent checkpoint
        checkpoint_epochs = [int(f.split("epoch")[-1].split(".")[0]) for f in checkpoint_files]
        most_recent_epoch = max(checkpoint_epochs)
        most_recent_checkpoint = [f for f in checkpoint_files if f"epoch{most_recent_epoch}" in f][0]
        checkpoint_path = os.path.join(os.path.dirname(save_path), most_recent_checkpoint)
        
        print(f"Loading checkpoint from epoch {most_recent_epoch}: {checkpoint_path}")
        
        
        # Create training state from checkpoint
        example_states = example_batch['states'][0]
        example_actions = example_batch['actions'][0]
        train_state = create_train_state(
            rng_key, 
            model, 
            args.learning_rate,
            example_states,
            example_actions
        )

        with open(checkpoint_path, 'rb') as f:
            target = {
                'params': train_state.params,
                'batch_stats': train_state.batch_stats,
                'epoch': most_recent_epoch,
                'loss': 0.0
            }
            checkpoint = flax.serialization.from_bytes(target, f.read())
        
        # Update training state with checkpoint values
        train_state = train_state.replace(
            params=checkpoint['params'],
            batch_stats=checkpoint['batch_stats']
        )
        
        start_epoch = checkpoint['epoch']
        epoch_losses = [checkpoint['loss']]  # Start with the last recorded loss
        print(f"Resuming training from epoch {start_epoch}")
    else:
        # Create new training state
        example_states = example_batch['states'][0]  # First agent's states
        example_actions = example_batch['actions'][0]  # First agent's actions
        train_state = create_train_state(
            rng_key, 
            model, 
            args.learning_rate,
            example_states,
            example_actions
        )
        epoch_losses = []
        print("No checkpoint found. Starting training from scratch.")

    # Training loop
    for epoch in tqdm(range(start_epoch, args.num_epochs), desc=f"Training"):            
        batch = next(dataloader)
        rng_key, step_key = jax.random.split(rng_key)
        train_state, loss = train_step(train_state, batch, step_key)
        epoch_losses.append(loss)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss:.4f}")
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint = {
                'params': train_state.params,
                'batch_stats': train_state.batch_stats,
                'epoch': epoch + 1,
                'loss': loss
            }
            with open(f"{save_path}_checkpoint_epoch{epoch+1}.msgpack", 'wb') as f:
                f.write(flax.serialization.to_bytes(checkpoint))
            
            # Plot and save training curve with sliding window average
            plt.figure(figsize=(10, 6))
            sns.set_context("paper", font_scale=1.5)
            sns.set_style("whitegrid")
            
            # Calculate sliding window average (window size = min(50, 10% of data points))
            window_size = min(50, max(5, len(epoch_losses) // 10))
            if len(epoch_losses) > window_size:
                # Calculate moving average
                smoothed_losses = np.convolve(epoch_losses, np.ones(window_size)/window_size, mode='valid')
                # Plot both raw and smoothed losses
                plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, alpha=0.3, color='blue', label='Raw Loss')
                plt.plot(range(window_size, len(epoch_losses) + 1), smoothed_losses, linewidth=2, color='red', 
                         label=f'Moving Avg (window={window_size})')
                plt.legend()
            else:
                # If we don't have enough data points yet, just plot the raw losses
                plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Loss')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss - {args.baseline_model}')
            plt.tight_layout()
            plt.savefig(f"{save_path}_training_loss.pdf")
            plt.close()
    
    # Save final model
    final_checkpoint = {
        'params': train_state.params,
        'batch_stats': train_state.batch_stats,
        'epoch': args.num_epochs,
        'loss': epoch_losses[-1]
    }
    with open(f"{save_path}_final.msgpack", 'wb') as f:
        f.write(flax.serialization.to_bytes(final_checkpoint))
        
    # Plot final training curve with sliding window average
    plt.figure(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")
    
    # Calculate sliding window average (window size = min(50, 10% of data points))
    window_size = min(50, max(5, len(epoch_losses) // 10))
    if len(epoch_losses) > window_size:
        # Calculate moving average
        smoothed_losses = np.convolve(epoch_losses, np.ones(window_size)/window_size, mode='valid')
        # Plot both raw and smoothed losses
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, alpha=0.3, color='blue', label='Raw Loss')
        plt.plot(range(window_size, len(epoch_losses) + 1), smoothed_losses, linewidth=2, color='red', 
                 label=f'Moving Avg (window={window_size})')
        plt.legend()
    else:
        # If we don't have enough data points yet, just plot the raw losses
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {args.baseline_model}')
    plt.tight_layout()
    plt.savefig(f"{save_path}_final_training_loss.pdf")
    plt.close()
        
    print("Training completed!")

if __name__ == "__main__":
    main()
    # args = parse_args()
    # dataloader = make_dataloader(args, num_agents_to_sample=args.num_agents_to_sample, num_datapoints_per_agent_to_sample=args.num_datapoints_per_agent_to_sample, overfit=args.overfit)
    
    # obs_traj = []
    # act_traj = []
    # for num_dat in range(25):
    #     batch = next(dataloader)
    #     obs = batch['states'].reshape(-1, *batch['states'].shape[-3:])
    #     act = batch['actions'].reshape(-1, *batch['actions'].shape[-1:])
    #     obs_traj.append(obs)
    #     act_traj.append(act)
    # obs_traj = jnp.concatenate(obs_traj, axis=0)
    # act_traj = jnp.concatenate(act_traj, axis=0)
    # # Save trajectories to pickle file
    # # Convert to numpy arrays
    # obs_array = np.array(obs_traj)
    # act_array = np.array(act_traj)
    
    # Create directory if it doesn't exist
    # save_dir = 'data/trajectories'
    # os.makedirs(save_dir, exist_ok=True)
    
    # Save with descriptive filename including num agents and datapoints
    # save_path = os.path.join(save_dir, f'dummy_trajectory.npz')
    
    # # Save arrays using npz format
    # np.savez(save_path, 
    #          observations=obs_array,
    #          actions=act_array)
        
    # print(f"Saved trajectories to {save_path}")


    # Load trajectories from npz file
    # data = np.load(save_path)
    # observations = data['observations']
    # actions = data['actions']

    # print(observations.shape)
    # print(actions.shape)

