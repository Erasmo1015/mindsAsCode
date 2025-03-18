import argparse
import msgpack
import flax
import os
import jax.numpy as jnp
import jax
import numpy as np
import random
from tqdm import tqdm
from environment import state_to_image_jit

from baselines.ToMnet import CharacterNet, MentalNet

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline models for automaticity.")
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="ToMnet",
        help="Baseline model to train. Currently only 'ToMnet' is implemented."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the dataset folders."
    )
    parser.add_argument('--num_agents', type=int, default=20, help='Number of agents in the dataset.')
    parser.add_argument('--num_datapoints_per_agent', type=int, default=100, help='Number of datapoints per agent in the dataset.')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps in the dataset.')
    parser.add_argument('--env_size', type=int, default=10, help='Size of the environment.')
    # parser.add_argument('--num_blocks', type=int, default=10, help='Number of blocks in the dataset.')
    # parser.add_argument('--num_walls', type=int, default=10, help='Number of walls in the dataset.')

    parser.add_argument('--as_images', type=bool, default=True, help='Whether to load the data as images.')
    
    args = parser.parse_args()
    
    # Check if the selected baseline model is implemented
    if args.baseline_model != "ToMnet":
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    return args

def make_dataloader(args, num_agents_to_sample: int = 2, num_datapoints_per_agent_to_sample: int = 20):
    """Load data from the dataset folders."""
    data_path = args.data_path
    as_images = args.as_images

    i = 0
    while True:
        i += 1
        num_blocks = random.choice(list(range(2, 22, 2)))
        num_walls = random.choice(list(range(2, 22, 2)))

        data_folder = f"{data_path}/num_blocks{num_blocks}/num_walls{num_walls}"
        data_file = f"{data_folder}/gt_fsm_traj_data_{args.num_agents}agents.msgpack"


        action_target = jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps, 1))
        agent_id_target = jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps))
        state_target = {
            'agent_id': agent_id_target,
            'agent_locations': jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps, 1, 2)),
            'agent_inventory': jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps, 1)),
            'agent_inventory_colors': jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps, 1, 3)),
            'block_colors': jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps, num_blocks, 3)),
            'block_locations': jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps, num_blocks, 2)),
            'time': jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps)),
            'terminal': jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps)),
            'wall_locations': jnp.zeros((args.num_agents, args.num_datapoints_per_agent, args.num_steps, num_walls + 2 * (args.env_size * 2 - 1) + 2, 2)),
        }
        target = {
            'states': state_target,
            'actions': action_target,
            'agent_ids': agent_id_target,
        }

        with open(data_file, "rb") as f:
            serialized_data = f.read()
            loaded_data = flax.serialization.from_bytes(target, serialized_data)

        # Validate shapes match target
        for key in target['states'].keys():
            if loaded_data['states'][key].shape != target['states'][key].shape:
                print(f"Shape mismatch for {key}:")
                print(f"Target shape: {target['states'][key].shape}")
                print(f"Loaded shape: {loaded_data['states'][key].shape}")
                continue
        if not as_images:
            yield loaded_data
        else:
            # sample batch_size random indices
            agent_indices = jax.random.randint(jax.random.PRNGKey(i), (num_agents_to_sample,), minval=0, maxval=loaded_data['states']['agent_locations'].shape[0])
            i += 1
            batch_indices = jax.random.randint(jax.random.PRNGKey(i), (num_datapoints_per_agent_to_sample,), minval=0, maxval=loaded_data['states']['agent_locations'].shape[1])  # choose from number of datapoints per agent

            sampled_data = jax.tree.map(lambda x: x[agent_indices], loaded_data)
            sampled_data = jax.tree.map(lambda x: x[:, batch_indices], sampled_data)
            # Reshape all arrays to (-1, *original_shape[3:])
            reshaped_state = jax.tree.map(
                lambda x: jnp.array(x).reshape(-1, *x.shape[3:]) if (isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray)) else x,
                sampled_data['states']
            )

            def convert_to_image(index, stacked_state):
                indexed_state = jax.tree.map(lambda x: x[index], stacked_state)
                return state_to_image_jit(indexed_state, args.env_size)
            
            stacked_images = jax.vmap(convert_to_image, in_axes=(0, None))(jnp.arange(reshaped_state['agent_locations'].shape[0]), reshaped_state)

            sampled_data['states'] = stacked_images.reshape(num_agents_to_sample, num_datapoints_per_agent_to_sample, args.num_steps, *stacked_images.shape[1:])
            yield sampled_data

def tomnet_train_step(character_model, mental_model, datapoint, rng_key):
    # Extract states from datapoint
    states = datapoint['states']  # Shape: (num_agents, num_datapoints_per_agent, num_steps, height, width, 3)
    actions = datapoint['actions']  # Shape: (num_agents, num_datapoints_per_agent, num_steps, 1)
    
    # Reshape states to match CharacterNet input requirements
    # We'll process one agent at a time
    agent_idx = 0  # Start with the first agent
    agent_states = states[agent_idx]  # (num_datapoints_per_agent, num_steps, height, width, 3)
    agent_actions = actions[agent_idx]  # (num_datapoints_per_agent, num_steps, 1)
    
    # Initialize model parameters
    rng_key, character_key, mental_key = jax.random.split(rng_key, 3)
    
    # Initialize the models with proper shapes
    character_variables = character_model.init(character_key, agent_states, agent_actions)
    mental_variables = mental_model.init(mental_key, agent_states, agent_actions)
    
    # Forward pass through the model
    mental_output, mental_updated_vars = mental_model.apply(
        mental_variables, 
        agent_states, 
        agent_actions,
        mutable=['batch_stats']  # Make batch_stats mutable
    )
    
    character_output, character_updated_vars = character_model.apply(
        character_variables, 
        agent_states, 
        agent_actions,
        mutable=['batch_stats']  # Make batch_stats mutable
    )
    breakpoint()
    # Here you would typically compute loss and update parameters
    # For now, just return the variables and output for debugging
    return character_updated_vars, mental_updated_vars, character_output

def main():
    """Main function to train baseline models."""
    args = parse_args()
    
    print(f"Training baseline model: {args.baseline_model}")
    dataloader = make_dataloader(args, num_agents_to_sample=2, num_datapoints_per_agent_to_sample=10)
    
    if args.baseline_model == "ToMnet":
        # Initialize the model
        character_model = CharacterNet(output_size=4)  # Adjust output_size based on your action space
        mental_model = MentalNet(output_channels=32)
        
        # Initialize random key for parameter initialization
        rng_key = jax.random.PRNGKey(0)
        
        # Get a batch of data
        datapoint = next(dataloader)
        
        # Train step
        character_updated_vars, mental_updated_vars, output = tomnet_train_step(character_model, mental_model, datapoint, rng_key)
        print(f"Model output shape: {output.shape}")

if __name__ == "__main__":
    main()
