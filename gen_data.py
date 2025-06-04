from environment import AutomaticityEnv, state_to_image_jit
from agent import AgentExecutionFramework
import numpy as np
import matplotlib.pyplot as plt
import imageio
import random
import jax.random as jrandom
import jax
import jax.numpy as jnp
import os
from flax.serialization import msgpack_serialize
from tqdm import tqdm

def initialize_agent_list(agent_id_list, num_agents=1, num_blocks=1, group=False):
    framework = AgentExecutionFramework()
    agent_list = []
    if group:
        extension = "_group"
    else:
        extension = ""
    for i in agent_id_list:
        with open(f"generated_outputs/fsm{extension}_agent_{i+1}.txt", "r") as f:
            agent_code = f.read()
        agent_list.append(framework.compile_agent(agent_code, num_agents=num_agents, num_blocks=num_blocks))
    return agent_list

def initialize_hand_designed_agent_list(num_agents=1, num_blocks=1, group=False):
    framework = AgentExecutionFramework()
    agent_list = []
    files = sorted(os.listdir("generated_outputs/hand_designed"))
    for i in range(len(files)):
        with open(f"generated_outputs/hand_designed/{files[i]}", "r") as f:
            agent_code = f.read()
        agent_list.append(framework.compile_agent(agent_code, num_agents=num_agents, num_blocks=num_blocks))
    return agent_list

def generate_trajectory(agent_id, agent_list, seed=0, num_agents=1, num_steps=100, num_blocks=1, num_walls=20, group=False):
    '''
    agent list should be a list of compiled agents
    '''
    assert len(agent_list) >= num_agents
    env = AutomaticityEnv(num_agents=num_agents, size=7, max_steps=num_steps, num_blocks=num_blocks, num_walls=num_walls)
    key = jrandom.PRNGKey(seed)
    np.random.seed(seed)

    obs, state = env.reset()
    assert len(state.block_locations) == num_blocks

    state_list = []
    action_list = []
    for i in range(num_steps):
        try:
            assert len(state.block_locations) == num_blocks
            assert len(obs[0]['agent_inventory']) >= num_agents
        except:
            breakpoint()
        if group:
            actions = agent_list[agent_id].act(obs[0])  # all agents follow group planner
        else:
            actions = [agent_list[agent_id].act(obs[0])]  # all agents follow same action for now
        assert len(actions) == num_agents
        
        next_obs, next_state = env.step(state, actions)

        state_list.append(obs[0])
        action_list.append(actions)
        obs = next_obs
        state = next_state
    return state_list, action_list, env

def convert_state_to_serializable_dict(state, agent_id, num_blocks):
    assert len(state.block_locations) == num_blocks
    return {
        'wall_locations': state.wall_locations,
        'agent_locations': state.agent_locations,
        'block_locations': state.block_locations,
        'agent_inventory': state.agent_inventory,
        'agent_inventory_colors': state.agent_inventory_colors,
        'block_colors': state.block_colors,
        'time': state.time,
        'terminal': state.terminal,
        'agent_id': agent_id
    }

def convert_state_list_to_serializable_list(state_list, agent_id, num_blocks):
    return [convert_state_to_serializable_dict(state, agent_id, num_blocks) for state in state_list]

def generate_dataset(num_datapoints_per_agent=100, num_steps=100, num_agents=1, num_block_steps=2, num_walls_steps=2, group=False):

    # Create directory if it doesn't exist
    os.makedirs(f"data/agent_num{num_agents}", exist_ok=True)
    for num_blocks_i in range(2, 22, num_block_steps):
        agent_list = initialize_agent_list(range(20), num_agents=num_agents, num_blocks=num_blocks_i, group=group)
        assert len(agent_list) >= num_agents
        for num_walls_i in range(2, 22, num_walls_steps):
            # Lists to collect data for this agent
            all_states = []
            all_actions = []
            all_agent_ids = []
            for agent_id in tqdm(range(len(agent_list))):

                # print(f"Generating dataset for agent {agent_id}, num_blocks={num_blocks_i}, num_walls={num_walls_i}, num_steps={num_steps}, num_datapoints={num_datapoints_per_agent}")
                agent_states = []
                agent_actions = []
                agent_agent_ids = []
                for datapoint in range(num_datapoints_per_agent):               
                    state_list, action_list, env = generate_trajectory(
                        agent_id, agent_list, seed=datapoint, 
                        num_agents=num_agents, num_steps=num_steps, 
                        num_blocks=num_blocks_i, num_walls=num_walls_i, group=group
                    )
                    
                    # Convert state_list to serializable dictionaries
                    serializable_states = convert_state_list_to_serializable_list(state_list, agent_id, num_blocks_i)
                    
                    # Convert to JAX arrays
                    jax_states = jax.tree.map(lambda x: jnp.array(x) if isinstance(x, (np.ndarray, list)) else x, serializable_states)
                    # Stack all items in jax_states using tree_map
                    # Stack all items in jax_states into a single pytree
                    try:
                        try:
                            jax_states = jax.tree.map(lambda *xs: jnp.stack(xs), *jax_states)
                        except Exception as e:
                            # Extract the attribute name from the error message
                            error_str = str(e)
                            print(f"Failed to stack states. Error: {error_str}")
                            # Print the shapes of each attribute to help debug
                            for key in jax_states[0].keys():
                                try:
                                    shapes = [s[key].shape for s in jax_states]
                                    print(f"Shapes for {key}: {shapes}")
                                except:
                                    print(f"Could not get shapes for {key}")
                            raise
                    except Exception as e:
                        print(f"Error stacking states: {e}")
                        breakpoint()
                    
                    # Convert action_list to JAX arrays
                    jax_actions = jnp.array(action_list)
                    
                    
                    # Create agent_id array (same id repeated for all steps)
                    jax_agent_id = jnp.full((len(state_list),), agent_id, dtype=jnp.int32)
                    
                    agent_states.append(jax_states)
                    agent_actions.append(jax_actions)
                    agent_agent_ids.append(jax_agent_id)
                stacked_agent_states = jax.tree.map(lambda *xs: jnp.stack(xs), *agent_states)
                all_states.append(stacked_agent_states)
                all_actions.append(jnp.stack(agent_actions))
                all_agent_ids.append(jnp.stack(agent_agent_ids))
            
            # Make sure the output directory exists
            os.makedirs(f"data/num_blocks{num_blocks_i}/num_walls{num_walls_i}", exist_ok=True)
            
            # Stack all data for this agent
            # For states, we need to be careful with the tree structure
            stacked_states = jax.tree.map(lambda *xs: jnp.stack(xs), *all_states)  # (num_agents, num_datapoints/agent, num_steps, *state_shapes)
            stacked_actions = jnp.stack(all_actions, axis=0)  # (num_agents, num_datapoints/agent, num_steps, num_agents)
            stacked_agent_ids = jnp.stack(all_agent_ids, axis=0)  # (num_agents, num_datapoints/agent, num_steps)
            
            # Create dataset dictionary
            dataset = {
                'states': stacked_states,
                'actions': stacked_actions,
                'agent_ids': stacked_agent_ids
            }
            
            # Serialize and save
            serialized_data = msgpack_serialize(dataset)
            if group:
                extension = "_group"
            else:
                extension = ""
            with open(f"data/num_blocks{num_blocks_i}/num_walls{num_walls_i}/gt_fsm{extension}_traj_data_{num_agents}agents.msgpack", "wb") as f:
                f.write(serialized_data)
            
            print(f"Saved dataset for blocks={num_blocks_i}, walls={num_walls_i}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        hand_designed_id = int(sys.argv[1])
    else:
        hand_designed_id = 0
    
    num_agents = 1  # number of agents
    num_datapoints_per_agent = 100  # number of trajectories to generate per agent
    num_steps = 50  # trajectory length
    num_block_steps = 2  # number of blocks to increment by
    num_walls_steps = 2  # number of walls to increment by

    # generate_dataset(num_datapoints_per_agent=num_datapoints_per_agent, num_steps=num_steps, num_agents=num_agents, num_block_steps=num_block_steps, num_walls_steps=num_walls_steps, group=True)



    agent_list = initialize_hand_designed_agent_list(num_agents=num_agents, num_blocks=1)


    state_list, action_list, env = generate_trajectory(hand_designed_id, agent_list, seed=42, num_agents=1, num_steps=100, num_blocks=8, num_walls=1)
    # Create frames list for RGB arrays
    stacked_states = jax.tree.map(lambda *xs: jnp.stack(xs), *state_list)

    def convert_to_image(index, stacked_state):
        indexed_state = jax.tree.map(lambda x: x[index], stacked_state)
        tile_size = 14
        grid_size = 7
        img_size = grid_size * tile_size

        img_gen_fn = jax.jit(state_to_image_jit, static_argnums=(1, 2, 3))
        render_fn = lambda x: img_gen_fn(x, img_size, grid_size, tile_size)
        return render_fn(indexed_state)
    
    frames = jax.vmap(convert_to_image, in_axes=(0, None))(jnp.arange(len(action_list)), stacked_states)
    # frames = [state_to_image(s, env.size) for s in state_list]
    # Save frames as GIF using imageio with loop=0 for infinite looping
    imageio.mimsave('environment_simulation.gif', frames, fps=5, loop=0)
    print("Saved GIF")