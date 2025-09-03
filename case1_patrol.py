from environment_jax import AutomaticityEnv, str_to_grid
from environment import state_to_image_jit
from agent import AgentExecutionFramework
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import math
import pickle
import copy
from tqdm import tqdm
import argparse
import os
import flax
# from jax_tqdm import scan_tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline models for automaticity.")
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="AutoToM",
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
    parser.add_argument('--num_datapoints_per_agent', type=int, default=5, help='Number of datapoints per agent in the dataset.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of steps in the dataset.')
    parser.add_argument('--env_size', type=int, default=7, help='Size of the environment.')
    # parser.add_argument('--num_blocks', type=int, default=10, help='Number of blocks in the dataset.')
    # parser.add_argument('--num_walls', type=int, default=10, help='Number of walls in the dataset.')

    parser.add_argument('--as_images', type=bool, default=False, help='Whether to load the data as images.')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save the model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--n_hypothesis', type=int, default=30, help='Number of hypothesis for thought trace.')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Name of the model to use.')  # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct or meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of tensor parallel size.')
    parser.add_argument('--dtype', type=str, default="float16", help='Data type.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization.')
    parser.add_argument('--overfit', type=bool, default=False, help='Whether to overfit on a single environment.')
    parser.add_argument('--bootstrap', action='store_true', help='Whether to use bootstrapping for hypothesis evaluation')
    parser.add_argument('--two_stage', action='store_true', help='Whether to use two-stage approach for FSM reasoning')
    parser.add_argument('--structured', type=str, default="False", choices=["False", "p1", "p2"], 
                        help='Structured prompting type for FSM reasoning: False, p1, or p2')
    parser.add_argument('--rejuvenation', action='store_true', help='Use rejuvenation for FSM model')
    parser.add_argument('--plot_gifs', action='store_true', help='Plot gifs for FSM model')
    parser.add_argument('--rejuvenation_threshold', type=float, default=1, help='Threshold for rejuvenation')
    parser.add_argument('--max_rejuvenation_attempts', type=int, default=2, help='Maximum number of rejuvenation attempts')
    parser.add_argument('--top_k', type=int, default=0, help='If > 0, only average over the top k most likely hypotheses')
    parser.add_argument('--multi_step_eval', type=bool, default=True, help='Perform multi-step evaluation for FSM')
    parser.add_argument('--num_steps_to_predict', type=int, default=20, help='Number of future steps to predict in multi-step eval')
    parser.add_argument('--flip_quarter', type=bool, default=True, help='reset the environment after 30 steps')
    parser.add_argument('--human_data', type=bool, default=False, help='Use human data')
    args = parser.parse_args()
    
    # Check if the selected baseline model is implemented
    if args.baseline_model not in ["ToMnet", 'BC', 'AutoToM', 'FSM', 'NLLM', 'Oracle']:
        raise NotImplementedError(f"Baseline model '{args.baseline_model}' is not implemented.")
    
    if args.baseline_model == 'AutoToM':
        os.environ['CURRENT_MODEL_NAME'] = args.model_name
    
    return args



grid_str = """
#######
## 1 ##
##   ##
## # ##
## A ##
## A ##
#######
"""

class MCTSAgent:
    def __init__(self, env, num_agents, num_blocks, num_walls, num_simulations=300, simulation_depth=100, prediction_model_type='gt'):
        self.env = env
        self.num_agents = num_agents
        self.num_blocks = num_blocks
        self.num_walls = num_walls
        self.num_actions = 6
        self.num_simulations = num_simulations
        self.simulation_depth = simulation_depth
        self.prediction_model_type = prediction_model_type
        self.jitted_step = jax.jit(self.env.step)
        self.jitted_get_observation = jax.jit(self.env.get_observation)
        self.action_deltas = jnp.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])

    def value_function(self, obs):
        '''
        makes a probability distribution over actions based on the heuristic of minimizing the distance to the block
        '''
        block_location = obs['block_locations'][0]
        sneak_location = obs['agent_locations'][1]

        def compute_distance(sneak_location, block_location, sneak_delta):
            new_sneak_location = sneak_location + sneak_delta
            distance = jnp.linalg.norm(new_sneak_location - block_location)
            return distance # for numerical stability
        
        distances = jax.vmap(compute_distance, in_axes=(None, None, 0))(sneak_location, block_location, self.action_deltas)
        action_probs = jnp.exp(-distances)
        action_probs = action_probs / jnp.sum(action_probs)
        return action_probs.astype(jnp.float32)

    '''
    TODO: make a separate get_action function for bc that uses jax's vmap to speed up the simulation
    '''
    def get_action_bc(self, rng_key, obs, state, predictor_model, first_patrol_obs):
        '''
        This function takes in the current observation, simulates many possible trajectories, and returns the action that maximizes the expected reward.
        '''
        action_reward_dict = {a: [] for a in range(self.num_actions)} # initialize the action reward dictionary
        def single_simulation(sim_id, rng_key, obs, state, predictor_model, first_patrol_obs):
            bc_model, stacked_bc_state = predictor_model
            param_to_use = jax.random.randint(rng_key, (1,), 0, 6)[0]
            param_state = jax.tree.map(lambda x: x[param_to_use], stacked_bc_state)
            rng_key, subkey = jax.random.split(rng_key)
            curr_sim_reward = 0
            action_probs = self.value_function(obs)
            sample_action_fn = lambda p, rng_key: jax.random.choice(rng_key, self.num_actions, p=p)
            uniform_function = lambda p, rng_key: jax.random.choice(rng_key, self.num_actions)
            first_step_action = jax.lax.cond(jnp.sum(action_probs) == 1, sample_action_fn, uniform_function, action_probs, rng_key)
            rng_key, subkey = jax.random.split(rng_key)
            
            # @scan_tqdm(self.simulation_depth)
            def inner_loop(carry, time_step):
                curr_obs, curr_patrol_obs, curr_state, reward, first_step_action, rng_key, param_state, was_terminal = carry

                # get sneak action
                def sample_action(o, rng_key):
                    action_probs = self.value_function(obs)
                    sample_action_fn = lambda p, rng_key: jax.random.choice(rng_key, self.num_actions, p=p)
                    uniform_function = lambda p, rng_key: jax.random.choice(rng_key, self.num_actions)
                    action = jax.lax.cond(jnp.sum(action_probs) == 1, sample_action_fn, uniform_function, action_probs, rng_key)
                    return action
                first_action_fn = lambda o, rng_key: first_step_action
                sneak_action = jax.lax.cond(time_step == 0, first_action_fn, sample_action, obs, rng_key)
                rng_key, subkey = jax.random.split(rng_key)


                # Get the action from the BC model
                patrol_action = bc_model.apply(param_state, curr_patrol_obs, None, training=False)
                patrol_action = patrol_action[0, -1]
                sample_patrol_action_fn = lambda p, rng_key: jax.random.choice(rng_key, self.num_actions, p=p)
                argmax_patrol_action_fn = lambda p, rng_key: jnp.argmax(p)
                patrol_action = jax.lax.cond(jnp.sum(patrol_action) == 1, sample_patrol_action_fn, argmax_patrol_action_fn, patrol_action, rng_key)
                rng_key, subkey = jax.random.split(rng_key)

                # execute the actions
                actions_to_execute = jnp.array([patrol_action, sneak_action])

                all_obs, curr_state, new_reward = self.jitted_step(curr_state, actions_to_execute)
                curr_obs = all_obs[1] # sneak agent's observation

                # if the state was terminal, set reward to 0
                new_reward = jnp.where(was_terminal, 0, new_reward)
                reward = reward + new_reward
                # if the state is terminal, set was_terminal to true, otherwise leave it as is
                was_terminal = jnp.where(curr_state.terminal, True, was_terminal)

                
                grid_size = 7
                tile_size = 10
                patrol_image = state_to_image_jit(all_obs[0], grid_size*tile_size, grid_size, tile_size=tile_size)
                patrol_image = patrol_image[None, None]
                patrol_obs = jnp.concatenate([curr_patrol_obs, patrol_image], axis=1)[:, 1:]

                new_carry = (curr_obs, patrol_obs, curr_state, reward, first_step_action, rng_key, param_state, was_terminal)
                return new_carry, new_reward
            
            curr_obs = obs
            curr_patrol_obs = first_patrol_obs
            curr_state = state
            reward = 0
            was_terminal = False
            carry = (curr_obs, curr_patrol_obs, curr_state, reward, first_step_action, rng_key, param_state, was_terminal)
            carry, _ = jax.lax.scan(inner_loop, carry, jnp.arange(self.simulation_depth), self.simulation_depth)
            reward = carry[3]
            action_values = jnp.zeros(self.num_actions)
            action_values = action_values.at[first_step_action].set(reward)
            return action_values
        
        action_values_stacked = jax.vmap(single_simulation, in_axes=(0, 0, None, None, None, None))(jnp.arange(self.num_simulations), jax.random.split(rng_key, self.num_simulations), obs, state, predictor_model, first_patrol_obs)
        action_values_dict = {a: action_values_stacked[:, a].mean() for a in range(self.num_actions)}
        best_action = max(action_values_dict, key=action_values_dict.get)
        return best_action

    def get_action_autoToM(self, rng, obs, state, predictor_model, first_patrol_obs, patrol_actions):
        '''
        This function takes in the current observation, simulates many possible trajectories, and returns the action that maximizes the expected reward.
        '''
        action_reward_dict = {a: [] for a in range(self.num_actions)} # initialize the action reward dictionary
        for _ in tqdm(range(self.num_simulations)):
            curr_sim_reward = 0
            sneak_action_probs = self.value_function(obs)
            try:
                first_step_action = np.random.choice(self.num_actions, p=sneak_action_probs)
            except Exception as e:
                # print(f"Error: {e}")
                sneak_action_probs = sneak_action_probs / np.sum(sneak_action_probs)
                first_step_action = np.random.choice(self.num_actions)

            for i in tqdm(range(self.simulation_depth)):
                if i == 0:
                    curr_obs = obs
                    curr_state = state
                    sneak_action = first_step_action
                    curr_stacked_states = first_patrol_obs
                else:
                    action_probs = self.value_function(curr_obs)
                    try:
                        sneak_action = np.random.choice(self.num_actions, p=action_probs)
                    except Exception as e:
                        # print(f"Error: {e}")
                        action_probs = action_probs / np.sum(action_probs)
                        sneak_action = np.random.choice(self.num_actions)
                
                # get the predicted action
                pred_action, pred_probs = predictor_model.predict_action(curr_stacked_states, patrol_actions, agent_id=0, timestep=None)
                try:
                    patrol_action = np.random.choice(self.num_actions, p=pred_probs)
                except ValueError:
                    patrol_action = pred_action
                
                actions_to_execute = jnp.array([int(patrol_action), int(sneak_action)])
                all_obs, curr_state, reward = self.jitted_step(curr_state, actions_to_execute)
                curr_obs = all_obs[1] # sneak agent's observation
                # first expand patrol obs
                expanded_patrol_obs = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), all_obs[0])
                # then concatenate the expanded patrol obs with curr_stacked_states
                curr_stacked_states = jax.tree.map(lambda *x: jnp.concatenate([x[0], x[1]], axis=0), curr_stacked_states, expanded_patrol_obs)
                # breakpoint()
                # remove the first element of curr_stacked_states
                # curr_stacked_states = jax.tree.map(lambda x: x[1:], curr_stacked_states)
                curr_sim_reward += reward
                if curr_state.terminal:
                    break
            action_reward_dict[first_step_action].append(curr_sim_reward)
        action_reward_dict = {k: np.mean(v) for k, v in action_reward_dict.items()}
        best_action = max(action_reward_dict, key=action_reward_dict.get)
        return best_action



        
    def get_action(self, obs, state, predictor_model, first_patrol_obs):
        '''
        This function takes in the current observation, simulates many possible trajectories, and returns the action that maximizes the expected reward.
        '''
        
        action_reward_dict = {a: [] for a in range(self.num_actions)} # initialize the action reward dictionary
        framework = AgentExecutionFramework()
        for _ in tqdm(range(self.num_simulations)):
            curr_sim_reward = 0
            action_probs = self.value_function(obs)
            try:
                first_step_action = np.random.choice(self.num_actions, p=action_probs)
            except ValueError:
                action_probs = action_probs / np.sum(action_probs)
                first_step_action = np.random.choice(self.num_actions)
            if self.prediction_model_type == 'gt':
                copy_of_model = copy.deepcopy(predictor_model) # make a copy of the base gt patrolling model
            for i in range(self.simulation_depth):
                if i == 0:
                    curr_obs = obs
                    curr_state = state
                    sneak_action = first_step_action
                    patrol_obs = first_patrol_obs
                else:
                    action_probs = self.value_function(curr_obs)
                    try:    
                        sneak_action = np.random.choice(self.num_actions, p=action_probs)
                    except ValueError:
                        action_probs = action_probs / np.sum(action_probs)
                        sneak_action = np.random.choice(self.num_actions)
                if self.prediction_model_type == 'gt':
                    patrol_action = framework.execute_agent(copy_of_model, patrol_obs)
                elif self.prediction_model_type == 'random':
                    patrol_action = np.random.randint(0, self.num_actions)
                elif self.prediction_model_type == 'bc':
                    bc_model, bc_state = predictor_model
                    params_to_use = np.random.randint(0, 6)
                    bc_param_state = jax.tree.map(lambda x: x[params_to_use], bc_state)
                    action_pred = bc_model.apply(bc_param_state, patrol_obs, None, training=False)
                    action_pred = action_pred[0, -1]
                    try:
                        patrol_action = np.random.choice(self.num_actions, p=action_pred)
                    except:
                        patrol_action = np.argmax(action_pred)
                else:
                    raise ValueError(f"Invalid prediction model type: {self.prediction_model_type}")
                actions_to_execute = jnp.array([int(patrol_action), int(sneak_action)])
                all_obs, curr_state, reward = self.jitted_step(curr_state, actions_to_execute)
                curr_obs = all_obs[1] # sneak agent's observation
                if self.prediction_model_type == 'bc':
                    grid_size = 7
                    tile_size = 10
                    patrol_image = state_to_image_jit(all_obs[0], grid_size*tile_size, grid_size, tile_size=tile_size)
                    patrol_image = patrol_image[None, None]
                    patrol_obs = jnp.concatenate([patrol_obs, patrol_image], axis=1)[:, 1:]
                else:
                    patrol_obs = all_obs[0] # patrol agent's observation
                curr_sim_reward += reward
                if curr_state.terminal:
                    break
            action_reward_dict[first_step_action].append(curr_sim_reward)
        action_reward_dict = {k: np.mean(v) for k, v in action_reward_dict.items()}
        best_action = max(action_reward_dict, key=action_reward_dict.get)
        return best_action


class PatrolEnv(AutomaticityEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, rng):
        obs, state = super().reset(rng)
        state = str_to_grid(grid_str)
        return obs, state
    
    def step(self, state, actions):
        obs, state = super().step(state, actions)
        # check to see if the two agents are close to each other
        patrol_agent_pos = state.agent_locations[0]
        sneak_agent_pos = state.agent_locations[1]
        collided = jnp.logical_and(jnp.linalg.norm(patrol_agent_pos - sneak_agent_pos) <= 1, sneak_agent_pos[1] < 5)  # did they collide and cross the engagement threshold?
        reached_block = jnp.all(sneak_agent_pos == state.block_locations[0])
        terminal = jnp.logical_or(collided, reached_block)
        state = state.replace(terminal=terminal)

        # Returns -100 if collided, +100 if reached block, -1 otherwise
        reward = jnp.where(collided, -100, -2)
        reward = jnp.where(reached_block, 100, reward)
        return obs, state, reward


def generate_trajectory(env, init_state, num_steps, agent_list, num_agents, prediction_model):
    img_frames = []
    patrol_obs = []
    actions_taken = []
    action_id_taken = []
    action_names = ["stay", "right", "left", "down", "up", "interact"]
    obs = env.get_observation(init_state)

    obs0 = jax.tree.map(lambda x: jnp.array(x), obs[0])
    obs1 = jax.tree.map(lambda x: jnp.array(x), obs[1])
    img_frames.append(state_to_image_jit(obs0,grid_size*tile_size, grid_size, tile_size=tile_size))
    patrol_obs.append(obs[0])
    
    framework = AgentExecutionFramework()
    state = init_state
    total_reward = 0
    trajectory = ['stay'] * 19
    for i in range(num_steps):
        actions = []
        action_taken_list = []
        for agent_id in range(num_agents):
            if agent_id < len(agent_list):
                agent = agent_list[agent_id]
                if type(agent) == MCTSAgent:
                    if i < 19:
                        chosen_action = action_names.index(trajectory[i])
                    else:
                        if agent.prediction_model_type == 'gt' or agent.prediction_model_type == 'random':
                            first_patrol_obs = obs[0]
                            chosen_action = agent.get_action(obs[1], state, prediction_model, first_patrol_obs)
                        elif agent.prediction_model_type == 'BC':
                            first_patrol_obs = jnp.stack(img_frames)[None]  # add batch dimension 
                            chosen_action = agent.get_action_bc(rng, obs[1], state, prediction_model, first_patrol_obs)
                        elif agent.prediction_model_type == 'AutoToM':
                            # stack the patrol obs
                            first_patrol_obs = jax.tree.map(lambda *x: jnp.stack(x), *patrol_obs)
                            patrol_actions = np.array(action_id_taken)
                            chosen_action = agent.get_action_autoToM(rng, obs[1], state, prediction_model, first_patrol_obs, patrol_actions)
                else:
                    chosen_action = framework.execute_agent(agent, obs[agent_id])
            else:
                chosen_action = trajectory[i]
                chosen_action = action_names.index(chosen_action)
            actions.append(chosen_action)
            action_taken_list.append(action_names[chosen_action])
        print(f"Patrol action: {action_taken_list[0]}, Sneak action: {action_taken_list[1]}")
        actions_taken.append(action_taken_list)
        action_id_taken.append(actions)
        obs, state, reward = env.step(state, jnp.array(actions))
        total_reward += reward
        obs0 = jax.tree.map(lambda x: jnp.array(x), obs[0])
        obs1 = jax.tree.map(lambda x: jnp.array(x), obs[1])
        img_frames.append(state_to_image_jit(obs0, grid_size*tile_size, grid_size, tile_size=tile_size))
        patrol_obs.append(obs[0])
        if reward != -2:
            break
    actions_taken.append(['Terminal', 'Terminal'])
    return img_frames, actions_taken, total_reward

def trajectory_to_gif(img_frames, actions_taken, gif_path):
    try:
        import imageio

        # Create a figure with two subplots - image on top, text below
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10),
                                       gridspec_kw={'height_ratios': [4, 1]})

        gif_frames = []
        # We'll use the axis limits from the second frame onwards for all frames
        zoom_xlim = None
        zoom_ylim = None

        for i, (frame, action) in enumerate(zip(img_frames, actions_taken)):
            # Clear the axes
            ax1.clear()
            ax2.clear()

            # Display the frame
            ax1.imshow(frame)
            ax1.set_title(f"Step {i+1}", fontsize=28)
            ax1.axis('off')  # Remove axis labels for the image

            # For the first frame, record the axis limits after imshow (zoomed out)
            if i == 0:
                zoom_xlim = ax1.get_xlim()
                zoom_ylim = ax1.get_ylim()
            else:
                # For subsequent frames, set the axis limits to match the first frame's zoomed-in view
                ax1.set_xlim(zoom_xlim)
                ax1.set_ylim(zoom_ylim)

            action_string = f"Agent 0: {action[0]}\nAgent 1: {action[1]}"
            ax2.text(
                0.5, 0.5, action_string,
                ha='center', va='center',
                fontsize=24,
                fontfamily='monospace'
            )
            ax2.axis('off')

            # Draw the figure and convert to image
            fig.canvas.draw()
            plt.tight_layout()

            # Convert figure to image properly
            img = np.array(fig.canvas.renderer.buffer_rgba())

            # Add this frame to our GIF
            gif_frames.append(img)

        # Save as GIF - ensure all frames are included
        imageio.mimsave(gif_path, gif_frames, fps=2)
        print(f"GIF saved as {gif_path} with {len(gif_frames)} frames")

    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("Try installing imageio with: pip install imageio")


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

if __name__ == "__main__":
    args = parse_args()
    num_agents = 2
    tile_size = 10
    grid_size = 7

    # bc_model, bc_state = load_bc_models(args)
    if args.baseline_model == 'AutoToM':
        from baselines.AutoToM.autoToM import AutoToM
        autoToM_model = AutoToM(model_name=args.model_name, tensor_parallel_size=1, dtype=torch.bfloat16, gpu_memory_utilization=0.95, group=False)
    elif args.baseline_model == 'BC':
        from baselines.BC.bc import BCNet
        bc_model, bc_state = load_bc_models(args)

    env = PatrolEnv(
        num_agents=num_agents,
        size=grid_size,
        max_steps=100,
        num_blocks=1,
        num_walls=1,
    )
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    patrolling_agent_path = 'generated_outputs/hand_designed/case1_patrol.txt'
    patrolling_agent_txt = open(patrolling_agent_path, 'r').read()
    patrolling_agent = AgentExecutionFramework().compile_agent(patrolling_agent_txt, num_agents=num_agents, num_blocks=1)

    mcts_agent = MCTSAgent(env, num_agents, num_blocks=1, num_walls=1, num_simulations=300, simulation_depth=10, prediction_model_type=args.baseline_model)

    # Generate trajectory with both agents
    if args.baseline_model == 'AutoToM':
        prediction_model = autoToM_model
    elif args.baseline_model == 'BC':
        prediction_model = [bc_model, bc_state]
    elif args.baseline_model == 'random' or args.baseline_model == 'gt':
        prediction_model = patrolling_agent

    img_frames, actions_taken, total_reward = generate_trajectory(env, state, 100, [patrolling_agent, mcts_agent], num_agents, prediction_model)
    trajectory_to_gif(img_frames, actions_taken, 'patrol_mcts.gif')
    print(f"Total reward: {total_reward}")