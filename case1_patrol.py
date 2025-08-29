from environment_jax import AutomaticityEnv, str_to_grid, state_to_image_jit
from agent import AgentExecutionFramework
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import random
import math
import pickle
import copy
from tqdm import tqdm

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
        return action_probs.astype(np.float32)
        
    def get_action(self, obs, state, predictor_model):
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
                    patrol_obs = self.jitted_get_observation(curr_state)[0]
                else:
                    action_probs = self.value_function(curr_obs)
                    try:    
                        sneak_action = np.random.choice(self.num_actions, p=action_probs)
                    except ValueError:
                        # breakpoint()
                        action_probs = action_probs / np.sum(action_probs)
                        sneak_action = np.random.choice(self.num_actions)
                if self.prediction_model_type == 'gt':
                    patrol_action = framework.execute_agent(copy_of_model, patrol_obs)
                elif self.prediction_model_type == 'random':
                    patrol_action = np.random.randint(0, self.num_actions)
                else:
                    raise ValueError(f"Invalid prediction model type: {self.prediction_model_type}")
                actions_to_execute = jnp.array([int(patrol_action), int(sneak_action)])
                all_obs, curr_state, reward = self.jitted_step(curr_state, actions_to_execute)
                curr_obs = all_obs[1] # sneak agent's observation
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


def generate_trajectory(env, init_state, num_steps, agent_list, num_agents):
    img_frames = []
    actions_taken = []
    action_names = ["stay", "right", "left", "down", "up", "interact"]
    obs = env.get_observation(init_state)

    obs0 = jax.tree.map(lambda x: jnp.array(x), obs[0])
    obs1 = jax.tree.map(lambda x: jnp.array(x), obs[1])
    img_frames.append(state_to_image_jit(obs0, grid_size, tile_size=tile_size))
    
    framework = AgentExecutionFramework()
    state = init_state
    total_reward = 0
    trajectory = ['stay', 'up', 'left', 'up', 'up', 'up', 'right']
    for i in range(num_steps):
        actions = []
        action_taken_list = []
        for agent_id in range(num_agents):
            if agent_id < len(agent_list):
                if type(agent_list[agent_id]) == MCTSAgent:
                    chosen_action = agent_list[agent_id].get_action(obs[agent_id], state, agent_list[0])
                else:
                    chosen_action = framework.execute_agent(agent_list[agent_id], obs[agent_id])
            else:
                chosen_action = trajectory[i]
                chosen_action = action_names.index(chosen_action)
            actions.append(chosen_action)
            action_taken_list.append(action_names[chosen_action])
        print(f"Patrol action: {action_taken_list[0]}, Sneak action: {action_taken_list[1]}")
        actions_taken.append(action_taken_list)
        obs, state, reward = env.step(state, jnp.array(actions))
        total_reward += reward
        obs0 = jax.tree.map(lambda x: jnp.array(x), obs[0])
        obs1 = jax.tree.map(lambda x: jnp.array(x), obs[1])
        img_frames.append(state_to_image_jit(obs0, grid_size, tile_size=tile_size))
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



if __name__ == "__main__":
    num_agents = 2
    tile_size = 10
    grid_size = 7

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

    mcts_agent = MCTSAgent(env, num_agents, num_blocks=1, num_walls=1, num_simulations=500, simulation_depth=50, prediction_model_type='gt')

    # Generate trajectory with both agents
    img_frames, actions_taken, total_reward = generate_trajectory(env, state, 100, [patrolling_agent, mcts_agent], num_agents)
    trajectory_to_gif(img_frames, actions_taken, 'patrol_mcts.gif')
    print(f"Total reward: {total_reward}")