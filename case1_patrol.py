from environment_jax import AutomaticityEnv, str_to_grid, state_to_image_jit
from agent import AgentExecutionFramework
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

grid_str = """
#######
## 1 ##
##   ##
## # ##
## A ##
## A ##
#######
"""


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
        reward = jnp.where(collided, -100, -1)
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
    for i in range(num_steps):
        actions = []
        action_taken_list = []
        for agent_id in range(num_agents):
            if agent_id < len(agent_list):
                chosen_action = framework.execute_agent(agent_list[agent_id], obs[agent_id])
            else:
                chosen_action = 0
            actions.append(chosen_action)
            action_taken_list.append(action_names[chosen_action])
        actions_taken.append(action_taken_list)
        obs, state, reward = env.step(state, jnp.array(actions))
        total_reward += reward
        obs0 = jax.tree.map(lambda x: jnp.array(x), obs[0])
        obs1 = jax.tree.map(lambda x: jnp.array(x), obs[1])
        img_frames.append(state_to_image_jit(obs0, grid_size, tile_size=tile_size))
    return img_frames[:-1], actions_taken, total_reward

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

    img_frames, actions_taken, total_reward = generate_trajectory(env, state, 100, [patrolling_agent], num_agents)
    trajectory_to_gif(img_frames, actions_taken, 'patrol.gif')
    print(f"Total reward: {total_reward}")