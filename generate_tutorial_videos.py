import os
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Import the environment and rendering tools from your existing files
from environment_jax import AutomaticityEnv, str_to_grid

# ### REVISION: Import the vetted functions from your video generation script ###
from video_generation_script import make_video, convert_to_image

# --- Configuration ---
OUTPUT_DIR = "human_vids_curated/tutorial"
GRID_SIZE = 7

# Action mappings from your environment (0:stay, 1:right, 2:left, 3:down, 4:up, 5:interact)
STAY = 0
RIGHT = 1
LEFT = 2
DOWN = 3
UP = 4
PICKUP_DROP = 5


def generate_trajectory_from_actions(env, initial_state, actions):
    """
    Generates a trajectory of states and observations by applying a predefined sequence of actions.
    """
    state_list = [initial_state]
    obs_list = [env.get_observation(initial_state)[0]]
    current_state = initial_state

    print("Simulating trajectory...")
    for action in tqdm(actions):
        # The environment expects a JAX array of actions for all agents
        action_array = jnp.array([action])
        obs, next_state = env.step(current_state, action_array)
        
        state_list.append(next_state)
        obs_list.append(obs[0]) # Get the observation for the first agent
        current_state = next_state
        
    return state_list, obs_list


if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use a single environment instance for stepping and rendering
    env = AutomaticityEnv(num_agents=1, size=GRID_SIZE, max_steps=50, num_blocks=5, num_walls=10)

    # --- Define all scenarios ---
    scenarios = {
        "movement": {
            "layout": '''
#######
#     #
#     #
#  A  #
#     #
#  1  #
#######
            ''',
            "actions": [UP, RIGHT, RIGHT, DOWN, DOWN, STAY, LEFT, LEFT, UP, STAY]
        },
        "walls": {
            "layout": '''
#######
#   1 #
#  #  #
#     #
#  A  #
#     #
#######
            ''',
            "actions": [UP, UP, UP]
        },
        "pickup_drop": {
            "layout": '''
#######
#     #
#     #
# A 1 #
#     #
#     #
#######
            ''',
            "actions": [RIGHT, RIGHT, PICKUP_DROP, UP, UP, PICKUP_DROP, LEFT, LEFT]
        },
        "collision": {
            "layout": '''
#######
#     #
#     #
# A1 2#
#     #
#     #
#######
            ''',
            "actions": [RIGHT, PICKUP_DROP, RIGHT, RIGHT, RIGHT]
        }
    }

    # --- Generate a video for each scenario ---
    for name, config in scenarios.items():
        print(f"\n{'='*40}")
        print(f"Generating video for: {name}.mp4")
        print(f"{'='*40}")

        # 1. Set up the initial state from the layout string
        initial_state = str_to_grid(config["layout"])
        action_sequence = config["actions"]

        # 2. Simulate the trajectory to get all states and observations
        _, obs_sequence = generate_trajectory_from_actions(env, initial_state, action_sequence)

        # 3. Convert the sequence of observations into images
        print("Converting observations to images...")
        image_sequence = np.array([convert_to_image(obs) for obs in tqdm(obs_sequence)])
        
        # 4. Use the make_video function to create the final, captioned video
        output_filename = os.path.join(OUTPUT_DIR, name) # No extension, make_video adds it
        make_video(image_sequence, np.array(action_sequence), output_filename)

    print("\n🎉 All tutorial videos have been successfully generated!")