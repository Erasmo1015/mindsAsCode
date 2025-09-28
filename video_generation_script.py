import os
import numpy as np
import jax
import jax.numpy as jnp
import cv2
import pickle
from tqdm import tqdm

# Import the tools for our curated environments
from environment_jax import AutomaticityEnv, state_to_image_jit, str_to_grid
from custom_environments import custom_grid_layouts, task_list
from agent import AgentExecutionFramework

# --- Constants ---
GRID_SIZE = 7
TILE_SIZE = 12
OUTPUT_DIR = "human_vids_curated"
FSM_DIR = "generated_outputs/hand_designed"

def initialize_hand_designed_agent_list(num_agents=1, num_blocks=5):
    """Initializes agents from hand-designed FSM files."""
    framework = AgentExecutionFramework()
    agent_list = []
    # Match the FSM file names to the task_list order
    files = [f"{task}.txt" for task in task_list]
    for filename in files:
        filepath = os.path.join(FSM_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: FSM file not found for task '{filename}'. Skipping.")
            continue
        with open(filepath, "r") as f:
            agent_code = f.read()
        agent_list.append(framework.compile_agent(agent_code, num_agents=num_agents, num_blocks=num_blocks))
    return agent_list, files

def generate_trajectory_from_string(agent_id, agent_list, grid_string, num_steps=20):
    """Generates a trajectory starting from a string-defined grid."""
    initial_state = str_to_grid(grid_string)
    env = AutomaticityEnv(
        num_agents=1, size=GRID_SIZE, max_steps=num_steps + 1,
        num_blocks=len(initial_state.block_locations), num_walls=len(initial_state.wall_locations)
    )
    state, obs = initial_state, env.get_observation(initial_state)
    state_list, action_list, obs_list = [state], [], [obs[0]]
    for _ in range(num_steps):
        try:
            actions = [agent_list[agent_id].act(obs[0])]
            next_obs, next_state = env.step(state, jnp.array(actions))
            action_list.append(actions[0])
            obs_list.append(next_obs[0])
            state_list.append(next_state)
            obs, state = next_obs, next_state
        except Exception as e:
            print(f"ERROR during trajectory generation: {e}")
            break
    return state_list, action_list, obs_list

def convert_to_image(obs, grid_size=GRID_SIZE, tile_size=TILE_SIZE):
    """Converts an observation to a rendered image."""
    img_gen_fn = jax.jit(state_to_image_jit, static_argnums=(1, 2))
    return np.array(img_gen_fn(obs, grid_size, tile_size))

def make_video(state_seq, action_seq, filename):
    """Creates a video with the caption on top and includes the final state."""
    action_to_name = ["NOOP", "Right", "Left", "Down", "Up", "Pick up/Drop"]
    h, w = state_seq.shape[1], state_seq.shape[2]
    
    # Configuration for the output video frames
    scale_factor = 3
    text_height = 50 # Space for the caption at the top
    
    scaled_h, scaled_w = h * scale_factor, w * scale_factor
    frame_height = scaled_h + text_height

    # Use a modern, compatible codec like H.264
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(f"{filename}.mp4", fourcc, 2.5, (scaled_w, frame_height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    frames = []

    # Generate frames for each action
    for i, action in enumerate(action_seq):
        img_state = state_seq[i]
        img = cv2.cvtColor(img_state, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        
        frame = np.full((frame_height, scaled_w, 3), 255, dtype=np.uint8)
        frame[text_height:, :, :] = img_resized
        
        text = f"Step {i}: {action_to_name[int(action)]}"
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        text_x = (scaled_w - text_size[0]) // 2
        text_y = (text_height + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, 0.6, (0, 0, 0), 2)
        
        frames.append(frame)

    # ### REVISION ### - Add the final state as the last frame of the video
    final_img_state = state_seq[-1]
    final_img = cv2.cvtColor(final_img_state, cv2.COLOR_RGB2BGR)
    final_img_resized = cv2.resize(final_img, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    
    final_frame = np.full((frame_height, scaled_w, 3), 255, dtype=np.uint8)
    final_frame[text_height:, :, :] = final_img_resized
    
    text = f"Step {len(action_seq)}: Final State"
    text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
    text_x = (scaled_w - text_size[0]) // 2
    text_y = (text_height + text_size[1]) // 2
    cv2.putText(final_frame, text, (text_x, text_y), font, 0.6, (0, 0, 0), 2)
    frames.append(final_frame)

    # Write all frames (including the final state) to video
    for frame in frames:
        out.write(frame)
    out.release()
    
    # Also save the final frame as a separate PNG image
    cv2.imwrite(f"{filename}.png", final_frame)


def generate_all_videos(num_steps_in_video=20, num_prediction_steps=5):
    """
    Generates videos and saves full trajectories including the 5 ground truth
    actions that occur after the video ends.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Initializing agents based on task list...")
    # agent_list, fsm_files = initialize_hand_designed_agent_list()

    # Total steps to simulate = steps for video + future steps for ground truth
    total_simulation_steps = num_steps_in_video + num_prediction_steps

    all_trajectories_data = [[] for _ in range(4)]

    print(f"Generating {len(task_list) * 4} videos (simulating {total_simulation_steps} steps each)...")
    for task_idx, task_name in enumerate(tqdm(task_list, desc="Processing Tasks")):
        grid_variants = custom_grid_layouts[task_name]
        for variant_idx, grid_string in enumerate(grid_variants):
            agent_list, fsm_files = initialize_hand_designed_agent_list()
            # 1. Generate the FULL, long trajectory
            state_list, action_list, obs_list = generate_trajectory_from_string(
                agent_id=task_idx, 
                agent_list=agent_list, 
                grid_string=grid_string, 
                num_steps=total_simulation_steps
            )
            
            if len(action_list) < total_simulation_steps:
                print(f"\n! Note: Trajectory for '{task_name}' (video_{variant_idx}_{task_idx}) finished early at step {len(action_list)}.")

            # 2. Create the video using only the first part of the trajectory
            video_obs_list = obs_list[:num_steps_in_video + 1]
            video_action_list = action_list[:num_steps_in_video]
            images = np.array([convert_to_image(obs) for obs in video_obs_list])
            filename = os.path.join(OUTPUT_DIR, f"video_{variant_idx}_{task_idx}")
            make_video(images, np.array(video_action_list), filename)

            # 3. Save the ENTIRE trajectory (including ground truth steps) to the data file
            stacked_states = jax.tree.map(lambda *x: jnp.stack(x), *state_list)
            padded_actions = jnp.pad(jnp.array(action_list), (0, total_simulation_steps + 1 - len(action_list)), 'constant', constant_values=0)[:total_simulation_steps+1]

            trajectory_data = {
                "states": stacked_states,
                "actions": padded_actions
            }
            all_trajectories_data[variant_idx].append(trajectory_data)


    output_data_file = os.path.join(OUTPUT_DIR, "fsm_curated_gameplay_data.pkl")
    with open(output_data_file, "wb") as f:
        pickle.dump(all_trajectories_data, f)
    
    print("-" * 60)
    print(f"Curated video and data generation complete!")
    print(f"All videos saved to: ./{OUTPUT_DIR}/")
    print(f"State data (including ground truth) saved to: {output_data_file}")


if __name__ == "__main__":
    # Pass the number of steps for the video and for prediction separately
    generate_all_videos(num_steps_in_video=20, num_prediction_steps=5)