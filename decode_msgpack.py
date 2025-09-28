import asyncio
from pprint import pp

import aiofiles
import msgpack
import os
from nicewebrl.logging import get_logger
import flax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

logger = get_logger(__name__)


def recursive_unpack(value):
    if not isinstance(value, dict):
        return value
    newdict = {}
    for k, v in value.items():
        if isinstance(v, dict):
            newdict[k] = recursive_unpack(v)
        elif isinstance(v, msgpack.ExtType):
            newdict[k] = flax.serialization.msgpack_restore(v.data)
        else:
            newdict[k] = v
    return newdict

async def read_msgpack_records(filepath: str):
    """Read length-prefixed msgpack records from a file.

    Args:
        filepath: Path to the file containing the records

    Yields:
        Decoded msgpack records one at a time
    """
    async with aiofiles.open(filepath, "rb") as f:
        while True:
            # Read length prefix (4 bytes)
            length_bytes = await f.read(4)
            if not length_bytes:  # End of file
                break

            # Convert bytes to integer
            length = int.from_bytes(length_bytes, byteorder="big")

            # Read the record data
            data = await f.read(length)
            if len(data) < length:  # Incomplete record
                logger.error(
                    f"Corrupt data in {filepath}: Expected {length} bytes but got {len(data)}"
                )
                break

            # Unpack and yield the record
            try:
                record = msgpack.unpackb(data)
                recursive_unpack(record)
                if "data" in record:
                    if "timestep" in record["data"]:
                        record["data"]["timestep"] = flax.serialization.msgpack_restore(record["data"]["timestep"])
                yield record
            except Exception as e:
                logger.error(f"Failed to unpack record in {filepath}: {e}")
                break
        # yield msgpack.unpackb(length_bytes + data)


async def read_file(filepath: str):
    datapoints = []
    async for line in read_msgpack_records(
        filepath
    ):
        datapoints.append(line)
    return datapoints

def load_and_stack_prediction_data(directory: str, username_list: list):
    actions = ["NOOP", "Right", "Left", "Down", "Up", "Interact"]
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.json')]
    files = [f for f in files if 'predict' in f]


    data_info = []
    for i, file in enumerate(files):
        filepath = os.path.join(directory, file)
        datapoints = asyncio.run(read_file(filepath))
        for datapoint in datapoints:
            if 'metadata' not in datapoint:
                continue
            if datapoint['metadata']['type'] == 'FeedbackStage':
                video_path = datapoint['data']['video_path']  # human_vids/video_userId_taskId.mp4
                # Extract userId and taskId from video path (format: human_vids/video_userId_taskId.mp4)
                video_parts = video_path.split('/')[-1].split('.')[0].split('_')
                user_id = int(video_parts[1])
                task_id = int(video_parts[2])
                username = username_list[user_id]
                task_str = datapoint['metadata']['task']
                predicted_action = datapoint['data']['predicted_action']
                predicted_action_idx = actions.index(predicted_action)
                rote_rating = datapoint['data']['rote_rating']
                informative_rating = datapoint['data']['informative_rating']
                goal_directed_rating = datapoint['data']['goal_directed_rating']
                random_rating = datapoint['data']['random_rating']
                thinking_rating = datapoint['data']['thinking_rating']
                complex_rating = datapoint['data']['complex_rating']
                planned_rating = datapoint['data']['planned_rating']
                behavior_description = datapoint['data']['behavior_description']

                relevant_info = {
                    'username': username,
                    'task_str': task_str,
                    'predicted_action_idx': predicted_action_idx,
                    'rote_rating': rote_rating,
                    'informative_rating': informative_rating,
                    'goal_directed_rating': goal_directed_rating,
                    'random_rating': random_rating,
                    'thinking_rating': thinking_rating,
                    'complex_rating': complex_rating,
                    'planned_rating': planned_rating,
                    'behavior_description': behavior_description,
                    'user_id': user_id,
                    'task_id': task_id,
                }
                data_info.append(relevant_info)
    return data_info

def load_and_stack_gameplay_data(directory: str):
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.json')]
    all_file_states = []
    all_file_actions = []
    all_file_agent_indices = []
    all_file_names = []
    for i, file in enumerate(files):
        filepath = os.path.join(directory, file)
        datapoints = asyncio.run(read_file(filepath))
        all_state_trajectories = []
        all_action_trajectories = []
        curr_state_trajectory = []
        curr_action_trajectory = []
        all_tasks = []
        current_task = ""
        for datapoint in datapoints:
            try:
                task = datapoint['metadata']['task']
                if 'Tutorial' in task:
                    continue
                if task != current_task:
                    current_task = task
                    all_tasks.append(current_task)
                    # stack the trajectories
                    if len(curr_state_trajectory) > 0:
                        stacked_state_trajectory = jax.tree.map(lambda *x: jnp.stack(x), *curr_state_trajectory)
                        stacked_action_trajectory = jnp.stack(curr_action_trajectory)
                        all_state_trajectories.append(stacked_state_trajectory)
                        all_action_trajectories.append(stacked_action_trajectory)
                    curr_state_trajectory = []
                    curr_action_trajectory = []
                action = jnp.array(datapoint['data']['action_idx'])
                state = datapoint['data']['timestep']['state']  # this is a dict. convert all leaves to jax arrays
                state = jax.tree.map(lambda x: jnp.array(x), state)
                curr_state_trajectory.append(state)
                curr_action_trajectory.append(action)
            except Exception as e:
                # logger.error(f"Failed to process {filepath}: {e}")
                continue
        
        # # Add the final task's trajectories
        # if len(curr_state_trajectory) > 0:
        #     stacked_state_trajectory = jax.tree.map(lambda *x: jnp.stack(x), *curr_state_trajectory)
        #     stacked_action_trajectory = jnp.stack(curr_action_trajectory)
        #     all_state_trajectories.append(stacked_state_trajectory)
        #     all_action_trajectories.append(stacked_action_trajectory)

        # task_to_idx = {task: i for i, task in enumerate(all_tasks)}
        from human_play_exp import task_list
        task_to_idx = jnp.array([task_list.index(task) for task in all_tasks])

        # Sort by task indices
        sort_indices = jnp.argsort(task_to_idx)
        all_state_trajectories = jax.tree.map(lambda *x: jnp.stack(x), *all_state_trajectories)  # (num_tasks, num_timesteps, *)
        all_action_trajectories = jnp.stack(all_action_trajectories)  # (num_tasks, num_timesteps)
        # Reindex trajectories based on sorted task order
        all_state_trajectories = jax.tree.map(lambda x: x[sort_indices], all_state_trajectories)
        all_action_trajectories = all_action_trajectories[sort_indices]
        all_agent_indices = jnp.arange(all_action_trajectories.shape[0])
        if all_action_trajectories.shape[0] == len(task_list):  # number of tasks
            all_file_states.append(all_state_trajectories)
            all_file_actions.append(all_action_trajectories)
            all_file_agent_indices.append(all_agent_indices)
            all_file_names.append(file)
    all_file_states = jax.tree.map(lambda *x: jnp.stack(x), *all_file_states)  # (num_files, num_tasks, num_timesteps, *)
    all_file_actions = jnp.stack(all_file_actions)  # (num_files, num_tasks, num_timesteps)
    all_file_agent_indices = jnp.stack(all_file_agent_indices)  # (num_files, num_tasks)
    return all_file_states, all_file_actions, all_file_agent_indices, all_file_names

def load_video_final_state_from_data(all_file_states, video_file_prefix):
    """
    Load the final state from video data based on the file prefix.
    This version is robust to different PyTree container types (dict or dataclass).
    """
    try:
        # Parse user_id and task_id from video filename (e.g., 'video_3_5')
        parts = video_file_prefix.split('_')
        if len(parts) != 3 or parts[0] != 'video':
            raise ValueError(f"Invalid video file prefix format: {video_file_prefix}")
        
        user_id = int(parts[1])
        task_id = int(parts[2])
        
        logger.info(f"Loading state for {video_file_prefix}: user_id={user_id}, task_id={task_id}")

        # Robustly get the shape from the first leaf array in the PyTree
        try:
            first_leaf = jax.tree.leaves(all_file_states)[0]
            shape_info = first_leaf.shape
            logger.info(f"all_file_states PyTree structure detected. Leaf shape: {shape_info}")
            max_users, max_tasks = shape_info[:2]
        except (IndexError, AttributeError) as e:
            raise TypeError(f"Could not determine shape from all_file_states. It might not be a valid PyTree of arrays. Error: {e}")
            
        if user_id >= max_users or task_id >= max_tasks:
            raise ValueError(f"Invalid user_id={user_id} or task_id={task_id}. Max users: {max_users}, max tasks: {max_tasks}")
            
        # Get the full trajectory for the specific user and task.
        state_sequence = jax.tree.map(lambda x: x[user_id, task_id], all_file_states)
        
        # The goal is to get the state corresponding to the *last* frame of the video.
        # Our video generation script creates N+1 states (e.g., 16 states for 15 steps).
        # We must select the state at the very last index.
        
        # Determine the length of the loaded trajectory from one of its leaf arrays.
        trajectory_length = jax.tree.leaves(state_sequence)[0].shape[0]
        
        if trajectory_length == 0:
            logger.warning(f"Loaded trajectory for {video_file_prefix} is empty.")
            return None
            
        # The index of the final state is the last available index.
        final_timestep_idx = trajectory_length - 1
        
        # Extract the state at that final timestep.
        final_state_slice = jax.tree.map(lambda x: x[final_timestep_idx], state_sequence)
        
        # Log for verification
        agent_pos = final_state_slice.agent_locations if hasattr(final_state_slice, 'agent_locations') else 'not found'
        time_step = final_state_slice.time if hasattr(final_state_slice, 'time') else 'not found'
        logger.info(f"Successfully loaded final state slice for {video_file_prefix}: index={final_timestep_idx}, time={time_step}, agent_pos={agent_pos}")
        
        return final_state_slice
        
    except Exception as e:
        logger.error(f"Failed to load state for {video_file_prefix}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None
    
def make_video(state_seq, action_seq, filename):
    '''
    state_seq: (num_timesteps, h, w, 3) images
    action_seq: (num_timesteps, ) action indices
    makes a video of the agents trajectory from 0 to num_timesteps - 2
    saves video to filename.mp4, and screenshot of final timestep to filename.png
    '''
    action_to_name = ["NOOP", "Right", "Left", "Down", "Up", "Interact"]
    import cv2
    import numpy as np

    # Get dimensions of the state images
    h, w = state_seq.shape[1], state_seq.shape[2]
    
    # Scale factor to make images larger
    scale_factor = 3
    scaled_h, scaled_w = h * scale_factor, w * scale_factor
    
    # Add space for text (smaller relative to the enlarged image)
    text_height = 50
    frame_height = scaled_h + text_height
    
    # Create a video writer with H.264 codec for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec instead of mp4v
    out = cv2.VideoWriter(filename + '.mp4', fourcc, 2.0, (scaled_w, frame_height))
    
    # Create font outside the loop
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Prepare all frames first
    frames = []
    for i in range(state_seq.shape[0] - 1):
        # Get the current image and action
        img = state_seq[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize the image to make it larger
        img_resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        
        action_idx = int(action_seq[i])
        action_name = action_to_name[action_idx]
        
        # Create a frame with space for text
        frame = np.zeros((frame_height, scaled_w, 3), dtype=np.uint8)
        
        # Add the image to the top part
        frame[:scaled_h, :, :] = img_resized
        
        # Add a white background for text
        frame[scaled_h:, :, :] = [255, 255, 255]
        
        # Add the action text
        text = f"{i}. Action: {action_name}"
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        text_x = (scaled_w - text_size[0]) // 2
        text_y = scaled_h + (text_height + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, 0.7, (0, 0, 0), 2)
        
        frames.append(frame)
    
    # Write all frames to the video
    for frame in frames:
        out.write(frame)
    
    out.release()

    # Save the final image with action text
    img = state_seq[-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize the final image
    img_resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    
    # Create a frame with space for text
    frame = np.zeros((frame_height, scaled_w, 3), dtype=np.uint8)
    frame[:scaled_h, :, :] = img_resized
    frame[scaled_h:, :, :] = [255, 255, 255]
    
    # Add the action text for the final frame
    action_idx = int(action_seq[-1])
    action_name = action_to_name[action_idx]
    text = f"{state_seq.shape[0]-1}. Action: {action_name}"
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = (scaled_w - text_size[0]) // 2
    text_y = scaled_h + (text_height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(filename + '.png', frame)
    
    # Add a fallback WebM version for better browser compatibility
    try:
        webm_out = cv2.VideoWriter(filename + '.webm', cv2.VideoWriter_fourcc(*'VP90'), 3.0, (scaled_w, frame_height))
        for frame in frames:
            webm_out.write(frame)
        webm_out.release()
    except Exception as e:
        print(f"Could not create WebM fallback: {e}")


