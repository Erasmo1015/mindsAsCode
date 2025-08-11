import asyncio
from pprint import pp

import aiofiles
import msgpack
import os
import flax
import jax
import jax.numpy as jnp
import numpy as np



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

task_list = [
    "Repeatedly go from the green-->blue-->purple blocks",
    "Patrol the border of the grid in a clockwise manner", 
    "Patrol the border of the grid in a counter-clockwise manner",
    "Alternate between going left until you hit a wall, then going right until you hit a wall", 
    "Pickup a blue block and place it next to another blue block",
    "Repeatedly move in an L-shape pattern", 
    "Pickup the green blocks and move them to a corner of the grid", 
    "Pickup the pink blocks and move them to a corner of the grid", 
    "Snake up and down across the rows of the grid",
    "Alternate between going up until you hit a wall, then going down until you hit a wall",
]

def load_and_save_human_gameplay_data(directory: str):
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
                        stacked_state_trajectory = jax.tree.map(lambda *x: np.stack(x), *curr_state_trajectory)
                        stacked_action_trajectory = np.stack(curr_action_trajectory)
                        task_id = task_list.index(current_task)
                        
                        # yield stacked_state_trajectory, stacked_action_trajectory, task_id, file, current_task

                        all_state_trajectories.append(stacked_state_trajectory)
                        all_action_trajectories.append(stacked_action_trajectory)
                    curr_state_trajectory = []
                    curr_action_trajectory = []
                action = np.array(datapoint['data']['action_idx'])
                state = datapoint['data']['timestep']['state']  # this is a dict. convert all leaves to numpy arrays
                state = jax.tree.map(lambda x: np.array(x), state)
                curr_state_trajectory.append(state)
                curr_action_trajectory.append(action)
            except Exception as e:
                # logger.error(f"Failed to process {filepath}: {e}")
                continue
        
        # # Add the final task's trajectories
        if len(curr_state_trajectory) > 0:
            stacked_state_trajectory = jax.tree.map(lambda *x: np.stack(x), *curr_state_trajectory)
            stacked_action_trajectory = np.stack(curr_action_trajectory)
            task_id = task_list.index(current_task)
            # yield stacked_state_trajectory, stacked_action_trajectory, task_id, file, current_task
            all_state_trajectories.append(stacked_state_trajectory)
            all_action_trajectories.append(stacked_action_trajectory)

        task_to_idx = {task: i for i, task in enumerate(all_tasks)}
        task_to_idx = np.array([task_list.index(task) for task in all_tasks])

        # Sort by task indices
        sort_indices = np.argsort(task_to_idx)

        not_complete = False
        for k_index, action_traj in enumerate(all_action_trajectories):
            if len(action_traj) < 30:
                not_complete = True
                break
        if not_complete:
            continue
        
        

        all_state_trajectories = jax.tree.map(lambda *x: np.stack(x), *all_state_trajectories)  # (num_tasks, num_timesteps, *)
        all_action_trajectories = np.stack(all_action_trajectories)  # (num_tasks, num_timesteps)
        
        # Reindex trajectories based on sorted task order
        all_state_trajectories = jax.tree.map(lambda x: x[sort_indices], all_state_trajectories)
        all_action_trajectories = all_action_trajectories[sort_indices]
        task_to_idx = task_to_idx[sort_indices]

        data_file = {
            'states': all_state_trajectories,
            'actions': all_action_trajectories,
            'tasks': all_tasks,
            'task_to_idx': task_to_idx,
            'sort_indices': sort_indices,
            'file': file,
        }
        # save data file to pkl with same filename but new extension
        import pickle

        pkl_file = file.replace('.json', '.pkl')
        with open(f'{directory}/{pkl_file}', 'wb') as f:
            pickle.dump(data_file, f)

        # all_agent_indices = np.arange(all_action_trajectories.shape[0])
        # if all_action_trajectories.shape[0] == len(task_list):  # number of tasks
        #     for i in range(all_action_trajectories.shape[0]):
        #         state_dat = jax.tree.map(lambda x: x[i], all_state_trajectories)
        #         action_dat = all_action_trajectories[i]
        #         agent_id = task_to_idx[i]
        #         breakpoint()
        #         yield state_dat, action_dat, agent_id, file, all_tasks[i]
        print(f'saved {file}')

def load_and_stack_human_gameplay_data(directory: str):
    import pickle
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.pkl')]
    for file in files:
        filepath = os.path.join(directory, file)
        with open(filepath, 'rb') as f:
            data_file = pickle.load(f)
            states = data_file['states']
            actions = data_file['actions']
            task_idx = data_file['sort_indices']
            filename = data_file['file']
            for i in range(actions.shape[0]):
                state_dat = jax.tree.map(lambda x: x[i], states)
                action_dat = actions[i]
                agent_id = task_idx[i]
                yield state_dat, action_dat, agent_id, filename, task_list[agent_id]

            


if __name__ == "__main__":
    # directory = '/Users/kunal/Code/UW/human_data'
    # load_and_save_human_gameplay_data('data/human_data_fix')
    load_and_stack_human_gameplay_data('data/human_data_fix')
    print('done')