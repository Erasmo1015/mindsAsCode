import asyncio
from pprint import pp

import aiofiles
import msgpack
import os
import flax
import jax
import jax.numpy as jnp



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
    'Always move right',
    'Wander randomly without any specific direction',
    'Always pick up the nearest block',
    'Move in a vertical line (up and down)',
    'Bounce off walls without moving beyond them',
    'Stay in place',
    'Always pick up purple blocks',
    'Only pick up the first block encountered',
    'Move towards the farthest block each time',
    'Follow a clockwise square pattern',
    'Snake through the grid (right, up, left, down)',
    'Collect blocks of a specific color',
    'Move left if possible, otherwise right',
    'Move in an L-shape pattern',
    'Oscillate between two points',
    'Follow a path to collect all blocks of a specific color',
    'Create a spiral movement pattern',
    'Move diagonally towards blocks',
    'Return to a specific location when possible',
    'Maximize the number of blocks collected frontally',
]

def load_and_stack_data(directory: str='./data/human_data'):
    while True:
        files = os.listdir(directory)
        files = [f for f in files if f.endswith('.json')]
        all_file_states = []
        all_file_actions = []
        all_file_agent_indices = []
        for i, file in enumerate(files):
            # print(f"loading file {i} of {len(files)}")
            filepath = os.path.join(directory, file)
            datapoints = asyncio.run(read_file(filepath))
            all_state_trajectories = []
            all_action_trajectories = []
            curr_state_trajectory = []
            curr_action_trajectory = []
            current_task = ""
            for datapoint in datapoints:
                # print("new datapoint")
                try:
                    task = datapoint['metadata']['task']
                    if task != current_task:
                        current_task = task
                        # stack the trajectories
                        if len(curr_state_trajectory) > 0:
                            stacked_state_trajectory = jax.tree.map(lambda *x: jnp.stack(x), *curr_state_trajectory)
                            stacked_action_trajectory = jnp.stack(curr_action_trajectory)
                            agent_id = task_list.index(task)
                            yield stacked_state_trajectory, stacked_action_trajectory, agent_id, file, task
                            del stacked_state_trajectory, stacked_action_trajectory
                            del agent_id
                            # all_state_trajectories.append(stacked_state_trajectory)
                            # all_action_trajectories.append(stacked_action_trajectory)
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
            # all_state_trajectories = jax.tree.map(lambda *x: jnp.stack(x), *all_state_trajectories)  # (num_tasks, num_timesteps, *)
            # all_action_trajectories = jnp.stack(all_action_trajectories)  # (num_tasks, num_timesteps)
            # all_agent_indices = jnp.arange(all_action_trajectories.shape[0])
            # if all_action_trajectories.shape[0] == 19:
            #     all_file_states.append(all_state_trajectories)
            #     all_file_actions.append(all_action_trajectories)
            #     all_file_agent_indices.append(all_agent_indices)
        # all_file_states = jax.tree.map(lambda *x: jnp.stack(x), *all_file_states)  # (num_files, num_tasks, num_timesteps, *)
        # all_file_actions = jnp.stack(all_file_actions)  # (num_files, num_tasks, num_timesteps)
        # all_file_agent_indices = jnp.stack(all_file_agent_indices)  # (num_files, num_tasks)
        # return all_file_states, all_file_actions, all_file_agent_indices


if __name__ == "__main__":
    # directory = '/Users/kunal/Code/UW/human_data'
    all_file_states, all_file_actions, all_file_agent_indices = load_and_stack_data()
    breakpoint()