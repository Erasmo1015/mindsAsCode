from functools import partial
from typing import Dict, List, Tuple

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from matplotlib import pyplot as plt

# generate 20 random colors
# Define block colors to alternate between green, blue, purple, orange, pink and cyan
block_colors = jnp.array([
    [0, 255, 0],     # green
    [0, 0, 255],     # blue
    [128, 0, 128],   # purple 
    [255, 192, 203], # pink
    [0, 255, 255],   # cyan
])

# Repeat the pattern to get 100 colors
block_colors = jnp.tile(block_colors, (20,1))[:100]

agent_colors = jnp.array([
    [255, 0, 0],     # red
    [0, 0, 255],     # blue 
    [0, 255, 0],     # green
    [255, 255, 0],   # yellow
    [128, 0, 128],   # purple
    [255, 165, 0],   # orange
    [165, 42, 42],   # brown
    [255, 192, 203], # pink
    [128, 128, 128], # gray
    [0, 255, 255]    # cyan
])



@struct.dataclass
class State:
    wall_locations: jnp.ndarray
    agent_locations: jnp.ndarray
    block_locations: jnp.ndarray
    agent_inventory: jnp.ndarray
    agent_inventory_colors: jnp.ndarray  # Add colors for blocks in inventory
    block_colors: jnp.ndarray
    time: int
    terminal: bool
    agent_id: int=-1

class AutomaticityEnv:
    def __init__(self, num_agents: int, size: int = 10, max_steps: int = 100, num_blocks: int = 10, num_walls: int = 20):
        self.size = size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.num_blocks = num_blocks
        self.num_walls = num_walls
        # Add interact action (0, 0, 1) - last digit indicates interact button
        self.actions = jnp.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]])  # stay, right, left, down, up, interact
        self.action_to_name = ["stay", "right", "left", "down", "up", "interact"]

        self.all_num_walls = (self.size * 4) + self.num_walls
        self.num_border_walls = self.size * 4
         
    @partial(jax.jit, static_argnums=0)
    def reset(self, rng: jax.random.PRNGKey) -> Tuple[Dict[str, np.ndarray], State]:
        # Wall locations (outer border)
        coords = jnp.arange(self.size)
        wall_edges = jnp.concatenate(
            [
                jnp.stack([coords, jnp.zeros_like(coords)], axis=1),  # left
                jnp.stack(
                    [coords, (self.size - 1) * jnp.ones_like(coords)], axis=1
                ),  # right
                jnp.stack([jnp.zeros_like(coords), coords], axis=1),  # top
                jnp.stack(
                    [(self.size - 1) * jnp.ones_like(coords), coords], axis=1
                ),  # bottom
            ],
            axis=0,
        )
        wall_locations = wall_edges

        # Interior wall positions
        interior_i, interior_j = jnp.meshgrid(
            jnp.arange(2, self.size - 2), jnp.arange(2, self.size - 2), indexing="ij"
        )
        interior_positions = jnp.stack([interior_i.ravel(), interior_j.ravel()], axis=1)

        rng, subrng = jax.random.split(rng)
        selected_wall_idxs = jax.random.choice(
            subrng, interior_positions.shape[0], shape=(self.num_walls,), replace=False
        )
        selected_interior_walls = interior_positions[selected_wall_idxs]
        wall_locations = jnp.concatenate(
            [wall_locations, selected_interior_walls], axis=0
        )

        # Get mask for all positions inside the wall (no border)
        all_i, all_j = jnp.meshgrid(
            jnp.arange(1, self.size - 1), jnp.arange(1, self.size - 1), indexing="ij"
        )
        all_positions = jnp.stack([all_i.ravel(), all_j.ravel()], axis=1)

        # Filter out wall positions
        def is_not_wall(pos, walls):
            return jnp.all(jnp.any(walls != pos, axis=1))

        def get_available_positions(all_pos, walls):
            return jax.vmap(lambda p: is_not_wall(p, walls))(all_pos)

        available_mask = get_available_positions(all_positions, wall_locations)
        available_mask_prob = available_mask / jnp.sum(available_mask)

        # Sample agent positions
        rng, subrng = jax.random.split(rng)
        agent_indices = jax.random.choice(
            subrng,
            all_positions.shape[0],
            shape=(self.num_agents,),
            replace=False,
            p=available_mask_prob,
        )
        agent_locations = all_positions[agent_indices]

        # Exclude agent positions to get available for blocks
        combined_used = jnp.concatenate([wall_locations, agent_locations], axis=0)
        available_mask = get_available_positions(all_positions, combined_used)
        available_mask_prob = available_mask / jnp.sum(available_mask)

        rng, subrng = jax.random.split(rng)
        block_indices = jax.random.choice(
            subrng,
            all_positions.shape[0],
            shape=(self.num_blocks,),
            replace=False,
            p=available_mask_prob,
        )
        block_locations = all_positions[block_indices]

        agent_inventory = -jnp.ones((self.num_agents,), dtype=jnp.int32)
        agent_inventory_colors = -jnp.ones((self.num_agents, 3), dtype=jnp.int32)

        state = State(
            wall_locations=wall_locations,
            agent_locations=agent_locations,
            block_locations=block_locations,
            agent_inventory=agent_inventory,
            agent_inventory_colors=agent_inventory_colors,
            block_colors=block_colors[: self.num_blocks],
            time=0,
            terminal=False,
        )

        obs = self.get_observation(state)  # You must jaxify this too

        return obs, state
    
    @partial(jax.jit, static_argnums=0)
    def step(
        self, state: State, actions: jnp.ndarray
    ) -> Tuple[List[Dict[str, jnp.ndarray]], State]:
        if len(actions) != self.num_agents:
            raise ValueError("Must provide actions for all agents")

        # Calculate new positions for all agents
        new_inventory = state.agent_inventory
        new_inventory_colors = state.agent_inventory_colors
        new_blocks = state.block_locations
        new_block_colors = state.block_colors

        # Track which blocks are being carried by agents
        def _update_carried_blocks(carried_blocks, agent_idx):
            def _inner_loop():
                agent_inventory_color = new_inventory_colors[agent_idx]
                color_matches = jnp.all(
                    new_block_colors == agent_inventory_color, axis=1
                )
                location_matches = jnp.all(  # need location matching to handle multiple blocks of same color
                    new_blocks == state.agent_locations[agent_idx], axis=1
                )
                block_idx = jnp.argmax(jnp.logical_and(color_matches, location_matches))
                return carried_blocks.at[block_idx].set(
                    jax.lax.select(
                        jnp.logical_and(color_matches[block_idx], location_matches[block_idx]), agent_idx, carried_blocks[block_idx]
                    )
                )

            carried_blocks = jax.lax.cond(
                new_inventory[agent_idx] != -1, _inner_loop, lambda: carried_blocks
            )

            return carried_blocks, None

        carried_blocks, _ = jax.lax.scan(
            _update_carried_blocks,
            jnp.full(len(new_block_colors), -1, dtype=int),
            jnp.arange(self.num_agents),
        )

        def _update_agent(carry, agent_idx):
            carried_blocks, new_inventory, new_inventory_colors, new_blocks = carry
            action = self.actions[actions[agent_idx]]
            current_pos = state.agent_locations[agent_idx]
            new_pos = current_pos + action[:2]

            # Check if new position would cross path with other agents
            def _agent_conflict(other_idx):
                different_agent = other_idx != agent_idx
                other_action = self.actions[actions[other_idx]]
                other_pos = state.agent_locations[other_idx]
                other_new_pos = other_pos + other_action[:2]
                move_into_occupied = jnp.array_equal(new_pos, other_pos)
                move_into_same = jnp.array_equal(new_pos, other_new_pos)
                other_agent_moved = agent_idx > other_idx
                would_swap_positions = jnp.array_equal(
                    new_pos, other_new_pos
                ) & jnp.array_equal(current_pos, other_pos)
                should_keep_pos = (
                    move_into_occupied
                    | (move_into_same & other_agent_moved)
                    | would_swap_positions
                )
                return should_keep_pos & different_agent

            has_conflict = jnp.any(
                jax.vmap(_agent_conflict)(jnp.arange(self.num_agents))
            )
            new_pos = jax.lax.select(has_conflict, current_pos, new_pos)

            # Handle interact action
            is_interact_action = action[2] == 1
            # if interacting, agent stays in the same place
            new_pos = jax.lax.select(is_interact_action, current_pos, new_pos)

            # Check if new position is valid (walls and bounds)
            touching_wall = jnp.any(jnp.all(new_pos == state.wall_locations, axis=1))
            within_bounds = jnp.all((new_pos >= 0) & (new_pos <= self.size))
            new_pos = jax.lax.select(
                touching_wall | ~within_bounds, current_pos, new_pos
            )

            # Check if new position has a block that's not being carried
            block_pos_match = jnp.all(new_blocks == new_pos, axis=1)
            block_not_carried = carried_blocks == -1
            can_use_block = block_pos_match & block_not_carried
            selected_block = jnp.argmax(can_use_block)
            inventory_full = new_inventory[agent_idx] != -1

            # If moving to a position with a block and have inventory, stay in place
            new_pos = jax.lax.select(
                can_use_block[selected_block] & inventory_full, current_pos, new_pos
            )

            # Handle dropping blocks - only when using interact action and have inventory
            def _drop_block():
                # Find which block this agent is carrying
                block_idx = new_inventory[agent_idx]
                # Place the block at the agent's current position
                new_blocks_dropped = new_blocks.at[block_idx].set(current_pos)
                # Remove from carried blocks
                new_carried_blocks = jnp.where(
                    carried_blocks == agent_idx, -1, carried_blocks
                )
                return (
                    new_carried_blocks,
                    new_inventory.at[agent_idx].set(-1),
                    new_inventory_colors.at[agent_idx].set([-1, -1, -1]),
                    new_blocks_dropped,
                    True,
                )

            has_inventory = new_inventory[agent_idx] != -1
            carried_blocks, new_inventory, new_inventory_colors, new_blocks, dropped_block = jax.lax.cond(
                is_interact_action & has_inventory,
                _drop_block,
                lambda: (carried_blocks, new_inventory, new_inventory_colors, new_blocks, False),
            )

            # Handle picking up blocks - only when using interact action and don't have inventory
            def _pickup_block():
                # Check if current position has a block that's not being carried
                block_pos_match = jnp.all(new_blocks == current_pos, axis=1)
                block_not_carried = carried_blocks == -1
                can_use_block = block_pos_match & block_not_carried
                selected_block = jnp.argmax(can_use_block)
                inventory_empty = new_inventory[agent_idx] == -1
                
                return jax.lax.cond(
                    can_use_block[selected_block] & inventory_empty,
                    lambda: (
                        new_inventory.at[agent_idx].set(selected_block),
                        new_inventory_colors.at[agent_idx].set(
                            new_block_colors[selected_block]
                        ),
                        carried_blocks.at[selected_block].set(agent_idx),
                        new_blocks,  # Keep block positions unchanged when picking up
                    ),
                    lambda: (new_inventory, new_inventory_colors, carried_blocks, new_blocks),
                )

            # Only try to pick up blocks if using interact action and don't have inventory
            inventory_empty = ~has_inventory
            new_inventory, new_inventory_colors, carried_blocks, new_blocks = jax.lax.cond(
                is_interact_action & inventory_empty,
                _pickup_block,
                lambda: (new_inventory, new_inventory_colors, carried_blocks, new_blocks),
            )

            return (carried_blocks, new_inventory, new_inventory_colors, new_blocks), new_pos

        (carried_blocks, new_inventory, new_inventory_colors, new_blocks), new_locations = (
            jax.lax.scan(
                _update_agent,
                (carried_blocks, new_inventory, new_inventory_colors, new_blocks),
                jnp.arange(self.num_agents),
            )
        )

        # Update positions of carried blocks to match their carriers
        new_blocks = jax.lax.select(
            jnp.broadcast_to(
                jnp.expand_dims(carried_blocks != -1, axis=1), new_blocks.shape
            ),
            new_locations[carried_blocks],
            new_blocks,
        )

        # Create new state
        new_state = State(
            wall_locations=state.wall_locations,
            agent_locations=new_locations,
            block_locations=new_blocks,
            agent_inventory=new_inventory,
            agent_inventory_colors=new_inventory_colors,
            block_colors=new_block_colors,
            time=state.time + 1,
            terminal=state.time + 1 >= self.max_steps,
        )

        return self.get_observation(new_state), new_state
    
    def get_observation(self, state: State) -> Dict[str, jnp.ndarray]:
        obs_list = []
        for agent_id in range(self.num_agents):
            obs = {
                "wall_locations": state.wall_locations,
                "agent_locations": state.agent_locations,
                "block_locations": state.block_locations,
                "agent_inventory": state.agent_inventory,
                "agent_inventory_colors": state.agent_inventory_colors,
                "block_colors": state.block_colors,
                "time": state.time,
                "terminal": state.terminal,
                "agent_id": agent_id,
            }
            obs_list.append(obs)
        return obs_list


def state_to_image(
    state: Dict[str, jnp.ndarray], size: int, tile_size: int = 10
) -> np.ndarray:
    """Convert a state into a RGB numpy array with shape (H, W, 3)."""
    # Create a blank white canvas
    image = np.ones((size * tile_size, size * tile_size, 3), dtype=np.uint8) * 255

    # Draw grid lines (thinner)
    for i in range(size + 1):
        # Draw single-pixel grid lines
        image[min(i * tile_size, image.shape[0] - 1), :] = [
            200,
            200,
            200,
        ]  # horizontal lines
        image[:, min(i * tile_size, image.shape[1] - 1)] = [
            200,
            200,
            200,
        ]  # vertical lines

    # Draw walls (gray) (jax array of num_walls x 2)
    for wall in state["wall_locations"]:
        x, y = wall
        image[
            y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size
        ] = [128, 128, 128]

    # Draw blocks (colored squares centered in cells) (jax array of num_blocks x 2)
    for idx, block in enumerate(state["block_locations"]):
        x, y = block
        color = state["block_colors"][idx]
        # Create centered square coordinates (smaller than the cell)
        block_size = int(tile_size * 0.6)  # 60% of tile size
        offset = (tile_size - block_size) // 2  # Center the block
        square_x1, square_y1 = x * tile_size + offset, y * tile_size + offset
        square_x2, square_y2 = square_x1 + block_size, square_y1 + block_size
        # Fill square with the block's color
        image[square_y1:square_y2, square_x1:square_x2] = color.tolist()

    # Draw agents (colored circles) (jax array of num_agents x 2)
    for idx, agent in enumerate(state["agent_locations"]):
        x, y = agent
        center = (int((x + 0.5) * tile_size), int((y + 0.5) * tile_size))
        # Draw agent circle with the agent's color
        agent_color = agent_colors[idx % len(agent_colors)]
        radius = max(2, tile_size // 5)  # Scale radius with tile size
        cv2.circle(image, center, radius, color=agent_color.tolist(), thickness=-1)

        # If agent has an item, draw small square inside with the block's color
        if state["agent_inventory"][idx] != -1:
            small_square_size = max(2, tile_size // 5)
            small_x1 = center[0] - small_square_size // 2
            small_y1 = center[1] - small_square_size // 2
            small_x2 = center[0] + small_square_size // 2
            small_y2 = center[1] + small_square_size // 2
            # Use the color from agent's inventory (jax array of num_agents x 3)
            block_color = state["agent_inventory_colors"][idx]
            image[small_y1:small_y2, small_x1:small_x2] = block_color.tolist()

    return image


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return jnp.logical_and(
            jnp.logical_and(x >= xmin, x <= xmax), jnp.logical_and(y >= ymin, y <= ymax)
        )

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return ((x - cx) ** 2 + (y - cy) ** 2) <= r**2

    return fn


def fill_coords(img, fn, color, normalize=True):
    y, x = jnp.meshgrid(
        jnp.arange(img.shape[0]), jnp.arange(img.shape[1]), indexing="ij"
    )
    if normalize:
        yf = (y + 0.5) / img.shape[0]
        xf = (x + 0.5) / img.shape[1]
    else:
        yf = y
        xf = x
    mask = fn(xf, yf)
    return jnp.where(mask[:, :, None], color, img)


def state_to_image_jit(state, grid_size, tile_size):
    """Convert a state into a RGB numpy array with shape (H, W, 3) using JAX for JIT compatibility."""
    img_size = grid_size * tile_size

    # Create a blank white canvas
    image = jnp.ones((img_size, img_size, 3), dtype=jnp.uint8) * 255

    # Draw grid lines
    grid_color = jnp.array([200, 200, 200], dtype=jnp.uint8)

    # Fix grid lines by making them thicker and properly scaled
    for i in range(grid_size + 1):
        # Calculate normalized position for grid lines
        pos = i * tile_size
        # Lines have thickness of 1

        # Horizontal lines
        fn = point_in_rect(0, img_size-1, pos, pos)
        image = fill_coords(image, fn, grid_color, normalize=False)

        # Vertical lines
        fn = point_in_rect(pos, pos, 0, img_size-1)
        image = fill_coords(image, fn, grid_color, normalize=False)

    # Draw walls (gray)
    wall_color = jnp.array([128, 128, 128], dtype=jnp.uint8)

    def draw_wall(carry, wall):
        img = carry
        x, y = wall[0] / grid_size, wall[1] / grid_size
        fn = point_in_rect(x, x + 1 / grid_size, y, y + 1 / grid_size)
        return fill_coords(img, fn, wall_color), None

    # Apply draw_wall to each wall location
    image, _ = jax.lax.scan(draw_wall, image, state["wall_locations"])

    # Draw blocks (colored squares centered in cells)
    def draw_block(carry, block_data):
        img = carry
        block, color = block_data
        x, y = block[0] / grid_size, block[1] / grid_size

        # Create centered square
        block_size = 0.6 / grid_size  # 60% of tile size
        offset = (1 / grid_size - block_size) / 2  # Center the block
        fn = point_in_rect(
            x + offset, x + offset + block_size, y + offset, y + offset + block_size
        )
        return fill_coords(img, fn, color).astype(jnp.uint8), None

    # Apply draw_block to each block
    image, _ = jax.lax.scan(
        draw_block, image, (state["block_locations"], state["block_colors"])
    )

    # Draw agents (colored circles)
    def draw_agent(carry, i):
        img = carry
        agent = state["agent_locations"][i]
        x, y = (agent[0] + 0.5) / grid_size, (agent[1] + 0.5) / grid_size

        # Get agent color
        agent_color = agent_colors[i % len(agent_colors)]

        # Draw agent circle
        # Fix: Adjust radius calculation to ensure it's always a circle
        # Use a fixed proportion of the cell size instead of a min value
        radius = 0.3 / grid_size  # 40% of cell size - consistent circle size
        fn = point_in_circle(x, y, radius)
        img = fill_coords(img, fn, agent_color).astype(jnp.uint8)

        # If agent has an item, draw small square inside with the block's color
        has_item = state["agent_inventory"][i] != -1

        def draw_inventory(img_with_agent):
            inventory_color = state["agent_inventory_colors"][i]

            # Small square inside the circle - also scale with grid size
            small_square_size = 0.2 / grid_size  # 20% of cell size
            fn = point_in_rect(
                x - small_square_size / 2,
                x + small_square_size / 2,
                y - small_square_size / 2,
                y + small_square_size / 2,
            )
            return fill_coords(img_with_agent, fn, inventory_color).astype(jnp.uint8)

        # Only draw inventory if agent has an item
        img = jax.lax.cond(has_item, draw_inventory, lambda x: x, img)
        return img.astype(jnp.uint8), None

    # Apply draw_agent to each agent using scan
    image, _ = jax.lax.scan(
        draw_agent, image, jnp.arange(len(state["agent_locations"]))
    )

    return image


sample_grid = '''
#######
#     #
#1A  2#
#     #
#3    #
#     #
#######
'''

def str_to_grid(state_str: str):
    lines = state_str.strip().split('\n')
    height = len(lines)
    width = len(lines[0])
    wall_locations = []
    agent_locations = []
    block_locations = []
    agent_inventory = []
    agent_inventory_colors = []
    block_colors_grid = []
    time = 0
    terminal = False
    agent_id = 0
    curr_agent_id = 0
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == '#':
                wall_locations.append((x, y))
            elif char == ' ':
                pass
            elif char == 'A':
                agent_locations.append((x, y))
                agent_inventory.append(-1)
                agent_inventory_colors.append(agent_colors[curr_agent_id])
                curr_agent_id += 1
            elif char.isdigit():
                block_locations.append((x, y))
                block_colors_grid.append(block_colors[int(char)-1])

    return State(
        wall_locations=jnp.array(wall_locations),
        agent_locations=jnp.array(agent_locations),
        block_locations=jnp.array(block_locations),
        agent_inventory=jnp.array(agent_inventory),
        agent_inventory_colors=jnp.array(agent_inventory_colors),
        block_colors=jnp.array(block_colors_grid),
        time=time,
        terminal=terminal,
        agent_id=agent_id
    )

if __name__ == "__main__":
    num_agents = 1
    tile_size = 10
    grid_size = 7

    env = AutomaticityEnv(
        num_agents=num_agents,
        size=grid_size,
        max_steps=100,
        num_blocks=6,
        num_walls=0,
    )
    rng = jax.random.PRNGKey(0)
    # jitted_reset = jax.jit(env.reset, static_argnums=0)
    obs, state = env.reset(rng)
    state = str_to_grid(sample_grid)
    obs = env.get_observation(state)


    obs = obs[0]
    obs = jax.tree.map(lambda x: jnp.array(x), obs)
    # img1 = state_to_image(obs, grid_size, tile_size=tile_size)
    # plt.imshow(img1)
    # plt.savefig("img1.png")

    # state = state.replace(agent_inventory=[1, 1, 1])

    obs, state = env.step(state, jnp.array([1]))
    obs = obs[0]
    obs = jax.tree.map(lambda x: jnp.array(x), obs)

    # Test the original function
    # image = state_to_image(obs, grid_size, tile_size=tile_size)
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.title("Original")
    # plt.imshow(image)

    # # Test the JIT-compatible function
    # jitted_render = jax.jit(state_to_image_jit, static_argnums=(1, 2))
    # img_size = grid_size * tile_size
    # jit_image = jitted_render(obs, grid_size, tile_size)
    # plt.subplot(1, 2, 2)
    # plt.title("JIT")
    # plt.imshow(jit_image)

    # plt.savefig("test.svg")
    
    image = state_to_image_jit(obs, grid_size, tile_size=tile_size)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("block_cycle.svg")