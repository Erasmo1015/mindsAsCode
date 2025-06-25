import numpy as np
from typing import Dict, Tuple, List
from flax import struct
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import cv2
import imageio
import jax
import jax.numpy as jnp
from functools import partial

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
        self.actions = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]  # stay, right, left, down, up, interact
        self.action_to_name = ["stay", "right", "left", "down", "up", "interact"]
        
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], State]:
        # Create outer walls
        wall_locations = []
        for i in range(self.size):
            wall_locations.extend([(i, 0), (i, self.size-1), (0, i), (self.size-1, i)])
        
        num_interior_walls = self.num_walls
        interior_positions = [(i, j) for i in range(2, self.size-2)   # walls leave at least 1 space
                            for j in range(2, self.size-2)]
        interior_walls = np.random.choice(len(interior_positions), 
                                        size=num_interior_walls, 
                                        replace=False)
        wall_locations.extend([interior_positions[i] for i in interior_walls])
        self.wall_locations = np.array(wall_locations)
        
        # Randomly place agents in non-wall positions
        available_positions = [(i, j) for i in range(1, self.size-1) 
                             for j in range(1, self.size-1) 
                             if (i, j) not in wall_locations]
        agent_positions = np.random.choice(len(available_positions), 
                                         size=self.num_agents, 
                                         replace=False)
        self.agent_locations = np.array([available_positions[i] for i in agent_positions])
        
        # Initialize colored blocks in remaining available positions
        used_positions = wall_locations + [tuple(pos) for pos in self.agent_locations]
        available_positions = [(i, j) for i in range(1, self.size-1) 
                             for j in range(1, self.size-1) 
                             if (i, j) not in used_positions]
        assert len(available_positions) >= self.num_blocks
        block_positions = np.random.choice(len(available_positions), 
                                         size=self.num_blocks, 
                                         replace=False)
        self.block_locations = np.array([available_positions[i] for i in block_positions])
        assert len(self.block_locations) == self.num_blocks
        
        # Assign fixed colors to blocks
        self.block_colors = block_colors[:self.num_blocks]
        
        # Initialize inventory and inventory colors for each agent
        self.agent_inventory = [-1] * self.num_agents
        self.agent_inventory_colors = np.full((self.num_agents, 3), -1)
        
        state = State(
            wall_locations=self.wall_locations,
            agent_locations=self.agent_locations,
            block_locations=self.block_locations,
            agent_inventory=np.array(self.agent_inventory),
            agent_inventory_colors=self.agent_inventory_colors,  # Add inventory colors
            block_colors=self.block_colors,
            time=0,
            terminal=False
        )
        
        return self.get_observation(state), state
    
    def step(self, state: State, actions: List[int]) -> Tuple[Dict[str, np.ndarray], State]:
        if len(actions) != self.num_agents:
            breakpoint()
            raise ValueError("Must provide actions for all agents")
        
        # Calculate new positions for all agents
        new_locations = []
        new_inventory = list(state.agent_inventory)
        new_inventory_colors = state.agent_inventory_colors.copy()
        new_blocks = state.block_locations.copy()
        new_block_colors = state.block_colors.copy()
        
        # Track which blocks are being carried by agents
        carried_blocks = {}  # Maps block index to agent index
        for agent_idx, inv in enumerate(new_inventory):
            if inv != -1:
                # Find the block this agent is carrying
                for block_idx, block_color in enumerate(new_block_colors):
                    color_match = np.array_equal(block_color, new_inventory_colors[agent_idx])
                    location_match = np.array_equal(new_blocks[block_idx], state.agent_locations[agent_idx])
                    if color_match and location_match:
                        carried_blocks[block_idx] = agent_idx
                        break
        
        for agent_idx, action_idx in enumerate(actions):
            if action_idx not in range(len(self.actions)):
                raise ValueError(f"Invalid action: {action_idx}")
            action = self.actions[action_idx]
            
            current_pos = state.agent_locations[agent_idx]
            new_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
            
            # Check if new position would cross paths with other agents
            for other_idx, other_pos in enumerate(state.agent_locations):
                if other_idx != agent_idx:
                    # Check if agents would swap positions
                    other_action = self.actions[actions[other_idx]]
                    other_new_pos = (other_pos[0] + other_action[0], other_pos[1] + other_action[1])
                    
                    if (np.array_equal(new_pos, other_pos) or  # Moving into occupied spot
                        (np.array_equal(new_pos, other_new_pos) and  # Moving to same spot
                         len(new_locations) > other_idx) or  # Other agent already moved
                        (np.array_equal(new_pos, other_new_pos) and  # Would swap positions
                         np.array_equal(current_pos, other_pos))):
                        new_pos = current_pos  # Stay in current position
                        break
            
            # Check if new position has a block that's not being carried
            for block_idx, block_pos in enumerate(new_blocks):
                if block_idx in carried_blocks:
                    continue  # Skip blocks that are already being carried
                
                if np.array_equal(new_pos, block_pos):
                    if new_inventory[agent_idx] == -1:
                        # Agent picks up the block
                        new_inventory[agent_idx] = block_idx
                        new_inventory_colors[agent_idx] = new_block_colors[block_idx]
                        carried_blocks[block_idx] = agent_idx
                    else:
                        # Can't move onto block if inventory is full
                        new_pos = current_pos
                    break
            
            
            # Handle interact action
            if action[2] == 1:  # Interact button pressed
                if new_inventory[agent_idx] != -1:
                    # Drop the block at current position
                    # The block is already in new_blocks, just update its status
                    for block_idx, agent_carrying in carried_blocks.items():
                        if agent_carrying == agent_idx:
                            # Block is no longer carried
                            carried_blocks.pop(block_idx)
                            break
                    new_inventory[agent_idx] = -1
                    new_inventory_colors[agent_idx] = np.array([-1, -1, -1])
                new_pos = current_pos  # Stay in place when interacting
            
            # Check if new position is valid
            if (any(np.array_equal(new_pos, wall) for wall in state.wall_locations) or
                not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size)):
                new_pos = current_pos
            

            new_locations.append(new_pos)
        
        # Update positions of carried blocks to match their carriers
        new_locations = np.array(new_locations)
        for block_idx, agent_idx in carried_blocks.items():
            new_blocks[block_idx] = new_locations[agent_idx]
        
        # Create new state
        new_state = State(
            wall_locations=state.wall_locations,
            agent_locations=new_locations,
            block_locations=new_blocks,
            agent_inventory=np.array(new_inventory),
            agent_inventory_colors=new_inventory_colors,
            block_colors=new_block_colors,
            time=state.time + 1,
            terminal=state.time + 1 >= self.max_steps
        )
        
        return self.get_observation(new_state), new_state
    
    def get_observation(self, state: State) -> Dict[str, np.ndarray]:
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
                "agent_id": agent_id
            }
            obs_list.append(obs)
        return obs_list
        

def state_to_image(state: Dict[str, np.ndarray], size: int, tile_size: int = 10) -> np.ndarray:
    """Convert a state into a RGB numpy array with shape (H, W, 3)."""
    # Create a blank white canvas
    image = np.ones((size * tile_size, size * tile_size, 3), dtype=np.uint8) * 255
    
    # Draw grid lines (thinner)
    for i in range(size + 1):
        # Draw single-pixel grid lines
        image[min(i * tile_size, image.shape[0] - 1), :] = [200, 200, 200]  # horizontal lines
        image[:, min(i * tile_size, image.shape[1] - 1)] = [200, 200, 200]  # vertical lines
    
    # Draw walls (gray) (jax array of num_walls x 2)
    for wall in state['wall_locations']:
        x, y = wall
        image[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] = [128, 128, 128]
    
    # Draw blocks (colored squares centered in cells) (jax array of num_blocks x 2)
    for idx, block in enumerate(state['block_locations']):
        x, y = block
        color = state['block_colors'][idx]
        # Create centered square coordinates (smaller than the cell)
        block_size = int(tile_size * 0.6)  # 60% of tile size
        offset = (tile_size - block_size) // 2  # Center the block
        square_x1, square_y1 = x*tile_size + offset, y*tile_size + offset
        square_x2, square_y2 = square_x1 + block_size, square_y1 + block_size
        # Fill square with the block's color
        image[square_y1:square_y2, square_x1:square_x2] = color.tolist()
    
    # Draw agents (colored circles) (jax array of num_agents x 2)
    for idx, agent in enumerate(state['agent_locations']):
        x, y = agent
        center = (int((x+0.5)*tile_size), int((y+0.5)*tile_size))
        # Draw agent circle with the agent's color
        agent_color = agent_colors[idx % len(agent_colors)]
        radius = max(2, tile_size // 5)  # Scale radius with tile size
        cv2.circle(image, center, radius, color=agent_color.tolist(), thickness=-1)
        
        # If agent has an item, draw small square inside with the block's color
        if state['agent_inventory'][idx] != -1:
            small_square_size = max(2, tile_size // 5)
            small_x1 = center[0] - small_square_size//2
            small_y1 = center[1] - small_square_size//2
            small_x2 = center[0] + small_square_size//2
            small_y2 = center[1] + small_square_size//2
            # Use the color from agent's inventory (jax array of num_agents x 3)
            block_color = state['agent_inventory_colors'][idx]
            image[small_y1:small_y2, small_x1:small_x2] = block_color.tolist()
    
    return image

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return jnp.logical_and(
            jnp.logical_and(x >= xmin, x <= xmax),
            jnp.logical_and(y >= ymin, y <= ymax)
        )
    return fn

def point_in_circle(cx, cy, r):
    def fn(x, y):
        return ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
    return fn

def fill_coords(img, fn, color):
    y, x = jnp.meshgrid(jnp.arange(img.shape[0]), jnp.arange(img.shape[1]), indexing='ij')
    yf = (y + 0.5) / img.shape[0]
    xf = (x + 0.5) / img.shape[1]
    mask = fn(xf, yf)
    return jnp.where(mask[:, :, None], color, img)


def state_to_image_jit(state, img_size, size, tile_size=10):
    """Convert a state into a RGB numpy array with shape (H, W, 3) using JAX for JIT compatibility."""
    
    # Create a blank white canvas
    image = jnp.ones((img_size, img_size, 3), dtype=jnp.uint8) * 255
    
    # Draw grid lines
    grid_color = jnp.array([200, 200, 200], dtype=jnp.uint8)
    
    # Fix grid lines by making them thicker and properly scaled
    for i in range(size + 1):
        # Calculate normalized position for grid lines
        pos = i / size
        # Make lines thicker for visibility
        thickness = 0.01  # Adjust thickness as needed
        
        # Horizontal lines
        fn = point_in_rect(0, 1, pos - thickness, pos + thickness)
        image = fill_coords(image, fn, grid_color)
        
        # Vertical lines
        fn = point_in_rect(pos - thickness, pos + thickness, 0, 1)
        image = fill_coords(image, fn, grid_color)
    
    # Draw walls (gray)
    wall_color = jnp.array([128, 128, 128], dtype=jnp.uint8)
    
    def draw_wall(carry, wall):
        img = carry
        x, y = wall[0] / size, wall[1] / size
        fn = point_in_rect(x, x + 1/size, y, y + 1/size)
        return fill_coords(img, fn, wall_color), None
    
    # Apply draw_wall to each wall location
    image, _ = jax.lax.scan(
        draw_wall,
        image,
        state['wall_locations']
    )

    # Draw blocks (colored squares centered in cells)
    def draw_block(carry, block_data):
        img = carry
        block, color = block_data
        x, y = block[0] / size, block[1] / size
        
        # Create centered square
        block_size = 0.6 / size  # 60% of tile size
        offset = (1/size - block_size) / 2  # Center the block
        fn = point_in_rect(
            x + offset, x + offset + block_size,
            y + offset, y + offset + block_size
        )
        return fill_coords(img, fn, color).astype(jnp.uint8), None
    
    # Apply draw_block to each block
    image, _ = jax.lax.scan(
        draw_block,
        image,
        (state['block_locations'], state['block_colors'])
    )
    
    # Draw agents (colored circles)
    def draw_agent(carry, i):
        img = carry
        agent = state['agent_locations'][i]
        x, y = (agent[0] + 0.5) / size, (agent[1] + 0.5) / size
        
        # Get agent color
        agent_color = agent_colors[i % len(agent_colors)]
        
        # Draw agent circle
        radius = max(0.02, 0.2 / size)  # Scale radius with tile size
        fn = point_in_circle(x, y, radius)
        img = fill_coords(img, fn, agent_color).astype(jnp.uint8)
        
        # If agent has an item, draw small square inside with the block's color
        has_item = state['agent_inventory'][i] != -1
        
        def draw_inventory(img_with_agent):
            inventory_color = state['agent_inventory_colors'][i]
            
            # Small square inside the circle
            small_square_size = max(0.02, 0.1 / size)
            fn = point_in_rect(
                x - small_square_size/2, x + small_square_size/2,
                y - small_square_size/2, y + small_square_size/2
            )
            return fill_coords(img_with_agent, fn, inventory_color).astype(jnp.uint8)
        
        # Only draw inventory if agent has an item
        img = jax.lax.cond(
            has_item,
            draw_inventory,
            lambda x: x,
            img
        )
        return img.astype(jnp.uint8), None
    
    # Apply draw_agent to each agent using scan
    image, _ = jax.lax.scan(
        draw_agent,
        image,
        jnp.arange(len(state['agent_locations']))
    )
    
    return image

if __name__ == "__main__":
    num_agents = 10
    tile_size = 8
    grid_size = 10

    env = AutomaticityEnv(num_agents=num_agents, size=grid_size, max_steps=100, num_blocks=10, num_walls=20)
    obs, state = env.reset()
    obs = obs[0]
    obs = jax.tree.map(lambda x: jnp.array(x), obs)
    
    # Test the original function
    image = state_to_image(obs, grid_size, tile_size=tile_size)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image)
    
    # Test the JIT-compatible function
    jitted_render = jax.jit(state_to_image_jit, static_argnums=(1, 2, 3))
    img_size = grid_size * tile_size
    jit_image = jitted_render(obs, img_size, grid_size, tile_size)
    plt.subplot(1, 2, 2)
    plt.title("JIT")
    plt.imshow(jit_image)
    
    plt.savefig("environment_comparison.png")
