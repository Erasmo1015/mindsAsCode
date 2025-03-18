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

# generate 20 random colors
color_key = jax.random.PRNGKey(1)
block_colors = jax.random.randint(color_key, (100, 3), 0, 256)
# shuffle the colors
shuffle_key = jax.random.PRNGKey(2)
block_colors = jax.random.permutation(shuffle_key, block_colors)
agent_colors = jnp.array([
    [255, 0, 0],    # red
    [0, 0, 255],    # blue 
    [0, 255, 0],    # green
    [255, 255, 0],  # yellow
    [128, 0, 128],  # purple
    [255, 165, 0],  # orange
    [165, 42, 42],  # brown
    [255, 192, 203],# pink
    [128, 128, 128],# gray
    [0, 255, 255]   # cyan
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
    
    def step(self, state: State, actions: List[Tuple[int, int, int]]) -> Tuple[Dict[str, np.ndarray], State]:
        if len(actions) != self.num_agents:
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
                    if np.array_equal(block_color, new_inventory_colors[agent_idx]):
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
            
            # Check if new position has a block that's not being carried
            for block_idx, block_pos in enumerate(new_blocks):
                if block_idx in carried_blocks:
                    continue  # Skip blocks that are already being carried
                
                if np.array_equal(new_pos, block_pos):
                    if new_inventory[agent_idx] == -1:
                        # Agent picks up the block
                        new_inventory[agent_idx] = 1
                        new_inventory_colors[agent_idx] = new_block_colors[block_idx]
                        carried_blocks[block_idx] = agent_idx
                    else:
                        # Can't move onto block if inventory is full
                        new_pos = current_pos
                    break
            
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
        

def state_to_image(state: Dict[str, np.ndarray], size: int) -> np.ndarray:
    """Convert a state into a RGB numpy array with shape (H, W, 3)."""
    # Create a blank white canvas
    image = np.ones((size * 50, size * 50, 3), dtype=np.uint8) * 255
    
    # Draw grid lines
    for i in range(size + 1):
        image[i * 50 - 1:i * 50 + 1, :] = [0, 0, 0]  # horizontal lines
        image[:, i * 50 - 1:i * 50 + 1] = [0, 0, 0]  # vertical lines
    
    # Draw walls (gray) (jax array of num_walls x 2)
    for wall in state['wall_locations']:
        x, y = wall
        image[y*50:(y+1)*50, x*50:(x+1)*50] = [128, 128, 128]
    
    # Draw blocks (colored triangles) (jax array of num_blocks x 2)
    for idx, block in enumerate(state['block_locations']):
        x, y = block
        color = state['block_colors'][idx]
        # Create triangle coordinates
        triangle_pts = np.array([
            [(x+0.5)*50, (y+0.2)*50],  # top
            [(x+0.2)*50, (y+0.8)*50],  # bottom left
            [(x+0.8)*50, (y+0.8)*50]   # bottom right
        ], dtype=np.int32)
        # Fill triangle with the block's color
        cv2.fillPoly(image, [triangle_pts], color=color.tolist())
    
    # Draw agents (colored circles) (jax array of num_agents x 2)
    for idx, agent in enumerate(state['agent_locations']):
        x, y = agent
        center = (int((x+0.5)*50), int((y+0.5)*50))
        # Draw agent circle with the agent's color
        agent_color = agent_colors[idx % len(agent_colors)]
        cv2.circle(image, center, 20, color=agent_color.tolist(), thickness=-1)
        
        # If agent has an item, draw small triangle inside with the block's color
        if state['agent_inventory'][idx] != -1:
            small_triangle_pts = np.array([
                [center[0], center[1]-10],      # top
                [center[0]-10, center[1]+10],   # bottom left
                [center[0]+10, center[1]+10]    # bottom right
            ], dtype=np.int32)
            # Use the color from agent's inventory (jax array of num_agents x 3)
            block_color = state['agent_inventory_colors'][idx]
            cv2.fillPoly(image, [small_triangle_pts], color=block_color.tolist())
    
    return image



def state_to_image_jit(state: Dict[str, jnp.ndarray], size: int) -> jnp.ndarray:
    """Convert a state into a RGB numpy array with shape (H, W, 3) using JAX for JIT compatibility.
    Uses vectorized operations for better performance."""
    tile_size = 12
    img_size = size * tile_size
    
    # Start with white background
    image = jnp.ones((img_size, img_size, 3), dtype=jnp.uint8) * 255
    
    # Draw grid lines
    for i in range(size + 1):
        # Create grid line masks
        h_mask = point_in_rect(0, img_size, i*tile_size-1, i*tile_size+1)
        v_mask = point_in_rect(i*tile_size-1, i*tile_size+1, 0, img_size)
        
        # Apply grid lines
        image = fill_coords(image, h_mask, jnp.array([0, 0, 0]))
        image = fill_coords(image, v_mask, jnp.array([0, 0, 0]))
    
    # Draw walls (gray)
    def draw_wall(i, img):
        wall = state['wall_locations'][i]
        x, y = wall[0], wall[1]
        wall_mask = point_in_rect(x*tile_size, (x+1)*tile_size, y*tile_size, (y+1)*tile_size)
        return fill_coords(img, wall_mask, jnp.array([128, 128, 128], dtype=jnp.uint8))
    
    image = jax.lax.fori_loop(0, len(state['wall_locations']), draw_wall, image)
    
    # Draw blocks (colored triangles)
    def draw_block(i, img):
        block_loc = state['block_locations'][i]
        color = state['block_colors'][i]
        x, y = block_loc[0], block_loc[1]
        
        # Create triangle points (normalized coordinates within cell)
        top = (x + 0.5, y + 0.2)
        bottom_left = (x + 0.2, y + 0.8)
        bottom_right = (x + 0.8, y + 0.8)
        
        # Create triangle mask
        triangle_mask = point_in_triangle(top, bottom_left, bottom_right)
        return fill_coords(img, triangle_mask, color)
    
    image = jax.lax.fori_loop(0, len(state['block_locations']), draw_block, image)
    
    # Draw agents (colored circles)
    def draw_agent(i, img):
        agent_loc = state['agent_locations'][i]
        x, y = agent_loc[0], agent_loc[1]
        agent_color = agent_colors[i % len(agent_colors)]
        
        # Create circle mask
        center_x, center_y = (x + 0.5), (y + 0.5)
        radius = 0.4  # 20px / 50px = 0.4 in normalized coordinates
        
        def circle_mask(px, py):
            return ((px - center_x)**2 + (py - center_y)**2) <= radius**2
        
        # Apply agent color
        img = fill_coords(img, circle_mask, agent_color)
        
        # Draw inventory triangle if agent has item
        has_item = state['agent_inventory'][i] != -1
        
        def draw_inventory():
            inventory_color = state['agent_inventory_colors'][i]
            
            # Small triangle inside the circle (normalized coordinates)
            tri_top = (center_x, center_y - 0.2)
            tri_bl = (center_x - 0.2, center_y + 0.2)
            tri_br = (center_x + 0.2, center_y + 0.2)
            
            # Create triangle mask
            tri_mask = point_in_triangle(tri_top, tri_bl, tri_br)
            return fill_coords(img, tri_mask, inventory_color)
        
        return jax.lax.cond(has_item, draw_inventory, lambda: img)
    
    image = jax.lax.fori_loop(0, len(state['agent_locations']), draw_agent, image)
    
    return image

def point_in_rect(xmin, xmax, ymin, ymax):
    """Returns a function that tests if a point is within a rectangle."""
    def fn(x, y):
        return jnp.logical_and(
            jnp.logical_and(x >= xmin, x <= xmax),
            jnp.logical_and(y >= ymin, y <= ymax)
        )
    return fn

def point_in_triangle(a, b, c):
    """Returns a function that tests if a point is within a triangle."""
    a, b, c = jnp.array(a), jnp.array(b), jnp.array(c)
    def fn(x, y):
        # Create a mesh grid of points
        y_indices, x_indices = jnp.meshgrid(jnp.arange(x.shape[0]), jnp.arange(x.shape[1]), indexing='ij')
        # Convert to normalized coordinates
        yf = y_indices / x.shape[0]
        xf = x_indices / x.shape[1]
        
        # Vectorized barycentric coordinate calculation
        v0 = c - a
        v1 = b - a
        
        # For each point in the grid
        points = jnp.stack([xf.ravel(), yf.ravel()], axis=1)
        v2 = points - a
        
        # Compute dot products
        dot00 = jnp.dot(v0, v0)
        dot01 = jnp.dot(v0, v1)
        dot11 = jnp.dot(v1, v1)
        
        # Vectorized dot products for all points
        dot02 = jnp.dot(v2, v0)
        dot12 = jnp.dot(v2, v1)
        
        # Compute barycentric coordinates
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Check if point is in triangle
        mask = jnp.logical_and(
            jnp.logical_and(u >= 0, v >= 0),
            (u + v) <= 1
        )
        
        return mask.reshape(x.shape)
    return fn

def fill_coords(img, fn, color):
    """Fill pixels in the image according to a mask function."""
    # Create coordinate grid
    y_indices, x_indices = jnp.meshgrid(jnp.arange(img.shape[0]), jnp.arange(img.shape[1]), indexing='ij')
    # Normalize coordinates to [0, 1] range
    yf = y_indices / img.shape[0]
    xf = x_indices / img.shape[1]
    
    # Apply mask function
    mask = fn(xf, yf)
    mask = jnp.expand_dims(mask, axis=-1)
    
    # Fill pixels where mask is True
    return jnp.where(mask, color, img)
