```python
import numpy as np

class FSMAgent:

    # Define the internal states
    states = ['idle', 'moving_right', 'moving_left', 'moving_down', 'moving_up', 'picking_block', 'dropping_block']

    def __init__(self, num_agents: int, num_blocks: int, num_actions: int=6):
        self.num_agents = num_agents
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.actions = [0,1,2,3,4,5]  # stay, right, left, down, up, interact
        self.action_to_name = ["stay", "right", "left", "down", "up", "interact"]
        self.current_state = 'idle'
        self.block_color_in_inventory = None
    
    def check_wall_collision(self, current_pos, new_pos):
        walls = np.array([[0,0], [6,0], [0,6], [6,6]])
        for wall in walls:
            if (wall == new_pos).all():
                return True
        return False

    def check_internal_collision(self, current_pos, new_pos, other_agents_positions):
        for position in other_agents_positions:
            if (position == new_pos).all():
                return True
        return False

    def pick_block(self, current_pos, block_locations):
        for i, block in enumerate(block_locations):
            if (block == current_pos).all() and self.block_color_in_inventory is None:
                self.block_color_in_inventory = tuple(block)
                return 5  # interact to pick up block
        return 0  # stay in place

    def drop_block(self, current_pos, block_locations):
        for i, block in enumerate(block_locations):
            if (block == current_pos).all() and self.block_color_in_inventory is not None:
                self.block_color_in_inventory = None
                return 5  # interact to drop off block
        return 0  # stay in place

    def move_to_block(self, current_pos, target_pos, block_locations):
        if tuple(target_pos) in block_locations:
            return self.drop_block(current_pos, block_locations)
        elif self.block_color_in_inventory is not None:
            return 5  # interact with block already held
        else:
            dx, dy = target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]
            new_pos = np.array([current_pos[0] + dx, current_pos[1] + dy])
            if not self.check_wall_collision(current_pos, new_pos) and not self.check_internal_collision(new_pos, [], []):
                return 1 if dx > 0 else 2 if dx < 0 else 3 if dy > 0 else 4  # right, left, down, up
            else:
                return 0  # stay in place

    def idle_state(self, current_pos, agent_locations, block_locations):
        closest_block = np.argmin(((np.array(agent_locations) - np.array(block_locations)) ** 2).sum(axis=1))
        target_pos = block_locations[closest_block]
        return self.move_to_block(current_pos, target_pos, block_locations)

    def moving_right_state(self, current_pos, agent_locations, block_locations):
        next_pos = [current_pos[0] + 1, current_pos[1]]
        if not self.check_wall_collision(current_pos, next_pos) and not self.check_internal_collision(next_pos, [], []):
            return 1
        else:
            return 0

    def moving_left_state(self, current_pos, agent_locations, block_locations):
        next_pos = [current_pos[0] - 1, current_pos[1]]
        if not self.check_wall_collision(current_pos, next_pos) and not self.check_internal_collision(next_pos, [], []):
            return 2
        else:
            return 0

    def moving_down_state(self, current_pos, agent_locations, block_locations):
        next_pos = [current_pos[0], current_pos[1] + 1]
        if not self.check_wall_collision(current_pos, next_pos) and not self.check_internal_collision(next_pos, [], []):
            return 3
        else:
            return 0

    def moving_up_state(self, current_pos, agent_locations, block_locations):
        next_pos = [current_pos[0], current_pos[1] - 1]
        if not self.check_wall_collision(current_pos, next_pos) and not self.check_internal_collision(next_pos, [], []):
            return 4
        else:
            return 0

    def picking_block_state(self, current_pos, block_locations):
        return self.pick_block(current_pos, block_locations)

    def dropping_block_state(self, current_pos, block_locations):
        return self.drop_block(current_pos, block_locations)

    def act(self, observation) -> int:
        current_pos = np.array(observation['agent_locations'][0])
        block_locations = np.array(observation['block_locations'])
        agent_locations = np.array(observation['agent_locations'])
        self.block_color_in_inventory = None

        if self.block_color_in_inventory is not None:
            self.current_state = 'dropping_block'
        elif len(block_locations) > 0:
            self.current_state = 'idle'

        if self.current_state == 'idle':
            return self.idle_state(current_pos, agent_locations, block_locations)
        elif self.current_state == 'moving_right':
            return self.moving_right_state(current_pos, agent_locations, block_locations)
        elif self.current_state == 'moving_left':
            return self.moving_left_state(current_pos, agent_locations, block_locations)
        elif self.current_state == 'moving_down':
            return self.moving_down_state(current_pos, agent_locations, block_locations)
        elif self.current_state == 'moving_up':
            return self.moving_up_state(current_pos, agent_locations, block_locations)
        elif self.current_state == 'picking_block':
            return self.picking_block_state(current_pos, block_locations)
        elif self.current_state == 'dropping_block':
            return self.dropping_block_state(current_pos, block_locations)

        # Default state: idle
        return self.idle_state(current_pos, agent_locations, block_locations)
```

This code implements a finite state machine (FSM) for the agent. The FSM is designed to handle the agent's actions based on its current state and the given observations. The agent starts in the 'idle' state and tries to pick up the nearest block when there are any available. If it is carrying a block, it tries to drop it at the first opportunity. The movement states (`moving_right`, `moving_left`, `moving_down`, `moving_up`) are used to navigate towards the desired location while avoiding collisions with walls or other agents.