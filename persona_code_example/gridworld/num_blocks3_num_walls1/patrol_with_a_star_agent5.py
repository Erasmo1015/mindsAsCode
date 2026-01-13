```python
import numpy as np

class FSMAgent:

    def __init__(self, num_agents: int, num_blocks: int, num_actions: int=6):
        self.num_agents = num_agents
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.actions = [0,1,2,3,4,5]  # stay, right, left, down, up, interact
        self.action_to_name = ["stay", "right", "left", "down", "up", "interact"]
        
        # Initialize states
        self.state = 'idle'
        self.block_to_pick_up = None
    
    def get_relative_position(self, target_location, current_location):
        delta_y = target_location[0] - current_location[0]
        delta_x = target_location[1] - current_location[1]
        if delta_y > 0:
            return 'down'
        elif delta_y < 0:
            return 'up'
        elif delta_x > 0:
            return 'right'
        elif delta_x < 0:
            return 'left'
    
    def update_state(self, current_location, block_locations, agent_inventory):
        if not self.block_to_pick_up is None and any((block_locations == self.block_to_pick_up).all(axis=1)):
            if agent_inventory[self.agent_id] == -1:
                self.state = 'moving_to_pick_up_block'
            else:
                self.state = 'idle'
        elif self.block_to_pick_up is not None:
            self.state = 'picked_up_block'
        else:
            self.state = 'idle'
    
    def act(self, observation) -> int:
        self.agent_id = observation['agent_id']
        self.block_locations = observation['block_locations']
        self.agent_locations = observation['agent_locations']
        self.agent_inventory = observation['agent_inventory']
        self.block_colors = observation['block_colors']
        self.time = observation['time']
        self.terminal = observation['terminal']
        
        self.update_state(self.agent_locations[self.agent_id], self.block_locations, self.agent_inventory)
        
        # Idle state: Move towards the block if no block is being held and no specific block is targeted
        if self.state == 'idle':
            if not -1 in self.agent_inventory:
                for i, block_location in enumerate(self.block_locations):
                    if block_location[0] == self.agent_locations[self.agent_id][0] and \
                       block_location[1] == self.agent_locations[self.agent_id][1]:
                        continue
                    if self.block_colors[i][0] != -1:  # Block is present
                        relative_pos = self.get_relative_position(block_location, self.agent_locations[self.agent_id])
                        if relative_pos == 'right':
                            action = 1  # right
                        elif relative_pos == 'left':
                            action = 2  # left
                        elif relative_pos == 'up':
                            action = 3  # up
                        elif relative_pos == 'down':
                            action = 4  # down
                        else:
                            action = 0  # stay
                        return action
        
        # Moving to pick up the block state
        if self.state == 'moving_to_pick_up_block':
            relative_pos = self.get_relative_position(self.block_to_pick_up, self.agent_locations[self.agent_id])
            if relative_pos == 'right':
                action = 1  # right
            elif relative_pos == 'left':
                action = 2  # left
            elif relative_pos == 'up':
                action = 3  # up
            elif relative_pos == 'down':
                action = 4  # down
            else:
                self.block_to_pick_up = None
                return 0  # idle after picking up the block
        
        # Picked up the block state
        if self.state == 'picked_up_block':
            for i, block_location in enumerate(self.block_locations):
                if block_location[0] == self.agent_locations[self.agent_id][0] and \
                   block_location[1] == self.agent_locations[self.agent_id][1]:
                    if self.block_colors[i][0] != -1:  # Check if there's a block there
                        self.block_to_pick_up = block_location
                        return 5  # interact
                        
            # Otherwise, move towards the dropped block location
            if not self.block_to_pick_up is None:
                relative_pos = self.get_relative_position(self.block_to_pick_up, self.agent_locations[self.agent_id])
                if relative_pos == 'right':
                    action = 1  # right
                elif relative_pos == 'left':
                    action = 2  # left
                elif relative_pos == 'up':
                    action = 3  # up
                elif relative_pos == 'down':
                    action = 4  # down
                else:
                    return 0  # idle
                
            else:
                return 0  # idle after dropping the block
                
        # Default: Stay in case of unexpected state
        return 0
```

In this implementation, we define three main states for the FSM: `idle`, `moving_to_pick_up_block`, and `picked_up_block`. The agent decides its next action based on its current state and the relative position of the block(s) in its environment. The agent interacts with the environment by picking up and dropping blocks, moving towards them, and checking if it should drop a block if it's already holding one.