```python
import numpy as np

class FSMAgent:

    def __init__(self, num_agents: int, num_blocks: int, num_actions: int=6):
        self.num_agents = num_agents
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.actions = [0,1,2,3,4,5]  # stay, right, left, down, up, interact
        self.action_to_name = ["stay", "right", "left", "down", "up", "interact"]
        self.current_location = None
        self.inventory = -1
        self.target_block_index = None
    
    def update_state(self, observation):
        self.current_location = np.array(observation['agent_locations'][0])
        self.inventory = observation['agent_inventory'][0]
        self.block_locations = np.array(observation['block_locations'])
        self.block_colors = np.array(observation['block_colors'])
        self.target_block_index = None

    def find_target_block(self):
        for i, block_loc in enumerate(self.block_locations):
            if np.array_equal(block_loc, self.current_location) and self.inventory == -1:
                self.target_block_index = i
                break
            elif self.inventory != -1 and np.array_equal(block_loc, self.block_locations[self.inventory]):
                self.target_block_index = -1
                break

    def act(self, observation) -> int:
        self.update_state(observation)
        self.find_target_block()

        if self.target_block_index is not None:
            target_loc = self.block_locations[self.target_block_index]
            if self.inventory == -1:
                if self.current_location[0] < target_loc[0]:  # move down
                    return 3
                elif self.current_location[0] > target_loc[0]:  # move up
                    return 1
                elif self.current_location[1] < target_loc[1]:  # move right
                    return 2
                elif self.current_location[1] > target_loc[1]:  # move left
                    return 4
                else:
                    self.inventory = self.target_block_index
                    return 5  # interact
            else:
                self.target_block_index = None
                self.inventory = -1
                return 0  # stay
        else:
            if self.inventory == -1:
                return 0  # stay
            else:
                if self.block_locations[self.inventory][0] < self.current_location[0]:  # move up
                    return 1
                elif self.block_locations[self.inventory][0] > self.current_location[0]:  # move down
                    return 3
                elif self.block_locations[self.inventory][1] < self.current_location[1]:  # move left
                    return 4
                elif self.block_locations[self.inventory][1] > self.current_location[1]:  # move right
                    return 2
                else:
                    self.inventory = -1
                    return 0  # stay

        # If none of the above conditions are met, default to stay
        return 0
```

This implementation of `FSMAgent` follows the FSM logic described in the experiences. It updates its state based on the current observation and then determines the next action based on whether it should pick up a block, drop a block, or move towards a target block. The `act` method returns the appropriate action index as specified in the problem statement.