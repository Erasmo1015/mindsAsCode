
import numpy as np

class FSMAgent:
    def __init__(self, num_agents: int, num_blocks: int, num_actions: int=6):
        self.num_agents = num_agents
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.actions = [0, 1, 2, 3, 4, 5]  # stay, right, left, down, up, interact
        self.action_to_name = ["stay", "right", "left", "down", "up", "interact"]
        self.state = "IDLE"  # Initial state

    def act(self, observation) -> int:
        agent_id = observation['agent_id']
        agent_location = observation['agent_locations'][agent_id]
        inventory = observation['agent_inventory'][agent_id]

        if self.state == "IDLE":
            # Check if there is a block at the agent's location and we can interact with it
            for block_location in observation['block_locations']:
                if np.array_equal(block_location, agent_location):
                    if inventory == -1:
                        self.state = "INTERACT"
                        break
            else:
                # No block at the agent's location, check for possible movements
                possible_actions = []
                for action in self.actions[:-1]:  # Exclude interact
                    new_location = self.apply_action(agent_location, action)
                    if not self.is_wall(new_location, observation['wall_locations']) and not self.is_other_agent(new_location, observation['agent_locations'], agent_id):
                        possible_actions.append(action)
                if possible_actions:
                    self.state = "MOVE"
                    self.target_action = np.random.choice(possible_actions)

        if self.state == "MOVE":
            self.state = "IDLE"  # Transition back to IDLE after moving
            return self.target_action

        if self.state == "INTERACT":
            self.state = "IDLE"  # Transition back to IDLE after interacting
            return 5  # Interact action

    def apply_action(self, location, action):
        if action == 1:  # right
            return [location[0], location[1] + 1]
        elif action == 2:  # left
            return [location[0], location[1] - 1]
        elif action == 3:  # down
            return [location[0] + 1, location[1]]
        elif action == 4:  # up
            return [location[0] - 1, location[1]]
        else:
            return location  # stay

    def is_wall(self, location, wall_locations):
        for wall in wall_locations:
            if np.array_equal(wall, location):
                return True
        return False

    def is_other_agent(self, location, agent_locations, agent_id):
        for i, agent_loc in enumerate(agent_locations):
            if i != agent_id and np.array_equal(agent_loc, location):
                return True
        return False
