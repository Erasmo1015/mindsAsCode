# import numpy as np

# class FSMAgent:
#     def __init__(self, num_agents: int, num_blocks: int, num_actions: int=6):
#         self.num_agents = num_agents
#         self.num_blocks = num_blocks
#         self.num_actions = num_actions
#         # Add interact action (0, 0, 1) - last digit indicates interact button
#         self.actions = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]  # stay, right, left, down, up, interact
#         self.action_to_name = ["stay", "right", "left", "down", "up", "interact"]
    
#     '''
#     Your helper functions for decision-making go here
#     '''
        
#     def act(self, observation) -> int:
#         '''
#         observation is a dictionary with the following keys:
#         - wall_locations: Array of wall coordinates (num_walls, 2)
#         - agent_locations: Array of agent coordinates (num_agents, 2)
#         - block_locations: Array of block coordinates (num_blocks, 2)
#         - agent_inventory: Array indicating if each agent has a block (num_agents)
#         - agent_inventory_colors: Array of colors for blocks in inventory (num_agents, 3)
#         - block_colors: Array of colors for blocks in environment (num_blocks, 3)
#         - time: int
#         - terminal: bool
#         - agent_id: int
#         '''

#         # TODO: Implement the agent's decision-making logic here
#         # This is a placeholder implementation that returns a random action
#         action = np.random.randint(self.num_actions)



#         ## DON'T CHANGE ANYTHING BELOW HERE
#         assert action in range(self.num_actions)
#         return action


llm_code = '''
```
import numpy as np

class FSMAgent:
    def __init__(self, num_agents: int=1, num_blocks: int=1, num_actions: int=6):
        self.num_agents = num_agents
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.actions = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
        self.action_to_name = ["stay", "right", "left", "down", "up", "interact"]
        
        self.goal_position = (8, 8)  # Delivery target in bottom-right
        self.last_action = None
        self.stuck_counter = 0

    def _find_nearest_block(self, current_pos, block_locations):
        if block_locations.size == 0:
            return None
        distances = np.sum(np.abs(block_locations - current_pos), axis=1)
        return tuple(block_locations[np.argmin(distances)])

    def _get_movement_action(self, current, target):
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 2  # Right or Left
        elif dy != 0:
            return 3 if dy > 0 else 4  # Down or Up
        else:
            return 0  # Stay if at target

    def _is_valid_move(self, action, current_pos, observation):
        dx, dy, _ = self.actions[action]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        # Wall collision check
        if any((new_pos[0], new_pos[1]) == tuple(wall) for wall in observation['wall_locations']):
            return False
            
        # Agent collision check
        for i, pos in enumerate(observation['agent_locations']):
            if i != observation['agent_id'] and tuple(pos) == new_pos:
                return False
                
        return True

    def act(self, observation) -> int:
        agent_id = observation['agent_id']
        current_pos = tuple(observation['agent_locations'][agent_id])
        has_block = observation['agent_inventory'][agent_id] != -1

        if not has_block:
            # Block acquisition phase
            if not observation['block_locations'].size:
                return 0  # No blocks available
            
            target = self._find_nearest_block(current_pos, observation['block_locations'])
            if target is None:
                return 0
        else:
            # Block delivery phase
            if current_pos == self.goal_position:
                return 5  # Drop block at goal
            target = self.goal_position

        desired_action = self._get_movement_action(current_pos, target)
        
        # Check if the agent is stuck
        if self.last_action == desired_action and not self._is_valid_move(desired_action, current_pos, observation):
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # If stuck, try alternative routes
        if self.stuck_counter > 2:
            alternative_actions = [a for a in range(5) if a != desired_action]
            np.random.shuffle(alternative_actions)
            for action in alternative_actions:
                if self._is_valid_move(action, current_pos, observation):
                    self.last_action = action
                    return action

        # If not stuck or no alternative found, proceed with desired action
        if self._is_valid_move(desired_action, current_pos, observation):
            self.last_action = desired_action
            return desired_action

        # If desired action is invalid, find any valid move
        for action in range(5):  # Try all moves except interact
            if self._is_valid_move(action, current_pos, observation):
                self.last_action = action
                return action

        # If no valid move, stay put
        self.last_action = 0
        return 0
```
'''

import re
import numpy as np
import traceback
from typing import Dict, Any
import sys
import contextlib

class NullWriter:
    def write(self, _): pass
    def flush(self): pass

@contextlib.contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = NullWriter()
    sys.stderr = NullWriter()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

class AgentExecutionFramework:
    """
    Framework for safely executing LLM-generated agent code with NumPy support
    """
    
    def __init__(self):
        self._execution_namespace = self._create_safe_namespace()
        
    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Create isolated namespace with allowed imports and modules"""
        return {
            'np': np,  # Pre-injected NumPy instance
            '__builtins__': {
                **__builtins__,
                'open': None,  # Disable file operations
                'importlib': None,
                'eval': None,
            }
        }
    
    def _extract_code(self, llm_output: str) -> str:
        """Extract Python code from markdown code blocks"""
        # Look for code between triple backticks
        code_pattern = r'```(?:python)?(.*?)```'
        matches = re.findall(code_pattern, llm_output, re.DOTALL)
        
        if matches:
            return '\n'.join([m.strip() for m in matches])
        return llm_output  # Fallback to raw output

    def _validate_agent_class(self, namespace: Dict[str, Any]) -> None:
        """Verify existence of valid Agent class with act method"""
        if 'FSMAgent' not in namespace:
            raise ValueError("LLM output does not contain Agent class definition")
            
        AgentClass = namespace['FSMAgent']
        if not hasattr(AgentClass, 'act') or not callable(AgentClass(1, 1).act):
            raise TypeError("Agent class missing required 'act' method")

    def compile_agent(self, llm_output: str, num_agents: int=1, num_blocks: int=1) -> Any:
        """
        Compile LLM output into executable Agent class
        Returns instantiated Agent object
        """
        try:
            # Extract and clean code
            raw_code = self._extract_code(llm_output)
            code = f'{raw_code}\n\n_AGENT_INSTANCE = FSMAgent({num_agents}, {num_blocks})'  # instantiate with num_agents and num_blocks
            
            # Execute in isolated namespace
            with suppress_output():
                exec(code, self._execution_namespace)
            self._validate_agent_class(self._execution_namespace)
            
            return self._execution_namespace['_AGENT_INSTANCE']
            
        except Exception as e:
            # traceback.print_exc()
            raise RuntimeError(f"Agent compilation failed: {str(e)}") from e

    def execute_agent(
        self,
        agent: Any,
        observation: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Execute agent's act method with observation validation"""
        if not isinstance(observation, dict):
            raise TypeError("Observation must be a dictionary")
                
        try:
            with suppress_output():
                return agent.act(observation)
        except Exception as e:
            # traceback.print_exc()
            raise RuntimeError(f"Agent execution failed: {str(e)}") from e

if __name__ == "__main__":
    agent = AgentExecutionFramework()
    agent_instance = agent.compile_agent(llm_code)
    action = agent.execute_agent(agent_instance, {})
    print(action)