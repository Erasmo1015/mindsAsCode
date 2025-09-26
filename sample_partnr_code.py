
import numpy as np

class FSMAgent:
    def __init__(self, num_agents: int=1, num_blocks: int=1):
        self.num_agents = num_agents
        self.num_blocks = num_blocks  # irrelevant, can ignore

    def parse_scene_graph(self, observation):
        for keys in observation['scene_graph']:
            if keys == 'furniture':
                for room_name, furniture_list in observation['scene_graph'][keys].items():
                    for furniture_piece in furniture_list:
                        pass  # each furniture_piece is a string
            if keys == 'objects':
                if type(observation['scene_graph'][keys]) == list and len(observation['scene_graph'][keys]) == 0:
                    pass  # no objects seen
                else:
                    for object, object_holder_list in observation['scene_graph'][keys].items():
                        for object_holder in object_holder_list:
                            pass # each object is either on or in an object holder
        return # do whatever is most helpful here

    def act(self, observation) -> int:
        '''
        observation is a dictionary with the following keys:
        - tool_list: List of tools available to the agent
        - tool_descriptions: Description of how each tool is used
        - scene_graph: Scene graph of the environment, dictionary with keys
            - "furniture" which maps to a dictionary with the keys
                - room description string (i.e. keys could be "living_room_1", "bathroom_1", etc.) that maps to list of 
                    - object_id string (i.e. table_21, chair_32, etc.)
            - "objects" which maps to a dictionary of 
                - object_id string (i.e. keys could be "plate_container_2", "vase_1" etc.) to list of 
                    - object_base string (i.e "table_14", "table_21")
                if type(observation['scene_graph']['objects']) == list, then you do not observe any objects
        - agent_state: Dictionary mapping to 
            - string of agent id (i.e. "0") maps to string describing what agent is doing
        '''
        agent_id = list(observation['agent_state'].keys())[0]
        agent_state = observation['agent_state'][agent_id]
        tool_list = observation['tool_list']

        if 'Explore' in tool_list:
            tool = 'Explore'
            target = list(observation['scene_graph']['furniture'].keys())[0]
        elif 'Pick' in tool_list and 'Standing' in agent_state:
            tool = 'Pick'
            targets = []
            for key in observation['scene_graph']['objects']:
                if 'agent_0' in observation['scene_graph']['objects'][key]:
                    targets.append(key)
            if targets:
                target = targets[0]
            else:
                target = None
        elif 'Place' in tool_list and 'Standing' in agent_state:
            tool = 'Place'
            target = None
            for key in observation['scene_graph']['objects']:
                if agent_id in observation['scene_graph']['objects'][key]:
                    target = key
                    break
            if not target:
                for key in observation['scene_graph']['furniture']:
                    for furniture_piece in observation['scene_graph']['furniture'][key]:
                        if agent_id in observation['scene_graph']['furniture'][key]:
                            target = key
                            break
            if not target:
                target = list(observation['scene_graph']['objects'].keys())[0]
        else:
            tool = 'Wait'
            target = None

        ## DON'T CHANGE ANYTHING BELOW HERE
        return (tool, target, None)
