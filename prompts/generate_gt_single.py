from vllm import LLM, SamplingParams
import torch
import pandas as pd
import time
import os
from rich.progress import track


# Load prompt from file
with open("stage2_gt_single_prompt.txt", "r") as f:
    base_prompt = f.read()

# Number of iterations to run
n_iterations = 20  # You can adjust this number

# Initialize the prompt with the base prompt
current_prompt = base_prompt

tensor_parallel_size = max(1, torch.cuda.device_count())

sampling_params = SamplingParams(max_tokens=5000, temperature=1.0)
print(f"Loading model")
llm = LLM(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    tensor_parallel_size=tensor_parallel_size,
    dtype=torch.bfloat16,
    gpu_memory_utilization=0.55
)
print(f"Warming up model")
llm.generate("This is me warming up the model", sampling_params=sampling_params)

# Save outputs to file
output_dir = "generated_outputs"
os.makedirs(output_dir, exist_ok=True)

import jax.random as random

# Initialize environment with 1 agent
from environment import AutomaticityEnv
from agent import AgentExecutionFramework
num_agents = 1
num_steps = 2
num_blocks = 3
env = AutomaticityEnv(num_agents=num_agents, size=10, max_steps=num_steps, num_blocks=num_blocks)
# Initialize agent execution framework
framework = AgentExecutionFramework()


# Loop through iterations
i = 0
while i  < n_iterations:
    print(f"Running iteration {i+1} of {n_iterations}")

    if current_prompt == base_prompt:
        additional_prompt = f'''\n Complete the above code skeleton to create a complete and working code for an FSM agent exhibiting behavior {i+1} from the list of behaviors above. 
        \n Return your response in a valid python code block surrounded by ```python and ```. \n'''
    else:
        additional_prompt = ''

    # Generate text based on current prompt
    current_prompt = current_prompt + additional_prompt
    outputs = llm.generate(current_prompt, sampling_params)
    
    # Extract generated text
    generated_text = outputs[0].outputs[0].text
    
    # Find the text after "</think>\n"
    think_marker = "class FSMAgent:"
    if think_marker in generated_text:
        # text_to_append = generated_text.split(think_marker, 1)[1]
        text_to_append = generated_text

        # try:
        #     fsm_agent = framework.compile_agent(text_to_append, num_agents=num_agents, num_blocks=num_blocks)
        #     obs, state = env.reset()
        #     action = fsm_agent.act(obs)
        #     obs, state = env.step(action)
        # except Exception as e: # making sure generation can execute
        #     breakpoint()
        #     print(f"Error: {e}")
        #     current_prompt = current_prompt + f"\n Ran into the following error with your last response: {e}\nDon't repeat this mistake and try again to make the FSM code. \n"
        #     continue
        current_prompt = base_prompt
        # Save the current state
        output_file = os.path.join(output_dir, f"fsm_agent_{i+1}.txt")
        with open(output_file, "w") as f:
            f.write(text_to_append)
        
        print(f"Completed iteration {i+1}, appended {len(text_to_append)} characters")
        i += 1
    else:
        print(f"Warning: Could not find '{think_marker}' in generated text for iteration {i+1}. Repeating")