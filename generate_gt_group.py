from vllm import LLM, SamplingParams
import torch
import pandas as pd
import time
import os
from rich.progress import track

phase = 3
if phase == 1:
    # Load prompt from file
    with open("prompts/stage1_gt_group_prompt.txt", "r") as f:
        base_prompt = f.read()

    # Number of iterations to run
    n_iterations = 20  # You can adjust this number

    tensor_parallel_size = max(1, torch.cuda.device_count())

    sampling_params = SamplingParams(max_tokens=5000, temperature=1.0)
    print(f"Loading model")
    llm = LLM(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.55
    )
    print(f"Warming up model")
    llm.generate("This is me warming up the model", sampling_params=sampling_params)

    # Save outputs to file
    output_dir = "generated_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Create a list of prompts (all using the same base prompt in this case)
    prompts = [base_prompt] * n_iterations
    
    # Generate all responses in a batch
    print(f"Generating {n_iterations} responses in batch")
    batch_outputs = llm.generate(prompts, sampling_params)
    
    # Save each response to a separate file
    for i, output in enumerate(batch_outputs):
        generated_text = output.outputs[0].text
        
        # Save the generated text
        output_file = os.path.join(output_dir, f"fsm_group_agent_{i+1}.txt")
        with open(output_file, "w") as f:
            f.write(generated_text)
        
        print(f"Saved response {i+1} of {n_iterations}")
elif phase == 2:
    # Load prompt from file
    with open("prompts/stage2_gt_group_prompt.txt", "r") as f:
        base_prompt = f.read()

    # Number of iterations to run
    n_iterations = 20  # You can adjust this number

    # Initialize the prompt with the base prompt
    current_prompt = base_prompt

    tensor_parallel_size = max(1, torch.cuda.device_count())

    sampling_params = SamplingParams(max_tokens=5000, temperature=1.0)
    print(f"Loading model")
    llm = LLM(
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )
    print(f"Warming up model")
    llm.generate("This is me warming up the model", sampling_params=sampling_params)

    import jax.random as random

    # Save outputs to file
    output_dir = "generated_outputs"
    os.makedirs(output_dir, exist_ok=True)
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
        
        think_marker = "class FSMAgent:"
        if think_marker in generated_text:
            # text_to_append = generated_text.split(think_marker, 1)[1]
            text_to_append = generated_text

            current_prompt = base_prompt
            # Save the current state
            output_file = os.path.join(output_dir, f"fsm_group_agent_{i+1}.txt")
            with open(output_file, "w") as f:
                f.write(text_to_append)
            
            print(f"Completed iteration {i+1}, appended {len(text_to_append)} characters")
            i += 1
        else:
            print(f"Warning: Could not find '{think_marker}' in generated text for iteration {i+1}. Repeating")
elif phase == 3:
    from environment import AutomaticityEnv
    from agent import AgentExecutionFramework
    num_agents = 4
    num_steps = 100
    num_blocks = 3
    n_iterations = 20
    env = AutomaticityEnv(num_agents=num_agents, size=10, max_steps=num_steps, num_blocks=num_blocks)
    

    def initialize_agent_list(agent_id_list, num_agents=1, num_blocks=1):
        framework = AgentExecutionFramework()
        agent_list = []
        for i in agent_id_list:
            print(f"Loading agent {i+1} of {n_iterations}")
            with open(f"generated_outputs/fsm_group_agent_{i+1}.txt", "r") as f:
                agent_code = f.read()
            agent_list.append(framework.compile_agent(agent_code, num_agents=num_agents, num_blocks=num_blocks))
        return agent_list
    
    agent_list = initialize_agent_list(list(range(n_iterations)), num_agents=num_agents, num_blocks=num_blocks)
    for i in range(n_iterations):
        print(f"Checking if agent {i+1} runs")
        obs, state = env.reset()
        agent = agent_list[i]
        for t in range(num_steps):
            actions = agent.act(obs[0])
            obs, state = env.step(state, actions)
    print(f"All agents run!")
