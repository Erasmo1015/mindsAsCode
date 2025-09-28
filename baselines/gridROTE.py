from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
# Add this at the top of your file, before any CUDA operations
torch.multiprocessing.set_start_method('spawn', force=True)
import pandas as pd
import time
import os
from rich.progress import track
import jax
import numpy as np
from tqdm import tqdm
from agent import AgentExecutionFramework
# Import vLLM
from vllm import LLM, SamplingParams
import traceback

openai_api_key = os.environ["OPENAI_API_KEY"]

class ROTEReasoner:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size: int = 1, device: str = "cuda", 
                 dtype: str = "float16", gpu_memory_utilization: float = 0.55, num_hypothesis: int = 4,
                 max_model_len: int = 2048, quantization: str = None, group: bool = False, two_stage: bool = False,
                 structured: str = "False", oracle: bool = False):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.device = device
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_hypothesis = num_hypothesis
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.two_stage = two_stage
        self.structured = structured
        self.oracle = oracle
        self.action_to_name = {
            0: "stay",
            1: "right",
            2: "left",
            3: "down",
            4: "up",
            5: "interact"
        }
        self.name_to_action = {v: k for k, v in self.action_to_name.items()}
        self.action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
        self.str_action_space = ["stay", "right", "left", "down", "up", "interact"]

        self.group = group
        if not group:
            self.dataset_name = "single_agent_dataset"
            prompt_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/infer_single_fsm.txt"
            if structured == "p1":
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_single_code_template.txt"
            elif structured == "p2":
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_p2_single_code_template.txt"
                self.first_stage_prompt = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_p1_single_code_template.txt"
            else:
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/single_code_template.txt"
        else:
            self.dataset_name = "group_agent_dataset"
            prompt_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/infer_group_fsm.txt"
            if structured == "p1":
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_group_code_template.txt"
            elif structured == "p2":
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_p2_group_code_template.txt"
                self.first_stage_prompt = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_p1_group_code_template.txt"
            else:
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/group_code_template.txt"
        refinement_1_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_1.txt"
        refinement_2_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_2.txt"
        refinement_3_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_3.txt"

        self.base_prompt = open(prompt_path, "r").read()
        self.code_template = open(code_template_path, "r").read()
        self.refinement_1 = open(refinement_1_path, "r").read()
        self.refinement_2 = open(refinement_2_path, "r").read()
        self.refinement_3 = open(refinement_3_path, "r").read()

        if self.oracle:
            program_dir = "/mmfs1/gscratch/socialrl/kjha/automaticity/generated_outputs/hand_designed"
            gt_programs = sorted(os.listdir(program_dir))
            self.gt_programs = [open(f"{program_dir}/{p}", "r").read() for p in gt_programs]


        # Convert dtype string to torch dtype
        torch_dtype = torch.float16
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32

        # Check if model is a GPT model
        if "gpt" in model_name.lower():
            # Override with GPT-4.1 Nano
            self.model_name = "gpt-4.1-nano"
            # Use OpenAI API instead of vLLM
            import openai
            openai.api_key = openai_api_key
            self.client = openai.OpenAI(api_key=openai_api_key)
            self.use_openai = True
        else:
            self.use_openai = False
            # Load model using vLLM with optimized settings
            vllm_kwargs = {
                "model": model_name,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "dtype": torch.bfloat16,
                "trust_remote_code": True,
                "max_num_batched_tokens": 40000,
                # "max_model_len": max_model_len,
            }
            
            # Add quantization if specified
            if quantization:
                vllm_kwargs["quantization"] = quantization
                
            self.llm = LLM(**vllm_kwargs)
            self.sampling_params = SamplingParams(temperature=1.0, max_tokens=2000)
        
        # Keep transformers implementation (commented out)
        # self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.llm_model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto"
        # )
        # Original transformers implementation
        # # Load model and tokenizer from transformers
        # self.llm = pipeline("text-generation", model=self.model_name, device=self.device)
    
    def convert_state_to_text(self, state):
        text = ""
        text += f"The agents' inventory is {state['agent_inventory']}.\n"
        text += f"The agents' inventory colors are {state['agent_inventory_colors']}.\n"
        text += f"The agents' location is {state['agent_locations']}.\n"
        text += f"The block colors are {state['block_colors']}.\n"
        text += f"The block locations are {state['block_locations']}.\n"
        text += f"The wall locations are {state['wall_locations']}.\n"
        return text
    
    def convert_states_actions_to_text(self, states, actions):
        state_strings = []
        action_strings = []
        for i in range(actions.shape[0]):
            state = jax.tree.map(lambda x: x[i], states)
            state_string = self.convert_state_to_text(state)
            action = [self.action_to_name[int(a)] for a in actions[i]]
            state_strings.append(f"{i+1}. State: {state_string}.")
            action_string = f"{i+1}."
            for aid, a in enumerate(action):
                action_string += f" Agent {aid}'s Action: {a}, "
            action_strings.append(action_string)
        state_action_strings = [f"{s} {a}" for s, a in zip(state_strings[:-1], action_strings[:-1])]
        state_action_strings.append(state_strings[-1])
        return "\n-------\n".join(state_action_strings)
    
    def generate_high_level_description(self, state_action_text):
        """
        Generate a high-level description of the trajectory from the detailed state-action text.
        """
        prompt = f"""Below is a detailed description of an agent's trajectory in a grid world environment.
Please provide a high-level summary of the agent's behavior pattern, focusing on:
1. The agent's overall goal or strategy
2. How the agent responds to different environmental features (blocks, walls)
3. Any patterns in movement or interaction

Detailed trajectory:
{state_action_text}

High-level description:"""

        if self.use_openai:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        else:
            # Format for vLLM
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
    
    def summarize_agent_code(self, agent_code):
        prompt = f"""Below is an agent code that implements a finite state machine (FSM) for a grid world environment.
Please provide a high-level summary of the agent's behavior pattern, focusing on:
1. The agent's overall goal or strategy
2. How the agent responds to different environmental features (blocks, walls)
3. Any patterns in movement or interaction

Provide a very short summary (5 words or less)with as few words as possible. For instance, if the code describes an agent which moves right constantly, your summary should be "Move right". If they choose a random action, your summary should be "Random". If they alternate between moving up and down until they hit a wall, your summary should be "Up/down until wall".

Agent code:
{agent_code}

Your high-level 5 word summary:"""

        if self.use_openai:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=5
            )
            return response.choices[0].message.content
        else:
            # Format for vLLM
            sampling_params = SamplingParams(temperature=0.7, max_tokens=10)
            outputs = self.llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text
    

    def predict_action_with_bootstrap(self, states, actions, training=False, episode_id=0, max_hypotheses=20, 
                                         rejuvenation_threshold=-10, max_rejuvenation_attempts=5, top_k=0,
                                         return_compiled_agents: bool = False, return_all_time_log_prob_list: bool = False, doing_rejuvenation=False):
        """
        Predicts actions with bootstrapping for different numbers of hypotheses.
        Returns predictions for hypotheses counts from 1 to max_hypotheses.
        
        Args:
            states: The states of the environment
            actions: The actions taken in the environment
            training: Whether this is being called during training
            episode_id: The ID of the episode
            max_hypotheses: Maximum number of hypotheses to generate
            top_k: If > 0, only average over the top k most likely hypotheses.
                  If 0, average over all hypotheses (default behavior).
            return_compiled_agents: If True, return compiled agents, their probabilities, and codes for max_hypotheses.
        """
        if self.oracle:
            max_hypotheses = len(self.gt_programs)  # use all oracle programs
            
        episode_name = f"{self.dataset_name}_{episode_id}"
        state_action_text = self.convert_states_actions_to_text(states, actions)
        
        if self.two_stage and not self.oracle:
            # Generate high-level description first
            high_level_description = self.generate_high_level_description(state_action_text)

            if self.structured == "p2":
                # First stage: Generate high-level FSM description
                first_stage_prompt = f"{self.base_prompt}\n{high_level_description}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if self.use_openai:
                    messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                               {'role': 'user', 'content': first_stage_prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500  # Limit to 100 tokens for high-level FSM
                    )
                    high_level_fsm = response.choices[0].message.content
                else:
                    # Format for vLLM with limited token generation
                    if "llama" in self.model_name.lower():
                        formatted_prompt = ""
                        messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                                   {'role': 'user', 'content': first_stage_prompt}]
                        for msg in messages:
                            if msg["role"] == "system":
                                formatted_prompt += f"<|system|>\n{msg['content']}\n"
                            elif msg["role"] == "user":
                                formatted_prompt += f"<|user|>\n{msg['content']}\n"
                            elif msg["role"] == "assistant":
                                formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
                        formatted_prompt += "<|assistant|>\n"
                    else:
                        formatted_prompt = first_stage_prompt
                    
                    # Create special sampling params with limited tokens
                    fsm_sampling_params = SamplingParams(temperature=0.7, max_tokens=500)
                    outputs = self.llm.generate([formatted_prompt], fsm_sampling_params)
                    high_level_fsm = outputs[0].outputs[0].text
                    
                    # # save high_level_fsm to a file
                    # with open(f"high_level_fsm_grid.txt", "w") as f:
                    #     f.write(high_level_fsm)
                    # exit()
                # Second stage: Use the high-level FSM in the final prompt
                full_prompt = f"{self.base_prompt}\n{high_level_description}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\n{self.code_template}"
            else:
                # Original approach for other structured options
                full_prompt = f"{self.base_prompt}\n{high_level_description}\n{self.code_template}"
        elif not self.oracle and not self.two_stage:
            # Not using two-stage approach
            if self.structured == "p2":
                # First stage: Generate high-level FSM description using state_action_text
                first_stage_prompt = f"{self.base_prompt}\n{state_action_text}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if self.use_openai:
                    messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                               {'role': 'user', 'content': first_stage_prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500  # Limit to 100 tokens for high-level FSM
                    )
                    high_level_fsm = response.choices[0].message.content
                else:
                    # Format for vLLM with limited token generation
                    if "llama" in self.model_name.lower():
                        formatted_prompt = ""
                        messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                                   {'role': 'user', 'content': first_stage_prompt}]
                        for msg in messages:
                            if msg["role"] == "system":
                                formatted_prompt += f"<|system|>\n{msg['content']}\n"
                            elif msg["role"] == "user":
                                formatted_prompt += f"<|user|>\n{msg['content']}\n"
                            elif msg["role"] == "assistant":
                                formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
                        formatted_prompt += "<|assistant|>\n"
                    else:
                        formatted_prompt = first_stage_prompt
                    
                    # Create special sampling params with limited tokens
                    fsm_sampling_params = SamplingParams(temperature=0.7, max_tokens=500)
                    outputs = self.llm.generate([formatted_prompt], fsm_sampling_params)
                    high_level_fsm = outputs[0].outputs[0].text
                # # save high_level_fsm to a file
                # with open(f"high_level_fsm_grid.txt", "w") as f:
                #     f.write(high_level_fsm)
                # exit()
                # Second stage: Use the high-level FSM in the final prompt
                full_prompt = f"{self.base_prompt}\n{state_action_text}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\n{self.code_template}"
            else:
                # Use original approach
                full_prompt = f"{self.base_prompt}\n{state_action_text}\n{self.code_template}"

        framework = AgentExecutionFramework()
        num_agents = 1 if not self.group else 4
        num_blocks = states['block_locations'].shape[1]
        agents = []
        log_prob_hypothesis_list = []
        final_action_pred_list = []
        agent_codes = []
        
        # Generate max_hypotheses instead of self.num_hypothesis
        if not self.oracle:
            formatted_prompts = []
            for hypothesis_id in range(max_hypotheses):
                messages = [{"role": "system", "content": "You are a helpful assistant."}, {'role': 'user', 'content': full_prompt}]
                
                # Format messages for vLLM or OpenAI
                if self.use_openai:
                    formatted_prompt = messages
                elif "llama" in self.model_name.lower():
                    # Format for Llama models
                    formatted_prompt = ""
                    for msg in messages:
                        if msg["role"] == "system":
                            formatted_prompt += f"<|system|>\n{msg['content']}\n"
                        elif msg["role"] == "user":
                            formatted_prompt += f"<|user|>\n{msg['content']}\n"
                        elif msg["role"] == "assistant":
                            formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
                    formatted_prompt += "<|assistant|>\n"
                else:
                    # Generic chat format
                    formatted_prompt = full_prompt
                    
                formatted_prompts.append(formatted_prompt)
        
            # Generate responses for each hypothesis
            if self.use_openai:
                # Process each hypothesis individually with OpenAI API
                outputs = []
                for prompt in formatted_prompts:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=prompt,
                        temperature=1.0,
                        max_tokens=2000
                    )
                    outputs.append(response.choices[0].message.content)
            else:
                # Batch generate with vLLM
                vllm_outputs = self.llm.generate(formatted_prompts, self.sampling_params)
                outputs = [output.outputs[0].text for output in vllm_outputs]
        else:
            outputs = self.gt_programs
        
        agent_codes = []
        # Process each hypothesis
        all_time_all_hyp_log_prob_list = []
        for hypothesis_id, agent_code in enumerate(outputs):
            # Define generate_response and revise_response functions based on model type
            if self.use_openai:
                def generate_response():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=formatted_prompt,
                        temperature=1.0,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content

                def revise_response(response, error_message, rejuvenation_attempt=False):
                    # prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}\nORIGINAL TASK:\n{full_prompt}"
                    prompt = formatted_prompts[0]
                    if rejuvenation_attempt:
                        prompt = f"{prompt}\n Here's the code you make last time {response}. Return a new program that is different from the last one.\n"
                    if self.use_openai:
                        revised = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=1.0,
                            max_tokens=2000
                        )
                        return revised.choices[0].message.content
                    else:
                        outputs = self.llm.generate([prompt], self.sampling_params)
                        return outputs[0].outputs[0].text
            else:
                def generate_response():
                    outputs = self.llm.generate([formatted_prompt], self.sampling_params)
                    return outputs[0].outputs[0].text

                def revise_response(response, error_message, rejuvenation_attempt=False):
                    # prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}\nORIGINAL TASK:\n{full_prompt}"
                    prompt = formatted_prompts[0]
                    outputs = self.llm.generate([prompt], self.sampling_params)
                    return outputs[0].outputs[0].text

            agent = None
            error = None
            trial = 0
            num_trials = 2
            num_rejuvenation_attempts = 0

            while trial < num_trials and num_rejuvenation_attempts < max_rejuvenation_attempts:
                try:
                    agent = framework.compile_agent(agent_code, num_agents, num_blocks)

                    log_prob_hypothesis = 1e-6
                    all_time_log_prob_list = []
                    # p(script | states, actions) = p(action | states, script) * prior(script)
                    for timestep in range(actions.shape[0] - 1):  
                        state = jax.tree.map(lambda x: x[timestep], states)
                        if len(state['agent_locations']) == 1:
                            state['agent_id'] = 0
                        if not self.group:
                            gt_action = actions[timestep][0]
                            def loop_body():
                                proposed_action = framework.execute_agent(agent, state)

                                try:
                                    if type(proposed_action) == tuple and proposed_action in self.action_space:
                                        proposed_action = self.action_space.index(proposed_action)
                                    elif type(proposed_action) == str:
                                        proposed_action = proposed_action.lower()
                                        proposed_action = self.str_action_space.index(proposed_action)
                                    correct = float(proposed_action == gt_action)
                                except Exception as e:
                                    trial += 1
                                    full_traceback = traceback.format_exc()
                                    # print(full_traceback)
                                    if trial == num_trials:
                                        correct = 0
                                    else:
                                        agent_code = revise_response(agent_code, full_traceback)
                                        correct = loop_body()
                                return correct
                            correct = loop_body()
                            log_prob_hypothesis += correct 
                            all_time_log_prob_list.append(log_prob_hypothesis)
                        else:
                            gt_actions = actions[timestep]
                            proposed_actions = framework.execute_agent(agent, state)
                            for a in proposed_actions:
                                if a in self.action_space:
                                    a = self.action_space.index(a)
                                elif type(a) == str and a.lower() in self.str_action_space:
                                    a = self.str_action_space.index(a)
                                assert a in range(6), "an action in proposed_actions is not an integer in range(num_actions)"
                            for i in range(len(proposed_actions)):
                                if proposed_actions[i] in self.action_space:
                                    proposed_actions[i] = self.action_space.index(proposed_actions[i])
                                log_prob_hypothesis += (proposed_actions[i] == gt_actions[i])

                    final_state = jax.tree.map(lambda x: x[-1], states)
                    if len(final_state['agent_locations']) == 1:
                        final_state['agent_id'] = 0
                    final_action = framework.execute_agent(agent, final_state)
  
                    
                    try:
                        assert type(log_prob_hypothesis) == float, "log_prob_hypothesis is not a float"
                    except Exception as e:
                        breakpoint()
                    
                    if log_prob_hypothesis >= rejuvenation_threshold or not doing_rejuvenation:
                        agent_codes.append(agent_code)
                        agents.append(agent)
                        log_prob_hypothesis_list.append(log_prob_hypothesis)
                        all_time_all_hyp_log_prob_list.append(all_time_log_prob_list)
                        final_action_pred_list.append(final_action)  # time t
                        break
                    else:
                        num_rejuvenation_attempts += 1
                        trial = 0
                        agent_code = revise_response(agent_code, "Rejuvenation attempt", rejuvenation_attempt=True)
                        log_prob_hypothesis = 1e-6
                        continue
                except Exception as e:
                    trial += 1
                    full_traceback = traceback.format_exc()
                    # print(full_traceback)
                    if trial == num_trials:
                        print(f"Failed to compile hypothesis {hypothesis_id} after {num_trials} trials")
                        break
                    agent_code = revise_response(agent_code, full_traceback)
        
        if len(log_prob_hypothesis_list) == 0:
            if return_compiled_agents:
                return None, None, None
            return None
        
        if return_compiled_agents:
            # Return the compiled agents, their normalized probabilities, and codes for the full set of max_hypotheses (or top_k)
            # These lists (agents, log_prob_hypothesis_list, agent_codes) are already populated
            
            current_agents_list = agents
            try:
                current_log_probs_np = np.array(log_prob_hypothesis_list)
            except Exception as e:
                breakpoint()
                current_log_probs_np = np.array(log_prob_hypothesis_list)
                
            current_agent_codes_list = agent_codes # This should be the codes for successfully compiled agents

            if len(current_log_probs_np) == 0:
                return None, None, None

            # Normalize probabilities
            current_log_probs_np = current_log_probs_np - np.max(current_log_probs_np)
            current_probs_np = np.exp(current_log_probs_np)
            current_probs_np = current_probs_np / np.sum(current_probs_np)

            final_agents_to_return = current_agents_list
            final_probs_to_return = current_probs_np
            final_codes_to_return = current_agent_codes_list

            if top_k > 0 and top_k < len(current_probs_np):
                top_k_indices = np.argsort(current_probs_np)[-top_k:]
                
                final_agents_to_return = [current_agents_list[i] for i in top_k_indices]
                final_probs_to_return = current_probs_np[top_k_indices]
                final_codes_to_return = [current_agent_codes_list[i] for i in top_k_indices]
                
                # Renormalize the filtered probs
                final_probs_to_return = final_probs_to_return / np.sum(final_probs_to_return)
            
            if return_all_time_log_prob_list:
                all_time_all_hyp_log_prob_list = np.array(all_time_all_hyp_log_prob_list)
                all_time_all_hyp_log_prob_list = all_time_all_hyp_log_prob_list.T
                all_time_all_hyp_log_prob_list = all_time_all_hyp_log_prob_list / np.sum(all_time_all_hyp_log_prob_list, axis=1, keepdims=True) 
                return final_agents_to_return, list(final_probs_to_return), final_codes_to_return, all_time_all_hyp_log_prob_list
            else:
                return final_agents_to_return, list(final_probs_to_return), final_codes_to_return

        # Store results for different numbers of hypotheses
        bootstrap_results = []
        # Also store weighted program lengths
        weighted_program_lengths = []
        
        # For each number of hypotheses from 1 to max_hypotheses (or as many as we have)
        for n_hyp in range(1, min(len(log_prob_hypothesis_list) + 1, max_hypotheses + 1)):
            # Use only the first n_hyp hypotheses
            curr_log_probs = np.array(log_prob_hypothesis_list[:n_hyp])
            curr_final_preds = np.array(final_action_pred_list[:n_hyp])
            curr_agent_codes = agent_codes[:n_hyp]
            
            # normalize the log probs
            curr_log_probs = curr_log_probs - np.max(curr_log_probs)
            curr_log_probs = np.exp(curr_log_probs)
            curr_log_probs = curr_log_probs / np.sum(curr_log_probs)  # (n_hyp,)
            
            # If top_k is specified and valid, only use the top k hypotheses
            if top_k > 0 and top_k < n_hyp:
                # Get indices of top k hypotheses by log probability
                top_k_indices = np.argsort(curr_log_probs)[-top_k:]
                # Filter log probs and action predictions to only include top k
                filtered_log_probs = curr_log_probs[top_k_indices]
                # Renormalize the filtered log probs
                filtered_log_probs = filtered_log_probs / np.sum(filtered_log_probs)
                
                # Filter the final action predictions and agent codes
                filtered_action_preds = curr_final_preds[top_k_indices]
                filtered_agent_codes = [curr_agent_codes[i] for i in top_k_indices]
                
                # Use the filtered lists for the weighted average
                curr_log_probs = filtered_log_probs
                curr_final_preds = filtered_action_preds
                curr_agent_codes = filtered_agent_codes
            
            # Calculate weighted program length
            program_lengths = np.array([len(code) for code in curr_agent_codes])
            weighted_length = np.sum(curr_log_probs * program_lengths)
            weighted_program_lengths.append(weighted_length)
            
            if not self.group:
                breakpoint()
                curr_final_preds = curr_final_preds + 1e-8
                curr_final_preds = np.clip(curr_final_preds, 1e-8, 1)
                curr_final_preds = curr_final_preds / np.sum(curr_final_preds, axis=1, keepdims=True)
                
                res_pi = np.sum(curr_log_probs * curr_final_preds.T, axis=1)  # (num_actions,)
            else:
                breakpoint()
                curr_final_preds = curr_final_preds + 1e-8
                curr_final_preds = np.clip(curr_final_preds, 1e-8, 1)
                curr_final_preds = curr_final_preds / np.sum(curr_final_preds, axis=-1, keepdims=True)
                
                # Reshape curr_log_probs to (n_hyp, 1, 1) for broadcasting
                weights = curr_log_probs[:, np.newaxis, np.newaxis]
                # Multiply and sum across hypotheses in one operation
                res_pi = np.sum(weights * curr_final_preds, axis=0)  # (num_agents, num_actions)
            
            bootstrap_results.append((res_pi, weighted_length))
        
        return bootstrap_results

    def predict_action(self, states, actions, training=False, episode_id=0, top_k=0):
        """Original predict_action method that uses self.num_hypothesis
        
        Args:
            states: The states of the environment
            actions: The actions taken in the environment
            training: Whether this is being called during training
            episode_id: The ID of the episode
            top_k: If > 0, only average over the top k most likely hypotheses.
                  If 0, average over all hypotheses (default behavior).
        """
        # If bootstrapping is requested, use the bootstrap method with the specified number of hypotheses
        if hasattr(self, 'bootstrap') and self.bootstrap:
            bootstrap_results = self.predict_action_with_bootstrap(states, actions, training, episode_id, self.num_hypothesis, top_k)
            if bootstrap_results is None:
                return None
            # Return the result for the specified number of hypotheses
            return bootstrap_results[-1][0]  # Last element corresponds to self.num_hypothesis, and [0] gets the action prediction
        
        # Original implementation follows
        episode_name = f"{self.dataset_name}_{episode_id}"
        state_action_text = self.convert_states_actions_to_text(states, actions)
        
        if self.two_stage:
            # Generate high-level description first
            high_level_description = self.generate_high_level_description(state_action_text)
            
            if self.structured == "p2":
                # First stage: Generate high-level FSM description
                first_stage_prompt = f"{self.base_prompt}\n{high_level_description}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if self.use_openai:
                    messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                               {'role': 'user', 'content': first_stage_prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=100  # Limit to 100 tokens for high-level FSM
                    )
                    high_level_fsm = response.choices[0].message.content
                else:
                    # Format for vLLM with limited token generation
                    if "llama" in self.model_name.lower():
                        formatted_prompt = ""
                        messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                                   {'role': 'user', 'content': first_stage_prompt}]
                        for msg in messages:
                            if msg["role"] == "system":
                                formatted_prompt += f"<|system|>\n{msg['content']}\n"
                            elif msg["role"] == "user":
                                formatted_prompt += f"<|user|>\n{msg['content']}\n"
                            elif msg["role"] == "assistant":
                                formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
                        formatted_prompt += "<|assistant|>\n"
                    else:
                        formatted_prompt = first_stage_prompt
                    
                    # Create special sampling params with limited tokens
                    fsm_sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
                    outputs = self.llm.generate([formatted_prompt], fsm_sampling_params)
                    high_level_fsm = outputs[0].outputs[0].text
                
                # Second stage: Use the high-level FSM in the final prompt
                full_prompt = f"{self.base_prompt}\n{high_level_description}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\n{self.code_template}"
            else:
                # Original approach for other structured options
                full_prompt = f"{self.base_prompt}\n{high_level_description}\n{self.code_template}"
        else:
            # Not using two-stage approach
            if self.structured == "p2":
                # First stage: Generate high-level FSM description using state_action_text
                first_stage_prompt = f"{self.base_prompt}\n{state_action_text}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if self.use_openai:
                    messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                               {'role': 'user', 'content': first_stage_prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=100  # Limit to 100 tokens for high-level FSM
                    )
                    high_level_fsm = response.choices[0].message.content
                else:
                    # Format for vLLM with limited token generation
                    if "llama" in self.model_name.lower():
                        formatted_prompt = ""
                        messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                                   {'role': 'user', 'content': first_stage_prompt}]
                        for msg in messages:
                            if msg["role"] == "system":
                                formatted_prompt += f"<|system|>\n{msg['content']}\n"
                            elif msg["role"] == "user":
                                formatted_prompt += f"<|user|>\n{msg['content']}\n"
                            elif msg["role"] == "assistant":
                                formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
                        formatted_prompt += "<|assistant|>\n"
                    else:
                        formatted_prompt = first_stage_prompt
                    
                    # Create special sampling params with limited tokens
                    fsm_sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
                    outputs = self.llm.generate([formatted_prompt], fsm_sampling_params)
                    high_level_fsm = outputs[0].outputs[0].text
                
                # Second stage: Use the high-level FSM in the final prompt
                full_prompt = f"{self.base_prompt}\n{state_action_text}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\n{self.code_template}"
            else:
                # Use original approach
                full_prompt = f"{self.base_prompt}\n{state_action_text}\n{self.code_template}"
        
        framework = AgentExecutionFramework()
        num_agents = 1 if not self.group else 4
        num_blocks = states['block_locations'].shape[1]
        agents = []
        log_prob_hypothesis_list = []
        final_action_pred_list = []
        
        # Prepare all prompts for batch inference
        formatted_prompts = []
        for hypothesis_id in range(self.num_hypothesis):
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {'role': 'user', 'content': full_prompt}]
            
            # Format messages for vLLM or OpenAI
            if self.use_openai:
                formatted_prompt = messages
            elif "llama" in self.model_name.lower():
                # Format for Llama models
                formatted_prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        formatted_prompt += f"<|system|>\n{msg['content']}\n"
                    elif msg["role"] == "user":
                        formatted_prompt += f"<|user|>\n{msg['content']}\n"
                    elif msg["role"] == "assistant":
                        formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
                formatted_prompt += "<|assistant|>\n"
            else:
                # Generic chat format
                formatted_prompt = full_prompt
                
            formatted_prompts.append(formatted_prompt)
        
        # Generate responses for each hypothesis
        if self.use_openai:
            # Process each hypothesis individually with OpenAI API
            outputs = []
            for prompt in formatted_prompts:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    temperature=1.0,
                    max_tokens=2000
                )
                outputs.append(response.choices[0].message.content)
        else:
            # Batch generate with vLLM
            vllm_outputs = self.llm.generate(formatted_prompts, self.sampling_params)
            outputs = [output.outputs[0].text for output in vllm_outputs]
        
        agent_codes = []
        # Process each hypothesis
        for hypothesis_id, agent_code in enumerate(outputs):
            # Define generate_response and revise_response functions based on model type
            if self.use_openai:
                def generate_response():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=formatted_prompt,
                        temperature=1.0,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content

                def revise_response(response, error_message):
                    prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}\nORIGINAL TASK:\n{full_prompt}"
                    if self.use_openai:
                        revised = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=1.0,
                            max_tokens=2000
                        )
                        return revised.choices[0].message.content
                    else:
                        outputs = self.llm.generate([prompt], self.sampling_params)
                        return outputs[0].outputs[0].text
            else:
                # vLLM implementation for generate_response
                def generate_response():
                    outputs = self.llm.generate([formatted_prompt], self.sampling_params)
                    return outputs[0].outputs[0].text

                # vLLM implementation for revise_response
                def revise_response(response, error_message):
                    prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}\nORIGINAL TASK:\n{full_prompt}"
                    outputs = self.llm.generate([prompt], self.sampling_params)
                    return outputs[0].outputs[0].text

            agent = None
            error = None
            trial = 0
            num_trials = 2
            while trial < num_trials:
                try:
                    agent = framework.compile_agent(agent_code, num_agents, num_blocks)

                    log_prob_hypothesis = 0
                    # p(script | states, actions) = p(action | states, script) * prior(script)
                    for timestep in range(actions.shape[0] - 1):  
                        state = jax.tree.map(lambda x: x[timestep], states)
                        if len(state['agent_locations']) == 1:
                            state['agent_id'] = 0
                        if not self.group:
                            gt_action = actions[timestep][0]
                            proposed_action = framework.execute_agent(agent, state)
                            log_prob_hypothesis += np.log(np.clip(proposed_action[gt_action], 1e-8, 1))
                        else:
                            gt_actions = actions[timestep]
                            proposed_actions = framework.execute_agent(agent, state)
                            for a in proposed_actions:
                                assert a in range(6), "an action in proposed_actions is not an integer in range(num_actions)"
                            for i in range(len(proposed_actions)):
                                proposed_pi = np.exp(np.array(proposed_pis[i]) + 1e-8) / np.sum(np.exp(np.array(proposed_pis[i]) + 1e-8))
                                log_prob_hypothesis += np.log(np.clip(proposed_pi[gt_actions[i]], 1e-8, 1))
                    
                    final_state = jax.tree.map(lambda x: x[-1], states)
                    if len(final_state['agent_locations']) == 1:
                        final_state['agent_id'] = 0
                    final_action, final_pi = framework.execute_agent(agent, final_state)
                    agent_codes.append(agent_code)
                    agents.append(agent)
                    log_prob_hypothesis_list.append(log_prob_hypothesis)
                    final_action_pred_list.append(final_pi)  # time t
                    break
                except Exception as e:
                    print(f"Error compiling agent {hypothesis_id}: {e}")
                    trial += 1
                    full_traceback = traceback.format_exc()
                    # print(full_traceback)
                    if trial == num_trials:
                        print(f"Failed to compile hypothesis {hypothesis_id} after {num_trials} trials")
                        print(full_traceback)
                        breakpoint()
                        break
                    agent_code = revise_response(agent_code, full_traceback)
            if agent is None:
                continue
        
        if len(log_prob_hypothesis_list) == 0:
            return None

        # normalize the log probs
        log_prob_hypothesis_list = np.array(log_prob_hypothesis_list)
        log_prob_hypothesis_list = log_prob_hypothesis_list - np.max(log_prob_hypothesis_list)
        log_prob_hypothesis_list = np.exp(log_prob_hypothesis_list)
        log_prob_hypothesis_list = log_prob_hypothesis_list / np.sum(log_prob_hypothesis_list)  # (num_hypothesis,)

        # Calculate weighted program length
        program_lengths = np.array([len(code) for code in agent_codes])
        weighted_length = np.sum(log_prob_hypothesis_list * program_lengths)
        self.weighted_program_length = weighted_length  # Store for later access

        # If top_k is specified and valid, only use the top k hypotheses
        if top_k > 0 and top_k < len(log_prob_hypothesis_list):
            # Get indices of top k hypotheses by log probability
            top_k_indices = np.argsort(log_prob_hypothesis_list)[-top_k:]
            # Filter log probs and action predictions to only include top k
            filtered_log_probs = log_prob_hypothesis_list[top_k_indices]
            # Renormalize the filtered log probs
            filtered_log_probs = filtered_log_probs / np.sum(filtered_log_probs)
            
            # Filter the final action predictions
            filtered_action_preds = [final_action_pred_list[i] for i in top_k_indices]
            
            # Use the filtered lists for the weighted average
            log_prob_hypothesis_list = filtered_log_probs
            final_action_pred_list = filtered_action_preds

        if not self.group:
            final_action_pred_list = np.array(final_action_pred_list) + 1e-8
            final_action_pred_list = np.clip(final_action_pred_list, 1e-8, 1)
            final_action_pred_list = final_action_pred_list / np.sum(final_action_pred_list, axis=1, keepdims=True)

            res_pi = np.sum(log_prob_hypothesis_list * final_action_pred_list.T, axis=1)  # (num_actions,)
        else:
            final_action_pred_list = np.array(final_action_pred_list) + 1e-8
            final_action_pred_list = np.clip(final_action_pred_list, 1e-8, 1)
            final_action_pred_list = final_action_pred_list / np.sum(final_action_pred_list, axis=-1, keepdims=True)   # (num_hypothesis, num_agents, num_actions)
            # Reshape log_prob_hypothesis_list to (num_hypothesis, 1, 1) for broadcasting
            weights = log_prob_hypothesis_list[:, np.newaxis, np.newaxis]
            # Multiply and sum across hypotheses in one operation
            res_pi = np.sum(weights * final_action_pred_list, axis=0)  # (num_agents, num_actions)
            
        return res_pi
            
        
        

