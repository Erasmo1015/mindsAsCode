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

openai_api_key = os.environ.get("OPENAI_API_KEY", "")

class FSMReasoner:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size: int = 1, device: str = "cuda", 
                 dtype: str = "float16", gpu_memory_utilization: float = 0.55, num_hypothesis: int = 4,
                 max_model_len: int = 2048, quantization: str = None, group: bool = False, two_stage: bool = False,
                 structured: str = "False"):
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

        self.group = group
        if not group:
            self.dataset_name = "single_agent_dataset"
        else:
            self.dataset_name = "group_agent_dataset"
        if not group:
            # self.dataset_name = "single_agent_dataset"
            prompt_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/infer_partnr_single.txt"
            if structured == "p1":
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_partnr_single_code_template.txt"
            elif structured == "p2":
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_partnr_p2_single_code_template.txt"
                self.first_stage_prompt = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_partnr_p1_single_code_template.txt"
            else:
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/single_partnr_code_template.txt"
        else:
            # self.dataset_name = "group_agent_dataset"
            prompt_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/infer_partnr_single.txt"
            if structured == "p1":
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_partnr_group_code_template.txt"
            elif structured == "p2":
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_p2_partnr_group_code_template.txt"
                self.first_stage_prompt = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/structured_p1_partnr_group_code_template.txt"
            else:
                code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/single_partnr_code_template.txt"
        refinement_1_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_1.txt"
        refinement_2_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_2.txt"
        refinement_3_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_3.txt"

        self.base_prompt = open(prompt_path, "r").read()
        self.code_template = open(code_template_path, "r").read()
        self.refinement_1 = open(refinement_1_path, "r").read()
        self.refinement_2 = open(refinement_2_path, "r").read()
        self.refinement_3 = open(refinement_3_path, "r").read()
        
        if structured == "p2":
            self.first_stage_prompt = open(self.first_stage_prompt, "r").read()

        # Convert dtype string to torch dtype
        torch_dtype = torch.float16
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32

        # Check if model is a GPT model
        if "gpt" in model_name.lower():
            # Override with specified GPT model
            self.model_name = model_name
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
        text += f"Scene Graph: {state['scene_graph']}\n"
        text += f"Agent State: {state['agent_state']}\n"
        return text
    
    def convert_states_actions_to_text(self, states, actions):
        state_strings = []
        action_strings = []
        tools = set()
        tool_descriptions = ""
        for i in range(len(actions)):
            state = states[i]
            action = actions[i]
            state_string = self.convert_state_to_text(state)
            state_strings.append(state_string)
            action_string = f"{i+1}. Agent 0 Action: {action}"
            action_strings.append(action_string)
            tools = tools.union(set(state['tool_list']))
            tool_descriptions = state['tool_descriptions']
        state_action_strings = [f"{s} {a} \n ----------\n" for s, a in zip(state_strings[:-2], action_strings[:-2])]
        state_action_strings.append(state_strings[-1])
        return "\n-------\n".join(state_action_strings), tools, tool_descriptions
    
    def generate_high_level_description(self, state_action_text):
        """
        Generate a high-level description of the trajectory from the detailed state-action text.
        """
        prompt = f"""Below is a detailed description of an agent's trajectory in an environment.
Please provide a high-level summary of the agent's behavior pattern, focusing on:
1. The agent's overall goal or strategy
2. How the agent responds to different environmental features
3. Any patterns in movement or interaction

Detailed trajectory:
{state_action_text}

High-level description:"""

        if hasattr(self, 'use_openai') and self.use_openai:
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

    def predict_action_with_bootstrap(self, states, actions, training=False, episode_id=0, max_hypotheses=20, rejuvenation_threshold=2, max_rejuvenation_attempts=2, top_k=0, use_rejuvenation=False):
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
        """
        episode_name = f"{self.dataset_name}_{episode_id}"
        state_action_text, tools, tool_descriptions = self.convert_states_actions_to_text(states, actions)
        
        if self.two_stage:
            # Generate high-level description first
            high_level_description = self.generate_high_level_description(state_action_text)
            
            if self.structured == "p2":
                # First stage: Generate high-level FSM description
                first_stage_prompt = f"{self.base_prompt}\n{high_level_description}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if hasattr(self, 'use_openai') and self.use_openai:
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
                full_prompt = f"{self.base_prompt}\n{high_level_description}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
            else:
                # Original approach for other structured options
                full_prompt = f"{self.base_prompt}\n{high_level_description}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
        else:
            # Not using two-stage approach
            if self.structured == "p2":
                # First stage: Generate high-level FSM description using state_action_text
                first_stage_prompt = f"{self.base_prompt}\n{state_action_text}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if hasattr(self, 'use_openai') and self.use_openai:
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
                full_prompt = f"{self.base_prompt}\n{state_action_text}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
            else:
                # Use original approach
                full_prompt = f"{self.base_prompt}\n{state_action_text}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"

        framework = AgentExecutionFramework()
        num_agents = 1 if not self.group else 4
        agents = []
        log_prob_hypothesis_list = []
        final_action_pred_list = []
        agent_codes = []  # Add this to track agent codes
        
        # Generate max_hypotheses instead of self.num_hypothesis
        formatted_prompts = []
        for hypothesis_id in range(max_hypotheses):
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {'role': 'user', 'content': full_prompt}]
            
            # Format messages for vLLM or OpenAI
            if hasattr(self, 'use_openai') and self.use_openai:
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
        if hasattr(self, 'use_openai') and self.use_openai:
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
        
        # Process each hypothesis
        for hypothesis_id, agent_code in enumerate(outputs):
            # Define generate_response and revise_response functions based on model type
            if hasattr(self, 'use_openai') and self.use_openai:
                def generate_response(prompt):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content

                def revise_response(response, error_message, prompt,rejuvenation_attempt=False):
                    # prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}"
                    if rejuvenation_attempt:
                        prompt = f"{prompt}\n Here's the code you make last time {response}. Return a new program that is different from the last one.\n"
                    revised = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        max_tokens=2000
                    )
                    return revised.choices[0].message.content
            else:
                # vLLM implementation for generate_response
                def generate_response(prompt):
                    outputs = self.llm.generate([prompt], self.sampling_params)
                    return outputs[0].outputs[0].text

                # vLLM implementation for revise_response
                def revise_response(response, error_message, prompt, rejuvenation_attempt=False):
                    # prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}"
                    if rejuvenation_attempt:
                        prompt = f"{prompt}\n Here's the code you make last time {response}. Return a new program that is different from the last one.\n"
                    outputs = self.llm.generate([prompt], self.sampling_params)
                    return outputs[0].outputs[0].text
            
            agent = None
            error = None
            trial = 0
            num_trials = 2
            rejuvenation_attempts = 0
            while trial < num_trials and rejuvenation_attempts < max_rejuvenation_attempts:
                try:
                    agent = framework.compile_agent(agent_code, num_agents)

                    log_prob_hypothesis = 1e-6
                    # p(script | states, actions) = p(action | states, script) * prior(script)
                    for timestep in range(len(actions) - 2):  
                        state = states[timestep]

                        gt_actions = actions[timestep]
                        proposed_actions = framework.execute_agent(agent, state)
                        
                        for i in range(1):  # just take tool use for now
                            pa = proposed_actions[i]
                            ga = gt_actions[i]
                            log_prob_hypothesis += (ga.lower() == pa.lower())  # did we get the sub part of the action right?
                    
                    final_state = states[-2]
                    final_action = framework.execute_agent(agent, final_state)

                    if log_prob_hypothesis < rejuvenation_threshold and use_rejuvenation:
                        rejuvenation_attempts += 1
                        agent_code = revise_response(agent_code, "Rejuvenation attempt", formatted_prompts[hypothesis_id], rejuvenation_attempt=True)
                        trial = 0
                        log_prob_hypothesis = 1e-6
                    else:
                        agent_codes.append(agent_code)  # Store the agent code
                        agents.append(agent)
                        log_prob_hypothesis_list.append(log_prob_hypothesis)
                        final_action_pred_list.append(final_action)  # time t
                    break
                except Exception as e:
                    trial += 1
                    full_traceback = traceback.format_exc()
                    # print(full_traceback)
                    if trial == num_trials:
                        break
                    agent_code = revise_response(agent_code, full_traceback, formatted_prompts[hypothesis_id], rejuvenation_attempt=False)
        
        if len(log_prob_hypothesis_list) == 0:
            return None
        
        # Store results for different numbers of hypotheses
        bootstrap_results = []
        # Also store weighted program lengths
        weighted_program_lengths = []
        
        # For each number of hypotheses from 1 to max_hypotheses (or as many as we have)
        for n_hyp in range(1, max_hypotheses + 1):

            actual_n_hyp = min(len(log_prob_hypothesis_list), n_hyp)
            # Use only the first n_hyp hypotheses
            curr_log_probs = np.array(log_prob_hypothesis_list[:actual_n_hyp])
            curr_final_preds = final_action_pred_list[:actual_n_hyp]
            curr_agent_codes = agent_codes[:actual_n_hyp]  # Get current subset of agent codes
            
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
                filtered_action_preds = [curr_final_preds[i] for i in top_k_indices]
                filtered_agent_codes = [curr_agent_codes[i] for i in top_k_indices]  # Filter agent codes too
                
                # Use the filtered lists for the weighted average
                curr_log_probs = filtered_log_probs
                curr_final_preds = filtered_action_preds
                curr_agent_codes = filtered_agent_codes
            
            # Calculate weighted program length
            program_lengths = np.array([len(code) for code in curr_agent_codes])
            weighted_length = np.sum(curr_log_probs * program_lengths)
            weighted_program_lengths.append(weighted_length)
            
            # sample index from curr_log_probs
            sampled_index = np.random.choice(range(len(curr_log_probs)), p=curr_log_probs)
            final_action_pred = curr_final_preds[sampled_index]
            
            bootstrap_results.append((final_action_pred, weighted_length))
        
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
        state_action_text, tools, tool_descriptions = self.convert_states_actions_to_text(states, actions)
        
        if self.two_stage:
            # Generate high-level description first
            high_level_description = self.generate_high_level_description(state_action_text)
            
            if self.structured == "p2":
                # First stage: Generate high-level FSM description
                first_stage_prompt = f"{self.base_prompt}\n{high_level_description}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if hasattr(self, 'use_openai') and self.use_openai:
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
                full_prompt = f"{self.base_prompt}\n{high_level_description}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
            else:
                # Original approach for other structured options
                full_prompt = f"{self.base_prompt}\n{high_level_description}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
        else:
            # Not using two-stage approach
            if self.structured == "p2":
                # First stage: Generate high-level FSM description using state_action_text
                first_stage_prompt = f"{self.base_prompt}\n{state_action_text}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if hasattr(self, 'use_openai') and self.use_openai:
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
                full_prompt = f"{self.base_prompt}\n{state_action_text}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
            else:
                # Use original approach
                full_prompt = f"{self.base_prompt}\n{state_action_text}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"

        framework = AgentExecutionFramework()
        num_agents = 1 if not self.group else 4
        agents = []
        log_prob_hypothesis_list = []
        final_action_pred_list = []
        
        # Prepare all prompts for batch inference
        formatted_prompts = []
        for hypothesis_id in range(self.num_hypothesis):
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {'role': 'user', 'content': full_prompt}]
            
            # Format messages for vLLM or OpenAI
            if hasattr(self, 'use_openai') and self.use_openai:
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
        if hasattr(self, 'use_openai') and self.use_openai:
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
        
        # Process each hypothesis
        for hypothesis_id, agent_code in enumerate(outputs):
            # Define generate_response and revise_response functions based on model type
            if hasattr(self, 'use_openai') and self.use_openai:
                def generate_response(prompt):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content

                def revise_response(response, error_message):
                    prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}"
                    revised = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        max_tokens=2000
                    )
                    return revised.choices[0].message.content
            else:
                # vLLM implementation for generate_response
                def generate_response(prompt):
                    outputs = self.llm.generate([prompt], self.sampling_params)
                    return outputs[0].outputs[0].text

                # vLLM implementation for revise_response
                def revise_response(response, error_message):
                    prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}"
                    outputs = self.llm.generate([prompt], self.sampling_params)
                    return outputs[0].outputs[0].text
            
            agent = None
            error = None
            trial = 0
            num_trials = 2
            while trial < num_trials:
                try:
                    agent = framework.compile_agent(agent_code, num_agents)

                    log_prob_hypothesis = 0
                    # p(script | states, actions) = p(action | states, script) * prior(script)
                    for timestep in range(len(actions) - 2):  
                        state = states[timestep]

                        gt_actions = actions[timestep]
                        proposed_actions = framework.execute_agent(agent, state)
                        
                        for i in range(1):  # just take tool use for now
                            pa = proposed_actions[i]
                            ga = gt_actions[i]
                            log_prob_hypothesis += (ga == pa)  # did we get the sub part of the action right?
                    
                    final_state = states[-2]
                    final_action = framework.execute_agent(agent, final_state)

                    agents.append(agent)
                    log_prob_hypothesis_list.append(log_prob_hypothesis)
                    final_action_pred_list.append(final_action)  # time t
                    break
                except Exception as e:
                    trial += 1
                    full_traceback = traceback.format_exc()
                    if trial == num_trials:
                        break
                    agent_code = revise_response(agent_code, full_traceback)
        
        if len(log_prob_hypothesis_list) == 0:
            return None

        # normalize the log probs
        log_prob_hypothesis_list = np.array(log_prob_hypothesis_list)
        log_prob_hypothesis_list = log_prob_hypothesis_list - np.max(log_prob_hypothesis_list)
        log_prob_hypothesis_list = np.exp(log_prob_hypothesis_list)
        log_prob_hypothesis_list = log_prob_hypothesis_list / np.sum(log_prob_hypothesis_list)  # (num_hypothesis,)

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

        # sample index from log_prob_hypothesis_list
        sampled_index = np.random.choice(range(len(log_prob_hypothesis_list)), p=log_prob_hypothesis_list)
        final_action_pred = final_action_pred_list[sampled_index]

        return final_action_pred

    def predict_action_with_rejuvenation(self, states, actions, training=False, episode_id=0, max_hypotheses=20, 
                                        rejuvenation_threshold=-10, max_rejuvenation_attempts=5, top_k=0):
        """
        Predicts actions with rejuvenation, discarding low-quality hypotheses and generating new ones.
        
        Args:
            states: The states of the environment
            actions: The actions taken in the environment
            training: Whether this is being called during training
            episode_id: The ID of the episode
            max_hypotheses: Maximum number of hypotheses to generate
            rejuvenation_threshold: Log probability threshold below which to rejuvenate hypotheses
            max_rejuvenation_attempts: Maximum number of rejuvenation attempts
            top_k: If > 0, only average over the top k most likely hypotheses.
                  If 0, average over all hypotheses (default behavior).
        """
        episode_name = f"{self.dataset_name}_{episode_id}"
        state_action_text, tools, tool_descriptions = self.convert_states_actions_to_text(states, actions)
        
        if self.two_stage:
            # Generate high-level description first
            high_level_description = self.generate_high_level_description(state_action_text)
            
            if self.structured == "p2":
                # First stage: Generate high-level FSM description
                first_stage_prompt = f"{self.base_prompt}\n{high_level_description}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if hasattr(self, 'use_openai') and self.use_openai:
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
                full_prompt = f"{self.base_prompt}\n{high_level_description}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
            else:
                # Original approach for other structured options
                full_prompt = f"{self.base_prompt}\n{high_level_description}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
        else:
            # Not using two-stage approach
            if self.structured == "p2":
                # First stage: Generate high-level FSM description using state_action_text
                first_stage_prompt = f"{self.base_prompt}\n{state_action_text}\n{self.first_stage_prompt}"
                
                # Format for the appropriate model with limited token generation
                if hasattr(self, 'use_openai') and self.use_openai:
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
                full_prompt = f"{self.base_prompt}\n{state_action_text}\nHIGH LEVEL FSM TO IMPLEMENT IN CODE: {high_level_fsm}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"
            else:
                # Use original approach
                full_prompt = f"{self.base_prompt}\n{state_action_text}\nThe available tools are: {tools}\nHere is a description of how each tool is used, which is important to implement the FSM correctly: {tool_descriptions}\n{self.code_template}"

        framework = AgentExecutionFramework()
        num_agents = 1 if not self.group else 4
        agents = []
        log_prob_hypothesis_list = []
        final_action_pred_list = []
        agent_codes = []  # Track agent codes
        
        # Format the prompt for the model
        if hasattr(self, 'use_openai') and self.use_openai:
            formatted_prompt = [{"role": "system", "content": "You are a helpful assistant."}, 
                               {'role': 'user', 'content': full_prompt}]
        elif "llama" in self.model_name.lower():
            formatted_prompt = ""
            messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                       {'role': 'user', 'content': full_prompt}]
            for msg in messages:
                if msg["role"] == "system":
                    formatted_prompt += f"<|system|>\n{msg['content']}\n"
                elif msg["role"] == "user":
                    formatted_prompt += f"<|user|>\n{msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
            formatted_prompt += "<|assistant|>\n"
        else:
            formatted_prompt = full_prompt
        
        # Define helper functions for generating and revising responses
        if hasattr(self, 'use_openai') and self.use_openai:
            def generate_response():
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_prompt,
                    temperature=1.0,
                    max_tokens=2000
                )
                return response.choices[0].message.content

            def revise_response(response, error_message, prompt):
                # prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}"
                revised = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                    max_tokens=2000
                )
                return revised.choices[0].message.content
        else:
            def generate_response():
                outputs = self.llm.generate([formatted_prompt], self.sampling_params)
                return outputs[0].outputs[0].text

            def revise_response(response, error_message, prompt):
                # prompt = f"{self.refinement_1}\n{response}\n{self.refinement_2}\n{error_message}\n{self.refinement_3}"
                outputs = self.llm.generate([prompt], self.sampling_params)
                return outputs[0].outputs[0].text
        
        # Generate hypotheses with rejuvenation
        hypothesis_id = 0
        rejuvenation_attempts = 0
        
        while hypothesis_id < max_hypotheses and rejuvenation_attempts < max_rejuvenation_attempts:
            # Generate a new hypothesis
            agent_code = generate_response()
            
            # Try to compile and evaluate the agent
            agent = None
            trial = 0
            num_trials = 2
            
            while trial < num_trials:
                try:
                    agent = framework.compile_agent(agent_code, num_agents)

                    # Calculate log probability of this hypothesis
                    log_prob_hypothesis = 0
                    # p(script | states, actions) = p(action | states, script) * prior(script)
                    for timestep in range(len(actions) - 2):  
                        state = states[timestep]

                        gt_actions = actions[timestep]
                        proposed_actions = framework.execute_agent(agent, state)
                        
                        for i in range(1):  # just take tool use for now
                            # breakpoint()
                            pa = proposed_actions[i]
                            ga = gt_actions[i]
                            log_prob_hypothesis += (ga.lower() == pa.lower())  # did we get the sub part of the action right?
                    
                    # Get final prediction
                    final_state = states[-2]
                    final_action = framework.execute_agent(agent, final_state)
                    
                    # Check if this hypothesis is good enough or needs rejuvenation
                    if log_prob_hypothesis < rejuvenation_threshold and hypothesis_id >= max_hypotheses // 2:
                        # This hypothesis is poor quality and we have enough hypotheses to be selective
                        # Try to generate a better one
                        rejuvenation_attempts += 1
                        print(f"Rejuvenating hypothesis {hypothesis_id}: log_prob = {log_prob_hypothesis:.4f}")
                        break  # Skip this hypothesis and try a new one
                    
                    # Accept this hypothesis
                    agent_codes.append(agent_code)
                    agents.append(agent)
                    log_prob_hypothesis_list.append(log_prob_hypothesis)
                    final_action_pred_list.append(final_action)
                    hypothesis_id += 1
                    break
                except Exception as e:
                    trial += 1
                    full_traceback = traceback.format_exc()
                    # print(full_tsraceback)
                    if trial == num_trials:
                        break
                    agent_code = revise_response(agent_code, full_traceback)
        
        if len(log_prob_hypothesis_list) == 0:
            return None
        
        # Process results for different numbers of hypotheses
        bootstrap_results = []
        # Also store weighted program lengths
        weighted_program_lengths = []
        
        # For each number of hypotheses from 1 to the number we generated
        for n_hyp in range(1, len(log_prob_hypothesis_list) + 1):
            # Use only the first n_hyp hypotheses
            curr_log_probs = np.array(log_prob_hypothesis_list[:n_hyp])
            curr_final_preds = final_action_pred_list[:n_hyp]
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
                filtered_action_preds = [curr_final_preds[i] for i in top_k_indices]
                filtered_agent_codes = [curr_agent_codes[i] for i in top_k_indices]
                
                # Use the filtered lists for the weighted average
                curr_log_probs = filtered_log_probs
                curr_final_preds = filtered_action_preds
                curr_agent_codes = filtered_agent_codes
            
            # Calculate weighted program length
            program_lengths = np.array([len(code) for code in curr_agent_codes])
            weighted_length = np.sum(curr_log_probs * program_lengths)
            weighted_program_lengths.append(weighted_length)
            
            # sample index from curr_log_probs
            sampled_index = np.random.choice(range(len(curr_log_probs)), p=curr_log_probs)
            final_action_pred = curr_final_preds[sampled_index]
            
            bootstrap_results.append((final_action_pred, weighted_length))
        
        return bootstrap_results
            
        
        

