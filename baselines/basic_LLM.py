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

class NaiveLLMReasoner:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size: int = 1, device: str = "cuda", 
                 dtype: str = "float16", gpu_memory_utilization: float = 0.55, num_hypothesis: int = 4,
                 max_model_len: int = 2048, quantization: str = None, group: bool = False, partnr: bool = False):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.device = device
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_hypothesis = num_hypothesis
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.action_to_name = {
            0: "stay",
            1: "right",
            2: "left",
            3: "down",
            4: "up",
            5: "interact"
        }
        self.name_to_action = {v: k for k, v in self.action_to_name.items()}

        self.group = group
        if partnr:
            prompt_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/basic_llm_partnr.txt"
        else:
            prompt_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/basic_llm.txt"
        if group:
            self.dataset_name = "group_agent_dataset"
        else:
            self.dataset_name = "single_agent_dataset"

        self.base_prompt = open(prompt_path, "r").read()
        self.partnr = partnr


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
            self.sampling_params = SamplingParams(temperature=0.75, max_tokens=10)
        
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
    
    def grid_convert_state_to_text(self, state):
        text = ""
        text += f"The agents' inventory is {state['agent_inventory']}.\n"
        text += f"The agents' inventory colors are {state['agent_inventory_colors']}.\n"
        text += f"The agents' location is {state['agent_locations']}.\n"
        text += f"The block colors are {state['block_colors']}.\n"
        text += f"The block locations are {state['block_locations']}.\n"
        text += f"The wall locations are {state['wall_locations']}.\n"
        return text
    
    def grid_convert_states_actions_to_text(self, states, actions):
        state_strings = []
        action_strings = []
        for i in range(actions.shape[0]):  # (num_timesteps, num_agents)
            state = jax.tree.map(lambda x: x[i], states)
            state_string = self.grid_convert_state_to_text(state)
            action = [self.action_to_name[int(a)] for a in actions[i]]
            state_strings.append(f"{i+1}. State: {state_string}.")
            action_string = f"{i+1}."
            for aid, a in enumerate(action):    # (num_agents,)
                action_string += f" Agent {aid}'s Action: {a}, "
            action_strings.append(action_string)
        state_action_strings = [f"{s} {a}" for s, a in zip(state_strings[:-1], action_strings[:-1])]
        state_action_strings.append(state_strings[-1])
        return "\n-------\n".join(state_action_strings)
    
    def partnr_convert_state_to_text(self, state):
        text = ""
        text += f"Scene Graph: {state['scene_graph']}\n"
        text += f"Agent State: {state['agent_state']}\n"
        return text
    
    def partnr_convert_states_actions_to_text(self, states, actions):
        state_strings = []
        action_strings = []
        tools = set()
        tool_descriptions = ""
        for i in range(len(actions)):
            state = states[i]
            action = actions[i]
            state_string = self.partnr_convert_state_to_text(state)
            state_strings.append(state_string)
            action_string = f"{i+1}. Agent 0 Action: {action}"
            action_strings.append(action_string)
            tools = tools.union(set(state['tool_list']))
            tool_descriptions = state['tool_descriptions']
        state_action_strings = [f"{s} {a}" for s, a in zip(state_strings[:-2], action_strings[:-2])]
        state_action_strings.append(state_strings[-2])
        return "\n-------\n".join(state_action_strings), tools, tool_descriptions
    
    def predict_action(self, states, actions, training=False, episode_id=0, agent_id=0, timestep=None):
        episode_name = f"{self.dataset_name}_{episode_id}"
        if self.partnr:
            state_action_text, tools, tool_descriptions = self.partnr_convert_states_actions_to_text(states, actions)
        else:
            state_action_text = self.grid_convert_states_actions_to_text(states, actions)

        if timestep is not None:
            # prompt = f"{self.base_prompt}\n{state_action_text}\nWhat action will agent {agent_id} take at timestep {timestep + 1}?"  
            prompt = f"{state_action_text}\nWhat action will agent {agent_id} take at timestep {timestep + 1}?"  
        else:
            # prompt = f"{self.base_prompt}\n{state_action_text}\nWhat action will agent {agent_id} take next?"
            prompt = f"{state_action_text}\nWhat action will agent {agent_id} take next?"  
        if not self.partnr:
            prompt += f" Choose from the following options: {self.action_to_name.values()}. Your answer should be in the following format: \nAction: <action>\n Your answer:"
        else:
            prompt += f" Here is a description of what each tool is and how to use them: Tools: {tools}\n Tool Descriptions: {tool_descriptions}. What tool will agent {agent_id} use next? Your answer should be in the following format: \nAction: <tool_name>\n For example, if the tool is 'clean', your answer should be: \nAction: clean\n If the tool is 'wait', your answer should be: \nAction: wait\n Only respond with the tool name, not any other text. Your answer:"
        full_prompt = prompt


        log_prob_hypothesis_list = []
        final_action_pred_list = []
        
        # Prepare all prompts for batch inference
        formatted_prompts = []
        for hypothesis_id in range(1):
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
        MAX_RETRIES = 2

        for i, prompt in enumerate(formatted_prompts):
            retries = 0
            success = False
            output = None
            while not success and retries < MAX_RETRIES:
                try:
                    if self.use_openai:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=prompt,
                            temperature=0.75,
                            max_tokens=10,
                            stop=["<|user|>", "<|system|>"]
                        )
                        output = response.choices[0].message.content.split("<|assistant|>\n")[-1].strip()
                    else:
                        vllm_output = self.llm.generate([prompt], self.sampling_params)[0]
                        output = vllm_output.outputs[0].text.split("<|assistant|>\n")[-1].strip()
                    
                    # Check if output is valid
                    if not self.partnr:
                        for action in self.action_to_name.values():
                            if action.lower() in output.lower():
                                final_action_pred_list.append(self.name_to_action[action])
                                success = True
                                break
                    else:
                        for tool in tools:
                            if tool.lower() in output.lower():
                                final_action_pred_list.append(tool)
                                success = True
                                break
                    if success:
                        log_prob_hypothesis_list.append(1)
                    else:
                        retries += 1
                except Exception as e:
                    retries += 1
                    # Optionally log the exception

            if not success:
                # Optionally log the failure
                continue


        if len(log_prob_hypothesis_list) == 0:
            return None

        # normalize the log probs
        log_prob_hypothesis_list = np.array(log_prob_hypothesis_list)
        log_prob_hypothesis_list = log_prob_hypothesis_list - np.max(log_prob_hypothesis_list)
        log_prob_hypothesis_list = np.exp(log_prob_hypothesis_list)
        log_prob_hypothesis_list = log_prob_hypothesis_list / np.sum(log_prob_hypothesis_list)  # (num_hypothesis,)

        if self.partnr:
            # For partnr, randomly sample prediction based on probabilities
            idx = np.random.choice(np.arange(len(log_prob_hypothesis_list)), p=log_prob_hypothesis_list)
            final_pred = final_action_pred_list[idx]
            return final_pred
        else:
            # Initialize distribution over all possible actions
            res_pi = np.zeros(len(self.action_to_name))  # (num_actions,)
            
            # Aggregate predictions weighted by their log probabilities
            for action_pred, prob in zip(final_action_pred_list, log_prob_hypothesis_list):
                res_pi[action_pred] += prob
                
            # Normalize to ensure valid probability distribution
            res_pi = res_pi / np.sum(res_pi)

            return res_pi
            
        
        

