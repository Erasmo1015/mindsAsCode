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

class FSMReasoner:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size: int = 1, device: str = "cuda", 
                 dtype: str = "float16", gpu_memory_utilization: float = 0.55, num_hypothesis: int = 4,
                 max_model_len: int = 2048, quantization: str = None, group: bool = False):
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
        if not group:
            self.dataset_name = "single_agent_dataset"
            prompt_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/infer_single_fsm.txt"
            code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/single_code_template.txt"
        else:
            self.dataset_name = "group_agent_dataset"
            prompt_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/infer_group_fsm.txt"
            code_template_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/group_code_template.txt"
        refinement_1_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_1.txt"
        refinement_2_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_2.txt"
        refinement_3_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/prompts/refinement_3.txt"

        self.base_prompt = open(prompt_path, "r").read()
        self.code_template = open(code_template_path, "r").read()
        self.refinement_1 = open(refinement_1_path, "r").read()
        self.refinement_2 = open(refinement_2_path, "r").read()
        self.refinement_3 = open(refinement_3_path, "r").read()


        # Convert dtype string to torch dtype
        torch_dtype = torch.float16
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32

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
            action = [self.action_to_name[a] for a in actions[i]]
            state_strings.append(f"{i+1}. State: {state_string}.")
            action_string = f"{i+1}."
            for aid, a in enumerate(action):
                action_string += f" Agent {aid}'s Action: {a}, "
            action_strings.append(action_string)
        state_action_strings = [f"{s} {a}" for s, a in zip(state_strings[:-1], action_strings[:-1])]
        state_action_strings.append(state_strings[-1])
        return "\n-------\n".join(state_action_strings)
    
    def predict_action(self, states, actions, training=False, episode_id=0):
        episode_name = f"{self.dataset_name}_{episode_id}"
        state_action_text = self.convert_states_actions_to_text(states, actions)
        prompt = f"{self.base_prompt}\n{state_action_text}\n{self.code_template}"
        full_prompt = prompt

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
            
            # Format messages for vLLM
            if "llama" in self.model_name.lower():
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
        
        # Batch generate with vLLM
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        
        # Process each hypothesis
        for hypothesis_id, output in enumerate(outputs):
            agent_code = output.outputs[0].text
            
            # Generate with transformers (commented out)
            # def generate_response(prompt):
            #     inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            #     outputs = self.llm_model.generate(
            #         **inputs,
            #         max_new_tokens=1000,
            #         temperature=1.0,
            #         pad_token_id=self.llm_tokenizer.eos_token_id
            #     )
            #     response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            #     response = response[len(prompt):]  # remove the prompt from the response
            #     return response
            
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
            num_trials = 5
            while trial < num_trials:
                try:
                    agent = framework.compile_agent(agent_code, num_agents, num_blocks)

                    log_prob_hypothesis = 0
                    # p(script | states, actions) = p(action | states, script) * prior(script)
                    for timestep in range(actions.shape[0] - 1):  
                        state = jax.tree.map(lambda x: x[timestep], states)
                        if not self.group:
                            gt_action = actions[timestep][0]
                            proposed_action, proposed_pi = agent.act(state)
                            proposed_pi = np.exp(proposed_pi + 1e-10) / np.sum(np.exp(proposed_pi + 1e-10))
                            log_prob_hypothesis += np.log(np.clip(proposed_pi[gt_action], 1e-10, 1))
                        else:
                            gt_actions = actions[timestep]
                            proposed_actions, proposed_pis = agent.act(state)
                            for a in proposed_actions:
                                assert a in range(6), "an action in proposed_actions is not an integer in range(num_actions)"
                            for i in range(len(proposed_actions)):
                                proposed_pi = np.exp(proposed_pis[i] + 1e-10) / np.sum(np.exp(proposed_pis[i] + 1e-10))
                                log_prob_hypothesis += np.log(np.clip(proposed_pi[gt_actions[i]], 1e-10, 1))
                    
                    final_state = jax.tree.map(lambda x: x[-1], states)
                    final_action, final_pi = agent.act(final_state)

                    agents.append(agent)
                    log_prob_hypothesis_list.append(log_prob_hypothesis)
                    final_action_pred_list.append(final_pi)  # time t
                    break
                except Exception as e:
                    # print(f"Error compiling agent {hypothesis_id}: {e}")
                    trial += 1
                    full_traceback = traceback.format_exc()
                    if trial == num_trials:
                        # print(f"Failed to compile hypothesis {hypothesis_id} after {num_trials} trials")
                        # print(full_traceback)
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

        if not self.group:
            final_action_pred_list = np.array(final_action_pred_list) + 1e-6
            final_action_pred_list = np.clip(final_action_pred_list, 1e-6, 1)
            final_action_pred_list = final_action_pred_list / np.sum(final_action_pred_list, axis=1, keepdims=True)

            res_pi = np.sum(log_prob_hypothesis_list * final_action_pred_list.T, axis=1)  # (num_actions,)
        else:
            final_action_pred_list = np.array(final_action_pred_list) + 1e-6
            final_action_pred_list = np.clip(final_action_pred_list, 1e-6, 1)
            final_action_pred_list = final_action_pred_list / np.sum(final_action_pred_list, axis=-1, keepdims=True)   # (num_hypothesis, num_agents, num_actions)
            # Reshape log_prob_hypothesis_list to (num_hypothesis, 1, 1) for broadcasting
            weights = log_prob_hypothesis_list[:, np.newaxis, np.newaxis]
            # Multiply and sum across hypotheses in one operation
            res_pi = np.sum(weights * final_action_pred_list, axis=0)  # (num_agents, num_actions)
            

        return res_pi
            
        
        

