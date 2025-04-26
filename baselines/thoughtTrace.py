from vllm import LLM, SamplingParams
import torch
import pandas as pd
import time
import os
from rich.progress import track


class ThoughtTrace:
    def __init__(self, n_hypothesis=4, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", tensor_parallel_size: int = 1, dtype: torch.dtype = torch.bfloat16, gpu_memory_utilization: float = 0.55):
        self.n_hypothesis = n_hypothesis
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization

        self.action_to_name = {
            0: "stay",
            1: "right",
            2: "left",
            3: "down",
            4: "up",
            5: "interact"
        }
        self.name_to_action = {v: k for k, v in self.action_to_name.items()}

        self.llm = LLM(
            self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization
        )
    
    def convert_state_action_to_text(self, state, action):
        breakpoint()
        return ""

    def predict_action(self, state, action, training=False):
        text = self.convert_state_action_to_text(state, action)
        return
    
        

