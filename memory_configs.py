"""
Memory-optimized configurations for FSM evaluation.
Choose the configuration that matches your GPU memory.
"""

# Configuration presets based on GPU memory
MEMORY_CONFIGS = {
    # For 8GB GPU (e.g., RTX 3070, RTX 2080)
    "8GB": {
        "gpu_memory_utilization": 0.6,
        "n_hypothesis": 5,
        "max_num_batched_tokens": 10000,
        "max_num_seqs": 128,
        "max_paddings": 1024,
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "max_tokens": 1500,
    },
    
    # For 12GB GPU (e.g., RTX 3080, RTX 4070)
    "12GB": {
        "gpu_memory_utilization": 0.7,
        "n_hypothesis": 8,
        "max_num_batched_tokens": 15000,
        "max_num_seqs": 192,
        "max_paddings": 1536,
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "max_tokens": 1800,
    },
    
    # For 16GB GPU (e.g., RTX 4080, RTX 3090)
    "16GB": {
        "gpu_memory_utilization": 0.75,
        "n_hypothesis": 12,
        "max_num_batched_tokens": 20000,
        "max_num_seqs": 256,
        "max_paddings": 2048,
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "max_tokens": 2000,
    },
    
    # For 24GB GPU (e.g., RTX 4090, RTX 3090 Ti)
    "24GB": {
        "gpu_memory_utilization": 0.8,
        "n_hypothesis": 15,
        "max_num_batched_tokens": 25000,
        "max_num_seqs": 320,
        "max_paddings": 2560,
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "max_tokens": 2000,
    },
    
    # For 40GB+ GPU (e.g., A100, H100)
    "40GB+": {
        "gpu_memory_utilization": 0.85,
        "n_hypothesis": 20,
        "max_num_batched_tokens": 40000,
        "max_num_seqs": 512,
        "max_paddings": 4096,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_tokens": 2000,
    },
}

def get_gpu_memory_size():
    """Detect GPU memory size using nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_mb = int(result.stdout.strip().split('\n')[0])
            memory_gb = memory_mb // 1024
            return memory_gb
    except Exception as e:
        print(f"Error detecting GPU memory: {e}")
    return None

def get_optimal_config(gpu_memory_gb=None):
    """Get optimal configuration based on GPU memory."""
    if gpu_memory_gb is None:
        gpu_memory_gb = get_gpu_memory_size()
    
    if gpu_memory_gb is None:
        print("Could not detect GPU memory. Using 8GB configuration as default.")
        return MEMORY_CONFIGS["8GB"]
    
    # Find the best matching configuration
    if gpu_memory_gb >= 40:
        config_key = "40GB+"
    elif gpu_memory_gb >= 24:
        config_key = "24GB"
    elif gpu_memory_gb >= 16:
        config_key = "16GB"
    elif gpu_memory_gb >= 12:
        config_key = "12GB"
    else:
        config_key = "8GB"
    
    print(f"Detected {gpu_memory_gb}GB GPU. Using {config_key} configuration.")
    return MEMORY_CONFIGS[config_key]

def print_config_help():
    """Print help information for memory configurations."""
    print("Memory-Optimized FSM Evaluation Configurations")
    print("=" * 50)
    print("Available configurations based on GPU memory:")
    for memory, config in MEMORY_CONFIGS.items():
        print(f"\n{memory} GPU:")
        for key, value in config.items():
            print(f"  --{key}: {value}")
    
    print(f"\nTo use optimal configuration for your GPU:")
    print("1. Run: python -c 'from memory_configs import get_optimal_config; print(get_optimal_config())'")
    print("2. Copy the parameters to your evaluation command")

if __name__ == "__main__":
    print_config_help()
    print("\n" + "=" * 50)
    print("Optimal configuration for your GPU:")
    optimal_config = get_optimal_config()
    for key, value in optimal_config.items():
        print(f"--{key} {value}") 