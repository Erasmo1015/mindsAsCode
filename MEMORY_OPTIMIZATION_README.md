# FSM Memory Optimization Guide

## Problem
FSM evaluation runs out of memory due to:
1. **High number of hypotheses** (default: 30) - Each hypothesis requires a separate LLM call
2. **High GPU memory utilization** (default: 0.9) - Leaves little buffer for operations
3. **Large batch token allocation** (default: 40000) - Allocates too much memory upfront
4. **Memory accumulation** - No cleanup between episodes/hypotheses
5. **Large model loading** - vLLM loads entire model into GPU memory

## Solutions Implemented

### 1. Reduced Default Parameters
- `n_hypothesis`: 30 → 10
- `gpu_memory_utilization`: 0.9 → 0.7
- `max_num_batched_tokens`: 40000 → 20000

### 2. Memory Cleanup
- Added `cleanup_memory()` function that calls `torch.cuda.empty_cache()` and `gc.collect()`
- Memory cleanup after each evaluation episode
- Memory cleanup after each hypothesis generation

### 3. vLLM Optimizations
- Added `max_num_seqs` and `max_paddings` limits
- Reduced batch token allocation
- Better memory management in hypothesis generation

## Quick Fix Commands

### For Immediate Use (Conservative Settings)
```bash
python eval_partnr.py \
    --baseline_model FSM \
    --n_hypothesis 5 \
    --gpu_memory_utilization 0.6 \
    --num_epochs 10
```

### For Different GPU Memory Sizes

#### 8GB GPU (RTX 3070, RTX 2080)
```bash
python eval_partnr.py \
    --baseline_model FSM \
    --n_hypothesis 5 \
    --gpu_memory_utilization 0.6 \
    --dtype float16
```

#### 12GB GPU (RTX 3080, RTX 4070)
```bash
python eval_partnr.py \
    --baseline_model FSM \
    --n_hypothesis 8 \
    --gpu_memory_utilization 0.7 \
    --dtype float16
```

#### 16GB GPU (RTX 4080, RTX 3090)
```bash
python eval_partnr.py \
    --baseline_model FSM \
    --n_hypothesis 12 \
    --gpu_memory_utilization 0.75 \
    --dtype float16
```

#### 24GB+ GPU (RTX 4090, A100, H100)
```bash
python eval_partnr.py \
    --baseline_model FSM \
    --n_hypothesis 15 \
    --gpu_memory_utilization 0.8 \
    --dtype float16
```

## Memory Monitoring

### Run Memory Monitor
```bash
# In a separate terminal
python memory_monitor.py
```

### Check GPU Memory
```bash
nvidia-smi
# or
watch -n 1 nvidia-smi
```

## Advanced Optimizations

### 1. Use Memory Configurations
```bash
# Get optimal config for your GPU
python memory_configs.py

# Use the recommended parameters
python eval_partnr.py --baseline_model FSM [copy parameters from output]
```

### 2. Reduce Model Size
```bash
# Use smaller models
--model_name "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"  # ~1.3B params
--model_name "microsoft/DialoGPT-medium"  # ~345M params
```

### 3. Enable Quantization (if supported)
```bash
# Add quantization to reduce memory usage
--quantization awq  # or gptq, sq
```

### 4. Use Tensor Parallelism (multi-GPU)
```bash
# If you have multiple GPUs
--tensor_parallel_size 2
```

## Troubleshooting

### Still Running Out of Memory?

1. **Reduce hypotheses further**:
   ```bash
   --n_hypothesis 3
   ```

2. **Lower GPU memory utilization**:
   ```bash
   --gpu_memory_utilization 0.5
   ```

3. **Use smaller model**:
   ```bash
   --model_name "microsoft/DialoGPT-medium"
   ```

4. **Enable gradient checkpointing** (if training):
   ```bash
   # Add to model loading
   --gradient_checkpointing
   ```

5. **Use CPU offloading** (slower but uses less GPU memory):
   ```bash
   # Modify vLLM config in llmFSM.py
   "gpu_memory_utilization": 0.3,
   "swap_space": 4,  # GB of CPU memory to use
   ```

### Memory Leaks

If you still experience memory leaks:

1. **Restart Python process** between large evaluations
2. **Use smaller batch sizes**:
   ```bash
   --num_epochs 5  # Evaluate fewer episodes at once
   ```
3. **Monitor with memory profiler**:
   ```bash
   pip install memory-profiler
   python -m memory_profiler eval_partnr.py
   ```

## Performance vs Memory Trade-offs

| Setting | Memory Usage | Performance | Quality |
|---------|-------------|-------------|---------|
| `n_hypothesis=3` | Low | Fast | Lower |
| `n_hypothesis=5` | Medium | Medium | Medium |
| `n_hypothesis=10` | High | Slow | Higher |
| `n_hypothesis=20+` | Very High | Very Slow | Highest |

## Best Practices

1. **Start with conservative settings** and increase gradually
2. **Monitor memory usage** during evaluation
3. **Use memory cleanup** between large operations
4. **Choose model size** based on your GPU memory
5. **Consider using smaller models** for initial testing
6. **Batch evaluations** to avoid memory accumulation

## Example Workflow

```bash
# 1. Check your GPU memory
nvidia-smi

# 2. Get optimal configuration
python memory_configs.py

# 3. Start memory monitor in another terminal
python memory_monitor.py

# 4. Run evaluation with recommended settings
python eval_partnr.py --baseline_model FSM [recommended_params]

# 5. Monitor and adjust if needed
```

This should resolve your memory issues while maintaining evaluation quality! 