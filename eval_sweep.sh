for model_arch in "FSM"
do
    for n_hypothesis in 2 3 4 5
    do
        for llm_model in "gpt-4.1" "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
        do
            sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis
        done
    done
done