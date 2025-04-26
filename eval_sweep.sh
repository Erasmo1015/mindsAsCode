for model_arch in "AutoToM"
do
    for n_hypothesis in 2
    do
        for llm_model in "meta-llama/Llama-3.1-8B-Instruct"
        do
            sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis
        done
    done
done