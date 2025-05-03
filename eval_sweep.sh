# for model_arch in "BC"
# do
#     for n_hypothesis in 2
#     do
#         for llm_model in "meta-llama/Llama-3.1-8B-Instruct"
#         do
#             for group in False
#             do
#                 sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis $group
#             done
#         done
#     done
# done


# for model_arch in "NLLM" "AutoToM"
# do
#     for n_hypothesis in 2
#     do
#         for llm_model in "meta-llama/Llama-3.1-8B-Instruct"
#         do
#             for group in False True
#             do
#                 sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis $group
#             done
#         done
#     done
# done

for model_arch in "FSM"
do
    for n_hypothesis in 1 2 3 4 5 6
    do
        for llm_model in "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" "meta-llama/Llama-3.1-8B-Instruct"
        do
            for group in False
            do
                sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis $group
            done
        done
    done
done
