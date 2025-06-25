# for model_arch in "BC"
# do
#     for n_hypothesis in 2
#     do
#         for llm_model in "meta-llama/Llama-3.1-8B-Instruct"
#         do
#             for group in True False
#             do
#                 for structured in "False"
#                 do
#                     for two_stage in False
#                     do
#                         for rejuvenation in False
#                         do
#                             for top_k in 0  # 0 means no top k, use all hypotheses
#                             do
#                                 # sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
                                
#                                 # Add partnr evaluation
#                                 # sbatch eval_partnr.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
#                                 sbatch eval_ms.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
                                
#                                 # if [ "$group" = "False" ]; then
#                                 #     sbatch eval_human.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
#                                 # fi
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done





# for model_arch in "NLLM" "AutoToM"
# do
#     for n_hypothesis in 2
#     do
#         for llm_model in "gpt-4.1-nano"
#         do
#             for group in False True
#             do
#                 sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis $group
#                 if [ "$group" = "False" ]; then
#                     sbatch eval_human.slurm $llm_model $model_arch $n_hypothesis $group
#                 fi
#                 # sbatch eval_partnr.slurm $llm_model $model_arch $n_hypothesis $group
#             done
#         done
#     done
# done


for model_arch in "AutoToM" "NLLM"
do
    for n_hypothesis in 2
    do
        # for llm_model in "deepseek-ai/DeepSeek-V2-Lite" "meta-llama/Llama-3.1-8B-Instruct" "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
        for llm_model in "meta-llama/Llama-3.1-8B-Instruct"
        do
            for group in False
            do
                for structured in "False"
                do
                    for two_stage in False
                    do
                        for rejuvenation in False
                        do
                            for top_k in 0  # 0 means no top k, use all hypotheses
                            do
                                for multi_step_eval in True
                                do
                                    # grid evaluation
                                    # sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
                                    
                                    # Add partnr evaluation
                                    # sbatch eval_partnr.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k

                                    # multi-step evaluation
                                    sbatch eval_ms.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k $multi_step_eval
                                    
                                    # if [ "$group" = "False" ]; then
                                        # sbatch eval_human.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
                                    # fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done




for model_arch in "FSM"
do
    for n_hypothesis in 30
    do
        # for llm_model in "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "deepseek-ai/DeepSeek-V2-Lite"
        for llm_model in "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
        do
            for group in False
            do
                # for structured in "False" "p1" "p2"
                for structured in "False"
                do
                    # for two_stage in False True
                    for two_stage in True
                    do
                        # for rejuvenation in False True
                        for rejuvenation in False True
                        do
                            # for top_k in 0 1 10 # 0 means no top k, use all hypotheses
                            for top_k in 0
                            do
                                # sbatch eval_cluster.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
                                
                                # Add partnr evaluation
                                # sbatch eval_partnr.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
                                sbatch eval_ms.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
                                
                                # if [ "$group" = "False" ]; then
                                    # sbatch eval_human.slurm $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k
                                # fi
                            done
                        done
                    done
                done
            done
        done
    done
done
