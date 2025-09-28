#!/bin/bash

#SBATCH --job-name=testToM
#SBATCH --mail-user=kj@cs.washington.edu

#SBATCH --account=socialrl
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --time=02-00:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --output=/mmfs1/gscratch/socialrl/kjha/mindsAsCode/logs/%x.%j.out
#SBATCH --error=/mmfs1/gscratch/socialrl/kjha/mindsAsCode/logs/%x.%j.err
#SBATCH --open-mode=append

# Add requeue flag
#SBATCH --requeue

cd /mmfs1/gscratch/socialrl/kjha/mindsAsCode

source ~/.bashrc
conda activate llm
module load cuda/12.4
module load gcc/10.2.0

llm_model=$1
model_arch=$2
n_hypothesis=$3
group=$4
structured=${5:-"False"}
two_stage=${6:-False}
rejuvenation=${7:-False}
top_k=${8:-0}
multi_step_eval=${9:-True}

if [ "$model_arch" = "BC" ]; then
    num_agents_to_sample=1
    num_datapoints_per_agent_to_sample=3
    num_epochs=100
else
    num_agents_to_sample=1
    num_datapoints_per_agent_to_sample=1
    num_epochs=100
fi

echo "model_arch: $model_arch"
echo "n_hypothesis: $n_hypothesis"
echo "llm_model: $llm_model"
echo "group: $group"
echo "structured: $structured"
echo "two_stage: $two_stage"
echo "rejuvenation: $rejuvenation"
echo "top_k: $top_k"
echo "num_agents_to_sample: $num_agents_to_sample"
echo "num_datapoints_per_agent_to_sample: $num_datapoints_per_agent_to_sample"
echo "num_epochs: $num_epochs"
structured_flag=""
two_stage_flag=""
rejuvenation_flag=""
top_k_flag=""

if [ "$model_arch" = "BC" ]; then
    env_size=10
else
    env_size=7
fi

if [ "$structured" != "False" ]; then
    structured_flag="--structured $structured"
fi

if [ "$two_stage" = "True" ]; then
    two_stage_flag="--two_stage"
fi

if [ "$rejuvenation" = "True" ]; then
    rejuvenation_flag="--rejuvenation"
fi

if [ "$top_k" -gt 0 ]; then
    top_k_flag="--top_k $top_k"
fi
    

# Run the main command and capture its exit status
if [ "$group" = "True" ]; then
    python plot_and_eval.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --num_agents 4 --group True $structured_flag $two_stage_flag $rejuvenation_flag $top_k_flag --num_agents_to_sample $num_agents_to_sample --num_datapoints_per_agent_to_sample $num_datapoints_per_agent_to_sample --num_epochs $num_epochs --bootstrap --multi_step_eval $multi_step_eval --env_size $env_size
else
    # Add structured, two_stage, and rejuvenation flags if they are True
    python plot_and_eval.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --bootstrap $structured_flag $two_stage_flag $rejuvenation_flag $top_k_flag --num_agents_to_sample $num_agents_to_sample --num_datapoints_per_agent_to_sample $num_datapoints_per_agent_to_sample --num_epochs $num_epochs --multi_step_eval $multi_step_eval  --num_steps_to_predict 20 --flip_quarter True --env_size $env_size
    # python eval_partnr.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --num_agents 20 --num_epochs 100 --num_agents_to_sample 1 --num_datapoints_per_agent_to_sample 1 --group False
    # python eval_human.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --num_agents 20 --num_epochs 120 --num_agents_to_sample $n_agents_to_sample --num_datapoints_per_agent_to_sample $n_datapoints_per_agent_to_sample
fi
EXIT_CODE=$?

# # If the job failed (non-zero exit code), resubmit it
# if [ $EXIT_CODE -ne 0 ]; then
#     echo "Job failed with exit code $EXIT_CODE. Requeuing..."
#     # Submit the same script again
#     sbatch $0 $llm_model $model_arch $n_hypothesis $group $structured $two_stage $rejuvenation $top_k $multi_step_eval
#     exit $EXIT_CODE
# fi

