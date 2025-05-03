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

#SBATCH --output=/mmfs1/gscratch/socialrl/kjha/automaticity/logs/%x.%j.out
#SBATCH --error=/mmfs1/gscratch/socialrl/kjha/automaticity/logs/%x.%j.err
#SBATCH --open-mode=append

# Add requeue flag
#SBATCH --requeue

source ~/.bashrc
conda activate llm
module load cuda/12.4
module load gcc/10.2.0

llm_model=$1
model_arch=$2
n_hypothesis=$3
group=$4

if [ "$model_arch" = "BC" ]; then
    n_agents_to_sample=3
    n_datapoints_per_agent_to_sample=3
    as_images=True
    num_epochs=33
else
    n_agents_to_sample=1
    n_datapoints_per_agent_to_sample=1
    as_images=False
    num_epochs=100
fi

echo "model_arch: $model_arch"
echo "n_hypothesis: $n_hypothesis"
echo "llm_model: $llm_model"
echo "group: $group"
echo "n_agents_to_sample: $n_agents_to_sample"
echo "n_datapoints_per_agent_to_sample: $n_datapoints_per_agent_to_sample"
echo "as_images: $as_images"
echo "num_epochs: $num_epochs"


# Run the main command and capture its exit status
if [ "$group" = "True" ]; then
    python eval_baselines.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --num_agents 4 --num_epochs $num_epochs --num_agents_to_sample $n_agents_to_sample --num_datapoints_per_agent_to_sample $n_datapoints_per_agent_to_sample --group True --as_images $as_images
else
    python eval_baselines.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --num_agents 20 --num_epochs $num_epochs --num_agents_to_sample $n_agents_to_sample --num_datapoints_per_agent_to_sample $n_datapoints_per_agent_to_sample --as_images $as_images
    # python eval_partnr.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --num_agents 20 --num_epochs 100 --num_agents_to_sample 1 --num_datapoints_per_agent_to_sample 1 --group False
    # python eval_human.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --num_agents 20 --num_epochs 120 --num_agents_to_sample $n_agents_to_sample --num_datapoints_per_agent_to_sample $n_datapoints_per_agent_to_sample
fi
EXIT_CODE=$?

# If the job failed (non-zero exit code), resubmit it
if [ $EXIT_CODE -ne 0 ]; then
    echo "Job failed with exit code $EXIT_CODE. Requeuing..."
    # Submit the same script again
    sbatch $0 $llm_model $model_arch $n_hypothesis $group
    exit $EXIT_CODE
fi

