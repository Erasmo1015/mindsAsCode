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

# Run the main command and capture its exit status
python eval_baselines.py --baseline_model $model_arch --n_hypothesis $n_hypothesis --model_name $llm_model --num_agents 20 --num_epochs 5 --num_agents_to_sample 3 --num_datapoints_per_agent_to_sample 3
