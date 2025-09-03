#!/bin/bash

#SBATCH --job-name=trainBC
#SBATCH --mail-user=kj@cs.washington.edu

#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --time=00-10:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --output=/mmfs1/gscratch/socialrl/kjha/automaticity/logs/%x.%j.out
#SBATCH --error=/mmfs1/gscratch/socialrl/kjha/automaticity/logs/%x.%j.err
#SBATCH --open-mode=append

# Add requeue flag
#SBATCH --requeue

source ~/.bashrc
conda activate llm
module load cuda/12.4
module load gcc/10.2.0

cd /mmfs1/gscratch/socialrl/kjha/automaticity/results

python partnr_embeddings.py

# # Run the main command and capture its exit status
# python train_baselines.py --baseline_model 'BC' --num_epochs 5000 --num_agents_to_sample 1 --num_datapoints_per_agent_to_sample 3 --seed $1
# EXIT_CODE=$?

# # If the job failed (non-zero exit code), resubmit it
# if [ $EXIT_CODE -ne 0 ]; then
#     echo "Job failed with exit code $EXIT_CODE. Requeuing..."
#     # Submit the same script again
#     sbatch $0 $1
#     exit $EXIT_CODE
# fi
