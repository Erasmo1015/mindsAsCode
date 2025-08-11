#!/bin/bash

#SBATCH --job-name=genSingleFSMData
#SBATCH --mail-user=kj@cs.washington.edu

#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40s
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --output=/mmfs1/gscratch/socialrl/kjha/automaticity/logs/%x.%j.out
#SBATCH --error=/mmfs1/gscratch/socialrl/kjha/automaticity/logs/%x.%j.err

source ~/.bashrc
conda activate llm
module load cuda/12.4
module load gcc/10.2.0

cd ..

python gen_data.py
