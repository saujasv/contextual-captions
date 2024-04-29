#!/bin/bash

#SBATCH --mem=48G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 12:00:00
#SBATCH --output slurm-out/run_inference-%j.out

HOST=`hostname`
echo $HOST

export CONDA_AUTO_ACTIVATE_BASE=false

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pgmproject2

python run_sampling.py data/imagecode_val_shard=4.json results/imagecode_val_shard=4.json 8
