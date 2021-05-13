#!/bin/bash

#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err
#SBATCH --gres=gpu:1

python3 run_qdess.py --dir_out delete_me --file_id_list 006 --num_iter 10
