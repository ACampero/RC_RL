#!/bin/bash
#SBATCH --gres=gpu:1 			#GPU capabilities
#SBATCH -n 1                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 6-12:00              # Runtime in D-HH:MM
#SBATCH -p fas_gpu       			# Partition to submit to
#SBATCH --mem=32000             # Memory pool for all cores (see also --mem-per-cpu)

source activate pt27
python run.py -game_name ${game_name} -doubleq 1 -trial_num 1 -level_switch sequential -lr 0.00025 -num_trials 1