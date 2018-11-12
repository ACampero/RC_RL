#!/bin/bash
#SBATCH --gres=gpu:1 			#GPU capabilities
#SBATCH -n 1                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 4-12:00              # Runtime in D-HH:MM
#SBATCH -p cox       			# Partition to submit to
#SBATCH --mem=16000             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/expt_push_boulders-seq-trial1.out      # File to which STDOUT will be written
#SBATCH -e errors/expt_push_boulders-seq-trial1.err      # File to which STDERR will be written

source activate pt27
python run.py -game_name expt_push_boulders -doubleq 1 -trial_num 1 -level_switch sequential -lr 0.00025 -num_trials 5