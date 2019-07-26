#!/bin/bash
#SBATCH --gres=gpu:1 			#GPU capabilities
#SBATCH -n 1                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 4-12:00              # Runtime in D-HH:MM
#SBATCH --mem=16000             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/plaqueattack-random-trial1.out      # File to which STDOUT will be written
#SBATCH -e errors/plaqueattack-random-trial1.err      # File to which STDERR will be written

set -e
root=/om2/user/$USER/vgdl_testing/RC_RL
module add openmind/singularity openmind/cudnn openmind/cuda
singularity exec --nv --bind $root VGDLContainer.py2.simg bash openmind/run_model_task.sh
