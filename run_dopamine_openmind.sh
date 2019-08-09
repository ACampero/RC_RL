#!/bin/bash

#SBATH --qos=tenenbaum
#SBATCH --gres=gpu:1 	        #GPU capabilities
#SBATCH -n 1                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 4-12:00              # Runtime in D-HH:MM
#SBATCH --mem=35000             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./output/prueba.out      # File to which STDOUT will be written
#SBATCH -e ./errors/prueba.err      # File to which STDERR will be written


declare -A array
array[1]=aliens

trial=203
name=${array[1]}

module add openmind/singularity openmind/cudnn openmind/cuda
cd /om2/user/campero/vgdl_testing/RC_RL/
singularity exec --nv --bind $(pwd) VGDLContainer.py2.simg bash run_dopamine_task.sh $name



