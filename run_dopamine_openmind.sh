#!/bin/bash

#SBATCH --gres=gpu:1 	        #GPU capabilities
#SBATCH -n 1                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 4-12:00              # Runtime in D-HH:MM
#SBATCH --mem=35000             # Memory pool for all cores (see also --mem-per-cpu)


array=( aliens zelda butterflies plaqueattack expt_antagonist expt_ee
avoidgeorge jaws watergame boulderdash expt_push_boulders expt_relational
survivezombies bait expt_preconditions expt_helper chase
variant_aliens_1 variant_aliens_2 variant_aliens_3 variant_aliens_4
variant_expt_antagonist_1 variant_expt_antagonist_2 variant_avoidgeorge_1
variant_avoidgeorge_2 variant_avoidgeorge_3 variant_avoidgeorge_4
variant_bait_1 variant_bait_2 bees_and_birds variant_bees_and_birds_1
variant_boulderdash_1 variant_boulderdash_2 variant_butterflies_1
variant_butterflies_2 variant_chase_1 variant_chase_2 variant_chase_3
closing_gates variant_closing_gates_1 corridor variant_corridor_1
variant_expt_ee_1 variant_expt_ee_2 variant_expt_ee_3 variant_frogs_1
variant_frogs_2 variant_frogs_3 variant_expt_helper_1 variant_expt_helper_2
variant_jaws_1 variant_jaws_2 lemmings variant_lemmings_1 variant_lemmings_2
variant_lemmings_3 missilecommand variant_missilecommand_1
variant_missilecommand_2 variant_missilecommand_3 variant_missilecommand_4
myAliens variant_myAliens_1 variant_myAliens_2
variant_plaqueattack_1 variant_plaqueattack_2 variant_plaqueattack_3
variant_portals_1 variant_portals_2
variant_expt_preconditions_1 variant_expt_preconditions_2
variant_expt_push_boulders_1 variant_expt_push_boulders_2
variant_expt_relational_1 variant_expt_relational_2
variant_sokoban_1 variant_sokoban_2
surprise variant_surprise_1 variant_surprise_2
variant_survivezombies_1 variant_survivezombies_2
variant_watergame_1 variant_watergame_2
variant_zelda_1 variant_zelda_2 variant_zelda_3
frogs portals sokoban)

name=${array[$1]}

module add openmind/singularity openmind/cudnn openmind/cuda
cd /om2/user/campero/vgdl_testing/RC_RL/
singularity exec --nv --bind $(pwd) VGDLContainer.py2.simg bash run_dopamine_task.sh $name



