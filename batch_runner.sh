#!/bin/bash
# declare -a arr=("string1" "string2" "string3")
game_name=$1
#!/bin/bash

# Add the full path processes to run to the array
PROCESSES_TO_RUN=(
# "variant_aliens_1" \
# "variant_aliens_2" \
# "variant_aliens_3" \
# "variant_aliens_4" \
# )
# "variant_bees_and_birds_1" \
# "variant_butterflies_1" \
# "variant_butterflies_2" \
# "variant_chase_1" \
# "variant_chase_2" \
# "variant_chase_3" \
# "variant_closing_gates_1" \
# "variant_corridor_1" \
# "variant_expt_ee_1" \
# "variant_expt_ee_2" \
# "variant_expt_ee_3" \
# "variant_lemmings_1" \
# "variant_lemmings_2" \
# "variant_lemmings_3" \
# "variant_missilecommand_1" \
# "variant_missilecommand_2" \
# "variant_missilecommand_3" \
# "variant_missilecommand_4" \
# "variant_myAliens_1" \
# "variant_myAliens_2" \
# "variant_surprise_1" \
# "variant_surprise_2" \
# "variant_survivezombies_1" \
# "variant_survivezombies_2" \
# "variant_zelda_1" \
# "variant_zelda_2" \
# "variant_zelda_3" \
# "variant_avoidgeorge_1" \
# "variant_avoidgeorge_2" \
# "variant_avoidgeorge_3" \
# "variant_avoidgeorge_4" \
# "variant_bait_1" \
# "variant_bait_2" \
# "variant_boulderdash_1" \
# "variant_boulderdash_2" \
# "variant_expt_antagonist_1" \
# "variant_expt_antagonist_2" \
# "variant_expt_helper_1" \
# "variant_expt_helper_2" \
# "variant_expt_preconditions_1" \
# "variant_expt_preconditions_2" \
# "variant_expt_push_boulders_1" \
# "variant_expt_push_boulders_2" \
# "variant_expt_relational_1" \
# "variant_expt_relational_2" \
# "variant_frogs_1" \
# "variant_frogs_2" \
# "variant_frogs_3" \
# "variant_jaws_1" \
# "variant_jaws_2" \
# "variant_portals_1" \
# "variant_portals_2" \
"variant_plaqueattack_1" \
"variant_plaqueattack_2" \
"variant_plaqueattack_3" \
)
# "variant_sokoban_1" \
# "variant_sokoban_2" \
# "variant_watergame_1" \
# "variant_watergame_2")
# You can keep adding processes to the array...

for game_name in ${PROCESSES_TO_RUN[@]}; do
    echo -e "Running Game: ${game_name}"
    export game_name
    sbatch rl_runscript.sbatch -o=output/${game_name}.out -e=errors/${game_name}.out
    # ${i%/*}/./${i##*/} > ${i}.log 2>&1 &
    # ${i%/*} -> Get folder name until the /
    # ${i##*/} -> Get the filename after the /
done

# Wait for the processes to finish
wait