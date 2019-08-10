
git pull
cd dopamine

array=( aliens zelda butteflies plaqueattack expt_antagonist expt_ee
avoidgeorge jaws watergame boulderdash expt_push_boulders expt_relational
survivezombies bait expt_preconditions expt_helper chase
variant_aliens1 variant_aliens_2 variant_aliens_3 variant_aliens_4
variant_expt_antagonist_1 variant_expt_antagonist_2 variant_avoidgeorge_1
variant_avoidgeorge_2 variant_avoidgeorge_3 variant_avoidgeorge_4 
variant_bait_1 variant_bait_2 bees_and_birds bees_and_birds_variant_1 
variant_boulderdash_1 variant_boulderdash_2 variant_butterflies_1 
variant_butterflies_2 variant_chase_1 variant_chase_2 variant_chase_3 
closing_gates variant_closing_gates_1 corridor variant_corridor_1 
variant_expt_ee_1 variant_expt_ee_2 variant_expt_ee_3 variant_frogs_1 
variant_frogs_2 variant_frogs_3 variant_expt_heper_1 variant_expt_heper_2 
variant_jaws_1 variant_jaws_2 lemmings variant_lemmings_1 variant_lemmings_2 
variant_lemmings_3 missilecommand variant_missilecommand_1
variant_missilecomand_2 variant_missilecomand_3 variant_missilecommand_4
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
variant_zelda_1 variant_zelda_2 variant_zelda_3)

echo $1 $2 
for i in {$1..$2}
do 
    aux=${array[$i]}
    directory=./tmp/dopamine/$aux
    game_name=\'create_atari_environment.game_name=\"VGDL_$aux\"\'
    
    python -um dopamine.discrete_domains.train \
    --base_dir=$directory
    --gin_files='dopamine/agents/rainbow/configs/rainbow_aaaiAndres.gin' \
    --gin_bindings=$game_name
    
    echo finished game
done

echo finished all games
