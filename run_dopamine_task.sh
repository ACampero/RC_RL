#!/bin/bash

source venv/bin/activate
cd dopamine

VGDLNAME="VGDL_$1"
aux1=gin_bindings='create_atari_environment.game_name="VGDL_$1"'

echo $aux1

python -um dopamine.discrete_domains.train \
--base_dir=./tmp/dopamine/$1 \
--gin_files='dopamine/agents/rainbow/configs/rainbow_aaaiAndres.gin' \
--$aux1
