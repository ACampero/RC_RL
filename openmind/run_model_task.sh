#!/bin/bash
set -e
cd /om2/user/$USER/vgdl_testing/RC_RL
source venv/bin/activate
python run.py -game_name plaqueattack -doubleq 1 -trial_num 1 -level_switch random -lr 0.00025
