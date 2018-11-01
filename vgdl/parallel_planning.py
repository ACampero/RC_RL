from pathos.multiprocessing import ProcessingPool
from main_agent import Agent
import time
import dill
import os
from IPython import embed
import argparse

parser = argparse.ArgumentParser(description='Process game number.')
parser.add_argument('--game_number', type=int, default=0, help='game number')

args = parser.parse_args()
game_number = args.game_number
# NOTE: fmin seems to fail with the hyperopt version installed by default
# as of 01/2018: it is best to install directly from the github repo with
# the command 'pip install git+https://github.com/hyperopt/hyperopt'

gameFileString = 'training_set_1'

gvggames = ['aliens', 'boulderdash', 'chase', 'frogs',  # 0-4
        	'missilecommand', 'portals']  # 5-9

local_games = ['expt_antagonist', 'expt_exploration_exploitation', 'expt_helper',  # 10-12
    'expt_preconditions', 'expt_push_boulders', 'expt_relational', 'video_debug']  # 13-15 to play a "local" game

# gvggames = ['aliens', 'boulderdash', 'chase', 'frogs',  # 0-3
        	# 'missilecommand', 'portals', 'sokoban', 'survivezombies']  # 4-7

# local_games = ['expt_exploration_exploitation',  # 8
    # 'expt_preconditions', 'expt_push_boulders', 'expt_relational']  # 9-11 to play a "local" game

def play_trainset(hyperparameters):
    start_time = time.time()


    # playing GVG-AI games
    if game_number < 6:
        gameName = gvggames[game_number]  # to play a gvgai game
        def read_gvgai_game(filename):
        	with open(filename, 'r') as f:
        		new_doc = []
        		g = gen_color()
        		for line in f.readlines():
        			new_line = (" ".join([string if string[:4]!="img="
        				else "color={}".format(next(g))
        				for string in line.split(" ")]))
        			new_doc.append(new_line)
        		new_doc = "\n".join(new_doc)
        	return new_doc

        def gen_color():
        	from vgdl.colors import colorDict
        	color_list = colorDict.values()
        	color_list = [c for c in color_list if c not in ['UUWSWF']]
        	for color in color_list:
        		yield color


        gvgname = "./{}/{}".format(gameFileString,gameName)

        gameString = read_gvgai_game('{}.txt'.format(gvgname))

        game_levels = [l for l in os.listdir(gameFileString) if gameName+'_lvl' in l]
        level_game_pairs = []
        for level_number in range(len(game_levels)):
        	with open('{}_lvl{}.txt'.format(gvgname, level_number), 'r') as level:
        		level_game_pairs.append([gameString, level.read()])

    # running local games
    else:
        level_game_pairs = None
        gameName = 'examples.gridphysics_new.{}'.format(local_games[game_number-6])

    agent = Agent('full', gameName, hyperparameter_sets=hyperparameters, parallel_planning=True)

    ##then pass this down for multiple episodes
    gameObject = None
    agent.playCurriculum(level_game_pairs=level_game_pairs)

    total_time = time.time() - start_time

    return total_time

hyperparameter_sets = [
    {## Worked for aliens. 
    ##shortHorizon=False,regrounding=3, selective_regrounding=True, avoid_danger=True, safeDistance=6, emptyPlansLimit=5, longHorizonObservationLimit=2
     #WBP.allowRollouts=True
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 1000,
     'sprite_negative_mult': .1,
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 0,#100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 0#50,
     },
    {
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 100,
     'sprite_negative_mult': 10.,
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 50
     },
    {
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 100,
     'sprite_negative_mult': .1,
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 50,
     }
]
play_trainset(hyperparameter_sets)
