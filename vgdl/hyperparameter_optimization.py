from hyperopt import fmin, tpe, hp
from pathos.multiprocessing import ProcessingPool
from main_agent import Agent
import time
import dill


# NOTE: fmin seems to fail with the hyperopt version installed by default
# as of 01/2018: it is best to install directly from the github repo with
# the command 'pip install git+https://github.com/hyperopt/hyperopt'


def play_trainset(hyperparameters, game_number):
    start_time = time.time()
    filename = "examples.gridphysics_2.expt_exploration_exploitation"

    level_game_pairs = None
    # Playing GVG-AI games
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

    gameName = gvggames[game_number]

    gvgname = "../gvgai/training_set_1/{}".format(gameName)

    gameString = read_gvgai_game('{}.txt'.format(gvgname))


    level_game_pairs = []
    for level_number in range(5):
    	with open('{}_lvl{}.txt'.format(gvgname, level_number), 'r') as level:
    		level_game_pairs.append([gameString, level.read()])

    #uncomment this line to run local games
    # gameName = filename

    agent = Agent('full', gameName, hyperparameters=hyperparameters)

    ##then pass this down for multiple episodes
    gameObject = None
    agent.playCurriculum(level_game_pairs=level_game_pairs)

    #and use this line
    # agent.playCurriculum(level_game_pairs=None)

    total_time = time.time() - start_time


    # Compute hyperopt loss
    # alpha = 1e3
    # beta = -1e6
    # loss = (beta * agent.levels_won) + (alpha * agent.total_game_steps) + agent.total_planner_steps

    return total_time

gvggames = ['aliens', 'boulderdash', 'butterflies', 'chase', 'frogs',  # 0-4
        	'missilecommand', 'portals', 'sokoban', 'survivezombies', 'zelda']  # 5-9

space = {
    'sprite_first_alpha': hp.loguniform('space_sprite_first_alpha', 1, 10),
    'sprite_second_alpha': hp.loguniform('space_sprite_second_alpha', 1, 10),
    'sprite_negative_mult': hp.loguniform('space_sprite_negative_mult', -2, 2),
    'multisprite_first_alpha': hp.loguniform('space_multisprite_first_alpha', 1, 10),
    'multisprite_second_alpha': hp.loguniform('space_multisprite_second_alpha', 1, 10),
    'novelty_first_alpha': hp.loguniform('space_novelty_first_alpha', 1, 10),
    'novelty_second_alpha': hp.loguniform('space_novelty_second_alpha_first_alpha', 1, 10)
}

def optimize_game(game_number):
    def play_game(hyperparameters):
        return play_trainset(hyperparameters=hyperparameters,
                             game_number=game_number)
    best = fmin(fn=play_game,
        space=space,
        algo=tpe.suggest,
        max_evals=10)

    gameName = gvggames[game_number]
    with open('{}.txt'.format(gameName), 'w') as f:
        f.write(str(best))

pool = ProcessingPool(nodes=10)
pool.map(optimize_game, range(10))
