
from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings
import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'
import pygame
import numpy as np
import pdb


def load_game(game_name, games_folder):
	def _load_level(gameString, levelString):

		headless = True

		rleCreateFunc = lambda: createRLInputGameFromStrings(gameString, levelString)

		# (self, gameDef, levelDef, observationType=OBSERVATION_GLOBAL, visualize=False, actionset=BASEDIRS, positions=None, **kwargs)

		rle = rleCreateFunc()
		# import pdb; pdb.set_trace()
		rle.visualize = True
		if headless:
			os.environ["SDL_VIDEODRIVER"] = "dummy"
		#pdb.set_trace()
		pygame.init()

		return rle

	def _gen_color():
		from vgdl.colors import colorDict
		color_list = colorDict.values()
		color_list = [c for c in color_list if c not in ['UUWSWF']]
		for color in color_list:
			yield color

	file_list = {}
	for file in os.listdir(games_folder):
		# import pdb; pdb.set_trace()
		if 'DS' not in file:

			if 'expt_ee' in game_name:
				if game_name in file:
					if 'lvl' not in file:
						level = file.split('desc_')[1][0]
						file_list['game_{}'.format(level)] = file
					else:
						level = file.split('_lvl')[1][0]
						file_list[int(level)] = file
			else:
				if game_name == file.split('.txt')[0] or game_name == file.split('_lvl')[0]:
					if 'lvl' not in file: 
						file_list['game'] = file
					else: 
						level = file.split('_lvl')[1][0]
						file_list[int(level)] = file

	# new_doc = ''
	# with open('{}/{}'.format(games_folder, file_list['game']), 'r') as f:
	# 	new_doc = []
	# 	g = _gen_color()
	# 	for line in f.readlines():
	# 		new_line = (" ".join([string if string[:4]!="img="
	# 			else "color={}".format(next(g))
	# 			for string in line.split(" ")]))
	# 		new_doc.append(new_line)
	# 	new_doc = "\n".join(new_doc)

	# import pdb; pdb.set_trace()

	if 'expt_ee' not in game_name:
		with open('{}/{}'.format(games_folder, file_list['game']), 'r') as game:
			gameString = game.read()

	env_list = {}

	num_levels = len(file_list.keys())-1
	if 'expt_ee' in game_name:
		num_levels = int(len(file_list.keys())/2)

	for lvl_idx in range(num_levels):

		if 'expt_ee' in game_name:

			with open('{}/{}'.format(games_folder, file_list['game_{}'.format(lvl_idx)]), 'r') as game:
				gameString = game.read()

		with open('{}/{}'.format(games_folder, file_list[lvl_idx]), 'r') as level:
			levelString = level.read()

		# import pdb; pdb.set_trace()

		env_list[lvl_idx] = _load_level(gameString, levelString)

	return env_list




























