
from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings
import pygame
import os
import numpy as np

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
		# import pdb; pdb.set_trace()
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
			if game_name == file.split('.txt')[0] or game_name == file.split('_lvl')[0]:
				if 'lvl' not in file: file_list['game'] = file
				else: 
					level = file.split('_lvl')[1][0]
					file_list[int(level)] = file

	# new_doc = ''
	# with open('all_games/{}'.format(file_list['game']), 'r') as f:
	# 	new_doc = []
	# 	g = _gen_color()
	# 	for line in f.readlines():
	# 		new_line = (" ".join([string if string[:4]!="img="
	# 			else "color={}".format(next(g))
	# 			for string in line.split(" ")]))
	# 		new_doc.append(new_line)
	# 	new_doc = "\n".join(new_doc)

	# import pdb; pdb.set_trace()
	with open('all_games/{}'.format(file_list['game']), 'r') as game:
		gameString = game.read()

	env_list = {}

	for lvl_idx in range(len(file_list.keys())-1):

		with open('all_games/{}'.format(file_list[lvl_idx]), 'r') as level:
			levelString = level.read()

		# import pdb; pdb.set_trace()

		env_list[lvl_idx] = _load_level(gameString, levelString)

	return env_list




























