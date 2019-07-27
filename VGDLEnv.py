
from vgdl import colors
import utils
import pygame
from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN, K_SPACE
from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings
import numpy as np
import imageio
from skimage.transform import resize
import pdb
# from core.env import env

#sample_usage

#self.env = VGDLEnv(game_path = '../gvgai_games/portals/')

#self.env.set_level(0)

#self.env.reset()

class VGDLEnv():

	def __init__(self, game_name, game_folder):

		self.game_name = game_name
		self.game_folder = game_folder

		self.env_list = utils.load_game(game_name, game_folder)

		self.lvl = 0

		self.set_level(self.lvl)

		self.actions = [0, K_RIGHT, K_LEFT, K_UP, K_DOWN, K_SPACE]

	def set_level(self, lvl):

		self.current_env = self.env_list[lvl]
		self.current_env.softReset()

	def step(self, action):

		# import pdb; pdb.set_trace()

		prev_score = self.current_env._game.score

		results = self.current_env.step(self.actions[action])

		# {'observation':observation, 'reward':reward, 'pcontinue':pcontinue, 'effectList':events, 'ended':ended, 'win':won, 'termination':termination}

		# import pdb; pdb.set_trace()

		reward = results['reward']

		ended = results['ended']

		win = results['win']

		# score = self.current_env._game.score

		# ended, win = self.current_env._isDone()

		# reward = score - prev_score

		# if reward > 0: import pdb; pdb.set_trace()

		return reward, ended, win

	def reset(self):

		self.env_list = utils.load_game(self.game_name, self.game_folder)
		self.set_level(self.lvl)
		self.current_env.softReset()

	def render(self, gif = False):

		game = self.current_env._game

		# import pdb; pdb.set_trace()

		# returns numpy array of pixel values based on the sprites in the game
		im = np.empty([game.screensize[1], game.screensize[0], 3], dtype=np.uint8)
		bg = np.array(colors.LIGHTGRAY, dtype=np.uint8) # background
		im[:] = bg

		for className in game.sprite_order:
			if className in game.sprite_groups:
				for sprite in game.sprite_groups[className]:
					r, c, h, w = sprite.rect.top , sprite.rect.left , sprite.rect.height , sprite.rect.width
					im[r:r+h, c:c+w, :] = np.array(sprite.color, dtype=np.uint8)

		if gif: im = resize(im, (64, 64, 3))
		return im











