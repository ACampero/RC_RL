from IPython import embed
import itertools
import numpy as np
from numpy import zeros
import pygame
from ontology import BASEDIRS
from core import VGDLSprite, colorDict, sys
from stateobsnonstatic import StateObsHandlerNonStatic
from rlenvironmentnonstatic import *
import argparse
import random
import math
from threading import Thread
from collections import defaultdict, deque
import time
# import ipdb
import copy
from threading import Lock
from Queue import Queue
from util import *
# import multiprocessing
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectObjectGoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, \
NoveltyRule, generateSymbolDict, ruleCluster, Theory, Game, writeTheoryToTxt, generateTheoryFromGame
from ontology import MovingAvatar, HorizontalAvatar, VerticalAvatar, FlakAvatar, AimedFlakAvatar, OrientedAvatar, \
	RotatingAvatar, RotatingFlippingAvatar, NoisyRotatingFlippingAvatar, ShootAvatar, AimedAvatar, \
		AimedFlakAvatar
from rlenvironmentnonstatic import createRLInputGame

# from line_profiler import LineProfiler
import cPickle

from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT
NONE = 0
ACTIONS = [K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, NONE]
actionDict = {K_SPACE: 'space', K_UP: 'up', K_DOWN: 'down', K_LEFT: 'left', K_RIGHT: 'right', NONE: 'wait'}

#############################################
# Sprites with 'DARKGRAY' colorName are not
# updated in _performAction and fastcopy
#############################################

## Base class for width-based planners (IW(k) and 2BFS)
class WBP():
	def __init__(self, rle, gameFilename, theory=None, fakeInteractionRules = [], seen_limits=[], annealing=1, max_nodes=100000, shortHorizon=False,
		firstOrderHorizon=False, conservative=False, hyperparameters={}, extra_atom=False, IW_k=2, display=False):
		self.rle = rle
		self.gameFilename = gameFilename
		self.hyperparameter_index = hyperparameters['idx']
		self.hyperparameters = dict((k, hyperparameters[k]) for k in hyperparameters.keys() if k not in ['idx'])
		self.T = len(rle._obstypes.keys())+1 #number of object types. Adding avatar, which is not in obstypes.
		self.vecDim = [rle.outdim[0]*rle.outdim[1], 2, self.T]
		self.trueAtoms = defaultdict(lambda:0) #set() ## set of atoms that have been true at some point thus far in the planner.
		self.objectTypes = rle._game.sprite_groups.keys()
		self.objectTypes.sort()
		self.phiSize = sum([len(rle._game.sprite_groups[k]) for k in rle._game.sprite_groups.keys() if k not in ['wall', 'avatar']])
		self.seen_limits = seen_limits
		self.IW_k = IW_k
		self.objIDs = {}
		self.solution = None
		self.trackTokens = False
		self.vecSize = None
		self.addWaitAction = True
		self.safeDistance = 3
		self.annealing = annealing
		self.statesEncountered = []
		self.padding = 5  ##5 is arbitrary; just to make sure we don't get overlap when we add positions
		self.objectNumberTrackingLimit = 200#50
		self.objectLocationTrackingLimit = 8
		self.max_nodes = max_nodes
		self.small_max_nodes = 100
		self.objectsWhoseLocationsWeIgnore = ['Flicker', 'Random']
		self.objectsWhosePresenceWeIgnore = ['Flicker']
		self.classesWhoseLocationsWeIgnore = []
		self.classesWhosePresenceWeIgnore = []
		self.allowRollouts = True
		self.quitting = False
		self.exhausted_novelty = True
		self.extra_atom = extra_atom
		self.gameString_array = []
		self.rolloutHyperparameters = dict([(k,v) if 'second' not in k else (k,0) for k,v in self.hyperparameters.items()])
		self.hypotenuse_squared = self.rle.outdim[0]**2 + self.rle.outdim[1]**2

		if theory == None:
			self.theory = generateTheoryFromGame(rle, alterGoal=False)
		else:
			self.theory=copy.deepcopy(theory)
			self.theory.interactionSet.extend(fakeInteractionRules)
			self.theory.updateTerminations()
	
		if any([t in str(s.vgdlType) for s in self.theory.spriteObjects.values() for t in ['Missile', 'Random', 'Chaser']]):
			movingTypesInGame = True
		else:
			movingTypesInGame = False
		
		if self.hyperparameter_index == 1:
			if movingTypesInGame:
				self.position_score_multiplier = -10
			else:
				self.position_score_multiplier = -1
		elif self.hyperparameter_index == 3:
			self.position_score_multiplier = -10
		else:
			print "Warning: haven't thought about position_score_multiplier for idx {}".format(self.hyperparameter_index)
			self.position_score_multiplier = -10

		#################
		#################
		self.display = False

		if self.display:
			print "In planner; MovingTypesInGame: {}. Planning with idx {} and position_multiplier {}".format(movingTypesInGame, self.hyperparameter_index, self.position_score_multiplier)


		if self.theory.classes['avatar'][0].args and 'stype' in self.theory.classes['avatar'][0].args:
			self.thingWeShoot = self.theory.classes['avatar'][0].args['stype']
		else:
			self.thingWeShoot = None

		self.boxes = []
		for rule in self.theory.interactionSet:
			if rule.interaction == 'bounceForward':
				self.boxes.append(rule.slot1)

		if self.display:
			print 'max nodes', self.max_nodes
			print "exta atom is {}".format(self.extra_atom)

		i=1
		for k in rle._game.all_objects.keys():
			self.objIDs[k] = i * 100 * (rle.outdim[0]*rle.outdim[1]+self.padding)
			i+=1

		self.pixel_size = self.rle._game.screensize[0]/self.rle._game.width
		self.visited_positions = np.zeros(np.array(self.rle._game.screensize)/
			self.pixel_size)

		self.killer_types = [inter.slot2 for inter in self.theory.interactionSet if inter.slot1=='avatar' and inter.interaction in ['killSprite']]
		if self.display and self.killer_types:
			print 'killer types', self.killer_types

		self.short_horizon = shortHorizon
		self.conservative = conservative
		self.winning_states = []
		self.total_nodes = 0

		self.getAvailableActions()
		if self.display:
			print "available actions:", self.actions

		if self.conservative:
			self.hyperparameters['sprite_negative_mult'] = 100
			if self.display:
				print "Planning conservatively. Switched sprite_negative_mult to {}".format(self.hyperparameters['sprite_negative_mult'])
		else:
			if self.display:
				print "Planning normally."
		## Ignore objects we don't want to track (i.e., non-moving immovables.)
		self.objectsToTrack = []
		for k in rle._game.sprite_groups.keys():
			if ((k in self.theory.classes.keys() and ('Resource' or 'Immovable') in str(self.theory.classes[k][0].vgdlType) and not \
			(('bounceForward' or 'pullWithIt') in [rule.interaction for rule in self.theory.interactionSet if k in [rule.slot1, rule.slot2]])) or
			len(rle._game.sprite_groups[k])>self.objectNumberTrackingLimit):
				pass# self.objectsToNotTrackInAtomList.append(k)
			else:
				self.objectsToTrack.append(k)

			## Don't track (in either way) objects that are very numerous; completely breaks calculateAtoms()
			## Also don't track projectiles we generate
			if (len(rle._game.sprite_groups[k])>self.objectNumberTrackingLimit) or k==self.thingWeShoot:
				self.classesWhosePresenceWeIgnore.append(k)
			if (len(rle._game.sprite_groups[k])>self.objectLocationTrackingLimit):
				self.classesWhoseLocationsWeIgnore.append(k)

		if self.display:
			print "ignoring presences for", self.classesWhosePresenceWeIgnore
			print "ignoring locations for", self.classesWhoseLocationsWeIgnore
		# Compute starting number of each SpriteCounter stype
		self.firstOrderHorizon = firstOrderHorizon
		self.starting_stype_n = {}
		for term in self.theory.terminationSet:
			if isinstance(term, SpriteCounterRule):
				stype = term.termination.stype
				objs = self.findObjectsInRLE(self.rle, stype)
				n_stypes = len(objs) if objs is not None else 0
				# n_stypes = len([0 for sprite in self.findObjectsInRLE(self.rle, stype) if self.findObjectsInRLE(self.rle, stype) is not None])
				self.starting_stype_n[stype] = n_stypes
			elif isinstance(term, MultiSpriteCounterRule):
				stypes = term.termination.stypes
				n_stypes = sum([len(self.findObjectsInRLE(self.rle, stype)) for stype in stypes if self.findObjectsInRLE(self.rle, stype)])
				self.starting_stype_n[tuple(stypes)] = n_stypes

	def findObjectsInRLE(self, rle, objName):
		try:
			objLocs = [rle._rect2pos(element.rect) for element in rle._game.sprite_groups[objName]
			if element not in rle._game.kill_list]
		except:
			# return None
			return []
		return objLocs

	def findAvatarInRLE(self, rle):
		avatar_loc = rle._rect2pos(rle._game.sprite_groups['avatar'][0].rect)
		return avatar_loc

	def getAvailableActions(self):
		## Note: if an object that isn't instantiated in the beginning is of a class that
		## spacebar applies to, we won't pick up on it here.
		# shootingClasses = ['MarioAvatar', 'ClimbingAvatar', 'ShootAvatar', 'Switch', 'FlakAvatar']
		# classes = [str(o[0].__class__) for o in self.rle._game.sprite_groups.values() if len(o)>0]
		# spacebarAvailable = False
		# for sc in shootingClasses:
		# 	if any([sc in c for c in classes]):
		# 		spacebarAvailable = True
		# 		break
		# embed()
		# if spacebarAvailable:
		# 	self.actions = [NONE, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT]
		# else:
		# 	self.actions = [NONE, K_UP, K_DOWN, K_LEFT, K_RIGHT]
		
		## get actions from avatar-type definition
		self.actions = self.rle._game.getAvatars()[0].declare_possible_actions().values()
		if self.conservative or self.addWaitAction:
			self.actions.append(NONE)
		self.actions = sorted(self.actions)		

		return

	def calculateAtoms(self, rle):
		lst = []

		## Track specific locations of objects
		
		kl_set = set(rle._game.kill_list)
		for k in self.objectsToTrack:
			## Don't track Flicker in atoms. The point is that the Flicker should have an effect on other objects, so atom novelty that would have been
			## a function of the Flicker's presence is being taken care of by that. Otherwise the agent can keep exploring states that have no actual effect
			## on the game state.
			if ((len(rle._game.sprite_groups[k])>0 and
					rle._game.sprite_groups[k][0].colorName in self.theory.spriteObjects.keys() and
					any([obj in str(self.theory.spriteObjects[rle._game.sprite_groups[k][0].colorName].vgdlType) for obj in self.objectsWhoseLocationsWeIgnore])) or
				k in self.classesWhoseLocationsWeIgnore) or k==self.thingWeShoot:

				pass
			else:
				for o in rle._game.sprite_groups[k]:
					if o not in kl_set:
						## turn location into vector position (rows appended one after the other.)
						# pos = rle._rect2pos(o.rect) #x,y
						# vecValue = pos[1] + pos[0]*rle.outdim[0] + 1
						pos = float(o.rect.left)/rle._game.block_size, float(o.rect.top)/rle._game.block_size
						vecValue = 10*pos[1] + 10*pos[0]*rle.outdim[0] + 10
					else:
						vecValue = 0
						# import ipdb; ipdb.set_trace()
					try:
						if k == rle._game.getAvatars()[0].stype:
							# Add avatar orientation to atom
							orientation = rle._game.sprite_groups[k][0].orientation
							if orientation[0] < 0 and orientation[1] == 0:
								vecValue += 0
							elif orientation[0] > 0 and orientation[1] == 0:
								vecValue += 100000
							elif orientation[0] == 0 and orientation[1] < 0:
								vecValue += 200000
							elif orientation[0] == 0 and orientation[1] > 0:
								vecValue += 300000
					except (IndexError, AttributeError) as e:
						pass

					objPosCombination = self.objIDs[o.ID] + vecValue
					# print("ObjId = {}, vecValue = {}".format(self.objIDs[o.ID], vecValue))
					lst.append(objPosCombination)

		## Track present/absent objects
		present = []
		for k in [t for t in self.objectTypes if t not in ['wall', 'avatar']]: ##maybe add the avatar to this global state

			## Don't track certain types (i.e., Flickers, Randoms) in atoms. The point is that the Flicker should have an effect on other objects, so atom novelty that would have been
			## a function of the Flicker's presence is being taken care of by that. Otherwise the agent can keep exploring states that have no actual effect
			## on the game state.
			if (len(rle._game.sprite_groups[k])>0 and
					rle._game.sprite_groups[k][0].colorName in self.theory.spriteObjects.keys() and
					any([obj in str(self.theory.spriteObjects[rle._game.sprite_groups[k][0].colorName].vgdlType) for obj in self.objectsWhosePresenceWeIgnore]) or
					k in self.classesWhosePresenceWeIgnore) or k==self.thingWeShoot:
				pass
			else:
				for o in sorted(rle._game.sprite_groups[k], key=lambda s:s.ID):
					if o not in kl_set:
						present.append(1)
					else:
						present.append(0)
		ind = sum([present[i]*2**i for i in range(len(present))])
		lst.append(ind)
		if not self.vecSize:
			self.vecSize = len(lst)
			# print "Vector is length {}".format(self.vecSize)

		if self.extra_atom:
			try:
				avatar_pos = self.findAvatarInRLE(rle)
				vecValue = avatar_pos[1] + avatar_pos[0]*rle.outdim[0] + 1
			except:
				vecValue = [0]

			stateIW1 = [vecValue] + rle.show_binary(self.thingWeShoot)
			lst.append(hash(tuple(stateIW1)))
		return set(lst)

	def compareDicts(self, d1,d2):
		## only tells us what is in d2 that isn't in d1, as well as differences in values between shared keys
		return [k for k in d2.keys() if (k not in d1.keys() or d1[k]!=d2[k])]

	def delta(self, node1, node2):
		if node1 is None:
			diff = node2.state
		else:
			diff = node2.state-node1.state
		return diff

	def noveltySelection(self, QNovelty, QReward):
		bestNodes = sorted(QNovelty, key=lambda n: (n.novelty, -n.intrinsic_reward))
		current = bestNodes.pop(0)
		QNovelty.remove(current)
		try:
			QReward.remove(current)
		except:
			pass
		return current

	def rewardSelection(self, QReward, QNovelty):
		## Use this for IW lesion
		# acceptableNodes = QReward
		#acceptableNodes = filter(lambda n: (not n.terminal or n.win), acceptableNodes)
		
		## normal mode
		# if not self.conservative:
		acceptableNodes = filter(lambda n: n.novelty<self.IW_k+1, QReward)
		## sort max to min for pop()
		bestNodes = sorted(acceptableNodes, key=lambda n: (-n.intrinsic_reward, n.novelty))
		# else:
		# 	acceptableNodes = QReward
		# 	## sort max to min for pop()
		# 	bestNodes = sorted(acceptableNodes, key=lambda n: (-n.intrinsic_reward, len(n.actionSeq)))
		# 	print "in conservative mode in reward selection"
		# 	# embed()



		try:
			
			current = bestNodes.pop(0)
			# print current.intrinsic_reward
			# current.ended, current.win = current.rle._isDone()
			if (current.terminal, current.win) == (True, False):
				print "rewardSelection picked a loss node!!"
				embed()
			if current.terminal and not current.win:
				print "rewardSelection picked a loss node!!"
				embed()
			if current.badOutcomes:
				print "picked a node with >0 badoutcomes"
				embed()
		except:
			# if self.display:
			print("RewardSelection didn't find a node that satisfied novelty criteria.")
			# embed()
			return 'pickMaxNode'
		
		# try:
		# 	for k,v in self.rle._game.getAvatars()[0].resources.items():
		# 		if v>3:
		# 			# print self.rle.show()
		# 			embed()
		# except:
		# 	pass

		QReward.remove(current)
		try:
			QNovelty.remove(current)
		except:
			pass

		return current

	"""
	def BFS_profiler(self):
		lp = LineProfiler()
		lp_wrapper = lp(self.BFS)
		lp_wrapper()
		lp.print_stats()
	"""

	def BFS(self):
		QNovelty, QReward = [], []
		visited, rejected = [], []
		start = Node(self.rle, self, [], None)
		start.rle = self.rle
		visited.append(start)
		start.eval()

		QNovelty.append(start)
		QReward.append(start)
		i=0

		while (len(QNovelty)>0 or len(QReward)>0) and i<self.max_nodes:
			if self.display and i%100==0:
				print "Searching node {}".format(i)

			current = self.rewardSelection(QReward, QNovelty)
			
			if current in [None, 'pickMaxNode']:

				if self.conservative:
					node = max(visited, key=lambda n:(n.intrinsic_reward, len(n.actionSeq)))
						# embed()
				else:
					if self.display:
						print "Failed to find a novel node. Quitting"
					node = start

				# if self.short_horizon:
				# 	if not self.conservative:
				# 		## Node has to have an actionseq
				# 		node = max(visited, key=lambda n:(n.actionSeq, n.intrinsic_reward))
				# 	else:
				# 		print "got pickMaxNode in shortHorizon"
				# 		embed()
				# 		node = max(visited, key=lambda n:(n.intrinsic_reward, len(n.actionSeq)))
				# 		# embed()
				# else:
				# 	if self.display:
				# 		print "Failed to find a novel node. Quitting"
				# 	node = start

				# node = max(visited, key=lambda n:(n.intrinsic_reward, len(n.actionSeq)))

				## if we didn't get any novelty-fulfilling nodes, just pick the best node 
				## sorted by reward and action-sequence length.
				# node = max(QReward, key=lambda n:(n.intrinsic_reward, len(n.actionSeq)))

				# parentNode = copy.deepcopy(node)
				parentNode = node
				self.solution = node.actionSeq

				if self.conservative and not self.solution:
					# print "in conservative mode. didn't get solution; trying to filter less aggressively"
					# print "you should never actually end up here"
					# embed()
					if QReward:
						node = max(QReward, key=lambda n:(n.intrinsic_reward, len(n.actionSeq)))
					else:
						## QReward only has nodes that didn't result in loss states. Return *some* plan here to make sure things don't break
						## This is a plan of taking a single 'wait' action.
						if self.display:
							print "QReward was empty -- returning a futile plan of a single 'none' action"
						start = Node(self.rle, self, [], None)
						start.rle = self.rle
						child = Node(self.rle, self, start.actionSeq+[0], start)
						child.eval()
						node = child

					# parentNode = copy.deepcopy(node)
					parentNode = node
					self.solution = node.actionSeq

				gameString_array, object_positions_array = [], []
				while parentNode is not None:
					gameString_array.append(parentNode.rle.show())
					object_positions_array.append(copy.deepcopy(parentNode.rle))
					parentNode = parentNode.parent
				self.gameString_array = gameString_array[::-1]
				self.object_positions_array = object_positions_array[::-1]

				self.quitting = True
				self.exhausted_novelty = True
				if self.display:
					print "was in None or PickMaxNode"
				# embed()
				return node, gameString_array, object_positions_array

			self.statesEncountered.append(current.rle._game.getFullState())

			# if self.display:
				# print current.rle.show(indent=True)

			try:
				(x, y) = np.array((current.rle._game.getAvatars()[0].rect.x,
					current.rle._game.getAvatars()[0].rect.y))/self.pixel_size
				self.visited_positions[x, y] += 1
			except IndexError:
				# print "adding to visited_positions failed"
				# embed()
				pass

			current.updateNoveltyDict(QNovelty, QReward)
			# embed()
			visited.append(current)

			current_actions = self.actions

			try:
				# If there's already a Missile on the screen
				# and the projectile class is a singleton
				# and the action chosen is shooting
				if (current.rle._game.getAvatars() and hasattr(current.rle._game.getAvatars()[0], 'stype') and
						'Missile' in str(self.theory.classes[current.rle._game.getAvatars()[0].stype][0].vgdlType) and
						self.findObjectsInRLE(current.rle, current.rle._game.getAvatars()[0].stype) and
						'singleton' in self.theory.classes[current.rle._game.getAvatars()[0].stype][0].args and
						bool(self.theory.classes[current.rle._game.getAvatars()[0].stype][0].args['singleton']) and
						len([s for s in current.rle._game.sprite_groups[current.rle._game.getAvatars()[0].stype] if s not in current.rle._game.kill_list])>0):
					current_actions = [0]
					avatar = current.rle._game.getAvatars()[0]
					killer_sprites = [s for k in self.killer_types for s in current.rle._game.sprite_groups[k]]
					if killer_sprites:
						nearest = findNearestSprite(avatar, killer_sprites)
						if manhattanDist(current.rle._rect2pos(avatar.rect), current.rle._rect2pos(nearest.rect))>3:
							current_actions = [0]
						else:
							current_actions = self.actions
							# current_actions = [0, K_LEFT, K_RIGHT, K_UP, K_DOWN]
							if self.display:
								print "didn't change current_actions; will plan normally"
								print "nearest dangerous sprite:", manhattanDist(current.rle._rect2pos(avatar.rect), current.rle._rect2pos(nearest.rect))

			except (IndexError, AttributeError, TypeError) as e:
				print "got triple except."
				embed()
				pass

			if self.display:
				print "________________"
				# try:
					# print current.rle._game.getAvatars()[0].resources
				# except:
					# print ""
				if current.actionSeq:
					print actionDict[current.actionSeq[-1]]
				print current.rle.show()
			# if self.killer_types:
				# embed()
			for a in current_actions:
				skipAction = False
				if not skipAction:
					child = Node(self.rle, self, current.actionSeq+[a], current)
					# print actionDict[a]
					child.eval()
					# embed()

					# print ""
					# ended, win = child.rle._isDone()
					ended, win = child.terminal, child.win
					# if a == K_SPACE:
						# embed()
					if self.firstOrderHorizon:
						# Return plan if first-order progress was made towards
						# a win condition
						foundWin = False
						for term in self.theory.terminationSet:
							if isinstance(term, SpriteCounterRule) and term.termination.win==True:
								stype = term.termination.stype
								n_stypes = len([0 for sprite in self.findObjectsInRLE(child.rle, stype)])
								if stype in self.starting_stype_n.keys() and self.starting_stype_n[stype] > n_stypes:
									if ended and not win:
										child.win, foundWin = False, False
									else:
										child.terminal = True
										child.win, foundWin = True, True
										if self.display:
											print "exiting early because progress was made toward", stype
											# embed()
							elif isinstance(term, MultiSpriteCounterRule) and term.termination.win==True:
								stypes = term.termination.stypes
								n_stypes = sum([len(self.findObjectsInRLE(child.rle, stype)) for stype in stypes if self.findObjectsInRLE(child.rle, stype)])
								if tuple(stypes) in self.starting_stype_n.keys() and self.starting_stype_n[tuple(stypes)] > n_stypes:
									if ended and not win:
										child.win, foundWin = False, False
									else:
										child.terminal = True
										child.win, foundWin = True, True
										if self.display:
											print "exiting early because progress was made toward", stypes
											# embed()
							if foundWin:
								break

					if child.win:
						# Get the gameString representation of the RLE at each
						# timestep in the chosen solution, so as to be able to
						# correct for stochasticity effects
						# compare it to the agent's RLE at execution time and
						self.winning_states.append(child)
						node = child
						gameString_array, object_positions_array = [], []
						while node is not None:
							gameString_array.append(node.rle.show(color='green'))
							object_positions_array.append(node.rle)
							node = node.parent
						self.gameString_array = gameString_array[::-1]
						self.object_positions_array = object_positions_array[::-1]
						ended, win, t = child.rle._isDone(getTermination=True)
						if self.display:
							# print child.rle.show()
							if t:
								print t.__dict__
							# embed()
							# embed()
						# if a == K_LEFT:
							# if t:
								# print t.__dict__
							# embed()
						self.solution = child.actionSeq
						self.statesEncountered.append(child.rle._game.getFullState())
						# if len(self.solution)>10:
							# print "got long solution. look into this."
							# embed()
						# if t:
							# print t.__dict__
						# if not child.rle._game.getAvatars():
							# print "Think we won but no avatars!?!?"
							# embed()
					else:
						if not (child.terminal and not child.win):
							QNovelty.append(child)
							QReward.append(child)
					# if current.rle._game.time==0 and self.killer_types:
						# print child.rle.show()
						# print child.terminal, child.win
						# embed()
			i+=1
			self.total_nodes = i

			if self.winning_states:
				# print "we have {} winning states".format(len(self.winning_states))
				# embed()
				bestNodes = sorted(self.winning_states, key=lambda n: (-n.intrinsic_reward))
				bestNode = bestNodes[0]
				if self.display:
					print "found winning states"
				return bestNode, gameString_array, object_positions_array

			# print i
		self.solution = []


		# if self.short_horizon:
		if self.conservative:
			if QReward:
				if self.display:
					print "In short-horizon mode; selecting highest-reward longest sequence"
				node = max(QReward, key=lambda n:(n.intrinsic_reward, len(n.actionSeq)))
			else:
				## QReward only has nodes that didn't result in loss states. Return *some* plan here to make sure things don't break
				## This is a plan of taking a single 'wait' action.
				if self.display:
					print "QReward was empty -- returning a futile plan of a single 'none' action"
				start = Node(self.rle, self, [], None)
				start.rle = self.rle
				child = Node(self.rle, self, start.actionSeq+[0], start)
				child.eval()
				node = child

			parentNode = node
			self.solution = node.actionSeq

			# print self.solution
			gameString_array, object_positions_array = [], []
			while parentNode is not None:
				gameString_array.append(parentNode.rle.show())
				object_positions_array.append(copy.deepcopy(parentNode.rle))
				parentNode = parentNode.parent
			self.gameString_array = gameString_array[::-1]
			self.object_positions_array = object_positions_array[::-1]
			# print "win"
			# embed()
			# print "in WBP, within short horizon"
			# embed()
			if not self.conservative and self.display:
				print "End of shorthorizon plan"
			elif self.conservative and self.display:
				print "End of conservative plan"
			# print "ended conservative plan"
			# embed()
			return node, gameString_array, object_positions_array
		# else:
		# 	if i>=self.max_nodes:
		# 		if self.display:
		# 			print "Got no plan after searching {} nodes".format(self.max_nodes)
		# print "reached end of BFS"
		# embed()
		return None, None, None

class Node():
	def __init__(self, rle, WBP, actionSeq, parent):
		self.rle = rle
		self.WBP = WBP
		self.actionSeq = actionSeq
		self.parent = parent
		self.state = {}
		self.candidates = set()
		self.novelty = None
		self.reward = None
		self.intrinsic_reward = 0
		self.metabolic_cost = 0
		self.children = None
		# self.lastState = None
		self.reconstructed=False
		self.expanded = False
		self.rolloutDepth = max(rle.outdim)
		if self.parent is not None:
			self.rolloutArray = parent.rolloutArray[1:]
		else:
			self.rolloutArray = []
		self.okOutcomes = None
		self.badOutcomes = None

	def fastcopy(self, rle):
		newRle = self.empty_copy(rle)
		for k,v in rle.__dict__.iteritems():
			ctype = str(type(getattr(rle,k)))
			if 'defaultdict' in ctype or 'dict' in ctype:
				newRle.__dict__[k] = v.copy()
			elif 'list' in ctype:
				newRle.__dict__[k] = v[:]
			else:
				newRle.__dict__[k] = v
		# newRle._obstypes = rle._obstypes.copy()
		# if hasattr(rle, '_gravepoints'):
		# 	newRle._gravepoints = rle._gravepoints.copy()
		# newRle.outdim = rle.outdim
		#ipdb.set_trace()
		# newRle.symbolDict = rle.symbolDict.copy()
		#newRle._other_types = rle._other_types[:]

		newRle._game = self.empty_copy(rle._game)
		ignoreKeys = ['spriteDistribution',
					  'object_token_spriteDistribution',
					  'spriteUpdateDict',
					  'movement_options',
					  'object_token_movement_options',
					  #'all_objects',
					  'uiud']
		sprite_attrs = ['ID', 'name','rect','x','y','orientation','stypes',
						'lastrect','lastmove','stypes', 'lastdisplacement',
						'speed','cooldown','direction','color','colorName']

		for k,v in rle._game.__dict__.iteritems():
			if k in ignoreKeys: continue

			ctype = str(type(getattr(rle._game,k)))

			if 'list' in ctype:
				if k != 'kill_list':
					newRle._game.__dict__[k] = v[:]
				else:
					newRle._game.kill_list = v[:]
			elif 'defaultdict' in ctype or 'dict' in ctype:
				if k != 'sprite_groups':
					newRle._game.__dict__[k] = ccopy(v)
				else:
					#print "embed"
					new_sprite_groups = defaultdict(list)
					for group_name, group in rle._game.sprite_groups.iteritems():
						for sprite in group:
							#embed()
							if sprite.colorName == 'DARKGRAY':
								new_sprite_groups[group_name].append(sprite)
							else:
								new_sprite = self.empty_copy(sprite)
								try:
									for attr in sprite.__dict__.keys():
										if hasattr(sprite, attr):
											setattr(new_sprite, attr, getattr(sprite, attr))
										#else:
											#print attr
									#new_sprite.__dict__ = sprite.__dict__.copy()
									setattr(new_sprite, 'resources', ccopy(sprite.__dict__['resources']))
								except:
									embed()
								new_sprite_groups[group_name].append(new_sprite)
					# newRle._game.sprite_groups = ccopy(v)
					newRle._game.sprite_groups = new_sprite_groups
			elif 'vgdl' in ctype:
				newRle._game.__dict__[k] = ccopy(v)
			else:
				setattr(newRle._game, k, ccopy(v))
		#newRle._game = ccopy(rle._game)
		return newRle

## when to trigger rollouts, if any
## rollout length
## repeating rollouts if death? e.g., are they optimistic?
## multiple samples??
	def metabolics(self, rle, events, action, n=10, mult=.3):

		# Computing reward unit to be used in metabolics
		sprite_first_alpha = self.WBP.hyperparameters['sprite_first_alpha']
		sprite_second_alpha = self.WBP.hyperparameters['sprite_second_alpha']
		sprite_negative_mult = self.WBP.hyperparameters['sprite_negative_mult']
		self.reward_unit = sprite_second_alpha
		if rle==None:
			rle = self.rle

		theory = self.WBP.theory
		for term in theory.terminationSet:
			if isinstance(term, SpriteCounterRule):
				self.compute_reward_unit(theory, term, term.termination.stype, rle,
					first_alpha=sprite_first_alpha, second_alpha=sprite_second_alpha,
					negative_mult=sprite_negative_mult)

		# print(self.reward_unit)

		# metabolic_cost = 1./n
		metabolic_cost = -.2 * self.reward_unit # multiplying second order incentive
		# if action==32:
		if action!=NONE or action!=32:
			metabolic_cost -= .0#1./n
			pass
		if action==32:
			metabolic_cost -= 0
		if len(events)>0:
			# metabolic_cost = .3
			if any([rle._game.sprite_groups['avatar'][0].ID in e and e[0]=='bounceForward' for e in events]):
				metabolic_cost = -.3 * self.reward_unit # multiplying second order incentive
				# metabolic_cost += .3#(1-1./n)*mult
				pass
			# if any([rle._game.sprite_groups['avatar'][0].ID in e and e[0]=='killSprite' for e in events]):
			# 	metabolic_cost += 0.3
		metabolic_cost = 0
		return metabolic_cost

	def rollout(self, Vrle, thingWeShoot):
		successfulRollout = False
		j=0
		while not successfulRollout:
			vrle = self.fastcopy(Vrle)
			# embed()
			potentialProjectiles = [s for s in vrle._game.sprite_groups[thingWeShoot] if vrle._game.sprite_groups[thingWeShoot] and s.lastmove==0]
			thingWeShot = potentialProjectiles[0] if potentialProjectiles else None
			# vrle = copy.deepcopy(Vrle)
			prevHeuristicVal = self.heuristics(vrle, **self.WBP.rolloutHyperparameters)
			rolloutArray = []
			i=0
			terminal, win = vrle._isDone()
			# print "in rollout"
			while i<self.rolloutDepth and thingWeShot not in vrle._game.kill_list and not terminal:
				a = random.choice([K_UP, K_DOWN, K_LEFT, K_RIGHT])
				# print a
				res = vrle.step(a, getTermination=True, getEffectList=True)
				# ended, win, t = res['ended'], res['win'], res['termination']
				if self.WBP.display:
					print vrle.show(indent=True, color='cyan')
				currHeuristicVal = self.heuristics(vrle, **self.WBP.rolloutHyperparameters)
				heuristicVal = currHeuristicVal-prevHeuristicVal
				rolloutArray.append(heuristicVal)
				prevHeuristicVal = currHeuristicVal
				terminal, win, t = vrle._isDone(getTermination=True)

				if self.WBP.firstOrderHorizon:
					# Return plan if first-order progress was made towards
					# a win condition
					foundWin = False
					for term in self.WBP.theory.terminationSet:
						if isinstance(term, SpriteCounterRule) and term.termination.win==True:
							stype = term.termination.stype
							n_stypes = len([0 for sprite in self.WBP.findObjectsInRLE(vrle, stype)])
							if stype in self.WBP.starting_stype_n.keys() and self.WBP.starting_stype_n[stype] > n_stypes:
								if not (terminal and not win):
									terminal, win = True, True
									if self.WBP.display:
										print "exiting rollout early because progress was made toward", stype
										# embed()
						elif isinstance(term, MultiSpriteCounterRule) and term.termination.win==True:
							stypes = term.termination.stypes
							n_stypes = sum([len(self.WBP.findObjectsInRLE(vrle, stype)) for stype in stypes if self.WBP.findObjectsInRLE(vrle, stype)])
							if tuple(stypes) in self.WBP.starting_stype_n.keys() and self.WBP.starting_stype_n[tuple(stypes)] > n_stypes:
								if not(terminal and not win):
									terminal, win = True, True
									if self.WBP.display:
										print "exiting rollout early because progress was made toward", stypes
										# embed()
						if win:
							break

				if terminal:
					try:
						if (t.name=='NoveltyTermination' and
								self.rle._game.getAvatars()[0].stype
								not in [t.s1, t.s2]):
							# If we have a novelty termination not involving
							# the projectile, ignore it
							terminal, win = False, False
						## if the thing we shot wasn't involved in any interaction, ignore it.
						if thingWeShot is not None:
							if not any([thingWeShot.ID in e for e in res['effectList']]):
								# print "ignoring rollout termination because it didn't have to do with our recent projectile"
								terminal, win = False, False
						if terminal:
							if self.WBP.display:
								print t.name, t.s1, t.s2
					except (IndexError, AttributeError) as e:
						# Avatar is dead or doesn't have projectile
						pass
				i+=1
			## we want optimistic estimates of the future value of a shot.
			## Take up to 100 samples but don't get caught in an infinite loop.
			if terminal and not win and j<100:
				successfulRollout = False
				# print "rolling out again"
				j+=1
				# embed()
			else:
				successfulRollout = True
		# print sum(rolloutArray)
		# embed()
		if win:
			if self.WBP.display:
				print "rolloutwin"
			# embed()
			self.terminal = terminal
			self.win = win
		return rolloutArray

	def compute_reward_unit(self, theory, term, stype, rle, first_alpha=10000.,
						  second_alpha=100, negative_mult=.1):

		# First order: progress in terms of number of sprites remaining.
		# Second order: distance to the closest instance of a target sprite type.

		# theory.interactionSet[16].generic=False
		# theory.interactionSet[16].rule='killSprite'
		# reward unit to be used for metabolic_cost; gets updated on compute_second_order

		val = 0
		compute_second_order = True

		# Check if condition is win or loss and multiply accordingly
		if term.termination.win:
			mult = -1
		else:
			compute_second_order = False
			mult = negative_mult

		# Get all types that kill or transform stype (the target)
		killer_types = [
			inter.slot2 for inter in theory.interactionSet
			if ((inter.interaction == 'killSprite' or
				 inter.interaction == 'transformTo') and
				 not inter.generic and
				 not inter.preconditions
				and inter.slot1 == stype)]

		## If you can shoot a Flicker, give yourself credit for being close to things it kills, but remove credit for that Flicker being close to those things.
		try:
			if rle._game.getAvatars() and hasattr(rle._game.getAvatars()[0], 'stype') and rle._game.getAvatars()[0].stype in killer_types:
				if rle._game.getAvatars()[0].stype in theory.classes:
					color = theory.classes[rle._game.getAvatars()[0].stype][0].color
				else:
					color = rle._game.sprite_groups[rle._game.getAvatars()[0].stype][0].colorName

				if 'Flicker' in str(theory.spriteObjects[color].vgdlType):
					killer_types.append(rle._game.getAvatars()[0].name)
					killer_types.remove(rle._game.getAvatars()[0].stype)

		except (IndexError, AttributeError) as e:
			# print "got exception in trying to assign Flicker bonus to avatar"
			pass

		# This list comprehension checks whether the avatar kills the stype with a preconditioned
		# interaction, and if so adds 'avatar' to the list as well as the precondition for that rule
		avatar_preconditions = [
			(inter.slot2, inter.preconditions) for inter in theory.interactionSet
			if ((inter.interaction == 'killSprite' or
				inter.interaction == 'killIfOtherHasMore' or
				 inter.interaction == 'transformTo') and
				 not inter.generic
				 and inter.preconditions
				and inter.slot1 == stype)]

		tmp_list = []

		## If we have preconditions, find the objects that we should go to given that we satisfy the relevant preconditions. E.g., if we have a key and want to know what
		## happens with item x, go to it.
		for avatar in avatar_preconditions:

			precondition = list(avatar[1])[0]
			item, num, negated, operator_name = precondition.item, precondition.num, precondition.negated, precondition.operator_name
			if negated:
				oppositeOperatorMap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}
				true_operator = oppositeOperatorMap[operator_name]
			else:
				true_operator = operator_name
			try:
				current_resource = rle._game.sprite_groups[avatar[0]][0].resources[precondition.item]
				## If we satisfy the precondiiton, append to tmp_list, then to killer_types (meaning we are capable of killing stype now)
				if eval("{}{}{}".format(current_resource, true_operator, num)):
					# if self.WBP.display:
						# print "reached resource limit"
						# embed()
					tmp_list.append(avatar)
			except (IndexError, KeyError) as e:
				pass

		for t in tmp_list:
			killer_types.append(t[0])
			avatar_preconditions.remove(t)

		else:
			## Normal case
			n_stypes = len([0 for sprite in self.WBP.findObjectsInRLE(rle, stype)]) if self.WBP.findObjectsInRLE(rle, stype) else 0


		# print "stype, n_stypes, distance_to_goal, val", stype, n_stypes, distance_to_goal, val
		if compute_second_order:
			## Get all positions of objects whose type is in killer_types; compute minimum distance
			## of each to the stypes we have to destroy. Return min over all mins.
			# embed()
			objs = [self.WBP.findObjectsInRLE(rle, ktype) for ktype in killer_types]
			objs = [obj for obj in objs if obj]

			try:
				if len(objs)>0:
					kill_positions = np.concatenate([o for o in objs if len(o)==max([len(obj) for obj in objs])])
				else:
					kill_positions = np.array(objs)
			except:
				import ipdb; ipdb.set_trace()

			possiblePairList = []
			stype_positions = self.WBP.findObjectsInRLE(rle, stype)
			try:
				# A consequence of the two-way generic interactions in the
				# theory is that minimum-distance object pairs whose interactions
				# were not yet observed will have their distance penalized twice
				# as much when none of those objects is an avatar. This implies
				# that avatar novel interactions will be favored over other ones
				possiblePairList = [manhattanDist(obj, pos)
					 for pos in kill_positions
					 for obj in stype_positions]

				distance = min(possiblePairList)
				# print("second order distance is {}".format(distance))
			except (ValueError, TypeError) as e:
				distance = 100

			if possiblePairList:
				n_sprites = len(possiblePairList) ## TODO: you're normalizing by the number of possible pairs of killer_sprites and target_sprites; you should just normalize by the number of targets
				# Normalize by number of sprites, enforcing a prior that encourages
				# goals that involve killing fewer objects
				self.reward_unit = min(self.reward_unit, abs(float(mult*second_alpha)/n_sprites**2))
				# print("reward unit is {}".format(self.reward_unit))

	def spritecounter_val(self, theory, term, stype, rle, first_alpha=10000.,
						  second_alpha=100, negative_mult=.1, surrogate_multisprite_counter=False):

		# First order: progress in terms of number of sprites remaining.
		# Second order: distance to the closest instance of a target sprite type.

		# theory.interactionSet[16].generic=False
		# theory.interactionSet[16].rule='killSprite'
		# reward unit to be used for metabolic_cost; gets updated on compute_second_order

		val = 0
		if not surrogate_multisprite_counter:
			compute_second_order = True
		else:
			compute_second_order = False

		# Check if condition is win or loss and multiply accordingly
		if term.termination.win:
			mult = -1
		else:
			# embed()
			# compute_second_order = False if not self.WBP.conservative else True
			mult = negative_mult

		# Get all types that kill or transform stype (the target)
		killer_types = [
			inter.slot2 for inter in theory.interactionSet
			if (inter.interaction in ['killSprite', 'transformTo', 'collectResource'] and
				 not inter.generic and
				 not inter.preconditions
				and inter.slot1 == stype)]

		## If you can shoot a Flicker, give yourself credit for being close to things it kills, but remove credit for that Flicker being close to those things.
		try:
			if rle._game.getAvatars()[0].stype in killer_types:
				if rle._game.getAvatars()[0].stype in theory.classes:
					color = theory.classes[rle._game.getAvatars()[0].stype][0].color
				else:
					color = rle._game.sprite_groups[rle._game.getAvatars()[0].stype][0].colorName

				if 'Flicker' in str(theory.spriteObjects[color].vgdlType) or 'Missile' in str(theory.spriteObjects[color].vgdlType):
					killer_types.append(rle._game.getAvatars()[0].name)
					killer_types.remove(rle._game.getAvatars()[0].stype)
					# print "made killer_type transition"

		except (IndexError, AttributeError) as e:
			# print "got exception in trying to assign Flicker bonus to avatar"
			pass

		# This list comprehension checks whether the avatar kills the stype with a preconditioned
		# interaction, and if so adds 'avatar' to the list as well as the precondition for that rule
		avatar_preconditions = [
			(inter.slot2, inter.preconditions) for inter in theory.interactionSet
			if (inter.interaction in ['killSprite', 'killIfOtherHasMore', 'transformTo']  and
				 not inter.generic
				 and inter.preconditions
				and inter.slot1 == stype)]

		tmp_list = []

		## If we have preconditions, find the objects that we should go to given that we satisfy the relevant preconditions. E.g., if we have a key and want to know what
		## happens with item x, go to it.
		for avatar in avatar_preconditions:

			precondition = list(avatar[1])[0]
			item, num, negated, operator_name = precondition.item, precondition.num, precondition.negated, precondition.operator_name
			if negated:
				oppositeOperatorMap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}
				true_operator = oppositeOperatorMap[operator_name]
			else:
				true_operator = operator_name
			try:
				current_resource = rle._game.sprite_groups[avatar[0]][0].resources[precondition.item]
				## If we satisfy the precondiiton, append to tmp_list, then to killer_types (meaning we are capable of killing stype now)
				if eval("{}{}{}".format(current_resource, true_operator, num)):
					# if self.WBP.display:
						# print "reached resource limit"
					tmp_list.append(avatar)
			except (IndexError, KeyError) as e:
				# print "print problem in avatar preconditions in spritecounter_val"
				pass

		for t in tmp_list:
			killer_types.append(t[0])
			avatar_preconditions.remove(t)

		# Get attributes from terminationSet
		limit = term.termination.limit

		if 'SpawnPoint' in str(theory.classes[stype][0].vgdlType) and not killer_types:
			distance_to_goal = 0
			## Special case, where you want to track whether that spawnPoint has a limit, etc.
			## Distance to goal here is how many sprites the spawnPoint still has to shoot before it expires.
			for o in rle._game.sprite_groups[stype]:
				distance_to_goal += abs(o.total-o.counter)
			val += mult * first_alpha * distance_to_goal
			return val
		else:
			## Normal case
			n_stypes = len([0 for sprite in self.WBP.findObjectsInRLE(rle, stype)]) if self.WBP.findObjectsInRLE(rle, stype) else 0
			distance_to_goal = abs(n_stypes - limit)
			# print("distance to goal {} is {}".format(stype, distance_to_goal))

		if distance_to_goal!=0:
			val -= float(mult * first_alpha) / distance_to_goal ## Penalize quadratically for classes for which we'd have to kill many instances.
		else:
			val -= mult*first_alpha ## we shouldn't go in here, as if we've actually destroyed the relevant sprite we'll trigger a win condition.

		# if self.WBP.theory.classes[stype][0].color=='SCJPNE':
			# print "stype, n_stypes, distance_to_goal, val", stype, n_stypes, distance_to_goal, val
		if compute_second_order:
			## Get all positions of objects whose type is in killer_types; compute minimum distance
			## of each to the stypes we have to destroy. Return min over all mins.
			# embed()
			objs = [self.WBP.findObjectsInRLE(rle, ktype) for ktype in killer_types]
			objs = [obj for obj in objs if obj]

			try:
				if len(objs)>0:
					kill_positions = np.concatenate([o for o in objs if len(o)==max([len(obj) for obj in objs])])
				else:
					kill_positions = np.array(objs)
			except TypeError:
				kill_positions = np.array([])

			possiblePairList = []
			stype_positions = self.WBP.findObjectsInRLE(rle, stype)
			try:
				# A consequence of the two-way generic interactions in the
				# theory is that minimum-distance object pairs whose interactions
				# were not yet observed will have their distance penalized twice
				# as much when none of those objects is an avatar. This implies
				# that avatar novel interactions will be favored over other ones
				possiblePairList = [manhattanDist(obj, pos)
					 for pos in kill_positions
					 for obj in stype_positions]

				distance = min(possiblePairList)
				# print("second order distance is {}".format(distance))
			except (ValueError, TypeError) as e:
				distance = 100

			# if objs and stype=='avatar':
				# embed()
			if possiblePairList:
				n_sprites = len(stype_positions) if stype!='avatar' else 20
				# n_sprites = len(possiblePairList) ## TODO: you're normalizing by the number of possible pairs of killer_sprites and target_sprites; you should just normalize by the number of targets
				# Normalize by number of sprites, enforcing a prior that encourages
				# goals that involve killing fewer objects
				# val += max(self.WBP.rle.outdim[0],self.WBP.rle.outdim[1])*float(mult * second_alpha * distance**2)/(n_sprites**2 * self.WBP.hypotenuse_squared)
				added_val = float(mult * second_alpha * distance)/n_sprites**2
			elif stype!='avatar':
				# This helps in cases in which either the stype or the killer_type is not always on the screen
				# Then, you should not be disincentivized to create it, which can be achieved through this high penalty
				# print "didn't find pair list"
				# if stype=='c5':
					# embed()
				distance = 101
				added_val = float(mult * second_alpha * distance)
			elif not stype_positions:
				## If we couldn't compute a second-order distance because the avatar is dead, give infinite penalty.
				added_val = -float('inf')
			else:
				added_val = 0.
			# if self.WBP.theory.classes[stype][0].color=='BROWN':
				# added_val=0
			val += added_val
			# if self.WBP.theory.classes[stype][0].color=='BLUE':
				# print "found blue: {}".format(added_val)
				# embed()
			# if self.WBP.conservative:
				# print distance, val
				# print rle.show()

			if avatar_preconditions:
				avatars = [self.WBP.findObjectsInRLE(rle, ktype[0]) for ktype in avatar_preconditions]

				resource_names = [list(resource[1])[0].item for resource in avatar_preconditions]

				resource_yielder_names = [[inter.slot2 if (inter.interaction=='changeResource' and inter.args['resource']==res) else 
						inter.slot1 if (inter.interaction=='collectResource' and res==inter.args['resource']==res) else None
						for inter in theory.interactionSet] for res in resource_names]

				resource_yielder_names = [[r for r in ryn if r] for ryn in resource_yielder_names] ## Remove 'None' yielded by last else condition above

				try:
					resource_positions = [np.concatenate([self.WBP.findObjectsInRLE(rle, yielder) for yielder in yielders]) for yielders in resource_yielder_names]
				except:
					print "problem with resource positions"
					resource_positions = []
					# embed()

				resource_limits = np.array([list(resource[1])[0].num + 1
					if list(resource[1])[0].operator_name == '>'
					else list(resource[1])[0].num
					for resource in avatar_preconditions])
				try:
					avatar_resource_quantities = np.array([rle._game.getAvatars()[0].resources[res] for res in resource_names])
				except IndexError:
					avatar_resource_quantities = np.array([0 for res in resource_names])
				precondition_distances = []
				try:
					for (obj1_positions, obj2_positions) in zip(avatars, resource_positions):
						# A consequence of the two-way generic interactions in the
						# theory is that minimum-distance object pairs whose interactions
						# were not yet observed will have their distance penalized twice
						# as much when none of those objects is an avatar. This implies
						# that avatar novel interactions will be favored over other ones
						try:
							possiblePairList = np.array([manhattanDist(obj1, obj2)
								for obj1 in obj1_positions
								for obj2 in obj2_positions])
						except:
							pass
							# print "failure with obj1_positions"
							# embed()

						precondition_distances.append(min(possiblePairList))

					# effective_distance = min(precondition_distances/(resource_limits-avatar_resource_quantities))
					physical_distance = min(precondition_distances)
					sprite_n_distance = abs(resource_limits-avatar_resource_quantities)
					# Normalize by number of sprites, enforcing a prior that encourages
					# goals that involve killing fewer objects
					val += float(mult * second_alpha * (physical_distance / 10.)) - 10000
					val += float(mult * second_alpha * sprite_n_distance) - 10000
					# print "doing precondition stuff for resources"
					# embed()

					# print distance
				except (ValueError, TypeError) as e:
					# if avatar_preconditions and avatars[0]:
						# print "valueError in spritecounter_val"
						# embed()
					pass
					# effective_distance = 0

				if not resource_positions:
					# print "didn't find resource positions"
					# This helps in cases in which either the stype or the killer_type is not always on the screen
					# Then, you should not be disincentivized to create it, which can be achieved through this high penalty
					distance = 100
					val += float(mult * second_alpha * distance) - 20000

		## empty space bonus
		# if stype!='avatar':
		# 	openspace_bonus = 0
		# 	locs = self.WBP.findObjectsInRLE(rle, stype)
		# 	for loc in locs:
		# 		transformedLoc = (loc[0]*30, loc[1]*30)
		# 		neighbors = [(transformedLoc[0]+30, transformedLoc[1]), (transformedLoc[0]-30, transformedLoc[1]), (transformedLoc[0], transformedLoc[1]+30), (transformedLoc[0], transformedLoc[1]-30)]
		# 		for neighbor in neighbors:
		# 			if neighbor not in rle._game.positionDict.keys() or (any([n.name in killer_types for n in rle._game.positionDict[neighbor]])):
		# 				openspace_bonus += 1
		# 	# if openspace_bonus==2:
		# 		# embed()
		# 	# print "openspace_bonus: {}".format(openspace_bonus*1000)
		# 	if locs:
		# 		val += (openspace_bonus*1000)/len(locs)**2
		return val

	def multispritecounter_val(self, theory, term, rle, first_alpha=10000,
							   second_alpha=100):
		##WARNING: THis will only work if term.termination.limit==0. Otherwise you could
		## end up with, say, count(stype)==1 for each constituent stype, meaning the terminations
		## would all be fulfilled, even though sum([count(stype) for stype in stypes]) != 1.

		val = 0
		# print "in multispritecounter"
		# embed()
		for stype in term.termination.stypes:
			val += self.spritecounter_val(theory, term, stype, rle,
				first_alpha=first_alpha, second_alpha=second_alpha, surrogate_multisprite_counter=True)
		val /= len(term.termination.stypes)#**2
			# print stype, val
		# val /= 10**len(term.termination.stypes)**2
		return val

	def noveltytermination_val(self, theory, term, s1, s2, rle, first_alpha=1000,
						  second_alpha=10):
		val = 0
		compute_second_order = True

		# print term.termination.win, term.termination.args
		# Check if condition is win or loss and multiply accordingly
		if term.termination.win:
			mult = -1
		else:
			compute_second_order = False
			mult = 1

		# ## Don't give heuristic bonus for using the flicker. But the agent is still incentivized to try to make the flicker interact with other objects
		# ## because of noveltyTerminationConditions.
		# if 'Flicker' in str(theory.classes[s1][0].vgdlType) or 'Flicker' in str(theory.classes[s2][0].vgdlType):
		# 	return 0, 10000

		## If the terminationRule is precondition-dependent, check that first. Don't give heuristic val if the preconditions aren't fulfilled.
		if term.termination.args:
			item, num, negated, operator_name = term.termination.args.item, term.termination.args.num, term.termination.args.negated, term.termination.args.operator_name
			if negated:
				oppositeOperatorMap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}
				true_operator = oppositeOperatorMap[operator_name]
			else:
				true_operator = operator_name

			try:
				resource_str = str(rle._game.getAvatars()[0].resources[item])
			except IndexError:
				# print "checking whether avatar preconditions are fulfilled"
				# print rle._game.getAvatars()[0].resources
				# return 0, 10000
				return 2 * mult * first_alpha, 10000

			if not eval(resource_str+true_operator+str(num)):
				# print "checking whether avatar preconditions are fulfilled 2"
				# print rle._game.getAvatars()[0].resources		
				return 0, 10000		
				# return 2 * mult * first_alpha, 10000



		if compute_second_order:
			## Get all positions of objects whose type is in killer_types; compute minimum distance
			## of each to the stypes we have to destroy. Return min over all mins.
			# if 'Flicker' in str(theory.classes[s1][0].vgdlType) or 'Flicker' in str(theory.classes[s2][0].vgdlType):
				# embed()

			###JOAO
			if 'Flicker' in str(theory.classes[s1][0].vgdlType) and not ('Flicker' in str(theory.classes[s2][0].vgdlType) or s2=='avatar'):
				s1 = s2
				s2 = 'avatar'
				#print("replaced flicker with avatar")

			s2_positions = self.WBP.findObjectsInRLE(rle, s2)
			s1_positions = self.WBP.findObjectsInRLE(rle, s1)

			# Second order lesion
			# if s1 != 'avatar' and s2 != 'avatar':
			# 	return 0, 10000
		
			## Don't return a value for novelty for the mere existence of a projectile that has novelty bonuses
			if self.WBP.thingWeShoot in theory.classes and self.WBP.thingWeShoot in [s1,s2]: #and 'Flicker' not in str(theory.classes[self.WBP.thingWeShoot][0].vgdlType)
				return 0, 10000

			n_sprites = len(s1_positions) if s1_positions else 0
			possiblePairList = []
			try:
				# A consequence of the two-way generic interactions in the
				# theory is that minimum-distance object pairs whose interactions
				# were not yet observed will have their distance penalized twice
				# as much when none of those objects is an avatar. This implies
				# that non-avatar novel interactions will be favored over others

				possiblePairList = [manhattanDist(obj, pos)
					 for pos in s2_positions
					 for obj in s1_positions
					 if manhattanDist(obj, pos) != 0]
				distance = min(possiblePairList)
					 # This is a trick to avoid getting distance 0 for objects
					 # of same type. If the list turns out to be empty, it will
					 # raise an error and set the distance to 0
				# print distance
			except (ValueError, TypeError) as e:
				distance = 0

			if possiblePairList:
				n_sprites = len(possiblePairList)
				# Normalize by number of sprites, enforcing a prior that encourages
				# goals that involve killing fewer objects
				# val += 100*(float(mult * second_alpha * distance**2)/(n_sprites**2 * self.WBP.hypotenuse_squared)) + second_alpha * max(self.rle.outdim[0], self.rle.outdim[1])
				
				## as you get closer to the item, this quantity increases, contributing to a higher overall score
				# val += float(mult * second_alpha * distance)/n_sprites**2 + second_alpha * max(self.rle.outdim[0], self.rle.outdim[1])
				val += float(mult * second_alpha * distance /max(self.rle.outdim[0], self.rle.outdim[1]) )#/n_sprites**2 #+ second_alpha * max(self.rle.outdim[0], self.rle.outdim[1])

		# if s1=='c6' and s2=='avatar':
			# print "novelty val for {}, {}: {}".format(s1, s2, val)
			# embed()
		# if term.termination.args and s1=='c4' and s2=='avatar' and self.rle._game.getAvatars():
		# 	for k,v in self.rle._game.getAvatars()[0].resources.items():
		# 		if v>3:
		# 			print "distance: {}. subtractand: {}".format(distance, (float(mult * second_alpha * distance)/n_sprites**2))
		# 			print "novelty val for {}, {}: {}".format(s1, s2, val)
		# 			print rle.show()
					# embed()

			# embed()
		if n_sprites==0:
			return val, 10000
		return val, distance*n_sprites

	def timeout_val(self, theory, term, rle):
		val = 0
		limit = term.termination.limit

		# Check if condition is win or loss and multiply accordingly
		if term.termination.win:
			mult = -1
		else:
			mult = 1

		time_elapsed = rle._game.time
		distance_to_goal = abs(time_elapsed - limit)

		val += mult * distance_to_goal

		return val

	def heuristics(self, rle=None, sprite_first_alpha=10000.,
		sprite_second_alpha=100, sprite_negative_mult=.1,
		multisprite_first_alpha=10000, multisprite_second_alpha=100,
		novelty_first_alpha=1000, novelty_second_alpha=10, time_alpha=10):
		if rle==None:
			rle = self.rle
		# print rle.show()
		theory = self.WBP.theory
		heuristicVal = 0
		avatarNoveltyVals = []
		for term in theory.terminationSet:
			if isinstance(term, SpriteCounterRule):
				spritecounter_val = self.spritecounter_val(theory, term, term.termination.stype, rle,
					first_alpha=sprite_first_alpha, second_alpha=sprite_second_alpha,
					negative_mult=sprite_negative_mult)
				# if spritecounter_val!=0:
					# print("spritecounter_val for {} is equal to {}".format(
						# term.termination.stype, spritecounter_val))
				heuristicVal += spritecounter_val

			elif isinstance(term, MultiSpriteCounterRule):
				multispritecounter_val = self.multispritecounter_val(theory, term, rle,
						first_alpha=multisprite_first_alpha, second_alpha=multisprite_second_alpha)  #500, 5 (normally)
				# if multispritecounter_val!=0:
					# print("multispritecounter_val for {} is equal to {}".format(
						# term.termination.stypes, multispritecounter_val))
				heuristicVal += multispritecounter_val

			elif isinstance(term, TimeoutRule):
				timeout_val = time_alpha * \
					self.timeout_val(theory, term, rle)
				heuristicVal += timeout_val

			elif isinstance(term, NoveltyRule):
				noveltytermination_val, ranking = self.noveltytermination_val(
					theory, term, term.termination.s1, term.termination.s2, rle,
					first_alpha=novelty_first_alpha, second_alpha=novelty_second_alpha)
				# if noveltytermination_val!=0:
					# print("noveltytermination_val for {} and {} is equal to {}".format(
						# term.termination.s1, term.termination.s2, noveltytermination_val))

				# if self.parent and self.parent.rle._game.score==0 and term.termination.args and term.termination.s1=='c6' and term.termination.s2=='avatar' and noveltytermination_val!=-5000:
					# ipdb.set_trace()
				if 'avatar' == term.termination.s2:
					avatarNoveltyVals.append([self.WBP.annealing*noveltytermination_val,
						ranking])
				else:
					heuristicVal += self.WBP.annealing * noveltytermination_val
					## Lesions
					## Warning: the tweak below for lesions isn't correct; it doesn't deal with the way we process novelty for avatar 
					## terminations (involving avatarNoveltyVals -- see below)
					# Exploit only
					# heuristicVal += 0 * self.WBP.annealing * noveltytermination_val
					# Explore only
					# heuristicVal += 1000 * self.WBP.annealing * noveltytermination_val

		# boxpenalty = 0
		# for boxType in self.WBP.boxes:
		# 	locs = self.WBP.findObjectsInRLE(rle, boxType)
		# 	for loc in locs:
		# 		transformedLoc = (loc[0]*30, loc[1]*30)
		# 		neighbors = [(transformedLoc[0]+30, transformedLoc[1]), (transformedLoc[0]-30, transformedLoc[1]), (transformedLoc[0], transformedLoc[1]+30), (transformedLoc[0], transformedLoc[1]-30)]
		# 		for neighbor in neighbors:
		# 			if neighbor in rle._game.positionDict.keys() and any([n.colorName=='DARKGRAY' for n in rle._game.positionDict[neighbor]]):
		# 				boxpenalty += 1

		if avatarNoveltyVals:
			# print "chosen avatar novelty val", min(avatarNoveltyVals, key= lambda x: x[1])[0]
			heuristicVal += min(avatarNoveltyVals, key= lambda x: x[1])[0]
		
		# boxpenalty_mult = 50
		# print "boxpenalty", boxpenalty*boxpenalty_mult
		# print "position", self.position_score(self.WBP.position_score_multiplier) 
		# print "game score", self.rle._game.score
		# print "sum:", heuristicVal+self.position_score(self.WBP.position_score_multiplier) + self.rle._game.score #+ boxpenalty*boxpenalty_mult
		# if self.actionSeq:
			# print actionDict[self.actionSeq[-1]]
		# print rle.show()

		# resource_bonus = 0
		# if self.parent:
		# 	try:
		# 		for k,v in self.rle._game.getAvatars()[0].resources.items():
		# 			if v==1 and self.parent.rle._game.getAvatars()[0].resources[k]==0:
		# 				resource_bonus += 5000
		# 	except:
		# 		pass
		# if resource_bonus>0:
		# 	print "got resource bonus"
		# 	print  "sum with resource: ", heuristicVal + self.position_score(self.WBP.position_score_multiplier) + self.rle._game.score + resource_bonus
		# 	embed()

		# self.intrinsic_reward = self.rle._game.score + self.heuristicVal + \
		# sum(self.rolloutArray) - self.metabolic_cost + self.(-250)

		# heuristicVal += sum(self.rolloutArray)

		# heuristicVal += self.rle._game.score*abs(heuristicVal)
		
		return heuristicVal

	def position_score(self, factor=1.):
		try:
			(x, y) = np.array((self.rle._game.getAvatars()[0].rect.x,
				self.rle._game.getAvatars()[0].rect.y))/self.WBP.pixel_size
			# print factor * self.WBP.visited_positions[x, y]
			return factor * self.WBP.visited_positions[x, y]**2
		except IndexError:
			# print "index error in position score"
			return 0

	def empty_copy(self, obj):
		class Empty(obj.__class__):
			def __init__(self): pass
		newcopy = Empty()
		newcopy.__class__ = obj.__class__
		return newcopy

	def getToCurrentState(self):
		if self.parent and self.parent.rle is not None:
			## try to copy parent lastState. Then take action and store as current lastState.
			## if that fails, replay from beginning and store as current lastState
			try:
				multipleSamples = False
				vrle = self.fastcopy(self.parent.rle)
				# vrle = copy.deepcopy(self.parent.rle)
				
				# if self.WBP.killer_types:
				# 	for k in self.WBP.killer_types:
				# 		## if we think it's stochastic
				# 		# if any([t in str(self.WBP.theory.classes[k][0].vgdlType) for t in ['Random', 'Chaser']]):
				# 		if True:
				# 			for s in vrle._game.sprite_groups[k]:
				# 				if manhattanDist(vrle._rect2pos(s.rect), vrle._rect2pos(vrle._game.getAvatars()[0].rect)) < self.WBP.safeDistance:
				# 					# print "closer than safeDistance away from {} {}. need to sample".format(k, self.WBP.theory.classes[k][0].vgdlType)
				# 					multipleSamples = True
				# 					break
				# multipleSamples = False
				if len(self.actionSeq)>0:
					a = self.actionSeq[-1]
					if multipleSamples:
						badOutcomeLimit = 0
						okOutcomes, badOutcomes = [], []
						for i in range(10):
							print "multiple samples"
							vrle = self.fastcopy(self.parent.rle)
							res = vrle.step(a, return_obs=True)
							terminal, win = res['ended'], res['win']
							
							# metabolic_cost = self.parent.metabolic_cost + self.metabolics(vrle, res['effectList'], a)
							metabolic_cost = 0

							# terminal, win = vrle._isDone()
							if (terminal, win) == (True, False):
								badOutcomes.append((vrle, terminal, win, metabolic_cost))
							else:
								okOutcomes.append((vrle, terminal, win, metabolic_cost))
							if len(badOutcomes)>badOutcomeLimit:
								break
						# if len(badOutcomes)>badOutcomeLimit:
							# print "too many bad outcomes"
							# embed()
						self.okOutcomes = okOutcomes
						self.badOutcomes = badOutcomes
						if len(badOutcomes)>badOutcomeLimit:
							# print "got badoutcomes"
							self.terminal, self.win, self.metabolic_cost = badOutcomes[0][1], badOutcomes[0][2], badOutcomes[0][3]
							return badOutcomes[0][0], self.terminal, badOutcomes[0][2] #vrle, terminal, win
						else:
							self.terminal, self.win, self.metabolic_cost = okOutcomes[0][1], okOutcomes[0][2], okOutcomes[0][3]
							return okOutcomes[0][0], self.terminal, okOutcomes[0][2] #vrle, terminal, win
					else:
						res = vrle.step(a, return_obs=True)
						self.terminal, self.win = res['ended'], res['win']
						# embed()
						# relevantEvents = [t for t in res['effectList'] if t[0] == 'changeResource']
						# self.metabolic_cost = self.parent.metabolic_cost + self.metabolics(vrle, res['effectList'], a)
						self.metabolic_cost = 0
						# self.terminal, self.win = vrle._isDone()
			except:
				print "conditions met but copy failed"
				embed()
		else:
			# print "in a reconstructed node"
			# embed()
			self.reconstructed=True
			
			# print "copy failed; replaying from top"
			vrle = self.fastcopy(self.rle)
			# vrle = copy.deepcopy(self.rle)
			self.terminal, self.win = vrle._isDone()
			i=0
			while not self.terminal and len(self.actionSeq)>i:
				# a = self.actionSeq[i]
				a = 0
				res = vrle.step(a, return_obs=True)
				# self.metabolic_cost += self.metabolics(vrle, res['effectList'], a)
				self.metabolic_cost = 0
				self.terminal, self.win = res['ended'], res['win']
				# self.terminal, self.win = vrle._isDone()
				i += 1

		return vrle, self.terminal, self.win

	"""
	def eval_profiler(self):
		lp = LineProfiler()
		lp_wrapper = lp(self.eval)
		lp_wrapper()
		lp.print_stats()
	"""

	def eval(self):
		# ## Evaluate current node, including calculating intrinsic reward: f(rewards, heuristics, etc.)

		self.rle, self.terminal, self.win = self.getToCurrentState()

		self.updateObjIDs(self.rle)

		self.state = self.WBP.calculateAtoms(self.rle)

		for i in range(1,self.WBP.IW_k+1):
			for c in itertools.combinations(self.state, i):
				c = tuple(sorted(c))
				if self.WBP.trueAtoms[c] == 0:
					self.candidates.add(c)

		self.updateNovelty()

		## Try rollouts for aliens?
		if self.WBP.allowRollouts and len(self.actionSeq)>0 and self.actionSeq[-1]==32:

			## if the thing we shoot is a missile, do a rollout
			if 'Missile' in str(self.WBP.theory.classes[self.WBP.thingWeShoot][0].vgdlType):
				self.rolloutArray = self.rollout(self.rle, self.WBP.thingWeShoot)
			# print self.rolloutArray
			# print "in rollout"

		self.heuristicVal = self.heuristics(**self.WBP.hyperparameters)

		## Old ways of incorporating rollout; keeping for reference.
		# print self.rle._game.score, self.heuristicVal, sum(self.rolloutArray), self.metabolic_cost, self.position_score()
		# self.intrinsic_reward = self.rle._game.score + self.heuristicVal + \
		# sum(self.rolloutArray) - self.metabolic_cost + self.(-250)
		# print("metabolic cost is {}".format(self.metabolic_cost))

		# resource_bonus = 0
		# if self.parent:
		# 	try:
		# 		for k,v in self.rle._game.getAvatars()[0].resources.items():
		# 			if v==1 and self.parent.rle._game.getAvatars()[0].resources[k]==0:
		# 				resource_bonus += 5000
		# 	except:
		# 		pass

		self.intrinsic_reward = self.heuristicVal + self.position_score(self.WBP.position_score_multiplier) + self.rle._game.score #+ resource_bonus

		## Debug printouts
		# print("heuristicVal {}".format(self.heuristicVal))
		# print("intrinsic_reward {}".format(self.intrinsic_reward))
		try:
			## Planner should return a plan when the agent has reached the limit of any particular resource (because we now should be curious about new objects, which we're taking care of in main_agent)
			if any([self.rle._game.getAvatars()[0].resources[k]==self.WBP.theory.resource_limits[k] for k in self.rle._game.getAvatars()[0].resources.keys() if k not in self.WBP.seen_limits]):
				# if self.WBP.display:
				print "resource limit win"
				self.win=True
		except IndexError:
			pass
		# if self.win:
			# embed()
		return self.win

	def updateNovelty(self):
		if len(self.candidates)==0:
			self.novelty = self.WBP.IW_k+1
		else:
			self.novelty = min([len(c) for c in self.candidates])
		return self.novelty

	def updateNoveltyDict(self, QNovelty, QReward):
		jointSet = list(set(QNovelty+QReward))
		for c in self.candidates:
			if self.WBP.trueAtoms[c] == 0:
				self.WBP.trueAtoms[c] = 1
				for n in jointSet:
					if c in n.candidates:
						n.candidates.remove(c)
		for n in jointSet:
			n.novelty = n.updateNovelty()
		return

	def updateObjIDs(self, vrle):
		i = 0
		for objType in vrle._game.sprite_groups:
			for s in vrle._game.sprite_groups[objType]:
				if s.ID not in self.WBP.objIDs.keys():
					if s.name=='bullet':
						s.ID = len([o for o in vrle._game.sprite_groups[objType] if o not in vrle._game.kill_list])
					else:
						s.ID = len(vrle._game.sprite_groups[objType])
					self.WBP.objIDs[s.ID] = (len(self.WBP.objIDs.keys())+1) * 100 * (self.rle.outdim[0]*self.rle.outdim[1]+self.WBP.padding)
					i+=1
		return

	def playBack(self, make_movie=False):
		vrle = copy.deepcopy(self.rle)
		# vrle = ccopy(self.rle) #5/10/18
		self.finalStatesEncountered = []
		terminal = vrle._isDone()[0]
		i=0
		if not make_movie:
			print vrle.show()
		while not terminal and i<len(self.actionSeq):
			a = self.actionSeq[i]
			vrle.step(a)
			if not make_movie:
				print actionDict[a]
				print vrle.show()
			else:
				self.finalStatesEncountered.append(vrle._game.getFullState())
			terminal = vrle._isDone()[0]
			i+=1
def gen_color():
	from vgdl.colors import colorDict
	color_list = colorDict.values()
	color_list = [c for c in color_list if c not in ['UUWSWF']]
	for color in color_list:
		yield color
	
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

hyperparameter_sets = [
	{'idx'           : 0,
	 'short_horizon' : False,
	 'first_order_horizon': True,
	 'sprite_first_alpha': 10000,
	 'sprite_second_alpha': 100,
	 'sprite_negative_mult': .1,
	 'multisprite_first_alpha': 10000,
	 'multisprite_second_alpha': 100,
	 'novelty_first_alpha': 5000,
	 'novelty_second_alpha': 50,
	 },
	{'idx'           : 1,
	 'short_horizon' : False,
	 'first_order_horizon': True,
	 'sprite_first_alpha': 10000,
	 'sprite_second_alpha': 100,
	 'sprite_negative_mult': 10.,
	 'multisprite_first_alpha': 10000,
	 'multisprite_second_alpha': 100,
	 'novelty_first_alpha': 5000,
	 'novelty_second_alpha': 50,
	 },
	{'idx'           : 2,
	 'short_horizon' : False,
	 'first_order_horizon': False,
	 'sprite_first_alpha': 10000,
	 'sprite_second_alpha': 100,
	 'sprite_negative_mult': .1,
	 'multisprite_first_alpha': 10000,
	 'multisprite_second_alpha': 100,
	 'novelty_first_alpha': 5000,
	 'novelty_second_alpha': 50,
	 },
	{'idx'           : 3,
	 'short_horizon' : True,
	 'first_order_horizon': True,
	 'sprite_first_alpha': 10000,
	 'sprite_second_alpha': 100,
	 'sprite_negative_mult': .1, #normally .1
	 'multisprite_first_alpha': 10000,
	 'multisprite_second_alpha': 100,
	 'novelty_first_alpha': 5000,
	 'novelty_second_alpha': 50,
	 },
	{'idx'           : 4,
	 'short_horizon' : True,
	 'first_order_horizon': True,
	 'sprite_first_alpha': 10000,
	 'sprite_second_alpha': 100,
	 'sprite_negative_mult': 10, #normally .1
	 'multisprite_first_alpha': 10000,
	 'multisprite_second_alpha': 100,
	 'novelty_first_alpha': 5000,
	 'novelty_second_alpha': 50,
	 }

]



if __name__ == "__main__":
	import argparse

	## Continuous physics games can't work right now. RLE is discretized, getSensors() relies on this, and a lot of the induction/planning
	## architecture depends on that. Will take some work to do this well. Best plan is to shrink the grid squares and increase speeds/strengths of
	## objects.
	# gameFilename = "examples.gridphysics.theorytest"
	# gameFilename = "examples.gridphysics.boulderdash"
	# gameFilename = "examples.continuousphysics.breakout_big"


	gameFileString = 'all_games'


	parser = argparse.ArgumentParser(description='Process game number.')
	parser.add_argument('--game_name', type=str, default=str(0), help='game name')
	parser.add_argument('--hyperparameter_index', type=int, default=3, help='hyperparameter_index')
	parser.add_argument('--level', type=int, default=0, help='level')


	args = parser.parse_args()
	game_name = args.game_name
	hyperparameter_index = args.hyperparameter_index
	level_num = args.level
	gvgname = "./{}/{}".format(gameFileString,game_name)
	gameString = read_gvgai_game('{}.txt'.format(gvgname))
	game_levels = [l for l in os.listdir(gameFileString) if l[0:len(game_name+'_lvl')] == game_name+'_lvl']
	print game_levels
	level_game_pairs = []
	for level_number in range(len(game_levels)):
		with open('{}_lvl{}.txt'.format(gvgname, level_number), 'r') as level:
			level_game_pairs.append([gameString, level.read()])

	hyperparameters = hyperparameter_sets[hyperparameter_index]
	planner_hyperparameters = dict((k, hyperparameters[k]) for k in hyperparameters.keys() if k not in ['short_horizon', 'first_order_horizon'])


	gameString, levelString = level_game_pairs[level_num]
	rleCreateFunc = lambda: createRLInputGameFromStrings(gameString, levelString)


	# gameFilename = "examples.gridphysics.theory_frogs"
	# gameString, levelString = defInputGame(gameFilename, randomize=True)
	# rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()

	# embed()
	max_nodes = 50 if hyperparameters['short_horizon'] else 10000
	
	# p = WBP(rle, 'sokoban', hyperparameters=planner_hyperparameters, extra_atom=True, IW_k=2)


	## Initialize planner
	p = WBP(rle, gameFilename, max_nodes=max_nodes, shortHorizon=hyperparameters['short_horizon'],
			firstOrderHorizon=hyperparameters['first_order_horizon'], conservative=False, 
			hyperparameters=planner_hyperparameters, extra_atom=False)
	# embed()
	t1 = time.time()
	bestNode, gameStringArray, objectPositionsArray = p.BFS()

	if bestNode is not None:
		solution = p.solution
		gameString_array = p.gameString_array
		objectPositionsArray = objectPositionsArray[::-1]
	if solution and not p.quitting:
		print "============================================="
		print "got solution of length", len(solution)
		print colored(p.gameString_array[0], 'green')
		for i,g in enumerate(p.gameString_array[1:]):
			print actionDict[solution[i]]
			print colored(g, 'green')
		print "============================================="

	print time.time()-t1
	
	# from core import VGDLParser
	# embed()
	# last.playBack(make_movie=True)
	# VGDLParser.playGame(gameString, levelString, p.statesEncountered, persist_movie=True, make_images=True, make_movie=True, movie_dir="videos/"+gameFilename, padding=0)
	# VGDLParser.playGame(gameString, levelString, last.finalStatesEncountered, persist_movie=True, make_images=True, make_movie=True, movie_dir="videos/"+gameFilename, padding=0)


	# embed()


#