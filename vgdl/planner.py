import numpy as np
from numpy import zeros
import pygame    
from ontology import BASEDIRS
from core import VGDLSprite, colorDict, sys
from stateobsnonstatic import StateObsHandlerNonStatic 
from rlenvironmentnonstatic import *
import argparse
import random
from IPython import embed
import math
from threading import Thread
from collections import defaultdict, deque
import time
import copy
from threading import Lock
from Queue import Queue
from util import *
import multiprocessing
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectObjectGoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, \
generateSymbolDict, ruleCluster, Theory, Game, writeTheoryToTxt, generateTheoryFromGame
from rlenvironmentnonstatic import createRLInputGame

#A hack to display things to the terminal conveniently.
np.core.arrayprint._line_width=250

ACTIONS = {(0,0):'stay',(0,-1):'up', (0,1):'down', (1,0):'right', (-1,0):'left', None:'none'}

class Planner:
	def __init__(self, rle, gameString, levelString, gameFilename, display=1):
		self.rle = rle
		self.gameString = gameString
		self.levelString = levelString
		self.gameFilename = gameFilename
		self.display = display
		self.actions = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
		self.maxPseudoReward = 1
		self.pseudoRewardDecay = .99
		self.heuristicDecay = .99
		self.immovables = []
		self.killerObjects = []
		# goalLoc = self.findObjectInRLE(rle, 'goal')
		# self.rewardDict = {goalLoc:self.maxPseudoReward}
		# self.scanDomainForMovementOptions()
		# self.propagateRewards(goalLoc)


	def scanDomainForMovementOptions(self):
		##TODO: Take a state, so that you can re-perform this scan as needed and take changes into account.
		##TODO: query VGDL description for penetrable/nonpenetrable objects, add to list.
		immovable_codes = []
		# immovables = ['wall']
		try:
			self.immovables = self.rle.immovables
			self.killerObjects = self.rle.killerObjects
			self.teleports = self.rle.teleports
			print "immovables", self.rle.immovables
		except:
			self.immovables = ['wall', 'poison']
			self.killerObjects = ['chaser']
			self.teleports = ['exit1', 'entry1']
			print "Using defaults as immovables", self.immovables

		for i in self.immovables:
			if i in self.rle._obstypes.keys():
				immovable_codes.append(2**(1+sorted(self.rle._obstypes.keys())[::-1].index(i)))

		# embed()
		#'exit':'entry' pairs.
		# teleport_partner_dict, teleport_partner_code_dict = {}, {}
		# teleport_partner_code_dict = {}
		# for i in self.teleports:
		# 	if i in self.rle._obstypes.keys():
		# 		if hasattr(self.rle._game.sprite_groups[i][0], 'stype'):
		# 			# we've found an entry
		# 			exitName = self.rle._game.sprite_groups[i][0].stype
		# 			# teleport_partner_dict[exitName] = i
		# 			exitcode = 2**(1+sorted(self.rle._obstypes.keys())[::-1].index(exitName))
		# 			entrycode = 2**(1+sorted(self.rle._obstypes.keys())[::-1].index(i))
		# 			teleport_partner_code_dict[exitcode] = entrycode


		actionDict = defaultdict(list)
		neighborDict = defaultdict(list)
		# action_superset = [(0,0),(-1,0), (1,0), (0,-1), (0,1)]
		action_superset = [(-1,0), (1,0), (0,-1), (0,1)]
		
		board = np.reshape(self.rle._getSensors(), self.rle.outdim)
		y,x=np.shape(board)
		for i in range(y):
			for j in range(x):
				##here
				if board[i,j] not in immovable_codes:
					for action in action_superset:
						nextPos = (i+action[1], j+action[0])
						## Don't look at positions off the board.
						if 0<=nextPos[0]<y and 0<=nextPos[1]<x:
							if board[nextPos] not in immovable_codes:
								actionDict[(i,j)].append(action)
								neighborDict[(i,j)].append(nextPos)
				# if board[i,j] in teleport_partner_code_dict.keys():
				# 	#single entry for each exit (but not other way around)
				# 	entry_loc = np.where(board==teleport_partner_code_dict[board[i,j]])
				# 	entry_loc = entry_loc[0][0], entry_loc[1][0]
				# 	neighborDict[i,j].append(entry_loc)


		self.actionDict = actionDict
		self.neighborDict = neighborDict
		return

	# def scanDomainForMovementOptions(self):
	# 	##TODO: Take a state, so that you can re-perform this scan as needed and take changes into account.
	# 	##TODO: query VGDL description for penetrable/nonpenetrable objects, add to list.
	# 	immovable_codes = []
	# 	# immovables = ['wall']
	# 	try:
	# 		self.immovables = self.rle.immovables
	# 		self.killerObjects = self.rle.killerObjects
	# 		print "immovables", self.rle.immovables
	# 	except:
	# 		self.immovables = ['wall', 'poison']
	# 		self.killerObjects = ['chaser']
	# 		print "Using defaults as immovables", self.immovables

	# 	for i in self.immovables:
	# 		if i in self.rle._obstypes.keys():
	# 			immovable_codes.append(2**(1+sorted(self.rle._obstypes.keys())[::-1].index(i)))


	# 	actionDict = defaultdict(list)
	# 	neighborDict = defaultdict(list)
	# 	# action_superset = [(0,0),(-1,0), (1,0), (0,-1), (0,1)]
	# 	action_superset = [(-1,0), (1,0), (0,-1), (0,1)]
		
	# 	board = np.reshape(self.rle._getSensors(), self.rle.outdim)
	# 	y,x=np.shape(board)
	# 	for i in range(y):
	# 		for j in range(x):
	# 			##here
	# 			if board[i,j] not in immovable_codes:
	# 				for action in action_superset:
	# 					nextPos = (i+action[1], j+action[0])
	# 					## Don't look at positions off the board.
	# 					if 0<=nextPos[0]<y and 0<=nextPos[1]<x:
	# 						if board[nextPos] not in immovable_codes:
	# 							actionDict[(i,j)].append(action)
	# 							neighborDict[(i,j)].append(nextPos)

	# 	self.actionDict = actionDict
	# 	self.neighborDict = neighborDict
	# 	return

	def propagateRewards(self, goalLoc):
		# This version works for teleports, too.
		rewardQueue = deque()
		processed = [goalLoc]
		rewardQueue.append(goalLoc)
		for n in self.neighborDict[goalLoc]:
			if n not in rewardQueue:
				rewardQueue.append(n)

		while len(rewardQueue)>0:
			loc = rewardQueue.popleft()
			if loc not in processed:
				valid_neighbors = [n for n in self.neighborDict[loc] if n in self.rewardDict.keys()]
				try:
					self.rewardDict[loc] = max([self.rewardDict[n] for n in valid_neighbors]) * self.pseudoRewardDecay
				except:
					print "problem with rewardDict"
					embed()
				processed.append(loc)
				for n in self.neighborDict[loc]:
					if n not in processed:
						rewardQueue.append(n)
		return	
	# def propagateRewards(self, goalLoc):
	# 	rewardQueue = deque()
	# 	processed = [goalLoc]
	# 	rewardQueue.append(goalLoc)
	# 	for n in self.neighborDict[goalLoc]:
	# 		if n not in rewardQueue:
	# 			rewardQueue.append(n)

	# 	while len(rewardQueue)>0:
	# 		loc = rewardQueue.popleft()
	# 		if loc not in processed:
	# 			valid_neighbors = [n for n in self.neighborDict[loc] if n in self.rewardDict.keys()]
	# 			self.rewardDict[loc] = max([self.rewardDict[n] for n in valid_neighbors]) * self.pseudoRewardDecay
	# 			processed.append(loc)
	# 			for n in self.neighborDict[loc]:
	# 				if n not in processed:
	# 					rewardQueue.append(n)
	# 	return

	def getPseudoReward(self, s, a):
		## returns pseudoreward of taking action a from location currentLoc.
		## gives pseudoReward[currentLoc] if a doesn't move states.

		currentLoc = self.findAvatarInState(s)
		if currentLoc:
			nextLoc = currentLoc[0]+a[1], currentLoc[1]+a[0] #again, locations are (y,x) and actions are (x,y)
		else:
			return 0.
		if nextLoc in self.rewardDict.keys():
			return self.rewardDict[nextLoc]
		elif currentLoc in self.rewardDict.keys():
			return self.rewardDict[currentLoc]
		else:
			return 0.

	def findObjectsInRLE(self, rle, objName):
		try:
			objLocs = [rle._rect2pos(element.rect) for element in rle._game.sprite_groups[objName] 
			if element not in rle._game.kill_list]
		except:
			return None
		return objLocs
	
	def findAvatarInRLE(self, rle):
		avatar_loc = rle._rect2pos(rle._game.sprite_groups['avatar'][0].rect)
		return avatar_loc

	def findAvatarInState(self, s):
		## takes a string representation of a state, returns avatar location
		state = np.reshape(np.fromstring(s,dtype=float), self.rle.outdim)
		avatar_code = 1
		if avatar_code in state:
			avatar_loc = np.where(state==avatar_code)
			avatar_loc = avatar_loc[0][0], avatar_loc[1][0]
		else:
			avatar_loc = None
		return avatar_loc

	def findObjectsInState(self, s, objName):
		##TODO: Finish last part of this function -- sometimes it can't access objloc[0][0], objloc[1][0]
		
		if objName in self.rle._game.sprite_groups.keys():
			return [self.rle._rect2pos(o.rect) for o in self.rle._game.sprite_groups[objName] if o not in self.rle._game.kill_list]
		else:
			return None


		# state = np.reshape(np.fromstring(s,dtype=float), self.rle.outdim)
		# if objName not in self.rle._obstypes.keys():
		# 	# print objName, "not in rle."
		# 	return None
		# objCode = 2**(1+sorted(self.rle._obstypes.keys())[::-1].index(objName))
		# objLocs = np.where(state==objCode)
		# objLocations = []
		# try:
		# 	for o in objLocs:
		# 		objLocations.append((o[1], o[0]))
		# except:
		# 	try:
		# 		objLocations = [(objLocs[0][0], objLocs[1][0])] #(y,x)
		# 	except:
		# 		print "findObjectsInState is failing."
		# 		embed()
		# return objLocations

