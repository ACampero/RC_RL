import numpy as np
from numpy import zeros
import pygame    
from ontology import BASEDIRS
from core import VGDLSprite
from stateobsnonstatic import StateObsHandlerNonStatic 
from rlenvironmentnonstatic import *
import argparse
import random
from IPython import embed
import math
from Queue import Queue
from threading import Thread
import time

"""
Run: python -m vgdl.basic_mcts
(from the top-level vgdl directory.)


Calling rle.step(a). Returns a dictionary with:
'reward', 'observation' and 'pcontinue': whether it was a terminal state


when you do rle.step(a), what happens to the state in other branches of the tree?

##Helps learning time to not use manhattan distance in bestchild.
## But manhattan distance is helpful for default policy.
"""

class Basic_MCTS:
	def __init__(self, decay_factor, rleCreateFunc, obsType, num_workers):
		# assumption: not starting on terminal state
		"""
		root = the root node of the MCTS tree 
		actions = the list of actions that could be taken
		treePolicy = the policy used to select the descendant node to expand 
		             in the selection step
		defaultPolicy = the policy used in the simulation step.
		"""
		self.decay_factor = decay_factor
		## Each time you call self.rleCreateFunc, it returns an rle (rl environment) to you.
		## We do this once per episode.
		self.rleCreateFunc = rleCreateFunc
		self.obsType = obsType
		## A few different ways to get observations of the game-state.
		## Observations of everything that's happening on the screen: OBSERVATION_GLOBAL
		## or just of the squares surrounding your avatar: some_other_keyword.
		rle = self.rleCreateFunc(OBSERVATION_GLOBAL)
		# always compute using a separate rle. This is only meant to be used for manhattan distance.
		self._obstypes = rle._obstypes
		self.outdim = rle.outdim
		## returns a representation of the current state.
		## numpy array. Each location in the array is a different grid cell.
		## Each sprite is a unique number. Empty:0, boxes can be 1, agent: 4
		## Assignments of types:number come from the rle instead
		## Different instances of same type have same number.
		## You can have multiple sprites on same square. Number we are shown 
		## is the sum of the IDs.
		## IDs are generated such that the objects are recoverable from the sum.
		self.actions = rle._actionset + [(0,0)]
		self.teleport_actions = [(0,-4), (-4, 0), (0, 4), (4, 0), (0,0)]
		self.root = MCTS_node(rle._getSensors(None), False, self.actions)
		self.currentNode = self.root
		self.defaultTime = 0
		self.treeTime = 0
		self.num_workers = num_workers

	def getManhattanDistanceComponents(self, state):
		"""
		expect avatar to be called 'avatar' in class section of theory
		expect goal to be called 'goal' in class section of theory
		currently expects the state observation to follow a grid string format (orignal default format)
		"""
		reshaped_state = np.reshape(state, self.outdim)
		# np_state = np.array([[j for j in i.split('\t')] for i in state.splitlines()])
		avatar = 1
		## Example: to find what ID a box would have, you'd just do ...index("box"). This is the
		## Schaul function for figuring the sprite IDs.
		goal = 2**(1+sorted(self._obstypes.keys())[::-1].index("goal"))
		avatar_loc = None
		goal_loc = None
		numRows, numCols = self.outdim
		for i in range(numRows): 
			for j in range(numCols):
				if (reshaped_state[i,j] / goal) % 2 == 1:
					goal_loc = (i,j)

				if (reshaped_state[i,j]/ avatar) % 2 == 1:
					avatar_loc = (i,j)

		return avatar_loc[0]-goal_loc[0], avatar_loc[1] - goal_loc[1]

	def getManhattanDistance(self, state):
		"""
		expect avatar to be called 'avatar' in class section of theory
		expect goal to be called 'goal' in class section of theory
		currently expects the state observation to follow a grid string format (orignal default format)
		"""
		deltaY, deltaX = self.getManhattanDistanceComponents(state)
		return abs(deltaX) + abs(deltaY)

	def startTrainingPhase(self, numTrainingCycles, step_horizon):
		# apparently the reset method is inefficient
		def createRLE(q, rle_total):
			for i in range(rle_total):
				q.put(self.rleCreateFunc(OBSERVATION_GLOBAL))

		oldTime = time.time()
		# q = Queue()
		# workers = []
		# for i in range(self.num_workers):
		# 	rle_total = (numTrainingCycles/self.num_workers) + (i < (numTrainingCycles % self.num_workers))
		# 	worker = Thread(target=createRLE, args=(q, rle_total,))
		# 	worker.setDaemon(True)
		# 	worker.start()
		# 	workers.append(worker)

		#track total iterations spent in treePolicy
		tree_policy_iters, default_policy_iters = 0, 0
		for i in range(numTrainingCycles):
			# rle._postInitReset()
			# rle._game.reset()
			rle = self.rleCreateFunc(OBSERVATION_GLOBAL)
			# rle = q.get()
			if i%10==0:
				print "Training cycle: %i"%i

			# rle = self.rleCreateFunc(self.obsType)
			reward, vl, iters = self.treePolicy(self.root, rle, step_horizon)
			tree_policy_iters += iters
			if not vl.terminal:
				reward, dPiters = self.defaultPolicy(vl, rle, step_horizon - iters)
				default_policy_iters += dPiters
			self.backup(vl, reward)

		# for worker in workers:
		# 	worker.join()
		print "Tree policy iters:", tree_policy_iters
		print "Default policy iters:", default_policy_iters
		print "Total time: %f"%(time.time()-oldTime)


	def getBestActionsForPlayout(self):
		v = self.root
		actions = []
		while v and not v.terminal:
			a, v = self.bestChild(v,0)
			actions.append(a)
			# bestVisitCount = 0
			# bestChild = None
			# for a,c in v.children.items():
			# 	if c.visitCount > bestVisitCount:
			# 		bestVisitCount = c.visitCount
			# 		bestAction = a
			# 		bestChild = c
			#
			# v = bestChild
			# actions.append(bestAction)
			

			# res = rle.step(a)
			# terminal = not res['pcontinue']
			# if terminal:
			# 	reward = res['reward']

		return actions

	def debug(self):
		v = self.root
		print np.reshape(v.state, self.outdim)
		actions, nodes = [], []
		while v and not v.terminal:
			# print v.children.iteritems()
			print [(k,c.qVal) for k,c in v.children.iteritems()]
			a, v = self.bestChild(v,0)
			actions.append(a)
			nodes.append(v)
			if v:
				print a
				print np.reshape(v.state, self.outdim)
				print ""

		return actions, nodes

	def teleport(self, rle, vector):
		## Teleports avatar to the position closest to AvatarPosition + vector 

		#get original object locations
		# rle._game.sprite_groups

		#translate vector to pixels:
		vector = rle._pos2rect(vector)
		avatarPos = rle._game.sprite_groups['avatar'][0].rect[0:2]
		#approximate teleport position
		outPos = (vector[0]+avatarPos[0], vector[1]+avatarPos[1])
		#find a nearby position
		outPos = self.findNearestEmptyBlock(rle, outPos)[0]
		#remove avatar
		rle._game.sprite_groups['avatar'].remove(rle._avatar)
		#re-place avatar
		rle._game._createSprite(['avatar'], outPos)
		# rle._game._createSprite(['avatar'], random.choice(rle._game.emptyBlocks()))
		# print np.reshape(rle._getSensors(), self.outdim)
		res = rle.step((0,0))
		return res

	def findNearestEmptyBlock(self, rle, pos):
		## TODO: probably can do this more efficiently.
		positions = rle._game.emptyBlocks()
		out = sorted(positions, key=lambda x:self.distance(x, pos))
		return out

	def distance(self, p1, p2):
		return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

	def treePolicy(self, v, rle, step_horizon):
		"""
		i = iteration number
		"""
		# embed()

		count = 0
		iters = 0
		while not v.terminal and iters < step_horizon:
			iters += 1
			self.treeTime += 1
			count += 1
			if not v.expanded:
				reward, c = self.expand(v, rle)
				# rle.step(a)
				return reward, c, iters

			else:
				Cp = 0.70710 # suggested exploration weight
				a, v = self.bestChild(v,Cp) 
				res = rle.step(a)
				terminal = not res['pcontinue']
				if terminal:
					reward = res['reward']
					return reward, v, iters


	def expand(self, v, rle):
		print "in expand"
		# embed()
		expan_action = None
		child = None
		reward = 0
		maxAxisDist = max([abs(el) for el in self.getManhattanDistanceComponents(v.state)])
		teleport_actions = [(0, -maxAxisDist), (-maxAxisDist, 0), (0, maxAxisDist), (maxAxisDist, 0), (0,0)]
		for a in teleport_actions:
		# for a in self.actions:
			if a not in v.children:
				expand_action = a
				# res = rle.step(a)
				res = self.teleport(rle, a)
				new_state = res["observation"]
				terminal = not res['pcontinue']
				if terminal:
					reward = res['reward']

				child = MCTS_node(new_state, terminal, self.actions, parent = v)

				v.createChild(a,child)
				break

		return reward, child

	def bestChild(self, v, Cp):
		def transform(x):
			coefficient = 0.
			slowdown_factor = 1./3
			return coefficient/(1+math.exp(-slowdown_factor * x)) # sigmoid

		maxFuncVal = -float('inf')
		bestChild = None
		bestAction = None
		for a,c in v.children.items():
			if v.equals(c):
				funcVal = -float('inf')
			elif c.visitCount == 0:
				funcVal = float('inf')
			else:
				if c.terminal:
					deltaY, deltaX = self.getManhattanDistanceComponents(v.state)
					manhattanDistance = abs(deltaX + a[0]) + abs(deltaY + a[1])
					if manhattanDistance:
						manhattanDistanceTransform = transform(manhattanDistance)
						funcVal = float(c.qVal)/c.visitCount + Cp * math.sqrt(2*math.log(v.visitCount)/c.visitCount) + float(manhattanDistanceTransform)/c.visitCount

					else:
						funcVal = float('inf')

				else:
					manhattanDistanceTransform = transform(self.getManhattanDistance(c.state))
					funcVal = float(c.qVal)/c.visitCount + Cp * math.sqrt(2*math.log(v.visitCount)/c.visitCount) + float(manhattanDistanceTransform)/c.visitCount

			if funcVal > maxFuncVal:
				maxFuncVal = funcVal
				bestAction = a
				bestChild = c

		return bestAction, bestChild

	def defaultPolicy(self, s, rle, step_horizon):
		"""
		i = iteration number
		"""
		reward = 0
		stepSize = 1 # try 13 later
		rotatedVecMap = {(0,1):(1,0), (1,0):(0,-1), (0,-1):(-1,0), (-1,0):(0,1)}
		vecDist = dict()
		temperature = 3
		terminal = False
		iters = 0
		state = s.state
		g = 1
		while not terminal and iters < step_horizon:
			iters += 1
			vecDistSum = 0
			for preRotatedVec in rotatedVecMap:
				rotatedVec = rotatedVecMap[preRotatedVec]
				for i in range(stepSize):
					vec = tuple(i*np.array(preRotatedVec) + (stepSize-i)*np.array(rotatedVec))
					comps = self.getManhattanDistanceComponents(state) # needs to change
					deltaY, deltaX = comps
					manhattanDistance = abs(deltaX + vec[0]) + abs(deltaY + vec[1])
					vecDist[vec] = math.exp(-temperature * manhattanDistance)
					vecDistSum += vecDist[vec]

			for vec in vecDist:
				vecDist[vec] /= vecDistSum

			samples = np.random.multinomial(1, vecDist.values(), size=1)
			sample_index = np.nonzero(samples)[1][0]
			sample = vecDist.keys()[sample_index]
			# print vecDist, samples, sample_index, sample
			# sample = np.random.choice(vecDist.keys(), 1, vecDist.values())[0]
			a = sample



				# a = self.actions[random.randint(0,len(self.actions)-1)] # COMMENT OUT
			res = rle.step(a)
			new_state = res["observation"]
			state = new_state
			terminal = not res['pcontinue']
			reward += g*res['reward']
			g *= self.decay_factor
			# if terminal:
			# 	reward = res['reward']

			# s_new = MCTS_node(new_state,terminal, rle._actionset, parent = s)
			# s.createChild(a,s_new)

			# s = s_new
			self.defaultTime += 1 # useless right now.
			# actionList.append(a)

			# embed()
			# stepSize = (stepSize + 1)/2

		return reward, iters

	def backup(self, v,reward):
		"""reward = 1 if win, -1 if loss"""
		while v:
			v.backProp(reward)
			reward *= self.decay_factor
			v = v.parent



class MCTS_node:
	def __init__(self,state, terminal, actions, parent=None):
		"""
		state = representation of the game state corresponding to this node
		self.children = a dictionary mapping each action to the child that results
		"""
		self.state = state
		self.terminal = terminal # boolean
		self.actions = actions
		self.visitCount = 0
		self.qVal = 0
		self.expanded = False
		self.parent = parent # set to None if root node
		self.children = dict()
		self.exploredChildren = dict()
		self.expanded = False

	def equals(self,v):
		return np.array_equal(self.state,v.state)

	def backProp(self,win):
		"""
		win = 1 if you won 
		or win = -1 if loss
		"""
		self.qVal+= win 
		self.visitCount += 1

	def createChild(self,action,child):
		# check the following if condition
		if action not in self.children:
		    self.children[action] = child
		    if len(self.children) == len(self.actions):
		    	self.expanded = True

	def getReward(self):
		if self.visitCount > 0:
			return float(self.qVal)/self.visitCount

		else:
			return -1



if __name__ == "__main__":
	obsType = OBSERVATION_GLOBAL
	# self.rleCreateFunc = createRLSimpleGame4
	## passing a function. That function contains things set in
	## 'rlenvironmentnonstatic' file
	## You have to make a function that creates the environment.
	## Make the game, then follow the layout in 'rlenvironmentnonstatic'
	rleCreateFunc = createRLSimpleGame4
	mcts = Basic_MCTS(1, rleCreateFunc, obsType, 1)
	mcts.startTrainingPhase(60, 20)
	# from vgdl.playback import VGDLParser
	from vgdl.core import VGDLParser
	from examples.gridphysics.simpleGame4 import box_level, push_game
	game = push_game
	level = box_level
	embed()
	# VGDLParser.playGame(game, level)
	# VGDLParser.playGame(game, level, mcts.getBestActionsForPlayout())
	# VGDLPlaybackParser.playGame(game, level, mcts.getBestActionsForPlayout())  

	# rewardSum = mcts.startTestingPhase(50)

