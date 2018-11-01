import numpy as np
from numpy import zeros
import pygame    
from ontology import BASEDIRS
from core import VGDLSprite, colorDict
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
import multiprocessing
from qlearner import *
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectObjectGoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, \
generateSymbolDict, ruleCluster, Theory, Game, writeTheoryToTxt, generateTheoryFromGame
from rlenvironmentnonstatic import createRLInputGame

#A hack to display things to the terminal conveniently.
np.core.arrayprint._line_width=250

ACTIONS = {(0,0):'stay',(0,-1):'up', (0,1):'down', (1,0):'right', (-1,0):'left', None:'none'}
class Basic_MCTS:
	def __init__(self, existing_rle=False, game = None, level = None, partitionWeights=[1,0,1],\
		         rleCreateFunc=False, obsType = OBSERVATION_GLOBAL, decay_factor=.8):
		if not existing_rle and not rleCreateFunc:
			print "You must pass either an existing rle or an rleCreateFunc"
			return

		if existing_rle:
			rle = existing_rle
		else:
			rle = rleCreateFunc(OBSERVATION_GLOBAL)
		self.rleCreateFunc = rleCreateFunc
		self.rle = rle
		self.obsType = obsType
		self.decay_factor = decay_factor
		self._obstypes = rle._obstypes
		self.outdim = rle.outdim
		self.actions = rle._actionset + [(0,0)]
		self.root = MCTS_node(self, rle._getSensors(None), False, self.actions)
		self.currentNode = self.root
		self.defaultTime = 0
		self.treeTime = 0
		self.neighborDict = {}
		self.rewardQueue = deque()
		self.num_solutions_found = 0
		self.visitedLocs = defaultdict(lambda:0)
		avatar_code = 1
		avatar_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==avatar_code)
		avatar_loc = avatar_loc[0][0], avatar_loc[1][0] ## (y,x)
		goal_code = 2**(1+sorted(self._obstypes.keys())[::-1].index("goal"))
		goal_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==goal_code)
		goal_loc = goal_loc[0][0], goal_loc[1][0] #(y,x)
		self.maxPseudoReward = 10
		self.pseudoRewardDecay = .99
		self.partitionWeights = [el/float(sum(partitionWeights)) for el in partitionWeights]
		self.printheuristicweight = self.partitionWeights[2]
		self.printexplorationweight = self.partitionWeights[1]
		self.rewardScaling = 1000
		self.rewardDict = {goal_loc:self.maxPseudoReward}
		self.processed = [goal_loc]
		self.actionDict = None ## gets initialized in scanDomainForMovementOptions
		self.neighborDict = None ## gets initialized in scanDomainForMovementOptions

		self.scanDomainForMovementOptions()
		self.propagateRewards(goal_loc)

	def scanDomainForMovementOptions(self):
		immovable_codes = []
		try:
			immovables = self.rle.immovables
		except:
			immovables = ['wall', 'poison']
			print "Using defaults as immovables", immovables

		for i in immovables:
			if i in self._obstypes.keys():
				immovable_codes.append(2**(1+sorted(self._obstypes.keys())[::-1].index(i)))

		actionDict = defaultdict(list)
		neighborDict = defaultdict(list)
		action_superset = [(0,0),(-1,0), (1,0), (0,-1), (0,1)]
		
		board = np.reshape(self.rle._getSensors(), self.outdim)
		y,x=np.shape(board)
		for i in range(y):
			for j in range(x):
				if board[i,j] not in immovable_codes:
					for action in action_superset:
						nextPos = (i+action[1], j+action[0])
						## Don't look at positions off the board.
						if 0<=nextPos[0]<y and 0<=nextPos[1]<x:
							if board[nextPos] not in immovable_codes:
								actionDict[(i,j)].append(action)
								neighborDict[(i,j)].append(nextPos)
		self.actionDict = actionDict
		self.neighborDict = neighborDict
		return

	def propagateRewards(self, goal_loc):
		self.rewardQueue.append(goal_loc)
		for n in self.neighborDict[goal_loc]:
			if n not in self.rewardQueue:
				self.rewardQueue.append(n)

		while len(self.rewardQueue)>0:
			loc = self.rewardQueue.popleft()
			if loc not in self.processed:
				valid_neighbors = [n for n in self.neighborDict[loc] if n in self.rewardDict.keys()]
				self.rewardDict[loc] = max([self.rewardDict[n] for n in valid_neighbors]) * self.pseudoRewardDecay
				self.processed.append(loc)
				for n in self.neighborDict[loc]:
					if n not in self.processed:
						self.rewardQueue.append(n)
		return

	def startTrainingPhase(self, numTrainingCycles, step_horizon, VRLE, mark_solution=False, solution_limit=20):

		#track total iterations spent in treePolicy
		tree_policy_iters, default_policy_iters = 0, 0
		rewards = []
		defaultPolicySolveStep = None
		# defaultPolicySolveStep stores the first iteration in which default policy solved the game
		for i in range(numTrainingCycles):
			Vrle = copy.deepcopy(VRLE)

			if i%100==0 and len(rewards)>0:
				print "Training cycle: %i"%i
				print "avg. rewards for last group of 100", np.mean(rewards[-100:])

			if defaultPolicySolveStep:
				reward, v, iters = self.treePolicy(self.root, Vrle, step_horizon, \
					                               solveSteps = i-defaultPolicySolveStep)
			else:
				reward, v, iters = self.treePolicy(self.root, Vrle, step_horizon)

			tree_policy_iters += iters

			if not v.terminal:
				reward, dPiters = self.defaultPolicy(v, Vrle, step_horizon, domain_knowledge=True)
				if reward > 0 and not defaultPolicySolveStep:
					defaultPolicySolveStep = i				
				default_policy_iters += dPiters
			elif v.terminal and mark_solution and reward==self.rewardScaling:
				self.num_solutions_found += 1
				if self.num_solutions_found > solution_limit:
					return self, i
			rewards.append(reward)
			self.backup(v, reward)

		return self, i

	def treePolicy(self, v, rle, step_horizon, solveSteps = None):
		count = 0
		iters = 0
		c = None
		# print rle.show()
		reward = 0
		while not v.terminal and iters < step_horizon:
			iters += 1
			count += 1
			if not v.expanded:
				reward, c = self.expand(v, rle, domain_knowledge=False)
				return reward, c, iters
			else:
				if solveSteps:
					a, v = self.bestChild(v,self.partitionWeights, solveSteps = solveSteps)
				else:
					a, v = self.bestChild(v,self.partitionWeights)

				res = rle.step(a)
				
				print rle.show()
				terminal = rle._isDone()[0]

				if terminal != v.terminal or not np.array_equal(v.state, rle._getSensors()):
					print "inconsistency in node and rle"
					embed()
				# terminal = (not res['pcontinue']) or (rle._avatar is None)
				if terminal:
					reward = res['reward']
					if reward==1:
						reward = self.rewardScaling

					return reward, v, iters

		return reward, v, iters
	def getBestActionsForPlayout(self, partitionWeights):
		v = self.root
		actions = []
		while v and not v.terminal and len(v.children.keys())>0:
			a, v = self.bestChild(v,partitionWeights)
			actions.append(a)
		return actions

