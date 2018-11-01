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

"""
Run: python -m vgdl.basic_mcts
(from the top-level vgdl directory.)

Then run: actions = planActLoop(max_actions_per_plan=10, planning_steps=100, defaultPolicyMaxSteps=50)

__

Calling rle.step(a). Returns a dictionary with:
'reward', 'observation' and 'pcontinue': whether it was a terminal state

Getting sprites:
mcts.rle._game.sprite_groups
"""

"""
OLETS conversion notes:
you're disabling pseudoreward propagations and have disabled reward scaling, as this was related to pseudorewards.
if you run getBestActionsForPlayout() you only really want to pick the first action.
#THEN: once you take it you should reuse that entire part of the tree.

TODO tuesday morning:
Look at results from more recent GVG-AI competitions; is there a better algo?
Compare to java source code. Why is your implementation so bad??
"""
ACTIONS = {(0,0):'stay',(0,-1):'up', (0,1):'down', (1,0):'right', (-1,0):'left', None:'none'}
class OLETS_agent:
	def __init__(self, existing_rle=False, game = None, level = None, partitionWeights=[1,0,1],\
		         rleCreateFunc=False, obsType = OBSERVATION_GLOBAL, decay_factor=.8, num_workers=1):
		if not existing_rle and not rleCreateFunc:
			print "You must pass either an existing rle or an rleCreateFunc"
			return
		# assumption: not starting on terminal state
		"""
		root = the root node of the MCTS tree 
		actions = the list of actions that could be taken
		treePolicy = the policy used to select the descendant node to expand 
		             in the selection step
		defaultPolicy = the policy used in the simulation step.
		subgoal_path_threshold = the max size possible for a path to a subgoal. If
					this has value None, then don't use subgoals.
		"""
		## A few different ways to get observations of the game-state.
		## Observations of everything that's happening on the screen: OBSERVATION_GLOBAL
		## or just of the squares surrounding your avatar: some_other_keyword.
		if existing_rle:
			rle = existing_rle
		else:
			rle = rleCreateFunc(OBSERVATION_GLOBAL)
		self.rleCreateFunc = rleCreateFunc
		self.rle = rle
		## Each time you call self.rleCreateFunc, it returns an rle (rl environment) to you.
		## We do this once per episode.
		self.obsType = obsType
		self.decay_factor = decay_factor

		# always compute using a separate rle. This is only meant to be used for manhattan distance.
		self._obstypes = rle._obstypes
		self.outdim = rle.outdim ## ensures all self.outdim and np.reshape() calls using it are (x,y)
		## returns a representation of the current state.
		## numpy array. Each location in the array is a different grid cell.
		## Each sprite is a unique number. Empty:0, boxes can be 1, agent: 4
		## Assignments of types:number come from the rle instead
		## Different instances of same type have same number.
		## You can have multiple sprites on same square. Number we are shown 
		## is the sum of the IDs.
		## IDs are generated such that the objects are recoverable from the sum.
		self.actions = rle._actionset + [(0,0)]

		self.defaultTime = 0
		self.treeTime = 0
		self.num_workers = num_workers
		self.neighborDict = {}
		self.rewardQueue = deque()

		avatar_code = 1
		avatar_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==avatar_code)
		avatar_loc = avatar_loc[0][0], avatar_loc[1][0] ## (y,x)
		## find location of goal, add to rewardDict.
		## also add neighbors of goal rewardQueue.
		##TODO: update this if goal moves!!
		goal_code = 2**(1+sorted(self._obstypes.keys())[::-1].index("goal"))
		goal_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==goal_code)
		goal_loc = goal_loc[0][0], goal_loc[1][0] #(y,x)

		self.visitedLocations = defaultdict(lambda:0)

		if 'avatar' in self._obstypes.keys():
			inverted_avatar_loc=self._obstypes['avatar'][0]
			avatar_loc = (inverted_avatar_loc[1], inverted_avatar_loc[0])
			self.avatar_code = np.reshape(self.rle._getSensors(), self.outdim)[avatar_loc[0]][avatar_loc[1]]
		else:
			self.avatar_code = 1

		if game and level:
			# self.pseudoRewardDecay = 0.8
			self.maxPseudoReward = 10 #1k
			self.pseudoRewardDecay = .8
			# self.maxPseudoReward = 1/((1-self.pseudoRewardDecay)*(self.pseudoRewardDecay**len(level)))
		else:
			self.maxPseudoReward = 100 #10k
			self.pseudoRewardDecay = .6

		self.rewardScaling = 1000
		self.rewardDict = {goal_loc:self.maxPseudoReward}
		self.processed = [goal_loc]
		self.actionDict = None ## gets initialized in scanDomainForMovementOptions
		self.neighborDict = None ## gets initialized in scanDomainForMovementOptions

		self.scanDomainForMovementOptions()
		self.root = node(self, False, self.actionDict[avatar_loc])
		self.currentNode = self.root
		# self.propagateRewards(goal_loc)

	def scanDomainForMovementOptions(self):
		##TODO: Take a state, so that you can re-perform this scan as needed and take changes into account.
		##TODO: query VGDL description for penetrable/nonpenetrable objects, add to list.
		immovable_codes = []
		# immovables = ['wall']
		try:
			immovables = self.rle.immovables
			# immovables = ['wall', 'poison']
			# print "immovables", immovables
		except:
			immovables = ['wall', 'poison']
			print "Using defaults as immovables", immovables

		for i in immovables:
			if i in self._obstypes.keys():
				immovable_codes.append(2**(1+sorted(self._obstypes.keys())[::-1].index(i)))

		actionDict = defaultdict(list)
		neighborDict = defaultdict(list)
		# action_superset = [(0,0),(-1,0), (1,0), (0,-1), (0,1)]
		action_superset = [(-1,0), (1,0), (0,-1), (0,1)]
		
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

	def getPathToGoal(self, avatar_loc, goal_loc):
		q = deque()
		# q stores tuples in which the first element is a node and the next
		# is the shortest path to that node
		q.append((avatar_loc, []))
		node = avatar_loc
		path = []
		seen = {avatar_loc}
		while q:
			node, path = q.popleft()
			seen.add(node)
			if node == goal_loc:
				break

			for neighbor in self.neighborDict[node]:
				if not neighbor in seen:
					q.append((neighbor, path + [neighbor]))

		if node != goal_loc:
			raise Exception("Didn't find a path to the goal location.")

		return path

	def findAvatarInRLE(self, rle):
		avatar_code = 1
		state = np.reshape(rle._getSensors(), self.rle.outdim)
		if avatar_code in state:
			avatar_loc = np.where(state==avatar_code)
			avatar_loc = avatar_loc[0][0], avatar_loc[1][0]
		else:
			avatar_loc = None
		return avatar_loc

	def OLETS(self, numTrainingCycles, step_horizon):
		rewards = []
		for i in range(numTrainingCycles):
			Vrle = copy.deepcopy(self.rle)

			if i%100==0 and len(rewards)>0:
				print "Training cycle: %i"%i
				print "avg. rewards for last group of 100", np.mean(rewards[-100:])

			reward, v, iters, terminal = self.runSimulation(self.root, Vrle, step_horizon)

		return self

	def getBestActionsForPlayout(self, partitionWeights, debug=False):
		v = self.root
		actions = []
		while len(v.children.keys())>0:
			a, v = self.pickAction(v)
			actions.append(a)
			v = v.children[a]
		return actions

	def pickAction(self, v, rle):
		bestVal, bestAction = -float('inf'), None

		avatarLoc = self.findAvatarInRLE(rle)
		actions = self.actionDict[avatarLoc]
		random.shuffle(actions)
		# embed()
		for a in actions:
			actionVal = self.OLE(v, a, avatarLoc)
			if actionVal > bestVal:
				bestVal = actionVal
				bestAction = a

		if bestAction == None:	## Tiebreaker
			bestAction = random.choice(v.children.keys())
		print bestAction, bestVal
		return bestAction

	def runSimulation(self, v, rle, step_horizon, solveSteps = None):
		count = 0
		iters = 0
		reward = 0
		terminal = False
		while not terminal and iters < step_horizon:
			iters += 1
			count += 1
			avatarLoc = self.findAvatarInRLE(rle)
			self.visitedLocations[avatarLoc] +=1
			if len(v.children.keys()) != len(self.actionDict[avatarLoc]):
				print "not expanded"
				print v.children.keys(), self.actionDict[avatarLoc]
				reward, c = self.expand(v, rle, domain_knowledge=True)
				return reward, c, iters, terminal
			else:
				# print 'selecting new node'
				# embed()
				a = self.pickAction(v, rle)
				res = rle.step(a)
				reward = res['reward']
				avatarLoc = self.findAvatarInRLE(rle)
				self.visitedLocations[avatarLoc] += 1
				print rle.show()
				if a not in v.children.keys():
					print a, "not in children.keys"
					embed()
				v = v.children[a]
				terminal = rle._isDone()[0]
		print "ended run"
		# embed()		
		v.n_e += 1
		v.R_e += reward
		self.backup(v)

		return reward, v, iters, terminal

	def expand(self, v, rle, domain_knowledge=True):

		if domain_knowledge:
			avatarLoc = self.findAvatarInRLE(rle)
			action_choices = self.actionDict[avatarLoc]
		else:
			action_choices = self.actions

		print "in expand", len(action_choices), len(v.children.keys())
		for a in action_choices:
			if a not in v.children.keys():
				expand_action = a
				res = rle.step(a)
				print a
				print rle.show()
				avatarLoc = self.findAvatarInRLE(rle)
				self.visitedLocations[avatarLoc] +=1
				new_state = res["observation"]

				terminal = rle._isDone()[0]
				reward = res['reward']

				child = node(self, terminal, self.actions, parent = v)

				if domain_knowledge:
					# print "creating w/ domain knowledge"
					v.createChild(a, child, avatarLoc, domain_knowledge)
				else:
					v.createChild(a, child)
				break
		
		return reward, child

	def backup(self, v):
		while v:
			# embed()
			v.n_s += 1
			try:
				v.R_m = v.R_e/v.n_s + ((1-v.n_e)/v.n_s) * max([v.children[k].R_m for k in v.children.keys()])
			except:
				v.R_m = v.R_e/v.n_s + ((1-v.n_e)/v.n_s)
			v = v.parent

	def OLE(self, node, action, avatarLoc):
		# print "in OLE"
		# embed()
		nextPos = (action[1]+avatarLoc[0], action[0]+avatarLoc[1])
		return node.R_m + math.sqrt(math.log(node.n_s)/node.n_s_a[action]) - .2*self.visitedLocations[nextPos]

class node:
	def __init__(self, tree, terminal, actions, parent=None):
		self.tree = tree
		self.terminal = terminal # boolean
		self.actions = actions
		self.expanded = False
		self.parent = parent # set to None if root node
		if parent is not None:
			self.depth = parent.depth+1
		else:
			self.depth = 0
		self.children = dict()
		self.exploredChildren = dict()

		self.n_s = 1. # num simulations that have passed through node
		self.n_e = 0. # num simulations that have ended in node
		self.R_e = 0.  #cumulative reward of simulations ending in node
		self.R_m = 0.
		self.n_s_a = defaultdict(lambda:1) #keys are actions. Number of times action a has been taken in this node

	def createChild(self,action,child, avatar_loc=False, domain_knowledge=False):
		# check the following if condition

		if action not in self.children:
		    self.children[action] = child
		    # if domain_knowledge:
		    self.expanded = len(self.children.keys()) == len(self.tree.actionDict[avatar_loc])
		    # print"creating Child"
		    # print action
		    # print self.children.keys(), self.tree.actionDict[avatar_loc]
		    # print len(self.children.keys()), len(self.tree.actionDict[avatar_loc])
			# else:
				# self.expanded = len(self.children.keys()) == len(self.actions)
		else:
			print "createChild got called but with an existing action."

if __name__ == "__main__":
	
	gameFilename = "examples.gridphysics.simpleGame_push_boulders"
	# gameFilename = "examples.gridphysics.waypointtheory" 
	# gameFilename = "examples.gridphysics.simpleGame_teleport"
	# gameFilename = "examples.gridphysics.movers3c"
	# gameFilename = "examples.gridphysics.scoretest" 
	# gameFilename = "examples.gridphysics.rivercross" 
	# gameFilename = "examples.gridphysics.simpleGame_small" 

	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()
	print rle.show()
	# rle.immovables = ['wall', 'poison1', 'poison2']
	print ""
	print "Initializing learner. Playing", gameFilename

	agent = OLETS_agent(rle, gameString, levelString)

	t1 = time.time()
	agent.OLETS(numTrainingCycles=100, step_horizon=100)
	t2 = time.time() - t1
	print "done in {} seconds".format(t2)

	self=agent
	v=agent.root
	avatarLoc = self.findAvatarInRLE(rle)
	actions = self.actionDict[avatarLoc]
	print [(a, self.OLE(v, a, avatarLoc)) for a in actions]
	embed()

	bestAction = agent.pickAction(agent.root, rle)
	rle.step(bestAction)
	agent.rle = rle
	agent.root = agent.root.children[bestAction]
	agent.OLETS(numTrainingCycles=500, step_horizon=100)
	print rle.show()
	v=agent.root
	avatarLoc = self.findAvatarInRLE(rle)
	actions = self.actionDict[avatarLoc]
	print [(a, self.OLE(v, a, avatarLoc)) for a in actions]
