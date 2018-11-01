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
from Queue import Queue
from threading import Thread
from collections import defaultdict, deque
import time
import copy
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectSubgoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, ruleCluster, Theory, Game, writeTheoryToTxt
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
ACTIONS = {(0,0):'stay',(0,-1):'up', (0,1):'down', (1,0):'right', (-1,0):'left', None:'none'}
class Basic_MCTS:
	def __init__(self, existing_rle=False, game = None, level = None, partitionWeights=[1,0,1], waypoint=None, rleCreateFunc=False, obsType = OBSERVATION_GLOBAL, decay_factor=.95, num_workers=1):
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
		self.root = MCTS_node(self, rle._getSensors(None), False, self.actions)
		self.currentNode = self.root
		self.defaultTime = 0
		self.treeTime = 0
		self.num_workers = num_workers
		self.neighborDict = {}
		self.rewardQueue = deque()
		self.num_solutions_found = 0

		## find location of goal, add to rewardDict.
		## also add neighbors of goal rewardQueue.
		##TODO: update this if goal moves!!
		goal_code = 2**(1+sorted(self._obstypes.keys())[::-1].index("goal"))
		goal_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==goal_code)
		goal_loc = goal_loc[0][0], goal_loc[1][0]

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

		self.waypoint = waypoint
		self.partitionWeights = partitionWeights ## Partition for bestChild: weights for (qValue, exploration, heuristic)
		self.partitionWeights = [el/float(sum(self.partitionWeights)) for el in self.partitionWeights]
		self.printheuristicweight = self.partitionWeights[2]
		self.printexplorationweight = self.partitionWeights[1]
		self.rewardScaling = 1000
		self.rewardDict = {goal_loc:self.maxPseudoReward}
		self.processed = [goal_loc]

		print "maxPseudoreward", self.maxPseudoReward
		print "rewardScaling", self.rewardScaling
		print "partitionWeights", self.partitionWeights
		self.scanDomainForMovementOptions()
		self.propagateRewards(goal_loc)

		# self.actionDict = {}
		# ##Populate dictionary for use in action-sampling in defaultPolicy. Not sampling (0,0).
		# for i in range(len(rle._actionset)):
		# 	self.actionDict[i] = rle._actionset[i]

	def scanDomainForMovementOptions(self):
		##TODO: Take a state, so that you can re-perform this scan as needed and take changes into account.
		##TODO: query VGDL description for penetrable/nonpenetrable objects, add to list.
		# print "in scanDomainForMovementOptions"
		immovable_codes = []
		# immovables = ['wall']
		try:
			immovables = self.rle.immovables
			# immovables = ['wall']
			print "immovables", immovables
		except:
			immovables = ['wall']
			print "Using defaults as immovables", immovables

		for i in immovables:
			if i in self._obstypes.keys():
				immovable_codes.append(2**(1+sorted(self._obstypes.keys())[::-1].index(i)))

		actionDict = defaultdict(list)
		neighborDict = defaultdict(list)
		action_superset = [(0,0),(-1,0), (1,0), (0,-1), (0,1)]
		
		board = np.reshape(self.rle._getSensors(), self.outdim)
		x,y=np.shape(board)
		for i in range(x):
			for j in range(y):
				if board[i,j] not in immovable_codes:
					for action in action_superset:
						nextPos = (i+action[0], j+action[1])
						## Don't look at positions off the board.
						if 0<=nextPos[0]<x and 0<=nextPos[1]<y:
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

		print "defaultPolicy steps", step_horizon
		if mark_solution:
			print "will stop once we find", solution_limit, "solutions"
		print ""
		#track total iterations spent in treePolicy
		tree_policy_iters, default_policy_iters = 0, 0
		rewards = []
		defaultPolicySolveStep = None
		# defaultPolicySolveStep stores the first iteration in which default policy solved the game
		for i in range(numTrainingCycles):
			Vrle = copy.deepcopy(VRLE)

			if i%10==0 and len(rewards)>0:
				print "Training cycle: %i"%i
				print "avg. rewards for last group of 10", np.mean(rewards[-10:])
				print "Partition weights", [self.partitionWeights[0], self.printexplorationweight, self.printheuristicweight]
			try:
				if defaultPolicySolveStep:
					reward, v, iters = self.treePolicy(self.root, Vrle, step_horizon, \
						                               solveSteps = i-defaultPolicySolveStep)
				else:
					reward, v, iters = self.treePolicy(self.root, Vrle, step_horizon)

			except TypeError:
				print "typeError in startTrainingphase"
				embed()

			tree_policy_iters += iters
			if not v.terminal:
				reward, dPiters = self.defaultPolicy(v, Vrle, step_horizon, domain_knowledge=True)
				if reward >0 and not defaultPolicySolveStep:
					defaultPolicySolveStep = i
					print "found reward at step", defaultPolicySolveStep


				loc = np.where(np.reshape(v.state, self.outdim)==self.avatar_code)
				
				# if len(loc[0])>0:
				# 	loc = loc[0][0], loc[1][0] 
				# 	if loc in self.rewardDict.keys():
				# 		reward = reward + self.rewardDict[loc]
				
				default_policy_iters += dPiters
			elif v.terminal and mark_solution and reward==self.rewardScaling:
				self.num_solutions_found += 1
				# print "found solution"
				if self.num_solutions_found > solution_limit:
					print ""
					print VRLE.show()
					print "found solution", solution_limit, "times in ", i, "rounds."
					actions = self.getBestActionsForPlayout((1,0,0))
					print "greedy path:", actions
					return self
			rewards.append(reward)
			self.backup(v, reward)

		return self

	def getBestActionsForPlayout(self, partitionWeights):
		v = self.root
		actions = []
		while v and not v.terminal and len(v.children.keys())>0:
			a, v = self.bestChild(v,partitionWeights, debug=False)
			actions.append(a)
		return actions

	def getBestStatesForPlayout(self, rleCreateFunc):
		rle = rleCreateFunc(OBSERVATION_GLOBAL)
		v = self.root
		states = [rle._game.getFullState()]
		while v and not v.terminal:
			a,v = self.maxChild(v)
			rle.step(a)
			states.append(rle._game.getFullState())

		return states

	def debug(self, rle, output=False, numActions=1):
		cntr=0
		v = self.root
		if output:
			print "current state"
			# rle.show()
			print np.reshape(v.state, rle.outdim)
		actions, nodes = [], []
		while v and not v.terminal and cntr<numActions:
			# print v.children.iteritems()
			if output:
				print "options"
				print [(ACTIONS[k],c.qVal) for k,c in v.children.iteritems()]
			# a, v = self.bestChild(v,0)
			a, v = self.bestChild(v,(1,0,0))

			actions.append(a)
			nodes.append(v)
			if output:
				if v:
					print "selected"
					print ACTIONS[a]
					print "resulted in"
					# Can't use rle.show() here, as it's doing a replay, rather than using the actual RLE.
					print np.reshape(v.state, rle.outdim)
					print ""
			cntr+=1
		# if v.terminal:
		# 	distance = 0
		else:
			state = nodes[-1].state
			# deltaX, deltaY = self.getManhattanDistanceComponents(state)
			# distance = abs(deltaX)+abs(deltaY)
		return actions, nodes#, distance


	def treePolicy(self, v, rle, step_horizon, solveSteps = None):
		count = 0
		iters = 0
		c = None # child for expansion - debug
		# print rle.show()
		reward = 0
		while not v.terminal and iters < step_horizon:
			iters += 1
			count += 1
			if not v.expanded:
				reward, c = self.expand(v, rle, domain_knowledge=False)
				# print rle.show()
				return reward, c, iters

			else:
				if solveSteps:
					a, v = self.bestChild(v,self.partitionWeights, solveSteps = solveSteps)
				else:
					a, v = self.bestChild(v,self.partitionWeights)

				res = rle.step(a) ## TODO: you're getting the bestChild and taking bestAction, but in a stochastic game you will end up in
									## a different state despite having taken the same action. Is this what you want?
				
				terminal = rle._isDone()[0]

				if self.waypoint: ## have to analyze for win states in a different way.
					# print "given waypoint", self.waypoint
					if not terminal:
						state = np.reshape(rle._getSensors(), self.outdim)
						avatar_loc = np.where(state==self.avatar_code)
						avatar_loc = (avatar_loc[0][0], avatar_loc[1][0])
						terminal = avatar_loc == self.waypoint
						if terminal:
							reward = self.rewardScaling
						# print avatar_loc, self.waypoint, terminal
					else:
						if rle._isDone()[1]: # ordinary win
							state = np.reshape(rle._getSensors(), self.outdim)
							avatar_loc = np.where(state==self.avatar_code)
							avatar_loc = (avatar_loc[0][0], avatar_loc[1][0])
							if avatar_loc == self.waypoint:
								reward = self.rewardScaling
							else:
								reward = res['reward']
							# print "ordinary win. Avatar, waypoint, reward", avatar_loc, self.waypoint, reward
						else: # ordinary loss
							reward = res['reward']
							# print "ordinary loss. Reward:", reward
					# print "reward, ", reward
					return reward, v, iters
				else:
					if v.terminal!=terminal or not np.array_equal(v.state,rle._getSensors()):
						print "in treePolicy"
						embed()

					# terminal = (not res['pcontinue']) or (rle._avatar is None)
					if terminal:
						reward = res['reward']
						if reward==1:
							reward = self.rewardScaling

					return reward, v, iters

		return reward, v, iters
		print "end of treepolicy"
		embed()

	def expand(self, v, rle, domain_knowledge=False):
		expand_action = None
		child = None
		reward = 0

		if domain_knowledge:
			state  = np.reshape(v.state, self.outdim)
			avatar_loc = np.where(state==self.avatar_code)
			avatar_loc = (avatar_loc[0][0], avatar_loc[1][0])
			action_choices = self.actionDict[avatar_loc]
		else:
			action_choices = self.actions

		for a in action_choices:
			if a not in v.children.keys():
				expand_action = a
				res = rle.step(a)
				new_state = res["observation"]

				##Buggy code on VGDL side forces us to also check the rle.
				# terminal = (not res['pcontinue']) or (rle._avatar is None)
				terminal = rle._isDone()[0]

				if self.waypoint: ## have to analyze for win states in a different way.
					# print "given waypoint", self.waypoint
					if not terminal:
						state = np.reshape(rle._getSensors(), self.outdim)
						avatar_loc = np.where(state==self.avatar_code)
						avatar_loc = (avatar_loc[0][0], avatar_loc[1][0])
						terminal = avatar_loc == self.waypoint
						if terminal:
							reward = self.rewardScaling
						# print avatar_loc, self.waypoint, terminal
					else:
						if rle._isDone()[1]: # ordinary win
							state = np.reshape(rle._getSensors(), self.outdim)
							avatar_loc = np.where(state==self.avatar_code)
							avatar_loc = (avatar_loc[0][0], avatar_loc[1][0])
							if avatar_loc == self.waypoint:
								reward = self.rewardScaling
							else:
								reward = res['reward']
							# print "ordinary win. Avatar, waypoint, reward", avatar_loc, self.waypoint, reward
						else: # ordinary loss
							reward = res['reward']
							# print "ordinary loss. Reward:", reward
					# print "reward, ", reward
				else:
					if terminal:
							reward = res['reward']
							if reward==1:
								reward = self.rewardScaling

				child = MCTS_node(self, new_state, terminal, self.actions, parent = v)

				if domain_knowledge:
					v.createChild(a, child, avatar_loc, domain_knowledge)
				else:
					v.createChild(a, child)
				break
		return reward, child

	def maxChild(self, v):
		tmp = np.where(np.reshape(v.state, self.rle.outdim)==1)
		avatar_loc = tmp[0][0], tmp[1][0]
		qVals = [v.children[a].qVal for a in v.children.keys()]
		if len(qVals)>0 and avatar_loc in self.neighborDict.keys() and len(qVals)>=len(self.neighborDict[avatar_loc])-1: #  -1, since (0,0) is not an action.
				maxVal = max(qVals)
				choices = [(a,c) for (a,c) in v.children.items() if c.qVal==maxVal]
				for (a,c) in v.children.items():
					print a, c.qVal
				print ""
				# for (a,c) in choices:
				# 	print a, c.qVal
				# 	print np.reshape(c.state, self.rle.outdim)
				# printchoices = [(a,c.qVal) for (a,c) in v.children.items() if c.qVal==maxVal]
				# print printchoices
				return random.choice(choices)
		else:
			return (None, None)

	def bestChild(self, v, partitionWeights, solveSteps = None, debug=False):
		"""
		solveSteps = the number of steps that have passed since default policy solved the game
		"""
		
		def transform(loc):
			slowdown_factor = 1 # 1./3
			distanceFunc = self.rewardDict[loc]
			return 1/(1+math.exp(-slowdown_factor * distanceFunc)) # sigmoid

		maxFuncVal = -float('inf')
		bestChild = None
		bestAction = None
		sumQVal = 0
		sumVisitCount = 0
		sumPseudoReward = 0
		heuristic_coefficient = partitionWeights[2]
		exploration_coefficient = partitionWeights[1]
		heuristic_decay_factor = 0.999
		exploration_decay_factor = 0.999
		if solveSteps:
			# print solveSteps
			heuristic_coefficient *= (heuristic_decay_factor**solveSteps)
			exploration_coefficient *= (exploration_decay_factor**solveSteps)
			self.printheuristicweight = heuristic_coefficient
			self.printexplorationweight = exploration_coefficient
			# print "heuristic coefficient is now", heuristic_coefficient

		for a,c in v.children.items():
			if v.equals(c):
				continue
			elif c.visitCount == 0:
				continue
			else:
				if c.terminal:
					vLoc = np.where(np.reshape(v.state, self.outdim)==self.avatar_code)
					if len(vLoc[0])>0:
							vLoc = vLoc[0][0], vLoc[1][0] 

					# cLoc = (vLoc[0] + a[0], vLoc[1] + vLoc[1])
					cLoc = (vLoc[0] + a[0], vLoc[1] + a[1])

					if cLoc in self.rewardDict:
						sumQVal += abs(float(c.qVal)/c.visitCount)
						sumVisitCount += abs(math.sqrt(2*math.log(v.visitCount)/c.visitCount))
						sumPseudoReward += abs(transform(cLoc))
						# sumPseudoReward += abs(transform(cLoc)/c.visitCount)

					else:
						continue

				else:
					loc = np.where(np.reshape(c.state, self.outdim)==self.avatar_code)
					if len(loc[0])>0:
						loc = loc[0][0], loc[1][0] 

					sumQVal += abs(float(c.qVal)/c.visitCount)
					sumVisitCount += abs(math.sqrt(2*math.log(v.visitCount)/c.visitCount))
					sumPseudoReward += abs(transform(loc))

					# sumPseudoReward += abs(transform(loc)/c.visitCount)

		# print ""
		# print np.reshape(v.state, self.outdim)
		for a,c in v.children.items():
			if v.equals(c):
				funcVal = -float('inf')
			elif c.visitCount == 0:
				funcVal = float('inf')
			else:
				if c.terminal:
					vLoc = np.where(np.reshape(v.state, self.outdim)==self.avatar_code)
					if len(vLoc[0])>0:
							vLoc = vLoc[0][0], vLoc[1][0] 

					# cLoc = (vLoc[0] + a[0], vLoc[1] + vLoc[1])
					cLoc = (vLoc[0] + a[0], vLoc[1] + a[1])

					if cLoc in self.rewardDict:
						qValFunction = 0
						if sumQVal == 0:
							qValFunction = 0
						else:
							qValFunction = float(c.qVal)/c.visitCount/sumQVal

						funcVal = partitionWeights[0]*qValFunction \
						        + exploration_coefficient*math.sqrt(2*math.log(v.visitCount)/c.visitCount)/sumVisitCount \
								+ heuristic_coefficient* transform(cLoc)/ sumPseudoReward

								# + partitionWeights[2]* (transform(cLoc)/c.visitCount) / sumPseudoReward
						# funcVal = float(c.qVal)/c.visitCount + Cp * math.sqrt(2*math.log(v.visitCount)/c.visitCount) \
						# 		+ Cp * transform(cLoc)/c.visitCount

					else:
						funcVal = -float('inf')

					# deltaY, deltaX = self.getManhattanDistanceComponents(v.state)
					# manhattanDistance = abs(deltaX + a[0]) + abs(deltaY + a[1])
					# if manhattanDistance:
						# manhattanDistanceTransform = transform(manhattanDistance)
						# funcVal = float(c.qVal)/c.visitCount + Cp * math.sqrt(2*math.log(v.visitCount)/c.visitCount)# + Cp*float(manhattanDistanceTransform)/c.visitCount
					
					## if you uncomment the above, remove the below line.
					
					# else:
						# funcVal = float('inf')
				else:
					# manhattanDistanceTransform = transform(self.getManhattanDistance(c.state))
					

					loc = np.where(np.reshape(c.state, self.outdim)==self.avatar_code)
					if len(loc[0])>0:
							loc = loc[0][0], loc[1][0] 

					qValFunction = 0
					if sumQVal == 0:
						qValFunction = 0
					else:
						qValFunction = (float(c.qVal)/c.visitCount)/sumQVal

					# print a, transform(loc), (transform(loc)/c.visitCount)/sumPseudoReward
					# print a, transform(loc)/sumPseudoReward	
					funcVal = partitionWeights[0]* qValFunction \
					        + exploration_coefficient*math.sqrt(2*math.log(v.visitCount)/c.visitCount)/sumVisitCount \
					        + heuristic_coefficient*transform(loc)/sumPseudoReward	
					        # + partitionWeights[2]*(transform(loc)/c.visitCount)/sumPseudoReward	
					# funcVal = float(c.qVal)/c.visitCount + Cp * math.sqrt(2*math.log(v.visitCount)/c.visitCount) \
					#         + Cp * transform(loc)/c.visitCount

					          # + Cp* float(self.rewardDict[loc])/c.visitCount

			if funcVal > maxFuncVal:
				maxFuncVal = funcVal
				bestAction = a
				bestChild = c
		if bestChild == None:	## Tiebreaker
			bestAction = random.choice(v.children.keys())
			bestChild = v.children[bestAction]

		return bestAction, bestChild

	def defaultPolicy(self, v, rle, step_horizon, domain_knowledge=False):

		reward = 0
		terminal = False
		iters = 0
		state = v.state
		g = 1
		
		terminal = rle._isDone()[0]

		while not terminal and iters < step_horizon:

			reshaped_state = np.reshape(state, self.outdim)
			avatar_loc = np.where(reshaped_state==self.avatar_code, True, False)
			avatar_loc = (avatar_loc[0][0], avatar_loc[1][0])

			iters += 1
			
			if domain_knowledge and avatar_loc in self.actionDict.keys():
				sample = random.choice(self.actionDict[avatar_loc])
			else:
				sample = random.choice([(-1,0), (1,0), (0,-1), (0,1)])
			
			a = sample

			res = rle.step(a)
			# print rle.show()
			new_state = res["observation"]
			state = new_state
			terminal = rle._isDone()[0]

			if self.waypoint: ## have to analyze for win states in a different way.
				# print "given waypoint", self.waypoint
				if not terminal:
					state = np.reshape(rle._getSensors(), self.outdim)
					avatar_loc = np.where(state==self.avatar_code)
					avatar_loc = (avatar_loc[0][0], avatar_loc[1][0])
					terminal = avatar_loc == self.waypoint
					if terminal:
						reward = g*self.rewardScaling
					# print avatar_loc, self.waypoint, terminal
				else:
					if rle._isDone()[1]: # ordinary win
						state = np.reshape(rle._getSensors(), self.outdim)
						avatar_loc = np.where(state==self.avatar_code)
						avatar_loc = (avatar_loc[0][0], avatar_loc[1][0])
						if avatar_loc == self.waypoint:
							reward = g*self.rewardScaling
						else:
							reward = g*res['reward']
						# print "ordinary win. Avatar, waypoint, reward", avatar_loc, self.waypoint, reward
					else: # ordinary loss
						reward = g*res['reward']
						# print "ordinary loss. Reward:", reward
				# print "reward, ", reward
			else:
				if terminal and res['reward']==1:
					reward += g*self.rewardScaling
				else:
					reward += g*res['reward']
			# reward += g*res['reward']
			
			g *= self.decay_factor

		return reward, iters

	def backup(self, v, reward):
		while v:
			v.backProp(reward)
			reward *= self.decay_factor
			v = v.parent


class MCTS_node:
	def __init__(self, tree, state, terminal, actions, parent=None):
		"""
		state = representation of the game state corresponding to this node
		self.children = a dictionary mapping each action to the child that results
		"""
		self.tree = tree
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

	def backProp(self, reward):
		self.qVal+= reward 
		self.visitCount += 1

	def createChild(self,action,child, avatar_loc=False, domain_knowledge=False):
		# check the following if condition
		if action not in self.children:
		    self.children[action] = child
		    if domain_knowledge:
		    	self.expanded = len(self.children) == len(self.tree.actionDict[avatar_loc])
		    else:
		    	self.expanded = len(self.children) == len(self.actions)

def translateEvents(events, all_objects):
	if events is None:
		return None
	# all_objects = rle._game.getObjects()

	def getObjectColor(objectID):
		return all_objects[objectID]['type']['color']

	outlist = []
	for event in events:
		if len(event)==3:
			outlist.append((event[0], getObjectColor(event[1]), getObjectColor(event[2])))
		elif len(event)==2:
			outlist.append((event[0], getObjectColor(event[1])))
	if len(outlist)>0:
		print outlist
	return outlist


def observe(rle, obsSteps):
	print "observing"
	for i in range(obsSteps):
		spriteInduction(rle._game, step=1)
		spriteInduction(rle._game, step=2)
		rle.step((0,0))
		spriteInduction(rle._game, step=3)
	return

def getToSubgoal(rle, vrle, subgoal, all_objects, finalEventList, verbose=True, 
	max_actions_per_plan=1, planning_steps=100, defaultPolicyMaxSteps=50, symbolDict=None):
	## Takes a real world, a theory (instantiated as a virtual world)
	## Moves the agent through the world, updating the theory as needed
	## Ends when subgoal is reached.
	## Right now will only properly work with max_actions_per_plan=1, as you want to re-plan when the theory changes.
	## Otherwise it will only replan every max_actions_per_plan steps.
	## Returns real world in its new state, as well as theory in its new state.
	## TODO: also return a trace of events and of game states for re-creation
	
	hypotheses = []
	terminal = rle._isDone()[0]
	goal_achieved = False

	def noise(action):
		prob=0.
		if random.random()<prob:
			return random.choice(BASEDIRS)
		else:
			return action

	## TODO: this will be problematic when new objects appear, if you don't update it.
	# all_objects = rle._game.getObjects()

	print ""
	print "object goal is", colorDict[str(subgoal.color)], rle._rect2pos(subgoal.rect)
	# actions_executed = []
	states_encountered = []
	while not terminal and not goal_achieved:
		mcts = Basic_MCTS(existing_rle=vrle)
		planner = mcts.startTrainingPhase(planning_steps, defaultPolicyMaxSteps, vrle)
		actions = mcts.getBestActionsForPlayout((1,0,0))

		for i in range(len(actions)):
			if not terminal and not goal_achieved:
				spriteInduction(rle._game, step=1)
				spriteInduction(rle._game, step=2)

				## Take actual step. RLE Updates all positions.
				res = rle.step(noise(actions[i])) ##added noise for testing, but prob(noise)=0 now.
				# actions_executed.append(actions[i])
				states_encountered.append(rle._game.getFullState())

				new_state = res['observation']
				terminal = rle._isDone()[0]
				
				# vrle_res = vrle.step(noise(actions[i]))
				# vrle_new_state = vrle_res['observation']

				effects = translateEvents(res['effectList'], all_objects) ##TODO: this gets object colors, not IDs.
				
				print ACTIONS[actions[i]]
				rle.show()

				# if symbolDict:
				# 	print rle.show()
				# else:
				# 	print np.reshape(new_state, rle.outdim)
				
				# Save the event and agent state
				try:
					agentState = dict(rle._game.getAvatars()[0].resources)
					rle.agentStatePrev = agentState
				# If agent is killed before we get agentState
				except Exception as e:	# TODO: how to process changes in resources that led to termination state?
					agentState = rle.agentStatePrev

				## If there were collisions, update history and perform interactionSet induction
				if effects:
					state = rle._game.getFullState()
					event = {'agentState': agentState, 'agentAction': actions[i], 'effectList': effects, 'gameState': rle._game.getFullStateColorized()}
					finalEventList.append(event)

					for effect in effects:
						rle._game.collision_objects.add(effect[1]) ##sometimes event is just (predicate, obj1)
						if len(effect)==3: ## usually event is (predicate, obj1, obj2)
							rle._game.collision_objects.add(effect[2])

					if colorDict[str(subgoal.color)] in [item for sublist in effects for item in sublist]:
						print "reached subgoal"
						goal_achieved = True
						if subgoal.name in rle._game.unknown_objects:
							rle._game.unknown_objects.remove(subgoal.name)
						goalLoc=None
					else:
						goalLoc = rle._rect2pos(subgoal.rect)

					## Sampling from the spriteDisribution makes sense, as it's
					## independent of what we've learned about the interactionSet.
					## Every timeStep, we should update our beliefs given what we've seen.
					# if not sample:
					sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
						
					g = Game(spriteInductionResult=sample)
					terminationCondition = {'ended': False, 'win':False, 'time':rle._game.time}
					trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState']) for e in finalEventList], terminationCondition)


					hypotheses = list(g.runInduction(sample, trace, 20))

					
					candidate_new_objs = []
					for interaction in hypotheses[0].interactionSet:
						if not interaction.generic:
							if interaction.slot1 != 'avatar':
								candidate_new_objs.append(interaction.slot1)
							if interaction.slot2 != 'avatar':
								candidate_new_objs.append(interaction.slot2)
					candidate_new_objs = list(set(candidate_new_objs))
					candidate_new_colors = []
					for o in candidate_new_objs:
						cols = [c.color for c in hypotheses[0].classes[o]]
						candidate_new_colors.extend(cols)

					## among the many things to fix:
					for e in finalEventList[-1]['effectList']:
						if e[1] == 'DARKBLUE':
							candidate_new_colors.append(e[2])
						if e[2] == 'DARKBLUE':
							candidate_new_colors.append(e[1])

					game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0], "./examples/gridphysics/theorytest.py", goalLoc=goalLoc)
					# all_immovables.extend(immovables)
					# print all_immovables
					vrle = createMindEnv(game, level, OBSERVATION_GLOBAL)
					vrle.immovables = immovables


					## TODO: You're re-running all of theory induction for every timestep
					## every time. Fix this.
					## if you fix it, note that you'd be passing a different g each time,
					## since you sampled (above).
					# hypotheses = list(g.runDFSInduction(trace, 20))

				spriteInduction(rle._game, step=3)
		if terminal:
			if rle._isDone()[1]:
				print "game won"
			else:
				print "Agent died."
	return rle, hypotheses, finalEventList, candidate_new_colors, states_encountered


def planActLoop(rleCreateFunc, filename, max_actions_per_plan, planning_steps, defaultPolicyMaxSteps, playback=False):
	
	rle = rleCreateFunc(OBSERVATION_GLOBAL)
	game, level = defInputGame(filename)
	outdim = rle.outdim
	print rle.show()
	
	terminal = rle._isDone()[0]
	
	i=0
	finalStates = [rle._game.getFullState()]
	while not terminal:
		mcts = Basic_MCTS(existing_rle=rle, game=game, level=level)
		mcts.startTrainingPhase(planning_steps, defaultPolicyMaxSteps, rle)
		# mcts.debug(mcts.rle, output=True, numActions=3)
		# break
		actions = mcts.getBestActionsForPlayout((1,0,0))

		# if len(actions)<max_actions_per_plan:
		# 	print "We only computed", len(actions), "actions."

		new_state = rle._getSensors()
		terminal = rle._isDone()[0]

		for j in range(min(len(actions), max_actions_per_plan)):
			if actions[j] is not None and not terminal:
				print ACTIONS[actions[j]]
				res = rle.step(actions[j])
				new_state = res["observation"]
				terminal = not res['pcontinue']
				print rle.show()
				finalStates.append(rle._game.getFullState())

		i+=1

	if playback:
		from vgdl.core import VGDLParser
		VGDLParser.playGame(game, level, finalStates)
		embed()


def planUntilSolved(rleCreateFunc, filename, defaultPolicyMaxSteps, waypoint=None, playback=False):
	
	rle = rleCreateFunc(OBSERVATION_GLOBAL)
	game, level = defInputGame(filename)
	outdim = rle.outdim
	print rle.show()
	
	terminal = rle._isDone()[0]
	
	i=0
	finalStates = [rle._game.getFullState()]
	

	mcts = Basic_MCTS(existing_rle=rle, game=game, level=level, partitionWeights=[2,1,0], waypoint = waypoint)
	mcts.startTrainingPhase(10000, defaultPolicyMaxSteps, rle, mark_solution=True, solution_limit=500)
	print "ended trainingphase"
	return mcts

	# while not terminal:
	# 	mcts = Basic_MCTS(existing_rle=rle, game=game, level=level)
	# 	mcts.startTrainingPhase(planning_steps, defaultPolicyMaxSteps, rle)
	# 	# mcts.debug(mcts.rle, output=True, numActions=3)
	# 	# break
	# 	actions = mcts.getBestActionsForPlayout()

	# 	# if len(actions)<max_actions_per_plan:
	# 	# 	print "We only computed", len(actions), "actions."

	# 	new_state = rle._getSensors()
	# 	terminal = rle._isDone()[0]

	# 	for j in range(min(len(actions), max_actions_per_plan)):
	# 		if actions[j] is not None and not terminal:
	# 			print ACTIONS[actions[j]]
	# 			res = rle.step(actions[j])
	# 			new_state = res["observation"]
	# 			terminal = not res['pcontinue']
	# 			print rle.show()
	# 			finalStates.append(rle._game.getFullState())

	# 	i+=1

	# if playback:
	# 	from vgdl.core import VGDLParser
	# 	VGDLParser.playGame(game, level, finalStates)
	# 	embed()
	# return finalStates

if __name__ == "__main__":
	## passing a function. That function contains things set in
	## 'rlenvironmentnonstatic' file
	## You have to make a function that creates the environment.
	## Make the game, then follow the layout in 'rlenvironmentnonstatic'
	
	filename = "examples.gridphysics.simpleGame4_small"
	game_to_play = lambda obsType: createRLInputGame(filename)
	
	embed()
	# planActLoop(game_to_play, filename, 5, 100, 50, playback=False)


