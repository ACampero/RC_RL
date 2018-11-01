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
ACTIONS = {(0,0):'stay',(0,-1):'up', (0,1):'down', (1,0):'right', (-1,0):'left', None:'none'}
class Basic_MCTS:
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
		self.root = MCTS_node(self, rle._getSensors(None), False, self.actions)
		self.currentNode = self.root
		self.defaultTime = 0
		self.treeTime = 0
		self.num_workers = num_workers
		self.neighborDict = {}
		self.rewardQueue = deque()
		self.num_solutions_found = 0

		avatar_code = 1
		avatar_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==avatar_code)
		avatar_loc = avatar_loc[0][0], avatar_loc[1][0] ## (y,x)
		## find location of goal, add to rewardDict.
		## also add neighbors of goal rewardQueue.
		##TODO: update this if goal moves!!
		goal_code = 2**(1+sorted(self._obstypes.keys())[::-1].index("goal"))
		goal_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==goal_code)
		goal_loc = goal_loc[0][0], goal_loc[1][0] #(y,x)

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

		self.partitionWeights = partitionWeights ## Partition for bestChild: weights for (qValue, exploration, heuristic)
		self.partitionWeights = [el/float(sum(self.partitionWeights)) for el in self.partitionWeights]
		self.printheuristicweight = self.partitionWeights[2]
		self.printexplorationweight = self.partitionWeights[1]
		self.rewardScaling = 1000
		self.rewardDict = {goal_loc:self.maxPseudoReward}
		self.processed = [goal_loc]
		self.actionDict = None ## gets initialized in scanDomainForMovementOptions
		self.neighborDict = None ## gets initialized in scanDomainForMovementOptions

		self.scanDomainForMovementOptions()
		self.propagateRewards(goal_loc)


	def getSubgoals(self, subgoal_path_threshold):

		avatar_code = 1
		avatar_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==avatar_code)
		avatar_loc = avatar_loc[0][0], avatar_loc[1][0]
		## find location of goal, add to rewardDict.
		## also add neighbors of goal rewardQueue.
		##TODO: update this if goal moves!!
		goal_code = 2**(1+sorted(self._obstypes.keys())[::-1].index("goal"))
		goal_loc = np.where(np.reshape(self.rle._getSensors(), self.outdim)==goal_code)
		goal_loc = goal_loc[0][0], goal_loc[1][0]
		self.subgoals = []
		path = self.getPathToGoal(avatar_loc, goal_loc)
		if subgoal_path_threshold > len(path):
			# don't use any subgoals in this case.
			return [goal_loc]
		else:
			subgoal_index = -1
			num_subgoals = int(math.ceil(float(len(path))/subgoal_path_threshold))
			for i in range(num_subgoals):
				if i < len(path) % num_subgoals:
					subgoal_index += (len(path)/num_subgoals + 1)
				else:
					subgoal_index += len(path)/num_subgoals

				self.subgoals.append(path[subgoal_index])
		return self.subgoals

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

				loc = np.where(np.reshape(v.state, self.outdim)==self.avatar_code)
				
				# if len(loc[0])>0:
				# 	loc = loc[0][0], loc[1][0] 
				# 	if loc in self.rewardDict.keys():
				# 		reward = reward + self.rewardDict[loc]
				
				default_policy_iters += dPiters
			elif v.terminal and mark_solution and reward==self.rewardScaling:
				self.num_solutions_found += 1
				if self.num_solutions_found > solution_limit:
					return self, i
			rewards.append(reward)
			self.backup(v, reward)

		return self, i

	def getBestActionsForPlayout(self, partitionWeights, debug=False):
		v = self.root
		actions = []
		while v and not v.terminal and len(v.children.keys())>0:
			a, v = self.bestChild(v,partitionWeights, debug=debug)
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
			print np.reshape(v.state, rle.outdim)
		actions, nodes = [], []
		while v and not v.terminal and cntr<numActions:
			if output:
				print "options"
				print [(ACTIONS[k],c.qVal) for k,c in v.children.iteritems()]
			a, v = self.bestChild(v,(1,0,0))

			actions.append(a)
			nodes.append(v)
			if output:
				if v:
					print "selected"
					print ACTIONS[a]
					print "resulted in"
					print np.reshape(v.state, rle.outdim)
					print ""
			cntr+=1
		# if v.terminal:
		# 	distance = 0
		else:
			state = nodes[-1].state
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
				# print "after expanding"
				# print rle.show()
				return reward, c, iters

			else:
				# Cp = 1.
				# Cp = 0.70710 # suggested exploration weight
				if solveSteps:
					a, v = self.bestChild(v,self.partitionWeights, solveSteps = solveSteps)
				else:
					a, v = self.bestChild(v,self.partitionWeights)

				# print "before bestChild action:"
				# print rle.show()

				res = rle.step(a) ## TODO: you're getting the bestChild and taking bestAction, but in a stochastic game you will end up in
									## a different state despite having taken the same action. Is this what you want?
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
		tmp = np.where(np.reshape(v.state, self.outdim)==1)
		avatar_loc = tmp[0][0], tmp[1][0]
		qVals = [v.children[a].qVal for a in v.children.keys()]
		if len(qVals)>0 and avatar_loc in self.neighborDict.keys() and len(qVals)>=len(self.neighborDict[avatar_loc])-1: #  -1, since (0,0) is not an action.
				maxVal = max(qVals)
				choices = [(a,c) for (a,c) in v.children.items() if c.qVal==maxVal]
				for (a,c) in v.children.items():
					print a, c.qVal
				print ""
				return random.choice(choices)
		else:
			return (None, None)

	def bestChild(self, v, partitionWeights, solveSteps = None, debug=False):
		"""
		solveSteps = the number of steps that have passed since default policy solved the game
		"""
		
		def transform(loc):
			slowdown_factor = 1 # 1./3
			if loc in self.rewardDict:
				distanceFunc = self.rewardDict[loc]
				return 1/(1+math.exp(-slowdown_factor * distanceFunc)) # sigmoid
			else:
				return 0.
			# try:
			# 	distanceFunc = self.rewardDict[loc]
			# except:
			# 	print loc, "not in self.rewardDict"
			# 	embed()

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
				if self.avatar_code not in [l%2 for l in c.state]: ##we're in a terminal losing state
																	## Don't give pseudoreward
					if not c.terminal:
						print "terminal state but avatar not in c.state"
						embed()

					# vLoc = np.where(np.reshape(v.state, self.outdim)%2==self.avatar_code)
					# if len(vLoc[0])>0:
					# 		vLoc = vLoc[0][0], vLoc[1][0] 

					# cLoc = (vLoc[0] + a[1], vLoc[1] + a[0])

					# if cLoc in self.rewardDict:
					# 	sumQVal += abs(float(c.qVal)/c.visitCount)
					# 	sumVisitCount += abs(math.sqrt(2*math.log(v.visitCount)/c.visitCount))
					# 	sumPseudoReward += abs(transform(cLoc))/c.visitCount
					# else:
					# 	continue

					sumQVal += abs(float(c.qVal)/c.visitCount)
					sumVisitCount += abs(math.sqrt(2*math.log(v.visitCount)/c.visitCount))
					sumPseudoReward += 0. #abs(transform(cLoc))/c.visitCount


				else:
					loc = np.where(np.reshape(c.state, self.outdim)%2==self.avatar_code)
					if len(loc[0])>0:
						loc = loc[0][0], loc[1][0] 
					try:
						sumQVal += abs(float(c.qVal)/c.visitCount)
						sumVisitCount += abs(math.sqrt(2*math.log(v.visitCount)/c.visitCount))
						sumPseudoReward += abs(transform(loc))/c.visitCount
					except TypeError:
						print "loc is weird type"
						embed()



		# print ""
		if debug:
			print np.reshape(v.state, self.outdim)
		for a,c in v.children.items():
			if v.equals(c):
				funcVal = -float('inf')
			elif c.visitCount == 0:
				funcVal = float('inf')
			else:
				if self.avatar_code not in [l%2 for l in c.state]: ##we're in a terminal losing state
																	## Don't give pseudoreward
					# vLoc = np.where(np.reshape(v.state, self.outdim)%2==self.avatar_code)
					# if len(vLoc[0])>0:
							# vLoc = vLoc[0][0], vLoc[1][0] 

					# cLoc = (vLoc[0] + a[1], vLoc[1] + a[0])
					# if cLoc in self.rewardDict:
					qValFunction = 0
					if sumQVal == 0:
						qValFunction = 0
					else:
						qValFunction = float(c.qVal)/c.visitCount/sumQVal

					funcVal = partitionWeights[0]*qValFunction \
					        + exploration_coefficient*math.sqrt(2*math.log(v.visitCount)/c.visitCount)/sumVisitCount \
							+ 0.


					# else:
					# 	funcVal = -float('inf')

					# qValFunction = 0
					# if sumQVal == 0:
					# 	qValFunction = 0
					# else:
					# 	qValFunction = float(c.qVal)/c.visitCount/sumQVal

					# funcVal = partitionWeights[0]*qValFunction \
					#         + exploration_coefficient*math.sqrt(2*math.log(v.visitCount)/c.visitCount)/sumVisitCount \
					# 		+ heuristic_coefficient* (transform(cLoc)/c.visitCount)/ sumPseudoReward


				else:
					loc = np.where(np.reshape(c.state, self.outdim)%2==self.avatar_code)
					if len(loc[0])>0:
							loc = loc[0][0], loc[1][0] 

					qValFunction = 0
					if sumQVal == 0:
						qValFunction = 0
					else:
						qValFunction = (float(c.qVal)/c.visitCount)/sumQVal


					funcVal = partitionWeights[0]* qValFunction \
					        + exploration_coefficient*math.sqrt(2*math.log(v.visitCount)/c.visitCount)/sumVisitCount \
					        + heuristic_coefficient*(transform(loc)/c.visitCount)/sumPseudoReward	
				if debug:
					print a, funcVal

			if funcVal > maxFuncVal:
				maxFuncVal = funcVal
				bestAction = a
				bestChild = c
		if bestChild == None:	## Tiebreaker
			bestAction = random.choice(v.children.keys())
			bestChild = v.children[bestAction]

		if debug:
			print ""
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



def getToObjectGoal(rle, vrle, plannerType, game_object, hypothesis, game, level, object_goal, all_objects, finalEventList, verbose=True,\
	defaultPolicyMaxSteps=50, symbolDict=None):
	## Takes a real world, a theory (instantiated as a virtual world)
	## Moves the agent through the world, updating the theory as needed
	## Ends when object_goal is reached.
	## Returns real world in its new state, as well as theory in its new state.
	## TODO: also return a trace of events and of game states for re-creation
	
	hypotheses = []
	terminal = rle._isDone()[0]
	goal_achieved = False
	outdim = rle.outdim

	def noise(action):
		prob=0.
		if random.random()<prob:
			return random.choice(BASEDIRS)
		else:
			return action

	## TODO: this will be problematic when new objects appear, if you don't update it.
	# all_objects = rle._game.getObjects()

	states_encountered = [rle._game.getFullState()]
	candidate_new_colors = []
	hypotheses = [hypothesis]
	while not terminal and not goal_achieved:

		theory_change_flag = False

		if not theory_change_flag: 
			if plannerType=='mcts':
				planner = Basic_MCTS(existing_rle=vrle, game=game, level=level, partitionWeights=[5,3,3])
				subgoals = planner.getSubgoals(subgoal_path_threshold=3)
			elif plannerType=='QLearning':
				planner = QLearner(vrle, gameString=game, levelString=level)
				subgoals = planner.getSubgoals(subgoal_path_threshold=10)
			print "subgoals", subgoals
			total_steps = 0
			for subgoal in subgoals:
				if not theory_change_flag and not goal_achieved:

					## write subgoal to theory; initialize VRLE.

					game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0], symbolDict, \
						"./examples/gridphysics/theorytest.py", subgoal)
					vrle = createMindEnv(game, level, output=False)
					vrle.immovables = immovables

					## Get actions that take you to goal.
					ignore, actions, steps = getToWaypoint(vrle, subgoal, plannerType, symbolDict, defaultPolicyMaxSteps, partitionWeights=[5,3,3], act=False)

					for action in actions:
						if not theory_change_flag and not goal_achieved:
							spriteInduction(rle._game, step=1)
							spriteInduction(rle._game, step=2)
							res = rle.step(noise(action))
							states_encountered.append(rle._game.getFullState())
							terminal = rle._isDone()[0]				
							effects = translateEvents(res['effectList'], all_objects)
							if symbolDict: 
								print rle.show()
							else:
								print np.reshape(new_state, rle.outdim)
							# Save the event and agent state
							try:
								agentState = dict(rle._game.getAvatars()[0].resources)
								rle.agentStatePrev = agentState
							# If agent is killed before we get agentState
							except Exception as e:	# TODO: how to process changes in resources that led to termination state?
								agentState = rle.agentStatePrev
					 		## If there were collisions, update history and perform interactionSet induction if the collisions were novel.
							if effects:
								state = rle._game.getFullState()
								event = {'agentState': agentState, 'agentAction': action, 'effectList': effects, 'gameState': rle._game.getFullStateColorized()}

								## Check if you reached object goal
								# if colorDict[str(object_goal.color)] in [item for sublist in effects for item in sublist]:
								# 	print "goal achieved?"
								# 	embed()
								# 	print "goal achieved"
								# 	goal_achieved = True
								for e in effects:
									if 'DARKBLUE' in e and colorDict[str(object_goal.color)] in e:
										print "goal achieved"
										# embed()
										goal_achieved = True

								## Sampling from the spriteDisribution makes sense, as it's
								## independent of what we've learned about the interactionSet.
								## Every timeStep, we should update our beliefs given what we've seen.
								
								## TODO: This crashed. Get it working again, then incorporate the sprite induction result.
								if len(rle._game.spriteDistribution)==0:
									print "before step3"
									embed()
								spriteInduction(rle._game, step=3)
								if len(rle._game.spriteDistribution)==0:
									print "after step3"
									embed()

								# if not sample:
								sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
								game_object = Game(spriteInductionResult=sample)

								## Get list of all effects we've seen. Only update theory if we're seeing something new.
								all_effects = [item for sublist in [e['effectList'] for e in finalEventList] for item in sublist]
								if not all([e in all_effects for e in effects]):## TODO: make sure you write this so that it works with simultaneous effects.
									finalEventList.append(event)
									terminationCondition = {'ended': False, 'win':False, 'time':rle._game.time}
									trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState']) for e in finalEventList], terminationCondition)
									theory_change_flag = True
									hypotheses = list(game_object.runInduction(game_object.spriteInductionResult, trace, 20)) ##if you resample or run sprite induction, this 
																								## should be g.runInduction

									## new colors that we have maybe learned about
									candidate_new_objs = []
									for interaction in hypotheses[0].interactionSet:
										if not interaction.generic:
											if interaction.slot1 != 'avatar':
												candidate_new_objs.append(interaction.slot1)
											if interaction.slot2 != 'avatar':
												candidate_new_objs.append(interaction.slot2)
									candidate_new_objs = list(set(candidate_new_objs))
									for o in candidate_new_objs:
										cols = [c.color for c in hypotheses[0].classes[o]]
										candidate_new_colors.extend(cols)

									## among the many things to fix:

									for e in finalEventList[-1]['effectList']:
										if e[1] == 'DARKBLUE':
											candidate_new_colors.append(e[2])
											print "appending", e[2], "to candidate_new_colors"
										if e[2] == 'DARKBLUE':
											candidate_new_colors.append(e[1])
											print "appending", e[1], "to candidate_new_colors"

									candidate_new_colors = list(set(candidate_new_colors))
									# print "candidate new colors", candidate_new_colors

									## update to incorporate what we've learned, keep the same subgoal for now; this will update at the top of the next loop.
									game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0], symbolDict, \
										"./examples/gridphysics/theorytest.py", goalLoc=(rle._rect2pos(object_goal.rect)[1], rle._rect2pos(object_goal.rect)[0]))

									print "updating internal theory"
									vrle = createMindEnv(game, level, output=False)
									vrle.immovables = immovables															
								else:
									finalEventList.append(event)
									terminationCondition = {'ended': False, 'win':False, 'time':rle._game.time}
									trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState']) for e in finalEventList], terminationCondition)
									## you need to figure out how to incorporate the result of sprite induction in cases where you don't do
									## interactionSet induction (i.e., here.)
									hypotheses = [hypothesis]


							
							if terminal:
								return rle, hypotheses, finalEventList, candidate_new_colors, states_encountered, game_object

					print "executed all actions."
			total_steps += steps
	return rle, hypotheses, finalEventList, candidate_new_colors, states_encountered, game_object


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
		actions = mcts.getBestActionsForPlayout()

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


def getToWaypoint(rle, subgoal, plannerType, symbolDict, defaultPolicyMaxSteps, partitionWeights, act=True):

	theory = generateTheoryFromGame(rle)

	theoryString, levelString, inverseMapping, immovables =\
	writeTheoryToTxt(rle, theory, symbolDict, "./examples/gridphysics/waypointtheory.py", subgoal)
	Vrle = createMindEnv(theoryString, levelString, output=False)
	Vrle.immovables = immovables

	print "mental map with subgoal:", subgoal
	print Vrle.show()
	print "planner type", plannerType
	if plannerType=='mcts':
		mcts = Basic_MCTS(existing_rle=Vrle, game=theoryString, level=levelString, partitionWeights=partitionWeights)
		# print "made mcts for subgoal,", subgoal
		# embed()
		m, steps = mcts.startTrainingPhase(1200, defaultPolicyMaxSteps, Vrle, mark_solution=True, solution_limit=20)
		actions = mcts.getBestActionsForPlayout((1,0,0), debug=False)
	elif plannerType=='QLearning':
		planner = QLearner(Vrle, gameString=theoryString, levelString=levelString)
		steps = planner.learn(500, satisfice=True)
		actions = planner.getBestActionsForPlayout()
	print "Found plan to subgoal. Actions", actions
	if act:
		for a in actions:
			rle.step(a)
			print rle.show()
	return rle, actions, steps


def planUntilSolved(rleCreateFunc, filename, defaultPolicyMaxSteps, partitionWeights, playback=False, maxEpisodes=700):
	
	rle = rleCreateFunc(OBSERVATION_GLOBAL)
	game, level = defInputGame(filename)
	outdim = rle.outdim
	symbolDict = generateSymbolDict(rle)
	print rle.show()

	goal_loc = np.where(np.reshape(rle._getSensors(), rle.outdim)==8)
	goal_loc = goal_loc[0][0], goal_loc[1][0]
	terminal = rle._isDone()[0]
	
	i=0
	finalStates = [rle._game.getFullState()]
	## Have to make this as a theory and then write it, so that you can find what the immovables are
	## then these can get incorporated when you look for subgoals.
	theory = generateTheoryFromGame(rle)
	# theory.display()
	theoryString, levelString, inverseMapping, immovables =\
	writeTheoryToTxt(rle, theory, symbolDict, "./examples/gridphysics/whatever.py", goal_loc)

	rle = createMindEnv(theoryString, levelString, output=False)
	rle.immovables = immovables

	mcts = Basic_MCTS(existing_rle=rle, game=game, level=level, partitionWeights=[5,2,3])
	subgoals = mcts.getSubgoals(subgoal_path_threshold=3)
	print "subgoals", subgoals

	
	total_steps = 0
	solved = True
	numActions = 0
	for subgoal in subgoals:
		rle, actions, steps = getToWaypoint(rle, subgoal, symbolDict, defaultPolicyMaxSteps, partitionWeights=[10,2,4])
		numActions += len(actions)
		print steps, "steps"
		total_steps += steps
		if total_steps > maxEpisodes:
			solved = False
			break

	if solved:
		print "Found and executed plan using", total_steps, "epiosodes of MCTS."
	else:
		print "didn't solve game even using %i episodes of MCTS"%total_steps

	return mcts, total_steps, solved, numActions

def parallelizedPlanUntilSolved(rleCreateFunc, filename, defaultPolicyMaxSteps, partitionWeightsList, numWorkers=4):
	"""
	partitionWeightsList = a list of partitionWeight tuples.
	numWorkers = the number of threads which are running planUntilSolved in parallel
	"""
	# m = multiprocessing.Manager()
	weightsQueue = multiprocessing.Queue()
	# contract: the weightsQueue will contain all the partition weights in the beginning
	# and a numWorkers number of DONE_MESSAGEs at the very end
	# to enssure each of the workers stops running.
	resultsQueue = multiprocessing.Queue()
	weightInfo = dict()
	DONE_MESSAGE = "DONE"
	def worker(weightsQue, resultsQue, DONE_MESSAGE):
		i = 0
		while not weightsQue.empty():
			message = weightsQue.get()
			if message == DONE_MESSAGE:
				break

			partitionWeights = message
			mcts, total_steps, solved, numActions = planUntilSolved(rleCreateFunc, filename, defaultPolicyMaxSteps, partitionWeights)
			resultsQueue.put((partitionWeights, {'total_steps': total_steps, 'solved': solved, 'numActions': numActions}))


	jobs = []
	for partitionWeights in partitionWeightsList:
		weightsQueue.put(partitionWeights)


	for i in range(numWorkers):
		weightsQueue.put(DONE_MESSAGE)
		p = multiprocessing.Process(target=worker, args=(weightsQueue, resultsQueue, DONE_MESSAGE))
		jobs.append(p)
		p.start()

	for j in jobs:
		j.join()

	while not resultsQueue.empty():
		(partitionWeights, result) = resultsQueue.get()
		weightInfo[partitionWeights] = result

	return weightInfo

if __name__ == "__main__":
	## passing a function. That function contains things set in
	## 'rlenvironmentnonstatic' file
	## You have to make a function that creates the environment.
	## Make the game, then follow the layout in 'rlenvironmentnonstatic'
	
	filename = "examples.gridphysics.simpleGame4"
	game_to_play = lambda obsType: createRLInputGame(filename)
	# planUntilSolved(game_to_play, filename, 50, [5,1,5])
	# partitionWeightsList = [(5,1,5), (5,3,3)]
	# partitionWeightsList = [(5,1,5)]
	partitionWeightsList = [(5,1,5),(5,3,3), (5,3,1), (5,1,3), (3,1,5), (3,5,1), (1,3,5), (5,5,1), (1,5,3)]
	weightInfoList = []
	totalWeightInfo = {k: {'solved': 0, 'total_steps': 0, 'numActions': 0} for k in partitionWeightsList}
	numIters = 8
	for i in range(numIters):
		weightInfo = parallelizedPlanUntilSolved(game_to_play, filename, 50, partitionWeightsList, numWorkers=4)
		weightInfoList.append(weightInfo)
		for k in totalWeightInfo:
			totalWeightInfo[k]['solved'] = totalWeightInfo[k]['solved'] + (weightInfo[k]['solved']/float(numIters))
			totalWeightInfo[k]['total_steps'] += weightInfo[k]['total_steps']/float(numIters)
			if weightInfo[k]['solved']:
				totalWeightInfo[k]['numActions'] += weightInfo[k]['numActions']

	for k in totalWeightInfo:
		totalWeightInfo[k]['numActions'] /= float(totalWeightInfo[k]['solved'])

	embed()

	# planActLoop(game_to_play, filename, 5, 100, 50, playback=False)