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
import multiprocessing
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectObjectGoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, \
generateSymbolDict, ruleCluster, Theory, Game, writeTheoryToTxt, generateTheoryFromGame
from rlenvironmentnonstatic import createRLInputGame
import curses

#A hack to display things to the terminal conveniently.
np.core.arrayprint._line_width=250

ACTIONS = {(0,0):'stay',(0,-1):'up', (0,1):'down', (1,0):'right', (-1,0):'left', None:'none'}

class QLearner:
	def __init__(self, rle, gameString, levelString, episodes=100, memory=None, alpha=1, epsilon=.1, gamma=.9):
		self.rle = rle
		self.gameString = gameString
		self.levelString = levelString
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma
		self.beta=.5
		self.episodes = episodes
		self.actions = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
		# self.actions = [(1,0), (-1,0), (0,1), (0,-1)]
		self.QVals = defaultdict(lambda: (1./(1-self.gamma)))
		self.V ={}
		self.counts = defaultdict(lambda:0.)
		self.memory = memory ## provide dicts of Q-values from previous runs. Use some function to smoothe
		self.maxPseudoReward = 10
		self.pseudoRewardDecay = .99
		self.partitionWeights = [20,1]
		self.heuristicDecay = .99
		self.immovables = []
		goalLoc = self.findObjectInRLE(rle, 'goal')
		self.goalLoc = goalLoc
		self.rewardQueue = deque()
		self.processed = [goalLoc]
		# goalLocs = self.findObjectInRLE(rle, 'probe')
		# self.rewardDict = defaultdict(lambda:0)
		# for goalLoc in goalLocs:
			# self.rewardDict[goalLoc] = self.maxPseudoReward
		self.rewardDict = {goalLoc:self.maxPseudoReward}
		self.scanDomainForMovementOptions()
		# self.getSubgoals(3)
		self.propagateRewards(goalLoc)
		# print "propagatedRewards"
		# embed()
	def scanDomainForMovementOptions(self):
		##TODO: Take a state, so that you can re-perform this scan as needed and take changes into account.
		##TODO: query VGDL description for penetrable/nonpenetrable objects, add to list.
		immovable_codes = []
		# immovables = ['wall']
		try:
			immovables = self.rle.immovables
			self.immovables = immovables
			print "immovables", immovables
		except:
			immovables = ['wall']#,'poison']
			self.immovables = immovables
			print "Using defaults as immovables", immovables

		for i in immovables:
			if i in self.rle._obstypes.keys():
				immovable_codes.append(2**(1+sorted(self.rle._obstypes.keys())[::-1].index(i)))

		actionDict = defaultdict(list)
		neighborDict = defaultdict(list)
		action_superset = [(0,0),(-1,0), (1,0), (0,-1), (0,1)]
		# action_superset = [(-1,0), (1,0), (0,-1), (0,1)]
		
		board = np.reshape(self.rle._getSensors(), self.rle.outdim)
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

	# def propagateRewards(self, goalLocs):
	# 	for goalLoc in goalLocs:
	# 		embed()
	# 		rewardQueue = deque()
	# 		processed = [goalLoc]
	# 		rewardQueue.append(goalLoc)
	# 		for n in self.neighborDict[goalLoc]:
	# 			if n not in rewardQueue:
	# 				rewardQueue.append(n)

	# 		while len(rewardQueue)>0:
	# 			loc = rewardQueue.popleft()
	# 			if loc not in processed:
	# 				valid_neighbors = [n for n in self.neighborDict[loc] if n in self.rewardDict.keys()]
	# 				self.rewardDict[loc] += max([self.rewardDict[n] for n in valid_neighbors]) * self.pseudoRewardDecay
	# 				processed.append(loc)
	# 				for n in self.neighborDict[loc]:
	# 					if n not in processed:
	# 						rewardQueue.append(n)
	# 	return

	def findObjectInRLE(self, rle, objName):
		if objName not in rle._obstypes.keys():
			print objName, "not in rle."
			# return None
		objLocs = []
		objCode = 2**(1+sorted(self.rle._obstypes.keys())[::-1].index(objName))
		objLoc = np.where(np.reshape(self.rle._getSensors(), self.rle.outdim)==objCode)
		objLoc = objLoc[0][0], objLoc[1][0] #(y,x)
		return objLoc
		# if type(objLoc[0])==np.ndarray:
		# 	objLocs = [(o[1], o[0]) for o in objLoc]
		# else:
		# 	objLoc = objLoc[0][0], objLoc[1][0] #(y,x)
		# 	objLocs = [objLoc]
		# print objLocs
		# return objLocs
	
	def findAvatarInRLE(self, rle):
		avatar_code = 1
		state = np.reshape(rle._getSensors(), self.rle.outdim)
		if avatar_code in state:
			avatar_loc = np.where(state==avatar_code)
			avatar_loc = avatar_loc[0][0], avatar_loc[1][0]
		else:
			avatar_loc = None
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

	def findObjectInState(self, s, objName):
		##TODO: Finish last part of this function -- sometimes it can't access objloc[0][0], objloc[1][0]
		state = np.reshape(np.fromstring(s,dtype=float), self.rle.outdim)
		if objName not in rle._obstypes.keys():
			print objName, "not in rle."
			return None
		objCode = 2**(1+sorted(self.rle._obstypes.keys())[::-1].index(objName))
		objLoc = np.where(state==objCode)
		try:
			objLoc = objLoc[0][0], objLoc[1][0] #(y,x)
		except:
			print "can't find objloc"
			embed()
		return objLoc

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
			print "didn't find path to goal in getSubgoals (in getPathToGoal)"
			return False
			# embed()
			# raise Exception("Didn't find a path to the goal location.")
		return path

	def getSubgoals(self, subgoal_path_threshold):
		avatar_loc = self.findAvatarInRLE(self.rle)
		## find location of goal, add to rewardDict.
		## also add neighbors of goal rewardQueue.
		##TODO: update this if goal moves!!
		if "goal" not in self.rle._obstypes.keys():
			print "no goal to get subgoals to"
			return []
		goal_code = 2**(1+sorted(self.rle._obstypes.keys())[::-1].index("goal"))
		killerObjectCodes = []
		if hasattr(self.rle, 'killerObjects'):
			for o in self.rle.killerObjects:
				if o in self.rle._obstypes.keys():
					killerObjectCodes.append(2**(1+sorted(self.rle._obstypes.keys())[::-1].index(o)))
		board = np.reshape(self.rle._getSensors(), self.rle.outdim)
		goal_loc = np.where(board==goal_code)
		goal_loc = goal_loc[0][0], goal_loc[1][0]
		self.subgoals = []
		# print "showing RLE we're getting path for."
		# print self.rle.show()
		path = self.getPathToGoal(avatar_loc, goal_loc)

		if not path:
			return [] ## so that you can try pursuing a different goal

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


				## Doesn't add subgoals that correspond to objects we know to be dangerous.
				if board[path[subgoal_index][0]][path[subgoal_index][1]] not in killerObjectCodes:
					self.subgoals.append(path[subgoal_index])
				else:
					for i in range(1, subgoal_path_threshold):
						# print "path fell on a killer object. changing path slightly."
						try:
							if path[subgoal_index-i] not in killerObjectCodes:
								self.subgoals.append(path[subgoal_index-i])
								# print "found altered path", path[subgoal_index-i]
								break
							elif path[subgoal_index+i] not in killerObjectCodes:
								self.subgoals.append(path[subgoal_index+i])
								# print "found altered path", path[subgoal_index+i]
								break
						except:
							print "indices didn't work out in looking for different path"
				# self.subgoals.append(path[subgoal_index])

		
		return self.subgoals

	# def selectAction(self, s, policy, partitionWeights = None, domainKnowledge=True, printout=False):
	# 	if policy == 'epsilonGreedy':
	# 		if random.random() < self.epsilon:
	# 			return random.choice(self.actions)
	# 		else:
	# 			bestQVal, bestA, QValsAreAllEqual = self.bestSA(s, partitionWeights, domainKnowledge = True)
	# 			return bestA
	# 	elif policy == 'greedy':
	# 		bestQVal, bestA, QValsAreAllEqual = self.bestSA(s, partitionWeights = [1,0], domainKnowledge = True)
	# 		if printout:
	# 			print bestQVal
	# 		if QValsAreAllEqual:
	# 			return None
	# 		else:
	# 			return bestA

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

	# def bestSA(self, s, partitionWeights, domainKnowledge=True, debug=False):
	# 	avatarLoc = self.findAvatarInState(s)
	# 	if domainKnowledge:
	# 		if len(self.actionDict[avatarLoc])>0:
	# 			actions = self.actionDict[avatarLoc]
	# 		else:
	# 			# set actions to full action set in case the actionDict was initialized with incorrect assumptions
	# 			# ... and thus thinks there's nothing you can do from the current state.
	# 			actions = self.actions
	# 	else:
	# 		actions = self.actions
		
	# 	random.shuffle(actions)
	# 	maxFuncVal = -float('inf')
	# 	sumQVal = 0.
	# 	sumPseudoReward = 0.
	# 	rewardCoefficient = partitionWeights[0]
	# 	heuristicCoefficient = partitionWeights[1]
	# 	# print heuristicCoefficient
	# 	bestAction = None
	# 	QValsAreAllEqual = False
	# 	if debug:
	# 		print "debugging bestSA"
	# 		embed()
	# 	for a in actions:
	# 		if (s,a) not in self.QVals.keys():
	# 			self.QVals[(s,a)] = 0.
	# 		sumQVal += abs(self.QVals[(s,a)])
	# 		sumPseudoReward += self.getPseudoReward(s, a)

	# 	for a in actions:

	# 		if sumQVal == 0.:
	# 			QValFunction = 0.
	# 		else:
	# 			QValFunction = self.QVals[(s,a)]/sumQVal

	# 		if sumPseudoReward == 0:
	# 			pseudoRewardFunction =0.
	# 		else:
	# 			pseudoRewardFunction = self.getPseudoReward(s,a)/sumPseudoReward

	# 		funcVal = rewardCoefficient*QValFunction + \
	# 					heuristicCoefficient*pseudoRewardFunction

	# 		if funcVal > maxFuncVal:
	# 			maxFuncVal = funcVal
	# 			bestAction = a
	# 			bestQVal = self.QVals[(s,a)]
		
	# 	if not bestAction:
	# 		try:
	# 			bestAction = random.choice(actions)
	# 			bestQVal = self.QVals[(s,a)]
	# 			QValsAreAllEqual = True
	# 		except:
	# 			print "actions array is empty. in bestSA"
	# 			print np.reshape(np.fromstring(s,dtype=float),self.rle.outdim)
	# 			embed()

	# 	# QVals = [self.QVals[(s,a)] for a in actions]
	# 	# if len(QVals)>0:
	# 	# 	bestQVal = max(QVals)
	# 	# 	QValsAreAllEqual = bestQVal == min(QVals)
	# 	# 	bestA = actions[QVals.index(bestQVal)]
	# 	# else:
	# 	# 	bestQVal, QValsAreAllEqual, bestA = 0., True, None
	# 	return bestQVal, bestAction, QValsAreAllEqual

	# def update(self, s, a, sPrime, r):
	# 	bestQVal, bestA, QValsAreAllEqual = self.bestSA(sPrime, self.partitionWeights)
	# 	bestQVal
	# 	# if (s,a) in self.QVals.keys():
	# 	try:
	# 		self.QVals[(s,a)] = self.QVals[(s,a)] + self.alpha * (r + self.gamma*bestQVal - self.QVals[(s,a)])
	# 	# else:
	# 	# 	self.QVals[(s,a)] = 0 + self.alpha * (r + self.gamma*bestQVal - self.QVals[(s,a)])
	# 	except:
	# 		print "didn't find qvals[s,a]"
	# 		embed()

	def H(self, avatarLoc, goalLoc):
		#returning negative number becuase we want to maximize reward+heuristic.
		return -1.*(abs(self.goalLoc[0]-avatarLoc[0]) + abs(self.goalLoc[1]-avatarLoc[1]))


	# def getVal(self, s, heuristicVal):
	# 	# For Barto RTDP implementation
	# 	if s in self.V.keys():
	# 		return self.V[s]
	# 	else:
	# 		try:
	# 			avatarLoc = self.findAvatarInState(sPrime)
	# 			goalLoc = self.findObjectInState(sPrime, 'goal')
	# 			return self.H(avatarLoc, goalLoc)
	# 		except:
	# 			return heuristicVal

	def getVal(self, s):
		avatarLoc = self.findAvatarInState(s)
		if avatarLoc is None:
				return -10.
		else:
			actions = self.actionDict[avatarLoc]
			return max([self.QVals[(s,a)] for a in actions])
	def exploration(self, avatarLoc, a, actions):
		if self.counts[(avatarLoc, a)] == 0:
			return self.beta
		else:
			# print avatarLoc, a, self.counts[(avatarLoc,a)]
			# print "beta and sqrt", self.beta, math.sqrt(self.counts[(avatarLoc, a)])
			return self.beta/math.sqrt(self.counts[(avatarLoc,a)])
		# allCounts = [self.counts[(avatarLoc,action)] for action in actions]
		# z = sum(allCounts)
		# if z>0:
		# 	return self.counts[(avatarLoc,a)]/z
		# else:
		# 	return 0.

	# def evaluateActions(self, rle, s):
	# 	## Partial implementation of Barto's RTDP algorithm. But exploration term was added later.
	# 	avatarLoc = self.findAvatarInState(s)
	# 	actions = self.actionDict[avatarLoc]
	# 	simulationResults = {}
	# 	for a in actions:
	# 		Vrle = copy.deepcopy(rle)
	# 		#calculate these before the next state, while we know that the avatar and goal exist
	# 		nextLoc = avatarLoc[0]+a[1], avatarLoc[1]+a[0] #y,x
	# 		goalLoc = self.findObjectInState(s, 'goal')
	# 		nextStateHeuristicVal = self.H(nextLoc, goalLoc) 
	# 		res = Vrle.step(a)
	# 		sPrime = Vrle._getSensors().tostring()
	# 		simulationResults[a]=Vrle
	# 		v = self.getVal(sPrime, nextStateHeuristicVal)
	# 		#Update equation for RTDP involves summing out the s', but we're in a deterministic domain.
	# 		self.QVals[(s,a)] = res['reward'] + self.gamma*v + self.exploration(avatarLoc, a, actions)
	# 		embed()
	# 		print a, res['reward'], v, self.exploration(avatarLoc,a,actions)
	# 	return simulationResults

	def evaluateActions(self, rle, s):
		avatarLoc = self.findAvatarInState(s)
		actions = self.actionDict[avatarLoc]
		simulationResults = {}
		for a in actions:
			Vrle = copy.deepcopy(rle)
			res = Vrle.step(a)
			sPrime = Vrle._getSensors().tostring()
			simulationResults[a]=Vrle
			v = self.getVal(sPrime)
			#Update equation for RTDP involves summing out the s', but we're in a deterministic domain.
			self.QVals[(s,a)] = res['reward'] + self.gamma*v + self.exploration(avatarLoc, a, actions)
			# embed()
			# print a, res['reward'], v, self.exploration(avatarLoc,a,actions)
		return simulationResults
	
	def selectAction(self, s):
		bestVal = -float('inf')
		bestActions = []
		avatarLoc = self.findAvatarInState(s)
		actions = self.actionDict[avatarLoc]
		for a in actions:
			if self.QVals[(s,a)]>bestVal:
				bestVal = self.QVals[(s,a)]
				bestActions = [a]
			elif self.QVals[(s,a)]==bestVal:
				bestActions.append(a)
		chosenAction = random.choice(bestActions)
		return chosenAction

	def runEpisode(self, stepLimit=float('inf')):
		rle = copy.deepcopy(self.rle)
		terminal = rle._isDone()[0]
		i=0
		total_reward = 0.
		s = rle._getSensors().tostring()
		while not terminal and i<stepLimit:
			# simualtionResults stores the result of taking each action in a copy of the RLE at that point.
			# we want to use what happened in the simulation (for which we stored values) to continue the episode.
			simulationResults = self.evaluateActions(rle, s)
			a = self.selectAction(s)
			avatarLoc = self.findAvatarInState(s)
			# print "counts before", self.counts[(avatarLoc,a)]
			self.counts[(avatarLoc, a)] += 1
			# print "counts after", self.counts[(avatarLoc,a)]
			# print ""
			self.V[s] = self.QVals[(s,a)]
			rle = simulationResults[a]
			## UNCOMMENT HERE IF YOU WANT TO WATCH Q-learner learning.
			print rle.show()
			terminal = rle._isDone()[0]
			i += 1
			if not terminal:
				s = rle._getSensors().tostring() #get actual resulting state


	def learn(self, episodes, satisfice=200):
		t1 = time.time()
		satisfice_episodes = 0
		for i in range(episodes):
			# sys.stdout.write("Episodes: {}\r".format(i) )
			# sys.stdout.flush()
			self.runEpisode(stepLimit=150)
			# satisfice_episodes +=1
			if i%10==0:
				s = self.rle._getSensors().tostring()
			# 	a = self.selectAction(s, policy='epsilonGreedy', partitionWeights = self.partitionWeights)
			# 	print i, self.QVals[(s,a)]
			# 	if satisfice: ## see if values have propagated to start state; if so, return.
			# 		actions = self.getBestActionsForPlayout()
			# 		if len(actions)>0:
			# 			if satisfice_episodes>satisfice:
			# 				return i
						
						# print "satisfice found actions in", time.time()-t1, "seconds."
						# return i

		return i

	def getBestActionsForPlayout(self, showActions = False):
		rle = copy.deepcopy(self.rle)
		terminal = rle._isDone()[0]
		s = rle._getSensors().tostring()
		actions = []
		# print rle.show()
		while not terminal:
			a = self.selectAction(s)

			actions.append(a)
			res = rle.step(a)
			if showActions:
				print rle.show()
			terminal = rle._isDone()[0]
			s = res['observation'].tostring()
		return actions

	def backwardsPlayback(self):
		lst = [(k,v) for k,v in self.QVals.iteritems()]
		slist = sorted(lst, key=lambda x:x[1])
		slist.reverse()
		for l in slist:
			if l[1]>0:
				print np.reshape(np.fromstring(l[0][0],dtype=float),self.rle.outdim)
				print l[1]

if __name__ == "__main__":
	
	# gameFilename = "examples.gridphysics.simpleGame_push_boulders_multigoal"
	# gameFilename = "examples.gridphysics.waypointtheory" 
	# gameFilename = "examples.gridphysics.simpleGame_teleport"
	# gameFilename = "examples.gridphysics.demo_dodge"
	# gameFilename = "examples.gridphysics.scoretest" 
	# gameFilename = "examples.gridphysics.demo_transform_small" 
	gameFilename = "examples.gridphysics.simpleGame_push_boulders"
	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()
	print rle.show()
	# rle.immovables = ['wall', 'poison1', 'poison2']
	print ""
	print "Initializing learner. Playing", gameFilename
	ql = QLearner(rle, gameString, levelString, alpha=1, epsilon=.5, gamma=.9, episodes=1000)
	# ql.getBestActionsForPlayout(False, True)
	# for x in range(10):
	# 	print '{0}\r'.format(x),
	# print
	# embed()
	t1 = time.time()
	# ql.runEpisode()
	ql.learn(50, satisfice=10)
	t2 = time.time() - t1
	print "done in {} seconds".format(t2)
	# ql.learn(100, satisfice=False)

	embed()