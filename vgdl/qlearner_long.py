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
from planner import *

#A hack to display things to the terminal conveniently.
np.core.arrayprint._line_width=250

ACTIONS = {(0,0):'stay',(0,-1):'up', (0,1):'down', (1,0):'right', (-1,0):'left', None:'none'}

class QLearner:
	def __init__(self, rle, gameString, levelString, episodes=100, memory=None, alpha=1, epsilon=.1, gamma=.9, partitionWeights = [20,1], stepLimit=100):
		self.rle = rle
		self.gameString = gameString
		self.levelString = levelString
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma
		self.episodes = episodes
		self.actions = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
		# self.actions = [(1,0), (-1,0), (0,1), (0,-1)]
		self.QVals = defaultdict(lambda:0)
		self.memory = memory ## provide dicts of Q-values from previous runs. Use some function to smoothe
		self.maxPseudoReward = 1
		self.pseudoRewardDecay = .99
		self.partitionWeights = partitionWeights
		self.stepLimit=stepLimit
		self.heuristicDecay = .99
		self.immovables = []
		goalLoc = self.findObjectInRLE(rle, 'goal')
		self.rewardDict = {goalLoc:self.maxPseudoReward}
		self.scanDomainForMovementOptions()
		# self.getSubgoals(3)
		self.propagateRewards(goalLoc)

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
			immovables = ['wall', 'poison']
			self.immovables = immovables
			print "Using defaults as immovables", immovables

		for i in immovables:
			if i in self.rle._obstypes.keys():
				immovable_codes.append(2**(1+sorted(self.rle._obstypes.keys())[::-1].index(i)))

		actionDict = defaultdict(list)
		neighborDict = defaultdict(list)
		# action_superset = [(0,0),(-1,0), (1,0), (0,-1), (0,1)]
		action_superset = [(-1,0), (1,0), (0,-1), (0,1)]
		
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
	
	def propagateRewards(self, goalLoc):
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
				self.rewardDict[loc] = max([self.rewardDict[n] for n in valid_neighbors]) * self.pseudoRewardDecay
				processed.append(loc)
				for n in self.neighborDict[loc]:
					if n not in processed:
						rewardQueue.append(n)
		return

	def findObjectInRLE(self, rle, objName):
		if objName not in rle._obstypes.keys():
			print objName, "not in rle."
			return None
		objCode = 2**(1+sorted(self.rle._obstypes.keys())[::-1].index(objName))
		objLoc = np.where(np.reshape(self.rle._getSensors(), self.rle.outdim)==objCode)
		objLoc = objLoc[0][0], objLoc[1][0] #(y,x)
		return objLoc
	
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


	def selectAction(self, s, policy, partitionWeights = None, domainKnowledge=True, printout=False):
		if policy == 'epsilonGreedy':
			if random.random() < self.epsilon:
				return random.choice(self.actions)
			else:
				bestQVal, bestA, QValsAreAllEqual = self.bestSA(s, partitionWeights, domainKnowledge = True)
				return bestA
		elif policy == 'greedy':
			bestQVal, bestA, QValsAreAllEqual = self.bestSA(s, partitionWeights = [1,0], domainKnowledge = True)
			if printout:
				print bestQVal
			if QValsAreAllEqual:
				return None
			else:
				return bestA

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

	def bestSA(self, s, partitionWeights, domainKnowledge=True, debug=False):
		avatarLoc = self.findAvatarInState(s)
		if domainKnowledge:
			if len(self.actionDict[avatarLoc])>0:
				actions = self.actionDict[avatarLoc]
			else:
				# set actions to full action set in case the actionDict was initialized with incorrect assumptions
				# ... and thus thinks there's nothing you can do from the current state.
				actions = self.actions
		else:
			actions = self.actions
		
		random.shuffle(actions)
		maxFuncVal = -float('inf')
		sumQVal = 0.
		sumPseudoReward = 0.
		rewardCoefficient = partitionWeights[0]
		heuristicCoefficient = partitionWeights[1]
		# print heuristicCoefficient
		bestAction = None
		QValsAreAllEqual = False
		if debug:
			print "debugging bestSA"
			embed()
		for a in actions:
			if (s,a) not in self.QVals.keys():
				self.QVals[(s,a)] = 0.
			sumQVal += abs(self.QVals[(s,a)])
			sumPseudoReward += self.getPseudoReward(s, a)

		for a in actions:

			if sumQVal == 0.:
				QValFunction = 0.
			else:
				QValFunction = self.QVals[(s,a)]/sumQVal

			if sumPseudoReward == 0:
				pseudoRewardFunction = 0.
			else:
				pseudoRewardFunction = self.getPseudoReward(s,a)/sumPseudoReward

			funcVal = rewardCoefficient*QValFunction + \
						heuristicCoefficient*pseudoRewardFunction

			if funcVal > maxFuncVal:
				maxFuncVal = funcVal
				bestAction = a
				bestQVal = self.QVals[(s,a)]
		
		if not bestAction:
			try:
				bestAction = random.choice(actions)
				bestQVal = self.QVals[(s,a)]
				QValsAreAllEqual = True
			except:
				print "actions array is empty. in bestSA"
				print np.reshape(np.fromstring(s,dtype=float),self.rle.outdim)
				embed()

		# QVals = [self.QVals[(s,a)] for a in actions]
		# if len(QVals)>0:
		# 	bestQVal = max(QVals)
		# 	QValsAreAllEqual = bestQVal == min(QVals)
		# 	bestA = actions[QVals.index(bestQVal)]
		# else:
		# 	bestQVal, QValsAreAllEqual, bestA = 0., True, None
		return bestQVal, bestAction, QValsAreAllEqual

	def update(self, s, a, sPrime, r):
		bestQVal, bestA, QValsAreAllEqual = self.bestSA(sPrime, self.partitionWeights)
		bestQVal
		try:
			self.QVals[(s,a)] = self.QVals[(s,a)] + self.alpha * (r + self.gamma*bestQVal - self.QVals[(s,a)])
		except:
			print "didn't find qvals[s,a]"
			embed()

	def runEpisode(self):
		rle = copy.deepcopy(self.rle)
		terminal = rle._isDone()[0]
		s = rle._getSensors().tostring()
		i=0
		total_reward = 0.

		while not terminal and i<self.stepLimit:
			

			a = self.selectAction(s, policy='epsilonGreedy', partitionWeights = self.partitionWeights)
			res = rle.step(a)
			sPrime, r = res['observation'].tostring(), res['reward']
			# if rle._isDone()[1]==True:
			# 	# print 'won'
			# 	r = 100
			## UNCOMMENT HERE IF YOU WANT TO WATCH Q-learner learning.
			# print rle.show()

			## lower epsilon once you've found winning states.
			# if rle._isDone()[1]:
				# self.partitionWeights[1] = self.partitionWeights[1]*self.heuristicDecay
				# self.epsilon = self.epsilon*self.heuristicDecay
				# print self.partitionWeights
				# print 'reward'

			self.update(s,a,sPrime,r)
			s = sPrime
			terminal = rle._isDone()[0]

			i += 1
			total_reward += r

		self.QVals[s] = 0.

	def learn(self, satisfice=0):
		t1 = time.time()
		satisfice_episodes = 0
		for i in range(self.episodes):
			# sys.stdout.write("Episodes: {}\r".format(i) )
			# sys.stdout.flush()
			self.runEpisode()
			satisfice_episodes +=1
			if i%10==0:
				s = self.rle._getSensors().tostring()
				a = self.selectAction(s, policy='greedy', partitionWeights = self.partitionWeights)
				print i, self.QVals[(s,a)]
				if satisfice: ## see if values have propagated to start state; if so, return.
					actions = self.getBestActionsForPlayout()
					if len(actions)>0:
						if satisfice_episodes>satisfice:
							return i
			if i%100==0:
				self.getBestActionsForPlayout(False, True)

		return i

	def getBestActionsForPlayout(self, aggressive=False, showActions = False):
		rle = copy.deepcopy(self.rle)
		terminal = rle._isDone()[0]
		s = rle._getSensors().tostring()
		actions = []
		# print rle.show()
		while not terminal:
			a = self.selectAction(s, policy='greedy', partitionWeights = None, domainKnowledge = None, printout = False)
			# print self.QVals[(s,a)]
			if aggressive:
				if a is None:
					return actions
			else:
				if a is None or self.QVals[(s,a)]<=0:
					# print "Negative q-values or no action. Breaking."
					return actions
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
	# gameFilename = "examples.gridphysics.demo_teleport"
	# gameFilename = "examples.gridphysics.movers3c"
	# gameFilename = "examples.gridphysics.scoretest" 
	# gameFilename = "examples.gridphysics.portals" 
	gameFilename = "examples.gridphysics.demo_transform_small" 

	# gameFilename = "examples.gridphysics.demo_dodge" 
	# gameFilename = "examples.gridphysics.rivercross" 
	# gameFilename = "examples.gridphysics.demo_chaser_transformer" 

	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()

	p = Planner(rle, gameString, levelString)
	embed()
	# print rle.show()
	# print ""
	# print "Initializing learner. Playing", gameFilename
	# ql = QLearner(rle, gameString, levelString, alpha=1, epsilon=.5, gamma=.9, \
	# 	episodes=200, partitionWeights=[20,0], stepLimit=100)
	# # ql.immovables = ['wall', 'poison']

	# # embed()
	# t1 = time.time()
	# ql.learn(satisfice=0)
	# time_elapsed = time.time()-t1
	# print "Game: {}".format(gameFilename)
	# print "Ran {} rounds in {} seconds".format(ql.episodes, time_elapsed)
	# print "Immovables: {}".format(ql.immovables)
	# print "Parameters: Epsilon: {}. Gamma: {}. PartitionWeights: {}".format(ql.epsilon, ql.gamma, ql.partitionWeights)
	# print ""

	# embed()