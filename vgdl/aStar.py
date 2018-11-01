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

class Node:
	def __init__(self, rle, state, parent, action, g, h, terminal=False, win=False):
		self.rle = rle
		self.state = state
		self.parent = parent
		self.action = action #action that got us to Node
		self.g = g
		self.h = h
		self.terminal = terminal
		self.win = win
	
	def f(self):
		return self.g + self.h
class AStar:
	def __init__(self, rle, gameString, levelString):
		self.rle = rle
		self.gameString = gameString
		self.levelString = levelString
		# self.actions = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
		self.actions = [(1,0), (-1,0), (0,1), (0,-1)]
		self.maxPseudoReward = 1
		self.pseudoRewardDecay = .8
		goalLoc = self.findObjectInRLE(rle, 'goal')
		self.rewardDict = {goalLoc:self.maxPseudoReward}
		self.open = set()
		self.closed = set()

		self.scanDomainForMovementOptions()
		self.propagateRewards(goalLoc)

	def scanDomainForMovementOptions(self):
		##TODO: Take a state, so that you can re-perform this scan as needed and take changes into account.
		##TODO: query VGDL description for penetrable/nonpenetrable objects, add to list.
		immovable_codes = []
		# immovables = ['wall']
		try:
			immovables = self.rle.immovables
			# immovables = ['wall', 'poison']
			print "immovables", immovables
		except:
			immovables = ['wall', 'poison']
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
	def findObjectInRLE(self, rle, objName):
		if objName not in rle._obstypes.keys():
			print objName, "not in rle."
			return None
		objCode = 2**(1+sorted(self.rle._obstypes.keys())[::-1].index(objName))
		objLoc = np.where(np.reshape(self.rle._getSensors(), self.rle.outdim)==objCode)
		objLoc = objLoc[0][0], objLoc[1][0] #(y,x)
		return objLoc
	
	def findAvatarInRLE(self, rle):
		# avatar_code = 1
		# state = np.reshape(rle._getSensors(), self.rle.outdim)
		# if avatar_code in state:
		# 	avatar_loc = np.where(state==avatar_code)
		# 	avatar_loc = avatar_loc[0][0], avatar_loc[1][0]
		# else:
		# 	avatar_loc = None
		if rle._avatar:
			avatar_loc = rle._rect2pos(rle._avatar.rect)
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

	# def getPseudoReward(self, s, a):
	# 	## returns pseudoreward of taking action a from location currentLoc.
	# 	## gives pseudoReward[currentLoc] if a doesn't move states.
	# 	currentLoc = self.findAvatarInState(s)
	# 	if currentLoc:
	# 		nextLoc = currentLoc[0]+a[1], currentLoc[1]+a[0] #again, locations are (y,x) and actions are (x,y)
	# 	else:
	# 		return 0.
	# 	if nextLoc in self.rewardDict.keys():
	# 		return self.rewardDict[nextLoc]
	# 	elif currentLoc in self.rewardDict.keys():
	# 		return self.rewardDict[currentLoc]
	# 	else:
	# 		return 0.

	def manhattanDistance(self, avatarLoc, goalLoc):
		return abs(avatarLoc[0]-goalLoc[0]) + abs(avatarLoc[1] - goalLoc[1])

	def bestNode(self):
		bestVal = 9999
		bestNode = None
		for node in self.open:
			if node.f() < bestVal:
				bestVal = node.f()
				bestNode = node
		# bestVal = min([n.f() for n in self.open])
		# options = [n for n in self.open if n.f() == bestVal]
		# if any([(o.terminal and not o.win) for o in options]):
		# 	print "found bad option"
		# 	embed()
		# return random.choice(options)
		return bestNode

	def makeNeighbors(self, node):
		neighbors = []
		s = self.findAvatarInState(node.state)
		for a in self.actionDict[s]:
			newRLE = copy.deepcopy(node.rle)
			newRLE.step(a)
			terminal, win = newRLE._isDone()[0], newRLE._isDone()[1]
			if not terminal:
				avatarLoc, goalLoc = self.findAvatarInRLE(newRLE), self.findObjectInRLE(newRLE, 'goal')
				try:
					h = self.manhattanDistance(avatarLoc, goalLoc)
				except:
					print "couldn't find h"
					embed()
				win = False
			else:
				if win:
					h = 0
				else:
					h = float('inf') ## TODO: probably not a good call.
			newNode = Node(newRLE, newRLE._getSensors().tostring(), node, a, node.g+1, h, terminal, win)
			neighbors.append(newNode)
		return neighbors

	def search(self):
		rle = copy.deepcopy(self.rle)
		terminal, win = rle._isDone()[0], rle._isDone()[1]
		s = rle._getSensors().tostring()
		bestChildSum, makeneighborSum, loopSum = 0, 0, 0
		i=0
		total_reward = 0.	
		avatarLoc, goalLoc = self.findAvatarInRLE(rle), self.findObjectInRLE(rle, 'goal')
		try:
			node = Node(rle, s, None, (0,0), 0., self.manhattanDistance(avatarLoc, goalLoc), terminal, win)
		except:
			print "couldn't make node becuase of manhattan distance"
			embed()
		self.open.add(node)

		while len(self.open)>0:
			t1 = time.time()
			current = self.bestNode()
			# print current.f(), len(self.open), [o.f() for o in self.open]
			bestChildSum += time.time() - t1
			if current.terminal and current.win:
				print 'bestChildSum, makeneighborsum, loopsum', bestChildSum, makeneighborSum, loopSum
				return self.constructPath(current)
			else:
				self.open.remove(current)
				self.closed.add(current)
				t2 = time.time()
				Neighbors = self.makeNeighbors(current)
				makeneighborSum = time.time() - t2
				for neighbor in Neighbors:
					if neighbor.state not in [n.state for n in self.closed]:
						# neighbor.f = neighbor.g + newNode.h
						if neighbor.state not in [n.state for n in self.open]:
							self.open.add(neighbor)
						else:
							neighbors = [n for n in self.open if n.state==neighbor.state]
							if len(neighbors)>1:
								print "found more than one neighbor"
								embed()
							openNeighbor = neighbors[0]

							# openNeighbor = [n for n in self.open if n.state==neighbor.state][0]

							if neighbor.g < openNeighbor.g:
								openNeighbor.g = neighbor.g
								openNeighbor.parent = neighbor.parent
			loopSum += time.time() - t1
			# print i
			i +=1
		print 'bestChildSum, makeneighborsum, loopsum', bestChildSum, makeneighborSum, loopSum
		return False

	def constructPath(self, node):
		path, actions = [], []
		path.append(node)
		while node.parent is not None:
			node = node.parent
			path.insert(0, node) ##prepend
			actions.insert(0, node.action)
		return path, actions[1:]

	def playPath(self, path):
		for p in path:
			print p.rle.show()

if __name__ == "__main__":
	
	# gameFilename = "examples.gridphysics.simpleGame4_small"
	# gameFilename = "examples.gridphysics.simpleGame_teleport"
	# gameFilename = "examples.gridphysics.simpleGame_many_poisons_huge"
	gameFilename = "examples.gridphysics.movers2b"
	print ""
	print "initializing AStar on game", gameFilename
	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()
	print rle.show()
	agent = AStar(rle, gameString, levelString)
	path, actions = agent.search()
	print "found path"

	embed()