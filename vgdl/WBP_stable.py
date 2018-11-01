from IPython import embed
from planner import *
import itertools

from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT
NONE = 0
ACTIONS = [K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, NONE]
actionDict = {K_SPACE: 'space', K_UP: 'up', K_DOWN: 'down', K_LEFT: 'left', K_RIGHT: 'right', NONE: 'wait'}

## Base class for width-based planners (IW(k) and 2BFS)
class WBP(Planner):
	def __init__(self, rle, gameString, levelString, gameFilename):
		Planner.__init__(self, rle, gameString, levelString, gameFilename, display=1)
		self.T = len(rle._obstypes.keys())+1 #number of object types. Adding avatar, which is not in obstypes.
		self.vecDim = [rle.outdim[0]*rle.outdim[1], 2, self.T]
		self.trueAtoms = set() ## set of atoms that have been true at some point thus far in the planner.
		self.objectTypes = rle._game.sprite_groups.keys()
		self.objectTypes.sort()
		self.phiSize = sum([len(rle._game.sprite_groups[k]) for k in rle._game.sprite_groups.keys() if k not in ['wall', 'avatar']])
		self.objIDs = {}
		self.maxNumObjects = 6
		self.trackTokens = False
		self.vecSize = None
		self.addWaitAction = True
		self.padding = 5  ##5 is arbitrary; just to make sure we don't get overlap when we add positions
		i=1
		for k in rle._game.all_objects.keys():
			self.objIDs[k] = i * (rle.outdim[0]*rle.outdim[1]+self.padding)
			i+=1
		self.addSpaceBarToActions()

	def addSpaceBarToActions(self):
		## Note: if an object that isn't instantiated in the beginning is of a class that 
		## spacebar applies to, we won't pick up on it here.
		shootingClasses = ['MarioAvatar', 'ClimbingAvatar', 'ShootAvatar', 'Switch', 'FlakAvatar']
		classes = [str(o[0].__class__) for o in self.rle._game.sprite_groups.values() if len(o)>0]
		spacebarAvailable = False
		for sc in shootingClasses:
			if any([sc in c for c in classes]):
				spacebarAvailable = True
				break
		if spacebarAvailable:
			self.actions = [K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT]
		else:
			self.actions = [K_UP, K_DOWN, K_LEFT, K_RIGHT]
		if self.addWaitAction:
			self.actions.append(NONE)
		return

	def calculateAtoms(self, rle):
		lst = []
		for k in rle._game.sprite_groups.keys():
			for o in rle._game.sprite_groups[k]:
				if o not in rle._game.kill_list:
					## turn location into vector posd2[ition (rows appended one after the other.)
					pos = rle._rect2pos(o.rect) #x,y
					vecValue = pos[1] + pos[0]*rle.outdim[0] + 1
				else:
					vecValue = 0
				objPosCombination = self.objIDs[o.ID] + vecValue
				lst.append(objPosCombination)
		present = []
		for k in [t for t in self.objectTypes if t not in ['wall', 'avatar']]: ##maybe add the avatar to this global state
			# for o in rle._game.sprite_groups[k]:
			for o in sorted(rle._game.sprite_groups[k], key=lambda s:s.ID):
				if o not in rle._game.kill_list:
					present.append(1)
				else:
					present.append(0)
		ind = sum([present[i]*2**i for i in range(len(present))])
		lst.append(ind)
		if not self.vecSize:
			self.vecSize = len(lst)
			print "Vector is length {}".format(self.vecSize)
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

	def novelty(self, node, k, update=False):
		# Returns number of k-tuples of atoms that are newly true in node.
		newAtoms = self.delta(node.parent, node)
		# print "in novelty fn"
		# embed()
		if len(self.trueAtoms) > 0:
			trueAtoms = node.state
			oldTrueAtoms = set(trueAtoms)-set(newAtoms)
			candidates = []
			for i in range(1,k+1):
				newPart = list(itertools.combinations(newAtoms, i))
				oldPart = list(itertools.combinations(oldTrueAtoms, k-i))
				unflattened = list(itertools.product(newPart, oldPart))
				flattened = [frozenset((u[0]+u[1])) for u in unflattened]	
				candidates.extend(flattened)
		else:
			candidates = [frozenset(p) for p in list(itertools.combinations(newAtoms, k))]
			# candidates = list(itertools.combinations(newAtoms, k))
		node.candidates = candidates
		newTuples = set()
		for aT in candidates:
			if aT not in self.trueAtoms:
				if update:
					self.trueAtoms.add(aT)
				newTuples.add(aT)
		# print 'in novelty'
		# print len(newTuples)
		# embed()
		return len(newTuples)
		
def noveltyHeuristic(lst, WBP, k, surrogateCall=False, threshold=False):
	#returns the node in lst that has highest novelty measure
	## you need to not change the noveltyDict when you evaluate the novelty. Only change the dict if you select the state!
	
	if threshold:
		choices = [n for n in lst if n.novelty>0]
		if len(choices)>0:
			return random.choice(choices)
		else:
			if not surrogateCall:
				return None
			else:
				return random.choice(lst)
	else:
		maxNovelty = max([n.novelty for n in lst])
		# print 'in novelty'
		# embed()
		if maxNovelty==0 and not surrogateCall:
			return None
		elif not surrogateCall:
			bestNodes = [n for n in lst if n.novelty==maxNovelty]
			# bestNodes = [n for n in lst if n.novelty>0]
			if len(bestNodes)==1:
				return bestNodes[0]
			elif len(bestNodes)>1:
			 	return rewardHeuristic(bestNodes, WBP, k, surrogateCall=True)
		elif surrogateCall:
			minNovelty = min([n.novelty for n in lst if n.novelty>0])
			bestNodes = [n for n in lst if n.novelty==minNovelty]
	 		return random.choice(bestNodes)
		else:
			print "found 0 nodes in noveltyHeuristic"
			embed()

def rewardHeuristic(lst, WBP, k, surrogateCall=False):
	maxReward = max([n.metabolic_reward for n in lst]) ## n.metabolic_reward includes the game reward
	bestNodes = [n for n in lst if n.metabolic_reward==maxReward]
	# maxReward = max([n.reward for n in lst])
	# bestNodes = [n for n in lst if n.reward==maxReward]
	# print 'in reward'
	# embed()
	if len(bestNodes)==1:
		return bestNodes[0]
	elif len(bestNodes)>1:
	 	if not surrogateCall:
	 		return noveltyHeuristic(bestNodes, WBP, k, surrogateCall=True)
	 	else:
	 		return random.choice(bestNodes)
	else:
		print "found 0 nodes in rewardHeuristic"
		embed()

def BFS_noNovelty(rle, WBP):
	Q = Queue()
	visited, rejected = [], []
	start = Node(rle, WBP, [], None)
	start.lastState = rle
	# visited.append(start)
	Q.put(start)
	while not Q.empty():
		current = Q.get()
		win = current.eval(updateNoveltyDict=True)
		if current.state not in visited:
			visited.append(current.state)
			if win:
				return current, visited, rejected
			for a in WBP.actions:
				child = Node(rle, WBP, current.actionSeq+[a], current)
				Q.put(child)
		else:
			rejected.append(current)
	print "no more states in queue"
	embed()
	return Q, visited, rejected

def BFS(rle, WBP):
	Q = Queue()
	visited, rejected = [], []
	start = Node(rle, WBP, [], None)
	start.lastState = rle
	# visited.append(start)
	Q.put(start)
	while not Q.empty():
		current = Q.get()
		win = current.eval(updateNoveltyDict=True)
		if current.novelty > 0:		
		# if current.state not in visited:
			visited.append(current)
			if win:
				return current, visited, rejected
			for a in WBP.actions:
				child = Node(rle, WBP, current.actionSeq+[a], current)
				Q.put(child)
		else:
			rejected.append(current)
	print "no more states in queue"
	embed()
	return Q, visited, rejected


def BFS3(rle, WBP):
	QNovelty, QReward = [], []
	visited, rejected = [], []
	start = Node(rle, WBP, [], None)
	start.lastState = rle
	visited.append(start)
	start.eval()
	QNovelty.append(start)
	QReward.append(start)
	i=0

	def noveltySelection():
		bestNodes = sorted(filter(lambda n: n.novelty>0, QNovelty), key=lambda n: (n.novelty, -n.metabolic_reward))
		# print [n.novelty for n in bestNodes]
		# embed()
		if len(bestNodes)>0:
			return bestNodes[0]
		else:
			return None

		# if len(QNovelty)==0:
			# return None
		# else:
			# acceptableNodes = [n for n in QNovelty if n.novelty>0]
		# print 'in noveltySelection'
		# embed()
		# return bestNodes[0]
		# maxNovelty = max([n.novelty for n in QNovelty])
		# acceptableNodes = [node for node in QNovelty if node.novelty>0]

		# if len(acceptableNodes) == 0:
		# 	return None
		# # if maxNovelty==0:
		# 	# return None
		# minNovelty = min([n.novelty for n in acceptableNodes])
		# bestNodes = [n for n in acceptableNodes if n.novelty==minNovelty]
		# # bestNodes = [n for n in QNovelty if n.novelty==maxNovelty]
		# if len(bestNodes)==1:
		# 	return bestNodes[0]
		# else:
		# 	maxReward = max([n.metabolic_reward for n in bestNodes])
		# 	return [n for n in bestNodes if n.metabolic_reward==maxReward][0]
			# return random.choice([n for n in bestNodes if n.metabolic_reward==maxReward])

	def rewardSelection():
		bestNodes = sorted(filter(lambda n:n.novelty>0, QReward), key=lambda n: (-n.metabolic_reward, n.novelty))
		# print "in rewardselection"
		# print [n.reward for n in bestNodes]
		# embed()
		if len(bestNodes)>0:
			return bestNodes[0]
		else:
			return None
		# maxReward = max([n.metabolic_reward for n in QReward])
		# bestNodes = [n for n in QReward if n.metabolic_reward==maxReward]
		# if len(bestNodes)==1:
		# 	return bestNodes[0]
		# else:
		# 	# acceptableNodes = [n for n in bestNodes if n.novelty>0]
		# 	minNovelty = min([n.novelty for n in bestNodes])
		# 	acceptableNodes = [n for n in bestNodes if n.novelty==minNovelty]
		# 	if len(acceptableNodes)>0:
		# 		return acceptableNodes[0]
		# 		# return random.choice(acceptableNodes)
		# 	else:
		# 		return None

	while len(QNovelty)>0 or len(QReward)>0:
		if i%2==0:
			current = noveltySelection()
		else:
			current = rewardSelection()

		## fix breakpoints.
		if current==None:
			pass
		else:
			if i%2==0:
				QNovelty.remove(current)		
			else:
				QReward.remove(current)
			print current.novelty
			print current.lastState.show()
			# QNovelty.remove(current)
			# QReward.remove(current)
			current.eval(updateNoveltyDict=True)
			visited.append(current)
			if current.win:
				return current, visited, rejected ##revisit
			else:
				for a in WBP.actions:
					child = Node(rle, WBP, current.actionSeq+[a], current)
					child.eval()
					# print child.novelty
					# print child.lastState.show()
					if child.novelty>0:
						QNovelty.append(child)		
					if child.metabolic_reward>0:
						QReward.append(child)
		i+=1
	return None, visited, rejected			

def BFS2(rle, WBP):
	Q = []
	visited, rejected = [], []
	start = Node(rle, WBP, [], None)
	start.lastState = rle
	visited.append(rle)
	start.eval()
	Q.append(start)
	i=0
	while len(Q)>0:
		if i%2==0:
			current = noveltyHeuristic(Q, WBP, WBP.k, surrogateCall=False, threshold=False)
			# print 'novelty'
		else:
			current = rewardHeuristic(Q, WBP, WBP.k, surrogateCall=False)
			# print 'reward'
		# print "_____"
		# print current.lastState.show()
		# print current.novelty, current.reward
		# embed()
		## This is not nec. right.
		if current is None:
			print "got no node"
			embed()
			return Q, visited, rejected
		else:
			print current.lastState.show()
			Q.remove(current)
			current.eval(updateNoveltyDict=True)
			visited.append(current)
			if current.win:
				return current, visited, rejected
			else:
				for a in WBP.actions:
					child = Node(rle, WBP, current.actionSeq+[a], current)
					child.eval()
					Q.append(child)		
			i+=1
	return Q, visited, rejected



class Node():
	def __init__(self, rle, WBP, actionSeq, parent):
		self.rle = rle
		self.WBP = WBP
		self.actionSeq = actionSeq
		self.parent = parent
		self.state = {}
		self.novelty = None
		self.reward = None
		self.children = None
		self.lastState = None
		self.reconstructed=False

	# try to copy parent lastState. Then take action and store as current lastState.
	## if that fails, replay from beginning and store as current lastState
	def eval(self, updateNoveltyDict=False):
		if self.parent and self.parent.lastState is not None:
			try:
				vrle = copy.deepcopy(self.parent.lastState)
				if len(self.actionSeq)>0:
					vrle.step(self.actionSeq[-1])
					terminal, win = vrle._isDone()
			except:
				# pass
				print "conditions met but copy failed"
				embed()
		else:
			self.reconstructed=True
			print "copy failed; replaying from top"
			vrle = copy.deepcopy(rle)
			terminal, win = vrle._isDone()
			i=0
			while not terminal and len(self.actionSeq)>i:
				vrle.step(self.actionSeq[i])
				terminal, win = vrle._isDone()
				i += 1
		# if len(self.actionSeq)>0:
			# print actionDict[self.actionSeq[-1]]
		self.updateObjIDs(vrle)
		# print vrle.show()
		# if len([o for o in vrle._game.sprite_groups['bullet'] if o not in vrle._game.kill_list]) > 1:
			# print "multiple bullets"
			# embed()

		self.state = self.WBP.calculateAtoms(vrle)
		self.lastState = vrle
		self.win = win
		self.novelty = self.WBP.novelty(self, self.WBP.k, update=updateNoveltyDict)
		self.reward = vrle._game.score
		self.metabolic_reward = vrle._game.metabolic_score
		return win

	def updateObjIDs(self, vrle):
		i = 0
		for objType in vrle._game.sprite_groups:
			for s in vrle._game.sprite_groups[objType]:
				if s.ID not in self.WBP.objIDs.keys():
					# print "in update IDs"
					# embed()
					if s.name=='bullet':
						s.ID = len([o for o in vrle._game.sprite_groups[objType] if o not in vrle._game.kill_list])
					else:
						s.ID = len(vrle._game.sprite_groups[objType])
					self.WBP.objIDs[s.ID] = (len(self.WBP.objIDs.keys())+1) * (self.rle.outdim[0]*self.rle.outdim[1]+self.WBP.padding)
					i+=1
		# print len(self.WBP.objIDs.keys())
		# print "updated {} objects".format(i)
		return

	def isTerminal(self):
		return self.rle._isDone()[0]

	def isWin(self):
		return self.rle._isDone()[1]

	def playBack(self):
		vrle = copy.deepcopy(self.rle)
		terminal = vrle._isDone()[0]
		i=0
		print vrle.show()
		while not terminal:
			a = self.actionSeq[i]
			print actionDict[a]
			vrle.step(a)
			print vrle.show()
			# vrle.step((0,0))
			# print vrle.show()
			# embed()
			terminal = vrle._isDone()[0]
			i+=1

class IW(WBP):
	def __init__(self, rle, gameString, levelString, gameFilename, k):
		WBP.__init__(self, rle, gameString, levelString, gameFilename)
		self.k = k

if __name__ == "__main__":
	
	# gameFilename = "examples.gridphysics.simpleGame4_small"

	## make better versions
	# gameFilename = "examples.gridphysics.demo_teleport" ##solved!!
	# gameFilename = "examples.gridphysics.movers3c" ##solved!!
	# gameFilename = "examples.gridphysics.rivercross" ## solved!!
	# gameFilename = "examples.gridphysics.simpleGame_push_boulders_multigoal" ## k=2 works!
	# gameFilename = "examples.gridphysics.demo_dodge"  ##solved!!
	# gameFilename = "examples.gridphysics.movers5" ##solved!!
	# gameFilename = "examples.gridphysics.demo_preconditions" ## k=2 works!
	# gameFilename = "examples.gridphysics.waterfall" ##solved!! 
	# gameFilename = "examples.gridphysics.frogs" ## worked with k=2.
	# gameFilename = "examples.gridphysics.pick_apples" ## worked with expanded phi!
	# gameFilename = "examples.gridphysics.scoretest" ##2BFS solves it!
	# gameFilename = "examples.gridphysics.demo_chaser"  ##easy version solved!
	# gameFilename = "examples.gridphysics.demo_helper"  ##easy version solved!
	gameFilename = "examples.gridphysics.demo_multigoal_and"  ##takes forever if you have many boxes and don't use 2BFS (with metabolic penalty)
	# gameFilename = "examples.gridphysics.simpleGame_push_boulders" 
	# gameFilename = "examples.gridphysics.chase" #yes!!!
	# gameFilename = "examples.gridphysics.survivezombies" # solvable, just not very fast if long timeout.
	# gameFilename = "examples.gridphysics.demo_transform_small" ## works

	# gameFilename = "examples.gridphysics.demo_helper"  ##
	# gameFilename = "examples.gridphysics.demo_transform" ##
	# gameFilename = "examples.gridphysics.simpleGame_missile" #later.
	# gameFilename = "examples.gridphysics.simpleGame_push_boulders2"

	# gameFilename = "examples.gridphysics.waypointtheory"  ##easy version solved!

	# gameFilename = "examples.gridphysics.demo_multigoal_and_score"  ##easy version solved!
	# gameFilename = "examples.gridphysics.demo_sokoban" #later
	# gameFilename = "examples.gridphysics.demo_sokoban_score" #later
	# gameFilename = "examples.gridphysics.portals" ## stochasticity breaks it

	## Continuous physics games can't work right now. RLE is discretized, getSensors() relies on this, and a lot of the induction/planning
	## architecture depends on that. Will take some work to do this well. Best plan is to shrink the grid squares and increase speeds/strengths of 
	## objects.
	# gameFilename = "examples.continuousphysics.mario"
	# gameFilename = "examples.gridphysics.boulderdash" #Game is buggy.
	# gameFilename = "examples.gridphysics.butterflies" #Game is buggy.


	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()
	p = IW(rle, gameString, levelString, gameFilename, k=2)
	# embed()
	# p.trackTokens = True
	t1 = time.time()
	# last, visited, rejected = BFS_noNovelty(rle, p)
	# last, visited, rejected = BFS(rle, p)
	# last, visited, rejected = BFS2(rle, p)
	last, visited, rejected = BFS3(rle, p)

	print time.time()-t1
	print len(visited), len(rejected)
	embed()
	# if not hasattr(last, 'actionSeq'):
	# 	print "Failed without tracking tokens. re-trying"
	# 	p.trackTokens = True
	# 	t1 = time.time()
	# 	last, visited, rejected, visitedStates = BFS(rle, p)
	# 	print time.time()-t1
	# 	print len(visited), len(rejected)
	# embed()

