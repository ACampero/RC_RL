from IPython import embed
from planner import *
import itertools

from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT
ACTIONS = [K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT]
actionDict = {K_SPACE: 'space', K_UP: 'up', K_DOWN: 'down', K_LEFT: 'left', K_RIGHT: 'right'}
# ACTIONS = [(1,0), (-1,0), (0,1), (0,-1)]

## Base class for width-based planners (IW(k) and 2BFS)
class WBP(Planner):
	def __init__(self, rle, gameString, levelString, gameFilename, display):
		Planner.__init__(self, rle, gameString, levelString, gameFilename, display)
		self.T = len(rle._obstypes.keys())+1 #number of object types. Adding avatar, which is not in obstypes.
		self.vecDim = [rle.outdim[0]*rle.outdim[1], 2, self.T]
		# self.noveltyDict = []
		self.trueAtoms = set() ## set of atoms that have been true at some point thus far in the planner.
		self.objectTypes = rle._game.sprite_groups.keys()
		self.objectTypes.sort()
		self.phiSize = sum([len(rle._game.sprite_groups[k]) for k in rle._game.sprite_groups.keys() if k not in ['wall', 'avatar']])
		self.objIDs = {}
		self.maxNumObjects = 6
		self.trackTokens = False
		self.vecSize = None
		self.padding = 5  ##5 is arbitrary; just to make sure we don't get overlap when we add positions
		print "If we track tokens we have an additional", 2**self.phiSize, "array elements."
		i=1
		for k in rle._game.all_objects.keys():
			self.objIDs[k] = i * (rle.outdim[0]*rle.outdim[1]+self.padding)
			i+=1

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
					# lst[o.ID] = vecValue
				# else:
				# 	lst[o.ID] = 'gone'
		present = []
		for k in [t for t in self.objectTypes if t not in ['wall', 'avatar']]:
			for o in rle._game.sprite_groups[k]:
				if o not in rle._game.kill_list:
					present.append(1)
				else:
					present.append(0)
		ind = sum([present[i]*2**i for i in range(len(present))])
		lst.append(ind)
		# lst['globalcount'] = ind
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
		print 'old true atoms', len(node.state-set(newAtoms))
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
def noveltyHeuristic(lst, WBP, k, surrogateCall=False):
	#returns the node in lst that has highest novelty measure
	## you need to not change the noveltyDict when you evaluate the novelty. Only change the dict if you select the state!
	maxNovelty = max([n.novelty for n in lst])
	# print 'in novelty'
	# embed()
	if maxNovelty==0 and not surrogateCall:
		return None
	else:
		bestNodes = [n for n in lst if n.novelty==maxNovelty]
		if len(bestNodes)==1:
			return bestNodes[0]
		elif len(bestNodes)>1:
		 	if not surrogateCall:
		 		return rewardHeuristic(bestNodes, WBP, k, surrogateCall=True)
		 	else:
		 		return random.choice(bestNodes)
		else:
			print "found 0 nodes in noveltyHeuristic"
			embed()

def rewardHeuristic(lst, WBP, k, surrogateCall=False):
	maxReward = max([n.reward for n in lst])
	bestNodes = [n for n in lst if n.reward==maxReward]
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


def BFS(rle, WBP):
	Q = Queue()
	visited, rejected= [], []
	start = Node(rle, WBP, [], None)
	start.lastState = rle
	visited.append(rle)
	Q.put(start)
	while not Q.empty():
		current = Q.get()
		
		win = current.eval(updateNoveltyDict=True)
		if win:
			return current, visited, rejected
		if current.novelty > 0:			
			visited.append(current)
			for a in ACTIONS:
				child = Node(rle, WBP, current.actionSeq+[a], current)
				Q.put(child)
		else:
			rejected.append(current)
	print "no more states in queue"
	# embed()
	return Q, visited, rejected

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
			current = noveltyHeuristic(Q, WBP, WBP.k, surrogateCall=False)
		else:
			current = rewardHeuristic(Q, WBP, WBP.k, surrogateCall=False)
		## This is not nec. right.
		if current is None:
			print "got no node"
			embed()
			return Q, visited, rejected
		else:
			Q.remove(current)
			current.eval(updateNoveltyDict=True)
			visited.append(current)
			if current.win:
				return current, visited, rejected
			else:
				for a in ACTIONS:
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
		print '-------------------------------------'
		if len(self.actionSeq)>0:
			print [actionDict[a] for a in self.actionSeq]
		self.updateObjIDs(vrle)
		print vrle.show()
		self.state = self.WBP.calculateAtoms(vrle)
		self.lastState = vrle
		self.win = win
		self.novelty = self.WBP.novelty(self, self.WBP.k, update=updateNoveltyDict)
		self.reward = vrle._game.score
		print 'novelty', self.novelty
		# raw_input("Press Enter to continue...")
		return win

	def updateObjIDs(self, vrle):
		i = 0
		for objType in vrle._game.sprite_groups:
			for s in vrle._game.sprite_groups[objType]:
				if s.ID not in self.WBP.objIDs.keys():
					self.WBP.objIDs[s.ID] = (len(self.WBP.objIDs.keys())+1) * (self.rle.outdim[0]*self.rle.outdim[1]+self.WBP.padding)
					i+=1
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
	def __init__(self, rle, gameString, levelString, gameFilename, k, display):
		WBP.__init__(self, rle, gameString, levelString, gameFilename, display)
		self.k = k

if __name__ == "__main__":
	
	# gameFilename = "examples.gridphysics.simpleGame_teleport" ##solved!
	# gameFilename = "examples.gridphysics.demo_teleport" ##solved!!
	# gameFilename = "examples.gridphysics.movers3c" ##solved!!
	# gameFilename = "examples.gridphysics.rivercross" ## solved!!
	# gameFilename = "examples.gridphysics.demo_dodge"  ##solved!!
	# gameFilename = "examples.gridphysics.simpleGame4_small"
	# gameFilename = "examples.gridphysics.movers5" ##solved!!

	gameFilename = "examples.gridphysics.simpleGame_push_boulders_multigoal" ## k=2 works!
	# gameFilename = "examples.gridphysics.demo_preconditions" ## k=2 works!
	# gameFilename = "examples.gridphysics.waterfall" ##solved!! 
	# gameFilename = "examples.gridphysics.frogs" ## worked with k=2.
	# gameFilename = "examples.gridphysics.pick_apples" ## worked with expanded phi!
	# gameFilename = "examples.gridphysics.scoretest" ##2BFS solves it!

	# gameFilename = "examples.gridphysics.simpleGame_push_boulders" ## k=2 works!


	# gameFilename = "examples.gridphysics.demo_chaser"  ##easy version solved!
	# gameFilename = "examples.gridphysics.portals" ## stochasticity breaks it
	# gameFilename = "examples.gridphysics.demo_helper"  ##easy version solved!

	# gameFilename = "examples.gridphysics.demo_multigoal_and"  ##takes forever.

	# gameFilename = "examples.gridphysics.demo_multigoal_and_score"  ##easy version solved!
	gameFilename = "examples.gridphysics.demo_sokoban"
	# gameFilename = "examples.gridphysics.demo_sokoban_score"
	# gameFilename = "examples.gridphysics.simpleGame_missile"

	## boulderdash: game freezes.
	# gameFilename = "examples.gridphysics.butterflies" #no
	# gameFilename = "examples.gridphysics.chase" no
	# gameFilename = "examples.gridphysics.survivezombies" # no

	# gameFilename = "examples.gridphysics.demo_transform_small" ## won't work until RLE can handle transformations.

	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()
	# embed()


	p = IW(rle, gameString, levelString, gameFilename, k=2, display=1)

	# p.trackTokens = True
	t1 = time.time()
	last, visited, rejected = BFS(rle, p)
	# last, visited, rejected = BFS2(rle, p)
	print
	print 'time', time.time()-t1
	print 'visited', len(visited)
	print 'rejected', len(rejected)
	# if not hasattr(last, 'actionSeq'):
	# 	print "Failed without tracking tokens. re-trying"
	# 	p.trackTokens = True
	# 	t1 = time.time()
	# 	last, visited, rejected, visitedStates = BFS(rle, p)
	# 	print time.time()-t1
	# 	print len(visited), len(rejected)
	# embed()

