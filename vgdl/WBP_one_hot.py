from IPython import embed
from planner import *
import itertools

ACTIONS = [(1,0), (-1,0), (0,1), (0,-1)]

## Base class for width-based planners (IW(k) and 2BFS)
class WBP(Planner):
	def __init__(self, rle, gameString, levelString, gameFilename, display):
		Planner.__init__(self, rle, gameString, levelString, gameFilename, display)
		self.T = len(rle._obstypes.keys())+1 #number of object types. Adding avatar, which is not in obstypes.
		self.vecDim = [rle.outdim[0]*rle.outdim[1], 2, self.T]
		self.trueAtoms = set() ## set of atoms that have been true at some point thus far in the planner.
		self.objectTypes = rle._game.sprite_groups.keys()
		self.objectTypes.sort()
		self.phiSize = sum([len(rle._game.sprite_groups[k]) for k in rle._game.sprite_groups.keys() if k not in ['wall', 'avatar']])
		self.maxNumObjects = 6
		self.trackTokens = False
		self.vecSize = None
		print "If we track tokens we have an additional", 2**self.phiSize, "array elements."

	def calculateAtoms(self, rle):
		## Converts rle state into a long list of atoms of length Nx2xT
		## (N: num of grid cells. 2: there / not there. T: Number of object types in the game).
		## For a 3x5 grid, we first flatten into a single column of len=15. (rle._getSensors() already has this representation)
		## atomList(m,n): whether object n is at location m.

		# TODO: Make implementation that considers avatar orientations. See p.3 of Geffner|Geffner paper.

		# vec = np.empty(self.vecDim)
		vec = []
		state = rle._getSensors()
		for i in range(len(state)):
			vec.extend(self.factorizeBoolean(rle, state[i]))
		
		if self.trackTokens:
			present = []
			for k in [t for t in self.objectTypes if t not in ['wall', 'avatar']]:
				for o in rle._game.sprite_groups[k]:
					if o not in rle._game.kill_list:
						present.append(1)
					else:
						present.append(0)
			ind = sum([present[i]*2**i for i in range(len(present))])
			nums = list(np.zeros(2**self.phiSize))
			nums[ind]=1
			vec = vec+nums

		if self.vecSize==None:
			print len(vec), "atoms"
			self.vecSize = len(vec)
		return np.array(vec)

	def factorize(self, rle, n):
		## Decomposes into a list of numbers that are incides of [avatar, rle._obstypes.keys()]
		## that correspond to which indices are present in n
		## this follows the convention of rle._getSensors(), which won't report two of the same number as being in a location, so
		## the decomposition is unique (e.g., if n=4, this is because the decomposition is [4], rather than having to worry that it
		## would be [2,2]).
		orig_n = n
		decomposition = []
		if n%2==1:
			decomposition.append(0)
			n = n-1
		i = len(rle._obstypes.keys())
		while i>0:
			if n>=2**i:
				decomposition.append(i)
				n = n-2**i
			i = i-1

		return decomposition

	def indicesToBooleans(self, listLen, indices):
		##turns a list of indices into two expanded lists.
		## phi_present: indicator function for presence of objects
		## phi_notPresent: indicator for absence of objects.
		## e.g., indicesToBooleans(8, [1,3,5]) = [0,1,0,1,0,1,0,0], [1,0,1,0,1,0,1,1]
		phi_present = []
		for i in range(listLen):
			if i in indices:
				phi_present.append(1)
			else:
				phi_present.append(0)
		phi_notPresent = [0 if p==1 else 1 for p in phi_present]
		mergedList = phi_present+phi_notPresent
		# return phi_present, phi_notPresent
		return mergedList

	def factorizeBoolean(self, rle, n):
		listLen = len(rle._obstypes.keys())+1
		return self.indicesToBooleans(self.T, self.factorize(rle, n))

	def findTrueTuples(self, node, k):
		## returns all combinations of k-tuples that are true in node.
		locTuples = np.where(node.state==1)[0]
		if k==1:
			return set(locTuples)

		kTuples = [c for c in list(itertools.combinations(locTuples, k))]

		return set(kTuples)

	def delta(self, node1, node2):
		if node1 is None:
			diff = np.where(node2.state==1)
		else:
			diff = np.where(node1.state!=node2.state)
		diffTuples = diff[0]
		diffTuples = [d for d in diffTuples if node2.state[d]==1]
		return set(diffTuples)


	def novelty(self, node, k, update=False):
		# Returns number of k-tuples of atoms that are newly true in node.
		newAtoms = self.delta(node.parent, node)
		# print "in novelty"
		# embed()
		if len(self.trueAtoms) > 0:
			trueAtoms = self.findTrueTuples(node,1)
			oldTrueAtoms = trueAtoms-newAtoms
			candidates = []
			for i in range(1,k+1):
				newPart = list(itertools.combinations(newAtoms, i))
				oldPart = list(itertools.combinations(oldTrueAtoms, k-i))
				unflattened = list(itertools.product(newPart, oldPart))
				flattened = [frozenset(u[0]+u[1]) for u in unflattened]	
				candidates.extend(flattened)
		else:
			candidates = [frozenset(p) for p in list(itertools.combinations(newAtoms, k))]

		node.candidates = candidates
		newTuples = set()
		for aT in candidates:
			if aT not in self.trueAtoms:
				if update:
					self.trueAtoms.add(aT)
				newTuples.add(aT)
		return len(newTuples)


def noveltyHeuristic(lst, rle, WBP, k, surrogateCall=False):
	#returns the node in lst that has highest novelty measure
	## you need to not change the noveltyDict when you evaluate the novelty. Only change the dict if you select the state!
	maxNovelty = max([n.novelty for n in lst])
	# print 'in novelty'
	# embed()
	if maxNovelty==0:
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
	 		return noveltyHeuristic(bestNodes, None, WBP, k, surrogateCall=True)
	 	else:
	 		return random.choice(bestNodes)
	else:
		print "found 0 nodes in rewardHeuristic"
		embed()

def BFS(rle, WBP, k):
	Q = Queue()
	visited, rejected, visitedStates= [], [], []
	start = Node(rle, WBP, [], None)
	start.lastState = rle
	visited.append(start)
	visitedStates.append(rle)
	Q.put(start)
	minDist = 100
	while not Q.empty():
		current = Q.get()
		win = current.eval(updateNoveltyDict=True)

		embed()
		if current.novelty > 0:
		# if current.state.tostring() not in visited:
			visited.append(current.state.tostring())
			visitedStates.append(current)
			if win:
				return current, visited, rejected, visitedStates
			for a in ACTIONS:
				child = Node(rle, WBP, current.actionSeq+[a], current)
				Q.put(child)
		else:
			rejected.append(current)

	return Q, visited, rejected, visitedStates

#when you expand a node, evaluate its children on both heuristics, then place in the queue.
## when you select a node from the queue, update the noveltyDict
def BFS2(rle, WBP, k):
	Q = []
	visited, rejected, visitedStates = [], [], []
	start = Node(rle, WBP, [], None)
	start.lastState = rle
	visited.append(start)
	visitedStates.append(rle)
	start.eval()
	Q.append(start)
	i=0
	while len(Q)>0:
		if i%2==0:
			#What do you do if you only have states with novelty 0?
			## you would pick from the rewardHeuristic, and would pick randomly if they all had value 0.
			current = noveltyHeuristic(Q, rle, WBP, k)
		else:
			## Similarly, if thre are tied rewards and you get states with novelty 0, you will still return one of these.
			## in the case where rewards were 0 and novelties were 0 we should stop.
			current = rewardHeuristic(Q, rle, WBP, k)

		## This is not nec. right.
		## TODO: make heuristics return None if they're forced to tie-break but rewards and novelties are 0.
		if current is None:
			print "current was None"
			embed()
			return Q, visited, rejected, visitedStates
		else:
			# embed()
			Q.remove(current)
			current.eval(updateNoveltyDict=True)
			visited.append(current.state.tostring())
			visitedStates.append(current)
			if current.win:
				return current, visited, rejected, visitedStates
			else:
				for a in ACTIONS:
					child = Node(rle, WBP, current.actionSeq+[a], current)
					child.eval()
					# embed()
					Q.append(child)		
			i+=1
	return Q, visited, rejected, visitedStates



class Node():
	def __init__(self, rle, WBP, actionSeq, parent):
		self.rle = rle
		self.WBP = WBP
		self.actionSeq = actionSeq
		self.parent = parent
		self.novelty = None
		self.reward = None
		self.children = None
		self.lastState = None
		self.reconstructed=False

	# try to copy parent lastState. Then take action and store as current lastState.
	## if that fails, replay from beginning and store as current lastState
	def eval(self, updateNoveltyDict=False):
		# try:
		if self.parent and self.parent.lastState is not None:
			try:
				vrle = copy.deepcopy(self.parent.lastState)
				if len(self.actionSeq)>0:
					vrle.step(self.actionSeq[-1])
					terminal, win = vrle._isDone()
					# terminal = vrle._isDone()[0]
			except:
				print "conditions met but copy failed"
				embed()
		else:
		# except:
			self.reconstructed=True
			print "copy failed; replaying from top"
			# embed()
			vrle = copy.deepcopy(rle)
			terminal, win = vrle._isDone()
			# terminal = vrle._isDone()[0]
			i=0
			while not terminal and len(self.actionSeq)>i:
				vrle.step(self.actionSeq[i])
				terminal, win = vrle._isDone()
				# terminal = vrle._isDone()[0]
				i += 1
		if len(self.actionSeq)>0:
			print self.actionSeq[-1]
		print vrle.show()
		# if len(vrle._game.sprite_groups['probe'])==0:
			# self.WBP.findAvatarInRLE(vrle) == (2,4):
			# embed()
		# print "depth", len(self.actionSeq)
		self.state = self.WBP.calculateAtoms(vrle)
		self.lastState = vrle
		self.win = win
		self.novelty = self.WBP.novelty(self, 2, update=updateNoveltyDict)
		self.reward = vrle._game.score
		return win

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
			vrle.step(a)
			print vrle.show()
			terminal = vrle._isDone()[0]
			i+=1



class IW(WBP):
	def __init__(self, rle, gameString, levelString, gameFilename, display, k):
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

	# gameFilename = "examples.gridphysics.simpleGame_push_boulders_multigoal" ## k=2 works!
	# gameFilename = "examples.gridphysics.demo_preconditions" ## k=2 works!
	# gameFilename = "examples.gridphysics.waterfall" ##solved!! 
	# gameFilename = "examples.gridphysics.frogs" ## worked with k=2.
	# gameFilename = "examples.gridphysics.pick_apples" ## worked with expanded phi!
	# gameFilename = "examples.gridphysics.scoretest" ##2BFS solves it!



	# gameFilename = "examples.gridphysics.demo_chaser"  ##easy version solved!
	# gameFilename = "examples.gridphysics.portals" ## stochasticity breaks it
	# gameFilename = "examples.gridphysics.demo_helper"  ##easy version solved!
	# gameFilename = "examples.gridphysics.demo_multigoal_and"  ##easy version solved!

	# gameFilename = "examples.gridphysics.demo_multigoal_and_score"  ##easy version solved!
	# gameFilename = "examples.gridphysics.demo_sokoban"
	gameFilename = "examples.gridphysics.demo_sokoban_score"

	## boulderdash: game freezes.
	# gameFilename = "examples.gridphysics.butterflies" no
	# gameFilename = "examples.gridphysics.chase" no
	# gameFilename = "examples.gridphysics.survivezombies" # no

	# gameFilename = "examples.gridphysics.demo_transform" ## won't work until RLE can handle transformations.
	# gameFilename = "examples.gridphysics.demo_multigoal_and"

	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()

	p = IW(rle, gameString, levelString, gameFilename, True, 1)
	p.trackTokens = True
	t1 = time.time()
	# last, visited, rejected, visitedStates = BFS(rle, p, 2)
	last, visited, rejected, visitedStates = BFS2(rle, p, 2)
	print time.time()-t1
	print len(visited), len(rejected)
	embed()
	if not hasattr(last, 'actionSeq'):
		print "Failed without tracking tokens. re-trying"
		p.trackTokens = True
		t1 = time.time()
		last, visited, rejected, visitedStates = BFS(rle, p, 2)
		print time.time()-t1
		print len(visited), len(rejected)
	# embed()

