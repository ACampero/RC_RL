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

		# nums = present+absent
		# print len(nums)
		## Now add new information: # For each type of item, are there 0, 1, 2, ... maxNum on the board?
		# for k in self.objectTypes:
		# 	numOnBoard = len([o for o in rle._game.sprite_groups[k] if o not in rle._game.kill_list])
		# 	lst = [1 if i==numOnBoard else 0 for i in range(self.maxNumObjects)]
		# 	if numOnBoard>=self.maxNumObjects:
		# 		lst.append(1)
		# 	else:
		# 		lst.append(0)
		# 	invlst = [1 if l==0 else 0 for l in lst]
		# 	both = lst+invlst
		# 	nums.extend(both)
		
		# embed()
		return np.array(vec)

	# def getNumInFactorizedState(self, factorizedState, objType):
	# 	phiSize = (self.maxNumObjects+1)*len(self.objectTypes)*2
	# 	len(factorizedState)
	# 	relevantPartOfState = factorizedState[-phiSize:]
	# 	ind = self.objectTypes.index(objType)
	# 	cut = relevantPartOfState[ind*(self.maxNumObjects+1)*2:ind*(self.maxNumObjects+1)*2+(self.maxNumObjects+1)]
	# 	return len(factorizedState)-phiSize+ind*(self.maxNumObjects+1)*2+np.where(cut==1)[0], cut

	# def getNumInFactorizedState(self, factorizedState, objType):

	# 	relevantPartOfState = factorizedState[-phiSize:]
	# 	ind = self.objectTypes.index(objType)
	# 	cut = relevantPartOfState[ind*(self.maxNumObjects+1)*2:ind*(self.maxNumObjects+1)*2+(self.maxNumObjects+1)]
	# 	return len(factorizedState)-phiSize+ind*(self.maxNumObjects+1)*2+np.where(cut==1)[0], cut

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
		# locs = np.where(node.state==1)
		# locTuples = [tuple([d[i] for d in locs]) for i in range(len(locs[0]))]
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
		# print "in delta"
		# embed()
		# diffTuples = [[d[i] for d in diff] for i in range(len(diff[0]))]
		# diffTuples = [tuple(d) for d in diffTuples if node2.state[d[0],d[1],d[2]]==1]
		return set(diffTuples)


	def novelty(self, node, k):
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

		# if 879 in newAtoms:
		# 	print "added 865 for the first time"
		# 	embed()
		# if frozenset((456,879)) in node.candidates:
		# 	print "adding 456,865"
		# 	embed()

		newTuples = set()
		for aT in candidates:
			if aT not in self.trueAtoms:
				self.trueAtoms.add(aT)
				newTuples.add(aT)
		return len(newTuples)


	# def calculateNewAtoms(self, factoredState):
	# 	newStates = set()
	# 	for loc in range(self.vecDim[0]):
	# 		for phi in range(self.vecDim[1]):
	# 			if np.any(factoredState[loc][phi]==1):
	# 				objs = np.where(factoredState[loc][phi]==1)[0]
	# 				for o in objs:
	# 					if (loc, phi, o) not in self.trueAtoms:
	# 						self.trueAtoms.add((loc, phi, o))
	# 						print (loc, phi, o)
	# 						newStates.add((loc, phi, o))
	# 			else:
	# 				objs = np.where(factoredState[loc][phi]==0)[0]
	# 	## number of states that are made true for the first time.
	# 	return len(newStates)


def BFS(rle, WBP, k):
	Q = Queue()
	visited = []#set()
	rejected = []#set()
	visitedStates = []
	start = Node(rle, WBP, [], None)
	start.lastState = rle
	visited.append(start)
	visitedStates.append(rle)
	Q.put(start)
	winNodes = []
	while not Q.empty():
		current = Q.get()
		win = current.eval()
		# embed()
		novelty = WBP.novelty(current, k)
		if novelty > 0:
		# if current.state.tostring() not in visited:
			visited.append(current.state.tostring())
			visitedStates.append(current)
			if win:
				# winNodes.append(current)
				return current, visited, rejected, visitedStates
			for a in ACTIONS:
				child = Node(rle, WBP, current.actionSeq+[a], current)
				Q.put(child)
		else:

			rejected.append(current)

	return Q, visited, rejected, visitedStates

# relevantVisited = [v for v in visitedStates if hasattr(v, 'lastState') and len(v.lastState._game.kill_list)==2]
# close = [v for v in relevantVisited if manhattanDist(p.findAvatarInRLE(v.lastState), p.findObjectInRLE(v.lastState, 'probe')) < 4]
# 
class Node():
	def __init__(self, rle, WBP, actionSeq, parent):
		self.rle = rle
		self.WBP = WBP
		self.actionSeq = actionSeq
		self.parent = parent
		self.children = None
		self.lastState = None
		self.reconstructed=False

	# try to copy parent lastState. Then take action and store as current lastState.
	## if that fails, replay from beginning and store as current lastState
	def eval(self):
		# try:
		if self.parent and self.parent.lastState is not None:
			try:
				vrle = copy.deepcopy(self.parent.lastState)
				if len(self.actionSeq)>0:
					vrle.step(self.actionSeq[-1])
					terminal = vrle._isDone()[0]
			except:
				print "conditions met but copy failed"
				embed()
		else:
		# except:
			self.reconstructed=True
			print "copy failed; replaying from top"
			# embed()
			vrle = copy.deepcopy(rle)
			terminal = vrle._isDone()[0]
			i=0
			while not terminal and len(self.actionSeq)>i:
				vrle.step(self.actionSeq[i])
				terminal = vrle._isDone()[0]
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
		return vrle._isDone()[1]

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
	# gameFilename = "examples.gridphysics.demo_chaser"  ##solved!
	gameFilename = "examples.gridphysics.simpleGame4_small"
	# gameFilename = "examples.gridphysics.movers5" ##solved!!

	# gameFilename = "examples.gridphysics.simpleGame_push_boulders_multigoal" ## k=2 works!
	# gameFilename = "examples.gridphysics.demo_preconditions" ## k=2 works!
	# gameFilename = "examples.gridphysics.waterfall" ##solved!! 
	# gameFilename = "examples.gridphysics.frogs" ## worked with k=2.
	# gameFilename = "examples.gridphysics.pick_apples" ## worked with expanded phi!

	# gameFilename = "examples.gridphysics.scoretest" 
	# gameFilename = "examples.gridphysics.portals" ## stochasticity breaks it

	# gameFilename = "examples.gridphysics.demo_transform" ## won't work until RLE can handle transformations.


	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()


	p = IW(rle, gameString, levelString, gameFilename, True, 1)
	# p.trackTokens = True
	t1 = time.time()
	last, visited, rejected, visitedStates = BFS(rle, p, 2)
	# winNodes = BFS(rle, p, 2)
	print time.time()-t1
	embed()
	# print len(visited), len(rejected)
	if not hasattr(last, 'actionSeq'):
		print "Failed without tracking tokens. re-trying"
		p.trackTokens = True
		t1 = time.time()
		# winNodes = BFS(rle, p, 2)
		last, visited, rejected, visitedStates = BFS(rle, p, 2)
		print time.time()-t1
		# print len(visited), len(rejected)
	embed()

