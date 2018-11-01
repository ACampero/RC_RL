from IPython import embed
from planner import *
from core import VGDLParser
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
		self.trueAtoms = defaultdict(lambda:0) #set() ## set of atoms that have been true at some point thus far in the planner.
		self.objectTypes = rle._game.sprite_groups.keys()
		self.objectTypes.sort()
		self.phiSize = sum([len(rle._game.sprite_groups[k]) for k in rle._game.sprite_groups.keys() if k not in ['wall', 'avatar']])
		self.objIDs = {}
		self.maxNumObjects = 6
		self.trackTokens = False
		self.vecSize = None
		self.addWaitAction = False
		self.padding = 5  ##5 is arbitrary; just to make sure we don't get overlap when we add positions
		self.theory = generateTheoryFromGame(rle, alterGoal=False)
		i=1
		for k in rle._game.all_objects.keys():
			self.objIDs[k] = i * (rle.outdim[0]*rle.outdim[1]+self.padding)
			i+=1
		self.addSpaceBarToActions()
		self.statesEncountered = []

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

def noveltySelection(QNovelty, QReward):
	bestNodes = sorted(QNovelty, key=lambda n: (n.novelty, -n.intrinsic_reward))
	current = bestNodes.pop(0)
	QNovelty.remove(current)
	try:
		QReward.remove(current)
	except:
		pass
	return current

def rewardSelection(QReward, QNovelty):
	# acceptableNodes = QReward
	acceptableNodes = filter(lambda n:n.novelty<3, QReward)
	# if len(acceptableNodes)==0:
		# acceptableNodes = QReward
		# print "Removed filter"
		# embed()
	bestNodes = sorted(acceptableNodes, key=lambda n: (-n.intrinsic_reward, n.novelty))
	current = bestNodes.pop(0)
	QReward.remove(current)

	try:
		QNovelty.remove(current)
	except:
		pass
	return current


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

	while len(QNovelty)>0 or len(QReward)>0:
		"""
		if i%2==0:
			current = noveltySelection(QNovelty, QReward)
		else:
		"""
		current = rewardSelection(QReward, QNovelty)
		# Add node state to states encountered
		WBP.statesEncountered.append(current.lastState._game.getFullState())

		print current.novelty, current.intrinsic_reward, current.heuristicVal
		# print len(QNovelty), len(QReward)
		# if current==None:
		# 	pass
		# else:
		# print current.novelty
		print current.lastState.show()

		current.updateNoveltyDict(QNovelty, QReward)
		# embed()
		visited.append(current)

		for a in WBP.actions:
			child = Node(rle, WBP, current.actionSeq+[a], current)
			child.eval()
			if child.win:
				return child, visited, rejected ##revisit
			else:
				QNovelty.append(child)
				QReward.append(child)
		i+=1
	return None, visited, rejected

class Node():
	def __init__(self, rle, WBP, actionSeq, parent):
		self.rle = rle
		self.WBP = WBP
		self.actionSeq = actionSeq
		self.parent = parent
		self.state = {}
		self.candidates = []
		self.novelty = None
		self.reward = None
		self.intrinsic_reward = 0
		self.metabolic_cost = 0
		self.children = None
		self.lastState = None
		self.reconstructed=False
		self.expanded = False
		self.rolloutDepth = max(rle.outdim)
		if self.parent is not None:
			self.rolloutArray = parent.rolloutArray[1:]
		else:
			self.rolloutArray = []


## when to trigger rollouts, if any
## rollout length
## repeating rollouts if death? e.g., are they optimistic?
## multiple samples??
	def metabolics(self, rle, events, action, n=10, mult=.25):

		metabolic_cost = 1./n
		if action==32:
			metabolic_cost += (1-1./n)*mult
		if len(events)>0:
			# if any([rle._game.sprite_groups['avatar'][0].ID in e and e[0]=='bounceForward' for e in events]):
				# metabolic_cost += (1-1./n)*mult
			if any([rle._game.sprite_groups['avatar'][0].ID in e and e[0]=='killSprite' for e in events]):
				metabolic_cost += 0.3
		return metabolic_cost

	def rollout(self, vrle):
		vrle = copy.deepcopy(vrle)
		prevHeuristicVal = self.heuristics(vrle)
		rolloutArray = []
		i=0
		terminal, win = vrle._isDone()
		while i<self.rolloutDepth and not terminal:
			a = random.choice([K_UP, K_DOWN, K_LEFT, K_RIGHT])
			vrle.step(a)
			currHeuristicVal = self.heuristics(vrle)
			heuristicVal = currHeuristicVal-prevHeuristicVal
			rolloutArray.append(heuristicVal)
			prevHeuristicVal = currHeuristicVal
			# print "in rollout"
			# print vrle.show()
			terminal, win = vrle._isDone()
			i+=1
		print "______"
		if terminal and not win:
			print "recursive call to rollout."
			return []
			# return self.rollout(vrle2)
		# else:
			# rolloutVal = self.heuristics(vrle)
			# if rolloutVal==100:
			# 	import ipdb; ipdb.set_trace()
			# print "rollout result", self.heuristics(vrle)
		return rolloutArray

	##TODO: have an initHeuristics() function do most of this work and return a simple function
	## that evaluates the heuristic value of a particular state.

	def spritecounter_val(self, theory, term, stype, rle, first_alpha=100,
						  second_alpha=1):
		val = 0
		compute_second_order = True

		# Check if condition is win or loss and multiply accordingly
		if term.termination.win:
			mult = -1
		else:
			compute_second_order = False
			mult = 1

		# Get all types that kill or transform stype
		killer_types = [
			inter.slot2 for inter in theory.interactionSet
			if ((inter.interaction == 'killSprite' or
				 inter.interaction == 'transformTo')
				and inter.slot1 == stype)]

		# Get attributes from terminationSet
		limit = term.termination.limit
		# embed()

		if 'SpawnPoint' in str(theory.classes[stype][0].vgdlType) and not killer_types:
			distance_to_goal = 0
			## Special case, where you want to track whether that spawnPoint has a limit, etc.
			## Distance to goal here is how many sprites the spawnPoint still has to shoot before it expires.
			for o in rle._game.sprite_groups[stype]:
				distance_to_goal += abs(o.total-o.counter)
			val += mult * first_alpha * distance_to_goal
			return val
		else:
			## Normal case
			n_sprites = len([0 for sprite in self.WBP.findObjectsInRLE(rle, stype)])
			distance_to_goal = abs(n_sprites - limit)

		val += mult * first_alpha * distance_to_goal

		if compute_second_order:
			## Get all positions of objects whose type is in killer_types; compute minimum distance
			## of each to the stypes we have to destroy. Return min over all mins.

			# import ipdb; ipdb.set_trace()
			kill_positions = np.concatenate([
				self.WBP.findObjectsInRLE(rle, ktype)
				for ktype in killer_types])
			stype_positions = self.WBP.findObjectsInRLE(rle, stype)
			try:
				distance = min([manhattanDist(obj, pos)
					 for pos in kill_positions
					 for obj in stype_positions])
			except ValueError:
				# embed()
				distance = 0

			val += mult * second_alpha * distance

		return val

	def multispritecounter_val(self, theory, term, rle, first_alpha=100,
							   second_alpha=1):
		val = 0
		for stype in term.termination.stypes:
			val += self.spritecounter_val(theory, term, stype, rle,
				first_alpha=first_alpha, second_alpha=second_alpha)
			# print stype, val
		return val

	def timeout_val(self, theory, term, rle):
		val = 0
		limit = term.termination.limit

		# Check if condition is win or loss and multiply accordingly
		if term.termination.win:
			mult = -1
		else:
			mult = 1

		time_elapsed = rle._game.time
		distance_to_goal = abs(time_elapsed - limit)

		val += mult * distance_to_goal

		return val

	def heuristics(self, rle=None, first_alpha=100, second_alpha=1, time_alpha=10):
		if rle==None:
			rle = self.lastState

		theory = self.WBP.theory
		heuristicVal = 0

		for term in theory.terminationSet:
			if isinstance(term, SpriteCounterRule):
				heuristicVal += self.spritecounter_val(theory, term, term.termination.stype, rle,
					first_alpha=first_alpha, second_alpha=second_alpha)

			elif isinstance(term, MultiSpriteCounterRule):
				heuristicVal += self.multispritecounter_val(theory, term, rle,
						first_alpha=first_alpha, second_alpha=second_alpha)

			elif isinstance(term, TimeoutRule):
				heuristicVal += time_alpha * \
					self.timeout_val(theory, term, rle)

		return heuristicVal

	def getToCurrentState(self):
		if self.parent and self.parent.lastState is not None:
			## try to copy parent lastState. Then take action and store as current lastState.
			## if that fails, replay from beginning and store as current lastState
			try:
				vrle = copy.deepcopy(self.parent.lastState)
				if len(self.actionSeq)>0:
					a = self.actionSeq[-1]
					res = vrle.step(a)
					self.metabolic_cost = self.parent.metabolic_cost + self.metabolics(vrle, res['effectList'], a)
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
				a = self.actionSeq[i]
				res = vrle.step(a)
				self.metabolic_cost += self.metabolics(vrle, res['effectList'], a)
				terminal, win = vrle._isDone()
				i += 1
		return vrle, win

	def eval(self):
		# ## Evaluate current node, including calculating intrinsic reward: f(rewards, heuristics, etc.)

		self.lastState, self.win = self.getToCurrentState()
		self.updateObjIDs(self.lastState)
		self.state = self.WBP.calculateAtoms(self.lastState)

		for i in range(1,3):
			for c in itertools.combinations(self.state, i):
				if self.WBP.trueAtoms[c] == 0:
					self.candidates.append(c)
		self.updateNovelty()

		# if self.win:
			# embed()

		## Try rollouts for aliens?
		if len(self.actionSeq)>0 and self.actionSeq[-1]==32:
			self.rolloutArray = self.rollout(self.lastState)

		self.heuristicVal = self.heuristics()

		self.intrinsic_reward = self.lastState._game.score + self.heuristicVal + \
		sum(self.rolloutArray) - self.metabolic_cost

		return self.win

	def updateNovelty(self):
		if len(self.candidates)==0:
			self.novelty = 3
		else:
			self.novelty = min([len(c) for c in self.candidates])
		return self.novelty

	def updateNoveltyDict(self, QNovelty, QReward):
		jointSet = list(set(QNovelty+QReward))
		for c in self.candidates:
			if self.WBP.trueAtoms[c] == 0:
				self.WBP.trueAtoms[c] = 1
				for n in jointSet:
					if c in n.candidates:
						n.candidates.remove(c)
		for n in jointSet:
			n.novelty = n.updateNovelty()
		return

	def updateObjIDs(self, vrle):
		i = 0
		for objType in vrle._game.sprite_groups:
			for s in vrle._game.sprite_groups[objType]:
				if s.ID not in self.WBP.objIDs.keys():
					if s.name=='bullet':
						s.ID = len([o for o in vrle._game.sprite_groups[objType] if o not in vrle._game.kill_list])
					else:
						s.ID = len(vrle._game.sprite_groups[objType])
					self.WBP.objIDs[s.ID] = (len(self.WBP.objIDs.keys())+1) * (self.rle.outdim[0]*self.rle.outdim[1]+self.WBP.padding)
					i+=1
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
			# vrle.step(0)
			print vrle.show()
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
	# gameFilename = "examples.gridphysics.demo_dodge"  ##solved!!
	# gameFilename = "examples.gridphysics.movers5" ##solved!!
	# gameFilename = "examples.gridphysics.demo_preconditions" ## k=2 works!
	# gameFilename = "examples.gridphysics.waterfall" ##solved!!
	# gameFilename = "examples.gridphysics.frogs" ## worked with k=2.
	gameFilename = "examples.gridphysics.pick_apples" ## worked with expanded phi!
	# gameFilename = "examples.gridphysics.scoretest" ##2BFS solves it!
	# gameFilename = "examples.gridphysics.demo_chaser"  ##easy version solved!
	# gameFilename = "examples.gridphysics.demo_helper"  ##easy version solved!
	# gameFilename = "examples.gridphysics.simpleGame_push_boulders"
	# gameFilename = "examples.gridphysics.chase" #yes!!!
	# gameFilename = "examples.gridphysics.survivezombies" # solvable, just not very fast if long timeout.
	# gameFilename = "examples.gridphysics.demo_transform_small" ## works

	# gameFilename = "examples.gridphysics.zelda_orig2" ## We can probably handle this, provided subgoal heuristics, once Chaser/A* are deterministic
	# gameFilename = "examples.gridphysics.missilecommand2" ## We can probably handle this, provided subgoal heuristics, once Chaser/A* are deterministic
	# gameFilename = "examples.gridphysics.chase2"
	# gameFilename = "examples.gridphysics.aliens2"  ##doesn't work. needs v. different heuristics


	# gameFilename = "examples.gridphysics.demo_helper"  ##
	# gameFilename = "examples.gridphysics.demo_transform" ##
	# gameFilename = "examples.gridphysics.simpleGame_missile" #later.
	# gameFilename = "examples.gridphysics.simpleGame_push_boulders2"

	# gameFilename = "examples.gridphysics.waypointtheory"  ##easy version solved!

	# gameFilename = "examples.gridphysics.simpleGame_push_boulders_multigoal" ## k=2 works!
	# gameFilename = "examples.gridphysics.simpleGame4"

	# gameFilename = "examples.gridphysics.simpleGame4_small"

	# gameFilename = "examples.gridphysics.demo_multigoal_and_score"  ##easy version solved!
	# gameFilename = "examples.gridphysics.demo_sokoban" #later
	# gameFilename = "examples.gridphysics.demo_sokoban_score" #later
	# gameFilename = "examples.gridphysics.butterflies" ## stochasticity breaks it


	# gameFilename = "examples.gridphysics.demo_multigoal_and"  ##takes forever if you have many boxes and don't use 2BFS (with metabolic penalty)


	## Continuous physics games can't work right now. RLE is discretized, getSensors() relies on this, and a lot of the induction/planning
	## architecture depends on that. Will take some work to do this well. Best plan is to shrink the grid squares and increase speeds/strengths of
	## objects.
	# gameFilename = "examples.continuousphysics.mario"
	# gameFilename = "examples.gridphysics.boulderdash" #Game is buggy.
	# gameFilename = "examples.gridphysics.butterflies"


	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()


	p = IW(rle, gameString, levelString, gameFilename, k=2)

	VGDLParser.playGame(gameString, levelString, p.statesEncountered, persist_movie=True, make_images=True, make_movie=False, movie_dir="videos/"+gameFilename, padding=10)
	# embed()
	t1 = time.time()
	last, visited, rejected = BFS3(rle, p)

	print time.time()-t1
	print len(visited), len(rejected)
	embed()


#
