from random import choice
import itertools, copy, scipy.misc
import numpy as np
from sampleVGDLString import *
from class_theory_template import *
from taxonomy import *
from IPython import embed
from ontology import *
from collections import defaultdict
# import ipdb
import operator
import time, math
from util import factorize, objectsToSymbol
from rlenvironmentnonstatic import createMindEnv

ALNUM = '0123456789bcdefhijklmnpqrstuvwxyzQWERTYUIOPSDFHJKLZXCVBNM,./;[]<>?:`-=~!@#$%^&*()_+'
AvatarTypes = [MovingAvatar, HorizontalAvatar, VerticalAvatar, FlakAvatar, AimedFlakAvatar, OrientedAvatar,
RotatingAvatar, RotatingFlippingAvatar, NoisyRotatingFlippingAvatar, ShootAvatar, AimedAvatar,
AimedFlakAvatar, InertialAvatar, MarioAvatar]

"""
Theory induction on VGDL Games
"""

'''
TODO 8/2:
- Implement tree viz of hypothesis generation --> save children of each node
- More informative printing for debugging/monitoring progress *
- When DFSinduction returns before the full tree is built, return in a way that let's us save the state of the function, and continue running to get more theories --> queue
- Run on more minimal example *
- Make DFSinduction more efficient *
	- fix the backward checks in likelihood (should you just check the most recently changed rules (anything in drying paint) against the past events?)
	- check that redundant events don't cost much extra
	- likelihood: check changed rules against all timesteps, check new timestep against all rules) --> calling checkRules / checkPredictions more targeted manner

NOTES:
Current assumptions:
	no grammar over preconditions
	preconditions limited to claims about a SINGLE object
	preconditions limited to simple comparison operators.
	Events that take place at same timestep can only be because of the same preconditions.

'''

class TimeStep:
	"""
	Everything that happened in a time step in the game.

	Ex.)
	TimeStep.agentAction = 'up'
	TimeStep.agentState = {'health':1, 'treasure':2}
	TimeStep.events = [(bounceForward, BLUE, ORANGE), (undoAll, ORANGE, BLACK)]
	TimeStep.t = 4  --> meaning all of this took place at t_4
	"""

	def __init__(self, agentAction, agentState, events, gameState, rle):
		self.agentAction = agentAction
		self.agentState = agentState # agent's backpack
		self.events = events
		self.t = False # Number timestep
		self.gameState = gameState
		self.rle = rle

	def display(self):
		print (self.agentAction, self.agentState, self.events, self.gameState)
		return (self.agentAction, self.agentState, self.events, self.gameState)


class Precondition(object):
	"""
	Appended to InteractionRules if conflicting effects occur from the same interaction, due to changed resources.
	"""
	def __init__(self, text, item, operator_name, num):
		self.text = text
		self.item = item
		self.operator_name = operator_name
		self.num = num
		self.negated = False

	def check(self, dictionary):
		if self.item not in dictionary.keys():
			dictionary[self.item] = 0

		if self.operator_name == '>':
			answer = dictionary[self.item] > self.num
		elif self.operator_name == '>=':
			answer = dictionary[self.item] >= self.num
		elif self.operator_name == '<':
			answer = dictionary[self.item] < self.num
		elif self.operator_name == '<=':
			answer = dictionary[self.item] <= self.num

		if self.negated:
			return not answer
		else:
			return answer

	def negate(self):
		self.negated = not self.negated
		self.text = 'not '+ self.text

	def display(self):
		print self.text

	def __eq__(self, other):
		try:
			return self.text == other.text
		except AttributeError:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)

class InteractionRule(object):
	"""
	Rule defining how 2 classes of objects interact with each other.
	# TODO: Should enforce proper syntax for interaction rules

	"""
	def __init__(self, interaction, c1, c2, args, preconditions=set(), generic=False):
		self.interaction = interaction
		self.slot1 = c1
		self.slot2 = c2
		self.args = args
		self.preconditions = preconditions
		self.generic = generic ## if generic, this interaction rule belongs to the generic prior that is meant to be overriden.


	def display(self):
		if not self.preconditions:
			print self.interaction, self.slot1, self.slot2, self.args, "generic: {}".format(self.generic)
		else:
			print self.interaction, self.slot1, self.slot2, self.args, [p.text for p in self.preconditions], "generic: {}".format(self.generic)
		return

	def asTuple(self):
		return (self.interaction, self.slot1, self.slot2, self.args) #TODO: Check that adding the value here doesn't mess up equality checks elsewhere

	def addPrecondition(self, precondition):
		"""
		TODO: Now that we've reimplemented preconditions as lambda functions,
		it can't properly check for equality of preconditions. You *may*
		be able to get around this by checking for the equality of precondition.text
		and making sure that precondition.text always reflects the functioning of the
		lambda function.
		"""
		curr_preconditions = [p.text for p in self.preconditions]
		if precondition.text not in curr_preconditions: #TODO: change equality for preconditions?
			self.preconditions = set([precondition]) #TODO: Need to change this, if we accept more than one precondition for an interaction rule

	def checkPreconditions(self, agentState):
		return all([p.check(agentState) for p in self.preconditions])

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return all([
				self.asTuple()==other.asTuple(),
				self.preconditions==other.preconditions
				])
		else:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)

class TerminationRule:
	"""
	TODO: eventually incorporate multiple sprite termination conditions and timeout termination conditions.
	At the moment, we assume single sprite conditions
	"""
	def isDone(self, game):
		return self.termination.isDone()

	def __eq__(self,other):
		return self.asTuple() == other.asTuple()

	def __hash__(self):
		return self._hash

class TimeoutRule(TerminationRule):
	def __init__(self, limit=0, win=False):
		self.termination = Timeout(limit=limit, win=win)
		self.ruleType = "TimeoutRule"
		self._hash = hash((self.ruleType, limit, win))

	def display(self):
		print (self.ruleType, self.termination.limit, self.termination.win)

	def asTuple(self):
		return (self.ruleType, self.termination.limit, self.termination.win)


class NoveltyRule(TerminationRule):
	""" Game ends when the number of sprites of type 'stype' hits 'limit' (or below). """
	def __init__(self,s1,s2,win,args=None):
		"""sclass = sprite class, snumber = sprite number, win = whether termination is a win"""
		self.termination = NoveltyTermination(s1=s1, s2=s2, win=win, args=args)
		if args is None:
			args = dict()
		self.ruleType = "NoveltyRule"
		self._hash = hash((self.ruleType, s1, s2, win, tuple(args)))

	def display(self):
		print self.ruleType, self.termination.s1, self.termination.s2, self.termination.win, self.termination.args
		return

	def asTuple(self):
		return (self.ruleType, self.termination.s1, self.termination.s2, self.termination.win, self.termination.args)

class SpriteCounterRule(TerminationRule):
	""" Game ends when the number of sprites of type 'stype' hits 'limit' (or below). """
	def __init__(self,stype,limit,win):
		"""sclass = sprite class, snumber = sprite number, win = whether termination is a win"""
		self.termination = SpriteCounter(limit=limit, stype=stype, win=win)
		self.ruleType = "SpriteCounterRule"
		self._hash = hash((self.ruleType, limit, stype, win))

	def display(self):
		print self.ruleType, self.termination.stype, self.termination.limit, self.termination.win
		return

	def asTuple(self):
		return (self.ruleType, self.termination.stype, self.termination.limit, self.termination.win)


class MultiSpriteCounterRule(TerminationRule):
	""" Game ends when the sum of all sprites of types 'stypes' hits 'limit'. """
	def __init__(self, limit=0, win=True, stypes = []):
		argList = dict((str(i), stype) for i, stype in enumerate(stypes))
		self.termination = MultiSpriteCounter(limit=limit,win=win, **argList)
		self.ruleType = "MultiSpriteCounterRule"
		self._hash = hash((self.ruleType, limit, win, tuple(sorted(argList.iteritems()))))

	def display(self):
		print self.ruleType, self.termination.stypes, self.termination.limit, self.termination.win
		return

	def asTuple(self):
		return (self.ruleType, set(self.termination.stypes), self.termination.limit, self.termination.win)


class ruleCluster(object):
	def __init__(self, interactionAndPreconditionList, pairList):
		self.clusteredRules = interactionAndPreconditionList
		self.pairs = pairList
		self.score = 0.

	def __eq__(self, other):
		if len(self.clusteredRules)!=len(other.clusteredRules):
			return False
		else:
			return all([r1 in [r2 for r2 in other.clusteredRules] for r1 in self.clusteredRules])

	def __ne__(self, other):
		return not self.__eq__(other)

## Helper print function
def printInteractionSet(interactionSet):
		print [i.display() for i in interactionSet]

class Theory(object):
	"""
	A VGDL description of a game
	"""
	def __init__(self, game):
		self.game = game
		self.parent = None
		self.children = []
		self.twins = [] #Not used now, but potentially use to keep better track of genealogy
		self.depth = 0

		# Following VGDL structure
		self.spriteSet = [] # Includes properties of sprites/objects
		self.levelMapping = [] # Map of the game
		self.interactionSet = [] # Interaction rules
		self.terminationSet = [] # Conditions that lead to game termination

		self.spriteObjects = {} # Maps sprite color -> Sprite object
		self.classes = {} # Maps classes -> objects
		self.predicates = set() # Types of possible interactions

		self.dryingPaint = set()
		self.inModification = {}

		self.falsified = []
		self.multi_falsified = []

		self.posterior = False

		self.goalColor = False ## TODO. Hack added 1/18/17 in lieu of termination set.

		self.resource_limits = defaultdict(lambda:1)

	def initializeSpriteSet(self, vgdlSpriteParse=False, spriteInductionResult=False):
		if not (vgdlSpriteParse or spriteInductionResult):
			print "You must provide either a vgdlSpriteParse or the result of having performed sprite induction."
			return
		if vgdlSpriteParse:
			self.spriteSet = vgdlSpriteParse
		if spriteInductionResult:
			self.spriteSet = spriteInductionResult

		# End of screen is a special object. Initialize it here.
		# eos = Sprite(core.VGDLSprite, 'ENDOFSCREEN', None, None)
		eos = Sprite(core.EOS, 'ENDOFSCREEN', None, None)
		self.spriteSet.append(eos)

		# Get mapping from sprite color to Sprite object
		for s in self.spriteSet:
			self.spriteObjects[s.color] = s
		# embed()
	"""Main functions"""

	def prior(self):

		def phi(numClasses, numRules, lamda):
			#TODO: Refine this to take into account the minimum necessary size of the ruleset.
			return lamda*numClasses + (1-lamda)*numRules

		#Mode is p(r-1) / (1-p). For now we pick p=.5, r=5 to reflect that phi=4 is modal.
		def negBin(k, r, p):
			return scipy.misc.comb(k+r-1, k) * p**k * (1-p)**r

		numClasses, numRules = len(self.classes.keys()), len(self.interactionSet)
		k = phi(numClasses, numRules, .5)

		return negBin(k,5,.5)


	def explainTimeStep(self, timestep, fullTimestep, timesteps, currTheories=False, override=False):
		"""
		Recursive function. Explains first event, then calls itself to explain the next events
		contingent on current explanations.
		Returns a set of theories that explain all the events that took place at timestep.
		currTheories can be passed as args to enable the explanation of multiple events in a single timestep.
		"""
		# print "in explainTimeStep. Explaining", timestep.events
		# if currTheories:
		# 	print "current theory:"
		# 	currTheories[0].display()
		theories = []

		# If we haven't provided theories that explain part of the time step, just explain the first event in the timestep
		if not currTheories:
			theories.extend(self.explainEvent(timestep.events[0], fullTimestep, timesteps, override=override))

		# Otherwise, you're now being passed the remainder of the timestep,
		# so timestep.events[0] is actually the first as-of-yet unexplained event.
		# Generate theories based on hypothetical theories (aka, currTheories)
		else:
			for theory in currTheories:
				newTheory = theory.explainEvent(timestep.events[0], fullTimestep, timesteps, override=override)
				theories.extend(newTheory)


		if len(timestep.events) == 1:							# Base Case
			# print "in base case of explainTimeStep"
			for t in theories:
				t.depth = self.depth+1
			# ## Falsify hypotheses
			# relevantEvents = [t for t in fullTimestep.events if 'killSprite' in t or 'transformTo' in t]
			# rle = fullTimestep.rle
			# for event in relevantEvents:
			# 	candidateSpriteType = [o for o in rle._game.sprite_groups if len(rle._game.sprite_groups[o])>0 and rle._game.sprite_groups[o][0].colorName == event[1]][0]
			# 	if len([o for o in rle._game.sprite_groups[candidateSpriteType] if o not in rle._game.kill_list]) == 0 and not rle._isDone()[0]:
			# 		self.falsified.append(SpriteCounterRule(self.colorToClassMapper(event[1]), 0, True))
			# 		self.falsified.append(SpriteCounterRule(self.colorToSpriteMapper(event[1]), 0, False))


			return theories
		else:													# Recursive Case
			# Create new timestep that consist of remaining unexpplained eventsl pass to the same function
			# print "in recursive case"
			updatedTimeStep = TimeStep(timestep.agentAction, timestep.agentState, timestep.events[1:], timestep.gameState, timestep.rle)
			return self.explainTimeStep(updatedTimeStep, fullTimestep, timesteps, currTheories=theories, override=override)

	def explainEvent(self, event, timestep, timesteps, override=False):
		"""
		Returns theories based on 'self' that explain the event, which is a tuple like:
			(bounceForward, BLUE, ORANGE)
		The theories will be Theory objects whose content may be something like:
			(bounceForward, c1, c2 IF health>1)

		"""
		theories = []


		likelihood = self.likelihood(timestep)

		if likelihood==1:
			# print "got likelihood=1"
			# embed()
			interpretedRule = self.interpret(event)
			matchingRule = [rule for rule in self.interactionSet if (interpretedRule.slot1, interpretedRule.slot2) == (rule.asTuple()[1], rule.asTuple()[2])][0]
			matchingRule.generic=False
			theories.append(self)
		else:
			failCase = self.getFailCases(event, timestep)

			# print event
			# print "\tFail case: ", failCase
			# print ""
			# embed()

			# This particular event is explained. Don't change anything.
			if failCase == 0:
				theories.append(self)
			# Add preconditions
			elif failCase in [1,2,3]:
				if failCase == 2 and event[0]=='killIfFromAbove': ## we're forgoing the process of doing proper precondition reasoning here; would be straightforward to do it.
					interpretation = self.interpret(event)
					theories.extend(self.addRules(event))
				else:
					theories.extend(self.addPreconditions(event, timestep, timesteps))
			# Add new rule
			elif failCase in [4]:
				theories.extend(self.addRules(event))

		return theories

	def colorToClassMapper(self,color):
		for c in self.classes:
			for c_class in self.classes[c]:
				if c_class.color == color:
					return c

		raise Exception("No corresponding class found for color")

	def makeGameStateWithClasses(self, gameState):
		classGameState = {k: 0 for k in self.classes.keys()}
		for c in self.classes:
			for s in self.classes[c]:
				if s.color in gameState:
					classGameState[c] += len(gameState[s.color])

		return classGameState


	def explainTermination(self, timestep, prevTimeSteps,result):
		"""
		adds all hypotheses about the termination conditions to the terminationSet
		params:
		timestep: the very last time step (at which termination occurs)
		prevTimeSteps: all time steps previous to the termination time step
		result: a dictionary for which the key 'win' is a boolean describing whether the game was won
		"""
		win = result['win']
		classesWithDiffAmounts = {} # objects which have different amounts in the termination time step from any previous timestep
		prevClassGameStates = [self.makeGameStateWithClasses(t.gameState['objects']) for t in prevTimeSteps]
		classGameState = self.makeGameStateWithClasses(timestep.gameState['objects'])
		for c in classGameState:
			timestep_amt = classGameState[c]
			timestep_amt_unique = not timestep_amt in [g[c] for g in prevClassGameStates]
			if timestep_amt_unique:
				classesWithDiffAmounts[c] = timestep_amt

		# print "IN TERMINATION CONDITIONS"
		# print timestep
		# # print timestep.events
		# print classesWithDiffAmounts
		# print {k:[c.asTuple() for c in v] for k,v in self.classes.items()}

		for event in timestep.events:
			for i in [1,2]:
				terminationClassColor = event[i] #self.getClass(event[i])
				terminationClassSymbol = self.colorToClassMapper(terminationClassColor)
				if terminationClassSymbol in classesWithDiffAmounts:
					timestep_amt = classesWithDiffAmounts[terminationClassSymbol]
					spriteCounterRule= SpriteCounterRule(terminationClassSymbol,timestep_amt,win)
					if not spriteCounterRule in self.terminationSet:
						self.terminationSet.append(spriteCounterRule)

		time = result["time"]
		timeoutRule = TimeoutRule(limit=time, win=win)
		if not timeoutRule in self.terminationSet:
			self.terminationSet.append(timeoutRule)


	def likelihood(self, timestep, sparse=False):
		"""
		Makes sure that:
			-all events in the timestep were covered by the ruleset
			-everything predicted in the ruleset happened.

		Right now returns only 1 or 0.
		"""
		# print "events in timestep {} | predictions in timestep {}".format(self.checkEventsInTimeStep(timestep), self.checkPredictionsInTimeStep(timestep))
		if self.checkEventsInTimeStep(timestep) and self.checkPredictionsInTimeStep(timestep, sparse):
			likelihood = 1.
		else:
			likelihood = 0.
		return likelihood

	def updateInteractionsPreconditions(self, resource, limit=None):
		if not limit: 
			new_precond = Precondition(
			text='new precondition for '+resource,
			item=resource, operator_name='>', num=0)
		else:
			new_precond = Precondition(
			text='new precondition for '+resource,
			item=resource, operator_name='>=', num=limit)

		# Add new generic rules for the avatar with preconditions
		newInteractionRules = []
		nonAvatars = [o for o in self.spriteSet if o.vgdlType not in AvatarTypes and o.color!='ENDOFSCREEN']
		for o in nonAvatars:
			rule = InteractionRule('killSprite', o.className, 'avatar', {}, set([new_precond]), generic=True)
			newInteractionRules.append(rule)
		# embed()

		return newInteractionRules

	def checkTerminationCounterInState(self, c, termCondition):
		"""
		c = game state with classes instead of colors
		"""
		return c[termCondition.termination.stype] == termCondition.termination.limit


	def getBadTerminationConditions(self, allTraces, verbose=False):
		"""
		Right now returns the list of termination conditions which are contradicted by the data.
		"""
		badTerminationConditions = []
		for termCondition in self.terminationSet:
			if termCondition.ruleType == "SpriteCounterRule":
				for timesteps,result in allTraces:
					for i in range(len(timesteps)):
						t = timesteps[i]
						c = self.makeGameStateWithClasses(t.gameState["objects"])
						if i == len(timesteps) - 1:
							if self.checkTerminationCounterInState(c, termCondition) and termCondition.termination.win != result["win"]:
								if not termCondition in badTerminationConditions:
									badTerminationConditions.append(termCondition)
						else:
							if self.checkTerminationCounterInState(c, termCondition): # if condition were true, would have ended on this time step.
								if not termCondition in badTerminationConditions:
									badTerminationConditions.append(termCondition)

			elif termCondition.ruleType == "TimeoutRule":
				for timesteps, result in allTraces:
					if result["time"] > termCondition.termination.limit:
						if not termCondition in badTerminationConditions:
							badTerminationConditions.append(termCondition)

		return badTerminationConditions


	def checkEventsInTimeStep(self, timestep):
		"""
		Check if all events in the timestep are covered by the interaction rule set.
		"""
		# print "events:", timestep.events
		interpretations = [self.interpret(event) for event in timestep.events]
		return all([self.checkEvents(i, timestep) for i in interpretations])


	def checkPredictions(self, event, timestep):
		"""
		Check if the relevant predictions to a specific event occurred.
		"""
		# TODO: Add comments here
		interpretations = [self.interpret(e).asTuple() for e in timestep.events if self.interpret(e) is not False]
		if interpretations:
			relevantRules = self.findRelevantRules(event, timestep.agentState, checkDryingPaint=True)
			if False in relevantRules:
				return ()
			if relevantRules:
				return all([rule.asTuple() in interpretations for rule in relevantRules])

		# If no interpretations, or if no relevant rules
		return ()


	def checkPredictionsInTimeStep(self, timestep, sparse=False):
		"""
		Check if all predictions for the timestep actually occurred.
		"""
		#Note: This fn cannot be exactly like checkPredictions(), becase here we don't care whether 'drying paint' is
		#T or F. We need to actually check all the predictions.
		interpretations = [self.interpret(event).asTuple() for event in timestep.events if self.interpret(event) is not False]

		relevantRules = []
		for event in timestep.events:
			relevantRules.extend(self.findRelevantRules(event, timestep.agentState, checkDryingPaint=False, sparse=sparse))

		# if set(['DARKBLUE', 'RED'])==set([event[1], event[2]]):
		# 	embed()
		if False in relevantRules:
			return False
		else:
			for rule in relevantRules:
				if rule.asTuple() not in interpretations:
					return False
			return True


	def getFailCases(self, event, timestep, verbose=False):
		"""
		Note: the only predictions we care about checking for here are the ones that are in the original theory.
		Predictions made by 'drying-paint' theories shouldn't be taken into account in the sense that all of these should receive
		the same treatment. That is, if we have (bf c1 c2) in the original theory, and are currently explaining the events:
		(ks c1 c2) (uA c1 c2),
		what we want to do is realize that (ks c1 c2) needs a precondition on it. Then we add this to a theory (as drying paint)
		and when we explain (uA c1 c2), we want to do exactly what we did with (ks c1 c2); recognize that it needs a single precondition.
		So checkPredictions only checks for theories that are not in dryingPaint.
		"""

		failCases = {(True, True): 	 [0, "Event likelihood = 1"],
					 (True, ()): [0, "Event likelihood = 1"],
					 (True, False):  [1, "Event likelihood failed because the interactionSet predicts things that didn't happen. "+
					 "Solution: Add preconditions to subset of interactionSet."],
					 (False, True):  [2, "Event likelihood failed because interpreted event is not in interactionSet. "+
					 "Solution: Add new rule with precondition on it."],
					 (False, False): [3, "Event likelihood failed both ways."+
					 "Solution: Add new rule with precondition on it; negate that precondition for other relevant rules."],
					 (False, ()):    [4, "Event likelihood failed because interactionSet hasn't seen the event."+
					 "Solution: AddRule()"]}


		(eventInRules, predictionsHappened) = self.checkEvents(self.interpret(event), timestep), self.checkPredictions(event, timestep)

		# print (eventInRules, predictionsHappened)
		# self.display()
		# print "event", event
		# print "interaction set:", [i.asTuple() for i in self.interactionSet]

		# if verbose:
		if (eventInRules, predictionsHappened) not in failCases.keys():
			print "weird fail case"
			embed()

		return failCases[(eventInRules, predictionsHappened)][0]


	def checkEvents(self, interpretation, timestep):
		"""
		Checks whether everything in the interpretation is accounted for by the interactionSet.
		"""

		if interpretation:
			interpretation = interpretation.asTuple()
			for rule in [r for r in self.interactionSet if not r.generic]:
				precon_list = [p.text for p in rule.preconditions]
				if rule.asTuple()==interpretation:

					# If no preconditions for the rule, then all is well.
					if not rule.preconditions:
						return True
					# Otherwise, make sure predonditions are met.
					else:
						preconditions_are_met = all([p.check(timestep.agentState) for p in rule.preconditions])

						return preconditions_are_met

		# If we've checked everything and found no matching rule or rule+precondition or couldn't even interpret the event, return False.
		return False


	def addRules(self, event, override=False):
		"""
		Search over possible assignments for classes; posit new classes if necessary
		Return theories that have either
		Try to make it fit according to the current rules by searching
		over possible class assignments for the objects.
		Returns a list of theories.
		"""
		# print 'in addRules...'
		newTheories = []
		possibleAssignments = self.searchForAssignments(event)
		try:
			## Events now have an optional last element of the tuple that is 'args': a dictionary of names-->values for things like
			## resources, stypes, values, etc.
			## These are relevant for interactions like teleportToExit, changeResource, etc.
			## Interactions in ontology.py that have arguments return at most two additional arguments. By convention, 'value' is always the
			## last of these.

			if 'stype' in event[3].keys():
				obj3 = self.spriteObjects[event[3]['stype']]
				# event[3]['stype'] = self.getClass(obj3)
				tmpEvent = copy.deepcopy(event)
				tmpEvent[3]['stype'] = self.getClass(obj3)
				args = tmpEvent[3]
			else:
				args = event[3]
		except:
			args = {}

		# if len(args.keys())>0:
		# 	print "found args in addRules for event", event
		# 	embed()
		obj1 = self.spriteObjects[event[1]]
		obj2 = self.spriteObjects[event[2]]

		if possibleAssignments:
			for assignment in possibleAssignments:
				class1, class2 = assignment[0], assignment[1]

				## Remove any relevant rules that are currently in the interaction set that are generic rules.
				rulesToRemove = [rule for rule in self.interactionSet if
					((class1==rule.slot1 and class2==rule.slot2) or (class1==rule.slot2 and class2==rule.slot1)) and rule.generic]

				# terminationsToRemove = [rule for rule in self.terminationSet if (class1 in rule.asTuple() or class2 in rule.asTuple()) \
				# and rule.ruleType=='SpriteCounterRule' and rule.generic]

				self.interactionSet = [rule for rule in self.interactionSet if rule not in rulesToRemove]
				# self.terminationSet = [rule for rule in self.terminationSet if rule not in terminationsToRemove]

				interaction = InteractionRule(event[0], assignment[0], assignment[1], args) #This isn't strictly necessary, but follows createChild requirements.
				# print "about to posit interactionRule"
				# embed()
				classAssignments = [(assignment[0], obj1), (assignment[1], obj2)]
				newTheory = self.createChild([interaction, classAssignments], override)
				# Checks and only adds to newTheories if the created theory was actually different.
				if newTheory:
					newTheories.append(newTheory)

		# print "adding {} theories with new assignments".format(len(newTheories))
		# embed()
		return newTheories

	def addPreconditions(self, event, timestep, timesteps):
		"""
		Creates preconditions based on the agentState that might help to explain the event.
		Returns a list of theories.
		"""

		newTheories = []

		obj1 = self.spriteObjects[event[1]]
		obj2 = self.spriteObjects[event[2]]

		classPair = (self.getClass(obj1), self.getClass(obj2))
		# If object classes are currently being modified in the same timestep, obtain the same preconditions as before
		if classPair in self.inModification.keys():
			p = self.inModification[classPair]
			interpretation = self.interpret(event)
			interpretation.addPrecondition(p) #TODO: maybe you should be only doing this if interpreting worked in the line above.
			newTheory = self.createChild([interpretation, False])
			if newTheory:
				newTheories.append(newTheory)
		else:
			# Create possible preconditions
			concepts = []
			for k in timestep.agentState.keys():
				concepts.extend(self.generateNumberConcepts(k, timestep.agentState[k])) #TODO: Combine generateNumberConcpets and makePreconditions
			generatedPreconditions = self.makePreconditions(concepts)
			for p in generatedPreconditions:

				# tmp_theory = copy.deepcopy(self)
				
				tmp_theory = self

				interpretation = tmp_theory.interpret(event)

				# ipdb.set_trace()
				# Find what rules you will need to negate
				relevantInteractionSetRules = tmp_theory.findRelatedRules(classPair, tmp_theory.interactionSet)
				relevantEvents = tmp_theory.findRelatedRules(classPair, [tmp_theory.interpret(e) for e in timestep.events])

				if len(relevantEvents)>len(relevantInteractionSetRules):
					# We want to add preconditions to the events that just happened that weren't predicted
					eventsToModify = [r for r in relevantEvents if r not in relevantInteractionSetRules]

					# And we need to make sure that the number concepts that we propose actually would have not been true in pervious cases (where this event didn't happen)
					relevantTimesteps = [t for t in timesteps[:-1] if [tmp_theory.interpret(event) for event in t.events if tmp_theory.interpret(event) in relevantEvents]]

					if all([not p.check(t.agentState) for t in relevantTimesteps]):
						for e in eventsToModify:
							e.addPrecondition(p)
							tmp_theory = tmp_theory.createChild([e, False])
						newTheories.append(tmp_theory)
						# interpretation.addPrecondition(p)
						# newTheories.append(tmp_theory)
						# newTheory = self.createChild([interpretation, False]) #TODO: make sure this is properly negating all other similar events

						# if newTheory:
							# newTheories.append(newTheory)
					# ipdb.set_trace()

				elif len(relevantInteractionSetRules)>len(relevantEvents):
					# We want to add preconditions to rules we have already put in the theory
					eventsToModify = [r for r in relevantInteractionSetRules if r not in relevantEvents]
					p.negate()
					## We're positing that the negated p (now called p) should have been true in that previous time step. If that's not true
					## it's because we generated a bad numberConcept, in which case we should just move on and not add it to the theory.
					relevantTimesteps = [t for t in timesteps[:-1] if [tmp_theory.interpret(event) for event in t.events if tmp_theory.interpret(event) in eventsToModify]]
					if all([p.check(t.agentState) for t in relevantTimesteps]):
						for e in eventsToModify:
							e.addPrecondition(p)
							# newTheory = self.createChild([e, False])
							newTheories.append(tmp_theory) ##TODO you didn't create a child theory, so your tracking of theory
														## genealogy will be off.
					# ipdb.set_trace()

				# else:
				# 	print "relevantEvents and relevantInteractionSetRules are disjoint but of same length"
				# 	embed()

		return newTheories


	def getNewClassName(self, color):
		existing_classes = [key for key in self.classes if key[0] == 'c']
		max_num = max([int(c[1:]) for c in existing_classes])
		class_num = max_num+1
		newClassName = 'c'+str(class_num)
		return newClassName

	def addSpriteToTheory(self, color, vgdlType='default', args=None):
		
		newSpriteName = self.getNewClassName(color)
		if vgdlType=='default':
			vgdlType = ResourcePack
		sprite = Sprite(vgdlType, color, className=newSpriteName)
		self.classes[newSpriteName] = [sprite]
		self.spriteSet.append(sprite)
		self.spriteObjects[color] = sprite
		for (o1,o2) in itertools.product([newSpriteName], self.classes.keys()):
			if o2=='avatar':
				rule1 = InteractionRule('killSprite', o1, o2, {}, set(), generic=True)
			else:
				rule1 = InteractionRule('nothing', o1, o2, {}, set(), generic=True)
			rule2 = InteractionRule('nothing', o2, o1, {}, set(), generic=True)
			self.interactionSet.append(rule1)
			self.interactionSet.append(rule2)
		return

	"""Helper functions"""
	def interpret(self, event):
		"""
		Looks up objects by their corresponding class under the theory,
		returns a corresponding interactionRule.

		If those objects aren't known, returns false.

		Ex) Event is a tuple: ('bounceForward', 'ORANGE', 'DARKBLUE') or ('changeResource', 'BLUE', 'RED', 1)
		If we know that ORANGE=c1 and DARKBLUE=c2, returns the InteractionRule
		that corresponds to (bounceForward, c1, c2)
		"""

		# try:
		# 	obj1 = self.spriteObjects[event[1]]
		# 	obj2 = self.spriteObjects[event[2]]
		# except:
		# 	print "couldn't find spriteObjects[event[k]] in interpret()"
		# 	embed()

		if event[1] in self.spriteObjects:
			obj1 = self.spriteObjects[event[1]]
		else:
			self.addSpriteToTheory(event[1])
			obj1 = self.spriteObjects[event[1]]

		if event[2] in self.spriteObjects:
			obj2 = self.spriteObjects[event[2]]
		else:
			self.addSpriteToTheory(event[2])
			obj2 = self.spriteObjects[event[2]]

		c1, c2 = self.getClass(obj1), self.getClass(obj2)

		# Check if there is an extra value argument in event. Also if there's an stype arg, get its class.
		try:
			if 'stype' in event[3].keys():
				obj3 = self.spriteObjects[event[3]['stype']]
				# event[3]['stype'] = self.getClass(obj3)
				tmpEvent = copy.deepcopy(event)
				tmpEvent[3]['stype'] = self.getClass(obj3)
				args = tmpEvent[3]
			else:
				args = event[3]
			# print "args", args
		except:
			args = {}


		#print 'classes:', c1, c2
		if c1 and c2:
			#print 'new interaction rule!'
			return InteractionRule(event[0], c1, c2, args)
		else:
			return False

	def findRelatedRules(self, classPair, interactionList):
		#needs to take a list of interpretations or a list of interaction rules
		return [interaction for interaction in interactionList if set(classPair) == set(interaction.asTuple()[1:3])]

	def negatePreconditions(self, unfulfilledPredictions):
		"""
		Given a list of Interaction rules, will add a negation to each rule, if rule is not in drying paint.
		"""
		if len(unfulfilledPredictions) > 0:
			print "in negatePreconditions"
			embed()
			# Iterate through relevant rules, negate them if they're not in the drying paint
			for r in unfulfilledPredictions:
				if r.asTuple() not in [new_r.asTuple() for new_r in self.dryingPaint]:
					precondition = r.preconditions # Single precondition object

					preconditionToNegate = copy.deepcopy(precondition)
					preconditionToNegate.negate()
					r.preconditions = [preconditionToNegate]

			# Generate new interaction set with new preconditioned rules
			newInteractionSet = []

			for r1 in self.interactionSet:
				for r2 in unfulfilledPredictions:
					if r1.asTuple()==r2.asTuple():
						newInteractionSet.append(r2)
					else:
						newInteractionSet.append(r1)


			self.interactionSet = newInteractionSet


	def makePreconditions(self, concepts):
		preconditions = []

		for c in concepts:
			text = c[0]
			item = c[1]
			operator_name = c[2]
			num = c[3]

			preconditions.append(Precondition(text, item, operator_name, num))
		return preconditions


	def createChild(self, proposal, override=False):
		"""
		Spawns a new child theory with the new proposal incorporated
		"""
		newTheory = copy.deepcopy(self)
		newTheory.depth = self.depth + 1
		newTheory.parent = self
		newTheory.children = []

		#TODO: Could copy over the spriteSet and self.classes?

		generatedNewTheory = newTheory.addProposal(proposal)

		# TODO: Fix this override; right now you're ignoring new assignments (though presumably if it gets called properly it won't be a problem)
		if override:
			rules_to_remove = [r for r in self.interactionSet if r.generic and r.slot1==proposal[0].slot1 and r.slot2==proposal[0].slot2]
			for rule in rules_to_remove:
				newTheory.interactionSet.remove(rule)
		if generatedNewTheory:
			self.children.append(newTheory)
			return newTheory
		else:
			return False

	def addProposal(self, proposal):
		"""
		Adds proposal to theory; takes care of rule and assignments
		"""

		## Helper functions
		def assignClass(classObjectPair):
			'''
			Adds object-class assignments; avoids duplicates.
			'''
			c, o = classObjectPair[0], classObjectPair[1]
			if c in self.classes.keys():
				if o not in self.classes[c]:
					self.classes[c].append(o)
					return True
				return False
			else:
				self.classes[c] = [o]
				return True

		def addInteractionRule(rule):
			"""
			Adds interactionRule if it is not in interactionSet.
			"""
			if rule.interaction not in self.predicates:
				self.predicates.add(rule.interaction)
			if not self.findRule(rule, self.interactionSet):
				self.interactionSet.append(rule)
				self.dryingPaint.add(rule)
				return True
			return False

		rule, assignments = proposal[0], proposal[1]
		# Add the proposed rule to the Theory's InteractionSet
		if rule:
			addedRule = addInteractionRule(rule)
		else:
			addedRule = False
		# Add the proposed class assignments to the Theory
		addedClass = False
		if assignments:
			addedClass = any([assignClass(assignment) for assignment in assignments])

		return (addedRule or addedClass)

	def updateTerminations(self, event=False, rle=None):
		self.terminationSet = [t for t in self.terminationSet
							   if t.ruleType=='SpriteCounterRule' and
							   not t.termination.win and all([not f.__eq__(t) for f in self.falsified])]

		if event:
			relevantEvents = [t for t in event['effectList'] if t[0] in ['killSprite', 'killIfHasLess', 'killIfHasMore', 'transformTo', 'collectResource']]
			# if relevantEvents:
				# ipdb.set_trace()
			rle = event['rle']
			for event in relevantEvents:
				candidateSpriteType = [o for o in rle._game.sprite_groups if len(rle._game.sprite_groups[o])>0 and rle._game.sprite_groups[o][0].colorName == event[1]][0]
				if len([o for o in rle._game.sprite_groups[candidateSpriteType] if o not in rle._game.kill_list]) == 0:

					## If the game didn't end, you can't win or lose based on this particular class being 0
					if not rle._isDone()[0]:
						self.falsified.append(SpriteCounterRule(self.colorToClassMapper(event[1]), 0, True))
						self.falsified.append(SpriteCounterRule(self.colorToClassMapper(event[1]), 0, False))
					else:
						## If you won, you can't lose based on this class being 0
						if rle._isDone()[1]:
							self.falsified.append(SpriteCounterRule(self.colorToClassMapper(event[1]), 0, False))
						else:
						## If you lost, you can't win based on this class being 0
							self.falsified.append(SpriteCounterRule(self.colorToClassMapper(event[1]), 0, True))

						## If you lost, maybe you lost because this class was 0. Check whether we'd already falsified this rule.
							loss_terminationRule = SpriteCounterRule(self.colorToClassMapper(event[1]), 0, False)
							if (all([not loss_terminationRule.__eq__(t) for t in self.terminationSet]) and
								all([not loss_terminationRule.__eq__(t) for t in self.falsified])):
									self.terminationSet.append(loss_terminationRule)


		if rle and not rle._isDone()[0]:
			knownColors = [sprite[0].color for sprite in self.classes.values()]
			presentColors = [rle._game.sprite_groups[o][0].colorName for o in rle._game.sprite_groups
							 if (len(rle._game.sprite_groups[o]) >
								len([dead_sprite for dead_sprite in rle._game.kill_list if dead_sprite.name==o])) and
							 rle._game.sprite_groups[o][0].colorName in knownColors]
			absentColors = [color for color in knownColors
							if color not in presentColors]

			for color in absentColors:
				## If the class is not at all present in this level,
				## you can't win or lose based on this particular class being 0
				self.falsified.append(SpriteCounterRule(self.colorToClassMapper(color), 0, True))
				self.falsified.append(SpriteCounterRule(self.colorToClassMapper(color), 0, False))

			for n in range(2, len(absentColors) + 1):
				for color_combination in itertools.combinations(absentColors, n):
					class_combination = [self.colorToClassMapper(color)
						for color in color_combination]
					self.multi_falsified.append(MultiSpriteCounterRule(stypes=class_combination))

		if 'avatar' in self.classes:
			if self.classes['avatar'][0].args and 'stype' in self.classes['avatar'][0].args:
				thingWeShoot = self.classes['avatar'][0].args['stype']
			else:
				thingWeShoot = None			
		else:
			if self.spriteObjects['DARKBLUE'].args and 'stype' in self.spriteObjects['DARKBLUE'].args:
				thingWeShoot = self.spriteObjects['DARKBLUE'].args['stype']
			else:
				thingWeShoot = None
		

		thingsThatCanBeKilled = []
		for rule in self.interactionSet:

			if rule.asTuple()[0] in ['killSprite', 'killIfHasLess', 'killIfHasMore', 'transformTo', 'nothing', 'collectResource']:
				if rule.generic and rule.preconditions:
					# if (    (rule.asTuple()[0]!='nothing' and not (rule.slot1==thingWeShoot and rule.slot2=='avatar')) or 
					# 		( rule.interaction == 'nothing' and not (rule.slot1==thingWeShoot and rule.slot2==thingWeShoot) and not ('Random' in str(self.classes[rule.slot1][0].vgdlType) or 'Chaser' in str(self.classes[rule.slot1][0].vgdlType)) and rule.slot2 not in ['avatar', thingWeShoot]) or
					# 		( rule.interaction == 'nothing' and not (rule.slot2==thingWeShoot and rule.slot1==thingWeShoot) and not ('Random' in str(self.classes[rule.slot2][0].vgdlType) or 'Chaser' in str(self.classes[rule.slot2][0].vgdlType)) and rule.slot1 not in ['avatar', thingWeShoot])
					# 		):
					if self.doWeMakeANoveltyRule(rule, thingWeShoot):
						terminationRule = NoveltyRule(rule.slot1, rule.slot2, True, copy.deepcopy(rule.preconditions))
						if (all([not ((t.termination.s2==rule.slot1) and (t.termination.s1==rule.slot2))
								for t in self.terminationSet if t.ruleType=='NoveltyRule']) and
							all([not terminationRule.__eq__(t) for t in self.terminationSet]) and
							all([not terminationRule.__eq__(t) for t in self.falsified])):
							self.terminationSet.append(terminationRule)
				elif rule.generic and not rule.preconditions:
					## Omit noveltytermination for randoms bumping into objects in the game; makes us disrupt plans even though we shouldnt't.
					# if ('Random' not in str(self.classes[rule.slot1][0].vgdlType)) and ('Random' not in str(self.classes[rule.slot2][0].vgdlType)) 
					# if (    (rule.asTuple()[0]!='nothing' and not (rule.slot1==thingWeShoot and rule.slot2=='avatar')) or 
					# 		( rule.interaction == 'nothing' and not (rule.slot1==thingWeShoot and rule.slot2==thingWeShoot) and not ('Random' in str(self.classes[rule.slot1][0].vgdlType) or 'Chaser' in str(self.classes[rule.slot1][0].vgdlType)) and rule.slot2 not in ['avatar', thingWeShoot]) or
					# 		( rule.interaction == 'nothing' and not (rule.slot2==thingWeShoot and rule.slot1==thingWeShoot) and not ('Random' in str(self.classes[rule.slot2][0].vgdlType) or 'Chaser' in str(self.classes[rule.slot2][0].vgdlType)) and rule.slot1 not in ['avatar', thingWeShoot])
					# 		):
					if self.doWeMakeANoveltyRule(rule, thingWeShoot):							
						terminationRule = NoveltyRule(rule.slot1, rule.slot2, True)
						if (all([not ((t.termination.s2==rule.slot1) and (t.termination.s1==rule.slot2))
								for t in self.terminationSet if t.ruleType=='NoveltyRule']) and
							all([not terminationRule.__eq__(t) for t in self.terminationSet]) and
							all([not terminationRule.__eq__(t) for t in self.falsified])):
							self.terminationSet.append(terminationRule)
				elif rule.asTuple()[0] in ['killSprite', 'killIfHasLess', 'killIfHasMore', 'transformTo', 'collectResource']:
					if rule.slot1 != 'avatar':
						thingsThatCanBeKilled.append(rule.slot1)
					terminationRule = SpriteCounterRule(rule.slot1, 0, True)
					if (all([not terminationRule.__eq__(t) for t in self.terminationSet]) and
						all([not terminationRule.__eq__(t) for t in self.falsified])):
						self.terminationSet.append(terminationRule)

			# if rule.slot2 == 'EOS' and rule.generic:
				# if not ('Random' in str(self.classes[rule.slot1][0].vgdlType)):
					# terminationRule = NoveltyRule(rule.slot1, rule.slot2, True)
					# self.terminationSet.append(terminationRule)
		# embed()
		falsified_win_stypes = set([sprite_rule.termination.stype for sprite_rule in self.falsified
			if (sprite_rule.termination.win and sprite_rule.termination.stype != 'EOS' and sprite_rule.termination.stype !='avatar')])

		try:
			falsified_win_stypes.remove(self.classes['avatar'][0].args['stype'])
		except:
			pass

		# if self.classes['avatar'][0].args and 'stype' in self.classes['avatar'][0].args.keys():
		# 	if sel
		# if any([rule.termination.stype=='explosion' for rule in self.falsified]):
		# 	embed()

		for n in range(2, len(falsified_win_stypes) + 1):
			for sprite_combination in itertools.combinations(falsified_win_stypes, n):
		# for n in range(2, len(thingsThatCanBeKilled) + 1):
			# for sprite_combination in itertools.combinations(thingsThatCanBeKilled, n):
				terminationRule = MultiSpriteCounterRule(stypes=sprite_combination)
				if (all([not terminationRule.__eq__(t) for t in self.terminationSet]) and
					all([not terminationRule.__eq__(t) for t in self.multi_falsified])):
					self.terminationSet.append(terminationRule)

		# if falsified_win_stypes:
		# 	embed()
		self.terminationSet = sorted(self.terminationSet, key=lambda t:t.ruleType)

		# if any([t.ruleType=='NoveltyRule' and t.termination.s1=='c7' and t.termination.s2=='avatar' and not t.termination.args for t in self.terminationSet]):
		# 	print "found c7 avatar w/o args"
		# 	embed()

		return

	def doWeMakeANoveltyRule(self, rule, thingWeShoot):
		if rule.asTuple()[0]!='nothing' and not (rule.slot1==thingWeShoot and rule.slot2=='avatar'):
			return True
		if rule.interaction == 'nothing':
			if (rule.slot1==thingWeShoot and rule.slot2 not in [thingWeShoot, 'avatar']) or (rule.slot2==thingWeShoot and rule.slot1 not in [thingWeShoot, 'avatar']) :
				return True
			if any([k in str(self.classes[rule.slot1][0].vgdlType) for k in ['Resource', 'Immovable', 'Missile']]) and any([k in str(self.classes[rule.slot2][0].vgdlType) for k in ['Resource', 'Immovable', 'Missile']]):
				return True
		return False

	def findRule(self, rule, lst):
		"""
		Finds if a rule is in the interaction set.
		"""
		for interactionRule in lst:
			if interactionRule == rule:
				return True
		return False


	def getClass(self, obj):
		"""
		Obtains the classes of the object; otherwise returns False if class not found.
		"""
		for classNum, objList in self.classes.iteritems(): #TODO: The issue is here w/ objects not found in the classes list
			for obj2 in objList:
				if obj == obj2:
					return classNum
		return False



	def findRelevantRules(self, event, agentState, checkDryingPaint=False, sparse=False):
		"""
		If an event involves c1 and c2, returns rules that use c1 and c2 in those slots.
		"""
		relevantRules = []

		# If both classes exist (whether predicate already exists doesn't matter)
		obj1 = self.spriteObjects[event[1]]
		obj2 = self.spriteObjects[event[2]]
		class1 = self.getClass(obj1)
		class2 = self.getClass(obj2)

		# if set([event[1], event[2]]) == set(['DARKBLUE', 'RED']):
		# 	embed()

		if class1 and class2:

			# This should not include any rules that don't satisfy the current preconditions
			rules = [rule for rule in self.interactionSet]

			if not sparse:
				#Default behavior
				if not checkDryingPaint:
					relevantRules.extend([rule for rule in rules if ((class1, class2) == (rule.asTuple()[1], rule.asTuple()[2]) or (class2, class1) == (rule.asTuple()[1], rule.asTuple()[2])) \
						and all([p.check(agentState) for p in rule.preconditions]) and not rule.generic])
				else:
					# Here we only return rules that are not in the drying paint.
					relevantRules.extend([rule for rule in rules if not self.findRule(rule, self.dryingPaint) \
						and ((class1, class2) == (rule.asTuple()[1], rule.asTuple()[2]) or (class2, class1) == (rule.asTuple()[1], rule.asTuple()[2]))  \
						and all([p.check(agentState) for p in rule.preconditions]) and not rule.generic])
			else:
				#'sparse' is passed when we check likelihood of lots of previous timesteps. The logic here is to
				#only check predictions for previous timesteps when the predictions may have changed. Meaning, only return rules that
				#are both relevant to the event *and* are new.
				relevantRules.extend([rule for rule in list(self.dryingPaint) if ((class1, class2) == (rule.asTuple()[1], rule.asTuple()[2]) or (class2, class1) == (rule.asTuple()[1], rule.asTuple()[2])) \
					and all([p.check(agentState) for p in rule.preconditions]) and not rule.generic])

		# If both classes don't exist
		else:
			return [False]

		return relevantRules



	def searchForPossibleClasses(self, obj_Sprite, newClasses=0):
		"""
		If the object has been assigned, return it. Otherwise return all
		possible classes. Optional argument can posit existence of a new class;
		user specifies whether to add 0, 1, or 2 new classes.
		"""
		gotNewClass = False

		# Get classes of sprites of the same VGDL Type
		possibleClasses = []
		for s in self.spriteSet:
			s_class = self.getClass(s)
			if s.vgdlType == obj_Sprite.vgdlType and s_class:
				if s_class not in possibleClasses:
					possibleClasses.append(s_class)

		# Class exists
		if self.getClass(obj_Sprite):
			return [self.getClass(obj_Sprite)], gotNewClass

		# Propose new classes and classes with sprites of the same vgdlType
		elif newClasses > 0:
			numClasses = len(self.classes.keys())
			for i in range(1, newClasses+1):
				possibleClasses.append('c'+str(numClasses+i)) # Classes that extend off number of existing classes
			gotNewClass = True
			return possibleClasses, gotNewClass

		else: return [], gotNewClass


	def searchForAssignments(self, event):
		obj1 = self.spriteObjects[event[1]]
		obj2 = self.spriteObjects[event[2]]


		x1, gotNewClass = self.searchForPossibleClasses(obj1, newClasses=1)
		if gotNewClass:
			x2 = self.searchForPossibleClasses(obj2, newClasses=2)[0]
		else:
			x2 = self.searchForPossibleClasses(obj2, newClasses=1)[0]

		if x1 and x2: # If both yielded possibilities
			classAssignments = []

			# If objects are the same type, any combo of classes is accepted
			if obj1.vgdlType == obj2.vgdlType:
				return list(itertools.product(x1,x2))

			 # If objects are diff type, want diff classes
			else:
				classAssignments = [(c1, c2) for c1 in x1 for c2 in x2 if c1!=c2]
				return classAssignments
		else: return False

	"""Prediction/generalization functions"""
	def findRuleClusters(self):
		ruleClusters = []
		uniquePairs = list(set([(rule.slot1, rule.slot2) for rule in self.interactionSet]))
		for pair in uniquePairs:
			rules = [(r.interaction, r.preconditions) for r in self.interactionSet if (r.slot1,r.slot2)==pair]
			ruleClusters.append(ruleCluster(rules, pair))
		return ruleClusters

	def predict(self, pair, lamda, tree, beta=1.,softmaxTemp=.1):

		#Takes a pair of object, generates a prediction (distribution over predicates) for what happens
		#if those collide.

		predicateList = ['killSprite', 'cloneSprite', 'stepBack', 'transformTo', 'undoAll',
		'bounceForward', 'conveySprite', 'windGust', 'slipForward', 'attractGaze', 'turnAround',
		'reverseDirection', 'flipDirection', 'bounceDirection', 'wallBounce', 'wallStop',
		'killIfSlow', 'killIfFromAbove', 'killIfAlive', 'collectResource', 'killIfHasMore',
		'killIfOtherHasMore', 'killIfHasLess', 'killIfOtherHasLess', 'wrapAround',
		'pullWithIt', 'teleportToExit']

		# print ""
		# print "predicting interactions for {} with parameters:".format(pair)
		# print "lambda = {}. beta = {}. tree = {}. softmax temp = {}".format(lamda, beta, tree.name, softmaxTemp)
		# print "(lambda: extrapolation (1) vs. guess (0) balance)"
		# print "(beta: ontology (1) vs. rule-similarity (0) balance)"
		# print ""

		#Get class memberships
		classes = (self.getClassFromColor(pair[0]), self.getClassFromColor(pair[1]))
		if False in classes:
			print "Can't make predictions; theory does not contain {}".format([el[0] for el in zip(pair, classes) if not el[1]])
			return False
		else:
			pair = classes

		inversePair = (pair[1], pair[0]) #a collision between cx and cy is the same as a collision between cy and cx. Locate both.
		# print 'making predictions for', pair[0], pair[1]

		knownRules = [rc for rc in self.findRuleClusters() if rc.pairs==pair or rc.pairs==inversePair]
		if len(knownRules)>0:
			#findRuleClusters will only return a single element if it works. It's a cluster, and contains all the matching rules.
			knownRules = knownRules[0].clusteredRules
			restOfRules = [p for p in predicateList if p not in [k[0] for k in knownRules]]

			knownRules = [[k, 1.] for k in knownRules]
			allRules = knownRules + [[r, 0.] for r in restOfRules]

			return allRules
		extrapolatedRules = [[r[0], r[1]*lamda] for r in self.extrapolateRule(pair, tree, beta)]
		guessedRules = [[r[0], r[1]*(1-lamda)] for r in self.guessRule(predicateList)]

		allRules = extrapolatedRules + guessedRules
		scores = softmax([r[1] for r in allRules], softmaxTemp)
		outList = [list(z) for z in zip([e[0] for e in allRules], scores)]

		#merge original extrapolated rules if they use the same predicates
		mergedRules = [outList[0]]
		for i in range(1, len(extrapolatedRules)):
			rule = outList[i]
			for m in mergedRules:
				if m[0]==rule[0]:
					m[1] += rule[1]
			if all([rule[0]!=m for m in [mR[0] for mR in mergedRules]]):
				mergedRules.append(rule)

		#convert ruleCluster rules to simple predicate form for ease of reading.
		#TODO: figure out what format you really want eventually, if you're going to
		#take actions, rather than just get a distribution over actions.
		mergedRules = [[m[0].clusteredRules, m[1]] for m in mergedRules]
		outList = mergedRules + outList[len(extrapolatedRules)+1:]

		# for o in outList:
		# 	print o
		return outList

	def guessRule(self, predicateList):
		#Currently returns interactions (no preconditions, and not in the form of interactionRules)
		#TODO: changeResource, spawnifHasMore require another argument. Add these and figure out how
		#to pass those args. Maybe this is best done in the step that creates interactionRules
		#in predict(). Also decide how to deal with values of optional args. Right now you'll
		#just make predictions based on default args.

		remainingPredicates = list(set(predicateList)-set([rule.interaction for rule in self.interactionSet]))
		scores = [1./len(remainingPredicates)]*len(remainingPredicates)
		return zip(remainingPredicates, scores)

	def extrapolateRule(self, pair, tree, beta=1.,softmaxTemp=False):
		#returns interactionRules (including preconditions) that are already in the interactionSet
		#weighted by their similarity to the provided pair.
		#TODO: think about default softmaxTemp.
		if len(self.interactionSet)==0:
			print "Can't extrapolate; our theory has no rules in the interactionSet!"
			return
		classPairs = list(set([(rule.slot1, rule.slot2) for rule in self.interactionSet]))
		similarityScores = [self.pairSimilarity(pair, classPair, tree, beta) for classPair in classPairs]
		similarityScores = normalize(similarityScores)
		if softmaxTemp:
			similarityScores = softmax(similarityScores,softmaxTemp)


		classSimilarities = zip(classPairs, similarityScores)

		ruleClusters = self.findRuleClusters()
		for ruleCluster in ruleClusters:
			ruleCluster.score = [cS[1] for cS in classSimilarities if cS[0]==ruleCluster.pairs][0]

		return ([[rc, rc.score] for rc in ruleClusters])

	def levenshtein(self, source, target):
		source, target = list(source), list(target)
		if max(len(source), len(target)) == 0:
			return 1.
		else:
			z = 1.*max(len(source), len(target))
			return 1. - self.levenshteinDistance(source, target)/z

	def levenshteinDistance(self, source, target):
		if len(source) < len(target):
			return self.levenshteinDistance(target, source)

		# So now we have len(source) >= len(target).
		if len(target) == 0:
			return len(source)

		# print 'source', source
		# We call tuple() to force strings to be used as sequences
		# ('c', 'a', 't', 's') - numpy uses them as values by default.
		source = np.array(tuple(source))
		target = np.array(tuple(target))
		# We use a dynamic programming algorithm, but with the
		# added optimization that we only need the last two rows
		# of the matrix.
		previous_row = np.arange(len(target) + 1)
		for s in source:
			# Insertion (target grows longer than source):
			current_row = previous_row + 1

			# Substitution or matching:
			# Target and source items are aligned, and either
			# are different (cost of 1), or are the same (cost of 0).

			current_row[1:] = np.minimum(
					current_row[1:],
					np.add(previous_row[:-1], [(t!=s).any() for t in target]))

			# Deletion (target grows shorter than source):
			current_row[1:] = np.minimum(
					current_row[1:],
					current_row[0:-1] + 1)

			previous_row = current_row

		return previous_row[-1]

	def levenshtein2(self, s1, s2):
		#Levenshtein (edit) distance. additions and deletions cost the same. No replacements.
		count = 0
		s1, s2 = list(s1), list(s2)
		for i in range(len(s1)):
			if s1[i] not in s2:
				s2.append(s1[i])
				count += 1
		to_remove = []
		for i in range(len(s2)):
			if s2[i] not in s1:
				to_remove.append(s2[i])
				count += 1
		for i in range(len(to_remove)):
			s2.remove(to_remove[i])
		return 1./(1+count)

	def ruleSimilarity(self, cx, cy):
		#Looks at rules in which cx participated in as slot 1, compares them to rules in which
		#cy participated as slot 1. Compares in terms of their edit distance.
		#Then does the same for slot 2.
		cxSlot1 = [(r.interaction, r.slot2, r.preconditions) for r in self.interactionSet
		if r.slot1==cx]
		cySlot1 = [(r.interaction, r.slot2, r.preconditions) for r in self.interactionSet
		if r.slot1==cy]

		cxSlot2 = [(r.interaction, r.slot1, r.preconditions) for r in self.interactionSet
		if r.slot2==cx]
		cySlot2 = [(r.interaction, r.slot1, r.preconditions) for r in self.interactionSet
		if r.slot2==cy]

		return .5*self.levenshtein(cxSlot1, cySlot1) + .5*self.levenshtein(cxSlot2, cySlot2)

	def pairSimilarity(self, pair1, pair2, tree, beta=1.):
		cx, cm, cy, cn = pair1[0], pair1[1], pair2[0], pair2[1]
		return (self.similarity(cx, cy, tree, beta) + self.similarity(cm, cn, tree, beta)) / 2.

	def similarity(self, cx, cy, tree, beta=1.):
		# Returns beta*treeSimilarity(c1,c2) + (1-beta)*ruleSimilarity(c1,c2)
		# Uses whatever tree is passed in. Currently we only have VGDLTree, which is
		# the original tree based on the VGDL ontology.
		n1, n2 = self.classes[cx][0].vgdlType, self.classes[cy][0].vgdlType
		treeSimilarity = tree.similarity(n1, n2)
		ruleSimilarity = self.ruleSimilarity(cx,cy)
		return beta*treeSimilarity + (1-beta)*ruleSimilarity

	def generateNumberConcepts(self, item, num): # TODO: Make this set of preconditions smaller
		"""
		Preconditions can be drawn from a pre-defined set of number concepts:
		n >= 0  --> any numbers from 0 to inf (having this amount of health is fine)
		n < 0 --> any negative numbers 		  (having this amount of health is bad)
		n >= 1 --> any numbers from 1 to inf  (having this amount of medicine and touching poison = safe)
		n < 1 --> any numbers from -inf to 0  (having this amount of medicine and touching poison = death)
		"""
		concepts = []

		if num<0:
			text = item+"<"+str(0)
			operator = '<'
			concepts.append((text,item,operator,0))
		if num<1:
			text = item+"<"+str(1)
			operator = '<'
			concepts.append((text,item,operator,1))
		if num>-1: #num>=0:
			text = item+">"+str(-1)
			operator = '>' # 			operator = '>='
			concepts.append((text,item,operator,-1))
		if num>0: #num>=1:
			text = item+">"+str(0)
			operator = '>'
			concepts.append((text,item,operator,0))

		## Check for limits of resources and add count(resource)==limit to the concepts.
		for rule in self.interactionSet:
			if 'resource' in rule.args.keys() and rule.args['resource']==item and 'limit' in rule.args.keys() and num>=rule.args['limit']:
				limit = rule.args['limit']
				##Also have a concept that is == num:
				text = item+">="+str(limit)
				operator = '>='
				concepts = [(text,item,operator,num)] ##superstition that you won because you had all of the limit items.
				break
		return concepts

	def getClassFromColor(self, color):
		for c in self.classes:
			if color in [cl.color for cl in self.classes[c]]:
				return c
		return False

	def displayRules(self):
		print ""
		print "InteractionSet:"
		for rule in self.interactionSet:
			rule.display()

	def displayClasses(self):
		print ""
		print "Class assignments:"
		for c in self.classes:
			class_list = [cl.color for cl in self.classes[c]]
			print "\t{}: {}: {}".format(c, class_list, self.spriteObjects[cl.color].vgdlType)
		#print self.classes
		print

	def displayTerminationSet(self):
		print ""
		print "TerminationSet:"
		for tc in self.terminationSet:
			tc.display()

	def display(self):
		print "_______"
		self.displayRules()
		self.displayClasses()
		self.displayTerminationSet() #TODO: Figure out why this isn't printing
		return

	def __eq__(self, other):
		if isinstance(other, self.__class__):

			# Must check interactionSet in this way, to use overloaded equality of InteractionRules
			interactionSetEqual = all(any(i1==i2 for i2 in other.interactionSet) for i1 in self.interactionSet)

			return all([
				self.spriteSet == other.spriteSet,
				self.levelMapping == other.levelMapping,
				interactionSetEqual, # TODO: Check if this uses InteractionRule overloaded __eq__
				self.classes == other.classes,
				self.terminationSet == other.terminationSet #may want to delete this
				]) # TODO: Add in termination set later
		else:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)


def normalize(array):
	z = float(sum(array))
	if z == 0:
		return [1./len(array)]*len(array) #if all items have the same score of 0, return the same score for all.
	else:
		return [a/z for a in array]

class Game(object):
	"""
	VGDL Game and Induction State.
	"""
	def __init__(self, vgdlString=False, spriteInductionResult=False):
		# Game states #TODO: May not need these
		#self.backpack = {}
		#self.trace = [] # list of TimeStep objects that happened during a gameplay

		self.vgdlString = vgdlString
		self.spriteInductionResult = spriteInductionResult
		if self.vgdlString:
			self.vgdlSpriteParse = self.makeSpriteParse()
		else:
			self.vgdlSpriteParse = False

		# Induction states
		self.hypothesisSpace = []
		self.theoryCount = 0

		#inherit ontology from VGDL
		self.VGDLTree = VGDLTree

		self.nodes_generated = 0
		self.nodes_eliminated = 0
		self.nodes_accepted = 0

	def display(self):
		print self.theoryCount

	def makeSpriteParse(self):
		s = SpriteParser()
		return s.parseGame(self.vgdlString)

	def posterior(self):
		#TODO: Consider allowing some amount of probability mass to uninstantiated hypotheses
		#The problem with this is it's not clear what the content of those hypotheses,
		#so it's unclear what you'd do with this new distribution.
		if len(self.hypothesisSpace)>0:
			z = 1.*sum([t.prior() for t in self.hypothesisSpace])
			for t in self.hypothesisSpace:
				t.posterior = t.prior()/z
			return [t.posterior for t in self.hypothesisSpace]
		else:
			print "Empty hypothesis space; can't give you a posterior."
	def entropy(self, theory):
		entropySum = 0
		numSpritesInClasses = float(sum([1 for c in theory.classes for i in c]))
		#print "\t num sprites total:", numSpritesInClasses
		for c in theory.classes:
			classLength = len(theory.classes[c])
			p = float(classLength/numSpritesInClasses)
			#print p
			entropySum += p * np.log2(p)
		return -1 * entropySum

	def orderHypotheses(self, hypotheses):
		temp_hypotheses = [(h, -1 * h.depth, self.entropy(h)) for h in hypotheses]
		temp_hypotheses = sorted(temp_hypotheses, key=operator.itemgetter(1,2))
		return [h[0] for h in temp_hypotheses]

	def explainTermination(self, theory, timestep, prevTimeSteps,result):
		"""
		adds all hypotheses about the termination conditions to the terminationSet
		params:
		theory: the theory that we are basing our new theories off of. Assume it's a member of hypothesis space.
		timestep: the very last time step (at which termination occurs)
		prevTimeSteps: all time steps previous to the termination time step
		result: a dictionary for which the key 'win' is a boolean describing whether the game was won
		"""

		win = result['win']
		classesWithDiffAmounts = {} # objects which have different amounts in the termination time step from any previous timestep
		try:
			prevClassGameStates = [theory.makeGameStateWithClasses(t.gameState['objects']) for t in prevTimeSteps]
		except TypeError:
			print "TypeError in explainTermination"
			embed()
		classGameState = theory.makeGameStateWithClasses(timestep.gameState['objects'])
		rulesToAdd = []
		for c in classGameState:
			timestep_amt = classGameState[c]
			timestep_amt_unique = not timestep_amt in [g[c] for g in prevClassGameStates]
			if timestep_amt_unique:
				classesWithDiffAmounts[c] = timestep_amt

		for event in timestep.events:
			# add sprite counter rules to the termination set, if applicable.
			# Infer potential sprite counter rules by looking at sprite counts for this timestep.
			for i in [1,2]:
				terminationClassColor = event[i] #self.getClass(event[i])
				terminationClassSymbol = theory.colorToClassMapper(terminationClassColor)
				if terminationClassSymbol in classesWithDiffAmounts:
					timestep_amt = classesWithDiffAmounts[terminationClassSymbol]
					spriteCounterRule= SpriteCounterRule(terminationClassSymbol,timestep_amt,win)
					if not spriteCounterRule in theory.terminationSet:
						rulesToAdd.append(spriteCounterRule)


		time = result["time"]
		timeoutRule = TimeoutRule(limit=time, win=win)
		# add a timeout rule to the termination set, if applicable. Use time at the end of this round.
		if not timeoutRule in theory.terminationSet:
			rulesToAdd.append(timeoutRule)

		theoryIsSufficient = len(rulesToAdd) > 0

		if not theoryIsSufficient:
			# parent theory's termination set was insufficient for explaining the termination
			# conditions of this time step. Need to add children theories to the hypothesis space.
			self.hypothesisSpace.remove(theory)
			for r in rulesToAdd:
				t = deepcopy(theory)
				t.terminationSet.add(r)
				self.hypothesisSpace.add(t)

	def completeTheory(self, theory, numSamples):

		def sampleCompletedTheory(game, theory):
			"""
			Assign all remaining sprites to a class for a given theory in a given game.
			This literally gives you a single *sample* from the possible ways you could extend the theory to include
			all seen objects.
			"""
			# Find all remaining sprites
			spritesLeft = []
			for sprite in theory.spriteSet:
				if not theory.getClass(sprite):
					spritesLeft.append(sprite)

			# For each sprite, assign it to a random possible class
			allClassAssignments = []			# Will save the class assignments here
			tempTheory = copy.deepcopy(theory) 	# Temporary theory
			for sprite in spritesLeft:
				possibleClasses,gotNewClass = tempTheory.searchForPossibleClasses(sprite, 1) # Second param is possible number of new classes
				sampledClass = choice(possibleClasses)

				classAssignments = [(sampledClass, sprite)]
				allClassAssignments.extend(classAssignments)
				tempTheory = tempTheory.createChild([None, classAssignments]) # Update the tempTheory; don't really want to save these theories

			# newHypothesisSpace = []
			# Finalize the temporary theory
			if tempTheory:
				newTheory = theory.createChild([None, allClassAssignments])
				# game.hypothesisSpace.append(newTheory)
				# newHypothesisSpace.append(newTheory)
			# return game.hypothesisSpace
			#used to return game.hypothesisSpace
			# return newHypothesisSpace
			return newTheory


		newHypothesisSpace = []
		for i in range(numSamples):
			newHypothesisSpace.append(sampleCompletedTheory(self, theory))
		return newHypothesisSpace

	def predict(self, pair, lamda, numCompletionSamples=10, tree=False, beta=1., softmaxTemp=.1):

		if not tree:
			tree = self.VGDLTree

		predictions = []

		# pairs = [(t.getClassFromColor(pair[0]), t.getClassFromColor(pair[1])) for t in self.hypothesisSpace]
		for theory in self.hypothesisSpace:
			prediction = theory.predict(pair, lamda, tree, beta, softmaxTemp)
			if prediction:
				predictions.append([prediction, theory.prior()])
			else: #prediction failed because we didn't have a complete theory
				newTheories = self.completeTheory(theory, numCompletionSamples)
				for t in newTheories:
					predictions.append([t.predict(pair, lamda, tree, beta, softmaxTemp), t.prior()])

		#TODO: The predicates are not always in the same order
		# and predicate list varies in size (becasue sometimes two things happen and
		#sometimes only one thing happens)

		predicates = [p[0] for p in predictions[0][0]]
		weights = [prediction[1] for prediction in predictions]
		z = sum(weights)
		weights = weights/z
		predLists = [prediction[0] for prediction in predictions]
		probs = [[p[1]*weights[i] for p in predLists[i]] for i in range(len(predictions))]
		print len(weights), len(probs), len(probs[0])
		sums = [sum([p[i] for p in probs]) for i in range(len(predicates))]

		return zip(predicates, sums)


	def DFSinduction(self, theory, timesteps, maxNumTheories, override=False, verbose=False):
		"""
		DFS implementation of induction function to deal with very long induction time.
		"""

		if verbose:
			print "\nStart hyp space length:", len(self.hypothesisSpace)
			print "running induction on theory"
			theory.display()

		# If still have time to generate more theories
		if len(self.hypothesisSpace) - 1 < maxNumTheories: # Subtracting one because of the initial hypothesis we must start out with to do induction
			ts_index = min(theory.depth, len(timesteps)-1) ## don't try to access nonexistent timesteps.

			if not timesteps:
				# If timesteps is an empty list, do nothing but return original theory.
				self.hypothesisSpace.append(theory)
				return

			if verbose:
				print "Current theory depth: ", ts_index
				print "Explaining event", timesteps[ts_index].events


			# Explain current timestep
			newTheories = theory.explainTimeStep(timesteps[ts_index], timesteps[ts_index], timesteps, override=override)

			self.nodes_generated += len(newTheories)
			if verbose:
				print "Possible new theories: ", len(newTheories)
				for theory in newTheories:
					theory.display()


			# If at the end of the timesteps list, add new theories to finalHypotheses
			if ts_index+1 == len(timesteps): # Need to add one, because you will create a theory of depth one greater than the length of the timesteps
				newTheoriesCount = 0
				for newTheory in newTheories:
					# if all(newTheory.likelihood(ts)==1.0 for ts in timesteps):
					if sum([newTheory.likelihood(ts) for ts in timesteps])/len(timesteps)>.5:
						self.nodes_accepted +=1
						newTheoriesCount += 1
						self.hypothesisSpace.append(newTheory)
					else:
						print "newTheory didn't explain all events"
						print "theory:"
						newTheory.display()
						self.nodes_eliminated +=1
				try:
					max_likelihood = np.unique([sum([h.likelihood(ts) for ts in timesteps]) for h in self.hypothesisSpace])[-1]
					self.hypothesisSpace = [h for h in self.hypothesisSpace if sum([h.likelihood(ts) for ts in timesteps]) == max_likelihood]
					# if len(timesteps)>6:
						# print "first max_likelihood"
						# embed()
				except IndexError:
					# timesteps is an empty list
					max_likelihood = 0
					print "WARNING: max_likelihood failed"
					# embed()
					
					self.hypothesisSpace = [theory]
				if verbose:
					print "New theories that passed likelihood tests: ", newTheoriesCount
					print "New hyp space length: ", len(self.hypothesisSpace)
					print "Nodes created: {}. Nodes eliminated: {}. Nodes accepted: {}".format(self.nodes_generated, self.nodes_eliminated, self.nodes_accepted)


			# If in the middle of timesteps:
			# We've made some new theories to explain the most recent timestep.
			# Make sure these new theories still explain all old timesteps.
			elif ts_index+1 != len(timesteps):
				acceptedTheories = []
				for t in newTheories:
					all_passed = True

					for ts in timesteps[:t.depth-1]: 			# Check that the theory can explain all timesteps
						if not t.likelihood(ts, sparse=True):
							self.nodes_eliminated +=1
							all_passed = False
							break

					# if all_passed:
						# self.nodes_accepted += 1
						# acceptedTheories.append(t)
					acceptedTheories.append(t) ##TODO: Now you're just adding every theory!

				newTheories = self.orderHypotheses(acceptedTheories)

				if verbose:
					print "New theories that passed likelihood tests: ", len(newTheories)
					for t in newTheories:
						t.display()
					print "Nodes created: {}. Nodes eliminated: {}. Nodes accepted: {}".format(self.nodes_generated, self.nodes_eliminated, self.nodes_accepted)

				for t in newTheories:
					t.dryingPaint = set()

				[self.DFSinduction(t, timesteps, maxNumTheories, override=override, verbose=verbose) for t in newTheories]


	def buildGenericTheory(self, spriteSample=True, vgdlSpriteParse=False):

		T = Theory(self)

		if spriteSample:
			T.initializeSpriteSet(vgdlSpriteParse=False, spriteInductionResult=spriteSample)
		else:
			T.initializeSpriteSet(vgdlSpriteParse = vgdlSpriteParse, spriteInductionResult=False)

		# Assign class names
		avatar = [o for o in T.spriteSet if o.vgdlType in AvatarTypes][0]
		nonAvatars = [o for o in T.spriteSet if o.vgdlType not in AvatarTypes and o.color!='ENDOFSCREEN']
		# wall = [o for o in T.spriteSet if o.color == "BLACK" or o.color=="GRAY"][0]
		# embed()
		allSprites = [avatar]+nonAvatars
		eos = [o for o in T.spriteSet if o.color=='ENDOFSCREEN'][0]

		# print "buildgenerictheory"
		# embed()
		avatar.className = 'avatar'
		T.classes[avatar.className] = [avatar]

		projectileName = ''
		try:
			projectileName = avatar.args['stype']
		except (TypeError, KeyError) as e:
			pass

		projectileTypes = [Flicker, OrientedFlicker, Missile]
		for i in range(len(nonAvatars)):
			if projectileName == nonAvatars[i].className:
				nonAvatars[i].className = projectileName
			else:
				nonAvatars[i].className = 'c'+str(i+2)

			T.classes[nonAvatars[i].className] = [nonAvatars[i]]
		T.classes['EOS'] = [eos] ##initialize EOS with special name, since it gets such special treatment in VGDL text files.

		for (o1, o2) in itertools.product(allSprites, allSprites):
			if (o1.vgdlType not in AvatarTypes and o2.vgdlType not in AvatarTypes) or (o1.className==projectileName and o2.vgdlType in AvatarTypes):
				rule = InteractionRule('nothing', o1.className, o2.className, {}, set(), generic=True)
				T.interactionSet.append(rule)
			elif o1.vgdlType not in AvatarTypes:
				rule = InteractionRule('killSprite', o1.className, o2.className, {}, set(), generic=True)
				T.interactionSet.append(rule)

		for s1 in nonAvatars + [avatar]:
			## append EOS rule
			rule = InteractionRule('stepBack', s1.className, 'EOS', {}, set(), generic=True)
			T.interactionSet.append(rule)

		rule =  SpriteCounterRule("avatar", 0, False)
		T.terminationSet.append(rule)

		T.updateTerminations()
		return T

	def addNewObjectsToTheory(self, theory, spriteSample):
		# Get the important objects in the theory names
		avatar = [o for o in theory.spriteSet if o.vgdlType in AvatarTypes][0]
		nonAvatars = [o for o in theory.spriteSet if o.vgdlType not in AvatarTypes and o.color!='ENDOFSCREEN']
		eos = [o for o in theory.spriteSet if o.color=='ENDOFSCREEN'][0]

		i = len(theory.classes)
		knownColors = [item.color for sublist in theory.classes.values() for item in sublist]
		for s in spriteSample:
			## If it's a sprite that's not in our theory, add it to the theory's classes
			## And intiialize all the generic rules.
			if s.color not in knownColors:
				s.className = 'c'+str(i)
				theory.classes[s.className] = [s]
				theory.spriteObjects[s.color] = s
				theory.spriteSet.append(s)
				rule = InteractionRule('killSprite', s.className, avatar.className, {}, set(), generic=True)
				theory.interactionSet.append(rule)
				rule = InteractionRule('stepBack', s.className, 'EOS', {}, set(), generic=True)
				theory.interactionSet.append(rule)
				for otherSprite in nonAvatars:
					rule = InteractionRule('nothing', s.className, otherSprite.className, {}, set(), generic=True)
					theory.interactionSet.append(rule)
					rule = InteractionRule('nothing', otherSprite.className, s.className, {}, set(), generic=True)
					theory.interactionSet.append(rule)
				i+=1
			else:
				## Since we're taking care of sprite property inference separately, update sprite info in the theory every time
				matchingSprite = [sprite for sprite in theory.spriteSet if sprite.color==s.color][0]
				s.className = matchingSprite.className
				theory.classes[s.className] = [s]
				theory.spriteObjects[s.color] = s
				theory.spriteSet.remove(matchingSprite)
				theory.spriteSet.append(s)
		return theory

		## decide how we're falsifying termination conditions, and tracking ones that weren't falsified.

	def runInduction(self, spriteSample, trace, maxNumTheories, verbose=False, existingTheories=False):
		# spriteSample: a particular assignment of sprite types. You can decide how you get this when you generate the sample, in getToSubgoal
		## Builds a generic theory and then overwrites it as it sees events in 'trace'.

		timesteps, result = trace

		## fiter for unique timesteps so that you don't waste time checking likelihoods, etc.
		try:
			unique_timesteps = [timesteps[0]]
		except IndexError:
			# timesteps is an empty list
			unique_timesteps = []

		for t in timesteps:
			if t.events not in [timestep.events for timestep in unique_timesteps]:
				unique_timesteps.append(t)

		timesteps=unique_timesteps

		if not existingTheories:
			# Start with fake theory (generic prior)
			T = self.buildGenericTheory(spriteSample)
			init_hypotheses = [T]
			self.hypothesisSpace = [] # Refresh the hypothesis space before DFS induction
		else:
			# Otherwise continue from the existing theories; work on the new events only.
			# print "had existing theory"
			## But first make sure we haven't seen a new object in the time step. if we have, it will be reflected in the spriteSample.
			for theory in existingTheories:
				theory = self.addNewObjectsToTheory(theory, spriteSample)
			init_hypotheses = existingTheories
			self.hypothesisSpace = []

		# This does DFS induction x times; not sure how to make it more like the behavior we want.
		for theory in init_hypotheses: 	# each of these theories has depth 1
			if verbose:
				theory.display()
			self.DFSinduction(theory, timesteps, maxNumTheories, override=True, verbose=verbose) ##override anything that was in the original set.


		# if any([self.hypothesisSpace[0].likelihood(ts)==0 for ts in timesteps]):
			# print "found 0 likelihood timestep"
			# embed()
		try:
			max_likelihood = np.unique([sum([h.likelihood(ts) for ts in timesteps]) for h in self.hypothesisSpace])[-1]
			self.hypothesisSpace = [h for h in self.hypothesisSpace if sum([h.likelihood(ts) for ts in timesteps]) == max_likelihood]
		except IndexError:
			# timesteps is an empty list
			max_likelihood = 0

		# Termination set induction
		## TODO: add this again.
		# if result:
		# 	hypothesisSpaceWithTermConditions = []
		# 	for theory in self.hypothesisSpace:
		# 		theory.explainTermination(timesteps[-1], timesteps[:-1], result)
		# 		hypothesisSpaceWithTermConditions.append(theory)

		# 	self.hypothesisSpace = hypothesisSpaceWithTermConditions

		if len(self.hypothesisSpace)==0:
			print "#################################################################"
			print "WARNING: no hypotheses. Returning the hypotheses we started with."
			print "#################################################################"
			self.hypothesisSpace = init_hypotheses
			# embed()


		return self.hypothesisSpace


	# def runDFSInduction(self, trace, maxNumTheories, override=False, verbose=False):
	# 	"""
	# 	"""

	# 	start = time.time()

	# 	timesteps, result = trace
	# 	temp_new_trace = ([timesteps[0]], None) # Just to run regular induction on first timestep

	# 	# Analyze first timestep (to get some sprites in theory classes so that entropy doesn't face divide by zero error)
	# 	self.induction(temp_new_trace, verbose=False)
	# 	self.cleanHypothesisSpace([timesteps[0]], 1)

	# 	init_hypotheses = self.orderHypotheses(self.hypothesisSpace)


	# 	self.hypothesisSpace = [] # Refresh the hypothesis space before DFS induction

	# 	# This does DFS induction x times; not sure how to make it more like the behavior we want.
	# 	for theory in init_hypotheses: 	# each of these theories has depth 1
	# 		if verbose:
	# 			theory.display()
	# 		self.DFSinduction(theory, timesteps, maxNumTheories, override, verbose=verbose)


	# 	# Termination set induction
	# 	if result:
	# 		hypothesisSpaceWithTermConditions = []
	# 		for theory in self.hypothesisSpace:
	# 			theory.explainTermination(timesteps[-1], timesteps[:-1], result)
	# 			hypothesisSpaceWithTermConditions.append(theory)

	# 		self.hypothesisSpace = hypothesisSpaceWithTermConditions

	# 	if verbose:
	# 		print "initial hypothesis space: ", len(self.hypothesisSpace)

	# 	end = time.time()
	# 	if verbose:
	# 		print "generated {} hypotheses in {} seconds".format(len(self.hypothesisSpace), end-start)

	# 	return self.hypothesisSpace



	def induction(self, trace, verbose=False, allTraces=None):
		"""
		Iterates through trace, performing theory induction on each timestep, contingent on theories inferred for the previous steps.
		"""
		T = Theory(self)

		##change this!
		if self.spriteInductionResult:
			T.initializeSpriteSet(vgdlSpriteParse=False, spriteInductionResult=self.spriteInductionResult)
			print "initialized from sprite induction result"
		elif self.vgdlSpriteParse:
			T.initializeSpriteSet(vgdlSpriteParse=self.vgdlSpriteParse, spriteInductionResult=False)
			print "initialized from sprite parse"

		self.hypothesisSpace = [T]
		newTheories = []

		# For every timestep
		timesteps, result = trace
		for i in range(len(timesteps)):
			timestep = timesteps[i]

			if verbose:
				print "explaining events {}".format(timestep.events)
				print "___________________________________________________________________"

			# For every theory
			for theory in self.hypothesisSpace:
				if theory.likelihood(timestep) < 1.0: 	# Theory needs to be changed
					newTheories.extend(theory.explainTimeStep(timestep, timestep, timesteps))

			# Make sure only to add unique theories
			#print "Iterating through new theories"
			for theory in newTheories:
				# theory.display()
				theoryIsNew = True
				for existingTheory in self.hypothesisSpace:
					if theory==existingTheory:
						theoryIsNew = False
						break
				if theoryIsNew:
					#print "ADDING NEW THEORIES IN INDUCTION --> now {} theories".format(len(self.hypothesisSpace))
					self.hypothesisSpace.append(theory) #TODO: numbering of theories should take place here.

			# if allTraces:
			# 	for theory in self.hypothesisSpace:
			# 		for timesteps,result in allTraces:
			# 			if result:
			# 				theory.explainTermination(timesteps[-1], timesteps[:-1], result)

			# 		badTerminationSet = theory.getBadTerminationConditions(allTraces)
			# 		for t in badTerminationSet:
			# 			theory.terminationSet.remove(t)


			self.cleanHypothesisSpace(timesteps[0:i+1], 1) #All timesteps up to now should be fully explained

			if verbose:
				print "{} hypotheses:".format(len(self.hypothesisSpace))

			# Sort hypotheses (right now by simple length metric), then print.
			hypotheses = sorted(self.hypothesisSpace, key=lambda x:len(x.interactionSet)*len(x.classes.keys()))

			if verbose:
				for h in hypotheses:
					h.display()
				print "___________________________________________________________________"
				print ""

		# Termination set induction
		if result:
			hypothesisSpaceWithTermConditions = []
			for theory in self.hypothesisSpace:
				theory.explainTermination(timesteps[-1], timesteps[:-1], result)
				hypothesisSpaceWithTermConditions.append(theory)

			self.hypothesisSpace = hypothesisSpaceWithTermConditions

		return self.hypothesisSpace

	def cleanHypothesisSpace(self, subtrace, threshold):
		"""
		Removes theories from hypothesisSpace if their likelihood for the timesteps
		passed in 'subtrace' is below threshold.
		"""
		# print "In cleanHypothesisSpace..."
		newHypothesisSpace = []

		#print "hypothesis space", self.hypothesisSpace
		for t in self.hypothesisSpace:
			#print "CHECKING THEORY:"
			#print " --> will check likelihood to see if the theory explains all of the timesteps (final check)"

			# print subtrace
			# for s in subtrace:
			# 	print "timestep: "
			# 	s.display()
			# 	print "likelihood:", t.likelihood(s)

			if all(t.likelihood(s)>=threshold for s in subtrace): #TODO: Issue might be here ?
				t.dryingPaint = set()
				newHypothesisSpace.append(t)


		self.hypothesisSpace = newHypothesisSpace
		# print "Done cleanHypothesisSpace...\n"
		return

def generateTheoryFromGame(rle, alterGoal=True):
	"""
	Given an rle, returns a very barebones theory object.
	This object has only 2 fields set: the interaction set, and the classes.
	"""
	theory = Theory(rle._game)

	inverseClasses = dict()
	for i,s in enumerate(rle._game.sprite_constr):
		(vgdlType, settings, _) = rle._game.sprite_constr[s]
		# Handle objects for which color is not declared
		# Should probably be done in a cleaner way when testing agent.py
		# since we suppose a 1:1 mapping from colors to objects
		try:
			color = colorDict[str(settings['color'])]
		except KeyError:
			color = 'noColor'

		if alterGoal and s=='goal':
			s = s[::-1] #reverse string. goal is to change names so as to not confuse anything with actual goal once you set it.
						# 'goal' is the only name that means something to all RLEs, so we're making sure to change this one.
		sprite = Sprite(vgdlType, color, className=s, args=settings) #classname was i
		theory.classes[s] = [sprite]
		inverseClasses[s] = i

	## Add EOS as a class, too.
	eos = Sprite(core.VGDLSprite, 'ENDOFSCREEN', None, None)
	theory.classes['EOS'] = [eos]

	for g1, g2, effect, kwargs in rle._game.collision_eff:
		if alterGoal:
			if g1=='goal':
				g1 = g1[::-1]
			if g2=='goal':
				g2 = g2[::-1]
		interaction = InteractionRule(effect.__name__, g1, g2, kwargs)
		# if not kwargs:
			# interaction = InteractionRule(effect.__name__, g1, g2, None, None)
		# 	interaction = InteractionRule(effect.__name__, g1, g2, None, None)
		# elif len(kwargs)==2:
		# 	interaction = InteractionRule(effect.__name__, g1, g2, kwargs.values()[0], kwargs.values()[1])
		# else:
		# 	print "Trying to generate theory from RLE. Got more args for collision than we can handle as of yet."
		# 	print "Embedding in generateTheoryFromGame()"
		# 	embed()


		# interaction = InteractionRule(effect.__name__, inverseClasses[g1], inverseClasses[g2], None, None)
		theory.interactionSet.append(interaction)

	# def __init__(self, limit=0, win=True, stypes = []):

	# Add termnation set
	for termination in rle._game.terminations:
		# No support for MultiSpriteCounterRule yet
		# Checking type with 'hasattr': ugly but isinstance breaks due to
		# relative imports
		if termination.name == 'SpriteCounter':
			if alterGoal and termination.stype=='goal':
				termination.stype='laog'
			spritecounter = SpriteCounterRule(limit=termination.limit,
											  stype=termination.stype,
											  win=termination.win)
			theory.terminationSet.append(spritecounter)
		elif termination.name == 'MultiSpriteCounter':
			if alterGoal:
				termination.stypes = ['laog' if t=='goal' else t for t in termination.stypes]
			multiSpriteCounter = MultiSpriteCounterRule(limit=termination.limit,
											  stypes=termination.stypes,
											  win=termination.win)
			theory.terminationSet.append(multiSpriteCounter)
		elif termination.name == 'Timeout':
			timeout = TimeoutRule(limit=termination.limit,
								  win=termination.win)
			theory.terminationSet.append(timeout)
		elif termination.name == 'NoveltyRule':
			noveltyrule = NoveltyRule(s1=termination.s1, s2=termination.s2, win=termination.win)
			theory.terminationSet.append(noveltyrule)

	return theory

def generateSymbolDict(rle):
	## run this once at the beginning of each game.
	## if new objects appear that are of an unknown type we have to be able to deal with this; writeTheoryToTxt should be
	## able to append to this dict if it finds any unknown objects.
	inverseMapping = dict()

	idx = 0
	try:
		colors = [colorDict[str(rle._game.sprite_constr[k][1]['color'])] for k in rle._obstypes.keys()]
	except:
		print "problem with generateSymbolDict"
		embed()
	try:
		colors.append(colorDict[str(rle._game.sprite_constr['avatar'][1]['color'])])
	except:
		colors.append(colorDict[str(rle._game.sprite_constr['avatar'][0].color)])
	possibilities = list(set([c for c in colors]))

	for p in possibilities:
		inverseMapping[p] = ALNUM[idx]
		idx+=1

	return inverseMapping

def getKeywordsFromOntology(interactionName):
	ontologyKeywordDict = \
	{'changeResource': ['resource', 'value', 'limit'],\
	'changeScore': ['value'],\
	'transformTo': ['stype'],\
	'transformToOnLanding': ['stype'],\
	'triggerOnLanding': ['strigger'],\
	'slipForward': ['prob'],\
	'attractGaze': ['prob'],\
	'reverseFloeIfActivated': ['strigger'],\
	'trigger': ['strigger'],\
	'detrigger': ['strigger'],\
	'bounceDirection': ['friction'],\
	'wallBounce': ['friction'],\
	'wallStop': ['friction'],\
	'killIfSlow': ['limitspeed'],\
	'spawnIfHasMore': ['resource', 'stype', 'limit'],\
	'killIfHasMore': ['resource', 'limit'],\
	'killOtherHasMore': ['resource', 'limit'],\
	'killIfHasLess': ['resource', 'limit'],\
	'killOtherHasLess': ['resource', 'limit'],\
	'wrapAround': ['offset']}
	if interactionName in ontologyKeywordDict.keys():
		return ontologyKeywordDict[interactionName]
	else:
		return []



def writeTheoryToTxt(rle, theory, symbolDict, txtFile, goalLoc = None):
	"""
	-need to be able to take an optional argument that tells you the location of the goal, and put that into the level string
	-assume that the goal sprite is getting killed
	-change the actual goal to be something else
	2 ways of swapping in knowledge:
	-cleanest way:
	"""
	def getClassNameFromSpriteString(spriteName):
		try:
			col = colorDict[str(rle._game.sprite_groups[spriteName][0].color)]
			try:
				className = [k for k in theory.classes.keys() if col in [c.color for c in theory.classes[k]]][0]
			except:
				print "couldn't find className"
				embed()
			return className
		except KeyError:
			if spriteName in theory.classes.keys():
				return spriteName
			else:
				try:
					## maybe we passed a color, so we should get the class.
					return theory.spriteObjects[spriteName].className
				except:
					print "failed to get spriteName color. In getClassNameFromSpriteString"
					embed()

	def buildArgsString(interactionRule):
		relevantArgNames = getKeywordsFromOntology(interactionRule.interaction)
		newInteractionName = interactionRule.interaction
		if interactionRule.interaction =='killSprite':
			oppositeOperatorMap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}
			precondition = list(set(interactionRule.preconditions))[0]
			if precondition:
				if precondition.negated:
					true_operator = oppositeOperatorMap[precondition.operator_name]
				else:
					true_operator = precondition.operator_name

				if true_operator in {"<", "<="}:
					newInteractionName = 'killIfHasLess' #example
					if true_operator == "<":
						limit = precondition.num - 2 ## used to be -2
					else:
						limit = precondition.num -1 ## used to be -1
					# embed()

				elif true_operator in {">", ">="}:
					newInteractionName = 'killIfOtherHasMore'
					if true_operator == ">":
						limit = precondition.num + 1
					else:
						limit = precondition.num

				argsString = " resource=%s limit=%s"%(precondition.item, str(limit))

		elif interactionRule.interaction=='teleportToExit':
			argsString = ""
		elif interactionRule.interaction == 'killIfFromAbove' or interactionRule.interaction == 'killIfFromBelow':
			argsString = ""
		else:
			if interactionRule.args:
				argsString = ""
				for k,v in interactionRule.args.items():
					if k in ['stype', 'strigger']:
						argsString += " %s=%s"%(k, getClassNameFromSpriteString(v))
					else:
						argsString += " %s=%s"%(k, v)

			else:
				# print "buildArgsString got called but no precondition"
				argsString = ""
				# embed()

		return argsString, newInteractionName

	# print "in initialize rle"
	# embed()

	DIRECTION_MAP = {(0,-1):'UP', (0,1):'DOWN', (1,0):'RIGHT', (-1,0):'LEFT'}

	for k,v in rle._game.sprite_groups.items():
		if k!='avatar' and k not in rle._obstypes:
			rle._obstypes[k] = [rle._sprite2state(sprite, oriented=False) for sprite in v if sprite not in rle._game.kill_list]


	_obstypes = rle._obstypes
	state = np.reshape(rle._getSensors(), rle.outdim)
	newGoalType, newGoalColor= None, None

	colorToSprite = {}

	for spriteType in rle._game.sprite_constr:
		if spriteType != "avatar":
			try:
				colorToSprite[colorDict[str(rle._game.sprite_constr[spriteType][1]['color'])]] = spriteType
			except KeyError:
				print "in writeTheoryToTxt, keyError"
				embed()

	if goalLoc:
		newGoalCode = state[goalLoc[0]][goalLoc[1]]
		if newGoalCode == 0:
			newGoalType = 'blank_space'
		else:
			newGoalIndex = int(round(math.log(newGoalCode,2)))-1
			newGoalType = sorted(_obstypes.keys())[::-1][newGoalIndex]
			newGoalColor = colorDict[str(rle._game.sprite_constr[newGoalType][1]['color'])]

	## teleport sprites have to be handled separately, as the spriteType is relational -- it depends on
	## what is in the interactionRules.
	if theory.interactionSet[0].args is not None:
		if any([len(i.args.keys()) for i in theory.interactionSet]):
			# print "found args in interactionRule"
			# embed()
			for interactionRule in theory.interactionSet:
				if interactionRule.interaction == 'teleportToExit':
					## second element in teleport tuple is the entrance; stype is the exit
					portalEntry = interactionRule.slot2
					portalExit = getClassNameFromSpriteString(interactionRule.args['stype'])

					theory.classes[portalEntry][0].vgdlType = Portal
					if theory.classes[portalEntry][0].args is None:
						theory.classes[portalEntry][0].args = {'stype':portalExit}
					else:
						theory.classes[portalEntry][0].args['stype'] = portalExit

					theory.classes[portalExit][0].vgdlType = Portal


	## TODO: Change.
	resourcesToAdd = set()
	for i in theory.interactionSet:
		if i.args is not None:
			for k,v in i.args.items():
				if k=='resource':
					resourcesToAdd.add(v)
			# if "resource" in i.args.keys():
			# 	resourcesToAdd.add(i.args["resource"])


	########### generating theory string
	theoryString = 'game = """\n'
	theoryString += "BasicGame\n"
	# first phase: the sprite rules
	theoryString += "\tSpriteSet\n"


	for c, sprites in theory.classes.items():
		if c == 'EOS':
			pass
		else:
			# embed()
			for s in sprites:
				unfilteredType = str(s.vgdlType)
				stype = unfilteredType[unfilteredType.find("vgdl.ontology.")+len("vgdl.ontology."): unfilteredType.find(">")-1]
				argsString = ""
				## Catch-all 'OTHER' s.vgdlType is causing a problem. replace for now with generic.
				if not stype:
					if unfilteredType == "OTHER":
						stype = 'ResourcePack'
					else:
						print "writetheorytotxt. stype problem"
						embed()

				if s.args:
					for k,v in s.args.items():
						if k == "color":
							continue
						elif k == "orientation":
							argsString += " %s=%s"%(k, DIRECTION_MAP[v])
						elif k == "speed":
							argsString += " %s=%s"%(k, v)
						else:
							argsString += " %s=%s"%(k, str(v))

				try:
					argsString += " %s=%s"%("speed", str(s.speed))
				except AttributeError:
					# print "couldn't find speed"
					# embed()
					pass

				try:
					argsString += " %s=%s"%("orientation", DIRECTION_MAP[s.orientation])
				except AttributeError:
					pass

				try:
					argsString += " %s=%s"%("fleeing", s.fleeing)
				except AttributeError:
					pass

				try:
					argsString += " %s=%s"%("cooldown", s.cooldown)
				except AttributeError:
					pass

				if hasattr(s, 'stype'):
					try:
						##when we initialized stypes in spriteInduction, we didn't have access to what we would call objects in the theory.
						colorConvertedToSType = theory.spriteObjects[s.stype].className
						# embed()
						argsString += " %s=%s"%("stype", colorConvertedToSType)
					except KeyError:
						print "in TheoryToTxt(), search for colorConvertedToSType"
						## TODO: If you, say, hypothesize that a missile is a Chaser and that it chases some random color but you don't have that color in your theory yet,
						## you can end up here.
						# embed()



				if "core" in stype:
					stype = stype[stype.find("core.")+len("core."):]

				if "avatar".lower() in stype.lower():
					theoryString += "\t\t%s > %s color=%s%s\n"%("avatar", stype, s.color, argsString)
				else:

					sname = c
					theoryString += "\t\t%s > %s color=%s%s\n"%(sname, stype, s.color, argsString)
					if goalLoc and newGoalType != 'blank_space' and s.color==newGoalColor:
						sname = colorToSprite[s.color]
						theoryString += "\t\t%s > %s color=%s%s\n"%("goal", stype, s.color, argsString)

	for resource in resourcesToAdd:
		theoryString += "\t\t%s > Resource color=RESOURCETOADD limit=%s\n"%(resource, theory.resource_limits[resource])

	if goalLoc:
		if newGoalType == 'blank_space':
			# we've selected an empty square to be the goal.
			theoryString += "\t\tgoal > Passive color=LIGHTRED\n"



	immovable_predicates = ['stepBack', 'undoAll']
	kill_predicates = ['killSprite']
	immovables, killerObjects = [], []
	# second phase: the interaction rules
	theoryString += "\tInteractionSet\n"
	added_rules = []
	sortedInteractionDict = {}

	## There's no reason to write 'nothing' interactions in the theory, except that we need
	## to use the eventHandler to detect collisions for collisions that we've never witnessed.
	## So, only write 'nothing' interactions if they're (a) generic, (b) you haven't seen the reversed classPair interaction already
	## The 'continue' statements below take care of this.

	# create a dict mapping interacting class pairs to their list of interactions
	for interactionRule in theory.interactionSet:
		
		c1 = interactionRule.slot1
		c2 = interactionRule.slot2
		if c1 > c2:
			c1, c2 = c2, c1 # flip order

		if not (c1,c2) in sortedInteractionDict:
			if interactionRule.interaction=='nothing' and not interactionRule.generic:
				continue
			sortedInteractionDict[(c1, c2)] = [interactionRule]
		else:
			if interactionRule.interaction=='nothing':
				continue
			sortedInteractionDict[(c1, c2)].append(interactionRule)
	# embed()
	sortedInteractions = []
	# For some games (e.g. boulderdash), the order of 'stepBack' interactions
	# matters: this list puts those that don't involve the avatar at the end
	nonAvatarStepBackInteractions = []
	for pair in sortedInteractionDict:
		(killIfHasLessInteractions, killInteractions, scoreChangeInteractions,
			nonKillInteractions, changeResourceInteractions) = [], [], [], [], []
		for interactionRule in sortedInteractionDict[pair]:
			if "kill" in interactionRule.interaction:
				precondition = list(set(interactionRule.preconditions))
				if precondition and precondition[0].operator_name in ['<', '<=']:
					killIfHasLessInteractions.append(interactionRule)
				else:
					# check whether this is a killing interaction
					killInteractions.append(interactionRule)
			elif "changeScore" in interactionRule.interaction:
				scoreChangeInteractions.append(interactionRule)
			elif "changeResource" in interactionRule.interaction:
				changeResourceInteractions.append(interactionRule)
			elif "stepBack" in interactionRule.interaction and 'avatar' not in [interactionRule.slot1, interactionRule.slot2]:
				nonAvatarStepBackInteractions.append(interactionRule)
			else:
				nonKillInteractions.append(interactionRule)

		sortedInteractions += (killIfHasLessInteractions +
			scoreChangeInteractions + changeResourceInteractions +
			killInteractions + nonKillInteractions)
		# make sure that killing interactions get processed before interactions
		# that don't kill.
		# EDIT: made killIfHasLess be processed first

	sortedInteractions += nonAvatarStepBackInteractions
	for interactionRule in sortedInteractions:

		if all([not interactionRule.__eq__(r) for r in added_rules]): ## don't duplicate rules.

			c1 = interactionRule.slot1
			c2 = interactionRule.slot2

			# if c2=='EOS' or c1=='EOS': ## 'EOS stepBack' is always being written at the end. Don't handle it here.
			# 	continue
			if (c1=='laog' and len(theory.classes[c1])==0) or (c2=='laog' and len(theory.classes[c2])==0):
				print "found laog"
				embed()

			for s1 in theory.classes[c1]:
				if c2 not in theory.classes.keys():
					embed()
				for s2 in theory.classes[c2]:
					argsString = ""

					if interactionRule.preconditions or interactionRule.args:
						args, interactionRule.interaction = buildArgsString(interactionRule)
						argsString += args

					if s1.color==newGoalColor:
						if not 'avatar' in str(s2.className): #only add actual goal object rule if it's not interacting with the avatar.
							theoryString += "\t\t%s %s > %s%s\n"%('goal', c2, interactionRule.interaction, argsString)
					elif s2.color==newGoalColor:
						if not 'avatar' in str(s1.className):#only add actual goal object rule if it's not interacting with the avatar.
							theoryString += "\t\t%s %s > %s%s\n"%(c1, 'goal', interactionRule.interaction, argsString)
					else:
						theoryString += "\t\t%s %s > %s%s\n"%(c1, c2, interactionRule.interaction, argsString)

					if 'avatar' in str(s1.vgdlType).lower():
						if interactionRule.interaction in immovable_predicates:
							# print "must add immovable"
							# embed()
							immovables.append(s2.className)
						if interactionRule.interaction in kill_predicates:
							killerObjects.append(s2.className) ##killSprite is not symmetrical; you to append things that are (avatar obj killSprite)
					elif 'avatar' in str(s2.vgdlType).lower():
						if interactionRule.interaction in immovable_predicates:
							# print "must add immovable"
							# embed()
							immovables.append(s1.className)
			added_rules.append(interactionRule)

	# if goal is an empty square
	# if newGoalType == 'blank_space':
	# theoryString += "\t\t%s %s > %s\n"%('goal', 'avatar', "killSprite")
	# theoryString += "\t\t%s %s > %s\n"%('avatar', 'EOS', "stepBack")

	# print "inwritetheory"
	# embed()
	## add EOS stepBack for all other sprites
	# for c in theory.classes.keys():
	# 	if c is not 'avatar':
	# 		theoryString += "\t\t%s %s > %s\n"%(c, 'EOS', "stepBack")


	# print "in writeTheory"
	# embed()
	immovables = list(set(immovables))
	killerObjects = list(set(killerObjects))


	# if theory.interactionSet[0].args is not None:
	# 	if any([len(i.args.keys()) for i in theory.interactionSet]):
	# 		print "inwritetheory"
	# 		embed()

	# third phase: the termination rules
	theoryString += "\tTerminationSet\n"
	goalConditionNotFound = True
	for terminationRule in theory.terminationSet:
		if terminationRule.ruleType == "TimeoutRule":
			theoryString += "\t\tTimeout limit=%s win=%s\n" % (str(terminationRule.termination.limit), str(terminationRule.termination.win))

		elif terminationRule.ruleType == "SpriteCounterRule":
			theoryString += "\t\tSpriteCounter stype=%s limit=%s win=%s\n" % \
						(terminationRule.termination.stype, \
						str(terminationRule.termination.limit), str(terminationRule.termination.win))
			if terminationRule.termination.stype == "goal":
				goalConditionNotFound = False
		elif terminationRule.ruleType == "NoveltyRule":

			theoryString += "\t\tNoveltyTermination s1=%s s2=%s win=%s" % \
						(terminationRule.termination.s1, terminationRule.termination.s2, str(terminationRule.termination.win))
			if terminationRule.termination.args:
				# print "found args in terminationrule"
				# embed()
				noveltyArgString = " args={item:%s,num:%s,negated:%s,operator_name:%s}" % \
				(terminationRule.termination.args.item, terminationRule.termination.args.num, terminationRule.termination.args.negated, terminationRule.termination.args.operator_name)
				theoryString += noveltyArgString

			theoryString +="\n"
		else:
			# multi sprite counter rule
			theoryString += "\t\tMultiSpriteCounter "
			for i in range(len(terminationRule.termination.stypes)):
				theoryString += "stype%i=%s " % (i, terminationRule.termination.stypes[i])

			theoryString += "limit=%s win=%s\n" % (str(terminationRule.termination.limit), str(terminationRule.termination.win))


	if goalLoc and goalConditionNotFound:
		# embed()
		theoryString += "\t\tSpriteCounter stype=goal limit=0 win=True\n"

	# # fourth phase: the level mapping

	mappedState = []
	for i in range(rle.outdim[0]):
		newEntry = []
		for j in range(rle.outdim[1]):
			newEntry.append(" ")

		mappedState.append(newEntry)

	for r in range(rle.outdim[0]):
		for c in range(rle.outdim[1]):
			if state[r][c] > 0:
				try:
					symbol = objectsToSymbol(rle, rle.getObjectsFromNumber(state[r][c]), symbolDict)
					mappedState[r][c] = symbol
				except:
					print "in map"
					embed()

			try:
				if mappedState[r][c] == " " and goalLoc == (r,c):
					# an empty square has been selected as the goal
					mappedState[r][c] = "G"
			except:
				print "mappedState problem2"
				embed()

	levelString = 'level="""\n'
	for mappedRow in mappedState:
		levelString += reduce(lambda a,b: a+b, mappedRow) + "\n"

	levelString += '"""\n'

	theoryString += "\tLevelMapping\n"

	for colors, symbol in symbolDict.items():
		if type(colors)==tuple:
			types = [theory.spriteObjects[c].className for c in colors if c in theory.spriteObjects.keys()]
			if len(types)==2:
				theoryString += "\t\t%s > %s %s\n"%(symbol, types[0], types[1])
			elif len(types)==3:
				theoryString += "\t\t%s > %s %s %s\n"%(symbol, types[0], types[1], types[2])
		elif type(colors)==str and colors in theory.spriteObjects.keys():
			c = theory.spriteObjects[colors].className
			theoryString += "\t\t%s > %s\n"%(symbol, c)

	# print "in writeTheory"
	# embed()
	# theoryString += "\t\tG > goal\n"
	theoryString += '"""\n'


	parserString = 'if __name__ == "__main__":\n\tfrom vgdl.core import VGDLParser\n\tVGDLParser.playGame(game, level)\n'

	gameString = levelString + theoryString + parserString
	with open(txtFile, 'w') as f:
		f.write(gameString)
	f.close()

	levelString = levelString[levelString.find('"""')+3:-4]
	theoryString = theoryString[theoryString.find('"""')+3:-4]
	return theoryString, levelString, symbolDict#, immovables, killerObjects
