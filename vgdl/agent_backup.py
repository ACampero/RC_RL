import multiprocessing as mp
from functools import partial
from util import *
import sys, traceback
from core import colorDict, VGDLParser, keyPresses
from ontology import *
from theory_template import Precondition, InteractionRule, TerminationRule, TimeoutRule, \
SpriteCounterRule, MultiSpriteCounterRule, Theory, Game, writeTheoryToTxt, generateSymbolDict, \
generateTheoryFromGame, expandLine, expandSprites, proposePredicates, getRuleSetsForClassPairPredicate,\
interateThresholds
from class_theory_template import Sprite
import os, subprocess, shutil
from collections import defaultdict
import importlib
import numpy as np
import ipdb, time
import os, subprocess, shutil
import copy
from math import ceil, floor
import warnings
from rlenvironmentnonstatic import createRLInputGame, createRLInputGameFromStrings, defInputGame, createMindEnv
from stateobsnonstatic import buildTracker, UNOBSERVABLE_PREDICATES
from termcolor import colored
from line_profiler import LineProfiler
from vgdl.util import manhattanDist, manhattanDist2, LinkedDict
from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT
from colors import colorDict
import copy_reg
import types
import heapq
from tqdm import tqdm, trange

import WBP
from termcolor import colored
from pathos.helpers import mp

ACTIONDICT = {K_UP: (0,1), K_DOWN: (0,-1),K_LEFT: (-1,0), K_RIGHT: (1,0), K_SPACE: (0,0), 0: (0,0)}

# This makes experience replay score a theory on all the one-step transitions we've seen
EXPERIENCE_REPLAY_METHOD = 'all'
# What average likelihood do we consider adequate for passing a theory on to the next generation?
ERRORCUTOFF = .3
# Not active now
NUM_SAMPLES_PER_HYPOTHESIS = 20
# how long to just watch before theorizing about the game
OBSERVATION_PERIOD_LENGTH = 10
initialErrorBuildup = []

class errorMapEntry:
	def __init__(self):
		self.diagnosis = []
		self.targetToken = None
		self.targetTokens = []
		self.targetClass = None
		self.targetColor = None
		self.intPairs = []
		self.componentsAddressed = []
		self.episodeStepGenerated = None
	
	def display(self):
		print ""
		print "diagnosis: {}".format(self.diagnosis)
		if self.targetTokens:
			print "targetTokens: {}".format(self.targetTokens)
		else:
			print "targetToken: {}".format(self.targetToken)
		print "targetClass: {}".format(self.targetClass)
		print "targetColor: {}".format(self.targetColor)
		print "intPairs: {}".format(self.intPairs)
		print "components addressed: {}".format(self.componentsAddressed)

	def copy(self):
		e                   	= errorMapEntry()
		e.diagnosis         	= self.diagnosis
		e.targetToken       	= ccopy(self.targetToken)
		e.targetTokens      	= ccopy(self.targetTokens)
		e.targetClass       	= self.targetClass
		e.targetColor 			= self.targetColor
		e.intPairs          	= self.intPairs
		e.componentsAddressed 	= self.componentsAddressed
		e.episodeStepGenerated	= self.episodeStepGenerated
		return e

	def __eq__(self, other):
		if set(self.diagnosis) == set(other.diagnosis) and self.targetClass == other.targetClass and sorted(self.intPairs)==sorted(other.intPairs):
			return True
		else:
			return False

class Agent:
	def __init__(self, modelType, gameFilename, hyperparameter_sets={}, parallel_planning=False):
		self.modelType = modelType
		self.gameFilename = gameFilename
		self.gameString = None
		self.levelString = None
		self.hyperparameter_sets = hyperparameter_sets
		self.parallel_planning = parallel_planning
		self.annealingFactor = 1.
		self.shortHorizon = False # How much do we search for a good plan before giving up?
		if self.shortHorizon == True:
			self.starting_max_nodes = 1000
			self.max_nodes_annealing = 1.05
		else:
			self.starting_max_nodes = 10000
			self.max_nodes_annealing = 10
		self.firstOrderHorizon = False # Makes you commit to a plan once first-order distances change (e.g., spritecounter values)
		self.regrounding = 3
		self.reground_for_killer_types = True # encourages safe behavior
		self.reground_for_stochastic_types = True # encourages replanning more often as these agents deviate from prediction
		self.safeDistance = 6
		self.emptyPlansLimit = 1#should be 5
		self.longHorizonObservationLimit = 2
		self.scores = []
		self.hypotheses = []
		self.symbolDict = None
		self.finalEventList = []
		self.statesEncountered = []
		self.fakeInteractionRules = []
		self.all_objects = {}
		self.spriteUpdateDict = defaultdict(lambda : 0)
		self.seen_resources = defaultdict(lambda : []) #key: a hypothesized avatar color. Value: list of colors of resources seen by that avatar.
		self.seen_limits = defaultdict(lambda: [])
		self.new_objects = {}
		self.memory = []
		self.rleHistory = []
		self.actionHistory = []
		self.allTheories = []
		self.theoryScoreHistory = []
		self.meanErrorHistory = []
		self.minStepError = []
		self.actionSet = [K_RIGHT, K_LEFT, K_UP, K_DOWN, K_SPACE]
		self.randomTheories = []
		self.benchmarkHistory = []
		self.resourceObservations = {'speed':[], 'changeResource':[]}
		self.observed_resources = set()
		self.distributions = {}
		self.history = {}
		self.lastObjectState = {}

		# Hyperopt output
		self.total_game_steps = 0
		self.total_planner_steps = 0
		self.levels_won = 0
		self.assumeZeroErrorTheoryExists = False

	def initializeEnvironment(self):
		if self.gameString == None or self.levelString == None:
			self.gameString, self.levelString = defInputGame(self.gameFilename, randomize=False)
		self.rleCreateFunc = lambda: createRLInputGameFromStrings(self.gameString, self.levelString)
		self.rle = self.rleCreateFunc()
		self.rle._game.spriteUpdateDict = self.spriteUpdateDict
		self.rle._game.observation = buildTracker(self.rle)
		return

	def initializeRLEFromGame(self):
		gameString, levelString = self.gameString, self.levelString
		if gameString == None or levelString == None:
			gameString, levelString = defInputGame(self.gameFilename, randomize=False)
		rleCreateFunc = lambda: createRLInputGameFromStrings(gameString, levelString)
		rle = rleCreateFunc()
		return rle

	def initializeHypotheses(self):

		spriteInduction(self.rle._game, step=1, action=None)

		spriteList = []
		colors = self.rle._game.observation['trackedObjects'].keys()
		for color in colors:
			s = Sprite(vgdlType=ResourcePack, colorName=color)
			spriteList.append(s)
		gameObject = Game(spriteInductionResult=spriteList)
		initialTheory = gameObject.buildGenericTheory(spriteList)
		initialTheory.terminationSet = [r for r in initialTheory.terminationSet if r.ruleType == 'SpriteCounterRule']
		initialTheory.mostRecentEdit = 'none'

		self.symbolDict = generateSymbolDict(self.rle)

		## Instantiate a hypothesis that each singleton class might be the avatar
		self.hypotheses = []
		## Grab all singleton classes and instantiate hypotheses that they are the avatar.
		for color in self.symbolDict.keys():
			if len(getSpritesByColor(self.rle._game, color)) == 1:
				newTheory = copy.deepcopy(initialTheory)
				oldClassName = newTheory.spriteObjects[color].className
				del newTheory.classes[oldClassName]
				newTheory.spriteObjects[color].className = 'avatar'
				newTheory.spriteObjects[color].vgdlType = MovingAvatar
				newTheory.classes['avatar'] = [newTheory.spriteObjects[color]]

				for rule in newTheory.interactionSet:
					if rule.slot1 == oldClassName:
						rule.slot1='avatar'
					if rule.slot2 == oldClassName:
						rule.slot2='avatar'

				## Rename classes to ensure canonical ordering: c2, c3, ...
				if min([int(k[1:]) for k in newTheory.classes.keys() if 'c' in k])>2:
					for s in newTheory.spriteSet:
						if s.className is not None and 'c' in s.className:
							tmpClassName = s.className
							del newTheory.classes[tmpClassName]
							s.className = 'c'+str(int(s.className[1:])-1)
							newTheory.classes[s.className] = [s]
					for rule in newTheory.interactionSet:
						if 'c' in rule.slot1:
							rule.slot1 = 'c'+str(int(rule.slot1[1:])-1)
						if 'c' in rule.slot2:
							rule.slot2 = 'c'+str(int(rule.slot2[1:])-1)

				self.hypotheses.append(newTheory)
				self.history[color] = {}
		return

	def observe(self, rle, episode_num, num_steps=1):
		for i in range(num_steps):
			action = 0
			bestScoresAndHypotheses = self.executeStep(episode_num, self.rleHistory, self.actionHistory, action, self.hypotheses)
			self.hypotheses = [item[1] for item in bestScoresAndHypotheses]
		# for i in range(num_steps):
			# spriteInduction(rle._game, step=1, action=None)
			# updateAllOptions(rle._game, rle._game, action=None)
		return

	def testCurriculum(self, level_game_pairs=None, actionSequences=None):
		if not level_game_pairs:
			module = importlib.import_module(self.gameFilename)
			level_game_pairs = module.level_game_pairs
		
		results = []
		for n_level, level_game in enumerate(level_game_pairs):

			print("Playing level {}".format(n_level))
			(self.gameString, self.levelString) = level_game

			for epoch in range(1):
				res = self.testEpisodes(epoch=epoch, actionSequences=actionSequences[n_level] if actionSequences else None)
				results.append(res)
		return results

	def playCurriculum(self, level_game_pairs=None, num_episodes_per_level=10):
		""" Plays a game level until it wins, then moves to the next one until
		completion. """
		if not level_game_pairs:
			level_game_pairs = importlib.import_module(self.gameFilename).level_game_pairs
		episodes = []
		
		num_levels = len(level_game_pairs)
		## for inference
		self.rleHistory = [[] for i in range(num_levels*num_episodes_per_level)]
		self.actionHistory = [[] for i in range(num_levels*num_episodes_per_level)]
		self.all_objects = [{} for i in range(num_levels*num_episodes_per_level)]
		
		episodes_played = 0
		for n_level, level_game in enumerate(level_game_pairs):
			self.gameString = level_game[0]
			self.levelString = level_game[1]

			episodes = []

			self.max_nodes = self.starting_max_nodes
			win = False
			i = 0
			# TODO: never used
			first_time_playing_level = True
			while not win and i < num_episodes_per_level:
				win, score, steps = self.playEpisode(n_level, episodes_played, win=win, first_time_playing_level=first_time_playing_level)
				self.total_game_steps += steps
				episodes.append((n_level, steps, win, score))
				episodes_played += 1
				if win:
					print 'won'
					break
				i += 1
			if i < num_episodes_per_level:
				self.levels_won += 1

		return

	def playEpisode(self, n_level, episode_num, flexible_goals=False, win=False, first_time_playing_level=False):
		
		self.initializeEnvironment()
		print "initializing RLE"
		# embed()
		steps, self.quits, self.longHorizonObservations = 0,0,0
		self.all_objects[episode_num] = self.rle._game.getAllObjects()
		ended, win = self.rle._isDone()
		annealing = 1

		self.statesEncountered.append(self.rle._game.getFullState())
		
		envReal = self.fastcopy(self.rle)
		if sum([len(episode) for episode in self.rleHistory]) <= OBSERVATION_PERIOD_LENGTH:
			# have to save extra info since we deal with the error maps after the step they occur
			copyGameInferenceInfo(envReal, self.rle)

		self.rleHistory[episode_num].append(envReal)
		
		if not self.hypotheses:
			if n_level!=0 and episode_num!=0:
				print "Have no hypotheses but not playing the first episode / first level!"
				embed()
			self.initializeHypotheses()
			updateTerminations(self.rle, self.hypotheses, addNoveltyRules=False)

		if first_time_playing_level:
			## Add defaults to theories for any new objects.
			## All hypotheses have the same number of classes / know about the same colors
			newColors = [k for k in envReal._game.observation['trackedObjects'].keys() if k not in self.hypotheses[0].spriteObjects]
			if newColors:
				self.observe(self.rle, episode_num, OBSERVATION_PERIOD_LENGTH)
				from vgdl.ontology import Resource
				for color in newColors:
					for h in self.hypotheses:
						existing_classes = [key for key in h.classes if key[0] == 'c']
						max_num = max([int(c[1:]) for c in existing_classes])
						class_num = max_num+1 
						newClassName = 'c'+str(class_num)
						h.addSpriteToTheory(newClassName, color, vgdlType=Resource)

		emptyPlans = 0
		while not ended:

			envReal = self.fastcopy(self.rle)

			## Select hypothesis/es to plan with.
			selectedHypotheses 	= [self.hypotheses[0]] # Only initialize as many theories as you are using parallel planners
			# selectedHypotheses = self.hypotheses
			hypothesesToPlanWith = [convertTheoryToSubgoalTheory(h) for h in selectedHypotheses]

			## Create fake incentives to reexplore previously-explored items while possessing resources
 			for h in hypothesesToPlanWith:
 				avatarColor = h.classes['avatar'][0].colorName
 				try:
 					a = envReal._game.observation['trackedObjects'][avatarColor][0]
 				except:
 					print "Didn't find avatar"
 					embed()

 				for k,v in envReal._game.observation['trackedObjects'][avatarColor][0].inventory.items():
 					resourceClass = h.spriteObjects[k].className
 					resourceAmount, limit = v[0], v[1]
 					resourceClass = h.spriteObjects[k].className
 					h.resource_limits[resourceClass] = limit

 					if resourceAmount>0:
 						h.fakeInteractionRules.extend(h.updateInteractionsPreconditions(resourceClass))
 					if resourceAmount==limit:
 						h.fakeInteractionRules.extend(h.updateInteractionsPreconditions(resourceClass, limit))

 					
 					if k not in self.seen_resources[avatarColor]:
 						self.seen_resources[avatarColor].append(k)
 					elif resourceClass not in self.seen_limits[avatarColor] and resourceAmount==limit:
 			 			self.seen_limits[avatarColor].append(resourceClass)

 					# if k not in self.seen_resources[avatarColor]:
 					# 	resourceClass = h.spriteObjects[k].className
 					# 	h.fakeInteractionRules.extend(h.updateInteractionsPreconditions(resourceClass))
 					# 	h.resource_limits[resourceClass] = limit
 					# 	self.seen_resources[avatarColor].append(k)
 					# elif resourceClass not in self.seen_limits[avatarColor] and resourceAmount==limit:
 			 	# 		h.fakeInteractionRules.extend(h.updateInteractionsPreconditions(resourceClass, limit))
 			 	# 		self.seen_limits[avatarColor].append(resourceClass)

			# embed()
 			[h.updateTerminations(addNoveltyRules=True) for h in hypothesesToPlanWith]

 			# Only initialize as many planner theories as you are using parallel planners
			plannerRLEs = VrleInitPhase(hypothesesToPlanWith, envReal)

 			# if envReal._game.observation['trackedObjects'][avatarColor][0].inventory:
 				# print "found inventory"
 				# embed()
			# embed()

			quitting = False
			if self.parallel_planning:
				pass
				# def WBP_wrapper(l):
				# 	hyperparameters, theory, queue = l
				# 	p = WBP.WBP(plannerRLEs[0], self.gameFilename, theory=theory, fakeInteractionRules = self.fakeInteractionRules,
				# 		seen_limits = self.seen_limits, annealing=annealing, max_nodes=self.max_nodes, shortHorizon=self.shortHorizon,
				# 		firstOrderHorizon=self.firstOrderHorizon, hyperparameters=hyperparameters)
				# 	return p
				# # # start planners
				# # print('#1')
				# result_queue = None
				# # print('#2')
				# pool = mp.Pool()
				# # print('#3')
				# res = pool.map_async(WBP_wrapper, [(h_set, self.hypotheses[0], result_queue) for h_set in self.hyperparameter_sets])
				# # print('#4')
				# pool.close()
				# # print('#5')
				# pool.join()
				# # print('#6')
				# best_index = np.argmin([p.total_nodes for p in res._value])
				# p = res._value[best_index]
			else:
				avatarColor = hypothesesToPlanWith[0].classes['avatar'][0].colorName

				p = WBP.WBP(plannerRLEs[0], self.gameFilename, theory=hypothesesToPlanWith[0], fakeInteractionRules = self.fakeInteractionRules,
					seen_limits = self.seen_limits[avatarColor], annealing=annealing, max_nodes=self.max_nodes, shortHorizon=self.shortHorizon,
					firstOrderHorizon=self.firstOrderHorizon, hyperparameters=self.hyperparameter_sets[0])
			
			bestNode, gameStringArray, predictedEnvs = p.BFS()

			# best_index = np.argmin([p.total_nodes for p in res._value])
			# bestNode, gameStringArray, predictedEnvs = res._value[best_index].BFS()
			self.total_planner_steps = p.total_nodes

			if bestNode is not None:
				solution = p.solution
				gameString_array = p.gameString_array
				predictedEnvs = predictedEnvs[::-1]
			else:
				solution = []

			if solution and not p.quitting:
				print "============================================="
				print "got solution of length", len(solution)
				for g in p.gameString_array:
					print colored(g, 'green')
				print "============================================="

			if self.shortHorizon:
				if not solution:
					emptyPlans +=1
				else:
					emptyPlans = 0
			else:
				if (not solution) or p.quitting:
					if self.longHorizonObservations<self.longHorizonObservationLimit:
						print "Didn't get solution or decided to quit. Observing, then replanning."
						# embed()
						self.observe(self.rle, episode_num, num_steps=5)
						solution = [] ## You may have gotten p.quitting but also a solution; make sure you don't try to act on that if the planner decided it wasn't worth it.
						self.longHorizonObservations += 1
					else:
						quitting = True

			if emptyPlans > self.emptyPlansLimit:
				print "observing"
				# embed()
				self.observe(self.rle, episode_num, num_steps=5)

			if not quitting:
				for action_num, action in enumerate(solution):
					bestScoresAndHypotheses = \
							self.executeStep(episode_num, self.rleHistory, self.actionHistory, action, self.hypotheses)
					
					## TODO: Prediction error only corresponds to self.hypotheses[0]. What you actually want is
					## checking for the predictions made by *each* of the hypotheses, and then if any give you prediction error,
					## you reground based on that.
					predictionError, regroundForKillerTypes, regroundForStochasticTypes = self.regroundOrNot(action_num, predictedEnvs, self.hypotheses[0])

					print bestScoresAndHypotheses
					self.hypotheses = [item[1] for item in bestScoresAndHypotheses]
					
					if selectedHypotheses[0]!=self.hypotheses[0]:
						# print "Best theory is no longer equal to selected theory"
						# embed()
						break

					steps +=1

					ended, win = self.rle._isDone()
					if regroundForKillerTypes or regroundForStochasticTypes: 
						print "got reground for killer or stochastic type. Replanning"
						break

					if ended:
						break

				if self.shortHorizon:
					self.max_nodes *= self.max_nodes_annealing
			else:
				## You failed the game either because you made a mistake you couldn't recover from or because you timed out in your search.
				## Search more deeply next time.
				self.max_nodes *= self.max_nodes_annealing
				print "You got quitting==True from planner. Embedding to debug."
				# embed()
				return False, self.rle._game.score, steps
		
			annealing *= self.annealingFactor
			ended, win = self.rle._isDone()

		score = self.rle._game.score
		output = "ended episode. Win={}                   						  ".format(win)
		if win:
			print colored('________________________________________________________________', 'white', 'on_green')
			print colored('________________________________________________________________', 'white', 'on_green')

			print colored(output, 'white', 'on_green')
			print colored('________________________________________________________________', 'white', 'on_green')
		else:
			print colored('________________________________________________________________', 'white', 'on_red')
			print colored(output, 'white', 'on_red')
			print colored('________________________________________________________________', 'white', 'on_red')

		return win, score, steps
	




	def testEpisodes(self, epoch=0, actionSequences=None):
		num_cores = mp.cpu_count()
		print "num cores: {}".format(num_cores)
		if num_cores<40:
			print "WARNING: running on < 40 cores."

		if actionSequences == None:
			actionSequences = [
				## TEST8
				# [K_LEFT, K_LEFT, K_LEFT, K_LEFT, K_LEFT, K_LEFT, K_LEFT, 0]
				## PUSH_BOULDERS_2
				# [K_RIGHT]*3, [K_RIGHT, K_RIGHT, K_UP]

				# [K_LEFT, K_LEFT, K_LEFT]
				[0]*6
			]

		# add mandatory observation period
		actionSequences[0] = [0]*OBSERVATION_PERIOD_LENGTH + actionSequences[0]

		self.rleHistory = [[] for i in range(len(actionSequences))]
		self.actionHistory = [[] for i in range(len(actionSequences))]
		self.all_objects = [{} for i in range(len(actionSequences))]

		totalTimeStart = time.time()

		for episode_num, actions in enumerate(actionSequences):
			print "initializing RLE. Epoch={}".format(epoch)

			self.initializeEnvironment()
			self.all_objects[episode_num] = self.rle._game.getAllObjects() ## we need to store all_objects across multiple episodes

			if episode_num == 0:
				self.initializeHypotheses()

			envReal = self.fastcopy(self.rle)
			self.rleHistory[episode_num].append(envReal)

			for num, action in enumerate(actions):
				if self.rle._isDone()[0]:
					print "Game is over."
					break
				print ">>> Step", num+1, "of", len(actions), "<<<"
				## initialize VRLEs
				theoryRLEs = VrleInitPhase(self.hypotheses, self.rle)
				t2 = time.time()
				scoresAndHypotheses = self.executeStep(episode_num, self.rleHistory, self.actionHistory, action, self.hypotheses)
				print ""
				print "executed step in {} seconds".format(time.time()-t2)
				print ""
				self.scores, self.hypotheses = zip(*scoresAndHypotheses)

			print ">>> Embedded at the end of episode {}".format(episode_num)
			embed()

		totalTime = time.time() - totalTimeStart

		return (totalTime, zip(self.scores, self.hypotheses))

	def manageNewObjects(self, episode_num, hypotheses, envRealPrev, action):
		# embed()
		## Add newly-seen objects.
		current_objects = self.rle._game.getAllObjects()
		newObjects = [k for k in current_objects if k not in self.rle._game.movement_options]
		if newObjects:
			spriteInduction(self.rle._game, step=1, action=action, specificSpritesToUpdate=[])

		return hypotheses

	def intersect(self, p1, p2):
		return (abs(p1[0] - p2[0]) <= self.rle._game.block_size and abs(p1[1] - p2[1]) <= self.rle._game.block_size)
	
	def fastcopy(self, rle):

		newRle = self.initializeRLEFromGame()
		newRle._obstypes = ccopy(rle._obstypes)
		if hasattr(rle, '_gravepoints'):
			newRle._gravepoints = ccopy(rle._gravepoints)
		newRle._game.sprite_groups = ccopy(rle._game.sprite_groups)
		newRle._game.kill_list = ccopy(rle._game.kill_list)
		newRle._game.lastcollisions = ccopy(rle._game.lastcollisions)
		newRle._game.time = ccopy(rle._game.time)
		newRle._game.score = ccopy(rle._game.score)
		newRle._game.keystate = ccopy(rle._game.keystate)
		newRle._game.observation = ccopy(rle._game.observation)
		newRle.symbolDict = ccopy(rle.symbolDict)
		newRle._game.sprite_groups['avatar'][0].resources = ccopy(rle._game.sprite_groups['avatar'][0].resources)

		return newRle

	def scoreAndFilterTheories(self, newTheories, episode_num, displayTheories=False):
		print "top of scoreAndFilterTheories"

		penalties, imaginedEffectsPerTheory = MultiEpisodeExperienceReplay(newTheories, self.rleHistory[:episode_num+1], \
				self.actionHistory[:episode_num+1], method=EXPERIENCE_REPLAY_METHOD, displayTheories=False, assumeZeroErrorTheoryExists=self.assumeZeroErrorTheoryExists)

		for n,imaginedEffects in enumerate(imaginedEffectsPerTheory):
			newTheories[n].setOfImaginedEffects = newTheories[n].setOfImaginedEffects.union(imaginedEffects)

		scoreAndTheoryTuples = zip(penalties, newTheories)
		scoreAndTheoryTuples = sorted(scoreAndTheoryTuples, key=lambda x: (x[0], x[1].prior()))

		for num, sh in reversed(list(enumerate(scoreAndTheoryTuples))):
			if num > 5:
				continue
			print "Theory: {} | Error: {}".format(num, sh[0])
			sh[1].display()
		scoreAndTheoryTuples = [s for s in scoreAndTheoryTuples if not hasattr(s[1],'trueTheory')]      

		scoresAndHypotheses = [(h[0],h[1]) for h in filterTheories(scoreAndTheoryTuples, percentile=30, max_num=30,
			proportionOfSpriteTheories=None, errorCutoff=ERRORCUTOFF, usePrior=True)]

		print "Experience replay complete."
		for num, sh in enumerate(scoresAndHypotheses):
			print "Theory: {} | Error: {}".format(num, sh[0])
		print ""
		print "{} survived".format(len(scoresAndHypotheses))

		return scoresAndHypotheses, scoreAndTheoryTuples

	def regroundOrNot(self, step_number, predictedEnvs, hypothesis):
		## Returns predictionError=True/False, regroundForKillerTypes=True/False, regroundForKillerTypes=True/False
		## NOTE: If predictionError=True, we aren't evaluating regroundForX
		## because we end up replanning no matter what.

		if not predictedEnvs:
			print "warning! empty predictedEnvs in regroundOrNot!"
			return False , False , False

		matchedEnvs, la, lb = matchEnvs(self.rle, predictedEnvs[step_number+1])

		if la:
			return True, False, False
		if lb:
			return True, False, False
		for match in matchedEnvs:
			if match[2]!=0:
				return True, False, False
		if self.reground_for_killer_types or self.reground_for_stochastic_types:
			killer_colors = [hypothesis.classes[killerType][0].colorName for killerType in hypothesis.killerTypes if \
					any([t in str(hypothesis.classes[killerType][0].vgdlType) for t in ['Random', 'Chaser', 'Missile']])]
			random_colors = [hypothesis.classes[c][0].colorName for c in hypothesis.classes if 'Random' in str(hypothesis.classes[c][0].vgdlType)]
			both_colors = list(set(killer_colors+random_colors))
			avatar_color = hypothesis.classes['avatar'][0].colorName
			avatar_sprite = self.rle._game.observation['trackedObjects'][avatar_color][0] if self.rle._game.observation['trackedObjects'][avatar_color] else None

			if avatar_sprite:
				for c in both_colors:
					if self.rle._game.observation['trackedObjects'][c]:
						for sprite in self.rle._game.observation['trackedObjects'][c]:
							if manhattanDist2(avatar_sprite, sprite)<self.safeDistance:
								regroundForKillerTypes = c in killer_colors
								regroundForStochasticTypes = c in random_colors
								return False, regroundForKillerTypes, regroundForStochasticTypes
		return False, False, False

	def propagateMissileOrientationBackwards(self, envReal, episode_num):
		## If we learn anything about orientation in this step for a sprite that was created in a previous step,
		## go back in time and assign that orientation to the previous steps that sprite was in. This is so that when you set the state to what you 
		## remember from the past, you can incorporate this knowledge.
		orientedSprites = [s for s in [item for sublist in envReal._game.observation['trackedObjects'].values() for item in sublist] if s.firstorientation]
		if orientedSprites:
			for i in list(reversed(range(len(self.rleHistory[episode_num])-1))):
				allSprites = [s for s in [item for sublist in self.rleHistory[episode_num][i]._game.observation['trackedObjects'].values() \
						for item in sublist]]
				madeChange = False
				for sprite in orientedSprites:
					matchedSprite = [s for s in allSprites if s==sprite]
					if matchedSprite and not matchedSprite[0].firstorientation:
						matchedSprite[0].firstorientation = sprite.firstorientation
						madeChange = True
				## Stop going back in time once you've gone to a timestep where you haven't been able to apply your knowledge from this step
				## (that is, when none of the sprites you learned about exist)
				if not madeChange:
					break		

	def executeStep(self, episode_num, rleHistories, actionHistories, action, hypotheses):

		theoryRLEs = VrleInitPhase(hypotheses, self.rle)
		envRealPrev = self.fastcopy(self.rle)
		actionHistories[episode_num].append(action)
		self.rle.step(action)
		envReal = self.fastcopy(self.rle)
		if sum([len(episode) for episode in rleHistories]) <= OBSERVATION_PERIOD_LENGTH:
			# have to save extra info since we deal with the error maps after the step they occur
			copyGameInferenceInfo(envReal, self.rle)

		hypotheses = self.manageNewObjects(episode_num, hypotheses, envRealPrev, action)

		## We are passing the real environment, but experienceReplay filters that rle through the processFrame function (via matchEnvs()).
		self.rleHistory[episode_num].append(envReal)
		self.propagateMissileOrientationBackwards(envReal, episode_num)

		_, new_sprites, _ = matchEnvs(envReal, envRealPrev)
		self.rle._game.sprite_appearances = new_sprites

		print ""
		print keyPresses[action]
		print self.rle.show(color='blue')

		print "evaluating {} old theories and proposing new ones".format(len(theoryRLEs))
		updateTerminations(self.rle, hypotheses, addNoveltyRules=False)
		
		## TODO: If you ever want to not always run testAndExpand, you should implement
		## whatever check you need here. Also, keep track of the scores corresponding to each hypothesis
		## so that you can return those when you don't do everything below the next 5 lines.
		newTheories = []
		for num, env in enumerate(theoryRLEs):
			theories = testAndExpand(env, hypotheses[num], action, self.rle, envRealPrev, self.rleHistory, \
					self.actionHistory, episode_num)
			newTheories.extend(theories)
		newTheories = list(set(newTheories))

		self.allTheories.extend(newTheories)
		print ""
		print "Tested and expanded {} theories to produce {} child theories".format(len(theoryRLEs), len(newTheories))

		if newTheories:
			# embed()
			# t1 = time.time()
			bestScoresAndHypotheses, scoreAndTheoryTuples = self.scoreAndFilterTheories(newTheories, episode_num)
			# print time.time()-t1
			# embed()

			if len(bestScoresAndHypotheses) == 0:	
				print "***** WARNING ***** 0 hypotheses survived filter ***** TRYING AGAIN *****"
				# print "Addressing remaining error maps for {} theories".format(len(newTheories))
				# embed()

				newerTheories = []

				if self.assumeZeroErrorTheoryExists:
					self.assumeZeroErrorTheoryExists = False
					# redo testAndExpand, but take your time and do it thoroughly
					for num, env in enumerate(theoryRLEs):
						theories = testAndExpand(env, hypotheses[num], action, self.rle, envRealPrev, self.rleHistory, \
								self.actionHistory, episode_num)
						newerTheories.extend(theories)
					newerTheories = list(set(newerTheories))

				if len(newerTheories) == 0:
					print "***** WARNING ***** still no hypotheses surviving filter when assumeZeroErrorTheoryExists is False, trying again"
					for t in newTheories:
						theories = addressRemainingErrorMaps(t, envRealPrev, self.rle, action, self.rleHistory, self.actionHistory)
						newerTheories.extend(theories)
					newerTheories = list(set(newerTheories))

				self.allTheories.extend(newerTheories)

				bestScoresAndHypotheses, scoreAndTheoryTuples = self.scoreAndFilterTheories(newerTheories, episode_num)
				for s , h in bestScoresAndHypotheses:
					h.dryingPaint = set()
			if len(bestScoresAndHypotheses) == 0:
				print "second attempt failed, 0 theories survived filter"
				embed()

		else:
			print "WARNING: You are returning 'bestScoresAndHypotheses' but these scores are fake and are\
					really the result of not scoring theories for the beginning observation period."
			bestScoresAndHypotheses = [(0.0, h) for h in hypotheses]

		self.statesEncountered.append(self.rle._game.getFullState())
		self.rle._game.sprite_appearances = []

		return bestScoresAndHypotheses

########################################################################
######## Other initialization METHODS                			########
########################################################################



########################################################################
######## RLE INITIALIZATION AND STATE-SETTING METHODS 			########
########################################################################

def setSpriteState(sprite, matchingSprite, hypothesis):

	if not matchingSprite:
		print "WARNING: didn't find matching sprite in setSpriteState; this shouldn't happen"
		embed()

	sprite.rect 		= pygame.Rect(matchingSprite.rect.left, matchingSprite.rect.top, matchingSprite.rect.width, matchingSprite.rect.height)
	sprite.lastrect 	= pygame.Rect(matchingSprite.lastrect.left, matchingSprite.lastrect.top, matchingSprite.lastrect.width, matchingSprite.lastrect.height)
	sprite.lastmove 	= matchingSprite.lastmove
	sprite.ID 			= matchingSprite.ID
	sprite.resources 	= defaultdict(int)
	for rcolor in matchingSprite.inventory.keys():
		if rcolor not in hypothesis.spriteObjects:
			continue
			# print 'in setSpriteState: next line is going to crash'
			# embed()
			# maybe do sprite induction here on purple?
		sprite.resources[hypothesis.spriteObjects[rcolor].className] = matchingSprite.inventory[rcolor][0]

	# in VGDL, only things which move passively have an orientation that isn't (0,0)
	if hypothesis.spriteObjects[matchingSprite.colorName].vgdlType == Missile:
		## Setting the Missile orientation to be consistent with the theory only makes sense for gridphysics games,
		## because in continuous games the orientation of a missile that is initially DOWN can change to
		## anything as a function of bounces. So doing it as below is actually ideal.
		# slight modification: if it's the missile's first move, orientation is dictated by the theory
		className = hypothesis.spriteObjects[matchingSprite.colorName].className
		classOrientation = hypothesis.classes[className][0].args['orientation']
		orientation = classOrientation if matchingSprite.lastmove <= 0 else matchingSprite.lastDisplacement

		## WrapAround rule conflicts with the normal way of setting sprite orientation. If we have this rule, just go with the prior
		## about a sprite's orientation.
		wrapAroundApplies = any([rule.interaction=='wrapAround' and rule.slot1==className for rule in hypothesis.interactionSet])
		if orientation == (0,0) or wrapAroundApplies:
			orientation = matchingSprite.firstorientation if matchingSprite.firstorientation else hypothesis.spriteObjects[matchingSprite.colorName].args['orientation']
		sprite.orientation = orientation

	else:
		## Convoluted logic. Some avatars have arrows drawn on them; for those we are letting ourselves 'see' the arrow.
		## This makes their orientation have some non-zero value. But if we're considering a hypothesis that 
		## we're something that doesn't passively update its position as a function of its orientation, we don't want to
		## set its orientation to this 'seen' value.
		unOrientedAvatars = ['Moving', 'Flack', 'Horizontal', 'Vertical']
		if all([avatarType not in str(hypothesis.spriteObjects[sprite.colorName].vgdlType) for avatarType in unOrientedAvatars]):
			sprite.orientation = matchingSprite.orientation
		else:
			sprite.orientation = (0,0)

	## Other aspects of state to potentially transfer
	# sprite.jumping = ccopy(matchingSprite.jumping)
	# sprite.wait_step = ccopy(matchingSprite.wait_step)
	# sprite.rope = ccopy(matchingSprite.rope)
	# sprite.gravity = ccopy(matchingSprite.gravity)
	# sprite.last_rope = ccopy(matchingSprite.last_rope)
	# sprite.last_gravity = ccopy(matchingSprite.last_gravity)
	# sprite.last_vy = ccopy(matchingSprite.last_vy)
	# sprite.speed = ccopy(matchingSprite.speed)
	return

def setVrleState(rle, Vrle, hypothesis, makeInitialVrle=False, debug=False):
	## Sets positions of objects in Vrle to what they were in the rle. Bypasses clunky VGDL level description.

	## This is now playing a dual role. Instead of building in theory-laden perception, we
	## are passing that functionality here. For example: processFrame() will never update the orientation of a sprite
	## but will remember its last displacement. When we set a state here, we can interpret that previous
	## observed displacement as a function of our hypothesis. E.g., if our last observed displacement for a
	## RotatingAvatar was UP but we know that our last action was RIGHT, setVrleState() can 'infer' that
	## this means the RotatingAvatar's orientation is actually RIGHT now, and can set it to that.
	## IMPORTANT: This means that each sprite's orientation will only be correct if this function is called between each time-step.
	if debug:
		print "in setVrleState"
		embed()
	if not makeInitialVrle:
		tmp_kill_list = []
		# note: there should at this point be no mismatch between the keys
		for classKey in Vrle._game.sprite_groups:
			if classKey=='wall':
				continue
			color = hypothesis.classes[classKey][0].colorName
			# first make the new (vrle) env have the correct number of each thing
			vrleSpriteCount = len(Vrle._game.sprite_groups[classKey]) - len([s for s in Vrle._game.kill_list if s.colorName==color])
			rleSpriteCount = len(rle._game.observation['trackedObjects'][color])

			if vrleSpriteCount > rleSpriteCount:
				# just delete extraneous ones from the end by adding them to the kill_list
				tmp_kill_list.extend(Vrle._game.sprite_groups[classKey][rleSpriteCount:])
			elif vrleSpriteCount < rleSpriteCount:
				# have to duplicate Vrle sprites so we have enough to copy all the rle sprites into
				try:
					## Make as many new sprites as you need and put them at (0,0); we'll set their position and state below.
					[Vrle._game._createSprite([classKey], (0,0)) for n in range(rleSpriteCount-vrleSpriteCount)]
				except:
					print "problem in setVrleState"
					embed()
			# Now copy over sprite state (if we have any left of that type)
			if not Vrle._game.sprite_groups[classKey]:
				continue

			for i in range(min(len(Vrle._game.sprite_groups[classKey]), rleSpriteCount)):
				setSpriteState(Vrle._game.sprite_groups[classKey][i], rle._game.observation['trackedObjects'][color][i], hypothesis)

		Vrle._game.kill_list = tmp_kill_list
		Vrle._game._eventHandling(UNOBSERVABLE_PREDICATES)

	Vrle._game.time = int(rle._game.time)
	Vrle._game.score = int(rle._game.score)
	Vrle._game.observation = buildTracker(Vrle)
	Vrle._game.observation['lastscore'] = rle._game.observation['lastscore']

	# Make sure that all_objects has the same IDs as the parent RLE.
	Vrle._game.all_objects = Vrle._game.getAllObjects()
	return

def initializeVrle(hypothesis, stateToSet, theoryRLE=None, makeInitialVrle=False, writeFile=False, debug=False):

	## World in agent's mind given 'hypothesis', including object goal
	gameString, levelString, symbolDict = writeTheoryToTxt(stateToSet, hypothesis,\
		 "./examples/gridphysics/theorytest.py", writeFile=writeFile, addAllObjects=makeInitialVrle)

	try:
		# Vrle = theoryRLE if theoryRLE else createMindEnv(gameString, levelString, output=False)
		Vrle = createMindEnv(gameString, levelString, output=False)
	except:
		print "in initializeVrle"
		embed()
	Vrle._game.colorToClassDict = {k:v.className for k,v in hypothesis.spriteObjects.items()}
	if makeInitialVrle:
		for k,v in Vrle._game.sprite_groups.iteritems():
			if v:
				Vrle._game.extra_sprites[k] = copy.deepcopy(v[0])

	## Don't do any of the rest if we have an ungrammatical hypothesis caused by num(avatars)>1.
	if len(stateToSet._game.observation['trackedObjects'][hypothesis.classes['avatar'][0].colorName])>1:
		print "Warning. In initializeVrle. Got more than one avatar. Returning None as Vrle."
		# embed()
		Vrle = None
		return Vrle
	
	## Initialize imaginary state to match real state.
	setVrleState(stateToSet, Vrle, hypothesis, makeInitialVrle)

	return Vrle

def convertTheoryToSubgoalTheory(theory):
	T = theory.copy()
	for rule in T.interactionSet:
		if rule.generic and (rule.interaction, rule.slot1, rule.slot2) not in theory.setOfImaginedEffects:
			if 'Avatar' not in str(T.classes[rule.slot1][0].vgdlType) and 'Avatar' not in str(T.classes[rule.slot2][0].vgdlType):
				rule.interaction = 'nothing'
			elif 'Avatar' not in str(T.classes[rule.slot1][0].vgdlType):
				rule.interaction = 'killSprite'
	imaginedEffectTuples = set([(eff[1], eff[2]) for eff in theory.setOfImaginedEffects])
	T.terminationSet = [rule for rule in T.terminationSet if rule.ruleType!='NoveltyRule' or (rule.termination.s1, rule.termination.s2) not in imaginedEffectTuples]

	return T

def VrleInitPhase(hypotheses, stateToSet, theoryRLEs=None, makeInitialVrle=False):
	## Initialize multiple VRLEs, each corresponding to one hypothesis in theories
	## Set their state to that of the provided RLE
	realVRLEs = []
	for num, hypothesis in enumerate(hypotheses):
		realVRLE = initializeVrle(hypothesis, stateToSet, theoryRLEs[num] if theoryRLEs else None, makeInitialVrle=makeInitialVrle, writeFile=True)
		realVRLEs.append(realVRLE)
	return realVRLEs

def findNearestSprite(sprite, spriteList):
	## returns the sprites in spriteList whose locations best match the location of sprite.
	if spriteList == []:
		return None
	else:
		minDist = float('inf')
		nearestSprites = []
		for x in spriteList:
			dist = abs(x.rect.left-sprite.rect.left)+abs(x.rect.top-sprite.rect.top)
			if dist < minDist:
				nearestSprites = [x]
				minDist = dist
			elif dist==minDist:
				nearestSprites.append(x)
		return nearestSprites

def findNearestSprites(sprite, spriteList, dist_function=manhattanDist2, skip_self=False):
	## returns a list of closest sprites where the distances are all equal
	if not spriteList:
		return []

	dist_map = defaultdict(lambda: [])
	min_dist = float('inf')
	for s in spriteList:
		if s == sprite and skip_self: continue
		dist = dist_function(s, sprite)
		if dist <= min_dist:
			min_dist = dist
			dist_map[dist].append(s)
	return dist_map[min_dist]


########################################################################
######## ERROR SIGNAL AND STATE-COMPARISON METHODS 				########
########################################################################


## Function generating penalty and error map
def errorSignal(envA, envB, theory, envPrev, p_dist=1, p_speed=1, p_miss=10, p_score=1, targetColor=None, penalty_only=False, earlyStopping=False):
	"""
	envA: hypothetical environment
	envB: real environment
	theory: corresponds to hypothetical
	p_dist: distance penalty per grid point
	p_speed: pentalty for distances arising from wrong speed
	p_miss: penalty for missing or additional sprite
	p_score: penalty for getting the score wrong
	penalty_only: return penalty, [] (empty list instead or errorMap)

	Calculates d_theory(envA, envB): distance between the states of the environments
	using the ontology of the supplied theory.

	Also returns errorMap, a dict that contains
	keys: (class1, class2). values: a diagnostic error signal
	"""

	# grid spacing
	d = 30.
	
	# likelihood version
	e_dist 			= 1e-10
	e_inventory 	= 1e-10
	e_disappearance = 1e-10
	e_score			= 1e-10

	# Initialization
	total_penalty = 0.
	errorMap = []

	## Check for an ungrammatical theory.
	if envA is None:
		print "Warning: got ungrammatical theory"
		e = errorMapEntry()
		e.diagnosis.append('ungrammatical theory')
		e.targetToken = None
		e.targetClass = None
		e.targetColor = None
		errorMap.append(e)
		total_penalty = 1. #likelihood version
		return total_penalty, errorMap

	# Match sprites in environments and get sprites that couldn't be matched
	matched_sprites, lonely_sprites_envA, lonely_sprites_envB = matchEnvs(envA, envB)

	if targetColor:
		try:
			matched_sprites = [m for m in matched_sprites if m[0].colorName == targetColor]
			lonely_sprites_envA = [s for s in lonely_sprites_envA if s.colorName == targetColor]
			lonely_sprites_envB = [s for s in lonely_sprites_envB if s.colorName == targetColor]

		except:
			print "targetClass filter in errorSignal failed"
			embed()

	## Penalize distance and additional/missing sprites
	for t in matched_sprites:
		
		## Flag used to skip errorMap creation if we're looking at the behavior of a
		## stochastic sprite whose observed position was within the space of possible positions
		## Prevents trying to fix the same problem over and over.
		reportError = True
		
		sA, sB = t[0], t[1] #sprites in envA, envB      
		dist = t[2] #distance to sprite in envB
		sPrev, dist_ts = find_sPrev(sB, envB, envPrev) #Previous location of our sprite

		sA_type = theory.spriteObjects[sA.colorName].vgdlType
		sA_class = theory.spriteObjects[sA.colorName].className
		
		## Calculate inventory penalty for the sprite
		inventory_penalty = 0
		keys = list(set(t[0].inventory.keys()+t[1].inventory.keys()))
		for k in keys:
			t0_k = t[0].inventory[k] if k in t[0].inventory.keys() else (0,0)
			t1_k = t[1].inventory[k] if k in t[1].inventory.keys() else (0,0)
			inventory_penalty += abs(t0_k[0]-t1_k[0])
		total_penalty += np.log((e_inventory)**inventory_penalty) #likelihood


		if dist>0 and 'flipDirection' in [r.interaction for r in theory.interactionSet if r.slot1==sA_class]:
			neighbors_prev = neighboringSprites(envPrev, sPrev,0)
			if neighbors_prev:
				neighbor_class = theory.spriteObjects[neighbors_prev[0].colorName].className
				if 'flipDirection' in [r.interaction for r in theory.interactionSet if r.slot1==sA_class and r.slot2==neighbor_class]:
					positionOptions = [(sPrev.rect.left-d, sPrev.rect.top), (sPrev.rect.left+d, sPrev.rect.top), \
							(sPrev.rect.left, sPrev.rect.top-d),(sPrev.rect.left, sPrev.rect.top+d)]
					if (sB.rect.left, sB.rect.top) in positionOptions:
						# multiply by len(positionOptions) because really this is less probable than reverseDirection if both have "zero" error
						total_penalty += np.log(1.-e_dist*len(positionOptions))
						continue
		## If a teleport event has taken place
		if dist>0 and 'teleportToExit' in [r.interaction for r in theory.interactionSet if r.slot1==sA_class]:
			sA_class = theory.spriteObjects[sA.colorName].className
			teleportEntryColors = [theory.classes[r.slot2][0].colorName for r in theory.interactionSet if r.interaction=='teleportToExit' and r.slot1==sA_class]
			teleportExitClasses = [theory.spriteObjects[colorName].args['stype'] for colorName in teleportEntryColors]
			teleportExitColors = [theory.classes[c][0].colorName for c in teleportExitClasses]
			teleportEntryLocs, teleportExitLocs = [],[]
			for entryType in teleportEntryColors:
				teleportEntryLocs.extend([s for s in envPrev._game.observation['trackedObjects'][entryType]])
			for exitType in teleportExitColors:
				teleportExitLocs.extend([(s.rect.left, s.rect.top) for s in envB._game.observation['trackedObjects'][exitType]])
			## If we've ended up at a purported exit and we could have gotten to an entry with a single step,
			## consider that we teleported and don't penalize the distance any other way.
			if (sB.rect.left, sB.rect.top) in teleportExitLocs and any([manhattanDist2(sPrev,s)<=sPrev.speed for s in teleportEntryLocs]):
				total_penalty += np.log(1.-e_dist)
				continue

		if any([stochasticType in str(sA_type) for stochasticType in ['Random', 'Chaser']]):
			
			# If RandomNPC: compare sB position to where it could have been given the hypothetical speed and random direction
			if 'Random' in str(sA_type):
				if 'speed' in theory.spriteObjects[sA.colorName].args.keys():
					sA_speed = theory.spriteObjects[sA.colorName].args['speed']
				elif 'speed' in theory.spriteObjects[sA.colorName].__dict__.keys():
					sA_speed = theory.spriteObjects[sA.colorName].speed
				else:
					## this only happens when you initialize the real theory for testing but haven't explicitly set the speed
					## in the VGDL description
					sA_speed = 1

				if sPrev is None:
					continue
				xB = sB.rect.left/d
				yB = sB.rect.top/d
				xPrev = sPrev.rect.left/d
				yPrev = sPrev.rect.top/d

				## if the sprite was allowed to move according to the theory
				if sA.lastmove%theory.spriteObjects[sA.colorName].args['cooldown']==0:
					positionOptions = [(xPrev, yPrev), (xPrev+sA_speed, yPrev), (xPrev-sA_speed, yPrev), (xPrev, yPrev+sA_speed), (xPrev, yPrev-sA_speed)]
				else:
					positionOptions = [(xPrev, yPrev)]
				
				if (xB,yB) in positionOptions:
					# total_penalty += np.log(1.-e_dist)
					total_penalty += np.log(1.-e_dist*len(positionOptions))
					reportError = False
				else:
					total_penalty += np.log(e_dist)
					errs = diagnosePosMismatch(sA, sB, sPrev, envA, envB, envPrev, dist_ts, theory)
			
			elif 'Chaser' in str(sA_type):

				xB = sB.rect.left/d
				yB = sB.rect.top/d
				if sPrev is None:
					continue

				stype = theory.spriteObjects[sA.colorName].args['stype']
				sA.stype = theory.classes[stype][0].colorName
				sA.fleeing = theory.spriteObjects[sA.colorName].args['fleeing']
				
				## the lastmove+1 is becuase of the *very* weird nature of the update function for Chaser.
				if (sA.lastmove+1)%theory.spriteObjects[sA.colorName].args['cooldown']==0:
					closestTargets = findChaserOptions(sA, sPrev, envPrev._game, fleeing=sA.fleeing)
					if not closestTargets:
						closestTargets = [(sPrev.rect.left/d, sPrev.rect.top/d)]
				else:
					closestTargets = [(sPrev.rect.left/d, sPrev.rect.top/d)]

				del sA.stype
				del sA.fleeing

				if (xB, yB) in closestTargets:
					# total_penalty += np.log(1.-e_dist)
					total_penalty += np.log(1.-e_dist*len(closestTargets))
					reportError = False
				else:
					total_penalty += np.log(e_dist)
					errs = diagnosePosMismatch(sA, sB, sPrev, envA, envB, envPrev, dist_ts, theory)

		# All of the other types are deterministic
		else:
			if t[2]==0.:
				total_penalty += np.log(1.-e_dist)
				reportError = False
			else:
				total_penalty += np.log(e_dist)

		try:
			if reportError:
				# Determine errorMapEntry object for position mismatch problem
				errs = diagnosePosMismatch(sA, sB, sPrev, envA, envB, envPrev, dist_ts, theory)
				errorMap.extend(errs)
		except:
			print "reportError problem"
			embed()

		if penalty_only and earlyStopping and 1.-np.exp(total_penalty) > 0.0001:
			# print "CUT OFF"
			return 1. , []

	# Missing/additional/transformation penalty
	total_penalty += np.log((e_disappearance)**( len(lonely_sprites_envA) + len(lonely_sprites_envB) )) #likelihood

	if envA._game.observation['score'] != envB._game.observation['score']:
		total_penalty += np.log(e_score)

	total_penalty = 1.-np.exp(total_penalty)
	if penalty_only:
		return total_penalty, []

	### Construct errorMap using previous state ###

	# Case B: Sprite moved in real environment, but we predicted a destruction
	# For this, we check if lonely envB sprite has match in envPrev (and pass to (2) if not)
	appeared_sprites_envB = []
	for sB in lonely_sprites_envB:
		# Find sprite corresponding to sB in previous time step
		sPrev, dist_ts = find_sPrev(sB, envB, envPrev)
		if sPrev == None: #sB has no match in envPrev
			appeared_sprites_envB.append(sB)
			continue 

		## Find erroneously destroyed sA by finding envA sprite closest to sPrev
		candidates_in_killList = [s for s in envA._game.observation['kill_list'] if s.colorName == sPrev.colorName]

		## These are both double-checking things that should have been taken care of better
		## by the sprite matching. But since it's imperfect given our limited knowledge, we're
		## being more thorough.

		if candidates_in_killList == []:
			if not sPrev: #there is no envA sprite where sPrev should have been
				print "empty killList in A, meaning the matching is wrong"
				## You need to figure out what to pass to diagnosePosMismatch for sA, since it
				## doesn't exist.
				embed()
				#appeared_sprites_envB.append(sB)
				continue
		else:
			sA = findNearestSprite(sPrev, candidates_in_killList)[0]
			if manhattanDist2(sA, sPrev)>1 and not sPrev: #there is no envA sprite where sPrev should have been
				## if there was a kill event and an appearance event somewhere far, we should really see this as
				## an appearance
				## Really, you should look at sprite matching better.
				print "manhattanDist2 > 1"
				embed()
				appeared_sprites_envB.append(sB)
				continue
		# Now we are completely sure that sprite in envA has been erroneously removed
		errs = diagnosePosMismatch(sA, sB, sPrev, envA, envB, envPrev, dist_ts, theory)
		errorMap.extend(errs)

	# 2) Unexpected destruction/appearance/transformation
	# 2.1) Transformation
	for iA,sA in enumerate(lonely_sprites_envA):
		for iB,sB in enumerate(appeared_sprites_envB):
			if manhattanDist2(sA, sB)<=2:
				e = errorMapEntry()
				e.diagnosis.append('transformation')
				e.targetToken = sA
				e.targetClass = sA.colorName
				e.targetColor = sA.colorName
				# Find sprite corresponding to sB in previous time step
				color = sB.colorName
				sB.colorName = sA.colorName
				matched_ts, _, _ = matchEnvs(envB, envPrev) #matches real env across timestep
				sB.colorName = color
				sPrev = [matched_ts[i][1] for i in range(len(matched_ts)) if matched_ts[i][0] == sB]

				if sPrev == []: #This was an appearance, pass to (2.3) below
					continue
				else: #This was indeed a transformation
					# print "WARNING: Found unexpected transformation"
					sPrev = sPrev[0]
					# Find neighbors of target sprite in the previous time step
					neighbors_prev = neighboringSpritesColors(envPrev, sPrev)
					# Write potential interaction pairs to error map entry
					for className in neighbors_prev:
						e.intPairs.append( (theory.spriteObjects[sPrev.colorName].className,className) )
					errorMap.append(e)

					# Remove transformed-sprite-pair from respective lists
					lonely_sprites_envA.pop(iA)
					appeared_sprites_envB.pop(iB)

	# 2.2) Destruction
	for sA in lonely_sprites_envA: #sA should have been destroyed
		e = errorMapEntry()
		e.targetClass = sA.colorName
		e.targetColor = sA.colorName
		candidates_in_killList = [s for s in envB._game.observation['kill_list'] if s.colorName == sA.colorName]
		sB = findNearestSprite(sA, candidates_in_killList)

		if not sB:
			print "WARNING: No target and interaction pair found in object destruction. You have not implemented anything resulting from that diagnosis."
			e.diagnosis.append('objectDidNotAppear')
			e.targetToken = None
			e.intPairs = []
			errorMap.append(e)
			continue
		else:
			sB = sB[0]
			e.diagnosis.append('objectDestruction')
			e.targetToken = sB
			# Find the sprite that was destroyed in envB from the kill_list
			# Find neighbors of target sprite in the previous time step
			sPrev = sB #sprite was destroyed but hasn't moved

		neighbors_prev = neighboringSpritesColors(envPrev, sPrev)
		neighbors_prev = [c for c in neighbors_prev if c!=sA.colorName]
		# Write potential interaction pairs to error map entry
		for className in neighbors_prev:
			e.intPairs.append( (sA.colorName,className) )
		errorMap.append(e)
	# 2.3) Appearance
	for sB in appeared_sprites_envB:
		print "WARNING: Found unexpected appearance"
		e = errorMapEntry()
		e.diagnosis.append('newObjectAppeared')
		e.targetToken = sB

		# Find class of new object by comparing colors, or give 'unknown' if unsuccessful
		sMatch = getObservedSpritesByColor(envA._game, sB.colorName)
		e.targetColor = sB.colorName
		if sMatch == []:
			e.targetClass = 'unknown'
		else:
			e.targetClass = sMatch[0].colorName

		# Find neighbors of target sprite in the real environment (envB) in the current time step -> could have caused appearance
		# And also in the previous time-step.
		# Simultaneously find culprit classes - an overlapping sprite could have launched the sprite due to its class
		
		neighbors_curr_and_prev = neighboringSpritesColors(envB, sB) + neighboringSpritesColors(envPrev, sB)
		nearestSprites = findNearestSprite(sB, [item for sublist in envA._game.observation['trackedObjects'].values() for item in sublist])

		for nearestSprite in nearestSprites:
			if nearestSprite.colorName in neighbors_curr_and_prev:
				e.intPairs.append((e.targetClass, nearestSprite.colorName))

		if not e.intPairs:
			print "Couldn't make intPairs in unexpected appearance case."
		errorMap.append(e)

	# 3) Inventory change
	for t in matched_sprites:
		inventory_penalty = 0
		keys = list(set(t[0].inventory.keys()+t[1].inventory.keys()))
		for k in keys:
			t0_k = t[0].inventory[k] if k in t[0].inventory.keys() else (0,0)
			t1_k = t[1].inventory[k] if k in t[1].inventory.keys() else (0,0)
			inventory_penalty += abs(t0_k[0]-t1_k[0])

		if inventory_penalty > 0:
			e = errorMapEntry()
			e.diagnosis.append('inventoryChange')
			sA, sB = t[0], t[1]
			e.targetToken = sB
			e.targetClass = sA.colorName
			e.targetColor = sB.colorName
			sPrev, dist_ts = find_sPrev(sB, envB, envPrev)
			neighbors_prev = neighboringSpritesColors(envPrev, sPrev)
			e.intPairs.extend([(e.targetClass, n) for n in neighbors_prev])
			errorMap.append(e)

	for sB in lonely_sprites_envB:
		inventory_penalty = 0
		sPrev, dist_ts = find_sPrev(sB, envB, envPrev)
		if not sPrev:
			continue
		keys = list(set(sB.inventory.keys()+sPrev.inventory.keys()))
		for k in keys:
			sB_k = sB.inventory[k] if k in sB.inventory.keys() else (0,0)
			sPrev_k = sPrev.inventory[k] if k in sPrev.inventory.keys() else (0,0)
			inventory_penalty += abs(sB_k[0]-sPrev_k[0])

		if inventory_penalty > 0:
			e = errorMapEntry()
			e.diagnosis.append('inventoryChange')
			e.targetToken = sB
			e.targetClass = sB.colorName
			e.targetColor = sB.colorName
			neighbors_prev = neighboringSpritesColors(envPrev, sPrev)
			e.intPairs.extend([(e.targetClass, n) for n in neighbors_prev])
			errorMap.append(e)

	# 4) Score change
	if envA._game.observation['score'] != envB._game.observation['score']:
		e = errorMapEntry()
		e.diagnosis.append('scoreChange')
		avatar_color = theory.classes['avatar'][0].colorName
		sA = envA._game.observation['trackedObjects'][avatar_color][0]
		sB = envB._game.observation['trackedObjects'][avatar_color][0]
		e.targetToken = sA
		e.targetClass = sA.colorName
		e.targetColor = sA.colorName
		sPrev, dist_ts = find_sPrev(sB, envB, envPrev)
		neighbors_prev = neighboringSpritesColors(envPrev, sPrev)
		e.intPairs.extend([(e.targetClass, n) for n in neighbors_prev])
		errorMap.append(e)

	## Share information across errorMap items and make a unique list
	# if len(errorMap) > 1:
	if errorMap:
		diagnosis_class_pairs = list(set([tuple(e.diagnosis+[e.targetClass]) for e in errorMap]))
		for dcp in diagnosis_class_pairs:
			relatedErrorMaps = [e for e in errorMap if tuple(e.diagnosis)==dcp[0:-1] and e.targetClass==dcp[-1]]
			# int_pairs = [item for sublist in [e.intPairs for e in errorMap if e.diagnosis[0] == dcp[0] and e.targetClass == dcp[1]] for item in sublist]
			int_pairs = [item for sublist in [e.intPairs for e in relatedErrorMaps] for item in sublist]
			int_pairs = list(set(int_pairs))
			targetTokens = list(set([e.targetToken for e in relatedErrorMaps]))
			## give int_pairs to each matching errorMap item.
			for e in errorMap:
				try:
					if tuple(e.diagnosis) == dcp[0:-1] and e.targetClass == dcp[-1]:
						e.intPairs = int_pairs
						e.targetTokens = targetTokens
				except:
					print "problem in errorMap consolidation in errorSignal"
					embed()
		lst = [errorMap[0]]
		for e in errorMap[1:]:
			if all([not(e.diagnosis == l.diagnosis and e.targetClass == l.targetClass) for l in lst]):
			# if all([not(e.diagnosis == l.diagnosis and e.targetClass == l.targetClass and e.targetToken == l.targetToken) for l in lst]):
				lst.append(e)
		# print "filtered errorMap"
		# embed()
		errorMap = lst

		## Sort so that you fix errors involving any new classes first when you build theories.
		errorMap = sorted(errorMap, key=lambda x: (x.targetClass!='unknown', 'inventoryChange' not in x.diagnosis) )

	# print "at end of errorSignal"
	# embed()
	return total_penalty, errorMap

def neighboringSpritesColors(env, sprite):
	"""
	returns colors of the neighboring sprites in env
	"""
	# Find potential interaction partners: neighboring sprites in previous step
	if sprite is None:
		return []
	neighbors = neighboringSprites(env, sprite)
	neighbors = list(set([n.colorName for n in neighbors]))
	return neighbors

def neighboringSprites(env, sprite, distanceThreshold=2):
	"""
	Function to find neighbors of sprite in the given environment (should be where the sprite came from)
	"""
	# Find potential interaction partners: neighboring sprites in previous step
	all_sprites = [item for sublist in env._game.observation['trackedObjects'].values() for item in sublist]

	# Neighbors of problematic sprite in real world in previous time step
	neighbors = [s for s in all_sprites if manhattanDist2(s, sprite)<=np.sqrt(distanceThreshold) and s!=sprite]
	return neighbors

def find_sPrev(sB, envB, envPrev):
	"""
	Find sprite in envPrev (previous environment) corresponding to a sprite in
	envB (current environment), and the distance that the sprite has traveled
	in the time step
	"""
	matched_ts, _, _ = matchEnvs(envB, envPrev) #matches real env across timestep
	dist_ts = [matched_ts[i][2] for i in range(len(matched_ts)) if matched_ts[i][0] == sB] #distance that sB has moved over timestep
	sPrev = [matched_ts[i][1] for i in range(len(matched_ts)) if matched_ts[i][0] == sB] #sB in previous step
	if sPrev == []:
		sPrev = None
		dist_ts = None
	else:
		sPrev = sPrev[0]
		dist_ts = dist_ts[0]
	return sPrev, dist_ts

def diagnosePosMismatch(sA, sB, sPrev, envA, envB, envPrev, dist_ts, theory):
	"""
	Returns errorMapEntry object containing the position mismatch error
	"""

	# Step through sub-problems
	e = errorMapEntry()
	e.targetToken = sB
	e.targetClass = sB.colorName
	e.targetColor = sB.colorName

	errorMaps = [e]
	
	# Find neighbors of target sprite in the previous time step
	neighbors_prev = neighboringSpritesColors(envPrev, sPrev)

	# Write potential interaction pairs to error map entry
	for className in neighbors_prev:
		e.intPairs.append( (sA.colorName,className) )
	# Determine mininum distance to neighbors in current real env -> to distinguish unexpectedPosition and unexpectedOverlap
	all_sprites_envB = [item for sublist in envB._game.observation['trackedObjects'].values() for item in sublist]
	nearest_sprite = findNearestSprite(sB, [s for s in all_sprites_envB if (s!=sB)])[0]

	nearest_dist = manhattanDist2(sB, nearest_sprite)
	
	# Determine orientation in current and previous step -> to detect orientation change
	if sB and sPrev:
		if theory.spriteObjects[sA.colorName].vgdlType in [Missile]:
			oB = sB.lastDisplacement
			oPrev = sPrev.lastDisplacement
		else:
			oB = sB.orientation
			oPrev = sPrev.orientation
	else:
		oB, oPrev = None, None

	## Categorize into sub-problem-class
	# 1.1) noMovement
	if dist_ts == 0:
		e.diagnosis.append('noMovement')
	# 1.2) orientationChange
	if dist_ts!=0 and oB!=None and oB!=oPrev:
		e.diagnosis.append('orientationChange')
	# 1.3) unexpectedPosition
	if dist_ts!=0 and nearest_dist>=1:
		e.diagnosis.append('unexpectedPosition')
	# 1.4) unexpectedOverlap
	if dist_ts!=0 and nearest_dist<1:
		e.diagnosis.append('unexpectedOverlap')
		# find sprite in envA that corresponds to covered sprite in envB
		color = nearest_sprite.colorName
		className_envA = ''
		for k in [key for key in envA._game.observation['trackedObjects'].keys() if envA._game.observation['trackedObjects'][key]]:
			if color == envA._game.observation['trackedObjects'][k][0].colorName:
				className_envA = k
		covered_sprite_envB = findNearestSprite(sB, [item for sublist in envB._game.observation['trackedObjects'].values() for item in sublist if sB!=item])[0]
		if covered_sprite_envB.colorName not in theory.spriteObjects:
			print "diagnosePosMismatch found a new color"
			e1 = errorMapEntry()
			e1.diagnosis.append('newClass')
			e1.targetToken = covered_sprite_envB
			e1.targetClass = 'unknown'
			e1.targetColor = covered_sprite_envB.colorName
			errorMaps.append(e1)
		e.intPairs = [(sA.colorName, covered_sprite_envB.colorName)] #overwrite interaction pair by the overlapping sprite pair
	if dist_ts>2:
		e2 = errorMapEntry()
		e2.targetToken = e.targetToken
		e2.targetClass = e.targetClass
		e2.targetColor = e.targetColor
		dx, dy = abs(sB.rect.left-sPrev.rect.left), abs(sB.rect.top-sPrev.rect.top)
		if (dx+sB.rect.width==envB._game.screensize[0] and dy==0) or (dx==0 and dy+sB.rect.height==envB._game.screensize[1]):
			e2.diagnosis.append('wrapAround')
			e2.intPairs.append((sA.colorName, 'ENDOFSCREEN'))
			pass
		else:
			e2.diagnosis.append('teleport')
			e2.intPairs = [(sA.colorName, n) for n in neighbors_prev]
		errorMaps.append(e2)
	
	# Return list of errorMapEntry objects
	return errorMaps

def matchEnvs(envA, envB):
	all_sprites_envA = [item for sublist in envA._game.observation['trackedObjects'].values() for item in sublist]
	all_sprites_envB = [item for sublist in envB._game.observation['trackedObjects'].values() for item in sublist]

	ID_dict = {}

	matched_sprites, lonely_sprites_envA, lonely_sprites_envB = [],[],[]
	for sprite in all_sprites_envA:
		ID_dict[sprite.ID] = [sprite]

	for sprite in all_sprites_envB:
		if sprite.ID in ID_dict:
			ID_dict[sprite.ID].append(sprite)
		else:
			lonely_sprites_envB.append(sprite)
	for v in ID_dict.values():
		if len(v)==2:
			matched_sprites.append((v[0], v[1], manhattanDist2(v[0], v[1])))
		elif len(v)==1:
			lonely_sprites_envA.append(v[0])

	new_matched_sprites, lonely_sprites_envA, lonely_sprites_envB = resolveUnmatchedSprites(lonely_sprites_envA, lonely_sprites_envB)

	matched_sprites += new_matched_sprites

	return matched_sprites, lonely_sprites_envA, lonely_sprites_envB

def resolveUnmatchedSprites(envA_sprites, envB_sprites, debug=False):
	'''
	Compares environment A to environment B, mapping sprites from A to sprites from B 1 to 1 (if it can)
	by comparing the positions of sprites in A to positions of sprites in B of the same color. 

	Returns mapping that minimizes distance between matching sprites (hopefully?)

	returns:

		the matched sprites as a list of tuples of sprites from A and sprites from B 
	and the manhatten distance between their positions: 
		[(s_A1, s_B1, d), (s_A2, s_A3, d), ...]

		the list of "lonely sprites" in A that don't map to any sprites in A: 
			[s_A5, s_A6, ..]

		the list of "lonely sprites" in B that don't map to any sprites in B: 
			[s_A7, s_A8, ..]
	'''

	# Start creating our data structures.
	matched_sprites, lonely_sprites_envA, lonely_sprites_envB = [],[],[]

	color_groupsA = defaultdict(lambda : [])
	color_groupsB = defaultdict(lambda : [])
	positions = set()

	# start with greedy algorithm
	# match objects with the same position/color to each other

	# map positions to objects
	pos_groupsA = defaultdict(lambda : [])
	pos_groupsB = defaultdict(lambda : [])

	matched_colors = defaultdict(lambda :LinkedDict())
	unmatchedA = set()
	unmatchedB = set()
	# is there any guarantee for the ordering of the sprites?
	# O(spritesA+spritesB) ~ O(n)
	for sprites, pos_groups, color_groups, unmatched in [(envA_sprites, pos_groupsA, color_groupsA, unmatchedA), 
													 (envB_sprites, pos_groupsB, color_groupsB, unmatchedB)]:
		for sprite in sprites:
			if sprite:
				color = sprite.colorName
				color_groups[color].append(sprite)
			
				pos = sprite.rect.topleft
				pos_groups[pos].append(sprite)
				positions.add(pos)
				unmatched.add(sprite)

	# Greedily matches sprites based on position first AND color
	# O(n^2) (but will likely be O(n) since not many sprites overlap)
	for pos in positions: # O(n)
		for spriteA in pos_groupsA[pos]: # O(max 5ish?)
			for spriteB in pos_groupsB[pos]: # O(max 5ish?)
				color = spriteA.colorName
				if color == spriteB.colorName:
					if matched_colors[color][spriteB]: continue # match already made
					unmatchedA.remove(spriteA)
					unmatchedB.remove(spriteB)
					matched_colors[color][spriteA] = spriteB
					# stop after first match and go on to match next one
					break

	# O(unmatchedA+unmatchedB) ~ O(n)
	unmatched_colorsA, unmatched_colorsB = defaultdict(lambda: set()), defaultdict(lambda: set())
	for unmatched, unmatched_colors in [(unmatchedA, unmatched_colorsA),
										(unmatchedB, unmatched_colorsB)]:
		for s in unmatched:
			unmatched_colors[s.colorName].add(s)

	# while it's still possible to make matches
	unmatched_colors = set(unmatched_colorsA).intersection(set(unmatched_colorsB))
	rematches = {}
	for color in unmatched_colors:
		# get all the sprites with the color in envB
		# and the unmatched sprites with the color in envAh

		pairing_paths = []
		for spriteA in unmatched_colorsA[color]:
			color_groupB = set(color_groupsB[color])
			unmatched_group = set(unmatched_colorsA[color])
			unmatched_group.remove(spriteA)
			match_dict = matched_colors[color]
			pairing_paths += [(0, spriteA, unmatched_group, color_groupB, [])]

		# color_groupB = set(color_groupsB[color])
		# unmatched_group = unmatched_colorsA[color].copy()
		# spriteA = unmatched_group.pop()
		# match_dict = matched_colors[color]
		# pairing_paths = [(0, spriteA, unmatched_group, color_groupB, [])]
		# pop one of the unmatched sprites from the unmatched color_groupA
			
		best_matches = None
		while pairing_paths:
			# rematch sprites until we've tried to match them all

			# BFS - grab last pairing. Sorted in decending order of dist
			sum_dist, spriteA, unmatched_group, color_groupB, matches = heapq.heappop(pairing_paths)

			# if not color_groupB:
			# 	pairing_paths.append((sum_dist, pairs, color_groupB))
			# 	break
			best_matches = matches
			if not (spriteA or unmatched_group):
				break

			if not spriteA:
				spriteA = unmatched_group.pop()

			nearest_sprites = findNearestSprites(spriteA, color_groupB, manhattanDist2)
			for spriteB in nearest_sprites: 
				color_group_copy = color_groupB.copy()
				unmatched_group_copy = unmatched_group.copy()
				matches_copy = matches[:]
				new_sum_dist = sum_dist

				new_spriteA = match_dict[spriteB]
				new_sum_dist += manhattanDist2(spriteA, spriteB)**2

				if new_spriteA:
					new_sum_dist -= manhattanDist2(new_spriteA, spriteB)**2

				color_group_copy.remove(spriteB)
				matches_copy.append((spriteA, spriteB))
				if new_spriteA:
					unmatched_group_copy.add(new_spriteA)

				next_sprite = findNearestSprites(spriteA, unmatched_group_copy)
				if next_sprite:
					new_spriteA = next_sprite[0]
					unmatched_group_copy.remove(next_sprite[0])

				heapq.heappush(pairing_paths, (new_sum_dist, new_spriteA, unmatched_group_copy, color_group_copy, matches_copy))

		if spriteA:
			unmatchedA.add(spriteA)
		for sA, sB in best_matches:
			if sA in unmatchedA:
				unmatchedA.remove(sA)
			if sB in unmatchedB:
				unmatchedB.remove(sB)

			matched_colors[color][sA] = sB

	lonely_sprites_envA = list(unmatchedA)
	lonely_sprites_envB = list(unmatchedB)
	matched_sprites = [(s1, s2, manhattanDist2(s1, s2)) for matched_color in matched_colors.values() for s1, s2 in matched_color.iteritems() ]

	return matched_sprites, lonely_sprites_envA, lonely_sprites_envB




########################################################################
######## EXPERIENCE REPLAY 										########
########################################################################


def subSampleStates(subsamplePercentage, actionsPerIndex, rleHistory):
	## Returns a random subsample of inidces in the rleHistory to test,
	## as well as how many actions per index to test

	## Note to self: it may happen that you sample: 
	## indices = [0,2,10], actionsPerIndex=5, 
	## in which case you'll double-penalize states 2,3,4.

	numStatesToSample = int(ceil(subsamplePercentage*len(rleHistory)))
	indices = list(np.random.choice(len(rleHistory)-1, numStatesToSample, replace=False))
	actionsPerIndex = actionsPerIndex

	return indices, actionsPerIndex

def getSalientStates(rleHistory):
	## make sure you don't sample the last state
	## get actionsPerIndex
	pass

def checkIfStatesAreDifferent(env1, env2):
	if (env1 and not env2) or (env2 and not env1):
		return True
	matchedEnvs, la, lb = matchEnvs(env1, env2)
	if la:
		return True
	if lb:
		return True
	for match in matchedEnvs:
		if match[2]!=0:
			return True
	return False

def singleTheoryExperienceReplay(rleHistory, actionHistory, method, targetColor, displayStates, hypotheses, returnAllErrors=False, assumeZeroErrorTheoryExists=False, errorCutoff=ERRORCUTOFF):

	# if there isn't a theory, it has error 1. (added for multiepisode experienceReplay)
	if not hypotheses[0]:
		return [1.] , set()

	cutoffThreshold = max(1, floor(errorCutoff * len(rleHistory)))+.0001 # aka n strikes, you're out
	if assumeZeroErrorTheoryExists:
		cutoffThreshold = .00001 # aka one strike you're out
	subsamplePercentage = .2
	actionsPerIndex = 2
	setOfImaginedEffects = set()
	cumulative_penalties = []

	if len(hypotheses)>1:
		print "got more than 1 hypothesis in singleTheoryExperienceReplay"
		embed()
	

	## This is only checking if you have replayed that exact sequence
	## but you have definitely replayed a similar shorter sequence. Grab that value and then only modify
	## it by the most recent step.
	## This is still not going to address the fact that when you change a theory you're deleting the whole history
	## How important it it actually to go all the way back and do full replay? As in,
	## How often will a new modification make something old far worse? Maybe this is just completely unnecessary.
	try:
		key = (method, targetColor, rleHistory[0].ID, len(rleHistory))
	except:
		print "key for singleTheoryExperienceReplay failed"
		embed()
	if not displayStates and key in hypotheses[0].experienceReplayRecord:
		return hypotheses[0].experienceReplayRecord[key], hypotheses[0].setOfImaginedEffects

	if method == 'all':
		if displayStates:
			print "Playing replay FORWARD. Default is backwards."
			indices = range(len(rleHistory))
		else:
			indices = list(reversed(range(len(rleHistory)-1)))
			# indices = indices[1:min(5, len(indices))]
			keyForPreviousSequence = (method, targetColor, rleHistory[0].ID, len(rleHistory)-1)
			if keyForPreviousSequence in hypotheses[0].experienceReplayRecord:
				# print "found key for shorter sequence; only testing most recent step"
				# embed()
				indices = indices[0:1]
				prevMeanError = hypotheses[0].experienceReplayRecord[keyForPreviousSequence][0]
				cumulative_penalties = [[prevMeanError] for i in range(len(rleHistory)-2)]
		actionsPerIndex = 1
	elif method == 'oneReplay':
		indices = [0]
		actionsPerIndex = len(actionHistory)
	elif method == 'screenLastStep':
		## Can't screen last step with fewer than two RLEs in history.
		if len(rleHistory)<2:
			actionsPerIndex = 0
			indices = [0]
			print "got screenLastStep on short sequence"
		else:
			indices = [-2]
			actionsPerIndex = 1
	elif method == 'subsample':
		indices, actionsPerIndex = subSampleStates(subsamplePercentage, actionsPerIndex, rleHistory)
	elif method == 'salient':
		indices, actionsPerIndex = getSalientStates(subsamplePercentage, actionsPerIndex, rleHistory)

	initialRLEs = VrleInitPhase(hypotheses, rleHistory[0], makeInitialVrle=True)
	theoryRLEs = [[]]
	for idx in indices:
		## 1. set imagined states to historical states  2. match IDs between real and theory RLEs
		t1 = time.time()

		if sum([c[0] for c in cumulative_penalties])>cutoffThreshold and (assumeZeroErrorTheoryExists or indices.index(idx)>3):
			# this hypothesis is so wrong it's not worth thinking about any more.
			# print "CUT OFF at index {} of {}, max={}".format(idx,indices, max(indices))
			mean_penalties = [1.]
			hypotheses[0].experienceReplayRecord[key] = mean_penalties
			return mean_penalties, setOfImaginedEffects

		# if any([checkIfStatesAreDifferent(tRLE, rleHistory[idx]) for tRLE in theoryRLEs]):
			# print "states were different; making new theoryRLEs"
			# theoryRLEs = VrleInitPhase(hypotheses, rleHistory[idx], theoryRLEs=initialRLEs)
		# else:
			# print "got same states; not making new theoryRLE"
		theoryRLEs = VrleInitPhase(hypotheses, rleHistory[idx], theoryRLEs=initialRLEs)

		## Take a predetermined number of actions starting from idx
		end = min(idx+actionsPerIndex, len(actionHistory))

		if displayStates:
			print "setting state to index {}. Grounding state looks like this:".format(idx)
			print rleHistory[idx].show()
			print "hypothetical states are in blue below; should match the black state above."
			for env in theoryRLEs:
				print env.show(color='blue')

		for n, action in enumerate(actionHistory[idx:end]):
			penalties = []
			if displayStates:
				print "after taking action {}, real state looked like this:".format(keyPresses[action])
				print rleHistory[idx+n+1].show()
			for num, env in enumerate(theoryRLEs):                      

				if env is not None:

					## Store resources the avatar had before the step
					## so we can update the dict of effects we thought we saw while having
					## 1 or lim of each resource
					avatarColor = hypotheses[num].classes['avatar'][0].colorName
					resourceDict = {}
					for k in env._game.sprite_groups['avatar'][0].resources.keys():
						resourceColor = hypotheses[num].classes[k][0].colorName	
						try:			
							resourceDict[k] = env._game.observation['trackedObjects'][avatarColor][0].inventory[resourceColor]
						except:
							print 'resourceDict problem'
							embed()

					imaginedEffects = env.step(action)['effectList']
					for effect in imaginedEffects:
						eff1Class = env._game.all_objects[effect[1]].name if effect[1] in env._game.all_objects else 'EOS'
						eff2Class = env._game.all_objects[effect[2]].name if effect[2] in env._game.all_objects else 'EOS'
						setOfImaginedEffects.add((effect[0], eff1Class, eff2Class))
						setOfImaginedEffects.add((effect[0], eff2Class, eff1Class))

						for k,v in resourceDict.items():
							if v[0]>0:
								setOfImaginedEffects.add((effect[0], eff1Class, eff2Class, k, True, False))
								setOfImaginedEffects.add((effect[0], eff2Class, eff1Class, k, True, False))
							if v[0]==v[1]:
								setOfImaginedEffects.add((effect[0], eff1Class, eff2Class, k, True, True))
								setOfImaginedEffects.add((effect[0], eff2Class, eff1Class, k, True, True))
				try:
					penalty, errorList = errorSignal(env, rleHistory[idx+n+1], hypotheses[num], 
						rleHistory[idx+n], targetColor=targetColor, penalty_only=True, earlyStopping=assumeZeroErrorTheoryExists)
					penalties.append(penalty)
				except:
					print "exception in experienceReplay"
					embed()
				if displayStates:
					print "resulting state incurred a penalty of {} and looks like this:".format(penalty)
					print env.show(color='green')
					embed()
			cumulative_penalties.append(penalties)
	
	if not cumulative_penalties:
		print "Warning: did not run experience replay."
		cumulative_penalties = [[0]*len(hypotheses)]

	cumulative_penalties = np.array(cumulative_penalties)
	mean_penalties = np.mean(cumulative_penalties, axis=0)
	hypotheses[0].experienceReplayRecord[key] = mean_penalties
	return mean_penalties, setOfImaginedEffects
	
def experienceReplay(hypotheses, rleHistory, actionHistory, method='all', targetColor=None, displayStates=False, displayTheories=False, assumeZeroErrorTheoryExists=False, errorCutoff=ERRORCUTOFF):
	# if len(hypotheses)>10:
		# print "Running experience replay on {} theories and {} time-steps".format(len(hypotheses), len(rleHistory))

	t1 = time.time()
	results, imaginedEffects = [], []
	# if len(hypotheses)>100:
		# print ">100 hypotheses"
		# embed()
	itr = trange(len(hypotheses)) if len(hypotheses) > 20 else range(len(hypotheses))
	for num in itr:
		h = hypotheses[num]
		if displayTheories:
			print "running experienceReplay on {}:".format(num)
			h.display()
		mean_penalties, setOfImaginedEffects = \
				singleTheoryExperienceReplay(rleHistory, actionHistory, method, targetColor, displayStates, [h],  assumeZeroErrorTheoryExists=assumeZeroErrorTheoryExists, errorCutoff=errorCutoff)
		# print "ran experienceReplay on {}. error: {}".format(num, mean_penalties[0])
		# if len(hypotheses)>400:
			# embed()

		results.append(mean_penalties)
		imaginedEffects.append(setOfImaginedEffects)

	# if len(hypotheses)>10:
		# print "Serial experience replay on {} theories and {} time-steps took {} seconds".format(len(hypotheses), len(rleHistory), time.time()-t1)

	# print "ran experienceReplay"
	# embed()
	mean_penalties = [r[0] for r in results]
	return mean_penalties, imaginedEffects


def MultiEpisodeExperienceReplay(hypotheses, rleHistories, actionHistories, method, targetColor=None, displayStates=False, displayTheories=False, assumeZeroErrorTheoryExists=False, errorCutoff=ERRORCUTOFF):
	'''
	Runs experience replay on multiple episodes with some action sequence for each episode and returns the penalties for the given theories (weighted on the number of actions)
	'''
	assert len(rleHistories) == len(actionHistories), 'rleHistories and actionHistories need to match'

	hypotheses = hypotheses[:] # so we can replace some with None if they're not worth continuing with (and not modify the list passed in)

	if sum([len(r) for r in rleHistories]) > 10 or len(hypotheses)>10:
		t1 = time.time()
		print "Running MultiEpisodeExperienceReplay on {} hypotheses, {} episodes and {} time-steps total".format(len(hypotheses), len(rleHistories), sum([len(r) for r in rleHistories]))

	multi_episode_mean_penalties = []
	imaginedEffectsPerTheory = [set() for i in range(len(hypotheses))]

	weight = 1./len(max(actionHistories, key=len))

	for rleHistory, actionHistory in zip(rleHistories, actionHistories):
		mean_penalties, imaginedEffects = \
				experienceReplay(hypotheses, rleHistory, actionHistory, method, targetColor, displayStates,\
						displayTheories, assumeZeroErrorTheoryExists=assumeZeroErrorTheoryExists, errorCutoff=errorCutoff)
		mean_penalties = np.array(mean_penalties)*weight*len(actionHistory)
		multi_episode_mean_penalties.append(mean_penalties)

		for i in range(len(hypotheses)):
			# if we have enough data, and it's very wrong, stop evaluating this theory for subsequent episodes
			if mean_penalties[i] > errorCutoff and (len(rleHistory) > 3 or assumeZeroErrorTheoryExists):
				hypotheses[i] = None
			else:
				imaginedEffectsPerTheory[i] = imaginedEffectsPerTheory[i].union(imaginedEffects[i])

	multi_episode_mean_penalties = np.mean(multi_episode_mean_penalties, axis=0)
	if sum([len(r) for r in rleHistories]) > 10 or len(hypotheses)>10:
		print "MultiEpisodeExperienceReplay on {} theories, {} episodes and {} time-steps took {} seconds".format(len(hypotheses), len(rleHistories), sum([len(r) for r in rleHistories]), time.time()-t1)

	return multi_episode_mean_penalties, imaginedEffectsPerTheory

########################################################################
######## THEORY MODIFICATION 									########
########################################################################

def updateTerminations(rle, hypotheses, addNoveltyRules=True):
	for h in hypotheses:
		h.updateTerminations(rle, addNoveltyRules=addNoveltyRules)
	return
	
def filterTheories(scoreAndTheoryTuples, percentile, max_num, proportionOfSpriteTheories, errorCutoff=None, usePrior=False):
	## Returns the max_num theories that are at percentile or greater, given their score.

	percentile = 100.-percentile
	scoreAndTheoryTuples = sorted(scoreAndTheoryTuples, key=lambda x: (x[0], x[1].prior()))
	cutoff = np.percentile([s[0] for s in scoreAndTheoryTuples], percentile)
	# WARNING: not the Right Thing -- do the Right Thing later
	cutoff = errorCutoff if errorCutoff else cutoff
	# end warning
	candidates = [s for s in scoreAndTheoryTuples if s[0]<=cutoff]
	if not candidates:
		return []

	## here begins new filtering method:
	# 	always keep first tier (by error) (as long as it made the maxmium allowed error cutoff)
	# 	if adding the second tier isn't too many, do that
	# 	if the first tier is too many, filter by prior
	filtered = []

	epsilon = .0001
	tierOneThresh = candidates[0][0]
	tierOne = [t for t in candidates if t[0] <= tierOneThresh + epsilon]
	filtered = tierOne # by default
	if len(tierOne) < max_num / 3 and len(tierOne) < len(candidates):
		# TODO: magic number ^
		# consider tier 2
		tierTwoThresh = candidates[len(tierOne)][0]
		# is it worth including them? TODO: magic number here
		if tierTwoThresh < 2 * tierOneThresh and tierTwoThresh < cutoff:
			tierTwo = [t for t in candidates if tierOneThresh + epsilon < t[0] <= tierTwoThresh + epsilon]
			if len(tierOne) + len(tierTwo) < max_num:
				filtered = tierOne + tierTwo
	elif len(tierOne) > max_num:
		# too many! have to filter by prior
		filtered = sorted(tierOne, key=lambda t: t[1].prior(granularity=1))[:max_num]

	# print "METHOD TWO FILTER:", [t[0] for t in filtered]

	if usePrior:
		filtered = filterByPrior(filtered)
	return filtered

def filterByPrior(scoreAndTheoryTuples, numPerLevel=1):
	# filter out any theory after the first numPerLevel whose added complexity does not improve its error
	#	i.e. between two theories of equal perfomance, ignore the less likely/more complex one
	precision = 16
	errorLevelToMinPrior = dict()
	for score, theory in scoreAndTheoryTuples:
		score = round(score, precision)
		if score not in errorLevelToMinPrior:
			errorLevelToMinPrior[score] = [999999999] * numPerLevel
		if theory.prior(granularity=1) < errorLevelToMinPrior[score][-1]:
			errorLevelToMinPrior[score][-1] = theory.prior(granularity=1)
			errorLevelToMinPrior[score].sort()
	return [sh for sh in scoreAndTheoryTuples if sh[1].prior(granularity=1) <= errorLevelToMinPrior[round(sh[0], precision)][-1]]

def expandTheories(theories, errorList, envRealPrev, envRealCurrent, prevAction, rleHistories, actionHistories, episode_num):
	# print "In expandTheories. errorList length: {}. Theories length {}".format(len(errorList), len(theories))

	# MEMOIZE!
	# lookup table (dict) which maps (classPair, predicateTuple) to all combinations of all possible rules involving those classes and predicates
	classPairPlusPredicateToRuleSets = dict()
	perColorErrorBaselines = dict()
	if sum([len(episode) for episode in rleHistories]) == OBSERVATION_PERIOD_LENGTH:
		assert len(theories)==1, "You got more than one theory in expandTheories while expecting only one."
		colors = set([e.targetColor for e in errorList])
		for color in colors:
			penalties, _ = MultiEpisodeExperienceReplay(theories, rleHistories, \
						actionHistories, method=EXPERIENCE_REPLAY_METHOD, targetColor = color)
			perColorErrorBaselines[color] = penalties[0]

	for errorMap in errorList:
		## Skip this whole step if you've already made changes for this theory. Just pass it on and you'll
		## evaluate it on the whole dataset in the outer loop.
		if len(theories) == 1 and any([errorMap == e for e in theories[0].errorMapHistory]):
			# errorMap.display()
			# print "we've addressed this error before (in expandTheories). Skipping it"
			newTheories = [theories[0]]
			theories = newTheories
			# FLAG: huh?
			continue

		# ONLY in the the single step where we'll have error maps from previous steps
		if sum([len(episode) for episode in rleHistories]) == OBSERVATION_PERIOD_LENGTH:
			envRealPrev = rleHistories[errorMap.episodeStepGenerated[0]][errorMap.episodeStepGenerated[1] - 1]
			envRealCurrent = rleHistories[errorMap.episodeStepGenerated[0]][errorMap.episodeStepGenerated[1]]
			prevAction = actionHistories[errorMap.episodeStepGenerated[0]][errorMap.episodeStepGenerated[1] - 1]

		# print "now dealing with errorMap"
		# embed()

		# print "In base case. Correcting error for {} for {} theories".format(errorMap.targetClass, len(theories))
		t1 = time.time()
		newTheories = []
		print "expanding theories for one errorMap"
		for theory in theories:
			newTheories.extend(expandTheoryForOneErrorMap(errorMap, envRealPrev, envRealCurrent, prevAction, rleHistories, actionHistories, 
					theory, classPairPlusPredicateToRuleSets))
		# print "Expanding {} theories took {} seconds".format(len(theories), time.time()-t1)

		t1 = time.time()
		# print "beforeFilter"
		# embed()
		newTheories = list(set(newTheories))
		if sum([len(episode) for episode in rleHistories]) == OBSERVATION_PERIOD_LENGTH:
			penalties, _ = MultiEpisodeExperienceReplay(newTheories, rleHistories, \
					actionHistories, method=EXPERIENCE_REPLAY_METHOD, targetColor=errorMap.targetColor, errorCutoff=.5)
			scoreAndTheoryTuples = zip(penalties, newTheories)
			print "{} theories before filtering".format(len(scoreAndTheoryTuples))

			scoreAndTheoryTupleCandidates = [tup for tup in scoreAndTheoryTuples if tup[0] < perColorErrorBaselines[errorMap.targetColor]]
			if scoreAndTheoryTupleCandidates:
				scoreAndTheoryTuples = scoreAndTheoryTupleCandidates

			print "{} theories after first filter".format(len(scoreAndTheoryTuples))
			scoreAndTheoryTuples = sorted(scoreAndTheoryTuples, key=lambda x: (x[0], x[1].prior()))

			# hyperparameters here:
			theoriesPerErrorLevel = 5
			medianDivisor = 3

			scoreAndTheoryTuples = filterByPrior(scoreAndTheoryTuples, numPerLevel=theoriesPerErrorLevel)
			print "{} theories after filtering by prior".format(len(scoreAndTheoryTuples))

			# update error threshold for this color
			med = scoreAndTheoryTuples[len(scoreAndTheoryTuples)//medianDivisor][0] + .000001 # to allow all infinitestimals
			print 'new baseline:' , med
			perColorErrorBaselines[errorMap.targetColor] = med
			scoreAndTheoryTuples = [tup for tup in scoreAndTheoryTuples if tup[0] < perColorErrorBaselines[errorMap.targetColor]]
			print "{} theories after filtering with new baseline".format(len(scoreAndTheoryTuples))

			print 'scores:' , [t[0] for t in scoreAndTheoryTuples]
			# embed()

		else:
			rleHistory, actionHistory = rleHistories[episode_num], actionHistories[episode_num]
			penalties, _ = MultiEpisodeExperienceReplay(newTheories, [rleHistory[-2:]], \
					[actionHistory[-1:]], method=EXPERIENCE_REPLAY_METHOD, targetColor = errorMap.targetColor)
			scoreAndTheoryTuples = zip(penalties, newTheories)
			scoreAndTheoryTuples = sorted(scoreAndTheoryTuples, key=lambda x: (x[0], x[1].prior()))
		
		newTheories = [s[1] for s in scoreAndTheoryTuples]
		theories = newTheories

	return theories

def addressRemainingErrorMaps(theory, envRealPrev, envRealCurrent, action, rleHistories, actionHistories):
	errorMaps = [e for e in theory.errorMapHistory if e.componentsAddressed=='spriteInduction']
	errorMaps = [e for e in errorMaps if all([d not in e.diagnosis for d in ['newObjectAppeared', 'newClass']])]
	newTheories = []
	for e in errorMaps:
		newTheories.extend(expandTheoryForOneErrorMap(e, envRealPrev,envRealCurrent,action,rleHistories,actionHistories,theory,{}))
	return newTheories

def expandTheoryForOneErrorMap(errorMap, envRealPrev, envRealCurrent, action, rleHistories, actionHistories, theory, classPairPlusPredicateToRuleSets):


	## Fixes the problems generated by a single errorMap entry.

	n = 1 # n is the number of allowed rules for a particular classpair-ordering, probably (TODO)

	errorMap = errorMap.copy()
	errorMap.targetClass = theory.spriteObjects[errorMap.targetClass].className if errorMap.targetClass in theory.spriteObjects.keys() else 'unknown'

	if errorMap.intPairs:
		newIntPairs = []
		for pair in errorMap.intPairs:
			
			if pair[0] in theory.spriteObjects.keys():
				p0 = theory.spriteObjects[pair[0]].className
			elif pair[0] in theory.classes.keys():
				p0 = pair[0]
			else:
				p0 = 'unknown'

			if pair[1] in theory.spriteObjects.keys():
				p1 = theory.spriteObjects[pair[1]].className
			elif pair[1] in theory.classes.keys():
				p1 = pair[1]
			else:
				p1 = 'unknown'
			pair = (p0, p1)
			newIntPairs.append(pair)
		errorMap.intPairs = newIntPairs

	## If we were about to make modifications we've made already, don't waste the time.
	if any([errorMap == e for e in theory.errorMapHistory]):
		errorMap.display()
		print "we've addressed this error before. Skipping it"
		newTheories = [theory]
		return newTheories

	theory.errorMapHistory.append(errorMap)
	newTheories = [theory.copy()]
	newErrorMaps = [errorMap]

	theory.experienceReplayRecord = {}

	updateAllOptions(envRealCurrent._game, envRealPrev._game, action=action)

	## If there are unknown colors in an inventory, add them to the theory here.
	if 'inventoryChange' in errorMap.diagnosis:
		from vgdl.ontology import Resource
		# print "got inventoryChange"
		# embed()
		for k in errorMap.targetToken.inventory:
			if k not in theory.spriteObjects.keys():
				color = k
				existing_classes = [key for key in theory.classes if key[0] == 'c']
				max_num = max([int(c[1:]) for c in existing_classes])
				class_num = max_num+1
				newClassName = 'c'+str(class_num)
				theory.addSpriteToTheory(newClassName, color, vgdlType=Resource, args={'limit':errorMap.targetToken.inventory[k][1]})
			else:
				theory.spriteObjects[k].vgdlType = Resource
				if theory.spriteObjects[k].args:
					theory.spriteObjects[k].args['limit'] = errorMap.targetToken.inventory[k][1]
				else:
					theory.spriteObjects[k].args = {'limit':errorMap.targetToken.inventory[k][1]}

	# print "about to check for new sprites"
	# embed()
	## If there are new objects on screen, add them to the thery or reason about related objects (e.g., spawnPoints)
	if errorMap.targetClass not in theory.classes.keys() or errorMap.targetToken in envRealCurrent._game.observation['new_sprites']:

		## If there are unknown colors on screen, add them to the theory here.		
		if errorMap.targetClass not in theory.classes.keys():
			if errorMap.targetColor in theory.spriteObjects:
				errorMap.targetClass = theory.spriteObjects[errorMap.targetColor].className
			else:
				from vgdl.ontology import Resource
				existing_classes = [key for key in theory.classes if key[0] == 'c']
				max_num = max([int(c[1:]) for c in existing_classes])
				class_num = max_num+1 
				newClassName = 'c'+str(class_num)
				errorMap.targetClass = newClassName
				theory.addSpriteToTheory(newClassName, errorMap.targetColor, vgdlType=Resource)
				print "Got unknown targetclass for {}. Added generic sprite to spriteSet and interactionSet".format(errorMap.targetToken.colorName)

			if 'newClass' in errorMap.diagnosis:
				newTheories = [theory]
				# embed()
				return newTheories
		
		## Now get overlapping/nearby classes and reassign the target class to the shooter/spawnpoint/etc. 
		## the next step will take care of not doing inference on these if we've done it already.
		neighbors = neighboringSprites(envRealCurrent, errorMap.targetToken, 1)

		# print "neighbors of new class are {}".format(neighbors)
		newErrorMaps = []
		for neighbor in neighbors:
			e = errorMap.copy()
			e.targetToken = neighbor
			e.targetTokens.append(e.targetToken)
			newPairs = []
			for num,pair in enumerate(e.intPairs):
				newPair = tuple([p if p!='unknown' else e.targetClass for p in list(pair)])
				newPairs.append(newPair)
			e.intPairs = newPairs
			e.targetClass = theory.spriteObjects[neighbor.colorName].className
			newErrorMaps.append(e)

	for eM in newErrorMaps:

		theoryCopy = theory.copy()

		if 'newObjectAppeared' in eM.diagnosis:
			if eM.targetClass in theoryCopy.expandedSprites:
				print "removing {} from theory.expandedSprites".format(eM.targetClass)
				theoryCopy.expandedSprites.remove(eM.targetClass)
		
		## SpriteSet induction step
		if eM.targetClass not in theoryCopy.expandedSprites:
			className, theories = expandSprites(envRealCurrent._game, theoryCopy, eM, 
					envRealPrev, envRealCurrent, action, percentile=20, max_num=30)
			theories = list(set(theories))
			# TODO: since we're not actually going to build on these, we haven't necessarily addressed the error
			# tomorrow: not sure if this is actually the problem
			# for t in theories:
			# 	t.errorMapHistory.pop(len(t.errorMapHistory)-1)
			# print "doing spriteInduction for {} generated {} theories".format(eM.targetClass, len(theories))
			newTheories.extend(theories)	

		## InteractionSet induction step
		for targetClassPair in eM.intPairs:

			singleIntPairErrorMap = eM.copy()
			singleIntPairErrorMap.intPairs = [targetClassPair]
			## If we have non-generic rules for this pair in the theory, then this has to involve some kind of precondition
			matchingRules = [rule for rule in theoryCopy.interactionSet if (rule not in list(theoryCopy.dryingPaint)) and 
				( targetClassPair == (rule.slot1, rule.slot2) or targetClassPair == (rule.slot2, rule.slot1) )]

			if any([not rule.generic for rule in matchingRules]):
				if ('objectDestruction' in singleIntPairErrorMap.diagnosis
							or any(['kill' in rule.interaction for rule in theoryCopy.interactionSet if eM.targetClass==rule.slot1]) ):
					singleIntPairErrorMap.diagnosis.append('conditionalKill')

			## Modify theory before the last step, then embed here to continue work
			## if the diagnosis involves objectDestruction and the targetClassPair has non-generic rules,
			## change the diagnosis here to conditionalKill such that you can propose preconditions in proposePredicates
			predicates = proposePredicates(singleIntPairErrorMap.diagnosis, envRealCurrent._game.observation)
			# if 'wrapAround' in eM.diagnosis:
				# print "got unexpectedOverlap"
				# embed()
			classPair, theories = expandLine(theoryCopy, singleIntPairErrorMap, targetClassPair, predicates,
				classPairPlusPredicateToRuleSets, envRealPrev, envRealCurrent, action, rleHistories, actionHistories, MultiEpisodeExperienceReplay, n=n, 
				observations=envRealCurrent._game.observation, generic=False)

			# this is because it would cause us to propose conditional stuff for later targetClassPairs
			if 'conditionalKill' in singleIntPairErrorMap.diagnosis:
				singleIntPairErrorMap.diagnosis.remove('conditionalKill')
			newTheories.extend(list(set(theories)))

	newTheories = list(set(newTheories))

	return newTheories

def testAndExpand(env, hypothesis, action, envReal, envRealPrev, rleHistories, actionHistories, episode_num):
	global initialErrorBuildup

	episodeStep = (episode_num, len(rleHistories[episode_num]) - 1)

	env.step(action)

	penalty, errorList = errorSignal(env, envReal, hypothesis, envRealPrev)

	# track where this error came from
	for e in errorList:
		e.episodeStepGenerated = episodeStep

	if sum([len(episode) for episode in rleHistories]) < OBSERVATION_PERIOD_LENGTH:
		initialErrorBuildup.extend(errorList)
		return []
	elif sum([len(episode) for episode in rleHistories]) == OBSERVATION_PERIOD_LENGTH:
		theories = expandTheories([hypothesis], initialErrorBuildup+errorList, None, None, None, rleHistories, actionHistories, episode_num)
	else:
		theories = expandTheories([hypothesis], errorList, 		   envRealPrev, envReal, action, rleHistories, actionHistories, episode_num)
	
	if len(theories)==1 and theories[0]==hypothesis:
		theories[0].experienceReplayRecord = hypothesis.experienceReplayRecord
	
	# if errorList:
		# hypothesis.display()
		# for e in errorList:
			# e.display()
			# print ""
	# else:
		# hypothesis.display()
		# print "No error"
		# embed()
	# print "expanding theories"

	return theories

def copyGameInferenceInfo(envReal, rle):
	envReal._game.spriteDistribution = ccopy(rle._game.spriteDistribution)
	envReal._game.object_token_spriteDistribution = rle._game.object_token_spriteDistribution
	envReal._game.movement_options = ccopy(rle._game.movement_options)
	envReal._game.orientation_options = ccopy(rle._game.orientation_options)
	envReal._game.sprite_appearance_predictions = ccopy(rle._game.sprite_appearance_predictions)
	envReal._game.object_token_movement_options = ccopy(rle._game.object_token_movement_options)
	envReal._game.sprite_appearances = ccopy(rle._game.sprite_appearances)






if __name__ == "__main__":

	##simpleGame_missile: no support for learning that it can shoot things.
	# filename = "examples.gridphysics.frogs"

	# filename = "examples.gridphysics.collect_resource"

	# filename = "examples.gridphysics.theorytest"
	# filename = "examples.continuousphysics.breakout_new"
	# filename = "examples.gridphysics.expt_push_boulders2"
	# filename = "examples.gridphysics.avatar_inference"
	# filename = "examples.gridphysics.testAll"
	
	filename = "examples.gridphysics.expt_antagonist"

	# filename = "examples.gridphysics.basics"

	level_game_pairs = None
	# Playing GVG-AI games
	def read_gvgai_game(filename):
		with open(filename, 'r') as f:
			new_doc = []
			g = gen_color()
			for line in f.readlines():
				new_line = (" ".join([string if string[:4]!="img="
					else "color={}".format(next(g))
					for string in line.split(" ")]))
				new_doc.append(new_line)
			new_doc = "\n".join(new_doc)
		return new_doc

	def gen_color():
		from vgdl.colors import colorDict
		color_list = colorDict.values()
		color_list = [c for c in color_list if c not in ['UUWSWF']]
		for color in color_list:
			yield color

	# gvggames = ['aliens', 'boulderdash', 'butterflies', 'chase', 'frogs',  # 0-4
	#   'missilecommand', 'portals', 'sokoban', 'survivezombies', 'zelda']  # 5-9

	# gameName = gvggames[6]

	# gvgname = "../gvgai/training_set_1/{}".format(gameName)

	# gameString = read_gvgai_game('{}.txt'.format(gvgname))


	# level_game_pairs = []
	# for level_number in range(5):
	#   with open('{}_lvl{}.txt'.format(gvgname, level_number), 'r') as level:
	#       level_game_pairs.append([gameString, level.read()])

	##uncomment this line to run local games
	gameName = filename


	module = importlib.import_module(gameName)
	level_game_pairs = module.level_game_pairs
	actionSequences = module.actionSequences if hasattr(module, 'actionSequences') else None

	multiTesting = False
	try:
		if module.multiTesting:
			multiTesting = True
			actionSequences = module.actionSequences
	except:
		pass


	results = []
	if multiTesting:
		# have to make a new agent each time since they're really different games all in one
		for num in xrange(len(level_game_pairs)):
			agent = Agent('full', gameName)

			try:
				results += agent.testCurriculum([level_game_pairs[num]], [actionSequences[num]])
			except:
				results += [("CRASHED :( here's all we know:", traceback.format_exc())]

	else:
		# normal mode

		agent = Agent('full', gameName)

		##For GVGAI games, use this line
		# agent.playCurriculum(level_game_pairs=level_game_pairs)

		##For local games, use this line
		# agent.playCurriculum(level_game_pairs=None)

		## For testing, use this line
		results = agent.testCurriculum(level_game_pairs, actionSequences)

	for num,res in enumerate(results):
		print '\n--------test {} produced the following:--------'.format(num+1)
		print res
		print ''

	embed()
