from util import *
from core import colorDict, VGDLParser, makeVideo, sys
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectObjectGoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, \
MultiSpriteCounterRule, ruleCluster, Theory, Game, writeTheoryToTxt, generateSymbolDict
from metaplanner import *
import importlib
from rlenvironmentnonstatic import createRLInputGame


## For now, only implementing version of agent that can deal with single goals.

class Agent:
	def __init__(self, gameFilename, plannerType):
		self.gameName = gameFilename[gameFilename.rfind('.')+1:] ## store just the game name
		self.gameFilename = gameFilename
		self.hypotheses = []
		self.symbolDict = None
		self.knownColors = []
		self.goalColor = []
		self.killerObjects = []
		self.finalEventList = []
		self.plannerType = plannerType
		self.initializeEnvironment(gameFilename)
	
	def initializeEnvironment(self, gameFilename):
		self.gameString, self.levelString = defInputGame(gameFilename, randomize=False)
		# self.rleCreateFunc = lambda: createRLInputGame(gameFilename)
		# rle = self.rleCreateFunc()
		self.rleCreateFunc = lambda: createRLInputGameFromStrings(self.gameString, self.levelString)
		rle = self.rleCreateFunc()
		rle.game_name = gameFilename
		# rle = RLEnvironmentNonStatic(self.gameString, self.levelString)
		avatarColor = colorDict[str(rle._game.sprite_groups['avatar'][0].color)]
		self.knownColors.append(avatarColor)
		return

	def pickNewLevel(self, index=False):
		self.gameString, self.levelString = defInputGame(self.gameFilename, randomize=False, index=index)
		self.rleCreateFunc = lambda: createRLInputGameFromStrings(self.gameString, self.levelString)
		rle = self.rleCreateFunc()

	def initializeHypotheses(self, rle, allObjects, learnSprites=True):
		if learnSprites:
			observe(rle, 10)
			spriteTypeHypothesis = sampleFromDistribution(rle._game.spriteDistribution, allObjects)
			gameObject = Game(spriteInductionResult=spriteTypeHypothesis)
			initialTheory = gameObject.buildGenericTheory(spriteTypeHypothesis)
		else:
			gameObject = Game(self.gameString)
			initialTheory = gameObject.buildGenericTheory(spriteSample=False, vgdlSpriteParse = gameObject.vgdlSpriteParse)

		self.hypotheses = [initialTheory]

		## Old: Used to check for Vrle and initialize accordingly.
		## New: assumption is Vrle needs to be initialized only if there are no hypotheses, so doing it all in
		## one chunk.

		self.symbolDict = generateSymbolDict(rle)

		return gameObject

	def completeHypotheses(self, rle, allObjects):
		observe(rle, 10)
		spriteTypeHypothesis = sampleFromDistribution(rle._game.spriteDistribution, allObjects)
		gameObject = Game(spriteInductionResult=spriteTypeHypothesis)
		newHypotheses = []
		for hypothesis in self.hypotheses:
			newHypotheses.append(gameObject.addNewObjectsToTheory(hypothesis, spriteTypeHypothesis))
		self.hypotheses = newHypotheses

	def getSpriteColor(self, sprite):
		return colorDict[str(sprite.color)]

	def getSpriteNameColor(self, spriteName, rle):
		try:
			return self.getSpriteColor(rle._game.sprite_groups[spriteName][0])
		except:
			print "getSpriteNameColor failed"
			embed()

	def objectSelectionPhase(self, unknownColors, allColors, rle):
		## TODO: this is contingent on only one goal existing.
		
		epsilon = .1
		## With probability 1-epsilon, select known goal if it's known, otherwise unkown object.
		if len(self.goalColor)>0 and \
		len([k for k in rle._game.sprite_groups.keys() if len(rle._game.sprite_groups[k])>0 and \
			self.getSpriteNameColor(k, rle) in self.goalColor])>0 and \
			random.random()>epsilon:
			key = random.choice([k for k in rle._game.sprite_groups.keys() if len(rle._game.sprite_groups[k])>0\
			 and self.getSpriteNameColor(k, rle) in self.goalColor])
			objectGoal = rle._game.sprite_groups[key][0]
			# actualGoal = objectGoal
			# objectGoalLocation = rle._rect2posFlipCoords(objectGoal.rect)
			print "Selecting from known goals", list(set(self.goalColor))
			print ""
		else:
			try:
				objectGoal = selectObjectGoal(rle, unknownColors, allColors, self.killerObjects, method="random_then_nearest")
				print ""
			except:
				print "no unknown objects and no goal? Embedding so you can debug."
				embed()
		objectGoalLocation = rle._rect2posFlipCoords(objectGoal.rect)
		print "object goal is", self.getSpriteColor(objectGoal), "at location", objectGoalLocation
		return objectGoal, objectGoalLocation

	def initializeVrle(self, hypothesis, objectGoalLocation, rle):
		## World in agent's head given 'hypothesis', including object goal
		gameString, levelString, symbolDict, immovables, killerObjects = writeTheoryToTxt(rle, hypothesis, self.symbolDict,\
				 "./examples/gridphysics/theorytest.py", objectGoalLocation)

		print "Initializing mental theory *with* object goal"
		Vrle = createMindEnv(gameString, levelString, output=True)
		Vrle.immovables, Vrle.killerObjects = immovables, killerObjects
		return Vrle

	def VrleInitPhase(self, objectGoalLocation, rle):
		## Initialize multiple VRLEs, each corresponding to one hypothesis in self.hypotheses
		VRLEs = []
		print "in VrleInitPhase.", len(self.hypotheses), "hypotheses"
		for hypothesis in self.hypotheses:
			VRLEs.append(self.initializeVrle(hypothesis, objectGoalLocation, rle))
		return VRLEs


	def playMultipleEpisodes(self, numEpisodes):
		
		tally, finalEventList, totalStatesEncountered = [], [], []

		gameObject = None
		for i in range(numEpisodes):
			print "Starting episode", i
			self.pickNewLevel(index=i)
			gameObject, won, eventList, statesEncountered = self.playEpisode(gameObject, finalEventList)
			finalEventList.extend(eventList)
			VGDLParser.playGame(self.gameString, self.levelString, statesEncountered, persist_movie=True, make_images=True, make_movie=False, movie_dir="videos/"+self.gameName, padding=10)
			totalStatesEncountered.append(statesEncountered)
			tally.append(won)
			print "Episode ended. Won:", won
			print "Have won", sum(tally), "out of", len(tally), "episodes"
		
		print "Won", sum(tally), "out of ", len(tally), "episodes."
		makeVideo(movie_dir="videos/"+self.gameName)
		# empty image directory
		# shutil.rmtree("images/tmp")
		# os.makedirs("images/tmp")
		return


	def playEpisode(self, gameObject, finalEventList):

		## Initialize external environment
		rle = self.rleCreateFunc()
		allObjects= rle._game.getObjects()
		# from core import colorDict
		# embed()
		# allColors = [colorDict[str(rle._game.sprite_groups[k][0].color)] for k in rle._game.sprite_groups.keys()]
		##select only non-moving objects as goals. Avoids chasing, which takes forever at the moment.
		allColors = [colorDict[str(rle._game.sprite_groups[k][0].color)] for k in rle._game.sprite_groups.keys() if len(rle._game.getSprites(k))>0 and rle._game.sprite_groups[k][0].speed is None ]
		allColors = [c for c in allColors if c!='DARKBLUE']
		unknownColors = [c for c in allColors if c not in self.knownColors]

		print "unknown colors:", unknownColors
		print "Known colors:", self.knownColors

		ended, won = rle._isDone()
		## Start storing encountered states.
		totalStatesEncountered = [rle._game.getFullState()]


		## initialize theory if necessary.
		if len(self.hypotheses) == 0:
			gameObject = self.initializeHypotheses(rle, allObjects, learnSprites=True)
			print "initializing hypotheses"
		else:
			gameObject = self.completeHypotheses(rle, allObjects)
			print "had hypotheses -- completing them."

		while not ended:

			## select explore / exploit goal
			objectGoal, objectGoalLocation = self.objectSelectionPhase(unknownColors, allColors, rle)

			## initialize one or many VRLEs according to hypothesis-selection method
			VRLEs = self.VrleInitPhase(objectGoalLocation, rle)

			## get to that goal

				## VRLEs, hypothesis-selection-method .....
				## figures out plan determined as above
				## carries out plan.
			print "calling getToObjecGoal"
			rle, self.hypotheses, finalEventList, candidateNewColors, statesEncountered, gameObject = \
				getToObjectGoal(rle, VRLEs[0], self.plannerType, gameObject, self.hypotheses[0], self.gameString, self.levelString, \
					objectGoal, allObjects, finalEventList, symbolDict=self.symbolDict)
			totalStatesEncountered.extend(statesEncountered)
			ended, won = rle._isDone()

			if won:
				for event in finalEventList[-1]['effectList']:
					if event[2]=='DARKBLUE':
						self.goalColor.append(event[1])
			else:
				for event in finalEventList[-1]['effectList']:
					if event[0]=='killSprite' and event[1]=='DARKBLUE':
						self.killerObjects.append(event[2])
			## update knowns/unknowns
			for col in candidateNewColors:
				if col not in self.knownColors:
					self.knownColors.append(col)
					print "added", col, "to knownColors"
			print "updated known colors", self.knownColors
			unknownColors = [c for c in unknownColors if c not in self.knownColors]
			print "updated unknownColors", unknownColors
		return gameObject, won, finalEventList, totalStatesEncountered

if __name__ == "__main__":
	# filename = "examples.gridphysics.simpleGame_resourceTest"

	# filename = "examples.gridphysics.simpleGame_preconditions" ## won't work until eventHandling() is corrected.
	# filename = "examples.gridphysics.simpleGame_inductionTest"
	# filename = "examples.gridphysics.simpleGame_missile2"	
	# filename = "examples.gridphysics.movers2d"	
	filename = "examples.gridphysics.movers3c"	
	# filename = "examples.gridphysics.movers4"	

	# filename = "examples.gridphysics.simpleGame_many_poisons_big"
	# filename = "examples.gridphysics.simpleGame_push_boulders2"
	# filename = "examples.gridphysics.pushtest"
	# filename = "examples.gridphysics.simpleGame_small"
	# filename = "examples.gridphysics.new_object_test"	
	# filename = "examples.gridphysics.demo_chaser"

	# filename = "examples.gridphysics.push_boulders_multigoal_incremental"	

	# filename = "examples.gridphysics.scoretest"	
	# filename = "examples.gridphysics.multigoal_and"	

	# filename = "examples.gridphysics.rivercross"	

	# filename = "examples.gridphysics.waterfall"	
	# filename = "examples.gridphysics.simpleGame_teleport"	


	# filename = "examples.gridphysics.movers5"	
	# filename = "examples.gridphysics.simpleGame_push_boulders_multigoal"	

	plannerType = "IW"
	# plannerType = "QLearning"
	# plannerType = "AStar"
	print ""
	print "Playing {} with {}".format(filename, plannerType)
	agent = Agent(filename, plannerType)
	t1 = time.time()
	numEpisodes = 5
	agent.playMultipleEpisodes(numEpisodes)
	print "Ended {} episodes of {} with planner {} in {} seconds".format(numEpisodes, filename, plannerType, time.time()-t1)
