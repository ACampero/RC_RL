from mcts_pseudoreward_heuristic import *
from util import *
from core import colorDict
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectObjectGoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, \
MultiSpriteCounterRule, ruleCluster, Theory, Game, writeTheoryToTxt, generateSymbolDict
import importlib
from rlenvironmentnonstatic import createRLInputGame


# from vgdl.mcts_pseudoreward_heuristic import *
# from vgdl.util import *
# from vgdl.core import colorDict
# from vgdl.ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
# from vgdl.ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectObjectGoal
# from vgdl.theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, ruleCluster, Theory, Game, writeTheoryToTxt
# import importlib
# from vgdl.rlenvironmentnonstatic import createRLInputGame




'''
## helpful functions or access methods:
rle.show()
rle._getSensors()
rle.step((0,0)) ## will actually move the gamestate if things are moving, though.
rle._game.sprite_groups ## dict of unique object types and their positions

for the equivalents in thought world, just do mcts.rle.whatever
'''

def playEpisode(rleCreateFunc, hypotheses=[], game_object=None, unknown_colors=False, goalColor=None, finalEventList=[], playback=False):

	rle = rleCreateFunc()																## Initialize rle the agent behaves in.
	all_objects = rle._game.getObjects()

	# spriteInduction(rle._game, step=0)													## Initialize sprite induction

	noHypotheses = len(hypotheses)==0

	print ""
	if unknown_colors==False:
		unknown_objects = [k for k in rle._game.sprite_groups.keys() if k!='avatar']
		unknown_colors = [colorDict[str(rle._game.sprite_groups[k][0].color)] for k in unknown_objects]
		print "unknown objects:", unknown_colors
	else:
		print "already know some objects. Unknown:"
		print unknown_colors

	ended, won = rle._isDone()
	
	total_states_encountered = [rle._game.getFullState()]							## Start storing encountered states.

	Vrle=None
	g = game_object
	while not ended:																	
		if noHypotheses:																	## Observe a few frames, then initialize sprite hypotheses
			observe(rle, 5)
			sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
			g = Game(spriteInductionResult=sample)
			t = g.buildGenericTheory(sample)
			hypotheses = [t]
			noHypotheses = False


		if not Vrle:	## Initialize world in agent's head.
			symbolDict = generateSymbolDict(rle)
			# for k,v in symbolDict.items():
			# 	print k, v
			# print ""
			# print "Initializing mental theory."
			game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0], symbolDict,\
			 "./examples/gridphysics/theorytest.py")

			Vrle = createMindEnv(game, level, output=False)
			Vrle.immovables = immovables
		if goalColor:																## Select known goal if it's known, otherwise unkown object.
			key = [k for k in rle._game.sprite_groups.keys() if \
			colorDict[str(rle._game.sprite_groups[k][0].color)]==goalColor][0]
			actual_goal = rle._game.sprite_groups[key][0]
			object_goal = actual_goal
			object_goal_location = Vrle._rect2pos(object_goal.rect)
			object_goal_location = object_goal_location[1], object_goal_location[0]
			print "goal is known:", goalColor
			print ""
		else:
			try:
				object_goal = selectObjectGoal(Vrle, unknown_colors, method="random_then_nearest")
				object_goal_location = Vrle._rect2pos(object_goal.rect)
				object_goal_location = object_goal_location[1], object_goal_location[0]
				print "object goal is", colorDict[str(object_goal.color)], "at location", (rle._rect2pos(object_goal.rect)[1], rle._rect2pos(object_goal.rect)[0])
				print ""
			except:
				print "no unknown objects and no goal? Embedding so you can debug."
				embed()

		game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0], symbolDict,\
		 "./examples/gridphysics/theorytest.py", object_goal_location)

		print "Initializing mental theory *with* object goal"
		# print "immovables", immovables
		Vrle = createMindEnv(game, level, output=True)							## World in agent's head, including object goal
		Vrle.immovables = immovables
																						## Plan to get to object goal
		rle, hypotheses, finalEventList, candidate_new_colors, states_encountered, g = \
		getToObjectGoal(rle, Vrle, g, hypotheses[0], game, level, object_goal, all_objects, finalEventList, symbolDict=symbolDict)
		

		if len(unknown_colors)>0:
			for col in candidate_new_colors:
				if col in unknown_colors:
					unknown_colors.remove(col)
			
		ended, won = rle._isDone()
		# actions_taken.extend(actions_executed)
		total_states_encountered.extend(states_encountered)
		
		## Hack to remember actual winning goal, until terminationSet is fixed.
		if won and not hypotheses[0].goalColor:
			# embed()
			goalColor = finalEventList[-1]['effectList'][0][1]		#fix. don't assume the second obj is the goal.
			hypotheses[0].goalColor=goalColor

	if playback:			## TODO: Aritro cleans this up.
		print "in playback"
		from vgdl.core import VGDLParser
		from examples.gridphysics.simpleGame4 import level, game
		playbackGame = game
		playbackLevel = level
		VGDLParser.playGame(playbackGame, playbackLevel, total_states_encountered)#, persist_movie=True)

	return hypotheses, g, won, unknown_colors, goalColor, finalEventList, total_states_encountered

if __name__ == "__main__":

	filename = "examples.gridphysics.simpleGame4"
	game_to_play = lambda: createRLInputGame(filename)

	thinking_steps = 50
	thinking_default_steps=50
	
	numEpisodes = 1

	hypotheses, tally, finalEventList = [], [], []
	unknown_colors = False
	goalColor = None
	game_object = None

	embed()
	# hypotheses, won, unknown_objects, goalColor, finalEventList, total_states_encountered = \
	# playEpisode(rleCreateFunc=game_to_play, hypotheses=hypotheses, \
	# 	unknown_objects=unknown_objects, goalColor=goalColor, finalEventList=finalEventList, \
	# 	playback=False)
	# print "ended episode"
	# embed()
	# goalColor='BROWN'
	for episode in range(numEpisodes):
		hypotheses, game_object, won, unknown_colors, goalColor, finalEventList, total_states_encountered = \
		playEpisode(rleCreateFunc=game_to_play, hypotheses=hypotheses, game_object=game_object,\
			unknown_colors=unknown_colors, goalColor=goalColor, finalEventList=finalEventList, \
			playback=True)
		tally.append(won)
		print "episode ended. Win:", won
		print "__________________________________________________"
	print "Won", sum(tally), "out of ", len(tally), "episodes."
	# embed()
