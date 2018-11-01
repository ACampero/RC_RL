from mcts import *
from util import *
from core import colorDict
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectSubgoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, ruleCluster, Theory, Game, writeTheoryToTxt
import importlib
from rlenvironmentnonstatic import createRLInputGame

'''
## helpful functions or access methods:
rle.show()
rle._getSensors()
rle.step((0,0)) ## will actually move the gamestate if things are moving, though.
rle._game.sprite_groups ## dict of unique object types and their positions

for the equivalents in thought world, just do mcts.rle.whatever
'''

def playEpisode(rleCreateFunc, hypotheses=[], unknown_objects=False, goalColor=None, finalEventList=[], playback=False):

																						## Initialize rle the agent behaves in.
	rle = rleCreateFunc()
	rle._game.unknown_objects = rle._game.sprite_groups.keys()
	rle._game.unknown_objects.remove('avatar') 											## For now we're asumming agent knows self.
	rle.agentStatePrev = {}
	all_objects = rle._game.getObjects()

	spriteInduction(rle._game, step=0)													## Initialize sprite induction


	noHypotheses = len(hypotheses)==0

		# sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)		## Initialize mental theory
		# g = Game(spriteInductionResult=sample)
		# t = g.buildGenericTheory(sample)
		# hypotheses = [t]



	## Fix this mess. Store the unknown categories. Select among those for a goal, and then provide that to selectToken.
	if unknown_categories==False:
		print "initializing unknown objects:"
		unknown_categories = [k for k in rle._game.sprite_groups.keys() if k!='avatar']
		print [colorDict[str(rle._game.sprite_groups[k][0].color)] for k in unknown_categories]
	else:
		print "already know some objects. Unknown:"
		print [colorDict[str(rle._game.sprite_groups[k][0].color)] for k in unknown_categories]


	# if unknown_objects==False: ##uninitialized
	# 	print "initializing unknown objects:"
	# 	unknown_objects=[rle._game.sprite_groups[k] for k in rle._game.sprite_groups.keys() if k!='avatar'] 	## Store instances of unknown objects
	# 	print [colorDict[str(o[0].color)] for o in unknown_objects]
	# else:
	# 	print "already know some objects. Unknown:"
	# 	print [colorDict[str(o[0].color)] for o in unknown_objects]
	

	##working hypothesis is hypotheses[0] for now.
	# unknown_objects= []
	# print [r.generic for r in hypotheses[0].interactionSet]
	# for rule in hypotheses[0].interactionSet:
	# 	##right now this only tries to learn about avatar touching things
	# 	##not things touching things
	# 	##Also sometimes you're going to randomly choose an unreachable object!
	# 	if rule.generic:
	# 		## make sure this is right if you have multiple objects in the class.
	# 		## e.g., review induction assumptions.

	# 		## you're assuming that generic rules always have avatar in slot2
	# 		col = hypotheses[0].classes[rule.slot1][0].color
	# 		key = [k for k in rle._game.sprite_groups.keys() if \
	# 		colorDict[str(rle._game.sprite_groups[k][0].color)]==col][0]
	# 		unknown_objects.append(rle._game.sprite_groups[key])
	

	ended, won = rle._isDone()
	
	total_states_encountered = [rle._game.getFullState()]

	Vrle=None


	while not ended:																	## Select known goal if it's known, otherwise unkown object.
		if noHypotheses:																	## Observe a few frames, then initialize sprite hypotheses
			observe(rle, 5)
			sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
			g = Game(spriteInductionResult=sample)
			t = g.buildGenericTheory(sample)
			hypotheses = [t]

																							## Initialize world in agent's head.
		if not Vrle:
			game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0],\
			 "./examples/gridphysics/theorytest.py")#, rle._rect2pos(subgoal.rect))

			Vrle = createMindEnv(game, level, OBSERVATION_GLOBAL)
			Vrle.immovables = immovables

		if goalColor:
			key = [k for k in rle._game.sprite_groups.keys() if \
			colorDict[str(rle._game.sprite_groups[k][0].color)]==goalColor][0]
			actual_goal = rle._game.sprite_groups[key][0]
			object_goal = actual_goal
			print "goal is known:", goalColor
		else:
			try:
				object_goal = random.choice(unknown_objects)
				embed()
				subgoalLocation = selectSubgoalToken(Vrle, 'wall', unknown_objects)
			except:
				print "no unknown objects and no goal? Embedding so you can debug."
				embed()


		# embed() 
		# subgoal = random.choice(rle._game.sprite_groups[object_goal.name])
		

		

		game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0],\
		 "./examples/gridphysics/theorytest.py", subgoalLocation)
		Vrle = createMindEnv(game, level, OBSERVATION_GLOBAL)							## World in agent's head.
		Vrle.immovables = immovables
		embed()

																						## Plan to get to subgoal
		rle, hypotheses, finalEventList, candidate_new_colors, states_encountered = \
		getToSubgoal(rle, Vrle, subgoal, all_objects, finalEventList, symbolDict=symbolDict)
		

		if len(unknown_objects)>0:
			for col in candidate_new_colors:
				obj = [o for o in unknown_objects if colorDict[str(o[0].color)]==col]
				if len(obj)>0:
					obj=obj[0]
					unknown_objects.remove(obj)
			
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
		playbackGame = push_game
		playbackLevel = box_level
		embed()
		VGDLParser.playGame(playbackGame, playbackLevel, total_states_encountered)

	return hypotheses, won, unknown_objects, goalColor, finalEventList, total_states_encountered

if __name__ == "__main__":

	finalEventList = []

	filename = "examples.gridphysics.simpleGame4"
	game_to_play = lambda: createRLInputGame(filename)

	thinking_steps = 50
	thinking_default_steps=50
	
	numEpisodes = 10

	hypotheses, tally = [], []
	unknown_objects = False
	goalColor = None
	# goalColor='BROWN'
	for episode in range(numEpisodes):
		hypotheses, won, unknown_objects, goalColor, finalEventList, total_states_encountered = \
		playEpisode(rleCreateFunc=game_to_play, hypotheses=hypotheses, \
			unknown_objects=unknown_objects, goalColor=goalColor, finalEventList=finalEventList, \
			playback=True)
		tally.append(won)
		print "episode ended. Win:", won
		print "__________________________________________________"
	print "Won", sum(tally), "out of ", len(tally), "episodes."
