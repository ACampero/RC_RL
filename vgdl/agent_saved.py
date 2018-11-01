from basic_mcts_domain import *
from util import *
from core import colorDict
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectSubgoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, ruleCluster, Theory, Game

'''
## helpful functions or access methods:
rle._getSensors()
rle.step((0,0)) ## will actually move the gamestate if things are moving, though.
rle._game.sprite_groups ## dict of unique object types and their positions

for the equivalents in thought world, just do mcts.rle.whatever
'''


if __name__ == "__main__":

	finalEventList = []
	obsType = OBSERVATION_GLOBAL
	thinking_steps = 50
	thinking_default_steps=50
	
	## Initialize rle the agent behaves in.
	rleCreateFunc = createRLSimpleGame4
	rle = rleCreateFunc(OBSERVATION_GLOBAL)
	rle._game.unknown_objects = rle._game.sprite_groups.keys()
	rle._game.unknown_objects.remove('avatar') 		## For now we're asumming agent knows self.
	rle.agentStatePrev = {}
	all_objects = rle._game.getObjects()

	spriteInduction(rle, step=0)					## Initialize sprite induction


	## When you restart episodes, reset the rle.agentStatePrev. Maybe some other things, too.
	
	print ""
	print ""
	print np.reshape(rle._getSensors(), rle.outdim)
	sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
	g = Game(spriteInductionResult=sample)
	## Temporary hack -- change as soon as we can write theory files.
	theories = [createRLVirtualGame, createRLVirtualGame2]
	goals = ['box1', 'box2']
	for i in range(2):
		# subgoal = selectSubgoal(rle, method='preselected')
		subgoal = random.choice(rle._game.sprite_groups[goals[i]])
		
		## Initialize some theory
		## Includes specifying the goal you selected as the termination condition. E.g., use pos.
		## This will merely overwrite the pre-existing file.

		rleVirtualFunc = theories[i] 		
		Vrle = rleVirtualFunc(OBSERVATION_GLOBAL)	##World in agent's head.

		## Plan to achieve that goal
		rle, hypotheses = getToSubgoal(rle, Vrle, subgoal, all_objects, finalEventList, sample)
		print "in agent loop"
		from rlenvironmentnonstatic import *
		from theory_template import writeTheoryToTxt
		# rle2= createRLtextTheory(obsType=OBSERVATION_GLOBAL)
		

		# sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
		# g = Game(spriteInductionResult=sample)
		# t = g.buildGenericTheory(sample)
		# embed()

		newsubgoal = random.choice(rle._game.sprite_groups['box2'])
		pos = rle._rect2pos(newsubgoal.rect)
		writeTheoryToTxt(rle,hypotheses[0], "./examples/gridphysics/textTheorytest.py", pos)
		embed()
		## Select hypothesis according to whichever method, make new VRLE
		## theory_to_world(hypotheses[0]) ## should write new .py file
		print ""

	# embed()

