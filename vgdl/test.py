from theory_template_080116 import Game, TimeStep
from sampleVGDLString import *
from taxonomy import *
from class_theory_template_071916 import *
from induction import runInduction_DFS
from IPython import embed
import time, ast


#TODO: If this is still an issue... Figure out issue where hypotheses only are found when you run this more than once; always breaks once the precondition event happens
#TODO: Figure out why DARKBLUE agent appears twice in the termination set
#TODO: Green agent often thought to be DARKBLUE, and then GREEN is considered a resource pack, but is actually an agent
#TODO: Any ways to split the termination conditions between "win" and "lose"?

def testTrace(vgdlFile, gameOutputFile, expectedHypotheses, name, maxNumTheories=100, verbose=False):
	"""
	Streamlined method to test DFS induction on a trace and see the number of outputted hypotheses.
	"""
	# Obtaining appropriate game information

	with open("../vgdl_text/{}".format(vgdlFile), 'r') as vf:
		vgdlString = ast.literal_eval(vf.read())

	with open("../output/{}".format(gameOutputFile), 'r') as f:
		output = f.readline()
		outputTuple = ast.literal_eval(output)

	start = time.time()
	g, hypotheses = runInduction_DFS(vgdlString, outputTuple, maxNumTheories)
	end = time.time()

	# Useful for quick view of test results
	print "########################"
	print "Checking {}...".format(name)
	print "TIME TO RUN: {}".format(end-start)
	print "Expected number of hypotheses {} = actual number of hypotheses {}? {}".format(expectedHypotheses, len(hypotheses), expectedHypotheses==len(hypotheses))
	
	if expectedHypotheses==len(hypotheses): 	# TODO: Are there other parameters which we want to check?
		print ">>>> PASS!"
	else:
		print ">>>> FAIL :("
	print "\n########################\n\n\n\n\n\n\n"

	return hypotheses


def testMultipleTraces(rawTraces, expectedHypotheses, names, verbose):
	"""
	"""

	# New game generated
	g = Game(push_game)
	g.VGDLTree = VGDLTree
	start = time.time()
	traces =[]
	for i in range(len(rawTraces)):
		rawTrace = rawTraces[i]
		# print rawTrace
		trace = ([TimeStep(tr['agentAction'], tr['agentState'], tr['effectList'], tr['gameState']) for tr in rawTrace[0]],rawTrace[1])
		traces.append(trace)

	hypotheses=list(g.inductionOverMultipleTraces(traces, verbose))
	end = time.time()
	print "########################"
	print "Checking {}...".format(names)
	print "TIME TO RUN: {}".format(end-start)
	print "Expected number of hypotheses {} = actual number of hypotheses {}? {}".format(expectedHypotheses, len(hypotheses), expectedHypotheses==len(hypotheses))
	if expectedHypotheses==len(hypotheses):
		print ">>>> PASS!"
	else:
		print ">>>> FAIL :("
	# TODO: Are there other parameters which we want to check?
	print "\n########################\n\n\n\n\n\n\n"
	# embed()
	return hypotheses


if __name__ == '__main__':

	#############################################
	
	## TEST CASES
	
	
	"""
	SimpleGames (short)
	"""
	simple1_output = ""
	max_theories = 100
	expectedHypotheses = 1
	simple1_info = ("../vgdl_text/simpleGame1.txt", simple1_output, expectedHypotheses, "simple1")


	"""
	Aliens

	TODO: 
		- seems to register missles as resourcepacks, and the agent get put in the same classes as different vgdltype classes
		- thinks that the moving avatar is a DARKBLUE character, but it should be GREEN
	"""
	aliens_output = ""
	max_theories = 100
	expectedHypotheses = 1
	aliens_info = ("../vgdl_text/aliens.txt", aliens_output, expectedHypotheses, "aliens")



	"""
	Boulderdash
	TODO: 
		- fix the following error: 
			  File "test.py", line 577, in <module>
			    hypotheses = testTrace(*params)
			  File "test.py", line 35, in testTrace
			    g, hypotheses = runInduction_DFS(vgdlString, outputTuple, maxNumTheories)
			  File "induction.py", line 26, in runInduction_DFS
			    hypotheses=list(g.runDFSInduction(trace, maxTheories, verbose))
			  File "theory_template.py", line 1376, in runDFSInduction
			    self.induction(temp_new_trace, verbose=False)
			  File "theory_template.py", line 1426, in induction
			    if theory.likelihood(timestep) < 1.0: 	# Theory needs to be changed
			  File "theory_template.py", line 390, in likelihood
			    if self.checkEventsInTimeStep(timestep) and self.checkPredictionsInTimeStep(timestep, sparse):
			  File "theory_template.py", line 437, in checkEventsInTimeStep
			    interpretations = [self.interpret(event) for event in timestep.events]
			  File "theory_template.py", line 649, in interpret
			    obj2 = self.spriteObjects[event[2]]
			KeyError: 'DARKGRAY'

	"""
	boulderdash_output = "boulderdash_2016_10_07_09_30_00.txt" # Lose case
	max_theories = 100
	expectedHypotheses = 1
	boulderdash_info = ("../vgdl_text/boulderdash.txt", boulderdash_output, expectedHypotheses, "boulderdash")






	"""
	Butterflies

	TODO: 
		
	"""
	butterflies_output = ""
	max_theories = 100
	expectedHypotheses = 1
	butterflies_info = ("../vgdl_text/butterflies.txt", butterflies_output, expectedHypotheses, "butterflies")




	"""
	Chase

	TODO:
		- killSprite doesn't show up as an event between the agent and the blue enemies
		- get a keyError for "DARKGRAY" items
	"""
	chase_output = ""
	max_theories = 100
	expectedHypotheses = 1
	chase_info = ("../vgdl_text/chase.txt", chase_output, expectedHypotheses, "chase")



	"""
	Dodge
	"""
	dodge_output = ""
	max_theories = 100
	expectedHypotheses = 1
	dodge_info = ("../vgdl_text/dodge.txt", dodge_output, expectedHypotheses, "dodge")



	"""
	Frogs 
	"""
	frogs_output = ""
	max_theories = 100
	expectedHypotheses = 1
	frogs_info = ("../vgdl_text/frogs.txt", frogs_output, expectedHypotheses, "frogs")




	"""
	MissileCommand
	"""
	misslecommand_output = ""
	max_theories = 100
	expectedHypotheses = 1
	missilecommand_info = ("../vgdl_text/missilecommand.txt", missilecommand_output, expectedHypotheses, "missilecommand")


	"""
	Mr. Pacman
	"""
	mrpacman_output = ""
	max_theories = 100
	expectedHypotheses = 1
	mrpacman_info = ("../vgdl_text/mrpacman.txt", mrpacman_output, expectedHypotheses, "mrpacman")



	"""
	Portals
	"""
	portals_output = ""
	max_theories = 100
	expectedHypotheses = 1
	portals_info = ("../vgdl_text/portals.txt", portals_output, expectedHypotheses, "portals")



	"""
	Sokoban
	"""
	sokoban_output = ""
	max_theories = 100
	expectedHypotheses = 1
	sokoban_info = ("../vgdl_text/sokoban.txt", sokoban_output, expectedHypotheses, "sokoban")



	"""
	Survive Zombies
	"""
	zombies_output = ""
	max_theories = 100
	expectedHypotheses = 1
	zombies_info = ("../vgdl_text/survivezombies.txt", zombies_info, expectedHypotheses, "survivezombies")


	"""
	Zelda
	"""
	zelda_output = ""
	max_theories = 100
	expectedHypotheses = 1
	zelda_info = ("../vgdl_text/zelda.txt", zelda_output, expectedHypotheses, "zelda")


	

	#############################################

	## TESTING SINGLE TRACES
	

	# DFS Format: 
	# def testTrace(vgdlFile, gameOutputFile, expectedHypotheses, name, maxNumTheories=100, verbose=False)
	DFS_traces = [
	# aliens_info,
	# boulderdash_info,
	# butterflies_info,
	# chase_info,
	# dodge_info,
	# frogs_info,
	# missilecommand_info,
	# mrpacman_info,
	# portals_info,
	# sokoban_info,
	# zombies_info,
	# zelda_info
	]

	for params in DFS_traces:
		params = list(params)
		name = params[3]
		hypotheses = testTrace(*params)
		generatedHypotheses[name] = hypotheses
	


	#############################################	
	## TESTING MULTIPLE TRACES

	# traces = [rawTrace_preconditions_simple, rawTrace_simple_win, rawTrace_simple_loss]
	# hypotheses = testMultipleTraces(traces, 1, ["rawTrace_simple_win","rawTrace_simple_loss", "rawTrace_preconditions_simple"], False)

	

	embed()

