from theory_template import Game, TimeStep
from sampleVGDLString import *
from taxonomy import *
from class_theory_template import *
from induction import runInduction_DFS
from IPython import embed
import time, ast


#TODO: If this is still an issue... Figure out issue where hypotheses only are found when you run this more than once; always breaks once the precondition event happens
#TODO: Figure out why DARKBLUE agent appears twice in the termination set
#TODO: Green agent often thought to be DARKBLUE, and then GREEN is considered a resource pack, but is actually an agent
#TODO: Any ways to split the termination conditions between "win" and "lose"?

def testTrace(vgdlFile, gameOutputFile, expectedHypotheses, name, maxNumTheories=100, verbose=True):
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




if __name__ == '__main__':

	###################################
	## TEST CASES
	###################################
	"""
	Artillery

	TODO:
	- KeyError: 'DARKGRAY'
	- Can't get a termination condition
	"""
	artillery_output = "artillery.txt"
	max_theories = 100
	expectedHypotheses = 1
	artillery_info = ("../vgdl_text/artillery.txt", artillery_output, expectedHypotheses, "artillery")


	"""
	Lander

	TODO:
	- hard to play!
	"""
	lander_output = ""
	max_theories = 100
	expectedHypotheses = 1
	lander_info = ("../vgdl_text/lander.txt", lander_output, expectedHypotheses, "lander")




	"""
	Mario

	Example output that works: 
		- Without pullWithIt predicate: "mario_without_pullwithit.txt"
		- With pullWithIt predicate: "mario_runme.txt"

	TODO: 
	- Termination conditions seem incorrect (they are generally about time limit, but should be about hitting an end of screen, for instance)
	- Had to stop printing/recording wallStop command because something to do with the repeated events / 2 different events about the same objects in a timestep; also induction theory depth is high, over >300 (use "mario_induction_forever.txt")
	- If you limit the max_theories for the "mario_induction_forever.txt", get: "RuntimeError: maximum recursion depth exceeded in cmp"
	"""
	mario_output = "mario_runme.txt"
	
	max_theories = 100
	expectedHypotheses = 1
	mario_info = ("../vgdl_text/mario.txt", mario_output, expectedHypotheses, "mario")



	"""
	Pong (2 player)
	"""
	pong_output = "pong_runme.txt"
	max_theories = 100
	expectedHypotheses = 1
	pong_info = ("../vgdl_text/pong.txt", pong_output, expectedHypotheses, "pong")


	"""
	PTSP

	TODO:
	- color issue "PURPLE"
	"""
	ptsp_output = "ptsp_2016_10_12_14_58_18.txt"
	max_theories = 100
	expectedHypotheses = 1
	ptsp_info = ("../vgdl_text/pong.txt", ptsp_output, expectedHypotheses, "ptsp")


	"""
	Tankwars

	TODO: 
	- color key error "DARKGRAY"
	"""
	tankwars_output = "tankwars_2016_10_12_15_01_57.txt"
	max_theories = 100
	expectedHypotheses = 1
	tankwars_info = ("../vgdl_text/tankwars.txt", tankwars_output, expectedHypotheses, "tankwars")



	###################################
	## TESTING SINGLE TRACES
	###################################

	# DFS Format: 
	# def testTrace(vgdlFile, gameOutputFile, expectedHypotheses, name, maxNumTheories=100, verbose=False)
	DFS_traces = [
	artillery_info, 
	#lander_info,
	mario_info#,
	#pong_info, 
	#ptsp_info,
	#tankwars_info
	]

	generatedHypotheses = {}

	for params in DFS_traces:
		params = list(params)
		name = params[3]
		hypotheses = testTrace(*params)
		generatedHypotheses[name] = hypotheses
	


	###################################
	## TESTING MULTIPLE TRACES
	###################################

	# traces = [rawTrace_preconditions_simple, rawTrace_simple_win, rawTrace_simple_loss]
	# hypotheses = testMultipleTraces(traces, 1, ["rawTrace_simple_win","rawTrace_simple_loss", "rawTrace_preconditions_simple"], False)

	

	embed()

