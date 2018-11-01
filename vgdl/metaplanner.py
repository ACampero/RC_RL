from ontology import distributionInitSetup
from WBP import *
from mcts import *
from qlearner import *
from aStar import *
import time
from termcolor import colored

def translateEvents(events, all_objects, rle):
	if events is None:
		return None
	# all_objects = rle._game.getObjects()

	def getObjectColor(objectID):
		if objectID is None:
			return None
		elif objectID == 'EOS':
			return 'ENDOFSCREEN'
		elif objectID in all_objects.keys():
			return all_objects[objectID]['type']['color']
		elif objectID in rle._game.getObjects().keys():
			return rle._game.getObjects()[objectID]['type']['color']
		elif objectID in [colorDict[k] for k in colorDict.keys()]:
			# If we were passed a color to begin with (i.e., in the case of EOS)
			return objectID
		elif objectID in rle._game.sprite_groups.keys():
			return colorDict[str(rle._game.sprite_groups[objectID][0].color)]
		elif objectID in [obj.ID for obj in rle._game.kill_list]:
			objectColor = [obj.color for obj in rle._game.kill_list
				if obj.ID==objectID][0]
			return colorDict[str(objectColor)]
		else:
			# for some reason we haven't been passed an ID but rather a sprite object
			objectName = objectID.name
			color = [all_objects[k]['type']['color'] for k in all_objects.keys() if all_objects[k]['sprite'].name==objectName][0]
			return color

	outlist = []
	for event in events:
		# if 'EOS' in event:
		# 	print "in translateEvents"
		# 	embed()
		try:
			# print 'in translateEvents', event
			if len(event) > 3:
				tmp = [event[0], getObjectColor(event[1]), getObjectColor(event[2])]
				for k in event[3].keys():
					if k=='stype':
						event[3][k] = getObjectColor(event[3][k])
				tmp.extend(event[3:])
				outlist.append(tuple(tmp))
			if len(event)==3:
				outlist.append((event[0], getObjectColor(event[1]), getObjectColor(event[2])))
			elif len(event)==2:
				outlist.append((event[0], getObjectColor(event[1])))
		except:
			print "translateEvents failed"
			embed()

	#Make sure events in timestep are unique (don't want to double-count things)
	uniqueEventList = []
	for o in outlist:
		if o not in uniqueEventList:
			uniqueEventList.append(o)
	if len(uniqueEventList)>0:
		print uniqueEventList
	return uniqueEventList


def observe(rle, obsSteps, bestSpriteTypeDict, display=False):
	if display:
		print "observing for {} steps".format(obsSteps)
	if obsSteps>0:
		for i in range(obsSteps):
			# print rle.show()
			# t1 = time.time()
			spriteInduction(rle._game, step=1, bestSpriteTypeDict=bestSpriteTypeDict)
			# print "step 1 took {} seconds".format(time.time()-t1)
			# t1 = time.time()
			spriteInduction(rle._game, step=2, bestSpriteTypeDict=bestSpriteTypeDict)
			# print "step 2 took {} seconds".format(time.time()-t1)
			rle.step((0,0))
			if display:
				print "score: {}, game tick: {}".format(rle._game.score, rle._game.time)
				print rle.show(color='blue')

			rle._game.nextPositions = {}
			for k, v in rle._game.all_objects.iteritems():
				rle._game.nextPositions[k] = (int(rle._game.all_objects[k]['sprite'].rect.x), int(rle._game.all_objects[k]['sprite'].rect.y))
				try:
					if rle._game.previousPositions[k] != rle._game.nextPositions[k]:
						rle._game.objectMemoryDict[k] = copy.deepcopy(rle._game.previousPositions[k])
				except KeyError:
					pass
			rle._game.previousPositions = copy.deepcopy(rle._game.nextPositions)

			# pinkID = [k for k in rle._game.all_objects.keys() if rle._game.all_objects[k]['features']['color']=='PINK'][0]
			# print "prev position", rle._game.previousPositions[pinkID]
			# print "memoryDict", rle._game.objectMemoryDict[pinkID]
			# print "curr position", rle._game.all_objects[pinkID]['sprite'].rect
			# t1 = time.time()
			spriteInduction(rle._game, step=3, bestSpriteTypeDict=bestSpriteTypeDict)
			# print "step 3 took {} seconds".format(time.time()-t1)
	else:
		spriteInduction(rle._game, step=1, bestSpriteTypeDict=bestSpriteTypeDict)
		spriteInduction(rle._game, step=2, bestSpriteTypeDict=bestSpriteTypeDict)
		# spriteInduction(rle._game, step=3)
	return


def planActLoop(rleCreateFunc, filename, max_actions_per_plan, planning_steps, defaultPolicyMaxSteps, playback=False):

	rle = rleCreateFunc(OBSERVATION_GLOBAL)
	game, level = defInputGame(filename)
	outdim = rle.outdim
	print rle.show()

	terminal = rle._isDone()[0]

	i=0
	finalStates = [rle._game.getFullState()]
	while not terminal:
		mcts = Basic_MCTS(existing_rle=rle, game=game, level=level)
		mcts.startTrainingPhase(planning_steps, defaultPolicyMaxSteps, rle)
		# mcts.debug(mcts.rle, output=True, numActions=3)
		# break
		actions = mcts.getBestActionsForPlayout()

		# if len(actions)<max_actions_per_plan:
		# 	print "We only computed", len(actions), "actions."

		new_state = rle._getSensors()
		terminal = rle._isDone()[0]

		for j in range(min(len(actions), max_actions_per_plan)):
			if actions[j] is not None and not terminal:
				print ACTIONS[actions[j]]
				res = rle.step(actions[j])
				new_state = res["observation"]
				terminal = not res['pcontinue']
				print rle.show()
				finalStates.append(rle._game.getFullState())

		i+=1

	if playback:
		from vgdl.core import VGDLParser
		VGDLParser.playGame(game, level, finalStates)
		embed()


def planUntilSolved(rleCreateFunc, filename, defaultPolicyMaxSteps, partitionWeights, playback=False, maxEpisodes=700):

	rle = rleCreateFunc(OBSERVATION_GLOBAL)
	game, level = defInputGame(filename)
	outdim = rle.outdim
	symbolDict = generateSymbolDict(rle)
	print rle.show()

	goal_loc = np.where(np.reshape(rle._getSensors(), rle.outdim)==8)
	goal_loc = goal_loc[0][0], goal_loc[1][0]
	terminal = rle._isDone()[0]

	i=0
	finalStates = [rle._game.getFullState()]
	## Have to make this as a theory and then write it, so that you can find what the immovables are
	## then these can get incorporated when you look for subgoals.
	theory = generateTheoryFromGame(rle)
	# theory.display()
	theoryString, levelString, inverseMapping, immovables, killerObjects =\
	writeTheoryToTxt(rle, theory, symbolDict, "./examples/gridphysics/whatever.py", goal_loc)

	rle = createMindEnv(theoryString, levelString, output=False)
	rle.immovables, rle.killerObjects = immovables, killerObjects

	mcts = Basic_MCTS(existing_rle=rle, game=game, level=level, partitionWeights=[5,2,3])
	subgoals = mcts.getSubgoals(subgoal_path_threshold=3)
	print "subgoals", subgoals


	total_steps = 0
	solved = True
	numActions = 0
	for subgoal in subgoals:
		rle, actions = getToWaypoint(rle, subgoal, symbolDict, defaultPolicyMaxSteps, partitionWeights=[10,2,4])
		numActions += len(actions)
		print steps, "steps"
		# total_steps += steps
		if total_steps > maxEpisodes:
			solved = False
			break

	if solved:
		print "Found and executed plan using", total_steps, "epiosodes of MCTS."
	else:
		print "didn't solve game even using %i episodes of MCTS"%total_steps

	return mcts, total_steps, solved, numActions

def parallelizedPlanUntilSolved(rleCreateFunc, filename, defaultPolicyMaxSteps, partitionWeightsList, numWorkers=4):
	"""
	partitionWeightsList = a list of partitionWeight tuples.
	numWorkers = the number of threads which are running planUntilSolved in parallel
	"""
	# m = multiprocessing.Manager()
	weightsQueue = multiprocessing.Queue()
	# contract: the weightsQueue will contain all the partition weights in the beginning
	# and a numWorkers number of DONE_MESSAGEs at the very end
	# to enssure each of the workers stops running.
	resultsQueue = multiprocessing.Queue()
	weightInfo = dict()
	DONE_MESSAGE = "DONE"
	def worker(weightsQue, resultsQue, DONE_MESSAGE):
		i = 0
		while not weightsQue.empty():
			message = weightsQue.get()
			if message == DONE_MESSAGE:
				break

			partitionWeights = message
			mcts, total_steps, solved, numActions = planUntilSolved(rleCreateFunc, filename, defaultPolicyMaxSteps, partitionWeights)
			resultsQueue.put((partitionWeights, {'total_steps': total_steps, 'solved': solved, 'numActions': numActions}))


	jobs = []
	for partitionWeights in partitionWeightsList:
		weightsQueue.put(partitionWeights)


	for i in range(numWorkers):
		weightsQueue.put(DONE_MESSAGE)
		p = multiprocessing.Process(target=worker, args=(weightsQueue, resultsQueue, DONE_MESSAGE))
		jobs.append(p)
		p.start()

	for j in jobs:
		j.join()

	while not resultsQueue.empty():
		(partitionWeights, result) = resultsQueue.get()
		weightInfo[partitionWeights] = result

	return weightInfo


def getToWaypoint(rle, subgoal, plannerType, symbolDict, defaultPolicyMaxSteps, partitionWeights, act=True):

	theory = generateTheoryFromGame(rle)

	print "in getToWayPoint"
	# embed()
	# print "making RLE in getToWaypoint, after generateTheoryFromGame()"
	theoryString, levelString, inverseMapping, immovables, killerObjects =\
	writeTheoryToTxt(rle, theory, symbolDict, "./examples/gridphysics/waypointtheory.py", subgoal)
	Vrle = createMindEnv(theoryString, levelString, output=False)
	Vrle.immovables, Vrle.killerObjects = immovables, killerObjects

	print "mental map with subgoal", subgoal
	print Vrle.show()
	print "planner type", plannerType
	if plannerType=='IW':
		planner = IW(rle=Vrle, gameString=theoryString, levelString=levelString, gameFilename=Vrle.game_name, k=2)
		planner.BFS(Vrle)
		actions = planner.solution.actionSeq
	elif plannerType=='mcts':
		mcts = Basic_MCTS(existing_rle=Vrle, game=theoryString, level=levelString, partitionWeights=partitionWeights)
		# print "made mcts for subgoal,", subgoal
		# embed()
		m, steps = mcts.startTrainingPhase(1200, defaultPolicyMaxSteps, Vrle, mark_solution=True, solution_limit=20)
		actions = mcts.getBestActionsForPlayout((1,0,0), debug=False)
	elif plannerType=='QLearning':
		planner = QLearner(Vrle, gameString=theoryString, levelString=levelString, alpha=1, epsilon=.5)
		steps = planner.learn(1000, satisfice=200)
		actions = planner.getBestActionsForPlayout()
	elif plannerType=='AStar':
		planner = AStar(Vrle, gameString=theoryString, levelString=levelString)
		path, actions = planner.search()
	print "Found plan to subgoal. Actions", actions
	if act:
		for a in actions:
			rle.step(a)
			print rle.show()
	return rle, actions

def objectGoalReached(effects, object_goal):
	## Check if you reached object goal
	goal_achieved = False
	for e in effects:
		if 'DARKBLUE' in e and colorDict[str(object_goal.color)] in e:
			print "goal achieved"
			goal_achieved = True
			break
	return goal_achieved

def updateCandidateColors(hypotheses, finalEventList):

	## new colors that we have maybe learned about
	candidate_new_objs, candidate_new_colors = [], []

	for interaction in hypotheses[0].interactionSet:
		if not interaction.generic:
			if interaction.slot1 != 'avatar':
				candidate_new_objs.append(interaction.slot1)
			if interaction.slot2 != 'avatar':
				candidate_new_objs.append(interaction.slot2)
	candidate_new_objs = list(set(candidate_new_objs))
	for o in candidate_new_objs:
		cols = [c.color for c in hypotheses[0].classes[o]]
		candidate_new_colors.extend(cols)

	## among the many things to fix:

	for e in finalEventList[-1]['effectList']:
		if e[1] == 'DARKBLUE':
			candidate_new_colors.append(e[2])
			# print "appending", e[2], "to candidate_new_colors"
		if e[2] == 'DARKBLUE':
			candidate_new_colors.append(e[1])
			# print "appending", e[1], "to candidate_new_colors"

	candidate_new_colors = list(set(candidate_new_colors))

	return candidate_new_colors

def getToObjectGoal(rle, vrle, plannerType, game_object, hypothesis, game, level, object_goal, all_objects, finalEventList, verbose=True,\
	defaultPolicyMaxSteps=50, symbolDict=None):
	## Takes a real world, a theory (instantiated as a virtual world)
	## Moves the agent through the world, updating the theory as needed
	## Ends when object_goal is reached.
	## Returns real world in its new state, as well as theory in its new state.
	## TODO: also return a trace of events and of game states for re-creation


	hypotheses = []
	terminal = rle._isDone()[0]
	goal_achieved = False
	outdim = rle.outdim
	candidate_new_colors = []

	def noise(action):
		prob=0.
		if random.random()<prob:
			return random.choice(BASEDIRS)
		else:
			return action

	## Add newly-seen objects.
	current_objects = rle._game.getObjects()
	for k in current_objects.keys():
		if k not in all_objects.keys():
			all_objects[k] = current_objects[k]

	states_encountered = [rle._game.getFullState()]
	hypotheses = [hypothesis]

	# print "at start of getToObjectGoal"
	# hypothesis.display()

	while not terminal and not goal_achieved:

		theory_change_flag = False
		resetSubgoals = False
		if not theory_change_flag:
			if plannerType == 'IW':
				planner = IW(rle=vrle, gameString=game, levelString=level, gameFilename=vrle.game_name, k=2)
				subgoals = planner.getSubgoals(subgoal_path_threshold=None)
			elif plannerType=='mcts':
				planner = Basic_MCTS(existing_rle=vrle, game=game, level=level, partitionWeights=[5,3,3])
				subgoals = planner.getSubgoals(subgoal_path_threshold=3)
			elif plannerType=='QLearning':
				planner = QLearner(vrle, gameString=game, levelString=level, alpha=1, epsilon=.4)
				subgoals = planner.getSubgoals(subgoal_path_threshold=4)
			elif plannerType=='AStar':
				planner = AStar(vrle, gameString=game, levelString=level)
				subgoals = planner.getSubgoals(subgoal_path_threshold=100) ##This finds subgoals by searching the entire space, so do this only once and then use the path.
			print "subgoals", subgoals

			## if you can't find subgoals that get you to the goal, exit
			if len(subgoals)==0:
				return rle, hypotheses, finalEventList, candidate_new_colors, states_encountered, game_object

			total_steps = 0

			for subgoal in subgoals:
				if not theory_change_flag and not goal_achieved and not resetSubgoals:

					## write subgoal to theory; initialize VRLE.
					print "at top of metaplanner loop -- making theory"
					game, level, symbolDict, immovables, killerObjects = writeTheoryToTxt(rle, hypotheses[0], symbolDict, \
						"./examples/gridphysics/theorytest.py", subgoal)
					vrle = createMindEnv(game, level, output=False)
					vrle.immovables, vrle.killerObjects = immovables, killerObjects


					## Get actions that take you to goal.
					ignore, actions = getToWaypoint(vrle, subgoal, plannerType, symbolDict, defaultPolicyMaxSteps, partitionWeights=[5,3,3], act=False)

					## Sometimes you can have a theory under which you can't get to a goal!
					## i.e., if you think that the objects around you will kill you (even though they won't in real life)
					## In this case, take a random action.
					if len(actions)==0:
						actions = [random.choice([(1,0), (-1,0), (0,1), (0,-1)])]
						resetSubgoals = True ## if you fail to find a plan, re-calculate subgoals, rather than moving on to the next subgoal after a single action.

					for action in actions:
						if not theory_change_flag and not goal_achieved:
							spriteInduction(rle._game, step=1)
							spriteInduction(rle._game, step=2)

							try:
								agentState = dict(rle._game.getAvatars()[0].resources)
								rle.agentStatePrev = agentState
							# If agent is killed before we get agentState
							except Exception as e:	# TODO: how to process changes in resources that led to termination state?
								# agentState = defaultdict(lambda: 0)
								agentState = rle.agentStatePrev
								print "didn't find agentState resources"
								embed()


							print "agentState", agentState
							res = rle.step(noise(action))


							## Add newly-seen objects.
							current_objects = rle._game.getObjects()
							for k in current_objects.keys():
								if k not in all_objects.keys():
									all_objects[k] = current_objects[k]
									distributionInitSetup(rle._game, k)
									rle._game.ignoreList.append(k) ## this is a hack -- the point is to prevent spriteInduction from
																	## trying to infer anything about newly-appeared sprites in this timestep.


							states_encountered.append(rle._game.getFullState())
							terminal = rle._isDone()[0]

							## The problem is you added the sprite itself to the effectList, and now translateEvents()
							## is trying to interpret that. You either want to make up a proper event
							## or don't add the sprite to the effectList at that point.
							## You also need to work out where/how to add the new sprite type to the theory.
							## POtentially, when you find the new sprite, change the stype argument in the effectList
							## and then if you see something whose color you don't know there, change the theory.
							# if len(res['effectList'])>0:
							# 	print "before translateEvents"
							# 	embed()
							effects = translateEvents(res['effectList'], all_objects, rle)

							k = random.choice(rle._game.spriteDistribution.keys())

							spriteInduction(rle._game, step=3)

							if symbolDict:
								print rle.show()
							else:
								print np.reshape(new_state, rle.outdim)

							## If there were collisions, update history and perform interactionSet induction if the collisions were novel.
							if effects:

								state = rle._game.getFullState()
								# print "about to do getFullStateColorized()"
								# embed()
								event = {'agentState': agentState, 'agentAction': action, 'effectList': effects, 'gameState': rle._game.getFullStateColorized(), 'rle': rle._game}

								goal_achieved = objectGoalReached(effects, object_goal)

								## Sampling from the spriteDisribution makes sense, as it's
								## independent of what we've learned about the interactionSet.
								## Every timeStep, we should update our beliefs given what we've seen.
								sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
								# print "just sampled"
								# embed()

								game_object = Game(spriteInductionResult=sample)


								## Get list of all effects we've seen. Only update theory if we're seeing something new.
								all_effects = [item for sublist in [e['effectList'] for e in finalEventList] for item in sublist]
								if not all([e in all_effects for e in effects]):## TODO: make sure you write this so that it works with simultaneous effects.

									print "new effects", [e for e in effects if not e in all_effects]

									finalEventList.append(event)
									terminationCondition = {'ended': False, 'win':False, 'time':rle._game.time}
									trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState'], e['rle']) for e in finalEventList], terminationCondition)
									theory_change_flag = True

									# print "about to run induction"
									# embed()
									hypotheses = list(game_object.runInduction(game_object.spriteInductionResult, trace, 20, verbose=False, existingTheories=hypotheses)) ##if you resample or run sprite induction, this

									# print "altered theory"
									# hypotheses[0].display()
									# embed()
									if len(hypotheses)>1:
										print "more than one hypothesis"
										embed()

									candidate_new_colors = updateCandidateColors(hypotheses, finalEventList)


									print "updating internal theory"
									# print "avatarLoc", planner.findAvatarInRLE(rle)
									## update to incorporate what we've learned, keep the same subgoal for now; this will update at the top of the next loop.
									game, level, symbolDict, immovables, killerObjects = writeTheoryToTxt(rle, hypotheses[0], symbolDict, \
										"./examples/gridphysics/theorytest.py", goalLoc=(rle._rect2pos(object_goal.rect)[1], rle._rect2pos(object_goal.rect)[0]))

									vrle = createMindEnv(game, level, output=True)
									vrle.immovables, vrle.killerObjects = immovables, killerObjects

									# If setting the new VRLE's resources fails, it's becuase there is no avatar, so don't worry about that here.
									try:
										vrle._game.getAvatars()[0].resources = rle._game.getAvatars()[0].resources
									except:
										pass

									# hypotheses[0].display()
									# print ""
								else:
									print "no new effects", effects
									finalEventList.append(event)
									terminationCondition = {'ended': False, 'win':False, 'time':rle._game.time}
									trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState']) for e in finalEventList], terminationCondition)
									## TODO: you need to figure out how to incorporate the result of sprite induction in cases where you don't do
									## interactionSet induction (i.e., here.)
									# hypotheses = [hypothesis]

							if terminal:
								return rle, hypotheses, finalEventList, candidate_new_colors, states_encountered, game_object

					print "executed all actions."
					## If you finish all actions, vrle needs to reflect most recent state.
					## goalLoc will be overwritten once you find new subgoals at the top.
					game, level, symbolDict, immovables, killerObjects = writeTheoryToTxt(rle, hypotheses[0], symbolDict, \
						"./examples/gridphysics/theorytest.py", goalLoc=(rle._rect2pos(object_goal.rect)[1], rle._rect2pos(object_goal.rect)[0]))
					vrle = createMindEnv(game, level, output=False)
					vrle.immovables, vrle.killerObjects = immovables, killerObjects
			# total_steps += steps
	return rle, hypotheses, finalEventList, candidate_new_colors, states_encountered, game_object
