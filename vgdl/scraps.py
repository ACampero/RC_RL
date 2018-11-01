
	## from WBP.py
	def calculateAtoms(self, rle):
		## Converts rle state into a long list of atoms of length Nx2xT
		## (N: num of grid cells. 2: there / not there. T: Number of object types in the game).
		## For a 3x5 grid, we first flatten into a single column of len=15. (rle._getSensors() already has this representation)
		## atomList(m,n): whether object n is at location m.

		# TODO: Make implementation that considers avatar orientations. See p.3 of Geffner|Geffner paper.

		# vec = np.empty(self.vecDim)
		vec = []
		state = rle._getSensors()
		for i in range(len(state)):
			vec.extend(self.factorizeBoolean(rle, state[i]))
		
		if self.trackTokens:
			present = []
			for k in [t for t in self.objectTypes if t not in ['wall', 'avatar']]:
				for o in rle._game.sprite_groups[k]:
					if o not in rle._game.kill_list:
						present.append(1)
					else:
						present.append(0)
			ind = sum([present[i]*2**i for i in range(len(present))])
			nums = list(np.zeros(2**self.phiSize))
			nums[ind]=1
			vec = vec+nums

		# nums = present+absent
		# print len(nums)
		## Now add new information: # For each type of item, are there 0, 1, 2, ... maxNum on the board?
		# for k in self.objectTypes:
		# 	numOnBoard = len([o for o in rle._game.sprite_groups[k] if o not in rle._game.kill_list])
		# 	lst = [1 if i==numOnBoard else 0 for i in range(self.maxNumObjects)]
		# 	if numOnBoard>=self.maxNumObjects:
		# 		lst.append(1)
		# 	else:
		# 		lst.append(0)
		# 	invlst = [1 if l==0 else 0 for l in lst]
		# 	both = lst+invlst
		# 	nums.extend(both)
		
		# embed()
		return np.array(vec)

	# def getNumInFactorizedState(self, factorizedState, objType):
	# 	phiSize = (self.maxNumObjects+1)*len(self.objectTypes)*2
	# 	len(factorizedState)
	# 	relevantPartOfState = factorizedState[-phiSize:]
	# 	ind = self.objectTypes.index(objType)
	# 	cut = relevantPartOfState[ind*(self.maxNumObjects+1)*2:ind*(self.maxNumObjects+1)*2+(self.maxNumObjects+1)]
	# 	return len(factorizedState)-phiSize+ind*(self.maxNumObjects+1)*2+np.where(cut==1)[0], cut

	# def getNumInFactorizedState(self, factorizedState, objType):

	# 	relevantPartOfState = factorizedState[-phiSize:]
	# 	ind = self.objectTypes.index(objType)
	# 	cut = relevantPartOfState[ind*(self.maxNumObjects+1)*2:ind*(self.maxNumObjects+1)*2+(self.maxNumObjects+1)]
	# 	return len(factorizedState)-phiSize+ind*(self.maxNumObjects+1)*2+np.where(cut==1)[0], cut

	def getManhattanDistance(self, state): ##used to be passed self, state
		"""
		expect avatar to be called 'avatar' in class section of theory
		expect goal to be called 'goal' in class section of theory
		currently expects the state observation to follow a grid string format (orignal default format)
		"""
		deltaY, deltaX = self.getManhattanDistanceComponents(state)
		return abs(deltaX) + abs(deltaY)

	def getManhattanDistanceComponents(self, state):
		
		reshaped_state = np.reshape(state, self.outdim)
		avatar = 1
		goal = 2**(1+sorted(self._obstypes.keys())[::-1].index("goal"))
		avatar_loc = np.where(reshaped_state==self.avatar_code)
		goal_loc = np.where(reshaped_state==goal)
		# if goal in reshaped_state and
		# print avatar_loc[0], goal_loc[0]
		if len(avatar_loc[0])>0 and len(goal_loc[0])>0:
			dist = avatar_loc[0][0]-goal_loc[0][0], avatar_loc[1][0]-goal_loc[1][0]
			return dist
		elif len(avatar_loc[0])==0 and len(goal_loc[0])>0:
			print "found a goal but no avatar"
			embed()
			return 100,100 ##TODO: hacked on 1/18. Fix
		elif len(avatar_loc[0])>0 and len(goal_loc[0])==0:
			return 0,0
		else:
			print "manhattanDistanceComponents. Weird 'else' case."
			embed()
			return 0,0

	def makeActionSet(self, n_samples):

		outList = []
		numActions = 4
		actionKeys = range(4)
		## Returns shuffled list of n_samples drawn from 'actions'.
		partition = np.random.dirichlet([1]*numActions,1)[0]
		outList = [self.actionDict[np.random.choice(actionKeys, p=partition)] for _ in range(n_samples)]
		return outList

	def defaultPolicyB(self, s, rle, step_horizon):
		"""
		Version that samples a chunk of actions.
		"""
		t1 = time.time()
		reward = 0
		stepSize = 1 # try 13 later

		temperature = 3
		terminal = False
		iters = 0
		state = s.state
		g = .5
		# reshaped_state = np.reshape(state, self.outdim)
		# avatar_initial_loc = np.where(reshaped_state==1)

		samples = self.makeActionSet(step_horizon)

		for i in range(len(samples)):
			iters += 1
			if not terminal:
				a = samples[i]
				t2 = time.time()
				res = rle.step(a)
				new_state = res["observation"]
				state = new_state
				terminal = not res['pcontinue']
				reward += g*res['reward']
				g *= self.decay_factor

				self.defaultTime += 1 # useless right now.
			else:
				break

		return reward, iters

		def getToSubgoal(rle, vrle, subgoal, all_objects, finalEventList, verbose=True, 
	max_actions_per_plan=1, planning_steps=100, defaultPolicyMaxSteps=50, symbolDict=None):
	## Takes a real world, a theory (instantiated as a virtual world)
	## Moves the agent through the world, updating the theory as needed
	## Ends when subgoal is reached.
	## Right now will only properly work with max_actions_per_plan=1, as you want to re-plan when the theory changes.
	## Otherwise it will only replan every max_actions_per_plan steps.
	## Returns real world in its new state, as well as theory in its new state.
	## TODO: also return a trace of events and of game states for re-creation
	
	hypotheses = []
	terminal = rle._isDone()[0]
	goal_achieved = False
	outdim = rle.outdim

	def noise(action):
		prob=0.
		if random.random()<prob:
			return random.choice(BASEDIRS)
		else:
			return action

	## TODO: this will be problematic when new objects appear, if you don't update it.
	# all_objects = rle._game.getObjects()

	print ""
	print "object goal is", colorDict[str(subgoal.color)], rle._rect2pos(subgoal.rect)
	# actions_executed = []
	states_encountered = []
	while not terminal and not goal_achieved:

		mcts = Basic_MCTS(existing_rle=rle, game=game, level=level, partitionWeights=[5,1,5])
		subgoals = mcts.getSubgoals(subgoal_path_threshold=4)
		total_steps = 0

		for subgoal in subgoals:
			rle, steps = getToWaypoint(rle, subgoal, defaultPolicyMaxSteps, partitionWeights=[5,1,3])
			print steps, "steps"
			total_steps += steps




		# mcts = Basic_MCTS(existing_rle=vrle)
		# planner, steps = mcts.startTrainingPhase(planning_steps, defaultPolicyMaxSteps, vrle)
		# actions = mcts.getBestActionsForPlayout((1,0,0))

		# for i in range(len(actions)):
		# 	if not terminal and not goal_achieved:
		# 		spriteInduction(rle._game, step=1)
		# 		spriteInduction(rle._game, step=2)

		# 		## Take actual step. RLE Updates all positions.
		# 		res = rle.step(noise(actions[i])) ##added noise for testing, but prob(noise)=0 now.
		# 		# actions_executed.append(actions[i])
		# 		states_encountered.append(rle._game.getFullState())

		# 		new_state = res['observation']
		# 		terminal = rle._isDone()[0]
				
		# 		# vrle_res = vrle.step(noise(actions[i]))
		# 		# vrle_new_state = vrle_res['observation']

		# 		effects = translateEvents(res['effectList'], all_objects) ##TODO: this gets object colors, not IDs.
				
		# 		print ACTIONS[actions[i]]
		# 		rle.show()

		# 		# if symbolDict:
		# 		# 	print rle.show()
		# 		# else:
		# 		# 	print np.reshape(new_state, rle.outdim)
				
		# 		# Save the event and agent state
		# 		try:
		# 			agentState = dict(rle._game.getAvatars()[0].resources)
		# 			rle.agentStatePrev = agentState
		# 		# If agent is killed before we get agentState
		# 		except Exception as e:	# TODO: how to process changes in resources that led to termination state?
		# 			agentState = rle.agentStatePrev

		# 		## If there were collisions, update history and perform interactionSet induction
		# 		if effects:
		# 			state = rle._game.getFullState()
		# 			event = {'agentState': agentState, 'agentAction': actions[i], 'effectList': effects, 'gameState': rle._game.getFullStateColorized()}
		# 			finalEventList.append(event)

		# 			for effect in effects:
		# 				rle._game.collision_objects.add(effect[1]) ##sometimes event is just (predicate, obj1)
		# 				if len(effect)==3: ## usually event is (predicate, obj1, obj2)
		# 					rle._game.collision_objects.add(effect[2])

		# 			if colorDict[str(subgoal.color)] in [item for sublist in effects for item in sublist]:
		# 				print "reached subgoal"
		# 				goal_achieved = True
		# 				if subgoal.name in rle._game.unknown_objects:
		# 					rle._game.unknown_objects.remove(subgoal.name)
		# 				goalLoc=None
		# 			else:
		# 				goalLoc = rle._rect2pos(subgoal.rect)

		# 			## Sampling from the spriteDisribution makes sense, as it's
		# 			## independent of what we've learned about the interactionSet.
		# 			## Every timeStep, we should update our beliefs given what we've seen.
		# 			# if not sample:
		# 			sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
						
		# 			g = Game(spriteInductionResult=sample)
		# 			terminationCondition = {'ended': False, 'win':False, 'time':rle._game.time}
		# 			trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState']) for e in finalEventList], terminationCondition)


		# 			hypotheses = list(g.runInduction(sample, trace, 20))

					
		# 			candidate_new_objs = []
		# 			for interaction in hypotheses[0].interactionSet:
		# 				if not interaction.generic:
		# 					if interaction.slot1 != 'avatar':
		# 						candidate_new_objs.append(interaction.slot1)
		# 					if interaction.slot2 != 'avatar':
		# 						candidate_new_objs.append(interaction.slot2)
		# 			candidate_new_objs = list(set(candidate_new_objs))
		# 			candidate_new_colors = []
		# 			for o in candidate_new_objs:
		# 				cols = [c.color for c in hypotheses[0].classes[o]]
		# 				candidate_new_colors.extend(cols)

		# 			## among the many things to fix:
		# 			for e in finalEventList[-1]['effectList']:
		# 				if e[1] == 'DARKBLUE':
		# 					candidate_new_colors.append(e[2])
		# 				if e[2] == 'DARKBLUE':
		# 					candidate_new_colors.append(e[1])

		# 			game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0], "./examples/gridphysics/theorytest.py", goalLoc=goalLoc)
		# 			# all_immovables.extend(immovables)
		# 			# print all_immovables
		# 			vrle = createMindEnv(game, level, OBSERVATION_GLOBAL)
		# 			vrle.immovables = immovables


		# 			## TODO: You're re-running all of theory induction for every timestep
		# 			## every time. Fix this.
		# 			## if you fix it, note that you'd be passing a different g each time,
		# 			## since you sampled (above).
		# 			# hypotheses = list(g.runDFSInduction(trace, 20))

		# 		spriteInduction(rle._game, step=3)
		# if terminal:
		# 	if rle._isDone()[1]:
		# 		print "game won"
		# 	else:
		# 		print "Agent died."
	return rle, hypotheses, finalEventList, candidate_new_colors, states_encountered


# def getToSubgoal(rle, vrle, subgoal, all_objects, finalEventList, verbose=True, 
# 	max_actions_per_plan=1, planning_steps=100, defaultPolicyMaxSteps=50, symbolDict=None):
# 	## Takes a real world, a theory (instantiated as a virtual world)
# 	## Moves the agent through the world, updating the theory as needed
# 	## Ends when subgoal is reached.
# 	## Right now will only properly work with max_actions_per_plan=1, as you want to re-plan when the theory changes.
# 	## Otherwise it will only replan every max_actions_per_plan steps.
# 	## Returns real world in its new state, as well as theory in its new state.
# 	## TODO: also return a trace of events and of game states for re-creation
	
# 	hypotheses = []
# 	terminal = rle._isDone()[0]
# 	goal_achieved = False

# 	def noise(action):
# 		prob=0.
# 		if random.random()<prob:
# 			return random.choice(BASEDIRS)
# 		else:
# 			return action

# 	## TODO: this will be problematic when new objects appear, if you don't update it.
# 	# all_objects = rle._game.getObjects()

# 	print ""
# 	print "object goal is", colorDict[str(subgoal.color)], rle._rect2pos(subgoal.rect)
# 	# actions_executed = []
# 	states_encountered = []
# 	while not terminal and not goal_achieved:
# 		mcts = Basic_MCTS(existing_rle=vrle)
# 		planner, steps = mcts.startTrainingPhase(planning_steps, defaultPolicyMaxSteps, vrle)
# 		actions = mcts.getBestActionsForPlayout((1,0,0))

# 		for i in range(len(actions)):
# 			if not terminal and not goal_achieved:
# 				spriteInduction(rle._game, step=1)
# 				spriteInduction(rle._game, step=2)

# 				## Take actual step. RLE Updates all positions.
# 				res = rle.step(noise(actions[i])) ##added noise for testing, but prob(noise)=0 now.
# 				# actions_executed.append(actions[i])
# 				states_encountered.append(rle._game.getFullState())

# 				new_state = res['observation']
# 				terminal = rle._isDone()[0]
				
# 				# vrle_res = vrle.step(noise(actions[i]))
# 				# vrle_new_state = vrle_res['observation']

# 				effects = translateEvents(res['effectList'], all_objects) ##TODO: this gets object colors, not IDs.
				
# 				print ACTIONS[actions[i]]
# 				rle.show()

# 				# if symbolDict:
# 				# 	print rle.show()
# 				# else:
# 				# 	print np.reshape(new_state, rle.outdim)
				
# 				# Save the event and agent state
# 				try:
# 					agentState = dict(rle._game.getAvatars()[0].resources)
# 					rle.agentStatePrev = agentState
# 				# If agent is killed before we get agentState
# 				except Exception as e:	# TODO: how to process changes in resources that led to termination state?
# 					agentState = rle.agentStatePrev

# 				## If there were collisions, update history and perform interactionSet induction
# 				if effects:
# 					state = rle._game.getFullState()
# 					event = {'agentState': agentState, 'agentAction': actions[i], 'effectList': effects, 'gameState': rle._game.getFullStateColorized()}
# 					finalEventList.append(event)

# 					for effect in effects:
# 						rle._game.collision_objects.add(effect[1]) ##sometimes event is just (predicate, obj1)
# 						if len(effect)==3: ## usually event is (predicate, obj1, obj2)
# 							rle._game.collision_objects.add(effect[2])

# 					if colorDict[str(subgoal.color)] in [item for sublist in effects for item in sublist]:
# 						print "reached subgoal"
# 						goal_achieved = True
# 						if subgoal.name in rle._game.unknown_objects:
# 							rle._game.unknown_objects.remove(subgoal.name)
# 						goalLoc=None
# 					else:
# 						goalLoc = rle._rect2pos(subgoal.rect)

# 					## Sampling from the spriteDisribution makes sense, as it's
# 					## independent of what we've learned about the interactionSet.
# 					## Every timeStep, we should update our beliefs given what we've seen.
# 					# if not sample:
# 					sample = sampleFromDistribution(rle._game.spriteDistribution, all_objects)
						
# 					g = Game(spriteInductionResult=sample)
# 					terminationCondition = {'ended': False, 'win':False, 'time':rle._game.time}
# 					trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState']) for e in finalEventList], terminationCondition)


# 					hypotheses = list(g.runInduction(sample, trace, 20))

					
# 					candidate_new_objs = []
# 					for interaction in hypotheses[0].interactionSet:
# 						if not interaction.generic:
# 							if interaction.slot1 != 'avatar':
# 								candidate_new_objs.append(interaction.slot1)
# 							if interaction.slot2 != 'avatar':
# 								candidate_new_objs.append(interaction.slot2)
# 					candidate_new_objs = list(set(candidate_new_objs))
# 					candidate_new_colors = []
# 					for o in candidate_new_objs:
# 						cols = [c.color for c in hypotheses[0].classes[o]]
# 						candidate_new_colors.extend(cols)

# 					## among the many things to fix:
# 					for e in finalEventList[-1]['effectList']:
# 						if e[1] == 'DARKBLUE':
# 							candidate_new_colors.append(e[2])
# 						if e[2] == 'DARKBLUE':
# 							candidate_new_colors.append(e[1])

# 					game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0], "./examples/gridphysics/theorytest.py", goalLoc=goalLoc)
# 					# all_immovables.extend(immovables)
# 					# print all_immovables
# 					vrle = createMindEnv(game, level, OBSERVATION_GLOBAL)
# 					vrle.immovables = immovables


# 					## TODO: You're re-running all of theory induction for every timestep
# 					## every time. Fix this.
# 					## if you fix it, note that you'd be passing a different g each time,
# 					## since you sampled (above).
# 					# hypotheses = list(g.runDFSInduction(trace, 20))

# 				spriteInduction(rle._game, step=3)
# 		if terminal:
# 			if rle._isDone()[1]:
# 				print "game won"
# 			else:
# 				print "Agent died."
# 	return rle, hypotheses, finalEventList, candidate_new_colors, states_encountered