def value_iteration_algorithm(states, actions, transition, rewardFunc, gamma, iterations):
	"""
	states = all states of the game
	actions = all possible actions within game
	transition = a dictionary mapping each tuple (s,a) to a dictionary.
	This dictionary maps each state s' that can be transitioned to to the probability p(s'|s,a) 
	(where s=original state, a=action, s'=new state)
	rewardFunc = a function taking as input the last state
	"""
	def single_value_iteration(oldValues, terminationStates):
		infinity = float('inf')
		# newValues = {k:0 for k,_ in oldValues.items()}
		newValues = dict()
		for prevState in states:
			if prevState in terminationStates:
				newValues[prevState] = oldValues[prevState]
				continue

			newValues[prevState] = -infinity
			for a in actions:
				q_val = 0
				newStateDist = transition[(prevState,a)]
				for newState, newStateProb in newStateDist.items():
					q_val += rewardFunc(prevState) + gamma * newStateProb * oldValues[newState]

				newValues[prevState] = max(newValues[prevState], q_val)

		return newValues

	values = {s: rewardFunc(s) for s in states}
	terminationStates = {s for s,v in values.items() if v != 0}
	for i in range(iterations):
		values = single_value_iteration(values,terminationStates)

	return values


def getAllStates(init_state, actions, transition):
	states = [init_state]
	seen_states = set()
	### Do DFS to get the set of all states
	while states:
		s = states.pop()
		if s in seen_states:
			continue

		seen_states.add(s)
		for a in actions:
			for newState,_ in transition[(s,a)]:
				if newState not in seen_states:
					states.append(newState)

	return seen_states

		