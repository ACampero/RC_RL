"""

Main terms:

trace:				A list of TimeStep objects
TimeStep:			Everything that happened in a time step in the game. E.g.
						TimeStep.agentAction = 'up'
						TimeStep.agentState = {'health':1, 'treasure':2}
						TimeStep.events = [(bounceForward, BLUE, ORANGE), (undoAll, ORANGE, BLACK)]
						TimeStep.t = 4 #meaning all of this took place at t_4
Theory:				A VGDL description. See class description in the code.
hypothetical:		A theory, passed as an argument to theory.explain. See description under
					definition of theory.explain
hypothesis space: 	A list of theories.


def induction(trace):
	#Takes a trace, a.k.a., a list of TimeStep objects. Performs theory induction,
	maintaining an hypothesis space of theories with nonzero posterior probability.
	For now, this maintains only theories that are fully explained by the data.


	#Start with a single theory. All new theories are descendants of this one.

	hypothesisSpace = [Theory()]

	for each timestep:
		for each theory in hypothesisSpace:
			if theory.likelihood(timestep) < 1.0: #theory needs to be changed
				newTheories = theory.explain(timestep)
				hypothesisSpace.extend(newTheories)
		hypothesisSpace.cleanup() #remove theories whose posterior probability is below some threshold

	return hypothesisSpace

___________

def theory.explain(timestep, hypotheticals):
	#Returns a set of theories that explain all the events that took place at timestep
	#Hypotheticals can be passed as args to enable the explanation of multiple events in a single timestep
	
	#E.g.: e0: (bounceForward BLUE ORANGE), e1: (undoAll ORANGE BLACK)
	#This first requires coming up with (bounceForward c1 c2), and then
	#using that hypothetical theory to also build (undoAll c2 c3).

	#So, if timestep.events = [e0, e1], this will go to the recursive case.
	#It will generate hypothetical theories for explain(e0), and then call explain(e1, hypotheticals)
	#The inner call will return a theory like (bounceForward c1 c2)
	#and the outer call will use that to build (undoAll c2 c3).


	#Base case
	if len(timestep.events) == 1:
		theories = []
		if no hypotheticals:
			theories.extend(generateTheories(timestep.events[0]))
		else:
			for hypothetical in hypotheticals:
				#Here we generate theories based on each hypothetical
				theories.extend(hypothetical.generateTheories(timestep.events[0])) 
		return theories

	#Recursive case
	else:
		hypotheticals = explain(timestep.events[0])
		return explain(timestep.events[1:], hypotheticals)

___________

also fix predicates

def theory.generateTheories(event):
	#Returns theories that explain the event, which is a tuple like:
	# (bounceForward, BLUE, ORANGE)

	eventName, obj1, obj2 = event[0], event[1], event[2]

	theories = []

	if likelihood(event) == 1:
		theories.append(self) #e.g., we will end up returning the current theory.

	Else, the problem is due to one of the following cases:

		Case 1: The interpreted event is in the interactionSet,
				but the current theory didn't predict what happened.
		Solution: 	Add preconditions
		Call: 		addPreconditions(event)

		Case 2: We know the event name and the object classes, 
				but the interpetation is not in the interactionSet
		Solution:	Add a line to the interactionSet
		Call:		addRules(event) <<?

		Case 3: We know the event name, but don't know at least one of the object classes.
		Solution:	Add assignments (e.g., assign objects to existing or new classes)
		Call:		addAssignments(event)
		
		Case 3: We don't know event name.
		Solution: 	Add a new line to the ruleset: (eventName, class(obj1), class(obj2)).
						class(obj1) and class(obj2) are either the known ones or new ones; function takes care of all proposals.
		Call: 		AddRules(event)

		Each of these appends to theories []

	return theories

______

def theory.addPreconditions(event, timestep):
	
	#For now we are assuming that the only allowable preconditions
	#are simple set-theoretic functions of what's in the 
	#AgentState (e.g., the backpack), e.g.:  (health>1),

	(leave existing code more or less as is).
	

def theory.addAssignmentsKeepRules(event):
	#Consider adding new assignments to any objects with unassigned classes.

	(leave existing code more or less as is).


def theory.keepAssignmentsAddRules()
	#Either assigns objects to existing classes such that they fit with existing rules,
	#or adds new classes and adds new rule.

	Generate cross product of all existing class names + 2 new ones
	Return those assignments.


def theory.addAssignmentsAddRules(event):
	eventName, obj1, obj2 = event[0], event[1], event[2]

	theories = []
	if len(theory.interactionSet) == 0:
		interaction = (eventName, c1, c2)  					#e.g., (bounceForward c1 c2)
		classAssignments = [('c1', obj1), ('c2', obj2)]		#e.g., c1=obj1, c2=obj2
		theories.append(Theory(interaction, classAssignments))
	else:
		assignments =  searchForAssignments(event)
		for assignment in assignments:
			theories.append(Theory(eventName, assignment))
	return theories
_____

def theory.likelihood(timestep):
	each event in timestep.events is a tuple: (eventName, obj1, obj2)

	For likelihood to be 1, the following have to hold:
		-all the events in the timestep have to be in the interaction set, e.g.,
			every (eventName, class(obj1), class(obj2)) has to be a rule in the interactionSet
		all relevant rules predicted in the theory's interactionSet have to have happened durning this time step

	return 1.0 or 0.0, accordingly

_____

def hypothesisSpace.cleanup():
	remove theories whose posterior<threshold
	remove or link duplicate theories

















"""