from planner import *

class QLearner(Planner):
	def __init__(self, rle, gameString, levelString, gameFilename, display, episodes=100, memory=None, \
		alpha=1, epsilon=.1, gamma=.9, partitionWeights = [20,1, 1], stepLimit=100, anneal=False):
		Planner.__init__(self, rle, gameString, levelString, gameFilename, display)
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma
		self.episodes = episodes
		self.actions = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
		self.QVals = defaultdict(lambda:0)
		self.memory = memory ## provide dicts of Q-values from previous runs. Use some function to smoothe
		self.maxPseudoReward = 1
		self.pseudoRewardDecay = .9
		self.partitionWeights = partitionWeights
		self.stepLimit=stepLimit
		self.heuristicDecay = .99
		self.anneal=anneal
		self.printSummary()

	def selectAction(self, s, policy, partitionWeights = None, domainKnowledge=True, printout=False):
		if policy == 'epsilonGreedy':
			if random.random() < self.epsilon:
				return random.choice(self.actions)
			else:
				bestQVal, bestA, QValsAreAllEqual = self.bestSA(s, partitionWeights, domainKnowledge = True)
				return bestA
		elif policy == 'greedy':
			bestQVal, bestA, QValsAreAllEqual = self.bestSA(s, partitionWeights = [1,0,0], domainKnowledge = True)
			if printout:
				print bestQVal
			if QValsAreAllEqual:
				return None
			else:
				return bestA

	def getAvoidanceScores(self, s, a):
		decay=.9
		avatarLoc = self.findAvatarInState(s)
		if avatarLoc and len(self.killerObjects)>0:
			enemyLocs = []
			for enemy in self.killerObjects:
				pos = self.findObjectsInState(s, enemy)
				if pos is not None:
					enemyLocs.extend(pos)
			nextLoc = avatarLoc[0]+a[1], avatarLoc[1]+a[0]
			penalties = [decay**manhattanDist(nextLoc, e) for e in enemyLocs]
			if len(penalties)>0:
				return -np.mean(penalties)
			else:
				return 0.
		else:
			return 0.

	def bestSA(self, s, partitionWeights, domainKnowledge=True, debug=False):
		avatarLoc = self.findAvatarInState(s)
		if domainKnowledge:
			if len(self.actionDict[avatarLoc])>0:
				actions = self.actionDict[avatarLoc]
			else:
				# set actions to full action set in case the actionDict was initialized with incorrect assumptions
				# ... and thus thinks there's nothing you can do from the current state.
				actions = self.actions
		else:
			actions = self.actions
		
		random.shuffle(actions)
		maxFuncVal = -float('inf')
		sumQVal = 0.
		sumPseudoReward = 0.
		sumAvoidanceScores = 0.
		rewardCoefficient = partitionWeights[0]
		heuristicCoefficient = partitionWeights[1]
		avoidanceCoefficient = partitionWeights[2]
		# print heuristicCoefficient
		bestAction = None
		QValsAreAllEqual = False

		# print self.rle.show()
		if debug:
			print "debugging bestSA"
			embed()
		for a in actions:
			if (s,a) not in self.QVals.keys():
				self.QVals[(s,a)] = 0.
			sumQVal += abs(self.QVals[(s,a)])
			sumPseudoReward += self.getPseudoReward(s, a)
			sumAvoidanceScores += abs(self.getAvoidanceScores(s,a))

		for a in actions:

			if sumQVal == 0.:
				QValFunction = 0.
			else:
				QValFunction = self.QVals[(s,a)]/sumQVal

			if sumPseudoReward == 0:
				pseudoRewardFunction = 0.
			else:
				pseudoRewardFunction = self.getPseudoReward(s,a)/sumPseudoReward

			if sumAvoidanceScores == 0:
				avoidanceFunction = 0.
			else:
				avoidanceFunction = self.getAvoidanceScores(s,a)/sumAvoidanceScores

			# print a
			# print rewardCoefficient*QValFunction, heuristicCoefficient*pseudoRewardFunction, avoidanceCoefficient*avoidanceFunction
			funcVal = rewardCoefficient*QValFunction + \
						heuristicCoefficient*pseudoRewardFunction + \
						avoidanceCoefficient*avoidanceFunction

			# print funcVal
			if funcVal > maxFuncVal:
				maxFuncVal = funcVal
				bestAction = a
				bestQVal = self.QVals[(s,a)]
		
		if not bestAction:
			try:
				bestAction = random.choice(actions)
				bestQVal = self.QVals[(s,a)]
				QValsAreAllEqual = True
			except:
				print "actions array is empty. in bestSA"
				print np.reshape(np.fromstring(s,dtype=float),self.rle.outdim)
				embed()

		# embed()
		return bestQVal, bestAction, QValsAreAllEqual

	def update(self, s, a, sPrime, r):
		bestQVal, bestA, QValsAreAllEqual = self.bestSA(sPrime, self.partitionWeights)
		bestQVal
		try:
			self.QVals[(s,a)] = self.QVals[(s,a)] + self.alpha * (r + self.gamma*bestQVal - self.QVals[(s,a)])
		except:
			print "didn't find qvals[s,a]"
			embed()

	def runEpisode(self):
		rle = copy.deepcopy(self.rle)
		terminal = rle._isDone()[0]
		s = rle._getSensors().tostring()
		i=0
		total_reward = 0.

		while not terminal and i<self.stepLimit:
			

			a = self.selectAction(s, policy='epsilonGreedy', partitionWeights = self.partitionWeights)
			res = rle.step(a)
			sPrime, r = res['observation'].tostring(), res['reward']
			if rle._isDone()[1]==True:
				r = 100

			## UNCOMMENT HERE IF YOU WANT TO WATCH Q-learner learning.
			if self.display:
				print rle.show()

			## lower epsilon once you've found winning states.
			if self.anneal and rle._isDone()[1]:
				self.partitionWeights[1] = self.partitionWeights[1]*self.heuristicDecay
				self.partitionWeights[2] = self.partitionWeights[2]*self.heuristicDecay
				self.epsilon = self.epsilon*self.heuristicDecay

				# print self.partitionWeights
				# print 'reward'

			self.update(s,a,sPrime,r)
			s = sPrime
			terminal = rle._isDone()[0]

			i += 1
			total_reward += r

		self.QVals[s] = 0.

	def learn(self, satisfice=0):
		t1 = time.time()
		satisfice_episodes = 0
		for i in range(self.episodes):
			# sys.stdout.write("Episodes: {}\r".format(i) )
			# sys.stdout.flush()
			self.runEpisode()
			satisfice_episodes +=1
			if i%10==0:
				s = self.rle._getSensors().tostring()
				a = self.selectAction(s, policy='greedy', partitionWeights = self.partitionWeights)
				print i, self.QVals[(s,a)]
				if satisfice: ## see if values have propagated to start state; if so, return.
					actions = self.getBestActionsForPlayout()
					if len(actions)>0:
						if satisfice_episodes>satisfice:
							return i
			if i%100==0:
				self.getBestActionsForPlayout(False, True)
		time_elapsed = time.time()-t1
		self.printSummary()
		print "Time to solution: {}".format(time_elapsed)
		print ""
		return i

	def getBestActionsForPlayout(self, aggressive=False, showActions = False):
		rle = copy.deepcopy(self.rle)
		terminal = rle._isDone()[0]
		s = rle._getSensors().tostring()
		actions = []
		# print rle.show()
		while not terminal:
			a = self.selectAction(s, policy='greedy', partitionWeights = [20,0,0], domainKnowledge = None, printout = False)
			# print self.QVals[(s,a)]
			if aggressive:
				if a is None:
					break
					# return actions
			else:
				if a is None or self.QVals[(s,a)]<=0:
					# print "Negative q-values or no action. Breaking."
					break
					# return actions
			actions.append(a)
			res = rle.step(a)
			if showActions:
				print rle.show()
			terminal = rle._isDone()[0]
			s = res['observation'].tostring()
		return actions

	def backwardsPlayback(self):
		lst = [(k,v) for k,v in self.QVals.iteritems()]
		slist = sorted(lst, key=lambda x:x[1])
		slist.reverse()
		for l in slist:
			if l[1]>0:
				print np.reshape(np.fromstring(l[0][0],dtype=float),self.rle.outdim)
				print l[1]

	def printSummary(self):
		print ""
		print "Game: {}".format(self.gameFilename)
		print "Immovables: {}".format(self.immovables)
		print "Enemies: {}".format(self.killerObjects)
		print "Parameters: Epsilon: {}. Gamma: {}. PartitionWeights: {}".format(self.epsilon, self.gamma, self.partitionWeights)
		return
if __name__ == "__main__":
	
	# gameFilename = "examples.gridphysics.simpleGame_teleport"
	# gameFilename = "examples.gridphysics.waypointtheory" 
	# gameFilename = "examples.gridphysics.demo_teleport"
	# gameFilename = "examples.gridphysics.movers3c"
	# gameFilename = "examples.gridphysics.scoretest" 
	# gameFilename = "examples.gridphysics.portals" 
	# gameFilename = "examples.gridphysics.pick_apples" 
	# gameFilename = "examples.gridphysics.demo_transform_small" 

	# gameFilename = "examples.gridphysics.demo_dodge" 
	# gameFilename = "examples.gridphysics.rivercross" 
	# gameFilename = "examples.gridphysics.demo_chaser" 

	gameFilename = "examples.gridphysics.simpleGame4_small"


	gameString, levelString = defInputGame(gameFilename, randomize=True)
	rleCreateFunc = lambda: createRLInputGame(gameFilename)
	rle = rleCreateFunc()
	# embed()
	# p = Planner(rle, gameString, levelString)
	# embed()
	# print rle.show()
	# print ""
	# print "Initializing learner. Playing", gameFilename
	ql = QLearner(rle, gameString, levelString, gameFilename, display=True, alpha=1, epsilon=.5, gamma=.9, \
		episodes=500, partitionWeights=[20,1,0], stepLimit=200, anneal=True)
	ql.killerObjects = ['chaser', 'random', 'missile1', 'missile2']

	embed()

	# # ql.immovables = ['wall', 'poison']
	# t1 = time.time()
	ql.learn(satisfice=0)
	embed()
	# print max([ql.QVals[k] for k in ql.QVals.keys()])

	# time_elapsed = time.time()-t1
	# print "Game: {}".format(gameFilename)
	# print "Ran {} rounds in {} seconds".format(ql.episodes, time_elapsed)
	# print "Immovables: {}".format(ql.immovables)
	# print "Parameters: Epsilon: {}. Gamma: {}. PartitionWeights: {}".format(ql.epsilon, ql.gamma, ql.partitionWeights)
	# print ""

	# embed()