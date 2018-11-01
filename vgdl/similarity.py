import numpy as np
from sklearn.preprocessing import normalize


#Simialrity of each class (rows) to each other class (columns)
#No need to normalize here.
	
# x = np.matrix([
# 			[1, .5, .2, .3],
# 			[.5, 1, .1, .8],
# 			[.2, .1, 1, .4],
# 			[.3, .8, .4, 1]
# 		   ])

# c1 = normalize([1, .5, .2, .3])
# c2 = normalize([.5, 1, .1, .8])
# c3 = normalize([.2, .1, 1, .4])
# c4 = normalize([.3, .8, .4, 1])

# x = np.matrix([c1[0],c2[0],c3[0],c4[0]])

# def sim(pair1, pair2):
# 	a,b = pair1[0], pair1[1]
# 	m,n = pair2[0], pair2[1]

# 	return (x[a,m]+x[b,n])/2.


c1 = set(['r1', 'r2', 'r3', 'r4', 'r5', 'r6'])
c2 = set(['r1'])
c3 = set(['r4'])
c5 = set([])
c4 = set(['r2', 'r3'])
def Tversky(self, s1, s2):
	alpha = .5
	beta = .5
	a = min(len(s1-s2), len(s2-s1))
	b = max(len(s1-s2), len(s2-s1))
	return len(s1&s2) / (len(s1&s2) + beta*(alpha*a+(1-alpha)*b))

def Levenshtein(s1, s2):
	count = 0
	s1, s2 = list(s1), list(s2)
	for i in range(len(s1)):
		if s1[i] not in s2:
			s2.append(s1[i])
			count += 1
	to_remove = []
	for i in range(len(s2)):
		if s2[i] not in s1:
			to_remove.append(s2[i])
			count += 1
	for i in range(len(to_remove)):
		s2.remove(to_remove[i])
	print 'count', count
	return 1./(1+count)