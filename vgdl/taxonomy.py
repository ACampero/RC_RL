# from theory_template_071416 import *
from ontology import *
from sampleVGDLString import *
# import pygraphviz as PG

'''
TODO:
Avatar inheritance is weirdly implemented in VGDL; some avatars are oriented sprites whereas some are
moving avatars. Distance function will be very wrong because of this, at least when it comes to avatars.

Make our own tree -- for now, just use our own intuitions.
'''
# A = PG.AGraph(directed=True, strict=True)

class Tree(object):
	def __init__(self, name, VGDLType, parent=False):
		self.name = name
		self.VGDLType = VGDLType
		self.parent = parent
		self.children = []
		self.head = self
		#maintain member list in head of tree
		if not self.parent:
			self.members = {str(VGDLType):self}
			self.depth = 0
			self.ancestors = set([self])
		else:
			self.depth = parent.depth + 1
			self.ancestors = set(parent.ancestors)
			self.ancestors.add(self)

	def addChild(self, sprite):
		if str(sprite) in self.members.keys():
			print "{} already in tree.".format(sprite)
			return
		#if the parent is in the tree	
		elif str(sprite.__base__) in self.members.keys():
			parent = self.members[str(sprite.__base__)]
			t = Tree(parent.name, sprite, parent)
			t.head = self.head #inherit head from head of tree
			parent.children.append(t)
			t.head.members[str(sprite)] = t
			# A.add_edge(str(parent.VGDLType), str(sprite))
			print "added self, {}".format(sprite)
		 #Otherwise, add the sprite's parent and then add the sprite.
		else:
			self.addChild(sprite.__base__)
			print "added parent, {}".format(sprite.__base__)
			self.addChild(sprite)

	def distance(self, n1, n2):
		#returns distance (nodes traversed) between node 1 and node 2. Example:
			# n1=VGDLTree.children[4].children[0]
			# n2=VGDLTree.children[4].children[1]
			# VGDLTree.distance(n1,n2)
		#TODO: if you pass a string as n1 and n2, e.g., 'c1','c2', this can convert those to sprite objects and ruin subsequent
		#function calls in theory_template. Copy here, then convert.
		#TODO: Refine to take depth in tree into account -- going from depth 2 to depth 3
		#should be less costly than traveling from depth 1 to 2.

		if str(n1) in self.members:
			n1 = self.members[str(n1)]
		if str(n2) in self.members:
			n2 = self.members[str(n2)]
		overlap = n1.ancestors & n2.ancestors
		overlap = sorted(list(overlap), key=lambda x:-x.depth)
		nearestParent = overlap[0]
		return (n1.depth-nearestParent.depth) + (n2.depth-nearestParent.depth)

	def similarity(self, n1, n2):
		return 1./(self.distance(n1, n2) + 1)


VGDLTree = Tree('VGDLTree', VGDLSprite)

#These are all the classes defined in ontology.py
types = [Immovable, Passive, ResourcePack, Flicker, Spreader, SpriteProducer, Portal, SpawnPoint,
RandomNPC, OrientedSprite, Conveyor, Missile, OrientedFlicker, Walker, WalkJumper,
RandomInertial, RandomMissile, ErraticMissile, Bomber, Chaser, Fleeing, AStarChaser,
MovingAvatar, HorizontalAvatar, VerticalAvatar, FlakAvatar, AimedFlakAvatar, OrientedAvatar, 
RotatingAvatar, RotatingFlippingAvatar, NoisyRotatingFlippingAvatar, ShootAvatar, AimedAvatar,
AimedFlakAvatar, InertialAvatar, MarioAvatar]

for vgdltype in types:
	VGDLTree.addChild(vgdltype)

#Make graph visualization
#A.write('VGDL_ontology.dot ')
#A.layout(prog='dot')