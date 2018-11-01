
import math
import core
from IPython import embed
import pygame
#from tools import logToFile

# Fix AStar 'get_walkable_tiles' to allow for a buffer around walls.
# Also, allow for a buffer around it's goal so that it can actually get to it.
# Make tiles less discritized (maybe based on speed, if we have access to that, which I think we do)
# See why it keeps returning empty paths???
class AStarNode(object):

	def __init__(self, index, vgdlSprite, parent = None):
		self.vgdlSprite = vgdlSprite
		self.sprite = vgdlSprite
		self.index = index
		self.parent = parent


class AStarWorld(object):

	def __init__(self, game):
		self.game = game
		#ghost_sprites = game.getSprites('ghost') 
		#pacman_sprite = game.getSprites('pacman')[0]

		self.food = game.getSprites('food')
		self.nest = game.getSprites('nest')
		self.moving = game.getSprites('moving')
		self.avatar = game.getSprites('avatar')
		self.empty = [core.VGDLSprite(pos, (self.game.block_size, self.game.block_size)) for pos in self.game.emptyBlocks()]

		##print "food=%s, nest=%s, moving=%s" %(len(food), len(nest), len(moving))
		##print "empty=%s"  %	(len(empty))
		##print "total=%s" %(len(food)+len(nest)+len(moving)+len(empty))

		##print "len(sprites)=%s" %len(sprites)
		#print "game.width=%s, game.height=%s" %(game.width, game.height)
		#print "pacman_sprite=%s" %(pacman_sprite)
		#print "x=%s, y=%s" %(pacman_sprite.rect.left/game.block_size, pacman_sprite.rect.top/game.block_size)

		self.save_walkable_tiles()

	def get_walkable_tiles(self):
		return self.food + self.nest + self.moving + self.empty

	def save_walkable_tiles(self):

		self.walkable_tiles = {}
		self.walkable_tile_indices = []

		combined = self.food + self.nest + self.moving + self.empty + self.avatar
		#print combined
		for sprite in combined:
			#print sprite
			tileX, tileY = self.get_sprite_tile_position(sprite)
			index = self.get_index(tileX, tileY)
			self.walkable_tile_indices.append(index)
			self.walkable_tiles[index] = AStarNode(index, sprite)

	

	def get_index(self, tileX, tileY):
		#return tileX  * self.game.width + tileY
		return tileY  * self.game.width + tileX

	def get_tile_from_index(self, index):
		return index/self.game.width, index%self.game.width

	def h(self, start, goal):
		"""
		Distance from start to goal; taxicab distance or euclidean distance.
		"""
		#return self.euclidean(start, goal)
		return self.distance(start, goal)

	def euclidean(self, node1, node2):
		x1, y1 = self.get_sprite_tile_position(node1.sprite)
		x2, y2 = self.get_sprite_tile_position(node2.sprite)

		#print "x1:%s, y1:%s, x2:%s, y2:%s" %(x1,y1,x2,y2)
		a = x2-x1
		b = y2-y1

		#print "a:%s, b:%s" %(a,b)
		return math.sqrt(a*a + b*b)


	def get_sprite_tile_position(self, sprite):
		# print sprite.speed
		tileX = sprite.rect.left/self.game.block_size
		tileY = sprite.rect.top/self.game.block_size

		return tileX, tileY


	def get_lowest_f(self, nodes, f_score):
		"""
		Searches for the node with the lowest f_score
		"""
		f_best = 9999 
		node_best = None
		for node in nodes:
			if f_score[node.index] < f_best:
				f_best = f_score[node.index]
				node_best = node

		return node_best


	def reconstruct_path(self, current):
		#print self.get_tile_from_index(current.index)
		# print current.sprite
		# raw_input('press enter to continue...')
		if current.parent:
			p = self.reconstruct_path(current.parent)
			p.append(current)
			return p
		else:
			return [current]


	def neighbor_nodes(self, node):
		sprite = node.sprite;
		return self.neighbor_nodes_of_sprite(sprite)
	
	def neighbor_nodes_of_sprite(self, sprite):
		tileX, tileY = self.get_sprite_tile_position(sprite)

		tiles = [ (tileX-1,tileY), (tileX+1, tileY), (tileX,tileY-1), (tileX, tileY+1)]
		neighbors = []
		for (tilex, tiley) in tiles:
			if (tilex >= 0 and tilex < self.game.width and tiley >= 0 and tiley < self.game.height):
				index = self.get_index(tilex, tiley)
				if index in self.walkable_tile_indices:
					neighbors.append(self.walkable_tiles[index])

		# neighbor_indices = [neighbor.index for neighbor in neighbors]
		# print 'neighbors(%s,%s):%s' %(tileX, tileY, map(self.get_tile_from_index, neighbor_indices))

		return neighbors

	def distance(self, node1, node2):
		"""
		Taxicab distance
		"""
		x1, y1 = self.get_sprite_tile_position(node1.sprite)
		x2, y2 = self.get_sprite_tile_position(node2.sprite)

		return abs(x2-x1) + abs(y2-y1)

	def getMoveFor(self, startSprite, target):
		tileX, tileY = self.get_sprite_tile_position(startSprite)
		index = self.get_index(tileX, tileY)
		startNode = AStarNode(index, startSprite)
		target = self.game.getSprites(target)[0]
		
		# if 'pacman' in self.game.sprite_groups:
		# 	target = self.game.getSprites('pacman')[0]
		# if 'avatar' in self.game.sprite_groups:
		# 	target = self.game.getSprites('avatar')[0]
		# elif 'hungry' in self.game.sprite_groups:
		# 	target = self.game.getSprites('hungry')[0]
		# elif 'powered' in self.game.sprite_groups:
		# 	target = self.game.getSprites('powered')[0]
		goalX, goalY = self.get_sprite_tile_position(target)
		goalIndex = self.get_index(goalX, goalY)
		goalNode = AStarNode(goalIndex, target)
		
		# logToFile('Goal: (%s,%s) --> (%s, %s)' %(tileX, tileY, goalX, goalY))
		return self.search(startNode, goalNode)

	def search(self, start, goal):
		# Initialize the variables.
		error = 1
		closedset = []
		openset = []
		came_from = {}
		g_score = {}
		f_score = {}

		openset = [start]
		
		g_score[start.index] = 0
		f_score[start.index] = g_score[start.index] + self.h(start, goal) 	# Score of start index is the distance between start to goal
		while openset:
			# print 'searching for path'
			current = self.get_lowest_f(openset, f_score) 					# Get node with lowest distance to goal
			# Reached the goal

			if current.index >= goal.index - error and current.index <= goal.index + error:

				# print came_from
				self.game.screen.fill((0, 0, 0))
				path = self.reconstruct_path(current)

				# path_sprites = [node.sprite for node in path]
				# pathh = map(self.get_sprite_tile_position, path_sprites)
				# print pathh

				for node in path[::3]:
					pygame.draw.rect(self.game.screen, (0, 255, 0), node.sprite.rect)
					pygame.display.flip()
				return path[::3]

			openset.remove(current)
			closedset.append(current)

			for neighbor in self.neighbor_nodes(current):
				temp_g = g_score[current.index] + self.distance(current, neighbor)				# g_score is the distance from the start node
				if self.nodeInSet(neighbor, closedset) and temp_g >= g_score[neighbor.index]:
					continue
				# New node 
				if not self.nodeInSet(neighbor, openset) or temp_g < g_score[neighbor.index]:
					neighbor.parent = current
					# print 'came_from[%s]=%s' % (self.get_tile_from_index(neighbor.index), self.get_tile_from_index(current.index))
					g_score[neighbor.index] = temp_g
					f_score[neighbor.index] = g_score[neighbor.index] + self.h(neighbor, goal)
					if neighbor not in openset:
						openset.append(neighbor)
		# print 'no path found'
		return []

	def nodeInSet(self, node, nodeSet):
		nodeSetIndices = [n.index for n in nodeSet]
		return node.index in nodeSetIndices






