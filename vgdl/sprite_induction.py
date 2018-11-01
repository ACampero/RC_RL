"""
Sprite Induction
"""
from ontology import *


# Create dictionary with transition updates: (TODO) should we do this manually, or can we do it automatically? 
sprite_types = [Immovable, Passive, Resource, 
ResourcePack, RandomNPC, Chaser, AStarChaser, 
OrientedSprite, Missile]

# Initialize distribution
def initializeDistribution():
	"""
	Creates a uniform distribution over all the sprite types.
	"""
	initial_distribution = {}
	for sprite in sprite_types:
		inital_distribution[sprite] = 1.0/len(sprite_types.keys()) # uniform distribution
	return initial_distribution

# Create function that takes in object last state and new state and updates the object distribution
def updateDistribution(curr_distribution, prev_state, curr_state):
	prev_game, prev_sprite = prev_state
	curr_game, curr_sprite = curr_state

	for sprite in curr_distribution:
		dist = sprite.updateOptions(prev_game)
		if curr_sprite.rect in dist:
			curr_distribution[sprite] *= dist[curr_sprite.rect]
		else:
			curr_distribution[sprite] = 0.0

	return curr_distribution




