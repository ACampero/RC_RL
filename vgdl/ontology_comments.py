'''
Video game description language -- ontology of concepts.

@author: Tom Schaul
'''

from random import choice, random
from math import sqrt
import pygame
from tools import triPoints, unitVector, vectNorm, oncePerStep
from ai import AStarWorld
from IPython import embed

# ---------------------------------------------------------------------
#     Constants
# ---------------------------------------------------------------------
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
RED = (200, 0, 0)
GRAY = (90, 90, 90)
WHITE = (250, 250, 250)
BROWN = (140, 120, 100)
BLACK = (0, 0, 0)
ORANGE = (250, 160, 0)
YELLOW = (250, 250, 0)
PINK = (250, 200, 200)
GOLD = (250, 212, 0)
LIGHTRED = (250, 50, 50)
LIGHTORANGE = (250, 200, 100)
LIGHTBLUE = (50, 100, 250)
LIGHTGREEN = (50, 250, 50)
LIGHTGRAY = (150, 150, 150)
DARKGRAY = (30, 30, 30)
DARKBLUE = (20, 20, 100)
PURPLE = (140, 20, 140)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

BASEDIRS = [UP, LEFT, DOWN, RIGHT]

colorDict = {str((0, 200, 0)): 'GREEN',\
            str((0, 0, 200)): 'BLUE',\
            str((200, 0, 0)): 'RED',\
            str((90, 90, 90)): 'GRAY',\
            str((250, 250, 250)): 'WHITE',\
            str((140, 120, 100)): 'BROWN',\
            str((0, 0, 0)): 'BLACK',\
            str((250, 160, 0)): 'ORANGE',\
            str((250, 250, 0)): 'YELLOW',\
            str((250, 200, 200)): 'PINK',\
            str((250, 212, 0)): 'GOLD',\
            str((250, 50, 50)): 'LIGHTRED',\
            str((250, 200, 100)): 'LIGHTORANGE',\
            str((50, 100, 250)): 'LIGHTBLUE',\
            str((50, 250, 50)): 'LIGHTGREEN',\
            str((150, 150, 150)): 'LIGHTGRAY',\
            str((30, 30, 30)): 'DARKGRAY',\
            str((20, 20, 100)): 'DARKBLUE',\
            str((140, 20, 140)): 'PURPLE',\
            }


# ---------------------------------------------------------------------
#     Types of physics
# ---------------------------------------------------------------------
class GridPhysics():
    """ Define actions and key-mappings for grid-world dynamics. """

    def passiveMovement(self, sprite):
        # print "passive movement for", sprite.name
        if sprite.speed is None:
            speed = 1
        else:
            speed = sprite.speed
        if speed != 0 and hasattr(sprite, 'orientation'):
            # print "in passive movement update for", sprite.name
            sprite._updatePos(sprite.orientation, speed * self.gridsize[0])
        # else:
            # print "no passive movement update for", sprite.name

    def calculatePassiveMovement(self, sprite):
        # print ""
        # print "in calculate passive movement for", sprite.name
        if sprite.speed is None:
            speed = 1
        else:
            speed = sprite.speed
        if speed != 0 and hasattr(sprite, 'orientation'):
            orientation = sprite.orientation
            speed = speed * self.gridsize[0]
            if not(self.cooldown > self.lastmove + 1 or abs(orientation[0])+abs(orientation[1])==0):
                # print "about to move", sprite.name
                coords = self.rect.move((orientation[0]*speed, orientation[1]*speed))
                return coords
            return sprite.rect
                  

    def activeMovement(self, sprite, action, speed=None):
        # print "action", action
        if speed is None:
            if sprite.speed is None:
                speed = 1
            else:
                speed = sprite.speed
        if speed != 0 and action is not None:
            # print "in active movement update for", sprite.name
            sprite._updatePos(action, speed * self.gridsize[0])
        # else:
            # print "no active movement update for", sprite.name
        # print "after activeMovement", (sprite.rect.left, sprite.rect.top) 
        # print "__"
        return (sprite.rect.left, sprite.rect.right) 

    def calculateActiveMovement(self, sprite, action, speed=None):
        """
        Calculate where the sprite would end up in a timestep, without actually updating its position.
        """
        if speed is None:
            if sprite.speed is None:
                speed = 1
            else:
                speed = sprite.speed
        if speed != 0 and action is not None:
            speed = speed * self.gridsize[0]
            if speed is None:
                speed = sprite.speed
            
            orientation = action

            # print "orientation", orientation, "cooldown", sprite.cooldown, "lastmove", sprite.lastmove
            # print "Before:", (sprite.rect.left, sprite.rect.right)

            ## 1. Why are we not entering this if statement
            ## 2. How does rect.move calculate where it's going to go
            ## 
            ##why is this not running? see activemovement function which is the original function
            ## also see updatepos in core.py Should be the same logic as that function, but this never goes into the if
            ## statement.
            ## how does rect.move work? we seem to not be getting the accurate numbers in terms of what it's calculating.
        
            if not(sprite.cooldown > sprite.lastmove + 1 or abs(orientation[0])+abs(orientation[1])==0): 
                # print "pseudo-updating pos for", sprite.ID

                # print "original coords", sprite.rect[0], sprite.rect[1]
                coords = round(sprite.rect[0]+orientation[0]*speed), round(sprite.rect[1]+orientation[1]*speed)
                # print "new coords", coords
                # coords = (sprite.rect.left+orientation[0]*speed, sprite.rect.right+orientation[1]*speed)
                # print "Our calculation:", (int(coords[0]), int(coords[1])) #TODO: Why are these different?
                # coords = sprite.rect.move((orientation[0]*speed, orientation[1]*speed))
                # print "Actual position:", (int(coords.left), int(coords.right))

                return coords
            else:
                # print "no active movement calculation"
                coords = (sprite.rect.left, sprite.rect.top)
        # print "Potential after 3:", (int(coords[0]), int(coords[1])) #TODO: Why are these different?

        return (sprite.rect[0], sprite.rect[1]) ##changed. original was returning sprite.rect

    def distance(self, r1, r2):
        """ Grid physics use Hamming distances. """
        return (abs(r1.top - r2.top)
                + abs(r1.left - r2.left))


class ContinuousPhysics(GridPhysics):
    gravity = 0.
    friction = 0.02

    def passiveMovement(self, sprite):
        if sprite.speed != 0 and hasattr(sprite, 'orientation'):
            sprite._updatePos(sprite.orientation, sprite.speed)
            if self.gravity > 0 and sprite.mass > 0:
                self.activeMovement(sprite, (0, self.gravity * sprite.mass))
            sprite.speed *= (1 - self.friction)

    def activeMovement(self, sprite, action, speed=None):
        """ Here the assumption is that the controls determine the direction of
        acceleration of the sprite. """
        if speed is None:
            speed = sprite.speed
        v1 = action[0] / float(sprite.mass) + sprite.orientation[0] * speed
        v2 = action[1] / float(sprite.mass) + sprite.orientation[1] * speed
        sprite.orientation = unitVector((v1, v2))
        sprite.speed = vectNorm((v1, v2)) / vectNorm(sprite.orientation)

    def distance(self, r1, r2):
        """ Continuous physics use Euclidean distances. """
        return sqrt((r1.top - r2.top) ** 2
                    + (r1.left - r2.left) ** 2)

class NoFrictionPhysics(ContinuousPhysics):
    friction = 0

class GravityPhysics(ContinuousPhysics):
    gravity = 0.8


# ---------------------------------------------------------------------
#     Sprite types
# ---------------------------------------------------------------------
from core import VGDLSprite, Resource
'''
In updateOptions function, object_info has form 
    {'position':(ob.rect.left, ob.rect.right), 'features':features, 'type': type_vector}
'''
class Immovable(VGDLSprite):
    """ A gray square that does not budge. """
    color = GRAY #TODO: can these be commented out?
    is_static = True

    def updateOptions(self, game):
        return {(self.rect.left, self.rect.top):1.}

class Passive(VGDLSprite):
    """ A square that may budge. """
    color = RED

    def updateOptions(self, game):
        return {(self.rect.left, self.rect.top):1.}

class ResourcePack(Resource):
    """ Can be collected, and in that case adds/increases a progress bar on the collecting sprite.
    Multiple resource packs can refer to the same type of base resource. """
    is_static = True

    def updateOptions(self, game):
        return {(self.rect.left, self.rect.top):1.}

class Flicker(VGDLSprite):
    """ A square that persists just a few timesteps. """
    color = RED
    limit = 1
    def __init__(self, **kwargs):
        self._age = 0
        VGDLSprite.__init__(self, **kwargs)

    def update(self, game):
        VGDLSprite.update(self, game)
        if self._age > self.limit:
            killSprite(self, None, game)
        self._age += 1

class Spreader(Flicker):
    """ Spreads to its four canonical neighbor positions, and replicates itself there,
    if these are unoccupied. """
    spreadprob = 1.
    def update(self, game):
        Flicker.update(self, game)
        if self._age == 2:
            for u in BASEDIRS:
                if random() < self.spreadprob:
                    game._createSprite([self.name], (self.lastrect.left + u[0] * self.lastrect.size[0],
                                                     self.lastrect.top + u[1] * self.lastrect.size[1]))

class SpriteProducer(VGDLSprite):
    """ Superclass for all sprites that may produce other sprites, of type 'stype'. """
    stype = None

class Portal(SpriteProducer):
    is_static = True
    color = BLUE

class SpawnPoint(SpriteProducer):
    prob = None
    total = None
    color = BLACK
    cooldown = None
    is_static = True
    def __init__(self, cooldown=1, prob=1, total=None, **kwargs):
        SpriteProducer.__init__(self, **kwargs)
        if prob:
            self.prob = prob
            self.is_stochastic = (prob > 0 and prob < 1)
        if cooldown:
            self.cooldown = cooldown
        if total:
            self.total = total
        self.counter = 0

    def update(self, game):
        if (game.time % self.cooldown == 0 and random() < self.prob):
            game._createSprite([self.stype], (self.rect.left, self.rect.top))
            self.counter += 1

        if self.total and self.counter >= self.total:
            killSprite(self, None, game)

class RandomNPC(VGDLSprite):
    """ Chooses randomly from all available actions each step. """
    speed = 1
    is_stochastic = True

    def update(self, game):
        VGDLSprite.update(self, game)
        self.direction = choice(BASEDIRS)
        self.physics.activeMovement(self, self.direction)

    def updateOptions(self, game):
        position_options = {}
        options = BASEDIRS
    
        for option in options:

            left, top = self.physics.calculateActiveMovement(self, option)
            if (left, top) in position_options.keys():
                position_options[(left, top)] += 1.0/len(options) 
            else:
                position_options[(left, top)] = 1.0/len(options)
        
        return position_options


class OrientedSprite(VGDLSprite): ##
    """ A sprite that maintains the current orientation. """
    draw_arrow = False
    orientation = RIGHT

    def _draw(self, game):
        """ With a triangle that shows the orientation. """
        VGDLSprite._draw(self, game)
        if self.draw_arrow:
            col = (self.color[0], 255 - self.color[1], self.color[2])
            pygame.draw.polygon(game.screen, col, triPoints(self.rect, unitVector(self.orientation)))


class Conveyor(OrientedSprite):
    """ A static object that used jointly with the 'conveySprite' interaction to move
    other sprites around."""
    is_static = True
    color = BLUE
    strength = 1
    draw_arrow = True

class Missile(OrientedSprite): ##
    """ A sprite that constantly moves in the same direction. """
    speed = 1
    color = PURPLE

class OrientedFlicker(OrientedSprite, Flicker):
    """ Preserves directionality """
    draw_arrow = True
    speed = 0

class Walker(Missile):
    """ Keep moving in the current horizontal direction. If stopped, pick one randomly. """
    airsteering = False
    is_stochastic = True
    def update(self, game):
        if self.airsteering or self.lastdirection[0] == 0:
            if self.orientation[0] > 0:
                d = 1
            elif self.orientation[0] < 0:
                d = -1
            else:
                d = choice([-1, 1])
            self.physics.activeMovement(self, (d, 0))
        Missile.update(self, game)

class WalkJumper(Walker):
    prob = 0.1
    strength = 10
    def update(self, game):
        if self.lastdirection[0] == 0:
            if self.prob < random():
                self.physics.activeMovement(self, (0, -self.strength))
        Walker.update(self, game)

class RandomInertial(OrientedSprite, RandomNPC):
    physicstype = ContinuousPhysics

class RandomMissile(Missile):
    def __init__(self, **kwargs):
        Missile.__init__(self, orientation=choice(BASEDIRS),
                         speed=choice([0.1, 0.2, 0.4]), **kwargs)

class ErraticMissile(Missile):
    """ A missile that randomly changes direction from time to time.
    (with probability 'prob' per timestep). """
    def __init__(self, prob=0.1, **kwargs):
        Missile.__init__(self, orientation=choice(BASEDIRS), **kwargs)
        self.prob = prob
        self.is_stochastic = (prob > 0 and prob < 1)

    def update(self, game):
        Missile.update(self, game)
        if random() < self.prob: 
            self.orientation = choice(BASEDIRS)

class Bomber(SpawnPoint, Missile):
    color = ORANGE
    is_static = False
    def update(self, game):
        Missile.update(self, game)
        SpawnPoint.update(self, game)

class Chaser(RandomNPC): ##
    """ Pick an action that will move toward the closest sprite of the provided target type. """
    stype = None
    fleeing = False

    def _closestTargets(self, game):
        bestd = 1e100
        res = []
        for target in game.getSprites(self.stype):
            d = self.physics.distance(self.rect, target.rect)
            if d < bestd:
                bestd = d
                res = [target]
            elif d == bestd:
                res.append(target)
        return res

    def _movesToward(self, game, target):
        """ Find the canonical direction(s) which move toward
        the target. """
        res = []
        basedist = self.physics.distance(self.rect, target.rect)
        for a in BASEDIRS:
            r = self.rect.copy()
            r = r.move(a)
            newdist = self.physics.distance(r, target.rect)
            if self.fleeing and basedist < newdist:
                res.append(a)
            if not self.fleeing and basedist > newdist:
                res.append(a)
        return res


    def update(self, game):
        VGDLSprite.update(self, game) # This makes the sprite start to move?

        options = []
        position_options = {}

        for target in self._closestTargets(game):
            options.extend(self._movesToward(game, target))
        if len(options) == 0:
            options = BASEDIRS
        # c = choice(options)
        # print "CHOICE: ", c

        for option in options:
            pos = self.physics.activeMovement(self, option)
            if pos in position_options.keys():
                position_options[pos] += 1.0/len(options) 
            else:
                position_options[pos] = 1.0/len(options)
        
        return position_options

    def updateOptions(self, game): #TODO: Need to make sure to feed in a copy of the game, so as not to actually update the current game? 
        VGDLSprite.update(self, game)
        options = []
        position_options = {}
        for target in self._closestTargets(game):
            options.extend(self._movesToward(game, target))
        if len(options) == 0:
            options = BASEDIRS

        for option in options:
            left, top = self.physics.calculateActiveMovement(self, option)
            if (left, top) in position_options.keys():
                position_options[(left, top)] += 1.0/len(options) 
            else:
                position_options[(left, top)] = 1.0/len(options)
        
        return position_options
        


class Fleeing(Chaser):
    """ Just reversing directions"""
    fleeing = True

class AStarChaser(RandomNPC): ##
    """ Move towards the character using A* search. """
    stype = None
    fleeing = False
    drawpath = None
    walkableTiles = None
    neighborNodes = None
    
    def _movesToward(self, game, target):
        """ Find the canonical direction(s) which move toward
            the target. """
        res = []
        basedist = self.physics.distance(self.rect, target.rect)
        for a in BASEDIRS:
            r = self.rect.copy()
            r = r.move(a)
            newdist = self.physics.distance(r, target.rect)
            if self.fleeing and basedist < newdist:
                res.append(a)
            if not self.fleeing and basedist > newdist:
                res.append(a)
        return res

    def _draw(self, game):
        """ With a triangle that shows the orientation. """
        RandomNPC._draw(self, game)
        
        if self.walkableTiles:
            col = pygame.Color(0, 0, 255, 100)
            for sprite in self.walkableTiles:
                pygame.draw.rect(game.screen, col, sprite.rect)
        
        if self.neighborNodes:
            #logToFile("len(neighborNodes)=%s" %len(self.neighborNodes))
            col = pygame.Color(0, 255, 255, 80)
            for node in self.neighborNodes:
                pygame.draw.rect(game.screen, col, node.sprite.rect)
    
        if self.drawpath:
            col = pygame.Color(0, 255, 0, 120)
            for sprite in self.drawpath[1:-1]:
                pygame.draw.rect(game.screen, col, sprite.rect)

    def _setDebugVariables(self, world, path):
        '''
            Sets the variables required for debug drawing of the paths
            resulting from the A-Star search.
            '''
        
        path_sprites = [node.sprite for node in path]
        
        self.walkableTiles = world.get_walkable_tiles()
        self.neighborNodes = world.neighbor_nodes_of_sprite(self)
        self.drawpath = path_sprites
    
    def update(self, game):
        VGDLSprite.update(self, game)
        
        world = AStarWorld(game)
        path = world.getMoveFor(self)
        
        # Uncomment below to draw debug paths.
        # self._setDebugVariables(world,path)
        
        if len(path)>1:
            move = path[1]
            
            nextX, nextY = world.get_sprite_tile_position(move.sprite)
            nowX, nowY = world.get_sprite_tile_position(self)
            
            movement = None
            
            if nowX == nextX:
                if nextY > nowY:
                    #logToFile('DOWN')
                    movement = DOWN
                else:
                    #logToFile('UP')
                    movement = UP
            else:
                if nextX > nowX:
                    #logToFile('RIGHT')
                    movement = RIGHT
                else:
                    #logToFile('LEFT')
                    movement = LEFT
                    
        self.physics.activeMovement(self, movement)

    def updateOptions(self, game):
        VGDLSprite.update(self, game)  #TODO: Need to make sure to feed in a copy of the game, so as not to actually update the current game? 

        world = AStarWorld(game)
        path = world.getMoveFor(self)
        
        # Uncomment below to draw debug paths.
        # self._setDebugVariables(world,path)
        
        if len(path)>1:
            move = path[1]
            
            nextX, nextY = world.get_sprite_tile_position(move.sprite)
            nowX, nowY = world.get_sprite_tile_position(self)
            
            movement = None
            
            if nowX == nextX:
                if nextY > nowY:
                    #logToFile('DOWN')
                    movement = DOWN
                else:
                    #logToFile('UP')
                    movement = UP
            else:
                if nextX > nowX:
                    #logToFile('RIGHT')
                    movement = RIGHT
                else:
                    #logToFile('LEFT')
                    movement = LEFT
        left, top = self.physics.calculateActiveMovement(self, movement)
        return {(left, top): 1.} 


# ---------------------------------------------------------------------
#     Avatars: player-controlled sprite types
# ---------------------------------------------------------------------
from core import Avatar

class MovingAvatar(VGDLSprite, Avatar):
    """ Default avatar, moves in the 4 cardinal directions. """
    color = WHITE
    speed = 1
    is_avatar = True
    alternate_keys=False


    def declare_possible_actions(self):
        from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN
        actions = {}
        actions["UP"] = K_UP
        actions["DOWN"] = K_DOWN
        actions["LEFT"] = K_LEFT
        actions["RIGHT"] = K_RIGHT
        return actions

    def _readAction(self, game):
        actions = self._readMultiActions(game)
        if actions:
            return actions[0]
        else:
            return None

    def _readMultiActions(self, game):
        """ Read multiple simultaneously pressed button actions. """
        from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, K_a, K_s, K_d, K_w
        res = []
        # res += [RIGHT]
        if self.alternate_keys:
            if   game.keystate[K_d]: res += [RIGHT]
            elif game.keystate[K_a]:  res += [LEFT]
            if   game.keystate[K_w]:    res += [UP]
            elif game.keystate[K_s]:  res += [DOWN]
        else:
            if   game.keystate[K_RIGHT]: res += [RIGHT]
            elif game.keystate[K_LEFT]:  res += [LEFT]
            if   game.keystate[K_UP]:    res += [UP]
            elif game.keystate[K_DOWN]:  res += [DOWN]
        return res

    def update(self, game):
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action:
            self.physics.activeMovement(self, action)

class HorizontalAvatar(MovingAvatar):
    """ Only horizontal moves.  """

    def declare_possible_actions(self):
        from pygame.locals import K_LEFT, K_RIGHT
        actions = {}
        actions["LEFT"] = K_LEFT
        actions["RIGHT"] = K_RIGHT
        return actions


    def update(self, game):
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action in [RIGHT, LEFT]:
            self.physics.activeMovement(self, action)

class VerticalAvatar(MovingAvatar):
    """ Only vertical moves.  """

    def declare_possible_actions(self):
        from pygame.locals import K_UP, K_DOWN
        actions = {}
        actions["UP"] = K_UP
        actions["DOWN"] = K_DOWN
        return actions

    def update(self, game):
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action in [UP, DOWN]:
            self.physics.activeMovement(self, action)

class FlakAvatar(HorizontalAvatar, SpriteProducer):
    """ Hitting the space button creates a sprite of the
    specified type at its location. """

    def declare_possible_actions(self):
        from pygame.locals import K_SPACE
        actions = HorizontalAvatar.declare_possible_actions(self)
        actions["SPACE"] = K_SPACE
        return actions

    color = GREEN
    def update(self, game):
        HorizontalAvatar.update(self, game)
        self._shoot(game)

    def _shoot(self, game):
        from pygame.locals import K_SPACE
        if self.stype and game.keystate[K_SPACE]:
            spawn = game._createSprite([self.stype], (self.rect.left, self.rect.top))
            

class OrientedAvatar(OrientedSprite, MovingAvatar):
    """ Avatar retains its orientation, but moves in cardinal directions. """
    draw_arrow = True
    def update(self, game):
        tmp = self.orientation
        self.orientation = (0, 0)
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action:
            self.physics.activeMovement(self, action)
        d = self.lastdirection
        if sum(map(abs, d)) > 0:
            # only update if the sprite moved.
            self.orientation = d
        else:
            self.orientation = tmp

class RotatingAvatar(OrientedSprite, MovingAvatar):
    """ Avatar retains its orientation, and moves forward/backward or rotates
    relative to that. """
    draw_arrow = True
    speed = 0
    def update(self, game):
        actions = self._readMultiActions(game)
        if UP in actions:
            self.speed = 1
        elif DOWN in actions:
            self.speed = -1
        if LEFT in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i + 1) % len(BASEDIRS)]
        elif RIGHT in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i - 1) % len(BASEDIRS)]
        VGDLSprite.update(self, game)
        self.speed = 0

class RotatingFlippingAvatar(RotatingAvatar):
    """ Uses a different action set: DOWN makes it spin around 180 degrees.
    Optionally, a noise level can be specified
    """

    noiseLevel = 0

    def update(self, game):
        actions = self._readMultiActions(game)
        if len(actions) > 0 and self.noiseLevel > 0:
            # pick a random one instead
            if random() < self.noiseLevel*4:
                actions = [choice([UP, LEFT, DOWN, RIGHT])]
        if UP in actions:
            self.speed = 1
        elif DOWN in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i + 2) % len(BASEDIRS)]
        elif LEFT in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i + 1) % len(BASEDIRS)]
        elif RIGHT in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i - 1) % len(BASEDIRS)]
        VGDLSprite.update(self, game)
        self.speed = 0

    @property
    def is_stochastic(self):
        return self.noiseLevel > 0

class NoisyRotatingFlippingAvatar(RotatingFlippingAvatar):
    noiseLevel = 0.1

class ShootAvatar(OrientedAvatar, SpriteProducer):
    """ Produces a sprite in front of it (e.g., Link using his sword). """
    ammo=None

    def __init__(self, stype=None, **kwargs):
        self.stype = stype
        OrientedSprite.__init__(self, **kwargs)

    def update(self, game):
        OrientedAvatar.update(self, game)
        if self._hasAmmo():
            self._shoot(game)

    def _hasAmmo(self):
        if self.ammo is None:
            return True
        elif self.ammo in self.resources:
            return self.resources[self.ammo] > 0
        return False

    def _reduceAmmo(self):
        if self.ammo is not None and self.ammo in self.resources:
            self.resources[self.ammo] -= 1

    def _shoot(self, game):
        from pygame.locals import K_SPACE
        if self.stype and game.keystate[K_SPACE]:
            u = unitVector(self.orientation)
            newones = game._createSprite([self.stype], (self.lastrect.left + u[0] * self.lastrect.size[0],
                                                       self.lastrect.top + u[1] * self.lastrect.size[1]))
            if len(newones) > 0  and isinstance(newones[0], OrientedSprite):
                newones[0].orientation = unitVector(self.orientation)
            self._reduceAmmo()


class AimedAvatar(ShootAvatar):
    """ Can change the direction of firing, but not move. """
    speed=0
    angle_diff=0.05
    def update(self, game):
        VGDLSprite.update(self, game)
        self._aim(game)
        self._shoot(game)

    def _aim(self, game):
        action = self._readAction(game)
        if action in [UP, DOWN]:
            if action == DOWN:
                angle = self.angle_diff
            else:
                angle = -self.angle_diff
            from math import cos, sin
            self.orientation = unitVector((self.orientation[0]*cos(angle)-self.orientation[1]*sin(angle),
                                           self.orientation[0]*sin(angle)+self.orientation[1]*cos(angle)))

class AimedFlakAvatar(AimedAvatar):
    """ Can move left and right """
    only_active=True
    speed=None

    def update(self, game):
        AimedAvatar.update(self, game)
        action = self._readAction(game)
        if action in [RIGHT, LEFT]:
            self.physics.activeMovement(self, action)

class InertialAvatar(OrientedAvatar):
    speed = 1
    physicstype = ContinuousPhysics
    def update(self, game):
        MovingAvatar.update(self, game)

class MarioAvatar(InertialAvatar):
    """ Mario can have two states: in contact with the ground, or in parabolic flight. """
    physicstype = GravityPhysics
    draw_arrow = False
    strength = 10
    airsteering = False
    def update(self, game):
        action = self._readAction(game)
        if action is None:
            action = (0, 0)
        from pygame.locals import K_SPACE
        if game.keystate[K_SPACE] and self.orientation[1] == 0:
            action = (action[0] * sqrt(self.strength), -self.strength)
        elif self.orientation[1] == 0 or self.airsteering:
            action = (action[0] * sqrt(self.strength), 0)
        else:
            action = (0, 0)
        self.physics.activeMovement(self, action)
        VGDLSprite.update(self, game)



# ---------------------------------------------------------------------
#     Termination criteria
# ---------------------------------------------------------------------
from core import Termination

class Timeout(Termination):
    def __init__(self, limit=0, win=False):
        self.limit = limit
        self.win = win

    def isDone(self, game):
        if game.time >= self.limit:
            return True, self.win
        else:
            return False, None

class SpriteCounter(Termination):
    """ Game ends when the number of sprites of type 'stype' hits 'limit' (or below). """
    def __init__(self, limit=0, stype=None, win=True):
        self.limit = limit
        self.stype = stype
        self.win = win

    def isDone(self, game):
        if game.numSprites(self.stype) <= self.limit:
            return True, self.win
        else:
            return False, None

class MultiSpriteCounter(Termination):
    """ Game ends when the sum of all sprites of types 'stypes' hits 'limit'. """
    def __init__(self, limit=0, win=True, **kwargs):
        self.limit = limit
        self.win = win
        self.stypes = kwargs.values()

    def isDone(self, game):
        if sum([game.numSprites(st) for st in self.stypes]) == self.limit:
            return True, self.win
        else:
            return False, None

# ---------------------------------------------------------------------
#     Helper functions
# ---------------------------------------------------------------------
def getColor(sprite):
    try:
        color_tuple = str( sprite.color)
        try: 
            return colorDict[color_tuple]
        except KeyError:
            return color_tuple

    except AttributeError:
        return None

# ---------------------------------------------------------------------
#     Effect types (invoked after an event).
# ---------------------------------------------------------------------
def killSprite(sprite, partner, game): ## FLAG
    """ Kill command """
    game.kill_list.append(sprite)
    if not None in {sprite, partner}:
        # sprite_info = colorDict[str(sprite.color)]
        # partner_info = colorDict[str(partner.color)]
        return ("killSprite",sprite.ID ,partner.ID) # partner = agent, sprite = what's being killed


def cloneSprite(sprite, partner, game):
    game._createSprite([sprite.name], (sprite.rect.left, sprite.rect.top))

def transformTo(sprite, partner, game, stype='wall'):
    newones = game._createSprite([stype], (sprite.rect.left, sprite.rect.top))
   
    if len(newones) > 0:
        if isinstance(sprite, OrientedSprite) and isinstance(newones[0], OrientedSprite):
            newones[0].orientation = sprite.orientation
        killSprite(sprite, partner, game)
    return ("transformTo",sprite.ID,partner.ID)

def stepBack(sprite, partner, game): 
    """ Revert last move. """
    sprite.rect = sprite.lastrect
    return ("stepBack", sprite.ID, partner.ID)

def undoAll(sprite, partner, game):
    """ Revert last moves of all sprites. """

    return ('undoAll', sprite.ID , partner.ID)

def bounceForward(sprite, partner, game): # FLAG
    """ The partner sprite pushed, so if possible move in the opposite direction. """
    # print "in beginning of bounceForward"
    # embed()
    sprite.physics.activeMovement(sprite, unitVector(partner.lastdirection))
    game._updateCollisionDict(sprite)
    # print "in second part of bounceForward"
    # embed()
    # return ('bounceForward', colorDict[str(partner.color)], colorDict[str(sprite.color)])
    return ('bounceForward', sprite.ID, partner.ID)


def conveySprite(sprite, partner, game):
    """ Moves the partner in target direction by some step size. """
    tmp = sprite.lastrect
    v = unitVector(partner.orientation)
    sprite.physics.activeMovement(sprite, v, speed=partner.strength)
    sprite.lastrect = tmp
    game._updateCollisionDict(sprite)
    # return ('conveySprite', colorDict[str(sprite.color)], colorDict[str(partner.color)])
    return ('conveySprite', sprite.ID, partner.ID)

def windGust(sprite, partner, game):
    """ Moves the partner in target direction by some step size, but stochastically
    (step, step-1 and step+1 are equally likely) """
    s = choice([partner.strength, partner.strength + 1, partner.strength - 1])
    if s != 0:
        tmp = sprite.lastrect.copy()
        v = unitVector(partner.orientation)
        sprite.physics.activeMovement(sprite, v, speed=s)
        sprite.lastrect = tmp
        game._updateCollisionDict(sprite)
        # return ("windGust", colorDict[str(sprite.color)], colorDict[str(partner.color)])
        return ('windGust', sprite.ID, partner.ID)

def slipForward(sprite, partner, game, prob=0.5):
    """ Slip forward in the direction of the current orientation, sometimes."""
    if prob > random():
        tmp = sprite.lastrect
        v = unitVector(sprite.orientation)
        sprite.physics.activeMovement(sprite, v, speed=1)
        sprite.lastrect = tmp
        game._updateCollisionDict(sprite)
        # return ("slipForward" , colorDict[str(sprite.color)], colorDict[str(partner.color)])
        return ('slipForward', sprite.ID, partner.ID)

def attractGaze(sprite, partner, game, prob=0.5):
    """ Turn the orientation to the value given by the partner. """
    if prob > random():
        sprite.orientation = partner.orientation
        # return ("attractGaze" , colorDict[str(sprite.color)], colorDict[str(partner.color)])
        return ('attractGaze', sprite.ID, partner.ID)

def turnAround(sprite, partner, game):
    sprite.rect = sprite.lastrect
    sprite.lastmove = sprite.cooldown
    sprite.physics.activeMovement(sprite, DOWN)
    sprite.lastmove = sprite.cooldown
    sprite.physics.activeMovement(sprite, DOWN)
    reverseDirection(sprite, partner, game)
    game._updateCollisionDict(sprite)
    return ('turnAround', sprite.ID, partner.ID)

def reverseDirection(sprite, partner, game): # FLAG
    sprite.orientation = (-sprite.orientation[0], -sprite.orientation[1])

    return ('reverseDirection', sprite.ID, partner.ID)


def flipDirection(sprite, partner, game): # FLAG
    sprite.orientation = choice(BASEDIRS)

    return ('flipDirection' , sprite.ID, partner.ID)

def bounceDirection(sprite, partner, game, friction=0): # FLAG
    """ The centers of the objects determine the direction"""
    stepBack(sprite, partner, game)
    inc = sprite.orientation
    snorm = unitVector((-sprite.rect.centerx + partner.rect.centerx,
                        - sprite.rect.centery + partner.rect.centery))
    dp = snorm[0] * inc[0] + snorm[1] * inc[1]
    sprite.orientation = (-2 * dp * snorm[0] + inc[0], -2 * dp * snorm[1] + inc[1])
    sprite.speed *= (1. - friction)
    return ('bounceDirection' , sprite.ID, partner.ID)


def wallBounce(sprite, partner, game, friction=0): # FLAG
    """ Bounce off orthogonally to the wall. """
    if not oncePerStep(sprite, game, 'lastbounce'):
        return
    sprite.speed *= (1. - friction)
    stepBack(sprite, partner, game)
    if abs(sprite.rect.centerx - partner.rect.centerx) > abs(sprite.rect.centery - partner.rect.centery):
        sprite.orientation = (-sprite.orientation[0], sprite.orientation[1])
    else:
        sprite.orientation = (sprite.orientation[0], -sprite.orientation[1])
    # return ('wallBounce', colorDict[str(partner.color)], colorDict[str(sprite.color)])
    ## TODO: Not printing for now   
    #return ('wallBounce' , sprite.ID, partner.ID)

def wallStop(sprite, partner, game, friction=0): # FLAG
    """ Stop just in front of the wall, removing that velocity component,
    but possibly sliding along it. """
    if not oncePerStep(sprite, game, 'laststop'):
        return
    stepBack(sprite, partner, game)
    if abs(sprite.rect.centerx - partner.rect.centerx) > abs(sprite.rect.centery - partner.rect.centery):
        sprite.orientation = (0, sprite.orientation[1] * (1. - friction))
    else:
        sprite.orientation = (sprite.orientation[0] * (1. - friction), 0)
    sprite.speed = vectNorm(sprite.orientation) * sprite.speed
    sprite.orientation = unitVector(sprite.orientation)
    ## TODO: Not printing for now   
    #return ('wallStop' , sprite.ID, partner.ID)

def killIfSlow(sprite, partner, game, limitspeed=1):
    """ Take a decision based on relative speed. """
    if sprite.is_static:
        relspeed = partner.speed
    elif partner.is_static:
        relspeed = sprite.speed
    else:
        relspeed = vectNorm((sprite._velocity()[0] - partner._velocity()[0],
                             sprite._velocity()[1] - partner._velocity()[1]))
    if relspeed < limitspeed:
        killSprite(sprite, partner, game)
        # return ('killIfSlow' , sprite.ID, partner.ID)


def killIfFromAbove(sprite, partner, game):
    """ Kills the sprite, only if the other one is higher and moving down. """
    if (sprite.lastrect.top > partner.lastrect.top
        and partner.rect.top > partner.lastrect.top):
        killSprite(sprite, partner, game)
        return ('killIfFromAbove' , parter.ID, sprite.ID)

def killIfAlive(sprite, partner, game):
    """ Perform the killing action, only if no previous collision effect has removed the partner. """
    if partner not in game.kill_list:
        killSprite(sprite, partner, game)
        return ('killIfAlive' , sprite.ID, partner.ID)

def collectResource(sprite, partner, game): # FLAG
    """ Adds/increments the resource type of sprite in partner """
    assert isinstance(sprite, Resource)
    r = sprite.resourceType
    partner.resources[r] = max(-1, min(partner.resources[r]+sprite.value, game.resources_limits[r]))
    #print 'Collected ', colorDict[str(sprite.color)]#partner.resources[r]
    # return ('collectResource', colorDict[str(partner.color)], colorDict[str(sprite.color)])
    return ('collectResource' , sprite.ID, partner.ID)

def changeResource(sprite, partner, resourceColor, game, resource, value=1):
    """ Increments a specific resource type in sprite """
    sprite.resources[resource] = max(-1, min(sprite.resources[resource]+value, game.resources_limits[resource]))
    #print resource, sprite.resources[resource]
    #print 'Changed ', colorDict[str(partner.color)]

    # NOTE: partner is the color of the resource (see _eventHandling() in core.py)
    # return ('changeResource', colorDict[str(sprite.color)], colorDict[str(partner.color)], resource, value)
    return ('changeResource' , sprite.ID, partner.ID)

def spawnIfHasMore(sprite, partner, game, resource, stype, limit=1):
    """ If 'sprite' has more than a limit of the resource type given, it spawns a sprite of 'stype'. """
    if sprite.resources[resource] >= limit:
        game._createSprite([stype], (sprite.rect.left, sprite.rect.top))
        return ('spawnIfHasMore', sprite.ID, partner.ID) ### NOTE - there is no default 'spawn' function we could return instead, but we should then make one

def killIfHasMore(sprite, partner, game, resource, limit=1):
    """ If 'sprite' has more than a limit of the resource type given, it dies. """
    if sprite.resources[resource] >= limit:
        killSprite(sprite, partner, game)
        # return ('killIfHasMore' , sprite.ID, partner.ID)

def killIfOtherHasMore(sprite, partner, game, resource, limit=1):
    """ If 'partner' has more than a limit of the resource type given, sprite dies. """
    if partner.resources[resource] >= limit:
        killSprite(sprite, partner, game)
        # return ('killIfOtherHasMore' , sprite.ID, partner.ID)

def killIfHasLess(sprite, partner, game, resource, limit=1):
    """ If 'sprite' has less than a limit of the resource type given, it dies. """
    if sprite.resources[resource] <= limit:
        killSprite(sprite, partner, game)
        # return ('killIfHasLess' , sprite.ID, partner.ID)

def killIfOtherHasLess(sprite, partner, game, resource, limit=1):
    """ If 'partner' has less than a limit of the resource type given, sprite dies. """
    if partner.resources[resource] <= limit:
        killSprite(sprite, partner, game)
        # return ('killIfOtherHasLess' , sprite.ID, partner.ID)

def wrapAround(sprite, partner, game, offset=0):
    """ Move to the edge of the screen in the direction the sprite is coming from.
    Plus possibly an offset. """
    if sprite.orientation[0] > 0:
        sprite.rect.left = offset * sprite.rect.size[1]
    elif sprite.orientation[0] < 0:
        sprite.rect.left = game.screensize[0] - sprite.rect.size[0] * (1 + offset)
    if sprite.orientation[1] > 0:
        sprite.rect.top = offset * sprite.rect.size[1]
    elif sprite.orientation[1] < 0:
        sprite.rect.top = game.screensize[1] - sprite.rect.size[1] * (1 + offset)
    sprite.lastmove = 0
    # return ('wrapAround' , sprite.ID, partner.ID)

def pullWithIt(sprite, partner, game):
    """ The partner sprite adds its movement to the sprite's. """
    if not oncePerStep(sprite, game, 'lastpull'):
        return
    tmp = sprite.lastrect
    v = unitVector(partner.lastdirection)
    sprite._updatePos(v, partner.speed * sprite.physics.gridsize[0])
    if isinstance(sprite.physics, ContinuousPhysics):
        sprite.speed = partner.speed
        sprite.orientation = partner.lastdirection
    sprite.lastrect = tmp
    return ('pullWithIt' , sprite.ID, partner.ID)

def teleportToExit(sprite, partner, game):
    e = choice(game.sprite_groups[partner.stype])
    sprite.rect = e.rect
    sprite.lastmove = 0
    return ('teleportToExit', sprite.ID, partner.ID)

# this allows us to determine whether the game has stochastic elements or not
stochastic_effects = [teleportToExit, windGust, slipForward, attractGaze, flipDirection]

# this allows is to determine which effects might kill a sprite
kill_effects = [killSprite, killIfSlow, transformTo, killIfOtherHasLess, killIfOtherHasMore, killIfHasMore, killIfHasLess,
                killIfFromAbove, killIfAlive]




# ---------------------------------------------------------------------
#     Sprite Induction
# ---------------------------------------------------------------------

# Create dictionary with transition updates: (TODO) should we do this manually, or can we do it automatically? 

## some syncing problem between screen time and the internal while loop that's calculating positions

def chaserClosestTargets(sprite, game):
    bestd = 1e100
    res = []
    for target in game.getSprites(sprite.stype):
        d = sprite.physics.distance(sprite.rect, target.rect)
        if d < bestd:
            bestd = d
            res = [target]
        elif d == bestd:
            res.append(target)
    return res

def chaserMovesToward(sprite, game, target):
    """ Find the canonical direction(s) which move toward
    the target. """
    res = []
    basedist = sprite.physics.distance(sprite.rect, target.rect)
    for a in BASEDIRS:
        r = sprite.rect.copy()
        r = r.move(a)
        newdist = sprite.physics.distance(r, target.rect)
        if sprite.fleeing and basedist < newdist:
            res.append(a)
        if not sprite.fleeing and basedist > newdist:
            res.append(a)
    return res


def updateOptions(game, sprite_type, current_sprite):
    """
    game - current game object
    sprite_type - the sprite type class
    current_sprite - the current sprite object
    """

    ##this won't work -- donneed to change objects[sprite]["position"] bc we don't know what 'objects' is from here.

    # if current_sprite.name == "angry": # For debugging
    #     print 
    #     print "Sprite: ", current_sprite.name
    #     print "Sprite type: ", sprite_type
    #     print "Current sprite position: ", objects[sprite]["position"]
    #     print "Options: ", options

    # Immovable, Passive, ResourcePack
    if (sprite_type == Immovable) or (sprite_type == Passive) or (sprite_type == ResourcePack):
        return {(current_sprite.rect.left, current_sprite.rect.top):1.} ##object stays in position

    #why are rect.left and rect.right the position?
    
    # Chaser
    elif sprite_type == Chaser:
        options = []
        position_options = {}

        ## TODO: Why not just use BASEDIRS from the beginning?
        for target in chaserClosestTargets(current_sprite, game):
            options.extend(chaserMovesToward(current_sprite, game, target))
            # print "new options:", options
        if len(options) == 0:
            options = BASEDIRS

        for option in options:
            ## this was rect = current_sprite....
            left, top = current_sprite.physics.calculateActiveMovement(current_sprite, option) #TODO: Check why this calculation isn't correct

            ## was rect.left, rect.top
            if (left, top) in position_options.keys():
                position_options[(left, top)] += 1.0/len(options) 
            else:
                position_options[(left, top)] = 1.0/len(options)

        return position_options

    # VGDLSprite
    else: 
        if not current_sprite.is_static and not current_sprite.only_active:
            rect = current_sprite.physics.calculatePassiveMovement(current_sprite)
            return {(rect.left, rect.top): 1.0}

# Initialize distribution
def initializeDistribution(sprite_types):
    """
    Creates a uniform distribution over all the sprite types.
    """
    initial_distribution = {}
    for sprite in sprite_types:
        initial_distribution[sprite] = 1.0/len(sprite_types) # uniform distribution
    return initial_distribution

# Create function that takes in object last state and new state and updates the object distribution
def updateDistribution(sprite, curr_distribution, movement_options, outcome):
    """
    Updates the sprite distribution for a given object in the game.

    Input:
        sprite - the current sprite ID
        curr_distribution - the current sprite distribution for the given object
        movement_options - possible next locations that the sprite of that sprite type can be in
        outcome - the resulting location that the sprite went to 
    Output:

    """
    for sprite_type in curr_distribution[sprite]:
        if outcome in movement_options[sprite][sprite_type]:
            curr_distribution[sprite_type] *=  movement_options[sprite][sprite_type][outcome]
        else:
            curr_distribution[sprite_type] = 0.0

    ##normalize.

    return curr_distribution

