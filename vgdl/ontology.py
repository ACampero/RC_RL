'''
Video game description language -- ontology of concepts.

@author: Tom Schaul
'''
import random
from random import choice
from copy import deepcopy
from colors import *
import itertools
from math import sqrt
import pygame
import numpy as np
import scipy.stats
from tools import triPoints, unitVector, vectNorm, oncePerStep
from ai import AStarWorld
from IPython import embed
import core
import copy
# import ipdb
import time
from pygame import Rect
from collections import defaultdict
# from line_profiler import LineProfiler

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

BASEDIRS = [UP, LEFT, DOWN, RIGHT]

spriteToParams = {'Resource': [], \
                'ResourcePack': [], \
                'RandomNPC': ['cooldown', 'speed'], \
                'Chaser': ['cooldown', 'fleeing', 'stype', 'speed'], \
                'AStarChaser': ['fleeing', 'speed', 'stype'], \
                'OrientedSprite': ['orientation'], \
                'Missile': ['speed', 'orientation', 'cooldown']} ##removed speed from chaser and randomNPC

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
            sprite._updatePos(sprite.orientation, speed * self.gridsize[0])

    def calculatePassiveMovement(self, sprite):
        """
        Calculate where the sprite would end up in a timestep, without actually updating its position.
        """
        ## This is where you could make hypotheses about speed, etc. for the object.
        if sprite.speed is None:
            speed = 1
        else:
            speed = sprite.speed
        if speed != 0 and hasattr(sprite, 'orientation'):
            orientation = sprite.orientation
            speed = speed * self.gridsize[0]
            if not((sprite.lastmove+1)%sprite.cooldown!=0 or abs(orientation[0])+abs(orientation[1])==0):
            # if not(sprite.cooldown > sprite.lastmove+1 or abs(orientation[0])+abs(orientation[1])==0):
                pos = sprite.rect.move((orientation[0]*speed, orientation[1]*speed))
                return pos.left, pos.top
        else:   # If object has speed = 0 or no 'orientation' attribute
            return None

    def calculatePassiveMovementGivenParams(self, sprite, speed, orientation):
        """
        Calculate where the sprite would end up in a timestep, without actually updating its position.
        """
        ## This is where you could make hypotheses about speed, etc. for the object.
        if speed is None:
            speed = 1

        if speed != 0:
            speed = speed * self.gridsize[0]
            if not((sprite.lastmove+1)%sprite.cooldown!=0 or abs(orientation[0])+abs(orientation[1])==0):
            # if not(sprite.cooldown > sprite.lastmove+1 or abs(orientation[0])+abs(orientation[1])==0):
                pos = sprite.rect.move((orientation[0]*speed, orientation[1]*speed))
                return pos.left, pos.top
            else:
                return sprite.rect.left, sprite.rect.top
        else:
            return None

    def activeMovement(self, sprite, action, speed=None):
        if speed is None:
            if sprite.speed is None:
                speed = 1.0
            else:
                speed = float(sprite.speed)
        if speed != 0 and action is not None:
            sprite._updatePos(action, speed * self.gridsize[0])

    def calculateActiveMovement(self, sprite, action, speed=None, is_chaser=False):
        """
        Calculate where the sprite would end up in a timestep, without actually updating its position.
        """
         ## This is where you could make hypotheses about speed, etc. for the object.
        if action is not None:
            orientation = action
        num = 1 if not is_chaser else 2
        if (sprite.lastmove+num)%sprite.cooldown==0 and abs(orientation[0])+abs(orientation[1])!=0:

            if speed is None:
                if sprite.speed is None:
                    speed = 1
                else:
                    speed = sprite.speed
            if speed != 0:# and action is not None:
                speed = float(speed) * self.gridsize[0]

            pos = sprite.rect.move((orientation[0]*speed, orientation[1]*speed))
            return pos.left, pos.top
        return(sprite.rect.left, sprite.rect.top)

    # using euclidian distance is also used here because it just works better
    # who uses hamming distance for anything where actual distance is needed?
    # No, seriously... I don't want to break anything
    def distance(self, r1, r2):
        """Euclidean distances. """
        if hasattr(r1, 'top'):
            return sqrt((r1.top - r2.top) ** 2
                        + (r1.left - r2.left) ** 2)
        else:
            return sqrt((r1[1]-r2[1])**2
                        + (r1[0]-r2[0])**2)

    # def distance(self, r1, r2):
    #     """ Grid physics use Hamming distances. """
    #     return (abs(r1.top - r2.top)
    #             + abs(r1.left - r2.left))


class ContinuousPhysics(GridPhysics):
    gravity = 0.
    friction = 0.02

    def passiveMovement(self, sprite):
        if sprite.speed != 0 and hasattr(sprite, 'orientation'):
            sprite._updatePos(sprite.orientation, sprite.speed)
            if self.gravity > 0 and sprite.mass > 0:
                self.activeMovement(sprite, (0, self.gravity * sprite.mass))
            sprite.speed *= (1 - self.friction)

    def calculatePassiveMovement(self, sprite):
        if sprite.speed != 0 and hasattr(sprite, 'orientation'):
            if not((sprite.lastmove+1) % sprite.cooldown != 0 or abs(sprite.orientation[0])+abs(sprite.orientation[1])==0):
                pos = sprite.rect.move((sprite.orientation[0]*sprite.speed, sprite.orientation[1]*sprite.speed))
            if self.gravity > 0 and sprite.mass > 0:
                return self.calculateActiveMovement(sprite, (0, self.gravity * sprite.mass))
        else:   # If object has speed = 0 or no 'orientation' attribute
            pos = rect
        return pos.left, pos.top


    def activeMovement(self, sprite, action, speed=None):
        print self.gridsize
        """ Here the assumption is that the controls determine the direction of
        acceleration of the sprite. """
        if speed is None:
            speed = sprite.speed
        v1 = action[0] / float(sprite.mass) + sprite.orientation[0] * speed
        v2 = action[1] / float(sprite.mass) + sprite.orientation[1] * speed
        sprite.orientation = unitVector((v1, v2))
        sprite.speed = vectNorm((v1, v2)) / vectNorm(sprite.orientation)

    def calculateActiveMovement(self, sprite, action, speed=None):
        """ Here the assumption is that the controls determine the direction of
        acceleration of the sprite. """
        if speed is None:
            speed = sprite.speed
        v1 = action[0] / float(sprite.mass) + sprite.orientation[0] * speed * self.gridsize[0]
        v2 = action[1] / float(sprite.mass) + sprite.orientation[1] * speed * self.gridsize[1]
        sprite.orientation = unitVector((v1, v2))
        sprite.speed = vectNorm((v1, v2)) / vectNorm(sprite.orientation)

        return sprite.rect.left, sprite.rect.top

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
# from core import VGDLSprite, Resource
VGDLSprite = core.VGDLSprite
Resource = core.Resource

class Immovable(VGDLSprite):
    """ A gray square that does not budge. """
    color = GRAY #TODO: can these be commented out?
    is_static = True

class Passive(VGDLSprite):
    """ A square that may budge. """
    color = RED

class ResourcePack(Resource):
    """ Can be collected, and in that case adds/increases a progress bar on the collecting sprite.
    Multiple resource packs can refer to the same type of base resource. """
    is_static = True

class Flicker(VGDLSprite):
    """ A square that persists just a few timesteps. """
    color = RED
    limit = 20
    def __init__(self, **kwargs):
        self._age = 0
        VGDLSprite.__init__(self, **kwargs)

    def update(self, game):
        VGDLSprite.update(self, game)
        if self._age >= self.limit:
            game.kill_list.append(self)
            # killSprite(self, None, game)
        else:
            self._age += 1

class Spreader(Flicker):
    """ Spreads to its four canonical neighbor positions, and replicates itself there,
    if these are unoccupied. """
    spreadprob = 1.
    def update(self, game):
        Flicker.update(self, game)
        if self._age == 2:
            for u in BASEDIRS:
                if random.random() < self.spreadprob:
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
    spawnCooldown = None
    is_static = True
    def __init__(self, spawnCooldown=1, prob=1, total=None, **kwargs):
        SpriteProducer.__init__(self, **kwargs)
        if prob:
            self.prob = prob
            self.is_stochastic = (prob > 0 and prob < 1)
        if spawnCooldown:
            self.spawnCooldown = spawnCooldown
        if total:
            self.total = total
        self.counter = 0

    def update(self, game):
        if self.total and self.counter >= self.total:
            killSprite(self, None, game)
            return

        if self.spawnCooldown < 11:
            if ((game.time+1) % self.spawnCooldown == 0 and random.random() < self.prob):
                game._createSprite([self.stype], (self.rect.left, self.rect.top))
                self.counter += 1
        else:
             if ((game.time+1) % self.spawnCooldown == 3 and random.random() < self.prob):
                game._createSprite([self.stype], (self.rect.left, self.rect.top))
                self.counter += 1
        self.lastmove += 1



class RandomNPC(VGDLSprite):
    """ Chooses randomly from all available actions each step. """
    speed = 1
    is_stochastic = True

    def update(self, game):
        self.lastmove -= 1
        VGDLSprite.update(self, game, random_npc=True)
        self.orientation = random.choice(BASEDIRS) #TODO: Make work with random direction
        self.physics.activeMovement(self, self.orientation)
        self.lastmove += 1


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

class Switch(VGDLSprite):
    activated = False
    wait_for_release = False
    can_switch = False
    def __init__(self, **kwargs):
        VGDLSprite.__init__(self, **kwargs)

    def update(self, game):
        if not self.can_switch: return
        from pygame.locals import K_SPACE

        if game.keystate[K_SPACE] and not self.wait_for_release:
            self.activated = True
            self.wait_for_release = True
        else:
            self.activated = False

        if not game.keystate[K_SPACE]:
            self.wait_for_release = False

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
                d = random.choice([-1, 1])
            self.physics.activeMovement(self, (d, 0))
        Missile.update(self, game)

class WalkJumper(Walker):
    prob = 0.1
    strength = 10
    def update(self, game):
        if self.lastdirection[0] == 0:
            if self.prob < random.random():
                self.physics.activeMovement(self, (0, -self.strength))
        Walker.update(self, game)

class RandomInertial(OrientedSprite, RandomNPC):
    physicstype = ContinuousPhysics

class RandomMissile(Missile):
    def __init__(self, **kwargs):
        Missile.__init__(self, orientation=random.choice(BASEDIRS),
                         speed=random.choice([0.1, 0.2, 0.4]), **kwargs)

class ErraticMissile(Missile):
    """ A missile that randomly changes direction from time to time.
    (with probability 'prob' per timestep). """
    def __init__(self, prob=0.1, **kwargs):
        Missile.__init__(self, orientation=random.choice(BASEDIRS), **kwargs)
        self.prob = prob
        self.is_stochastic = (prob > 0 and prob < 1)

    def update(self, game):
        Missile.update(self, game)
        if random.random() < self.prob:
            self.orientation = random.choice(BASEDIRS)

class Bomber(SpawnPoint, Missile):
    color = ORANGE
    is_static = False
    #lastmove = 0
    def update(self, game):
        self.lastmove -= 1
        Missile.update(self, game)
        SpawnPoint.update(self, game)


class Chaser(RandomNPC): ##
    """ Pick an action that will move toward the closest sprite of the provided target type. """
    stype = None
    fleeing = False

    # is_stochastic=False
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
        VGDLSprite.update(self, game) # This increments self.lastmove by 1

        options = []
        position_options = {}

        for target in self._closestTargets(game):
            options.extend(self._movesToward(game, target))
        if len(options) == 0:
            options = BASEDIRS
        # self.physics.activeMovement(self, options[0])
        self.physics.activeMovement(self, random.choice(options))


class Fleeing(Chaser):
    """ Just reversing directions"""
    fleeing = True

class AStarChaser(VGDLSprite): ##
    """ Move towards the character using A* search. """
    stype = None
    speed = .1
    fleeing = False
    drawpath = None
    walkableTiles = None
    neighborNodes = None
    path = []
    next_move = None
    last_move = None

    # def _movesToward(self, game, target):
    #     print target
    #     """ Find the canonical direction(s) which move toward
    #         the target. """
    #     res = []
    #     basedist = self.physics.distance(self.rect, target.rect)
    #     for a in BASEDIRS:
    #         r = self.rect.copy()
    #         r = r.move(a)
    #         newdist = self.physics.distance(r, target.rect)
    #         if self.fleeing and basedist < newdist:
    #             res.append(a)
    #         if not self.fleeing and basedist > newdist:
    #             res.append(a)
    #     return res

    # def _draw(self, game):
    #     """ With a triangle that shows the orientation. """
    #     RandomNPC._draw(self, game)

    #     if self.walkableTiles:
    #         col = pygame.Color(0, 0, 255, 100)
    #         for sprite in self.walkableTiles:
    #             pygame.draw.rect(game.screen, col, sprite.rect)

    #     if self.neighborNodes:
    #         #logToFile("len(neighborNodes)=%s" %len(self.neighborNodes))
    #         col = pygame.Color(0, 255, 255, 80)
    #         for node in self.neighborNodes:
    #             pygame.draw.rect(game.screen, col, node.sprite.rect)

    #     if self.drawpath:
    #         col = pygame.Color(0, 255, 0, 120)
    #         for sprite in self.drawpath[1:-1]:
    #             pygame.draw.rect(game.screen, col, sprite.rect)

    # def _setDebugVariables(self, world, path):
    #     '''
    #         Sets the variables required for debug drawing of the paths
    #         resulting from the A-Star search.
    #         '''

    #     path_sprites = [node.sprite for node in path]

    #     self.walkableTiles = world.get_walkable_tiles()
    #     self.neighborNodes = world.neighbor_nodes_of_sprite(self)
    #     self.drawpath = path_sprites


    def update(self, game):
        VGDLSprite.update(self, game)
        world = AStarWorld(game)
        error = 10

        # Will not update AStarChaser if there is nothing to chase
        killed = [s.name for s in game.kill_list]
        if 'avatar' in killed:
            print "avatar is dead"
            return

        if game.time % 5 == 0:
            self.path = world.getMoveFor(self, self.target)
        # print path
        # print 'in astar', [world.get_sprite_tile_position(p.sprite) for p in path]
        # Uncomment below to draw debug paths.
        # self._setDebugVariables(world,path)
        print 'updating'
        print len(self.path)
        if self.path:
            # n = min(5, len(self.path)-1)


            if self.next_move == None:
                print 'popping off next path node'
                # self.path.pop(0)
                self.next_move = self.path.pop(0)
                print self.next_move.sprite.rect, self.rect

            next_x, next_y = self.next_move.sprite.rect.x, self.next_move.sprite.rect.y
            self_x, self_y = self.rect.x, self.rect.y


            print next_x, next_y
            print self_x, self_y

            dx = abs(next_x - self_x)
            dy = abs(next_y - self_y)

            if dx >= dy:
                movement = [LEFT, RIGHT][next_x > self_x]
            else:
                movement = [UP, DOWN][next_y > self_y]

            if dx < error and dy < error:
                self.last_move = self.next_move
                self.next_move = None
            print dx, dy, movement

            self.physics.activeMovement(self, movement)



# ---------------------------------------------------------------------
#     Avatars: player-controlled sprite types
# ---------------------------------------------------------------------
# from core import Avatar
Avatar = core.Avatar

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
            if random.random() < self.noiseLevel*4:
                actions = [random.choice([UP, LEFT, DOWN, RIGHT])]
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

    def declare_possible_actions(self):
        from pygame.locals import K_SPACE
        actions = MovingAvatar.declare_possible_actions(self)
        actions["SPACE"] = K_SPACE
        return actions


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
    physicstype = GravityPhysics
    draw_arrow = False
    strength = 1
    movestrength = sqrt(strength)
    vx_max = 10
    airsteering = False
    last_vy = 0
    jumping = False
    wait_step = 0
    airstrength = 1
    decay = 0 #.5

    def declare_possible_actions(self):
        from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN
        actions = {}
        #actions["UP"] = K_UP
        #actions["DOWN"] = K_DOWN
        actions["LEFT"] = K_LEFT
        actions["RIGHT"] = K_RIGHT
        return actions

    def update(self, game):
        from pygame.locals import K_SPACE

        if self.lastrect == self.rect and not self.jumping:
            self.speed = self.speed * self.orientation[0]
            self.orientation = (1,0)

        action = self._readAction(game)

        if action == None:
            action = [0, 0]
        action = list(action)
        action[1]=0
        # presumibly, this means the sprite is 'landed'
        self.airstrength *= (1-self.decay)
        if self.last_vy == self.lastrect.y - self.rect.y:
            #print "are equal"
            self.wait_step += 1
            if not self.jumping:
                #print "no"
                #action[0] = action[0] * self.movestrength
                action = [action[0] * self.movestrength,0]
                if game.keystate[K_SPACE] and not self.jumping:
                    action[1] = -self.strength
                    self.jumping = True
                    self.wait_step = 0
                    self.airstrength = 1
            else:
                #print "yes"
                action[0] = action[0] * self.movestrength * self.airstrength
                #action[0] = 0

        else:
            self.wait_step = 0
            action[0] = 0

        # this is pretty hacky. What if sprite doesn't move very fast?
        if self.wait_step > 2:
            self.jumping = False

        self.physics.activeMovement(self, action)
        #changes speed


        vx = self.orientation[0]*self.speed

        #print((vx,self.orientation[1]*self.speed))

        if abs(vx) > self.vx_max:
            # vx always greater than zero at this point
            sign = abs(vx)/vx
            vx = sign*self.vx_max
            vy = self.orientation[1]*self.speed
            self.orientation = unitVector((vx, vy))
            self.speed = vectNorm((vx, vy))/ vectNorm(self.orientation)

        # a less precise vy, but this is useful
        self.last_vy = self.lastrect.y-self.rect.y
        VGDLSprite.update(self, game)

class ClimbingAvatar(MarioAvatar, MovingAvatar):
    climbing = False
    saved_gravity = GravityPhysics.gravity
    saved_steering = MarioAvatar.airsteering
    jumping = False
    def update(self, game):
        action = self._readAction(game)
        if action is None:
            action = (0, 0)
        from pygame.locals import K_SPACE, K_UP, K_DOWN

        if self.climbing:
            self.physics.gravity = 0
            self.airsteering = True
        else:
            self.physics.gravity = self.saved_gravity
            self.airsteering = self.saved_steering

        if game.keystate[K_SPACE] and self.orientation[1] == 0:
            self.climbing = False
            self.jumping = True
            self.physicstype = GravityPhysics
            action = (action[0] * sqrt(self.strength), -self.strength)
        elif game.keystate[K_UP] and self.climbing:
            climbing = True
        elif game.keystate[K_DOWN] and self.climbing:
            climbing = True
        elif self.orientation[1] == 0 or self.airsteering:
            action = (action[0] * sqrt(self.strength), 0)
        else:
            action = (0, 0)
            if self._velocity()[1] > 0:
                self.jumping = False
        self.climbing = False
        self.physics.activeMovement(self, action)
        VGDLSprite.update(self, game)


class FrostBiteAvatar(HorizontalAvatar, InertialAvatar):
    physicstype = GravityPhysics
    draw_arrow = False
    strength = 6
    airsteering = True
    speed = .25
    solid = True
    jumping = False

    def update(self, game):
        action = self._readAction(game)
        if action is None:
            action = (0, 0)
        from pygame.locals import K_UP, K_DOWN
        if game.keystate[K_UP] and self.orientation[1] == 0:
            action = (action[0] * sqrt(self.strength), -self.strength)
            self.jumping = True
        elif game.keystate[K_DOWN] and self.orientation[1] == 0:
            self.solid = False
        elif self.orientation[1] == 0 or self.airsteering:
            action = (action[0] * sqrt(self.strength), 0)
        else:
            action = (0, 0)
        if self._velocity()[1] > 0:
            self.jumping = False

        self.physics.activeMovement(self, action)
        HorizontalAvatar.update(self, game)
        VGDLSprite.update(self, game)

class Floe(Missile, Switch):
    speed = 0.05
    def update(self, game):
        Missile.update(self, game)
        Switch.update(self, game)

class FrostbiteIgloo(SpawnPoint, Switch):
    offsets = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]]
    total = None
    triggered = False
    detriggered = False
    triggers = 0
    def __init__(self, platforms=8, **kwargs):
        SpawnPoint.__init__(self, **kwargs)
        Switch.__init__(self, **kwargs)
        self.total = 5
        self.platforms = platforms
        self.last_count = 0


    def update(self, game):
        Switch.update(self, game)
        new_count = (self.triggers * self.total) / (self.platforms)

        if new_count > self.last_count:
            self.last_count = new_count
            SpawnPoint.update(self, game)

        if new_count < self.last_count:
            game.kill_list.append(self.last_sprites.pop())
            self.counter -= 1
            self.last_count = new_count

        if self.triggered and self.counter < self.total:
            self.xoffset = self.offsets[new_count][0]
            self.yoffset = self.offsets[new_count][1]

            self.triggers += 1
            self.triggered = False

        if self.detriggered and self.triggers > 0:
            self.triggers -= 1
            self.detriggered = False

# ---------------------------------------------------------------------
#     Conditional criteria
# ---------------------------------------------------------------------
# from core import Conditional
Conditional = core.Conditional

class SpriteCount(Conditional):
    ops = {'equ': lambda x, y: x == y,
           'lss': lambda x, y: x < y,
           'grt': lambda x, y: x > y,
           'leq': lambda x, y: x <= y,
           'geq': lambda x, y: x >= y,
           'neq': lambda x, y: x != y
           }
    def __init__(self, stype=None, count=0, op='equ'):
        self.stype = stype
        self.count = count
        self.op = op
    def condition(self, game):
        if self.ops[self.op](game.numSprites(self.stype), self.count):
            return True
        else:
            return False

class OnStart(Conditional):
    def condition(self, game):
        if game.started:
            return True
        return False

# ---------------------------------------------------------------------
#     Termination criteria
# ---------------------------------------------------------------------
# from core import Termination
Termination = core.Termination

class Timeout(Termination):
    def __init__(self, limit=0, win=False, bonus=0):
        self.limit = limit
        self.win = win
        self.name = 'Timeout'
        self.bonus = bonus

    def isDone(self, game):
        if game.time >= self.limit:
            return True, self.win
        else:
            if game.time > game.timeout_bonus_granted_on_timestep:
                game.score += self.bonus
                game.timeout_bonus_granted_on_timestep = game.time
            return False, None

class SpriteCounter(Termination):
    """ Game ends when the number of sprites of type 'stype' hits 'limit' (or below). """
    def __init__(self, limit=0, stype=None, win=True, bonus=0):
        self.limit = limit
        self.stype = stype
        self.win = win
        self.name = 'SpriteCounter'
        self.bonus=bonus

    def isDone(self, game):
        if game.numSprites(self.stype) <= self.limit:
            if game.time > game.sprite_bonus_granted_on_timestep:
                game.score += self.bonus
                game.sprite_bonus_granted_on_timestep = game.time
            return True, self.win
        else:
            return False, None

class MultiSpriteCounter(Termination):
    """ Game ends when the sum of all sprites of types 'stypes' hits 'limit'. """
    def __init__(self, limit=0, win=True, bonus=0, **kwargs):
        self.limit = limit
        self.win = win
        self.bonus = bonus
        self.stypes = kwargs.values()
        self.name = 'MultiSpriteCounter'

    def isDone(self, game):
        if sum([game.numSprites(st) for st in self.stypes]) == self.limit:
            if game.time > game.sprite_bonus_granted_on_timestep:
                game.score += self.bonus
                game.sprite_bonus_granted_on_timestep = game.time
            return True, self.win
        else:
            return False, None

class NoveltyTermination(Termination):
    def __init__(self, s1, s2, win=True, args=None):
        self.s1 = s1
        self.s2 = s2
        self.win = win
        self.name = 'NoveltyTermination'
        self.args = args
        if self.args is not None:
            # print "found args in noveltytermination"
            if type(self.args) == set:
                self.args = list(self.args)[0]
            elif type(self.args) == dict:
                # embed()
                pass

    def isDone(self, game):

        resourcePassed = False
        ## self.args lets us do precondition-dependent terminations.
        if self.args:
            # print "got noveltytermination args"
            # embed()
            if type(self.args)==dict:
                item, num, negated, operator_name = self.args['item'], self.args['num'], eval(self.args['negated']), self.args['operator_name']
            else:
                item, num, negated, operator_name = self.args.item, self.args.num, self.args.negated, self.args.operator_name
            if negated:
                oppositeOperatorMap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}
                true_operator = oppositeOperatorMap[operator_name]
            else:
                true_operator = operator_name
            try:
                # resource_str = str(game.getAvatars()[0].resources[item])
                if item in game.lastAvatarResources:
                    resource_str = str(game.lastAvatarResources[item])
                else:
                    resource_str = str(0)
            except IndexError:
                return False, None

            if not eval(resource_str+true_operator+str(num)):
                # if self.s1=='c5' and self.s2=='avatar':
                    # print "precondition for {} is {} and failed".format((self.s1,self.s2), self.args)
                    # print game.getAvatars()[0].resources
                return False, None
            else:
                # if self.s1=='c5' and self.s2=='avatar':
                    # print "precondition for {} is {} and PASSED".format((self.s1,self.s2), self.args)
                resourcePassed = True
                # print ""

            # else:
            #     print "found correct preconditions"
                # embed()
        # if game.effectList:
            # embed()
        # print len(game.effectListByClass), self.s1, self.s2
        for e in game.effectListByClass:
            if (e[0] in ['killSprite', 'transformTo', 'nothing']) and len(e) > 2:
                if (e[1], e[2]) in [(self.s1, self.s2), (self.s2, self.s1)]:
                    # print "noveltyTermination between", self.s1, self.s2
                    return True, self.win
        return False, None


        # for e in game.effectList:
        #     id_not_found = False
        #     class1, class2 = 'none', 'none'
        #     if (e[0] in ['killSprite', 'transformTo', 'nothing']) and len(e) > 2:
        #         try:
        #             name1 = game.getAllObjects()[e[1]].name
        #             class1 = str(game.getAllObjects()[e[1]].__class__)
        #             # if 'RandomNPC' in class1  and e[2]  != 'avatar':
        #                 # return False, None
        #             # if 'Flicker' in class1  and e[2]  == 'avatar':
        #                 # return False, None       
        #         except KeyError:
        #             if e[1]=='ENDOFSCREEN':
        #                 name1 = 'EOS'
        #             elif e[1] in [obj.ID for obj in game.kill_list]:
        #                 name1 = [obj.name for obj in game.kill_list
        #                     if obj.ID==e[1]][0]
        #                 class1 = str([obj.__class__ for obj in game.kill_list
        #                     if obj.ID==e[1]][0])
        #                 # if 'RandomNPC' in class1  and e[2]  != 'avatar':
        #                     # return False, None
        #                 # if 'Flicker' in class1  and e[2]  == 'avatar':
        #                     # return False, None
        #             elif e[1] in game.getObjects().keys():
        #                 name1 = game.getObjects()[e[1]]['sprite'].name
        #                 class1 =  str(game.getObjects()[e[1]]['sprite'].__class__)
        #                 # if 'RandomNPC' in class1  and e[2]  != 'avatar':
        #                     # return False, None
        #                 # if 'Flicker' in class1  and e[2]  == 'avatar':
        #                     # return False, None
        #             else:
        #                 id_not_found = True
        #                 # print "id_not_found 1"
        #                 # embed()
        #                 ## This happens when we shoot an object and IDs are mismatched; default to the thing we shoot.
        #                 ## We've confirmed that this isn't due to other objects shot by other objects.
        #                 try:
        #                     name1 = game.getAvatars()[0].stype
        #                     if e[2] == 'avatar':
        #                         return False, None
        #                 except (AttributeError, IndexError) as err:
        #                     # Avatar dead or doesn't have stype
        #                     name1 = ''
        #         except IndexError:
        #             print("IndexError in game.all_objects (1)")
        #             embed()
        #         except TypeError:
        #             print("TypeError in game.all_objects (1)")
        #             embed()
        #         try:
        #             name2 = game.getAllObjects()[e[2]].name
        #             class2 = str(game.getAllObjects()[e[2]].__class__)
        #             # if 'RandomNPC' in class2  and e[1]  != 'avatar':
        #                 # return False, None
        #             # if 'Flicker' in class2  and e[1]  == 'avatar':
        #                 # return False, None   
        #         except KeyError:
        #             if e[2]=='ENDOFSCREEN':
        #                 name2 = 'EOS'
        #             elif e[2] in [obj.ID for obj in game.kill_list]:
        #                 # candidates = [obj for obj in game.kill_list]
        #                 name2 = [obj.name for obj in game.kill_list
        #                     if obj.ID==e[2]][0]
        #                 class2 = str([obj.__class__ for obj in game.kill_list
        #                     if obj.ID==e[2]][0])
        #                 # if 'RandomNPC' in class2  and e[1]  != 'avatar':
        #                     # return False, None
        #                 # if 'Flicker' in class2  and e[1]  == 'avatar':
        #                     # return False, None
        #             elif e[2] in game.getObjects().keys():
        #                 name2 = game.getObjects()[e[2]]['sprite'].name
        #                 class2 =  str(game.getObjects()[e[2]]['sprite'].__class__)
        #                 # if 'RandomNPC' in class2  and e[1]  != 'avatar':
        #                     # return False, None
        #                 # if 'Flicker' in class2  and e[1]  == 'avatar':
        #                     # return False, None
        #             else:
        #                 id_not_found = True
        #                 # print "id_not_found 2"
        #                 # embed()
        #                 ## This happens when we shoot an object and IDs are mismatched; default to the thing we shoot.
        #                 ## We've confirmed that this isn't due to other objects shot by other objects.
        #                 try:
        #                     name2 = game.getAvatars()[0].stype
        #                 except (AttributeError, IndexError) as err:
        #                     # Avatar dead or doesn't have stype
        #                     name2 = ''
        #         except IndexError:
        #             print("IndexError in game.all_objects (2)")
        #             embed()
        #         except TypeError:
        #             print("TypeError in game.all_objects (2)")
        #             embed()
        #         if name1==self.s1 and name2==self.s2:
        #             # if id_not_found:
        #                 # print "id_not_found"
        #                 # embed()
        #                 # pass
        #             print("NoveltyTermination with {} and {}".format(
        #                 name1, name2))
        #             # if name1=='c7' and name2=='avatar':
        #             #     ipdb.set_trace()
        #             # print("Classes are {} and {}".format(class1, class2))
        #             return True, self.win
        #     elif len(e) > 2 and e[2]=='ENDOFSCREEN':
        #         name2 = 'EOS'
        #         try:
        #             name1 = game.getAllObjects()[e[1]].name
        #             # Don't get a noveltyTermination from RandomNPCs
        #             # if 'RandomNPC' in str(game.all_objects[e[1]]['sprite'].__class__) and e[2]  != 'avatar':
        #                 # return False, None
        #             # if 'Flicker' in str(game.all_objects[e[1]]['sprite'].__class__)  and e[2]  == 'avatar':
        #                 # return False, None
        #         except KeyError:
        #             if e[1]=='ENDOFSCREEN':
        #                 name1 = 'EOS'
        #             elif e[1] in [obj.ID for obj in game.kill_list]:
        #                 name1 = [obj.name for obj in game.kill_list
        #                     if obj.ID==e[1]][0]
        #             elif e[1] in game.getObjects().keys():
        #                 name1 = game.getObjects()[e[1]]['sprite'].name
        #             else:
        #                 # print "Couldn't find object in NoveltyTermination"
        #                 id_not_found = True
        #                 # embed()
        #                 try:
        #                     name1 = game.getAvatars()[0].stype
        #                 except (AttributeError, IndexError) as err:
        #                     # Avatar dead or doesn't have stype
        #                     name1 = ''
        #         except IndexError:
        #             print("IndexError in game.all_objects")
        #             # embed()
        #             pass
        #         except:
        #             print("AttributeError in game.all_objects")
        #             embed()
        #         # self.s2 returns a type for the EOS for some reason, so the
        #         # check has to be performed like this
        #         if name1==self.s1 and name2 in str(self.s2):
        #             if id_not_found:
        #                 print "id_not_found"
        #                 embed()
        #                 pass
        #             print("NoveltyTermination with {} and {}".format(
        #                 name1, name2))
        #             # if name1=='c7' and name2=='avatar':
        #             #     ipdb.set_trace()
        #             return True, self.win
        # return False, None

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
def nothing(sprite, partner, game):
    """ Returns no interaction """
    # print ("nothing", sprite.rect, partner.rect, sprite.name, partner.name)
    return ("nothing", sprite.ID, partner.ID)

def killSprite(sprite, partner, game):
    """ Kill command """
    game.kill_list.append(sprite)
    sprite.deathage = game.time
    if not None in {sprite, partner}:
        return ("killSprite", sprite.ID, partner.ID) # partner = agent, sprite = what's being killed

def cloneSprite(sprite, partner, game):
    newones = game._createSprite([sprite.name], (sprite.rect.left, sprite.rect.top))
    # try:
    #     if len(newones) > 0:
    #         if isinstance(sprite, OrientedSprite) and isinstance(newones[0], OrientedSprite):
    #             newones[0].orientation = sprite.orientation
    #         game.kill_list.append(sprite)
    #         game.dead.append(sprite)
    # except:
    #     pass
    return ("cloneSprite", sprite.ID, partner.ID)

def transformTo(sprite, partner, game, stype='wall'):
    newones = game._createSprite([stype], (sprite.rect.left, sprite.rect.top))
    if len(newones) > 0:
        if isinstance(sprite, OrientedSprite) and isinstance(newones[0], OrientedSprite):
            newones[0].orientation = sprite.orientation
            newones[0].resources = sprite.resources
        sprite.deathage = game.time
        game.kill_list.append(sprite)
    args = {'stype':stype}
    return ("transformTo", sprite.ID, partner.ID, args)

def transformToOnLanding(sprite, partner, game, stype='wall'):
    """sprite will be transformed to stype when partner (avatar) lands on it from above"""
    if partner.speed*partner.orientation[1] == 0 and partner.lastrect.y != partner.rect.y:
        transformTo(sprite, partner, game, stype)
        args = {'stype':stype}
        ##Decide whether it's "fair" to know this was transformToOnLanding as opposed to transform
        return ("transformToOnLanding", sprite.ID, partner.ID, args)

def triggerOnLanding(sprite, partner, game, strigger=None):
    '''triggers a triggerable sprite. triggerable is interesting. should change this?'''
    if partner.speed*partner.orientation[1] == 0 and partner.lastrect.y != partner.rect.y:
        trigger(sprite, partner, game, strigger)
        args = {'strigger':strigger}
        return ("trigger", sprite.ID, partner.ID, args)

def stepBack(sprite, partner, game):
    """ Revert last move. """
    sprite.rect = sprite.lastrect

    if partner:
        try:
            return ("stepBack", sprite.ID, partner.ID)
        except:
            ## happens most likely with EOS events.
            return ("stepBack", sprite.ID, partner)

def undoAll(sprite, partner, game):
    """ Revert last moves of all sprites. """
    for s in game:
        s.rect = s.lastrect

    return ('undoAll', sprite.ID , partner.ID)

def bounceForward(sprite, partner, game): # FLAG
    """ The partner sprite pushed, so if possible move in the opposite direction. """
    # print "in beginning of bounceForward"
    sprite.physics.activeMovement(sprite, unitVector(partner.lastdirection))
    game._updateCollisionDict(sprite)
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
    s = random.choice([partner.strength, partner.strength + 1, partner.strength - 1])
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
    if prob > random.random():
        tmp = sprite.lastrect
        v = unitVector(sprite.orientation)
        sprite.physics.activeMovement(sprite, v, speed=1)
        sprite.lastrect = tmp
        game._updateCollisionDict(sprite)
        # return ("slipForward" , colorDict[str(sprite.color)], colorDict[str(partner.color)])
        return ('slipForward', sprite.ID, partner.ID)

def attractGaze(sprite, partner, game, prob=0.5):
    """ Turn the orientation to the value given by the partner. """
    if prob > random.random():
        sprite.orientation = partner.orientation
        # return ("attractGaze" , colorDict[str(sprite.color)], colorDict[str(partner.color)])
        return ('attractGaze', sprite.ID, partner.ID)

def turnAround(sprite, partner, game):
    sprite.rect = sprite.lastrect
    sprite.lastmove = sprite.cooldown -1 ## Needed because updatePos looks for lastmove+1%cooldown==0
    firstspeed = sprite.speed
    # if firstspeed<1:
        # sprite.speed = 1.1 ## speed <1 weird creates a weird overlap in aliens, possibly in other games.
    sprite.physics.activeMovement(sprite, DOWN)
    reverseDirection(sprite, partner, game)
    # sprite.speed = firstspeed
    game._updateCollisionDict(sprite)
    if partner == None:
        return ('turnAround', sprite.ID)
    return ('turnAround', sprite.ID, partner.ID)

def turn(sprite, partner, game):
    sprite.rect = sprite.lastrect
    sprite.lastmove = sprite.cooldown -1 ## Needed because updatePos looks for lastmove+1%cooldown==0
    # sprite.physics.activeMovement(sprite, DOWN)
    # sprite.lastmove = sprite.cooldown
    # sprite.physics.activeMovement(sprite, DOWN)
    reverseDirection(sprite, partner, game)
    game._updateCollisionDict(sprite)
    if partner == None:
        return ('turn', sprite.ID)
    return ('turn', sprite.ID, partner.ID)

def reverseDirection(sprite, partner, game): # FLAG
    sprite.orientation = (-sprite.orientation[0], -sprite.orientation[1])
    if partner == None:
        return ('reverseDirection', sprite.ID)
    return ('reverseDirection', sprite.ID, partner.ID)

##TODO: add event labels for the below effects
def reverseFloeIfActivated(sprite, partner, game, strigger=None):
    '''sprite is Floe, partner is FrostbiteAvatar'''
    if sprite.activated:
        detrigger(sprite, partner, game, strigger)
        reverseDirection(sprite, partner, game)
        sprite.activated = False
    ## returning the below is likely too much. Only adding this now for consistency of return statements.
    args = {'strigger':strigger}
    return ('reverseFloeIfActivated', sprite.ID, partner.ID, args)

def trigger(sprite, partner, game, strigger=None):
    if strigger == None:
        triggers = [sprite]
    else:
        triggers = game.getSprites(strigger)

    for sprite in triggers:
        sprite.triggered = True
    return ('trigger', sprite.ID, partner.ID, strigger)

def detrigger(sprite, partner, game, strigger=None):
    if strigger == None:
        triggers = [sprite]
    else:
        triggers = game.getSprites(strigger)

    for sprite in triggers:
        sprite.detriggered = True
    args = {'strigger':strigger}
    return ('detrigger', sprite.ID, partner.ID, args)


def flipDirection(sprite, partner, game): # FLAG
    sprite.orientation = random.choice(BASEDIRS)

    return ('flipDirection', sprite.ID, partner.ID)

def bounceDirection(sprite, partner, game, friction=0): # FLAG
    """ The centers of the objects determine the direction"""
    stepBack(sprite, partner, game)
    inc = sprite.orientation
    snorm = unitVector((-sprite.rect.centerx + partner.rect.centerx,
                        - sprite.rect.centery + partner.rect.centery))
    dp = snorm[0] * inc[0] + snorm[1] * inc[1]
    sprite.orientation = (-2 * dp * snorm[0] + inc[0], -2 * dp * snorm[1] + inc[1])
    sprite.speed *= (1. - friction)
    return ('bounceDirection', sprite.ID, partner.ID)


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
    return ('wallBounce', sprite.ID, partner.ID)

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
        return killSprite(sprite, partner, game)
        # return ('killIfSlow' , sprite.ID, partner.ID)


def killIfFromAbove(sprite, partner, game):
    """ Kills the sprite, only if the other one is higher and moving down. """
    if (sprite.lastrect.top > partner.lastrect.top
        and partner.rect.top > partner.lastrect.top):

        game.kill_list.append(sprite)
        if not None in {sprite, partner}:
         # sprite_info = colorDict[str(sprite.color)]
            return ('killIfFromAbove', sprite.ID, partner.ID)

def killIfAlive(sprite, partner, game):
    """ Perform the killing action, only if no previous collision effect has removed the partner. """
    if partner not in game.kill_list:
        return killSprite(sprite, partner, game)
        # return ('killIfAlive' , sprite.ID, partner.ID)

def collectResource(sprite, partner, game, resource=None, value=1, limit=None): # FLAG
    """ Adds/increments the resource type of sprite in partner """
    if resource is None:
        if isinstance(sprite, Resource) or isinstance(sprite, ResourcePack):
            r = sprite.resourceType
            partner.resources[r] = max(-1, min(partner.resources[r]+sprite.value, game.resources_limits[r]))
        else:
            r = partner.resources.keys()[0]
            value=1
            partner.resources[r] = max(-1, min(partner.resources[r]+value, game.resources_limits[r]))
    else:
        try:
            partner.resources[resource] = max(-1, min(partner.resources[resource]+sprite.value, game.resources_limits[resource]))
        except:
            partner.resources[resource] = max(-1, min(partner.resources[resource]+value, game.resources_limits[resource]))

    killSprite(sprite, partner, game)
    args = {'resource':sprite.name, 'value':value, 'limit':game.resources_limits[sprite.name]}
    print "collectResource", partner.resources
    #print 'Collected ', colorDict[str(sprite.color)]#partner.resources[r]
    # return ('collectResource', colorDict[str(partner.color)], colorDict[str(sprite.color)])
    return ('collectResource' , sprite.ID, partner.ID, args)

def changeResource(sprite, partner, resourceColor, game, resource, value=1, limit=None):
    """ Increments a specific resource type in sprite """

    sprite.resources[resource] = max(-1, min(sprite.resources[resource]+value, game.resources_limits[resource]))
    # NOTE: partner is the color of the resource (see _eventHandling() in core.py)
    args = {'resource':resource, 'value':value, 'limit':game.resources_limits[resource]}
    # print args
    return ('changeResource', sprite.ID, partner.ID, args)

def changeScore(sprite, partner, game, value):
    game.score += value
    # print "score", game.score
    args = {'value': value}
    return ('changeScore', sprite.ID, partner.ID, args)

def spawnIfHasMore(sprite, partner, game, resource, stype, limit=1):
    """ If 'sprite' has more than a limit of the resource type given, it spawns a sprite of 'stype'. """
    if sprite.resources[resource] >= limit:
        game._createSprite([stype], (sprite.rect.left, sprite.rect.top))
        # Note: returning the resource doesn't seem like something the agent should have access to, so we're not returning it.
        args = {'stype':stype}
        return ('spawnIfHasMore', sprite.ID, partner.ID, args) ### NOTE - there is no default 'spawn' function we could return instead, but we should then make one

def killIfHasMore(sprite, partner, game, resource, limit=1):
    """ If 'sprite' has more than a limit of the resource type given, it dies. """
    if sprite.resources[resource] >= limit:
        return killSprite(sprite, partner, game)
        # return ('killIfHasMore' , sprite.ID, partner.ID)

def killIfOtherHasMore(sprite, partner, game, resource, limit=1):
    """ If 'partner' has more than a limit of the resource type given, sprite dies. """
    # print partner.resources, limit
    if partner.resources[resource] >= limit:
        return killSprite(sprite, partner, game)
        # return ('killIfOtherHasMore' , sprite.ID, partner.ID)

def killIfHasLess(sprite, partner, game, resource, limit=1):
    """ If 'sprite' has less than a limit of the resource type given, it dies. """
    # print sprite.resources[resource], limit
    if sprite.resources[resource] <= limit:
        return killSprite(sprite, partner, game)
        # return ('killIfHasLess' , sprite.ID, partner.ID)

def killIfOtherHasLess(sprite, partner, game, resource, limit=1):
    """ If 'partner' has less than a limit of the resource type given, sprite dies. """
    if partner.resources[resource] <= limit:
        return killSprite(sprite, partner, game)
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
    args = {'offset':offset}
    # print ('wrapAround', sprite.colorName, partner.colorName, args)
    return ('wrapAround', sprite.ID, partner.ID, args)

def pullWithIt(sprite, partner, game):
    """ The partner sprite adds its movement to the sprite's. """
    if not oncePerStep(sprite, game, 'lastpull'):
        return

    tmp = sprite.lastrect
    v = unitVector(partner.lastdirection)
    try:
        sprite._updatePos(v, partner.speed * sprite.physics.gridsize[0])
    except:
        print "problem in pullwithit"
        embed()
    if isinstance(sprite.physics, ContinuousPhysics):
        sprite.speed = partner.speed
        sprite.orientation = partner.lastdirection
    sprite.lastrect = tmp

    return ('pullWithIt' , sprite.ID, partner.ID)

def collideFromAbove(sprite, partner, game):
    """ Allows the sprite to pass through the bottom and collide with the top."""
    if (sprite.lastrect.top < partner.lastrect.top
        and sprite.lastrect.bottom < partner.lastrect.bottom) and sprite.solid and not sprite.jumping:
        pullWithIt(sprite, partner, game)
    elif (sprite.lastrect.bottom > partner.lastrect.bottom or
        sprite.lastrect.right < partner.lastrect.left or
        sprite.lastrect.left > partner.lastrect.right) and not(sprite.solid):
        sprite.solid = True
    return ('collideFromAbove', sprite.ID, partner.ID)

def killSpriteOnLanding(sprite, partner, game):
    """ kills the sprite given the collision condition from collide from above"""
    if (sprite.lastrect.top < partner.lastrect.top
        and sprite.lastrect.bottom < partner.lastrect.bottom
         and sprite.solid and not sprite.jumping):
        killSprite(sprite, partner, game)
    return ('killSpriteOnLanding', sprite.ID, partner.ID)

def teleportToExit(sprite, partner, game):
    try:
        e = random.choice(game.sprite_groups[partner.stype])
    except:
        ## if the above fails, it's because the theory has specified the partner.stype as the goal
        ## and so there is no game.sprite_groups[partner.stype]. send avatar to the goal.
        e = random.choice(game.sprite_groups['goal'])
    sprite.rect = e.rect
    #sprite.lastmove = 0
    args = {'stype':partner.stype}
    return ('teleportToExit', sprite.ID, partner.ID, args)

# this allows us to determine whether the game has stochastic elements or not
stochastic_effects = [teleportToExit, windGust, slipForward, attractGaze, flipDirection]

# this allows is to determine which effects might kill a sprite
kill_effects = [killSprite, killIfSlow, transformTo, killIfOtherHasLess, killIfOtherHasMore, killIfHasMore, killIfHasLess,
                killIfFromAbove, killIfAlive]


def canActivateSwitch(sprite, partner, game):
    sprite.can_switch = True
    return ('canActivateSwitch', sprite.ID, partner.ID)

def cannotActivateSwitch(sprite, partner, game):
    sprite.can_switch = False
    return ('cannotActivateSwitch', sprite.ID, partner.ID)

# ---------------------------------------------------------------------
#     Sprite Induction
# ---------------------------------------------------------------------
## TODO: Make sure you put these other types back when you fix sprite induction!!
sprite_types = [ResourcePack, RandomNPC, Missile, Chaser] #removed Resource, Immovable, Passive, AStarChaser,


def getSpeed(params):
    """
    params = a dict mapping sprite attributes to values
    sprite = the VGDL sprite.
    """
    if 'speed' in params:
        return params['speed']
    else:
        return 1
        # default speed value

def getFleeing(params):
    """
    params = a dict mapping sprite attributes to values
    sprite = the VGDL sprite.
    """
    if 'fleeing' in params:
        return params['fleeing']
    else:
        return False

def getOrientation(params):
    """
    params = a dict mapping sprite attributes to values
    sprite = the VGDL sprite.
    """
    if 'orientation' in params:
        return params['orientation']

def getStype(params):
    """
    params = a dict mapping sprite attributes to values
    sprite = the VGDL sprite.
    """
    if 'stype' in params:
        return params['stype']

def getCooldown(params):
    if 'cooldown' in params:
        return params['cooldown']
    else:
        return 1

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

def chaserMovesToward(sprite, game, target, fleeing):
    """ Find the canonical direction(s) which move toward
    the target. """
    
    if (sprite, target, fleeing) in game.chaserMovesTowardDict:
        return game.chaserMovesTowardDict[(sprite, target, fleeing)]

    res = []
    basedist = sprite.physics.distance(sprite.rect, target.rect)

    for a in BASEDIRS:
        r = sprite.rect.copy()
        r = r.move(a)
        newdist = sprite.physics.distance(r, target.rect)

        if fleeing and basedist < newdist:
            res.append(a)
        if not fleeing and basedist > newdist:
            res.append(a)
    game.chaserMovesTowardDict[(sprite, target, fleeing)] = res
    return res

def setSpriteParams(param, sprite):
    """
    param = a dict mapping parameters to values
    sprite = the vgdl sprite
    """
    for p in param:
        if p == "speed":
            sprite.speed = param[p]
        elif p == "fleeing":
            sprite.fleeing = param[p]
        elif p == "orientation":
            sprite.orientation = param[p]
        elif p == "stype":
            sprite.stype = param[p]
        elif p == "cooldown":
            sprite.cooldown = param[p]


    # def calculateActiveMovement(self, sprite, action, speed=None):
    #     """
    #     Calculate where the sprite would end up in a timestep, without actually updating its position.
    #     """
    #      ## This is where you could make hypotheses about speed, etc. for the object.
    #     if action is not None:
    #         orientation = action
    #     if (sprite.lastmove+1)%sprite.cooldown==0 and abs(orientation[0])+abs(orientation[1])!=0:

    #         if speed is None:
    #             if sprite.speed is None:
    #                 speed = 1
    #             else:
    #                 speed = sprite.speed
    #         if speed != 0:# and action is not None:
    #             speed = float(speed) * self.gridsize[0]

    #         pos = sprite.rect.move((orientation[0]*speed, orientation[1]*speed))
    #         return pos.left, pos.top
    #     return(sprite.rect.left, sprite.rect.top)



def calculateSpriteMove(game, sprite, speed, orientation):
    if abs(orientation[0])+abs(orientation[1])!=0:
        if speed is None:
            if sprite.speed is None:
                speed = 1
            else:
                speed = sprite.speed
        if speed != 0:# and action is not None:
            speed = float(speed) * game.block_size
        newPos = sprite.rect.left+orientation[0]*speed, sprite.rect.top+orientation[1]*speed
        return (newPos[0], newPos[1])
    return(sprite.rect.left, sprite.rect.top)

def getTargets(game, targetColor):
    t1 = time.time()
    if targetColor not in game.targetColorDict:
        try:
            targetName = [k for k in game.sprite_groups.keys() if game.sprite_groups[k] and game.sprite_groups[k][0].colorName==targetColor][0]
            targets = [s for s in game.sprite_groups[targetName] if s not in game.kill_list]
            # print "target name: {}. target length: {}".format(targetName, len(targets))
        except:
            targets = []
        game.targetColorDict[targetColor] = targets

    return game.targetColorDict[targetColor]

# def updateOptionsProfiler(game, sprite_type_tuple, current_sprite, params={}, missileOrientationClustering=False):
#     lp = LineProfiler()
#     lp_wrapper = lp(updateOptions)
#     d1, d2 = lp_wrapper(game, sprite_type_tuple, current_sprite, params, missileOrientationClustering)
#     lp.print_stats()
#     return d1, d2


def updateOptions(game, sprite_type_tuple, current_sprite, params={}, missileOrientationClustering=False):
    """
    This method gets all of the parameter information from the params variable
    instead of directly accessing the parameters in current_sprite.
    game - current game object
    sprite_type - the sprite type class hypothesis
    current_sprite - the current sprite object
    params - inferred params of the sprite. A dict mapping parameters (as strings) to their values.
    The default value of params is an empty dictionary - if that's the value passed, then the method will
    assume default values for each attribute.
    """
    sprite_type = sprite_type_tuple[1]

    # if current_sprite.name!='wall':
    #     embed()
    # Immovable, Passive, ResourcePack
    if sprite_type in [Immovable, Passive, ResourcePack, Resource, 'OTHER']:
    # if (sprite_type == Immovable) or (sprite_type == Passive) or (sprite_type == ResourcePack) or (sprite_type == Resource) or (sprite_type=='OTHER'):
        return {(current_sprite.rect.left, current_sprite.rect.top): 1.}, {(current_sprite.rect.left, current_sprite.rect.top): 1.} ##object stays in position

    # Chaser
    elif sprite_type == Chaser:
        speed = getSpeed(params)
        fleeing = getFleeing(params)
        targetColor = getStype(params)
        cooldown = getCooldown(params)

        realCooldown = int(current_sprite.cooldown)
        current_sprite.cooldown = cooldown
        
        targets = getTargets(game, targetColor)

        options = []
        position_options = {}

        try:
            for target in targets:
                options.extend(chaserMovesToward(current_sprite, game, target, fleeing))
            if len(options) == 0:
                options = BASEDIRS

            for option in options:
                # left, top = calculateSpriteMove(game, current_sprite, speed, option)
                left, top = current_sprite.physics.calculateActiveMovement(current_sprite, option, speed=speed, is_chaser=True)
                # if left!=left1 or top!=top1:
                    # print "got different positions"
                    # embed()
                if (left, top) in position_options.keys():
                    position_options[(left, top)] += 1.0/len(options)
                else:
                    position_options[(left, top)] = 1.0/len(options)

        except AttributeError: # deals with following error: 'Immovable' object has no attribute 'stype'
            position_options = {(current_sprite.rect.left, current_sprite.rect.top): 1.}

        current_sprite.cooldown = realCooldown
        return position_options, position_options

    # Random NPC
    elif sprite_type == RandomNPC:

        realCooldown = int(current_sprite.cooldown)
        speed, cooldown = getSpeed(params), getCooldown(params)
        current_sprite.cooldown = cooldown
        position_options = {}

        # if (current_sprite.lastmove+1)%cooldown!=0:
        #     position_options = {(current_sprite.rect.left, current_sprite.rect.top): 1.}
        #     return position_options, position_options


        for option in BASEDIRS:
            # left, top = calculateSpriteMove(game, current_sprite, speed, option)
            left, top = current_sprite.physics.calculateActiveMovement(current_sprite, option, speed=speed)
            if (left, top) in position_options.keys():
                position_options[(left, top)] += 1.0/len(BASEDIRS)
            else:
                position_options[(left, top)] = 1.0/len(BASEDIRS)

        current_sprite.cooldown = realCooldown
        return position_options, position_options

    # Missile or OrientedSprite
    elif sprite_type in [Missile, OrientedSprite]:

        if not current_sprite.is_static and not current_sprite.only_active:
            # NOTE: we might want to consider having is_static and only_active be
            # parameters that we have to infer, rather than things we get for free.
            # (i.e. make these fields in the params variable)
            speed = getSpeed(params)
            orientation = getOrientation(params)
            cooldown = getCooldown(params)
            realCooldown = int(current_sprite.cooldown)
            current_sprite.cooldown = cooldown

            # if (current_sprite.lastmove+1)%cooldown!=0:
            #     position_options = {(current_sprite.rect.left, current_sprite.rect.top): 1.}
            #     return position_options, position_options
            
            coords = current_sprite.physics.calculatePassiveMovementGivenParams(current_sprite, speed, orientation)
            
            # If object has speed = 0 or no 'orientation' attribute
            position_options, clustered_position_options = {}, {}
            
            if coords == None:
                return position_options, position_options

            position_options[(coords[0], coords[1])] = 1.

            if missileOrientationClustering:
                # if current_sprite.colorName=='RED' and cooldown==1 and speed==.1:
                    # embed()

                clustered_position_options[(coords[0], coords[1])] = 1. ##8/1 hack

                epsilon_prob = 0.005
                # clustered_position_options[(coords[0], coords[1])] = .5 + epsilon_prob
                #flip orientation
                orientation = (orientation[0]*-1, orientation[1]*-1)

                coords = current_sprite.physics.calculatePassiveMovementGivenParams(current_sprite, speed, orientation)
                clustered_position_options[(coords[0], coords[1])] = 1.-epsilon_prob   ##8/1 hack                

                # if (coords[0], coords[1]) in clustered_position_options.keys():
                    # clustered_position_options[(coords[0], coords[1])] += .5 - epsilon_prob
                # else:
                    # clustered_position_options[(coords[0], coords[1])] = .5 - epsilon_prob

            # if current_sprite.colorName=='RED' and cooldown==1 and speed==1 and params['orientation']==(1,0):
            #     print "found our sprite"
            #     print current_sprite, current_sprite.lastmove, params
            #     print position_options, clustered_position_options
            #     embed()
            # if current_sprite.colorName=='RED' and cooldown==5 and speed==1 and params['orientation']==(1,0):
            #     print "found the foil hypothesis"
            #     print current_sprite, current_sprite.lastmove, params
            #     print clustered_position_options
            #     embed()


            current_sprite.cooldown = realCooldown
            return position_options, clustered_position_options

        # Catches objects that can't be Oriented Sprite and Missile types b/c fails the if-statement
        return {}, {}

def initializeDistribution(sprite_types, objectColors):
    """
    Creates a uniform distribution over all parameter combinations
    """
    catch_all_prior = .000001
    outList = []
    for sprite_type in sprite_types:
            paramList = initializeDistributionArgs(sprite_type, objectColors)
            for element in itertools.product(*paramList):
                outList.append(tuple([('vgdlType', sprite_type)]+sorted(element)))
    # z = len(outList)
    initial_distribution = {k:1.0 for k in outList}
    initial_distribution[(('vgdlType', 'OTHER'), )] = catch_all_prior
    return initial_distribution


def initializeDistributionArgs(sprite_type, objectColors):
    """
    Given a sprite type, this returns a distribution over the kinds of args (parameters) belonging
    to that sprite type.
    NOTE - this initializes args for speed, orientation and fleeing. It does not handle cooldown yet.
    """

    def initializeSpeed():
        speedValues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        # speedValues = [0.2, 1.]
        return [('speed', v) for v in speedValues]

    def initializeOrientation():
        orientationValues = [LEFT, RIGHT, UP, DOWN]
        return [('orientation', v) for v in orientationValues]

    def initializeFleeing():
        fleeingValues = [True, False]
        return [('fleeing', v) for v in fleeingValues]

    def initializeStype():
        stypeValues = [o for o in objectColors if o not in ['BLACK', 'DARKGRAY', 'MPUYEI', 'NUPHKK', 'SCJPNE']]
        return [('stype', v) for v in stypeValues]

    def initializeCooldown():
        stypeValues = [1, 2, 3, 4, 5, 6]
        # stypeValues = [1,5]
        return [('cooldown', v) for v in stypeValues]

    paramList = []
    spriteParams = spriteToParams[sprite_type.__name__]

    for s in spriteParams:
        if s == "speed":
            paramList.append(initializeSpeed())
        elif s == "fleeing":
            paramList.append(initializeFleeing())
        elif s == "orientation":
            paramList.append(initializeOrientation())
        elif s=='stype':
            paramList.append(initializeStype())
        elif s=='cooldown':
            paramList.append(initializeCooldown())

    return paramList


def distributionInitSetup(game, sprite):
    """
    Does setup for initializing distribution
    'sprite' is an object ID
    """
    objectColors = set()
    for k in game.sprite_constr.keys():
        try:
            if game.sprite_constr[k][1]['color'] not in ['BLACK', 'DARKGRAY', 'MPUYEI', 'NUPHKK', 'SCJPNE']:
                objectColors.add(colorDict[str(game.sprite_constr[k][1]['color'])])
        except KeyError:
            continue
    objectColors = list(objectColors)
    if 'DTIZDF' in objectColors:
        print "found DTIZDF"
        embed()
    game.spriteDistribution[sprite] = initializeDistribution(sprite_types, objectColors) # Indexed by object ID
    # game.object_token_spriteDistribution[sprite] = initializeDistribution(sprite_types, objectColors) # Indexed by object ID
    if sprite not in game.all_objects.keys():
        game.all_objects[sprite] = game.getObjects()[sprite]

    game.movement_options[sprite] = {k:{} for k in game.spriteDistribution[sprite].keys()}
    # game.object_token_movement_options[sprite] = {k:{} for k in game.spriteDistribution[sprite].keys()}


def updateDistribution(game, sprite, curr_distribution, movement_options, outcome, specialID=None, missileOrientationClustering=False):

    epsilon_prob = 0.000005

    # For computing the new normalized likelihoods, we proceed as follows:

    # The normalized likelihood for a given observation sequence o_1, ... o_t-1 given a parameter p_j is:
    #   p(o_1, ..., o_t-1|p_j) / sum_i(p(o_1, .., o_t-1|p_i))
    #
    # We want to arrive at the new normalized likelihoods p(o_1, ..., o_t-1, o_t|p_j) / sum_i(p(o_1, .., o_t-1, o_t|p_i))
    #
    # We first compute the ratio between the normalization constants:
    # sum_i(p(o_1, .., o_t-1, o_t|p_i)) / sum_j(p(o_1, .., o_t-1|p_j)) =
    # sum_i(p(o_1, .., o_t-1|p_i) * p(o_t|p_i)) / sum_j(p(o_1, .., o_t-1|p_j))
    #
    # Now, we can get the new normalized likelihood by doing:
    # p(o_1, ..., o_t-1, o_t|p_j) / sum_i(p(o_1, .., o_t-1, o_t|p_i)) =
    #   p(o_1, ..., o_t-1|p_j) / sum_i(p(o_1, .., o_t-1|p_i)) *
    #   p(o_t|p_j)
    #   sum_i(p(o_1, .., o_t-1|p_i) * p(o_t|p_i)) / sum_k(p(o_1, .., o_t-1|p_k))

# (('vgdlType', <class 'vgdl.ontology.Chaser'>), ('cooldown', 1), ('fleeing', False), ('speed', 0.1), ('stype', 'RED')

    normalization_ratio = 0
    alpha = 1.

    # if missileOrientationClustering and game.all_objects[sprite]['features']['color']=='RED' and game.all_objects[sprite]['position'][0]>250:
    #     print "BEFORE UPDATE"
    #     print game.all_objects[sprite]['position'], outcome
    #     # print movement_options[sprite][(('vgdlType', RandomNPC), ('cooldown', 3), ('speed', 0.2))]
    #     print "missile", movement_options[sprite][(('vgdlType', Missile), ('cooldown', 1), ('orientation', (1,0)), ('speed', 0.1))]
    #     print "missile prob", curr_distribution[sprite][(('vgdlType', Missile), ('cooldown', 1), ('orientation', (1,0)), ('speed', 0.1))]
    #     print "chaser", movement_options[sprite][(('vgdlType', Chaser), ('cooldown', 1),  ('fleeing', False), ('speed', 0.1), ('stype', 'RED'))]
    #     print "chaser prob", curr_distribution[sprite][(('vgdlType', Chaser), ('cooldown', 1),  ('fleeing', False), ('speed', 0.1), ('stype', 'RED'))]
    #     embed()

    # if game.all_objects[sprite]['features']['color']=='ORANGE':
        # print "BEFORE UPDATE"
        # print game.all_objects[sprite]['position'], outcome
        # print game.spriteUpdateDict[sprite]
        # print movement_options[sprite][(('vgdlType', Chaser),('cooldown', 2),('fleeing', False), ('speed', 0.1), ('stype', 'DARKBLUE'))]
    #     print "missile", movement_options[sprite][(('vgdlType', ResourcePack),)]
    #     print "missile prob", curr_distribution[sprite][(('vgdlType', ResourcePack),)]
        # embed()


    if sprite in curr_distribution.keys():
        for param_combination in curr_distribution[sprite].keys():
            if outcome in movement_options[sprite][param_combination].keys():
                if missileOrientationClustering and 'Missile' in str(param_combination[0][1]):
                    normalization_ratio += curr_distribution[sprite][param_combination] * (movement_options[sprite][param_combination][outcome]**alpha)
                else:
                    normalization_ratio += curr_distribution[sprite][param_combination] * movement_options[sprite][param_combination][outcome]
            else:
                normalization_ratio += curr_distribution[sprite][param_combination] * epsilon_prob

    if sprite in curr_distribution.keys():
        for param_combination in curr_distribution[sprite].keys():
            if outcome in movement_options[sprite][param_combination].keys():
                if missileOrientationClustering and 'Missile' in str(param_combination[0][1]):
                    curr_distribution[sprite][param_combination] *= ((movement_options[sprite][param_combination][outcome]**alpha) / normalization_ratio)
                else:
                    curr_distribution[sprite][param_combination] *= (movement_options[sprite][param_combination][outcome] / normalization_ratio)
            else:
                curr_distribution[sprite][param_combination] *= (epsilon_prob / normalization_ratio)

    # if missileOrientationClustering and game.all_objects[sprite]['features']['color']=='GREEN':
    #     print "AFTER UPDATE"
    #     print game.all_objects[sprite]['position'], outcome
    #     print game.spriteUpdateDict[sprite]
    #     # print movement_options[sprite][(('vgdlType', RandomNPC), ('cooldown', 3), ('speed', 0.2))]
    #     print "missile", movement_options[sprite][(('vgdlType', ResourcePack),)]
    #     print "missile prob", curr_distribution[sprite][(('vgdlType', ResourcePack),)]
        # embed()


    # if missileOrientationClustering and game.all_objects[sprite]['features']['color']=='RED' and game.all_objects[sprite]['position'][0]>250:
    #     print "AFTER UPDATE"
    #     print game.all_objects[sprite]['position'], outcome
    #     # print movement_options[sprite][(('vgdlType', RandomNPC), ('cooldown', 3), ('speed', 0.2))]
    #     print "missile", movement_options[sprite][(('vgdlType', Missile), ('cooldown', 1), ('orientation', (1,0)), ('speed', 0.1))]
    #     print "missile prob", curr_distribution[sprite][(('vgdlType', Missile), ('cooldown', 1), ('orientation', (1,0)), ('speed', 0.1))]
    #     print "chaser", movement_options[sprite][(('vgdlType', Chaser), ('cooldown', 1),  ('fleeing', False), ('speed', 0.1), ('stype', 'RED'))]
    #     print "chaser prob", curr_distribution[sprite][(('vgdlType', Chaser), ('cooldown', 1),  ('fleeing', False), ('speed', 0.1), ('stype', 'RED'))]
    #     embed()
    return curr_distribution

# def updateDistribution(game, sprite, curr_distribution, movement_options, outcome, specialID=None, missileOrientationClustering=False):
#     """
#     Updates the sprite distribution for a given object in the game.

#     Input:
#         sprite - the current sprite ID
#         curr_distribution - the current sprite distribution for all objects
#         movement_options - possible next locations that the sprite of that sprite type can be in
#         outcome - the sprite's resulting location after the update

#     Output:
#         curr_distribution - renormalized updated distribution over sprite types for a given object

#     Here, we update both the distribution over sprite types as well as the distribution over the parameters
#     belonging to that sprite type.
#     Now we discuss the mathematical equations used for updating.
#     Let o = the outcome observed for a particular sprite object.
#     Let s = the sprite type of a particular sprite object.
#     Let p_1, ..., p_i, ..., p_k represent the parameters corresponding to
#     a particular sprite type.

#     The values stored in curr_distribution are p(s) and p(p_1 | s),..., p(p_i | s),..., p(p_k | s)
#     We would like to update them to their posterior values, i.e. p(s|o) and p(p_i | s, o)
#     (1) first we find the update equation for p(s|o).
#     Note that p(s|o) is proportional to
#     p(o|s)p(s) = p(s) * sum_{all combinations of p_1,..._p_k}{p(o|s,p_1,...,p_k)*p(p_1|s)*...*p(p_k|s)}
#     (2) Now we find the update equation for p(p_i | s, o):
#     Note that p(p_i | s, o) is proportional to
#     p(o|p_i, s)*p(p_i|s) =
#     p(p_i|s)*sum_{all combinations of values of p_1,...,p_k except p_i}{p(o|s,p_1,...,p_k)*p(p_1|s)*...*p(p_{i-1}|s)*p(p_{i+1}|s)*...*p(p_k|s)}
#     = sum_{all combinations of values of p_1,...,p_k except p_i}{p(o|s,p_1,...,p_k)*p(p_1|s)*...*p(p_k|s)}

#     The key observation for both of these update equations is that p(o|s,p_1,...,p_k) can be accessed
#     by using movement_options[sprite][sprite_type][param][outcome] (where param = a choice of p_1,...,p_k,
#     o = outcome, s = sprite_type, and sprite is the sprite object we are computing the new distribution for).
#     """

#     if sprite in curr_distribution.keys():
#         # if sprite in game.all_objects.keys() and game.all_objects[sprite]['sprite'].colorName=='ORANGE':
#             # ipdb.set_trace()
#         for sprite_type in curr_distribution[sprite].keys():
#             # if sprite_type == "OTHER":
#                 # sprite type is unknown.
#                 # movement_options[sprite][sprite_type][()] = {outcome: 1.0/5} #up down left right stay


#             if curr_distribution[sprite][sprite_type]['prob'] > 0:

#                 spriteTypeLikelihood = 0.
#                 newParameterLikelihood = {}
#                 for p in curr_distribution[sprite][sprite_type]['args']:
#                     newParameterLikelihood[p] = \
#                     {val: 0 for val in curr_distribution[sprite][sprite_type]['args'][p]}

#                 for param in movement_options[sprite][sprite_type]:
#                 # If the outcome is an option for the sprite type, update probability
#                     if outcome in movement_options[sprite][sprite_type][param].keys():
#                         attributeProduct = 1.
#                         for p, val in param:
#                             attributeProduct *= curr_distribution[sprite][sprite_type]['args'][p][val]

#                         spriteTypeLikelihood += movement_options[sprite][sprite_type][param][outcome] * attributeProduct

#                         for p, val in param:
#                             if 'Missile' in str(sprite_type) and missileOrientationClustering:
#                                 newParameterLikelihood[p][val] += movement_options[sprite][sprite_type][param][outcome]**2 * attributeProduct
#                             else:
#                                 newParameterLikelihood[p][val] += movement_options[sprite][sprite_type][param][outcome]*attributeProduct

#                 curr_distribution[sprite][sprite_type]['prob'] *= spriteTypeLikelihood
#                 curr_distribution[sprite][sprite_type]['args'] = newParameterLikelihood

#             # ch = [k for k in movement_options[sprite].keys() if 'Chaser' in str(k)][0]
#             # rp = [k for k in movement_options[sprite].keys() if 'Resource' in str(k)][0]

#             # if sprite==specialID and sprite_type==ch:
#             #     print specialID

#         epsilon_prob = 0.005
#         # want to make sure we don't entirely rule out certain sprite types and parameters
#         # but only place a small probability on them. In order to do that, we add a small positive value
#         # called epsilon_prob to each one of the probabilities in the distribution. This ensures
#         # that we don't entirely rule out a sprite type or parameter choice (by making its probability nonzero)
#         for sprite_type in curr_distribution[sprite].keys():
#             curr_distribution[sprite][sprite_type]['prob'] += epsilon_prob
#             for p in curr_distribution[sprite][sprite_type]['args']:
#                 for pval in curr_distribution[sprite][sprite_type]['args'][p]:
#                     curr_distribution[sprite][sprite_type]['args'][p][pval] += epsilon_prob


#         # Re-normalize the distribution
#         z = sum([curr_distribution[sprite][sprite_type]['prob'] for sprite_type in curr_distribution[sprite]])
#         for sprite_type in curr_distribution[sprite].keys():
#             curr_distribution[sprite][sprite_type]['prob'] /= z

#         for sprite_type in curr_distribution[sprite]:
#             for p in curr_distribution[sprite][sprite_type]['args']:
#                 z = sum(curr_distribution[sprite][sprite_type]['args'][p].values())
#                 if z > 0:
#                     for val in curr_distribution[sprite][sprite_type]['args'][p]:
#                         curr_distribution[sprite][sprite_type]['args'][p][val] /= z

#     return curr_distribution

def sampleFromDistribution(game, curr_distribution, all_objects, spriteUpdateDict, bestSpriteTypeDict, oldSpriteSet = None, skipInduction=False, display=False):

    import random
    import numpy as np
    from class_theory_template import Sprite
    from ontology import ResourcePack

    distributionsHaveChanged = False

    sample = []
    exceptions = []

    ##remove avatar. For now let's just assume we know which one it is.
    ##TODO: You need to do avatarInduction, unless there's a generic type that can cover all types.
    non_avatar_keys = []
    for k in all_objects.keys():
        if all_objects[k]['sprite'].name != 'avatar':
            non_avatar_keys.append(k)
        else:
            # from ontology import MovingAvatar, HorizontalAvatar, VerticalAvatar, FlakAvatar, AimedFlakAvatar, OrientedAvatar, \
            #     RotatingAvatar, RotatingFlippingAvatar, NoisyRotatingFlippingAvatar, ShootAvatar, AimedAvatar, \
            #         AimedFlakAvatar

            avatar_type = all_objects[k]['sprite'].__class__
            if avatar_type in [FlakAvatar, AimedFlakAvatar, ShootAvatar, AimedAvatar, AimedFlakAvatar]:
                try:
                    ## Add avatar, and add the attached arguments, i.e., what the avatar shoots.
                    sample.append(Sprite(vgdlType=all_objects[k]['sprite'].__class__, color=all_objects[k]['type']['color'], args={'stype':all_objects[k]['sprite'].stype}))

                    ## Get the object the Avatar shoots, add that.
                    projectile_name = all_objects[k]['sprite'].stype
                    ao = game.sprite_constr[projectile_name]
                    ao_vgdl_type = ao[0]
                    ao_color = colorDict[str(ao[1]['color'])]
                    ao_args = ao[1]
                    if projectile_name in game.singletons:
                        ao_args.update({'singleton': 'True'})
                    sample.append(Sprite(vgdlType=ao_vgdl_type, color=ao_color, className=all_objects[k]['sprite'].stype, args=ao_args))
                    exceptions.append(ao_color)

                except AttributeError:
                    print "tried and failed to add a shooting avatar type"
                    # embed()
                    # No args in avatar
                    sample.append(Sprite(vgdlType=MovingAvatar, color=all_objects[k]['type']['color']))
            else:
                sample.append(Sprite(vgdlType=avatar_type, color=all_objects[k]['type']['color']))



    ##unique types. TODO: Change to type index, not color. See note in runInduction_DFS for details.
    types = list(set([all_objects[k]['type']['color'] for k in non_avatar_keys]) - set(exceptions)) ## We are treating (for now) the object shot by a ShootAvatar, FlakAvatar, etc. separately
                                                                                                    ## and not doing inference about it.
    best_params = {}

    if skipInduction:
        for obj_type in types:
            s = Sprite(vgdlType=ResourcePack, color=obj_type)
            sample.append(s)
        return sample, exceptions, distributionsHaveChanged, best_params

    for obj_type in types:
        
        if obj_type in ['DARKGRAY', 'MPUYEI', 'NUPHKK', 'SCJPNE']:
            s = Sprite(vgdlType=ResourcePack, color=obj_type)
            sample.append(s)
            continue

        ## Integrate evidence across all episodes; pick best hypothesis.
        # try:
        #     param_product = {k:1. for k in bestSpriteTypeDict[obj_type].values()[0].keys()}

        # except IndexError:
        #     # bestSpriteTypeDict has yet to be populated for this object type
        #     for k, v in game.getObjects().items():
        #         if v['features']['color'] == obj_type:
        #             bestSpriteTypeDict[obj_type][k] = game.spriteDistribution[k]

        #     param_product = {k:1. for k in bestSpriteTypeDict[obj_type].values()[0].keys()}

        # spriteUpdateNormalizer = 0
        # for k in bestSpriteTypeDict[obj_type].keys():
            # spriteUpdateNormalizer += spriteUpdateDict[k]
        # if obj_type=='RED':
            # embed()
        
        ######

        k1 = (('vgdlType', ResourcePack),)
        

        ## Integrate evidence across all episodes; pick best hypothesis.

        numDict = defaultdict(lambda:[])
        for k in bestSpriteTypeDict[obj_type].keys():
            numDict[spriteUpdateDict[k]].append(k)

        try:
            param_sum = {k:0. for k in bestSpriteTypeDict[obj_type].values()[0].keys()}
        except IndexError:
            # bestSpriteTypeDict has yet to be populated for this object type
            for k, v in game.getObjects().items():
                if v['features']['color'] == obj_type:
                    bestSpriteTypeDict[obj_type][k] = game.spriteDistribution[k]
            param_sum = {k:1. for k in bestSpriteTypeDict[obj_type].values()[0].keys()}

        param_z = 0

        for num, IDs in numDict.items():
            # param_product = {k:1. for k in bestSpriteTypeDict[obj_type].values()[0].keys()}
            for param in param_sum.keys():
                # param_sum[param] += num*np.prod([bestSpriteTypeDict[obj_type][ID][param] for ID in IDs])/float(len(IDs))
                # param_z += num*np.prod([bestSpriteTypeDict[obj_type][ID][param] for ID in IDs])/float(len(IDs))

                tmp_prod = 1.
                for ID in IDs:
                    if param in bestSpriteTypeDict[obj_type][ID]:
                        tmp_prod *= bestSpriteTypeDict[obj_type][ID][param]
                    elif 'Chaser' in str(param[0][1]):
                        cooldown = [p[1] for p in param if p[0]=='cooldown']
                        cooldown = cooldown[0] if cooldown else 1
                        speed = [p[1] for p in param if p[0]=='speed']
                        speed = speed[0] if speed else 1
                        randomnpc_param = (
                            ('vgdlType', RandomNPC),
                            ('cooldown', cooldown),
                            ('speed', speed)
                        )
                        tmp_prod *= bestSpriteTypeDict[obj_type][ID][randomnpc_param]
                    else:
                        print "problem in param_product"
                        embed()

                param_sum[param] += num*tmp_prod
                param_z += num*tmp_prod
                # param_z += param_product[param]
                # param_sum[param] += param_product[param]
                # param_z += param_product[param]
        
        if param_z != 0:
            for param,val in param_sum.items():
                param_sum[param] /= param_z


        best_param = max(param_sum, key=param_sum.get)
        # if obj_type=='GREEN':
            # embed()
        ######

        # z = 0.
        # param_z = 0.
        # for k in bestSpriteTypeDict[obj_type].keys():
        #     # if obj_type == 'RED':
        #     #     print "before update"
        #         # k1 = (('vgdlType', Missile), ('cooldown', 1), ('orientation', (1,0)), ('speed', 0.1))
        #         # k2 = (('vgdlType', Chaser), ('cooldown', 1),  ('fleeing', False), ('speed', 0.1), ('stype', 'RED'))
        #         # print "missile", bestSpriteTypeDict['RED'][k][k1]
        #         # print "chaser", bestSpriteTypeDict['RED'][k][k2]
        #     #     embed()
        #     if obj_type == 'GREEN':
        #         print "before update"
        #         k1 = (('vgdlType', ResourcePack),)
        #         print spriteUpdateDict[k], bestSpriteTypeDict[obj_type][k][k1]
        #         # print "missile", bestSpriteTypeDict['GREEN'][k][k1]
        #         # embed()

        #     for param in param_product.keys():
        #         try:
        #             param_product[param] *= spriteUpdateDict[k]*bestSpriteTypeDict[obj_type][k][param]
        #             param_z += param_product[param]

        #         except KeyError:
        #             print("Got key error when updating param_product")
        #             # If we landed here because we're trying to update the hypothesis
        #             # for the sprite chasing an object that wasn't around before,
        #             # this hypothesis should borrow the parameters from RandomNPC
        #             # with the same speed, as that's what Chaser behaviour defaults
        #             # to in the absence of its chasee.
        #             if 'Chaser' in str(param[0][1]):
        #                 cooldown = [p[1] for p in param if p[0]=='cooldown']
        #                 cooldown = cooldown[0] if cooldown else 1
        #                 speed = [p[1] for p in param if p[0]=='speed']
        #                 speed = speed[0] if speed else 1
        #                 randomnpc_param = (
        #                     ('vgdlType', RandomNPC),
        #                     ('cooldown', cooldown),
        #                     ('speed', speed)
        #                 )
        #                 param_product[randomnpc_param] *= spriteUpdateDict[k]*bestSpriteTypeDict[obj_type][k][randomnpc_param]
        #                 param_z += param_product[randomnpc_param]

        #             else:  # if this is not a new chaser, we shouldn't be landing here
        #                 embed()
        #     z += spriteUpdateDict[k]

        # # for k in param_product:
        # #     try:
        # #         param_product[k] /= z
        # #     except ZeroDivisionError:
        # #         pass

        # if param_z != 0:
        #     for param,val in param_product.items():
        #         param_product[param] /= param_z


        # if obj_type == 'GREEN':
        #     k1 = (('vgdlType', ResourcePack),)
        #     k2 = (('vgdlType', Chaser), ('cooldown', 1),  ('fleeing', False), ('speed', 0.1), ('stype', 'RED'))
        #     print "after update"
        #     embed()

        # if obj_type == 'BROWN':
        #     k1 = (('vgdlType', Missile), ('cooldown', 1), ('orientation', (1,0)), ('speed', 0.1))
        #     k2 = (('vgdlType', Chaser), ('cooldown', 1),  ('fleeing', False), ('speed', 0.1), ('stype', 'RED'))
        #     print "after update"
        #     embed()

        # null_hypothesis = [k for k in param_product.keys() if 'Resource' in str(k[0][1])][0]
        # best_param = max(param_product, key=param_product.get)
        

        # if param_product[null_hypothesis]==0 and param_product[best_param]==0:
        #     best_param = null_hypothesis
        #     # print "setting {} to null: {}".format(obj_type, null_hypothesis)
        # else:
        #     if best_param!=null_hypothesis and param_product[null_hypothesis]!=0 and (param_product[best_param]/param_product[null_hypothesis] > 2.):
        #         # print best_param, param_product[best_param]
        #         best_param = best_param
        #     else:
        #         # if obj_type=='RED':
        #             # embed()
        #             # print best_param, param_product[best_param], null_hypothesis, param_product[null_hypothesis]
        #         # print "setting {} to null, case 2: {}".format(obj_type, null_hypothesis)
        #         best_param = null_hypothesis
        # if obj_type=='GOLD':
            # embed()

        # Use for debugging sprite-type inference.
        # if obj_type=='RED':
        #     # goldobjs = [game.sprite_groups[k] for k in game.sprite_groups.keys() if game.sprite_groups[k] and game.sprite_groups[k][0].colorName=='BROWN']
        #     # print [g.rect for g in goldobjs[0]]
        #     for i,k in enumerate(sorted(param_sum, key=param_sum.get, reverse=True)):
        #         print(k, param_sum[k])
        #         if i>10:
        #             break
        #     print ""
        #     print best_param
            # embed()

        sprite_type = best_param[0][1]

        color = obj_type

        if sprite_type=='OTHER':
            # from ontology import RandomNPC
            # sprite_type = RandomNPC
            sprite_type = ResourcePack

        s = Sprite(vgdlType=sprite_type, color=color)

        param = dict(best_param[1:])
        setSpriteParams(param, s) # set the parameters for sprite s
        sample.append(s)
        ## Find matching object in the existing hypothesis
        try:
            if oldSpriteSet:
                if s.color in [sprite.color for sprite in oldSpriteSet]:
                    matchingSprite = [sprite for sprite in oldSpriteSet if s.color==sprite.color][0]
                    # If types are different, distributionsHaveChanged is true
                    if s.vgdlType!=matchingSprite.vgdlType:
                        distributionsHaveChanged = True
                        if display:
                            print ("Distributions for {} have changed from sprite type {} to {}".format(s.color, matchingSprite.vgdlType, s.vgdlType))
                    # If one of the args is None but not the other,
                    # distributionsHaveChanged is true
                    elif ((s.args==None and matchingSprite.args!=None) or
                        (s.args!=None and matchingSprite.args==None)):
                        distributionsHaveChanged = True
                        if display:
                            print ("Distribution args for {} have changed from {} to {}".format(s.color, s.args, matchingSprite.args))
                    elif (s.args and matchingSprite.args) != None:
                        # If args are different, except for the case where only an
                        # orientation is reversed (e.g. turnAround), then
                        # distributionsHaveChanged is true
                        for key in s.args.keys() + matchingSprite.args.keys():
                            try:
                                if not ((s.args[key] and matchingSprite.args[key])
                                    in ([LEFT, RIGHT] or [UP, DOWN])):
                                    if s.args[key] != matchingSprite.args[key]:
                                        distributionsHaveChanged = True
                                    if display:
                                        print ("Distribution args for {} have changed from {} to {}".format(s.color, s.args, matchingSprite.args))
                            except KeyError:
                                # If the new sprite has an arg that the old one
                                # doesn't, or vice-versa, then
                                # distributionsHaveChanged is true
                                distributionsHaveChanged = True
                                if display:
                                    print ("Distribution args for {} have changed from {} to {}".format(s.color, s.args, matchingSprite.args))

                else:
                    # print s.color, oldSpriteSet
                    # embed()
                    distributionsHaveChanged = True
        except:
            print "failed to find matching object in sampleFromDistribution"
            embed()

        # param = dict(best_param[1:])
        # setSpriteParams(param, s) # set the parameters for sprite s
        # sample.append(s)

    return sample, exceptions, distributionsHaveChanged, best_params

def checkIfDistributionsHaveChanged(game, spriteUpdateDict, bestSpriteTypeDict):

    all_objects = game.all_objects
    curr_distribution = game.spriteDistribution
    changes = False
    exceptions = []

    ## We don't do sprite inference for the avatar and for Flak
    non_avatar_keys = []
    for k in all_objects.keys():
        if all_objects[k]['sprite'].name is not 'avatar':
            non_avatar_keys.append(k)
        # else:
            # exceptions.append('BLUE')

    ##unique types. TODO: Change to type index, not color. See note in runInduction_DFS for details.
    types = list(set([all_objects[k]['type']['color'] for k in non_avatar_keys]) - set(exceptions)) ## We are treating (for now) the object shot by a ShootAvatar, FlakAvatar, etc. separately
                                                                                                    ## and not doing inference about it.
    for obj_type in types:
        ## find the most-updated object, use that one for the sprite hypothesis.
        options = [k for k in all_objects.keys() if all_objects[k]['type']['color'] == obj_type]
        k = max(options, key=lambda x:spriteUpdateDict[x])

        oldDistribution = bestSpriteTypeDict[obj_type]['distribution']

        if spriteUpdateDict[k] >= bestSpriteTypeDict[obj_type]['count']: ## If we have more observations in the current episode than in our memory, use the current distribution
            # k = random.choice(options)
            ## always alphabetize the keys
            ## sample multinomially from the spriteDistribution[key] dictionary, to get the spriteType
            ## add that to the color info for that object.
            if k not in curr_distribution.keys():
                print k, "not in curr_distribution"
                embed()
            sprite_possibilities = curr_distribution[k]
        else:
            sprite_possibilities = bestSpriteTypeDict[obj_type]['distribution']

        newDistribution = sprite_possibilities

    return False

def getKL(spriteDistribution1, spriteDistribution2):
    d1, d2 = [v['prob'] for v in spriteDistribution1.values()], [v['prob'] for v in spriteDistribution2.values()]
    return scipy.stats.entropy(d1,d2)


# def spriteInductionProfiler(game, step, bestSpriteTypeDict, oldSpriteSet=None, old_outcome=None):
#     lp = LineProfiler()
#     lp_wrapper = lp(spriteInduction)
#     distributionsHaveChanged = lp_wrapper(game, step, bestSpriteTypeDict, oldSpriteSet, old_outcome)
#     lp.print_stats()
#     return distributionsHaveChanged


def spriteInduction(game, step, bestSpriteTypeDict, oldSpriteSet=None, old_outcome=None):
    """
    An explanation of important data structures used in this function:
    game = a BasicGame object
    game.spriteDistribution is a dictionary of the following form:
    {sprite: {sprite_type: {'prob': PROBABILITY OF SPRITE TYPE, 'args': {'speed': {A VALUE OF SPEED: PROBABILITY OF THAT VALUE}}},
    ...}, ...}
    game.spriteDistribution tells you the probability of a sprite being being a particular type. It also
    tells you the probability distribution over values for each parameter (e.g. speed, orientation).
    game.movement_options is a dictionary of the following form:
    {sprite: {sprite_type: {attributeTuple: {sprite position: probability of that sprite position},...},
    ...}, ...}
    game.movement_options tells you the probability of a sprite being in a particular position, given a certain
    setting of its attributes (e.g. specific values for speed, orientation, etc.) and also given sprite type.
    """
    distributionsHaveChanged = False
    # embed()
    if step==0:
    ## Prep for sprite induction
        objects = game.getObjects()
        for sprite in objects:
            if objects[sprite]['sprite'].colorName not in ['DARKGRAY', 'MPUYEI', 'NUPHKK', 'SCJPNE']:
                distributionInitSetup(game, sprite)
    elif step==1:
        ## Sprite Induction Part 1:
        ## every time you act, make sure there aren't new objects
        ## if there are, update spriteDistribution etc.
        objects = game.getObjects()
        kill_list_keys = [s.ID for s in game.kill_list]
        # print "step1"
        # print objects.keys()
        spritestoupdate = 0
        for sprite in objects:
            if objects[sprite]['sprite'].colorName not in ['DARKGRAY', 'MPUYEI', 'NUPHKK', 'SCJPNE'] and sprite not in game.spriteDistribution:
                spritestoupdate+=1
                game.all_objects[sprite] = objects[sprite]
                distributionInitSetup(game, sprite)
        # print "sprites to update step 1:", spritestoupdate
    elif step == 2:
        ## See the update options for each sprite type the sprite could be
        objects = game.getObjects()
        # notUpdated = [s for s in objects.keys() if objects[s]['sprite'].colorName not in ['DARKGRAY', 'MPUYEI', 'NUPHKK', 'SCJPNE'] and s not in game.spriteDistribution.keys()]

        game = game                                               # Save game state
        sprite_count, param_count=0, 0
        # kill_list_keys = [s.ID for s in game.kill_list]
        # print "len spriteDistribution: {}, objects: {}".format(len(game.spriteDistribution.keys()), len(objects.keys()))
        # print "len filtered keys: {}".format(len([s for s in game.spriteDistribution.keys() if s not in kill_list_keys]))

        for sprite in [s for s in game.spriteDistribution.keys() if s in objects.keys()]:                  # Keys are the IDs of the game objects
            sprite_count +=1
            sprite_obj = objects[sprite]["sprite"]

            if sprite_obj.name !='avatar':
                # print "Updating {} because it moved. Lastrect:{}, rect: {}".format(sprite_obj.colorName, sprite_obj.lastrect, sprite_obj.rect)
                for param_combination in game.spriteDistribution[sprite].keys(): # Check each potential sprite type
                    if game.spriteDistribution[sprite][param_combination]> 0:    # Make sure sprite_type is an option for sprite, and sprite is not killed
                        param_count +=1
                        # sprite_obj = objects[sprite]["sprite"]

                        sprite_type = param_combination[0]
                        attributeDict = {k:v for k,v in param_combination[1:]}

                        # Get potential next positions for sprite if it were that sprite type
                        # if sprite_obj.name!='avatar':
                        ##we are sprite_obj, and we are updating the options for where it could be next contingent on its being 'sprite_type'
                        # given a set of potential attribute values, update the movement options
                        # for this attribute tuple (i.e. candidate set of parameters)

                        ## missileOrientationClustering: considers left/right and up/down to be equivalent options in the likelihood
                        ## so that when objects bounce off walls it doesn't dramatically reduce the probability that they are straight-moving
                        ## objects
                        # if 'Chaser' in str(sprite_type[1]):
                            # print param_combination
                            # game.object_token_movement_options[sprite][param_combination], game.movement_options[sprite][param_combination] = \
                            # updateOptionsProfiler(game, sprite_type, sprite_obj, params=attributeDict, missileOrientationClustering=True)
                        # else:


                        # game.object_token_movement_options[sprite][param_combination], game.movement_options[sprite][param_combination] = \
                            # updateOptions(game, sprite_type, sprite_obj, params=attributeDict, missileOrientationClustering=True)

                        _, game.movement_options[sprite][param_combination] = \
                            updateOptions(game, sprite_type, sprite_obj, params=attributeDict, missileOrientationClustering=True)

                # if sprite_obj.colorName=='RED':
                #     randKey = [k for k in game.object_token_movement_options[sprite].keys() if 'Random' in str(k[0][1]) and k[1][1]==1 and k[2][1]==1][0]
                #     print game.object_token_movement_options[sprite][randKey]
                #     embed()
        game.targetColorDict = dict()
        game.chaserMovesTowardDict = dict()
        # print "step 2 updated {} sprites and {} param combinations".format(sprite_count, param_count)

    elif step==3:
        ## Update sprite distribution based on observations
        objects = game.getObjects()
        # notUpdated = [s for s in objects.keys() if objects[s]['sprite'].colorName not in ['DARKGRAY', 'MPUYEI', 'NUPHKK', 'SCJPNE'] and s not in game.spriteDistribution.keys()]
        # t1 = time.time()
        for sprite in [s for s in game.spriteDistribution.keys() if s in objects.keys() and s not in [k.ID for k in game.kill_list]]:        # Keys are the IDs of the game objects
            sprite_obj = objects[sprite]["sprite"]
            # sprite_count +=1

            if all([sprite not in e for e in game.effectList if e[0]!='nothing']) and sprite not in game.ignoreList and sprite_obj.name != 'avatar':
                # only update the distribution in this fashion if there are no events for this
                # time step involving this sprite.
                # calls_to_update_distribution +=1
                outcome = objects[sprite]["position"]
                # if sprite_obj.lastmove>0:
                    # embed()
                game.spriteDistribution = updateDistribution(game, sprite, game.spriteDistribution, \
                                          game.movement_options, outcome, missileOrientationClustering=True)
                # game.object_token_spriteDistribution = updateDistribution(game, sprite, game.object_token_spriteDistribution, \
                                          # game.object_token_movement_options, outcome)

                game.spriteUpdateDict[sprite] += 1
        
                # if game.all_objects[sprite]['features']['color']=='RED':
                #     print "missile prob", game.spriteDistribution[sprite][(('vgdlType', Missile), ('cooldown', 1), ('orientation', (1,0)), ('speed', 0.1))]
                #     print "chaser prob", game.spriteDistribution[sprite][(('vgdlType', Chaser), ('cooldown', 1),  ('fleeing', False), ('speed', 0.1), ('stype', 'RED'))]
                #     embed()
        # print "step 3 updated {} sprites and {} param combinations, took {} seconds.".format(sprite_count, calls_to_update_distribution, t1-time.time())
        ## Update the global memory
        for k in game.spriteDistribution.keys():
            try:
                color = game.all_objects[k]['type']['color']
            except KeyError:
                print("got key error when trying to access sprite color")
                embed()
            bestSpriteTypeDict[color][k] = game.spriteDistribution[k]

        # t1 = time.time()
        sample, exceptions, distributionsHaveChanged, _ = sampleFromDistribution(game, game.spriteDistribution, game.all_objects, game.spriteUpdateDict, bestSpriteTypeDict, oldSpriteSet = oldSpriteSet)
        # print "sampleFromDistribution took {} seconds".format(t1-time.time())
    ## Reset ignoreList so that next time around you do inference.
    game.ignoreList = []
    return distributionsHaveChanged



def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def selectObjectGoal(rle, unknown_colors, all_colors, exclude_colors, method):
    def dist(a,b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])

    epsilon = .2
    unknown_colors = [c for c in unknown_colors if c not in exclude_colors]
    safe_colors = [c for c in all_colors if c not in exclude_colors]
    if method=='random_then_nearest':
        if len(unknown_colors)>0 and random.random()>epsilon:
            print "selecting an unknown color"
            object_color = random.choice(unknown_colors)
        else:
            # in case we've interacted with everything once but want to randomly try things again
            print "sometimes with probability", epsilon, "we select randomly from all safe colors. This just happened."
            object_color = random.choice(safe_colors)
        choices = [item for sublist in rle._game.sprite_groups.values() for item in sublist if colorDict[str(item.color)]==object_color]
        avatar_loc = rle._rect2pos(rle._game.sprite_groups['avatar'][0].rect)
        choices = [(dist(rle._rect2pos(c.rect), avatar_loc), c) for c in choices]
        choices = sorted(choices, key=lambda c:c[0])
        nearest_dist = min([c[0] for c in choices])
        nearest = [c for c in choices if c[0]==nearest_dist]
        return random.choice(nearest)[1]
