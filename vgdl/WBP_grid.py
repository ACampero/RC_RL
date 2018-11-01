from IPython import embed
import itertools
import numpy as np
from numpy import zeros
import pygame
from ontology import BASEDIRS
from core import VGDLSprite, colorDict, sys
from stateobsnonstatic import StateObsHandlerNonStatic
from rlenvironmentnonstatic import *
import argparse
import random
import math
from threading import Thread
from collections import defaultdict, deque
import time
import ipdb
import copy
from threading import Lock
from Queue import Queue
from util import *
import multiprocessing
from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution, spriteInduction, selectObjectGoal
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, SpriteCounterRule, MultiSpriteCounterRule, \
NoveltyRule, generateSymbolDict, ruleCluster, Theory, Game, writeTheoryToTxt, generateTheoryFromGame
from rlenvironmentnonstatic import createRLInputGame

# from line_profiler import LineProfiler
import cPickle

from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT
NONE = 0
ACTIONS = [K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, NONE]
actionDict = {K_SPACE: 'space', K_UP: 'up', K_DOWN: 'down', K_LEFT: 'left', K_RIGHT: 'right', NONE: 'wait'}

## Base class for width-based planners (IW(k) and 2BFS)
class WBP():
    def __init__(self, rle, gameFilename, theory=None, fakeInteractionRules = [], seen_limits=[], annealing=1, max_nodes=100000, shortHorizon=False,
        firstOrderHorizon=False):
        self.rle = rle
        self.gameFilename = gameFilename
        self.T = len(rle._obstypes.keys())+1 #number of object types. Adding avatar, which is not in obstypes.
        self.vecDim = [rle.outdim[0]*rle.outdim[1], 2, self.T]
        self.trueAtoms = defaultdict(lambda:0) #set() ## set of atoms that have been true at some point thus far in the planner.
        self.objectTypes = rle._game.sprite_groups.keys()
        self.objectTypes.sort()
        self.phiSize = sum([len(rle._game.sprite_groups[k]) for k in rle._game.sprite_groups.keys() if k not in ['wall', 'avatar']])
        self.seen_limits = seen_limits
        self.objIDs = {}
        self.solution = None
        self.trackTokens = False
        self.vecSize = None
        self.addWaitAction = False
        self.annealing = annealing
        self.statesEncountered = []
        self.padding = 5  ##5 is arbitrary; just to make sure we don't get overlap when we add positions
        self.objectNumberTrackingLimit = 50
        self.objectLocationTrackingLimit = 8
        self.max_nodes = max_nodes
        self.objectsWhoseLocationsWeIgnore = ['Flicker', 'Random']
        self.objectsWhosePresenceWeIgnore = ['Flicker']
        self.classesWhoseLocationsWeIgnore = []
        self.classesWhosePresenceWeIgnore = []
        self.allowRollouts = True
        self.quitting = False
        self.gameString_array = []
        if theory == None:
            self.theory = generateTheoryFromGame(rle, alterGoal=False)
        else:
            self.theory=copy.deepcopy(theory)
            self.theory.interactionSet.extend(fakeInteractionRules)
            self.theory.updateTerminations()
        print 'max nodes', self.max_nodes

        # for rule in self.theory.interactionSet:
        #     if 'stepBack'==rule.interaction:
        #         ipdb.set_trace()
        i=1
        for k in rle._game.all_objects.keys():
            self.objIDs[k] = i * 100 * (rle.outdim[0]*rle.outdim[1]+self.padding)
            i+=1
        self.addSpaceBarToActions()
        self.pixel_size = self.rle._game.screensize[0]/self.rle._game.width
        self.visited_positions = np.zeros(np.array(self.rle._game.screensize)/
            self.pixel_size)

        self.short_horizon = shortHorizon
        self.winning_states = []
        self.trueAtomsIW1 = []

        ## Ignore objects we don't want to track (i.e., non-moving immovables.)
        self.objectsToTrack = []
        for k in rle._game.sprite_groups.keys():
            if ((k in self.theory.classes.keys() and ('Resource' or 'Immovable') in str(self.theory.classes[k][0].vgdlType) and not \
            (('bounceForward' or 'pullWithIt') in [rule.interaction for rule in self.theory.interactionSet if k in [rule.slot1, rule.slot2]])) or
            len(rle._game.sprite_groups[k])>self.objectNumberTrackingLimit):
                pass# self.objectsToNotTrackInAtomList.append(k)
            else:
                self.objectsToTrack.append(k)

            ## Don't track (in either way) objects that are very numerous; completely breaks calculateAtoms()
            if len(rle._game.sprite_groups[k])>self.objectNumberTrackingLimit:
                self.classesWhosePresenceWeIgnore.append(k)
            if len(rle._game.sprite_groups[k])>self.objectLocationTrackingLimit:
                self.classesWhoseLocationsWeIgnore.append(k)

        print "ignoring presences for", self.classesWhosePresenceWeIgnore
        print "ignoring locations for", self.classesWhoseLocationsWeIgnore
        # Compute starting number of each SpriteCounter stype
        self.firstOrderHorizon = firstOrderHorizon
        self.starting_stype_n = {}
        for term in self.theory.terminationSet:
            if isinstance(term, SpriteCounterRule):
                stype = term.termination.stype
                n_stypes = len([0 for sprite in self.findObjectsInRLE(self.rle, stype)])
                self.starting_stype_n[stype] = n_stypes

    def findObjectsInRLE(self, rle, objName):
        try:
            objLocs = [rle._rect2pos(element.rect) for element in rle._game.sprite_groups[objName]
            if element not in rle._game.kill_list]
        except:
            return None
        return objLocs

    def findAvatarInRLE(self, rle):
        avatar_loc = rle._rect2pos(rle._game.sprite_groups['avatar'][0].rect)
        return avatar_loc

    def addSpaceBarToActions(self):
        ## Note: if an object that isn't instantiated in the beginning is of a class that
        ## spacebar applies to, we won't pick up on it here.
        shootingClasses = ['MarioAvatar', 'ClimbingAvatar', 'ShootAvatar', 'Switch', 'FlakAvatar']
        classes = [str(o[0].__class__) for o in self.rle._game.sprite_groups.values() if len(o)>0]
        spacebarAvailable = False
        for sc in shootingClasses:
            if any([sc in c for c in classes]):
                spacebarAvailable = True
                break
        if spacebarAvailable:
            self.actions = [NONE, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT]
        else:
            self.actions = [NONE, K_UP, K_DOWN, K_LEFT, K_RIGHT]
        if self.addWaitAction:
            self.actions.append(NONE)
        return

    def calculateAtoms(self, rle):
        lst = []

        ## Track specific locations of objects
        for k in self.objectsToTrack:
            ## Don't track Flicker in atoms. The point is that the Flicker should have an effect on other objects, so atom novelty that would have been
            ## a function of the Flicker's presence is being taken care of by that. Otherwise the agent can keep exploring states that have no actual effect
            ## on the game state.
            if ((len(rle._game.sprite_groups[k])>0 and
                    rle._game.sprite_groups[k][0].colorName in self.theory.spriteObjects.keys() and
                    any([obj in str(self.theory.spriteObjects[rle._game.sprite_groups[k][0].colorName].vgdlType) for obj in self.objectsWhoseLocationsWeIgnore])) or
                k in self.classesWhoseLocationsWeIgnore):

                pass
            else:
                # if rle._game.sprite_groups[k]:
                    # print rle._game.sprite_groups[k][0].colorName
                for o in rle._game.sprite_groups[k]:
                    if o not in rle._game.kill_list:
                        ## turn location into vector position (rows appended one after the other.)
                        # pos = rle._rect2pos(o.rect) #x,y
                        # vecValue = pos[1] + pos[0]*rle.outdim[0] + 1
                        pos = float(o.rect.left)/rle._game.block_size, float(o.rect.top)/rle._game.block_size
                        vecValue = 10*pos[1] + 10*pos[0]*rle.outdim[0] + 10
                    else:
                        vecValue = 0
                    try:
                        if k == rle._game.getAvatars()[0].stype:
                            # Add avatar orientation to atom
                            orientation = rle._game.sprite_groups[k][0].orientation
                            if orientation[0] < 0 and orientation[1] == 0:
                                vecValue += 0
                            elif orientation[0] > 0 and orientation[1] == 0:
                                vecValue += 100000
                            elif orientation[0] == 0 and orientation[1] < 0:
                                vecValue += 200000
                            elif orientation[0] == 0 and orientation[1] > 0:
                                vecValue += 300000
                    except (IndexError, AttributeError) as e:
                        pass
                    objPosCombination = self.objIDs[o.ID] + vecValue
                    # print("ObjId = {}, vecValue = {}".format(self.objIDs[o.ID], vecValue))
                    lst.append(objPosCombination)

        ## Track present/absent objects
        present = []
        for k in [t for t in self.objectTypes if t not in ['wall', 'avatar']]: ##maybe add the avatar to this global state

            ## Don't track certain types (i.e., Flickers, Randoms) in atoms. The point is that the Flicker should have an effect on other objects, so atom novelty that would have been
            ## a function of the Flicker's presence is being taken care of by that. Otherwise the agent can keep exploring states that have no actual effect
            ## on the game state.
            if (len(rle._game.sprite_groups[k])>0 and
                    rle._game.sprite_groups[k][0].colorName in self.theory.spriteObjects.keys() and
                    any([obj in str(self.theory.spriteObjects[rle._game.sprite_groups[k][0].colorName].vgdlType) for obj in self.objectsWhosePresenceWeIgnore]) or
                    k in self.classesWhosePresenceWeIgnore):
                pass
            else:
                for o in sorted(rle._game.sprite_groups[k], key=lambda s:s.ID):
                    if o not in rle._game.kill_list:
                        present.append(1)
                    else:
                        present.append(0)
        ind = sum([present[i]*2**i for i in range(len(present))])
        lst.append(ind)
        if not self.vecSize:
            self.vecSize = len(lst)
            # print "Vector is length {}".format(self.vecSize)
        return set(lst)

    def compareDicts(self, d1,d2):
        ## only tells us what is in d2 that isn't in d1, as well as differences in values between shared keys
        return [k for k in d2.keys() if (k not in d1.keys() or d1[k]!=d2[k])]

    def delta(self, node1, node2):
        if node1 is None:
            diff = node2.state
        else:
            diff = node2.state-node1.state
        return diff

    def noveltySelection(self, QNovelty, QReward):
        bestNodes = sorted(QNovelty, key=lambda n: (n.novelty, -n.intrinsic_reward))
        current = bestNodes.pop(0)
        QNovelty.remove(current)
        try:
            QReward.remove(current)
        except:
            pass
        return current

    def rewardSelection(self, QReward, QNovelty):
        # acceptableNodes = QReward
        acceptableNodes = filter(lambda n:n.novelty<3, QReward)
        acceptableNodes = filter(lambda n: (not n.terminal or n.win), acceptableNodes)
        print "accetable:", len(acceptableNodes)
        # if len(acceptableNodes)==0:
            # acceptableNodes = QReward
            # print "Removed filter"
            # embed()
        bestNodes = sorted(acceptableNodes, key=lambda n: (-n.intrinsic_reward, n.novelty))
        try:
            current = bestNodes.pop(0)
        except:
            print('reward selection error')
            # embed()
            return 'pickMaxNode'
        QReward.remove(current)
        try:
            QNovelty.remove(current)
        except:
            pass
        # self.trueAtomsIW1.append(current.stateIW1)
        return current

    """
    def BFS_profiler(self):
        lp = LineProfiler()
        lp_wrapper = lp(self.BFS)
        lp_wrapper()
        lp.print_stats()
    """


    def BFS(self):
        QNovelty, QReward = [], []
        visited, rejected = [], []
        start = Node(self.rle, self, [], None)
        start.rle = self.rle
        visited.append(start)
        start.eval()
        QNovelty.append(start)
        QReward.append(start)
        i=0

        while (len(QNovelty)>0 or len(QReward)>0) and i<self.max_nodes:
            """
            if i%2==0:
            else:
            """
            # current = self.noveltySelection(QNovelty, QReward)
            current = self.rewardSelection(QReward, QNovelty)
            print "visited:", len(visited)
            # print("node chosen has position score {}".format(current.position_score()))
            # print embed()
            if current in [None, 'pickMaxNode']:
                node = max(visited, key=lambda n:n.intrinsic_reward)

                parentNode = copy.deepcopy(node)
                self.solution = node.actionSeq

                gameString_array, object_positions_array = [], []
                while parentNode is not None:
                    gameString_array.append(parentNode.rle.show())
                    object_positions_array.append(copy.deepcopy(parentNode.rle))
                    parentNode = parentNode.parent
                self.gameString_array = gameString_array[::-1]
                self.object_positions_array = object_positions_array[::-1]

                self.quitting = True
                # return None
                return node, gameString_array, object_positions_array

            try:
                (x, y) = np.array((current.rle._game.getAvatars()[0].rect.x,
                    current.rle._game.getAvatars()[0].rect.y))/self.pixel_size
                self.visited_positions[x, y] += 1
            except IndexError:
                pass

            self.statesEncountered.append(current.rle._game.getFullState())

            print current.rle.show(indent=True)

            current.updateNoveltyDict(QNovelty, QReward)
            # embed()
            visited.append(current)

            for a in self.actions:
                skipAction = False
                try:
                    # If there's already a projectile on the screen
                    # and the projectile class is a singleton
                    # and the action chosen is shooting
                    if (self.findObjectsInRLE(current.rle, current.rle._game.getAvatars()[0].stype) and
                        bool(self.theory.classes[current.rle._game.getAvatars()[0].stype][0].args['singleton']) and
                        a == K_SPACE):
                        # Then skip the action
                        skipAction = True

                except (IndexError, AttributeError, TypeError) as e:
                    # embed()
                    pass

                if not skipAction:
                    child = Node(self.rle, self, current.actionSeq+[a], current)
                    child.eval()

                    if self.firstOrderHorizon:
                        # Return plan if first-order progress was made towards
                        # a win condition
                        foundWin = False
                        for term in self.theory.terminationSet:
                            if isinstance(term, SpriteCounterRule) and term.termination.win==True:
                                stypes = [term.termination.stype]
                            elif isinstance(term, MultiSpriteCounterRule) and term.termination.win==True:
                                stypes = term.termination.stypes
                            else:
                                stypes = []
                            for stype in stypes:
                                n_stypes = len([0 for sprite in self.findObjectsInRLE(child.rle, stype)])
                                if stype in self.starting_stype_n.keys() and self.starting_stype_n[stype] > n_stypes:
                                    child.terminal, child.win = True, True
                                    foundWin = True
                                    break
                            if foundWin:
                                break

                    if child.win:
                        # Get the gameString representation of the RLE at each
                        # timestep in the chosen solution, so as to be able to
                        # correct for stochasticity effects
                        # compare it to the agent's RLE at execution time and
                        self.winning_states.append(child)
                        node = child
                        gameString_array, object_positions_array = [], []
                        while node is not None:
                            gameString_array.append(node.rle.show(color='green'))
                            object_positions_array.append(node.rle)
                            node = node.parent
                        self.gameString_array = gameString_array[::-1]
                        self.object_positions_array = object_positions_array[::-1]

                        ended, win, t = child.rle._isDone(getTermination=True)
                        self.solution = child.actionSeq
                        self.statesEncountered.append(child.rle._game.getFullState())
                        print "win"
                        # print t
                        # embed()
                        # return child, gameString_array
                    else:
                        QNovelty.append(child)
                        QReward.append(child)
            i+=1

            if self.winning_states:
                print "we have {} winning states".format(len(self.winning_states))
                bestNodes = sorted(self.winning_states, key=lambda n: (-n.intrinsic_reward))
                bestNode = bestNodes[0]
                # gameString_array.append(bestNode.rle.show())
                # object_positions_array.append(copy.deepcopy(bestNode.rle))
                return bestNode, gameString_array, object_positions_array

            # print i
        self.solution = []#Node(self.rle, self, [], None)
        if i>=self.max_nodes:
            if self.short_horizon:
                print "playing with short horizon; reached max of {} nodes".format(self.max_nodes)
                node = max(visited, key=lambda n:n.intrinsic_reward)
                parentNode = copy.deepcopy(node)
                self.solution = node.actionSeq
                print self.solution
                gameString_array, object_positions_array = [], []
                while parentNode is not None:
                    gameString_array.append(parentNode.rle.show())
                    object_positions_array.append(copy.deepcopy(parentNode.rle))
                    parentNode = parentNode.parent
                self.gameString_array = gameString_array[::-1]
                self.object_positions_array = object_positions_array[::-1]
                # print "win"
                # embed()
                return node, gameString_array, object_positions_array
            else:
                # self.quitting = True
                print "Got no plan after searching {} nodes".format(self.max_nodes)
        return None, None, None

class Node():
    def __init__(self, rle, WBP, actionSeq, parent):
        self.rle = rle
        self.WBP = WBP
        self.actionSeq = actionSeq
        self.parent = parent
        self.state = {}
        self.candidates = []
        self.novelty = None
        self.reward = None
        self.intrinsic_reward = 0
        self.metabolic_cost = 0
        self.children = None
        # self.lastState = None
        self.reconstructed=False
        self.expanded = False
        self.rolloutDepth = 13#max(rle.outdim)
        if self.parent is not None:
            self.rolloutArray = parent.rolloutArray[1:]
        else:
            self.rolloutArray = []


## when to trigger rollouts, if any
## rollout length
## repeating rollouts if death? e.g., are they optimistic?
## multiple samples??
    def metabolics(self, rle, events, action, n=10, mult=.3):

        # metabolic_cost = 1./n
        metabolic_cost = 0
        # if action==32:
        if action!=NONE or action!=32:
            metabolic_cost += 20#1./n
            pass
        if len(events)>0:
            # metabolic_cost = .3
            if any([rle._game.sprite_groups['avatar'][0].ID in e and e[0]=='bounceForward' for e in events]):
                # metabolic_cost += .3#(1-1./n)*mult
                pass
            # if any([rle._game.sprite_groups['avatar'][0].ID in e and e[0]=='killSprite' for e in events]):
            #     metabolic_cost += 0.3
        # metabolic_cost = 0
        return metabolic_cost

    def rollout(self, Vrle):
        successfulRollout = False
        j=0
        while not successfulRollout:
            vrle = copy.deepcopy(Vrle)
            prevHeuristicVal = self.heuristics(vrle)
            rolloutArray = []
            i=0
            terminal, win = vrle._isDone()
            print "in rollout"
            while i<self.rolloutDepth and not terminal:
                a = random.choice([K_UP, K_DOWN, K_LEFT, K_RIGHT])
                # print a
                vrle.step(a)
                print vrle.show(indent=True)
                currHeuristicVal = self.heuristics(vrle)
                heuristicVal = currHeuristicVal-prevHeuristicVal
                rolloutArray.append(heuristicVal)
                prevHeuristicVal = currHeuristicVal
                # print vrle.show()
                terminal, win, t = vrle._isDone(getTermination=True)
                if terminal:
                    try:
                        if (t.name=='noveltyTermination' and
                            self.rle._game.getAvatars()[0].stype
                            not in [t.s1, t.s2]):
                            # If we have a novelty termination not involving
                            # the projectile, ignore it
                            terminal = False
                    except (IndexError, AttributeError) as e:
                        # Avatar is dead or doesn't have projectile
                        pass
                i+=1
            # embed()
            ## we want optimistic estimates of the future value of a shot. Take up to 100 samples but don't get caught in an infinite loop.
            if terminal and not win and j<100:
                successfulRollout = False
                print "rolling out again"
                j+=1
                # embed()
            else:
                successfulRollout = True

        if win:
            self.terminal = terminal
            self.win = win
        return rolloutArray

    def spritecounter_val(self, theory, term, stype, rle, first_alpha=10000.,
                          second_alpha=100):

        # First order: progress in terms of number of sprites remaining.
        # Second order: distance to the closest instance of a target sprite type.

        val = 0
        compute_second_order = True

        # Check if condition is win or loss and multiply accordingly
        if term.termination.win:
            mult = -1
        else:
            compute_second_order = False
            mult = .1

        # Get all types that kill or transform stype (the target)
        killer_types = [
            inter.slot2 for inter in theory.interactionSet
            if ((inter.interaction == 'killSprite' or
                 inter.interaction == 'transformTo') and
                 not inter.generic and
                 not inter.preconditions
                and inter.slot1 == stype)]

        ## If you can shoot a Flicker, give yourself credit for being close to things it kills, but remove credit for that Flicker being close to those things.
        try:
            if (rle._game.getAvatars()[0].stype in killer_types and
                'Flicker' in str(theory.spriteObjects[rle._game.sprite_groups[rle._game.getAvatars()[0].stype][0].colorName].vgdlType)):
                    killer_types.append(rle._game.getAvatars()[0].name)
                    killer_types.remove(rle._game.getAvatars()[0].stype)
        except (IndexError, AttributeError) as e:
            pass

        # This list comprehension checks whether the avatar kills the stype with a preconditioned
        # interaction, and if so adds 'avatar' to the list as well as the precondition for that rule
        avatar_preconditions = [
            (inter.slot2, inter.preconditions) for inter in theory.interactionSet
            if ((inter.interaction == 'killSprite' or
                inter.interaction == 'killIfOtherHasMore' or
                 inter.interaction == 'transformTo') and
                 not inter.generic
                 and inter.preconditions
                and inter.slot1 == stype)]

        tmp_list = []

        ## If we have preconditions, find the objects that we should go to given that we satisfy the relevant preconditions. E.g., if we have a key and want to know what
        ## happens with item x, go to it.
        for avatar in avatar_preconditions:

            precondition = list(avatar[1])[0]
            item, num, negated, operator_name = precondition.item, precondition.num, precondition.negated, precondition.operator_name
            if negated:
                oppositeOperatorMap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}
                true_operator = oppositeOperatorMap[operator_name]
            else:
                true_operator = operator_name
            try:
                current_resource = rle._game.sprite_groups[avatar[0]][0].resources[precondition.item]
                ## If we satisfy the precondiiton, append to tmp_list, then to killer_types (meaning we are capable of killing stype now)
                if eval("{}{}{}".format(current_resource, true_operator, num)):
                    print "reached resource limit"
                    tmp_list.append(avatar)
            except IndexError:
                pass

        for t in tmp_list:
            killer_types.append(t[0])
            avatar_preconditions.remove(t)

        # Get attributes from terminationSet
        limit = term.termination.limit

        if 'SpawnPoint' in str(theory.classes[stype][0].vgdlType) and not killer_types:
            distance_to_goal = 0
            ## Special case, where you want to track whether that spawnPoint has a limit, etc.
            ## Distance to goal here is how many sprites the spawnPoint still has to shoot before it expires.
            for o in rle._game.sprite_groups[stype]:
                distance_to_goal += abs(o.total-o.counter)
            val += mult * first_alpha * distance_to_goal
            return val
        else:
            ## Normal case
            n_stypes = len([0 for sprite in self.WBP.findObjectsInRLE(rle, stype)])

            distance_to_goal = abs(n_stypes - limit)

        if distance_to_goal!=0:
            val -= float(mult * first_alpha) / distance_to_goal**2 ## Penalize quadratically for classes for which we'd have to kill many instances.
        else:
            val -= mult*first_alpha ## we shouldn't go in here, as if we've actually destroyed the relevant sprite we'll trigger a win condition.

        # print "stype, n_stypes, distance_to_goal, val", stype, n_stypes, distance_to_goal, val
        if compute_second_order:
            ## Get all positions of objects whose type is in killer_types; compute minimum distance
            ## of each to the stypes we have to destroy. Return min over all mins.
            # embed()
            objs = [self.WBP.findObjectsInRLE(rle, ktype) for ktype in killer_types]

            if len(objs)>0:
                kill_positions = np.concatenate([o for o in objs if len(o)==max([len(obj) for obj in objs])])
            else:
                kill_positions = np.array(objs)

            stype_positions = self.WBP.findObjectsInRLE(rle, stype)
            try:
                # A consequence of the two-way generic interactions in the
                # theory is that minimum-distance object pairs whose interactions
                # were not yet observed will have their distance penalized twice
                # as much when none of those objects is an avatar. This implies
                # that avatar novel interactions will be favored over other ones
                possiblePairList = [manhattanDist(obj, pos)
                     for pos in kill_positions
                     for obj in stype_positions]

                distance = min(possiblePairList)
            except ValueError:
                distance = 0

            if possiblePairList:
                n_sprites = len(possiblePairList) ## TODO: you're normalizing by the number of possible pairs of killer_sprites and target_sprites; you should just normalize by the number of targets
                # Normalize by number of sprites, enforcing a prior that encourages
                # goals that involve killing fewer objects
                val += float(mult * second_alpha * distance)/n_sprites**2
            else:
                # This helps in cases in which either the stype or the killer_type is not always on the screen
                # Then, you should not be disincentivized to create it, which can be achieved through this high penalty
                distance = 100
                val += float(mult * second_alpha * distance)

            if avatar_preconditions:
                avatars = [self.WBP.findObjectsInRLE(rle, ktype[0]) for ktype in avatar_preconditions]

                resource_names = [list(resource[1])[0].item for resource in avatar_preconditions]

                try:
                    resource_yielder_names = [[inter.slot2 if (inter.interaction=='changeResource' and inter.args['resource']==res) else res if (inter.interaction=='collectResource' and res==inter.slot1) else None
                    for inter in theory.interactionSet] for res in resource_names]
                except:
                    print "failure with resource_yielder_names"
                    embed()

                resource_yielder_names = [[r for r in ryn if r] for ryn in resource_yielder_names] ## Remove 'None' yielded by last else condition above

                resource_positions = [np.concatenate([self.WBP.findObjectsInRLE(rle, yielder) for yielder in yielders]) for yielders in resource_yielder_names]

                resource_limits = np.array([list(resource[1])[0].num + 1
                    if list(resource[1])[0].operator_name == '>'
                    else list(resource[1])[0].num
                    for resource in avatar_preconditions])
                try:
                    avatar_resource_quantities = np.array([rle._game.getAvatars()[0].resources[res] for res in resource_names])
                except IndexError:
                    avatar_resource_quantities = np.array([0 for res in resource_names])
                precondition_distances = []
                try:
                    for (obj1_positions, obj2_positions) in zip(avatars, resource_positions):
                        # A consequence of the two-way generic interactions in the
                        # theory is that minimum-distance object pairs whose interactions
                        # were not yet observed will have their distance penalized twice
                        # as much when none of those objects is an avatar. This implies
                        # that avatar novel interactions will be favored over other ones
                        try:
                            possiblePairList = np.array([manhattanDist(obj1, obj2)
                                for obj1 in obj1_positions
                                for obj2 in obj2_positions])
                        except:
                            print "failure with obj1_positions"
                            embed()

                        precondition_distances.append(min(possiblePairList))

                    # effective_distance = min(precondition_distances/(resource_limits-avatar_resource_quantities))
                    physical_distance = min(precondition_distances)
                    sprite_n_distance = abs(resource_limits-avatar_resource_quantities)
                    # Normalize by number of sprites, enforcing a prior that encourages
                    # goals that involve killing fewer objects
                    val += float(mult * second_alpha * (physical_distance / 10.)) - 10000
                    val += float(mult * second_alpha * sprite_n_distance) - 10000

                    # print distance
                except ValueError:
                    if avatar_preconditions and avatars[0]:
                        print "valueError in spritecounter_val"
                        embed()
                    pass
                    # effective_distance = 0

                if not resource_positions:
                    # This helps in cases in which either the stype or the killer_type is not always on the screen
                    # Then, you should not be disincentivized to create it, which can be achieved through this high penalty
                    distance = 100
                    val += float(mult * second_alpha * distance) - 20000


            # if stype == 'avatar':
            #     # if the avatar's death depends on a precondition
            #     preconditions = [inter.preconditions for inter in theory.interactionSet if ((inter.interaction == 'killSprite') and (not inter.generic) and (inter.slot1 == stype) and (inter.preconditions))]
            #     for precondition_set in preconditions:
            #         precondition = list(precondition_set)[0]
            #         # Give intrinsic reward based on resource distance to kill value
            #         if precondition.negated:
            #             oppositeOperatorMap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}
            #             true_operator = oppositeOperatorMap[precondition.operator_name]
            #         else:
            #             true_operator = precondition.operator_name
            #         resource = precondition.item
            #         current_val = self.WBP.rle._game.getAvatars()[0].resources[resource]
            #         if true_operator in {"<", "<="}:
            #             val += mult * second_alpha * (precondition.num-current_val)
            #         elif true_operator in {">", ">="}:
            #             val += mult * second_alpha * (current_val-precondition.num)

        return val

    def multispritecounter_val(self, theory, term, rle, first_alpha=10000,
                               second_alpha=100):
        val = 0
        # print "in multispritecounter"
        # embed()
        for stype in term.termination.stypes:
            val += self.spritecounter_val(theory, term, stype, rle,
                first_alpha=first_alpha, second_alpha=second_alpha)
            # print stype, val
        return val

    def noveltytermination_val(self, theory, term, s1, s2, rle, first_alpha=1000,
                          second_alpha=10):
        val = 0
        compute_second_order = True

        # Check if condition is win or loss and multiply accordingly
        if term.termination.win:
            mult = -1
        else:
            compute_second_order = False
            mult = 1

        ## Don't give heuristic bonus for using the flicker. But the agent is still incentivized to try to make the flicker interact with other objects
        ## because of noveltyTerminationConditions.
        if 'Flicker' in str(theory.classes[s1][0].vgdlType) or 'Flicker' in str(theory.classes[s2][0].vgdlType):
            return 0, 10000

        ## If the terminationRule is precondition-dependent, check that first. Don't give heuristic val if the preconditions aren't fulfilled.
        if term.termination.args:
            item, num, negated, operator_name = term.termination.args.item, term.termination.args.num, term.termination.args.negated, term.termination.args.operator_name
            if negated:
                oppositeOperatorMap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}
                true_operator = oppositeOperatorMap[operator_name]
            else:
                true_operator = operator_name

            try:
                resource_str = str(rle._game.getAvatars()[0].resources[item])
            except IndexError:
                return 2 * mult * first_alpha, 10000

            if not eval(resource_str+true_operator+str(num)):
                return 2 * mult * first_alpha, 10000



        if compute_second_order:
            ## Get all positions of objects whose type is in killer_types; compute minimum distance
            ## of each to the stypes we have to destroy. Return min over all mins.
            # embed()
            s2_positions = self.WBP.findObjectsInRLE(rle, s2)
            s1_positions = self.WBP.findObjectsInRLE(rle, s1)

            # Second order lesion
            if s1 != 'avatar' and s2 != 'avatar':
                return 0, 10000

            n_sprites = len(s1_positions)
            try:
                # A consequence of the two-way generic interactions in the
                # theory is that minimum-distance object pairs whose interactions
                # were not yet observed will have their distance penalized twice
                # as much when none of those objects is an avatar. This implies
                # that non-avatar novel interactions will be favored over others
                possiblePairList = [manhattanDist(obj, pos)
                     for pos in s2_positions
                     for obj in s1_positions
                     if manhattanDist(obj, pos) != 0]
                distance = min(possiblePairList)
                     # This is a trick to avoid getting distance 0 for objects
                     # of same type. If the list turns out to be empty, it will
                     # raise an error and set the distance to 0
                # print distance
            except ValueError:
                # embed()
                distance = 0

            if possiblePairList:
                n_sprites = len(possiblePairList)
                # Normalize by number of sprites, enforcing a prior that encourages
                # goals that involve killing fewer objects
                val += (float(mult * second_alpha * distance)/n_sprites**2) + second_alpha * max(self.rle.outdim[0], self.rle.outdim[1])

        if n_sprites==0:
            return val, 10000
        return val, distance*n_sprites

    def timeout_val(self, theory, term, rle):
        val = 0
        limit = term.termination.limit

        # Check if condition is win or loss and multiply accordingly
        if term.termination.win:
            mult = -1
        else:
            mult = 1

        time_elapsed = rle._game.time
        distance_to_goal = abs(time_elapsed - limit)

        val += mult * distance_to_goal

        return val

    def heuristics(self, rle=None, first_alpha=1000., second_alpha=1,
                   time_alpha=10):
        if rle==None:
            rle = self.rle

        theory = self.WBP.theory
        heuristicVal = 0
        avatarNoveltyVals = []
        for term in theory.terminationSet:
            if isinstance(term, SpriteCounterRule):
                spritecounter_val = self.spritecounter_val(theory, term, term.termination.stype, rle,
                    first_alpha=5000, second_alpha=0)
                # if spritecounter_val!=0:
                    # print("spritecounter_val for {} is equal to {}".format(
                        # term.termination.stype, spritecounter_val))
                heuristicVal += spritecounter_val

            elif isinstance(term, MultiSpriteCounterRule):
                multispritecounter_val = self.multispritecounter_val(theory, term, rle,
                        first_alpha=500, second_alpha=0)  #500, 5 (normally)
                # if multispritecounter_val!=0:
                    # print("multispritecounter_val for {} is equal to {}".format(
                        # term.termination.stypes, multispritecounter_val))
                heuristicVal += multispritecounter_val

            elif isinstance(term, TimeoutRule):
                timeout_val = time_alpha * \
                    self.timeout_val(theory, term, rle)
                heuristicVal += timeout_val

            elif isinstance(term, NoveltyRule):
                noveltytermination_val, ranking = self.noveltytermination_val(
                    theory, term, term.termination.s1, term.termination.s2, rle,
                    first_alpha=2500, second_alpha=0)
                # if noveltytermination_val!=0:
                    # print("noveltytermination_val for {} and {} is equal to {}".format(
                        # term.termination.s1, term.termination.s2, noveltytermination_val))

                # if self.parent and self.parent.rle._game.score==0 and term.termination.args and term.termination.s1=='c6' and term.termination.s2=='avatar' and noveltytermination_val!=-5000:
                    # ipdb.set_trace()
                if 'avatar' == term.termination.s2:
                    avatarNoveltyVals.append([self.WBP.annealing*noveltytermination_val,
                        ranking])
                else:
                    heuristicVal += self.WBP.annealing * noveltytermination_val
                    # Exploit only
                    # heuristicVal += 0 * self.WBP.annealing * noveltytermination_val
                    # Explore only
                    # heuristicVal += 1000 * self.WBP.annealing * noveltytermination_val

        if avatarNoveltyVals:
            # print noveltyVals
            heuristicVal += min(avatarNoveltyVals, key= lambda x: x[1])[0]
        # print "sum:", heuristicVal

        return heuristicVal

    def position_score(self, factor=0):
        try:
            (x, y) = np.array((self.rle._game.getAvatars()[0].rect.x,
                self.rle._game.getAvatars()[0].rect.y))/self.WBP.pixel_size
            # print factor * self.WBP.visited_positions[x, y]
            return factor * self.WBP.visited_positions[x, y]
        except IndexError:
            print "index error in position score"
            return 0

    """
    def getTo_profiler(self):
        lp = LineProfiler()
        lp_wrapper = lp(self.getToCurrentState)
        output = lp_wrapper()
        lp.print_stats()
        return output
    """

    def getToCurrentState(self):
        if self.parent and self.parent.rle is not None:
            ## try to copy parent lastState. Then take action and store as current lastState.
            ## if that fails, replay from beginning and store as current lastState
            try:
                vrle = cPickle.loads(cPickle.dumps(self.parent.rle, -1))
                # vrle = copy.deepcopy(self.parent.rle)
                if len(self.actionSeq)>0:
                    a = self.actionSeq[-1]
                    # print a
                    res = vrle.step(a)
                    relevantEvents = [t for t in res['effectList'] if t[0] == 'changeResource']
                    self.metabolic_cost = self.parent.metabolic_cost + self.metabolics(vrle, res['effectList'], a)
                    self.terminal, self.win = vrle._isDone()
            except:
                print "conditions met but copy failed"
                embed()
        else:
            self.reconstructed=True
            # print "copy failed; replaying from top"
            vrle = cPickle.loads(cPickle.dumps(self.rle, -1))
            # vrle = copy.deepcopy(self.rle)
            self.terminal, self.win = vrle._isDone()
            i=0
            while not self.terminal and len(self.actionSeq)>i:
                a = self.actionSeq[i]
                res = vrle.step(a)
                self.metabolic_cost += self.metabolics(vrle, res['effectList'], a)
                self.terminal, self.win = vrle._isDone()
                i += 1
        return vrle, self.win

    """
    def eval_profiler(self):
        lp = LineProfiler()
        lp_wrapper = lp(self.eval)
        lp_wrapper()
        lp.print_stats()
    """

    def eval(self):
        # ## Evaluate current node, including calculating intrinsic reward: f(rewards, heuristics, etc.)


        self.rle, self.win = self.getToCurrentState()

        self.updateObjIDs(self.rle)

        self.state = self.WBP.calculateAtoms(self.rle)

        for i in range(1,3):
            for c in itertools.combinations(self.state, i):
                c = tuple(sorted(c))
                if self.WBP.trueAtoms[c] == 0:
                    self.candidates.append(c)
        self.updateNovelty()

        """
        try:
            avatar_pos = self.WBP.findAvatarInRLE(self.rle)
            vecValue = avatar_pos[1] + avatar_pos[0]*self.rle.outdim[0] + 1
            self.stateIW1 = [vecValue]
        except:
            vecValue = [0]

        self.stateIW1.append([pos for pos, char in enumerate(self.rle.show()) if char== ' '])
        self.updateNoveltyIW1()
        """

        ## Try rollouts for aliens?
        if self.WBP.allowRollouts and len(self.actionSeq)>0 and self.actionSeq[-1]==32:

            self.rolloutArray = self.rollout(self.rle)
            # print self.rolloutArray
            # print "in rollout"

        self.heuristicVal = self.heuristics()

        # print self.rle._game.score, self.heuristicVal, sum(self.rolloutArray), self.metabolic_cost, self.position_score()

        self.intrinsic_reward = self.rle._game.score + self.heuristicVal + \
        sum(self.rolloutArray) - self.metabolic_cost + self.position_score(-250)

        try:
            ## Planner should return a plan when the agent has reached the limit of any particular resource (because we now should be curious about new objects, which we're taking care of in main_agent)
            if any([self.rle._game.getAvatars()[0].resources[k]==self.WBP.theory.resource_limits[k] for k in self.rle._game.getAvatars()[0].resources.keys() if k not in self.WBP.seen_limits]):
                self.win=True
        except IndexError:
            pass

        return self.win

    def updateNovelty(self):
        if len(self.candidates)==0:
            self.novelty = 3
        else:
            self.novelty = min([len(c) for c in self.candidates])
        return self.novelty

    def updateNoveltyIW1(self):
        for state in self.WBP.trueAtomsIW1:
            if self.stateIW1 == state:
                self.novelty = 1
                return self.novelty
        self.novelty = 0
        return self.novelty

    def updateNoveltyDict(self, QNovelty, QReward):
        jointSet = list(set(QNovelty+QReward))
        for c in self.candidates:
            if self.WBP.trueAtoms[c] == 0:
                self.WBP.trueAtoms[c] = 1
                for n in jointSet:
                    if c in n.candidates:
                        n.candidates.remove(c)
        for n in jointSet:
            n.novelty = n.updateNovelty()
        return

    def updateNoveltyDictIW1(self, QNovelty, QReward):
        jointSet = list(set(QNovelty+QReward))
        for n in jointSet:
            n.novelty = n.updateNoveltyIW1()
        return

    def updateObjIDs(self, vrle):
        i = 0
        for objType in vrle._game.sprite_groups:
            for s in vrle._game.sprite_groups[objType]:
                if s.ID not in self.WBP.objIDs.keys():
                    if s.name=='bullet':
                        s.ID = len([o for o in vrle._game.sprite_groups[objType] if o not in vrle._game.kill_list])
                    else:
                        s.ID = len(vrle._game.sprite_groups[objType])
                    self.WBP.objIDs[s.ID] = (len(self.WBP.objIDs.keys())+1) * 100 * (self.rle.outdim[0]*self.rle.outdim[1]+self.WBP.padding)
                    i+=1
        return

    def isTerminal(self):
        return self.rle._isDone()[0]

    def isWin(self):
        return self.rle._isDone()[1]

    def playBack(self, make_movie=False):
        vrle = copy.deepcopy(self.rle)
        self.finalStatesEncountered = []
        terminal = vrle._isDone()[0]
        i=0
        if not make_movie:
            print vrle.show()
        while not terminal and i<len(self.actionSeq):
            a = self.actionSeq[i]
            vrle.step(a)
            if not make_movie:
                print actionDict[a]
                print vrle.show()
            else:
                self.finalStatesEncountered.append(vrle._game.getFullState())
            terminal = vrle._isDone()[0]
            i+=1


if __name__ == "__main__":


    ## Continuous physics games can't work right now. RLE is discretized, getSensors() relies on this, and a lot of the induction/planning
    ## architecture depends on that. Will take some work to do this well. Best plan is to shrink the grid squares and increase speeds/strengths of
    ## objects.
    # gameFilename = "examples.gridphysics.theorytest"
    # gameFilename = "examples.gridphysics.boulderdash"
    gameFilename = "examples.gridphysics.theorytest"
    # gameFilename = "examples.continuousphysics.breakout_big"

    gameString, levelString = defInputGame(gameFilename, randomize=True)
    rleCreateFunc = lambda: createRLInputGame(gameFilename)
    rle = rleCreateFunc()
    embed()
    p = WBP(rle, gameFilename)


    # embed()
    t1 = time.time()
    last, gameString_array = p.BFS()
    from core import VGDLParser
    # embed()
    last.playBack(make_movie=True)
    # VGDLParser.playGame(gameString, levelString, p.statesEncountered, persist_movie=True, make_images=True, make_movie=True, movie_dir="videos/"+gameFilename, padding=0)
    # VGDLParser.playGame(gameString, levelString, last.finalStatesEncountered, persist_movie=True, make_images=True, make_movie=True, movie_dir="videos/"+gameFilename, padding=0)


    print time.time()-t1
    # embed()


#
