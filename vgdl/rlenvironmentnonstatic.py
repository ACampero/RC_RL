'''
Created on 2014 4 02

@author: Dylan Banarse (dylski@google.com)

Wrappers for games to interface them with artificial players.
This interface is a generic one for interfacing with RL agents.
'''
import numpy as np
from numpy import zeros
import pygame
from ontology import BASEDIRS
from core import VGDLSprite
from stateobsnonstatic import StateObsHandlerNonStatic
from collections import defaultdict
import argparse
from IPython import embed
import random
import math
import importlib
from colors import *
from util import factorize, objectsToSymbol
from pygame.locals import K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT
from termcolor import colored
import time
import cPickle
# from line_profiler import LineProfiler

OBSERVATION_LOCAL = 'local'
OBSERVATION_GLOBAL = 'global'

class RLEnvironmentNonStatic( StateObsHandlerNonStatic):
    """ Wrapping a VGDL game with a generic interface suitable for reinforcement learning.
    """

    name = "VGDL-RLEnvironment"
    description = "RLEnvironment interface to VGDL."

    # If the visualization is enabled, all actions will be reflected on the screen.
    visualize = False
    # visualize = True
    # In that case, optionally wait a few milliseconds between actions?
    actionDelay = 0

    # Recording events (in slightly redundant format state-action-nextstate)
    recordingEnabled = False

    def __init__(self, gameDef, levelDef, observationType=OBSERVATION_GLOBAL, visualize=False, actionset=BASEDIRS, **kwargs):
        game = _createVGDLGame( gameDef, levelDef )
        StateObsHandlerNonStatic.__init__(self, game, **kwargs)
        self._actionset = actionset
        self.visualize = visualize
        self._initstate = self.getState()
        #
        # Total output dimensions are:
        #   #object_types * ( #neighbours + center )
        #
        # Note that _obstypes is an array of arrays for object types and their positions, e.g.
        # {
        #  'wall': [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
        #  'goal': [(4, 1)]
        # }
        self.observationType=observationType
        if observationType == OBSERVATION_LOCAL:
            # Array of grid indices around the agent
            ns = self._stateNeighbors(self._initstate)
            self.outdim = [(len(ns) + 1) * len(self._obstypes), 1]
        else:
            self.nsAllCells = []
            self.outdim = [game.height, game.width]
            for y in range(0, game.height):
                for x in range(0, game.width):
                    self.nsAllCells.append( (x, y) )
        self._postInitReset()
        self._game.reset()
        self._game.all_objects = self._game.getObjects() # Save all objects, some which may be killed in game
        self._game.exceptedObjects = []
        self.makeSymbolDict()
        self._game.ignoreList = [] ## another way to mark objects that shouldn't be processed when doing induction (that is, collision objects)
        self._game.keystate = defaultdict(bool)
        self._game.metabolic_score = 0
        self.game_name = None

    # Get definition of the observation data expected
    def observationSpec(self):
        return{ 'scheme':'Doubles', 'size':self.outdim }

    def getObjectsFromNumber(self, n):
        indices = factorize(self, n)
        allItems = ['avatar']+sorted(self._obstypes.keys())[::-1]
        return [allItems[i] for i in indices]


    def makeSymbolDict(self):
        inverseMapping = dict()
        colorMapping = dict()
        numbers = '0123456789'
        alnum = numbers + 'abcdefghijklmnopqrstuvwxyz'
        idx = 0
        OLD_GOAL = "oldGl"
        # embed()
        for s in self._obstypes.keys():
            # colorMapping[s] = colorDict[str(self._game.sprite_constr[s][1]['color'])].lower()
            if not s == "goal":
                inverseMapping[s] = alnum[idx]
                idx+=1
            elif s=="goal":
                inverseMapping["goal"] = "G"
            else:
                inverseMapping[OLD_GOAL] = "O" # old goal

        inverseMapping['avatar'] = 'A'
        # if "goal" in self._obstypes:
        #     inverseMapping["goal"] = "G"

        self.symbolDict = inverseMapping
        # self.colorMapping = colorMapping
        return

    def show_binary(self, thingWeShoot):
        """
        symbolDict = a dict mapping each sprite name to its symbol.
        If there's no sprite overlap, then returns a string. Else returns numpy array.
        """        
        mappedState = np.ones((self.outdim[1]*self.outdim[0]))
        kl_set = set(self._game.kill_list)
        for k, lst in self._game.sprite_groups.items():
            if k!= thingWeShoot:
                for sprite in lst:
                    if sprite not in kl_set:
                        y,x = sprite.rect.top/30, sprite.rect.left/30
                        try:
                            mappedState[x+self.outdim[1]*y] = 0
                        except:
                            pass
        return mappedState

    def show(self, indent=False, showArrays=False, binary=False, color='grey'):
        """
        symbolDict = a dict mapping each sprite name to its symbol.
        If there's no sprite overlap, then returns a string. Else returns numpy array.
        """
        ## faster version, but need to figure out how to display 

        locs = defaultdict(lambda:[])
        if binary:
            mappedState = [[1 for x in range(self.outdim[1])] for y in range(self.outdim[0])]
        else:
            mappedState = [[' ' for x in range(self.outdim[1])] for y in range(self.outdim[0])]

        for lst in self._game.sprite_groups.values():
            for sprite in lst:
                if sprite not in self._game.kill_list:
                    y,x = sprite.rect.top/30, sprite.rect.left/30
                    locs[(y,x)].append(sprite.name)

        for k,v in locs.iteritems():
            if binary:
                symbol = 0
            else:
                if len(v)>1:
                    if 'avatar' in v:
                        symbol = 'X'
                    else:
                        symbol = '$'
                else:
                    if 'avatar' in v:
                        symbol = 'A'
                    else:
                        symbol = objectsToSymbol(self, v, self.symbolDict)
                
                if symbol in ['A', 'X']:
                    symbol = colored(symbol, 'red')
                else:
                    if color != 'grey':
                        symbol = colored(symbol, color)

            try:
                mappedState[k[0]][k[1]] = symbol
            except:
                # print "mappedState problem in rlenvironmentNonStatic"
                ## if you define rules poorly, objects can go off screen, in which case they can't be assigned to an on-screen loc!
                continue

        if binary:
            gameString = []
            for mappedRow in mappedState:
                gameString.extend(mappedRow)
        else:
            gameString = ""
            for mappedRow in mappedState:
                gameString += reduce(lambda a,b: a+b, mappedRow) + "\n"
        return gameString

    # Get definition of the actions that are accepted
    def actionSpec(self):
        return{ 'scheme':'Integer', 'N':4 }

    # Reset the game between episodes.
    # Currently it is not recommended that this is called hundreds of times
    # cause things start to slow down exponentially (being looked at). The
    # recommended process is to re-create this class for each episode
    # (i.e. call the constructor for this class each episode) and call softReset
    # to get the starting observations.
    def reset(self):
        self._postInitReset(True)
        return self.step(None)

    # Reset after constructor
    # Like reset() but does not re-initialise state. This can be called after the
    # class has been constructed to get the starting observations
    def softReset(self):
        self._postInitReset(False)
        return self.step(None)

    # Reset game data and optionally the state
    def _postInitReset(self, performStateResetTesting=False):
        if self.visualize:
            self._game._initScreen(self._game.screensize, not self.visualize)

        # Calling self.setState(self._initstate) hundreds of times causes massive slowdown.
        if performStateResetTesting:
            self.setState(self._initstate)

        # if no avatar starting location is specified, the default one will be to place it randomly
        self._game.randomizeAvatar()

        self._game.kill_list = []
        if self.visualize:
            pygame.display.flip()
        if self.recordingEnabled:
            self._last_state = self.getState()
            self._allEvents = []

    def close():
        pass

    def _isDone(self, getTermination=False):
        # remember reward if the final state ends the game
        # self._game.terminations.sort(key=lambda x: 0 if (x.name=='SpriteCounter' and x.stype=='avatar' and x.win==  False) else 1 if x.name=='SpriteCounter' else 2)
        self._game.terminations.sort(key=lambda x: 0 if (x.name=='NoveltyTermination') else 1 if (x.name=='SpriteCounter' and x.stype=='avatar' and x.win==  False) else 2 if x.name=='SpriteCounter' else 3)
        for t in self._game.terminations:
            # import pdb; pdb.set_trace()
            # Convention: the first criterion is for keyboard-interrupt termination
            # Breaking convention here
            ended, win = t.isDone(self._game)
            if ended:
                # if t.name=='NoveltyTermination':
                #     print t.s1, t.s2
                # elif t.name=='SpriteCounter':
                #     print t.stype
                # elif t.name=='MultiSpriteCounter':
                #     print t.stypes
                if getTermination:
                    return ended, win, t
                else:
                    return ended, win
        if getTermination:
            return False, False, None
        else:
            return False, False

    """
    def sensors_profiler(self, state=None):
        lp = LineProfiler()
        lp_wrapper = lp(self._getSensors)
        output = lp_wrapper(state)
        lp.print_stats()

        return output
    """

    def _getSensors(self, state=None):
        # Get position and orientation
        if state is None:
            # state = { x, y, (rot?) }
            state = self.getState()
        if self.orientedAvatar:
            pos = (state[0], state[1])
        else:
            pos = state

        res = zeros(self.outdim[0]*self.outdim[1])
        # Get sensor data given current state (i.e. position)
        # and whether local state or whole game state is required
        if self.observationType == OBSERVATION_LOCAL:
            # A 1D array of ints, each representing presence or absense of
            # object type at local positions (e.g. Centre, Top, Left, Down,
            # Right) around avatar. First set of ints represent the first
            # object type in _obstypes, the next len(BASEDIRS) ints are for
            # the next object type, etc.
            # e.g. where object type A is present left and below,
            # and object type B is not visible, the observation would be:
            # 00110 00000
            ns = [pos] + self._stateNeighbors(state)
            for i, n in enumerate(ns):
                os = self._rawSensor(n)
                # slice os (e.g. [True,False]) into 'res' at position i and i+len(ns)
                # where len(ns) is number of sensor areas per sensor
                #print("i="+str(i)+" res="+str(res)+" res[..]="+str(res[i::len(ns)]))
                res[i::len(ns)] = os
        else:
            # OBSERVATION_GLOBAL
            # Returns 2D array of ints where bits set represent object types
            # present at that position. Bit 1 = Avatar. The other bits are set
            # in order that they are set in _obstypes (stateobs.py)
            # e.g. for avatar (1) in walled area (2) with goal at top right (4)
            # 222222
            # 200042
            # 200002
            # 210002
            # 222222
            ns = self.nsAllCells
            for i, n in enumerate(ns):
                # check if the avatar is here
                if n[0]==pos[0] and n[1]==pos[1]:
                    res[i] = 1
                os = self._rawSensor(n)
                for s in range(0,len(os)):
                    if os[s]==True:
                        res[i] = int(res[i]) | (2<<s)
        return res

    def _performAction(self, action=[], onlyavatar=False):

        """ Action is an index for the actionset.  """
        # take action and compute consequences
        # replace the method that reads multiple action keys with a fn that just
        # returns the currently desired action
        # if action == (0,0) or action == None:
        #     return

        # if action != (0,0) and self._avatar:
        #     self._avatar._readMultiActions = lambda *x: [action]

        # self._avatar._readMultiActions = lambda *x: [self._actionset[action]] # old
        possible_actions = [K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT]


        if action in possible_actions:
            self._game.keystate[action] = True


        if self.visualize:
            self._game._clearAll(self.visualize)

        self._game.new_sprites = []
        # update sprites
        if onlyavatar:
            if action != 0:
                self._avatar.update(self._game)

        else:
            for s in self._game:
                if action == 0 and s == self._avatar:
                        continue
                if s not in self._game.kill_list:
                        s.update(self._game)

        events = self._game._eventHandling()

        ## get events (e.g., (stepBack obj1ID, obj2ID))

        # self._gravepoints[(skey, self._rect2pos(s.rect))] = True


        ## Added 5/2, to correct for the fact that some gmaes don't have _gravepoints by default
        if not hasattr(self, '_gravepoints'):
            self._gravepoints = {}

        # ### BEGINNING OF CHANGES
        for skey in self._other_types:
            ss = self._game.sprite_groups[skey]
            self._obstypes[skey] = [self._sprite2state(sprite, oriented=False)
                                        for sprite in ss]

        ## Added 4/31
        ## Logic (I think) was to make sure everything that could exist was in gravepoints because
        ## getState (defined in stateobsnonstatic) uses it to populate getState, getSensors, etc.
        for k in self._game.sprite_groups:
            for sprite in self._game.sprite_groups[k]:
                if (k, self._rect2pos(sprite.rect)) not in self._gravepoints.keys():
                    self._gravepoints[(k, self._rect2pos(sprite.rect))] = True
        # print "after adding gravepoints"
        # embed()
        return events

        # if self.visualize:
        #     self._game._clearAll(self.visualize)

        # # update screen
        # if self.visualize:
        #     self._game._drawAll()
        #     pygame.display.update(VGDLSprite.dirtyrects)
        #     VGDLSprite.dirtyrects = []
        #     pygame.time.wait(self.actionDelay)

        # if self.recordingEnabled:
        #     self._previous_state = self._last_state
        #     self._last_state = self.getState()
        #     self._allEvents.append((self._previous_state, action, self._last_state))

    """
    def step_profiler(self, action):
        lp = LineProfiler()
        lp_wrapper = lp(self.step)
        output = lp_wrapper(action)
        lp.print_stats()

        return output
    """

    def step(self, action, return_obs=False, getTermination=False, getEffectList=False):
        if action == ('space'):
            self._game.keystate[32] = True
            action = (0,0)
        pre_step_score = self._game.score
        # t1 = time.time()
        events = self._performAction(action)
        # embed()
        # observation = self._getSensors()

        self._game.time+=1
        observation = self._getSensors() if return_obs else None
        if getTermination:
            (ended, won, termination) = self._isDone(getTermination=True)
        else:
            (ended, won) = self._isDone()
            termination = []
        
        self._game.ended, self._game.win = ended, won

        dScore = self._game.score - pre_step_score
        if ended:
            pcontinue = 0
            if won:
                reward = 1
            else:
                reward = -1
        else:
            pcontinue = 1
            ## this is where you need to give the reward for doing non-terminal actions, and then your agent can process this.
            reward = dScore
        for k in self._game.keystate:
            self._game.keystate[k] = False

        self._game.positionDict = dict()
        for k,v in self._game.sprite_groups.items():
            for sprite in v:
                if sprite not in self._game.kill_list:
                    loc = (sprite.rect.left, sprite.rect.top)
                    if loc in self._game.positionDict.keys():
                        self._game.positionDict[loc].append(sprite)
                    else:
                        self._game.positionDict[loc] = [sprite]
        # print time.time()-t1
        # if getEffectList:
            # return{'observation':observation, 'reward':reward, 'pcontinue':pcontinue, 'effectList':events}
        # else:
            # return {}
        return{'observation':observation, 'reward':reward, 'pcontinue':pcontinue, 'effectList':events, 'ended':ended, 'win':won, 'termination':termination}

## the game in the agent's 'head'
def defTheoryTest():
    from examples.gridphysics.theorytest import game, level
    print level[0]
    print game[0]
    return (game, level)

def defVirtualGame():
    from examples.gridphysics.virtualGame import game, level
    return (game, level)

def defVirtualGame2():
    from examples.gridphysics.virtualGame2 import game, level
    return (game, level)

def defMaze():
    from examples.gridphysics.mazes import maze_game, maze_level_1
    return( maze_game, maze_level_1 )

def defSimpleGame1():
    from examples.gridphysics.simpleGame1 import game, level
    return (game, level)

def defSimpleGame3():
    from examples.gridphysics.simpleGame3 import push_game, box_level
    return (push_game, box_level)

def defSimpleGame4(r=False):
    if r:
        from examples.gridphysics.simpleGame4 import game, level1, level2, level3
        level = random.choice([level1, level2, level3])
    else:
        from examples.gridphysics.simpleGame4 import game, level
    return (game, level)

def defSimpleGame5():
    from examples.gridphysics.simpleGame5 import game, level
    return (game, level)

def defSimpleGame_missile():
    from examples.gridphysics.simpleGame_missile import game, level
    return (game, level)

def defspriteInduction4():
    from examples.gridphysics.spriteInduction4 import game, level
    return (game, level)

def deftextTheory():
    from examples.gridphysics.textTheory import game, level
    return (game, level)

def defFrogs():
    from examples.gridphysics.frogs import frog_game, frog_level
    return (frog_game, frog_level)

def defAliens():
    from examples.gridphysics.aliens import aliens_game, aliens_level
    return (aliens_game, aliens_level)

def try_int(s):
    "Convert to integer if possible."
    try: return int(s)
    except: return s

def natsort_key(s):
    "Used internally to get a tuple by which s is sorted."
    import re
    return map(try_int, re.findall(r'(\d+|\D+)', s))

def natcmp(a, b):
    "Natural string comparison, case sensitive."
    return cmp(natsort_key(a), natsort_key(b))

def natcasecmp(a, b):
    "Natural string comparison, ignores case."
    return natcmp(a.lower(), b.lower())

def defInputGame(filename, randomize=False, index=None):
    game_file = importlib.import_module(filename)
    levels = [k for k in game_file.__dict__.keys() if 'level' in k]
    levels.sort(natcasecmp)
    # print levels
    if randomize:
        level = random.choice(levels)
        return (game_file.game, game_file.__dict__[level])
    elif index>=0:
        if index<len(levels):
            print index
            level = levels[index]
        else:
            level = random.choice(levels)
        return (game_file.game, game_file.__dict__[level])
    else:
        return (game_file.game, game_file.level)

def _createVGDLGame( gameSpec, levelSpec ):
    import uuid
    from vgdl.core import VGDLParser
    # parse, run and play.
    # import pdb; pdb.set_trace()
    game = VGDLParser().parseGame(gameSpec)
    game.buildLevel(levelSpec)
    game.uiud = uuid.uuid4()
    return game

def playTestMaze():
    game = _createVGDLGame( *defMaze() )
    headless = False
    persist_movie = False
    game.startGame(headless,persist_movie)

def playTestSimpleGame1():
    game = _createVGDLGame( *defSimpleGame1() )
    headless = False
    persist_movie = False
    game.startGame(headless,persist_movie)

def playTestSimpleGame3():
    game = _createVGDLGame( *defSimpleGame3() )
    headless = False
    persist_movie = False
    game.startGame(headless,persist_movie)

def playTestFrogs():
    game = _createVGDLGame( *defFrogs() )
    headless = False
    persist_movie = False
    game.startGame(headless,persist_movie)

def playTestAliens():
    game = _createVGDLGame( *defAliens() )
    headless = False
    persist_movie = False
    game.startGame(headless,persist_movie)

# Test some of the observation and action specs
def testSpecs():
    game = _createVGDLGame( *defMaze() )
    rle = RLEnvironmentNonStatic( *defMaze() )
    if rle.actionSpec() != {'scheme': 'Integer', 'N': 4}:
        print "FAILED actionSpec"
        print rle.actionSpec()
    if rle.observationSpec() != {'scheme': 'Doubles', 'size': [10, 1]}:
        print "FAILED observationSpec"
        print rle.observationSpec()

# Verify that observation received matches target observation
def _verify( obs, targetObs ):
    if obs["pcontinue"] != targetObs["pcontinue"]:
        print "FAILED pcontinue"
        return False
    if obs["reward"] != targetObs["reward"]:
        print "FAILED reward"
        return False
    match = True
    i=0
    for ob in targetObs["observation"]:
        if float(obs["observation"][i]) != float(targetObs["observation"][i]):
            match = False
        i = i+1

    if match==False:
        print ""
        print "FAILED observation"
        print obs["observation"]
        print targetObs["observation"]
        print match
        return False
    return True

##TODO: Add these functions here to make a new game.
## That is, Make the def createRLSimpleGame1...
##          and define defSimpleGame1...
## Star in these args unzips the tuple.
# simple maze test, moved to goal and win


def createMindEnv(game, level, output=False, obsType=OBSERVATION_GLOBAL ):
    if output:
        print game
        print level
    return RLEnvironmentNonStatic( game, level, observationType=obsType )

def createRLVirtualGame( obsType=OBSERVATION_GLOBAL ):
    return RLEnvironmentNonStatic( *defVirtualGame(), observationType=obsType )

def createRLVirtualGame2( obsType=OBSERVATION_GLOBAL ):
    return RLEnvironmentNonStatic( *defVirtualGame2(), observationType=obsType )

def createRLMaze( obsType=OBSERVATION_LOCAL ):
    return RLEnvironmentNonStatic( *defMaze(), observationType=obsType )

def createRLSimpleGame1( obsType=OBSERVATION_GLOBAL ):
    return RLEnvironmentNonStatic( *defSimpleGame1(), observationType=obsType )

def createRLSimpleGame3( obsType=OBSERVATION_GLOBAL ):
    return RLEnvironmentNonStatic( *defSimpleGame3(), observationType=obsType )

def createRLSimpleGame4( obsType=OBSERVATION_GLOBAL ):
    return RLEnvironmentNonStatic( *defSimpleGame4(r=False), observationType=obsType )

def createRLSimpleGame4_random( obsType=OBSERVATION_GLOBAL ):
    return RLEnvironmentNonStatic( *defSimpleGame4(r=True), observationType=obsType )

def createRLSimpleGame5( obsType=OBSERVATION_GLOBAL ):
    return RLEnvironmentNonStatic( *defSimpleGame5(), observationType=obsType )

def createRLSimpleGame_missile( obsType=OBSERVATION_GLOBAL ):
    return RLEnvironmentNonStatic( *defSimpleGame_missile(), observationType=obsType )

def createRLspriteInduction4(obsType = OBSERVATION_GLOBAL):
    return RLEnvironmentNonStatic( *defspriteInduction4(), observationType=obsType)

def createRLtextTheory(obsType = OBSERVATION_GLOBAL):
    return RLEnvironmentNonStatic( *deftextTheory(), observationType=obsType)

def createRLFrogs( obsType=OBSERVATION_LOCAL ):
    return RLEnvironmentNonStatic( *defFrogs(), observationType=obsType )

def createRLAliens( obsType=OBSERVATION_LOCAL ):
    return RLEnvironmentNonStatic( *defAliens(), observationType=obsType )

def createRLInputGame(filename, obsType=OBSERVATION_GLOBAL):
    game_file = importlib.import_module(filename)
    return RLEnvironmentNonStatic(game_file.game, game_file.level, \
            observationType = obsType)


def createRLInputGameFromStrings(game, level):
    return RLEnvironmentNonStatic(game, level, \
            observationType = OBSERVATION_GLOBAL)


def testMaze(numEpisodes, numJogOnSpot, verify, reuseGame, obsType):
    rle = createRLMaze( obsType )

    # uncomment following two lines to see the walk (causes internal warning)
    #rle.visualize = True
    for i in range(0,numEpisodes):

        if reuseGame:
            # Purely for testing: reuse the game and by calling _postInitReset(True).
            # This should be faster but self.setState(_initstate) in _postInitReset()
            # causes the game to slow down with hunreds of calls.
             rle._postInitReset(True)
        else:
            # Re-create the game.
            rle = createRLMaze( obsType )

        res = rle.step(0) #up
        if verify:
            _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]} )
        res = rle.step(1) #left (there's a wall so expect no change in observations)
        if verify:
            _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]} )
        res = rle.step(3) #right
        if verify:
            _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]} )

        # Hop backwards and forwards
        for j in range (0,int(numJogOnSpot)):
            res = rle.step(1) #left
            if verify:
                _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]} )
            res = rle.step(3) #right
            if verify:
                _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]} )

        res = rle.step(3) #right
        if verify:
            _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]} )
        res = rle.step(3) #right
        if verify:
            _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.]} )
        res = rle.step(0) #up
        if verify:
            _verify( res, {'pcontinue': 0, 'reward': 1, 'observation': [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]} )

def testSimpleGame1(numEpisodes, numJogOnSpot, verify, reuseGame, obsType):
    rle = createRLSimpleGame1 ( obsType )

    # uncomment following two lines to see the walk (causes internal warning)
    #rle.visualize = True
    for i in range(0,numEpisodes):

        if reuseGame:
            # Purely for testing: reuse the game and by calling _postInitReset(True).
            # This should be faster but self.setState(_initstate) in _postInitReset()
            # causes the game to slow down with hunreds of calls.
             rle._postInitReset(True)
        else:
            # Re-create the game.
            rle = createRLSimpleGame1( obsType )

        # rle = createRLSimpleGame1( obsType )
        print "in testSimpleGame1"
        embed()



def testFrogs(numEpisodes, numJogOnSpot, verify, reuseGame, obsType):
    rle = createRLFrogs ( obsType )

    # uncomment following two lines to see the walk (causes internal warning)
    #rle.visualize = True
    for i in range(0,numEpisodes):

        if reuseGame:
            # Purely for testing: reuse the game and by calling _postInitReset(True).
            # This should be faster but self.setState(_initstate) in _postInitReset()
            # causes the game to slow down with hunreds of calls.
             rle._postInitReset(True)
        else:
            # Re-create the game.
            rle = createRLFrogs( obsType )

        # rle = createRLSimpleGame1( obsType )
        print "in testFrogs"
        embed()

def testSimpleGame_missile(numEpisodes, numJogOnSpot, verify, reuseGame, obsType):
    rle = createRLSimpleGame_missile( obsType )

    # uncomment following two lines to see the walk (causes internal warning)
    #rle.visualize = True
    for i in range(0,numEpisodes):

        if reuseGame:
            # Purely for testing: reuse the game and by calling _postInitReset(True).
            # This should be faster but self.setState(_initstate) in _postInitReset()
            # causes the game to slow down with hunreds of calls.
             rle._postInitReset(True)
        else:
            # Re-create the game.
            rle = createRLSimpleGame1( obsType )

        # rle = createRLSimpleGame1( obsType )
        print "in testSimpleGame1"
        embed()

def testAliens(numEpisodes, numJogOnSpot, verify, reuseGame, obsType):
    rle = createRLAliens ( obsType )

    # uncomment following two lines to see the walk (causes internal warning)
    #rle.visualize = True
    for i in range(0,numEpisodes):

        if reuseGame:
            # Purely for testing: reuse the game and by calling _postInitReset(True).
            # This should be faster but self.setState(_initstate) in _postInitReset()
            # causes the game to slow down with hunreds of calls.
             rle._postInitReset(True)
        else:
            # Re-create the game.
            rle = createRLAliens( obsType )

        print "in testAliens"
        embed()

        # res = rle.step(0) #up
        # if verify:
        #     _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]} )
        # res = rle.step(1) #left (there's a wall so expect no change in observations)
        # if verify:
        #     _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]} )
        # res = rle.step(3) #right
        # if verify:
        #     _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]} )

        # # Hop backwards and forwards
        # for j in range (0,int(numJogOnSpot)):
        #     res = rle.step(1) #left
        #     if verify:
        #         _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]} )
        #     res = rle.step(3) #right
        #     if verify:
        #         _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]} )

        # res = rle.step(3) #right
        # if verify:
        #     _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]} )
        # res = rle.step(3) #right
        # if verify:
        #     _verify( res, {'pcontinue': 1, 'reward': 0, 'observation': [ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.]} )
        # res = rle.step(0) #up
        # if verify:
        #     _verify( res, {'pcontinue': 0, 'reward': 1, 'observation': [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]} )


def defaultTest():
    print("testSpecs()")
    testSpecs()
    print("testMaze(1, 0, True, True, OBSERVATION_LOCAL)")
    testMaze(1, 0, True, True, OBSERVATION_LOCAL)
    print("testMaze(1, 0, True, False, OBSERVATION_LOCAL)")
    testMaze(1, 0, True, False, OBSERVATION_LOCAL)
    print("testMaze(2, 0, True, False, OBSERVATION_LOCAL)")
    testMaze(2, 0, True, False, OBSERVATION_LOCAL)
    print("testMaze(1, 2, True, False, OBSERVATION_LOCAL)")
    testMaze(1, 2, True, False, OBSERVATION_LOCAL)

if __name__ == "__main__":
    # playTestSimpleGame1()
    parser = argparse.ArgumentParser()
    parser.add_argument("--numEpisodes", default=1, help="Number of episodes to run",
                    type=int)
    parser.add_argument("--profile", help="profile a set of episode runs",
                    action="store_true")
    parser.add_argument("--reuse-game", help="EXPERIMENTAL: don't re-create game each episode, results in slow-down bug",
                    action="store_true")
    parser.add_argument("--jog-on-spot", type=int, default=0, help="half the extra number of steps to add")
    parser.add_argument("--test", help="run tests",
                    action="store_true")

    parser.add_argument("--observation-type", help="'local' for neighbors or 'global' for whole game area", default='local')
    parser.add_argument("--play-test", help="Interactively play the test maze", default=False, action='store_true')
    args = parser.parse_args()

    if args.profile:
        import cProfile
        command = 'testMaze('+str(args.numEpisodes)+','+str(args.jog_on_spot)+',False,'+str(args.reuse_game)+',"'+args.observation_type+'")'
        cProfile.run(command)
    elif args.play_test:
        playTestMaze()
    else:
        # defaultTest()
        # testMaze(args.numEpisodes, args.jog_on_spot, True, args.reuse_game, args.observation_type)
        # testMaze(1, 0, True, True, OBSERVATION_GLOBAL)
        testSimpleGame_missile(1, 0, True, True, OBSERVATION_GLOBAL)
        # testSimpleGame1(1, 0, True, True, OBSERVATION_GLOBAL) # to uncomment
        # testFrogs(1, 0, True, True, OBSERVATION_GLOBAL)
        # testAliens(1, 0, True, True, OBSERVATION_GLOBAL)
