'''
Video game description language -- parser, framework and core game classes.

@author: Tom Schaul
'''
import pygame
from random import choice
from tools import Node, indentTreeParser
from collections import defaultdict
from tools import roundedPoints
from colors import *
import os, shutil
import datetime
import uuid
import subprocess
import glob
import ipdb
from copy import deepcopy
import logging
import numpy as np
import sys
import re
from IPython import embed
import time
import os
import uuid

# ---------------------------------------------------------------------
#     Constants
# ---------------------------------------------------------------------


disableContinuousKeyPress = True
actionToKeyPress = {(-1,0): pygame.K_LEFT, (1,0): pygame.K_RIGHT,
                    (0,1): pygame.K_DOWN, (0,-1): pygame.K_UP}

keyPresses = {273: 'up', 274: 'down', 276: 'left', 275: 'right', 32: 'spacebar', 0:'none'}
emptyKeyState = tuple([0]*323) #keyState when no keys are pressed



class VGDLParser(object):
    """ Parses a string into a Game object. """
    verbose = False


    @staticmethod
    def playGame(game_str, map_str, playback_states = None, headless = False, persist_movie = False, make_images=False, make_movie=False, movie_dir = "./tmpl", padding=0):
        """ Parses the game and level map strings, and starts the game. """
        g = VGDLParser().parseGame(game_str)

        g.buildLevel(map_str)
        g.uiud = uuid.uuid4()
        if playback_states:
            g.playback_states = playback_states
        if(headless):
            g.startGameExternalPlayer(headless, persist_movie, movie_dir)
            #g.startGame(headless,persist_movie)
        else:
            if playback_states:
                g.startPlaybackGame(headless, persist_movie, make_images, make_movie, movie_dir, padding)
            else:
                g.startGame(headless, persist_movie)

        return g


    @staticmethod
    def playSubjectiveGame(game_str, map_str):
        from pybrain.rl.experiments.episodic import EpisodicExperiment
        from interfaces import GameTask
        from subjective import SubjectiveGame
        from agents import InteractiveAgent, UserTiredException
        g = VGDLParser().parseGame(game_str)
        g.buildLevel(map_str)
        senv = SubjectiveGame(g, actionDelay=100, recordingEnabled=True)
        task = GameTask(senv)
        iagent = InteractiveAgent()
        exper = EpisodicExperiment(task, iagent)
        try:
            exper.doEpisodes(1)
        except UserTiredException:
            pass

    def parseGame(self, tree):
        """ Accepts either a string, or a tree. """
        if not isinstance(tree, Node):
            tree = indentTreeParser(tree).children[0]
        sclass, args = self._parseArgs(tree.content)
        self.game = sclass(**args)
        for c in tree.children:
            if c.content == "SpriteSet":
                self.parseSprites(c.children)
            if c.content == "InteractionSet":
                self.parseInteractions(c.children)
            if c.content == "LevelMapping":
                self.parseMappings(c.children)
            if c.content == "TerminationSet":
                self.parseTerminations(c.children)
            if c.content == "ConditionalSet":
                self.parseConditions(c.children)
        return self.game

    def _eval(self, estr):
        """ Whatever is visible in the global namespace (after importing the ontologies)
        can be used in the VGDL, and is evaluated.
        """
        from ontology import * # @UnusedWildImport
        return eval(estr)

    def parseInteractions(self, inodes):
        for inode in inodes:
            if ">" in inode.content:
                pair, edef = [x.strip() for x in inode.content.split(">")]
                eclass, args = self._parseArgs(edef)
                class1, class2 = [x.strip() for x in pair.split(" ") if len(x)>0]
                self.game.collision_eff.append(tuple([class1, class2, eclass, args]))
                if self.verbose:
                    print "Collision", pair, "has effect:", edef
        #print self.game.collision_eff

    def parseTerminations(self, tnodes):
        # if any(['Multi' in tnode.content for tnode in tnodes]):
            # print("found MultiSpriteCounter in parseTerminations")
        # ipdb.set_trace()
        for tn in tnodes:
            sclass, args = self._parseArgs(tn.content)
            if self.verbose:
                print "Adding:", sclass, args
            self.game.terminations.append(sclass(**args))

    def parseConditions(self, cnodes):
        for cnode in cnodes:
            if ">" in cnode.content:
                conditional, interaction = [x.strip() for x in cnode.content.split(">")]
                cclass, cargs = self._parseArgs(conditional)
                eclass, eargs = self._parseArgs(interaction)
                self.game.conditions.append([cclass(**cargs), [eclass, eargs]])

    def parseSprites(self, snodes, parentclass=None, parentargs={}, parenttypes=[]):
        for sn in snodes:
            assert ">" in sn.content
            key, sdef = [x.strip() for x in sn.content.split(">")]
            sclass, args = self._parseArgs(sdef, parentclass, parentargs.copy())
            stypes = parenttypes+[key]
            if 'singleton' in args:
                if args['singleton']==True:
                    self.game.singletons.append(key)
                args = args.copy()
                del args['singleton']

            if len(sn.children) == 0:
                if self.verbose:
                    print "Defining:", key, sclass, args, stypes
                self.game.sprite_constr[key] = (sclass, args, stypes)
                # self.game.sprite_groups[key] = [] ##Added 4/30
                if key in self.game.sprite_order:
                    # last one counts
                    self.game.sprite_order.remove(key)
                self.game.sprite_order.append(key)
            else:
                self.parseSprites(sn.children, sclass, args, stypes)

    def parseMappings(self, mnodes):
        for mn in mnodes:
            c, val = [x.strip() for x in mn.content.split(">")]
            assert len(c) == 1, "Only single character mappings allowed."
            # a char can map to multiple sprites
            keys = [x.strip() for x in val.split(" ") if len(x)>0]
            if self.verbose:
                print "Mapping", c, keys
            self.game.char_mapping[c] = keys

    def _parseArgs(self, s,  sclass=None, args=None):
        if not args:
            args = {}
        sparts = [x.strip() for x in s.split(" ") if len(x) > 0]
        if len(sparts) == 0:
            return sclass, args
        if not '=' in sparts[0]:
            sclass = self._eval(sparts[0])
            sparts = sparts[1:]
        # if any(['args' in sp for sp in sparts]):
        #     extraArgs = sparts[['args' in sp for sp in sparts].index(True)]
        for sp in sparts:
            ## this is failing once you've written a theory with args
            if 'args' not in sp:
                k, val = sp.split("=")
            else:
                k='args'
                val=sp[sp.find('{'):]
            if k=='args':
                argsDict = {}
                vals=val[1:-1].split(',')
                for v in vals:
                    v1,v2 = v.split(':')
                    argsDict[v1] = v2
                val=argsDict


            try:
                args[k] = self._eval(val)
            except:
                args[k] = val
        return sclass, args




class BasicGame(object):
    """ This regroups all the components of a game's dynamics, after parsing. """
    MAX_SPRITES = 10000

    default_mapping = {'w': ['wall'],
                       'A': ['avatar'],
                       }

    lastcollisions = {}
    block_size = 10
    frame_rate = 20
    load_save_enabled = True

    def __init__(self, **kwargs):
        from ontology import Immovable, DARKGRAY, BLACK, MovingAvatar, GOLD
        for name, value in kwargs.iteritems():
            # print "NAME: ", name
            if hasattr(self, name):
                self.__dict__[name] = value
            else:
                print "WARNING: undefined parameter '%s' for game! "%(name)

        # contains mappings to constructor (just a few defaults are known)
        self.sprite_constr = {'wall': (Immovable, {'color': DARKGRAY}, ['wall']),
                              'avatar': (MovingAvatar, {}, ['avatar']),
                              }
        # z-level of sprite types (in case of overlap)
        self.sprite_order  = ['wall',
                              'avatar',
                              ]
        # contains instance lists
        self.sprite_groups = defaultdict(list)
        # which sprite types (abstract or not) are singletons?
        self.singletons = []
        # collision effects (ordered by execution order)
        self.collision_eff = []

        self.playback_states = []
        self.playback_index = 0
        # for reading levels
        self.char_mapping = {}
        # termination criteria
        self.terminations = [] #[Termination()]
        # conditional criteria
        self.conditions = []
        # resource properties
        self.resources_limits = defaultdict(int)
        self.resources_colors = defaultdict(str)

        self.is_stochastic = False
        self._lastsaved = None
        self.win = None
        self.effectList = [] # list of effects that happened this current timestep
        self.spriteDistribution = {}
        self.object_token_spriteDistribution = {}
        self.spriteUpdateDict = defaultdict(int) ## track how many times we have run spriteType updates to each particular object
        self.movement_options = {}
        self.object_token_movement_options = {}
        self.all_objects = None

        self.EOS = EOS((-1, -1))

        self.reset()

    def reset(self):
        self.score = 0
        self.time = 0
        self.ended = False
        self.num_sprites = 0
        self.kill_list=[]
        self.all_killed=[] # All items that have been killed

    def buildLevel(self, lstr):
        from ontology import stochastic_effects
        lines = [l for l in lstr.split("\n") if len(l)>0]
        lengths = map(len, lines)
        assert min(lengths)==max(lengths), "Inconsistent line lengths."
        self.width = lengths[0]
        self.height = len(lines)
        assert self.width > 1 and self.height > 1, "Level too small."
        # assert self.width%2 == 0, "Level has odd-numbered width."
        # assert self.height%2==0, "Level has odd-numbered height."
        # rescale pixels per block to adapt to the level
        # self.block_size = max(2,int(800./max(self.width, self.height)))
        self.block_size = 30
        self.screensize = (self.width*self.block_size, self.height*self.block_size)

        # set up resources
        for res_type, (sclass, args, _) in self.sprite_constr.iteritems():
            if issubclass(sclass, Resource):
                if 'res_type' in args:
                    res_type = args['res_type']
                if 'color' in args:
                    self.resources_colors[res_type] = args['color']
                if 'limit' in args:
                    self.resources_limits[res_type] = args['limit']
            else:
                self.sprite_groups[res_type] = []

        # create sprites
        for row, l in enumerate(lines):
            for col, c in enumerate(l):
                if c in self.char_mapping:
                    pos = (col*self.block_size, row*self.block_size)
                    self._createSprite(self.char_mapping[c], pos)
                elif c in self.default_mapping:
                    pos = (col*self.block_size, row*self.block_size)
                    self._createSprite(self.default_mapping[c], pos)


        self.kill_list=[]

        for _, _, effect, _ in self.collision_eff:
            if effect in stochastic_effects:
                self.is_stochastic = True

        # guarantee that avatar is always visible
        self.sprite_order.remove('avatar')
        self.sprite_order.append('avatar')


    def emptyBlocks(self):
        alls = [s for s in self]
        res = []
        for col in range(self.width):
            for row in range(self.height):
                r = pygame.Rect((col*self.block_size, row*self.block_size), (self.block_size, self.block_size))
                free = True
                for s in alls:
                    if r.colliderect(s.rect):
                        free = False
                        break
                if free:
                    res.append((col*self.block_size, row*self.block_size))
        return res

    def randomizeAvatar(self):
        if len(self.getAvatars()) == 0:
            self._createSprite(['avatar'], choice(self.emptyBlocks()))

    def _createSprite(self, keys, pos):
        res = []

        for key in keys:
            if self.num_sprites > self.MAX_SPRITES:
                print "Sprite limit reached."
                return
            sclass, args, stypes = self.sprite_constr[key]
            # verify the singleton condition
            anyother = False
            for pk in stypes[::-1]:
                if pk in self.singletons:
                    if self.numSprites(pk) > 0:
                        anyother = True
                        break
            if anyother:
                continue
            s = sclass(pos=pos, size=(self.block_size, self.block_size), name=key, **args)
            s.stypes = stypes
            self.sprite_groups[key].append(s)
            self.num_sprites += 1
            if s.is_stochastic:
                self.is_stochastic = True
            res.append(s)

        return res

    def _createSprite_cheap(self, key, pos):
        """ The same, but without the checks, which speeds things up during load/saving"""
        sclass, args, stypes = self.sprite_constr[key]
        s = sclass(pos=pos, size=(self.block_size, self.block_size), name=key, **args)
        s.stypes = stypes
        self.sprite_groups[key].append(s)
        self.num_sprites += 1
        return s

    def _initScreen(self, size,headless):
        if(headless):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.init()
            self.screen = pygame.display.set_mode((1,1))
            self.background = pygame.Surface(size)
        else:
            from ontology import LIGHTGRAY
            pygame.init()
            self.screen = pygame.display.set_mode(size)
            self.background = pygame.Surface(size)
            self.background.fill(LIGHTGRAY)
            self.screen.blit(self.background, (0,0))

    def set_caption(self, text):
        pygame.display.set_caption(str(text))


    def __iter__(self):
        """ Iterator over all sprites (ordered) """
        for key in self.sprite_order:
            if key not in self.sprite_groups:
                # abstract type
                continue
            for s in self.sprite_groups[key]:
                yield s

    def numSprites(self, key):
        """ Abstract sprite groups are computed on demand only """
        deleted = len([s for s in self.kill_list if key in s.stypes])
        if key in self.sprite_groups:
            return len(self.sprite_groups[key])-deleted
        else:
            return len([s for s in self if key in s.stypes])-deleted

    def getSprites(self, key):
        if key in self.sprite_groups:
            return [s for s in self.sprite_groups[key] if s not in self.kill_list]
        else:
            return [s for s in self if key in s.stypes and s not in self.kill_list]

    def getAvatars(self):
        """ The currently alive avatar(s) """
        res = []
        for ss in self.sprite_groups.values():
            if ss and isinstance(ss[0], Avatar):
                res.extend([s for s in ss if s not in self.kill_list])
        return res

    ignoredattributes = ['stypes',
                             'name',
                             'lastmove',
                             'color',
                             'lastrect',
                             'resources',
                             'physicstype',
                             'physics',
                             'rect',
                             'alternate_keys',
                             'res_type',
                             'stype',
                             'ammo',
                             'draw_arrow',
                             'shrink_factor',
                             'prob',
                             'is_stochastic',
                             'cooldown',
                             'total',
                             'is_static',
                             'noiseLevel',
                             'angle_diff',
                             'only_active',
                             'airsteering',
                             'strength',
                             ]
    def getObjects(self):
        """
        Return dictionary with all the objects, and their parameters, from the full state.
        """
        obj_list = {}
        fs = self.getFullState()
        obs = fs['objects']

        for ob_type in obs:
            for ob in self.getSprites(ob_type):
                features = {'color':colorDict[str(ob.color)], 'row':(ob.rect.top)}
                type_vector = {'color':colorDict[str(ob.color)], 'row':(ob.rect.top)}
                sprite = ob
                obj_list[ob.ID] = {'sprite': sprite, 'position':(ob.rect.left, ob.rect.top), 'features':features, 'type': type_vector}
        return obj_list

    def getFullState(self,as_string = False):
        """ Return a dictionary that allows full reconstruction of the game state,
        e.g. for the load/save functionality. """
        # TODO: make sure this list is complete/correct -- maybe a naming convention would be easier,
        # if it distinguished in-game-mutable form immutable attributes!
        ias = self.ignoredattributes
        obs = {}
        for key in self.sprite_groups:
            ss = {}
            obs[key] = ss
            for s in self.getSprites(key):
                pos = (s.rect.left, s.rect.top)
                attrs = {}
                while pos in ss:
                    # two objects of the same type in the same location, we need to disambiguate
                    pos = (pos, None)
                if(as_string):
                    ss[str(pos)] = attrs
                else:
                    ss[pos] = attrs
                for a, val in s.__dict__.iteritems():
                    if a not in ias:
                        attrs[a] = val
                if s.resources:
                    attrs['resources'] = dict(s.resources)

        fs = {'score': self.score,
              'ended': self.ended,
              'win': self.win,
              'objects': obs}
        return fs

    def setFullState(self, fs,as_string = False):
        """ Reset the game to be exactly as defined in the fullstate dict. """
        self.reset()
        self.score = fs['score']
        self.ended = fs['ended']
        for key, ss in fs['objects'].iteritems():
            self.sprite_groups[key] = [] ## Added 4/31/17
            for ID, attrs in ss.iteritems():
                try:
                    p = attrs['x'], attrs['y']
                except:
                    p = attrs[x], attrs[y]
                s = self._createSprite_cheap(key, p)
                for a, val in attrs.iteritems():
                    if a == 'resources':
                        for r, v in val.iteritems():
                            s.resources[r] = v
                    else:
                        s.__setattr__(a, val)

    def getFullStateColorized(self,as_string=False):
        fs = self.getFullState(as_string=as_string)
        fs_colorized = deepcopy(fs)
        fs_colorized['objects'] = {}
        for sprite_name in fs['objects']:
            try:
                sclass, args, stypes = self.sprite_constr[sprite_name]
                fs_colorized['objects'][colorDict[str(args['color'])]] = fs['objects'][sprite_name]
            except: # Object color isn't immediately available
                sprite_type = []
                if stypes[0] in self.sprite_groups: # be CAREFUL. self.sprite_groups is a defaultdict. caused bugs for mario.
                    sprite_type = self.sprite_groups[stypes[0]]

                if sprite_type:
                    sprite_rep = sprite_type[0]
                    fs_colorized['objects'][colorDict[str(sprite_rep.color)]] = fs['objects'][sprite_name]

                # No more sprites left?
                else:
                    #print self.sprite_groups[stypes[0]]
                    pass

        return fs_colorized



    def _clearAll(self, onscreen=True):
        for s in set(self.kill_list):
            self.all_killed.append(s)
            if onscreen:
                s._clear(self.screen, self.background, double=True)
            self.sprite_groups[s.name].remove(s)
        if onscreen:
            for s in self:
                s._clear(self.screen, self.background)
        self.kill_list = []

    def _drawAll(self):
        for s in self:
            s._draw(self)

    def _updateCollisionDict(self, changedsprite):
        for key in changedsprite.stypes:
            if key in self.lastcollisions:
                del self.lastcollisions[key]

    def _eventHandling(self):
        self.lastcollisions = {}
        push_effect = 'bounceForward'
        back_effect = 'stepBack'
        force_collisions = []
        collision_set = set()
        new_collisions = True
        self.effectList = []
        dead = self.kill_list[:] # copy kill list
        created = []



        self.collision_eff.sort(key=lambda x:1 if x[2].__name__ in ['bounceForward','stepBack']
            else (2 if x[2].__name__ in ['killSprite'] else (3 if x[2].__name__ in ['changeResource', 'changeScore'] else 0)), reverse=True)
        # build the current sprite lists (if not yet available)
        # for class1, class2, effect, kwargs in self.collision_eff:
        while new_collisions:
            new_collisions = set()
            new_effects = []
            for class1, class2, effect, kwargs in self.collision_eff:
                for sprite_class in [class1, class2]:
                    if sprite_class not in self.lastcollisions:
                        if sprite_class in self.sprite_groups:
                            sprite_group = self.sprite_groups[sprite_class]
                        else:
                            sprite_group = []
                            for key in self.sprite_groups:
                                sprite = self.sprite_groups[key]
                                if sprite and sprite_class in sprite[0].stypes:
                                    sprite_group.extend(sprite)
                        self.lastcollisions[sprite_class] = (sprite_group[:], len(sprite_group))

                # special case for end-of-screen
                if class2 == "EOS":
                    ss1, l1 = self.lastcollisions[class1]

                    for s1 in ss1:
                        if not pygame.Rect((0,0), self.screensize).contains(s1.rect):
                            new_effects.append(effect(s1, self.EOS, self, **kwargs))
                    continue

                # iterate over the shorter one
                sprite_list1 = self.lastcollisions[class1][0][:]
                sprite_list2 = self.lastcollisions[class2][0][:]

                # score argument is not passed along to the effect function
                score = 0
                if 'scoreChange' in kwargs:
                    kwargs = kwargs.copy()
                    score = kwargs['scoreChange']
                    del kwargs['scoreChange']

                dim = None
                if 'dim' in kwargs:
                    kwargs = kwargs.copy()
                    dim = kwargs['dim']
                    del kwargs['dim']

                # do collision detection
                for sprite1 in sprite_list1:
                    for collision_index in sprite1.rect.collidelistall(sprite_list2):
                        sprite2 = sprite_list2[collision_index]
                        if (sprite1 == sprite2
                            or sprite1 in dead
                            or sprite2 in dead
                            or (sprite1, sprite2) in collision_set):
                            continue
                        new_collisions.add((sprite1, sprite2))

                        # deal with the collision effects
                        if score:
                            self.score += score
                        # hacky ways of adding to the language, should fix
                        if 'applyto' in kwargs:
                            kwargs = kwargs.copy()
                            stype = kwargs['applyto']
                            del kwargs['applyto']
                            for sC in self.getSprites(stype):
                                new_effects.append((effect, sC, sprite1, self, kwargs))
                                spritesActedOn.add(sC)
                            continue

                        if dim:
                            sprites = self.getSprites(classprite1)
                            spritesFiltered = filter(lambda sprite: sprite.__dict__[dim] == sprite2.__dict__[dim], sprites)
                            for sC in spritesFiltered:
                                new_effects.append((effect, sprite1, sC, self, kwargs))
                                spritesActedOn.add(sprite1)
                            continue

                        if effect.__name__ == 'changeResource':
                            resource = kwargs['resource']
                            (sclass, args, stypes) = self.sprite_constr[resource]
                            resource_color = args['color']
                            new_effects.append(effect(sprite1, sprite2, resource_color, self, **kwargs))

                        elif effect.__name__ == 'transformTo':

                            new_effects.append(effect(sprite1, sprite2, self, **kwargs))
                            new_sprite = self.getSprites(kwargs['stype'])[-1]
                            new_collisions.add((sprite1, new_sprite))
                            dead.append(sprite1)

                        # Deal with push effects
                        elif effect.__name__ == push_effect:
                            for collision in force_collisions:
                                if sprite2 in collision: # object is getting pushed by another object that got pushed
                                    collision.add(sprite1)
                            else:
                                force_collisions.append(set([sprite1, sprite2])) # first time object is pushed (probably avatar pushing)

                            new_effects.append(effect(sprite1, sprite2, self, **kwargs)) # apply effect
                        elif effect.__name__ == back_effect:
                            # possibly need to undo all effects
                            for collision in force_collisions:
                                if sprite1 in collision: # check if sprite1 got pushed back
                                    for sprite in collision:
                                        effect(sprite, sprite2, self, **kwargs) # apply push back to all sprites in that set
                            else: # if there were no sprites in the collision, do normal thing
                                new_effects.append(effect(sprite1, sprite2, self, **kwargs))

                        else:
                            new_effects.append(effect(sprite1, sprite2, self, **kwargs))

            self.effectList += [new_effect for new_effect in new_effects if new_effect]
            collision_set = collision_set.union(new_collisions)

        self.kill_list = list(set(self.kill_list))

        ## Remove duplicates
        new_collision_eff = []
        for element in self.effectList:
            if element not in new_collision_eff:
                new_collision_eff.append(element)
        self.effectList = new_collision_eff


        # self.kill_list = dead[:]
        # if len(self.effectList) > 0:
            # print 'effectList', self.effectList
        # self.effectList = list(set(self.effectList))

        return self.effectList

    def startPlaybackGame(self, headless, persist_movie, make_images=False, make_movie=False, movie_dir=False, padding=0):
        """
        Main method to run game.
        """
        # ----------- Initialization ----------


        self._initScreen(self.screensize,headless)
        pygame.display.flip()
        self.reset()
        clock = pygame.time.Clock()
        self.frame_rate = 5



        win = False
        i = 0

        lastKeyPress=(0,0,1) # PT: initialize to fake keypress index
        lastKeyPressTime=0 #PT

        # Logging
        f = sys.argv[0]
        m = re.search('([A-Za-z0-9]+)\.py', f)
        name = m.group(1)
        gamelog = "{}.log".format(name)
        #logging.basicConfig(filename=gamelog, level=logging.INFO)
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        game_output = "output/{}_{}.txt".format(name, timestamp)
        sprite_output = "output/{}_{}_sprites.txt".format(name,timestamp)

        # --------- Game-play ------------
        from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
        from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution
        from ontology import spriteInduction
        # from theory_template import *
        finalEventList = []
        agentStatePrev = {}
        agentState = dict(self.getAvatars()[0].resources)
        keyPressPrev = None

        ##uncomment to write output
        # f_sprite = open(sprite_output,"w")

        # Prep for Sprite Induction
        sprite_types = [Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile]
        self.all_objects = self.getObjects() # Save all objects, some which may be killed in game

        ##figure out keypress type:
        disableContinuousKeyPress = False#all([self.all_objects[k]['sprite'].physicstype.__name__=='GridPhysics' for k in self.all_objects.keys()])

        objects = self.getObjects()
        self.spriteDistribution = {}
        self.movement_options = {}
        allStates = [self.getFullState()]


        while self.playback_index < len(self.playback_states):
            clock.tick(self.frame_rate)
            self.time += 1

            self._clearAll()

            try:
                self.setFullState(self.playback_states[self.playback_index])
            except:
                print "playback is failing"
                embed()

            # Save the event and agent state
            try:
                agentState = dict(self.getAvatars()[0].resources)
                agentStatePrev = agentState
                keyPressPrev = keyPressType

            # If agent is killed before we get agentState
            except Exception as e:              # TODO: how to process changes in resources that led to termination state?
                agentState = agentStatePrev
                keyPressType = keyPressPrev

            collision_objects = set()

            self._drawAll()
            pygame.display.update(VGDLSprite.dirtyrects)
            allStates.append(self.getFullState())

            #if(headless):
            if(make_images):
                tmp_dir = "images/tmp/"
                # tmpl = '{tmp_dir}%09d-{name}-{g_id}.png'.format(i, tmp_dir = tmp_dir, name="VGDL-GAME", g_id=self.uiud)
                img_index = len([d for d in os.listdir(tmp_dir) if d != '.DS_Store'])
                tmpl = '{tmp_dir}%09d.png'.format(img_index, tmp_dir = tmp_dir)
                if padding and (i==0 or i==len(self.playback_states)-1): ## add padding to first and last frame.
                    for j in range(padding):
                        pygame.image.save(self.screen, tmpl%(img_index+j))
                else:
                    pygame.image.save(self.screen, tmpl%img_index)

                i+=1

            VGDLSprite.dirtyrects = []
            allStates.append(self.getFullState())

            self.playback_index += 1


        # Print entire history of effects
        terminationCondition = {'ended': True, 'win':win, 'time':self.time}

        if win:
            self.score += 1
            self.win = True
            print "Game won, with score %s" % self.score
        else:
            self.score -= 1
            self.win = False
            print "Playback is incomplete, or game is lost. Score=%s" % self.score

        # ipdb.set_trace()

        # pause a few frames for the player to see the final screen.
        pygame.time.wait(10)
        return win, self.score


    def startGame(self, headless, persist_movie, make_images=False, make_movie=False):
        """
        Main method to run game.
        """
        # ----------- Initialization ----------
        self._initScreen(self.screensize,headless)
        pygame.display.flip()
        self.reset()
        t1 = time.time()
        self.actions = []
        clock = pygame.time.Clock()
        if self.playback_states:
            self.frame_rate = 1

        win = False
        i = 0

        lastKeyPress=(0,0,1) # PT: initialize to fake keypress index
        lastKeyPressTime=0 #PT

        # Logging
        f = sys.argv[0]
        m = re.search('([A-Za-z0-9]+)\.py', f)
        name = m.group(1)
        gamelog = "{}.log".format(name)
        #logging.basicConfig(filename=gamelog, level=logging.INFO)
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        game_output = "output/{}_{}.txt".format(name, timestamp)
        sprite_output = "output/{}_{}_sprites.txt".format(name,timestamp)

        # --------- Game-play ------------
        from ontology import Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile
        from ontology import initializeDistribution, updateDistribution, updateOptions, sampleFromDistribution
        from ontology import spriteInduction
        # from theory_template import *
        finalEventList = []
        agentStatePrev = {}
        agentState = dict(self.getAvatars()[0].resources)
        keyPressPrev = None

        ##uncomment to write output
        # f_sprite = open(sprite_output,"w")

        # Prep for Sprite Induction
        sprite_types = [Immovable, Passive, Resource, ResourcePack, RandomNPC, Chaser, AStarChaser, OrientedSprite, Missile]
        self.all_objects = self.getObjects() # Save all objects, some which may be killed in game

        ##figure out keypress type:
        disableContinuousKeyPress = all([self.all_objects[k]['sprite'].physicstype.__name__=='GridPhysics' for k in self.all_objects.keys()])

        objects = self.getObjects()
        self.spriteDistribution = {}
        self.movement_options = {}
        allStates = [self.getFullState()]
        # spriteInduction(self, step=0)
        # for sprite in objects:
        #     self.spriteDistribution[sprite] = initializeDistribution(sprite_types) # Indexed by object ID
        #     self.movement_options[sprite] = {"OTHER":{}}
        #     for sprite_type in sprite_types:
        #         self.movement_options[sprite][sprite_type] = {}

        self.collision_eff.sort(key=lambda x:1 if x[2].__name__ in ['bounceForward','stepBack']
            else (2 if x[2].__name__ in ['killSprite'] else (3 if x[2].__name__ in ['changeResource', 'changeScore'] else 0)), reverse=True)


        while not self.ended:
            clock.tick(self.frame_rate)
            self.time += 1

            self._clearAll()

            # gather events
            pygame.event.pump()

            # get action pressed
            self.keystate = pygame.key.get_pressed()

            # # PT: Disables mistaken contiguous key presses, prints to terminal
            if disableContinuousKeyPress and not self.playback_states:
                keyPressType = None
                if self.keystate != emptyKeyState:
                    if (self.time-lastKeyPressTime)<2 and self.keystate==lastKeyPress:
                        self.keystate = emptyKeyState
                    else:
                        lastKeyPress = self.keystate
                        # if self.keystate[pygame.K_RETURN] and self.playback_actions:
                        #     self.keystate = list(self.keystate)
                        #     self.keystate[actionToKeyPress[self.playback_actions[self.playback_index]]] = True
                        #     self.keystate = tuple(self.keystate)
                        #     self.playback_index += 1

                        if lastKeyPress.index(1) in keyPresses.keys():
                            keyPressType = keyPresses[lastKeyPress.index(1)]
                            # print keyPressType


                    lastKeyPressTime = self.time


            # # load/save handling
            # if self.load_.save_enabled:
            #     from pygame.locals import K_1, K_2
            #     if self.keystate[K_2] and self._lastsaved is not None:
            #         self.setFullState(self._lastsaved)
            #         self._initScreen(self.screensize,headless)
            #         pygame.display.flip()
            #     if self.keystate[K_1]:
            #         self._lastsaved = self.getFullState()


            # Save the event and agent state
            try:
                agentState = dict(self.getAvatars()[0].resources)
                agentStatePrev = agentState
                keyPressPrev = keyPressType

            # If agent is killed before we get agentState
            except Exception as e:              # TODO: how to process changes in resources that led to termination state?
                agentState = agentStatePrev
                keyPressType = keyPressPrev

            if keyPressType is not None:
                self.actions.append(keyPressType)
            collision_objects = set()

            if self.effectList:
                state = self.getFullState()
                event = {'agentState': agentState, 'agentAction': keyPressType, 'effectList': self.effectList, 'gameState': self.getFullStateColorized()}
                finalEventList.append(event)

                # Get objects involved in the effectList
                for effect in event['effectList']:
                    if len(effect) == 3:
                        collision_objects.add(effect[1])
                        collision_objects.add(effect[2])
                    elif len(effect) == 2:
                        collision_objects.add(effect[1])


            # Termination #1
            for t in self.terminations:
                self.ended, win = t.isDone(self)
                if self.ended:
                    if win:
                        self.score += 1
                        # winning a game always gives a positive score.
                        # if self.score <= 0:
                        #     self.score = 1

                        self.win = True
                        print time.time()-t1, len(self.actions), win, self.score
                        print "Game won, with score %s" % self.score
                    else:
                        self.win = False
                        self.score -=1 ## Added 3/16/17
                        print time.time()-t1, len(self.actions), win, self.score
                        print "Game lost. Score=%s" % self.score
                    np.save("temp_data.npy", [time.time()-t1, len(self.actions), self.win, self.score])
                    allStates.append(self.getFullState())

                    pygame.time.wait(10)
                    print len(self.actions), win, self.score
                    return win, self.score
                    # pygame.quit()
                    # sys.exit()
                    # break

            # Conditional Criteria
            for conditional in self.conditions:
                condition, eclass = conditional
                effect, kwargs = eclass

                if condition.condition(self):
                    stype = kwargs['applyto']
                    kwargs_use = deepcopy(kwargs)
                    kwargs_use.pop('applyto')
                    for sC in self.getSprites(stype):

                        effect(sC, sC, self, **kwargs_use)


            ## Update actual sprite positions.
            for s in self:
                s.update(self)

            # handle collision effects
            self._eventHandling()

            # Termination #2 : Avatars have been killed
            if len(self.getAvatars()) == 0:
                break

            self._drawAll()
            pygame.display.update(VGDLSprite.dirtyrects)
            allStates.append(self.getFullState())

            #if(headless):
            if(persist_movie):
                tmp_dir = "./temp/"
                tmpl = '{tmp_dir}%09d-{name}-{g_id}.png'.format(i,tmp_dir = tmp_dir, name="VGDL-GAME", g_id=self.uiud)
                pygame.image.save(self.screen, tmpl%i)
                i+=1
            VGDLSprite.dirtyrects = []
            allStates.append(self.getFullState())

        if(persist_movie):
            print "Creating Movie"
            self.video_file = "./videos/" +  str(self.uiud) + ".mp4"
            subprocess.call(["ffmpeg","-y",  "-r", "30", "-b", "800", "-i", tmpl, self.video_file ])
            [os.remove(f) for f in glob.glob(tmp_dir + "*" + str(self.uiud) + "*")]

        # Print entire history of effects
        terminationCondition = {'ended': True, 'win':win, 'time':self.time}
        # logging.info((finalEventList, terminationCondition))

        # Recording results into files
        # with open(game_output, 'w') as f:
        #     f.write(str((finalEventList, terminationCondition)))
        # f_sprite.write(str(self.all_objects) + "\n")
        # f_sprite.write(str(self.spriteDistribution))
        # f_sprite.close()

        # print "Expecting {} events".format(len(finalEventList))

        if win:
            # winning a game always gives a positive score.
            # if self.score <= 0:
                # self.score = 1
            self.score +=1 # Added 3/16/17
            self.win = True
            print "Game won, with score %s" % self.score
            np.save("temp_data.npy", [time.time()-t1, len(self.actions), self.win, self.score])

        else:
            self.win = False
            self.score -=1 # Added 3/16/17
            print "Game lost. Score=%s" % self.score
            np.save("temp_data.npy", [time.time()-t1, len(self.actions), self.win, self.score])


        # pause a few frames for the player to see the final screen.
        pygame.time.wait(10)
        print len(self.actions), win, self.score
        return win, self.score


    def getPossibleActions(self):
        return self.getAvatars()[0].declare_possible_actions()

    def startGameExternalPlayer(self, headless, persist_movie, movie_dir):
        print "in startgameexternalplayer"
        embed()
        self._initScreen(self.screensize, headless)
        pygame.display.flip()
        self.reset()
        self.clock = pygame.time.Clock()
        self.tmp_dir = movie_dir
        self.video_tmpl = '{tmp_dir}%09d-{name}-{g_id}.png'.format(self.time,tmp_dir = self.tmp_dir, name="VGDL-GAME", g_id=self.uiud)


    def tick(self,action,headless=True, persist_movie=False):

        win = False
        self.screen.fill(LIGHTGRAY)
        #self.clock.tick(self.frame_rate)
        self.time += 1
        if not headless:
            self._clearAll()

        # gather events
        pygame.event.pump()
        self.keystate = list(pygame.key.get_pressed())

        self.keystate[action] = 1

            # load/save handling
        #if self.load_save_enabled:
        #        from pygame.locals import K_1, K_2
        #        if self.keystate[K_2] and self._lastsaved is not None:
        #            self.setFullState(self._lastsaved)
        #            self._initScreen(self.screensize,headless)
        #            pygame.display.flip()
        #        if self.keystate[K_1]:
        #            self._lastsaved = self.getFullState()

            # termination criteria
        for t in self.terminations:
                self.ended, win = t.isDone(self)
                if self.ended:
                    return win, self.score
            # update sprites
        #print action

        for s in self:
            s.update(self)


        # handle collision effects
        self._eventHandling()

        self._drawAll()
        if not headless:
            self._drawAll()
            pygame.display.update(VGDLSprite.dirtyrects)
            VGDLSprite.dirtyrects = []

        return None, None




class VGDLSprite(object):
    """ Base class for all sprite types. """
    name = None
    COLOR_DISC = [20,80,140,200]
    dirtyrects = []
    is_static= False
    only_active =False
    is_avatar= False
    is_stochastic = False
    color    = None
    cooldown = 1
    # cooldown = 0 # pause ticks in-between two moves
    speed    = None
    mass     = 1
    physicstype=None
    shrinkfactor=0

    def __init__(self, pos, size=(10,10), color=None, speed=None, cooldown=None, physicstype=None, **kwargs):
        from ontology import GridPhysics
        self.rect = pygame.Rect(pos, size)
        self.x = pos[0]
        self.y = pos[1]
        self.lastrect = self.rect
        self.physicstype = physicstype or self.physicstype or GridPhysics
        self.physics = self.physicstype()
        self.physics.gridsize = size
        self.speed = speed or self.speed
        self.cooldown = cooldown or self.cooldown
        # self.ID = id(self) # TODO: Make sure that these are unique, maintained during the lifetime of the object
        self.ID = uuid.uuid1()
        self.direction = None
        #TODO: change the choice to be from colors that are not taken?
        self.color = color or self.color or PURPLE#(140, 20, 140)
        if self.color == ENDOFSCREEN:
            self.ID = 'ENDOFSCREEN'
        self.colorName = colorDict[str(self.color)]
        # print 'color', self.color


        #self.color = color or self.color or (choice(self.COLOR_DISC), choice(self.COLOR_DISC), choice(self.COLOR_DISC))
        for name, value in kwargs.iteritems():
            try:
                self.__dict__[name] = value
            except:
                print "WARNING: undefined parameter '%s' for sprite '%s'! "%(name, self.__class__.__name__)
        # how many timesteps ago was the last move?
        self.lastmove = 0

        # management of resources contained in the sprite
        self.resources = defaultdict(int)

    def update(self, game, random_npc=False):
        """ The main place where subclasses differ. """
        self.x = self.rect.x
        self.y = self.rect.y
        self.lastrect = self.rect
        # no need to redraw if nothing was updated
        self.lastmove += 1
        # if self.colorName == 'RED':
            # ipdb.set_trace()
        if not self.is_static and not self.only_active and not random_npc:
            self.physics.passiveMovement(self)

    def _updatePos(self, orientation, speed=None):
        if speed is None:
            speed = self.speed
        if (self.lastmove+1)%self.cooldown==0 and abs(orientation[0])+abs(orientation[1])!=0:
        # if not( ((self.lastmove+1) % self.cooldown != 0) or abs(orientation[0])+abs(orientation[1])==0): ##used this until 9/14
            self.rect = self.rect.move((orientation[0]*speed, orientation[1]*speed))
            # self.lastmove = 0


    def _velocity(self):
        """ Current velocity vector. """
        if self.speed is None or self.speed==0 or not hasattr(self, 'orientation'):
            return (0,0)
        else:
            return (self.orientation[0]*self.speed, self.orientation[1]*self.speed)

    @property
    def lastdirection(self):
        return (self.rect[0]-self.lastrect[0], self.rect[1]-self.lastrect[1])

    def _draw(self, game):
        from ontology import LIGHTGREEN
        screen = game.screen
        if self.shrinkfactor != 0:
            shrunk = self.rect.inflate(-self.rect.width*self.shrinkfactor,
                                       -self.rect.height*self.shrinkfactor)
        else:
            shrunk = self.rect

        if self.is_avatar:
            rounded = roundedPoints(shrunk)
            pygame.draw.polygon(screen, self.color, rounded)
            pygame.draw.lines(screen, LIGHTGREEN, True, rounded, 2)
            r = self.rect.copy()
        elif not self.is_static:
            rounded = roundedPoints(shrunk)
            pygame.draw.polygon(screen, self.color, rounded)
            r = self.rect.copy()
        else:
            r = screen.fill(self.color, shrunk)
        if self.resources:
            self._drawResources(game, screen, shrunk)
        VGDLSprite.dirtyrects.append(r)

    def _drawResources(self, game, screen, rect):
        """ Draw progress bars on the bottom third of the sprite """
        from ontology import BLACK
        tot = len(self.resources)
        barheight = rect.height/3.5/tot
        offset = rect.top+2*rect.height/3.
        for r in sorted(self.resources.keys()):
            wiggle = rect.width/10.
            try:
                prop = max(0,min(1,self.resources[r] / float(game.resources_limits[r])))
            except ZeroDivisionError:
                prop = max(1,min(1,self.resources[r]))
            filled = pygame.Rect(rect.left+wiggle/2, offset, prop*(rect.width-wiggle), barheight)
            rest   = pygame.Rect(rect.left+wiggle/2+prop*(rect.width-wiggle), offset, (1-prop)*(rect.width-wiggle), barheight)
            screen.fill(game.resources_colors[r], filled)
            screen.fill(BLACK, rest)
            offset += barheight

    def _clear(self, screen, background, double=False):
        r = screen.blit(background, self.rect, self.rect)
        VGDLSprite.dirtyrects.append(r)
        if double:
            r = screen.blit(background, self.lastrect, self.lastrect)
            VGDLSprite.dirtyrects.append(r)

    def __repr__(self):
        return str(self.name)+" at (%s,%s)"%(self.rect.left, self.rect.top)

class EOS(VGDLSprite):
    color = ENDOFSCREEN

class Avatar(object):
    """ Abstract superclass of all avatars. """
    shrinkfactor=0.15

    def __init__(self):
        self.actions = self.declare_possible_actions()

class Resource(VGDLSprite):
    """ A special type of object that can be present in the game in two forms, either
    physically sitting around, or in the form of a counter inside another sprite. """
    value=1
    limit=2
    res_type = None

    @property
    def resourceType(self):
        if self.res_type is None:
            return self.name
        else:
            return self.res_type

class Termination(object):
    """ Base class for all termination criteria. """
    # def __init__(self):
    #     self.name = 'Generic'
    def isDone(self, game):
        """ returns whether the game is over, with a win/lose flag """
        from pygame.locals import K_ESCAPE, QUIT
        if game.keystate[K_ESCAPE] or pygame.event.peek(QUIT):
            return True, False
        else:
            return False, None

class Conditional(object):
    """ Base class for all conditional criteria"""
    def condition(self, game):
        """ returns true if condition is met. default returns false"""
        return False

# def makeVideo(movie_dir):
#     import os
#     print "Creating Movie"
#     # self.video_file = "videos/" +  str(self.uiud) + ".mp4"
#     if not os.path.exists(movie_dir):
#         print movie_dir, "didn't exist. making new dir"
#         os.makedirs(movie_dir)
#     round_index = len([d for d in os.listdir(movie_dir) if d != '.DS_Store'])
#     video_dirname = movie_dir+"/round"+str(round_index)+".mp4"
#     images_dir = "images/tmp/%09d.png"
#     com = "ffmpeg -i " +images_dir+ " -pix_fmt yuv420p -filter:v 'setpts=4.0*PTS' "+ video_dirname
#     command = "{}".format(com)
#     subprocess.call(command, shell=True)
#     # empty image directory
#     shutil.rmtree("images/tmp")
#     os.makedirs("images/tmp")
