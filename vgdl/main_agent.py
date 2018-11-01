# from IPython import embed
from util import *
from core import colorDict, VGDLParser, sys, keyPresses
from ontology import *
from theory_template import TimeStep, Precondition, InteractionRule, TerminationRule, TimeoutRule, \
SpriteCounterRule, MultiSpriteCounterRule, ruleCluster, Theory, Game, writeTheoryToTxt, generateSymbolDict, \
generateTheoryFromGame
import os, subprocess, shutil
from collections import defaultdict
import WBP
import importlib
import numpy as np
import random
import cPickle
import time
from datetime import datetime
import copy
from metaplanner import translateEvents, observe
from rlenvironmentnonstatic import createRLInputGame, createRLInputGameFromStrings, defInputGame, createMindEnv
from termcolor import colored
from pygame import K_LEFT, K_UP, K_RIGHT, K_DOWN, K_SPACE

# from line_profiler import LineProfiler

MAX_STEPS = 1000
actionDict = {K_SPACE: 'space', K_UP: 'up', K_DOWN: 'down', K_LEFT: 'left', K_RIGHT: 'right', 0:'none'}
AvatarTypes = [MovingAvatar, HorizontalAvatar, VerticalAvatar, FlakAvatar, AimedFlakAvatar, OrientedAvatar,
RotatingAvatar, RotatingFlippingAvatar, NoisyRotatingFlippingAvatar, ShootAvatar, AimedAvatar,
AimedFlakAvatar, InertialAvatar, MarioAvatar]

def playCurriculum(agent, level_game_pairs):
    # Necessary to define a top-level function for playCurriculum so that
    # hyperopt.mongoexpt can correctly pickle the objective function
    start_time = time()
    agent.playCurriculum(level_game_pairs)
    end_time = time() - start_time
    return end_time

hyperparameter_sets = [
    {'idx'           : 0,
     'short_horizon' : False,
     'first_order_horizon': True,
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 100,
     'sprite_negative_mult': .1,
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 50,
     },
    {'idx'           : 1,
     'short_horizon' : False,
     'first_order_horizon': False,
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 100,
     'sprite_negative_mult': 10.,
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 50,
     },
    {'idx'           : 2,
     'short_horizon' : False,
     'first_order_horizon': False,
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 100,
     'sprite_negative_mult': .1,
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 50,
     },
    {'idx'           : 3,
     'short_horizon' : True,
     'first_order_horizon': True,
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 100,
     'sprite_negative_mult': 10, #normally .1
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 50,
     },
    {'idx'           : 4,
     'short_horizon' : True,
     'first_order_horizon': True,
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 100,
     'sprite_negative_mult': .1, #normally .1
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 10,
     }
]


class Agent:
    def __init__(self, modelType, gameFilename, hyperparameter_sets, hyperparameter_index=3, IW_k=2, extra_atom_allowed=True):
        self.modelType = modelType
        self.gameFilename = gameFilename
        self.gameString = None
        self.levelString = None
        self.display_text = False
        self.display_states = True
        self.record_states = True
        self.record_video_info = True
        self.hyperparameter_sets = hyperparameter_sets
        self.hyperparameter_index = hyperparameter_index
        self.hyperparameters = hyperparameter_sets[hyperparameter_index]
        self.annealingFactor = 1.
        self.shortHorizon = self.hyperparameters['short_horizon']#False
        self.firstOrderHorizon = self.hyperparameters['first_order_horizon'] #True ## Makes you commit to a plan once first-order distances change (e.g., spritecounter values)
        self.IW_k = IW_k
        self.extra_atom_allowed = extra_atom_allowed ## for analysis, allows for toggling whether we allow the below.
        self.extra_atom = False
        self.param_ID = "params__IW={}__ea={}".format(self.IW_k, self.extra_atom_allowed)
        if self.shortHorizon == True:
            self.starting_max_nodes = 500
            self.max_nodes_annealing = 1.05
        else:
            self.starting_max_nodes = 1000
            self.max_nodes_annealing = 2.
        self.conservative = False
        self.regrounding = 1
        self.selective_regrounding = True
        self.reground_for_npcs = False
        self.safeDistance = 3
        self.emptyPlansLimit = 5
        self.longHorizonObservationLimit = 2
        self.hypotheses = []
        self.symbolDict = None
        self.finalEventList = []
        self.finalEffectList = set()
        self.finalTimeStepList = []
        self.statesEncountered = []
        self.rleHistory = []
        self.episodeRecord = []
        self.fakeInteractionRules = []
        self.all_objects = {}
        self.bestSpriteTypeDict = defaultdict(lambda : {})
        self.spriteUpdateDict = defaultdict(lambda : 0)
        self.max_game_time_observed = 0
        self.best_params = None
        self.seen_resources = []
        self.seen_limits = []
        self.new_objects = {}
        self.actionSeqLength = 0.
        self.skipInduction = False

        # Hyperopt output
        self.total_game_steps = 0
        self.total_planner_steps = 0
        self.levels_won = 0

        self.todo_delete = True

    def hyperparameterSwitch(self, new_index):
        if new_index!=self.hyperparameter_index:
            self.hyperparameter_index = new_index
            self.hyperparameters = self.hyperparameter_sets[new_index]
            self.shortHorizon = self.hyperparameters['short_horizon']
            self.firstOrderHorizon = self.hyperparameters['first_order_horizon'] ## Makes you commit to a plan once first-order distances change (e.g., spritecounter values)
            if self.shortHorizon == True:
                self.starting_max_nodes = random.choice([200, 500])
                self.max_nodes_annealing = 1.05
            else:
                self.starting_max_nodes = 1000
                self.max_nodes_annealing = 2. 
            self.max_nodes = self.starting_max_nodes
            self.stored_max_nodes = self.max_nodes

            if self.display_text:
                print "Switching hyperparameters to {}".format(new_index)
        planner_hyperparameters = dict((k, self.hyperparameters[k]) for k in self.hyperparameters.keys() if k not in ['short_horizon', 'first_order_horizon'])
        return planner_hyperparameters

    def initializeEnvironment(self):
        if self.gameString==None or self.levelString==None:
            self.gameString, self.levelString = defInputGame(self.gameFilename, randomize=False)
        self.rleCreateFunc = lambda: createRLInputGameFromStrings(self.gameString, self.levelString)
        self.rle = self.rleCreateFunc()
        self.rle._game.spriteUpdateDict = self.spriteUpdateDict
        return

    def initializeRLEFromGame(self):
        gameString, levelString = self.gameString, self.levelString
        if gameString == None or levelString == None:
            gameString, levelString = defInputGame(self.gameFilename, randomize=False)
        rleCreateFunc = lambda: createRLInputGameFromStrings(gameString, levelString)
        rle = rleCreateFunc()
        return rle

    def fastcopy(self, rle):

        newRle = self.initializeRLEFromGame()
        newRle._obstypes = ccopy(rle._obstypes)
        if hasattr(rle, '_gravepoints'):
            newRle._gravepoints = ccopy(rle._gravepoints)
        newRle._game.sprite_groups = ccopy(rle._game.sprite_groups)
        newRle._game.kill_list = ccopy(rle._game.kill_list)
        # newRle._game.lastcollisions = ccopy(rle._game.lastcollisions)
        newRle._game.time = ccopy(rle._game.time)
        newRle._game.score = ccopy(rle._game.score)
        newRle._game.keystate = ccopy(rle._game.keystate)
        newRle.symbolDict = ccopy(rle.symbolDict)
        newRle._game.sprite_groups['avatar'][0].resources = ccopy(rle._game.sprite_groups['avatar'][0].resources)

        return newRle

    def getSpritesByColor(self, rle, color):
        outList = []
        for k in rle._game.sprite_groups.keys():
            if rle._game.sprite_groups[k] and rle._game.sprite_groups[k][0].colorName==color:
                outList.extend(rle._game.sprite_groups[k])
        if outList:
            return outList
        else:
            return None

    def findNearestSprite(self, sprite, spriteList):
        ## returns the sprite in spriteList whose location best matches the location of sprite.
        return sorted(spriteList, key=lambda x:abs(x.rect[0]-sprite.rect[0])+abs(x.rect[1]-sprite.rect[1]))[0]

    def setSpritePositions(self, rle, Vrle, hypothesis):
        ## Sets positions of objects in Vrle to what they were in the rle. Bypasses clunky VGDL level description.

        old_sprite_groups = Vrle._game.sprite_groups
        for k in old_sprite_groups.keys():
            if old_sprite_groups[k]:
                color = Vrle._game.sprite_groups[k][0].colorName
                matchingSpritesInRLE = self.getSpritesByColor(rle, color)
                for sprite in old_sprite_groups[k]:
                    matchingSprite = self.findNearestSprite(sprite, matchingSpritesInRLE)
                    sprite.rect = matchingSprite.rect
                    sprite.lastmove = matchingSprite.lastmove
                    sprite.ID2 = matchingSprite.ID
                    if 'Missile' in str(hypothesis.classes[sprite.name][0].vgdlType) and self.best_params!=None:
                        try:
                            ## Enforce consistency: inferred value for individual orientations has to be consistent with what we're saying the horizontal/vertical orientation is of the entire group.
                            # embed()

                            orientation = tuple(np.sign(np.array(self.rle._game.previousPositions[matchingSprite.ID]) - np.array(self.rle._game.objectMemoryDict[matchingSprite.ID])))
                            # if color=='RED':
                                # embed()
                            if orientation == (0,0):
                                # print "found 0,0 orientation. Using generic missile orientation:", sprite.orientation, sprite.speed, sprite.cooldown
                                pass
                            #     embed()

                            else:
                                sprite.orientation = orientation

                        except KeyError:
                            # print "Failed to get params for Missile in main_agent"
                            # embed()
                            pass
        for k,v in Vrle._game.sprite_groups.items():
            for sprite in v:
                if sprite not in Vrle._game.kill_list:
                    loc = (sprite.rect.left, sprite.rect.top)
                    if loc in Vrle._game.positionDict.keys():
                        Vrle._game.positionDict[loc].append(sprite)
                    else:
                        Vrle._game.positionDict[loc] = [sprite]
        return


    def initializeVrle(self, hypothesis):
        ## World in agent's head given 'hypothesis', including object goal
        # gameString, levelString, symbolDict = writeTheoryToTxt(self.rle, hypothesis, self.symbolDict,\
        #          "./theory_files/hyperparameter_idx_{}/{}.py".format(self.hyperparameters['idx'], self.gameFilename))
        gameString, levelString, symbolDict = writeTheoryToTxt(self.rle, hypothesis, self.symbolDict,\
                 "./theory_files/{}.py".format(self.gameFilename))
        Vrle = createMindEnv(gameString, levelString, output=False)

        self.setSpritePositions(self.rle, Vrle, hypothesis)
        try:
            Vrle._game.getAvatars()[0].resources = copy.deepcopy(self.rle._game.getAvatars()[0].resources)
            Vrle._game.getAvatars()[0].orientation = copy.deepcopy(self.rle._game.getAvatars()[0].orientation)
        except (IndexError, AttributeError) as e:
            pass
        return Vrle

    def VrleInitPhase(self, flexible_goals=False):
        ## Initialize multiple VRLEs, each corresponding to one hypothesis in self.hypotheses
        VRLEs = []

        for hypothesis in self.hypotheses[0:1]:
            tempHypothesis = copy.deepcopy(hypothesis)
            tmpFakeInteractionRules = copy.deepcopy(self.fakeInteractionRules)
            tempHypothesis.interactionSet.extend(tmpFakeInteractionRules)
            if not flexible_goals:
                tempHypothesis.updateTerminations()
            VRLEs.append(self.initializeVrle(tempHypothesis))

        return VRLEs

    def initializeHypotheses(self, allObjects, statesEncountered, compactStates, learnSprites=True):
        if learnSprites:
            if not self.skipInduction:
                self.observe(self.rle, 15, self.bestSpriteTypeDict, statesEncountered, compactStates, display=self.display_states)
            else:
                self.observe(self.rle, 1, self.bestSpriteTypeDict, statesEncountered, compactStates, display=self.display_states)                
            spriteTypeHypothesis, exceptedObjects, _, self.best_params = sampleFromDistribution(self.rle._game, \
                self.rle._game.spriteDistribution, allObjects, self.rle._game.spriteUpdateDict, self.bestSpriteTypeDict, skipInduction=self.skipInduction)
            self.rle._game.exceptedObjects = exceptedObjects
            gameObject = Game(spriteInductionResult=spriteTypeHypothesis)
            initialTheory = gameObject.buildGenericTheory(spriteTypeHypothesis)
            # embed()
        else:
            gameObject = Game(self.gameString)
            initialTheory = gameObject.buildGenericTheory(spriteSample=False, vgdlSpriteParse = gameObject.vgdlSpriteParse)

        # Handle wall vs. projectile interaction (hacky)
        avatar = [o for o in initialTheory.spriteSet if o.vgdlType in AvatarTypes][0]
        self.hypotheses = [initialTheory]
        self.symbolDict = generateSymbolDict(self.rle)
        return gameObject

    def completeHypotheses(self, allObjects, statesEncountered, compactStates, first_time_playing_level):
        previous_colors = [o['type']['color'] for o in self.previous_objects.values()]
        current_colors = [o['type']['color'] for o in allObjects.values()]
        if all([c in previous_colors for c in current_colors]):
            self.observe(self.rle, 0, self.bestSpriteTypeDict, statesEncountered, compactStates, display=self.display_states) ## observe a couple steps so that you're not completely clueless about object movements when you're restarting a level.
        else:
            self.observe(self.rle, 5, self.bestSpriteTypeDict, statesEncountered, compactStates, display=self.display_states) ## observe many steps so that you're not completely clueless about object movements for the new level

        ## Make sure any objects that appeared while we were observing are reflected in allObjects
        for k,v in self.rle._game.getObjects().items():
            if k not in allObjects:
                allObjects[k] = v

        spriteTypeHypothesis, exceptedObjects, _, self.best_params= sampleFromDistribution(self.rle._game, self.rle._game.spriteDistribution, allObjects, self.rle._game.spriteUpdateDict, 
                self.bestSpriteTypeDict, self.hypotheses[0].spriteSet, skipInduction=self.skipInduction)
        gameObject = Game(spriteInductionResult=spriteTypeHypothesis)
        newHypotheses = []
        for hypothesis in self.hypotheses:
            newHypotheses.append(gameObject.addNewObjectsToTheory(hypothesis, spriteTypeHypothesis))
        self.hypotheses = newHypotheses


    def playCurriculum(self, heatmap=False, level_game_pairs=None, make_movie=False):
        """ Plays a game level until it wins, then moves to the next one until
        completion. """
        starttime = time.time()
        if not level_game_pairs:
            level_game_pairs = importlib.import_module(self.gameFilename).level_game_pairs
        episodes = []
        allEffectsEncountered = []
        self.make_movie = make_movie

        ## used for time-stamping data related to this particular run of the model.
        timestamp = datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d__%H_%M')

        if self.record_states:
            dirname = "results/{}/{}/".format(self.param_ID, self.gameFilename)
            filename = "{}{}_{}".format(dirname, self.gameFilename, timestamp)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        if self.record_video_info:
            dirname = "raw_video_info/{}/{}/".format(self.param_ID, self.gameFilename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        if self.make_movie:
            if 'images' in os.listdir('.') and 'tmp' in os.listdir('images') and self.gameFilename in os.listdir('images/tmp'):
                shutil.rmtree("images/tmp/"+self.gameFilename)
            os.makedirs("images/tmp/"+self.gameFilename)

        j=0
        flexible_goals = False
        fullStateEpisodes, episodeCompactStates = {}, {}
        for n_level, level_game in enumerate(level_game_pairs):

            print("Playing level {}".format(n_level))
            (self.gameString, self.levelString) = level_game
            self.max_nodes = self.starting_max_nodes
            self.stored_max_nodes = self.max_nodes
            win = False
            gameObject = None
            i=0
            levelEffectsEncountered = []
            allStatesEncountered = []
            allCompactStates = []
            t1 = time.time()
            first_time_playing_level = True
            while not win and i<15:
                gameObject, win, score, steps, statesEncountered, effectsEncountered, compactStates = self.playEpisode(gameObject, flexible_goals, win, first_time_playing_level)
                
                self.total_game_steps += steps
                allCompactStates.append(compactStates)
                episode_results = (n_level, steps, win, score, self.total_planner_steps)
                episodes.append(episode_results)

                # write progressively to file
                # output = {'modelType':self.param_ID,
                #             'gameName': self.gameFilename,
                #             'condition': 'normal',
                #             'episodes' : [episode_results]}
                # # write_to_csv('hyperparameter_idx_'+str(self.hyperparameters['idx']), str(self.gameFilename)+'.csv', output)
                # write_to_csv('',str(self.gameFilename)+'.csv', output)

                if self.make_movie:
                    self.statesEncountered = statesEncountered
                    self.makeImages()
                
                if self.record_video_info:
                    allStatesEncountered.extend(statesEncountered)

                # if make_movie:
                    # allStatesEncountered.extend(statesEncountered)
                    # VGDLParser.playGame(self.gameString, self.levelString, statesEncountered,
                    # persist_movie=True, make_images=True, make_movie=False, movie_dir="videos/"+self.gameFilename, padding=10)
                # levelEffectsEncountered.append(effectsEncountered)
                
                # if self.total_game_steps > MAX_STEPS:
                    # return

                first_time_playing_level = False
                i += 1
                print "Finished in ", time.time() - t1

                episodeCompactStates[n_level] = allCompactStates
                fullStateEpisodes[n_level] = allStatesEncountered

                ## will write all previous episodes to the file at the end of each episode.
                if self.record_states:
                    gameInfo = {'gameString':self.gameString, 'levelString':self.levelString, 'gameName':self.gameFilename}
                    episodeList = [v for k,v in sorted(episodeCompactStates.items())]
                    print "n_level", n_level
                    print len(episodeList)
                    # embed()
                    with open(filename, 'wb') as f:
                        cPickle.dump({'gameInfo':gameInfo,'modelParams':self.param_ID, 'episodes':episodeList, 'time_elapsed':time.time()-starttime}, f)
                    f.close()

                ## will write video data at the end of each episode
                if self.record_video_info:
                    videofilename = "{}{}_{}".format(dirname, self.gameFilename, timestamp)
                    gameInfo = {'gameString':self.gameString, 'levelString':self.levelString, 'gameName':self.gameFilename}
                    fullStateList = [v for k,v in sorted(fullStateEpisodes.items())]
                    with open(videofilename, 'wb') as f:
                        cPickle.dump({'gameInfo':gameInfo,'modelParams':self.param_ID, 'episodes':fullStateList, 'time_elapsed':time.time()-starttime}, f)
                    f.close()

            # if i < 10:
                # self.levels_won += 1

            # if heatmap:
            #     self.makeHeatmap(allStatesEncountered, '{}_{}_level{}_heatmap.pdf'.format(
            #         # self.gameFilename[self.gameFilename.find('expt'):],
            #         gvgname[gvgname.find('set_1/')+6:],
            #         self.modelType, n_level))

            # allEffectsEncountered.append(levelEffectsEncountered)

            ## Uncomment if you want to run flexible goals version.
            # j+=1
            # if j>0:
            #     flexible_goals=True

            if flexible_goals:
                ## When you embed, you can manually input changes in theory. See flexible_goals.py for an example.
                print "in main_agent; playing with flexible_goals"
                embed()

        endtime = time.time()
        ## put timestamp on filenames
        # if self.record_states:
        #     dirname = "results/{}/{}/".format(self.param_ID, self.gameFilename)
        #     filename = "{}{}_{}".format(dirname, self.gameFilename, timestamp)
        #     if not os.path.exists(dirname):
        #         os.makedirs(dirname)
        #     gameInfo = {'gameString':self.gameString, 'levelString':self.levelString, 'gameName':self.gameFilename}
        #     with open(filename, 'wb') as f:
        #         cPickle.dump({'gameInfo':gameInfo,'modelParams':self.param_ID, 'episodes':episodeCompactStates, 'time_elapsed':endtime-starttime}, f)
        #     f.close()

        # if self.record_video_info:
        #     dirname = "raw_video_info/{}/{}/".format(self.param_ID, self.gameFilename)
        #     if not os.path.exists(dirname):
        #         os.makedirs(dirname)
        #     filename = "{}{}_{}".format(dirname, self.gameFilename, timestamp)
        #     gameInfo = {'gameString':self.gameString, 'levelString':self.levelString, 'gameName':self.gameFilename}
        #     with open(filename, 'wb') as f:
        #         cPickle.dump({'gameInfo':gameInfo,'modelParams':self.param_ID, 'episodes':fullStateEpisodes, 'time_elapsed':endtime-starttime}, f)
        #     f.close()

        # if self.make_movie:
            # self.makeMovie()

    def compactify(self, rle, planner_nodes=0):
        gameObject = rle._game
        ended, win = rle._isDone()
        state = {'timestep': gameObject.time,
                 'score': gameObject.score,
                 'planner_settings': self.hyperparameter_index,
                 'planner_nodes': planner_nodes, ## how many nodes were searched to determine this particular action? 0 if this is resulting from a cached plan.
                 'ended': ended,
                 'win': win,
                 'objects': [(colorDict[str(s.color)], (s.rect.left/gameObject.block_size, s.rect.top/gameObject.block_size), s.resources if s.name=='avatar' else {}) 
                        for sublist in gameObject.sprite_groups.values() for s in sublist if s not in gameObject.kill_list]
                 }
        return state

    def makeHeatmap(self, statesEncountered, filename):
        from vgdl.plotting import featurePlot
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullLocator
        import numpy as np

        states = [s['objects']['avatar'].keys()[0] for s in statesEncountered
                  if s['objects']['avatar'].keys()]
        width, height = self.rle._game.width, self.rle._game.height
        correction_factor = self.rle._game.screensize[0]/width
        corrected_states = [(s[0]/correction_factor, s[1]/correction_factor) for s in states]

        m = np.zeros((width, height))
        Xs, Ys = [],[]
        im = plt.imread('flexible_goals.png')
        implot = plt.imshow(im)
        w, h = implot.get_extent()[1], implot.get_extent()[2]
        block_size = w/width

        for s in corrected_states:
            x = s[0]
            y = s[1]
            m[x, y] += 1
            Xs.append(x*block_size+block_size/2.)
            Ys.append(y*block_size+block_size/2.)
        plt.scatter(x=Xs, y=Ys, alpha=.5, edgecolor='')
        # plt.imshow(m.T, cmap='viridis')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def makeSummaryPlot(self, allEffectsEncountered):
        import matplotlib.pyplot as plt
        import importlib

        mod = importlib('vgdl.colors')
        colors = [cl[0].color for cl in self.hypotheses[0].classes.values()]
        for color in colors:
            times_touched_per_level = []
            for level in allEffectsEncountered:
                times_touched = len([effect
                    for attempt in level
                    for effect in attempt
                    if ((effect[1]==color and effect[2]=='DARKBLUE')
                        or (effect[2]==color and effect[1]=='DARKBLUE'))])
                times_touched_per_level.append(times_touched)
            color_to_plot = [float(value)/255 for value in getattr(mod, color)]
            plt.plot(times_touched_per_level, color=color_to_plot)
        plt.show()


    def makeImages(self):
        VGDLParser.playGame(self.gameString, self.levelString, self.statesEncountered, \
            persist_movie=True, make_images=True, make_movie=False, movie_dir="videos/"+self.gameFilename, gameName = self.gameFilename, parameter_string=self.param_ID, padding=10)

    def makeMovie(self):

        print "Creating Movie"
        movie_dir = "videos/{}/{}".format(self.param_ID, self.gameFilename)

        if not os.path.exists(movie_dir):
            print movie_dir, "didn't exist. making new dir"
            os.makedirs(movie_dir)
        round_index = len([d for d in os.listdir(movie_dir) if d != '.DS_Store'])
        video_dirname = movie_dir+"/round"+str(round_index)+".mp4"
        images_dir = "images/tmp/{}/%09d.png".format(self.gameFilename)
        com = "ffmpeg -i " +images_dir+ " -pix_fmt yuv420p -filter:v 'setpts=4.0*PTS' "+ video_dirname
        command = "{}".format(com)
        subprocess.call(command, shell=True)
        # empty image directory
        shutil.rmtree("images/tmp/"+self.gameFilename)
        os.makedirs("images/tmp/"+self.gameFilename)
        return

    def playMultipleEpisodes(self, num_episodes):
        i=0
        gameObject = None
        wins, scores = [], []
        win = False
        while i<num_episodes:
            gameObject, win, score, statesEncountered, _ = self.playEpisode(gameObject, flexible_goals=False,first_time_playing_level=False)
            wins.append(win)
            scores.append(score)
            i+=1
        VGDLParser.playGame(self.gameString, self.levelString, self.statesEncountered, \
            persist_movie=True, make_images=True, make_movie=False, movie_dir="videos/"+self.gameFilename, padding=10)
        print "Won {} out of {} episodes.".format(sum(wins), i)

        """
        def playEpisodeProfiler(self, gameObject, flexible_goals=False, first_time_playing_level=False):
            lp = LineProfiler()
            lp_wrapper = lp(self.playEpisode)
            gameObject, win, score, steps, statesEncountered, effectsEncountered = lp_wrapper(gameObject, flexible_goals, first_time_playing_level)
            lp.print_stats()
            return gameObject, win, score, steps, statesEncountered, effectsEncountered
        """

    def playEpisode(self, gameObject, flexible_goals=False, win=False, first_time_playing_level=False, pool=None):
        from vgdl.util import manhattanDist

        ## Initialize external environment
        self.initializeEnvironment()
        if self.display_text:
            print "initializing RLE"
        print "Game name:", self.gameFilename
        print self.rle.show(color='blue')

        self.quits = 0
        self.longHorizonObservations = 0
        self.previous_objects = self.all_objects if self.all_objects else {}
        self.all_objects= self.rle._game.getObjects()
        annealing = 1
        ## Start storing encountered states.
        effectsEncountered = []
        statesEncountered = []
        compactStates = [] ## for analysis

        # self.rleHistory.append(copy.deepcopy(self.rle._game))
        # self.statesEncountered.append(self.rle._game.getFullState())
        if self.make_movie or self.record_video_info:
            statesEncountered.append(self.rle._game.getFullState())
        
        if self.record_states:
            compactStates.append(self.compactify(self.rle))
        ## Initialize memory of object positions
        self.rle._game.objectMemoryDict, self.rle._game.previousPositions = {}, {}
        for k, v in self.rle._game.all_objects.iteritems():
            self.rle._game.objectMemoryDict[k] = (int(self.rle._game.all_objects[k]['sprite'].rect.x), int(self.rle._game.all_objects[k]['sprite'].rect.y))
            self.rle._game.previousPositions[k] = (int(self.rle._game.all_objects[k]['sprite'].rect.x), int(self.rle._game.all_objects[k]['sprite'].rect.y))

        ## initialize theory if necessary.
        if len(self.hypotheses) == 0:
            gameObject = self.initializeHypotheses(self.all_objects, statesEncountered, compactStates, learnSprites=True)
            if self.display_text:
                print "initializing hypotheses"
        else:
            gameObject = self.completeHypotheses(self.all_objects, statesEncountered, compactStates, first_time_playing_level)
            if self.display_text:
                print "had hypotheses -- completing them."
            # If theory is being carried over, falsify termination hypotheses
            # given new level state
            if not flexible_goals:
                [t.updateTerminations(rle=self.rle) for t in self.hypotheses]


        ## Do beginning-of-episode resource-management.
        resources = self.rle._game.getAvatars()[0].resources
        for resource, val in resources.items():
            if resource not in self.seen_resources and val>0:
                self.seen_resources.append(resource)
                self.hypotheses[0].resource_limits[resource] = self.rle._game.resources_limits[resource]
            if resource not in self.seen_limits and val==self.rle._game.resources_limits[resource]:
                self.seen_limits.append(resource)

        ended, win = self.rle._isDone()
        # if ended and win:
        #     print "ended and won 0"
        #     embed()
        steps = self.rle._game.time
        emptyPlans = 0
        while not ended:

            self.max_nodes = self.stored_max_nodes

            # if self.display_text:
            print "planning with hyperparameter index {}".format(self.hyperparameter_index)
            print "max_nodes: {}, short_horizon: {}".format(self.max_nodes, self.shortHorizon)

            ## initialize one or many VRLEs according to hypothesis-selection method
            theoryRLEs = self.VrleInitPhase(flexible_goals)

            quitting = False

            planner_hyperparameters = dict((k, self.hyperparameters[k]) for k in self.hyperparameters.keys() if k not in ['short_horizon', 'first_order_horizon'])

            ## also, you commented out the bottom part of the planner, where it will still return a high-reward sequence in shortHorizon. This could have a very detrimental effect on short-horizon games...

            ## Initialize planner
            p = WBP.WBP(theoryRLEs[0], self.gameFilename, theory=self.hypotheses[0], fakeInteractionRules = self.fakeInteractionRules,
                seen_limits = self.seen_limits, annealing=annealing, max_nodes=self.max_nodes, shortHorizon=self.shortHorizon,
                firstOrderHorizon=self.firstOrderHorizon, conservative=self.conservative, hyperparameters=planner_hyperparameters, extra_atom=self.extra_atom, IW_k=self.IW_k)
            p_quitting = p.quitting
            bestNode, gameStringArray, objectPositionsArray = p.BFS()
            self.total_planner_steps += p.total_nodes

            if bestNode is not None:
                solution = p.solution
                gameString_array = p.gameString_array
                objectPositionsArray = objectPositionsArray[::-1]
                if solution and self.display_text:
                    print "got solution"
            else:
                solution = []

            if not solution:
                ## If we're repeatedly dying in the same way, just switch hyperparameters blindly.
                if self.checkForRepeatedDeaths(self.episodeRecord, 2):
                    if self.hyperparameter_index == 1:
                        if self.display_text:
                            print "Repeated deaths. Switching to short-range planning"
                        new_index = 3 
                        conservative = False
                    elif self.hyperparameter_index == 3:
                        if self.display_text:
                            print "Repeated deaths. Switching to long-range planning"
                        new_index = 1
                        conservative = False
                    planner_hyperparameters = self.hyperparameterSwitch(new_index=new_index)

                elif self.hyperparameter_index == 3:
                    movingTypes = self.checkForMovingTypes(self.rle, self.hypotheses[0])
                    if self.rle._game.time>compactStates[-1]['timestep']:
                        scoreChange = self.rle._game.score!=compactStates[-1]['score']
                    else:
                        scoreChange = True
                    # if self.display_text:
                    print "moving types: {}".format(movingTypes)
                    print "noNewObjectsInAWhile: {}".format(self.noNewObjectsInAWhile(self.rle, 55))
                    print "scoreChange: {}".format(scoreChange)
                    # print "self.max_game_time_observed>501: {}".format(self.max_game_time_observed>501)
                    if self.noNewObjectsInAWhile(self.rle, 55) and \
                            (not movingTypes or (movingTypes and not scoreChange)):
                            # (not movingTypes or (movingTypes and self.max_game_time_observed>501)):
                        # if self.display_text:
                        print "switching to long-range planning"
                        ## switch to long-range planning
                        new_index = 1
                        planner_hyperparameters = self.hyperparameterSwitch(new_index=new_index)
                        conservative = False
                        # embed()
                    else:
                        # if self.display_text:
                        print "planning conservatively"
                        new_index = 3
                        planner_hyperparameters = self.hyperparameterSwitch(new_index=new_index)
                        conservative = True
                        self.stored_max_nodes = self.max_nodes ##taking annealing into account
                        self.max_nodes = 50
                else:
                    conservative = False

                if self.display_text:
                    print "planning with hyperparameter index {}".format(self.hyperparameter_index)
                    print "max_nodes: {}, short_horizon: {}, conservative: {}".format(self.max_nodes, self.shortHorizon, conservative)
                # embed()

                if conservative:
                    ## Replan in new mode
                    p = WBP.WBP(theoryRLEs[0], self.gameFilename, theory=self.hypotheses[0], fakeInteractionRules = self.fakeInteractionRules,
                        seen_limits = self.seen_limits, annealing=annealing, max_nodes=self.max_nodes, shortHorizon=self.shortHorizon,
                        firstOrderHorizon=self.firstOrderHorizon, conservative=conservative, hyperparameters=planner_hyperparameters, extra_atom=self.extra_atom, IW_k=self.IW_k)
                    p_quitting = p.quitting
                    bestNode, gameStringArray, objectPositionsArray = p.BFS()
                    self.total_planner_steps += p.total_nodes
                    if bestNode is not None:
                        solution = p.solution
                        gameString_array = p.gameString_array
                        objectPositionsArray = objectPositionsArray[::-1]
                        if solution and self.display_text:
                            print "got solution"
                    else:
                        solution = []

            if (not solution) or p_quitting:
                # Here we make a distinction between quitting because you've
                # exhausted the number of nodes you can visit or because you
                # ran out of novelty. In the first case, you only wait longer,
                # in the second case, you also add a new atom to IW
                if p.exhausted_novelty and self.extra_atom_allowed:
                    print "turning on extra atom"
                    self.extra_atom = True
                if self.longHorizonObservations<self.longHorizonObservationLimit:
                    if self.display_text:
                        print "Didn't get solution. Observing, then replanning."
                    self.observe(self.rle, 5, self.bestSpriteTypeDict, statesEncountered, compactStates)
                    solution = [] ## You may have gotten p.quitting but also a solution; make sure you don't try to act on that if the planner decided it wasn't worth it.
                    self.longHorizonObservations += 1
                else:
                    quitting = True
            # if K_SPACE in solution:
                # embed()

            
            self.actionSeqLength += len(solution)

            if solution and not p.quitting and self.display_states:
                print "============================================="
                print "got solution of length", len(solution)
                print colored(p.gameString_array[0], 'green')
                for i,g in enumerate(p.gameString_array[1:]):
                    print actionDict[solution[i]]
                    print colored(g, 'green')
                print "============================================="

            ##new 6/30/18
            # if emptyPlans > self.emptyPlansLimit:
            #     self.max_nodes *= self.max_nodes_annealing
            #     print "reached emptyPlansLimit of {}. Annealing max nodes to {}".format(self.emptyPlansLimit, self.max_nodes)

            # if emptyPlans > self.emptyPlansLimit:
                # print "got too many empty plans"
                # observe(self.rle, 5, self.bestSpriteTypeDict)

            if not quitting:
                for i, action in enumerate(solution):
                    self.hypotheses[0].dryingPaint = set()
                    # if action==K_SPACE:
                        # print "about to take a shot"
                        # embed()
                    # envPrev = copy.deepcopy(self.rle)
                    if self.display_text:
                        t1 = time.time()
                    effects = []
                    plannerNodes = p.total_nodes if i==0 else 0
                    hypotheses, theory_change_flag, effects = self.executeStep(action, self.hypotheses, statesEncountered, compactStates, plannerNodes,
                        run_induction = not flexible_goals)
                    
                    if self.display_text:
                        print "executeStep took {} seconds".format(time.time()-t1)
                    sys.stdout.flush()
                    
                    self.rle._game.nextPositions = {}
                    for k, v in self.rle._game.all_objects.iteritems():
                        self.rle._game.nextPositions[k] = (int(self.rle._game.all_objects[k]['sprite'].rect.x), int(self.rle._game.all_objects[k]['sprite'].rect.y))
                        try:
                            if self.rle._game.previousPositions[k] != self.rle._game.nextPositions[k]:
                                self.rle._game.objectMemoryDict[k] = copy.deepcopy(self.rle._game.previousPositions[k])
                        except KeyError:
                            pass
                    self.rle._game.previousPositions = copy.deepcopy(self.rle._game.nextPositions)

                    effectsEncountered.extend(effects)
                    steps +=1
                    if theory_change_flag:
                        self.hypotheses = hypotheses
                        break
                    ended, win = self.rle._isDone()
                    # if ended and win:
                    #     print "ended and won 1"
                    #     embed()

                    self.max_game_time_observed = max(self.max_game_time_observed, self.rle._game.time)
                    if ended:
                        # print "episode ended"
                        # embed()
                        break
                    # if self.total_game_steps > MAX_STEPS:
                        # score = self.rle._game.score
                        # return gameObject, win, score, steps, statesEncountered, effectsEncountered
                    # if self.rle._game.time>13:
                        # embed()

                    ## Make sure you're far enough from unpredictable dangerous objects.
                    # Check for disparities between plan and reality
                    # (e.g. stochastic effects)
                    if (i+1)%self.regrounding==0:

                        if self.checkForDangerOrAvatarMisLocation(self.rle, hypotheses[0], objectPositionsArray, i):
                            break

                    if self.reground_for_npcs: ## this is just exercising caution when near random objects, irrespective of whether they kill us or not
                        try:
                            random_npc_colors = [self.hypotheses[0].classes[k][0].color for k in self.hypotheses[0].classes.keys() if self.hypotheses[0].classes[k] and 'Random' in str(self.hypotheses[0].classes[k][0].vgdlType)]
                            random_npc_classes = [k for k in self.rle._game.sprite_groups.keys() if self.rle._game.sprite_groups[k] and self.rle._game.sprite_groups[k][0].colorName in random_npc_colors]
                            random_npc_positions = []

                            for c in random_npc_classes:
                                for element in self.rle._game.sprite_groups[c]:
                                    if element not in self.rle._game.kill_list:
                                        random_npc_positions.append(self.rle._rect2pos(element.rect))

                            avatar_positions = [self.rle._rect2pos(avatar.rect)
                                 for avatar in self.rle._game.getAvatars()]

                            possiblePairList = [manhattanDist(avatar, random)
                                for avatar in avatar_positions
                                for random in random_npc_positions]
                            if min(possiblePairList) < self.safeDistance:
                                print("Close to RandomNPC, regrounding")
                                # embed()
                                break

                        except ValueError:
                            # print("error in avoid_danger: is the avatar dead?")
                            pass
            else:
                ## You failed the game either because you made a mistake you couldn't recover from or because you timed out in your search.
                ## Search more deeply next time.
                self.max_nodes *= self.max_nodes_annealing
                self.stored_max_nodes = self.max_nodes
                win, effects = False, []
                self.episodeRecord.insert(0, (win, effects))
                # self.updateMemory(self.rle)
                print colored('________________________________________________________________', 'white', 'on_red')
                print colored("Quitting", 'white', 'on_red')
                print colored('________________________________________________________________', 'white', 'on_red')
                return gameObject, False, self.rle._game.score, steps, statesEncountered, effectsEncountered, compactStates


            annealing *= self.annealingFactor
            ended, win = self.rle._isDone()
            # if ended and win:
            #     print "ended and won 2"
            #     embed()
            
            if ended:
                self.episodeRecord.insert(0, (win, effects))
            
            if ended and not win and self.rle._game.time==2000:
                print "lost on timeout. switching hyperparameters"
                self.hyperparameterSwitch(new_index=1)


        ## Update global memory of updates
        # for k in game.spriteUpdateDict:
            # self.spriteUpdateDict[k] = game.spriteUpdateDict[k]

        score = self.rle._game.score

        output =          "ended episode. Win={}                                           ".format(win)
        if win:
            print colored('________________________________________________________________', 'white', 'on_green')
            print colored('________________________________________________________________', 'white', 'on_green')

            print colored(output, 'white', 'on_green')
            print colored('________________________________________________________________', 'white', 'on_green')
        else:
            print colored('________________________________________________________________', 'white', 'on_red')
            print colored(output, 'white', 'on_red')
            print colored('________________________________________________________________', 'white', 'on_red')


        return gameObject, win, score, steps, statesEncountered, effectsEncountered, compactStates

    def checkForRepeatedDeaths(self, episodeRecord, cutoff):
        count = 1
        for i in range(1, len(episodeRecord)):
            if episodeRecord[i][0]==False and episodeRecord[i][1]==episodeRecord[i-1][1]:
                count+=1
            else:
                break
        if count>cutoff:
            return True
        else:
            return False

    def noNewObjectsInAWhile(self, rle, age_cutoff):
        if self.hypotheses[0].classes['avatar'][0].args and 'stype' in self.hypotheses[0].classes['avatar'][0].args:
            thingWeShoot = self.hypotheses[0].classes['avatar'][0].args['stype']
        else:
            thingWeShoot = None         
        
        min_age = min([item.lastmove for sublist in self.rle._game.sprite_groups.values() for item in sublist if (item not in self.rle._game.kill_list and item.name not in [thingWeShoot, 'avatar'])])

        try:
            time_since_last_kill = self.rle._game.time - max([item.deathage for item in self.rle._game.kill_list if item.name!=thingWeShoot])
        except:
            time_since_last_kill = self.rle._game.time

        if (min_age > age_cutoff) and (time_since_last_kill > age_cutoff):
            return True
        else:
            return False

    def checkForMovingTypes(self, rle, hypothesis):
        if self.hypotheses[0].classes['avatar'][0].args and 'stype' in self.hypotheses[0].classes['avatar'][0].args:
            thingWeShoot = self.hypotheses[0].classes['avatar'][0].args['stype']
        else:
            thingWeShoot = None    
        moving_types = [k for k in hypothesis.classes.keys() if k!=thingWeShoot and any([t in str(hypothesis.classes[k][0].vgdlType) for t in ['Missile', 'Random', 'Chaser']])]
        moving_colors = [hypothesis.classes[k][0].color for k in moving_types]
        movingTypes = False
        if moving_colors:
            for s in [item for sublist in self.rle._game.sprite_groups.values() for item in sublist if item not in self.rle._game.kill_list]:
                if s.colorName in moving_colors:
                    movingTypes = True
                    break
        return movingTypes

    def checkForMovingKillerTypes(self, rle, hypothesis):
        killer_types = [inter.slot2 for inter in hypothesis.interactionSet if inter.slot1=='avatar' and inter.interaction in ['killSprite']]
        moving_killer_types = [k for k in killer_types if any([t in str(hypothesis.classes[k][0].vgdlType) for t in ['Missile', 'Random', 'Chaser']])]
        killer_colors = [hypothesis.classes[k][0].color for k in moving_killer_types]
        danger = False
        if killer_colors:
            for s in [item for sublist in self.rle._game.sprite_groups.values() for item in sublist if item not in self.rle._game.kill_list]:
                if s.colorName in killer_colors:
                    danger = True
                    break
        if danger and random.random()>.5:
            return True
        else:
            return False

    def checkForDangerOrAvatarMisLocation(self, rle, hypothesis, objectPositionsArray, i):
        regroundingFlag = False

        rleDict, hypDict = {}, {}

        for s in [item for sublist in objectPositionsArray[i+1]._game.sprite_groups.values() for item in sublist if item not in objectPositionsArray[i+1]._game.kill_list]:
            hypDict[s.ID2] = s

        killer_types = [inter.slot2 for inter in hypothesis.interactionSet if inter.slot1=='avatar' and inter.interaction in ['killSprite']]
        killer_colors = [hypothesis.classes[k][0].color for k in killer_types]

        for s in [item for sublist in self.rle._game.sprite_groups.values() for item in sublist if item not in self.rle._game.kill_list]:
            ## If the object isn't in our predicted environment or the positions vary
            ## if it's an object we're worried about
            if s.name=='avatar' or s.colorName in killer_colors:
                if s.ID not in hypDict and manhattanDist(s.rect, self.rle._game.getAvatars()[0].rect) < self.safeDistance*s.rect.width:

                    regroundingFlag=True
                    print colored("Regrounding because we didn't predict the appearance of {} and it's too close for comfort".format(s), 'white', 'on_yellow')
                    break
                if s.ID in hypDict and s.rect!=hypDict[s.ID].rect and manhattanDist(s.rect, self.rle._game.getAvatars()[0].rect) < self.safeDistance*s.rect.width:
                    print colored("Regrounding because distance between {} and {} is {}, which is less than the safe distance of {}. We thought it would be at {}".format(
                            s, self.rle._game.getAvatars()[0], manhattanDist(s.rect, self.rle._game.getAvatars()[0].rect), self.safeDistance*s.rect.width, hypDict[s.ID]),
                            'white', 'on_yellow')
                    regroundingFlag=True
                    break
                rleDict[s.ID] = s
        return regroundingFlag

    def matchEventToRuleByIDAndSpriteName(self, event, rule):
        # Check if the two objects involved in the
        # event are the same as those in the novelty
        # termination rule (invariant by order)

        if event[1] not in self.hypotheses[0].spriteObjects:
            self.hypotheses[0].addSpriteToTheory(event[1])
        if event[2] not in self.hypotheses[0].spriteObjects:
            self.hypotheses[0].addSpriteToTheory(event[2])
    
        try:
            hypSlot1 = self.hypotheses[0].spriteObjects[event[1]].className
            hypSlot2 = self.hypotheses[0].spriteObjects[event[2]].className
        except:
            print "hypslot problem in main agent"
            embed()


        if set([hypSlot1, hypSlot2]) == set([rule.slot1, rule.slot2]):
            if not rule.preconditions:
                return True
            else:
                if not all([p.check(self.rle.agentStatePrev) for p in list(rule.preconditions)]):
                    return False
                else:
                    return True
        else:
            return False

    def manageNewObjects(self, hypotheses):
        ## Add newly-seen objects.
        current_objects = self.rle._game.getObjects()
        for k in current_objects.keys():
            spriteName = current_objects[k]['sprite'].name
            if spriteName not in [self.all_objects[key]['sprite'].name for key in self.all_objects.keys()]:
                if self.display_text:
                    print "new object", spriteName
                self.all_objects[k] = current_objects[k]
                distributionInitSetup(self.rle._game, k)
                ## prevent spriteInduction from trying to infer anything about newly-appeared sprites in this timestep.
                self.rle._game.ignoreList.append(k)
                self.new_objects[spriteName] = 0


            # if k not in theory.spriteObjects.keys():
            #     color = k
            #     existing_classes = [key for key in theory.classes if key[0] == 'c']
            #     max_num = max([int(c[1:]) for c in existing_classes])
            #     class_num = max_num+1
            #     newClassName = 'c'+str(class_num)
            #     theory.addSpriteToTheory(newClassName, color, vgdlType=Resource, args={'limit':errorMap.targetToken.inventory[k][1]})

        # for k in self.new_objects.keys():
        #     self.new_objects[k] += 1

        # if any([self.new_objects[k]>5 for k in self.new_objects.keys()]):
        #     # if self.new_objects[k] > 5:
        #     spriteTypeHypothesis, exceptedObjects, _, self.best_params = sampleFromDistribution(self.rle._game, self.rle._game.spriteDistribution, self.all_objects, self.rle._game.spriteUpdateDict, self.bestSpriteTypeDict, self.hypotheses[0].spriteSet)
        #     gameObject = Game(spriteInductionResult=spriteTypeHypothesis)

        #     newHypotheses = []
        #     for hypothesis in hypotheses:
        #         newHypotheses.append(gameObject.addNewObjectsToTheory(hypothesis, spriteTypeHypothesis))
        #     hypotheses = newHypotheses

        # [self.new_objects.pop(k, None) for k in self.new_objects.keys() if self.new_objects[k]>5] ## don't track items once we've updated the theory
        return hypotheses

    """
    def executeStepProfiler(self, action, hypotheses, statesEncountered, run_induction=True):
        lp = LineProfiler()
        lp_wrapper = lp(self.executeStep)
        hypotheses, theory_change_flag, effects = lp_wrapper(action, hypotheses, statesEncountered, run_induction)
        lp.print_stats()
        return hypotheses, theory_change_flag, effects
    """
    def observe(self, rle, obsSteps, bestSpriteTypeDict, statesEncountered, compactStates, display=False):
        if display:
            print "observing for {} steps".format(obsSteps)
        if obsSteps>0:
            for i in range(obsSteps):
                spriteInduction(rle._game, step=1, bestSpriteTypeDict=bestSpriteTypeDict)
                spriteInduction(rle._game, step=2, bestSpriteTypeDict=bestSpriteTypeDict)
                rle.step((0,0))
                if self.make_movie or self.record_video_info:
                    statesEncountered.append(self.rle._game.getFullState())
                if self.record_states:
                    compactStates.append(self.compactify(self.rle))
                if display:
                    print "score: {}, game tick: {}".format(rle._game.score, rle._game.time)
                    print rle.show(color='blue')

                rle._game.nextPositions = {}
                for k, v in rle._game.all_objects.iteritems():
                    rle._game.nextPositions[k] = (int(rle._game.all_objects[k]['sprite'].rect.x), int(rle._game.all_objects[k]['sprite'].rect.y))
                    try:
                        if rle._game.previousPositions[k] != rle._game.nextPositions[k]:
                            rle._game.objectMemoryDict[k] = copy.deepcopy(rle._game.previousPositions[k])
                    except KeyError:
                        pass
                rle._game.previousPositions = copy.deepcopy(rle._game.nextPositions)
                spriteInduction(rle._game, step=3, bestSpriteTypeDict=bestSpriteTypeDict)
        else:
            spriteInduction(rle._game, step=1, bestSpriteTypeDict=bestSpriteTypeDict)
            spriteInduction(rle._game, step=2, bestSpriteTypeDict=bestSpriteTypeDict)
        return

    def executeStep(self, action, hypotheses, statesEncountered, compactStates, plannerNodes, run_induction=True):

        theory_change_flag = False

        if not self.skipInduction:
            # t1 = time.time()
            spriteInduction(self.rle._game, step=1, bestSpriteTypeDict=self.bestSpriteTypeDict, oldSpriteSet=hypotheses[0].spriteSet)
            # print "induction step 1 took {} seconds.".format(time.time()-t1)
            t1 = time.time()
            spriteInduction(self.rle._game, step=2, bestSpriteTypeDict=self.bestSpriteTypeDict, oldSpriteSet=hypotheses[0].spriteSet)
            # print "induction step 2 took {} seconds".format(time.time()-t1)


        try:
            agentState = copy.deepcopy(self.rle._game.getAvatars()[0].resources)
            # agentState = ccopy(self.rle._game.getAvatars()[0].resources)

        except IndexError:
            agentState = defaultdict(lambda: 0)

        lastScore = self.rle._game.score
        res = self.rle.step(action)

        try:
            agentState = copy.deepcopy(self.rle._game.getAvatars()[0].resources)
            # print agentState

            for e in res['effectList']:
                if 'changeResource' in e:
                    changes = e[3]
                    if changes['value'] < 0:
                        # undo one negative change to account for eventhandler ordering
                        agentState[changes['resource']] -= changes['value']
                        break

            self.rle.agentStatePrev = agentState

        # If agent is killed before we get agentState
        except (IndexError, AttributeError) as e:
            ignored_negative_change = False
            for e in res['effectList']:
                if 'changeResource' in e:
                    changes = e[3]
                    if changes['value'] > 0 or ignored_negative_change:
                        agentState[changes['resource']] += changes['value']
                    else:
                        agentState[changes['resource']] += 0
                        ignored_negative_change = True
            self.rle.agentStatePrev = agentState
        for k,v in agentState.items():
            agentState[k] = max(0, v)
        # print "post-step understanding of pre-step agentState (passed to induction): {}".format(agentState)
        # print "agentState stuff: {}".format(time.time()-t1)
        # embed()

        t1 = time.time()
        hypotheses = self.manageNewObjects(hypotheses)

        if self.make_movie or self.record_video_info:
            statesEncountered.append(self.rle._game.getFullState())
        if self.record_states:
            compactStates.append(self.compactify(self.rle, plannerNodes))

        # print "manage new objects and getFullState: {}".format(time.time()-t1)

        t1 = time.time()
        if not self.skipInduction:
            distributionsHaveChanged = spriteInduction(self.rle._game, step=3, bestSpriteTypeDict=self.bestSpriteTypeDict, oldSpriteSet=hypotheses[0].spriteSet)
        else:
            distributionsHaveChanged = False
        # print "sprite induction step 3: {}".format(time.time()-t1)
 
        # effects = translateEvents(res['effectList'], self.all_objects, self.rle)
        effects = self.rle._game.effectListByColor
        # if effects:
        #     print effects
        #     print alternateEffects
        #     embed()

        if self.display_states:
            print "score: {}, game tick: {}".format(self.rle._game.score, self.rle._game.time)
        
        # t1 = time.time()
        if self.display_states:
            print ""
            print keyPresses[action]
            print self.rle.show(color='blue')
        # print "rle.show: {}".format(time.time()-t1)
        
        # event = {'agentState': agentState, 'agentAction': action, 'effectList': effects, \
        #     'gameState': self.rle._game.getFullStateColorized(), 'rle': self.rle}
        event = {'agentState': agentState, 'agentAction': action, 'effectList': effects, \
            'gameState': None, 'rle': self.rle}
        # if len(effects)>4:
            # embed()
        # t1 = time.time()
        newEffects = False
        # print "{} effects in this time-step".format(len(effects))
        # print "finalEffectList:"
        # print self.finalEffectList
        if effects:
            if self.display_text:
                print effects
            # #  PRECONDITIONS HANDLING
            # # Current assumptions:
            # # - Only one resource can change for each timestep
            # # - The first time a resource changes, it goes from 0 to a positive
            # #   value
            for change_resource_effect in [e[3] for e in event['effectList'] if ('changeResource' in e)] + [e[3] for e in event['effectList'] if ('collectResource' in e)]:
                resource = change_resource_effect['resource']
                val = change_resource_effect['value']
                limit = change_resource_effect['limit']
                if (resource not in self.seen_resources and val>0):
                    self.fakeInteractionRules.extend(hypotheses[0].updateInteractionsPreconditions(resource))
                    self.fakeInteractionRules = list(set(self.fakeInteractionRules))
                    self.seen_resources.append(resource)
                    hypotheses[0].resource_limits[resource] = limit
                    theory_change_flag = True
                    newEffects = True
                    self.finalEffectList = set()

                if agentState[resource]>=limit and resource not in self.seen_limits:
                    self.fakeInteractionRules.extend(hypotheses[0].updateInteractionsPreconditions(resource, limit))
                    self.fakeInteractionRules = list(set(self.fakeInteractionRules))
                    self.seen_limits.append(resource)

                    theory_change_flag = True
                    newEffects = True
                    self.finalEffectList = set()

            self.finalEventList.append(event)
            newTimeStep = TimeStep(event['agentAction'], event['agentState'], event['effectList'], event['gameState'], event['rle'])
            self.finalTimeStepList.append(newTimeStep)
            for e in effects:
                compactEvent = (e[0], e[1], e[2])
                if compactEvent not in self.finalEffectList:
                    self.finalEffectList.add(compactEvent)
                    if self.display_text:
                        print "New event: {}".format(compactEvent)
                    newEffects = True
            # newEffects = True
        
        self.fakeInteractionRules = [r for r in self.fakeInteractionRules if
            not any([self.matchEventToRuleByIDAndSpriteName(e, r) for e in event['effectList']])]

        # print "finalEffectList length: {}".format(len(self.finalEffectList))
        # print "set prep took {} seconds".format(time.time()-t1)

        ## For games with moving objects you should do a quick-and-dirty evaluation of whether to change theories.
        ## For the games where we're the only ones to cause effects, we can afford to do the full thing.
        # if not any([t in str(s.vgdlType) for s in self.hypotheses[0].spriteObjects.values() for t in ['Random', 'Missile', 'Chaser']]):
            # newEffects = len(event['effectList'])
        # if (event['effectList'] and run_induction) or distributionsHaveChanged:
        # distributionsHaveChanged = False

        if ((newEffects or (random.random()<.2 and len(self.finalTimeStepList)<300)) and run_induction) or distributionsHaveChanged:
            # print "event", (not all([e in all_effects for e in effects])), "distributions changed", distributionsHaveChanged
            if self.display_text:
                print "new event", newEffects, "distributions changed", distributionsHaveChanged

            ## Delete fake interaction rules for events that were witnessed in this time step.
            # oldFakeInteractionRules = copy.deepcopy(self.fakeInteractionRules)

            # self.fakeInteractionRules = [r for r in self.fakeInteractionRules if
            #     not any([self.matchEventToRuleByIDAndSpriteName(e, r) for e in event['effectList']])]

            # if (not all([e in all_effects for e in effects])) or distributionsHaveChanged:
            if newEffects or distributionsHaveChanged:
                theory_change_flag = True

            t1 = time.time()
            sample, exceptedObjects, _, self.best_params= sampleFromDistribution(self.rle._game, self.rle._game.spriteDistribution, self.all_objects, 
                    self.rle._game.spriteUpdateDict, self.bestSpriteTypeDict, self.hypotheses[0].spriteSet, skipInduction=self.skipInduction, display=self.display_text)

            game_object = Game(spriteInductionResult=sample)
            # print "sampleFromDistribution: {}".format(time.time()-t1)
            
            terminationCondition = {'ended': False, 'win':False, 'time':self.rle._game.time}
            # trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState'], e['rle']) \
                # for e in self.finalEventList], terminationCondition)
            trace = (self.finalTimeStepList, terminationCondition)

            t1 = time.time()
            hypotheses = list(game_object.runInduction(game_object.spriteInductionResult, trace, 20, \
            verbose=False, existingTheories=hypotheses))
            # print "induction took {} seconds".format(time.time()-t1)
            if hypotheses[0].__dict__ != self.hypotheses[0].__dict__:
                theory_change_flag = True

            # #  PRECONDITIONS HANDLING
            # # Current assumptions:
            # # - Only one resource can change for each timestep
            # # - The first time a resource changes, it goes from 0 to a positive
            # #   value
            # for change_resource_effect in [e[3] for e in event['effectList'] if ('changeResource' in e)] + [e[3] for e in event['effectList'] if ('collectResource' in e)]:
            #     resource = change_resource_effect['resource']
            #     val = change_resource_effect['value']
            #     limit = change_resource_effect['limit']

            #     if (resource not in self.seen_resources and val>0):
            #         self.fakeInteractionRules.extend(hypotheses[0].updateInteractionsPreconditions(resource))
            #         self.fakeInteractionRules = list(set(self.fakeInteractionRules))
            #         self.seen_resources.append(resource)
            #         hypotheses[0].resource_limits[resource] = limit
            #         theory_change_flag = True

            #     if agentState[resource]==limit and resource not in self.seen_limits:
            #         self.fakeInteractionRules.extend(hypotheses[0].updateInteractionsPreconditions(resource, limit))
            #         self.fakeInteractionRules = list(set(self.fakeInteractionRules))
            #         self.seen_limits.append(resource)

            #         theory_change_flag = True
            #         # print "reached resource limit for", resource

        ## We need to update termination conditions even when we haven't seen a new event,
        ## because the state is informative about termination conditions.
        # t1 = time.time()
        # oldhypothesis = copy.deepcopy(hypotheses[0])
        oldTerminationSet = set(hypotheses[0].terminationSet)
        if event['effectList'] and run_induction:
            [t.updateTerminations(event=event) for t in hypotheses]

        # if hypotheses[0].__dict__ != oldhypothesis.__dict__:
        if set(hypotheses[0].terminationSet) != oldTerminationSet:
            if self.display_text:
                print "terminationSet Change"
            theory_change_flag = True
            # embed()

        # if action == K_LEFT:
            # embed()
        # print "updateTerminations took {} seconds".format(time.time()-t1)
        if theory_change_flag and not distributionsHaveChanged and self.display_text:
            print "changed theory:"
            hypotheses[0].display()

        return hypotheses, theory_change_flag, effects



if __name__ == "__main__":

    ##simpleGame_missile: no support for learning that it can shoot things.
    # filename = "examples.gridphysics.demo_helper"


    # filename = "examples.gridphysics.expt_physics_sharpshooter"
    # filename = "examples.gridphysics.demo_transform_relational"
    # filename = "examples.gridphysics.simpleGame_push_boulders"
    # filename = "examples.gridphysics.pick_apples"
    # filename = "examples.gridphysics.expt_exploration_exploitation_debugging"

    filename = "examples.gridphysics.theory_overload"

    level_game_pairs = None
    # Playing GVG-AI games
    def read_gvgai_game(filename):
        with open(filename, 'r') as f:
            new_doc = []
            g = gen_color()
            for line in f.readlines():
                new_line = (" ".join([string if string[:4]!="img="
                    else "color={}".format(next(g))
                    for string in line.split(" ")]))
                new_doc.append(new_line)
            new_doc = "\n".join(new_doc)
        return new_doc

    def gen_color():
        from vgdl.colors import colorDict
        color_list = colorDict.values()
        color_list = [c for c in color_list if c not in ['UUWSWF']]
        for color in color_list:
            yield color

    gvggames = ['aliens', 'boulderdash', 'butterflies', 'chase', 'frogs',  # 0-4
        'missilecommand', 'portals', 'sokoban', 'survivezombies', 'zelda']  # 5-9

    # gameName = gvggames[5]

    # gvgname = "../gvgai/training_set_1/{}".format(gameName)

    # gameString = read_gvgai_game('{}.txt'.format(gvgname))


    # level_game_pairs = []
    # for level_number in range(5):
        # with open('{}_lvl{}.txt'.format(    gvgname, level_number), 'r') as level:
            # level_game_pairs.append([gameString, level.read()])

    ##uncomment this line to run local games
    gameName = filename

    hyperparameter_sets = [{'idx'           : 2,
     'short_horizon' : False,
     'first_order_horizon': False,
     'sprite_first_alpha': 10000,
     'sprite_second_alpha': 100,
     'sprite_negative_mult': .1,
     'multisprite_first_alpha': 10000,
     'multisprite_second_alpha': 100,
     'novelty_first_alpha': 5000,
     'novelty_second_alpha': 50,
     }]

    agent = Agent('full', gameName, hyperparameter_sets[0])

    ##then pass this down for multiple episodes
    gameObject = None
    agent.playCurriculum(level_game_pairs=level_game_pairs)

    ##and use this line
    # agent.playCurriculum(level_game_pairs=None)
