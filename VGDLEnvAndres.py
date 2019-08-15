import random
from gym import spaces
from collections import namedtuple, defaultdict
import numpy as np
from PIL import Image
import pdb
from scipy import misc
import imageio
import sys
from VGDLEnv import VGDLEnv
import csv
import cloudpickle
import cv2


import os
from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN, K_SPACE

class VGDLEnvAndres(object):
    def __init__(self, game_name):

        ###CONFIGS
        self.game_name = game_name
        self.game_name_short = game_name[5:]
        self.level_switch = 'sequential'
        self.trial_num = 1000
        self.criteria = '1/1'
        self.timeout = 2000
        games_folder = '../all_games'

        ##FOR RECORDING
        self.record_flag = 1 #record_flag
        #pdb.set_trace()
        self.reward_histories_folder = '../reward_histories'
        self.object_interaction_histories_folder = '../object_interaction_histories'
        self.picklefilepath = '../pickleFiles/{}.csv'.format(self.game_name_short)

        self.Env = VGDLEnv(self.game_name_short, games_folder)
        self.Env.set_level(0)
        self.action_space = spaces.Discrete(len(self.Env.actions))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(84,84,3))

        self.game_over = 0
        self.screen_history = []
        self.steps = 0
        self.episode_steps = 0
        self.episode = 0
        self.episode_reward = 0
        self.event_dict = defaultdict(lambda: 0)
        self.recent_history = [0] * int(self.criteria.split('/')[1])

        if self.record_flag:
            with open('{}/{}_reward_history_{}_trial{}.csv'.format(self.reward_histories_folder,self.game_name_short,
                                                                             self.level_switch,
                                                                             self.trial_num), "ab") as file:
                writer = csv.writer(file)
                writer.writerow(["level", "steps", "ep_reward", "win", "game_name", "criteria"])

            with open('{}/{}_object_interaction_history_{}_trial{}.csv'.format(
                    self.object_interaction_histories_folder,self.game_name_short, self.level_switch, self.trial_num), "wb") as file:
                interactionfilewriter = csv.writer(file)
                interactionfilewriter.writerow(
                    ['agent_type', 'subject_ID', 'modelrun_ID', 'game_name', 'game_level', 'episode_number', 'event_name',
                     'count'])


    ### FOR Gym API
    def set_level(self, intended_level, intended_steps):
        self.Env.lvl = intended_level
        self.Env.set_level(self.Env.lvl)
        self.steps = intended_steps

    def get_level(self):
        return self.Env.lvl

    def step(self, action):
        if self.steps>= 1000000:
            sys.exit()
        self.steps += 1
        self.episode_steps += 1
        self.append_gif()
        self.reward , self.game_over, self.win = self.Env.step(action)
        self.avatar_position_data['episodes'][-1].append((self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                                     self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                                     self.Env.current_env._game.time,
                                                     self.Env.lvl))
        ## PEDRO: 2. Store events that occur at each timestep
        timestep_events = set()
        for e in self.Env.current_env._game.effectListByClass:
            ## because event handling is so weird in Frogs, we need to filter out these events.
            ## Avatar-water and avatar-log collisions will still be reported from the (killSprite avatar water) interaction and (pullWithIt avatar log) interaction
            ## which is what a player perceives when they play
            if e in [('changeResource', 'avatar', 'water'), ('changeResource', 'avatar', 'log')]:
                pass
            else:
                timestep_events.add(tuple(sorted((e[1], e[2]))))
        for e in timestep_events:
            self.event_dict[e] += 1

        self.episode_reward += self.reward
        #self.reward = max(-1.0, min(self.reward, 1.0))
        #self.last_screen = self.current_screen
        self.state = self.get_screen()

        if self.game_over or self.episode_steps > self.timeout:
            if self.episode_steps > self.timeout: print("Game Timed Out")
            ## PEDRO: 3. At the end of each episode, write events to csv
            if self.record_flag:
                with open('{}/{}_object_interaction_history_{}_trial{}.csv'.format(
                        self.object_interaction_histories_folder, self.game_name_short, self.level_switch, self.trial_num), "ab") as file:
                    interactionfilewriter = csv.writer(file)
                    for event_name, count in self.event_dict.items():
                        row = ('DDQN', 'NA', 'NA', self.game_name_short, self.Env.lvl, self.episode, event_name, count)
                        interactionfilewriter.writerow(row)
            self.episode += 1
            print("Level {}, episode reward at step {}: {}".format(self.Env.lvl, self.steps, self.episode_reward))
            sys.stdout.flush()
            episode_results = [self.Env.lvl, self.steps, self.episode_reward, self.win, self.game_name_short,
                                 int(self.criteria.split('/')[0])]

            self.recent_history.insert(0, self.win)
            self.recent_history.pop()
            if self.level_step():
                if self.record_flag:
                    with open('{}/{}_reward_history_{}_trial{}.csv'.format( self.reward_histories_folder,
                                                                                         self.game_name_short,
                                                                                         self.level_switch,
                                                                                         self.trial_num),
                              "ab") as file:
                        writer = csv.writer(file)
                        writer.writerow(episode_results)
                    print('{{}'.format(1))
                    return self.state, self.reward, self.game_over, 0
            self.episode_reward = 0

            if self.episode % 2 == 0 and self.record_flag:
                with open(self.picklefilepath, 'wb') as f:
                    cloudpickle.dump(self.avatar_position_data, f)

            if self.record_flag:
                with open('{}/{}_reward_history_{}_trial{}.csv'.format(self.reward_histories_folder ,self.game_name_short,
                                                                                     self.level_switch,
                                                                                     self.trial_num),

                          "ab") as file:
                    writer = csv.writer(file)
                    writer.writerow(episode_results)
            self.screen_history = []
        return self.state, self.reward, self.game_over, 0

    def reset(self):
        self.Env.reset()
        self.avatar_position_data = {'game_info': (self.Env.current_env._game.width, self.Env.current_env._game.height),
                                'episodes': [[(self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                               self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                               self.Env.current_env._game.time,
                                               self.Env.lvl)]]}
        self.episode_steps = 0
        #self.last_screen = self.get_screen()
        #self.current_screen = self.get_screen()
        #self.state = current_screen - last_screen
        self.state = self.get_screen()
        return self.state

    ####Screen functions from player.py
    def save_screen(self):
        misc.imsave('original.png', self.Env.render())
        misc.imsave('altered.png', np.rollaxis(self.get_screen().cpu().numpy()[0], 0, 3))

    def get_screen(self):
        # imageio.imsave('sample.png', self.Env.render())
        screen = self.Env.render()
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = cv2.resize(screen, dsize=(84,84), interpolation=cv2.INTER_CUBIC)
        screen_1channel = np.mean(screen, axis=2)
        #return screen
        return screen_1channel

    def save_gif(self):
        imageio.mimsave('screens/{}_frame{}.gif'.format(self.game_name_short, self.steps), self.screen_history)

    def append_gif(self):
        frame = self.Env.render(gif=True)
        self.screen_history.append(frame)

	###Auxiliary functions from player.py
    def level_step(self):
        if self.level_switch == 'sequential':
            if sum(self.recent_history) == int(self.criteria.split('/')[0]):  # if level is 'won'
                if self.Env.lvl == len(self.Env.env_list) - 1:  # if this is the last training level
                    print("Learning Finished")
                    return 1
                else:  # if this isn't the last level
                    self.Env.lvl += 1
                    self.Env.set_level(self.Env.lvl)
                    print("Next Level!")
                    self.recent_history = [0] * int(self.criteria.split('/')[1])
                    return 0
        ##ANDRES Note that nothing happens otherwise
        elif self.level_switch == 'random':
            # else:
            self.Env.lvl = np.random.choice(range(len(self.Env.env_list) - 1))
            self.Env.set_level(self.Env.lvl)
            return 0
        else:
            raise Exception('level switch not specified.')
