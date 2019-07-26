import random
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
    def __init__(self, config, record_flag=1):
        self.config = config
        self.Env = VGDLEnv(self.config.game_name, 'all_games')
        self.Env.set_level(0)
        self.game_size = np.shape(self.Env.render())
        self.input_channels = self.game_size[2]
        self.action_space.n = len(self.Env.actions)
        self.action_space = self.Env.actions

        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))
        self.ended = 0
        self.num_episodes = config.num_episodes
        self.screen_history = []
        self.steps = 0
        self.episode_steps = 0
        self.episode = 0
        self.episode_reward = 0
        self.record_flag = record_flag

        if self.record_flag:
            with open('reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config.game_name,
                                                                             self.config.level_switch,
                                                                             self.config.trial_num), "wb") as file:
                writer = csv.writer(file)
                writer.writerow(["level", "steps", "ep_reward", "win", "game_name", "criteria"])

            with open('object_interaction_histories/{}_object_interaction_history_{}_trial{}.csv'.format(
                    self.config.game_name, self.config.level_switch, self.config.trial_num), "wb") as file:
                interactionfilewriter = csv.writer(file)
                interactionfilewriter.writerow(
                    ['agent_type', 'subject_ID', 'modelrun_ID', 'game_name', 'game_level', 'episode_number', 'event_name',
                     'count'])

            ## PEDRO: Rename as needed
            picklefilepath = 'pickleFiles/{}.csv'.format(self.config.game_name)
            self.recent_history = [0] * int(self.config.criteria.split('/')[1])


    ### FOR Gym API
    def step(self, action):
        self.steps += 1
        self.episode_steps += 1
        self.append_gif()
        self.reward , self.ended, self.win = Env.step(action)
        avatar_position_data['episodes'][-1].append((self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
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
            event_dict[e] += 1

        self.episode_reward += self.reward
        #self.reward = max(-1.0, min(self.reward, 1.0))
        #self.last_screen = self.current_screen
        self.state = self.get_screen()

        #self.next_state = self.current_screen - self.last_screen
        #if not self.ended:
        #    self.next_state = current_screen - last_screen
        #else:
        #    self.next_state = None
        # Store the transition in memory
        self.memory.push(self)
        # Move to the next state
        #self.state = self.next_state

        if self.ended or self.episode_steps > self.config.timeout:
            if self.episode_steps > self.config.timeout: print("Game Timed Out")
            ## PEDRO: 3. At the end of each episode, write events to csv
            if self.record_flag:
                with open('object_interaction_histories/{}_object_interaction_history_{}_trial{}.csv'.format(
                        self.config.game_name, self.config.level_switch, self.config.trial_num), "ab") as file:
                    interactionfilewriter = csv.writer(file)
                    for event_name, count in event_dict.items():
                        row = ('DDQN', 'NA', 'NA', self.config.game_name, self.Env.lvl, self.episode, event_name, count)
                        interactionfilewriter.writerow(row)
            self.episode += 1
            print("Level {}, episode reward at step {}: {}".format(self.Env.lvl, self.steps, self.episode_reward))
                sys.stdout.flush()
            episode_results = [self.Env.lvl, self.steps, self.episode_reward, self.win, self.config.game_name,
                                 int(self.config.criteria.split('/')[0])]
            self.recent_history.insert(0, self.win)
            self.recent_history.pop()

            if self.level_step():
                if self.record_flag:
                    with open('reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config.game_name,
                                                                                         self.config.level_switch,
                                                                                         self.config.trial_num),
                              "ab") as file:
                        writer = csv.writer(file)
                        writer.writerow(episode_results)
                    break
            self.episode_reward = 0

            if self.episode % 2 == 0 and record_flag:
                with open(picklefilepath, 'wb') as f:
                    cloudpickle.dump(avatar_position_data, f)

            if record_flag:
                with open('reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config.game_name,
                                                                                     self.config.level_switch,
                                                                                     self.config.trial_num),
                          "ab") as file:
                    writer = csv.writer(file)
                    writer.writerow(episode_results)
            self.screen_history = []
        return self.state, self.reward, self.ended, _

    def reset(self):
        self.Env.reset()
        avatar_position_data['episodes'].append([(self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                                          self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                                          self.Env.current_env._game.time, self.Env.lvl)])
        event_dict = defaultdict(lambda: 0)
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
        return screen

    def save_gif(self):
        imageio.mimsave('screens/{}_frame{}.gif'.format(self.config.game_name, self.steps), self.screen_history)

    def append_gif(self):
        frame = self.Env.render(gif=True)
        self.screen_history.append(frame)

	###Auxiliary functions from player.py
    def level_step(self):
        if self.config.level_switch == 'sequential':
            if sum(self.recent_history) == int(self.config.criteria.split('/')[0]):  # if level is 'won'
                if self.Env.lvl == len(self.Env.env_list) - 1:  # if this is the last training level
                    print("Learning Finished")
                    return 1
                else:  # if this isn't the last level
                    self.Env.lvl += 1
                    self.Env.set_level(self.Env.lvl)
                    print("Next Level!")
                    self.recent_history = [0] * int(self.config.criteria.split('/')[1])
                    return 0
        ##ANDRES Note that nothing happens otherwise
        elif self.config.level_switch == 'random':
            # else:
            self.Env.lvl = np.random.choice(range(len(self.Env.env_list) - 1))
            self.Env.set_level(self.Env.lvl)
            return 0
        else:
            raise Exception('level switch not specified.')
