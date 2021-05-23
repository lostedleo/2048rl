#!/usr/bin/env python
# coding=utf-8

import env
import gym
import time
import getopt
import sys
import numpy as np

from pypprof.net_http import start_pprof_server
from env.env_wrappers import make_gym_env

def test(game_size, norm):
    #  start_pprof_server(port=8081)
    env = gym.make('game2048-v0', size=game_size, norm=norm)
    obs = env.reset()
    rewards = 0
    step = 0

    for _ in range(1):
        start = time.time() * 1000
        while True:
           # if render for every step
           #  env.render()
           action = env.action_space.sample()
           obs, reward, done, info = env.step(action)
           rewards += reward
           step += 1
           if done:
               escape = time.time() * 1000 - start
               env.render()
               print(f'obs: {obs}')
               print(f'play games steps: {step} reward: {rewards} info: {info}'
                       + f' use {escape:.3f}ms speed: {(step * 1000 / escape):.3f}ops/s')
               time.sleep(0.5)

               step = 0
               rewards = 0
               start = time.time() * 1000
               env.reset()

def test_vec(game_size, norm):
    vec_num = 10
    vec_env = make_gym_env('game2048-v0', vec_num, 0, game_size, norm)
    episodes = 1000

    obs = vec_env.reset()
    start = time.time()
    rewards = np.zeros(vec_num)
    for i in range(episodes):
        actions = np.random.randint(4, size=vec_num)
        obs, reward, done, info = vec_env.step(actions)
        vec_env.render()
        rewards += reward

    used = (time.time() - start) * 1000
    print(f'play vec games steps: {episodes} rewards: {rewards}'
            + f' used: {used:.3f}ms speed: {(episodes*vec_num*1000/used):.3f}ops/s')

def usage():
    print(sys.argv[0] + ' -s game_size -l learning_rate -d reward_decay')
    print(sys.argv[0] + ' -h get help info')

def parse_args():
    opts, args = getopt.getopt(sys.argv[1:], 'hs:l:d:v:n:',
            ['help', 'size', 'vec', 'norm'])
    game_size = 4
    vec_env = False
    norm = False
    for opt, value in opts:
        if opt == '-s' or opt == '--size':
            game_size = int(value)
        elif opt == '-v' or opt == '--vec':
            vec_env = True
        elif opt == '-n' or opt == '--norm':
            norm = True
        elif opt == '-h' or opt == 'help':
            usage()
            exit(-1)

    if vec_env:
        test_vec(game_size, norm)
    else:
        test(game_size, norm)

if __name__ == "__main__":
    parse_args()
