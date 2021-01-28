#!/usr/bin/env python
# coding=utf-8

import env
import gym
import time
import getopt
import sys

def test():
    env = parse_args()
    obs = env.reset()
    rewards = 0
    step = 0

    for _ in range(1):
        while True:
           # if render for every step
           env.render()
           start = time.time() * 1000
           action = env.action_space.sample()
           obs, reward, done, info = env.step(action)
           rewards += reward
           step += 1
           if done:
               escape = time.time() * 1000 - start
               env.render()
               print(f'play games steps: {step} reward: {rewards} info: {info} use {escape}ms')
               time.sleep(0.5)

               step = 0
               rewards = 0
               start = time.time() * 1000
               env.reset()

def usage():
    print(sys.argv[0] + ' -s game_size -l learning_rate -d reward_decay')
    print(sys.argv[0] + ' -h get help info')

def parse_args():
    opts, args = getopt.getopt(sys.argv[1:], 'hs:l:d:',
            ['help', 'size='])
    game_size = 4
    for opt, value in opts:
        if opt == '-s' or opt == '--size':
            game_size = int(value)
        elif opt == '-h' or opt == 'help':
            usage()
            exit(-1)
    env = gym.make('game2048-v0', size=game_size)
    return env

if __name__ == "__main__":
    test()
