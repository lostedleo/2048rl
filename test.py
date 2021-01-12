#!/usr/bin/env python
# coding=utf-8

import time

from env.game_2048 import *

env = Game2048()
obs = env.reset()
print(obs)

rewards = 0
step = 0

for _ in range(1):
    while True:
       env.render()
       action = env.action_space.sample()
       obs, reward, done, info = env.step(action)
       rewards += reward
       step += 1
       if done:
           print(f'play games steps: {step} reward: {rewards} info: {info}')
           time.sleep(1)
           env.reset()

