#!/usr/bin/env python
# coding=utf-8

import env
import gym
import time

env = gym.make('game2048-v0', size=2)
obs = env.reset()
rewards = 0
step = 0

for _ in range(1):
    while True:
       # if render for every step
       # env.render()
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

