#!/usr/bin/env python
# coding=utf-8

import env
import gym
import model
import numpy as np
from collections import deque
import getopt
import sys
import matplotlib.pyplot as plt

highest = {}
targets = {}
action_info = ['UP', 'RIGHT', 'DOWN', 'LEFT']
WINDOWS_SIZE=100

def train_ddpg(size):
    env = gym.make('game2048-v0', size=size)
    agent = model.DDPGAgent(size * size, 4, 0)
    total_steps = 0
    total_scores = 0
    highest_score = 0
    trials = 1 * 100000000 * (size ** 2)
    scores_window = deque(maxlen=WINDOWS_SIZE)
    scores = []

    for trial in range(1, trials+1):
        obs = env.reset()
        obs = obs.reshape(size * size).copy()
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            total_steps += 1
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            obs_ = obs_.reshape(size * size).copy()
            agent.step(obs, action, reward, obs_, done)
            obs = obs_
            rewards += reward
            if done:
                break

        scores_window.append(rewards)
        scores.append(rewards)
        #env.render()
        #  print(f'Completed in {trial} use {stepno} steps highest: \
#  {env.highest()} rewards: {rewards}')
        if env.get_score() > highest_score:
            highest_score = env.get_score()
        total_scores += env.get_score()
        print('\rEpisode {}\t Average Score: {:.2f}'.format(trial, np.mean(scores_window)), end="")
        if trial % WINDOWS_SIZE == 0:
            print('\rEpisode {}\t Average Score: {:.2f}'.format(trial, np.mean(scores_window)))
        if np.mean(scores_window) >= 35.0:
            import torch
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    eval(env, agent, 1000, render=False)
    print(f'steps: {total_steps} avg_score: {total_scores / trials} \
highest_score: {highest_score} at size: {size}')
    plot_score(scores, [])


def plot_score(scores, max_tiles):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(max_tiles)), max_tiles)
    plt.ylabel('Score')
    plt.xlabel('training steps')
    plt.show()

def eval(env, agent, times=1000, render=False):
    highest_score = 0
    total_scores = 0
    size = env.get_size()
    scores = []
    max_tiles = []

    for i in range(times):
        obs = env.reset()
        obs = obs.reshape(size * size).copy()

        while True:
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            obs_ = obs_.reshape(size * size).copy()
            if render:
                print(f'action is: {action} {obs} {obs_}')
                env.render()
            if str(obs_) == str(obs):
                #env.render()
                agent.step(obs, action, reward, obs_, done)
#                  print(f'this should not happend {obs} action: {action} q_value: {agent.q_table[obs]} explore: \
#  {agent.q_table_explore[obs]}')
            obs = obs_
            if done:
                break

        env.render()
        scores.append(env.get_score())
        max_tiles.append(env.highest())
        if env.get_score() > highest_score:
            highest_score = env.get_score()
        total_scores += env.get_score()

    if times > 0:
        plot_score(scores, max_tiles)
        print(f'eval avg_score: {total_scores / times} highest_score: {highest_score}')

def usage():
    print(sys.argv[0] + ' -s game_size')
    print(sys.argv[0] + ' -h get help info')

def main():
    opts, args = getopt.getopt(sys.argv[1:], 'hs:',
            ['help', 'size='])
    game_size = 2
    for opt, value in opts:
        if opt == '-s' or opt == '--size':
            game_size = int(value)
        elif opt == '-h' or opt == 'help':
            usage()
            exit(-1)

    train_ddpg(game_size)

if __name__ == "__main__":
    main()
