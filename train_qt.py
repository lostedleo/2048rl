#!/usr/bin/env python
# coding=utf-8

import env
import gym
import model
import numpy as np
import getopt
import sys

scores = []
highest = {}
targets = {}
epsilon = 0.9

def train_ql(size, lr, rd):
    env = gym.make('game2048-v0', size)
    agent = model.QLearning(env.action_space, learning_rate=lr, reward_decay=rd)
    total_steps = 0
    trials = 1 * 1000 * (size ** 2)

    for trial in range(trials):
        obs = env.reset()
        obs = str(obs.reshape(size ** 2).tolist())
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            total_steps += 1
            action = agent.choose_action(str(obs))
            obs_, reward, done, _ = env.step(action)
            obs_ = str(obs_.reshape(size ** 2).tolist())
            if done:
                obs_ = 'terminal'
            agent.learn(obs, action, reward, obs_)
            obs = obs_
            rewards += reward
            if done:
                break

        env.render()
        print(f'Completed in {trial} use {stepno} steps highest: \
{env.highest()} rewards: {rewards}')
        scores.append(env.get_score())
        stepno = 0
        rewards = 0

    print(f'table_len: {len(agent.q_table)} steps: {total_steps}')
    plot_score()

def train_sarsa(size, lr, rd):
    env = gym.make('game2048-v0', size=size)
    agent = model.Sarsa(env.action_space, learning_rate=lr, reward_decay=rd)
    greedy = False
    total_steps = 0
    total_scores = 0
    highest_score = 0
    trials = 1 * 1000 * (size ** 2)

    for trial in range(trials):
        if not greedy and trial > int(trials * epsilon):
            greedy = True
        obs = env.reset()
        obs = str(obs.reshape(size ** 2).tolist())
        action = agent.choose_action(obs, greedy)
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            total_steps += 1
            obs_, reward, done, _ = env.step(action)
            obs_ = str(obs_.reshape(size ** 2).tolist())
            action_ = agent.choose_action(obs_, greedy)
            if done:
                obs_ = 'terminal'
            agent.learn(obs, action, reward, obs_, action_)
            obs = obs_
            action = action_
            rewards += reward
            if done:
                break

        env.render()
        print(f'Completed in {trial} use {stepno} steps highest: \
{env.highest()} rewards: {rewards}')
        if env.highest() >= 2 ** (size ** 2 - 1):
            highest[trial] = env.highest()
            if env.highest() >= 2 ** (size ** 2):
                targets[trial] = env.highest()
        scores.append(env.get_score())
        if env.get_score() > highest_score:
            highest_score = env.get_score()
        total_scores += env.get_score()
        stepno = 0
        rewards = 0

    print(f'table_len: {len(agent.q_table)} steps: {total_steps} avg_score: {total_scores / trials} \
highest_score: {highest_score}')
    print(f'highest len: {len(highest)} prob: {len(highest) * 1.0 / trials} \
target len: {len(targets)} prob: {len(targets) * 1.0 / trials}')
    plot_score()

def train_sl(size, lr, rd):
    env = gym.make('game2048-v0', size=size)
    agent = model.SarsaLambda(env.action_space)
    trials = 1 * 1000 * (size ** 2)

    for trial in range(trials):
        obs = env.reset()
        obs = str(obs.reshape(size ** 2).tolist())
        action = agent.choose_action(obs)
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            obs_, reward, done, _ = env.step(action)
            obs_ = str(obs_.reshape(size ** 2).tolist())
            action_ = agent.choose_action(obs_)
            if done:
                obs_ = 'terminal'
            agent.learn(obs, action, reward, obs_, action_)
            obs = obs_
            action = action_
            rewards += reward
            if done:
                break

        env.render()
        print(f'Completed in {trial} use {stepno} steps highest: \
{env.highest()} rewards: {rewards}')
        scores.append(env.get_score())
        stepno = 0
        rewards = 0

    print(len(agent.q_table))
    plot_score()

def plot_score():
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('training steps')
    plt.show()

def usage():
    print(sys.argv[0] + ' -s game_size -l learning_rate -d reward_decay')
    print(sys.argv[0] + ' -h get help info')

def main():
    opts, args = getopt.getopt(sys.argv[1:], 'hs:l:d:m:',
            ['help', 'size=', 'learning_rate=', 'decay=', 'model='])
    game_size = 2
    learning_rate = 0.05
    reward_decay = 0.9
    model = 0
    for opt, value in opts:
        if opt == '-s' or opt == '--size':
            game_size = int(value)
        elif opt == '-l' or opt == '--learning_rate':
            learning_rate = float(value)
        elif opt == '-d' or opt == '--decay':
            reward_decay = float(value)
        elif opt == 'm' or opt == '--model':
            model = int(value)
        elif opt == '-h' or opt == 'help':
            usage()
            exit(-1)

    if model == 0:
        train_sarsa(game_size, learning_rate, reward_decay)
    elif model == 1:
        train_ql(game_size, learning_rate, reward_decay)
    elif model == 2:
        train_sl(game_size, learning_rate, reward_decay)

if __name__ == "__main__":
    main()
