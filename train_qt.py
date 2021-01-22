#!/usr/bin/env python
# coding=utf-8

import env
import gym
import model
import numpy as np
import getopt
import sys

highest = {}
targets = {}

def train_ql(size, lr, rd):
    env = gym.make('game2048-v0', size=size)
    agent = model.QLearning(env.action_space, learning_rate=lr, reward_decay=rd)
    total_steps = 0
    total_scores = 0
    highest_score = 0
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
        if env.get_score() > highest_score:
            highest_score = env.get_score()
        total_scores += env.get_score()
        stepno = 0
        rewards = 0

    eval(env, agent, 1000, render=False)
    print(f'table_len: {len(agent.q_table)} steps: {total_steps} avg_score: {total_scores / trials} \
highest_score: {highest_score} at size: {size} lr: {lr} reward_decay: {rd}')
    print(f'table_len: {len(agent.q_table)} steps: {total_steps}')

def train_sarsa(size, lr, rd):
    env = gym.make('game2048-v0', size=size)
    agent = model.Sarsa(env.action_space, learning_rate=lr, reward_decay=rd)
    total_steps = 0
    total_scores = 0
    highest_score = 0
    trials = 1 * 1000 * (size ** 2)

    for trial in range(trials):
        obs = env.reset()
        obs = str(obs.reshape(size ** 2).tolist())
        action = agent.choose_action(obs)
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            total_steps += 1
            obs_, reward, done, _ = env.step(action)
            obs_ = str(obs_.reshape(size ** 2).tolist())
            action_ = agent.choose_action(obs_, True)
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
        if env.get_score() > highest_score:
            highest_score = env.get_score()
        total_scores += env.get_score()
        stepno = 0
        rewards = 0

    eval(env, agent, render=False)
    print(f'table_len: {len(agent.q_table)} steps: {total_steps} avg_score: {total_scores / trials} \
highest_score: {highest_score} at size: {size} lr: {lr} reward_decay: {rd}')
    print(f'highest len: {len(highest)} prob: {len(highest) * 1.0 / trials} \
target len: {len(targets)} prob: {len(targets) * 1.0 / trials}')

def train_sl(size, lr, rd):
    env = gym.make('game2048-v0', size=size)
    agent = model.SarsaLambda(env.action_space)
    trials = 1 * 10000 * (size ** 2)

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
        stepno = 0
        rewards = 0

    print(len(agent.q_table))

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
        obs = str(obs.reshape(size ** 2).tolist())

        while True:
            action = agent.choose_action(obs, True)
            obs_, reward, done, _ = env.step(action)
            obs_ = str(obs_.reshape(size ** 2).tolist())
            if render:
                print(f'action is: {action} {obs} {obs_}')
                env.render()
            if obs_ == obs:
                env.render()
                agent.learn(obs, action, reward, obs_)
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
    print(f'Writing... explore info')
    with open('explore.file', 'wb+') as f:
        for k, v in agent.q_table_explore.items():
            f.write(bytes(k + ',' + str(v) + ',' + str(agent.q_table[k]) + '\n', 'utf-8'))
    print(f'Ended write explore info')

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
        elif opt == '-m' or opt == '--model':
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
