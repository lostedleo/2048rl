#!/usr/bin/env python
# coding=utf-8

import env
import gym
import model
import numpy as np

GAME_SIZE=3
scores = []
trials = 4 * 1000 * 100

def train_ql():
    env = gym.make('game2048-v0', size=GAME_SIZE)
    agent = model.QLearning(env.action_space)

    for trial in range(trials):
        obs = env.reset()
        obs = str(obs.reshape(GAME_SIZE ** 2).tolist())
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            action = agent.choose_action(str(obs))
            obs_, reward, done, _ = env.step(action)
            obs_ = str(obs_.reshape(GAME_SIZE ** 2).tolist())
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

    print(len(agent.q_table))
    plot_score()

def train_sarsa():
    env = gym.make('game2048-v0', size=GAME_SIZE)
    agent = model.Sarsa(env.action_space)

    for trial in range(trials):
        obs = env.reset()
        obs = str(obs.reshape(GAME_SIZE ** 2).tolist())
        action = agent.choose_action(obs)
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            obs_, reward, done, _ = env.step(action)
            obs_ = str(obs_.reshape(GAME_SIZE ** 2).tolist())
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

def train_sl():
    env = gym.make('game2048-v0', size=GAME_SIZE)
    agent = model.SarsaLambda(env.action_space)

    for trial in range(trials):
        obs = env.reset()
        obs = str(obs.reshape(GAME_SIZE ** 2).tolist())
        action = agent.choose_action(obs)
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            obs_, reward, done, _ = env.step(action)
            obs_ = str(obs_.reshape(GAME_SIZE ** 2).tolist())
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

if __name__ == "__main__":
    train_ql()
