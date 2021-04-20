#!/usr/bin/env python
# coding=utf-8

from absl import app
from absl import flags
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

flags.DEFINE_integer('size', 3, '2048 game size')
flags.DEFINE_integer('agt', 1, 'agent type')
flags.DEFINE_integer('memory_size', int(1e5), 'dueling_DQN memory size')
flags.DEFINE_boolean('test', False, 'if test model or not')
flags.DEFINE_boolean('double_q', False, 'if double dqn')
flags.DEFINE_boolean('dueling', False, 'if dueling dqn')

FLAGS = flags.FLAGS

def train_dqn(size, agt, eps_start=1.0, eps_end=0.08, eps_decay=0.99999):
    env = gym.make('game2048-v0', size=size, norm=True)
    env.seed(1)

    agent = model.DQNAgent(size * size, 4, 0, FLAGS.double_q, FLAGS.dueling)
    total_steps = 0
    total_scores = 0
    highest_score = 0
    #  trials = 1 * 100000000 * (size ** 2)
    trials = 6000
    eps = eps_start
    scores_window = deque(maxlen=WINDOWS_SIZE)
    rewards_window = deque(maxlen=WINDOWS_SIZE)
    scores = []
    sd_name = 'model_%dx%d.checkpoint'%(size, size)

    for trial in range(1, trials+1):
        obs = env.reset()
        stepno = 0
        rewards = 0
        loss = 0
        while True:
            stepno += 1
            total_steps += 1
            action, _ = agent.choose_action(obs, eps)
            obs_, reward, done, _ = env.step(action)
            loss = agent.step(obs, action, reward, obs_, done)
            obs = obs_
            rewards += reward
            if done:
                break

        eps = max(eps_end, eps_decay * eps)
        rewards_window.append(rewards)
        scores_window.append(env.get_score())
        scores.append(rewards)
        #  env.render()
        if env.get_score() > highest_score:
            highest_score = env.get_score()
        total_scores += env.get_score()
        print('\rEpisode {}\t Steps: {}\t\t Average Reward: {:.2f}\t\t Average Scores: {:.2f}\t loss: {}\t eps: {}'.
                format(trial, total_steps, np.mean(rewards_window), np.mean(scores_window), loss, eps), end="")
        if trial % WINDOWS_SIZE == 0:
            print('\rEpisode {}\t Steps: {}\t\t Average Reward: {:.2f}\t\t Average Scores: {:.2f}\t loss: {}\t eps: {}'.
                format(trial, total_steps, np.mean(rewards_window), np.mean(scores_window), loss, eps))
        if trial % 1000 == 0:
            agent.save(sd_name)

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
    eps = 0.0

    for i in range(times):
        obs = env.reset()

        while True:
            action, action_values = agent.choose_action(obs, eps)
            if eps > 0.0:
                eps = 0.0
            obs_, reward, done, _ = env.step(action)
            if render:
                env.render()
            if str(obs_) == str(obs):
                #env.render()
                print(f'action is: {action} {reward} {action_values} {obs} {obs_}')
                agent.step(obs, action, reward, obs_, done)
                eps = 1.0
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

def test_dqn(size):
    import torch
    env = gym.make('game2048-v0', size=size)
    agent = model.DQNAgent(size * size, 4, 0, FLAGS.double_q, FLAGS.dueling)

    sd_name = 'model_%dx%d.checkpoint'%(size, size)
    agent.load(sd_name)
    eval(env, agent, 1000, render=False)

def test_dueling_dqn(size):
    pass

def main(_):
    if FLAGS.test:
        test_dqn(FLAGS.size)
        return

    train_dqn(FLAGS.size, FLAGS.agt)

if __name__ == "__main__":
    app.run(main)
