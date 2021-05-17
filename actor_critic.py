#!/usr/bin/env python
# coding=utf-8

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from IPython.display import clear_output
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from common.multiprocessing_env import SubprocVecEnv

import env

num_envs = 32
#  env_name = "CartPole-v0"
env_name = "game2048-v0"

def make_env():
    def _thunk():
        env = gym.make(env_name, size=3, norm=True)
        return env

    return _thunk

env = make_env()()

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, hd2_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hd2_size),
            nn.ReLU(),
            nn.Linear(hd2_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hd2_size),
            nn.ReLU(),
            nn.Linear(hd2_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def test_env(model, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward, env.get_score()

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


#Hyper params:
hidden_size = 128
hd2_size = 64
lr          = 3e-4
num_steps   = 5

def main():
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n

    model = ActorCritic(num_inputs, num_outputs, hidden_size, hd2_size).to(device)
    optimizer = optim.Adam(model.parameters())

    max_frames   = 10000
    frame_idx    = 0
    test_rewards = []

    state = envs.reset()

    while frame_idx < max_frames:

        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            state = next_state
            frame_idx += 1

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        print(f'\rframe: {frame_idx}\t loss: {loss}', end='')
        if frame_idx % 100 == 0:
            rewards, scores = map(list,zip(*((test_env(model, False) for _ in range(10)))))
            avg_rewards = np.mean(rewards)
            avg_scores = np.mean(scores)
            print(f'\rframe: {frame_idx}\t avg_rewards: {avg_rewards:.2f}\t avg_scores: {avg_scores:.2f}\t loss: {loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ((test_env(model, True) for _ in range(10)))
    envs.close()

if __name__ == '__main__':
    main()
