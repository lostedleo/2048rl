#!/usr/bin/env python
# coding=utf-8

import env
import gym
import model

GAME_SIZE=4

def main():
    max_tile = []
    env = gym.make('game2048-v0', size=GAME_SIZE)
    agent = model.QLearning(env.action_space)
    trials = 1000

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
        max_tile.append(env.highest())
        stepno = 0
        rewards = 0

    print(max_tile)
    print(len(agent.q_table))

if __name__ == "__main__":
    main()
