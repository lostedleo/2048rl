#!/usr/bin/env python
# coding=utf-8

import env
import gym
import model

GAME_SIZE=4

def main():
    max_tile = []
    env = gym.make('game2048-v0', size=GAME_SIZE)
    agent = model.DeepQNetwork(env.action_space.n,
                               GAME_SIZE ** 2,
                               memory_size = 2000)
    trials = 1000

    for trial in range(trials):
        obs = env.reset()
        obs = obs.reshape(GAME_SIZE ** 2)
        stepno = 0
        rewards = 0
        while True:
            stepno += 1
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            obs_ = obs_.reshape(GAME_SIZE ** 2)
            agent.store_transition(obs, action, reward, obs_)
            if (stepno > 200) and (stepno % 5 == 0):
                agent.learn()

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
    agent.plot_cost()

if __name__ == "__main__":
    main()
