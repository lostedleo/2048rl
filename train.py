#!/usr/bin/env python
# coding=utf-8

from env.game_2048 import Game2048
from model.dqn import *

def main():
    max_tile = []
    env = Game2048()
    dqn_agent = DQN()
    trials = 1000

    for trial in range(trials):
        cur_state = env.reset()
        cur_state = cur_state.reshape(4, 4)
        stepno = 0
        while True:
            stepno += 1
            action = dqn_agent.get_action(cur_state)
            new_state, reward, done, _ = env.step(action)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            if stepno % 10 == 0:
                dqn_agent.replay()
                dqn_agent.target_train()

            cur_state = new_state
            if done:
                break

        if env.highest() >= 2048:
            print(f'Completed in {trial} use {stepno} steps highest: \
{env.highest()}')
            print(env.get_board())
            max_tile.append(env.highest())
            dqn_agent.save_model("success.model")
        else:
            print(f'Trial number {trial} Failed to complete use {stepno} \
steps highest: {env.highest()}')
            print(env.get_board())
            max_tile.append(env.highest())
            if env.highest() >= 512:
                dqn_agent.save_model('trial num-{}.model'.format(trial))

    print(max_tile)

if __name__ == "__main__":
    main()
