#!/usr/bin/env python
# coding=utf-8

import sys
import env
import gym
import numpy as np
import random
import math
import getopt
import sys
from collections import defaultdict, deque
import matplotlib.pyplot as plt

def update_Q_sarsa(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
    """Returns updated Q-value for the most recent experience."""
    current = Q[state][action]  # estimate in Q-table (for current state, action pair)
    # get value of state, action pair at next time step
    Qsa_next = Q[next_state][next_action] if next_state is not None else 0
    target = reward + (gamma * Qsa_next)               # construct TD target
    new_value = current + (alpha * (target - current)) # get updated value
    return new_value

def epsilon_greedy(env, Q, state, nA, eps):
    """Selects epsilon-greedy action for supplied state.

    Params
    ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
    """
    if random.random() > eps: # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:                     # otherwise, select an action randomly
        return random.choice(np.arange(env.action_space.n))

def sarsa(size, num_episodes, alpha, gamma=1.0, plot_every=100):
    env = gym.make('game2048-v0', size=size)
    nA = env.action_space.n                # number of actions
    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays

    # monitor performance
    tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes

    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        score = 0                                             # initialize score
        state = env.reset()                                   # start episode
        state = str(state.reshape(size ** 2).tolist())

        eps = 1.0 / i_episode                                 # set value of epsilon
        action = epsilon_greedy(env, Q, state, nA, eps)            # epsilon-greedy action selection

        while True:
            next_state, reward, done, info = env.step(action) # take action A, observe R, S'
            next_state = str(next_state.reshape(size **2).tolist())
            score += reward                                   # add reward to agent's score
            if not done:
                next_action = epsilon_greedy(env, Q, next_state, nA, eps) # epsilon-greedy action
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, \
                        state, action, reward, next_state, next_action)

                state = next_state     # S <- S'
                action = next_action   # A <- A'
            if done:
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, \
                        state, action, reward)
                tmp_scores.append(score)    # append score
                break
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Q length: %d Episodes: ' % (plot_every, len(Q))), np.max(avg_scores))
    return Q

def update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state=None):
    """Returns updated Q-value for the most recent experience."""
    current = Q[state][action]  # estimate in Q-table (for current state, action pair)
    Qsa_next = np.max(Q[next_state]) if next_state is not None else 0  # value of next state
    target = reward + (gamma * Qsa_next)               # construct TD target
    new_value = current + (alpha * (target - current)) # get updated value
    return new_value

def q_learning(size, num_episodes, alpha, gamma=1.0, plot_every=100):
    env = gym.make('game2048-v0', size=size)
    """Q-Learning - TD Control

    Params
    ======
        num_episodes (int): number of episodes to run the algorithm
        alpha (float): learning rate
        gamma (float): discount factor
        plot_every (int): number of episodes to use when calculating average score
    """
    nA = env.action_space.n                # number of actions
    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays

    # monitor performance
    tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes

    for i_episode in range(1, num_episodes+1):
        # monitor progress
        score = 0                                              # initialize score
        state = env.reset()                                    # start episode
        state = str(state.reshape(size ** 2).tolist())
        eps = 1.0 / i_episode                                  # set value of epsilon

        while True:
            action = epsilon_greedy(env, Q, state, nA, eps)         # epsilon-greedy action selection
            next_state, reward, done, info = env.step(action)  # take action A, observe R, S'
            next_state = str(next_state.reshape(size ** 2).tolist())
            score += reward                                    # add reward to agent's score
            Q[state][action] = update_Q_sarsamax(alpha, gamma, Q, \
                    state, action, reward, next_state)
            state = next_state                                 # S <- S'
            if done:
                tmp_scores.append(score)                       # append score
                break

        print("\rEpisode {}/{}\t Average Score: {:.2f}".format(i_episode, num_episodes, np.mean(tmp_scores)), end="")
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes))
            sys.stdout.flush()
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Q length: %d Episodes: ' % (plot_every, len(Q))), np.max(avg_scores))
    return Q

def update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state=None):
    """Returns updated Q-value for the most recent experience."""
    current = Q[state][action]         # estimate in Q-table (for current state, action pair)
    policy_s = np.ones(nA) * eps / nA  # current policy (for next state S')
    policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA) # greedy action
    Qsa_next = np.dot(Q[next_state], policy_s)         # get value of state at next time step
    target = reward + (gamma * Qsa_next)               # construct target
    new_value = current + (alpha * (target - current)) # get updated value
    return new_value

def expected_sarsa(size, num_episodes, alpha, gamma=1.0, plot_every=100):
    env = gym.make('game2048-v0', size=size)
    """Expected SARSA - TD Control

    Params
    ======
        num_episodes (int): number of episodes to run the algorithm
        alpha (float): step-size parameters for the update step
        gamma (float): discount factor
        plot_every (int): number of episodes to use when calculating average score
    """
    nA = env.action_space.n                # number of actions
    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays

    # monitor performance
    tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes

    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        score = 0             # initialize score
        state = env.reset()   # start episode
        state = str(state.reshape(size ** 2).tolist())
        eps = 0.005           # set value of epsilon

        while True:
            action = epsilon_greedy(env, Q, state, nA, eps)         # epsilon-greedy action selection
            next_state, reward, done, info = env.step(action)  # take action A, observe R, S'
            next_state = str(next_state.reshape(size ** 2).tolist())
            score += reward                                    # add reward to agent's score
            # update Q
            Q[state][action] = update_Q_expsarsa(alpha, gamma, nA, eps, Q, \
                    state, action, reward, next_state)
            state = next_state              # S <- S'
            if done:
                tmp_scores.append(score)    # append score
                break
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Q length: %d Episodes: ' % (plot_every, len(Q))), np.max(avg_scores))
    return Q

def usage():
    print(sys.argv[0] + ' -s game_size -l learning_rate -d reward_decay')
    print(sys.argv[0] + ' -h get help info')

def main():
    opts, args = getopt.getopt(sys.argv[1:], 'hs:l:d:m:e:',
            ['help', 'size=', 'learning_rate=', 'decay=', 'model=', 'episodes='])
    game_size = 2
    learning_rate = 0.01
    reward_decay = 1.0
    episodes = 5000
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
        elif opt == '-e' or opt == '--episodes=':
            episodes = int(value)
        elif opt == '-h' or opt == 'help':
            usage()
            exit(-1)

    if model == 0:
        sarsa(game_size, episodes, learning_rate, reward_decay)
    elif model == 1:
        q_learning(game_size, episodes, learning_rate, reward_decay)
    elif model == 2:
        expected_sarsa(game_size, episodes, learning_rate, reward_decay)

if __name__ == "__main__":
    main()

