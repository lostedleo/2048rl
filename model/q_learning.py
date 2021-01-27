"""
This part of code is the Q learning, which is a brain of the agent.
All decisions are made in here.
"""

import numpy as np

class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = dict()
        self.q_table_explore = dict()

    def check_state_exist(self, state):
        if state not in self.q_table:
            # add new state to q table
            self.q_table[state] = np.zeros(self.actions.n, dtype=np.float64)
            self.q_table_explore[state] = np.zeros(self.actions.n, dtype=np.int)

    def choose_action(self, observation, eps=0.):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() > eps:
            # choose best action
            state_action = self.q_table[observation]
            # some actions may have the same value, randomly choose on in these actions
            action = np.argmax(state_action)
        else:
            # choose random action
            action = self.actions.sample()
        return action

    def learn(self, *args):
        pass

    def print_q_table(self):
        print(self.q_table)

# off-policy
class QLearning(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearning, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[s_].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table[s][a] += self.lr * (q_target - q_predict)  # update
        self.q_table_explore[s][a] += 1

# on-policy
class Sarsa(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(Sarsa, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[s_][a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table[s][a] += self.lr * (q_target - q_predict)  # update

# backward eligibility traces
class SarsaLambda(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambda, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table:
            # add new state to q table
            self.q_table[state] = np.zeros(self.actions.n, dtype=np.float64)
            self.eligibility_trace[state] = np.zeros(self.actions.n, dtype=np.float64)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[s_][a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace[s][a] += 1

        # Method 2:
        self.eligibility_trace[s] *= 0
        self.eligibility_trace[s][a] = 1

        # Q update
        self.q_table[s] += self.lr * error * self.eligibility_trace[s]

        # decay eligibility trace after update
        self.eligibility_trace[s] *= self.gamma*self.lambda_
