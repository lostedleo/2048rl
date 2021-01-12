#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam

from env.game_2048 import Game2048

#Converting observations in range (0,1) using log(n)/log(max) so that gradients don't vanish
def process_log(observation):
    observation = np.reshape(observation, (4, 4))
    observation_temp = np.where(observation <= 0, 1, observation)
    processed_observation = np.log2(observation_temp)/np.log2(65536)
    return processed_observation.reshape(1,4,4)

def get_grids_next_step(grid):
    #Returns the next 4 states s' from the current state s
    grids_list = []

    for movement in range(4):
        grid_before = grid.copy()
        env = Game2048()
        env.set_board(grid_before)
        try:
            _ = env.move(movement)
        except:
            pass
        grid_after = env.get_board()
        grids_list.append(grid_after)

    return grids_list

class DQN(object):
    def __init__(self, gamma=0.90, epsilon=1.0, learning_rate=0.005,
            epsilon_decay=0.995, tau=0.125):
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = 4

        model.add(Flatten(input_shape=(4, 4)))
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=4))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state):
        #Epsilon value decays as model gains experience
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        else:
            #Getting the 4 future states
            allstates = get_grids_next_step(state)
            res = []
            for i in range(len(allstates)):
                if (allstates[i] == state).all():
                    res.append(0)
                else:
                    processed_state = process_log(allstates[i])
                    #max from the 4 future Q_Values is appended in res
                    res.append(np.max(self.model.predict(processed_state)))

            a = self.model.predict(process_log(state))
            #Final Q_Values are the sum of Q_Values of current state andfuture states
            final = np.add(a, res)

            return np.argmax(final)

    def remember(self, state, action, reward, new_state, done):
        #Replay Memory stores tuple(S, A, R, S')
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size=32
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(process_log(state))
            if done:
                target[0][action] = reward
            else:
                #Bellman Equation for update
                Q_future = max(self.target_model.predict(process_log(new_state))[0])

                #The move which was selected, its Q_Value gets updated
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit((process_log(state)), target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, filepath):
        self.model.save(filepath)
