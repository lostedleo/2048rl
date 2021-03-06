import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import logging
import random

import itertools
from six import StringIO
import sys
from math import log2
from math import pow

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

class Game2048(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, config=None, size=4, norm=False):
        # Definitions for game. Board-matrix must be square.
        self.size = size
        self.norm = norm
        if config is not None:
            self.size = config['size']
            self.norm = config['norm']

        self.w = self.size
        self.h = self.size
        self.penalty = -1
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards.
        self.score = 0
        self.illegal_actions = 0

        # Members for gym implementation:
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board-matrix.
        if self.norm:
            self.observation_space = spaces.Box(0, 1, (self.squares + 2, self.size, self.size), dtype=np.int)
            self.ret_obs = self.norm_obs
            self.ret_func = self._norm_ret
        else:
            self.observation_space = spaces.Box(0, pow(2, self.squares + 1), (self.size, self.size), dtype=np.int)
            self.ret_obs = lambda x: x.copy()
            self.ret_func = self._default_ret

        # Initialise the random seed of the gym environment.
        self.seed()

        # Reset the board-matrix, ready for a new game.
        self.reset()

    def _default_ret(self, obs, reward, done, info):
        return self.ret_obs(obs), reward, done, info

    def _norm_ret(self, obs, reward, done, info):
        if reward > 0:
            reward = log2(reward)
        else:
            reward = -0.5

        return self.ret_obs(obs), reward, done, info

    def seed(self, seed=None):
        """Set the random seed for the gym environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Implementation of gym interface:
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        add_val = 0
        try:
            score = float(self.move(action))
            self.score += score
            add_val = self.add_tile()
            done = self.isend()
            reward = float(score)
            self.illegal_actions = 0
        except IllegalMove as e:
            logging.debug("Illegal move")
            done = False
            reward = self.penalty # No reward for an illegal move. We could even set a negative value to penalize the agent.
            self.illegal_actions +=1

        observation = self.Matrix
        # info (dictionary):
        #    - can be used to store further information to the caller after executing each step/movement in the game
        #    - it is useful for testing and for monitoring the agent (via callback functions) while it is training
        info = {"max_tile": self.highest(), "score": self.score}
        if done:
            reward = 0
        else:
            if self.illegal_actions > 10:
                done = True
                info['illegal_actions'] = True

        # Return observation (board-matrix state), reward, done and info dictionary
        return self.ret_func(observation, reward, done, info)

    def norm_obs(self, obs):
        value = obs.copy()
        value *= 2
        index = value == 0
        value[index] = 2
        value = np.log2(value).astype(np.int32)
        value -= 1

        ret = np.zeros((self.squares + 2, self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                ret[value[i, j], i, j] = 1

        return ret

    def reset(self):
        """Reset the game board-matrix and add 2 tiles."""
        self.Matrix = np.zeros((self.h, self.w), np.int)
        self.score = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return self.ret_obs(self.Matrix)

    def render(self, mode='human'):
        """Rendering for standard output of score, highest tile reached and
        board-matrix of game."""
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\t'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        s += "{}\n".format(self.Matrix)
        outfile.write(s)
        return outfile

    # Implementation of game logic for 2048:
    def add_tile(self):
        """Add a tile with value 2 or 4 with different probabilities."""
        val = 0
        if self.np_random.random_sample() > 0.9:
            val = 4
        else:
            val = 2
        empties = self.empties()
        assert empties
        empty_idx = self.np_random.choice(len(empties))
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)
        return val

    def get(self, x, y):
        """Get the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a list of tuples of the location of empty squares."""
        empties = list()
        for y in range(self.h):
            for x in range(self.w):
                if self.get(x, y) == 0:
                    empties.append((x, y))
        return empties

    def highest(self):
        """Report the highest tile on the board-matrix."""
        highest = 0
        for y in range(self.h):
            for x in range(self.w):
                highest = max(highest, self.get(x, y))
        return highest

    def get_score(self):
        return self.score

    def get_size(self):
        return self.size

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        #assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Check if the game is ended. Game ends if there is a 2048 tile or
        there are no legal moves. If there are empty spaces then there must
        be legal moves."""
        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Get the whole board-matrix, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Set the whole board-matrix, useful for testing."""
        self.Matrix =new_board


