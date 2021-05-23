from collections import deque

import numpy as np
import gym
from gym import spaces

from .vec_env.dummy_vec_env import DummyVecEnv
from .vec_env.subproc_vec_env import SubprocVecEnv

def make_gym(env_id, size, norm):
    """
    Create a wrapped atari Environment
    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = gym.make(env_id, size=size, norm=norm)
    return env

def make_gym_env(env_id, num_env, seed, size, norm,
                 start_index=0, allow_early_resets=True,
                 start_method=None, use_subprocess=False):
    """
    Create a wrapped, monitored VecEnv for Atari.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.  See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    def make_env(rank):
        def _thunk():
            env = make_gym(env_id, size, norm)
            env.seed(seed + rank)
            return env

        return _thunk

    # set_global_seeds(seed)

    # When using one environment, no need to start subprocesses
    if num_env == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)],
                         start_method=start_method)
