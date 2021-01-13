#!/usr/bin/env python
# coding=utf-8

from env.game_2048 import Game2048
from gym.envs.registration import register

register(id='game2048-v0', entry_point='env:Game2048')
