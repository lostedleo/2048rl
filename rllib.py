#!/usr/bin/env python
# coding=utf-8
import gym
import env
from ray import tune

tune.run(
    "A3C",
    config={
        "env": env.Game2048,
        "num_workers": 11,
        "env_config": {"size": 2}})
