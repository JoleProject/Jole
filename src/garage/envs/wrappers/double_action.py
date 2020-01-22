"""Clip reward for gym.Env."""
import gym
import numpy as np
import random

class DoubleAction(gym.Wrapper):
    """Clip the reward by its sign."""

    def step(self, ac):
        """gym.Env step function."""
        r = random.uniform(0, 1)
        if r>0.8:
            ac = ac * 2
        obs, reward, done, info = self.env.step(ac)
        return obs, np.sign(reward), done, info

    def reset(self):
        """gym.Env reset."""
        return self.env.reset()
