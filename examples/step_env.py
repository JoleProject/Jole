#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment."""
import gym

# Construct the environment
env = gym.make('MountainCar-v0')

# Reset the environment and launch the viewer
env.reset()
env.render()

# Step randomly until interrupted
try:
    print('Press Ctrl-C to stop...')
    while True:
        env.step(env.action_space.sample())
        env.render()
except KeyboardInterrupt:
    print('Exiting...')
    env.close()
