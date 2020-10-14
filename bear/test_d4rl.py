import numpy as np
import gym
import d4rl # Import required to register environments

# Create the environment
env = gym.make('maze2d-large-v1')

# d4rl abides by the OpenAI gym interface
obs = env.reset()
#env.render()
obs, rew, done, info = env.step(env.action_space.sample())
#env.render()


# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations'].shape) # An N x dim_observation Numpy array of observations (N = 1e6)
print(dataset['actions'].shape)
print(dataset['rewards'].shape)

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)

'''
import gym

env = gym.make('HalfCheetah-v2')

while True:
    obs = env.reset()
    env.render()

    env.step(env.action_space.sample())
    env.render()
'''




