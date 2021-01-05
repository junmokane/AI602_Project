import gym
from rlkit.envs.point_robot import PointEnv
import numpy as np
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv

print(ENVS)

env = NormalizedBoxEnv(ENVS['point-robot']())

traj = []

obs = env.reset()
traj.append(obs)
while True:
    action = env.action_space.sample()
    # action = np.array([2.0, 2.0])
    print(action)
    obs, rew, d, _ = env.step(action)
    print(obs, rew)
    traj.append(obs)
    env.render()

    if d:
        break

print(traj)
states = np.array(traj)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import csv
import pickle
import os
import colour

gr = 0.1 # goal radius, for visualization purposes
g = np.array([1.0, 1.0])
plt.figure(figsize=(8,8))
axes = plt.axes()
axes.set(aspect='equal')
plt.axis([-0.25, 1.25, -0.25, 1.25])
circle = plt.Circle((g[0], g[1]), radius=gr)
axes.add_artist(circle)
plt.plot(states[:-1, 0], states[:-1, 1], '-o')
plt.plot(states[-1, 0], states[-1, 1], '-x', markersize=20)
plt.show()

