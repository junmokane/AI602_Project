import numpy as np
from gym import spaces
from gym import Env

from . import register_env


@register_env('point-robot')
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane
     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self):
        self._goal = np.array([1.0, 1.0])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self):
        self._state = np.array([0.0, 0.0])
        return self._state

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        if - reward < 0.1:
            done = True
        ob = self._state
        return ob, reward, done, dict()

    def render(self):
        print('current state:', self._state)