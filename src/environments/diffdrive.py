from typing import Optional, Union, Tuple

import gym
import numpy as np
from gym.core import ObsType, ActType
import matplotlib.pyplot as plt


class DiffDriveRobot(gym.Env):

    def __init__(self, timestep=0.01, process_noise=np.array([0.1, 0.1, 0.1, 0]), measurement_noise= np.array([0.5, 0.2])):
        self._state = np.array([0., 0., 0., 0.])
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._timestep = timestep
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        x_t = self._state
        x_tp1 = np.zeros_like(x_t)
        x_tp1[0] = x_t[0] + x_t[3] * np.cos(x_t[2]) * self._timestep
        x_tp1[1] = x_t[1] + x_t[3] * np.sin(x_t[2]) * self._timestep
        x_tp1[2] = x_t[2] + action[1] * self._timestep
        x_tp1[3] = action[0]
        self._state = x_tp1 + np.random.normal(0, self._process_noise ** 2)
        obs = x_tp1[:2] + np.random.normal(0, self._measurement_noise)
        return obs, 0.0, False, dict(x=x_tp1)

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[ObsType, Tuple[ObsType, dict]]:
        self._state = np.array([0., 0., 0., 0.])
        return self._state[:2]

    def render(self, mode="human"):
        pass
