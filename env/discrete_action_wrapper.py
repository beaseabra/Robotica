import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Maps Discrete(6) actions to the continuous Box(2,) of WebotsVehicleEnv.

    Used for DQN (discrete control). PPO uses the raw continuous env directly.

    Actions:
        0 — forward          [ 0.0,  0.6]
        1 — strong left      [-0.4,  0.6]
        2 — strong right     [ 0.4,  0.6]
        3 — slight left      [-0.2,  0.6]
        4 — slight right     [ 0.2,  0.6]
        5 — stop / brake     [ 0.0,  0.0]
    """

    ACTIONS = [
        np.array([ 0.0,  0.6], dtype=np.float32),   # 0 forward
        np.array([-0.4,  0.6], dtype=np.float32),   # 1 strong left
        np.array([ 0.4,  0.6], dtype=np.float32),   # 2 strong right
        np.array([-0.2,  0.6], dtype=np.float32),   # 3 slight left
        np.array([ 0.2,  0.6], dtype=np.float32),   # 4 slight right
        np.array([ 0.0,  0.0], dtype=np.float32),   # 5 stop
    ]

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

    def action(self, action: int) -> np.ndarray:
        return self.ACTIONS[int(action)]
