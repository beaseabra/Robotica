import gymnasium as gym
import numpy as np


class LiDARNoiseWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper that injects noise into LiDAR observations (C2 / C4 / C5).

    Applied independently at every timestep:
      1. Gaussian noise:  ray += N(0, noise_std)
      2. Ray dropout:     each ray zeroed with probability dropout_prob

    Camera and all other observation keys are passed through unchanged.

    Default parameters (match the study specification):
        noise_std    = 0.1   (sigma of Gaussian perturbation)
        dropout_prob = 0.05  (5% of rays dropped per timestep)
    """

    def __init__(
        self,
        env,
        noise_std: float = 0.1,
        dropout_prob: float = 0.05,
        lidar_max_range: float = 100.0,
    ):
        super().__init__(env)
        self.noise_std       = noise_std
        self.dropout_prob    = dropout_prob
        self.lidar_max_range = lidar_max_range

    def observation(self, obs):
        noisy = dict(obs)
        lidar = obs["lidar"].copy()

        # 1 — Gaussian noise
        lidar += np.random.normal(0.0, self.noise_std, size=lidar.shape).astype(np.float32)

        # 2 — Ray dropout (simulate sensor failure)
        mask = np.random.rand(*lidar.shape) < self.dropout_prob
        lidar[mask] = 0.0

        # Clip to valid sensor range
        noisy["lidar"] = np.clip(lidar, 0.0, self.lidar_max_range).astype(np.float32)
        return noisy
