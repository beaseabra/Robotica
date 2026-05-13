from env.webots_env import WebotsVehicleEnv
from env.critical_obstacles import CriticalObstacleManager


class WebotsCriticalEnv(WebotsVehicleEnv):
    """
    WebotsVehicleEnv extended with critical dynamic obstacles (C3 / C4).

    Drop-in replacement for WebotsVehicleEnv.
    Wrap with LiDARNoiseWrapper on top for C4:

        base = WebotsCriticalEnv(random_reset=True)
        env  = LiDARNoiseWrapper(base, noise_std=0.1, dropout_prob=0.05)

    IMPORTANT: Open worlds/city_obstacles.wbt in Webots before running,
    as it contains the PEDESTRIAN_1 and VEHICLE_1 DEF nodes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obstacle_manager = CriticalObstacleManager(
            supervisor=self.robot,
            vehicle_translation_field=self.translation_field,
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.obstacle_manager.reset()
        return obs, info

    def step(self, action):
        real_action = self._denormalize_action(action)
        self._apply_action(real_action)

        # move dynamic obstacles before advancing simulation
        self.obstacle_manager.step()

        if self.robot.step(self.timestep) == -1:
            return {}, 0.0, True, False, {}

        obs = self._get_observations()
        self._check_stuck(real_action)
        reward, done, collision = self._compute_reward(obs, real_action)

        return obs, float(reward), done, False, {"collision": collision}
