import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
WEBOTS_PYTHON_PATH = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
if WEBOTS_PYTHON_PATH not in sys.path:
    sys.path.append(WEBOTS_PYTHON_PATH)

from controller import Supervisor


class WebotsVehicleEnv(gym.Env):
    """
    Gymnasium environment for autonomous vehicle lane following in Webots.

    Action Space (PPO — continuous):
        Box(2,): normalized [steering, throttle/brake] in [-1, 1]
        Real range: steering in [-0.5, 0.5], throttle in [-1, 1]

    Observation Space:
        Dict:
          - "lidar": Box(180,) — LiDAR distance readings (0 to 100 m)
          - "camera": Box(64, 128, 3) — RGB image from front camera

    Conditions:
        C1 — WebotsVehicleEnv()                           (baseline)
        C2 — LiDARNoiseWrapper(WebotsVehicleEnv())        (sensor noise)
        C3 — WebotsCriticalEnv()                          (dynamic obstacles)
        C4 — LiDARNoiseWrapper(WebotsCriticalEnv())       (combined)
    """

    def __init__(self, random_reset: bool = False, heading_noise_deg: float = 10.0):
        super().__init__()

        # ── Simulator ──────────────────────────────────────────────
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # ── Sensors ────────────────────────────────────────────────
        self.lidar = self.robot.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        # ── Vehicle node & fields ──────────────────────────────────
        self.vehicle_node = self.robot.getSelf()
        self.translation_field = self.vehicle_node.getField("translation")
        self.rotation_field    = self.vehicle_node.getField("rotation")

        self.initial_translation = list(self.translation_field.getSFVec3f())
        self.initial_rotation    = list(self.rotation_field.getSFRotation())

        # ── Reset strategy ─────────────────────────────────────────
        self.random_reset       = random_reset
        self.heading_noise_rad  = np.deg2rad(heading_noise_deg)

        # ── Action space ───────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_low  = np.array([-0.5, -1.0], dtype=np.float32)
        self.action_high = np.array([ 0.5,  1.0], dtype=np.float32)

        # ── Observation space ──────────────────────────────────────
        cam_h = self.camera.getHeight()
        cam_w = self.camera.getWidth()
        lidar_w = self.lidar.getHorizontalResolution()

        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(
                low=0.0, high=100.0, shape=(lidar_w,), dtype=np.float32
            ),
            "camera": spaces.Box(
                low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8
            ),
        })

        # ── Actuators ──────────────────────────────────────────────
        self.left_steering  = self.robot.getDevice("left_steer")
        self.right_steering = self.robot.getDevice("right_steer")

        self.wheels = []
        for name in ["left_front_wheel", "right_front_wheel"]:
            w = self.robot.getDevice(name)
            w.setPosition(float("inf"))
            w.setVelocity(0.0)
            self.wheels.append(w)

        # ── Episode state ──────────────────────────────────────────
        self.stuck_step_count        = 0
        self.stuck_step_limit        = 500
        self.stuck_distance_threshold = 0.001
        self.previous_position       = np.array(
            self.translation_field.getSFVec3f(), dtype=np.float32
        )

        self.lost_line_steps    = 0
        self.max_lost_line_steps = 10

        self.current_step    = 0
        self.max_episode_steps = 2000

        # cumulative lane deviation for metrics
        self.cumulative_lane_deviation = 0.0
        self.lane_deviation_count      = 0

        # ── Viewpoint (optional follow camera) ────────────────────
        vp = self.robot.getFromDef("VIEWPOINT")
        if vp is not None:
            self._vp_pos   = vp.getField("position")
            self._vp_ori   = vp.getField("orientation")
            self._vp_pos0  = self._vp_pos.getSFVec3f()
            self._vp_ori0  = self._vp_ori.getSFRotation()
        else:
            self._vp_pos = self._vp_ori = None

    # ── Helpers ────────────────────────────────────────────────────

    def _denormalize_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def _perturbed_rotation(self):
        """Return initial rotation with a small random heading perturbation."""
        rot = list(self.initial_rotation)
        noise = np.random.uniform(-self.heading_noise_rad, self.heading_noise_rad)
        rot[3] = rot[3] + noise
        return rot

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ── Restore vehicle pose ───────────────────────────────────
        self.translation_field.setSFVec3f(self.initial_translation)
        if self.random_reset:
            self.rotation_field.setSFRotation(self._perturbed_rotation())
        else:
            self.rotation_field.setSFRotation(self.initial_rotation)

        # ── Restore viewpoint ──────────────────────────────────────
        if self._vp_pos is not None:
            self._vp_pos.setSFVec3f(self._vp_pos0)
            self._vp_ori.setSFRotation(self._vp_ori0)

        self.vehicle_node.resetPhysics()
        self.robot.step(self.timestep)

        for w in self.wheels:
            w.setVelocity(0.0)

        # ── Reset episode counters ─────────────────────────────────
        self.stuck_step_count = 0
        self.previous_position = np.array(
            self.translation_field.getSFVec3f(), dtype=np.float32
        )
        self.lost_line_steps = 0
        self.current_step    = 0
        self.cumulative_lane_deviation = 0.0
        self.lane_deviation_count      = 0

        return self._get_observations(), {}

    # ── Sensor reading ─────────────────────────────────────────────

    def get_camera_image(self):
        raw = self.camera.getImage()
        img = np.frombuffer(raw, np.uint8).reshape(
            (self.camera.getHeight(), self.camera.getWidth(), 4)
        )
        return img[:, :, [2, 1, 0]]   # BGRA → RGB

    def _get_observations(self):
        lidar = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar = np.nan_to_num(lidar, nan=100.0, posinf=100.0, neginf=0.0)
        return {"lidar": lidar, "camera": self.get_camera_image()}

    # ── Lane detection ─────────────────────────────────────────────

    def _extract_yellow_line_features(self, camera_image):
        """
        Detect the yellow lane line using colour segmentation.

        Returns:
            line_visible (bool), lane_error (float in [-1,1]), yellow_ratio (float)
        """
        h, w, _ = camera_image.shape
        roi  = camera_image[int(h * 0.45):, :]
        r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]

        mask = (
            (r > 120) & (g > 100) & (b < 100) &
            (r > b * 1.4) & (g > b * 1.4)
        )
        pts = np.argwhere(mask)
        ratio = pts.shape[0] / mask.size

        if pts.shape[0] == 0:
            return False, 1.0, 0.0

        cx = np.mean(pts[:, 1])
        error = float(np.clip((cx - w / 2.0) / (w / 2.0), -1.0, 1.0))
        return True, error, float(ratio)

    # ── Action application ─────────────────────────────────────────

    def _apply_action(self, action):
        steer    = float(action[0])
        velocity = float(action[1]) * 50.0
        self.left_steering.setPosition(steer)
        self.right_steering.setPosition(steer)
        for w in self.wheels:
            w.setVelocity(velocity)

    # ── Stuck detection ────────────────────────────────────────────

    def _check_stuck(self, real_action):
        pos = np.array(self.translation_field.getSFVec3f(), dtype=np.float32)
        movement = np.linalg.norm(pos - self.previous_position)
        if abs(real_action[1]) > 0.05 and movement < self.stuck_distance_threshold:
            self.stuck_step_count += 1
        else:
            self.stuck_step_count = 0
        self.previous_position = pos

    # ── Reward & termination ───────────────────────────────────────

    def _compute_reward(self, obs, action):
        lidar   = obs["lidar"]
        camera  = obs["camera"]
        steer   = float(action[0])
        throttle = float(action[1])

        line_visible, lane_error, yellow_ratio = self._extract_yellow_line_features(camera)

        # track lane deviation for metrics
        if line_visible:
            self.cumulative_lane_deviation += abs(lane_error)
            self.lane_deviation_count += 1

        reward = 0.0
        done   = False
        collision = False

        # ── LiDAR obstacle zone ────────────────────────────────────
        n      = len(lidar)
        ctr    = n // 2
        front  = lidar[max(0, ctr - 10): min(n, ctr + 10)]
        d_min  = float(np.min(front))
        obstacle_near      = d_min < 4.0
        obstacle_very_close = d_min < 1.2

        forward_reward = max(0.0, throttle)

        if line_visible:
            self.lost_line_steps = 0
            center_reward = 1.0 - abs(lane_error)

            if obstacle_near:
                reward += forward_reward * 0.8
                reward += min(abs(steer), 0.5) * 1.0
                reward += center_reward * 0.5
                reward -= abs(lane_error) * 0.3
                if obstacle_very_close:
                    reward += min(abs(steer), 0.5) * 2.0
                    reward -= 2.0
            else:
                reward += center_reward * 3.0
                reward += forward_reward * 0.5
                reward -= abs(lane_error) * 2.0
                reward -= abs(steer) * 0.2
                if abs(lane_error) < 0.15:
                    reward += 2.0

            if yellow_ratio < 0.001:
                reward -= 1.0
        else:
            self.lost_line_steps += 1
            if obstacle_near:
                reward -= 1.0
                reward += forward_reward * 0.5
                reward += min(abs(steer), 0.5) * 0.5
            else:
                reward -= 5.0

        # reverse penalty
        if throttle < 0:
            reward += throttle * 0.2

        # time penalty
        reward -= 0.01

        # ── Termination conditions ─────────────────────────────────
        if self.lost_line_steps >= self.max_lost_line_steps:
            reward -= 100.0
            done = True

        if self.stuck_step_count >= self.stuck_step_limit:
            reward -= 100.0
            done = True

        # collision: any LiDAR ray < 0.45 m
        if np.min(lidar) < 0.45:
            reward -= 100.0
            done      = True
            collision = True

        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True

        return reward, done, collision

    # ── Step ───────────────────────────────────────────────────────

    def step(self, action):
        real_action = self._denormalize_action(action)
        self._apply_action(real_action)

        if self.robot.step(self.timestep) == -1:
            return {}, 0.0, True, False, {}

        obs = self._get_observations()
        self._check_stuck(real_action)
        reward, done, collision = self._compute_reward(obs, real_action)

        info = {"collision": collision}
        return obs, float(reward), done, False, info

    def mean_lane_deviation(self):
        """Average lateral error (normalised, 0–1) over the episode."""
        if self.lane_deviation_count == 0:
            return 1.0
        return self.cumulative_lane_deviation / self.lane_deviation_count
