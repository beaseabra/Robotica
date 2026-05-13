"""
Train DQN — Condition C2 (Gaussian noise + dropout, static obstacles only).
Same timesteps as C1 for fair comparison.
"""
import sys, os
import numpy as np
import torch

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
sys.path.insert(0, os.path.join(WEBOTS_HOME, "lib", "controller", "python"))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from env.webots_env import WebotsVehicleEnv
from env.discrete_action_wrapper import DiscreteActionWrapper
from env.lidar_noise_wrapper import LiDARNoiseWrapper

LOG_DIR    = "./logs/dqn_c2/"
SAVE_DIR   = "./models/dqn_c2/"
BEST_DIR   = "./models/dqn_c2/best/"
FINAL_PATH = os.path.join(SAVE_DIR, "dqn_c2_final")

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

base_env = WebotsVehicleEnv(random_reset=True, heading_noise_deg=10.0)
noisy    = LiDARNoiseWrapper(base_env, noise_std=0.1, dropout_prob=0.05)
env      = DiscreteActionWrapper(noisy)
check_env(env, warn=True)

model = DQN(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    seed=SEED,
    tensorboard_log=LOG_DIR,
)

checkpoint_cb = CheckpointCallback(
    save_freq=10_000, save_path=SAVE_DIR, name_prefix="dqn_c2"
)
eval_cb = EvalCallback(
    env,
    best_model_save_path=BEST_DIR,
    log_path=os.path.join(LOG_DIR, "eval"),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

print("=== DQN C2 — Sensor Noise Training ===")
model.learn(
    total_timesteps=100_000,
    callback=CallbackList([checkpoint_cb, eval_cb]),
    progress_bar=True,
    reset_num_timesteps=False,
)
model.save(FINAL_PATH)
print(f"Saved: {FINAL_PATH}.zip")
