"""
Train PPO — Condition C1 (clean sensors, static obstacles only).
"""
import sys, os
import numpy as np
import torch

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
sys.path.insert(0, os.path.join(WEBOTS_HOME, "lib", "controller", "python"))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from env.webots_env import WebotsVehicleEnv

LOG_DIR    = "./logs/ppo_c1/"
SAVE_DIR   = "./models/ppo_c1/"
BEST_DIR   = "./models/ppo_c1/best/"
FINAL_PATH = os.path.join(SAVE_DIR, "ppo_c1_final")

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

env = WebotsVehicleEnv(random_reset=True, heading_noise_deg=10.0)
check_env(env, warn=True)

model = PPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    seed=SEED,
    tensorboard_log=LOG_DIR,
)

checkpoint_cb = CheckpointCallback(
    save_freq=10_000, save_path=SAVE_DIR, name_prefix="ppo_c1"
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

print("=== PPO C1 — Baseline Training ===")
model.learn(
    total_timesteps=100_000,
    callback=CallbackList([checkpoint_cb, eval_cb]),
    progress_bar=True,
    reset_num_timesteps=False,
)
model.save(FINAL_PATH)
print(f"Saved: {FINAL_PATH}.zip")
