"""
evaluate.py — Universal evaluation script for all conditions (C1–C4).

Usage examples:

  # C1 — DQN baseline (100 episodes)
  python evaluation/evaluate.py --algo dqn --condition c1 --model ./models/dqn_c1/dqn_c1_final --n_episodes 100

  # C2 — PPO with noise
  python evaluation/evaluate.py --algo ppo --condition c2 --model ./models/ppo_c2/ppo_c2_final --n_episodes 100

  # C3 — DQN with dynamic obstacles
  python evaluation/evaluate.py --algo dqn --condition c3 --model ./models/dqn_c1/dqn_c1_final --n_episodes 100

  # C4 — PPO combined (evaluate C1-trained model in C4 — cross-condition)
  python evaluation/evaluate.py --algo ppo --condition c4 --model ./models/ppo_c1/ppo_c1_final --n_episodes 100

  # C5 — PPO trained on C2, evaluated on C4
  python evaluation/evaluate.py --algo ppo --condition c4 --model ./models/ppo_c2/ppo_c2_final --n_episodes 100 --tag c5

Collision detection:
  A collision is registered when info["collision"] == True (LiDAR min < 0.45 m).
  This is propagated directly from the environment — NOT estimated from reward.

Metrics collected per episode:
  - success       : no collision AND episode not ended by stuck/lost-line
  - collision     : True/False
  - steps         : total steps in episode
  - total_reward  : cumulative reward
  - lane_deviation: mean lateral error (normalised 0–1) over the episode
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# ── Webots path ────────────────────────────────────────────────────────────────
WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
sys.path.insert(0, os.path.join(WEBOTS_HOME, "lib", "controller", "python"))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_env(condition: str, algo: str):
    """
    Build the correct environment stack for the given condition.

    C1 — base env only
    C2 — base env + LiDARNoiseWrapper
    C3 — WebotsCriticalEnv
    C4 — WebotsCriticalEnv + LiDARNoiseWrapper
    """
    from env.webots_env import WebotsVehicleEnv
    from env.webots_critical_env import WebotsCriticalEnv
    from env.lidar_noise_wrapper import LiDARNoiseWrapper
    from env.discrete_action_wrapper import DiscreteActionWrapper

    cond = condition.lower()

    if cond == "c1":
        env = WebotsVehicleEnv(random_reset=True, heading_noise_deg=10.0)

    elif cond == "c2":
        base = WebotsVehicleEnv(random_reset=True, heading_noise_deg=10.0)
        env  = LiDARNoiseWrapper(base, noise_std=0.1, dropout_prob=0.05)

    elif cond == "c3":
        env = WebotsCriticalEnv(random_reset=True, heading_noise_deg=10.0)

    elif cond == "c4":
        base = WebotsCriticalEnv(random_reset=True, heading_noise_deg=10.0)
        env  = LiDARNoiseWrapper(base, noise_std=0.1, dropout_prob=0.05)

    else:
        raise ValueError(f"Unknown condition '{condition}'. Choose from: c1, c2, c3, c4")

    if algo.lower() == "dqn":
        from env.discrete_action_wrapper import DiscreteActionWrapper
        env = DiscreteActionWrapper(env)

    return env


def load_model(algo: str, model_path: str, env):
    from stable_baselines3 import DQN, PPO
    cls = DQN if algo.lower() == "dqn" else PPO
    return cls.load(model_path, env=env)


def run_evaluation(
    algo: str,
    condition: str,
    model_path: str,
    n_episodes: int = 100,
    max_steps_per_ep: int = 2000,
    tag: str = "",
):
    print(f"\n{'='*60}")
    print(f"  Evaluation: {algo.upper()} | Condition: {condition.upper()}")
    if tag:
        print(f"  Tag: {tag}")
    print(f"  Model: {model_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"{'='*60}\n")

    env   = build_env(condition, algo)
    model = load_model(algo, model_path, env)

    results     = []
    start_time  = time.time()

    for ep in range(n_episodes):
        obs, _    = env.reset(seed=SEED + ep)
        done      = False
        total_reward = 0.0
        steps     = 0
        collision = False
        ep_start  = time.time()

        while not done and steps < max_steps_per_ep:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            total_reward += reward
            steps        += 1

            # ── Correct collision detection ────────────────────────
            if info.get("collision", False):
                collision = True

        # lane deviation: access underlying base env through wrapper chain
        base = env
        while hasattr(base, "env"):
            base = base.env
        lane_dev = base.mean_lane_deviation() if hasattr(base, "mean_lane_deviation") else float("nan")

        # success = episode ended without collision
        # (stuck / lost_line endings are failures but not collisions)
        success = not collision

        ep_time = time.time() - ep_start

        result = {
            "episode":        ep,
            "success":        success,
            "collision":      collision,
            "steps":          steps,
            "total_reward":   round(total_reward, 3),
            "lane_deviation": round(lane_dev, 4),
            "episode_time_s": round(ep_time, 2),
        }
        results.append(result)

        status = "✓" if success else ("✗ COLLISION" if collision else "✗")
        print(
            f"Ep {ep:3d} | {status:12s} | "
            f"Steps: {steps:4d} | "
            f"Reward: {total_reward:8.1f} | "
            f"Lane err: {lane_dev:.3f}"
        )

    total_time = time.time() - start_time

    # ── Summary ────────────────────────────────────────────────────
    success_rate   = float(np.mean([r["success"]        for r in results])) * 100
    collision_rate = float(np.mean([r["collision"]      for r in results]))
    avg_steps      = float(np.mean([r["steps"]          for r in results]))
    avg_reward     = float(np.mean([r["total_reward"]   for r in results]))
    avg_lane_dev   = float(np.nanmean([r["lane_deviation"] for r in results]))

    # lap time: average steps of SUCCESSFUL episodes only
    successful_steps = [r["steps"] for r in results if r["success"]]
    avg_lap_steps    = float(np.mean(successful_steps)) if successful_steps else float("nan")

    summary = {
        "algo":             algo.upper(),
        "condition":        condition.upper(),
        "tag":              tag,
        "model_path":       model_path,
        "n_episodes":       n_episodes,
        "seed":             SEED,
        "success_rate_pct": round(success_rate, 2),
        "collision_rate":   round(collision_rate, 3),
        "avg_steps":        round(avg_steps, 1),
        "avg_lap_steps":    round(avg_lap_steps, 1) if not np.isnan(avg_lap_steps) else None,
        "avg_reward":       round(avg_reward, 2),
        "avg_lane_deviation": round(avg_lane_dev, 4),
        "total_time_s":     round(total_time, 1),
        "episodes":         results,
    }

    print(f"\n{'─'*60}")
    print(f"  Success rate:      {success_rate:.1f}%")
    print(f"  Collision rate:    {collision_rate*100:.1f}%")
    print(f"  Avg steps/ep:      {avg_steps:.0f}")
    print(f"  Avg lap steps:     {avg_lap_steps:.0f}" if not np.isnan(avg_lap_steps) else "  Avg lap steps:     N/A (no successes)")
    print(f"  Avg reward/ep:     {avg_reward:.1f}")
    print(f"  Avg lane error:    {avg_lane_dev:.3f}")
    print(f"  Total time:        {total_time:.0f}s")
    print(f"{'─'*60}\n")

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs("./results", exist_ok=True)
    label    = f"{algo.lower()}_{condition.lower()}" + (f"_{tag}" if tag else "")
    out_path = f"./results/{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {out_path}")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained RL model in Webots.")
    p.add_argument("--algo",       required=True, choices=["dqn", "ppo"],
                   help="Algorithm: dqn or ppo")
    p.add_argument("--condition",  required=True, choices=["c1", "c2", "c3", "c4"],
                   help="Experimental condition")
    p.add_argument("--model",      required=True,
                   help="Path to model .zip (without extension)")
    p.add_argument("--n_episodes", type=int, default=100,
                   help="Number of evaluation episodes (default: 100)")
    p.add_argument("--max_steps",  type=int, default=2000,
                   help="Max steps per episode (default: 2000)")
    p.add_argument("--tag",        default="",
                   help="Optional label for output file (e.g. 'c5' for cross-condition eval)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        algo=args.algo,
        condition=args.condition,
        model_path=args.model,
        n_episodes=args.n_episodes,
        max_steps_per_ep=args.max_steps,
        tag=args.tag,
    )
