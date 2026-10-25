"""Evaluation script for trained Atari DQN models.

Usage:
    python evaluate.py --checkpoint checkpoints/dqn_ALE_Pong-v5_final.pt --episodes 10
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym

from src.wrappers import make_atari_env
from src.agent import DQNAgent
from src.utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DQN on Atari")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Gymnasium Atari environment ID")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--record-dir", type=str, default="recordings", help="Directory to save evaluation videos")
    parser.add_argument("--record", action="store_true", help="Whether to record video of evaluation")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    return parser.parse_args()


def evaluate(args):
    set_seed(args.seed)
    device = get_device()

    render_mode = "rgb_array" if args.record else None
    env = make_atari_env(
        args.env,
        clip_rewards=False,       # Measure unclipped raw game score
        episodic_life=False,      # Measure true full episodes
        render_mode=render_mode,
    )

    if args.record:
        os.makedirs(args.record_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.record_dir,
            episode_trigger=lambda ep_id: True,
            name_prefix="dqn_eval",
        )

    num_actions = env.action_space.n
    agent = DQNAgent(num_actions=num_actions, device=device)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    scores = []
    lengths = []

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated
            state = next_state

        scores.append(ep_reward)
        lengths.append(ep_length)
        print(f"Episode {ep:2d}/{args.episodes}: Reward = {ep_reward:6.1f} | Length = {ep_length:5d} steps")

    env.close()

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print("\n" + "=" * 45)
    print(f"Evaluation Results over {args.episodes} episodes:")
    print(f"  Mean Score:   {mean_score:6.2f} +/- {std_score:5.2f}")
    print(f"  Min / Max:    {np.min(scores):6.1f} / {np.max(scores):6.1f}")
    print(f"  Mean Steps:   {np.mean(lengths):6.1f}")
    print("=" * 45)


if __name__ == "__main__":
    evaluate(parse_args())
