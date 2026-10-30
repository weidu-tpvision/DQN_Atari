"""Compare 2048 checkpoints on the same held-out games.

The evaluations during training use 10-20 games, which is too noisy to rank checkpoints (and
`best_model.pt` is picked on those same games). This plays every checkpoint greedily on the same
fixed set of seeds, disjoint from the training-time eval seeds, and reports the standard error.

    .venv/bin/python3 eval_2048.py checkpoints_2048/best_model.pt checkpoints_2048_exp/*/dqn_2048_final.pt
"""

import argparse

import numpy as np

from src.agent import DQNAgent
from src.model_2048 import DQN2048
from src.utils import get_device
from train_2048 import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate 2048 checkpoints on fixed held-out games")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files to evaluate")
    parser.add_argument("--episodes", type=int, default=200, help="Games per checkpoint")
    parser.add_argument("--seed", type=int, default=50_000, help="Seed of the first game (training evals start at 1000)")
    args = parser.parse_args()

    device = get_device()
    width = max(len(p) for p in args.checkpoints)
    for path in args.checkpoints:
        agent = DQNAgent(num_actions=4, device=device, model_fn=lambda: DQN2048(in_channels=16, num_actions=4))
        agent.load_weights(path)
        r = evaluate_policy(agent, num_episodes=args.episodes, seed=args.seed)
        se = r["std_score"] / np.sqrt(args.episodes)
        print(f"{path:{width}s}  mean {r['mean_score']:7.0f} ±{se:4.0f}  "
              f">=512 {r['pct_512']:5.1f}%  >=1024 {r['pct_1024']:5.1f}%  >=2048 {r['pct_2048']:4.1f}%  "
              f"max tile {r['highest_tile']}", flush=True)


if __name__ == "__main__":
    main()
