"""Tournament Benchmark script to evaluate and rank multiple Atari DQN checkpoints.

Plays identical matches across fixed seeds for each candidate model (the seed fixes the
emulator, sticky actions and the number of reset no-ops), evaluating:
  - Match Win Rate (% of games won to 21 points)
  - Mean Net Score (+21 to -21)
  - Average Points Conceded to CPU (Defense quality)
  - Shutouts (Games with <= 3 conceded points)
  - Average match step duration

Usage:
    # Run tournament across the top trained checkpoints (default 5 matches each):
    python tournament.py

    # Run deep tournament across 10 matches:
    python tournament.py --matches 10

    # Test specific checkpoints:
    python tournament.py --checkpoints checkpoints/best_model.pt checkpoints/dqn_ALE_Pong-v5_step_1250000.pt
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from src.wrappers import make_atari_env
from src.agent import DQNAgent
from src.utils import get_device


def get_default_checkpoints(ckpt_dir: str = "checkpoints") -> list[str]:
    """Auto-discover the top elite milestone checkpoints across the entire training history."""
    preferred = [
        "dqn_ALE_Pong-v5_step_5100000.pt",
        "dqn_ALE_Pong-v5_step_9700000.pt",
        "dqn_ALE_Pong-v5_step_10000000.pt",
        "dqn_ALE_Pong-v5_step_4600000.pt",
        "dqn_ALE_Pong-v5_step_3900000.pt",
        "dqn_ALE_Pong-v5_step_2350000.pt",
        "dqn_ALE_Pong-v5_step_2200000.pt",
        "dqn_ALE_Pong-v5_step_1450000.pt",
        "dqn_ALE_Pong-v5_step_1350000.pt",
        "best_model.pt",
    ]
    candidates = []
    for name in preferred:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            candidates.append(path)
    return candidates


def parse_args():
    parser = argparse.ArgumentParser(description="Atari DQN Tournament Benchmark")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="List of checkpoint paths to evaluate. Defaults to top trained checkpoints.",
    )
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Atari environment ID")
    parser.add_argument("--matches", type=int, default=5, help="Number of 21-point matches per checkpoint")
    parser.add_argument("--base-seed", type=int, default=100, help="Base seed (identical across models)")
    parser.add_argument("--output-csv", type=str, default="checkpoints/tournament_results.csv", help="Output CSV path")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Generate leaderboard visualization")
    return parser.parse_args()


def play_match(env, agent, device, seed: int) -> dict:
    """Play a single full game of Pong to 21 points with greedy policy."""
    state, _ = env.reset(seed=seed)
    agent_pts = 0
    cpu_pts = 0
    steps = 0

    while True:
        steps += 1
        with torch.no_grad():
            s_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
            q_vals = agent.online_net(s_tensor)
            action = int(q_vals.argmax(dim=-1).item())

        state, reward, terminated, truncated, _ = env.step(action)
        if reward > 0:
            agent_pts += int(reward)
        elif reward < 0:
            cpu_pts += int(-reward)

        if terminated or truncated:
            break

    won = agent_pts > cpu_pts
    net_score = agent_pts - cpu_pts
    return {
        "agent_pts": agent_pts,
        "cpu_pts": cpu_pts,
        "net_score": net_score,
        "won": won,
        "steps": steps,
        "shutout": cpu_pts <= 3,
    }


def main():
    args = parse_args()
    device = get_device()
    print(f"\n=======================================================")
    print(f"       ATARI DQN CHAMPIONSHIP TOURNAMENT")
    print(f"=======================================================")
    print(f"Device: {device} | Matches per candidate: {args.matches}")
    print(f"Environment: {args.env} (unclipped raw score, full to 21)")

    ckpt_list = args.checkpoints if args.checkpoints else get_default_checkpoints()
    if not ckpt_list:
        print("Error: No valid checkpoints found.")
        return

    print(f"Contenders ({len(ckpt_list)} checkpoints):")
    for idx, c in enumerate(ckpt_list, 1):
        print(f"  [{idx}] {os.path.basename(c)}")
    print(f"-------------------------------------------------------\n")

    env = make_atari_env(args.env, clip_rewards=False, episodic_life=False)
    num_actions = env.action_space.n
    agent = DQNAgent(num_actions=num_actions, device=device)

    leaderboard = []

    for c_idx, ckpt_path in enumerate(ckpt_list, 1):
        name = os.path.basename(ckpt_path)
        print(f">>> [{c_idx}/{len(ckpt_list)}] Benchmarking: {name} ...", flush=True)

        try:
            agent.load(ckpt_path)
        except Exception as e:
            print(f"    Failed to load {ckpt_path}: {e}")
            continue

        match_records = []
        for m in range(args.matches):
            seed = args.base_seed + m
            res = play_match(env, agent, device, seed=seed)
            match_records.append(res)
            res_tag = "WON" if res["won"] else "LOST"
            print(f"    Match {m+1:2d} (Seed {seed}): Agent {res['agent_pts']:2d} - {res['cpu_pts']:2d} CPU (Net: {res['net_score']:+3d}) -> {res_tag}", flush=True)

        # Aggregate stats for this model
        wins = sum(1 for r in match_records if r["won"])
        win_rate = (wins / args.matches) * 100.0
        net_scores = [r["net_score"] for r in match_records]
        cpu_scores = [r["cpu_pts"] for r in match_records]
        agent_scores = [r["agent_pts"] for r in match_records]
        shutouts = sum(1 for r in match_records if r["shutout"])
        avg_steps = np.mean([r["steps"] for r in match_records])

        leaderboard.append({
            "model": name,
            "path": ckpt_path,
            "win_rate": win_rate,
            "wins": wins,
            "matches": args.matches,
            "mean_net_score": float(np.mean(net_scores)),
            "std_net_score": float(np.std(net_scores)),
            "min_net_score": int(np.min(net_scores)),
            "max_net_score": int(np.max(net_scores)),
            "avg_agent_pts": float(np.mean(agent_scores)),
            "avg_cpu_conceded": float(np.mean(cpu_scores)),
            "shutouts": shutouts,
            "avg_steps": float(avg_steps),
        })

    env.close()

    # Sort leaderboard: 
    # 1. Win Rate (descending)
    # 2. Conceded points to CPU (ascending, lower is better defense)
    # 3. Mean Net Score (descending)
    leaderboard.sort(key=lambda x: (-x["win_rate"], x["avg_cpu_conceded"], -x["mean_net_score"]))

    # Print Leaderboard Table
    print(f"\n==========================================================================================================")
    print(f"                                      FINAL TOURNAMENT LEADERBOARD                                        ")
    print(f"==========================================================================================================")
    header = f"{'Rank':<5} | {'Checkpoint':<32} | {'Win Rate':<10} | {'Mean Net':<10} | {'CPU Conceded':<12} | {'Shutouts':<9} | {'Avg Steps':<9}"
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(leaderboard, 1):
        trophy = " 🥇" if rank == 1 else (" 🥈" if rank == 2 else (" 🥉" if rank == 3 else "   "))
        net_str = f"{row['mean_net_score']:+.1f} +/- {row['std_net_score']:.1f}"
        print(f"{rank:<2}{trophy} | {row['model']:<32} | {row['win_rate']:>6.1f}%   | {net_str:<10} | {row['avg_cpu_conceded']:>7.1f} pts  | {row['shutouts']:>3d}/{row['matches']:<3d}   | {row['avg_steps']:>7.0f}")

    print("=" * len(header))

    champ = leaderboard[0]
    print(f"\n🏆 TOURNAMENT CHAMPION: {champ['model']}")
    print(f"   * Match Win Rate: {champ['win_rate']:.1f}% ({champ['wins']}/{champ['matches']})")
    print(f"   * Mean Net Score: {champ['mean_net_score']:+.1f} (Peak: {champ['max_net_score']:+d})")
    print(f"   * Best Defense: Only conceded {champ['avg_cpu_conceded']:.1f} points per game to the CPU!")
    print(f"   * Shutout Matches: {champ['shutouts']} of {champ['matches']} games conceded <= 3 points.\n")

    # Save to CSV
    out_dir = os.path.dirname(args.output_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(leaderboard)
    df.to_csv(args.output_csv, index=False)
    print(f"--> Saved detailed tournament statistics to: {args.output_csv}")

    # Optional plot
    if args.plot and len(leaderboard) > 1:
        plot_tournament_leaderboard(leaderboard, os.path.join(out_dir, "tournament_leaderboard.png"))


def plot_tournament_leaderboard(leaderboard: list[dict], save_path: str):
    """Generate a clean visual ranking chart of the tournament contenders."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [r["model"].replace("dqn_ALE_Pong-v5_", "").replace(".pt", "") for r in leaderboard]
    net_scores = [r["mean_net_score"] for r in leaderboard]
    net_errors = [r["std_net_score"] for r in leaderboard]
    cpu_conceded = [r["avg_cpu_conceded"] for r in leaderboard]

    x = np.arange(len(models))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Panel 1: Net Score (Higher is better)
    bars1 = ax1.bar(x, net_scores, yerr=net_errors, capsize=4, width=width, color="#2ecc71", edgecolor="#27ae60", alpha=0.85)
    ax1.set_ylabel("Mean Net Score (Agent - CPU)", fontsize=11, fontweight="bold")
    ax1.set_title("Atari Pong DQN Tournament: Contender Comparison", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(min(0.0, min(n - e for n, e in zip(net_scores, net_errors)) - 1.0), 22)
    ax1.axhline(0.0, color="#34495e", linewidth=0.8)
    ax1.axhline(21.0, color="#7f8c8d", linestyle="--", alpha=0.6, label="Perfect Game (+21.0)")
    ax1.legend(loc="upper right")

    # Add bar values
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Conceded Points (Lower is better defense)
    bars2 = ax2.bar(x, cpu_conceded, width=width, color="#e74c3c", edgecolor="#c0392b", alpha=0.85)
    ax2.set_ylabel("Avg CPU Points Conceded\n(Lower = Better Defense)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Checkpoint", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax2.set_ylim(0, max(15.0, max(cpu_conceded) + 2.0))

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"--> Saved visual tournament leaderboard to: {save_path}")


if __name__ == "__main__":
    main()
