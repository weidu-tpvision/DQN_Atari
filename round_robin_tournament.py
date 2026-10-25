"""Round-Robin Head-to-Head Tournament between top 10 Atari DQN checkpoints.

All 10 contenders play every other contender in direct PvP battles.
Each pair plays 2 matches (home and away side-swap) to ensure 100% fair evaluation.

Rules:
    - 10 Contenders -> 45 Pairings x 2 Games = 90 Total Matches
    - First to target-score (default: 5 points per match)
    - Full league table: Wins, Losses, Win Rate, Points For (PF), Points Against (PA), Differential (+/-)
    - 10x10 Head-to-Head Cross Table Heatmap and Leaderboard Chart

Usage:
    python round_robin_tournament.py
    python round_robin_tournament.py --target-score 5
"""

import os
import glob
import re
import itertools
import argparse
import numpy as np
import pandas as pd
import torch
from collections import deque

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

from pettingzoo.atari import pong_v3
from src.agent import DQNAgent
from src.utils import get_device


def process_frame(rgb: np.ndarray, flip: bool = False) -> np.ndarray:
    """Preprocess 210x160 RGB into 84x84 grayscale, optionally horizontally flipped for Player 2."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    if flip:
        resized = cv2.flip(resized, 1)
    return resized


def get_top_10_checkpoints(ckpt_dir: str = "checkpoints") -> list[dict]:
    """Retrieve top 10 unique checkpoints from previous leaderboard or rank them."""
    # Preferred order based on empirical tournament ranking across 10M steps
    preferred_names = [
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

    selected = []
    for name in preferred_names:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            label = name.replace("dqn_ALE_Pong-v5_", "").replace(".pt", "")
            selected.append({"name": name, "label": label, "path": p})

    # Fallback if any missing
    if len(selected) < 10:
        all_pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        for p in reversed(all_pts):
            name = os.path.basename(p)
            if name not in [s["name"] for s in selected] and name != "latest.pt" and name != "champion_model.pt":
                label = name.replace("dqn_ALE_Pong-v5_", "").replace(".pt", "")
                selected.append({"name": name, "label": label, "path": p})
            if len(selected) == 10:
                break

    return selected[:10]


def parse_args():
    parser = argparse.ArgumentParser(description="Round-Robin DQN PvP Tournament")
    parser.add_argument("--target-score", type=int, default=5, help="Points needed to win each match")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Slight exploration (2%) to prevent infinite deterministic loops")
    parser.add_argument("--base-seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    return parser.parse_args()


def play_pvp_game(env, agent_right, agent_left, device: str, target_score: int, epsilon: float, seed: int) -> tuple[int, int]:
    """Play a single head-to-head game between agent_right (P1) and agent_left (P2)."""
    obs, _ = env.reset(seed=seed)
    right_stack = deque([process_frame(obs["first_0"], flip=False)] * 4, maxlen=4)
    left_stack = deque([process_frame(obs["second_0"], flip=True)] * 4, maxlen=4)

    right_pts = 0
    left_pts = 0

    max_steps = 15000
    steps = 0

    while right_pts < target_score and left_pts < target_score and steps < max_steps:
        steps += 1

        # Agent actions
        if epsilon > 0.0 and np.random.rand() < epsilon:
            act_right = np.random.randint(6)
        else:
            s_r = torch.as_tensor(np.array(right_stack), device=device).unsqueeze(0)
            with torch.no_grad():
                act_right = int(agent_right.online_net(s_r)[0].argmax().item())

        if epsilon > 0.0 and np.random.rand() < epsilon:
            act_left = np.random.randint(6)
        else:
            s_l = torch.as_tensor(np.array(left_stack), device=device).unsqueeze(0)
            with torch.no_grad():
                act_left = int(agent_left.online_net(s_l)[0].argmax().item())

        # Frame skip 4
        for _ in range(4):
            obs, rewards, term, trunc, _ = env.step({"first_0": act_right, "second_0": act_left})
            if rewards["first_0"] > 0:
                right_pts += int(rewards["first_0"])
            if rewards["second_0"] > 0:
                left_pts += int(rewards["second_0"])

            if term["first_0"] or trunc["first_0"] or term["second_0"] or trunc["second_0"]:
                break

        right_stack.append(process_frame(obs["first_0"], flip=False))
        left_stack.append(process_frame(obs["second_0"], flip=True))

    return right_pts, left_pts


def main():
    args = parse_args()
    device = get_device()

    contenders = get_top_10_checkpoints()
    n = len(contenders)

    print(f"\n==========================================================================================")
    print(f"               ATARI PONG: TOP 10 HEAD-TO-HEAD ROUND-ROBIN TOURNAMENT                     ")
    print(f"==========================================================================================")
    print(f"Contenders ({n} models):")
    for idx, c in enumerate(contenders, 1):
        print(f"  [{idx:2d}] {c['label']:<20} ({c['name']})")

    pairings = list(itertools.combinations(range(n), 2))
    total_matches = len(pairings) * 2
    print(f"\nFormat: Full Round-Robin | 45 Pairings x 2 Games (Side-Swapped) = {total_matches} Total Matches")
    print(f"Match Rules: First to {args.target_score} points | Epsilon: {args.epsilon*100:.1f}% | Device: {device}")
    print(f"------------------------------------------------------------------------------------------\n")

    # Preload all 10 agents onto GPU for fast execution
    agents = []
    for c in contenders:
        ag = DQNAgent(num_actions=6, device=device)
        ag.load(c["path"])
        agents.append(ag)

    env = pong_v3.parallel_env(render_mode=None)

    # Cross-table matrix: cross_wins[i][j] = wins of i against j
    cross_wins = np.zeros((n, n), dtype=int)
    cross_points = np.zeros((n, n), dtype=int)

    # Overall stats
    wins = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    pts_for = np.zeros(n, dtype=int)
    pts_against = np.zeros(n, dtype=int)

    match_num = 0
    match_records = []

    for idx1, idx2 in pairings:
        c1 = contenders[idx1]
        c2 = contenders[idx2]

        # Game A: c1 on Right, c2 on Left
        match_num += 1
        r_pts, l_pts = play_pvp_game(
            env, agents[idx1], agents[idx2], device, args.target_score, args.epsilon, seed=args.base_seed + match_num
        )
        pts_for[idx1] += r_pts
        pts_against[idx1] += l_pts
        pts_for[idx2] += l_pts
        pts_against[idx2] += r_pts
        cross_points[idx1][idx2] += r_pts
        cross_points[idx2][idx1] += l_pts

        if r_pts > l_pts:
            wins[idx1] += 1
            losses[idx2] += 1
            cross_wins[idx1][idx2] += 1
            winner = c1["label"]
        else:
            wins[idx2] += 1
            losses[idx1] += 1
            cross_wins[idx2][idx1] += 1
            winner = c2["label"]

        match_records.append({
            "match": match_num,
            "p1": c1["label"],
            "p2": c2["label"],
            "score": f"{r_pts}-{l_pts}",
            "winner": winner,
            "side": f"{c1['label']} (Right) vs {c2['label']} (Left)",
        })
        print(f"[{match_num:2d}/{total_matches}] Game 1: {c1['label']} {r_pts} - {l_pts} {c2['label']} -> {winner} won", flush=True)

        # Game B: Side Swap! c2 on Right, c1 on Left
        match_num += 1
        r_pts, l_pts = play_pvp_game(
            env, agents[idx2], agents[idx1], device, args.target_score, args.epsilon, seed=args.base_seed + match_num
        )
        # Note: in Game B, c2 is Right, c1 is Left
        pts_for[idx2] += r_pts
        pts_against[idx2] += l_pts
        pts_for[idx1] += l_pts
        pts_against[idx1] += r_pts
        cross_points[idx2][idx1] += r_pts
        cross_points[idx1][idx2] += l_pts

        if r_pts > l_pts:
            wins[idx2] += 1
            losses[idx1] += 1
            cross_wins[idx2][idx1] += 1
            winner = c2["label"]
        else:
            wins[idx1] += 1
            losses[idx2] += 1
            cross_wins[idx1][idx2] += 1
            winner = c1["label"]

        match_records.append({
            "match": match_num,
            "p1": c2["label"],
            "p2": c1["label"],
            "score": f"{r_pts}-{l_pts}",
            "winner": winner,
            "side": f"{c2['label']} (Right) vs {c1['label']} (Left)",
        })
        print(f"[{match_num:2d}/{total_matches}] Game 2: {c2['label']} {r_pts} - {l_pts} {c1['label']} -> {winner} won", flush=True)

    env.close()

    # Compile Leaderboard
    table = []
    total_games_per_agent = (n - 1) * 2  # 9 * 2 = 18 games each
    for i in range(n):
        diff = pts_for[i] - pts_against[i]
        win_pct = (wins[i] / total_games_per_agent) * 100.0
        table.append({
            "model": contenders[i]["name"],
            "label": contenders[i]["label"],
            "matches": total_games_per_agent,
            "wins": int(wins[i]),
            "losses": int(losses[i]),
            "win_rate": float(win_pct),
            "points_for": int(pts_for[i]),
            "points_against": int(pts_against[i]),
            "diff": int(diff),
        })

    # Sort leaderboard: 1. Win Rate, 2. Goal Differential, 3. Points For
    table.sort(key=lambda x: (-x["win_rate"], -x["diff"], -x["points_for"]))

    # Print Final League Standings
    print(f"\n==========================================================================================================")
    print(f"                                   ROUND-ROBIN PvP FINAL STANDINGS                                        ")
    print(f"==========================================================================================================")
    header = f"{'Rank':<5} | {'Model Checkpoint':<20} | {'Record (W-L)':<12} | {'Win Rate':<10} | {'PF':<5} | {'PA':<5} | {'Diff':<6}"
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(table, 1):
        medal = " 🥇" if rank == 1 else (" 🥈" if rank == 2 else (" 🥉" if rank == 3 else "   "))
        rec_str = f"{row['wins']}-{row['losses']}"
        diff_str = f"{row['diff']:+d}"
        print(f"{rank:<2}{medal} | {row['label']:<20} | {rec_str:<12} | {row['win_rate']:>6.1f}%   | {row['points_for']:<5} | {row['points_against']:<5} | {diff_str:>6}")

    print("=" * len(header))

    champ = table[0]
    print(f"\n🏆 PvP GRAND CHAMPION: {champ['label']} ({champ['model']})")
    print(f"   * Dominant Record: {champ['wins']} Wins - {champ['losses']} Losses ({champ['win_rate']:.1f}% Win Rate)")
    print(f"   * Point Differential: {champ['diff']:+d} ({champ['points_for']} Scored, only {champ['points_against']} Conceded)")
    print(f"   * Competed in {champ['matches']} Head-to-Head games against 9 other trained checkpoints!\n")

    # Save CSVs
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "round_robin_results.csv")
    pd.DataFrame(table).to_csv(csv_path, index=False)
    print(f"--> Saved standings to: {csv_path}")

    # Build and Save Cross-Table DataFrame
    labels = [c["label"] for c in contenders]
    cross_df = pd.DataFrame(cross_wins, index=labels, columns=labels)
    cross_csv_path = os.path.join(args.output_dir, "round_robin_crosstable.csv")
    cross_df.to_csv(cross_csv_path)
    print(f"--> Saved Head-to-Head Cross Table to: {cross_csv_path}")

    # Plot visual chart
    plot_round_robin_charts(table, cross_wins, labels, os.path.join(args.output_dir, "round_robin_leaderboard.png"))


def plot_round_robin_charts(table: list[dict], cross_wins: np.ndarray, labels: list[str], save_path: str):
    """Plot visual standings and cross-table heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # 1. Bar Chart of Win Rates & Points
    models = [r["label"] for r in table]
    win_rates = [r["win_rate"] for r in table]
    diffs = [r["diff"] for r in table]

    y_pos = np.arange(len(models))
    bars = ax1.barh(y_pos, win_rates, color="#3498db", edgecolor="#2980b9", alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=10, fontweight="bold")
    ax1.invert_yaxis()  # Top rank on top
    ax1.set_xlabel("Win Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Round-Robin PvP: Win Rate & Differential", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlim(0, 105)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        diff_val = diffs[i]
        diff_txt = f"+{diff_val}" if diff_val > 0 else str(diff_val)
        ax1.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}% ({table[i]['wins']}-{table[i]['losses']}, Diff: {diff_txt})", va="center", fontsize=9, fontweight="bold")

    # 2. Cross Table Heatmap
    im = ax2.imshow(cross_wins, cmap="YlGnBu", vmin=0, vmax=2)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_title("Head-to-Head Cross Table (Row Wins vs Column)", fontsize=13, fontweight="bold", pad=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax2.text(j, i, "-", ha="center", va="center", color="gray", fontsize=10, fontweight="bold")
            else:
                ax2.text(j, i, str(cross_wins[i, j]), ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"--> Saved visual tournament chart to: {save_path}")


if __name__ == "__main__":
    main()
