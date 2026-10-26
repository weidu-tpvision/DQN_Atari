"""Round-Robin Head-to-Head Tournament between up to 10 Atari DQN checkpoints.

Every contender plays every other contender in direct PvP battles.
Each pair plays 2 matches (home and away side-swap) so neither side is favored.

Rules:
    - N Contenders -> N(N-1)/2 Pairings x 2 Games (10 contenders: 45 x 2 = 90 matches)
    - First to target-score (default: 5 points per match); a match still level after
      15,000 agent steps is a draw (half a win for each side in the win rate)
    - Full league table: Wins, Draws, Losses, Win Rate, Points For (PF), Points Against (PA), Differential (+/-)
    - NxN Head-to-Head Cross Table Heatmap and Leaderboard Chart

Usage:
    python round_robin_tournament.py
    python round_robin_tournament.py --target-score 5
"""

import os
import glob
import itertools
import argparse
import numpy as np
import pandas as pd

from src.agent import DQNAgent
from src.pong2p import TwoPlayerPong, NUM_ACTIONS, policy_action
from src.utils import get_device, set_seed


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
    parser.add_argument("--epsilon", type=float, default=0.02, help="Slight exploration (2%%) to prevent infinite deterministic loops")
    parser.add_argument("--base-seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    return parser.parse_args()


def play_pvp_game(game: TwoPlayerPong, agent_right, agent_left, target_score: int, epsilon: float, seed: int,
                  max_steps: int = 15000) -> tuple[int, int]:
    """Play a single head-to-head game between agent_right (P1) and agent_left (P2)."""
    s_r, s_l = game.reset(seed=seed)
    right_pts = 0
    left_pts = 0

    for _ in range(max_steps):
        act_right = policy_action(agent_right, s_r, epsilon)
        act_left = policy_action(agent_left, s_l, epsilon)
        r_gain, l_gain, done = game.step(act_right, act_left)
        right_pts += r_gain
        left_pts += l_gain
        s_r, s_l = game.observe()
        if done or right_pts >= target_score or left_pts >= target_score:
            break

    return right_pts, left_pts


def record_result(stats: dict, cross_wins: np.ndarray, right: int, left: int, r_pts: int, l_pts: int) -> int:
    """Update league stats for one game; returns the winner index or -1 for a draw."""
    stats["pts_for"][right] += r_pts
    stats["pts_against"][right] += l_pts
    stats["pts_for"][left] += l_pts
    stats["pts_against"][left] += r_pts
    if r_pts == l_pts:
        stats["draws"][right] += 1
        stats["draws"][left] += 1
        return -1
    winner, loser = (right, left) if r_pts > l_pts else (left, right)
    stats["wins"][winner] += 1
    stats["losses"][loser] += 1
    cross_wins[winner][loser] += 1
    return winner


def main():
    args = parse_args()
    set_seed(args.base_seed)
    device = get_device()

    contenders = get_top_10_checkpoints()
    n = len(contenders)

    print(f"\n==========================================================================================")
    print(f"               ATARI PONG: HEAD-TO-HEAD ROUND-ROBIN TOURNAMENT                            ")
    print(f"==========================================================================================")
    print(f"Contenders ({n} models):")
    for idx, c in enumerate(contenders, 1):
        print(f"  [{idx:2d}] {c['label']:<20} ({c['name']})")

    if n < 2:
        print("Error: need at least 2 checkpoints for a round robin.")
        return

    pairings = list(itertools.combinations(range(n), 2))
    total_matches = len(pairings) * 2
    print(f"\nFormat: Full Round-Robin | {len(pairings)} Pairings x 2 Games (Side-Swapped) = {total_matches} Total Matches")
    print(f"Match Rules: First to {args.target_score} points | Epsilon: {args.epsilon*100:.1f}% | Device: {device}")
    print(f"------------------------------------------------------------------------------------------\n")

    # Preload all agents onto the device for fast execution
    agents = []
    for c in contenders:
        ag = DQNAgent(num_actions=NUM_ACTIONS, device=device)
        ag.load_weights(c["path"])
        agents.append(ag)

    game = TwoPlayerPong()

    # Cross-table matrix: cross_wins[i][j] = wins of i against j
    cross_wins = np.zeros((n, n), dtype=int)
    stats = {k: np.zeros(n, dtype=int) for k in ("wins", "draws", "losses", "pts_for", "pts_against")}

    match_num = 0
    for idx1, idx2 in pairings:
        # Game 1: idx1 on Right, idx2 on Left. Game 2: side swap.
        for game_no, (right, left) in enumerate([(idx1, idx2), (idx2, idx1)], 1):
            match_num += 1
            r_pts, l_pts = play_pvp_game(
                game, agents[right], agents[left], args.target_score, args.epsilon, seed=args.base_seed + match_num
            )
            winner = record_result(stats, cross_wins, right, left, r_pts, l_pts)
            outcome = "draw" if winner < 0 else f"{contenders[winner]['label']} won"
            print(f"[{match_num:2d}/{total_matches}] Game {game_no}: {contenders[right]['label']} {r_pts} - {l_pts} "
                  f"{contenders[left]['label']} -> {outcome}", flush=True)

    game.close()

    # Compile Leaderboard
    table = []
    total_games_per_agent = (n - 1) * 2
    for i in range(n):
        diff = stats["pts_for"][i] - stats["pts_against"][i]
        win_pct = (stats["wins"][i] + 0.5 * stats["draws"][i]) / total_games_per_agent * 100.0
        table.append({
            "model": contenders[i]["name"],
            "label": contenders[i]["label"],
            "matches": total_games_per_agent,
            "wins": int(stats["wins"][i]),
            "draws": int(stats["draws"][i]),
            "losses": int(stats["losses"][i]),
            "win_rate": float(win_pct),
            "points_for": int(stats["pts_for"][i]),
            "points_against": int(stats["pts_against"][i]),
            "diff": int(diff),
        })

    # Sort leaderboard: 1. Win Rate, 2. Goal Differential, 3. Points For
    table.sort(key=lambda x: (-x["win_rate"], -x["diff"], -x["points_for"]))

    # Print Final League Standings
    print(f"\n==========================================================================================================")
    print(f"                                   ROUND-ROBIN PvP FINAL STANDINGS                                        ")
    print(f"==========================================================================================================")
    header = f"{'Rank':<5} | {'Model Checkpoint':<20} | {'Record (W-D-L)':<14} | {'Win Rate':<10} | {'PF':<5} | {'PA':<5} | {'Diff':<6}"
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(table, 1):
        medal = " 🥇" if rank == 1 else (" 🥈" if rank == 2 else (" 🥉" if rank == 3 else "   "))
        rec_str = f"{row['wins']}-{row['draws']}-{row['losses']}"
        diff_str = f"{row['diff']:+d}"
        print(f"{rank:<2}{medal} | {row['label']:<20} | {rec_str:<14} | {row['win_rate']:>6.1f}%   | {row['points_for']:<5} | {row['points_against']:<5} | {diff_str:>6}")

    print("=" * len(header))

    champ = table[0]
    print(f"\n🏆 PvP GRAND CHAMPION: {champ['label']} ({champ['model']})")
    print(f"   * Record: {champ['wins']} Wins - {champ['draws']} Draws - {champ['losses']} Losses ({champ['win_rate']:.1f}% Win Rate)")
    print(f"   * Point Differential: {champ['diff']:+d} ({champ['points_for']} Scored, only {champ['points_against']} Conceded)")
    print(f"   * Competed in {champ['matches']} Head-to-Head games against {n - 1} other trained checkpoints!\n")

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
        ax1.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}% ({table[i]['wins']}-{table[i]['draws']}-{table[i]['losses']}, Diff: {diff_txt})", va="center", fontsize=9, fontweight="bold")

    # 2. Cross Table Heatmap
    ax2.imshow(cross_wins, cmap="YlGnBu", vmin=0, vmax=2)
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
