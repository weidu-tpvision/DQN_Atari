"""Plot and visualize DQN training and evaluation metrics from CSV logs.

Usage:
    # Plot a single run (automatically finds training_metrics.csv and eval_metrics.csv in dir):
    python plot_metrics.py --dir checkpoints --output learning_progress.png

    # Or specify explicit files:
    python plot_metrics.py --train-log checkpoints/training_metrics.csv --eval-log checkpoints/eval_metrics.csv
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot and visualize DQN training & evaluation metrics")
    parser.add_argument("--dir", type=str, default="checkpoints", help="Directory containing metric CSVs")
    parser.add_argument("--train-log", type=str, default=None, help="Path to training_metrics.csv")
    parser.add_argument("--eval-log", type=str, default=None, help="Path to eval_metrics.csv")
    parser.add_argument("--output", type=str, default="learning_progress.png", help="Path to save output figure")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Display the plot in a window on desktop (--no-show to only save)")
    return parser.parse_args()


def main():
    args = parse_args()
    train_log = args.train_log or os.path.join(args.dir, "training_metrics.csv")
    eval_log = args.eval_log or os.path.join(args.dir, "eval_metrics.csv")

    has_train = os.path.exists(train_log) and os.path.getsize(train_log) > 0
    has_eval = os.path.exists(eval_log) and os.path.getsize(eval_log) > 0

    if not has_train and not has_eval:
        print(f"Error: Neither {train_log} nor {eval_log} could be found!")
        return

    train_df = pd.read_csv(train_log) if has_train else None
    eval_df = pd.read_csv(eval_log) if has_eval else None

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # 1. Performance / Score Panel: Training Rolling Score vs. True Greedy Evaluation Score
    ax1 = axes[0]
    if has_train and "avg_reward_100" in train_df.columns:
        ax1.plot(train_df["step"], train_df["avg_reward_100"], label="Training (100-Game Rolling, Clipped Rewards, with Exploration)", color="#1f77b4", alpha=0.8, linewidth=2)
    if has_eval and "mean_score" in eval_df.columns:
        ax1.plot(eval_df["step"], eval_df["mean_score"], label="Evaluation (Greedy Policy, Epsilon=0)", color="#2ca02c", marker="o", linewidth=2.5)
        if "std_score" in eval_df.columns:
            ax1.fill_between(
                eval_df["step"],
                eval_df["mean_score"] - eval_df["std_score"],
                eval_df["mean_score"] + eval_df["std_score"],
                color="#2ca02c",
                alpha=0.2,
                label="Eval +/- 1 StdDev",
            )
    ax1.set_title("Agent Score: Training vs. Evaluation", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Environment Steps", fontsize=11)
    ax1.set_ylabel("Raw Atari Game Score", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    # 2. Bellman TD Huber Loss
    ax2 = axes[1]
    if has_train and "loss" in train_df.columns:
        ax2.plot(train_df["step"], train_df["loss"], color="#d62728", alpha=0.7, linewidth=1.5)
        # Add smoothed rolling mean if many points
        if len(train_df) > 20:
            smoothed = train_df["loss"].rolling(window=10, min_periods=1).mean()
            ax2.plot(train_df["step"], smoothed, color="#8b0000", linewidth=2, label="Smoothed Loss (10-window)")
            ax2.legend()
    ax2.set_title("Bellman TD Huber Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Environment Steps", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 3. Predicted Q-Value (Action-Value)
    ax3 = axes[2]
    if has_train and "mean_q" in train_df.columns:
        ax3.plot(train_df["step"], train_df["mean_q"], color="#9467bd", linewidth=2)
    ax3.set_title("Mean Estimated Action-Value (Q-Value)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Environment Steps", fontsize=11)
    ax3.set_ylabel("Average Q(s, a)", fontsize=11)
    ax3.grid(True, alpha=0.3)

    # 4. Epsilon Decay & Throughput
    ax4 = axes[3]
    if has_train and "epsilon" in train_df.columns:
        ax4.plot(train_df["step"], train_df["epsilon"], color="#ff7f0e", linestyle="--", linewidth=2, label="Epsilon (Exploration)")
        ax4.set_ylabel("Epsilon", color="#ff7f0e", fontsize=11)
        ax4.tick_params(axis="y", labelcolor="#ff7f0e")
        ax4.set_ylim(-0.05, 1.05)

    if has_train and "fps" in train_df.columns:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(train_df["step"], train_df["fps"], color="#7f7f7f", alpha=0.4, linestyle=":", label="Throughput (FPS)")
        ax4_twin.set_ylabel("Throughput (FPS)", color="#7f7f7f", fontsize=11)
        ax4_twin.tick_params(axis="y", labelcolor="#7f7f7f")

    ax4.set_title("Exploration Schedule (Epsilon)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Environment Steps", fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"--> Saved training & evaluation progress plot to: {args.output}")

    # Display plot in a window if DISPLAY is available and --show is True
    if args.show and os.environ.get("DISPLAY"):
        try:
            import cv2
            img = cv2.imread(args.output)
            if img is not None:
                window_name = "DQN Training & Evaluation Dashboard"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1100, 750)
                cv2.imshow(window_name, img)
                print("--> Displaying plot window on your screen (Press 'q', ESC, or any key to close)...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Could not open display window ({e}). You can view {args.output} directly.")


if __name__ == "__main__":
    main()
