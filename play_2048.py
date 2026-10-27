"""Interactive 2048 Player and AI Visualizer.

Modes:
1. Human Play:
    python play_2048.py --mode human

2. Watch Trained AI:
    python play_2048.py --mode ai --checkpoint checkpoints_2048/best_model.pt --delay 0.15
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from src.games.game_2048 import Game2048
from src.games.env_2048 import Env2048
from src.model_2048 import DQN2048
from src.agent import DQNAgent
from src.utils import get_device

# ANSI color codes for 2048 tiles
TILE_COLORS = {
    0: "\033[90m",        # Empty (gray)
    2: "\033[37m",        # White
    4: "\033[97m",        # Bright White
    8: "\033[93m",        # Light Yellow
    16: "\033[33m",       # Dark Yellow / Orange
    32: "\033[91m",       # Light Red
    64: "\033[31m",       # Red
    128: "\033[96m",      # Light Cyan
    256: "\033[36m",      # Cyan
    512: "\033[92m",      # Light Green
    1024: "\033[32m",     # Green
    2048: "\033[1;93;41m",# Bold Yellow on Red
    4096: "\033[1;97;45m",# Bold White on Magenta
}
RESET_COLOR = "\033[0m"


def print_colored_board(board: np.ndarray, score: int, moves: int, last_act: str = ""):
    """Print beautifully styled terminal board with tile colors."""
    os.system("cls" if os.name == "nt" else "clear")
    print(f"\n  ========================================")
    print(f"               2048 ARENA               ")
    print(f"  ========================================")
    print(f"   Score: {score:<7} | Moves: {moves:<5} | Last: {last_act}")
    print(f"  +---------+---------+---------+---------+")

    for row in board:
        line_top = "  |"
        line_mid = "  |"
        line_bot = "  |"
        for val in row:
            color = TILE_COLORS.get(val, "\033[1;97;44m")
            if val == 0:
                s = "       "
            else:
                s = f"{val:^7}"
            line_top += f"         |"
            line_mid += f" {color}{s}{RESET_COLOR} |"
            line_bot += f"         |"
        print(line_top)
        print(line_mid)
        print(line_bot)
        print(f"  +---------+---------+---------+---------+")
    print()


def play_human():
    """Run interactive human game using terminal input."""
    import termios
    import tty

    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # Arrow keys
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                return f"{ch}{ch2}{ch3}"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    env = Env2048()
    obs, info = env.reset()
    last_action_str = "START"

    key_map = {
        "w": Game2048.UP,
        "s": Game2048.DOWN,
        "a": Game2048.LEFT,
        "d": Game2048.RIGHT,
        "\x1b[A": Game2048.UP,
        "\x1b[B": Game2048.DOWN,
        "\x1b[D": Game2048.LEFT,
        "\x1b[C": Game2048.RIGHT,
    }

    try:
        while True:
            print_colored_board(env.game.board, env.game.score, env.game.moves, last_action_str)
            if env.game.is_game_over():
                print(f"  GAME OVER! Final Score: {env.game.score} | Highest Tile: {env.game.get_max_tile()}\n")
                break

            print("  Controls: [W/A/S/D] or Arrow Keys to move | [Q] to quit: ", end="", flush=True)
            key = get_key()
            if key.lower() == "q" or key == "\x03":
                print("\nQuitting...")
                break

            if key in key_map:
                act = key_map[key]
                obs, reward, done, trunc, info = env.step(act)
                last_action_str = Game2048.ACTION_NAMES[act]
                if not info["changed"]:
                    last_action_str += " (ILLEGAL MOVE)"
    except KeyboardInterrupt:
        print("\nExiting.")


def play_ai(checkpoint_path: str, delay: float = 0.15):
    """Watch trained DQN agent play 2048."""
    device = get_device()
    agent = DQNAgent(
        num_actions=4,
        device=device,
        model_fn=lambda: DQN2048(in_channels=16, num_actions=4),
    )
    print(f"Loading trained weights from {checkpoint_path}...")
    agent.load_weights(checkpoint_path)

    env = Env2048()
    obs, info = env.reset()
    done = False
    last_action_str = "START"

    try:
        while not done:
            print_colored_board(env.game.board, env.game.score, env.game.moves, last_action_str)
            mask = info["legal_mask"]
            act = agent.select_action(obs, deterministic=True, action_mask=mask)
            
            # Read Q-values for display
            with torch.no_grad():
                s_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                q_vals = agent.online_net(s_t).squeeze(0).cpu().numpy()
            
            q_str = " | ".join(f"{Game2048.ACTION_NAMES[i]}: {q_vals[i]:5.2f}" for i in range(4))
            print(f"  AI Q-Values:  {q_str}")
            print(f"  Action Chosen: \033[1;32m{Game2048.ACTION_NAMES[act]}\033[0m")
            time.sleep(delay)

            obs, reward, done, trunc, info = env.step(act)
            last_action_str = Game2048.ACTION_NAMES[act]

        print_colored_board(env.game.board, env.game.score, env.game.moves, last_action_str)
        print(f"  GAME FINISHED! Final Score: {env.game.score} | Max Tile: {env.game.get_max_tile()}\n")

    except KeyboardInterrupt:
        print("\nExiting AI visualizer.")


def main():
    parser = argparse.ArgumentParser(description="2048 Game & AI Visualizer")
    parser.add_argument("--mode", type=str, choices=["human", "ai"], default="human", help="Play mode")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_2048/best_model.pt", help="Path to checkpoint for AI mode")
    parser.add_argument("--delay", type=float, default=0.15, help="Step delay in seconds for AI mode")
    args = parser.parse_args()

    if args.mode == "human":
        play_human()
    else:
        play_ai(args.checkpoint, delay=args.delay)


if __name__ == "__main__":
    main()
