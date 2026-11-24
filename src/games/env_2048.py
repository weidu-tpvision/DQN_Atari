"""Gymnasium environment wrapper for 2048."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.games.game_2048 import Game2048


def board_to_one_hot(board: np.ndarray, num_classes: int = 16) -> np.ndarray:
    """Encode 4x4 integer board into (num_classes, 4, 4) uint8 one-hot tensor.
    
    Channel 0 corresponds to empty cell (0).
    Channel k (1..15) corresponds to tile 2^k (e.g. 1 -> 2, 2 -> 4, ..., 11 -> 2048).
    """
    one_hot = np.zeros((num_classes, 4, 4), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            val = board[r, c]
            if val == 0:
                idx = 0
            else:
                power = int(round(np.log2(val)))
                idx = min(power, num_classes - 1)
            one_hot[idx, r, c] = 1
    return one_hot


class Env2048(gym.Env):
    """Gymnasium Environment for 2048.

    Reward per move (`reward`):
        "log"    log2 of the points gained by the move (0 if nothing merged). Compresses big merges:
                 a 2048 merge is worth 11, a 4 merge 2.
        "linear" the points gained (the game score increment), times `reward_scale`. Keeps big
                 late-game merges large, so they still matter after discounting.
    Illegal moves (board unchanged) give -0.5 in both modes; they never happen with action masking.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 5000,
                 reward: str = "log", reward_scale: float = 1.0):
        super().__init__()
        if reward not in ("log", "linear"):
            raise ValueError(f"reward must be 'log' or 'linear', got {reward!r}")
        self.reward_mode = reward
        self.reward_scale = reward_scale
        self.game = Game2048()
        self.action_space = spaces.Discrete(4)  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        
        # (16, 4, 4) binary one-hot tensor stored as uint8
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16, 4, 4), dtype=np.uint8
        )
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        board = self.game.reset(seed=seed)
        self.current_step = 0
        info = {
            "score": self.game.score,
            "max_tile": self.game.get_max_tile(),
            "legal_mask": self.game.get_legal_actions_mask(),
        }
        return self._get_obs(board), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        changed, points, done = self.game.step(action)

        if not changed:
            reward = -0.5
        elif points == 0:
            reward = 0.0
        elif self.reward_mode == "log":
            reward = float(np.log2(points)) * self.reward_scale
        else:
            reward = float(points) * self.reward_scale

        truncated = (self.current_step >= self.max_steps)
        info = {
            "score": self.game.score,
            "max_tile": self.game.get_max_tile(),
            "moves": self.game.moves,
            "points_gained": points,
            "legal_mask": self.game.get_legal_actions_mask(),
            "changed": changed,
        }
        return self._get_obs(self.game.board), reward, done, truncated, info

    def legal_actions_mask(self) -> np.ndarray:
        """Returns boolean mask of shape (4,) for legal moves."""
        return self.game.get_legal_actions_mask()

    def _get_obs(self, board: np.ndarray) -> np.ndarray:
        return board_to_one_hot(board, num_classes=16)

    def render(self) -> Optional[str]:
        """Render board to terminal."""
        ansi_str = "\n" + "=" * 25 + "\n"
        ansi_str += f" Score: {self.game.score:<6} | Moves: {self.game.moves:<5}\n"
        ansi_str += "-" * 25 + "\n"
        for row in self.game.board:
            row_str = "| " + " ".join(f"{val:4d}" if val > 0 else "   ." for val in row) + " |\n"
            ansi_str += row_str
        ansi_str += "=" * 25 + "\n"
        if self.render_mode == "human":
            print(ansi_str)
        return ansi_str
