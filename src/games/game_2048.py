"""Core game engine for 2048 implemented in pure NumPy.

Rules & Mechanics:
- 4x4 grid of tiles.
- Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT.
- Tiles slide as far as they can in the chosen direction.
- Adjacent identical tiles merge into one with double the value.
- Merges occur once per tile per move in the direction of the slide.
- After every legal move that changes the board, a new tile is spawned
  in an empty cell (90% chance of 2, 10% chance of 4).
- Game over occurs when no empty cells remain and no adjacent merges are possible.
"""

from typing import Tuple, List
import numpy as np


class Game2048:
    """High-performance 2048 engine in pure NumPy."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        """Reset board to two initial random tiles."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.board.fill(0)
        self.score = 0
        self.moves = 0
        self._spawn_tile()
        self._spawn_tile()
        return self.board.copy()

    def _spawn_tile(self) -> bool:
        """Spawn a tile in a random empty cell (90% 2, 10% 4). Returns True if spawned."""
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            return False
        idx = self.rng.randint(0, len(empty_cells))
        r, c = empty_cells[idx]
        val = 2 if self.rng.random_sample() < 0.9 else 4
        self.board[r, c] = val
        return True

    @staticmethod
    def _slide_and_merge_row(row: np.ndarray) -> Tuple[np.ndarray, int]:
        """Slide and merge a single 1D row to the left.
        
        Example:
            [2, 0, 2, 4] -> [4, 4, 0, 0], gained_score = 4
            [2, 2, 2, 2] -> [4, 4, 0, 0], gained_score = 8
            [4, 0, 0, 0] -> [4, 0, 0, 0], gained_score = 0
        """
        # 1. Compress non-zeros to the left
        non_zeros = row[row != 0]
        if len(non_zeros) == 0:
            return np.zeros(4, dtype=np.int32), 0

        new_row = np.zeros(4, dtype=np.int32)
        score_gained = 0
        write_idx = 0
        skip = False

        for i in range(len(non_zeros)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i + 1]:
                merged_val = non_zeros[i] * 2
                new_row[write_idx] = merged_val
                score_gained += merged_val
                skip = True
            else:
                new_row[write_idx] = non_zeros[i]
            write_idx += 1

        return new_row, score_gained

    def simulate_move(self, action: int) -> Tuple[np.ndarray, int, bool]:
        """Simulate an action on a copy of the board.
        
        Returns:
            new_board: 4x4 array
            points_gained: score delta from merges
            changed: whether the board actually changed
        """
        temp_board = self.board.copy()
        points = 0

        # Rotate board so the slide direction aligns with sliding rows to the left
        # Action mappings:
        # LEFT: 0 rotations
        # UP: rotate 90 deg counter-clockwise (1 rot90) -> rows become columns
        # RIGHT: flip horizontal (or 2 rot90)
        # DOWN: rotate 270 deg counter-clockwise (3 rot90)
        if action == self.LEFT:
            rot = 0
        elif action == self.UP:
            rot = 1
        elif action == self.RIGHT:
            rot = 2
        elif action == self.DOWN:
            rot = 3
        else:
            raise ValueError(f"Invalid action: {action}")

        # Rotate to slide LEFT
        rotated = np.rot90(temp_board, rot)
        new_rotated = np.zeros_like(rotated)

        for r in range(4):
            new_row, gained = self._slide_and_merge_row(rotated[r])
            new_rotated[r] = new_row
            points += gained

        # Rotate back
        new_board = np.rot90(new_rotated, -rot)
        changed = not np.array_equal(self.board, new_board)
        return new_board, points, changed

    def step(self, action: int) -> Tuple[bool, int, bool]:
        """Execute a move on the board.
        
        Returns:
            changed: whether the board changed
            points: score gained
            game_over: whether no legal moves remain
        """
        new_board, points, changed = self.simulate_move(action)
        if changed:
            self.board = new_board
            self.score += points
            self.moves += 1
            self._spawn_tile()

        done = self.is_game_over()
        return changed, points, done

    def get_legal_actions_mask(self) -> np.ndarray:
        """Returns boolean array of shape (4,) where True means the move is legal."""
        mask = np.zeros(4, dtype=bool)
        for act in range(4):
            _, _, changed = self.simulate_move(act)
            mask[act] = changed
        return mask

    def is_game_over(self) -> bool:
        """Game is over if no empty cells exist and no adjacent tiles can merge."""
        # 1. Any empty cells?
        if np.any(self.board == 0):
            return False

        # 2. Any horizontal merges possible?
        if np.any(self.board[:, :-1] == self.board[:, 1:]):
            return False

        # 3. Any vertical merges possible?
        if np.any(self.board[:-1, :] == self.board[1:, :]):
            return False

        return True

    def get_max_tile(self) -> int:
        """Return the highest tile on the board."""
        return int(np.max(self.board))
