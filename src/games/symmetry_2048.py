"""The 8 board symmetries of 2048 (rotations and reflections) for data augmentation.

2048 plays the same on a rotated or mirrored board, provided the move is rotated/mirrored too
(e.g. mirror the board left-right and LEFT becomes RIGHT). Training on randomly transformed copies
of replayed transitions gives up to 8x the data for rare late-game positions.

The move remapping isn't written by hand: `ACTION_PERMS` is derived from the game engine itself
(move a on the original board == move ACTION_PERMS[t][a] on the transformed board).
"""

from typing import List, Tuple

import numpy as np
import torch

from src.games.game_2048 import Game2048

# (number of 90-degree counter-clockwise rotations, mirror left-right afterwards)
SYMMETRIES: List[Tuple[int, bool]] = [(k, flip) for flip in (False, True) for k in range(4)]


def transform_board(board: np.ndarray, t: int) -> np.ndarray:
    """Apply symmetry t to the last two (row, col) axes of a numpy array."""
    k, flip = SYMMETRIES[t]
    out = np.rot90(board, k, axes=(-2, -1))
    return np.flip(out, axis=-1) if flip else out


def transform_tensor(x: torch.Tensor, t: int) -> torch.Tensor:
    """Same as `transform_board` for a torch tensor (..., 4, 4)."""
    k, flip = SYMMETRIES[t]
    out = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(out, dims=(-1,)) if flip else out


def _derive_action_perms(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    game = Game2048(seed=seed)
    perms = np.zeros((len(SYMMETRIES), 4), dtype=np.int64)
    for t in range(len(SYMMETRIES)):
        for a in range(4):
            candidates = set(range(4))
            # Keep only moves consistent on several random boards until one remains
            while len(candidates) > 1:
                board = np.where(rng.random((4, 4)) < 0.3, 0, 2 ** rng.integers(1, 5, (4, 4))).astype(np.int32)
                game.board = board
                expected = transform_board(game.simulate_move(a)[0], t)
                game.board = np.ascontiguousarray(transform_board(board, t))
                candidates = {b for b in candidates if np.array_equal(game.simulate_move(b)[0], expected)}
            perms[t, a] = candidates.pop()
    return perms


ACTION_PERMS = _derive_action_perms()


def augment_batch(
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    next_masks: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an independent random symmetry to each transition of a batch.

    Rewards and dones are unchanged by a symmetry, so they aren't touched. Legal-move masks are
    remapped like actions: new_mask[perm[a]] = mask[a].
    """
    tids = torch.randint(0, len(SYMMETRIES), (states.shape[0],), device=states.device)
    states, next_states, actions = states.clone(), next_states.clone(), actions.clone()
    next_masks = next_masks.clone() if next_masks is not None else None
    perms = torch.as_tensor(ACTION_PERMS, device=states.device)
    for t in range(1, len(SYMMETRIES)):  # t = 0 is the identity
        idx = (tids == t).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        states[idx] = transform_tensor(states[idx], t)
        next_states[idx] = transform_tensor(next_states[idx], t)
        actions[idx] = perms[t][actions[idx]]
        if next_masks is not None:
            remapped = torch.zeros_like(next_masks[idx])
            remapped[:, perms[t]] = next_masks[idx]
            next_masks[idx] = remapped
    return states, actions, next_states, next_masks
