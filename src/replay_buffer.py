"""Experience Replay Buffer for DQN.

Reference:
    Lin, L.-J. "Self-improving reactive agents based on reinforcement learning,
    planning and mental practice." Machine Learning 8, 293–321 (1992).
    Mnih et al., Nature 518, 529–533 (2015).
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity circular replay buffer storing transitions in pre-allocated numpy arrays.

    Using numpy arrays with uint8 storage keeps RAM overhead low and makes batch sampling
    vectorized and extremely fast.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        state_shape: Tuple[int, ...] = (4, 84, 84),
        device: torch.device = torch.device("cpu"),
        action_mask_dim: Optional[int] = None,
    ):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            state_shape: Shape of the stacked observation tensor, e.g. (4, 84, 84).
            device: PyTorch device to move sampled batches to.
            action_mask_dim: If set, also store the legal-action mask of each next state
                (for games with illegal moves); `sample()` then returns it as a 6th element.
        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate contiguous memory buffers
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_masks = np.ones((capacity, action_mask_dim), dtype=np.bool_) if action_mask_dim else None

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition (s, a, r, s', done[, legal mask of s']) in the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        if self.next_masks is not None:
            self.next_masks[self.ptr] = True if next_mask is None else next_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 32) -> Tuple[torch.Tensor, ...]:
        """Sample a uniform random minibatch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) transferred to `self.device`,
            plus next_masks (batch, action_mask_dim) bool if the buffer stores action masks.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.bool, device=self.device)

        if self.next_masks is not None:
            next_masks = torch.as_tensor(self.next_masks[indices], dtype=torch.bool, device=self.device)
            return states, actions, rewards, next_states, dones, next_masks
        return states, actions, rewards, next_states, dones

    def save(self, path: str, *, checkpoint_id: str = "") -> None:
        """Save replay buffer state and arrays to compressed npz file atomically."""
        tmp_path = path + ".tmp.npz"
        arrays = {k: getattr(self, k)[:self.size] for k in self._keys()}
        try:
            np.savez_compressed(tmp_path, checkpoint_id=checkpoint_id, ptr=self.ptr, size=self.size, **arrays)
            os.replace(tmp_path, path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def load(self, path: str, *, checkpoint_id: str | None = None) -> None:
        """Load replay buffer state from compressed npz file.

        Handles a capacity different from the one the buffer was saved with: the most recent
        min(saved_size, capacity) transitions are kept in chronological order.
        """
        with np.load(path, allow_pickle=False) as data:
            if checkpoint_id is not None and (
                "checkpoint_id" not in data or data["checkpoint_id"].item() != checkpoint_id
            ):
                raise ValueError(f"Replay file {path} does not match the checkpoint that references it.")
            self._load_arrays(data)

    def _load_arrays(self, data) -> None:
        saved_ptr = int(data["ptr"])
        loaded_size = len(data["actions"])
        if data["states"].shape[1:] != self.states.shape[1:]:
            raise ValueError(
                f"Replay buffer state shape {data['states'].shape[1:]} does not match {self.states.shape[1:]}"
            )

        keys = self._keys()
        if self.next_masks is not None and "next_masks" not in data.files:
            # Legacy buffers may have been saved with action masking disabled.
            self.next_masks.fill(True)
            keys = keys[:-1]
        same_layout = loaded_size == self.capacity or (loaded_size < self.capacity and saved_ptr == loaded_size)
        if same_layout:
            for k in keys:
                getattr(self, k)[:loaded_size] = data[k]
            self.ptr = saved_ptr % self.capacity
            self.size = loaded_size
            return

        # Reorder oldest -> newest (a full saved buffer starts at saved_ptr), keep the newest part
        order = np.roll(np.arange(loaded_size), -saved_ptr)
        keep = order[-min(loaded_size, self.capacity):]
        for k in keys:
            getattr(self, k)[:len(keep)] = data[k][keep]
        self.size = len(keep)
        self.ptr = self.size % self.capacity

    def _keys(self) -> Tuple[str, ...]:
        keys = ("states", "next_states", "actions", "rewards", "dones")
        return keys + ("next_masks",) if self.next_masks is not None else keys

    def save_for_checkpoint(self, checkpoint_path: str) -> dict:
        path = Path(checkpoint_path)
        checkpoint_id = uuid4().hex
        replay_path = path.with_name(f"{path.stem}_replay_{checkpoint_id}.npz")
        self.save(str(replay_path), checkpoint_id=checkpoint_id)
        return {"file": replay_path.name, "id": checkpoint_id}

    @staticmethod
    def discard_checkpoint_snapshot(checkpoint_path: str, replay: dict) -> None:
        """Remove an owned versioned snapshot without touching legacy or external replay files."""
        if not isinstance(replay, dict):
            return
        path = Path(checkpoint_path)
        name = replay.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            return
        if name != f"{path.stem}_replay_{replay.get('id')}.npz":
            return
        try:
            path.with_name(name).unlink(missing_ok=True)
        except OSError:
            # A cleanup failure should not invalidate a published checkpoint.
            pass

    def restore_for_checkpoint(self, checkpoint_path: str, checkpoint: dict, override: str | None) -> bool:
        if override is not None:
            self.load(override)
            return True

        replay = checkpoint.get("replay")
        if replay is None:
            return False

        name = replay["file"]
        if Path(name).name != name:
            raise ValueError(f"Invalid replay filename in checkpoint: {name}")
        replay_path = Path(checkpoint_path).with_name(name)
        self.load(str(replay_path), checkpoint_id=replay["id"])
        return True

    def __len__(self) -> int:
        return self.size
