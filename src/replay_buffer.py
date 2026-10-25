"""Experience Replay Buffer for DQN.

Reference:
    Lin, L.-J. "Self-improving reactive agents based on reinforcement learning,
    planning and mental practice." Machine Learning 8, 293–321 (1992).
    Mnih et al., Nature 518, 529–533 (2015).
"""

from typing import Tuple
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
    ):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
            state_shape: Shape of the stacked observation tensor, e.g. (4, 84, 84).
            device: PyTorch device to move sampled batches to.
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

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition (s, a, r, s', done) in the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a uniform random minibatch of transitions.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) transferred to `self.device`.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.bool, device=self.device)

        return states, actions, rewards, next_states, dones

    def save(self, path: str) -> None:
        """Save replay buffer state and arrays to compressed npz file atomically."""
        tmp_path = path + ".tmp.npz"
        np.savez_compressed(
            tmp_path,
            ptr=self.ptr,
            size=self.size,
            states=self.states[:self.size],
            next_states=self.next_states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            dones=self.dones[:self.size],
        )
        import os
        os.replace(tmp_path, path)

    def load(self, path: str) -> None:
        """Load replay buffer state from compressed npz file."""
        data = np.load(path)
        self.ptr = int(data["ptr"])
        self.size = int(data["size"])
        loaded_size = len(data["actions"])
        self.states[:loaded_size] = data["states"]
        self.next_states[:loaded_size] = data["next_states"]
        self.actions[:loaded_size] = data["actions"]
        self.rewards[:loaded_size] = data["rewards"]
        self.dones[:loaded_size] = data["dones"]

    def __len__(self) -> int:
        return self.size
