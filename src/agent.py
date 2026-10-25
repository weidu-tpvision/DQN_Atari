"""Deep Q-Network Agent (Nature 2015).

Manages:
- Online Network Q(s, a; θ) and Target Network Q(s, a; θ^-)
- Epsilon-greedy action selection with linear decay
- Bellman loss computation with Huber Loss (Smooth L1)
- Periodic target network weight synchronization
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from src.model import NatureDQN


class DQNAgent:
    """Nature 2015 DQN Agent."""

    def __init__(
        self,
        num_actions: int,
        in_channels: int = 4,
        gamma: float = 0.99,
        lr: float = 1e-4,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_decay_steps: int = 1_000_000,
        target_sync_freq: int = 10_000,
        double_dqn: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_sync_freq = target_sync_freq
        self.double_dqn = double_dqn
        self.device = device

        # Online network: updated at every train step
        self.online_net = NatureDQN(in_channels=in_channels, num_actions=num_actions).to(device)

        # Target network: frozen parameters θ^-, synchronized periodically every C steps
        self.target_net = NatureDQN(in_channels=in_channels, num_actions=num_actions).to(device)
        self.sync_target_network()
        self.target_net.eval()

        # Optimizer: Adam with lr=1e-4 provides faster and more stable convergence than vanilla RMSprop
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # Huber loss matches the [-1, 1] gradient clipping used in Mnih et al. (2015)
        self.loss_fn = nn.SmoothL1Loss()

        self.total_steps = 0

    def get_epsilon(self) -> float:
        """Calculate current epsilon using linear annealing schedule."""
        fraction = min(1.0, self.total_steps / float(self.eps_decay_steps))
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action via epsilon-greedy policy.
        
        Args:
            state: Array of shape (4, 84, 84) (uint8).
            deterministic: If True, always select greedy action (for evaluation).
            
        Returns:
            Action index (int).
        """
        epsilon = 0.0 if deterministic else self.get_epsilon()

        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)

        # Greedy action via Q-network
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one step of gradient descent on a sampled minibatch.
        
        Args:
            states: (batch_size, 4, 84, 84)
            actions: (batch_size,)
            rewards: (batch_size,)
            next_states: (batch_size, 4, 84, 84)
            dones: (batch_size,)
            
        Returns:
            Dictionary with training metrics (loss, mean_q).
        """
        self.online_net.train()

        # 1. Compute current Q(s, a; θ)
        q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Compute Target y
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: action selection via online_net, evaluation via target_net
                best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            else:
                # Vanilla Nature 2015 DQN
                next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + (1.0 - dones.float()) * self.gamma * next_q

        # 3. Compute Huber Loss
        loss = self.loss_fn(q_pred, targets)

        # 4. Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for numerical stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 5. Periodic target network update
        self.total_steps += 1
        if self.total_steps % self.target_sync_freq == 0:
            self.sync_target_network()

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_pred.mean().item()),
            "epsilon": self.get_epsilon(),
        }

    def sync_target_network(self) -> None:
        """Copy weights from online network to target network: θ^- ← θ."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str, extra_state: dict = None) -> None:
        """Save model checkpoint atomically with optional training loop state."""
        checkpoint = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }
        if extra_state:
            checkpoint.update(extra_state)
        
        import os
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path: str) -> dict:
        """Load model checkpoint and return checkpoint dict for resumption."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        return checkpoint
