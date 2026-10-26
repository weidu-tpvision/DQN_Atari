"""Deep Q-Network Agent (Nature 2015).

Manages:
- Online Network Q(s, a; θ) and Target Network Q(s, a; θ^-)
- Epsilon-greedy action selection with linear decay over environment steps
- Bellman loss computation with Huber Loss (Smooth L1)
- Periodic target network weight synchronization (every C gradient updates)

Counters:
- `env_steps`: environment (agent) steps taken so far. Drives the epsilon schedule and is
  set by the training loop each step. Nature 2015 anneals epsilon over agent steps.
- `num_updates`: gradient updates performed so far. Drives target network syncs. Nature 2015
  measures the target update frequency C in parameter updates.
"""

import os
from typing import Dict
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
        target_sync_freq: int = 1_000,
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

        # Target network: frozen parameters θ^-, synchronized every `target_sync_freq` updates
        self.target_net = NatureDQN(in_channels=in_channels, num_actions=num_actions).to(device)
        self.sync_target_network()
        self.target_net.eval()

        # Optimizer: Adam with lr=1e-4 provides faster and more stable convergence than vanilla RMSprop
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # Huber loss matches the [-1, 1] gradient clipping used in Mnih et al. (2015)
        self.loss_fn = nn.SmoothL1Loss()

        self.env_steps = 0
        self.num_updates = 0

    def get_epsilon(self) -> float:
        """Linear annealing from eps_start to eps_end over `eps_decay_steps` environment steps."""
        fraction = min(1.0, self.env_steps / float(self.eps_decay_steps))
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

        if epsilon > 0.0 and np.random.rand() < epsilon:
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
        self.num_updates += 1
        if self.num_updates % self.target_sync_freq == 0:
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
            "env_steps": self.env_steps,
            "num_updates": self.num_updates,
        }
        if extra_state:
            checkpoint.update(extra_state)

        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def save_weights(self, path: str) -> None:
        """Save only the online network weights (small, loadable with weights_only=True)."""
        tmp_path = path + ".tmp"
        torch.save({"online_net": self.online_net.state_dict()}, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path: str) -> dict:
        """Load full training state (weights, optimizer, counters) and return the checkpoint dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["online_net"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Older checkpoints stored the update count as "total_steps" and the env step as "global_step"
        self.num_updates = checkpoint.get("num_updates", checkpoint.get("total_steps", 0))
        self.env_steps = checkpoint.get("env_steps", checkpoint.get("global_step", 0))
        return checkpoint

    def load_weights(self, path: str) -> None:
        """Load only network weights (fresh optimizer and counters), e.g. to warm-start or to play."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["online_net"] if "online_net" in checkpoint else checkpoint
        self.online_net.load_state_dict(state_dict)
        self.sync_target_network()
