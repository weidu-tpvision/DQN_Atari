"""Nature 2015 Deep Q-Network (DQN) Architecture.

Reference:
    Mnih et al., "Human-level control through deep reinforcement learning",
    Nature 518, 529-533 (2015).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureDQN(nn.Module):
    """Convolutional Neural Network architecture from Mnih et al. (2015).
    
    Accepts preprocessed 4-frame grayscale stack of size (4, 84, 84).
    Outputs a Q-value for each discrete action.
    """

    def __init__(self, in_channels: int = 4, num_actions: int = 6):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions

        # Feature extraction layers (3 convolutional layers with ReLU)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        # Input size: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal/Kaiming initialization as commonly used in modern PyTorch RL."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 4, 84, 84).
               Can be uint8 in [0, 255] or float32 in [0, 1].
               
        Returns:
            Q-values for each action of shape (batch, num_actions).
        """
        # If input is in uint8 [0, 255], cast to float and normalize to [0.0, 1.0] on device
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten to (batch, 3136)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values
