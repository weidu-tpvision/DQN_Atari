"""Neural Network Architecture for 2048 DQN.

Uses multi-scale directional convolutions tailored for the 4x4 grid:
- Horizontal (1x2) kernels for row-wise adjacent tile relationships
- Vertical (2x1) kernels for column-wise adjacent tile relationships
- 2x2 kernels for local square patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN2048(nn.Module):
    """Multi-scale convolutional Q-network for 2048."""

    def __init__(self, in_channels: int = 16, num_actions: int = 4):
        super().__init__()
        # Horizontal features: scans pairs across rows
        self.conv_h = nn.Conv2d(in_channels, 128, kernel_size=(1, 2))
        
        # Vertical features: scans pairs across columns
        self.conv_v = nn.Conv2d(in_channels, 128, kernel_size=(2, 1))
        
        # Square 2x2 features
        self.conv_sq = nn.Conv2d(in_channels, 128, kernel_size=(2, 2))

        # Flatten dimensions:
        # conv_h: (4, 4) with (1, 2) kernel -> (4, 3) = 12 * 128 = 1536
        # conv_v: (4, 4) with (2, 1) kernel -> (3, 4) = 12 * 128 = 1536
        # conv_sq: (4, 4) with (2, 2) kernel -> (3, 3) = 9 * 128 = 1152
        # Total concatenated = 1536 + 1536 + 1152 = 4224
        concat_dim = 1536 + 1536 + 1152

        self.fc1 = nn.Linear(concat_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_actions)

        # Initialize weights
        for m in [self.conv_h, self.conv_v, self.conv_sq, self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.constant_(m.bias, 0.0)
        nn.init.uniform_(self.out.weight, -0.01, 0.01)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize/cast to float if passed as uint8
        if x.dtype == torch.uint8:
            x = x.float()

        h = F.relu(self.conv_h(x)).flatten(start_dim=1)
        v = F.relu(self.conv_v(x)).flatten(start_dim=1)
        sq = F.relu(self.conv_sq(x)).flatten(start_dim=1)

        merged = torch.cat([h, v, sq], dim=1)
        feat = F.relu(self.fc1(merged))
        feat = F.relu(self.fc2(feat))
        return self.out(feat)
