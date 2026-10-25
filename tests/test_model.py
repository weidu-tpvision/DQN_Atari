"""Unit tests for Nature DQN model."""

import torch
import pytest
from src.model import NatureDQN


def test_nature_dqn_output_shape():
    batch_size = 4
    num_actions = 6
    model = NatureDQN(in_channels=4, num_actions=num_actions)
    
    # Test with uint8 tensor in [0, 255]
    x_uint8 = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
    out = model(x_uint8)
    assert out.shape == (batch_size, num_actions)
    assert out.dtype == torch.float32

    # Test with float32 tensor in [0.0, 1.0]
    x_float = torch.rand((batch_size, 4, 84, 84), dtype=torch.float32)
    out_float = model(x_float)
    assert out_float.shape == (batch_size, num_actions)


def test_nature_dqn_conv_dimensions():
    model = NatureDQN(in_channels=4, num_actions=4)
    # Check layer parameters
    assert model.conv1.in_channels == 4
    assert model.conv1.out_channels == 32
    assert model.conv2.out_channels == 64
    assert model.conv3.out_channels == 64
    assert model.fc1.out_features == 512
    assert model.fc2.out_features == 4
