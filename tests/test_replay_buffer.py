"""Unit tests for Experience Replay Buffer."""

import numpy as np
import torch
import pytest
from src.replay_buffer import ReplayBuffer


def test_replay_buffer_push_and_sample():
    capacity = 100
    batch_size = 16
    buffer = ReplayBuffer(capacity=capacity, state_shape=(4, 84, 84))

    assert len(buffer) == 0

    # Add transitions
    for i in range(50):
        s = np.full((4, 84, 84), fill_value=i, dtype=np.uint8)
        a = i % 4
        r = float(i)
        s_next = np.full((4, 84, 84), fill_value=i + 1, dtype=np.uint8)
        d = (i % 10 == 0)
        buffer.push(s, a, r, s_next, d)

    assert len(buffer) == 50

    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=batch_size)
    assert states.shape == (batch_size, 4, 84, 84)
    assert next_states.shape == (batch_size, 4, 84, 84)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert dones.shape == (batch_size,)


def test_replay_buffer_wrap_around():
    capacity = 10
    buffer = ReplayBuffer(capacity=capacity, state_shape=(4, 84, 84))

    for i in range(25):
        s = np.zeros((4, 84, 84), dtype=np.uint8)
        buffer.push(s, 0, 1.0, s, False)

    assert len(buffer) == 10
    assert buffer.ptr == 5
