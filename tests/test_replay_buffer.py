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


def _filled(capacity, n):
    buffer = ReplayBuffer(capacity=capacity, state_shape=(1, 2, 2))
    for i in range(n):
        s = np.full((1, 2, 2), i, dtype=np.uint8)
        buffer.push(s, i % 4, float(i), s, False)
    return buffer


def test_save_load_round_trip(tmp_path):
    buffer = _filled(10, 25)
    path = str(tmp_path / "buf.npz")
    buffer.save(path)
    other = ReplayBuffer(capacity=10, state_shape=(1, 2, 2))
    other.load(path)
    assert (other.ptr, other.size) == (buffer.ptr, buffer.size)
    np.testing.assert_array_equal(other.rewards, buffer.rewards)


def test_load_into_smaller_buffer_keeps_newest(tmp_path):
    buffer = _filled(10, 25)  # holds rewards 15..24, ptr=5
    path = str(tmp_path / "buf.npz")
    buffer.save(path)
    small = ReplayBuffer(capacity=4, state_shape=(1, 2, 2))
    small.load(path)
    assert len(small) == 4
    np.testing.assert_array_equal(small.rewards, [21.0, 22.0, 23.0, 24.0])
    assert small.ptr == 0


def test_load_into_larger_buffer_appends_in_order(tmp_path):
    buffer = _filled(10, 25)
    path = str(tmp_path / "buf.npz")
    buffer.save(path)
    big = ReplayBuffer(capacity=20, state_shape=(1, 2, 2))
    big.load(path)
    assert (len(big), big.ptr) == (10, 10)
    np.testing.assert_array_equal(big.rewards[:10], np.arange(15, 25, dtype=np.float32))


def test_load_rejects_wrong_shape(tmp_path):
    buffer = _filled(10, 5)
    path = str(tmp_path / "buf.npz")
    buffer.save(path)
    other = ReplayBuffer(capacity=10, state_shape=(1, 3, 3))
    with pytest.raises(ValueError):
        other.load(path)
