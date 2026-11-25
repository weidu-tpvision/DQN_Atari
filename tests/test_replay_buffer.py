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


def test_action_masks_stored_sampled_and_saved(tmp_path):
    buffer = ReplayBuffer(capacity=8, state_shape=(1, 2, 2), action_mask_dim=4)
    s = np.zeros((1, 2, 2), dtype=np.uint8)
    buffer.push(s, 0, 0.0, s, False, next_mask=np.array([True, False, True, False]))
    buffer.push(s, 1, 1.0, s, True, next_mask=np.array([False, False, False, False]))
    batch = buffer.sample(16)
    assert len(batch) == 6 and batch[5].shape == (16, 4)
    path = str(tmp_path / "buf.npz")
    buffer.save(path)
    other = ReplayBuffer(capacity=8, state_shape=(1, 2, 2), action_mask_dim=4)
    other.load(path)
    np.testing.assert_array_equal(other.next_masks[:2], buffer.next_masks[:2])


def test_plain_buffer_still_returns_five_and_loads_into_masked(tmp_path):
    plain = _filled(10, 3)
    assert len(plain.sample(4)) == 5
    path = str(tmp_path / "buf.npz")
    plain.save(path)
    masked = ReplayBuffer(capacity=10, state_shape=(1, 2, 2), action_mask_dim=4)
    masked.load(path)  # file has no masks: everything stays legal
    assert len(masked) == 3 and masked.next_masks.all()


def test_checkpoint_replay_preserves_masks_and_rejects_replacement(tmp_path):
    buffer = ReplayBuffer(capacity=3, state_shape=(1, 2, 2), action_mask_dim=4)
    state = np.zeros((1, 2, 2), dtype=np.uint8)
    mask = np.array([True, False, True, False])
    buffer.push(state, 0, 0.0, state, False, next_mask=mask)
    checkpoint_path = str(tmp_path / "dqn_2048_step_100000.pt")
    checkpoint = {"replay": buffer.save_for_checkpoint(checkpoint_path)}

    restored = ReplayBuffer(capacity=2, state_shape=(1, 2, 2), action_mask_dim=4)
    assert restored.restore_for_checkpoint(checkpoint_path, checkpoint, None)
    np.testing.assert_array_equal(restored.next_masks[0], mask)

    replay_path = tmp_path / checkpoint["replay"]["file"]
    buffer.save(str(replay_path), checkpoint_id="different-checkpoint")
    with pytest.raises(ValueError, match="does not match"):
        restored.restore_for_checkpoint(checkpoint_path, checkpoint, None)


def test_failed_checkpoint_publish_preserves_previous_replay(tmp_path, monkeypatch):
    from src.agent import DQNAgent

    path = tmp_path / "latest.pt"
    agent = DQNAgent(num_actions=2)
    buffer = _filled(5, 2)
    agent.save(str(path), replay_buffer=buffer)
    previous = agent.load(str(path))
    previous_replay = tmp_path / previous["replay"]["file"]

    def fail_save(*args):
        raise OSError("model write failed")

    state = np.zeros((1, 2, 2), dtype=np.uint8)
    buffer.push(state, 0, 10.0, state, False)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_save)
        with pytest.raises(OSError, match="model write failed"):
            agent.save(str(path), replay_buffer=buffer)

    restored = ReplayBuffer(capacity=5, state_shape=(1, 2, 2))
    assert restored.restore_for_checkpoint(str(path), agent.load(str(path)), None)
    assert len(restored) == 2
    assert list(tmp_path.glob("latest_replay_*.npz")) == [previous_replay]

    agent.save(str(path), replay_buffer=buffer)
    assert restored.restore_for_checkpoint(str(path), agent.load(str(path)), None)
    assert len(restored) == 3
    assert not previous_replay.exists()


def test_legacy_checkpoint_can_restore_fixed_replay_filename(tmp_path):
    buffer = _filled(5, 2)
    replay_path = tmp_path / "latest_replay.npz"
    buffer.save(str(replay_path), checkpoint_id="legacy")
    restored = ReplayBuffer(capacity=5, state_shape=(1, 2, 2))
    checkpoint = {"replay": {"file": replay_path.name, "id": "legacy"}}
    assert restored.restore_for_checkpoint(str(tmp_path / "latest.pt"), checkpoint, None)
    assert len(restored) == 2


def test_explicit_replay_path_checks_game_and_checkpoint_pair(tmp_path):
    shape = (1, 2, 2)
    source = ReplayBuffer(capacity=2, state_shape=shape, replay_context="ALE/Pong-v5")
    path = str(tmp_path / "moved_replay.npz")
    source.save(path, checkpoint_id="pong")
    filename = str(tmp_path / "latest.pt")

    wrong_game = ReplayBuffer(capacity=2, state_shape=shape, replay_context="ALE/SpaceInvaders-v5")
    with pytest.raises(ValueError, match="Replay context"):
        wrong_game.restore_for_checkpoint(filename, {}, path)

    matching_game = ReplayBuffer(capacity=2, state_shape=shape, replay_context="ALE/Pong-v5")
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        matching_game.restore_for_checkpoint(filename, {"replay": {"id": "other"}}, path)
    assert matching_game.restore_for_checkpoint(filename, {"replay": {"id": "pong"}}, path)
