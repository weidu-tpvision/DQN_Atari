"""Round-trip and boundary coverage for frame-based Atari replay."""

import numpy as np
import pytest

from src.atari_replay import AtariReplayBuffer
from src.replay_buffer import ReplayBuffer


STATE_SHAPE = (4, 2, 3)


def transitions(count, *, start=0):
    for step in range(start, start + count):
        if step == start or step % 7 == 0:
            state = np.full(STATE_SHAPE, step + 80, dtype=np.uint8)
        next_state = np.concatenate((state[1:], np.full((1, 2, 3), step, dtype=np.uint8)))
        if step % 13 == 5:
            next_state = np.arange(np.prod(STATE_SHAPE), dtype=np.uint8).reshape(STATE_SHAPE)
        # Some terminals leave history intact, and some resets follow a nonterminal.
        yield state, step % 4, float(step), next_state, step % 3 == 0
        state = next_state


def assert_transitions(buffer, expected, monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda low, high, size: np.arange(high))
    states, actions, rewards, next_states, dones = buffer.sample(len(buffer))
    assert sorted(rewards.tolist()) == [transition[2] for transition in expected]
    by_reward = {transition[2]: transition for transition in expected}
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones, strict=True):
        expected_state, expected_action, _, expected_next, expected_done = by_reward[float(reward)]
        np.testing.assert_array_equal(state.numpy(), expected_state)
        np.testing.assert_array_equal(next_state.numpy(), expected_next)
        assert int(action) == expected_action and bool(done) == expected_done


@pytest.mark.parametrize("capacity", [1, 3, 13])
def test_atari_replay_reconstructs_observations_after_wrap(capacity, monkeypatch):
    replay = AtariReplayBuffer(capacity, STATE_SHAPE)
    records = list(transitions(60))
    for record in records:
        replay.push(*record)
    assert replay.ptr == 60 % capacity
    assert_transitions(replay, records[-capacity:], monkeypatch)


@pytest.mark.parametrize("capacity", [1, 5, 13, 70])
def test_compact_snapshot_resizes_and_can_be_saved_again(tmp_path, monkeypatch, capacity):
    monkeypatch.setattr(AtariReplayBuffer, "_BLOCK_SIZE", 3)
    original = AtariReplayBuffer(13, STATE_SHAPE)
    records = list(transitions(60))
    for record in records:
        original.push(*record)
    snapshot = tmp_path / "compact.npz"
    original.save(str(snapshot))
    with np.load(snapshot, allow_pickle=False) as data:
        assert "states" not in data.files and "next_states" not in data.files
        assert sum(len(data[key]) for key in data.files if key.startswith("frames_")) == 17

    restored = AtariReplayBuffer(capacity, STATE_SHAPE)
    restored.load(str(snapshot))
    kept = records[-min(13, capacity):]
    assert_transitions(restored, kept, monkeypatch)
    if capacity == 13:
        assert restored.ptr == original.ptr
        np.testing.assert_array_equal(restored.rewards, original.rewards)

    added = list(transitions(9, start=60))
    for record in added:
        restored.push(*record)
    expected = (kept + added)[-capacity:]
    restored.save(str(snapshot))
    with np.load(snapshot, allow_pickle=False) as data:
        assert sum(len(data[key]) for key in data.files if key.startswith("frames_")) == len(expected) + 4
    reopened = AtariReplayBuffer(capacity, STATE_SHAPE)
    reopened.load(str(snapshot))
    assert reopened.ptr == restored.ptr
    assert_transitions(reopened, expected, monkeypatch)


@pytest.mark.parametrize("capacity", [5, 13, 20])
def test_legacy_stacked_checkpoint_restores_into_compact_replay(tmp_path, monkeypatch, capacity):
    monkeypatch.setattr(AtariReplayBuffer, "_BLOCK_SIZE", 3)
    legacy = ReplayBuffer(13, STATE_SHAPE)
    records = list(transitions(60))
    for record in records:
        legacy.push(*record)
    checkpoint_path = str(tmp_path / "latest.pt")
    checkpoint = {"replay": legacy.save_for_checkpoint(checkpoint_path)}

    restored = AtariReplayBuffer(capacity, STATE_SHAPE)
    assert restored.restore_for_checkpoint(checkpoint_path, checkpoint, None)
    assert_transitions(restored, records[-min(13, capacity):], monkeypatch)
    if capacity == 13:
        assert restored.ptr == legacy.ptr
        np.testing.assert_array_equal(restored.rewards, legacy.rewards)


def test_empty_compact_snapshot_can_be_restored(tmp_path):
    path = str(tmp_path / "empty.npz")
    replay = AtariReplayBuffer(13, STATE_SHAPE)
    replay.save(path)
    restored = AtariReplayBuffer(5, STATE_SHAPE)
    restored.load(path)
    assert len(restored) == 0 and restored.ptr == 0
