"""Snapshot pool for self-play: size cap, numbering, PFSP sampling, persistence."""

import os
import random
import sys

import numpy as np
import pytest
import torch

import train_selfplay
from src.agent import DQNAgent
from train_selfplay import SnapshotPool


@pytest.fixture(scope="module")
def agent():
    return DQNAgent(num_actions=6)


def test_cap_retires_oldest_and_keeps_numbering(tmp_path, agent):
    pool = SnapshotPool(str(tmp_path), torch.device("cpu"), max_size=3)
    for step in range(5):
        pool.add_snapshot(agent, step)
    names = [os.path.basename(p) for p in pool.snapshots]
    assert names == ["snapshot_002_step_2.pt", "snapshot_003_step_3.pt", "snapshot_004_step_4.pt"]
    assert len(os.listdir(tmp_path)) == 5  # retired files stay on disk

    reopened = SnapshotPool(str(tmp_path), torch.device("cpu"), max_size=3)
    assert reopened.snapshots == pool.snapshots
    reopened.add_snapshot(agent, 5)
    assert os.path.basename(reopened.snapshots[-1]) == "snapshot_005_step_5.pt"


def test_numeric_order_beyond_999(tmp_path, agent):
    for idx in (998, 999, 1000):
        torch.save({"online_net": agent.online_net.state_dict()}, tmp_path / f"snapshot_{idx:03d}_step_{idx}.pt")
    pool = SnapshotPool(str(tmp_path), torch.device("cpu"), max_size=2)
    assert [os.path.basename(p) for p in pool.snapshots] == ["snapshot_999_step_999.pt", "snapshot_1000_step_1000.pt"]
    assert pool.next_index == 1001


def test_pfsp_prefers_opponents_the_learner_loses_to(tmp_path, agent):
    pool = SnapshotPool(str(tmp_path), torch.device("cpu"), max_size=0, sampling="pfsp", latest_prob=0.0)
    easy, hard, newest = (pool.add_snapshot(agent, s) for s in (0, 1, 2))
    for _ in range(20):
        pool.record_result(easy, 1.0)
        pool.record_result(hard, 0.0)
    random.seed(0)
    picks = [pool._choose(for_eval=False) for _ in range(3000)]
    assert picks.count(hard) > 10 * picks.count(easy)
    # Evaluation sampling stays uniform
    eval_picks = [pool._choose(for_eval=True) for _ in range(3000)]
    assert min(eval_picks.count(p) for p in (easy, hard, newest)) > 800


def test_pool_state_round_trip_drops_retired(tmp_path, agent):
    pool = SnapshotPool(str(tmp_path), torch.device("cpu"), max_size=2)
    a, b = pool.add_snapshot(agent, 0), pool.add_snapshot(agent, 1)
    pool.record_result(a, 1.0)
    pool.record_result(b, 0.0)
    state = pool.state_dict()
    pool.add_snapshot(agent, 2)  # retires a
    restored = SnapshotPool(str(tmp_path), torch.device("cpu"), max_size=2)
    restored.load_state_dict(state)
    assert set(restored.results) == {os.path.basename(b)}
    assert restored.win_rate(b) == pytest.approx(1 / 3)


def test_resume_copies_active_opponents_and_restores_results(tmp_path, agent):
    source = tmp_path / "source"
    original = SnapshotPool(str(source / "pool"), torch.device("cpu"), max_size=2)
    original.add_snapshot(agent, 0)
    previous = original.add_snapshot(agent, 1)
    original.record_result(previous, 0.0)
    checkpoint = {"pool_snapshots": original.snapshot_names, "pool_state": original.state_dict()}
    original.add_snapshot(agent, 2)

    destination = tmp_path / "destination" / "pool"
    restored = SnapshotPool(str(destination), torch.device("cpu"), max_size=2)
    restored.restore(str(source / "latest.pt"), checkpoint, step=1)

    assert restored.snapshot_names == checkpoint["pool_snapshots"]
    assert all((destination / name).is_file() for name in restored.snapshot_names)
    assert restored.win_rate(previous) == pytest.approx(1 / 3)
    assert restored.next_index == 2


def test_selfplay_trains_and_resumes_with_saved_settings(tmp_path, monkeypatch):
    class ShortMatch:
        def __init__(self):
            self.state = np.zeros((4, 84, 84), dtype=np.uint8)
            self.right_reward = 1.0
            self.left_reward = -1.0
            self.truncated = False

        def reset(self, seed=None):
            return self.observe()

        def step(self, action_right, action_left):
            return 1, 0, True

        def observe(self):
            return self.state, self.state

        def close(self):
            pass

    monkeypatch.setattr(train_selfplay, "TwoPlayerPong", ShortMatch)
    monkeypatch.setattr(train_selfplay, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train_selfplay.signal, "signal", lambda *_: None)

    def run(directory, *options):
        monkeypatch.setattr(sys, "argv", ["train_selfplay.py", "--save-dir", str(directory), *options])
        train_selfplay.main()
        return torch.load(directory / "latest.pt", weights_only=True)

    source = tmp_path / "source"
    first = run(
        source, "--total-steps", "4", "--buffer-size", "6", "--replay-start-size", "2", "--batch-size", "2",
        "--train-freq", "2", "--gamma", "0.95", "--target-sync-freq", "2", "--no-double-dqn", "--eval-freq", "0",
        "--lr", "0.0003", "--pool-size", "3", "--opponent-sampling", "uniform", "--latest-prob", "0",
    )
    assert first["num_updates"] == 2

    dest = tmp_path / "resumed"
    resumed = run(dest, "--resume", str(source), "--additional-steps", "2", "--double-dqn")
    assert resumed["env_steps"] == 6 and resumed["num_updates"] == 3
    assert resumed["pool_snapshots"] == first["pool_snapshots"]
    assert all((dest / "pool" / name).is_file() for name in resumed["pool_snapshots"])
    assert resumed["config"]["double_dqn"] is True
    assert resumed["optimizer"]["param_groups"][0]["lr"] == 0.0003
    for option in ("gamma", "target_sync_freq", "eval_freq", "pool_size", "opponent_sampling", "latest_prob"):
        assert resumed["config"][option] == first["config"][option]
