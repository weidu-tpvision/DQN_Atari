"""Snapshot pool for self-play: size cap, numbering, PFSP sampling, persistence."""

import csv
import hashlib
import os
import random
import sys

import numpy as np
import pytest
import torch

import train_selfplay
from src.agent import DQNAgent
from src.utils import capture_rng_state
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


@pytest.mark.parametrize("with_anchors", [False, True])
def test_selfplay_trains_and_resumes_with_saved_settings(tmp_path, monkeypatch, agent, with_anchors):
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
    evaluation = ["--eval-freq", "0"]
    if with_anchors:
        anchor_paths = [tmp_path / "first, anchor.pt", tmp_path / "second anchor.pt"]
        for anchor_path in anchor_paths:
            agent.save_weights(str(anchor_path))
        evaluation = [
            "--eval-freq", "2", "--eval-matches", "2", "--eval-seed", "5030",
            *[flag for path in anchor_paths for flag in ("--eval-anchor", str(path))],
        ]
    first = run(
        source, "--total-steps", "4", "--buffer-size", "6", "--replay-start-size", "2", "--batch-size", "2",
        "--train-freq", "2", "--gamma", "0.95", "--target-sync-freq", "2", "--no-double-dqn",
        "--lr", "0.0003", "--pool-size", "3", "--opponent-sampling", "uniform", "--latest-prob", "0",
        *evaluation,
    )
    assert first["num_updates"] == 2

    dest = tmp_path / "resumed"
    resumed = run(dest, "--resume", str(source), "--additional-steps", "2", "--double-dqn")
    assert resumed["env_steps"] == 6 and resumed["num_updates"] == 3
    assert resumed["pool_snapshots"] == first["pool_snapshots"]
    assert all((dest / "pool" / name).is_file() for name in resumed["pool_snapshots"])
    assert resumed["config"]["double_dqn"] is True
    assert resumed["optimizer"]["param_groups"][0]["lr"] == 0.0003
    for option in ("gamma", "target_sync_freq", "eval_freq", "pool_size", "opponent_sampling", "latest_prob", "eval_anchors"):
        assert resumed["config"][option] == first["config"][option]

    if with_anchors:
        for directory, steps in ((source, ["2", "4"]), (dest, ["6"])):
            with (directory / "selfplay_anchor_metrics.csv").open(encoding="utf-8", newline="") as metrics:
                rows = list(csv.DictReader(metrics))
            assert [(row["step"], row["opponent"]) for row in rows] == [
                (step, str(anchor_path)) for step in steps for anchor_path in anchor_paths
            ]
            for row in rows:
                with open(row["opponent"], "rb") as model:
                    assert row["opponent_sha256"] == hashlib.sha256(model.read()).hexdigest()
                assert row["first_seed"] == "5030" and row["matches"] == "2"


def test_pool_evaluation_is_repeatable_and_preserves_training_rng(tmp_path, monkeypatch, agent):
    matches = []
    state = np.zeros((4, 84, 84), dtype=np.uint8)

    class RecordingMatch:
        def reset(self, seed):
            matches.append([seed])
            return self.observe()

        def step(self, action_right, action_left):
            matches[-1].append((action_right, action_left))
            return 1, 0, True

        def observe(self):
            return state, state

        def close(self):
            pass

    monkeypatch.setattr(train_selfplay, "TwoPlayerPong", RecordingMatch)
    pool = SnapshotPool(str(tmp_path), torch.device("cpu"))
    pool.add_snapshot(agent, 0)
    pool.add_snapshot(agent, 1)
    agent.online_net.train()
    before = capture_rng_state()
    options = {"num_matches": 8, "epsilon": 1.0, "seed": 600}
    result = train_selfplay.evaluate_learner_vs_pool(agent, pool, **options)
    assert pool.loaded_agents
    assert train_selfplay.evaluate_learner_vs_pool(agent, pool, **options) == result
    after = capture_rng_state()
    assert before["python"] == after["python"]
    np.testing.assert_equal(before["numpy"], after["numpy"])
    assert torch.equal(before["torch"], after["torch"])
    assert agent.online_net.training
    assert matches[:8] == matches[8:]
    assert [match[0] for match in matches[:8]] == list(range(600, 608))
