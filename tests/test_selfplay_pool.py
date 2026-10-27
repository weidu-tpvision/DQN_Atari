"""Snapshot pool for self-play: size cap, numbering, PFSP sampling, persistence."""

import os
import random

import pytest
import torch

pytest.importorskip("pettingzoo")
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
