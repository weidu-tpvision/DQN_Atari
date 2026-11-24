import argparse
import sys

import numpy as np
import pytest
import torch
from gymnasium.spaces import Discrete

import evaluate
import train


class ShortAtariEnv:
    def __init__(self):
        self.action_space = Discrete(6)
        self.actions = []
        self.observation = np.zeros((4, 84, 84), dtype=np.uint8)

    def reset(self, seed=None):
        return self.observation, {}

    def step(self, action):
        self.actions.append(action)
        return self.observation, 0.0, len(self.actions) == 24, False, {}

    def close(self):
        pass


def test_training_resume_preserves_config_and_allows_overrides(tmp_path, monkeypatch):
    agents = []
    create_agent = train.DQNAgent

    def record_agent(**kwargs):
        agent = create_agent(**kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(train, "make_atari_env", lambda *args, **kwargs: ShortAtariEnv())
    monkeypatch.setattr(train, "DQNAgent", record_agent)
    monkeypatch.setattr(train, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train.signal, "signal", lambda *args: None)

    def run(save_dir, *options):
        monkeypatch.setattr(sys, "argv", ["train.py", "--save-dir", str(save_dir), "--eval-freq", "0", *options])
        train.train(train.parse_args())
        return torch.load(save_dir / "latest.pt", weights_only=True)

    initial_dir = tmp_path / "initial"
    first = run(
        initial_dir, "--total-steps", "4", "--buffer-size", "6", "--replay-start-size", "2",
        "--batch-size", "2", "--double-dqn", "--gamma", "0.95", "--target-sync-freq", "2",
        "--train-freq", "2", "--lr", "0.0003",
    )
    resume = ("--resume", str(initial_dir / "latest.pt"), "--additional-steps", "2")
    second = run(tmp_path / "resumed", *resume)
    resumed = agents[-1]
    assert resumed.double_dqn
    assert resumed.gamma == 0.95 and resumed.target_sync_freq == 2
    assert resumed.optimizer.param_groups[0]["lr"] == 0.0003
    assert second["num_updates"] == first["num_updates"] + 1

    run(tmp_path / "overrides", *resume, "--no-double-dqn", "--lr", "0.0002")
    assert agents[-1].double_dqn is False
    assert agents[-1].optimizer.param_groups[0]["lr"] == 0.0002


def test_training_rejects_impossible_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--buffer-size", "4", "--replay-start-size", "8", "--save-dir", str(tmp_path / "run"),
    ])
    with pytest.raises(ValueError, match="--replay-start-size"):
        train.train(train.parse_args())
    assert not (tmp_path / "run").exists()


def test_atari_random_actions_follow_evaluation_and_training_seed(tmp_path, monkeypatch):
    envs = []

    def make_env(*args, **kwargs):
        env = ShortAtariEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(train, "make_atari_env", make_env)
    monkeypatch.setattr(train.signal, "signal", lambda *args: None)
    monkeypatch.setattr(evaluate, "make_atari_env", make_env)
    monkeypatch.setattr(evaluate.DQNAgent, "load_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(evaluate, "get_device", lambda: torch.device("cpu"))

    for run in range(2):
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--total-steps", "16", "--buffer-size", "20", "--replay-start-size", "20",
            "--save-dir", str(tmp_path / str(run)), "--no-save-buffer", "--seed", "99",
        ])
        train.train(train.parse_args())
        evaluate.evaluate(argparse.Namespace(
            seed=99, env="ALE/Pong-v5", record=False, checkpoint="unused.pt",
            trust_checkpoint=False, episodes=1, epsilon=1.0,
        ))

    assert envs[0].actions == envs[2].actions
    assert envs[1].actions == envs[3].actions
