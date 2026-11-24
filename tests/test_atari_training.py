import argparse
import sys

import numpy as np
import pytest
import torch
from gymnasium.spaces import Discrete

import evaluate
import train
import train_2048
import train_selfplay


class ShortAtariEnv:
    def __init__(self):
        self.action_space = Discrete(6)
        self.actions = []
        self.reset_seeds = []
        self.observation = np.zeros((4, 84, 84), dtype=np.uint8)

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.episode_steps = 0
        return self.observation, {}

    def step(self, action):
        self.actions.append(action)
        self.episode_steps += 1
        return self.observation, 0.0, self.episode_steps == 24, False, {}

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
    assert first["num_updates"] == 2
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


@pytest.mark.parametrize("trainer", [train, train_2048, train_selfplay])
def test_training_rejects_impossible_warmup(tmp_path, monkeypatch, trainer):
    monkeypatch.setattr(sys, "argv", [
        trainer.__name__, "--buffer-size", "4", "--replay-start-size", "8", "--save-dir", str(tmp_path / "run"),
    ])
    with pytest.raises(ValueError, match="--replay-start-size"):
        if trainer is train_selfplay:
            trainer.main()
        else:
            trainer.train(trainer.parse_args())
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


def test_atari_checkpoint_evaluation_reuses_seeds_and_resets_best_when_they_change(tmp_path, monkeypatch):
    evaluations = []

    def score_policy(agent, env_id, num_episodes, seed):
        evaluations.append((num_episodes, seed))
        score = 11 - len(evaluations)
        return {"mean_score": score, "std_score": 0, "min_score": score, "max_score": score, "mean_length": 24}

    monkeypatch.setattr(train, "make_atari_env", lambda *args, **kwargs: ShortAtariEnv())
    monkeypatch.setattr(train, "evaluate_policy", score_policy)
    monkeypatch.setattr(train, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train.signal, "signal", lambda *args: None)

    def run(directory, *options):
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--save-dir", str(directory), "--eval-freq", "2", "--patience", "5", *options,
        ])
        train.train(train.parse_args())
        return torch.load(directory / "latest.pt", weights_only=True)

    source = tmp_path / "source"
    first = run(
        source, "--total-steps", "4", "--buffer-size", "6", "--replay-start-size", "2",
        "--eval-episodes", "3", "--eval-seed", "1100",
    )
    assert evaluations == [(3, 1100), (3, 1100)]
    assert first["best_eval_score"] == 10 and first["patience_counter"] == 1

    resume = ("--resume", str(source / "latest.pt"), "--additional-steps", "2")
    second = run(tmp_path / "same-seeds", *resume)
    assert evaluations[-1] == (3, 1100)
    assert second["best_eval_score"] == 10 and second["patience_counter"] == 2

    changed = tmp_path / "changed-seeds"
    third = run(changed, *resume, "--eval-seed", "1120")
    assert evaluations[-1] == (3, 1120)
    assert third["best_eval_score"] == 7 and third["patience_counter"] == 0
    assert (changed / "best_model.pt").is_file()


def test_standalone_atari_evaluation_starts_at_held_out_seed(monkeypatch):
    env = ShortAtariEnv()
    monkeypatch.setattr(evaluate, "make_atari_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(evaluate, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(evaluate.DQNAgent, "load_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", "unused.pt", "--episodes", "2", "--epsilon", "1"])
    evaluate.evaluate(evaluate.parse_args())
    assert env.reset_seeds == [50_000, 50_001]
