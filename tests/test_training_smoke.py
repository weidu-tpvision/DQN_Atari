"""Train and resume through the public CLI using real Atari environments."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    ("script", "environment_options"),
    [
        pytest.param("train.py", ("--env", "ALE/Pong-v5"), id="pong"),
        pytest.param("train.py", ("--env", "ALE/Breakout-v5"), id="breakout"),
        pytest.param("train_selfplay.py", (), marks=pytest.mark.multiplayer, id="selfplay"),
    ],
)
def test_atari_train_and_resume_makes_optimizer_updates(tmp_path, script, environment_options):
    if script == "train_selfplay.py":
        pytest.importorskip("multi_agent_ale_py")
        pytest.importorskip("pettingzoo")

    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    latest = tmp_path / "latest.pt"

    def run(*options):
        result = subprocess.run(
            [sys.executable, "-B", script, "--eval-freq", "0", "--save-dir", str(tmp_path), *options],
            cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        with np.load(tmp_path / checkpoint["replay"]["file"], allow_pickle=False) as replay:
            assert "atari_replay_version" in replay.files
        return checkpoint

    first = run(
        *environment_options, "--total-steps", "8", "--buffer-size", "6",
        "--replay-start-size", "2", "--batch-size", "2", "--train-freq", "2",
    )
    assert first["env_steps"] == 8 and first["num_updates"] == 4
    assert first["optimizer"]["state"]

    resumed = run("--resume", str(latest), "--additional-steps", "4")
    assert resumed["env_steps"] == 12 and resumed["num_updates"] == 6
    assert resumed["config"]["buffer_size"] == 6
