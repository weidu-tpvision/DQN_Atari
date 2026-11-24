"""Smoke tests: every script's --help must work."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "train.py", "train_selfplay.py", "train_2048.py", "play_2048.py",
    "evaluate.py", "tournament.py", "round_robin_tournament.py",
    "pvp_battle.py", "play_remote.py", "record_demo.py", "plot_metrics.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help(script):
    result = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_boolean_flags_can_be_disabled():
    result = subprocess.run([sys.executable, "train_selfplay.py", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    for flag in ("--no-save-buffer", "--no-double-dqn"):
        assert flag in result.stdout
