"""Unit tests for Atari Preprocessing Wrappers."""

import pytest
import numpy as np


def test_atari_env_pipeline():
    try:
        import gymnasium as gym
        import ale_py
        from src.wrappers import make_atari_env
    except ImportError:
        pytest.skip("Gymnasium or ALE not yet installed")

    env = make_atari_env("ALE/Pong-v5", clip_rewards=True, episodic_life=True)
    obs, info = env.reset(seed=42)

    # Verify observation shape is (4, 84, 84)
    assert obs.shape == (4, 84, 84), f"Expected shape (4, 84, 84), got {obs.shape}"
    assert obs.dtype == np.uint8, f"Expected dtype uint8, got {obs.dtype}"

    # Step environment
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (4, 84, 84)
    assert reward in [-1.0, 0.0, 1.0], f"Expected clipped reward in [-1, 0, 1], got {reward}"
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    env.close()
