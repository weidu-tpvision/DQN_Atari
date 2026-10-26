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


def test_noop_reset_is_seeded():
    pytest.importorskip("ale_py")
    from src.wrappers import make_atari_env

    def first_obs(seed):
        env = make_atari_env("ALE/Pong-v5", clip_rewards=False, episodic_life=False)
        obs, _ = env.reset(seed=seed)
        for _ in range(30):
            obs, *_ = env.step(0)
        env.close()
        return obs

    np.testing.assert_array_equal(first_obs(11), first_obs(11))


def test_breakout_eval_env_fires_after_life_loss():
    """With episodic_life=False a policy that never presses FIRE must still see play resume."""
    pytest.importorskip("ale_py")
    from src.wrappers import make_atari_env, FireOnLifeLossEnv

    env = make_atari_env("ALE/Breakout-v5", clip_rewards=False, episodic_life=False)
    wrapper = env
    found = False
    while hasattr(wrapper, "env"):
        if isinstance(wrapper, FireOnLifeLossEnv):
            found = True
        wrapper = wrapper.env
    assert found

    env.reset(seed=0)
    steps = 0
    done = False
    while not done and steps < 5000:
        _, _, term, trunc, _ = env.step(0)  # NOOP forever
        done = term or trunc
        steps += 1
    env.close()
    # All 5 lives are lost quickly instead of idling until the 27k-step time limit
    assert done and steps < 5000


def test_episodic_life_env_not_added_to_eval_pipeline_for_pong():
    pytest.importorskip("ale_py")
    from src.wrappers import make_atari_env, EpisodicLifeEnv

    env = make_atari_env("ALE/Pong-v5", episodic_life=False)
    wrapper = env
    while hasattr(wrapper, "env"):
        assert not isinstance(wrapper, EpisodicLifeEnv)
        wrapper = wrapper.env
    env.close()
