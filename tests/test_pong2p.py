"""Tests for the shared 2-player Pong pipeline (src/pong2p.py)."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
ale_py = pytest.importorskip("ale_py")
pytest.importorskip("pettingzoo")

from src.pong2p import TwoPlayerPong, process_frame
from src.wrappers import WarpFrame


def _gray_levels(frames):
    return set(np.unique(np.concatenate([f.ravel() for f in frames])).tolist())


def test_same_palette_as_gym_pong():
    """PettingZoo frames must give the same gray levels the Gym-trained networks saw."""
    gym.register_envs(ale_py)
    genv = gym.make("ALE/Pong-v5", frameskip=1)
    obs, _ = genv.reset(seed=0)
    warp = WarpFrame(genv)
    gym_frames = []
    for t in range(400):
        obs, *_ = genv.step(genv.action_space.sample())
        if t % 10 == 0:
            gym_frames.append(warp.observation(obs))
    genv.close()

    game = TwoPlayerPong()
    s_r, _ = game.reset(seed=0)
    pz_frames = []
    for t in range(100):
        game.step(np.random.randint(6), np.random.randint(6))
        if t % 3 == 0:
            pz_frames.append(game.observe()[0][-1])
    game.close()

    # Court background and paddle grays must be identical; exclude resize-blended edge pixels
    def dominant(frames):
        vals, counts = np.unique(np.concatenate([f.ravel() for f in frames]), return_counts=True)
        return set(vals[counts > counts.sum() * 0.001].tolist())

    assert dominant(pz_frames) <= dominant(gym_frames) | {0}
    assert 87 in dominant(pz_frames)  # court background (144, 72, 17)


def test_observations_shape_and_mirroring():
    game = TwoPlayerPong()
    s_r, s_l = game.reset(seed=1)
    assert s_r.shape == (4, 84, 84) and s_r.dtype == np.uint8
    assert s_l.shape == (4, 84, 84)
    for _ in range(10):
        game.step(0, 0)
    s_r, s_l = game.observe()
    np.testing.assert_array_equal(s_l, s_r[:, :, ::-1])
    game.close()


def test_reset_seed_is_reproducible():
    game = TwoPlayerPong()
    a, _ = game.reset(seed=7)
    for _ in range(20):
        game.step(2, 3)
    a = game.observe()[0]
    b, _ = game.reset(seed=7)
    for _ in range(20):
        game.step(2, 3)
    b = game.observe()[0]
    np.testing.assert_array_equal(a, b)
    game.close()


def test_process_frame_matches_warpframe():
    rgb = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
    gym.register_envs(ale_py)
    warp = WarpFrame(gym.make("ALE/Pong-v5", frameskip=1))
    np.testing.assert_array_equal(process_frame(rgb), warp.observation(rgb))


def test_serve_penalty_reported_without_points():
    """Stalling the serve gives the server a -1 reward but scores no point.

    (The opening serve of a game is automatic, so one point can still happen while idle.)
    """
    game = TwoPlayerPong(noop_max=0)
    game.reset(seed=0)
    penalties = 0
    for _ in range(1000):
        r_pts, l_pts, done = game.step(0, 0)
        assert not done
        if r_pts == 0 and l_pts == 0:
            penalties += (game.right_reward < 0) + (game.left_reward < 0)
    assert penalties >= 5
    game.close()


def test_point_rewards_are_zero_sum():
    game = TwoPlayerPong(noop_max=0)
    game.reset(seed=0)
    for _ in range(2000):
        r_pts, l_pts, done = game.step(1, 0)  # right serves, left never moves
        if r_pts:
            assert game.right_reward == 1.0 and game.left_reward == -1.0
            break
    else:
        raise AssertionError("no point scored")
    game.close()
