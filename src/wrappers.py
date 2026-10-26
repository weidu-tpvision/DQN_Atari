"""Atari 2600 Preprocessing Wrappers for Gymnasium.

Implements the standard preprocessing pipeline defined in Mnih et al. (2015):
- NoopResetEnv: Random NOOPs at start of episode for stochasticity.
- FireResetEnv: Press FIRE on start for games that require it.
- MaxAndSkipEnv: Frame skip (4) + element-wise max over last 2 frames to remove flicker.
- EpisodicLifeEnv: End episode on life loss to encourage survival in bootstrapping.
- FireOnLifeLossEnv: (Evaluation only) press FIRE after a life is lost in games that need it.
- WarpFrame: Grayscale conversion and resize to 84x84.
- FrameStack: Stack last 4 frames into shape (4, 84, 84).
- ClipRewardEnv: Clip rewards to {-1, 0, +1}.
"""

from collections import deque
from typing import Tuple, Dict, Any
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import ale_py


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        self._rng = np.random.default_rng()
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Own generator, re-seeded by reset(seed=...), so a seed fully determines the no-op count
        if kwargs.get("seed") is not None:
            self._rng = np.random.default_rng(kwargs["seed"])
        obs, info = self.env.reset(**kwargs)
        noops = int(self._rng.integers(1, self.noop_max + 1))
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for games that require FIRE to start."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert len(action_meanings) >= 3 and action_meanings[1] == "FIRE"

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)  # 1 = FIRE
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # 2 = RIGHT (Pong: paddle up)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame and max-pool over the last 2 frames to remove flicker."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)
        self._skip = skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            if term or trunc:
                # Episode ended early: make sure no frame from the previous step is pooled
                if i < self._skip - 2:
                    self._obs_buffer[0] = obs
                    self._obs_buffer[1] = obs
                elif i == self._skip - 2:
                    self._obs_buffer[1] = obs
                terminated = term
                truncated = trunc
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to sign(reward): -1, 0, or +1."""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


class WarpFrame(gym.ObservationWrapper):
    """Convert RGB frame to grayscale (84x84)."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.
    
    Done so that the agent learns value of avoiding death during Q-value bootstrapping.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Lost a life: signal termination for bootstrapping
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Step NOOP to advance past death screen without full game reset
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireOnLifeLossEnv(gym.Wrapper):
    """Press FIRE after a life is lost so play resumes in games like Breakout.

    Only needed when EpisodicLifeEnv is off (evaluation). During training, EpisodicLifeEnv turns
    every life loss into a reset() and FireResetEnv already presses FIRE there. Without this, a
    greedy policy that never picks FIRE idles until the 108k-frame time limit.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if not (terminated or truncated) and 0 < lives < self.lives:
            obs, fire_reward, terminated, truncated, info = self.env.step(1)  # 1 = FIRE
            reward += fire_reward
            lives = self.env.unwrapped.ale.lives()
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FrameStack(gym.Wrapper):
    """Stack k consecutive 84x84 grayscale frames into a (k, 84, 84) observation tensor."""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        orig_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(k, *orig_shape), dtype=np.uint8
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.array(self.frames, dtype=np.uint8)


def make_atari_env(
    env_id: str = "ALE/Pong-v5",
    clip_rewards: bool = True,
    episodic_life: bool = True,
    noop_max: int = 30,
    frame_skip: int = 4,
    stack_k: int = 4,
    render_mode: str = None,
) -> gym.Env:
    """Construct full Atari preprocessing pipeline matching Nature 2015."""
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    if frame_skip > 1:
        env = MaxAndSkipEnv(env, skip=frame_skip)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    
    # Check if FIRE action is required to start
    action_meanings = env.unwrapped.get_action_meanings()
    if len(action_meanings) >= 3 and action_meanings[1] == "FIRE":
        env = FireResetEnv(env)
        if not episodic_life:
            env = FireOnLifeLossEnv(env)

    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=stack_k)
    return env
