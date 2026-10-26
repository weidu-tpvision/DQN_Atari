"""Two-player Atari Pong (PettingZoo pong_v3) with the same preprocessing as `make_atari_env`.

Single source of truth for every 2-player script (self-play training, PvP arena,
round-robin league, human-vs-AI play), so all of them feed the networks identical inputs:
- 1-30 random NOOPs on reset (NoopResetEnv)
- 4-frame action repeat with element-wise max over the last 2 frames (MaxAndSkipEnv)
- RGB -> grayscale -> 84x84 INTER_AREA (WarpFrame)
- 4-frame stacks (FrameStack)

PettingZoo and Gymnasium ALE/Pong-v5 render the court with the same palette
(background (144,72,17), paddles (92,186,92) / (213,130,74)), so no color remapping is needed.

Players:
- first_0  = right paddle (green), the side single-agent Gym Pong trains on.
- second_0 = left paddle (orange). Its stack is mirrored horizontally so a policy trained on the
  right paddle sees itself on the right.

Rules that differ from single-agent Gym Pong-v5 (ALE mode 4 instead of mode 0):
- Serving is not automatic: the player holding the ball must press FIRE. A player who doesn't
  serve within ~2 seconds receives a -1 reward (no point is scored) and the timer restarts.
  `step()` reports points and raw rewards separately so trainers can learn from the penalty.
- No sticky actions (PettingZoo uses repeat_action_probability=0.0, Gym v5 uses 0.25).
- The opponent is another policy rather than the built-in CPU.
"""

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from pettingzoo.atari import pong_v3

NUM_ACTIONS = 6
RIGHT = "first_0"
LEFT = "second_0"


def process_frame(rgb: np.ndarray) -> np.ndarray:
    """Preprocess a 210x160 RGB frame to 84x84 grayscale (matches WarpFrame)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


def policy_action(agent, state: np.ndarray, epsilon: float = 0.0) -> int:
    """Greedy action from `agent`, replaced by a uniformly random one with probability epsilon."""
    if epsilon > 0.0 and np.random.rand() < epsilon:
        return int(np.random.randint(NUM_ACTIONS))
    return agent.select_action(state, deterministic=True)


class TwoPlayerPong:
    """PettingZoo pong_v3 with Nature-DQN preprocessing for both paddles."""

    def __init__(self, render_mode: Optional[str] = None, noop_max: int = 30, frame_skip: int = 4, stack_k: int = 4):
        self.env = pong_v3.parallel_env(render_mode=render_mode)
        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.stack_k = stack_k
        self._rng = np.random.default_rng()
        self._obs_buffer = np.zeros((2, 210, 160, 3), dtype=np.uint8)
        self.right_stack: deque = deque(maxlen=stack_k)
        self.left_stack: deque = deque(maxlen=stack_k)
        # Raw PettingZoo rewards summed over the last step (points +-1, serve penalty -1)
        self.right_reward = 0.0
        self.left_reward = 0.0
        # True if the last step ended by PettingZoo's max_cycles time limit rather than game over
        self.truncated = False

    @staticmethod
    def _ended(term: dict, trunc: dict) -> bool:
        return bool(term[RIGHT] or trunc[RIGHT] or term[LEFT] or trunc[LEFT])

    def _push_frame(self, rgb: np.ndarray) -> None:
        gray = process_frame(rgb)
        self.right_stack.append(gray)
        self.left_stack.append(cv2.flip(gray, 1))

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the match. Returns (right_state, left_state), each (4, 84, 84) uint8."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, _ = self.env.reset(seed=seed)
        noops = int(self._rng.integers(1, self.noop_max + 1)) if self.noop_max > 0 else 0
        for _ in range(noops):
            obs, _, term, trunc, _ = self.env.step({RIGHT: 0, LEFT: 0})
            if self._ended(term, trunc):
                obs, _ = self.env.reset()
        for _ in range(self.stack_k):
            self._push_frame(obs[RIGHT])
        return self.observe()

    def observe(self) -> Tuple[np.ndarray, np.ndarray]:
        """Current (right_state, left_state) stacks."""
        return np.array(self.right_stack), np.array(self.left_stack)

    def step(self, action_right: int, action_left: int) -> Tuple[int, int, bool]:
        """Repeat both actions for `frame_skip` frames.

        Returns:
            (right_points, left_points, done): points each paddle scored during this step and
            whether the underlying game ended (21 points or PettingZoo max_cycles). The raw
            per-player rewards of the step are in `right_reward` / `left_reward`, and
            `truncated` tells whether `done` came from the time limit.
        """
        right_pts = left_pts = 0
        self.right_reward = self.left_reward = 0.0
        self.truncated = False
        done = False
        skip = self.frame_skip
        for i in range(skip):
            obs, rewards, term, trunc, _ = self.env.step({RIGHT: action_right, LEFT: action_left})
            right_pts += max(0, int(rewards[RIGHT]))
            left_pts += max(0, int(rewards[LEFT]))
            self.right_reward += float(rewards[RIGHT])
            self.left_reward += float(rewards[LEFT])
            frame = obs[RIGHT]
            if i == skip - 2:
                self._obs_buffer[0] = frame
            elif i == skip - 1:
                self._obs_buffer[1] = frame
            if self._ended(term, trunc):
                # Ended early: make sure no frame from the previous step is pooled
                if i < skip - 2:
                    self._obs_buffer[0] = frame
                    self._obs_buffer[1] = frame
                elif i == skip - 2:
                    self._obs_buffer[1] = frame
                done = True
                self.truncated = not (term[RIGHT] or term[LEFT])
                break
        pooled = self._obs_buffer.max(axis=0) if skip > 1 else frame
        self._push_frame(pooled)
        return right_pts, left_pts, done

    def render(self) -> np.ndarray:
        """RGB frame of the full screen (requires render_mode='rgb_array')."""
        return self.env.render()

    def close(self) -> None:
        self.env.close()
