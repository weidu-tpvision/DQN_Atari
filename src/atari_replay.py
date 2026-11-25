"""Atari replay that reconstructs overlapping observation stacks when sampled."""

from collections.abc import Iterator
from typing import BinaryIO

import numpy as np
import torch

from src.replay_buffer import ReplayBuffer


class AtariReplayBuffer(ReplayBuffer):
    """A ring of newest frames with sparse copies of history at discontinuities.

    The transition and frame rings have independent cursors. Extra frame slots
    retain the history needed to sample the oldest transition after wraparound.
    Observation continuity is separate from the Bellman terminal flag. A point
    can end bootstrapping without clearing frame history, and a time-limit reset
    can break history while the previous transition still bootstraps.
    """

    _SNAPSHOT_VERSION = 1
    _BLOCK_SIZE = 512

    def __init__(
        self,
        capacity: int = 100_000,
        state_shape: tuple[int, int, int] = (4, 84, 84),
        device: torch.device = torch.device("cpu"),
        *,
        replay_context: str | None = None,
    ):
        if capacity <= 0 or len(state_shape) != 3 or any(dimension <= 0 for dimension in state_shape):
            raise ValueError("Atari replay requires a positive capacity and a (stack, height, width) shape")
        super().__init__(capacity, state_shape, device, replay_context=replay_context)

    def _init_observations(self) -> None:
        self._stack_size = self.state_shape[0]
        self.frames = np.empty((self.capacity + self._stack_size, *self.state_shape[1:]), dtype=np.uint8)
        self._history_lengths = np.empty(self.capacity, dtype=np.int32)
        self._prefixes: dict[int, np.ndarray] = {}
        self._next_overrides: dict[int, np.ndarray] = {}
        self._previous_next = np.empty(self.state_shape, dtype=np.uint8)
        self._has_previous = False
        self._steps = 0

    def _store_observations(self, state: np.ndarray, next_state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.uint8)
        next_state = np.asarray(next_state, dtype=np.uint8)
        if state.shape != self.state_shape or next_state.shape != self.state_shape:
            raise ValueError(f"Atari replay expects observations of shape {self.state_shape}")

        step = self._steps
        if self._has_previous and np.array_equal(state, self._previous_next):
            previous = self._history_lengths[(self.ptr - 1) % self.capacity]
            self._history_lengths[self.ptr] = min(int(previous) + 1, self._stack_size)
        else:
            # The normal reset stack repeats one frame. Preserve a full prefix
            # when starting from a non-reset observation, such as in old replay.
            repeats = np.all(state == state[0])
            self._prefixes[step] = state[:1].copy() if repeats else state.copy()
            self._history_lengths[self.ptr] = 0

        self.frames[step % len(self.frames)] = next_state[-1]
        shifted = np.array_equal(next_state[:-1], state[1:])
        if not shifted:
            self._next_overrides[step] = next_state.copy()
        np.copyto(self._previous_next, next_state)
        self._has_previous = shifted
        self._steps += 1

        oldest = self._steps - min(self.size + 1, self.capacity)
        self._prefixes.pop(oldest - self._stack_size, None)
        self._next_overrides.pop(oldest - 1, None)

    def _sample_observations(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        steps = self._steps - 1 - ((self.ptr - 1 - indices) % self.capacity)
        offsets = np.arange(-self._stack_size, 0)
        histories = self._history_lengths[indices]
        frame_steps = steps[:, None] + np.maximum(offsets, -histories[:, None])
        states = self.frames[frame_steps % len(self.frames)]
        for row in np.flatnonzero(histories < self._stack_size):
            history = int(histories[row])
            prefix = self._prefixes[int(steps[row]) - history]
            states[row, :self._stack_size - history] = prefix if len(prefix) == 1 else prefix[history:]

        next_states = np.empty_like(states)
        next_states[:, :-1] = states[:, 1:]
        next_states[:, -1] = self.frames[steps % len(self.frames)]
        if self._next_overrides:
            for row, step in enumerate(steps):
                override = self._next_overrides.get(int(step))
                if override is not None:
                    next_states[row] = override
        return states, next_states

    def _snapshot_arrays(self) -> dict:
        first_frame = max(0, self._steps - self.size - self._stack_size)
        arrays = {
            "atari_replay_version": self._SNAPSHOT_VERSION,
            "state_shape": self.state_shape,
            "steps": self._steps,
            "capacity": self.capacity,
            "first_frame": first_frame,
            "block_size": self._BLOCK_SIZE,
            "history_lengths": self._history_lengths[:self.size],
            "actions": self.actions[:self.size],
            "rewards": self.rewards[:self.size],
            "dones": self.dones[:self.size],
        }
        for block, step in enumerate(range(first_frame, self._steps, self._BLOCK_SIZE)):
            start = step % len(self.frames)
            stop = start + min(self._BLOCK_SIZE, self._steps - step)
            frames = self.frames[start:stop]
            if stop > len(self.frames):
                frames = np.concatenate((frames, self.frames[:stop - len(self.frames)]))
            arrays[f"frames_{block}"] = frames
        for name, entries in (("prefix", self._prefixes), ("next_override", self._next_overrides)):
            arrays[f"{name}_steps"] = list(entries)
            arrays[f"{name}_lengths"] = [len(frames) for frames in entries.values()]
            arrays[f"{name}_frames"] = (
                np.concatenate(list(entries.values())) if entries else self.frames[:0]
            )
        return arrays

    def _load_arrays(self, data) -> None:
        if "atari_replay_version" not in data.files:
            self._load_legacy_stacks(data)
            return
        if int(data["atari_replay_version"]) != self._SNAPSHOT_VERSION:
            raise ValueError("Unsupported Atari replay snapshot version")
        if tuple(data["state_shape"]) != self.state_shape:
            raise ValueError(f"Atari replay snapshot must have observation shape {self.state_shape}")

        saved_size = int(data["size"])
        saved_ptr = int(data["ptr"])
        saved_capacity = int(data["capacity"])
        steps = int(data["steps"])
        saved_first_frame = int(data["first_frame"])
        block_size = int(data["block_size"])
        if (
            not 0 <= saved_size <= min(saved_capacity, steps)
            or not 0 <= saved_ptr < saved_capacity
            or (saved_size < saved_capacity and saved_ptr != saved_size)
            or saved_first_frame != max(0, steps - saved_size - self._stack_size)
            or block_size <= 0
        ):
            raise ValueError("Invalid Atari replay snapshot layout")

        self._steps = steps
        self.size = min(saved_size, self.capacity)
        oldest = self._steps - self.size

        if self.capacity == saved_capacity:
            rows = slice(0, self.size)
            self.ptr = saved_ptr
        else:
            rows = (saved_ptr - self.size + np.arange(self.size)) % saved_capacity
            self.ptr = self.size % self.capacity
        self.actions[:self.size] = data["actions"][rows]
        self.rewards[:self.size] = data["rewards"][rows]
        self.dones[:self.size] = data["dones"][rows]
        self._history_lengths[:self.size] = data["history_lengths"][rows]

        step = max(0, oldest - self._stack_size)
        while step < self._steps:
            target = step % len(self.frames)
            block_number, offset = divmod(step - saved_first_frame, block_size)
            block = data[f"frames_{block_number}"]
            expected_size = min(block_size, self._steps - saved_first_frame - block_number * block_size)
            if block.shape != (expected_size, *self.state_shape[1:]) or block.dtype != np.uint8:
                raise ValueError(f"Invalid frame block {block_number} in Atari replay snapshot")
            length = min(len(block) - offset, len(self.frames) - target, self._steps - step)
            self.frames[target:target + length] = block[offset:offset + length]
            step += length

        self._prefixes = self._load_sparse(data, "prefix", oldest - self._stack_size + 1)
        self._next_overrides = self._load_sparse(data, "next_override", oldest)
        self._has_previous = False

    @staticmethod
    def _load_sparse(data, name: str, first_step: int) -> dict[int, np.ndarray]:
        frames = data[f"{name}_frames"]
        offset = 0
        entries = {}
        for step, length in zip(data[f"{name}_steps"], data[f"{name}_lengths"]):
            if step >= first_step:
                entries[int(step)] = frames[offset:offset + length].copy()
            offset += length
        return entries

    def _load_legacy_stacks(self, data) -> None:
        """Convert old NPZ stacks one chunk at a time, keeping their latest transitions."""
        actions, rewards, dones = (data[key] for key in ("actions", "rewards", "dones"))
        saved_size = len(actions)
        saved_ptr = int(data["ptr"])
        keep = min(saved_size, self.capacity)
        first = (saved_ptr - keep) % saved_size if saved_size else 0

        with data.zip.open("states.npy") as states, data.zip.open("next_states.npy") as next_states:
            state_offset = self._legacy_offset(states, saved_size)
            next_offset = self._legacy_offset(next_states, saved_size)
            self._steps = self.size = 0
            # Preserve physical sampling indices at equal capacity.
            self.ptr = first if keep == saved_size == self.capacity else 0
            self._has_previous = False
            self._prefixes.clear()
            self._next_overrides.clear()

            for start, stop in ((first, min(first + keep, saved_size)), (0, max(0, first + keep - saved_size))):
                state_chunks = self._legacy_chunks(states, state_offset, start, stop)
                next_chunks = self._legacy_chunks(next_states, next_offset, start, stop)
                row = start
                for before, after in zip(state_chunks, next_chunks, strict=True):
                    for state, next_state in zip(before, after, strict=True):
                        self.push(state, actions[row], rewards[row], next_state, dones[row])
                        row += 1

    def _legacy_offset(self, stream: BinaryIO, size: int) -> int:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported legacy observation array version {version}")
        if shape != (size, *self.state_shape) or fortran or dtype != np.dtype(np.uint8):
            raise ValueError(f"Legacy Atari replay must store contiguous uint8 stacks of shape {self.state_shape}")
        return stream.tell()

    def _legacy_chunks(self, stream: BinaryIO, offset: int, start: int, stop: int) -> Iterator[np.ndarray]:
        if start >= stop:
            return
        row_bytes = int(np.prod(self.state_shape))
        stream.seek(offset + start * row_bytes)
        for row in range(start, stop, self._BLOCK_SIZE):
            count = min(self._BLOCK_SIZE, stop - row)
            yield np.frombuffer(stream.read(count * row_bytes), dtype=np.uint8).reshape(count, *self.state_shape)
