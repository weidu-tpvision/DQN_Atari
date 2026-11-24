"""Utility functions for DQN."""

import random
from collections import deque
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_exploration(agent, args, checkpoint=None, *, warm_start=False) -> None:
    """Apply explicit CLI options over the checkpoint's effective epsilon schedule."""
    checkpoint = checkpoint or {}
    schedule = checkpoint.get("epsilon_schedule")
    if schedule is None:
        schedule = dict(checkpoint.get("config", {}))
        if schedule.get("base_model"):
            schedule["eps_start"] = min(schedule.get("eps_start", agent.eps_start), 0.20)

    for option, attribute in (
        ("eps_start", "eps_start"),
        ("eps_end", "eps_end"),
        ("eps_decay", "eps_decay_steps"),
    ):
        value = getattr(args, option)
        if value is None:
            value = schedule.get(option, getattr(agent, attribute))
        setattr(agent, attribute, value)
    if warm_start:
        agent.eps_start = min(agent.eps_start, 0.20)


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    name, keys, *rest = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *rest))
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def get_device() -> torch.device:
    """Detect and return the fastest functional PyTorch compute device."""
    if torch.cuda.is_available():
        try:
            # Test a small tensor operation to verify CUDA capability compatibility
            test_tensor = torch.zeros(1, device="cuda")
            _ = test_tensor + 1
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        except Exception as e:
            print(f"CUDA device detected, but kernel execution failed: {e}")
            print("Falling back to high-performance CPU (using multi-core PyTorch).")

    device = torch.device("cpu")
    print(f"Using CPU with {torch.get_num_threads()} PyTorch threads")
    return device


class MetricTracker:
    """Tracks windowed running averages for training metrics.

    Only the last `window_size` values per metric are kept, so checkpoints that store
    `history` stay a constant size.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.history:
                self.history[k] = deque(maxlen=self.window_size)
            self.history[k].append(v)

    def mean(self, key: str) -> float:
        if key not in self.history or len(self.history[key]) == 0:
            return 0.0
        return float(np.mean(self.history[key]))

    def state_dict(self) -> dict:
        return {key: list(values) for key, values in self.history.items()}

    def load(self, history: dict) -> None:
        """Restore history from a checkpoint (accepts old unbounded lists)."""
        self.history = {k: deque(list(v)[-self.window_size:], maxlen=self.window_size) for k, v in history.items()}
