"""Utility functions for DQN."""

import os
import random
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
            print(f"CUDA device detected ({torch.cuda.get_device_name(0)}), but kernel execution failed: {e}")
            print("Falling back to high-performance CPU (using multi-core PyTorch).")

    device = torch.device("cpu")
    print(f"Using CPU with {torch.get_num_threads()} PyTorch threads")
    return device


class MetricTracker:
    """Tracks running averages and windowed stats for training metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def mean(self, key: str) -> float:
        if key not in self.history or len(self.history[key]) == 0:
            return 0.0
        window = self.history[key][-self.window_size:]
        return float(np.mean(window))
