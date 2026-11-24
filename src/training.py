"""Configuration and checkpoint state shared by the DQN training entry points."""

import argparse

from src.utils import restore_rng_state


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Leave resumable options unset so a checkpoint can supply their values."""
    parser.add_argument("--buffer-size", type=int, help="Replay buffer capacity")
    parser.add_argument("--replay-start-size", type=int, help="Warm-up steps before training starts")
    parser.add_argument("--batch-size", type=int, help="Minibatch size for training")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--lr", type=float, help="Learning rate (Adam)")
    parser.add_argument("--eps-start", type=float, help="Initial exploration rate")
    parser.add_argument("--eps-end", type=float, help="Final exploration rate")
    parser.add_argument("--eps-decay", type=int, help="Environment steps to anneal epsilon")
    parser.add_argument("--target-sync-freq", type=int, help="Gradient updates between target network syncs")
    parser.add_argument("--train-freq", type=int, help="Agent steps between training updates")
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=None, help="Use Double DQN targets")
    parser.add_argument("--seed", type=int, help="Random seed")


def resolve_training_config(args, checkpoint: dict, **trainer_defaults) -> None:
    """Resolve explicit options, saved settings, and trainer defaults in that order."""
    defaults = {
        "buffer_size": 100_000,
        "replay_start_size": 10_000,
        "batch_size": 32,
        "gamma": 0.99,
        "lr": 1e-4,
        "target_sync_freq": 1_000,
        "train_freq": 4,
        "double_dqn": False,
        "seed": 42,
        **trainer_defaults,
    }
    saved = checkpoint.get("config") or {}
    for name, default in defaults.items():
        if getattr(args, name) is None:
            saved_value = saved.get(name)
            setattr(args, name, default if saved_value is None else saved_value)

    if args.buffer_size <= 0 or not 0 <= args.replay_start_size <= args.buffer_size:
        raise ValueError("--buffer-size must be positive and --replay-start-size must be between 0 and --buffer-size")

    for name in ("batch_size", "train_freq", "target_sync_freq", "save_freq", "log_freq"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def restore_training_state(agent, tracker, args, checkpoint: dict, *, lr_override: float | None) -> None:
    """Restore the learner and common runtime state, honoring an explicit learning rate."""
    agent.load_training_state(checkpoint)
    if lr_override is not None:
        for group in agent.optimizer.param_groups:
            group["lr"] = lr_override
    args.lr = agent.optimizer.param_groups[0]["lr"]

    if "tracker_history" in checkpoint:
        tracker.load(checkpoint["tracker_history"])
    if "rng_state" in checkpoint:
        try:
            restore_rng_state(checkpoint["rng_state"])
        except Exception as exc:
            print(f"Note: Could not restore full RNG state ({exc}), continuing with fresh RNG.")
