"""Self-Play DQN Training for Atari Pong (PettingZoo pong_v3).

Implements Fictitious Self-Play (FSP) with an evolving Opponent Snapshot Pool:
  1. Learner agent plays against frozen historical snapshots of itself.
  2. The learner's side is chosen at random each match (50% Right paddle, 50% Left paddle).
  3. Left player observation is horizontally flipped (cv2.flip) for vision symmetry.
  4. Only the learner's own transitions are stored in the replay buffer. The reward is the game's
     own per-player reward (points, plus PettingZoo's -1 penalty for not serving in time).
  5. When the learner's win-rate against the pool reaches the promotion threshold, a new
     snapshot is promoted to the pool. The pool keeps the newest --pool-size snapshots.
  6. Opponents are sampled by Prioritized Fictitious Self-Play (default): mostly the snapshots the
     learner still loses to, plus the newest snapshot --latest-prob of the time.
  7. Atomic checkpointing and graceful shutdown on SIGINT (Ctrl+C).

Preprocessing is shared with every other 2-player script via src/pong2p.py and matches
train.py (see that module for the remaining differences, e.g. no sticky actions).

Usage:
    # Train self-play from scratch:
    python train_selfplay.py --total-steps 500000

    # Train self-play initialized from an existing Pong checkpoint (weights only):
    python train_selfplay.py --base-model checkpoints/champion_model.pt --total-steps 500000
"""

import os
import sys
import time
import csv
import filecmp
import hashlib
import re
import shutil
from pathlib import Path
import json
import random
import signal
import argparse
from collections import deque
from typing import Callable, List, Tuple, Dict

import numpy as np
import torch

from src.agent import DQNAgent, add_checkpoint_trust_flag, read_checkpoint
from src.atari_replay import AtariReplayBuffer
from src.pong2p import TwoPlayerPong, NUM_ACTIONS, policy_action
from src.training import add_training_arguments, resolve_training_config, restore_training_state
from src.utils import (
    MetricTracker, capture_rng_state, configure_exploration, get_device, set_seed,
)


def _snapshot_index(path: str) -> int:
    match = re.match(r"snapshot_(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else -1


def load_frozen_opponent(path: str, device: torch.device, *, trusted: bool) -> DQNAgent:
    # The models initialize on CPU before being moved to the requested device.
    # Loading earlier during evaluation should not change RNG use during training.
    with torch.random.fork_rng(devices=[]):
        opponent = DQNAgent(num_actions=NUM_ACTIONS, device=device)
    opponent.load_weights(path, trusted=trusted)
    opponent.online_net.eval()
    return opponent


class SnapshotPool:
    """Frozen past copies of the learner: the opponents for self-play.

    - At most `max_size` snapshots are active (0 = unlimited). Adding one beyond the cap retires
      the oldest; its file stays on disk but it is no longer played against.
    - Training opponents are sampled with probability `latest_prob` from the newest snapshot,
      otherwise according to `sampling`:
        "pfsp"    Prioritized Fictitious Self-Play (Vinyals et al. 2019, AlphaStar): weight
                  (1 - p)^2, where p is the learner's recent win rate against that snapshot, so
                  opponents the learner still struggles with are met more often.
        "uniform" every active snapshot equally likely (classic fictitious self-play).
    - Evaluation opponents (`for_eval=True`) are always uniform over the active pool, so the
      promotion test measures strength against the whole pool.
    """

    def __init__(self, pool_dir: str, device: torch.device, max_size: int = 50, sampling: str = "pfsp",
                 latest_prob: float = 0.2, result_window: int = 20, *, trusted: bool = False):
        self.pool_dir = pool_dir
        self.device = device
        self.max_size = max_size
        self.sampling = sampling
        self.latest_prob = latest_prob
        self.result_window = result_window
        os.makedirs(pool_dir, exist_ok=True)
        files = sorted(Path(pool_dir).glob("snapshot_*.pt"), key=_snapshot_index)
        self.next_index = _snapshot_index(files[-1]) + 1 if files else 0
        self.trusted = trusted
        self.snapshots: List[str] = [str(path) for path in (files[-max_size:] if max_size > 0 else files)]
        self.loaded_agents: Dict[str, DQNAgent] = {}
        # Recent match results of the learner against each snapshot: 1 win, 0.5 draw, 0 loss
        self.results: Dict[str, deque] = {}

    @property
    def snapshot_names(self) -> List[str]:
        return [os.path.basename(path) for path in self.snapshots]

    def restore(self, checkpoint_path: str, checkpoint: dict, step: int) -> None:
        source_dir = Path(checkpoint_path).resolve().parent / "pool"
        names = checkpoint.get("pool_snapshots")
        if names is None:
            # Legacy checkpoints only recorded the count. Exclude later promotions.
            snapshots = []
            for source in source_dir.glob("snapshot_*.pt"):
                match = re.fullmatch(r"snapshot_(\d+)_step_(\d+)\.pt", source.name)
                if match and int(match[2]) <= step:
                    snapshots.append((int(match[1]), int(match[2]), source.name))
            names = [name for _, _, name in sorted(snapshots)]
            count = checkpoint.get("pool_size", len(names))
            names = names[-count:] if count else []

        target_dir = Path(self.pool_dir).resolve()
        restored = []
        if self.max_size > 0:
            names = names[-self.max_size:]
        for name in names:
            if not re.fullmatch(r"snapshot_\d+_step_\d+\.pt", name):
                raise ValueError(f"Invalid opponent snapshot filename: {name}")
            source = source_dir / name
            target = target_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing opponent snapshot listed in checkpoint: {source}")
            if source != target:
                if not target.exists():
                    temporary = target.with_suffix(".pt.tmp")
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, target)
                elif not filecmp.cmp(source, target, shallow=False):
                    raise FileExistsError(f"Opponent snapshot differs from the checkpoint's copy: {target}")
            restored.append(str(target))
            self.next_index = max(self.next_index, _snapshot_index(name) + 1)
        self.snapshots = restored
        self.loaded_agents.clear()
        self.load_state_dict(checkpoint.get("pool_state", {}))

    def add_snapshot(self, agent: DQNAgent, step: int) -> str:
        """Save the learner's current weights into the pool, retiring the oldest if over the cap."""
        path = os.path.join(self.pool_dir, f"snapshot_{self.next_index:03d}_step_{step}.pt")
        self.next_index += 1
        agent.save_weights(path)
        self.snapshots.append(path)
        retired = []
        while self.max_size > 0 and len(self.snapshots) > self.max_size:
            old = self.snapshots.pop(0)
            self.loaded_agents.pop(old, None)
            self.results.pop(os.path.basename(old), None)
            retired.append(os.path.basename(old))
        msg = f"\n[POOL PROMOTION] Added new snapshot to pool: {os.path.basename(path)} (Pool Size: {len(self.snapshots)})"
        if retired:
            msg += f" | retired {', '.join(retired)}"
        print(msg)
        return path

    def win_rate(self, path: str) -> float:
        """Learner's recent win rate vs a snapshot, with a uniform Beta(1, 1) prior."""
        res = self.results.get(os.path.basename(path), ())
        return (sum(res) + 1.0) / (len(res) + 2.0)

    def record_result(self, path: str, learner_score: float) -> None:
        """Record a finished training match: learner_score is 1 (win), 0.5 (draw) or 0 (loss)."""
        key = os.path.basename(path)
        self.results.setdefault(key, deque(maxlen=self.result_window)).append(learner_score)

    def _choose(self, for_eval: bool, rng: random.Random | None = None) -> str:
        if not self.snapshots:
            raise RuntimeError("Snapshot pool is empty!")
        if for_eval:
            return (rng or random).choice(self.snapshots)
        if len(self.snapshots) == 1:
            return random.choice(self.snapshots)
        if random.random() < self.latest_prob:
            return self.snapshots[-1]
        if self.sampling == "uniform":
            return random.choice(self.snapshots)
        weights = [(1.0 - self.win_rate(p)) ** 2 for p in self.snapshots]
        return random.choices(self.snapshots, weights=weights)[0]

    def sample_opponent(self, for_eval: bool = False, rng: random.Random | None = None) -> Tuple[DQNAgent, str]:
        """Sample an opponent; returns (agent, snapshot path)."""
        chosen_path = self._choose(for_eval, rng)

        # Cache loaded agents for fast switching
        if chosen_path not in self.loaded_agents:
            self.loaded_agents[chosen_path] = load_frozen_opponent(chosen_path, self.device, trusted=self.trusted)

        return self.loaded_agents[chosen_path], chosen_path

    def state_dict(self) -> dict:
        return {"results": {k: list(v) for k, v in self.results.items()}}

    def load_state_dict(self, state: dict) -> None:
        active = {os.path.basename(p) for p in self.snapshots}
        self.results = {
            k: deque(v, maxlen=self.result_window) for k, v in state.get("results", {}).items() if k in active
        }


def evaluate_learner_vs_pool(
    learner: DQNAgent,
    pool: SnapshotPool,
    num_matches: int = 20,
    target_score: int = 7,
    epsilon: float = 0.02,
    max_steps: int = 10_000,
    seed: int = 5000,
) -> Dict[str, float]:
    """Evaluate against uniformly sampled snapshots for the promotion decision."""
    opponent_rng = random.Random(seed)
    return evaluate_matches(
        learner, lambda: pool.sample_opponent(for_eval=True, rng=opponent_rng)[0],
        num_matches, target_score, epsilon, max_steps, seed,
    )


def evaluate_matches(
    learner: DQNAgent,
    opponent_for_match: Callable[[], DQNAgent],
    num_matches: int = 20,
    target_score: int = 7,
    epsilon: float = 0.02,
    max_steps: int = 10_000,
    seed: int = 5000,
) -> Dict[str, float]:
    """Evaluate with alternating sides and an independent random stream for each match."""
    game = TwoPlayerPong()
    wins = 0
    draws = 0
    total_pts_for = 0
    total_pts_against = 0

    was_training = learner.online_net.training
    learner.online_net.eval()
    try:
        for m in range(num_matches):
            opp_agent = opponent_for_match()
            learner_is_right = (m % 2 == 0)
            right_agent = learner if learner_is_right else opp_agent
            left_agent = opp_agent if learner_is_right else learner

            match_seed = seed + m
            action_rng = np.random.default_rng(match_seed)
            s_r, s_l = game.reset(seed=match_seed)
            r_pts, l_pts = 0, 0
            for _ in range(max_steps):
                a_r = policy_action(right_agent, s_r, epsilon, rng=action_rng)
                a_l = policy_action(left_agent, s_l, epsilon, rng=action_rng)
                r_gain, l_gain, done = game.step(a_r, a_l)
                r_pts += r_gain
                l_pts += l_gain
                s_r, s_l = game.observe()
                if done or r_pts >= target_score or l_pts >= target_score:
                    break

            learner_pts = r_pts if learner_is_right else l_pts
            opp_pts = l_pts if learner_is_right else r_pts
            total_pts_for += learner_pts
            total_pts_against += opp_pts
            if learner_pts > opp_pts:
                wins += 1
            elif learner_pts == opp_pts:
                draws += 1
    finally:
        learner.online_net.train(was_training)
        game.close()
    return {
        "win_rate": float((wins + 0.5 * draws) / num_matches * 100.0),
        "wins": wins,
        "draws": draws,
        "matches": num_matches,
        "pts_for": total_pts_for,
        "pts_against": total_pts_against,
    }


def evaluate_anchors(learner, anchors, args, step: int) -> None:
    path = Path(args.save_dir) / "selfplay_anchor_metrics.csv"
    with path.open("a", encoding="utf-8", newline="") as metrics:
        writer = csv.writer(metrics)
        if metrics.tell() == 0:
            writer.writerow([
                "step", "opponent", "opponent_sha256", "first_seed", "target_score", "matches",
                "wins", "draws", "win_rate", "pts_for", "pts_against",
            ])
        for opponent_path, digest, opponent in anchors:
            result = evaluate_matches(
                learner, lambda: opponent, num_matches=args.eval_matches,
                target_score=args.eval_target_score, seed=args.eval_seed,
            )
            writer.writerow([
                step, opponent_path, digest, args.eval_seed, args.eval_target_score, result["matches"],
                result["wins"], result["draws"], f"{result['win_rate']:.2f}",
                result["pts_for"], result["pts_against"],
            ])
            print(
                f"[FIXED OPPONENT @ Step {step:7d}] {opponent_path} | "
                f"Win rate: {result['win_rate']:.1f}% | Score: {result['pts_for']}-{result['pts_against']}",
                flush=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Play DQN Training for Atari Pong")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total environment interaction steps")
    parser.add_argument("--additional-steps", type=int, default=None, help="Train for an additional N steps from checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (or directory with latest.pt) to resume self-play from")
    parser.add_argument("--base-model", help="Warm-start from model weights and cap initial epsilon at 0.20")
    parser.add_argument("--replay-path", help="Explicit replay file to use on resume")
    add_training_arguments(parser)
    parser.add_argument("--save-freq", type=int, default=50_000, help="Step interval for saving periodic checkpoints")
    parser.add_argument("--eval-freq", type=int, help="Step interval to evaluate vs pool and promote (<=0 to disable)")
    parser.add_argument("--eval-matches", type=int, help="Matches per pool evaluation (more = less noisy promotion)")
    parser.add_argument("--eval-target-score", type=int, help="Points needed to win an evaluation match")
    parser.add_argument("--eval-seed", type=int, help="First seed of the fixed evaluation set (default: 5000)")
    anchors = parser.add_mutually_exclusive_group()
    anchors.add_argument("--eval-anchor", dest="eval_anchors", action="append", metavar="CHECKPOINT",
                         help="Evaluate against this fixed checkpoint at every evaluation (repeatable)")
    anchors.add_argument("--no-eval-anchors", dest="eval_anchors", action="store_const", const=[],
                         help="Clear fixed opponents inherited from a checkpoint")
    parser.add_argument("--log-freq", type=int, default=1_000, help="Step interval for console progress")
    parser.add_argument("--promotion-threshold", type=float, help="Win rate %% needed to add snapshot to pool")
    parser.add_argument("--pool-size", type=int, help="Max active snapshots; oldest retired beyond this (0 = unlimited)")
    parser.add_argument("--opponent-sampling", choices=["pfsp", "uniform"],
                        help="pfsp: favor opponents the learner loses to; uniform: classic fictitious self-play")
    parser.add_argument("--latest-prob", type=float, help="Probability of playing the newest snapshot")
    parser.add_argument("--save-dir", type=str, default="checkpoints_selfplay", help="Directory to save selfplay data")
    parser.add_argument("--save-buffer", action=argparse.BooleanOptionalAction, default=True, help="Save replay buffer with checkpoints")
    add_checkpoint_trust_flag(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    resume_path = args.resume
    ckpt = {}
    if resume_path:
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "latest.pt")
        print(f"\n--> Resuming Self-Play training from checkpoint: {resume_path}")
        ckpt = read_checkpoint(resume_path, device, trusted=args.trust_checkpoint)
    lr_override = args.lr
    resolve_training_config(
        args, ckpt, double_dqn=True, eval_freq=25_000, eval_matches=20, eval_target_score=7,
        eval_seed=5_000, eval_anchors=(), promotion_threshold=65.0,
        pool_size=50, opponent_sampling="pfsp", latest_prob=0.2,
    )
    if args.eval_freq > 0 and (args.eval_matches <= 0 or args.eval_target_score <= 0 or args.eval_seed < 0):
        raise ValueError("Evaluation requires positive --eval-matches and --eval-target-score and a nonnegative --eval-seed")
    args.eval_anchors = list(dict.fromkeys(str(Path(path).resolve()) for path in args.eval_anchors))
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    pool_dir = os.path.join(args.save_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"       ATARI PONG: SELF-PLAY DQN TRAINING ENGINE       ")
    print(f"=======================================================")
    print(f"Total Steps:   {args.total_steps:,} | Buffer: {args.buffer_size:,}")
    print(f"Double DQN:    {args.double_dqn} | Promotion Win-Rate: {args.promotion_threshold}%")
    print(f"Pool:          max {args.pool_size} | sampling {args.opponent_sampling} | latest {args.latest_prob:.0%}")
    print(f"Save Dir:      {args.save_dir} | Device: {device}")
    if args.base_model:
        print(f"Base Model:    {args.base_model} (warm-starting policy)")
    print(f"-------------------------------------------------------\n")

    # Initialize Learner
    learner = DQNAgent(
        num_actions=NUM_ACTIONS,
        in_channels=4,
        gamma=args.gamma,
        lr=args.lr,
        target_sync_freq=args.target_sync_freq,
        double_dqn=args.double_dqn,
        device=device,
    )

    configure_exploration(learner, args, ckpt, warm_start=bool(args.base_model and not resume_path))

    if args.base_model and not args.resume:
        if not os.path.exists(args.base_model):
            sys.exit(f"Base model not found: {args.base_model}")
        # Weights only: keep a fresh optimizer and counters so the epsilon schedule starts at eps_start
        print(f"Loading initial policy weights from base model: {args.base_model}")
        learner.load_weights(args.base_model, trusted=args.trust_checkpoint)

    # Initialize Replay Buffer
    replay_buffer = AtariReplayBuffer(
        capacity=args.buffer_size,
        state_shape=(4, 84, 84),
        device=device,
    )

    tracker = MetricTracker(window_size=100)
    global_step = 0
    matches_played = 0

    # Handle Resumption
    if resume_path:
        restore_training_state(learner, tracker, args, ckpt, lr_override=lr_override)
        global_step = ckpt.get("global_step", learner.env_steps)
        matches_played = ckpt.get("matches_played", 0)

        print(f"Resumed at step: {global_step:,} | Matches played so far: {matches_played:,} | Epsilon: {learner.get_epsilon():.3f}")

        if replay_buffer.restore_for_checkpoint(resume_path, ckpt, args.replay_path):
            print(f"Replay buffer restored with {len(replay_buffer):,} transitions.")
        else:
            print("No associated replay buffer. Collecting fresh transitions.")

    # Determine target total steps
    if args.additional_steps is not None:
        target_total_steps = global_step + args.additional_steps
        print(f"--> Training for an additional {args.additional_steps:,} steps (target: {target_total_steps:,}).")
    else:
        target_total_steps = args.total_steps

    if global_step >= target_total_steps:
        print(f"[INFO] Current step ({global_step:,}) already meets or exceeds target steps ({target_total_steps:,}).")
        return

    pool = SnapshotPool(
        pool_dir, device, max_size=args.pool_size, sampling=args.opponent_sampling,
        latest_prob=args.latest_prob, trusted=args.trust_checkpoint,
    )
    if resume_path:
        pool.restore(resume_path, ckpt, global_step)
        if not pool.snapshots:
            print("No historical opponent snapshots found. Seeding the pool from the resumed learner.")
    if not pool.snapshots:
        pool.add_snapshot(learner, step=global_step)

    anchor_opponents = []
    if args.eval_freq > 0:
        for path in args.eval_anchors:
            opponent = load_frozen_opponent(path, device, trusted=args.trust_checkpoint)
            digest = hashlib.sha256()
            with open(path, "rb") as checkpoint_file:
                for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(block)
            anchor_opponents.append((path, digest.hexdigest(), opponent))

    # Logging setup
    log_file = open(os.path.join(args.save_dir, "selfplay_training.log"), "a", buffering=1)
    log_file.write(f"Config: {json.dumps(vars(args))}\n")

    metrics_path = os.path.join(args.save_dir, "selfplay_metrics.csv")
    csv_exists = os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0
    csv_file = open(metrics_path, "a", buffering=1)
    if not csv_exists:
        csv_file.write("step,matches_played,win_rate_vs_pool,pool_size,loss,mean_q,epsilon,fps,elapsed_seconds\n")
        csv_file.flush()

    step_csv_path = os.path.join(args.save_dir, "selfplay_step_metrics.csv")
    step_csv_exists = os.path.exists(step_csv_path) and os.path.getsize(step_csv_path) > 0
    step_csv_file = open(step_csv_path, "a", buffering=1)
    if not step_csv_exists:
        step_csv_file.write("step,matches_played,pool_size,loss,mean_q,epsilon,fps,elapsed_seconds\n")
        step_csv_file.flush()

    def close_files():
        log_file.close()
        csv_file.close()
        step_csv_file.close()

    interrupted = False
    def sigint_handler(signum, frame):
        nonlocal interrupted
        print("\n[Interrupt] Caught signal! Saving self-play state before exit...", flush=True)
        interrupted = True

    signal.signal(signal.SIGINT, sigint_handler)

    game = TwoPlayerPong()

    # Checkpoint helper function
    def save_checkpoint(step_num: int, is_latest: bool = False):
        extra = {
            "global_step": step_num,
            "matches_played": matches_played,
            "pool_state": pool.state_dict(),
            "pool_snapshots": pool.snapshot_names,
            "tracker_history": tracker.state_dict(),
            "config": vars(args),
            "rng_state": capture_rng_state(),
        }
        name = "latest.pt" if is_latest else f"dqn_selfplay_step_{step_num}.pt"
        ckpt_path = os.path.join(args.save_dir, name)
        save_replay = args.save_buffer and (is_latest or step_num % 100_000 == 0)
        learner.save(ckpt_path, extra_state=extra, replay_buffer=replay_buffer if save_replay else None)
        print(f"--> Saved checkpoint to {ckpt_path}", flush=True)

    t0 = time.time()
    session_start_step = global_step

    try:
        while global_step < target_total_steps and not interrupted:
            matches_played += 1
            opp_agent, opp_path = pool.sample_opponent()
            learner_is_right = (random.random() < 0.5)

            s_r, s_l = game.reset(seed=args.seed + matches_played)
            match_done = False
            r_pts, l_pts = 0, 0

            while not match_done and global_step < target_total_steps and not interrupted:
                global_step += 1
                learner.env_steps = global_step

                # Learner explores with its schedule; the opponent plays greedily with 5% noise
                # to avoid deterministic cycles
                if learner_is_right:
                    a_r = learner.select_action(s_r, deterministic=False)
                    a_l = policy_action(opp_agent, s_l, epsilon=0.05)
                else:
                    a_l = learner.select_action(s_l, deterministic=False)
                    a_r = policy_action(opp_agent, s_r, epsilon=0.05)

                r_gain, l_gain, game_over = game.step(a_r, a_l)
                r_pts += r_gain
                l_pts += l_gain
                match_done = game_over or r_pts >= 21 or l_pts >= 21
                next_s_r, next_s_l = game.observe()

                # Clipped raw game reward per player: +1 point won, -1 point lost, and -1 when the
                # player holding the ball fails to serve in time (PettingZoo's serve timer)
                r_rew = float(np.sign(game.right_reward))
                l_rew = float(np.sign(game.left_reward))

                # Every point (won or lost) ends the bootstrapped trajectory, as does game over.
                # A PettingZoo time-limit cutoff is not a real ending, so it still bootstraps.
                game_ended = match_done and not game.truncated
                point_terminal = (r_gain > 0 or l_gain > 0 or game_ended)
                if learner_is_right:
                    replay_buffer.push(s_r, a_r, r_rew, next_s_r, point_terminal)
                else:
                    replay_buffer.push(s_l, a_l, l_rew, next_s_l, point_terminal)

                s_r, s_l = next_s_r, next_s_l

                # Gradient Descent Update
                if len(replay_buffer) >= args.replay_start_size and global_step % args.train_freq == 0:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(args.batch_size)
                    update_metrics = learner.update(b_s, b_a, b_r, b_ns, b_d)
                    tracker.update(loss=update_metrics["loss"], mean_q=update_metrics["mean_q"])

                if match_done:
                    learner_pts, opp_pts = (r_pts, l_pts) if learner_is_right else (l_pts, r_pts)
                    pool.record_result(opp_path, 1.0 if learner_pts > opp_pts else 0.5 if learner_pts == opp_pts else 0.0)

                # Periodic Console Logging
                if global_step % args.log_freq == 0:
                    elapsed = time.time() - t0
                    fps = (global_step - session_start_step) / max(1.0, elapsed)
                    loss_val = tracker.mean("loss")
                    q_val = tracker.mean("mean_q")
                    eps_val = learner.get_epsilon()
                    log_str = (
                        f"Step {global_step:7d} | Matches: {matches_played:5d} | "
                        f"Pool: {len(pool.snapshots):2d} | Avg Q: {q_val:6.2f} | "
                        f"Loss: {loss_val:7.4f} | Eps: {eps_val:.3f} | FPS: {fps:4.0f}"
                    )
                    print(log_str, flush=True)
                    log_file.write(log_str + "\n")

                    step_csv_file.write(
                        f"{global_step},{matches_played},{len(pool.snapshots)},"
                        f"{loss_val:.6f},{q_val:.4f},{eps_val:.4f},{fps:.1f},{elapsed:.1f}\n"
                    )

                # Periodic Evaluation & Pool Promotion
                if args.eval_freq > 0 and global_step % args.eval_freq == 0 and len(replay_buffer) >= args.replay_start_size:
                    eval_res = evaluate_learner_vs_pool(
                        learner, pool,
                        num_matches=args.eval_matches,
                        target_score=args.eval_target_score,
                        seed=args.eval_seed,
                    )
                    eval_str = (
                        f"\n>>> [SELF-PLAY EVAL @ Step {global_step:7d}] "
                        f"Win-Rate vs Pool: {eval_res['win_rate']:5.1f}% "
                        f"({eval_res['wins']}W/{eval_res['draws']}D of {eval_res['matches']}) | "
                        f"Score: {eval_res['pts_for']}-{eval_res['pts_against']}\n"
                    )
                    print(eval_str, flush=True)
                    log_file.write(eval_str + "\n")

                    if anchor_opponents:
                        evaluate_anchors(learner, anchor_opponents, args, global_step)

                    # Check for promotion to pool
                    if eval_res["win_rate"] >= args.promotion_threshold:
                        pool.add_snapshot(learner, step=global_step)

                    # Save latest checkpoint
                    save_checkpoint(global_step, is_latest=True)

                    elapsed = time.time() - t0
                    fps = (global_step - session_start_step) / max(1.0, elapsed)
                    csv_file.write(
                        f"{global_step},{matches_played},{eval_res['win_rate']:.2f},"
                        f"{len(pool.snapshots)},{tracker.mean('loss'):.6f},{tracker.mean('mean_q'):.4f},"
                        f"{learner.get_epsilon():.4f},{fps:.1f},{elapsed:.1f}\n"
                    )

                # Periodic Checkpointing
                if global_step % args.save_freq == 0:
                    save_checkpoint(global_step, is_latest=True)
                    save_checkpoint(global_step, is_latest=False)

    except Exception as e:
        print(f"\n[CRASH DETECTED] Unhandled exception occurred at Step {global_step}: {e}", flush=True)
        print("Saving emergency recovery checkpoint before exit...", flush=True)
        save_checkpoint(global_step, is_latest=True)
        close_files()
        game.close()
        raise

    save_checkpoint(global_step, is_latest=True)
    game.close()
    if interrupted:
        print("Checkpoint saved successfully. Exiting gracefully.", flush=True)
        close_files()
        sys.exit(0)

    final_path = os.path.join(args.save_dir, "selfplay_final.pt")
    learner.save(final_path, {
        "global_step": global_step,
        "matches_played": matches_played,
        "pool_snapshots": pool.snapshot_names,
        "pool_state": pool.state_dict(),
        "config": vars(args),
    })
    print(f"\n[DONE] Self-Play training concluded at step {global_step:,}. Final model saved to {final_path}")
    close_files()


if __name__ == "__main__":
    main()
