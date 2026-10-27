"""High-performance DQN training script for 2048.

Features:
- Multi-scale directional convolutional Q-Network (DQN2048)
- Action masking: prevents illegal moves during exploration and greedy selection, and excludes
  them from the bootstrap target (the replay buffer stores each next state's legal mask)
- Resumable checkpoints with RNG state and replay buffer serialization
- Real-time tracking of max tile distributions (% reaching 256, 512, 1024, 2048)
- Throughput: ~600-1,200 steps/s while training on GPU (the game alone runs ~10k steps/s)
"""

import os
import sys
import json
import time
import random
import signal
import argparse
from typing import Dict, Any
from collections import deque

import numpy as np
import torch

from src.games.env_2048 import Env2048
from src.model_2048 import DQN2048
from src.replay_buffer import ReplayBuffer
from src.agent import DQNAgent
from src.utils import set_seed, get_device, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on 2048")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Total environment interaction steps")
    parser.add_argument("--additional-steps", type=int, default=None, help="Train for an additional N steps from checkpoint")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--replay-start-size", type=int, default=5_000, help="Warm-up steps before training starts")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size for SGD")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (Adam)")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--eps-end", type=float, default=0.02, help="Final exploration rate")
    parser.add_argument("--eps-decay", type=int, default=200_000, help="Environment steps to decay epsilon to min")
    parser.add_argument("--target-sync-freq", type=int, default=1_000, help="Target network update frequency (gradient updates)")
    parser.add_argument("--train-freq", type=int, default=2, help="Agent steps between training updates")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Step interval for saving checkpoints")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Step interval for greedy policy evaluation")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of greedy evaluation episodes (same fixed seeds every time)")
    parser.add_argument("--log-freq", type=int, default=1_000, help="Step interval for console progress")
    parser.add_argument("--save-dir", type=str, default="checkpoints_2048", help="Directory to save checkpoints")
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=True, help="Enable Double DQN")
    parser.add_argument("--action-masking", action=argparse.BooleanOptionalAction, default=True, help="Mask illegal moves")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--save-buffer", action=argparse.BooleanOptionalAction, default=True, help="Save replay buffer on checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate_policy(agent: DQNAgent, num_episodes: int = 10, action_masking: bool = True, seed: int = 1000) -> Dict[str, Any]:
    """Run greedy evaluation episodes on 2048."""
    env = Env2048()
    scores = []
    moves_list = []
    max_tiles = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_score = 0
        ep_moves = 0
        max_t = 0

        while not done:
            mask = info["legal_mask"] if action_masking else None
            action = agent.select_action(obs, deterministic=True, action_mask=mask)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            ep_score = info["score"]
            ep_moves = info["moves"]
            max_t = info["max_tile"]

        scores.append(ep_score)
        moves_list.append(ep_moves)
        max_tiles.append(max_t)

    env.close()
    max_tiles_arr = np.array(max_tiles)
    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "mean_moves": float(np.mean(moves_list)),
        "highest_tile": int(np.max(max_tiles_arr)),
        "pct_256": float(np.mean(max_tiles_arr >= 256) * 100.0),
        "pct_512": float(np.mean(max_tiles_arr >= 512) * 100.0),
        "pct_1024": float(np.mean(max_tiles_arr >= 1024) * 100.0),
        "pct_2048": float(np.mean(max_tiles_arr >= 2048) * 100.0),
    }


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    print(f"\n=======================================================")
    print(f"             2048 DQN TRAINING ENGINE                  ")
    print(f"=======================================================")
    print(f"Target Steps:   {args.total_steps:,} | Buffer: {args.buffer_size:,}")
    print(f"Batch Size:     {args.batch_size} | LR: {args.lr} | Gamma: {args.gamma}")
    print(f"Double DQN:     {args.double_dqn} | Action Masking: {args.action_masking}")
    print(f"Save Directory: {args.save_dir} | Device: {device}")
    print(f"-------------------------------------------------------\n")

    env = Env2048()

    # Experience replay buffer stores (16, 4, 4) uint8 binary one-hot tensors
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_shape=(16, 4, 4),
        device=device,
        action_mask_dim=4 if args.action_masking else None,
    )

    # Initialize DQN Agent with custom DQN2048 architecture
    agent = DQNAgent(
        num_actions=4,
        gamma=args.gamma,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay,
        target_sync_freq=args.target_sync_freq,
        double_dqn=args.double_dqn,
        device=device,
        model_fn=lambda: DQN2048(in_channels=16, num_actions=4),
    )

    tracker = MetricTracker(window_size=100)
    start_step = 1
    episode_count = 0
    best_eval_score = -float("inf")

    # Logging setup
    log_path = os.path.join(args.save_dir, "training.log")
    log_file = open(log_path, "a", buffering=1)
    log_file.write(f"Config: {json.dumps(vars(args))}\n")

    csv_path = os.path.join(args.save_dir, "training_metrics.csv")
    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    csv_file = open(csv_path, "a", buffering=1)
    if not csv_exists:
        csv_file.write("step,episode,avg_score_100,avg_moves_100,max_tile_100,loss,mean_q,epsilon,fps,elapsed_seconds\n")
        csv_file.flush()

    eval_csv_path = os.path.join(args.save_dir, "eval_metrics.csv")
    eval_csv_exists = os.path.exists(eval_csv_path) and os.path.getsize(eval_csv_path) > 0
    eval_csv_file = open(eval_csv_path, "a", buffering=1)
    if not eval_csv_exists:
        eval_csv_file.write("step,mean_score,max_score,mean_moves,highest_tile,pct_256,pct_512,pct_1024,pct_2048,elapsed_seconds\n")
        eval_csv_file.flush()

    # Resumption
    if args.resume:
        print(f"\n--> Resuming training from checkpoint: {args.resume}")
        ckpt = agent.load(args.resume)  # restores env_steps (epsilon schedule) and num_updates
        start_step = agent.env_steps + 1
        episode_count = ckpt.get("episode_count", 0)
        best_eval_score = ckpt.get("best_eval_score", -float("inf"))
        if "tracker_history" in ckpt:
            tracker.load(ckpt["tracker_history"])
        if "rng_state" in ckpt:
            rng = ckpt["rng_state"]
            try:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                if rng.get("torch") is not None:
                    torch.set_rng_state(rng["torch"].clone().detach().cpu())
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(rng["cuda"].clone().detach().cpu())
            except Exception as e:
                print(f"Note: Could not restore full RNG state ({e}), continuing with fresh RNG.")
        print(f"Resumed at step {start_step:,} | Episodes: {episode_count:,} | Epsilon: {agent.get_epsilon():.3f}")

        # Restore replay buffer
        buf_cand = [
            os.path.join(os.path.dirname(args.resume), "latest_replay.npz"),
            os.path.join(args.save_dir, "latest_replay.npz"),
        ]
        buf_file = next((b for b in buf_cand if os.path.exists(b)), None)
        if buf_file:
            print(f"Restoring replay buffer from {buf_file}...")
            replay_buffer.load(buf_file)

    if args.additional_steps is not None:
        target_total_steps = (start_step - 1) + args.additional_steps
    else:
        target_total_steps = args.total_steps

    interrupted = False
    def sigint_handler(signum, frame):
        nonlocal interrupted
        print("\n[Interrupt] Caught signal! Saving 2048 checkpoint before exit...", flush=True)
        interrupted = True

    signal.signal(signal.SIGINT, sigint_handler)

    def save_checkpoint(step_num: int, is_latest: bool = False):
        extra = {
            "global_step": step_num,
            "episode_count": episode_count,
            "best_eval_score": best_eval_score,
            "tracker_history": tracker.history,
            "config": vars(args),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        name = "latest.pt" if is_latest else f"dqn_2048_step_{step_num}.pt"
        ckpt_path = os.path.join(args.save_dir, name)
        agent.save(ckpt_path, extra_state=extra)

        if args.save_buffer and (is_latest or step_num % 100_000 == 0):
            buf_name = "latest_replay.npz" if is_latest else f"replay_step_{step_num}.npz"
            buf_path = os.path.join(args.save_dir, buf_name)
            replay_buffer.save(buf_path)
            print(f"--> Saved checkpoint & replay buffer to {ckpt_path} and {buf_path}", flush=True)
        else:
            print(f"--> Saved checkpoint to {ckpt_path}", flush=True)

    # The game in progress isn't checkpointed, so a resumed run starts a fresh (differently seeded) game
    state, info = env.reset(seed=args.seed + start_step - 1)
    recent_scores = deque(maxlen=100)
    recent_moves = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    t0 = time.time()
    current_step = start_step

    try:
        for step in range(start_step, target_total_steps + 1):
            current_step = step

            if interrupted:
                # Step `step` has not run yet, so the last completed step is step - 1
                save_checkpoint(step - 1, is_latest=True)
                print("Checkpoint saved successfully. Exiting gracefully.", flush=True)
                sys.exit(0)

            agent.env_steps = step

            # Action selection
            mask = info["legal_mask"] if args.action_masking else None
            if len(replay_buffer) < args.replay_start_size:
                if mask is not None:
                    legal_acts = np.where(mask)[0]
                    action = int(np.random.choice(legal_acts)) if len(legal_acts) > 0 else 0
                else:
                    action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False, action_mask=mask)

            next_state, reward, done, trunc, next_info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done, next_mask=next_info["legal_mask"])

            state = next_state
            info = next_info

            if done or trunc:
                episode_count += 1
                recent_scores.append(info["score"])
                recent_moves.append(info["moves"])
                recent_max_tiles.append(info["max_tile"])
                state, info = env.reset()

            # Gradient update
            if len(replay_buffer) >= args.replay_start_size and step % args.train_freq == 0:
                # With action masking the batch also carries next-state legal masks for the target
                metrics = agent.update(*replay_buffer.sample(args.batch_size))
                tracker.update(loss=metrics["loss"], mean_q=metrics["mean_q"])

            # Console logging
            if step % args.log_freq == 0 and len(replay_buffer) >= args.replay_start_size:
                elapsed = time.time() - t0
                fps = int((step - start_step + 1) / max(1e-3, elapsed))
                avg_sc = np.mean(recent_scores) if recent_scores else 0.0
                avg_mv = np.mean(recent_moves) if recent_moves else 0.0
                max_t = np.max(recent_max_tiles) if recent_max_tiles else 0
                loss_v = tracker.mean("loss")
                q_v = tracker.mean("mean_q")
                eps_v = agent.get_epsilon()

                log_str = (
                    f"Step {step:7d} | Ep: {episode_count:5d} | "
                    f"Avg Score: {avg_sc:6.0f} | Max Tile: {max_t:4d} | "
                    f"Avg Q: {q_v:6.2f} | Loss: {loss_v:7.4f} | "
                    f"Eps: {eps_v:.3f} | FPS: {fps:4d}"
                )
                print(log_str, flush=True)
                log_file.write(log_str + "\n")
                log_file.flush()

                csv_file.write(
                    f"{step},{episode_count},{avg_sc:.1f},{avg_mv:.1f},{max_t},"
                    f"{loss_v:.6f},{q_v:.4f},{eps_v:.4f},{fps},{elapsed:.1f}\n"
                )
                csv_file.flush()

            # Greedy evaluation
            if args.eval_freq > 0 and step % args.eval_freq == 0 and len(replay_buffer) >= args.replay_start_size:
                elapsed = time.time() - t0
                # Fixed evaluation seeds so scores are comparable across evaluations (best-model selection)
                res = evaluate_policy(agent, num_episodes=args.eval_episodes, action_masking=args.action_masking, seed=10_000 + args.seed)
                eval_str = (
                    f"\n>>> [2048 EVAL @ Step {step:7d}] "
                    f"Mean Score: {res['mean_score']:6.0f} (Max: {res['max_score']}) | "
                    f"Highest Tile: {res['highest_tile']:4d} | "
                    f">=512: {res['pct_512']:4.0f}% | >=1024: {res['pct_1024']:4.0f}% | >=2048: {res['pct_2048']:4.0f}%\n"
                )
                print(eval_str, flush=True)
                log_file.write(eval_str + "\n")
                log_file.flush()

                eval_csv_file.write(
                    f"{step},{res['mean_score']:.1f},{res['max_score']},{res['mean_moves']:.1f},"
                    f"{res['highest_tile']},{res['pct_256']:.1f},{res['pct_512']:.1f},"
                    f"{res['pct_1024']:.1f},{res['pct_2048']:.1f},{elapsed:.1f}\n"
                )
                eval_csv_file.flush()

                if res["mean_score"] > best_eval_score:
                    best_eval_score = res["mean_score"]
                    best_path = os.path.join(args.save_dir, "best_model.pt")
                    agent.save(best_path, {"global_step": step, "best_eval_score": best_eval_score, "config": vars(args)})
                    print(f"--> [NEW BEST MODEL] High score: {best_eval_score:.0f} saved to {best_path}!\n", flush=True)

            # Checkpointing
            if step % args.save_freq == 0:
                save_checkpoint(step, is_latest=True)
                save_checkpoint(step, is_latest=False)

    except Exception as e:
        print(f"\n[CRASH DETECTED] Unhandled exception at step {current_step}: {e}", flush=True)
        save_checkpoint(current_step, is_latest=True)
        log_file.close()
        csv_file.close()
        eval_csv_file.close()
        env.close()
        raise

    # Final save
    save_checkpoint(current_step, is_latest=True)
    final_path = os.path.join(args.save_dir, "dqn_2048_final.pt")
    agent.save(final_path, {"global_step": current_step, "best_eval_score": best_eval_score, "config": vars(args)})
    print(f"\n[DONE] 2048 training complete at step {current_step:,}. Final model saved to {final_path}")
    log_file.close()
    csv_file.close()
    eval_csv_file.close()
    env.close()


if __name__ == "__main__":
    train(parse_args())
