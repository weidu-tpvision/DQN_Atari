"""Training script for Atari DQN (Nature 2015 / Double DQN).

Features:
- Fully resumable from checkpoint (--resume)
- Saves/loads replay buffer state
- Graceful shutdown on Ctrl+C (saves state before exiting)
- Optional Double DQN flag (--double-dqn)
"""

import os
import sys
import time
import random
import signal
import csv
import argparse
import numpy as np
import torch

from src.wrappers import make_atari_env
from src.model import NatureDQN
from src.replay_buffer import ReplayBuffer
from src.agent import DQNAgent
from src.utils import set_seed, get_device, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Atari")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Gymnasium Atari environment ID")
    parser.add_argument("--total-steps", type=int, default=1_500_000, help="Total environment interaction steps")
    parser.add_argument("--additional-steps", type=int, default=None, help="Train for an additional N steps from checkpoint")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--replay-start-size", type=int, default=10_000, help="Warm-up steps before training starts")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size for SGD")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Adam)")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--eps-end", type=float, default=0.02, help="Final exploration rate")
    parser.add_argument("--eps-decay", type=int, default=250_000, help="Steps to decay epsilon to min")
    parser.add_argument("--target-sync-freq", type=int, default=1_000, help="Target network update frequency")
    parser.add_argument("--train-freq", type=int, default=4, help="Agent steps between training updates")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Step interval for saving checkpoints")
    parser.add_argument("--eval-freq", type=int, default=25_000, help="Step interval to run greedy policy evaluation (<=0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of greedy evaluation episodes")
    parser.add_argument("--log-freq", type=int, default=1_000, help="Step interval for console progress")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--double-dqn", action="store_true", help="Enable Double DQN target calculation")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--replay-path", type=str, default=None, help="Path to replay buffer npz to resume from")
    parser.add_argument("--save-buffer", action="store_true", default=True, help="Whether to save replay buffer on checkpoints")
    parser.add_argument("--early-stop-score", type=float, default=None, help="Stop early if mean evaluation score reaches this threshold (None to disable)")
    parser.add_argument("--patience", type=int, default=0, help="Stop early if no improvement over best evaluation score for N evaluations (0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate_policy(agent, env_id: str, num_episodes: int = 5, seed: int = 123) -> dict:
    """Run deterministic greedy policy evaluation on full episodes with raw unclipped rewards."""
    eval_env = make_atari_env(env_id, clip_rewards=False, episodic_life=False, render_mode=None)
    scores = []
    lengths = []
    for ep in range(num_episodes):
        state, _ = eval_env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_length = 0
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, term, trunc, _ = eval_env.step(action)
            ep_reward += reward
            ep_length += 1
            done = term or trunc
            state = next_state
        scores.append(ep_reward)
        lengths.append(ep_length)
    eval_env.close()
    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "mean_length": float(np.mean(lengths)),
    }


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    # Create training environment
    env = make_atari_env(args.env, clip_rewards=True, episodic_life=True)
    num_actions = env.action_space.n
    print(f"Environment: {args.env} | Action Space: {num_actions} actions | Device: {device}")
    if args.double_dqn:
        print("Double DQN enabled: Action selection via online network, evaluation via target network.")

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_shape=(4, 84, 84),
        device=device,
    )

    # Initialize DQN Agent
    agent = DQNAgent(
        num_actions=num_actions,
        in_channels=4,
        gamma=args.gamma,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay,
        target_sync_freq=args.target_sync_freq,
        double_dqn=args.double_dqn,
        device=device,
    )

    tracker = MetricTracker(window_size=100)
    start_step = 1
    episode_count = 0

    # Handle Resumption
    if args.resume:
        print(f"\n--> Resuming training from checkpoint: {args.resume}")
        checkpoint_data = agent.load(args.resume)
        start_step = checkpoint_data.get("global_step", checkpoint_data.get("total_steps", 0)) + 1
        episode_count = checkpoint_data.get("episode_count", 0)
        episode_reward = checkpoint_data.get("episode_reward", 0.0)
        episode_length = checkpoint_data.get("episode_length", 0)
        agent.total_steps = checkpoint_data.get("total_steps", start_step)

        # Restore metric tracker history so running averages are continuous
        if "tracker_history" in checkpoint_data:
            tracker.history = checkpoint_data["tracker_history"]

        # Restore RNG states for exact pseudo-random sequence continuity
        if "rng_state" in checkpoint_data:
            rng = checkpoint_data["rng_state"]
            try:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                if rng.get("torch") is not None:
                    torch.set_rng_state(rng["torch"].clone().detach().cpu())
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(rng["cuda"].clone().detach().cpu())
            except Exception as e:
                print(f"Note: Could not restore full RNG state ({e}), continuing with fresh RNG.")

        print(f"Resumed at Step: {start_step} | Episodes: {episode_count} | Epsilon: {agent.get_epsilon():.3f}")

        # Check for replay buffer file intelligently:
        ckpt_dir = os.path.dirname(args.resume) or "."
        base_no_ext = os.path.splitext(args.resume)[0]
        step_num = checkpoint_data.get("global_step")

        candidates = [
            args.replay_path,
            f"{base_no_ext}_replay.npz",
            os.path.join(ckpt_dir, f"replay_step_{step_num}.npz") if step_num else None,
            os.path.join(ckpt_dir, "latest_replay.npz"),
            os.path.join(args.save_dir, "latest_replay.npz"),
        ]
        buffer_file = next((c for c in candidates if c and os.path.exists(c)), None)

        if buffer_file:
            print(f"Loading replay buffer from {buffer_file}...")
            replay_buffer.load(buffer_file)
            print(f"Replay buffer restored with {len(replay_buffer)} transitions.")
        else:
            print("No saved replay buffer file found for this checkpoint.")
            print("Fast-populating buffer using loaded policy to avoid cold-start random noise...")
            s, _ = env.reset(seed=args.seed)
            for _ in range(min(args.replay_start_size, args.buffer_size)):
                a = agent.select_action(s, deterministic=False)
                ns, r, term, trunc, _ = env.step(a)
                replay_buffer.push(s, a, r, ns, term)
                s = ns if not (term or trunc) else env.reset()[0]
            print(f"Buffer populated with {len(replay_buffer)} initial transitions.")

    # Determine target total steps
    if args.additional_steps is not None:
        target_total_steps = (start_step - 1) + args.additional_steps
        print(f"--> Training for an additional {args.additional_steps} steps (target global step: {target_total_steps}).")
    else:
        target_total_steps = args.total_steps

    # Guard against target step already passed
    if start_step > target_total_steps:
        print(f"\n[INFO] Checkpoint is already at step {start_step - 1}, which already meets or exceeds target steps ({target_total_steps}).")
        print(f"Nothing to train. To train further, specify a higher --total-steps (e.g. --total-steps {start_step + 100_000}) or use --additional-steps.")
        env.close()
        return

    # Graceful shutdown handler
    interrupted = False
    def sigint_handler(signum, frame):
        nonlocal interrupted
        print("\n[Interrupt] Caught signal! Saving latest progress before exit...", flush=True)
        interrupted = True

    signal.signal(signal.SIGINT, sigint_handler)

    log_file_path = os.path.join(args.save_dir, "training.log")
    log_file = open(log_file_path, "a", buffering=1)

    csv_file_path = os.path.join(args.save_dir, "training_metrics.csv")
    csv_exists = os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0
    csv_file = open(csv_file_path, "a", buffering=1)
    if not csv_exists:
        csv_file.write("step,episode,avg_reward_100,avg_length_100,loss,mean_q,epsilon,fps,elapsed_seconds\n")
        csv_file.flush()

    eval_csv_path = os.path.join(args.save_dir, "eval_metrics.csv")
    eval_csv_exists = os.path.exists(eval_csv_path) and os.path.getsize(eval_csv_path) > 0
    eval_csv_file = open(eval_csv_path, "a", buffering=1)
    if not eval_csv_exists:
        eval_csv_file.write("step,mean_score,std_score,min_score,max_score,mean_length,elapsed_seconds\n")
        eval_csv_file.flush()

    best_eval_score = -float("inf")
    if args.resume and "checkpoint_data" in locals():
        best_eval_score = checkpoint_data.get("best_eval_score", -float("inf"))
    patience_counter = 0

    state, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_length = 0
    start_time = time.time()

    if start_step <= args.replay_start_size and len(replay_buffer) < args.replay_start_size:
        print("\nStarting Warmup (filling replay buffer)...", flush=True)

    def save_checkpoint(step_num: int, is_latest: bool = False):
        extra = {
            "global_step": step_num,
            "episode_count": episode_count,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "best_eval_score": best_eval_score,
            "total_steps": agent.total_steps,
            "tracker_history": tracker.history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        # Save model
        name = "latest.pt" if is_latest else f"dqn_{args.env.replace('/', '_')}_step_{step_num}.pt"
        ckpt_path = os.path.join(args.save_dir, name)
        agent.save(ckpt_path, extra_state=extra)
        
        # Save replay buffer for complete resumption
        if args.save_buffer and (is_latest or step_num % 100_000 == 0):
            buf_name = "latest_replay.npz" if is_latest else f"replay_step_{step_num}.npz"
            buf_path = os.path.join(args.save_dir, buf_name)
            replay_buffer.save(buf_path)
            print(f"--> Saved complete training state & buffer to {ckpt_path} and {buf_path}", flush=True)
        else:
            print(f"--> Saved checkpoint to {ckpt_path}", flush=True)

    current_step = start_step
    try:
        for step in range(start_step, target_total_steps + 1):
            current_step = step
            if interrupted:
                save_checkpoint(step, is_latest=True)
                print("Checkpoint saved successfully. Exiting gracefully.")
                sys.exit(0)

            # Action selection (random during warm-up if buffer not yet filled, epsilon-greedy afterwards)
            if len(replay_buffer) < args.replay_start_size:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition in buffer
            replay_buffer.push(state, action, reward, next_state, terminated)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                episode_count += 1
                tracker.update(ep_reward=episode_reward, ep_length=episode_length)
                state, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0

            # Training step
            if len(replay_buffer) >= args.replay_start_size and step % args.train_freq == 0:
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(args.batch_size)
                metrics = agent.update(b_states, b_actions, b_rewards, b_next_states, b_dones)
                tracker.update(loss=metrics["loss"], mean_q=metrics["mean_q"])

            # Logging & Comparison CSV Writing
            if step % args.log_freq == 0 and len(replay_buffer) >= args.replay_start_size:
                elapsed = time.time() - start_time
                fps = int((step - start_step + 1) / max(1e-3, elapsed))
                avg_rew = tracker.mean("ep_reward")
                avg_len = tracker.mean("ep_length")
                avg_loss = tracker.mean("loss")
                avg_q = tracker.mean("mean_q")
                eps = agent.get_epsilon()

                # Console log
                log_str = (
                    f"Step {step:7d} | Ep: {episode_count:4d} | "
                    f"Avg Rew (100 ep): {avg_rew:6.2f} | "
                    f"Avg Q: {avg_q:6.2f} | Loss: {avg_loss:7.4f} | "
                    f"Eps: {eps:5.3f} | FPS: {fps:4d}"
                )
                print(log_str, flush=True)
                log_file.write(log_str + "\n")
                log_file.flush()

                # Structured CSV log for comparisons across runs/checkpoints
                csv_line = f"{step},{episode_count},{avg_rew:.4f},{avg_len:.2f},{avg_loss:.6f},{avg_q:.4f},{eps:.4f},{fps},{elapsed:.2f}\n"
                csv_file.write(csv_line)
                csv_file.flush()
                os.fsync(csv_file.fileno())

            # Periodic Evaluation (Deterministic Greedy Policy)
            if args.eval_freq > 0 and step % args.eval_freq == 0 and len(replay_buffer) >= args.replay_start_size:
                elapsed = time.time() - start_time
                eval_res = evaluate_policy(agent, args.env, num_episodes=args.eval_episodes, seed=args.seed + step)
                eval_str = (
                    f"\n>>> [EVALUATION @ Step {step:7d}] Greedy Test Score ({args.eval_episodes} ep): "
                    f"{eval_res['mean_score']:6.2f} +/- {eval_res['std_score']:5.2f} "
                    f"(Min: {eval_res['min_score']:5.1f} | Max: {eval_res['max_score']:5.1f} | Avg Len: {eval_res['mean_length']:5.1f})\n"
                )
                print(eval_str, flush=True)
                log_file.write(eval_str + "\n")
                log_file.flush()

                eval_csv_line = f"{step},{eval_res['mean_score']:.4f},{eval_res['std_score']:.4f},{eval_res['min_score']:.1f},{eval_res['max_score']:.1f},{eval_res['mean_length']:.2f},{elapsed:.2f}\n"
                eval_csv_file.write(eval_csv_line)
                eval_csv_file.flush()
                os.fsync(eval_csv_file.fileno())

                # Checkpoint best performing model
                if eval_res["mean_score"] > best_eval_score:
                    best_eval_score = eval_res["mean_score"]
                    patience_counter = 0
                    best_path = os.path.join(args.save_dir, "best_model.pt")
                    agent.save(
                        best_path,
                        extra_state={
                            "global_step": step,
                            "episode_count": episode_count,
                            "best_eval_score": best_eval_score,
                        },
                    )
                    print(f"--> [NEW BEST MODEL] High score: {best_eval_score:6.2f} saved to {best_path}!\n", flush=True)
                else:
                    if args.patience > 0:
                        patience_counter += 1
                        print(f"--> [PLATEAU CHECK] Evaluations without new high: {patience_counter}/{args.patience} (Best: {best_eval_score:.2f})\n", flush=True)

                # Early Stopping: Target superhuman ceiling reached (if configured)
                if args.early_stop_score is not None and eval_res["mean_score"] >= args.early_stop_score:
                    print(f"\n🎯 [TARGET CEILING REACHED] Evaluation score {eval_res['mean_score']:.2f} >= {args.early_stop_score:.2f}! Terminating early.", flush=True)
                    break

                # Early Stopping: Plateau reached (if configured)
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"\n⏹️ [CONVERGENCE PLATEAU REACHED] No improvement for {patience_counter} consecutive evaluations ({patience_counter * args.eval_freq} steps). Training complete!\n", flush=True)
                    break

            # Checkpointing
            if step % args.save_freq == 0:
                save_checkpoint(step, is_latest=True)
                save_checkpoint(step, is_latest=False)

    except Exception as e:
        print(f"\n[CRASH DETECTED] Unhandled exception occurred at Step {current_step}: {e}", flush=True)
        print("Saving emergency recovery checkpoint before exit...", flush=True)
        save_checkpoint(current_step, is_latest=True)
        log_file.close()
        csv_file.close()
        eval_csv_file.close()
        env.close()
        raise e

    # Final save
    final_step = current_step
    save_checkpoint(final_step, is_latest=True)
    final_path = os.path.join(args.save_dir, f"dqn_{args.env.replace('/', '_')}_final.pt")
    agent.save(final_path, extra_state={"global_step": final_step, "episode_count": episode_count, "best_eval_score": best_eval_score})
    print(f"\nTraining completed! Saved final model to {final_path}", flush=True)
    log_file.close()
    csv_file.close()
    eval_csv_file.close()
    env.close()


if __name__ == "__main__":
    train(parse_args())
