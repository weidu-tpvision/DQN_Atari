"""Script to run a live visual demo or generate an animated GIF/video of the trained DQN playing Pong.

Usage:
    # 1. To view live on screen (requires X11/GUI):
    python record_demo.py --mode human

    # 2. To generate an animated GIF with HUD overlay:
    python record_demo.py --mode gif --output pong_demo.gif --frames 1000
"""

import os
import time
import argparse
import numpy as np
import cv2
import torch
import gymnasium as gym

# Avoid indirect OpenGL/GLX errors over SSH X11 forwarding
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

from src.wrappers import make_atari_env
from src.agent import DQNAgent
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained DQN playing Pong")
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/Pong-v5",
        help="Gymnasium Atari environment ID",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/champion_model.pt"
            if os.path.exists("checkpoints/champion_model.pt")
            else (
                "checkpoints/best_model.pt"
                if os.path.exists("checkpoints/best_model.pt")
                else "checkpoints/dqn_ALE_Pong-v5_final.pt"
            )
        ),
        help="Path to trained model weights",
    )
    parser.add_argument("--mode", type=str, choices=["human", "gif", "mp4"], default="human")
    parser.add_argument("--output", type=str, default="pong_demo.gif")
    parser.add_argument("--frames", type=int, default=0, help="Max frames (0 = unlimited for human mode, 1500 default for gif/mp4)")
    parser.add_argument("--fps", type=int, default=35, help="Playback FPS")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Use rgb_array so we can render via pure X11 (cv2.imshow) avoiding Pygame GLX issues
    env = make_atari_env(
        args.env,
        clip_rewards=False,
        episodic_life=False,
        render_mode="rgb_array",
    )

    agent = DQNAgent(num_actions=env.action_space.n, device=device)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from: {args.checkpoint}")

    state, _ = env.reset(seed=args.seed)
    recorded_frames = []
    
    agent_pts = 0
    cpu_pts = 0
    matches_won = 0
    matches_played = 0
    total_steps = 0
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

    max_steps = args.frames if (args.frames > 0 or args.mode != "human") else (1500 if args.mode != "human" else 10**9)

    if args.mode == "human":
        print("Starting live GUI window (Press 'q' or ESC on the window to exit)...")
        print("Agent will play full matches continuously. Press 'q' on the game window anytime to quit.")
        cv2.namedWindow("DQN Pong - Nature 2015", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DQN Pong - Nature 2015", 560, 420)

    step = 0
    while step < max_steps:
        step += 1
        # Forward pass to get Q-values and greedy action
        with torch.no_grad():
            s_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
            q_vals = agent.online_net(s_tensor).cpu().numpy()[0]
            action = int(np.argmax(q_vals))

        # Capture RGB frame from emulator
        rgb = env.unwrapped.render()
        
        # Scale up frame and add HUD overlay
        frame = cv2.resize(rgb, (560, 420), interpolation=cv2.INTER_NEAREST)
        
        # HUD overlay banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (560, 52), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # Text info
        diff = agent_pts - cpu_pts
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        cv2.putText(frame, f"Agent: {agent_pts}  |  CPU: {cpu_pts}  ({diff_str})", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if diff >= 0 else (0, 100, 255), 2)
        cv2.putText(frame, f"Action: {action_names[action]}  |  Matches: {matches_won}/{matches_played}", (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(frame, f"Max Q: {q_vals[action]:.2f}", (400, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        if args.mode == "human":
            # Display frame via OpenCV standard X11 window
            cv2.imshow("DQN Pong - Nature 2015", frame)
            key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC to break
                print("Exiting visual demo...")
                break
        else:
            recorded_frames.append(frame)

        next_state, reward, terminated, truncated, _ = env.step(action)
        if reward > 0:
            agent_pts += int(reward)
        elif reward < 0:
            cpu_pts += int(-reward)
        total_steps += 1
        state = next_state

        if terminated or truncated:
            matches_played += 1
            won = agent_pts > cpu_pts
            if won:
                matches_won += 1
            result_str = "WON" if won else "LOST"
            print(f"Match {matches_played:2d} Finished: Agent {agent_pts} - CPU {cpu_pts} (Diff: {diff_str}) -> {result_str} | Total Won: {matches_won}/{matches_played}")
            state, _ = env.reset()
            agent_pts = 0
            cpu_pts = 0

    if args.mode == "human":
        cv2.destroyAllWindows()
    env.close()

    if args.mode == "gif":
        print(f"Encoding {len(recorded_frames)} frames to animated GIF: {args.output}...")
        import imageio
        # Downsample slightly to keep GIF file size manageable
        gif_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in recorded_frames[::2]]
        imageio.mimsave(args.output, gif_frames, fps=args.fps // 2, loop=0)
        print(f"--> Successfully created GIF: {args.output} (Size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB)")
    elif args.mode == "mp4":
        print(f"Encoding {len(recorded_frames)} frames to MP4: {args.output}...")
        h, w, _ = recorded_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for f in recorded_frames:
            writer.write(f)
        writer.release()
        print(f"--> Successfully created MP4: {args.output}")


if __name__ == "__main__":
    main()
