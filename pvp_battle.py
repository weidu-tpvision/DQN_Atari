"""Head-to-Head PvP Arena: Two DQN Neural Networks battle each other in Atari Pong.

Both paddles are controlled by deep neural network checkpoints:
  - Player 1 (Right green paddle): Trained DQN checkpoint
  - Player 2 (Left orange paddle): Trained DQN checkpoint (with mirrored visual perception)

Usage:
    # 1. Live interactive match on screen over X11 forwarding:
    python pvp_battle.py --mode human --fps 35

    # 2. Battle between two specific checkpoints:
    python pvp_battle.py --p1 checkpoints/champion_model.pt --p2 checkpoints/dqn_ALE_Pong-v5_step_1200000.pt --mode human

    # 3. Fast tournament benchmark (alternating sides for fairness):
    python pvp_battle.py --matches 5 --target-score 7 --mode headless

    # 4. Export an animated GIF of the duel:
    python pvp_battle.py --mode gif --output pvp_battle.gif --frames 1200
"""

import os
import sys
import argparse
import numpy as np
import cv2

from src.agent import DQNAgent
from src.pong2p import TwoPlayerPong, NUM_ACTIONS, policy_action
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-Head DQN vs DQN Pong Arena")
    parser.add_argument(
        "--p1",
        type=str,
        default="checkpoints/champion_model.pt" if os.path.exists("checkpoints/champion_model.pt") else "checkpoints/best_model.pt",
        help="Checkpoint for Player 1 (Right paddle / Green by default)",
    )
    parser.add_argument(
        "--p2",
        type=str,
        default="checkpoints/dqn_ALE_Pong-v5_step_1200000.pt" if os.path.exists("checkpoints/dqn_ALE_Pong-v5_step_1200000.pt") else "checkpoints/dqn_ALE_Pong-v5_final.pt",
        help="Checkpoint for Player 2 (Left paddle / Orange by default)",
    )
    parser.add_argument("--mode", type=str, choices=["human", "headless", "gif", "mp4"], default="human")
    parser.add_argument("--matches", type=int, default=5, help="Number of matches to play")
    parser.add_argument("--target-score", type=int, default=7, help="Points needed to win a match (e.g. 7, 11, or 21)")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Slight exploration (2%%) to break infinite deterministic volley loops")
    parser.add_argument("--frames", type=int, default=1500, help="Max frames for gif/mp4 recording")
    parser.add_argument("--fps", type=int, default=35, help="Playback speed in human mode")
    parser.add_argument("--output", type=str, default="pvp_battle.gif", help="Output file path for gif/mp4")
    parser.add_argument("--swap-sides", action=argparse.BooleanOptionalAction, default=True, help="Alternate paddle sides between matches for fair play")
    parser.add_argument("--seed", type=int, default=200, help="Base seed for match resets and epsilon noise")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    p1_name = os.path.basename(args.p1).replace(".pt", "")
    p2_name = os.path.basename(args.p2).replace(".pt", "")

    print(f"\n==========================================================================")
    print(f"               DQN HEAD-TO-HEAD CHAMPIONSHIP (PvP ARENA)                  ")
    print(f"==========================================================================")
    print(f"Player 1: {p1_name}  (Right / Green initially)")
    print(f"Player 2: {p2_name}  (Left / Orange initially)")
    print(f"Rules: First to {args.target_score} points | Epsilon: {args.epsilon*100:.1f}% | Device: {device}")
    print(f"Side Swapping: {'Enabled (Alternating every game)' if args.swap_sides else 'Disabled'}")
    print(f"Mode: {args.mode}")
    print(f"--------------------------------------------------------------------------\n")

    # Load both agents
    agent1 = DQNAgent(num_actions=NUM_ACTIONS, device=device)
    agent1.load_weights(args.p1)

    agent2 = DQNAgent(num_actions=NUM_ACTIONS, device=device)
    agent2.load_weights(args.p2)

    display_action_map = {0: "NOOP", 1: "FIRE", 2: "UP", 3: "DOWN", 4: "UPFIRE", 5: "DOWNFIRE"}

    if args.mode == "human":
        if not os.environ.get("DISPLAY"):
            print("\n[ERROR] No graphical display detected (environment variable $DISPLAY is not set).")
            print("To view a live window over SSH:")
            print("  1. Reconnect with X11 forwarding: 'ssh -X user@host' or 'ssh -Y user@host'")
            print("  2. Alternatively, view live terminal scores: '--mode headless'")
            print("  3. Or render an animated replay to watch directly: '--mode gif --output pvp_duel.gif'\n")
            sys.exit(1)
        # Avoid indirect OpenGL/GLX errors over SSH X11 forwarding
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        cv2.namedWindow("DQN Head-to-Head Arena", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DQN Head-to-Head Arena", 640, 480)
        print("Starting live GUI window. Press 'q' or ESC on the game window anytime to exit.\n")

    render = args.mode in ["human", "gif", "mp4"]
    game = TwoPlayerPong(render_mode="rgb_array" if render else None)

    recorded_frames = []
    matches_played = 0
    p1_match_wins = 0
    p2_match_wins = 0
    draws = 0

    target_matches = args.matches if args.mode in ["headless", "human"] else 1
    total_frames_recorded = 0

    while matches_played < target_matches:
        # Check side assignment for this match
        swap = args.swap_sides and (matches_played % 2 == 1)
        # first_0 is ALWAYS Right paddle (Green), second_0 is ALWAYS Left paddle (Orange)
        right_agent = agent2 if swap else agent1
        left_agent = agent1 if swap else agent2
        right_label = p2_name if swap else p1_name
        left_label = p1_name if swap else p2_name

        right_state, left_state = game.reset(seed=args.seed + matches_played)
        right_pts = 0
        left_pts = 0

        print(f">>> STARTING MATCH {matches_played + 1}/{target_matches}:")
        print(f"    [Left / Orange]  {left_label}")
        print(f"    [Right / Green]  {right_label}")

        while True:
            total_frames_recorded += 1

            act_right = policy_action(right_agent, right_state, args.epsilon)
            act_left = policy_action(left_agent, left_state, args.epsilon)

            r_right, r_left, done = game.step(act_right, act_left)
            right_state, left_state = game.observe()

            if r_right > 0:
                right_pts += r_right
                print(f"    Point! [{right_label} scored] Scoreboard: {left_label} {left_pts} - {right_pts} {right_label}", flush=True)
            if r_left > 0:
                left_pts += r_left
                print(f"    Point! [{left_label} scored] Scoreboard: {left_label} {left_pts} - {right_pts} {right_label}", flush=True)

            # Visual Rendering (OpenCV expects BGR)
            if render:
                frame = cv2.cvtColor(game.render(), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)

                # HUD Overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (640, 60), (15, 15, 15), -1)
                frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

                # Left Player (Orange)
                l_act = display_action_map.get(act_left, "NOOP")
                cv2.putText(frame, f"[LEFT] {left_label[:14]}", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                cv2.putText(frame, f"Pts: {left_pts} | Act: {l_act}", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Scoreboard Center
                cv2.putText(frame, f"{left_pts}  -  {right_pts}", (275, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f"Match {matches_played+1} (To {args.target_score})", (270, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

                # Right Player (Green)
                r_act = display_action_map.get(act_right, "NOOP")
                cv2.putText(frame, f"[RIGHT] {right_label[:14]}", (440, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(frame, f"Pts: {right_pts} | Act: {r_act}", (440, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if args.mode == "human":
                    cv2.imshow("DQN Head-to-Head Arena", frame)
                    key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
                    if key == ord("q") or key == 27:
                        print("\nExiting PvP Arena...")
                        game.close()
                        cv2.destroyAllWindows()
                        return
                else:
                    recorded_frames.append(frame)

            # Check match victory condition (first to target_score)
            if right_pts >= args.target_score or left_pts >= args.target_score or done:
                break
            if args.mode in ["gif", "mp4"] and total_frames_recorded >= args.frames:
                break

        matches_played += 1
        p1_pts = left_pts if swap else right_pts
        p2_pts = right_pts if swap else left_pts

        if p1_pts > p2_pts:
            p1_match_wins += 1
            result = f"{p1_name} WINS!"
        elif p2_pts > p1_pts:
            p2_match_wins += 1
            result = f"{p2_name} WINS!"
        else:
            draws += 1
            result = "DRAW"

        print(f"--> MATCH {matches_played} FINAL: {p1_name} {p1_pts} - {p2_pts} {p2_name} | {result} (Series: {p1_name} {p1_match_wins} - {p2_match_wins} {p2_name}, {draws} draws)\n")

        if args.mode in ["gif", "mp4"] and total_frames_recorded >= args.frames:
            break

    game.close()
    if args.mode == "human":
        cv2.destroyAllWindows()

    print(f"==========================================================================")
    print(f"                       FINAL TOURNAMENT RESULTS                           ")
    print(f"==========================================================================")
    print(f"Total Matches Played: {matches_played}")
    print(f"  * {p1_name}: {p1_match_wins} match wins ({p1_match_wins / max(1, matches_played) * 100:.1f}%)")
    print(f"  * {p2_name}: {p2_match_wins} match wins ({p2_match_wins / max(1, matches_played) * 100:.1f}%)")
    if draws:
        print(f"  * Draws: {draws}")

    if p1_match_wins > p2_match_wins:
        print(f"\n🏆 HEAD-TO-HEAD GRAND CHAMPION: {p1_name}!\n")
    elif p2_match_wins > p1_match_wins:
        print(f"\n🏆 HEAD-TO-HEAD GRAND CHAMPION: {p2_name}!\n")
    else:
        print("\n🤝 THE HEAD-TO-HEAD DUEL ENDS IN A PERFECT TIE!\n")

    if args.mode == "gif":
        print(f"Encoding {len(recorded_frames)} frames to animated GIF: {args.output}...")
        import imageio
        gif_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in recorded_frames[::2]]
        imageio.mimsave(args.output, gif_frames, fps=args.fps // 2, loop=0)
        print(f"--> Saved PvP GIF: {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.2f} MB)")
    elif args.mode == "mp4":
        print(f"Encoding {len(recorded_frames)} frames to MP4: {args.output}...")
        h, w, _ = recorded_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for f in recorded_frames:
            writer.write(f)
        writer.release()
        print(f"--> Saved PvP MP4: {args.output}")


if __name__ == "__main__":
    main()
