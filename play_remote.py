"""Interactive Playable Atari Pong: Play against the trained DQN Agent with Remote Control.

Supports two interactive modes:
  1. Desktop X11 Window Mode (--mode x11):
     Play directly on your desktop via OpenCV window over SSH X11 forwarding.
     Controls:
       [W] or [UP Arrow]:    Move Paddle UP
       [S] or [DOWN Arrow]:  Move Paddle DOWN
       [SPACE]:              Serve / Fire
       [Q] or [ESC]:         Quit

  2. Web Browser Remote Control Mode (--mode web):
     Opens a local web server (http://localhost:8080).
     Play in any web browser with live 60 FPS video stream and interactive remote control buttons / keyboard!

Usage:
    # Desktop X11 window:
    python play_remote.py --mode x11

    # Web browser remote control:
    python play_remote.py --mode web --port 8080
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import cv2
import torch
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

from pettingzoo.atari import pong_v3
from src.agent import DQNAgent
from src.utils import get_device


def process_frame(rgb: np.ndarray, flip: bool = False) -> np.ndarray:
    """Preprocess 210x160 RGB into 84x84 grayscale."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    if flip:
        resized = cv2.flip(resized, 1)
    return resized


# Global state for web mode
class GameServerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_frame = None
        self.human_action = 0  # 0: NOOP, 1: FIRE, 2: UP, 3: DOWN
        self.last_action_time = 0
        self.human_pts = 0
        self.ai_pts = 0
        self.matches_won_human = 0
        self.matches_won_ai = 0
        self.ai_model_name = "champion_model"
        self.running = True
        self.paused = False

state = GameServerState()


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Atari Pong - Human vs AI Remote Arena</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            background-color: #0f111a;
            color: #e0e6ed;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            margin: 0;
            padding: 15px;
        }
        h1 { color: #00e676; margin-bottom: 5px; font-size: 24px; text-transform: uppercase; letter-spacing: 2px; }
        .subtitle { color: #8892b0; margin-top: 0; font-size: 14px; }
        .arena-container {
            display: inline-block;
            background: #1a1e2e;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            border: 1px solid #2d3748;
        }
        #game-screen {
            border: 3px solid #334155;
            border-radius: 8px;
            width: 640px;
            height: 480px;
            image-rendering: pixelated;
            display: block;
            margin: 0 auto;
        }
        .controls-panel {
            margin-top: 15px;
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
        }
        .btn {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            user-select: none;
            transition: all 0.1s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .btn:active, .btn.active {
            background: #1d4ed8;
            transform: scale(0.95);
            box-shadow: 0 2px 4px rgba(0,0,0,0.4);
        }
        .btn-up { background: #059669; }
        .btn-up:active, .btn-up.active { background: #047857; }
        .btn-down { background: #d97706; }
        .btn-down:active, .btn-down.active { background: #b45309; }
        .btn-fire { background: #dc2626; }
        .btn-fire:active, .btn-fire.active { background: #b91c1c; }
        .instructions {
            margin-top: 15px;
            color: #94a3b8;
            font-size: 13px;
            line-height: 1.6;
        }
        .badge {
            background: #334155;
            padding: 2px 6px;
            border-radius: 4px;
            color: #38bdf8;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <h1>Pong AI Remote Arena</h1>
    <p class="subtitle">You (Orange Paddle, Left) vs Trained DQN Master (Green Paddle, Right)</p>

    <div class="arena-container">
        <img id="game-screen" src="/stream.mjpg" alt="Live Pong Screen">

        <div class="controls-panel">
            <button class="btn btn-up" id="btn-up" onmousedown="sendAction(2)" onmouseup="sendAction(0)" ontouchstart="sendAction(2)" ontouchend="sendAction(0)">▲ MOVE UP</button>
            <button class="btn btn-fire" id="btn-fire" onmousedown="sendAction(1)" onmouseup="sendAction(0)" ontouchstart="sendAction(1)" ontouchend="sendAction(0)">● SERVE / FIRE</button>
            <button class="btn btn-down" id="btn-down" onmousedown="sendAction(3)" onmouseup="sendAction(0)" ontouchstart="sendAction(3)" ontouchend="sendAction(0)">▼ MOVE DOWN</button>
        </div>

        <div class="instructions">
            Keyboard Controls: Press <span class="badge">W</span> or <span class="badge">▲ Up</span> to Move Up | <span class="badge">S</span> or <span class="badge">▼ Down</span> to Move Down | <span class="badge">Space</span> to Serve
        </div>
    </div>

    <script>
        function sendAction(act) {
            fetch('/action?act=' + act).catch(e => {});
        }

        // Keyboard listeners
        const activeKeys = {};
        window.addEventListener('keydown', function(e) {
            if (activeKeys[e.code]) return;
            activeKeys[e.code] = true;

            if (e.code === 'KeyW' || e.code === 'ArrowUp') {
                sendAction(2);
                document.getElementById('btn-up').classList.add('active');
            } else if (e.code === 'KeyS' || e.code === 'ArrowDown') {
                sendAction(3);
                document.getElementById('btn-down').classList.add('active');
            } else if (e.code === 'Space') {
                sendAction(1);
                document.getElementById('btn-fire').classList.add('active');
            }
        });

        window.addEventListener('keyup', function(e) {
            activeKeys[e.code] = false;
            if (e.code === 'KeyW' || e.code === 'ArrowUp') {
                document.getElementById('btn-up').classList.remove('active');
                sendAction(0);
            } else if (e.code === 'KeyS' || e.code === 'ArrowDown') {
                document.getElementById('btn-down').classList.remove('active');
                sendAction(0);
            } else if (e.code === 'Space') {
                document.getElementById('btn-fire').classList.remove('active');
                sendAction(0);
            }
        });
    </script>
</body>
</html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(HTML_PAGE)))
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif url.path == "/action":
            qs = parse_qs(url.query)
            act = int(qs.get("act", [0])[0])
            with state.lock:
                state.human_action = act
                state.last_action_time = time.time()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif url.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while state.running:
                    frame = None
                    with state.lock:
                        if state.current_frame is not None:
                            frame = state.current_frame.copy()
                    if frame is not None:
                        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b"\r\n")
                    time.sleep(0.025)  # ~40 FPS stream
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logging to keep terminal clean


def run_web_server(port: int):
    server = HTTPServer(("0.0.0.0", port), StreamHandler)
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Play against trained DQN Pong AI")
    parser.add_argument(
        "--ai-checkpoint",
        type=str,
        default="checkpoints/champion_model.pt" if os.path.exists("checkpoints/champion_model.pt") else "checkpoints/best_model.pt",
        help="Path to AI model weights",
    )
    parser.add_argument("--mode", type=str, choices=["x11", "web"], default="x11", help="Control mode: x11 (desktop window) or web (browser)")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (for web mode)")
    parser.add_argument("--fps", type=int, default=35, help="Playback speed for X11 mode")
    parser.add_argument("--target-score", type=int, default=11, help="Points needed to win a match (default: 11)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    ai_name = os.path.basename(args.ai_checkpoint).replace(".pt", "")
    print(f"\n=======================================================")
    print(f"       ATARI PONG: HUMAN VS AI PLAYABLE ARENA")
    print(f"=======================================================")
    print(f"Human Player: [LEFT PADDLE / ORANGE]")
    print(f"AI Opponent:  [RIGHT PADDLE / GREEN] ({ai_name})")
    print(f"Target Score: First to {args.target_score} points | Device: {device}")
    print(f"Mode:         {args.mode.upper()}")
    if args.mode == "web":
        print(f"Web Arena:    http://localhost:{args.port}")
    print(f"-------------------------------------------------------\n")

    # Load AI
    ai_agent = DQNAgent(num_actions=6, device=device)
    ai_agent.load(args.ai_checkpoint)

    env = pong_v3.parallel_env(render_mode="rgb_array")
    obs, _ = env.reset(seed=42)

    ai_stack = deque([process_frame(obs["first_0"], flip=False)] * 4, maxlen=4)

    if args.mode == "web":
        # Launch web server in daemon thread
        web_thread = threading.Thread(target=run_web_server, args=(args.port,), daemon=True)
        web_thread.start()
        print(f"--> Web server started on http://0.0.0.0:{args.port}")
        print(f"--> Open http://localhost:{args.port} in your browser to play!")
    elif args.mode == "x11":
        cv2.namedWindow("Pong: Human vs AI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pong: Human vs AI", 640, 480)
        print("Starting X11 Window. Click on the game window to control!")
        print("Controls: [W] / [UP] = Move Up | [S] / [DOWN] = Move Down | [SPACE] = Serve | [Q] = Quit\n")

    human_pts = 0
    ai_pts = 0
    matches_human = 0
    matches_ai = 0

    action_label_map = {0: "STILL", 1: "SERVE", 2: "UP", 3: "DOWN"}

    try:
        while state.running:
            # Determine human action
            human_action = 0

            if args.mode == "x11":
                # Poll OpenCV keyboard event
                key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
                if key in [ord("w"), ord("W"), 82]:  # W or UP arrow
                    human_action = 2
                elif key in [ord("s"), ord("S"), 84]:  # S or DOWN arrow
                    human_action = 3
                elif key == 32:  # SPACE
                    human_action = 1
                elif key in [ord("q"), ord("Q"), 27]:  # Q or ESC
                    print("\nExiting Human vs AI arena...")
                    break
            elif args.mode == "web":
                time.sleep(1.0 / args.fps)
                with state.lock:
                    human_action = state.human_action
                    # Auto-reset action if not refreshed recently (key release)
                    if time.time() - state.last_action_time > 0.15:
                        state.human_action = 0

            # Determine AI action (Right paddle / first_0)
            s_ai = torch.as_tensor(np.array(ai_stack), device=device).unsqueeze(0)
            with torch.no_grad():
                ai_q = ai_agent.online_net(s_ai)[0].cpu().numpy()
                ai_action = int(np.argmax(ai_q))

            # Step environment (frame skip 4)
            r_ai_total = 0
            r_human_total = 0
            done = False

            for _ in range(4):
                # first_0 is AI (Right), second_0 is Human (Left)
                obs, rewards, term, trunc, _ = env.step({"first_0": ai_action, "second_0": human_action})
                if rewards["first_0"] > 0:
                    r_ai_total += int(rewards["first_0"])
                if rewards["second_0"] > 0:
                    r_human_total += int(rewards["second_0"])

                if term["first_0"] or trunc["first_0"] or term["second_0"] or trunc["second_0"]:
                    done = True
                    break

            ai_pts += r_ai_total
            human_pts += r_human_total
            ai_stack.append(process_frame(obs["first_0"], flip=False))

            # Render frame
            rgb = env.render()
            frame = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_NEAREST)

            # Draw HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 60), (15, 15, 15), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

            # Left: Human (Orange)
            h_act_str = action_label_map.get(human_action, "STILL")
            cv2.putText(frame, "[YOU] Human Player", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1)
            cv2.putText(frame, f"Pts: {human_pts} | Act: {h_act_str}", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Center: Scoreboard
            cv2.putText(frame, f"{human_pts}  -  {ai_pts}", (275, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Match: You {matches_human} - {matches_ai} AI", (260, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

            # Right: AI (Green)
            cv2.putText(frame, f"[AI] {ai_name[:14]}", (450, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
            cv2.putText(frame, f"Pts: {ai_pts} | Max Q: {ai_q[ai_action]:.2f}", (450, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if args.mode == "x11":
                cv2.imshow("Pong: Human vs AI", frame)
            elif args.mode == "web":
                with state.lock:
                    state.current_frame = frame

            # Check point termination
            if human_pts >= args.target_score or ai_pts >= args.target_score or done:
                if human_pts > ai_pts:
                    matches_human += 1
                    print(f"Match Finished: YOU WON! {human_pts} - {ai_pts} (Series: You {matches_human} - {matches_ai} AI)")
                else:
                    matches_ai += 1
                    print(f"Match Finished: AI WON! {ai_pts} - {human_pts} (Series: You {matches_human} - {matches_ai} AI)")
                obs, _ = env.reset()
                ai_stack = deque([process_frame(obs["first_0"], flip=False)] * 4, maxlen=4)
                human_pts = 0
                ai_pts = 0

    except KeyboardInterrupt:
        print("\nStopping Human vs AI arena...")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        state.running = False
        env.close()
        if args.mode == "x11":
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
