# Deep Q-Networks (DQN) for Atari (Nature 2015)

A high-performance, modular, and readable PyTorch implementation of the landmark Deep Reinforcement Learning algorithm from Google DeepMind:
> **Human-level control through deep reinforcement learning**  
> *Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg & Demis Hassabis.*  
> **Nature 518, 529–533 (2015)** | [Paper](https://www.nature.com/articles/nature14236) | [NIPS 2013 arXiv](https://arxiv.org/abs/1312.5602)

---

## Key Features

- **Nature 2015 CNN Architecture**: 3 convolutional layers ($8\times 8$ stride 4, $4\times 4$ stride 2, $3\times 3$ stride 1) followed by a 512-unit fully connected layer and linear Q-value heads.
- **Gymnasium + ALE Atari Integration**: Clean, modern environment wrappers handling frame skipping ($k=4$), sprite flicker removal (max-pooling over 2 frames), grayscale conversion, downsampling ($84\times 84$), frame stacking ($k=4$), and reward clipping ($\{-1, 0, +1\}$).
- **Episodic Life Loss**: End-of-life treated as terminal for Bellman bootstrapping, encouraging survival without forcing an emulator reset.
- **Compact Experience Replay**: Observations stored as `uint8` arrays (normalized on the GPU), with fast vectorized sampling.
- **Huber Loss (Smooth L1)**: Matches the paper's $[-1, 1]$ TD-error gradient clipping for numerical stability.
- **Frozen Target Network ($\theta^-$)**: Hard-sync every $C$ gradient updates to eliminate moving-target feedback loops.

### Deviations from the Nature 2015 setup

| Setting | Nature 2015 | This repo (defaults) |
| :--- | :--- | :--- |
| Environment | deterministic ALE (no sticky actions) | `ALE/Pong-v5`: sticky actions, `repeat_action_probability=0.25` |
| Optimizer | RMSProp, lr 2.5e-4 | Adam, lr 1e-4, plus gradient-norm clipping at 10 |
| Replay memory / start | 1,000,000 / 50,000 | 100,000 / 10,000 |
| Target update $C$ | 10,000 gradient updates | 1,000 gradient updates (= 4,000 env steps) |
| Epsilon | 1.0 → 0.1 over 1M steps | 1.0 → 0.02 over 1M env steps |
| Evaluation epsilon | 0.05 | 0 (greedy); `evaluate.py --epsilon 0.05` for the paper protocol |
| Weight init | default | Kaiming normal |
| Replay storage | — | full 4-frame stacks for $s$ and $s'$ (~5.6 GB for 100k transitions) |

---

## Architecture & Data Flow

```
Raw Atari Frame (210x160 RGB @ 60Hz)
  │
  ▼ [WarpFrame & MaxAndSkipEnv]
84x84 Grayscale (flicker-removed, 15Hz effective)
  │
  ▼ [FrameStack]
Stacked State: (4, 84, 84) uint8
  │
  ▼
NatureDQN:
  ├── Conv2D(in=4, out=32, k=8x8, s=4) + ReLU   ──> (32, 20, 20)
  ├── Conv2D(in=32, out=64, k=4x4, s=2) + ReLU  ──> (64, 9, 9)
  ├── Conv2D(in=64, out=64, k=3x3, s=1) + ReLU  ──> (64, 7, 7)
  ├── Flatten (3136 units)
  ├── Linear(3136 -> 512) + ReLU
  └── Linear(512 -> num_actions)                ──> Q(s, a) for all actions
```

---

## Directory Structure

```
atari/
├── .venv/                          # Isolated Python virtual environment (CUDA 13 + PyTorch)
├── DQN_ATARI_IMPLEMENTATION_PLAN.md# Comprehensive engineering blueprint
├── README.md                       # Complete documentation & usage guide
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git exclusion rules for large models/buffers
├── pytest.ini
├── src/
│   ├── __init__.py
│   ├── model.py                    # NatureDQN ConvNet with on-device normalization
│   ├── wrappers.py                 # Preprocessing & frame stacking wrappers
│   ├── pong2p.py                   # Shared 2-player Pong (PettingZoo) pipeline for all PvP/self-play scripts
│   ├── replay_buffer.py            # Vectorized uint8 circular replay buffer with atomic serialization
│   ├── agent.py                    # DQNAgent: Double DQN, Huber loss, target sync, atomic saving
│   └── utils.py                    # Seeding, GPU/device detection, metric tracker
├── tests/                          # Unit, CLI and web-server tests (pytest)
├── checkpoints/                    # Pong run: metrics CSVs + plots (checkpoints & buffers are git-ignored)
├── checkpoints_breakout/           # Breakout run metrics
├── checkpoints_selfplay/           # Self-play run metrics
├── train.py                        # Resumable training loop with crash resilience & atomic saves
├── train_selfplay.py               # Fictitious self-play trainer (PettingZoo pong_v3)
├── train_2048.py / play_2048.py    # DQN for the 2048 puzzle (src/games/, src/model_2048.py)
├── evaluate.py                     # Deterministic greedy evaluation script
├── record_demo.py                  # Live X11 GUI visualizer with match HUD overlay
├── plot_metrics.py                 # 4-panel training and evaluation telemetry dashboard
├── tournament.py                   # Automated benchmark tournament vs Atari CPU
├── pvp_battle.py                   # Head-to-Head 2-Player DQN vs DQN Arena (PettingZoo)
├── round_robin_tournament.py       # 10-player Round-Robin League (90 matches, cross-table matrix)
└── play_remote.py                  # Playable Human vs AI Arena with desktop & web remote controls
```

---

## Experimental Results, Telemetry & Empirical Findings

The agent was trained from scratch for **10,000,000 environment interaction steps** (equivalent to **40,000,000 emulator frames** at frame-skip 4) on an **NVIDIA GB10 (Blackwell architecture, sm_121)**.

![Training Telemetry](learning_progress.png)

### 1. Training Progression & Performance Milestones

| Milestone Steps | Greedy Eval Score (Mean ± Std) | Win Rate vs CPU | Conceded Pts / Game | Performance State & Observed Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **0 – 250k** | `-21.00 ± 0.00` | 0.0% | 21.0 | Mostly random play (epsilon ≈ 0.76 at 250k; it reached 0.02 at ~1.01M steps). |
| **500k** | `-13.80 ± 2.40` | 0.0% | 21.0 | First returns: agent begins tracking paddle vertically to intercept slow volleys. |
| **1.5M** | `+15.10 ± 2.91` | **100.0%** | ~5.9 | **Game Victory Achieved**: agent wins every match (e.g., 21–6), though still drops balls during extended rallies. |
| **2.2M** | `+18.00 ± 1.41` | **100.0%** | ~3.0 | **Angle Mastery**: agent learns to strike balls with the edge of the paddle, slicing sharp returns into CPU blind spots. |
| **3.9M** | `+19.20 ± 1.17` | **100.0%** | ~1.8 | High consistency; rallies are ended swiftly with aggressive corner shots. |
| **5.1M – 9.7M** | **`+20.40 ± 0.49`** | **100.0%** | **~0.6** | **All-Time Peak Evaluation**: near-flawless shutouts (21–0 or 21–1 in nearly every match). |
| **10.0M (Final)** | `+19.20 ± 1.17` | **100.0%** | **1.8** | **Complete Saturation**: monotonic convergence vs. Atari CPU, sweeping tournament 5/5 with perfect shutouts. |

---

### 2. Telemetry & Curve Interpretations

The 4-panel dashboard in `learning_progress.png` illustrates the learning dynamics:

1. **Episode Reward Curve**: Shows a steep sigmoid ascent between 500k and 1.5M steps, crossing the zero-threshold at ~1.0M steps and saturating near +18 to +20 by 3.5M–5M steps.
2. **Huber TD Loss**: Spikes during the initial transition from random exploration to early policy learning (200k–500k steps) where Bellman errors are highest, before decaying smoothly and stabilizing around ~0.001–0.002.
3. **Q-Value Growth**: Mean action-values increase steadily from 0 towards ~1.5–2.0, with no sign of runaway overestimation.
4. **Greedy Evaluation vs. Exploration**: Deterministic greedy evaluation ($\epsilon = 0$) consistently outscores rolling training reward by +1 to +2 points because greedy play completely eliminates the 2% random exploratory mistakes ($\epsilon = 0.02$) that cause occasional dropped balls.

---

### 3. Key Observations & Algorithmic Explanations

#### A. Perception vs. Reality: The "Losing" Illusion
In Pong, a match is played to 21 points (a net score of `+14` means winning 21–7). When a human spectator watches early trained models (e.g., 1.5M steps), seeing the agent miss 7 balls during rallies can give the visual illusion that *"it is losing often"*, even though the agent won the match 21 to 7. 
Extended training from 1.5M to 10M steps was specifically required to eliminate those final 5–6 unforced errors per game, driving conceded points from ~6 down to under 2, achieving genuine visual shutouts.

#### B. Single-Agent Specialization vs. Multi-Agent Non-Transitivity
A striking empirical phenomenon emerged when comparing the **CPU Tournament** vs. the **10-Player Round-Robin PvP League**:
- **Against the Atari Stella CPU (Deterministic Heuristic)**: Performance scaled monotonically with training steps. The final `step_10000000.pt` checkpoint was the absolute master, conceding only 1.8 points/game.
- **In Head-to-Head Neural Network PvP**: The intermediate checkpoint **`step_2200000.pt`** won the 90-match Round-Robin League (12 Wins - 6 Losses, `+24` point differential), outperforming the 10M step model. 
- **Hypothesis**: Over 10M steps, the policy may have specialized against the return angles of the Atari Stella CPU, while intermediate checkpoints kept a more varied repertoire. Treat this as tentative: the league had only 2 games per pairing (18 per model), the top two finished level at 12–6, and it ran on an older 2-player pipeline without max-pooling. Rerun `round_robin_tournament.py` (ideally with more games per pairing) before drawing conclusions.

#### C. Zero-Shot 2-Player Control via Symmetric Vision
In PettingZoo Pong, Player 1 controls the right paddle (Green) and Player 2 controls the left paddle (Orange). By applying a horizontal flip (`cv2.flip(frame, 1)`) to Player 2's stacked frames, the exact same Nature DQN policy trained solely on the right paddle was able to control the left paddle with zero weight changes or fine-tuning.

---

### 4. Conclusions

1. **Architectural Validation**: The Nature 2015 3-Conv + 2-FC architecture combined with canonical Atari wrappers reliably solves `ALE/Pong-v5` to superhuman level.
2. **Stability**: Q-values stayed bounded and policy improvement was smooth. (The run configuration was not recorded at the time, so whether `--double-dqn` was enabled can't be confirmed; runs now log their full config to `training.log` and every checkpoint.)
3. **Training Horizons**: While game-level victory is established by ~1.5M steps, attaining true superhuman mastery and sub-2-point conceded averages requires 5M to 10M interaction steps.

---

## Visual Demo & Live Gameplay

### 1. Watch the Trained Agent Live on Screen
Run the live X11 GUI simulation over SSH forwarding (`ssh -Y`):
```bash
.venv/bin/python3 record_demo.py --mode human --fps 35
```
* Automatically loads the champion checkpoint `checkpoints/champion_model.pt` (a symlink to `dqn_ALE_Pong-v5_step_5100000.pt`).
* Displays live in-game scoreboard: `Agent: 21 | CPU: 7 (Diff: +14) | Matches: W/P`.
* Plays full matches continuously until you press `q` or `ESC`.

### 2. Plot Training & Evaluation Metrics
Generate the 4-panel telemetry dashboard:
```bash
.venv/bin/python3 plot_metrics.py
```
Outputs [`learning_progress.png`](learning_progress.png) showing Episode Rewards, Training Loss, Epsilon Annealing, and Greedy Evaluation Curves.

---

## Competitive Tournaments & Head-to-Head PvP

### 1. Tournament Against CPU Opponent
Benchmark trained checkpoints on the same game seeds (a seed fixes the emulator, sticky actions and reset no-ops) to find the best policy:
```bash
.venv/bin/python3 tournament.py --matches 5
```
* **CPU Champion**: `step_10000000.pt` (Mean Score: `+19.2 +/- 1.2`, only conceded 1.8 points per game, 5/5 shutouts).
* **Runner-Up**: `step_9700000.pt` (Mean Score: `+19.0 +/- 1.4`, conceded 2.0 points per game).
* Saves results to `checkpoints/tournament_results.csv` and visual chart to `checkpoints/tournament_leaderboard.png`.

### 2. Head-to-Head PvP Arena (Agent vs. Agent)
Pit two neural networks directly against each other with symmetric vision and side-swapping (`--no-swap-sides` to disable). Drawn matches are reported as draws:
```bash
# Watch the duel live:
.venv/bin/python3 pvp_battle.py \
    --p1 checkpoints/dqn_ALE_Pong-v5_step_2200000.pt \
    --p2 checkpoints/dqn_ALE_Pong-v5_step_10000000.pt \
    --mode human --fps 35
```

### 3. Top 10 Round-Robin League
Run a full league (10 checkpoints: 90 matches) where every checkpoint plays every other one home and away:
```bash
.venv/bin/python3 round_robin_tournament.py --target-score 5
```
Results of the original run (older pipeline, see the caveat in section 3.B):
* **PvP Grand Champion**: `step_2200000.pt` (12 Wins - 6 Losses, `+24` point differential).
* **PvP Runner-Up**: `best_model.pt` (12 Wins - 6 Losses, `+16` point differential).
* **PvP Third Place**: `step_10000000.pt` (10 Wins - 8 Losses, `+14` point differential).
* Generates the $10\times 10$ Head-to-Head Cross Table and visual standings chart at `checkpoints/round_robin_leaderboard.png`.

---

## Playable Human vs. AI Arena (Remote Control)

Test your skills directly against the trained DQN agent! `play_remote.py` lets a human player take the left paddle (Orange) against the AI (Green):

```bash
# Option A: Native Desktop X11 window (Controls: [W]/[S]/[SPACE]):
.venv/bin/python3 play_remote.py --mode x11

# Option B: Web Browser Remote Control (Interactive UI on port 8080, bound to 127.0.0.1):
.venv/bin/python3 play_remote.py --mode web --port 8080
#   from another machine: ssh -L 8080:localhost:8080 <host>, then open http://localhost:8080
#   or expose it on the LAN (no authentication!): --host 0.0.0.0
```
- Real-time match telemetry showing human actions, points, and AI Q-values.
- Supports both keyboard input and virtual touch/click buttons.

---

## Installation

Tested with Python 3.12. `requirements.txt` pins the exact environment of the reported runs, including a PyTorch nightly for CUDA 13 that is not on PyPI (see the comment at the top of the file for how to install it, or replace the torch/CUDA pins with a stable build for your GPU). Create the virtual environment and install dependencies:

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running Unit Tests

Run the test suite using `pytest`:

```bash
.venv/bin/python3 -m pytest tests/
```

---

## Training: Single-Agent vs. Atari CPU (`train.py`)

To train DQN on Pong (or any other Atari environment) against the built-in Atari Stella CPU heuristic:

```bash
.venv/bin/python3 train.py \
    --env ALE/Pong-v5 \
    --total-steps 1500000 \
    --buffer-size 100000 \
    --replay-start-size 10000 \
    --batch-size 32 \
    --lr 0.0001 \
    --eps-decay 1000000 \
    --target-sync-freq 1000 \
    --save-dir checkpoints
```

### Key Hyperparameters:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--env` | `ALE/Pong-v5` | Environment ID in Gymnasium / ALE |
| `--total-steps` | `1,500,000` | Total environment interaction steps |
| `--buffer-size` | `100,000` | Capacity of experience replay memory |
| `--replay-start-size` | `10,000` | Warm-up steps with random actions before training |
| `--batch-size` | `32` | Minibatch size for stochastic gradient descent |
| `--lr` | `1e-4` | Learning rate (Adam optimizer) |
| `--eps-start` / `--eps-end` | `1.0` / `0.02` | Epsilon exploration bounds |
| `--eps-decay` | `1,000,000` | Environment steps to anneal epsilon from start to end |
| `--target-sync-freq` | `1,000` | Gradient updates between target network syncs ($\theta^- \leftarrow \theta$); with `--train-freq 4` that is every 4,000 env steps |
| `--train-freq` | `4` | Agent steps between gradient descent updates |
| `--save-freq` | `50,000` | Step interval for saving periodic checkpoints |
| `--eval-freq` | `25,000` | Step interval for greedy evaluation episodes |
| `--double-dqn` | off | Double DQN targets |
| `--save-buffer` / `--no-save-buffer` | on | Save the replay buffer with checkpoints |

> **Note on older runs:** before this was fixed, epsilon was annealed over *gradient updates*, so the
> `--eps-decay 250000` used for the reported runs actually annealed over ~1.01M env steps. The new
> default of 1,000,000 env steps reproduces that schedule. Old checkpoints load and resume unchanged.

Under `EpisodicLifeEnv` (e.g. Breakout) a life loss ends the bootstrapped episode, but the logged
`avg_reward_100` / `avg_length_100` are per full game (clipped rewards). Evaluation runs full games
and presses FIRE after a lost life (`FireOnLifeLossEnv`) so games like Breakout don't stall.

---

## Self-Play Training: Multi-Agent League (`train_selfplay.py`)

In single-agent training, the agent optimizes against a **fixed, static program** (the Stella CPU heuristic), which can cause the policy to overfit to CPU return patterns. To discover generalized, robust policies that converge toward a minimax Nash equilibrium, **Fictitious Self-Play (FSP)** trains the learner against an evolving pool of its own historical snapshots:

```bash
# Train self-play from scratch:
.venv/bin/python3 train_selfplay.py \
    --total-steps 1000000 \
    --save-dir checkpoints_selfplay

# Resume self-play training:
.venv/bin/python3 train_selfplay.py \
    --resume checkpoints_selfplay/latest.pt \
    --additional-steps 500000

# Warm-start from a single-agent model (weights only; epsilon restarts at min(--eps-start, 0.20)):
.venv/bin/python3 train_selfplay.py \
    --base-model checkpoints/champion_model.pt \
    --total-steps 1000000
```

Promotion: every `--eval-freq` steps the learner plays `--eval-matches` (default 20) matches to
`--eval-target-score` (default 7) against snapshots drawn uniformly from the pool, alternating sides,
with 2% epsilon on both sides and a step cap. Draws count as half a win. If the win rate reaches
`--promotion-threshold` (default 65%), the learner's weights are added to the pool. Only the
learner's own transitions go into the replay buffer.

Opponent pool:
- **Size cap** (`--pool-size`, default 50): only the newest snapshots are played against; adding one
  beyond the cap retires the oldest (its file stays in `pool/`). Without a cap the first 10M-step run
  collected 290 snapshots and, sampling them uniformly, spent most matches against weak early copies.
- **Prioritized Fictitious Self-Play** (`--opponent-sampling pfsp`, default; AlphaStar): the newest
  snapshot is played `--latest-prob` (default 20%) of the time; otherwise a snapshot is chosen with
  weight (1 − p)², where p is the learner's win rate over its last 20 finished matches against it
  (Beta(1,1) prior). Opponents the learner still loses to come up more often. `--opponent-sampling
  uniform` restores classic fictitious self-play. The win-rate table is saved in checkpoints.

The training reward is the 2-player game's own per-player reward, clipped: +1 for a point won,
−1 for a point lost, and −1 when the player holding the ball fails to serve within ~2 seconds
(PettingZoo's serve timer: unlike single-player Pong, serving needs FIRE). Each point ends the
bootstrapped trajectory, while PettingZoo's time-limit cutoff does not.

### Preprocessing: `train.py` vs. `train_selfplay.py`

All 2-player scripts (`train_selfplay.py`, `pvp_battle.py`, `round_robin_tournament.py`,
`play_remote.py`) share one pipeline, `src/pong2p.py`, which reproduces `make_atari_env`'s
preprocessing:

| Dimension | Single-Agent (`train.py`) | 2-player (`src/pong2p.py`) | Same? |
| :--- | :--- | :--- | :--- |
| Network, observation tensor | Nature CNN, $(4, 84, 84)$ `uint8` | same | Yes |
| Frame skip / flicker removal | 4, max over last 2 frames | 4, max over last 2 frames | Yes |
| Color palette | court gray 87, paddles 147/148, ball 236 | identical (PettingZoo renders the same palette, no remap) | Yes |
| Player perspective | paddle on the right | right paddle as-is, left paddle mirrored with `cv2.flip` | Yes |
| Reset stochasticity | 1–30 NOOPs | 1–30 NOOPs | Yes |
| Sticky actions | 0.25 (`Pong-v5` default) | 0.0 (PettingZoo) | **No** |
| Terminal for bootstrapping | only game end (Pong has no lives) | every point (won or lost) | **No** — Q-value scales differ |
| Double DQN | off by default | on by default (`--no-double-dqn` to disable) | Configurable |
| Exploration | 1.0 → 0.02 over 1M env steps | same (0.20 → 0.02 with `--base-model`) | Yes |
| Opponent | Atari CPU | evolving snapshot pool | Independent variable |

> **Palette note:** an earlier version remapped PettingZoo colors to `(109,118,43)` / `(101,111,228)`.
> Gym Pong uses that palette only on the single frame returned by `reset()`, which training never sees
> (it is followed by at least one NOOP), so the remap *introduced* a domain shift. It has been removed.
> Self-play checkpoints and PvP results produced with the remap (including the committed
> `checkpoints_selfplay/` run) should be regenerated.

---

## 2048 (`train_2048.py`, `play_2048.py`)

A single-player puzzle (no opponent, so no self-play) solved with the same `DQNAgent`:

- **Game** (`src/games/game_2048.py`): pure-NumPy 4×4 engine, standard rules (slide, merge once per
  tile, spawn a 2 (90%) or 4 (10%) only after a move that changes the board).
- **Environment** (`src/games/env_2048.py`): observation is a (16, 4, 4) one-hot of log2 tile values;
  reward is log2 of the points gained by the move (0 if no merge, −0.5 for an illegal move);
  `info["legal_mask"]` gives the legal moves.
- **Network** (`src/model_2048.py`): parallel 1×2, 2×1 and 2×2 convolutions → 512 → 256 → 4.
- **Action masking** (`--action-masking`, default on): illegal moves are never chosen, and the replay
  buffer stores each next state's legal mask so the bootstrap target (max, or Double DQN argmax) only
  considers legal moves. Without that, untrained Q-values of never-taken illegal moves leak into
  the targets.

```bash
.venv/bin/python3 train_2048.py --total-steps 500000          # checkpoints_2048/
.venv/bin/python3 train_2048.py --resume checkpoints_2048/latest.pt --additional-steps 500000
.venv/bin/python3 play_2048.py --mode ai --checkpoint checkpoints_2048/best_model.pt
.venv/bin/python3 play_2048.py --mode human                    # W/A/S/D or arrow keys
```

Evaluation uses the same fixed seeds every time, so `best_model.pt` is chosen on comparable scores.

Options that matter most for reaching big tiles (a 2048 tile takes ~1,000 moves):
- `--reward {log,linear}` / `--reward-scale` (default 1 for log, 1/1024 for linear). With
  discounting, log reward makes distant big merges almost worthless (a 2048 merge 500 moves away is
  worth ~3% of one move's reward at γ = 0.99); linear keeps them about one move's worth.
- `--gamma`: the planning horizon is roughly 1/(1 − γ) moves (100 at 0.99, 200 at 0.995).
- `--eps-end`: random moves late in a game can wreck a corner strategy; keep it small.
- `--symmetry-augment`: each sampled transition is replaced by one of its 8 rotations/reflections
  (move and legal mask remapped, derived from the game engine), multiplying rare late-game data.

---

## Metrics Logging & Comparison

At every `--log-freq` steps (e.g., every 1,000 steps), structured metrics are streamed and flushed to disk in `checkpoints/training_metrics.csv`:
- `step`: Global environment step
- `episode`: Total completed episodes
- `avg_reward_100`: Rolling 100-game average of the clipped training reward
- `avg_length_100`: Rolling 100-game average game length (agent steps)
- `loss`: Bellman Huber TD loss
- `mean_q`: Average estimated action-value
- `epsilon`: Current exploration rate
- `fps`: Current throughput
- `elapsed_seconds`: Total runtime

To plot or compare multiple runs:

```bash
.venv/bin/python3 plot_metrics.py \
    --train-log checkpoints/training_metrics.csv \
    --eval-log checkpoints/eval_metrics.csv \
    --output learning_curves.png --no-show
```

---

## Crash Resilience & Atomic Saving

- **Atomic File Writing**: Checkpoints write to `.tmp` first before an atomic rename (`os.replace`), preventing corrupt files if interrupted mid-write.
- **Immediate Disk Sync**: The CSV writer calls `os.fsync()` after every log interval so no telemetry is lost if the machine restarts.
- **Exception Trap**: Unhandled exceptions trigger an emergency checkpoint save before terminating.
- **SIGINT Trap**: Pressing `Ctrl+C` cleanly saves `latest.pt` and `latest_replay.npz` before exiting (both trainers).

---

## Resuming Training

Training is fully resumable:

```bash
# Option A: Train until a specific total step target (e.g. 1.5M steps total)
.venv/bin/python3 train.py \
    --resume checkpoints/latest.pt \
    --total-steps 1500000 \
    --save-dir checkpoints

# Option B: Train for an additional N steps from where the checkpoint left off
.venv/bin/python3 train.py \
    --resume checkpoints/latest.pt \
    --additional-steps 500000 \
    --save-dir checkpoints
```

This restores:
- Network weights (`online_net` and `target_net`)
- Optimizer state (Adam momentum & variance)
- Global step counter, gradient-update counter & episode count
- Epsilon annealing schedule position
- Best evaluation score and early-stopping patience counter
- Python/NumPy/Torch RNG states
- Replay buffer state (from `latest_replay.npz`; a different `--buffer-size` keeps the newest transitions) or warm-populates using the loaded policy

Not restored: the emulator state (a resumed run starts a fresh game) and the ALE's internal RNG,
so a resumed run is close to, but not bit-identical with, an uninterrupted one.

---

## Double DQN Support

To eliminate overestimation bias and accelerate convergence to human/superhuman level:

```bash
.venv/bin/python3 train.py \
    --resume checkpoints/dqn_ALE_Pong-v5_final.pt \
    --double-dqn \
    --total-steps 1500000
```

To evaluate a saved model checkpoint across 10 full episodes (add `--epsilon 0.05` for the Nature evaluation protocol):

```bash
.venv/bin/python3 evaluate.py \
    --checkpoint checkpoints/dqn_ALE_Pong-v5_step_500000.pt \
    --episodes 10
```

To record an evaluation video:

```bash
.venv/bin/python3 evaluate.py \
    --checkpoint checkpoints/dqn_ALE_Pong-v5_final.pt \
    --episodes 3 \
    --record \
    --record-dir recordings
```

---

## References

1. **Mnih, V., Kavukcuoglu, K., Silver, D. et al.** *Human-level control through deep reinforcement learning.* Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
2. **Mnih, V. et al.** *Playing Atari with Deep Reinforcement Learning.* NIPS Deep Learning Workshop (2013). https://arxiv.org/abs/1312.5602
