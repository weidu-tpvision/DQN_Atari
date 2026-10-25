# Detailed Implementation Plan: Deep Q-Networks (DQN) for Atari (Nature 2015)

## 1. Executive Summary & Objective

This project implements the foundational Deep Reinforcement Learning algorithm introduced by Google DeepMind in the landmark Nature 2015 paper:
> **Mnih et al., "Human-level control through deep reinforcement learning", Nature 518, 529–533 (2015).**

The goal is to build a high-fidelity, modular, clean, and computationally efficient PyTorch implementation capable of mastering Atari 2600 games (benchmarked on `ALE/Pong-v5` / `PongNoFrameskip-v4`) directly from raw pixel input.

---

## 2. Theoretical Blueprint & Exact Paper Specifications

### 2.1 The Bellman Optimality Equation with Function Approximation
The agent learns the optimal action-value function $Q^*(s, a)$ parameterised by network weights $\theta$:

$$Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{P}(\cdot|s, a)} \left[ r + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \;\middle|\; s, a \right]$$

### 2.2 Loss Function & Gradient
At iteration $i$, a mini-batch of transitions $(s, a, r, s', d)$ is sampled uniformly from the replay buffer $\mathcal{D}$. The target $y$ is computed using the frozen **Target Network** $\theta^-$:

$$y_i = \begin{cases} 
r & \text{if } d = \text{True (terminal)} \\
r + \gamma \max_{a'} Q(s', a'; \theta^-) & \text{otherwise}
\end{cases}$$

The objective is minimizing the Mean Squared Bellman Error with Huber Loss (to match the paper's $[-1, 1]$ gradient clipping):

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \mathcal{H}_\delta\left( y - Q(s, a; \theta) \right) \right]$$

where $\mathcal{H}_\delta$ is the Huber loss ($\delta = 1.0$):

$$\mathcal{H}_\delta(u) = \begin{cases}
\frac{1}{2} u^2 & \text{for } |u| \le 1 \\
|u| - \frac{1}{2} & \text{otherwise}
\end{cases}$$

---

## 3. Critical Architectural & Engineering Components

### 3.1 Network Architecture (Nature DQN CNN)
The network takes stacked grayscale frames $(4 \times 84 \times 84)$ and maps them to a $|A|$-dimensional vector of $Q$-values in a single forward pass:

```
Input: (Batch, 4, 84, 84) normalized to [0.0, 1.0]
  │
  ▼
Conv2D(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0) + ReLU
  │ Output: (Batch, 32, 20, 20)
  ▼
Conv2D(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0) + ReLU
  │ Output: (Batch, 64, 9, 9)
  ▼
Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0) + ReLU
  │ Output: (Batch, 64, 7, 7)
  ▼
Flatten() -> 64 * 7 * 7 = 3136 units
  │
  ▼
Linear(in_features=3136, out_features=512) + ReLU
  │
  ▼
Linear(in_features=512, out_features=num_actions) (Linear activation)
```

### 3.2 Preprocessing & Atari Wrappers
Standard Atari games run at $210 \times 160$ RGB at 60Hz. The canonical Atari preprocessing pipeline handles:
1. **`NoopResetEnv`**: Executes random $k \sim \text{Uniform}(1, 30)$ NO-OP actions on environment reset to provide non-deterministic starting states.
2. **`FireResetEnv`**: For games like Pong or Breakout requiring a FIRE button press to initiate play.
3. **`MaxAndSkipEnv`**:
   - Frame skipping: repeats the chosen action for 4 consecutive frames.
   - Max-pooling: takes the element-wise maximum across the last 2 frames to eliminate Atari 2600 sprite flickering (30Hz scanline phosphor alternation).
4. **`EpisodicLifeEnv`**:
   - Treats loss of life as a terminal state for the Q-learning bootstrapping update ($d=\text{True}$ in TD target), but does not execute a full environment reset until all lives are lost.
5. **`WarpFrame`**: Converts RGB to luminance grayscale and resizes to $84 \times 84$.
6. **`FrameStack`**: Stacks 4 consecutive frames to form an $(84 \times 84 \times 4)$ observation, capturing movement direction and speed.
7. **`ClipRewardEnv`**: Clips environment rewards to $\text{sign}(r) \in \{-1, 0, +1\}$.

### 3.3 Memory-Efficient Experience Replay Buffer
- **Naive approach**: Storing $(s, a, r, s')$ where $s, s'$ are 4-frame float32 tensors consumes $\sim 28 \text{ GB}$ of RAM for $10^6$ frames, which can trigger out-of-memory crashes.
- **Optimized approach**: Store individual frames as `np.uint8` in a circular ring buffer ($84 \times 84 \times 1$). Transitions share consecutive frames, reducing memory to $\approx 7 \text{ GB}$ for $10^6$ frames or $\approx 700 \text{ MB}$ for $10^5$ frames. Frames are only converted to float32 and normalized on the GPU during mini-batch sampling.

---

## 4. Hyperparameters Reference Table (Nature 2015 Spec)

| Hyperparameter | Nature 2015 Value | Notes / Practical Recommendation |
| :--- | :--- | :--- |
| **Minibatch size** | $32$ | Standard size |
| **Replay memory size** | $1,000,000$ | Scaled to $100,000$ - $300,000$ for rapid testing |
| **Agent history length** | $4$ | 4 stacked frames |
| **Target network update freq ($C$)** | $10,000$ steps | Hard update: $\theta^- \leftarrow \theta$ every $C$ frames |
| **Discount factor ($\gamma$)** | $0.99$ | Standard RL discount |
| **Action repeat / skip ($k$)** | $4$ | Frame skipping |
| **Update frequency** | Every $4$ steps | Train step performed every 4 agent steps |
| **Initial exploration ($\epsilon_{start}$)** | $1.0$ | Random exploratory actions |
| **Final exploration ($\epsilon_{end}$)** | $0.1$ or $0.02$ | Purely greedy / minimal exploration |
| **Exploration decay duration** | $1,000,000$ frames | Linear schedule |
| **Replay start size** | $50,000$ steps | Collect random transitions before training |
| **Optimizer** | RMSprop / Adam | Paper used RMSprop; Adam ($lr=1e-4$) converges faster |
| **Loss** | Huber Loss | Smooth L1 loss ($\delta=1.0$) |

---

## 5. File Structure

```
atari/
├── .venv/                      # Isolated Python 3.12 environment
├── requirements.txt            # Frozen dependency list
├── DQN_ATARI_IMPLEMENTATION_PLAN.md
├── src/
│   ├── __init__.py
│   ├── wrappers.py             # Atari preprocessing & frame stacking wrappers
│   ├── model.py                # Nature DQN ConvNet architecture in PyTorch
│   ├── replay_buffer.py        # Circular uint8 replay buffer
│   ├── agent.py                # DQNAgent (action selection, target net update, optimization)
│   └── utils.py                # Seeding, device detection, logging helpers
├── tests/
│   ├── test_wrappers.py        # Verify observation shapes (4, 84, 84) and reset logic
│   ├── test_model.py           # Verify tensor shapes and forward pass
│   └── test_replay_buffer.py   # Verify sampling correctness and memory bounds
├── train.py                    # Main training loop with metrics logging
└── evaluate.py                 # Evaluation script with video capture and score plotting
```

---

## 6. Execution & Verification Roadmap

### Phase 1: Environment & Tooling Setup
1. Create virtual environment `.venv/`.
2. Install required packages:
   - `torch`, `torchvision` (CUDA-enabled)
   - `gymnasium[atari]`, `gymnasium[accept-rom-license]`, `ale-py`
   - `opencv-python`, `numpy`, `matplotlib`
3. Generate `requirements.txt`.

### Phase 2: Core Components Implementation
1. **`wrappers.py`**: Assemble the full Atari wrapper pipeline compatible with modern Gymnasium and ALE.
2. **`model.py`**: Implement `NatureDQN` PyTorch module with orthogonal or He initialization.
3. **`replay_buffer.py`**: Implement ring buffer with uint8 storage and vectorized sampling.
4. **`agent.py`**: Implement `DQNAgent` managing $\theta$ and $\theta^-$, $\epsilon$-greedy schedule, and Bellman target computations.

### Phase 3: Unit Testing & Integrity Checks
1. Test wrapper pipeline on `ALE/Pong-v5` to verify $(4, 84, 84)$ float tensors in $[0, 1]$.
2. Test model forward pass with dummy batches.
3. Test replay buffer push and sample operations for consistency.

### Phase 4: Training & Convergence Validation
1. Run `train.py` on `ALE/Pong-v5`.
2. Monitor key metrics:
   - Episode Reward (Pong scores from $-21$ to $+21$).
   - Average $Q$-value estimation.
   - TD Loss stability.
   - FPS / step throughput.
3. Target: Demonstrate positive learning curve (e.g., reaching $+18$ to $+21$ on Pong).

### Phase 5: Evaluation & Inspection
1. Run `evaluate.py` to test the greedy policy without exploration noise.
2. Render episode video and plot reward learning curves.
