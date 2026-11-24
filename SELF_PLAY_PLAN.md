# Self-Play vs. Fixed-Opponent DQN — Findings and Concrete Plan

**Date:** 2026-10-26
**Status:** proposal; nothing in Phases 1–5 below is implemented yet. Bug fixes from `CODE_REVIEW.md` are already in the working tree.

---

## 1. Goal

The repository started as a re-implementation of the Nature 2015 DQN paper. That part works: the
single-agent agent scores about **+20 against the Atari CPU** in Pong.

The next question, inspired by AlphaGo/AlphaZero:

> **Does an agent trained by self-play end up stronger and more robust than an agent trained against
> a fixed opponent, with everything else held equal?**

Two things are needed to answer it:
1. **Both training methods as good as we can make them**, with the *opponent* as the only difference.
2. **Evaluation that can actually tell strong agents apart.** The CPU score tops out at +21, so it can't.

---

## 2. Background: what self-play means here (and what it doesn't)

### 2.1 AlphaGo / AlphaZero vs. this project

| | AlphaZero | This project |
|---|---|---|
| Game type | turn-based, perfect information (Go, chess, shogi) | real-time, simultaneous moves, from pixels |
| Learning algorithm | MCTS planning + policy/value network trained on search results | model-free value learning (DQN), no planning |
| Simulator use | perfect rules model, search thousands of positions per move | emulator only steps forward (no search) |
| Opponent | the current network itself (with search) | a pool of frozen past copies of the learner |
| Theory | self-play + search converges toward minimax play | DQN learns a *best response* to whatever opponents fill its replay buffer |

The parts that transfer are **learning from scratch with no human data** and **opponents that get
stronger as the learner gets stronger**. The planning part does not transfer, since there is no
cheap tree search over Pong frames.

The closest published relatives of what we'll do are model-free:
- **Fictitious Self-Play / NFSP** (Heinrich & Silver 2015/2016): a DQN best-responds to the
  *average* of past opponents; NFSP adds a supervised "average policy" network that approximates a Nash equilibrium.
- **OpenAI Five** (2019): 80% of games against the latest self, 20% against past selves.
- **AlphaStar league** (Vinyals et al. 2019): Prioritized Fictitious Self-Play (PFSP), meaning the
  learner meets the past opponents it loses to more often, plus dedicated "exploiter" agents.
- **Bansal et al. 2018**, "Emergent Complexity via Multi-Agent Competition": opponent sampling from
  past snapshots prevents overfitting to the current opponent.

### 2.2 Which Atari games can be learned by self-play?

The intuition is right. Self-play needs **a second player whose policy can be learned**. Breakout,
Space Invaders (1P), Montezuma's Revenge, etc. are a player against the game's physics and scripted
enemies, so there is no opponent to replace with a copy of yourself.

PettingZoo provides 24 multi-player Atari games. **13 of them also exist as single-agent ALE games**
against a built-in CPU or environment (measured from the installed `pettingzoo` and `ale-py`). Those
13 are the ones where "fixed opponent vs. self-play" can be compared on the same cartridge:

| Game | Type | Self-play suitability | Notes |
|---|---|---|---|
| **Pong** | symmetric, zero-sum (+ serve-timer penalty) | ⭐ **Start here** | Mirror symmetry (left↔right) makes sharing one network trivial; already working |
| **Boxing** | symmetric, zero-sum | ⭐ Next | Top-down view; players are told apart by **color**, not side, so it needs a color swap rather than a mirror |
| **Surround** | symmetric, zero-sum (Tron/snake) | ⭐ Good | Simple, short games; strong opponent modelling matters |
| **Tennis** | symmetric, zero-sum | Good but harder | Players **change ends** during the match; single-agent DQN is known to learn to stall |
| **Ice Hockey**, **Double Dunk** | team sports, zero-sum | Harder | Multiple controlled characters, long horizons |
| **Space War**, **Flag Capture** | symmetric competitive | Possible | Less studied |
| **Othello**, **Video Checkers** | turn-based board games | AlphaZero-like in spirit | Through ALE pixels they're slow and awkward; a real board-game environment would suit AlphaZero better |
| **Mario Bros**, **Space Invaders**, **Wizard of Wor** | mixed cooperative/competitive | Research-y | Partner and enemy at once; zero-sum assumptions don't hold |
| Breakout, Montezuma, Seaquest, ... | single-player vs. environment | ❌ No opponent | "Self-play" ideas don't apply (single-player variants like *ranked rewards* exist but are a different topic) |

**Plan:** do the full study on **Pong**. Once the pipeline is proven, extend it to **Boxing** and **Surround** (Phase 7).

---

## 3. Findings: why the two trainers were not comparable

The bug fixes in `CODE_REVIEW.md` are already applied (palette remap removed, schedules fixed,
shared `src/pong2p.py`, etc.). Investigating the comparison turned up the following
**additional differences** between `train.py` (vs. CPU) and `train_selfplay.py` (vs. self).
Items marked **[measured]** were checked in the installed environments.

### F1. Serve timer is invisible to the self-play learner [measured] — affects self-play quality — ✅ fixed
In PettingZoo Pong the player holding the ball **must press FIRE to serve**. If they don't serve
within ~2 s they receive a **−1 reward** (no point is scored), and the timer repeats. With both
players idle for 4,000 frames: **32 serve penalties, zero points scored**. Gym's single-player Pong
serves automatically.

`TwoPlayerPong.step()` counts only *positive* rewards (points), and the self-play reward is
`sign(r_gain − l_gain)`. So:
- the learner never sees the penalty, and nothing except random exploration teaches it to serve;
- a greedy model that doesn't serve stalls until the step cap, and evaluation scores it as a draw.

**Fix:** keep counting *points* from positive rewards (for scoreboards), but train on each
player's **raw reward** (point won +1, point lost −1, stalling −1), clipped per agent step.

### F2. Different game modes [measured]
Gym `ALE/Pong-v5` runs **mode 0** (1 player vs CPU). PettingZoo runs **mode 4** (the first
2-player mode). Single-player modes can't be loaded in 2-player and vice versa.
- Measured equal: horizontal ball speed (1 px/frame), paddle height (16 px), palette.
- Not proven equal: deflection angles, serve angles, the serve mechanic (F1).

**Consequence:** the original "Gym vs CPU" baseline can never be identical to self-play. A
**controlled** fixed-opponent condition must run in the *same* 2-player environment (Section 4).

### F3. Sticky actions differ [measured]
Gym v5 uses `repeat_action_probability = 0.25`: each frame, with probability 0.25, the previous
action is repeated. PettingZoo **hard-codes 0.0** (`base_atari_env.py:105`). Sticky actions make
the game stochastic and policies more robust.

**Fix:** implement sticky actions per player inside `TwoPlayerPong` (default 0.25, seeded), so all
conditions see the same stochasticity.

### F4. Different definitions of "terminal"
`train.py` ends the bootstrapped episode only at game end (Pong has no lives). `train_selfplay.py`
ends it **after every point**. This changes what Q-values mean (value of the rest of the game vs.
value of the current rally) and changes learning dynamics.

**Fix:** one `--point-terminal / --no-point-terminal` flag in both trainers with the **same
default** (proposed: off, matching the proven single-agent run).

### F5. Time-limit handling in the 2-player env — ✅ fixed
PettingZoo truncates after `max_cycles = 100,000` frames, and the self-play loop treats that as
terminal (no bootstrap). Gym truncates at 108,000 frames and `train.py` correctly bootstraps.

**Fix:** `TwoPlayerPong(max_frames=108_000)` returns a separate `truncated` flag, and the trainer
stores `done=False` on truncation.

### F6. Defaults differ
- **Double DQN:** off in `train.py`, on in `train_selfplay.py`. Proposed: on in both.
- **Opponent noise:** the self-play opponent plays with ε = 0.05. The CPU has no ε (but has sticky
  actions). Proposed: one `--opponent-epsilon` used by every opponent type.

### F7. Learner side
The single-agent learner always plays the **right** paddle. The self-play learner plays both
sides, seeing the left side mirrored. Moves are mirror-safe ([measured]: action 2 = up and
3 = down for *both* paddles, so the mirrored left view needs no action remapping). But the
self-play network does learn a slightly different distribution (it serves from both ends).

**Fix:** in every 2-player condition the learner's side is random, 50/50. Evaluations report both sides.

### F8. The evaluation can't separate strong agents
- **Against the CPU, scores top out at +21.** The current champion is already at +20.4; a much
  better player can't score higher.
- `best_model.pt` is chosen from 5 greedy episodes with **different seeds each evaluation**
  (`train.py:333`). That is noisy and biased toward lucky checkpoints.
- The existing round robin used 2 games per pair, and the top two finished level.

**Fix:** the evaluation protocol in Section 5: fixed seeds, many games, ratings with confidence
intervals, and an **exploitability** test.

### F9. Shifting opponents and the replay buffer (a known self-play issue, not a bug)
The replay buffer (100k transitions ≈ the last ~65 matches) mixes games against many past
opponents. A DQN therefore learns a best response to the **recent mixture** of opponents. That is
the fictitious-self-play idea, and is acceptable. We will **log which opponent produced each
match** so effects can be analysed, and keep the buffer size identical across conditions.

### F10. The scripted opponent can be built from RAM [measured]
Pong RAM: ball x = `ram[49]`, ball y = `ram[54]`, right paddle y = `ram[51]`, left paddle y = `ram[50]`.
Action 2 moves a paddle up (y decreases) and action 3 moves it down. A ball-tracking bot similar to
the Atari CPU is therefore straightforward to build (Phase 1.5).

**How the real Atari CPU plays [measured]** (Gym mode 0, 20,000 frames, RAM `ram[50]` = CPU paddle):
- Per-frame paddle movement is only **0 or ±2 px** (plus rare jumps when a point resets the paddle).
  The CPU moves toward the ball's height at a **capped speed of 2 px/frame**. Otherwise it holds
  still (the ball is within ±5 px of the paddle 92% of the time).
- The naive rule `move = 2·sign(ball_y − cpu_y − 1)` matches the CPU's move direction on **91% of frames**.
  The remaining mismatches are likely a reaction delay and/or reading the ball one frame late. To be fitted.
- This explains the game: the CPU is nearly perfect on slow balls and only loses to returns whose
  vertical speed exceeds 2 px/frame. That's exactly what the C0 agent learned to exploit.

**Cloning limits:** in mode 4 the bot doesn't set its paddle position directly as the CPU does. It
presses up/down like a player, and player paddles accelerate (holding a button moves ~2.4 px/frame
on average). So the bot is a **CPU-like policy using the same controls as the learner**. That is
fair to the learner, but it isn't bit-identical to the CPU, so its fidelity is measured (Phase 1.5).

---

## 4. Experimental Design

### 4.0 Rule: every compared method trains and is tested on the same game

A controlled comparison requires **the same emulator, ROM, game mode, rules, preprocessing and
stochasticity** for every method; only the opponent may differ. Both Gym and PettingZoo run the
same Stella emulator and the same Pong ROM, but in **different game modes with different rules**:

| | Gym `Pong-v5` | PettingZoo `pong_v3` |
|---|---|---|
| ALE game mode | 0 (1 player) | 4 (2 players) |
| Left paddle | built-in CPU, **cannot be controlled** | controlled by a second agent |
| Serving | automatic | FIRE required, −1 penalty for stalling |
| Sticky actions | 0.25 | 0.0 (we add 0.25 ourselves, F3) |

Neither environment can host both kinds of training:
- Self-play is impossible in mode 0, because the CPU owns the left paddle.
- The built-in CPU doesn't exist in mode 4, because both paddles are player-controlled.

Therefore:
- **The real experiment (C1–C5) runs entirely in PettingZoo mode 4**, through one code path. The
  "fixed agent" is a scripted CPU-like bot (C1) or a frozen network (C2), playing in the *same* game as self-play.
- **C0 (Gym vs CPU) is kept only as the paper-reproduction reference.** It is never used for the
  head-to-head conclusion, because it was trained on a different mode.
- **Evaluation applies the same rule.** All headline comparisons (E2–E5) are in mode 4. E1 (vs the
  Gym CPU) is a secondary, *cross-game* check that every model takes equally (all of C1–C5 trained
  in mode 4; only C0 is on its home game there, which must be stated when reporting it).

### 4.1 Conditions

All conditions train **from scratch** (no `--base-model`), for the **same number of learner
environment steps**, with **identical hyperparameters** (4.2). Conditions C1–C5 run in the same
2-player environment through the same code path (`train_selfplay.py`), differing only in the
opponent sampler.

| ID | Name | Environment | Opponent during training | Role |
|---|---|---|---|---|
| **C0** | Classic DQN vs CPU | Gym `Pong-v5` (mode 0) | built-in Atari CPU | Paper-style reference (what you already have) |
| **C1** | Fixed scripted opponent | PettingZoo (mode 4) | CPU-like RAM ball-tracker (F10), never changes | **Main "fixed agent" control** |
| **C2** | Fixed learned opponent | PettingZoo | frozen strong DQN (the C0 champion), never changes | Fixed but *strong* opponent |
| **C3** | Naive self-play | PettingZoo | latest learner copy, refreshed every 50k steps | Ablation: expected to cycle or forget |
| **C4** | Fictitious self-play (FSP) | PettingZoo | pool of past snapshots: 80% uniform, 20% latest; gated promotion | **Main self-play condition** (current trainer, fixed) |
| **C5** | Prioritized FSP (PFSP) — ✅ implemented (`--opponent-sampling pfsp`, now the default) | PettingZoo | pool, sampled ∝ (1 − win-rate vs. opponent)² | AlphaStar-style improvement |

**Core set (do first): C0, C1, C4.** Extended set: C2, C3, C5.

### 4.2 Shared hyperparameters (all conditions)

| Setting | Value |
|---|---|
| Learner env steps | 10,000,000 (40M frames) |
| Seeds | 3 per condition (42, 43, 44) |
| Network | Nature CNN, Kaiming init |
| Optimizer | Adam, lr 1e-4, batch 32, grad-norm clip 10, Huber loss |
| Discount | 0.99 |
| Replay | 100,000 transitions, start after 10,000 |
| Train frequency | every 4 env steps |
| Target sync | every 1,000 gradient updates |
| Exploration | ε 1.0 → 0.02 linearly over 1M env steps |
| Double DQN | on |
| Frame skip / max-pool / stack | 4 / last 2 frames / 4 |
| Sticky actions | 0.25 (Gym native; added to `TwoPlayerPong`) |
| Terminal | game end only (`--no-point-terminal`); truncation at 108k frames bootstraps |
| Reward | per-player raw reward, clipped to {−1, 0, +1} per agent step (includes serve penalty in 2P) |
| Learner side (2P) | random 50/50 per match, left view mirrored |
| Opponent ε | 0.05 for every opponent type |
| Checkpoints | every 1M steps (kept) + final; **no "best model" selection** used for the comparison |

### 4.3 What gets compared
- **Main result:** **final** checkpoints (10M) of every seed of every condition, evaluated by every protocol in Section 5.
- **Learning curves:** the 1M-step checkpoints, evaluated against the anchors (E1, E2) and in the league (E3).

---

## 5. Evaluation Protocol

Every evaluation uses **fixed seeds**, a **greedy policy with ε = 0.05** (the Nature protocol;
it avoids deterministic loops) and, as a secondary result, ε = 0.

| ID | Evaluation | How | Metric | Why |
|---|---|---|---|---|
| **E1** | vs Atari CPU | `tournament.py`, Gym `Pong-v5`, 30 games to 21, seeds 1000–1029, right paddle | mean net score ± 95% bootstrap CI, win rate | Paper-comparable; tops out at +21 |
| **E2** | vs scripted bot | 2P env, 30 games to 21, both sides | mean net score ± CI | Same-environment fixed reference |
| **E3** | **League (head-to-head)** | all final models (3 seeds × 6 conditions = 18) + scripted bot; every pair plays 20 games to 21 (10 per side) | **Bradley–Terry / Elo** ratings with bootstrap 95% CI; cross-play win matrix | Directly answers "who is stronger" |
| **E4** | **Exploitability** | for each final model M: freeze it, train a *fresh* exploiter (C1 mode with opponent = M) for 2M steps; evaluate the exploiter vs M every 250k steps | exploiter win rate vs M at 2M steps; area under the curve | Measures how easily a policy is beaten by someone who studies it. **The key test where self-play should win** |
| **E5** | Generalization (cross-seed) | from E3: games of each model against models from *other* conditions/seeds it never trained against | mean score vs. unseen opponents | Checks for overfitting to the training opponent |

Statistics: report per-seed values plus mean ± 95% CI over seeds. For E3, fit Bradley–Terry on all
games and bootstrap over games (1,000 resamples). Differences count as real only if the CIs don't overlap.

### Hypotheses (decided before running)
- **H1 (E1):** all of C0, C1, C4 reach ≥ +18 vs CPU. The CPU score alone won't separate them.
- **H2 (E3):** C4 (and C5) rate higher than C1 and C0.
- **H3 (E4):** fixed-opponent policies (C0, C1, C2) are exploited faster than FSP/PFSP policies.
- **H4:** naive self-play (C3) shows cycling: its rating against fixed anchors goes up and down during training, and it rates below C4.
- **H5 (E1 vs E3):** the strongest model vs CPU is not necessarily the strongest head-to-head (the effect hinted at by the old round robin).

---

## 6. Implementation Plan

Each task lists files and an **acceptance check** (a test or measurement).

### Phase 1 — Make the 2-player environment equivalent (≈ 1 day)

| # | Task | Files | Acceptance |
|---|---|---|---|
| 1.1 | `TwoPlayerPong.step()` also returns each player's **raw clipped reward** (includes serve penalty); points stay positive-only | `src/pong2p.py` | Test: both-idle for 500 steps → server's reward < 0, points 0 |
| 1.2 | Per-player **sticky actions** (`repeat_action_probability=0.25`, own seeded RNG, applied per frame like ALE) | `src/pong2p.py` | Test: with p=1 the first action repeats forever; same seed → identical trajectories |
| 1.3 | **`max_frames=108_000`** truncation, returned separately from game over | `src/pong2p.py` | Test with small cap: `truncated=True, terminated=False` |
| 1.4 | `--point-terminal/--no-point-terminal` in **both** trainers (default off); Double DQN on by default in both; `--opponent-epsilon` | `train.py`, `train_selfplay.py` | `--help` shows flags; config logged |
| 1.5 | **Re-created CPU** `CpuLikePongBot`: reads RAM and emits up/down/FIRE actions. (a) Fit the CPU rule from Gym mode-0 RAM logs: reaction lag k, centre offset, dead zone, speed cap. (b) Choose the action that brings the paddle closest to the CPU's target position given player-paddle acceleration. (c) Serve (FIRE) with the same delay as Gym's automatic serve (measured) | `src/pong2p.py`, `tools/fit_cpu.py` | (1) Rule fidelity on held-out Gym logs: move-direction agreement ≥ 97% (naive rule: 91%). (2) Behavioral match: the C0 champion scores within ±2 vs the bot (mode 4) of its score vs the real CPU (mode 0), over 30 games each. (3) Random policy vs bot ≈ −21 |
| 1.6 | `evaluate_policy` in `train.py`: fixed seeds, `--eval-episodes` default 10, `--eval-epsilon` | `train.py` | Same checkpoint evaluated twice → identical numbers |

### Phase 2 — One trainer, pluggable opponents (≈ 1–2 days)

For two-player games there is **one training script**. The only switch is where the opponent comes
from: a frozen fixed opponent (the re-created CPU or a frozen network), or frozen snapshots of the
learner itself (self-play). In neither case is the opponent trained by gradients; in self-play it
is replaced by newer copies of the learner. `train.py` remains the trainer for single-player games
(Breakout etc.) and the C0 paper reference.

`train_selfplay.py` (to be renamed `train_2p.py`) gets `--opponent-mode {scripted, fixed, latest, fsp, pfsp}`:

```
OpponentSampler (interface)
  ├─ ScriptedOpponent()                 C1
  ├─ FixedOpponent(checkpoint)          C2, and the E4 exploiter
  ├─ LatestOpponent(refresh_every)      C3
  ├─ FSPPool(p_latest=0.2, gated)       C4 (current behavior)
  └─ PFSPPool(weight=(1-p)^2)           C5 (win-rate table updated from training matches + evals)
```

| # | Task | Acceptance |
|---|---|---|
| 2.1 | Sampler classes + CLI; the learner loop is identical for every mode | Unit tests per sampler (distribution of picks over 10k samples) |
| 2.2 | Per-match log `matches.csv`: step, opponent id, learner side, score, length, serve penalties | File written; one row per match |
| 2.3 | **Anchor evaluations during training** (every 250k steps): 5 games vs scripted bot (2P) and 3 games vs Gym CPU. Win rate vs pool is relative and hovers around the threshold, so it can't show absolute progress | `anchors.csv` with both scores |
| 2.4 | Pool: gated promotion (20 matches, ≥ 65%) **plus** a forced snapshot if nothing was promoted for 500k steps; weights-only snapshots; pool capped at 50 (oldest dropped evenly) | Tests |
| 2.5 | Same CSV columns and config logging for every mode, so `analyze_results.py` treats them alike | — |

### Phase 3 — Evaluation suite (≈ 1–2 days)

| # | Task | Files |
|---|---|---|
| 3.1 | `tournament.py`: `--epsilon`, `--matches 30`, bootstrap CI columns | `tournament.py` |
| 3.2 | `league.py` (replaces `round_robin_tournament.py`): `--checkpoints` (globs), `--include-bot`, `--games-per-pair 20`, `--target-score 21`; uses all CPU cores (one process per pairing); outputs game-level CSV, cross-play matrix, Bradley–Terry ratings ± CI, plots | `league.py` |
| 3.3 | `exploitability.py`: for each model, launches `train_selfplay.py --opponent-mode fixed --opponent <M> --total-steps 2M`, then collects exploiter win-rate curves | `exploitability.py` |
| 3.4 | `analyze_results.py`: aggregates everything into the final tables/plots (per condition: mean ± CI over seeds) | `analyze_results.py` |

### Phase 4 — Pilot (≈ ½ day of compute)
One seed each of C0, C1, C4 for **1M steps**. Checks:
- the learner actually serves (serve penalties per match fall toward 0);
- anchor scores rise in every condition;
- the self-play pool grows, and win rate vs pool stays in a sane band (40–75%);
- throughput with 5 runs in parallel (to finalize the schedule in 4.5/Phase 5).

### Phase 5 — Full runs (compute schedule)

Measured throughput: ~600–700 steps/s for `train.py`; ~400–600 steps/s for self-play (short
run, small buffer). Assume about **5–7 h per 10M-step run**. Each run needs ~6 GB RAM for the
replay buffer plus one CPU core for the emulator. The machine has 20 cores, 121 GB RAM and one GB10 GPU, so
**~5 parallel runs** is a reasonable starting point (confirmed in the pilot).

| Wave | Runs | Wall clock (est.) |
|---|---|---|
| 1 | C0, C1, C4 × seed 42, 43 (6 runs) | ~8 h |
| 2 | C0, C1, C4 × seed 44 + C2, C5 × seed 42 (5 runs) | ~7 h |
| 3 | C2, C3, C5 remaining seeds (≈ 7 runs) | ~8 h |
| E3 league | 18 models + bot → 171 pairs × 20 games × ~1,500 steps | ~1–2 h (parallel) |
| E4 exploitability | 18 exploiters × 2M steps | ~8 h (parallel) |
| **Total** | | **~1.5–2 days of machine time**; the core set alone ≈ 15 h |

### Phase 6 — Write-up
README section "Self-play vs fixed opponent". It should contain:
- the method table (4.1/4.2);
- E1–E5 results with CIs and an Elo table;
- exploitability curves;
- learning curves vs anchors;
- what was confirmed or refuted among H1–H5;
- limitations.

### Phase 7 — Extensions (optional, after Pong)
1. **NFSP**: add an average-policy network (supervised on a reservoir buffer of the learner's own
   actions). Evaluate the *average* policy, which in theory approaches a Nash equilibrium.
2. **League with exploiters** (AlphaStar-style): dedicated agents trained only to beat the current main agent, whose snapshots enter the pool.
3. **Boxing / Surround**: generalize `TwoPlayerPong` into a `TwoPlayerAtari` with per-game
   symmetry (mirror vs color swap), palette checks, and a RAM-based or frozen-DQN fixed opponent.
4. Faster data collection (vectorized environments) and a frame-sharing replay buffer, needed before scaling to more games and seeds.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Self-play DQN is unstable or cycles | Pool (C4) and PFSP (C5) instead of naive self-play; anchor evaluations detect regressions; C3 measures how bad the naive version is |
| Gated promotion stalls (pool never grows) | Forced snapshot after 500k steps without promotion |
| Scripted bot too weak or too strong to be a meaningful fixed opponent | Calibrate against the Gym CPU (Phase 1.5); optionally add a difficulty knob (reaction lag / noise) |
| C0 vs C1–C5 still differ in game mode and serve mechanic (F1, F2) | That's why C1 exists; C0 is reported as the reference, not as the controlled comparison |
| Seed variance swamps differences | 3 seeds minimum; report CIs; add seeds for any close comparison |
| Compute | Core set first (≈ 15 h); extended set only if the core results are interesting |

---

## 8. Decisions Needed Before Implementing

1. **Terminal-on-point:** proposed **off** for all conditions (matches the proven single-agent run). Alternative: on for all.
2. **Double DQN on** in all conditions (`train.py` default changes; the old 10M run stays as a "legacy" reference).
3. **Sticky actions 0.25** in all conditions (alternative: 0.0 everywhere, i.e. retrain C0 with `repeat_action_probability=0`).
4. **Scope:** core set (C0, C1, C4 × 3 seeds) first, or everything (C0–C5).
5. Keep the existing artifacts (`checkpoints/`, `checkpoints_selfplay/`) as legacy, and write new runs to `runs/<condition>/seed_<n>/`.

---

## References

- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016.
- Machado et al., *Revisiting the Arcade Learning Environment* (sticky actions, evaluation protocol), JAIR 2018.
- Heinrich, Lanctot & Silver, *Fictitious Self-Play in Extensive-Form Games*, ICML 2015.
- Heinrich & Silver, *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games* (NFSP), 2016.
- Silver et al., *Mastering the game of Go without human knowledge* (AlphaGo Zero), Nature 2017.
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* (AlphaZero), Science 2018.
- Bansal et al., *Emergent Complexity via Multi-Agent Competition*, ICLR 2018.
- Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning* (AlphaStar, PFSP, league), Nature 2019.
- OpenAI et al., *Dota 2 with Large Scale Deep Reinforcement Learning* (OpenAI Five), 2019.
- Terry et al., *PettingZoo: Gym for Multi-Agent Reinforcement Learning*, NeurIPS 2021.
