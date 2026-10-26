# Code & Documentation Review — Atari DQN (Nature 2015) Repository

**Date:** 2026-10-26
**Scope:** all tracked source (`src/`, top-level scripts, `tests/`), `README.md`, `DQN_ATARI_IMPLEMENTATION_PLAN.md`, `requirements.txt`, `.gitignore`, and the committed metric CSVs.
**Method:** read every file in full, then checked the suspected bugs against the real environments in `.venv` (ALE 0.12.1, PettingZoo 1.27, Gymnasium 1.4). Findings marked **[verified]** were reproduced by running code or are backed by the committed CSV logs. The other findings come from reading the code.

---

## 0. Executive Summary

The core DQN (network, Huber/TD target, target network, replay sampling, truncation-vs-termination handling in `train.py`) is correct and readable, and the Pong result is real. Most of the problems are in the **2-player / PettingZoo tooling**, in **scheduling semantics**, and in the **README making claims the code doesn't support**.

### Top issues (fix first)

| # | Severity | Issue | Where |
|---|---|---|---|
| 1 | 🔴 Critical | **The palette remap creates the domain shift it claims to remove.** Gym `Pong-v5` and PettingZoo `pong_v3` use *identical* colors. The "single-agent" colors the remap targets appear in Gym only on the first frame after `reset()`. **[verified]** | `train_selfplay.py:41-50`, `pvp_battle.py:36-45`, `README.md:292, 306-312` |
| 2 | 🔴 Critical | **Web play mode is broken.** `HTTPServer` is single-threaded, so once the MJPEG stream is open no `/action` request is ever served. **[verified]** | `play_remote.py:257-259` |
| 3 | 🔴 High | **Epsilon decay and target sync count gradient *updates*, not env steps.** `--eps-decay 250000` really means ~1.01M env steps, and `--target-sync-freq 1000` means 4,000 env steps. The README/help text says "steps". **[verified via CSV]** | `src/agent.py:58-61, 131-133`, `train.py:39-40`, `README.md:88, 258-259, 299-300` |
| 4 | 🔴 High | **Self-play resume corrupts the schedule:** `learner.total_steps = global_step` puts env steps into an update counter, so epsilon jumps by about 4×. | `train_selfplay.py:310` |
| 5 | 🔴 High | **Breakout greedy evaluation stalls until the 108k-frame time limit**, because FIRE is never pressed after a life is lost when `episodic_life=False`. Eval episode length ≈ 27,000 agent steps = 108,000 frames. **[verified via CSV]** | `src/wrappers.py:190-193`, `train.py:59`, `evaluate.py:34`, `checkpoints_breakout/eval_metrics.csv` |
| 6 | 🟠 High | **The five PettingZoo scripts preprocess differently** (remap vs. no remap, max-pool vs. no max-pool). The PvP/round-robin/human-play results are not comparable with each other. | `round_robin_tournament.py:34-41,131`, `play_remote.py:43-49,371`, `pvp_battle.py`, `train_selfplay.py` |
| 7 | 🟠 High | **R/B channels swapped in every on-screen and video output:** RGB frames go to OpenCV BGR APIs. | `pvp_battle.py:211-234,287`, `record_demo.py:101-120,155`, `play_remote.py:374-396,240` |
| 8 | 🟠 Medium | **`--save-buffer`, `--double-dqn` (self-play), `--swap-sides`, `--plot` can't be turned off** (`store_true` with `default=True`). **[verified]** | `train.py:50`, `train_selfplay.py:212-213`, `pvp_battle.py:79`, `tournament.py:69` |
| 9 | 🟠 Medium | **`round_robin_tournament.py --help` crashes** (unescaped `%` in the help string). **[verified]** | `round_robin_tournament.py:84` |
| 10 | 🟠 Medium | **`record_demo.py --mode gif/mp4` without `--frames` records 0 frames**, and mp4 then crashes with `IndexError`. | `record_demo.py:83` |

---

## Fix Status (2026-10-26)

Fixed in the working tree, with tests (`pytest`: 38 passed) and end-to-end smoke runs of every script:

| Finding | Status | How |
|---|---|---|
| 1.1 palette remap | ✅ Fixed | Remap deleted; new `src/pong2p.py` reproduces `make_atari_env` preprocessing; `tests/test_pong2p.py` checks gray levels against Gym |
| 1.2 web mode | ✅ Fixed | `ThreadingHTTPServer`, byte-accurate `Content-Length`, action validation (400 on bad input), `--host` (default `127.0.0.1`), JS re-sends held actions every 100 ms; `tests/test_play_remote.py` |
| 1.3 epsilon/target units | ✅ Fixed | Epsilon anneals over `agent.env_steps` (set by the loop); target sync stays per gradient update (as in Nature) and is documented as such. `--eps-decay` default 1,000,000 env steps reproduces the old effective schedule. Old checkpoints load (legacy keys mapped) |
| 1.4 self-play resume / base model | ✅ Fixed | Counters come from the checkpoint; `--base-model` loads weights only (fresh optimizer/counters, ε starts at 0.20) |
| 1.5 Breakout eval stall + per-life logging | ✅ Fixed | `FireOnLifeLossEnv` in non-episodic FIRE games (Breakout eval: 702 steps/episode, was ~27,000); training logs per-game rewards |
| 1.6 divergent 2-player copies | ✅ Fixed | All four 2-player scripts use `TwoPlayerPong` |
| 1.7 RGB/BGR | ✅ Fixed | Convert after `render()` in `pvp_battle.py`, `record_demo.py`, `play_remote.py` |
| 2.1 un-disableable flags | ✅ Fixed | `BooleanOptionalAction` |
| 2.2 round-robin `--help` | ✅ Fixed | `%%`; `tests/test_cli.py` runs `--help` for every script |
| 2.3 record_demo frames | ✅ Fixed | 1500 default for gif/mp4; action names from the env |
| 2.4 ties | ✅ Fixed | Draws counted in PvP, round robin (W-D-L, half-win), human play, self-play eval |
| 2.5 self-play docstring | ✅ Fixed | Docstring now says only learner transitions are stored (adding opponent transitions remains an option) |
| 2.6 self-play misc | ✅ Fixed | Ctrl+C saves `latest.pt` and exits (no "final"); `--eval-matches` (default 20) with ε=0.02 and a step cap; session FPS; weights-only pool snapshots |
| 2.7 sticky actions | 📝 Documented | Difference kept and listed in README tables |
| 2.8 reproducibility | ✅ Fixed (mostly) | `NoopResetEnv` and `TwoPlayerPong` reseed their own RNG on `reset(seed)`; PvP/round robin seed NumPy; README states what resume can't restore |
| 2.9 train.py misc | ✅ Fixed | Config saved in log and checkpoints, patience restored, SIGINT off-by-one, dead code/unused imports removed |
| 2.10 checkpoint bloat | ✅ Fixed | `MetricTracker` keeps only the window (`deque`) |
| 2.11 replay buffer | ✅ Partly | `load()` validates shape and handles capacity changes. Frame-sharing storage and buffer retention **not done** |
| 2.12 tournament | ✅ Fixed | Plot path follows `--output-csv`, y-limits fit the data, contender counts dynamic |
| §3 low items | ✅ Mostly | Docstrings/comments fixed; `weights_only` artifacts via `save_weights()`. Not changed: init scheme, `x.max()` branch, logging framework |
| §4 requirements | ✅ Partly | Removed `torchvision`, `pygame`, `pygame-ce`; documented the torch nightly. Not split into direct/lock files |
| §6 docs | ✅ Fixed | README sections rewritten (deviations table, consistency table, palette note, flags, links, tree); plan doc marked historical |

**Still to do (needs compute or a decision):**
- Retrain self-play: the committed `checkpoints_selfplay/` run used the bad remap.
- Rerun `round_robin_tournament.py`/`pvp_battle.py` for the README's PvP numbers (they came from the old pipeline).
- Frame-sharing replay buffer, vectorized envs, CI setup.

---

## 1. Critical / High-Severity Bugs

### 1.1 🔴 `remap_pz_to_gym` maps colors in the wrong direction [verified]

`train_selfplay.py:41-50` and `pvp_battle.py:36-45` assume PettingZoo uses a different palette from Gym and remap
`(144,72,17)→(109,118,43)` (background) and both paddles → `(101,111,228)`.

Measured over 6,000+ frames of each env:

| Env | Background | Right paddle | Left paddle | Ball/walls |
|---|---|---|---|---|
| Gym `ALE/Pong-v5` | `(144,72,17)` gray 87 | `(92,186,92)` gray 147 | `(213,130,74)` gray 148 | `(236,236,236)` |
| PettingZoo `pong_v3` | `(144,72,17)` gray 87 | `(92,186,92)` gray 147 | `(213,130,74)` gray 148 | `(236,236,236)` |

The palette `(109,118,43)` / `(101,111,228)` *does* exist in Gym Pong, but only on **one frame per episode**: the frame returned by `reset()` (background changed at steps 0, 3831, 6887 … i.e. right after each reset, then immediately back to 144). `NoopResetEnv` always takes ≥1 NOOP, so the training pipeline **never sees that palette**. The remap was probably derived from looking at a raw `reset()` observation.

**Consequences**
- Every PvP game in `pvp_battle.py` feeds the Gym-trained models frames they never saw in training (court gray 107 instead of 87; paddles 121 instead of 147/148).
- `train_selfplay.py` trains on the off-distribution palette. A self-play model is therefore *not* compatible with `train.py` models or with `tournament.py`/`evaluate.py`, and `--base-model` warm-starting starts from a shifted input distribution.
- `README.md:292` ("Color Normalization … Identical") and the whole section "The Color Palette Domain Shift & Resolution" (`README.md:306-312`) are factually wrong.
- Ironically, `round_robin_tournament.py` and `play_remote.py`, which *don't* remap, are the correct ones here.

**Fix:** delete `remap_pz_to_gym` (or reduce it to a no-op). Add a unit test asserting that the processed PettingZoo frame equals the processed Gym frame for the same emulator state (or at least that the gray-level sets match). Retrain/re-evaluate self-play, rerun the PvP results, and rewrite the README section.

### 1.2 🔴 Web mode: controls never reach the game [verified]

`play_remote.py:257-259` uses `http.server.HTTPServer`, which handles **one request at a time**. The `/stream.mjpg` handler (`:229-249`) loops forever, so after the page opens the stream, every `fetch('/action?...')` waits in the socket backlog. Reproduced: with the stream connected, `GET /action?act=2` timed out after 3 s.

**Fix:** `from http.server import ThreadingHTTPServer` and use it in `run_web_server`.

Related bugs in the same mode:
- **Holding a key/button only moves the paddle for 0.15 s.** The JS sends one `keydown` (auto-repeat is suppressed by `if (activeKeys[e.code]) return;`, `:176`), and the server resets the action to NOOP 0.15 s after the last request (`:343`). Either keep-alive the action from JS with `setInterval` while the key is held, or drop the timeout and rely on the explicit `keyup → sendAction(0)`.
- **Truncated HTML** [verified]: `Content-Length` is `len(HTML_PAGE)` in *characters* (5213), but the UTF-8 body is 5223 bytes because of `▲ ● ▼`, so the browser drops the last 10 bytes. Encode first, then use `len(body)`.
- **Unvalidated input crashes the game:** `int(qs.get("act")[0])` raises on non-integers, and any integer outside `0..5` is passed straight to `env.step`. That raises in the main loop and ends the session (`:221`, `:359`). Clamp/validate.
- **Binds to `0.0.0.0` with no auth** (`:258`), so anyone on the LAN can control/kill the game. Default to `127.0.0.1` and add a `--host` flag.
- `GameServerState` fields `human_pts`, `ai_pts`, `matches_won_*`, `ai_model_name`, `paused` are never used.

### 1.3 🔴 Epsilon / target-sync units are "updates", not "steps" [verified]

`DQNAgent.total_steps` is incremented only inside `update()` (`src/agent.py:131`), and both `get_epsilon()` and target sync use it. With `train_freq=4` and `replay_start_size=10k`:

- epsilon reaches `eps_end` at env step `10,000 + 4 × 250,000 = 1,010,000`, not 250,000;
- the target net syncs every 4,000 env steps, not 1,000.

Evidence: `checkpoints_breakout/training_metrics.csv` at step 500,000 shows `epsilon=0.5198` = `1 − 0.98 × (490,000/4)/250,000` exactly. The self-play log ends with `epsilon=0.0298` at 1M steps, so the self-play learner was more than 50% random for the first half of its run.

Impacts:
- `--help` text ("Steps to decay epsilon"), `README.md:88` ("0–250k: epsilon decaying from 1.0 down to 0.02"), `README.md:258-259`, and the consistency table rows at `:299-300` are all wrong.
- The Nature 2015 schedule is defined in frames/agent steps, so comparisons with the paper are off by 4×.

**Fix (choose one, then document it):**
- Rename the counter to `num_updates`, and pass `env_step` explicitly to `get_epsilon(env_step)`/`select_action(..., env_step)`. Measure `target_sync_freq` in env steps (or rename the flag `--target-sync-updates`).
- Or keep the behavior and change all help strings and docs to say "gradient updates".

### 1.4 🔴 Self-play resume and `--base-model` break the schedule

- **Resume** (`train_selfplay.py:310`): `learner.total_steps = global_step`. The checkpoint already stored the correct update count in `"total_steps"` (`:357`), and `learner.load()` already restored it. This line overwrites it with an env-step count about 4× larger, so epsilon instantly drops to `eps_end` and the target-sync phase shifts. **Delete the line.**
- **`--base-model`** (`:238, 252-255`): `learner.load()` restores the base model's `total_steps` (≈2.5M updates for the 10M-step model), its optimizer state, and its target net. So:
  - the `min(eps_start, 0.20)` cap is moot, because epsilon is `eps_end` from step 1;
  - the help text "(or 0.15 if base-model)" (`:202`) disagrees with the code (0.20).
  For a warm start you probably want to copy only `online_net` weights (and sync the target), reset `total_steps` to 0, and create a fresh optimizer.
- `train.py:131` has the same redundant pattern (`agent.total_steps = checkpoint_data.get("total_steps", start_step)`). It is harmless today because the key exists, but the fallback `start_step` uses the wrong unit.

### 1.5 🔴 Breakout: greedy evaluation never re-serves after losing a life [verified]

`FireResetEnv` presses FIRE only inside `reset()`. During training `EpisodicLifeEnv` turns every life loss into a `reset()`, so FIRE is pressed. In evaluation (`episodic_life=False`) nobody presses FIRE after a life is lost. A deterministic greedy policy that never chooses FIRE then idles until the 108,000-frame limit.

Evidence: `checkpoints_breakout/eval_metrics.csv` `mean_length` = 21,657–26,995 agent steps (≈ 87k–108k frames), against a training episode length of ~73 steps.

**Fix options:**
- a `FireOnLifeLossEnv` wrapper that steps FIRE whenever `ale.lives()` decreases (used in eval only);
- or evaluate with ε = 0.05 as in the Nature paper (`evaluate_policy` and `evaluate.py` use ε = 0);
- also add a per-episode step cap to eval loops.

Related: in `train.py:290-295` an "episode" ends on every life loss, so for Breakout `avg_reward_100`/`avg_length_100` are **per-life clipped** rewards, not game scores. Track true game score separately, e.g. accumulate until `env.was_real_done`, or use Gymnasium's `RecordEpisodeStatistics` placed *inside* `EpisodicLifeEnv`.

### 1.6 🟠 Five divergent copies of the PettingZoo game loop

| Script | Color remap | 2-frame max-pool | Early-termination buffer fix | NOOP reset | ε |
|---|---|---|---|---|---|
| `train_selfplay.py` (train loop) | yes ✗ | yes | yes | yes | learner ε, opp 5% |
| `train_selfplay.py::evaluate_learner_vs_pool` | yes ✗ | yes | **no** | yes | **0 / 0** |
| `pvp_battle.py` | yes ✗ | yes | yes | **no** | 2% |
| `round_robin_tournament.py` | no | **no** | n/a | **no** | 2% |
| `play_remote.py` | no | **no** | n/a | n/a | AI 0 |

So the same checkpoint pair can give different results depending on the script, and the README's round-robin conclusions (`README.md:115-119`) came from a different input pipeline than `pvp_battle.py`.

**Fix:** move `process_frame`, the 4-frame skip + max-pool step, the NOOP reset, and the frame stacks into one module (e.g. `src/pong2p.py` with a `TwoPlayerPong` class exposing `reset() -> (obs_right, obs_left)` and `step(a_r, a_l) -> (obs_r, obs_l, r_gain, l_gain, done)`). Every script should use it.

Additional problems in these copies:
- `evaluate_learner_vs_pool` runs both agents fully greedy with no step cap, so a deterministic volley loop runs until PettingZoo's `max_cycles=100000` (`train_selfplay.py:138-172`). Add ε or a cap, like `round_robin`'s `max_steps`.
- `round_robin_tournament.py:102-132` never checks `term/trunc` in the outer loop, so after the env ends it keeps calling `step()`. It is only safe because `max_steps*4 < max_cycles`.

### 1.7 🟠 RGB vs BGR in all visual outputs

`env.render()` returns **RGB**, but `cv2.imshow`, `cv2.VideoWriter` and `cv2.imencode` expect **BGR**. The HUD colors are written in BGR (e.g. `(0,165,255)` = orange), so the HUD looks right but the game image has red and blue swapped (brown court shows as blue, orange paddle shows as blue). The GIF path then calls `cv2.cvtColor(f, COLOR_BGR2RGB)`, which swaps the game pixels back but turns the HUD text the wrong color.

Affected: `pvp_battle.py:211-234, 287, 293-296`, `record_demo.py:101-120, 155, 160-164`, `play_remote.py:374-396`, `:240`.
**Fix:** `frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)` right after `render()`, and keep the existing BGR→RGB conversion for imageio.

---

## 2. Medium-Severity Bugs

### 2.1 Boolean flags that can't be disabled [verified]
`action="store_true", default=True` always yields `True`:
- `train.py:50` `--save-buffer`
- `train_selfplay.py:212` `--save-buffer`, `:213` `--double-dqn` (self-play is always Double DQN, but `train.py` defaults to vanilla, which contradicts the "Identical" row at `README.md:297`)
- `pvp_battle.py:79` `--swap-sides`
- `tournament.py:69` `--plot`
- `plot_metrics.py:23` works only because of the separate `--no-show`.

**Fix:** `action=argparse.BooleanOptionalAction, default=True` (gives `--x/--no-x`).

### 2.2 `round_robin_tournament.py --help` crashes [verified]
`:84` help text `"(2%) to prevent…"` gives `ValueError: unsupported format character ')'`. Use `%%`, as `pvp_battle.py:75` does.

### 2.3 `record_demo.py` default frame budget is 0
`:83` `max_steps = args.frames if (args.frames > 0 or args.mode != "human") else …`. For `--mode gif` with the default `--frames 0` this gives `max_steps = 0`, so nothing is recorded (an empty GIF, or `IndexError` at `recorded_frames[0]` for mp4). The help text promises 1500. Suggested:
```python
max_steps = args.frames if args.frames > 0 else (10**9 if args.mode == "human" else 1500)
```
Also: `action_names` (`:81`) is Pong-specific and will raise `IndexError` for envs with more than 6 actions; use `env.unwrapped.get_action_meanings()`. In addition, `diff_str` printed at match end (`:142`) is from *before* the final point.

### 2.4 Ties are awarded to one side
- `pvp_battle.py:254-259`: `p1_pts == p2_pts` → P2 wins (possible when a GIF/MP4 frame cap or env truncation ends a match early).
- `round_robin_tournament.py:195-204, 229-238`: tie at `max_steps` → the *left* player wins.
- `play_remote.py:403-408`: tie → AI wins.

Count draws explicitly.

### 2.5 Self-play docstring vs. behavior
`train_selfplay.py:7` says "Both perspectives generate valid zero-sum transitions for the shared replay buffer (2x data efficiency)". The code pushes **only the learner's transition** (`:464-467`). Since Q-learning is off-policy, pushing the opponent-perspective transition too is valid and would double the data. Implement it or fix the docstring.

### 2.6 Self-play: other issues
- **Dead interrupt branch** (`:405-410`): the `while` condition already includes `not interrupted`, so the inner `if interrupted:` never runs. Ctrl+C falls through to the "final" save and writes `selfplay_final.pt`, which mislabels an interrupted run as finished.
- **Promotion is noisy.** 6 matches to 7 points means win-rate granularity is 16.7%, and the 65% threshold needs 4/6. With p = 0.5, P(≥4/6) ≈ 34%, so about one in three evals promotes by chance. This matches the log: 20 promotions in 40 evals, often alternating between 33% and 67%. Use ≥20 matches, a confidence bound (e.g. Wilson lower bound > 55%), or a Bernoulli SPRT.
- **FPS after resume is inflated** (`:481, 520`): `global_step / elapsed` includes steps from before the resume. Track `steps_this_session`.
- **Snapshots store full agents** (online + target + Adam state ≈ 27 MB each), and `SnapshotPool.loaded_agents` caches every snapshot on the device without eviction. Save `online_net.state_dict()` only, and load a bare `NatureDQN` for opponents.
- `num_actions=6` is hard-coded in `SnapshotPool.sample_opponent` (`:94`) and the learner (`:240`).
- `if global_step >= target_total_steps: return` (`:347-349`) leaks the open log/CSV files. Use `with` blocks or `contextlib.ExitStack`.
- **Point-terminal semantics differ from `train.py`.** Self-play ends the bootstrapped trajectory on every point (`:463`). In `train.py` Pong points are *not* terminal: `EpisodicLifeEnv` is a no-op because Pong has no lives (`ale.lives() == 0`). So `README.md:301` ("Point loss ends transition (EpisodicLifeEnv) … Identical") is wrong, and Q-values from the two trainers have different scales. Self-play also marks a point *won* as terminal, which is a reasonable choice but should be documented.

### 2.7 Sticky actions differ between Gym and PettingZoo [verified]
`gym.make("ALE/Pong-v5")` defaults to `repeat_action_probability=0.25` (sticky actions, v5 protocol, Machado et al. 2018). PettingZoo `pong_v3` reports `0.0`. So:
- `train.py` is **not** "exact Nature 2015", which used `NoFrameskip-v4`-style deterministic emulation (no sticky actions);
- single-agent and self-play/PvP envs differ in dynamics (another row the README's "Identical" table leaves out).

Make it an explicit `make_atari_env(..., repeat_action_probability=...)` argument and document it.

### 2.8 Reproducibility claims
- `NoopResetEnv` uses the **global** `np.random` (`src/wrappers.py:33`), not `self.np_random`. `env.reset(seed=…)` therefore doesn't control the number of no-ops, and `tournament.py` never calls `set_seed`. The claim "identical set of randomized game seeds" (`tournament.py:3`, `README.md:157`) is false: each model gets different no-op counts depending on RNG consumption. Use `self.unwrapped.np_random.integers(1, noop_max + 1)`.
- `pvp_battle.py` and `round_robin_tournament.py` never seed `np.random`, which drives their ε-actions (`--base-seed` only seeds env resets).
- `train.py` resume restores Python/NumPy/Torch RNG but **not** ALE state (sticky-action RNG, emulator), and re-seeds the env with the same `args.seed` on every resume (`:226`). A resumed run is therefore not equivalent to an uninterrupted one, contrary to "exact pseudo-random sequence continuity" (`:137`).

### 2.9 `train.py` smaller issues
- `episode_reward`/`episode_length` are restored from the checkpoint (`:129-130`) and then immediately overwritten with 0 (`:227-228`). The env is also reset, so the half-finished episode is lost anyway. Remove the dead restore.
- `if args.resume and "checkpoint_data" in locals()` (`:222`) is fragile. Initialize `checkpoint_data = {}` up front.
- `patience_counter` is not saved, so early-stopping patience resets on resume.
- On SIGINT the checkpoint is saved at the *top* of iteration `step` before the step runs, but resume starts at `global_step + 1`. Off-by-one: one step is skipped.
- FPS (`:306`) includes evaluation time, so FPS dips after every eval.
- `best_model.pt` is chosen from a 5-episode greedy eval with varying seeds (`seed=args.seed + step`, `:333`). That is noisy and not comparable across evals; use a fixed eval seed set.
- The run configuration (`vars(args)`) is not saved anywhere. The "Double DQN enabled" line is printed only to stdout and not to `training.log`, so it's **impossible to tell from the artifacts whether the 10M-step run used Double DQN**, yet the README credits Double DQN for the stability (`README.md:104, 129`). Save `args` in every checkpoint and write a `config.json` in `save_dir`.
- Unused imports: `NatureDQN` (`train.py:21`), `gym` (`tournament.py:28`, `record_demo.py:17`), `glob`/`re` (`tournament.py:22-23`, `round_robin_tournament.py:18-19`), `Tuple`/`Any` (`src/agent.py:10`).

### 2.10 Checkpoint bloat: `tracker_history` grows forever [verified]
`MetricTracker.update` appends every loss/Q value to a Python list (`src/utils.py:46-50`), and the whole history is pickled into every checkpoint (`train.py:242`). Checkpoint size grows from **27 MB at step 100k to 70 MB at step 10M**. Use `collections.deque(maxlen=window_size)`; only the window is needed for the running mean.

### 2.11 Replay buffer
- **Memory isn't optimized the way the docs say.** `states` and `next_states` are stored separately as full 4-frame stacks (`src/replay_buffer.py:40-41`), so every frame is stored 8 times: 100k × 2 × 28,224 B ≈ **5.6 GB** of RAM. The plan doc (`DQN_ATARI_IMPLEMENTATION_PLAN.md` §3.3) describes a frame-ring buffer (~700 MB for 1e5), and the README calls the replay "Memory-Optimized". Storing single frames plus indices (with care at episode boundaries) gives ~8× savings and makes the Nature 1M-transition buffer (~7 GB) practical.
- `load()` (`:98-108`) assumes the saved size ≤ the current capacity and doesn't check `state_shape`. Resuming with a smaller `--buffer-size` crashes with a broadcasting error; with a larger one, `ptr` is wrong when the saved buffer was full. Validate and handle both cases.
- 97 `replay_step_*.npz` files (~32 MB each, ~3 GB total) are kept forever (`train.py:256`). Add a retention policy (`--keep-buffers N`).
- `sample()` makes two separate host→device copies of `uint8` stacks. Pinned memory with `non_blocking=True`, or a single `torch.from_numpy(...).to(device, non_blocking=True)`, helps throughput.

### 2.12 `tournament.py`
- The plot path is hard-coded to `checkpoints/tournament_leaderboard.png` (`:213`) and ignores `--output-csv`'s directory.
- `ax1.set_ylim(0, 22)` (`:237`) hides negative net scores, and `ax2.set_ylim(0, 15)` (`:252`) clips teams conceding more than 15. Weak checkpoints are drawn wrongly.
- The default contender list is hard-coded to one particular training run (`:37-48`, duplicated in `round_robin_tournament.py:47-58`). Move it to a shared helper or a JSON file.
- `round_robin_tournament.py:153, 293` hard-code "45 Pairings" / "9 other checkpoints" even when fewer than 10 checkpoints are found.

---

## 3. Low-Severity / Code-Quality Findings

| Location | Finding | Suggestion |
|---|---|---|
| `src/model.py:38` | Docstring says "Orthogonal/Kaiming", but only Kaiming is used. The output layer `fc2` gets Kaiming-ReLU init, which gives large initial Q-values. | Use a small init (e.g. `std=0.01`) or default init for the head, and fix the docstring. |
| `src/model.py:62` | `elif x.max() > 1.0` forces a GPU→CPU sync on every float forward pass and silently rescales data. | Accept `uint8` only (all callers pass uint8), or make normalization explicit. |
| `src/agent.py:156`, `replay_buffer.py:95` | `import os` inside functions. | Import at module top. |
| `src/agent.py:163` | `torch.load(..., weights_only=False)` runs arbitrary pickle code. Fine for your own files, unsafe for shared checkpoints (PvP / `--ai-checkpoint` / pool). | Save a weights-only artifact for distribution (`online_net.state_dict()`) and load it with `weights_only=True`. Keep the full pickle only for `latest.pt`. |
| `src/agent.py:104` | `online_net.train()` on every update and no `eval()` in `select_action`. This is a no-op today (no BN/dropout) but a trap later. | Set modes once, or call `eval()` in action selection. |
| `src/wrappers.py:72-84` | `MaxAndSkipEnv`: if the episode ends at `i < skip-2`, the max-pool uses stale frames from the previous step (self-play handles this case, the wrapper doesn't). The buffer is also not cleared on reset. | Mirror the fix from `train_selfplay.py:439-444`. |
| `src/wrappers.py:54` | `FireResetEnv` comment "2 = UP": action 2 is `RIGHT` in both Pong and Breakout. | Fix the comment. |
| `src/utils.py:31` | In the CUDA fallback `except`, `torch.cuda.get_device_name(0)` can itself raise. | Wrap it, or drop the name from the message. |
| `src/utils.py:16-17` | `cudnn.deterministic=True` / `benchmark=False` are always set, which costs speed, while Adam/scatter ops can still be nondeterministic. | Add a `--deterministic` flag. |
| `plot_metrics.py:23` | `--show` with `store_true, default=True` is redundant. Showing a PNG through `cv2.imshow` is roundabout. | `plt.show()` or `BooleanOptionalAction`. |
| `plot_metrics.py:63` | The training curve is labeled "Raw Atari Game Score", but training rewards are **clipped** and per-life (§1.5). | Relabel as "clipped training return". |
| `pvp_battle.py:63-70`, `play_remote.py:267`, `record_demo.py:38-46` | Defaults that depend on files existing (`os.path.exists` at parse time). | Fine for personal use; better to fail with a clear message. |
| `pvp_battle.py:83` | `device: str` annotation, but a `torch.device` is passed. | Fix the type hint. |
| `train_selfplay.py:33` et al. | `os.environ["LIBGL_ALWAYS_SOFTWARE"]="1"` set globally in headless training scripts. | Only needed for GUI modes; set it inside the GUI branch. |
| Everywhere | `print` + manual file writes for logging. | `logging` module, or TensorBoard/W&B for metrics. |
| Everywhere | Duplicate checkpoint/RNG/save/resume code in `train.py` and `train_selfplay.py`. | Extract a `TrainingState` / `Checkpointer` helper. |

---

## 4. Dependencies (`requirements.txt`)

1. **`torch==2.16.0.dev20261023+cu130`** is a nightly dev build that isn't available from PyPI, so `pip install -r requirements.txt` fails on a fresh machine. Pin a stable release and document the CUDA index URL (`--extra-index-url https://download.pytorch.org/whl/cu1xx`).
2. **`torchvision==0.20.1`** is unused and ABI-incompatible with torch 2.16 (0.20 pairs with torch 2.5). Remove it.
3. **`pygame==2.6.1` and `pygame-ce==2.5.8`** both install the `pygame` package and overwrite each other. Neither is imported. Remove both, or keep one.
4. Pinned CUDA/NVIDIA wheels (`cuda-toolkit`, `nvidia-*`, `triton`) make the file Linux + CUDA 13 + specific GPU only. Split it into `requirements.txt` (direct deps: `torch`, `gymnasium[atari]`, `ale-py`, `pettingzoo[atari]`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `imageio`, `pytest`) and a `requirements-lock.txt` for the exact environment.
5. `README.md:205` says "Python 3.10+". Check that against the pins (`numpy 2.5`, `pandas 3.1` very likely need ≥3.11) and state the version that was actually tested.
6. The plan doc mentions `gymnasium[accept-rom-license]`. This is obsolete: ROMs ship with `ale-py ≥ 0.9`.

---

## 5. Tests

Current state: 5 tests, all passing (`pytest -q` → `5 passed in 0.76s`). Coverage is limited to shapes and the buffer pointer. None of the bugs above would have been caught.

Recommended additions (in order of value):
1. **Preprocessing parity test:** a PettingZoo frame processed by the 2-player pipeline should produce the same gray levels as a Gym frame (catches §1.1).
2. **Agent target test:** with a hand-built tiny net/batch, check `targets = r + γ(1-d)·max Q_target` and the Double-DQN variant, and that `dones` mask the bootstrap.
3. **Schedule test:** `get_epsilon` after N env steps matches the documented semantics (§1.3), and target sync happens at the documented interval.
4. **Save/load round-trip** for `DQNAgent` (weights, optimizer, counters) and `ReplayBuffer` (including capacity-mismatch handling).
5. **Wrapper tests:** `EpisodicLifeEnv` on Breakout (terminal on life loss, no real reset), `FireResetEnv`, `MaxAndSkipEnv` early-termination frame, `NoopResetEnv` seeding.
6. **CLI smoke tests:** `--help` for every script (catches §2.2), plus a 200-step `train.py` run with a tiny buffer into `tmp_path`.
7. **Web server test:** the `/action` endpoint responds while a stream is open (catches §1.2).
8. `test_wrappers.py` catches `ImportError` and skips. Now that deps are pinned, let it fail loudly instead.

Also add CI (GitHub Actions: CPU-only torch, `pytest`, `ruff`), and a `pyproject.toml` so `src` is an installable package instead of relying on `pytest.ini: pythonpath = .`.

---

## 6. Documentation Review

### 6.1 `README.md` — factual errors

| Line(s) | Claim | Reality |
|---|---|---|
| 306-312 | PettingZoo switches to "Color Scheme #5"; single-agent court is olive `(109,118,43)`, paddles purple `(101,111,228)` | False: both envs use the same palette (§1.1). Remove the section. |
| 292 | "Color Normalization … Identical" | The remap *causes* a mismatch. |
| 284-304 | "100% consistent across every dimension" | Not identical: sticky actions (0.25 vs 0.0), Double DQN default (off vs forced on), point-terminal semantics, ε start with `--base-model`. |
| 301 | Single-agent: "Point loss ends transition (EpisodicLifeEnv)" | Pong has no lives, so `EpisodicLifeEnv` does nothing; points are not terminal in `train.py`. |
| 88, 258-259, 299-300 | ε anneals over 250k steps; target sync every 1,000 steps | ε anneals over ~1.01M env steps; sync every 4,000 env steps (§1.3). |
| 12-17, title | "Exact Nature 2015" | Deviations: Adam (not RMSProp), v5 sticky actions, buffer 100k (not 1M), replay start 10k (not 50k), target sync 4k env steps (not 10k updates = 40k steps), ε_end 0.02 (not 0.1), Kaiming init, extra grad-norm clipping at 10, eval ε = 0 (not 0.05). Add a **"Deviations from the paper"** table. |
| 15 | "Memory-Optimized Experience Replay" | Stores s and s′ separately, 8 copies per frame, ~5.6 GB (§2.11). |
| 104, 129 | Q-value stability "guaranteed by Double DQN"; conclusion credits Double DQN | `--double-dqn` defaults to off, and the run config wasn't recorded, so this can't be confirmed. "Guaranteed" overstates it in any case: Double DQN reduces overestimation, it doesn't guarantee its absence. |
| 157 | "identical set of randomized game seeds" | No-op counts aren't seeded (§2.8). |
| 334-336 | `plot_metrics.py --logs …` | No `--logs` flag exists. Use `--dir`, `--train-log`, `--eval-log`. |
| 346, 374 | Ctrl+C "saves `latest.pt` and `latest_replay.npz`" | True for `train.py`. In self-play, Ctrl+C goes down the "final" path and also writes `selfplay_final.pt`. |
| 368-374 | Resume restores everything | The in-progress episode, ALE RNG, and patience counter are not restored (§2.8, §2.9). |
| 115-119 | Non-transitivity "Root Cause" explanation | Speculative. It rests on 18 games per model with 2 games per pair, and the top two are tied 12-6 (decided by point differential), with the round robin using unpooled frames (§1.6). Present it as a hypothesis. Add confidence intervals or Elo ± CI from many more games. |
| 189 | Web mode "Supports keyboard input and touch buttons" | Broken (§1.2). |

### 6.2 `README.md` — structure / polish
- The directory tree (`:46-72`) is missing `train_selfplay.py`, `checkpoints*/`, and `pytest.ini`.
- Absolute `file:///home/weidu/...` links (`:141, :150`) are broken for anyone else. Use relative links.
- Duplicate `---` separators (`:74-76`, `:314-316`).
- Results are presented before Installation/Quickstart. A newcomer-friendly order is: Overview → Install → Quickstart (train/eval) → Results → Tools (tournament/PvP/play) → Internals → Deviations → References.
- Show the command lines used to produce the reported 10M-step run (exact flags, seed, whether `--double-dqn` was used), so the results can be reproduced.
- The Breakout run (`checkpoints_breakout/`) and the self-play results (`checkpoints_selfplay/selfplay_progress.png`) are committed but not documented.
- `pvp_best_model_right.gif` / `pvp_real_duel.gif` are in the working tree but git-ignored, so the README can't reference them. Either commit a small GIF on purpose (the `!` exception exists for PNGs) or drop them.

### 6.3 `DQN_ATARI_IMPLEMENTATION_PLAN.md`
- §3.3 describes a frame-sharing ring buffer that was never implemented (§2.11). Mark it as TODO, or implement it.
- §4 "Target network update freq: 10,000 steps … every C frames": the paper measures C in parameter updates. The table mixes frames, steps and updates throughout. Pick one unit and label it.
- Phase 3 step 1 says the wrapper should produce "(4, 84, 84) float tensors in [0, 1]". The implementation (correctly) produces `uint8` and normalizes in the model.
- §5 file structure is outdated (none of the tournament/PvP/self-play scripts are listed).
- Mentions `PongNoFrameskip-v4` as a benchmark, but the code only supports v5 semantics (sticky actions). Clarify.
- It reads as a pre-implementation plan. Either freeze it as "historical design doc" with a header note, or update it to match reality.

### 6.4 In-code docstrings
- `train_selfplay.py:7` describes two-perspective replay that isn't implemented (§2.5).
- `tournament.py:3` "Plays identical matches across fixed seeds": not identical (§2.8).
- `round_robin_tournament.py:3-10` hard-codes "10 contenders / 45 pairings / 90 matches".
- `src/model.py:38` "Orthogonal/Kaiming": only Kaiming.
- The argparse `help` strings for `--eps-decay`, `--target-sync-freq` (both trainers) and `--eps-start` (self-play, "0.15" vs code 0.20) are wrong.

---

## 7. Suggestions for Improvement (beyond bug fixes)

### Correctness / methodology
1. **Evaluation protocol:** fixed seed set, ε = 0.05 (paper) or report both, ≥30 episodes for final numbers, mean ± 95% CI, per-episode step cap.
2. **PvP rating:** replace win-rate with Elo/TrueSkill or Bradley-Terry fitted over ≥20 games per pair with side swapping, and report CIs. The current 2 games per pair can't separate the top checkpoints.
3. **Self-play:** fix §1.1/§1.4/§2.5. Consider prioritized sampling of opponents (PFSP: favor opponents the learner loses to), cap the pool size, and evaluate the self-play learner against the CPU too (via `tournament.py`) to detect regressions.
4. **Record config + git SHA** in every checkpoint and log directory.

### Performance
Measured throughput is ~260–750 FPS (single-agent) and ~100 FPS (self-play).
1. **Vectorized environments** (`gymnasium.vector.AsyncVectorEnv` or `envpool`) with batched action selection. This is usually a 5–20× gain for DQN data collection.
2. In self-play, `remap_pz_to_gym` (3 × `np.all` over 210×160×3 per frame, twice per step) is a large share of the CPU cost. Deleting it (§1.1) fixes correctness and speed together. Also compute the gray frame once and flip it, instead of processing the RGB frame twice.
3. Frame-ring replay buffer (§2.11), pinned memory, `torch.backends.cudnn.benchmark=True` when determinism isn't required, optional `torch.compile` / AMP for the update step.
4. Batch the learner and opponent forward passes into one call per step in self-play/PvP.

### Engineering
1. Package layout: `pyproject.toml`, `src/dqn_atari/…`, console entry points (`dqn-train`, `dqn-eval`, `dqn-pvp`, …).
2. Shared modules: `pong2p.py` (§1.6), `checkpointing.py` (resume/RNG/atomic save), `rendering.py` (HUD overlay + RGB/BGR + GIF/MP4 writer, currently duplicated in 3 scripts).
3. Config via dataclasses + `argparse` (or Hydra/`tyro`) so one config object drives both trainers and is saved alongside checkpoints.
4. Use `with` / `ExitStack` for all log files so early returns don't leak handles.
5. Add `ruff` + `mypy` (the code already uses type hints) and pre-commit hooks.
6. Checkpoint retention policy (keep last N + every M-th + best) and a "weights-only" export for sharing (safer with `weights_only=True`, ~4× smaller).

---

## 8. Things Done Well

- Correct **termination vs. truncation** handling: `train.py:284` stores `terminated` (not `terminated or truncated`) as `done`, so time-limit truncations still bootstrap. Many DQN implementations get this wrong.
- Atomic checkpoint writes (`tmp` + `os.replace`) for both model and replay buffer.
- `uint8` storage with on-device normalization, and a vectorized sampler.
- Graceful SIGINT handling and emergency checkpoint on crash in `train.py`.
- Clean, compact Nature CNN and agent. Double DQN is a one-flag switch.
- Good attention to max-pool frame alignment on early termination in `train_selfplay.py` / `pvp_battle.py` (it just needs to be shared everywhere, including `MaxAndSkipEnv`).
- Structured CSV metrics with `fsync`, and a useful 4-panel plotting script.

---

## Appendix A — Verification Commands Used

```bash
# Palette comparison (Gym vs PettingZoo), sticky-action defaults, action meanings
.venv/bin/python -c "import gymnasium as gym, ale_py; ..."   # see review session
# -> both envs: background (144,72,17), paddles (92,186,92)/(213,130,74)
# -> Gym (109,118,43) background only on the first frame after reset()
# -> Gym Pong-v5 repeat_action_probability=0.25, PettingZoo pong_v3 = 0.0

.venv/bin/python round_robin_tournament.py --help
# -> ValueError: unsupported format character ')' (0x29) at index 22

# Web server: open /stream.mjpg, then GET /action?act=2 with 3 s timeout
# -> TimeoutError; HTML Content-Length 5213 vs 5223 UTF-8 bytes

.venv/bin/python -m pytest -q        # -> 5 passed

# Epsilon schedule check from logs:
# checkpoints_breakout/training_metrics.csv @500000: eps 0.5198 = 1-0.98*((500000-10000)/4)/250000
# checkpoints_breakout/eval_metrics.csv mean_length ~21.6k-27.0k agent steps (= 108k-frame cap)
# checkpoint size: step_100000.pt 27 MB vs step_10000000.pt 70 MB (tracker_history growth)
```
