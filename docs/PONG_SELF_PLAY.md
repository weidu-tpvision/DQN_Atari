# Pong self-play

The current trainer learns against frozen copies of its own policy in PettingZoo’s two-player Pong. This guide describes the implemented trainer, what the committed results can establish, and the controlled comparison proposed in the original self-play plan. For commands and flags, see the [README](../README.md#self-play-training-multi-agent-league-train_selfplaypy).

## Current implementation

| Area | Current behavior |
| --- | --- |
| **Game and observations** | [`TwoPlayerPong`](../src/pong2p.py#L51) runs PettingZoo Pong in ALE two-player mode. It uses 1–30 reset no-ops, four-frame action repeat, two-frame max pooling, and four grayscale frames at `84 × 84`. The left view is mirrored so the policy sees its own paddle on the right. |
| **Training** | [`train_selfplay.py`](../train_selfplay.py#L439) chooses one opponent and a random learner side per match. The learner uses its ε schedule and the opponent uses a greedy policy with ε = 0.05. Replay stores only the learner’s transitions. Double DQN is on by default. |
| **Rewards and terminals** | The learner receives clipped raw game rewards, including a penalty if it holds the ball and fails to serve. The penalty changes the reward without awarding a point. A scored point or a true game ending cuts off bootstrapping. A time-limit cutoff alone still bootstraps. |
| **Opponent sampling** | The active [snapshot pool](../train_selfplay.py#L56) defaults to 50 weights-only snapshots. Training picks the latest with probability 0.2. Otherwise it samples uniformly or uses the default PFSP weights `(1 − p)²`, where `p` is the smoothed learner win rate over up to 20 recent matches against that opponent. Retired snapshots stay on disk. |
| **Promotion** | Every 25,000 steps by default, [evaluation](../train_selfplay.py#L193) plays 20 matches targeting seven points, alternating sides and sampling opponents uniformly from the active pool. Both policies use ε = 0.02. The match cap is 10,000 agent steps, and a draw counts as half a win. A score of 65% or higher promotes a new snapshot. |
| **Resume and warm-start** | Resume restores learner weights and optimizer state, step counters, the saved ε schedule, and the active snapshot list and match records. Full checkpoints can also restore matching replay and saved RNG state. Keep the referenced replay snapshot and adjacent `pool/` with the checkpoint. Warm-start loads weights, starts fresh optimizer and counters, and caps the starting ε at 0.20. [Resume code](../train_selfplay.py#L322) · [Checkpoint guide](../README.md#resuming-training) |

## Comparing with the built-in CPU

| | Single-player ALE `Pong-v5` | Two-player PettingZoo `pong_v3` |
| --- | --- | --- |
| **Game mode and opponent** | Mode 0, with the built-in CPU on the left. | Mode 4, with a policy or human controlling each paddle. |
| **Serving** | Automatic in single-player Pong. | The player holding the ball must press FIRE. Waiting too long incurs a reward penalty. |
| **Sticky actions** | 0.25 with the default ALE v5 environment. | 0.0 in the current two-player environment. |
| **Training bootstrap** | Continues across points and stops at a true game ending. | Stops at every scored point or true game ending. |
| **Default DQN target** | Vanilla DQN. | Double DQN. |

The shared two-player pipeline uses the matching active-play palette and grayscale transformation. The game mode, serving, sticky actions, and bootstrap rules still change the learning problem. A match against the built-in CPU is useful as a separate compatibility check. A comparison that isolates opponent selection needs fixed-opponent and self-play agents trained and tested in the same two-player environment. [Single-player environment](../src/wrappers.py#L210) · [Two-player environment](../src/pong2p.py#L1) · [Nature paper comparison](DQN_PAPER_COMPARISON.md)

## What the committed results show

- [Self-play metrics](../checkpoints_selfplay/selfplay_metrics.csv) cover a historical run through 1 million agent steps. The last row records 632 matches, 21 snapshots, and a 66.67% win rate against that run’s pool. The run predates the removal of the palette remap and the current PFSP defaults. Pool membership changed as the run progressed, so each reported win rate measures performance against the pool active at that checkpoint.
- The [historical round robin](../checkpoints/round_robin_results.csv) played two games per pairing, one on each side, with an older preprocessing path. The top two checkpoints finished level on wins. The result motivates a larger comparison under the shared pipeline. The [current tournament code](../round_robin_tournament.py#L114) retains the two-games-per-pair format.
- The [single-player Pong run](../checkpoints/eval_metrics.csv) reached a peak mean of +20.4 against the built-in CPU, close to the +21 scoring ceiling. This leaves little room for the CPU benchmark to separate strong checkpoints.

## Proposed controlled experiment

The current trainer supports pooled uniform and PFSP sampling. A controlled fixed-opponent comparison still needs a scripted or frozen-opponent mode in the shared trainer. The original plan identified a RAM ball tracker as a candidate. It would use ball coordinates at RAM bytes 49 and 54 and paddle heights at 50 and 51, then issue normal player actions. The original measurements came from mode 0, so the bot would need calibration in mode 4 before it can serve as a comparison baseline. See the [historical self-play plan](https://github.com/weidu-tpvision/DQN_Atari/blob/08d531c7008cdb9e3245cc48cbdc14f4c89a5/SELF_PLAY_PLAN.md).

| Training condition | Status |
| --- | --- |
| Single-player DQN against the built-in CPU | Implemented. Use as a separate cross-mode reference. |
| Two-player DQN against a fixed scripted opponent | Proposed primary control. |
| Two-player DQN against a frozen learned policy | Proposed second control. |
| Two-player DQN against a regularly refreshed latest copy | Proposed ablation for cycling and forgetting. |
| Two-player DQN against a uniform snapshot pool | Implemented with `--opponent-sampling uniform`. Use `--latest-prob 0` for strictly uniform training samples. |
| Two-player DQN against a prioritized snapshot pool | Implemented and enabled by default. |

The comparison would hold preprocessing, game rules, sticky actions, terminal semantics, optimizer, and learner-step budgets constant within two-player runs. The original proposal used three matched training seeds, 10 million learner steps per run, and saved checkpoints at every million steps. A shorter pilot would establish that each condition can learn to serve and that the controls behave as expected. Compare policies at the same training step and report results per training seed with uncertainty across seeds.

| Evaluation | Proposed measurement |
| --- | --- |
| Fixed two-player opponent | Run 30 games against the same scripted anchor with fixed reset seeds and balanced sides. Track score margin and win rate. |
| Head-to-head league | Use at least 20 games per pairing with balanced sides. Report the cross-play matrix and estimated strengths with uncertainty. |
| Unseen opponents | Break out results against policies trained with other seeds or opponent-selection methods. |
| Exploitability | Freeze each final policy and train a fresh opponent against it. Compare how quickly the new opponent learns to win under an equal step budget. |
| Built-in CPU | Report scores as a separate cross-mode compatibility check. Use the [paper comparison](DQN_PAPER_COMPARISON.md#evaluation-and-reporting-differences) for the differences between the current evaluator and the Nature benchmark. |
