"""Unit tests for 2048 game engine, gymnasium environment, and neural network."""

import numpy as np
import torch
import pytest

from src.games.game_2048 import Game2048
from src.games.env_2048 import Env2048, board_to_one_hot
from src.model_2048 import DQN2048


class TestGame2048:
    def test_slide_and_merge_row(self):
        # 1. Simple merge: [2, 0, 2, 4] -> [4, 4, 0, 0], points = 4
        row = np.array([2, 0, 2, 4], dtype=np.int32)
        new_row, pts = Game2048._slide_and_merge_row(row)
        assert np.array_equal(new_row, [4, 4, 0, 0])
        assert pts == 4

        # 2. Quad merge: [2, 2, 2, 2] -> [4, 4, 0, 0], points = 8
        row = np.array([2, 2, 2, 2], dtype=np.int32)
        new_row, pts = Game2048._slide_and_merge_row(row)
        assert np.array_equal(new_row, [4, 4, 0, 0])
        assert pts == 8

        # 3. Triple merge (left-associative): [2, 2, 2, 0] -> [4, 2, 0, 0], points = 4
        row = np.array([2, 2, 2, 0], dtype=np.int32)
        new_row, pts = Game2048._slide_and_merge_row(row)
        assert np.array_equal(new_row, [4, 2, 0, 0])
        assert pts == 4

        # 4. No-op row: [4, 8, 16, 32] -> unchanged, points = 0
        row = np.array([4, 8, 16, 32], dtype=np.int32)
        new_row, pts = Game2048._slide_and_merge_row(row)
        assert np.array_equal(new_row, [4, 8, 16, 32])
        assert pts == 0

    def test_directional_moves(self):
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 2],
        ], dtype=np.int32)

        # Slide LEFT
        new_b, pts, changed = game.simulate_move(Game2048.LEFT)
        assert changed
        assert pts == 8
        assert new_b[0, 0] == 4 and new_b[3, 0] == 4

        # Slide UP
        new_b_up, pts_up, changed_up = game.simulate_move(Game2048.UP)
        assert changed_up
        assert pts_up == 8
        assert new_b_up[0, 0] == 4 and new_b_up[0, 3] == 4

    def test_legal_moves_and_game_over(self):
        game = Game2048(seed=123)
        # Full board with no adjacent matches -> game over
        game.board = np.array([
            [2,  4,  2,  4],
            [4,  2,  4,  2],
            [2,  4,  2,  4],
            [4,  2,  4,  2],
        ], dtype=np.int32)
        assert game.is_game_over()
        mask = game.get_legal_actions_mask()
        assert not np.any(mask)

        # Full board WITH an adjacent match -> not game over
        game.board[0, 0] = 4
        assert not game.is_game_over()
        mask = game.get_legal_actions_mask()
        assert np.any(mask)


class TestEnv2048:
    def test_one_hot_encoding(self):
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = 2   # 2^1 -> channel 1
        board[0, 1] = 4   # 2^2 -> channel 2
        board[0, 2] = 2048 # 2^11 -> channel 11
        board[3, 3] = 0   # 0 -> channel 0

        one_hot = board_to_one_hot(board, num_classes=16)
        assert one_hot.shape == (16, 4, 4)
        assert one_hot[1, 0, 0] == 1
        assert one_hot[2, 0, 1] == 1
        assert one_hot[11, 0, 2] == 1
        assert one_hot[0, 3, 3] == 1

    def test_gym_interface(self):
        env = Env2048()
        obs, info = env.reset(seed=42)
        assert obs.shape == (16, 4, 4)
        assert "legal_mask" in info
        assert "score" in info

        # Step a legal move
        legal_actions = np.where(info["legal_mask"])[0]
        assert len(legal_actions) > 0
        act = legal_actions[0]
        next_obs, rew, done, trunc, next_info = env.step(act)
        assert next_obs.shape == (16, 4, 4)
        assert isinstance(rew, float)
        assert isinstance(done, bool)
        assert isinstance(trunc, bool)


class TestDQN2048Model:
    def test_forward_shape(self):
        model = DQN2048(in_channels=16, num_actions=4)
        batch = torch.zeros((8, 16, 4, 4), dtype=torch.uint8)
        out = model(batch)
        assert out.shape == (8, 4)


class TestGame2048Rules:
    def test_down_and_right(self):
        game = Game2048(seed=0)
        game.board = np.array([
            [2, 2, 4, 0],
            [0, 0, 4, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        down, pts, changed = game.simulate_move(Game2048.DOWN)
        assert changed and pts == 4 + 8
        np.testing.assert_array_equal(down[3], [4, 2, 8, 0])
        right, pts, changed = game.simulate_move(Game2048.RIGHT)
        assert changed and pts == 4
        np.testing.assert_array_equal(right[0], [0, 0, 4, 4])
        np.testing.assert_array_equal(right[2], [0, 0, 0, 2])

    def test_spawn_only_after_changing_move(self):
        game = Game2048(seed=1)
        game.board = np.array([[2, 0, 0, 0]] + [[0] * 4] * 3, dtype=np.int32)
        changed, _, _ = game.step(Game2048.LEFT)  # tile already at the left wall: illegal
        assert not changed and np.count_nonzero(game.board) == 1
        changed, _, _ = game.step(Game2048.RIGHT)
        assert changed and np.count_nonzero(game.board) == 2


def test_train_and_resume_smoke(tmp_path):
    """Short real training run, then a resume that continues from the saved step."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    common = ["--replay-start-size", "100", "--buffer-size", "500", "--batch-size", "16", "--eval-freq", "200",
              "--eval-episodes", "1", "--save-freq", "200", "--log-freq", "100", "--save-dir", str(tmp_path)]
    run = lambda extra: subprocess.run([sys.executable, "train_2048.py", *common, *extra], cwd=root,
                                       capture_output=True, text=True, timeout=300)
    first = run(["--total-steps", "400"])
    assert first.returncode == 0, first.stderr
    second = run(["--resume", str(tmp_path / "latest.pt"), "--additional-steps", "200"])
    assert second.returncode == 0, second.stderr
    assert "Resumed at step 401" in second.stdout
    ckpt = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert ckpt["global_step"] == 600 and ckpt["env_steps"] == 600 and "config" in ckpt


class TestSymmetry:
    def test_torch_and_numpy_transforms_agree(self):
        from src.games.symmetry_2048 import SYMMETRIES, transform_board, transform_tensor
        x = np.arange(2 * 16 * 16).reshape(2, 16, 4, 4)
        for t in range(len(SYMMETRIES)):
            np.testing.assert_array_equal(transform_tensor(torch.as_tensor(x), t).numpy(), transform_board(x, t))

    def test_action_perms_are_equivariant_on_random_boards(self):
        from src.games.symmetry_2048 import ACTION_PERMS, SYMMETRIES, transform_board
        assert sorted(map(tuple, ACTION_PERMS.tolist())) == sorted(set(map(tuple, ACTION_PERMS.tolist())))
        rng = np.random.default_rng(7)
        game = Game2048(seed=0)
        for _ in range(300):
            board = np.where(rng.random((4, 4)) < 0.3, 0, 2 ** rng.integers(1, 7, (4, 4))).astype(np.int32)
            for t in range(len(SYMMETRIES)):
                for a in range(4):
                    game.board = board
                    moved, pts, _ = game.simulate_move(a)
                    game.board = np.ascontiguousarray(transform_board(board, t))
                    moved_t, pts_t, _ = game.simulate_move(int(ACTION_PERMS[t][a]))
                    np.testing.assert_array_equal(moved_t, transform_board(moved, t))
                    assert pts_t == pts

    def test_augment_batch_keeps_transitions_consistent(self):
        """An augmented transition must still be a valid game transition (before the spawn)."""
        from src.games.symmetry_2048 import augment_batch
        rng = np.random.default_rng(3)
        game = Game2048(seed=0)
        boards, actions, nexts, masks = [], [], [], []
        while len(boards) < 64:
            board = np.where(rng.random((4, 4)) < 0.3, 0, 2 ** rng.integers(1, 6, (4, 4))).astype(np.int32)
            game.board = board
            legal = game.get_legal_actions_mask()
            if not legal.any():
                continue
            a = int(rng.choice(np.where(legal)[0]))
            moved, _, _ = game.simulate_move(a)
            game.board = moved
            boards.append(board); actions.append(a); nexts.append(moved.copy()); masks.append(game.get_legal_actions_mask())
        enc = lambda bs: torch.as_tensor(np.stack([board_to_one_hot(b) for b in bs]))
        s, a, ns, m = augment_batch(enc(boards), torch.tensor(actions), enc(nexts), torch.as_tensor(np.stack(masks)))
        decode = lambda x: np.where(x.argmax(0) > 0, 2 ** x.argmax(0), 0).astype(np.int32)
        for i in range(64):
            game.board = decode(s[i].numpy())
            moved, _, changed = game.simulate_move(int(a[i]))
            assert changed
            np.testing.assert_array_equal(moved, decode(ns[i].numpy()))
            game.board = decode(ns[i].numpy())
            np.testing.assert_array_equal(game.get_legal_actions_mask(), m[i].numpy())


def test_linear_and_log_rewards():
    for mode, scale, expected in [("log", 1.0, 3.0), ("linear", 1 / 1024, 8 / 1024), ("linear", 1.0, 8.0)]:
        env = Env2048(reward=mode, reward_scale=scale)
        env.reset(seed=0)
        env.game.board = np.array([[4, 4, 0, 0]] + [[0] * 4] * 3, dtype=np.int32)
        _, reward, _, _, info = env.step(Game2048.LEFT)
        assert info["points_gained"] == 8
        assert reward == pytest.approx(expected)
    with pytest.raises(ValueError):
        Env2048(reward="sqrt")
