"""Unit tests for the DQN agent: TD targets, schedules, and checkpoint round-trips."""

import torch
import pytest
from src.agent import DQNAgent


def _batch(batch_size=8, num_actions=3, dones=None):
    states = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
    next_states = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
    actions = torch.randint(0, num_actions, (batch_size,))
    rewards = torch.randn(batch_size)
    if dones is None:
        dones = torch.zeros(batch_size, dtype=torch.bool)
    return states, actions, rewards, next_states, dones


@pytest.mark.parametrize("double_dqn", [False, True])
def test_td_target_matches_bellman_equation(double_dqn):
    torch.manual_seed(0)
    agent = DQNAgent(num_actions=3, gamma=0.9, double_dqn=double_dqn)
    # Make online and target nets differ so Double DQN is distinguishable from vanilla
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.add_(0.05 * torch.randn_like(p))

    dones = torch.tensor([False, True, False, True, False, False, True, False])
    states, actions, rewards, next_states, dones = _batch(dones=dones)

    with torch.no_grad():
        q_next_target = agent.target_net(next_states)
        if double_dqn:
            best = agent.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = q_next_target.gather(1, best).squeeze(1)
        else:
            next_q = q_next_target.max(dim=1)[0]
        expected = rewards + 0.9 * (~dones).float() * next_q
        q_pred = agent.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        expected_loss = torch.nn.functional.smooth_l1_loss(q_pred, expected)

    metrics = agent.update(states, actions, rewards, next_states, dones)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), rel=1e-5)


def test_epsilon_anneals_over_env_steps():
    agent = DQNAgent(num_actions=4, eps_start=1.0, eps_end=0.1, eps_decay_steps=1000)
    agent.env_steps = 0
    assert agent.get_epsilon() == pytest.approx(1.0)
    agent.env_steps = 500
    assert agent.get_epsilon() == pytest.approx(0.55)
    agent.env_steps = 5000
    assert agent.get_epsilon() == pytest.approx(0.1)
    # Gradient updates must not move the schedule
    agent.num_updates = 10**6
    agent.env_steps = 500
    assert agent.get_epsilon() == pytest.approx(0.55)


def test_target_sync_every_n_updates():
    agent = DQNAgent(num_actions=3, target_sync_freq=3)
    batch = _batch()
    for i in range(1, 4):
        agent.update(*batch)
        same = all(torch.equal(a, b) for a, b in zip(agent.online_net.state_dict().values(),
                                                     agent.target_net.state_dict().values()))
        assert same == (i == 3)
    assert agent.num_updates == 3


def test_save_load_round_trip(tmp_path):
    agent = DQNAgent(num_actions=3)
    agent.update(*_batch())
    agent.env_steps, agent.num_updates = 1234, 77
    path = str(tmp_path / "ckpt.pt")
    agent.save(path, extra_state={"global_step": 1234})

    other = DQNAgent(num_actions=3)
    ckpt = other.load(path)
    assert ckpt["global_step"] == 1234
    assert (other.env_steps, other.num_updates) == (1234, 77)
    for a, b in zip(agent.online_net.parameters(), other.online_net.parameters()):
        assert torch.equal(a, b)


def test_load_legacy_checkpoint_keys(tmp_path):
    """Old checkpoints stored the update count as total_steps and env steps as global_step."""
    agent = DQNAgent(num_actions=3)
    path = str(tmp_path / "old.pt")
    torch.save({
        "online_net": agent.online_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "total_steps": 250,
        "global_step": 1010,
    }, path)
    other = DQNAgent(num_actions=3)
    other.load(path)
    assert other.num_updates == 250
    assert other.env_steps == 1010


def test_load_weights_resets_training_state(tmp_path):
    agent = DQNAgent(num_actions=3)
    agent.env_steps, agent.num_updates = 10**7, 10**6
    full = str(tmp_path / "full.pt")
    agent.save(full)
    weights = str(tmp_path / "weights.pt")
    agent.save_weights(weights)

    for path in (full, weights):
        fresh = DQNAgent(num_actions=3, eps_start=0.2)
        fresh.load_weights(path)
        assert fresh.env_steps == 0 and fresh.num_updates == 0
        assert fresh.get_epsilon() == pytest.approx(0.2)
        for a, b in zip(agent.online_net.parameters(), fresh.target_net.parameters()):
            assert torch.equal(a, b)

    # The weights-only file is safe to load with weights_only=True
    torch.load(weights, weights_only=True)


@pytest.mark.parametrize("double_dqn", [False, True])
def test_masked_target_ignores_illegal_actions(double_dqn):
    """Bootstrap max/argmax must only consider actions legal in the next state."""
    from src.model_2048 import DQN2048

    torch.manual_seed(0)
    agent = DQNAgent(num_actions=4, gamma=0.9, double_dqn=double_dqn, model_fn=lambda: DQN2048())
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    b = 6
    states = torch.randint(0, 2, (b, 16, 4, 4), dtype=torch.uint8)
    next_states = torch.randint(0, 2, (b, 16, 4, 4), dtype=torch.uint8)
    actions = torch.randint(0, 4, (b,))
    rewards = torch.randn(b)
    dones = torch.tensor([False, False, False, False, False, True])
    masks = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        q_t = agent.target_net(next_states)
        select = agent.online_net(next_states) if double_dqn else q_t
        select = select.masked_fill(~masks, -float("inf"))
        best = select.argmax(dim=1)
        next_q = q_t.gather(1, best.unsqueeze(1)).squeeze(1)
        next_q[~masks.any(dim=1)] = 0.0  # no legal move: no bootstrap
        expected = rewards + 0.9 * (~dones).float() * next_q
        q_pred = agent.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        expected_loss = torch.nn.functional.smooth_l1_loss(q_pred, expected)

    metrics = agent.update(states, actions, rewards, next_states, dones, next_masks=masks)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), rel=1e-5)


def test_masked_select_action_never_picks_illegal():
    from src.model_2048 import DQN2048

    agent = DQNAgent(num_actions=4, model_fn=lambda: DQN2048(), eps_start=1.0, eps_end=1.0)
    state = torch.randint(0, 2, (16, 4, 4), dtype=torch.uint8).numpy()
    mask = torch.tensor([False, True, False, False]).numpy()
    for deterministic in (True, False):
        for _ in range(50):
            assert agent.select_action(state, deterministic=deterministic, action_mask=mask) == 1
