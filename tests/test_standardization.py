"""Unit tests for Phase 7: GNN Message Passing Policy and PettingZoo Wrapper."""

import numpy as np
import pytest
import torch
from urbanmarl.envs.pettingzoo_wrapper import UrbanPettingZooEnv
from urbanmarl.models.gnn_policy import VectorizedGNNMessagePassing


def test_gnn_message_passing():
    """Tests GNN message-passing layer forward pass and output embedding shapes."""
    b, n, node_dim = 2, 4, 12
    gnn = VectorizedGNNMessagePassing(
        node_dim=node_dim, hidden_dim=32, out_dim=32, r_comm=100.0
    )

    features = torch.randn(b, n, node_dim)
    positions = torch.tensor(
        [
            [
                [0.0, 0.0, 50.0],
                [30.0, 0.0, 50.0],
                [0.0, 200.0, 50.0],
                [500.0, 500.0, 50.0],
            ],
            [
                [0.0, 0.0, 50.0],
                [20.0, 20.0, 50.0],
                [40.0, 40.0, 50.0],
                [60.0, 60.0, 50.0],
            ],
        ]
    )
    velocities = torch.randn(b, n, 3)

    embeddings = gnn(features, positions, velocities)
    assert embeddings.shape == (b, n, 32)
    assert not torch.isnan(embeddings).any()


def test_pettingzoo_wrapper_api():
    """Verifies that UrbanPettingZooEnv conforms to standard multi-agent reset and step APIs."""
    pz_env = UrbanPettingZooEnv(
        scenario="uav_navigation",
        num_uavs=3,
        num_ues=5,
        volume_size=(300, 300, 100),
        max_steps=10,
    )

    assert len(pz_env.possible_agents) == 3
    assert pz_env.possible_agents == ["uav_0", "uav_1", "uav_2"]

    obs, infos = pz_env.reset(seed=42)
    assert len(obs) == 3
    for agent in pz_env.possible_agents:
        assert agent in obs
        assert pz_env.observation_space(agent).contains(obs[agent])

    # Sample actions from action space
    actions = {
        agent: pz_env.action_space(agent).sample() for agent in pz_env.possible_agents
    }
    next_obs, rewards, term, trunc, next_infos = pz_env.step(actions)

    assert len(next_obs) == 3
    assert len(rewards) == 3
    assert len(term) == 3
    assert len(trunc) == 3

    for agent in pz_env.possible_agents:
        assert isinstance(rewards[agent], float)
        assert isinstance(term[agent], bool)
        assert isinstance(trunc[agent], bool)

    pz_env.close()
