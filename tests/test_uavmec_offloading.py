"""Unit tests for UAV-MEC Offloading Scenario."""

import pytest
import torch
from tensordict import TensorDict
from urbanmarl.envs.base_env import UrbanEnv
from urbanmarl.scenarios import load_scenario


@pytest.fixture
def mec_config():
    """Provides configuration dictionary for UAV-MEC offloading scenario."""
    return {
        "num_uavs": 3,
        "num_ues": 12,
        "volume_size": (500, 500, 200),
        "max_steps": 50,
        "max_horizontal_speed": 40.0,
        "max_vertical_speed": 10.0,
        "task_arrival_rate": 5.0,
        "task_data_size_min": 500000.0,
        "task_data_size_max": 1500000.0,
        "cpu_cycles_per_bit": 1000.0,
        "max_latency_deadline": 1.0,
        "ue_tx_power": 0.5,
        "uav_cpu_freq": 20.0e9,
        "uav_num_cores": 4,
        "distance_weight": 0.7,
        "extended_mec_obs": True,
        "has_state": True,
        "has_global_info": True,
    }


def test_uavmec_offloading_initialization(mec_config):
    """Verifies that the UAV-MEC scenario loads and sets configuration properly."""
    scenario = load_scenario("uavmec_offloading", mec_config)
    assert scenario is not None
    assert scenario.task_arrival_rate == 5.0
    assert scenario.uav_num_cores == 4
    assert scenario.extended_mec_obs is True


def test_uavmec_specs(mec_config):
    """Verifies observation, action, reward, and state specs for UAV-MEC."""
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=42,
        device=torch.device("cpu"),
        scenario="uavmec_offloading",
        **mec_config,
    )

    # Observation spec should be 8-dim when extended_mec_obs is True
    obs_spec = env.scenario.observation_spec(env, "agents")
    assert obs_spec.shape == torch.Size([8])

    # Action spec should be 3-dim [vh, phi, vz]
    act_spec = env.scenario.action_spec(env, "agents")
    assert act_spec.shape == torch.Size([3])

    # State spec should be defined and non-empty
    state_spec = env.scenario.state_spec(env)
    assert state_spec is not None
    assert state_spec.shape[0] > 0


def test_uavmec_step_and_rewards(mec_config):
    """Tests that a full environment step computes transmission, queuing, and rewards."""
    env = UrbanEnv(
        num_envs=4,
        continuous_actions=True,
        seed=10,
        device=torch.device("cpu"),
        scenario="uavmec_offloading",
        **mec_config,
    )

    td = env.reset()
    group = list(env.group_map.keys())[0]
    assert group in td.keys()
    assert td[(group, "observation")].shape == (4, 3, 8)

    # Verify initial MEC tensors exist
    assert hasattr(env, "uav_mec_utilization")
    assert hasattr(env, "uav_completed_tasks")
    assert hasattr(env, "uav_ue_sojourn")

    # Sample actions and step
    actions = env.action_spec.sample()
    td.update(actions)
    next_td = env.step(td)

    # Check next observation and rewards
    next_obs = next_td[("next", group, "observation")]
    reward = next_td[("next", group, "reward")]
    assert next_obs.shape == (4, 3, 8)
    assert reward.shape == (4, 3, 1)
    assert not torch.isnan(next_obs).any()
    assert not torch.isnan(reward).any()

    # Check that MEC queuing and task completion were computed
    assert env.uav_mec_utilization.shape == (4, 3)
    assert (env.uav_mec_utilization >= 0.0).all()
    assert (env.uav_completed_tasks >= 0.0).all()

    # Check global info
    info = env.scenario.info_global(env)
    assert "completed_tasks" in info
    assert "mean_system_time" in info
    assert "mean_utilization" in info
    assert "sojourn_time" in info


def test_uavmec_sojourn_time_accumulation(mec_config):
    """Verifies that sojourn time accumulates across consecutive LoS steps."""
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=24,
        device=torch.device("cpu"),
        scenario="uavmec_offloading",
        **mec_config,
    )

    td = env.reset()
    group = list(env.group_map.keys())[0]
    initial_sojourn = env.uav_ue_sojourn.clone()
    assert (initial_sojourn == 0.0).all()

    # Take step with zero velocity (hovering)
    hover_action = torch.zeros((2, 3, 3), device=env.device)
    td[(group, "action")] = hover_action
    next_td = env.step(td)

    # At least some UEs with LoS should have sojourn_time >= 1.0
    los = env.uav_ue_los
    if los.any():
        assert (env.uav_ue_sojourn[los] >= 1.0).all()
