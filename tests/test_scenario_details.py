import pytest
import torch
from tensordict import TensorDict
from urbanmarl.envs.base_env import UrbanEnv
from urbanmarl.scenarios import load_scenario


@pytest.fixture
def env_config():
    """Provides base environment configuration dictionary."""
    return {
        "num_uavs": 3,
        "num_ues": 6,
        "volume_size": (400, 400, 150),
        "max_steps": 10,
        "max_horizontal_speed": 40.0,
        "max_vertical_speed": 10.0,
        "max_transmit_power": 4.0,
        "has_state": True,
        "has_global_info": True,
    }


def test_battery_consumption_kinematics(env_config):
    """
    Tests that UAV motion action results in battery energy consumption
    based on horizontal and vertical speed components.
    """
    env = UrbanEnv(
        num_envs=1,
        continuous_actions=True,
        seed=42,
        device=torch.device("cpu"),
        scenario="uav_navigation",
        **env_config
    )

    td = env.reset()
    initial_battery = env.uav_battery.clone()

    # Create motion action: positive velocity in horizontal (speed=20.0) and vertical direction (speed=5.0)
    group_name = list(env.group_map.keys())[0]
    action_dict = {
        group_name: TensorDict(
            {"action": torch.tensor([[[20.0, 0.0, 5.0],
                                      [20.0, 0.0, 5.0],
                                      [20.0, 0.0, 5.0]]])},
            batch_size=torch.Size([1, 3]),
            device=env.device,
        )
    }

    td_input = td.update(action_dict)
    _ = env.step(td_input)

    # Battery must decrease after movement
    assert (env.uav_battery < initial_battery).all()


def test_done_termination_on_max_steps(env_config):
    """
    Tests that environment truncation flag becomes True when current_step reaches max_steps.
    """
    env_config_short = env_config.copy()
    env_config_short["max_steps"] = 3

    env = UrbanEnv(
        num_envs=1,
        continuous_actions=True,
        seed=1,
        device=torch.device("cpu"),
        scenario="uav_navigation",
        **env_config_short
    )

    td = env.reset()
    action_spec = env.action_spec

    for step in range(3):
        mock_action = action_spec.rand()
        td = env.step(td.update(mock_action))

    assert env.current_step[0].item() == 3
    done, terminated, truncated = env.scenario.done(env)
    assert truncated[0].item() is True or done[0].item() is True


def test_coverage_scenario_distance_penalty(env_config):
    """
    Tests CoverageScenario distance penalty calculation when UAVs are placed closer
    than min_distance_threshold.
    """
    coverage_config = env_config.copy()
    coverage_config["min_distance_threshold"] = 100.0

    env = UrbanEnv(
        num_envs=1,
        continuous_actions=True,
        seed=10,
        device=torch.device("cpu"),
        scenario="coverage",
        **coverage_config
    )

    env.reset()

    # Manually place UAV 0 and UAV 1 very close (distance = 10m < 100m)
    env.uav_agents_pos[0, 0] = torch.tensor([0.0, 0.0, 50.0])
    env.uav_agents_pos[0, 1] = torch.tensor([5.0, 5.0, 50.0])
    env.uav_agents_pos[0, 2] = torch.tensor([200.0, 200.0, 50.0])

    group_name = list(env.group_map.keys())[0]
    reward = env.scenario.reward(env, group=group_name)

    # Reward tensor shape (batch_size, n_uavs, 1)
    assert reward.shape == (1, 3, 1)
    # UAV 0 and UAV 1 should receive distance penalty reduction
    assert reward[0, 0].item() < reward[0, 2].item()


def test_global_info_dictionary_structure(env_config):
    """
    Tests that info_global(env) returns a valid dictionary with urban_params,
    collisions, velocity, and los keys.
    """
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=77,
        device=torch.device("cpu"),
        scenario="uav_navigation",
        **env_config
    )

    env.reset()
    info = env.scenario.info_global(env)

    assert isinstance(info, dict)
    assert "urban_params" in info
    assert "collisions" in info
    assert "velocity" in info
    assert "los" in info

    assert info["urban_params"].shape == (2, 4)
    assert info["collisions"].shape == (2, 1) or info["collisions"].shape == (2,)
