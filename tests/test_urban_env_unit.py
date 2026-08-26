import pytest
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from urbanmarl.envs.base_env import UrbanEnv, UrbanEnvBase


@pytest.fixture
def mock_env_config():
    """Returns standard environment configuration dict."""
    return {
        "num_uavs": 3,
        "num_ues": 8,
        "volume_size": (300, 300, 100),
        "max_steps": 20,
        "max_horizontal_speed": 35.0,
        "max_vertical_speed": 8.0,
        "max_transmit_power": 3.0,
    }


@pytest.fixture
def mock_urban_env(mock_env_config):
    """Instantiates a 2-environment vectorized UrbanEnv on CPU."""
    return UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=100,
        device=torch.device("cpu"),
        scenario="default",
        **mock_env_config,
    )


def test_urban_env_initialization(mock_urban_env, mock_env_config):
    """
    Verifies UrbanEnv initialization, batch size, group map creation, and spec construction.
    """
    assert isinstance(mock_urban_env, EnvBase)
    assert isinstance(mock_urban_env, UrbanEnvBase)
    assert mock_urban_env.batch_size == torch.Size([2])
    assert mock_urban_env.n_uavs == mock_env_config["num_uavs"]
    assert mock_urban_env.n_ues == mock_env_config["num_ues"]
    assert mock_urban_env.n_agents == mock_env_config["num_uavs"]
    group = list(mock_urban_env.group_map.keys())[0]
    assert group in mock_urban_env.group_map


def test_urban_env_reset_tensordict_mock(mock_urban_env):
    """
    Tests that reset() generates a valid root TensorDict containing required keys
    ('done', 'terminated', 'truncated', group observations, and state/info if specified).
    """
    td = mock_urban_env.reset()

    assert isinstance(td, TensorDict)
    assert td.batch_size == torch.Size([2])
    assert "done" in td.keys()
    assert "terminated" in td.keys()
    assert "truncated" in td.keys()

    group_name = list(mock_urban_env.group_map.keys())[0]
    assert group_name in td.keys()

    group_td = td.get(group_name)
    assert "observation" in group_td.keys()
    assert group_td["observation"].shape[:2] == torch.Size([2, mock_urban_env.n_uavs])


def test_urban_env_step_with_mock_action_tensordict(mock_urban_env):
    """
    Tests mocking UAV action inputs using PyTorch tensors and TensorDict,
    and executing step(td) without a full BenchMARL training loop.
    """
    td = mock_urban_env.reset()

    # Create mock action matching action spec
    action_spec = mock_urban_env.action_spec
    mock_action = action_spec.rand()

    td_input = td.update(mock_action)
    td_step = mock_urban_env.step(td_input)

    assert isinstance(td_step, TensorDict)
    assert "done" in td_step.keys()

    # Verify group next rewards are populated
    for group in mock_urban_env.group_map.keys():
        assert ("next", group, "reward") in td_step.keys(True, True) or "reward" in td_step.get(("next", group), {}).keys()


def test_urban_env_fake_tensordict(mock_urban_env):
    """
    Verifies that fake_tensordict() creates a synthetic TensorDict matching environment specs
    that can be safely stepped through the environment.
    """
    mock_urban_env.reset()
    fake_td = mock_urban_env.fake_tensordict()
    assert isinstance(fake_td, TensorDict)
    assert fake_td.batch_size == mock_urban_env.batch_size

    mock_action = mock_urban_env.action_spec.rand()
    fake_td_input = fake_td.update(mock_action)

    result_td = mock_urban_env.step(fake_td_input)
    assert isinstance(result_td, TensorDict)


def test_urban_env_rollout(mock_urban_env):
    """
    Tests multi-step rollout execution for a specified step count (e.g. n_steps=3).
    """
    n_steps = 3
    rollout_td = mock_urban_env.rollout(max_steps=n_steps)

    assert isinstance(rollout_td, TensorDict)
    assert rollout_td.shape[0] == 2
    assert rollout_td.shape[1] <= n_steps and rollout_td.shape[1] >= 1
    assert "done" in rollout_td.keys()
