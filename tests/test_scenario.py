import pytest

import sys
import os
from pathlib import Path

# Add project root to path (adjust if notebook is in a subfolder)
project_root = Path.cwd()#.parent  # if notebook is in experiments/ or similar
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torchrl.envs import EnvBase
from urbanmarl.envs.base_env import UrbanEnv
from urbanmarl.scenarios import _registry, load_scenario

# ---------- Fixtures ----------

@pytest.fixture
def base_config():
    """Minimal configuration valid for all scenarios."""
    return {
        "num_uavs": 2,
        "num_ues": 5,
        "volume_size": (500, 500, 200),
        "max_time_slots": 10,
        "max_horizontal_speed": 49.0,
        "max_vertical_speed": 12.0,
        "max_transmit_power": 5.0,
        "frequency_ghz": 29.0,
        "g2a_bandwidth": 10e6,
        "noise_figure_db": 7.0,
    }

@pytest.fixture(params=list(_registry.keys()))
def scenario_name(request):
    """Parametrize over all registered scenarios."""
    return request.param

@pytest.fixture
def env(scenario_name, base_config):
    """Create an environment instance for the given scenario."""
    return UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=42,
        device="cpu",
        scenario=scenario_name,
        **base_config,
    )

# ---------- Tests ----------

def test_env_creation(env, scenario_name):
    """Environment should be created with correct batch size and scenario name."""
    assert env.batch_size == torch.Size((2,))
    assert env.scenario_name == scenario_name
    assert isinstance(env, EnvBase)


def test_reset(env):
    """Reset should return a valid TensorDict with required keys."""
    td = env.reset()
    assert td is not None
    assert "done" in td.keys()
    assert "agents" in td.keys()
    group = list(env.group_map.keys())[0]
    assert "observation" in td.get(group).keys()
    # state and info are optional, but if present they must be in the root
    if env.scenario.has_state:
        assert "state" in td.keys()
    if env.scenario.has_global_info:
        assert "info" in td.keys()


def test_step(env):
    """Step should consume an action and return a new TensorDict."""
    tensordict = env.reset()
    action = env.action_spec.sample()
    new_td = tensordict.update(action)
    td = env.step(new_td)
    assert td is not None
    assert "done" in td.keys()
    assert "agents" in td.keys()
    # Ensure reward is present for each group
    for group in env.group_map.keys():
        assert "reward" in td.get(("next", group)).keys()


def test_rollout(env):
    """Rollout should run for the requested number of steps without errors."""
    n_steps = 3
    rollout = env.rollout(n_steps)
    assert rollout.shape[1] == n_steps
    # Check that done is present and has the correct batch size
    assert "done" in rollout.keys()
    assert rollout["done"].shape == torch.Size((env.batch_size[0], n_steps, 1))


def test_check_env_specs(env):
    """The environment's specs must be internally consistent (no errors)."""
    assert env.check_env_specs() == None


def test_fake_tensordict(env):
    from tensordict import TensorDictBase
    """fake_tensordict should match the specs and be usable in step."""
    _ = env.reset()
    fake = env.fake_tensordict()
    action = env.action_spec.sample()
    fake = fake.update(action)
    # Step with fake action should not crash
    assert isinstance(env.step(fake), TensorDictBase)


def test_global_info_present_when_has_flag(env):
    """If has_global_info is True, 'info' must be present in reset/step outputs."""
    if env.scenario.has_global_info:
        td = env.reset()
        assert "info" in td.keys()
        # Additionally, info should be a TensorDict with specs matching
        info_spec = env.full_observation_spec_unbatched.get("info")
        assert info_spec is not None
        # Check that actual info shape matches spec (if spec defines shape)
        # We can't check all, but we can at least verify it's not empty.
        assert len(td["info"].keys()) > 0
    else:
        pytest.skip(f"Scenario: {env.scenario_name} does not have global info.")


def test_agent_info_present_when_has_flag(env):
    """If has_agent_info is True, each group must contain an 'info' key."""
    if env.scenario.has_agent_info:
        td = env.reset()
        for group in env.group_map.keys():
            group_td = td.get(group)
            assert "info" in group_td.keys()
        # Also test after a step
        action = env.action_spec.sample()
        td_step = env.step(action)
        for group in env.group_map.keys():
            group_td = td_step.get(group)
            assert "info" in group_td.keys()
    else:
        pytest.skip(f"Scenario: {env.scenario_name} does not have agent info.")


def test_scenario_specs_match_actual_data(env):
    """
    Verify that the data returned by the scenario matches the declared specs.
    This checks that observation, action, reward, info (global & agent) shapes
    and dtypes are consistent.
    """
    # Reset and get a sample
    td = env.reset()
    # Check observation spec vs actual
    for group, agent_names in env.group_map.items():
        obs_spec = env.scenario.observation_spec(env, group)
        obs_data = td.get(group)["observation"]
        # Check shape (ignoring batch dims)
        assert obs_data.shape[2:] == obs_spec.shape, \
            f"Observation shape mismatch for group {group}: spec {obs_spec.shape}, data {obs_data.shape[2:]}"
        assert obs_data.dtype == obs_spec.dtype, \
            f"Observation dtype mismatch for group {group}: spec {obs_spec.dtype}, data {obs_data.dtype}"

    # Check global state if present
    if env.scenario.has_state:
        state_spec = env.scenario.stat_spec(env)
        state_data = td["state"]
        assert state_data.shape[1:] == state_spec.shape, \
            f"State shape mismatch: spec {state_spec.shape}, data {state_data.shape[1:]}"
        assert state_data.dtype == state_spec.dtype

    # Check global info if present
    if env.scenario.has_global_info:
        info_spec = env.scenario.info_global_spec(env)
        info_data = td["info"]
        # info_spec is a Composite; we check each subkey
        for key, spec in info_spec.items():
            data = info_data[key]
            assert data.shape[1:] == spec.shape, \
                f"Global info '{key}' shape mismatch: spec {spec.shape}, data {data.shape[1:]}"
            assert data.dtype == spec.dtype

    # Check agent info if present
    if env.scenario.has_agent_info:
        info_spec = env.scenario.info_agent_spec(env, group)  # same for all groups in simple case
        for group, agent_names in env.group_map.items():
            info_data = td.get(group)["info"]
            for key, spec in info_spec.items():
                data = info_data[key]
                # data shape: (batch, n_agents_in_group, ...)
                assert data.shape[2:] == spec.shape, \
                    f"Agent info '{key}' shape mismatch for group {group}: spec {spec.shape}, data {data.shape[2:]}"
                assert data.dtype == spec.dtype


# ---------- Optional: Test specific scenario edge cases ----------
# If some scenarios require special handling, add them here.

def test_navigate_scenario_specific(env, scenario_name):
    """Additional checks for the 'navigate' scenario."""
    # if scenario_name != "navigate":
    #     pytest.skip("Only for 'navigate' scenario")
    # Check that agent info keys are as expected (collision, velocity_norm)
    if env.scenario.has_agent_info:
        td = env.reset()
        for group in env.group_map.keys():
            info = td.get(group)["info"]
            assert "collision" in info.keys()
            assert "velocity_norm" in info.keys() 
