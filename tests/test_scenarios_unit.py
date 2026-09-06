import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite
from urbanmarl.scenarios import (
    _registry,
    auto_register_scenarios,
    get_scenario_class,
    list_scenarios,
    load_scenario,
    register_scenario,
    UrbanScenario,
)


@pytest.fixture
def base_scenario_config():
    """Minimal configuration valid for all scenarios."""
    return {
        "num_uavs": 3,
        "num_ues": 10,
        "volume_size": (400, 400, 150),
        "max_steps": 50,
        "max_horizontal_speed": 40.0,
        "max_vertical_speed": 10.0,
        "max_transmit_power": 4.0,
        "has_state": True,
        "has_agent_info": True,
        "has_global_info": True,
    }


@pytest.mark.parametrize("scenario_name", list(_registry.keys()))
def test_scenario_loader(scenario_name, base_scenario_config):
    """
    Verifies that load_scenario successfully loads registered scenario classes
    and initializes them with scenario configurations.
    """
    scenario = load_scenario(scenario_name, base_scenario_config)
    assert isinstance(scenario, UrbanScenario)
    assert scenario.config == base_scenario_config


def test_invalid_scenario_loader():
    """
    Verifies that load_scenario raises ValueError when given an unknown scenario name.
    """
    with pytest.raises(ValueError, match="not found in registry"):
        load_scenario("non_existent_scenario_123")


@pytest.mark.parametrize("scenario_name", list(_registry.keys()))
def test_scenario_specs(scenario_name, base_scenario_config):
    """
    Tests that observation_spec and action_spec return valid Composite specs
    for each registered scenario.
    """
    from urbanmarl.envs.base_env import UrbanEnv

    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=42,
        device=torch.device("cpu"),
        scenario=scenario_name,
        **base_scenario_config,
    )

    group = list(env.group_map.keys())[0]

    obs_spec = env.scenario.observation_spec(env, group)
    act_spec = env.scenario.action_spec(env, group)

    from torchrl.data.tensor_specs import TensorSpec

    assert isinstance(obs_spec, TensorSpec)
    assert isinstance(act_spec, TensorSpec)


@pytest.mark.parametrize("scenario_name", list(_registry.keys()))
def test_scenario_mocked_step_execution(scenario_name, base_scenario_config):
    """
    Tests processing actions and computing step outputs (rewards, observations, done flags)
    using TensorDict mock inputs without requiring a full BenchMARL training loop.
    """
    from urbanmarl.envs.base_env import UrbanEnv

    batch_size = 2
    env = UrbanEnv(
        num_envs=batch_size,
        continuous_actions=True,
        seed=123,
        device=torch.device("cpu"),
        scenario=scenario_name,
        **base_scenario_config,
    )

    # 1. Mock Reset
    td_reset = env.reset()
    assert isinstance(td_reset, TensorDict)
    assert "done" in td_reset.keys()

    # 2. Mock Actions using PyTorch and TensorDict from action spec
    mock_action = env.action_spec.rand()
    mock_action_td = td_reset.update(mock_action)

    # 3. Step Environment with Mocked Actions
    td_next = env.step(mock_action_td)

    assert isinstance(td_next, TensorDict)
    assert "done" in td_next.keys()

    # 4. Check group rewards and observations
    for group in env.group_map.keys():
        group_td = td_next.get(("next", group), default=td_next.get(group))
        assert "reward" in group_td.keys() or ("next", group, "reward") in td_next.keys(
            True, True
        )


def test_dynamic_auto_registration():
    """Verifies that auto_register_scenarios finds and registers concrete scenarios."""
    scenarios = list_scenarios()
    expected = [
        "coverage",
        "default",
        "mec_offloading",
        "uav_lidar_navigation",
        "uav_mobile_ue",
        "uav_navigation",
        "uav_ue_los",
        "uavmec_advanced_physics",
        "uavmec_offloading",
    ]
    for exp in expected:
        assert exp in scenarios
        cls = get_scenario_class(exp)
        assert issubclass(cls, UrbanScenario)


def test_custom_scenario_registration(base_scenario_config):
    """Verifies that a custom subclass of UrbanScenario can be dynamically registered and loaded."""
    from urbanmarl.scenarios.default import Scenario as DefaultScenario

    class CustomTestScenario(DefaultScenario):
        pass

    register_scenario("custom_test_env", CustomTestScenario, aliases=["custom_alias"])
    assert "custom_test_env" in list_scenarios()
    assert "custom_alias" in list_scenarios()

    inst = load_scenario("custom_test_env", base_scenario_config)
    assert isinstance(inst, CustomTestScenario)

    inst_alias = load_scenario("custom_alias", base_scenario_config)
    assert isinstance(inst_alias, CustomTestScenario)


def test_invalid_registration():
    """Verifies that invalid or abstract classes raise TypeError on registration."""

    class NotAScenario:
        pass

    with pytest.raises(TypeError, match="must be a subclass of UrbanScenario"):
        register_scenario("invalid_cls", NotAScenario)

    with pytest.raises(TypeError, match="Cannot register abstract scenario class"):
        register_scenario("abstract_base", UrbanScenario)


def test_backward_compatibility_getattr():
    """Verifies that legacy PascalCase scenario class names are accessible via __getattr__."""
    import urbanmarl.scenarios as scenarios_mod

    assert hasattr(scenarios_mod, "NavigationScenario")
    assert hasattr(scenarios_mod, "DefaultScenario")
    assert hasattr(scenarios_mod, "UAVMECScenario")
    assert hasattr(scenarios_mod, "UavMecAdvancedPhysicsScenario")
    assert hasattr(scenarios_mod, "CoverageScenario")
    assert hasattr(scenarios_mod, "MecOffloadingScenario")
