"""Unit tests for Vectorized User Mobility models and UAV_MOBILE_UE scenario."""

import pytest
import torch
from tensordict import TensorDict
from urbanmarl.envs.base_env import UrbanEnv
from urbanmarl.models.mobility import VectorizedUserMobility


@pytest.fixture
def mobility_config():
    """Environment configuration for mobile UE scenario testing."""
    return {
        "num_uavs": 2,
        "num_ues": 6,
        "volume_size": (400, 400, 150),
        "max_steps": 15,
        "max_horizontal_speed": 40.0,
        "max_vertical_speed": 10.0,
        "mobility_model": "manhattan",
        "ue_speed_min": 1.0,
        "ue_speed_max": 2.5,
    }


def test_user_mobility_engine_models():
    """Tests that all mobility models (Manhattan, RWP, Gauss-Markov, Hotspot) step properly."""
    volume_size = (400.0, 400.0, 150.0)

    for model in ["manhattan", "rwp", "gauss_markov", "hotspot"]:
        engine = VectorizedUserMobility(
            volume_size=volume_size,
            model_type=model,
            speed_min=0.5,
            speed_max=2.0,
        )

        b, m = 2, 8
        init_vel = engine.initialize_velocities(b, m)
        assert init_vel.shape == (b, m, 3)
        assert torch.all(init_vel[..., 2] == 0.0)

        if model == "hotspot":
            engine.initialize_hotspots(b, num_hotspots=2)

        init_pos = torch.zeros((b, m, 3))
        init_pos[..., 2] = 1.5

        new_pos, new_vel = engine.step(init_pos, init_vel, dt=1.0)
        assert new_pos.shape == (b, m, 3)
        assert new_vel.shape == (b, m, 3)

        # Ensure ground plane constraint is maintained
        assert torch.allclose(new_pos[..., 2], torch.tensor(1.5))

        # Ensure bounds are respected
        assert torch.all(new_pos[..., 0] >= -200.0) and torch.all(
            new_pos[..., 0] <= 200.0
        )
        assert torch.all(new_pos[..., 1] >= -200.0) and torch.all(
            new_pos[..., 1] <= 200.0
        )


def test_uav_mobile_ue_scenario_integration(mobility_config):
    """Verifies that uav_mobile_ue scenario integrates with UrbanEnv and updates UE positions."""
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=101,
        device=torch.device("cpu"),
        scenario="uav_mobile_ue",
        **mobility_config,
    )

    td = env.reset()
    assert td is not None

    initial_ue_pos = env.ue_user_pos.clone()

    # Step the environment with random action
    action = env.action_spec.sample()
    td_step = td.update(action)
    next_td = env.step(td_step)

    assert next_td is not None
    # Ground users must have moved!
    assert not torch.allclose(env.ue_user_pos, initial_ue_pos)

    # Check that rewards and done are valid in the step transition
    next_data = next_td.get("next")
    assert ("uav", "reward") in next_data.keys(
        True, True
    ) or "reward" in next_data.keys()
    assert "done" in next_td.keys()
    assert not torch.isnan(next_td.get("done")).any()


def test_uav_mobile_ue_sojourn_accumulation(mobility_config):
    """Verifies that sojourn time is tracked over mobile UEs."""
    env = UrbanEnv(
        num_envs=1,
        continuous_actions=True,
        seed=42,
        device=torch.device("cpu"),
        scenario="uav_mobile_ue",
        **mobility_config,
    )
    env.reset()

    # Force artificial LoS to True
    env.uav_ue_los = torch.ones_like(env.uav_ue_los, dtype=torch.bool)

    action = env.action_spec.sample()
    fake_td = env.reset().update(action)
    env.step(fake_td)

    assert hasattr(env, "uav_ue_sojourn")
    assert env.uav_ue_sojourn.max() <= 1.0
