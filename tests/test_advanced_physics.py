"""Unit tests for high-fidelity aerodynamics, 3D channel models, and UAVMEC_ADVANCED_PHYSICS scenario."""

import pytest
import torch
from urbanmarl.envs.base_env import UrbanEnv
from urbanmarl.models.aerodynamics import VectorizedAerodynamics
from urbanmarl.models.channel_advanced import AdvancedChannelModel


@pytest.fixture
def physics_config():
    """Environment configuration for high-fidelity physics scenario."""
    return {
        "num_uavs": 2,
        "num_ues": 5,
        "volume_size": (400, 400, 150),
        "max_steps": 15,
        "max_horizontal_speed": 40.0,
        "max_vertical_speed": 10.0,
    }


def test_aerodynamics_power_model():
    """Tests rotary-wing UAV power consumption across hover, forward flight, and vertical climb."""
    aero = VectorizedAerodynamics()

    # 1. Hover condition (zero velocity)
    v_hover = torch.zeros((1, 1, 3))
    p_hover = aero.compute_propulsion_power(v_hover).item()
    # Expected hover power around P0 + Pi = 79.86 + 88.63 = 168.49 W
    assert 160.0 <= p_hover <= 180.0

    # 2. Fast forward flight (v_h = 25 m/s)
    v_fast = torch.tensor([[[25.0, 0.0, 0.0]]])
    p_fast = aero.compute_propulsion_power(v_fast).item()
    assert p_fast > p_hover

    # 3. Climbing flight (v_z = 5 m/s)
    v_climb = torch.tensor([[[0.0, 0.0, 5.0]]])
    p_climb = aero.compute_propulsion_power(v_climb).item()
    assert p_climb > p_hover


def test_advanced_channel_model():
    """Tests 3GPP 3D directional antenna gain and achievable data rates."""
    channel = AdvancedChannelModel(frequency_ghz=29.0, bandwidth_hz=10e6)

    # UAV at (0, 0, 50)
    tx_pos = torch.tensor([[[0.0, 0.0, 50.0]]])
    # UE directly below at (0, 0, 0)
    rx_nadir = torch.tensor([[[0.0, 0.0, 0.0]]])
    # UE far away at (200, 200, 0)
    rx_edge = torch.tensor([[[200.0, 200.0, 0.0]]])

    gain_nadir = channel.compute_3d_antenna_gain(tx_pos, rx_nadir).item()
    gain_edge = channel.compute_3d_antenna_gain(tx_pos, rx_edge).item()

    # Boresight gain must exceed far-angle gain
    assert gain_nadir > gain_edge

    rate_nadir = channel.compute_data_rates(tx_pos, rx_nadir, tx_power=2.0).item()
    rate_edge = channel.compute_data_rates(tx_pos, rx_edge, tx_power=2.0).item()

    assert rate_nadir > rate_edge
    assert rate_edge > 0.0


def test_uavmec_advanced_physics_integration(physics_config):
    """Verifies that uavmec_advanced_physics scenario executes properly with UrbanEnv."""
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=101,
        device=torch.device("cpu"),
        scenario="uavmec_advanced_physics",
        **physics_config,
    )

    td = env.reset()
    assert td is not None

    obs = env._get_obs()
    assert obs.shape[-1] == 8

    # Step environment
    action = env.action_spec.sample()
    stepped = env.step(td.update(action))
    assert stepped is not None

    assert hasattr(env, "uav_mec_utilization")
    assert hasattr(env, "uav_completed_tasks")
    assert env.uav_completed_tasks.shape == (2, 2)
