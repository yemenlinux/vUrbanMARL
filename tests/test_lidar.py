"""Unit tests for VectorizedLiDAR and UAV_LIDAR_NAVIGATION scenario."""

import pytest
import torch
from urbanmarl.envs.base_env import UrbanEnv
from urbanmarl.models.lidar import VectorizedLiDAR


@pytest.fixture
def lidar_config():
    """Environment configuration for lidar navigation scenario."""
    return {
        "num_uavs": 3,
        "num_ues": 5,
        "volume_size": (400, 400, 150),
        "max_steps": 15,
        "max_horizontal_speed": 40.0,
        "max_vertical_speed": 10.0,
        "num_lidar_beams": 8,
        "lidar_max_range": 50.0,
    }


def test_vectorized_lidar_scan():
    """Tests ray-marching LiDAR scans with mock UAV positions."""
    lidar = VectorizedLiDAR(num_beams=8, max_range=50.0, n_steps=10)
    uav_pos = torch.tensor([[[0.0, 0.0, 50.0], [10.0, 10.0, 30.0]]])

    # Class with mock heightmaps
    class MockMap:
        sim_x = 400
        sim_y = 400
        volume_size = (400, 400, 150)
        height_maps = torch.zeros((1, 400, 400))

    mock_map = MockMap()
    # Place a building near the second UAV
    mock_map.height_maps[0, 215:225, 205:215] = 60.0

    ranges = lidar.scan(uav_pos, mock_map)
    assert ranges.shape == (1, 2, 8)
    assert torch.all(ranges >= 0.0) and torch.all(ranges <= 1.0)


def test_uav_lidar_navigation_integration(lidar_config):
    """Verifies that uav_lidar_navigation runs with UrbanEnv and updates observations."""
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=101,
        device=torch.device("cpu"),
        scenario="uav_lidar_navigation",
        **lidar_config,
    )

    td = env.reset()
    assert td is not None

    obs = env._get_obs()
    # 4 pos/battery + 3 target_disp + 8 lidar = 15
    assert obs.shape[-1] == 15

    # Step environment
    action = env.action_spec.sample()
    stepped = env.step(td.update(action))
    assert stepped is not None
    assert "uav_lidar_ranges" in dir(env)
    assert env.uav_lidar_ranges.shape == (2, 3, 8)
