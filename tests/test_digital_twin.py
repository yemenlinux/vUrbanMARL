"""Unit tests for Network Digital Twin (NDT) and Radio Environment Map (REM)."""

import json

import numpy as np
import pytest
import torch
from urbanmarl.envs.base_env import UrbanEnv
from urbanmarl.envs.rendering import Urban3DRenderer, UrbanRenderConfig
from urbanmarl.models.digital_twin import (
    NDTTelemetryFrame,
    NetworkDigitalTwin,
    RadioEnvironmentMap,
)


@pytest.fixture
def env_config():
    """Minimal environment configuration for digital twin tests."""
    return {
        "num_uavs": 3,
        "num_ues": 8,
        "volume_size": (400, 400, 150),
        "max_steps": 20,
        "max_horizontal_speed": 40.0,
        "max_vertical_speed": 10.0,
    }


def test_radio_environment_map_generation():
    """Tests that RadioEnvironmentMap generates spatial SINR grids and coverage ratios."""
    rem_gen = RadioEnvironmentMap(
        volume_size=(400.0, 400.0, 150.0),
        grid_resolution=25,
        frequency_ghz=29.0,
        bandwidth_hz=10e6,
        sinr_threshold_db=0.0,
    )

    uav_positions = torch.tensor(
        [
            [-50.0, -50.0, 60.0],
            [50.0, 50.0, 70.0],
            [0.0, 0.0, 80.0],
        ]
    )

    rem = rem_gen.compute_rem(uav_positions, tx_power=2.0)

    assert "sinr_grid" in rem
    assert "coverage_mask" in rem
    assert "coverage_ratio" in rem
    assert "mean_sinr_db" in rem

    assert rem["sinr_grid"].shape == (25, 25)
    assert rem["coverage_mask"].shape == (25, 25)
    assert 0.0 <= rem["coverage_ratio"] <= 1.0
    assert not torch.isnan(rem["sinr_grid"]).any()


def test_network_digital_twin_capture(env_config):
    """Verifies that NetworkDigitalTwin captures synchronized 4-layer telemetry frames."""
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=123,
        device=torch.device("cpu"),
        scenario="uavmec_offloading",
        **env_config,
    )
    env.reset()

    ndt = NetworkDigitalTwin(
        volume_size=(400.0, 400.0, 150.0),
        rem_resolution=20,
    )

    frame = ndt.capture_frame(env, env_idx=0, step=0, compute_rem_heatmap=True)

    assert isinstance(frame, NDTTelemetryFrame)
    assert frame.frame_id == 0
    assert frame.timestamp_s == 0.0

    # Geospatial layer
    assert frame.geospatial.volume_size == [400, 400, 150]
    assert frame.geospatial.alpha > 0.0

    # REM layer
    assert 0.0 <= frame.rem.coverage_ratio <= 1.0
    assert frame.rem.los_link_count >= 0

    # Compute layer
    assert frame.compute.mean_utilization >= 0.0
    assert frame.compute.total_completed_tasks >= 0

    # Mobility layer
    assert frame.mobility.n_uavs == 3
    assert frame.mobility.n_ues == 8
    assert frame.mobility.mean_battery_pct == 100.0

    # Telemetry and links
    assert len(frame.uav_telemetry) == 3
    assert len(frame.active_links) == 3 * 8


def test_ndt_json_serialization(env_config):
    """Tests serialization of NDT telemetry frames to JSON string and file export."""
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=42,
        device=torch.device("cpu"),
        scenario="uavmec_offloading",
        **env_config,
    )
    env.reset()

    ndt = NetworkDigitalTwin(volume_size=(400.0, 400.0, 150.0))
    frame = ndt.capture_frame(env, env_idx=0, step=1)

    json_str = ndt.export_telemetry_json(frame)
    data = json.loads(json_str)

    assert data["frame_id"] == 1
    assert "geospatial" in data
    assert "rem" in data
    assert "compute" in data
    assert "mobility" in data
    assert "uav_telemetry" in data
    assert len(data["uav_telemetry"]) == 3


def test_renderer_telemetry_hud(env_config):
    """Tests that Urban3DRenderer renders telemetry HUD and REM contours without errors."""
    config = UrbanRenderConfig(
        show_telemetry_hud=True,
        show_rem_contours=True,
    )
    renderer = Urban3DRenderer(config=config)

    state = {
        "volume_size": [400, 400, 150],
        "uav_positions": np.array([[0, 0, 50], [50, 50, 60]]),
        "ue_positions": np.array([[10, 10, 0], [20, 20, 0]]),
        "telemetry": {
            "completed_tasks": 15,
            "dropped_tasks": 1,
            "utilization": 0.42,
            "system_time": 0.018,
            "coverage_ratio": 0.85,
            "sojourn_time": 4.5,
        },
        "rem_sinr": np.random.uniform(-5.0, 25.0, (20, 20)),
    }

    img = renderer.render(state, mode="rgb_array")
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3
    renderer.close()
