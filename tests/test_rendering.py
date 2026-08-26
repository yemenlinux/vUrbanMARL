import pytest
import numpy as np
import torch
from urbanmarl.envs.rendering import Urban3DRenderer, UrbanRenderConfig


def test_urban_render_config_defaults():
    """
    Tests that UrbanRenderConfig initializes with valid aesthetic defaults.
    """
    config = UrbanRenderConfig()
    assert config.dpi == 100
    assert config.building_alpha == 0.2
    assert config.camera_elev == 45.0
    assert config.camera_azim == -120.0


def test_urban_3d_renderer_init_and_close():
    """
    Verifies Urban3DRenderer lifecycle management (initialization and cleanup).
    """
    renderer = Urban3DRenderer()
    assert renderer.config is not None
    assert renderer._fig is None
    renderer.close()
    assert renderer._fig is None


def test_urban_3d_renderer_render_rgb_array():
    """
    Tests rendering a simulated environment state into an RGB numpy array (mode='rgb_array')
    using mock positions for UAVs, UEs, and building structures.
    """
    renderer = Urban3DRenderer()

    mock_state = {
        "volume_size": [300, 300, 100],
        "current_frame": 0,
        "uav_positions": np.array([[0.0, 0.0, 50.0], [50.0, 50.0, 60.0]]),
        "ue_positions": np.array([[-50.0, -50.0, 0.0], [20.0, 30.0, 0.0]]),
        "buildings": [
            {"position": [10, 10, 0], "size": [20, 20, 30]}
        ],
        "links": [
            {"source": [0.0, 0.0, 50.0], "target": [-50.0, -50.0, 0.0], "los": True}
        ],
        "title": "Unit Test Render View",
    }

    try:
        rgb_array = renderer.render(mock_state, mode="rgb_array")
        assert isinstance(rgb_array, np.ndarray)
        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3  # RGB channels
        assert rgb_array.dtype == np.uint8
    finally:
        renderer.close()
