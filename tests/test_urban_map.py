import pytest
import torch
from urbanmarl.models.urban_map import VectorizedUrbanMap


@pytest.fixture
def urban_map():
    """Provides a VectorizedUrbanMap instance for testing procedural map logic."""
    batch_size = 2
    volume_size = (200, 200, 100)
    device = torch.device("cpu")
    umap = VectorizedUrbanMap(batch_size=batch_size, volume_size=volume_size, device=device)
    umap.generate_batch_maps()
    return umap


def test_urban_map_init():
    """
    Verifies VectorizedUrbanMap initialization, coordinate bounds, and height maps allocation.
    """
    batch_size = 3
    volume_size = (300, 300, 150)
    device = torch.device("cpu")
    umap = VectorizedUrbanMap(batch_size=batch_size, volume_size=volume_size, device=device)

    assert umap.batch_size == 3
    assert umap.volume_size == (300, 300, 150)
    assert umap.x_min == -150 and umap.x_max == 150
    assert umap.y_min == -150 and umap.y_max == 150
    assert umap.sim_x == 300 and umap.sim_y == 300
    assert umap.height_maps.shape == (3, 300, 300)
    assert (umap.height_maps == 0.0).all()


def test_generate_batch_maps_procedural(urban_map):
    """
    Tests procedural map generation using generate_batch_maps, verifying height maps values
    are non-negative and info metadata tensors are initialized with (B, 7) properties.
    """
    assert (urban_map.height_maps >= 0.0).all()
    assert urban_map.info is not None
    assert urban_map.info.shape == (urban_map.batch_size, 7)
    # Check info label count matches info second dimension
    assert len(urban_map.get_info_labels()) == 7


def test_check_los_batch_clear_vs_blocked(urban_map):
    """
    Tests check_los_batch line-of-sight status detection.
    High altitude points above building tops should maintain LoS clearance (True).
    """
    batch_size = urban_map.batch_size
    num_agents = 2
    num_targets = 2

    # High altitude positions above any procedural building height (z = 200.0)
    p1 = torch.tensor([[[0.0, 0.0, 200.0], [50.0, 50.0, 200.0]]]).expand(batch_size, num_agents, 3)
    p2 = torch.tensor([[[10.0, 10.0, 200.0], [60.0, 60.0, 200.0]]]).expand(batch_size, num_targets, 3)

    los_matrix = urban_map.check_los_batch(p1, p2, n_steps=10)

    assert isinstance(los_matrix, torch.Tensor)
    assert los_matrix.shape == (batch_size, num_agents, num_targets)
    assert los_matrix.dtype == torch.bool
    # At high altitude, line of sight should be completely unblocked
    assert los_matrix.all()


def test_check_collision_batch(urban_map):
    """
    Tests check_collision_batch verifying collision tensor returns boolean masks
    of shape (B, N, 1).
    """
    batch_size = urban_map.batch_size
    num_agents = 3

    # High altitude motion (no collisions expected)
    p1 = torch.rand(batch_size, num_agents, 3) * 50.0
    p1[..., 2] = 200.0
    p2 = p1 + 5.0

    collisions = urban_map.check_collision_batch(p1, p2, n_steps=10)

    assert isinstance(collisions, torch.Tensor)
    assert collisions.shape == (batch_size, num_agents, 1)
    assert collisions.dtype == torch.bool
    assert not collisions.any()


def test_position_normalization_roundtrip(urban_map):
    """
    Tests position normalization (norm_pos) and denormalization (denorm_pos),
    verifying exact spatial recovery.
    """
    positions = torch.tensor([[[0.0, 0.0, 50.0], [50.0, -50.0, 25.0]]]).expand(urban_map.batch_size, 2, 3)

    norm_p = urban_map.norm_pos(positions)
    assert (norm_p[..., 0] >= 0.0).all() and (norm_p[..., 0] <= 1.0).all()
    assert (norm_p[..., 1] >= 0.0).all() and (norm_p[..., 1] <= 1.0).all()

    denorm_p = urban_map.denorm_pos(norm_p)
    assert torch.allclose(positions, denorm_p, atol=1e-4)


def test_outdoor_position_generation(urban_map):
    """
    Verifies that gen_outdoor_pos generates positions that lie strictly on outdoor terrain
    (where height map building height is zero).
    """
    num_pos = 5
    positions = urban_map.gen_outdoor_pos(num_pos=num_pos, min_z=0.0, max_z=10.0)

    assert positions.shape == (urban_map.batch_size, num_pos, 3)
    grid_coords = urban_map.pos_to_grid(positions)

    for b in range(urban_map.batch_size):
        for i in range(num_pos):
            gx = grid_coords[b, i, 0].item()
            gy = grid_coords[b, i, 1].item()
            assert urban_map.height_maps[b, gx, gy].item() == 0.0


def test_pos_to_grid_clamping(urban_map):
    """
    Tests pos_to_grid physical position to grid index conversion with boundary clamping.
    """
    # Out of bounds positions
    positions = torch.tensor([[[-1000.0, -1000.0, 0.0], [1000.0, 1000.0, 0.0]]]).expand(urban_map.batch_size, 2, 3)
    grid = urban_map.pos_to_grid(positions)

    assert (grid[..., 0] >= 0).all() and (grid[..., 0] < urban_map.sim_x).all()
    assert (grid[..., 1] >= 0).all() and (grid[..., 1] < urban_map.sim_y).all()


def test_map_reset_and_reset_at(urban_map):
    """
    Tests reset() and reset_at() methods for single or batch environment regeneration.
    """
    initial_map = urban_map.height_maps.clone()

    # Reset environment at index 0
    urban_map.reset_at(env_idx=0)
    assert urban_map.height_maps.shape == initial_map.shape

    # Reset all environments
    urban_map.reset()
    assert urban_map.has_envs is True
