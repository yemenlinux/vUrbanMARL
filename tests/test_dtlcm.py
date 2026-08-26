import pytest
import torch
from urbanmarl.models.dtlcm import compute_batched_dtlcm_assignment


def test_dtlcm_output_shape_and_range():
    """
    Verifies that compute_batched_dtlcm_assignment outputs a tensor of shape (B, M)
    containing long integer UAV indices in the range [0, N-1].
    """
    batch_size = 2
    num_uavs = 4
    num_ues = 10

    uav_pos = torch.rand(batch_size, num_uavs, 3) * 500.0
    ue_pos = torch.rand(batch_size, num_ues, 3) * 500.0
    uav_caps = torch.full((batch_size, num_uavs), 10.0)
    uav_loads = torch.zeros(batch_size, num_uavs)
    task_workloads = torch.full((batch_size, num_ues), 1.0)

    assignments = compute_batched_dtlcm_assignment(
        uav_pos, ue_pos, uav_caps, uav_loads, task_workloads, distance_weight=0.7
    )

    assert isinstance(assignments, torch.Tensor)
    assert assignments.shape == (batch_size, num_ues)
    assert assignments.dtype == torch.int64 or assignments.dtype == torch.long
    assert (assignments >= 0).all()
    assert (assignments < num_uavs).all()


def test_dtlcm_spatial_distance_priority():
    """
    Tests that when distance_weight is 1.0, the assignment selects the UAV with the smallest
    horizontal distance to each UE.
    """
    batch_size = 1
    num_uavs = 3
    num_ues = 1

    # Place UAV 0 at (0, 0), UAV 1 at (100, 100), UAV 2 at (500, 500)
    uav_pos = torch.tensor([[[0.0, 0.0, 50.0],
                             [100.0, 100.0, 50.0],
                             [500.0, 500.0, 50.0]]])
    # Place UE at (10.0, 10.0) - closest to UAV 0
    ue_pos = torch.tensor([[[10.0, 10.0, 0.0]]])

    uav_caps = torch.tensor([[10.0, 10.0, 10.0]])
    uav_loads = torch.tensor([[0.0, 0.0, 0.0]])
    task_workloads = torch.tensor([[1.0]])

    assignments = compute_batched_dtlcm_assignment(
        uav_pos, ue_pos, uav_caps, uav_loads, task_workloads, distance_weight=1.0
    )

    assert assignments[0, 0].item() == 0


def test_dtlcm_capability_priority():
    """
    Tests that when distance_weight is 0.0, assignment prioritizes UAVs with higher
    computational capacity relative to task workloads.
    """
    batch_size = 1
    num_uavs = 2
    num_ues = 1

    # UAV 0 is close, but has low capacity (1.0). UAV 1 is far, but has high capacity (100.0).
    uav_pos = torch.tensor([[[0.0, 0.0, 50.0],
                             [500.0, 500.0, 50.0]]])
    ue_pos = torch.tensor([[[0.0, 0.0, 0.0]]])

    uav_caps = torch.tensor([[1.0, 100.0]])
    uav_loads = torch.tensor([[0.0, 0.0]])
    task_workloads = torch.tensor([[50.0]])

    assignments = compute_batched_dtlcm_assignment(
        uav_pos, ue_pos, uav_caps, uav_loads, task_workloads, distance_weight=0.0
    )

    assert assignments[0, 0].item() == 1


def test_dtlcm_single_uav_edge_case():
    """
    Tests the single UAV edge case (N=1) where all UEs must be assigned to index 0.
    """
    batch_size = 2
    num_uavs = 1
    num_ues = 5

    uav_pos = torch.rand(batch_size, num_uavs, 3) * 500.0
    ue_pos = torch.rand(batch_size, num_ues, 3) * 500.0
    uav_caps = torch.full((batch_size, num_uavs), 10.0)
    uav_loads = torch.zeros(batch_size, num_uavs)
    task_workloads = torch.full((batch_size, num_ues), 2.0)

    assignments = compute_batched_dtlcm_assignment(
        uav_pos, ue_pos, uav_caps, uav_loads, task_workloads
    )

    assert (assignments == 0).all()
