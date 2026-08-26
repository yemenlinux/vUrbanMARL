"""UrbanMARL Dynamic Distance-Task-Load Capacity Matching (DTLCM).

Provides continuous tensor heuristic matching for UAV-MEC task offloading assignments.
"""

import torch


def compute_batched_dtlcm_assignment(
    uav_pos: torch.Tensor,
    ue_pos: torch.Tensor,
    uav_caps: torch.Tensor,
    uav_loads: torch.Tensor,
    task_workloads: torch.Tensor,
    distance_weight: float = 0.7,
) -> torch.Tensor:
    """Executes batched heuristic DTLCM assignment for task offloading.

    Args:
        uav_pos (torch.Tensor): UAV 3D positions of shape (B, N, 3).
        ue_pos (torch.Tensor): UE 3D positions of shape (B, M, 3).
        uav_caps (torch.Tensor): UAV compute capacities of shape (B, N).
        uav_loads (torch.Tensor): Current UAV load levels of shape (B, N).
        task_workloads (torch.Tensor): Offloaded task workloads of shape (B, M).
        distance_weight (float): Weight assigned to spatial distance score (default: 0.7).

    Returns:
        torch.Tensor: Long tensor of assigned UAV indices per UE of shape (B, M).
    """
    B, N, _ = uav_pos.shape
    _, M, _ = ue_pos.shape

    diff = uav_pos.unsqueeze(2) - ue_pos.unsqueeze(1)
    dist = torch.norm(diff[..., :2], dim=-1)
    max_dist = 1000.0
    norm_dist = torch.clamp(dist / max_dist, 0.0, 1.0)
    dist_score = 1.0 - norm_dist

    cap_exp = uav_caps.unsqueeze(2).expand(B, N, M)
    workload_exp = task_workloads.unsqueeze(1).expand(B, N, M)
    cap_score = torch.clamp(cap_exp / (workload_exp + 1e-6), 0.0, 1.0)

    load_penalty = (uav_loads / (uav_caps + 1e-6)).unsqueeze(2).expand(B, N, M)

    total_scores = (
        (distance_weight * dist_score)
        + ((1.0 - distance_weight) * cap_score)
        - (0.1 * load_penalty)
    )

    assigned_uavs = torch.argmax(total_scores, dim=1)
    return assigned_uavs
