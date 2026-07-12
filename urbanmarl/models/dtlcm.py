import torch

def compute_batched_dtlcm_assignment(uav_pos: torch.Tensor, ue_pos: torch.Tensor, 
                                     uav_caps: torch.Tensor, uav_loads: torch.Tensor,
                                     task_workloads: torch.Tensor, distance_weight: float = 0.7) -> torch.Tensor:
    """
    Executes a continuous tensor approximation of the heuristic DTLCM assignment step.
    uav_pos: (B, N, 3), ue_pos: (B, M, 3)
    uav_caps: (B, N), uav_loads: (B, N), task_workloads: (B, M)
    Returns: assignment matrix (B, M) representing assigned UAV index mapping.
    """
    B, N, _ = uav_pos.shape
    _, M, _ = ue_pos.shape
    
    # 1. Normalized Spatial Distance Matrix
    diff = uav_pos.unsqueeze(2) - ue_pos.unsqueeze(1) # (B, N, M, 3)
    dist = torch.norm(diff[..., :2], dim=-1) # Focus purely on horizontal layout properties
    max_dist = 1000.0
    norm_dist = torch.clamp(dist / max_dist, 0.0, 1.0) # (B, N, M)
    dist_score = 1.0 - norm_dist
    
    # 2. Computational Capability Adaptability Profile
    # Expand properties ensuring full multi-dimensional calculation capacity
    cap_exp = uav_caps.unsqueeze(2).expand(B, N, M)
    workload_exp = task_workloads.unsqueeze(1).expand(B, N, M)
    cap_score = torch.clamp(cap_exp / (workload_exp + 1e-6), 0.0, 1.0)
    
    # 3. Dynamic Runtime Balancing Penalties
    load_penalty = (uav_loads / (uav_caps + 1e-6)).unsqueeze(2).expand(B, N, M)
    
    # Complete multi-criteria attention tensor construction
    total_scores = (distance_weight * dist_score) + ((1.0 - distance_weight) * cap_score) - (0.1 * load_penalty)
    
    # Extract targeted optimal processing execution parameters using argmax assignments
    assigned_uavs = torch.argmax(total_scores, dim=1) # Shape (B, M)
    return assigned_uavs 
