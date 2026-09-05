"""UrbanMARL Vectorized LiDAR Rangefinder Sensor Model.

Provides GPU-accelerated, pure PyTorch ray-marching LiDAR simulations for UAVs:
- Casts radial horizontal rays in 360-degree azimuth around each UAV.
- Samples 3D building heightmaps along ray trajectories.
- Detects distance to nearest obstacle building face.
- Outputs normalized proximity vectors for collision-free decentralized POMDP navigation.
"""

from typing import Any, Union

import torch


class VectorizedLiDAR:
    """Vectorized 3D LiDAR proximity rangefinder.

    Simulates radial rangefinder beams around each UAV to detect building facades.
    """

    def __init__(
        self,
        num_beams: int = 8,
        max_range: float = 60.0,
        n_steps: int = 15,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initializes the VectorizedLiDAR sensor.

        Args:
            num_beams (int): Number of radial azimuth beams (e.g., 8 or 16). Defaults to 8.
            max_range (float): Maximum detection range in meters. Defaults to 60.0.
            n_steps (int): Number of ray-marching sampling intervals along each beam. Defaults to 15.
            device (Union[torch.device, str]): PyTorch compute device.
        """
        self.num_beams = num_beams
        self.max_range = max_range
        self.n_steps = n_steps
        self.device = torch.device(device)

        # Precompute azimuth ray angles in [0, 2*pi)
        angles = torch.linspace(0, 2 * torch.pi, num_beams + 1, device=self.device)[:-1]
        self.dir_x = torch.cos(angles)  # (num_beams,)
        self.dir_y = torch.sin(angles)  # (num_beams,)

        # Relative step fractions along ray: shape (n_steps,)
        self.step_fractions = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=device)

    def scan(
        self,
        uav_positions: torch.Tensor,
        urban_map: Any,
    ) -> torch.Tensor:
        """Performs ray-marching LiDAR scans for all UAVs in parallel.

        Args:
            uav_positions (torch.Tensor): UAV 3D coordinates of shape (B, N, 3).
            urban_map: VectorizedUrbanMap instance containing 3D heightmaps.

        Returns:
            torch.Tensor: Normalized distance readings of shape (B, N, num_beams).
                Values are in [0.0, 1.0], where 1.0 represents clear/max range,
                and values approaching 0.0 indicate proximity to an obstacle.
        """
        b, n, _ = uav_positions.shape
        device = uav_positions.device

        # If no heightmap available, return full clearance (1.0)
        if not hasattr(urban_map, "height_maps"):
            return torch.ones((b, n, self.num_beams), device=device)

        sim_x = urban_map.sim_x
        sim_y = urban_map.sim_y
        x_min = -urban_map.volume_size[0] / 2
        y_min = -urban_map.volume_size[1] / 2

        # Beam unit direction vectors: shape (1, 1, num_beams, 2)
        dir_xy = torch.stack(
            [self.dir_x.to(device), self.dir_y.to(device)], dim=-1
        ).view(1, 1, self.num_beams, 2)

        # Distances along ray: shape (1, 1, 1, n_steps, 1)
        step_dists = (self.step_fractions.to(device) * self.max_range).view(
            1, 1, 1, self.n_steps, 1
        )

        # Ray sample points (x, y): shape (B, N, num_beams, n_steps, 2)
        # UAV xy: shape (B, N, 1, 1, 2)
        uav_xy = uav_positions[..., :2].unsqueeze(2).unsqueeze(3)
        ray_xy = uav_xy + dir_xy.unsqueeze(3) * step_dists

        # Clamp grid coordinates to map boundaries
        grid_x = torch.clamp((ray_xy[..., 0] - x_min).long(), 0, sim_x - 1)
        grid_y = torch.clamp((ray_xy[..., 1] - y_min).long(), 0, sim_y - 1)

        # Broadcast batch index
        b_idx = (
            torch.arange(b, device=device)
            .view(b, 1, 1, 1)
            .expand(b, n, self.num_beams, self.n_steps)
        )

        # Sample height at each step: shape (B, N, num_beams, n_steps)
        sampled_heights = urban_map.height_maps[b_idx, grid_x, grid_y]

        # Obstacle collision: if building height >= UAV z-coordinate
        uav_z = uav_positions[..., 2:].unsqueeze(2)  # (B, N, 1, 1)
        is_hit = sampled_heights >= uav_z  # (B, N, num_beams, n_steps)

        # Find first collision step index along each ray
        # If no hit, default to n_steps
        hit_float = is_hit.float()
        has_hit = hit_float.any(dim=-1)  # (B, N, num_beams)

        # Argmax on hit_float gives index of first occurrence of 1
        first_hit_step = torch.argmax(hit_float, dim=-1)  # (B, N, num_beams)

        # Compute normalized range: (step_index + 1) / n_steps
        range_frac = (first_hit_step.float() + 0.5) / float(self.n_steps)
        normalized_ranges = torch.where(
            has_hit, range_frac, torch.ones_like(range_frac)
        )

        return normalized_ranges
