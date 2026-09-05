"""UrbanMARL Vectorized Ground User (UE) Mobility Engine.

Provides GPU-accelerated, pure PyTorch mobility models for dynamic ground users:

1. Street-Constrained Manhattan Mobility: Constrains user motion to street corridors
   between ITU-R P.1410 building footprints.
2. Random Waypoint (RWP) & Gauss-Markov Mobility: Smooth continuous velocity and direction
   updates with boundary reflection.
3. Dynamic Hotspot & Crowd Migration Mobility: Models temporal crowd clustering around
   spatially drifting hotspot centers.
"""

from typing import Any, Optional, Tuple, Union

import torch


class VectorizedUserMobility:
    """Tensor-accelerated user mobility simulation engine.

    Attributes:
        volume_size (Tuple[float, float, float]): Urban dimensions (sim_x, sim_y, sim_z).
        device (torch.device): PyTorch compute device.
        model_type (str): Mobility model identifier ('manhattan', 'rwp', 'gauss_markov', 'hotspot').
        speed_min (float): Minimum user speed in m/s. Defaults to 0.5 (pedestrian).
        speed_max (float): Maximum user speed in m/s. Defaults to 3.0 (brisk walk/runner).
    """

    def __init__(
        self,
        volume_size: Tuple[float, float, float] = (500.0, 500.0, 200.0),
        model_type: str = "manhattan",
        speed_min: float = 0.5,
        speed_max: float = 3.0,
        alpha_memory: float = 0.75,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initializes the VectorizedUserMobility engine.

        Args:
            volume_size (Tuple[float, float, float]): Urban simulation boundary dimensions.
            model_type (str): Mobility algorithm name ('manhattan', 'rwp', 'gauss_markov', 'hotspot').
            speed_min (float): Minimum speed bound in m/s. Defaults to 0.5.
            speed_max (float): Maximum speed bound in m/s. Defaults to 3.0.
            alpha_memory (float): Gauss-Markov memory coefficient in [0, 1]. Defaults to 0.75.
            device (Union[torch.device, str]): PyTorch compute device.
        """
        self.volume_size = volume_size
        self.model_type = model_type.lower()
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.alpha = alpha_memory
        self.device = torch.device(device)

        self.x_min, self.x_max = -volume_size[0] / 2, volume_size[0] / 2
        self.y_min, self.y_max = -volume_size[1] / 2, volume_size[1] / 2

        # Hotspot centers: shape (B, K, 2) initialized during reset if used
        self.hotspot_centers: Optional[torch.Tensor] = None
        self.hotspot_velocities: Optional[torch.Tensor] = None

    def initialize_velocities(self, batch_size: int, n_ues: int) -> torch.Tensor:
        """Initializes random 2D ground velocities for UEs.

        Args:
            batch_size (int): Number of parallel environments.
            n_ues (int): Number of ground UEs per environment.

        Returns:
            torch.Tensor: Initial velocity tensor of shape (B, M, 3) with vz=0.
        """
        speeds = (
            torch.rand((batch_size, n_ues, 1), device=self.device)
            * (self.speed_max - self.speed_min)
            + self.speed_min
        )

        if self.model_type == "manhattan":
            # Discrete cardinal directions: East, North, West, South
            angles = torch.randint(
                0, 4, (batch_size, n_ues, 1), device=self.device
            ).float() * (torch.pi / 2.0)
        else:
            angles = (
                torch.rand((batch_size, n_ues, 1), device=self.device) * 2.0 * torch.pi
            )

        vx = speeds * torch.cos(angles)
        vy = speeds * torch.sin(angles)
        vz = torch.zeros_like(vx)

        velocities = torch.cat([vx, vy, vz], dim=-1)
        return velocities

    def initialize_hotspots(self, batch_size: int, num_hotspots: int = 2) -> None:
        """Initializes spatial hotspot cluster centers and drift velocities.

        Args:
            batch_size (int): Parallel environment batch count.
            num_hotspots (int): Number of simultaneous crowd hotspots. Defaults to 2.
        """
        cx = (
            torch.rand((batch_size, num_hotspots, 1), device=self.device)
            * (self.x_max - self.x_min)
            * 0.6
            + self.x_min * 0.6
        )
        cy = (
            torch.rand((batch_size, num_hotspots, 1), device=self.device)
            * (self.y_max - self.y_min)
            * 0.6
            + self.y_min * 0.6
        )
        self.hotspot_centers = torch.cat([cx, cy], dim=-1)

        # Drifting velocities: 0.2 to 1.0 m/s
        h_speeds = (
            torch.rand((batch_size, num_hotspots, 1), device=self.device) * 0.8 + 0.2
        )
        h_angles = (
            torch.rand((batch_size, num_hotspots, 1), device=self.device)
            * 2.0
            * torch.pi
        )
        self.hotspot_velocities = torch.cat(
            [h_speeds * torch.cos(h_angles), h_speeds * torch.sin(h_angles)], dim=-1
        )

    def step(
        self,
        ue_pos: torch.Tensor,
        ue_vel: torch.Tensor,
        dt: float = 1.0,
        urban_map: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Steps ground user positions and updates velocities.

        Args:
            ue_pos (torch.Tensor): Current UE positions of shape (B, M, 3).
            ue_vel (torch.Tensor): Current UE velocities of shape (B, M, 3).
            dt (float): Simulation time step duration in seconds. Defaults to 1.0.
            urban_map: Optional VectorizedUrbanMap instance for street boundary collision.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated (new_positions, new_velocities).
        """
        b, m, _ = ue_pos.shape

        if self.model_type == "gauss_markov":
            # Autoregressive velocity update
            noise = torch.randn_like(ue_vel[..., :2]) * 0.5
            new_v_xy = self.alpha * ue_vel[..., :2] + (1.0 - self.alpha) * noise
            speeds = torch.norm(new_v_xy, dim=-1, keepdim=True).clamp(
                self.speed_min, self.speed_max
            )
            angles = torch.atan2(new_v_xy[..., 1:], new_v_xy[..., :1])
            ue_vel = torch.cat(
                [
                    speeds * torch.cos(angles),
                    speeds * torch.sin(angles),
                    torch.zeros((b, m, 1), device=self.device),
                ],
                dim=-1,
            )

        elif self.model_type == "hotspot" and self.hotspot_centers is not None:
            # Update drifting hotspots
            self.hotspot_centers += self.hotspot_velocities * dt
            # Bounce hotspots at boundaries
            x_out = (self.hotspot_centers[..., 0] < self.x_min * 0.8) | (
                self.hotspot_centers[..., 0] > self.x_max * 0.8
            )
            y_out = (self.hotspot_centers[..., 1] < self.y_min * 0.8) | (
                self.hotspot_centers[..., 1] > self.y_max * 0.8
            )
            self.hotspot_velocities[..., 0] = torch.where(
                x_out, -self.hotspot_velocities[..., 0], self.hotspot_velocities[..., 0]
            )
            self.hotspot_velocities[..., 1] = torch.where(
                y_out, -self.hotspot_velocities[..., 1], self.hotspot_velocities[..., 1]
            )

            # Attractive drift towards nearest hotspot
            # Centers: (B, K, 2), Pos: (B, M, 2)
            disp = self.hotspot_centers.unsqueeze(1) - ue_pos[..., :2].unsqueeze(2)
            dists = torch.norm(disp, dim=-1)  # (B, M, K)
            nearest_k = torch.argmin(dists, dim=-1)  # (B, M)

            b_idx = torch.arange(b, device=self.device).unsqueeze(1).expand(b, m)
            target_centers = self.hotspot_centers[b_idx, nearest_k]  # (B, M, 2)

            vec_to_hotspot = target_centers - ue_pos[..., :2]
            angles_to_hotspot = torch.atan2(
                vec_to_hotspot[..., 1:], vec_to_hotspot[..., :1]
            )

            # Blend random walk with attraction to hotspot (60% pull)
            speeds = torch.norm(ue_vel[..., :2], dim=-1, keepdim=True).clamp(
                self.speed_min, self.speed_max
            )
            cur_angles = torch.atan2(ue_vel[..., 1:2], ue_vel[..., :1])
            new_angles = 0.4 * cur_angles + 0.6 * angles_to_hotspot

            ue_vel = torch.cat(
                [
                    speeds * torch.cos(new_angles),
                    speeds * torch.sin(new_angles),
                    torch.zeros((b, m, 1), device=self.device),
                ],
                dim=-1,
            )

        elif self.model_type == "manhattan":
            # Probabilistic turns at grid intersections (15% chance to turn 90 deg)
            turn_prob = torch.rand((b, m, 1), device=self.device) < 0.15
            turn_dir = (
                torch.randint(0, 2, (b, m, 1), device=self.device).float() * 2.0 - 1.0
            ) * (torch.pi / 2.0)
            cur_angles = torch.atan2(ue_vel[..., 1:2], ue_vel[..., :1])
            new_angles = torch.where(turn_prob, cur_angles + turn_dir, cur_angles)
            speeds = torch.norm(ue_vel[..., :2], dim=-1, keepdim=True).clamp(
                self.speed_min, self.speed_max
            )
            ue_vel = torch.cat(
                [
                    speeds * torch.cos(new_angles),
                    speeds * torch.sin(new_angles),
                    torch.zeros((b, m, 1), device=self.device),
                ],
                dim=-1,
            )

        # Tentative position update
        new_pos = ue_pos + ue_vel * dt
        new_pos[..., 2] = 1.5  # Ground plane constraint

        # Street constraint: if urban heightmap indicates building obstacle, bounce back!
        if urban_map is not None and hasattr(urban_map, "height_maps"):
            # Sample height at new position
            gx = torch.clamp(
                (new_pos[..., 0] - self.x_min).long(), 0, urban_map.sim_x - 1
            )
            gy = torch.clamp(
                (new_pos[..., 1] - self.y_min).long(), 0, urban_map.sim_y - 1
            )
            b_exp = torch.arange(b, device=self.device).unsqueeze(1).expand(b, m)
            sampled_h = urban_map.height_maps[b_exp, gx, gy]

            # Obstacle hit: if height > 0, reverse velocity and cancel movement
            hit_building = sampled_h > 0.0
            new_pos[..., 0] = torch.where(hit_building, ue_pos[..., 0], new_pos[..., 0])
            new_pos[..., 1] = torch.where(hit_building, ue_pos[..., 1], new_pos[..., 1])
            ue_vel[..., :2] = torch.where(
                hit_building.unsqueeze(-1), -ue_vel[..., :2], ue_vel[..., :2]
            )

        # Volume boundary bounce
        bounce_x = (new_pos[..., 0] < self.x_min) | (new_pos[..., 0] > self.x_max)
        bounce_y = (new_pos[..., 1] < self.y_min) | (new_pos[..., 1] > self.y_max)

        new_pos[..., 0] = torch.clamp(new_pos[..., 0], self.x_min, self.x_max)
        new_pos[..., 1] = torch.clamp(new_pos[..., 1], self.y_min, self.y_max)

        ue_vel[..., 0] = torch.where(bounce_x, -ue_vel[..., 0], ue_vel[..., 0])
        ue_vel[..., 1] = torch.where(bounce_y, -ue_vel[..., 1], ue_vel[..., 1])

        return new_pos, ue_vel
