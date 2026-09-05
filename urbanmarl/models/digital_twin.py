"""UrbanMARL Network Digital Twin (NDT) Architecture & Telemetry Model.

Provides a unified, 4-layer synchronized digital twin model for 6G multi-UAV
assisted Mobile Edge Computing (MEC) networks:
1. Geospatial & Structural Twin: 3D urban terrain, ITU-R P.1410 parameters, buildings.
2. Radio Environment Map (REM) Twin: Spatial SINR coverage, LoS/NLoS paths, channel capacity.
3. Computational & Queuing Twin: M/M/c server utilization, queue lengths, task latency.
4. Aerial Mobility & Energy Twin: UAV kinematics, battery state, aerodynamic energy, sojourn time.

Includes vectorized Radio Environment Map (REM) generators, live telemetry extraction,
and JSON-compliant serialization for 3D digital twin visualization engines.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class GeospatialTwinState:
    """Geospatial and structural digital twin layer state."""

    volume_size: List[float]
    alpha: float
    beta: float
    gamma: float
    e_param: float
    n_buildings: int
    building_width: float
    street_width: float
    building_count: int


@dataclass
class REMTwinState:
    """Radio Environment Map (REM) and wireless propagation digital twin state."""

    frequency_ghz: float
    bandwidth_hz: float
    coverage_ratio: float
    mean_sinr_db: float
    los_link_count: int
    nlos_link_count: int
    mean_data_rate_mbps: float


@dataclass
class ComputeQueuingTwinState:
    """MEC computation and queuing dynamics digital twin state."""

    mean_utilization: float
    max_utilization: float
    mean_queue_length: float
    mean_waiting_time_ms: float
    mean_system_time_ms: float
    total_completed_tasks: int
    total_dropped_tasks: int


@dataclass
class AerialMobilityTwinState:
    """UAV swarm kinematics, energy, and sojourn coverage digital twin state."""

    n_uavs: int
    n_ues: int
    mean_uav_altitude: float
    mean_uav_speed: float
    mean_battery_pct: float
    mean_propulsion_power_w: float
    mean_sojourn_time_steps: float
    collision_count: int


@dataclass
class NDTTelemetryFrame:
    """Unified Network Digital Twin (NDT) telemetry snapshot for a single step."""

    frame_id: int
    timestamp_s: float
    geospatial: GeospatialTwinState
    rem: REMTwinState
    compute: ComputeQueuingTwinState
    mobility: AerialMobilityTwinState
    uav_positions: List[List[float]]
    ue_positions: List[List[float]]
    uav_telemetry: List[Dict[str, Any]]
    active_links: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Converts telemetry frame into standard Python dictionary."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serializes telemetry frame into JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class RadioEnvironmentMap:
    """Vectorized 2D/3D Radio Environment Map (REM) generator.

    Computes spatial received signal power, interference, and Signal-to-Interference-plus-Noise
    Ratio (SINR) across a uniform spatial evaluation grid over the urban volume.
    """

    def __init__(
        self,
        volume_size: Tuple[float, float, float],
        grid_resolution: int = 50,
        frequency_ghz: float = 29.0,
        bandwidth_hz: float = 10e6,
        noise_figure_db: float = 7.0,
        sinr_threshold_db: float = 0.0,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initializes the RadioEnvironmentMap generator.

        Args:
            volume_size (Tuple[float, float, float]): Urban dimensions (sim_x, sim_y, sim_z).
            grid_resolution (int): Spatial grid points along each horizontal axis. Defaults to 50.
            frequency_ghz (float): Carrier frequency in GHz. Defaults to 29.0.
            bandwidth_hz (float): Channel bandwidth in Hz. Defaults to 10e6.
            noise_figure_db (float): Receiver noise figure in dB. Defaults to 7.0.
            sinr_threshold_db (float): Minimum SINR in dB for valid coverage. Defaults to 0.0.
            device (Union[torch.device, str]): PyTorch compute device.
        """
        self.volume_size = volume_size
        self.grid_res = grid_resolution
        self.freq_ghz = frequency_ghz
        self.bandwidth = bandwidth_hz
        self.noise_figure_db = noise_figure_db
        self.sinr_threshold_db = sinr_threshold_db
        self.device = torch.device(device)

        k_b = 1.380649e-23
        t_k = 290.0
        thermal_noise = k_b * t_k * self.bandwidth
        self.noise_power = thermal_noise * (10 ** (self.noise_figure_db / 10.0))

        c = 299792458.0
        self.wavelength = c / (self.freq_ghz * 1e9)

        # Precompute 2D grid coordinates on ground plane (z = 1.5m)
        x_coords = torch.linspace(
            -volume_size[0] / 2, volume_size[0] / 2, grid_resolution, device=device
        )
        y_coords = torch.linspace(
            -volume_size[1] / 2, volume_size[1] / 2, grid_resolution, device=device
        )
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        grid_z = torch.full_like(grid_x, 1.5)
        # Shape: (grid_res * grid_res, 3)
        self.grid_points = torch.stack(
            [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1
        )

    def compute_rem(
        self,
        uav_positions: torch.Tensor,
        tx_power: float = 2.0,
        urban_map: Optional[Any] = None,
        env_idx: int = 0,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Computes spatial SINR and coverage ratio for active UAV positions.

        Args:
            uav_positions (torch.Tensor): UAV 3D coordinates of shape (N, 3) or (B, N, 3).
            tx_power (float): Transmit power per UAV in Watts. Defaults to 2.0.
            urban_map: Optional VectorizedUrbanMap instance for LoS ray-casting.
            env_idx (int): Batch index if uav_positions has batch dimension.

        Returns:
            Dict[str, Union[torch.Tensor, float]]:
                - 'sinr_grid': 2D tensor of shape (grid_res, grid_res) in dB.
                - 'coverage_mask': 2D boolean tensor indicating SINR >= threshold.
                - 'coverage_ratio': Float percentage in [0.0, 1.0].
                - 'mean_sinr_db': Average SINR across all grid points in dB.
        """
        if uav_positions.ndim == 3:
            uav_pos = uav_positions[env_idx]
        else:
            uav_pos = uav_positions

        n_uavs = uav_pos.shape[0]
        n_points = self.grid_points.shape[0]

        # Distances: (N, n_points)
        diff = uav_pos.unsqueeze(1) - self.grid_points.unsqueeze(0)
        distances = torch.clamp(torch.norm(diff, dim=-1), min=1.0)

        # Free space path loss: (N, n_points)
        fspl = (4.0 * torch.pi * distances / self.wavelength) ** 2

        # Line-of-sight status if urban_map available
        if urban_map is not None and hasattr(urban_map, "check_los_batch"):
            # Shape (1, N, 3) and (1, n_points, 3)
            los_mask = urban_map.check_los_batch(
                uav_pos.unsqueeze(0),
                self.grid_points.unsqueeze(0),
                n_steps=12,
            )[0]
        else:
            los_mask = torch.ones(
                (n_uavs, n_points), dtype=torch.bool, device=self.device
            )

        pl_exponent = torch.where(los_mask, 2.0, 3.5)
        pathloss = fspl * (distances ** (pl_exponent - 2.0))

        # Received power: (N, n_points)
        rx_power = tx_power / pathloss

        # Best server association: max received power
        max_rx_power, best_uav = torch.max(rx_power, dim=0)
        total_rx_power = torch.sum(rx_power, dim=0)
        interference = total_rx_power - max_rx_power

        sinr_linear = max_rx_power / (interference + self.noise_power + 1e-12)
        sinr_db = 10.0 * torch.log10(torch.clamp(sinr_linear, min=1e-6))

        coverage_mask = sinr_db >= self.sinr_threshold_db
        coverage_ratio = float(coverage_mask.float().mean().item())
        mean_sinr = float(sinr_db.mean().item())

        sinr_grid = sinr_db.view(self.grid_res, self.grid_res)
        cov_grid = coverage_mask.view(self.grid_res, self.grid_res)

        return {
            "sinr_grid": sinr_grid,
            "coverage_mask": cov_grid,
            "coverage_ratio": coverage_ratio,
            "mean_sinr_db": mean_sinr,
        }


class NetworkDigitalTwin:
    """Central Network Digital Twin (NDT) manager for UrbanMARL simulations.

    Synchronizes physical, wireless, computing, and mobility layers and extracts
    structured telemetry frames and serialization exports.
    """

    def __init__(
        self,
        volume_size: Tuple[float, float, float] = (500.0, 500.0, 200.0),
        frequency_ghz: float = 29.0,
        bandwidth_hz: float = 10e6,
        noise_figure_db: float = 7.0,
        rem_resolution: int = 40,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initializes the NetworkDigitalTwin manager.

        Args:
            volume_size (Tuple[float, float, float]): 3D urban simulation bounds.
            frequency_ghz (float): mmWave carrier frequency.
            bandwidth_hz (float): Channel bandwidth.
            noise_figure_db (float): Receiver noise figure.
            rem_resolution (int): Spatial grid resolution for Radio Environment Maps.
            device (Union[torch.device, str]): Compute device.
        """
        self.volume_size = list(volume_size)
        self.frequency_ghz = frequency_ghz
        self.bandwidth_hz = bandwidth_hz
        self.noise_figure_db = noise_figure_db
        self.device = torch.device(device)

        self.rem_generator = RadioEnvironmentMap(
            volume_size=volume_size,
            grid_resolution=rem_resolution,
            frequency_ghz=frequency_ghz,
            bandwidth_hz=bandwidth_hz,
            noise_figure_db=noise_figure_db,
            device=device,
        )

        self._frame_history: List[NDTTelemetryFrame] = []

    def capture_frame(
        self,
        env: Any,
        env_idx: int = 0,
        step: Optional[int] = None,
        compute_rem_heatmap: bool = False,
    ) -> NDTTelemetryFrame:
        """Captures a complete digital twin telemetry frame from an active UrbanEnv.

        Args:
            env: UrbanEnv environment instance.
            env_idx (int): Environment batch index to monitor. Defaults to 0.
            step (Optional[int]): Current episode horizon step index.
            compute_rem_heatmap (bool): Whether to calculate full spatial REM SINR grid.

        Returns:
            NDTTelemetryFrame: Structured telemetry snapshot.
        """
        step_idx = (
            step
            if step is not None
            else (
                int(env.current_step[env_idx].item())
                if hasattr(env, "current_step")
                else len(self._frame_history)
            )
        )
        dt = float(getattr(env, "dt", 1.0))
        timestamp = float(step_idx * dt)

        # 1. Geospatial & Structural Layer
        info_tensor = env._env.info[env_idx] if hasattr(env, "_env") else None
        if info_tensor is not None and info_tensor.shape[0] >= 7:
            alpha = float(info_tensor[0].item())
            beta = float(info_tensor[1].item())
            gamma = float(info_tensor[2].item())
            e_param = float(info_tensor[3].item())
            n_b = int(info_tensor[4].item())
            b_w = float(info_tensor[5].item())
            s_w = float(info_tensor[6].item())
        else:
            alpha, beta, gamma, e_param, n_b, b_w, s_w = (
                0.3,
                500.0,
                15.0,
                0.0,
                50,
                25.0,
                20.0,
            )

        b_count = (
            len(env._env.building_data[env_idx])
            if hasattr(env, "_env") and hasattr(env._env, "building_data")
            else n_b
        )

        geo_state = GeospatialTwinState(
            volume_size=list(env.volume_size),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            e_param=e_param,
            n_buildings=n_b,
            building_width=b_w,
            street_width=s_w,
            building_count=b_count,
        )

        # 2. Wireless & REM Propagation Layer
        uav_pos = env.uav_agents_pos[env_idx]
        ue_pos = env.ue_user_pos[env_idx]
        los_tensor = env.uav_ue_los[env_idx] if hasattr(env, "uav_ue_los") else None

        los_count = int(los_tensor.sum().item()) if los_tensor is not None else 0
        total_links = (
            int(los_tensor.numel())
            if los_tensor is not None
            else env.n_uavs * env.n_ues
        )
        nlos_count = total_links - los_count

        if compute_rem_heatmap:
            rem_data = self.rem_generator.compute_rem(
                uav_positions=uav_pos,
                tx_power=float(getattr(env, "max_power", 2.0)),
                urban_map=getattr(env, "_env", None),
                env_idx=env_idx,
            )
            coverage_ratio = float(rem_data["coverage_ratio"])
            mean_sinr = float(rem_data["mean_sinr_db"])
        else:
            coverage_ratio = float(los_count / max(1, total_links))
            mean_sinr = 15.0 if los_count > 0 else -5.0

        rem_state = REMTwinState(
            frequency_ghz=self.frequency_ghz,
            bandwidth_hz=self.bandwidth_hz,
            coverage_ratio=coverage_ratio,
            mean_sinr_db=mean_sinr,
            los_link_count=los_count,
            nlos_link_count=nlos_count,
            mean_data_rate_mbps=float(coverage_ratio * 50.0),
        )

        # 3. Computation & Queuing Layer
        if hasattr(env, "uav_mec_utilization"):
            uav_util = env.uav_mec_utilization[env_idx]
            uav_ql = env.uav_mec_queue_length[env_idx]
            uav_wt = env.uav_mec_waiting_time[env_idx]
            uav_st = env.uav_mec_system_time[env_idx]
            completed = (
                int(env.uav_completed_tasks[env_idx].sum().item())
                if hasattr(env, "uav_completed_tasks")
                else 0
            )
            dropped = (
                int(env.uav_dropped_tasks[env_idx].sum().item())
                if hasattr(env, "uav_dropped_tasks")
                else 0
            )

            compute_state = ComputeQueuingTwinState(
                mean_utilization=float(uav_util.mean().item()),
                max_utilization=float(uav_util.max().item()),
                mean_queue_length=float(uav_ql.mean().item()),
                mean_waiting_time_ms=float(uav_wt.mean().item() * 1000.0),
                mean_system_time_ms=float(uav_st.mean().item() * 1000.0),
                total_completed_tasks=completed,
                total_dropped_tasks=dropped,
            )
        else:
            compute_state = ComputeQueuingTwinState(
                mean_utilization=0.0,
                max_utilization=0.0,
                mean_queue_length=0.0,
                mean_waiting_time_ms=0.0,
                mean_system_time_ms=0.0,
                total_completed_tasks=0,
                total_dropped_tasks=0,
            )

        # 4. Mobility & Kinematic Layer
        uav_alt = float(uav_pos[:, 2].mean().item())
        uav_vel = (
            env.uav_velocity[env_idx]
            if hasattr(env, "uav_velocity")
            else torch.zeros_like(uav_pos)
        )
        uav_speed = float(torch.norm(uav_vel, dim=-1).mean().item())
        battery = (
            float(env.uav_battery[env_idx].mean().item())
            if hasattr(env, "uav_battery")
            else 100.0
        )
        prop_power = 0.1 * uav_speed * 10.0 + 5.0
        sojourn = (
            float(env.uav_ue_sojourn[env_idx].mean().item())
            if hasattr(env, "uav_ue_sojourn")
            else 0.0
        )
        collisions = (
            int(env.uav_collisions[env_idx].sum().item())
            if hasattr(env, "uav_collisions")
            else 0
        )

        mobility_state = AerialMobilityTwinState(
            n_uavs=env.n_uavs,
            n_ues=env.n_ues,
            mean_uav_altitude=uav_alt,
            mean_uav_speed=uav_speed,
            mean_battery_pct=battery,
            mean_propulsion_power_w=prop_power,
            mean_sojourn_time_steps=sojourn,
            collision_count=collisions,
        )

        # Entity coordinates & active communication links
        uav_pos_list = uav_pos.detach().cpu().numpy().tolist()
        ue_pos_list = ue_pos.detach().cpu().numpy().tolist()

        uav_telemetry = []
        for i in range(env.n_uavs):
            uav_info = {
                "id": f"uav_{i}",
                "position": uav_pos_list[i],
                "velocity": uav_vel[i].detach().cpu().numpy().tolist(),
                "battery": float(
                    env.uav_battery[env_idx, i].item()
                    if hasattr(env, "uav_battery")
                    else 100.0
                ),
            }
            if hasattr(env, "uav_mec_utilization"):
                uav_info["utilization"] = float(
                    env.uav_mec_utilization[env_idx, i].item()
                )
                uav_info["system_time_ms"] = float(
                    env.uav_mec_system_time[env_idx, i].item() * 1000.0
                )
                uav_info["completed_tasks"] = int(
                    env.uav_completed_tasks[env_idx, i].item()
                )
            uav_telemetry.append(uav_info)

        active_links = []
        if los_tensor is not None:
            for u in range(env.n_uavs):
                for m in range(env.n_ues):
                    is_los = bool(los_tensor[u, m].item())
                    active_links.append(
                        {
                            "uav_id": f"uav_{u}",
                            "ue_id": f"ue_{m}",
                            "los": is_los,
                            "uav_pos": uav_pos_list[u],
                            "ue_pos": ue_pos_list[m],
                        }
                    )

        frame = NDTTelemetryFrame(
            frame_id=step_idx,
            timestamp_s=timestamp,
            geospatial=geo_state,
            rem=rem_state,
            compute=compute_state,
            mobility=mobility_state,
            uav_positions=uav_pos_list,
            ue_positions=ue_pos_list,
            uav_telemetry=uav_telemetry,
            active_links=active_links,
        )

        self._frame_history.append(frame)
        return frame

    def export_telemetry_json(
        self, frame: NDTTelemetryFrame, filepath: Optional[str] = None
    ) -> str:
        """Exports a single telemetry frame to JSON string or file.

        Args:
            frame (NDTTelemetryFrame): Telemetry snapshot.
            filepath (Optional[str]): Target file path to write to.

        Returns:
            str: Serialized JSON payload string.
        """
        payload = frame.to_json()
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    def export_telemetry_history(
        self, filepath: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Exports all recorded telemetry frames in chronological order.

        Args:
            filepath (Optional[str]): Target file path for JSON array.

        Returns:
            List[Dict[str, Any]]: List of dictionary telemetry snapshots.
        """
        history_dicts = [f.to_dict() for f in self._frame_history]
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(history_dicts, f, indent=2)
        return history_dicts

    def clear_history(self) -> None:
        """Clears recorded frame history."""
        self._frame_history.clear()
