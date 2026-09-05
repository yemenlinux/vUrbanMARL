"""UrbanMARL UAV MEC Offloading Scenario.

Provides a fully vectorized, closed-loop multi-agent reinforcement learning scenario
for multi-UAV assisted Mobile Edge Computing (MEC) networks.

Integrates:

* Vectorized mmWave radio channel propagation (Shannon transmission capacity).
* Vectorized M/M/c queuing dynamics (server utilization, queue waiting time, system delay).
* Dynamic Task-Distance-Load Capacity Matching (DTLCM) heuristic offloading.
* Sojourn time tracking and deadline violation penalization based on Dr. Basheer Raddwan's
  research ("Quantify the joint effect of mobility and urban environment on computation
  offloading to multi-UAV MEC network: Sojourn time", Ad Hoc Networks, 2025).
"""

from typing import Optional, Tuple

import torch
from tensordict import TensorDictBase
from torchrl.data import BoundedContinuous, Composite, Unbounded

from urbanmarl.models.channel import VectorizedChannelModel
from urbanmarl.models.dtlcm import compute_batched_dtlcm_assignment
from urbanmarl.models.mec_queue import VectorizedMECQueue
from urbanmarl.scenarios.base import UrbanScenario


class Scenario(UrbanScenario):
    """UAV MEC Offloading scenario for UrbanMARL environments.

    Multi-UAV base stations collaboratively navigate a 3D urban environment to
    maximize task offloading completion, minimize total system latency and energy,
    avoid building collisions, and maintain persistent sojourn coverage over active users.
    """

    def __init__(self, config: dict) -> None:
        """Initializes the UAV MEC Offloading scenario.

        Args:
            config (dict): Scenario configuration parameters.
        """
        super().__init__(config)
        self.has_state = config.get("has_state", True)
        self.has_agent_info = config.get("has_agent_info", False)
        self.has_global_info = config.get("has_global_info", True)

        # Radio channel parameters
        self.channel_config = {
            "frequency_ghz": config.get("frequency_ghz", 29.0),
            "g2a_bandwidth": config.get("g2a_bandwidth", 10e6),
            "noise_figure_db": config.get("noise_figure_db", 7.0),
        }

        # MEC task workload parameters
        self.task_arrival_rate = config.get(
            "task_arrival_rate", 5.0
        )  # tasks/sec per UE
        self.data_size_min = config.get("task_data_size_min", 0.5e6)  # 0.5 Mbits
        self.data_size_max = config.get("task_data_size_max", 2.0e6)  # 2.0 Mbits
        self.mean_data_size = (self.data_size_min + self.data_size_max) / 2.0
        self.cpu_cycles_per_bit = config.get("cpu_cycles_per_bit", 1000.0)  # cycles/bit
        self.max_latency_deadline = config.get("max_latency_deadline", 1.0)  # seconds
        self.ue_tx_power = config.get("ue_tx_power", 0.5)  # Watts

        # UAV MEC server compute parameters
        self.uav_cpu_freq = config.get("uav_cpu_freq", 20.0e9)  # 20 GHz aggregate
        self.uav_num_cores = config.get("uav_num_cores", 4)  # 4 cores per UAV
        self.distance_weight = config.get("distance_weight", 0.7)

        # Multi-objective reward weights
        self.w_tasks = config.get("w_tasks", 1.0)
        self.w_latency = config.get("w_latency", 0.5)
        self.w_energy = config.get("w_energy", 0.1)
        self.w_collision = config.get("w_collision", 2.0)
        self.w_sojourn = config.get("w_sojourn", 0.3)

        # Observation configuration: default 8-dim MEC state
        self.extended_mec_obs = config.get("extended_mec_obs", True)

        self.channel_model: Optional[VectorizedChannelModel] = None
        self.mec_queue: Optional[VectorizedMECQueue] = None

    def _ensure_models_initialized(self, env) -> None:
        """Lazily initializes the vectorized channel and queuing models on env device."""
        if self.channel_model is None or self.channel_model.device != env.device:
            self.channel_model = VectorizedChannelModel(
                self.channel_config, device=env.device
            )
        if self.mec_queue is None or self.mec_queue.device != env.device:
            self.mec_queue = VectorizedMECQueue(device=env.device, max_cores=64)

    def _init_mec_tensors(self, env) -> None:
        """Initializes internal MEC queuing and offloading tracking tensors."""
        b = env.batch_size[0]
        device = env.device

        env.uav_mec_utilization = torch.zeros((b, env.n_uavs), device=device)
        env.uav_mec_queue_length = torch.zeros((b, env.n_uavs), device=device)
        env.uav_mec_waiting_time = torch.zeros((b, env.n_uavs), device=device)
        env.uav_mec_system_time = torch.zeros((b, env.n_uavs), device=device)
        env.uav_completed_tasks = torch.zeros((b, env.n_uavs), device=device)
        env.uav_dropped_tasks = torch.zeros((b, env.n_uavs), device=device)
        env.uav_assigned_tasks = torch.zeros((b, env.n_uavs), device=device)

        # Sojourn time matrix tracking consecutive LoS steps between UAVs and UEs
        env.uav_ue_sojourn = torch.zeros((b, env.n_uavs, env.n_ues), device=device)

        # Task properties per UE
        env.ue_task_data_size = torch.full(
            (b, env.n_ues), self.mean_data_size, device=device
        )
        env.ue_task_cpu_cycles = env.ue_task_data_size * self.cpu_cycles_per_bit
        env.ue_assigned_uav = torch.zeros(
            (b, env.n_ues), dtype=torch.long, device=device
        )

    def _reset_all(
        self, env, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets all environment instances across the batch."""
        self._ensure_models_initialized(env)
        b = env.batch_size[0]

        env.uav_agents_pos = env._env.gen_pos(
            num_pos=env.n_uavs, min_z=20.0, max_z=150.0, outdoor=True
        )
        env.uav_battery = torch.full((b, env.n_uavs, 1), 100.0, device=env.device)
        env.uav_velocity = torch.zeros((b, env.n_uavs, 3), device=env.device)
        env.uav_collisions = torch.zeros(
            (b, env.n_uavs, 1), dtype=torch.bool, device=env.device
        )
        env.ue_user_pos = env._env.gen_pos(
            num_pos=env.n_ues, min_z=1.5, max_z=1.5, outdoor=True
        )
        env.ue_battery = torch.full((b, env.n_ues, 1), 100.0, device=env.device)
        env.current_step = torch.zeros((b, 1), dtype=torch.int32, device=env.device)
        env.done = torch.zeros((b, 1), dtype=torch.bool, device=env.device)

        env.uav_ue_los = env._env.check_los_batch(env.uav_agents_pos, env.ue_user_pos)
        self._init_mec_tensors(env)

    def _reset_at(
        self, env, env_index: int, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets a specific environment instance in the batch."""
        self._ensure_models_initialized(env)

        env.uav_agents_pos[env_index] = env._env.gen_pos(
            num_pos=env.n_uavs,
            min_z=20.0,
            max_z=150.0,
            batch_idx=env_index,
            outdoor=True,
        )
        env.uav_battery[env_index] = torch.full(
            (env.n_uavs, 1), 100.0, device=env.device
        )
        env.uav_velocity[env_index] = torch.zeros((env.n_uavs, 3), device=env.device)
        env.uav_collisions[env_index] = torch.zeros(
            (env.n_uavs, 1), dtype=torch.bool, device=env.device
        )
        env.ue_user_pos[env_index] = env._env.gen_pos(
            num_pos=env.n_ues,
            min_z=1.5,
            max_z=1.5,
            batch_idx=env_index,
            outdoor=True,
        )
        env.ue_battery[env_index] = torch.full((env.n_ues, 1), 100.0, device=env.device)
        env.current_step[env_index] = torch.zeros(
            (1,), dtype=torch.int32, device=env.device
        )
        env.done[env_index] = torch.zeros((1,), dtype=torch.bool, device=env.device)

        env.uav_ue_los[env_index] = env._env.check_los_batch(
            env.uav_agents_pos[env_index : env_index + 1],
            env.ue_user_pos[env_index : env_index + 1],
        )[0]

        if hasattr(env, "uav_mec_utilization"):
            env.uav_mec_utilization[env_index].zero_()
            env.uav_mec_queue_length[env_index].zero_()
            env.uav_mec_waiting_time[env_index].zero_()
            env.uav_mec_system_time[env_index].zero_()
            env.uav_completed_tasks[env_index].zero_()
            env.uav_dropped_tasks[env_index].zero_()
            env.uav_assigned_tasks[env_index].zero_()
            env.uav_ue_sojourn[env_index].zero_()

    def process_actions(self, env, tensordict: TensorDictBase) -> None:
        """Executes agent actions and runs the end-to-end MEC offloading pipeline.

        Steps:
        1. Updates 3D UAV kinematics and collision/LoS states.
        2. Computes wireless transmission data rates (Shannon capacity).
        3. Matches UE tasks to UAV servers using DTLCM heuristic.
        4. Calculates M/M/c queuing dynamics (utilization, waiting, response delay).
        5. Tracks sojourn time and task completion within SLA deadlines.
        """
        self._ensure_models_initialized(env)
        b = env.batch_size[0]

        # 1. Kinematics & Spatial Movement
        env.uav_collisions.zero_()
        for group, _agent_names in env.group_map.items():
            if group.lower() in ("uav", "agents"):
                group_action = tensordict.get((group, "action"), None)
                if group_action is None:
                    group_action = tensordict.get(
                        ("agents", "action"),
                        tensordict.get("action", None),
                    )
                if group_action is None:
                    continue

                dx = group_action[..., 0] * torch.cos(group_action[..., 1])
                dy = group_action[..., 0] * torch.sin(group_action[..., 1])
                dz = group_action[..., 2]
                delta_pos = torch.stack([dx, dy, dz], dim=-1)

                env.previous_uav_pos = env.uav_agents_pos.clone()
                env.uav_agents_pos += delta_pos

                env.uav_agents_pos[..., 0] = torch.clamp(
                    env.uav_agents_pos[..., 0],
                    -env.volume_size[0] / 2,
                    env.volume_size[0] / 2,
                )
                env.uav_agents_pos[..., 1] = torch.clamp(
                    env.uav_agents_pos[..., 1],
                    -env.volume_size[1] / 2,
                    env.volume_size[1] / 2,
                )
                env.uav_agents_pos[..., 2] = torch.clamp(
                    env.uav_agents_pos[..., 2], 0.0, env.volume_size[2]
                )

                env.uav_velocity = (env.uav_agents_pos - env.previous_uav_pos) / env.dt
                env.uav_collisions = env._env.check_collision_batch(
                    env.uav_agents_pos, env.previous_uav_pos
                )
                env.uav_ue_los = env._env.check_los_batch(
                    env.uav_agents_pos, env.ue_user_pos
                )

                # Battery dissipation: propulsion flight power
                horizontal_speed = torch.norm(
                    env.uav_velocity[..., :2], dim=-1, keepdim=True
                )
                vertical_speed = torch.abs(env.uav_velocity[..., 2:])
                propulsion_power = 0.1 * horizontal_speed + 0.2 * vertical_speed
                env.uav_battery -= propulsion_power * env.dt

        # 2. Dynamic Task Workload Generation per UE
        rand_sizes = torch.rand((b, env.n_ues), device=env.device)
        env.ue_task_data_size = self.data_size_min + rand_sizes * (
            self.data_size_max - self.data_size_min
        )
        env.ue_task_cpu_cycles = env.ue_task_data_size * self.cpu_cycles_per_bit

        # 3. mmWave Radio Transmission Data Rates
        ue_tx_powers = torch.full((b, env.n_ues), self.ue_tx_power, device=env.device)
        # channel rates: shape (B, n_uavs, n_ues)
        rates = self.channel_model.compute_data_rates(
            tx_pos=env.ue_user_pos,
            rx_pos=env.uav_agents_pos,
            tx_power=ue_tx_powers,
            los_mask=env.uav_ue_los,
        )

        # 4. DTLCM Task Offloading Assignment
        uav_caps = torch.full((b, env.n_uavs), self.uav_cpu_freq, device=env.device)
        uav_loads = env.uav_mec_utilization * uav_caps
        assigned_uavs = compute_batched_dtlcm_assignment(
            uav_pos=env.uav_agents_pos,
            ue_pos=env.ue_user_pos,
            uav_caps=uav_caps,
            uav_loads=uav_loads,
            task_workloads=env.ue_task_cpu_cycles,
            distance_weight=self.distance_weight,
        )
        env.ue_assigned_uav = assigned_uavs

        # 5. Uplink Transmission Latency
        b_indices = torch.arange(b, device=env.device).unsqueeze(1).expand(b, env.n_ues)
        m_indices = (
            torch.arange(env.n_ues, device=env.device).unsqueeze(0).expand(b, env.n_ues)
        )
        rate_assigned = rates[b_indices, assigned_uavs, m_indices]
        los_assigned = env.uav_ue_los[b_indices, assigned_uavs, m_indices]

        tx_delay = env.ue_task_data_size / torch.clamp(rate_assigned, min=1e3)

        # 6. Aggregate Task Arrivals & M/M/c Queuing Delays per UAV
        arrival_rates = torch.zeros((b, env.n_uavs), device=env.device)
        assigned_counts = torch.zeros((b, env.n_uavs), device=env.device)

        for u_idx in range(env.n_uavs):
            mask = (assigned_uavs == u_idx) & los_assigned
            assigned_counts[:, u_idx] = mask.float().sum(dim=1)
            # Total arrival rate = assigned UEs * per-UE Poisson arrival rate
            arrival_rates[:, u_idx] = assigned_counts[:, u_idx] * (
                self.task_arrival_rate / max(1, env.n_ues)
            )

        mean_cycles_per_task = self.mean_data_size * self.cpu_cycles_per_bit
        service_rates_per_core = (
            self.uav_cpu_freq / self.uav_num_cores
        ) / mean_cycles_per_task
        service_rates = torch.full(
            (b, env.n_uavs), service_rates_per_core, device=env.device
        )
        num_cores = torch.full(
            (b, env.n_uavs), self.uav_num_cores, dtype=torch.long, device=env.device
        )

        queue_delays = self.mec_queue.compute_delays(
            arrival_rates=arrival_rates,
            service_rates=service_rates,
            num_cores=num_cores,
        )

        env.uav_mec_utilization = queue_delays["utilization"]
        env.uav_mec_queue_length = queue_delays["avg_queue_length"]
        env.uav_mec_waiting_time = queue_delays["avg_waiting_time"]
        env.uav_mec_system_time = queue_delays["avg_system_time"]
        env.uav_assigned_tasks = assigned_counts

        # 7. Total End-to-End Latency & SLA Deadline Verification
        uav_sys_time_per_ue = env.uav_mec_system_time[b_indices, assigned_uavs]
        uav_util_per_ue = env.uav_mec_utilization[b_indices, assigned_uavs]
        total_delay = tx_delay + uav_sys_time_per_ue

        task_success_mask = (
            los_assigned
            & (total_delay <= self.max_latency_deadline)
            & (uav_util_per_ue < 0.99)
        )

        for u_idx in range(env.n_uavs):
            uav_mask = assigned_uavs == u_idx
            success_count = (task_success_mask & uav_mask).float().sum(dim=1)
            dropped_count = (~task_success_mask & uav_mask).float().sum(dim=1)
            env.uav_completed_tasks[:, u_idx] = success_count
            env.uav_dropped_tasks[:, u_idx] = dropped_count

        # 8. Computing Energy Consumption
        # Dynamic power = kappa * freq^3 * active_cores, energy = power * dt
        kappa = 1e-28
        comp_power = (
            kappa
            * (self.uav_cpu_freq**3)
            * env.uav_mec_utilization
            * self.uav_num_cores
        )
        env.uav_battery -= (comp_power.unsqueeze(-1) * 0.001) * env.dt

        # 9. Sojourn Time Accumulation
        env.uav_ue_sojourn = torch.where(
            env.uav_ue_los,
            env.uav_ue_sojourn + 1.0,
            torch.zeros_like(env.uav_ue_sojourn),
        )

    def observation_spec(self, env, group: str) -> Composite:
        """Returns observation spec for UAV agents."""
        if self.extended_mec_obs:
            # 8-dim spec: [x, y, z, battery, utilization, avg_system_time, assigned_ratio, los_ratio]
            low = torch.tensor(
                [
                    -env.volume_size[0] / 2,
                    -env.volume_size[1] / 2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                device=env.device,
            )
            high = torch.tensor(
                [
                    env.volume_size[0] / 2,
                    env.volume_size[1] / 2,
                    env.volume_size[2],
                    100.0,
                    1.0,
                    10.0,
                    1.0,
                    1.0,
                ],
                device=env.device,
            )
            return BoundedContinuous(low=low, high=high, shape=torch.Size([8]))
        else:
            # Baseline 4-dim spec: [x, y, z, battery]
            low = torch.tensor(
                [-env.volume_size[0] / 2, -env.volume_size[1] / 2, 0.0, 0.0],
                device=env.device,
            )
            high = torch.tensor(
                [
                    env.volume_size[0] / 2,
                    env.volume_size[1] / 2,
                    env.volume_size[2],
                    100.0,
                ],
                device=env.device,
            )
            return BoundedContinuous(low=low, high=high, shape=torch.Size([4]))

    def observation(self, env) -> torch.Tensor:
        """Constructs observation tensor for all UAV agents."""
        pos = env.uav_agents_pos
        battery = env.uav_battery

        if self.extended_mec_obs:
            util = env.uav_mec_utilization.unsqueeze(-1)
            sys_time = torch.clamp(env.uav_mec_system_time.unsqueeze(-1), 0.0, 10.0)
            assigned_ratio = env.uav_assigned_tasks.unsqueeze(-1) / max(1, env.n_ues)
            los_ratio = env.uav_ue_los.float().mean(dim=-1, keepdim=True)
            return torch.cat(
                [
                    pos,
                    battery,
                    util,
                    sys_time,
                    assigned_ratio,
                    los_ratio,
                ],
                dim=-1,
            )
        else:
            return torch.cat([pos, battery], dim=-1)

    def action_spec(self, env, group: str) -> Composite:
        """Continuous 3D velocity action spec: [v_h, phi, v_z]."""
        from torchrl.data.tensor_specs import Bounded

        max_h_speed = float(env.max_h_speed)
        max_v_speed = float(env.max_v_speed)
        return Bounded(
            low=torch.tensor(
                [-max_h_speed, -torch.pi, -max_v_speed], device=env.device
            ),
            high=torch.tensor([max_h_speed, torch.pi, max_v_speed], device=env.device),
            shape=torch.Size([3]),
            dtype=torch.float32,
            device=env.device,
        )

    def reward_spec(self, env, group: str) -> Composite:
        """Unbounded scalar reward specification."""
        return Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=env.device)

    def reward(self, env, group: str) -> torch.Tensor:
        """Calculates multi-objective reward balancing tasks, latency, energy, and sojourn time.

        Args:
            env: UrbanEnv environment instance.
            group (str): Agent group name.

        Returns:
            torch.Tensor: Reward tensor of shape (batch_size, n_uavs, 1).
        """
        # 1. Task throughput reward: completed task ratio
        completed_ratio = env.uav_completed_tasks.unsqueeze(-1) / max(1, env.n_ues)

        # 2. System latency penalty: normalized by deadline
        latency_penalty = torch.clamp(
            env.uav_mec_system_time.unsqueeze(-1) / self.max_latency_deadline,
            0.0,
            2.0,
        )

        # 3. Collision penalty
        collision_penalty = env.uav_collisions.float()

        # 4. Energy consumption penalty (propulsion + computation)
        speed = torch.norm(env.uav_velocity, dim=-1, keepdim=True)
        energy_penalty = speed / max(1.0, float(env.max_h_speed))

        # 5. Sojourn Time Bonus (Dr. Raddwan's formulation)
        mean_sojourn = env.uav_ue_sojourn.mean(dim=-1, keepdim=True)
        sojourn_bonus = torch.tanh(0.1 * mean_sojourn)

        reward = (
            (self.w_tasks * completed_ratio)
            - (self.w_latency * latency_penalty)
            - (self.w_energy * energy_penalty)
            - (self.w_collision * collision_penalty)
            + (self.w_sojourn * sojourn_bonus)
        )
        return reward

    def done(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes episode termination flags (battery depletion, collision, horizon)."""
        terminated = env.uav_collisions.any(dim=1) | (env.uav_battery <= 0.0).any(dim=1)
        truncated = env.current_step >= env.max_steps
        dones = terminated | truncated
        return dones, terminated, truncated

    def state_spec(self, env) -> Optional[Composite]:
        """Global centralized state spec for CTDE algorithms (e.g. MAPPO)."""
        n_env_param = 4  # alpha, beta, gamma, E
        pos_dim = 3
        obs_dim = 8 if self.extended_mec_obs else 4
        mec_extra_dim = (
            4  # utilization, queue_len, waiting_time, completed_tasks per UAV
        )

        state_dim = (
            n_env_param
            + (env.n_uavs * (obs_dim + mec_extra_dim))
            + (env.n_ues * pos_dim)
        )
        return Unbounded(
            shape=torch.Size([state_dim]), dtype=torch.float32, device=env.device
        )

    def state(self, env) -> Optional[torch.Tensor]:
        """Constructs global centralized state tensor."""
        b = env.batch_size[0]
        n_env_param = 4

        obs = self.observation(env).view(b, -1)
        mec_state = torch.cat(
            [
                env.uav_mec_utilization,
                env.uav_mec_queue_length,
                env.uav_mec_system_time,
                env.uav_completed_tasks,
            ],
            dim=-1,
        )
        ue_pos = env.ue_user_pos.view(b, -1)
        urban_params = env._env.info[:, :n_env_param].view(b, -1)

        state = torch.cat([urban_params, obs, mec_state, ue_pos], dim=-1)
        return state

    def info_global_spec(self, env) -> Optional[Composite]:
        """Specifies keys recorded in global info dictionary."""
        return Composite(
            {
                "completed_tasks": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "dropped_tasks": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "mean_system_time": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "mean_utilization": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "sojourn_time": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "collisions": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "los": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
            }
        )

    def info_agent_spec(self, env, group: str) -> Optional[Composite]:
        """Specifies agent-level info dictionary if enabled."""
        if not self.has_agent_info:
            return None
        return Composite(
            {
                "utilization": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "system_time": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
            }
        )

    def info_agent(self, env, group: str) -> Optional[dict]:
        """Returns agent-level info dictionary if enabled."""
        if not self.has_agent_info:
            return None
        return {
            "utilization": env.uav_mec_utilization.unsqueeze(-1),
            "system_time": env.uav_mec_system_time.unsqueeze(-1),
        }

    def info_global(self, env) -> Optional[dict]:
        """Extracts scalar metrics for BenchMARL and TensorBoard/CSV logging."""
        return {
            "completed_tasks": env.uav_completed_tasks.sum(dim=-1, keepdim=True),
            "dropped_tasks": env.uav_dropped_tasks.sum(dim=-1, keepdim=True),
            "mean_system_time": env.uav_mec_system_time.mean(dim=-1, keepdim=True),
            "mean_utilization": env.uav_mec_utilization.mean(dim=-1, keepdim=True),
            "sojourn_time": env.uav_ue_sojourn.mean(dim=(-1, -2)).unsqueeze(-1),
            "collisions": env.uav_collisions.float().sum(dim=1),
            "los": env.uav_ue_los.float().mean(dim=(-1, -2)).unsqueeze(-1),
        }

    def _render(self, env, mode: str = "rgb_array") -> dict:
        """Returns renderable digital twin state payload."""
        if not hasattr(self, "render_idx"):
            self.render_idx = 0

        r_idx = self.render_idx
        uav_positions = env.uav_agents_pos[r_idx].cpu()
        ue_positions = env.ue_user_pos[r_idx].cpu()
        los = env.uav_ue_los[r_idx]

        los_links = []
        collisions = []

        for uav in range(env.n_uavs):
            if env.uav_collisions[r_idx, uav]:
                collisions.append(uav_positions[uav])
            for ue in range(env.n_ues):
                los_links.append(
                    {
                        "source": uav_positions[uav],
                        "target": ue_positions[ue],
                        "los": los[uav, ue].item(),
                    }
                )

        ndt = {
            "uav_positions": uav_positions,
            "ue_positions": ue_positions,
            "links": los_links,
            "collisions": collisions,
            "mec_telemetry": {
                "completed_tasks": env.uav_completed_tasks[r_idx].cpu().tolist(),
                "system_time": env.uav_mec_system_time[r_idx].cpu().tolist(),
                "utilization": env.uav_mec_utilization[r_idx].cpu().tolist(),
            },
        }
        return ndt
