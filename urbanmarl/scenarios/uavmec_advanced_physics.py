"""UrbanMARL High-Fidelity UAV-MEC Task Offloading Scenario.

NEW standalone scenario combining:
1. High-fidelity rotary-wing UAV aerodynamic propulsion power (Zeng et al., IEEE TWC).
2. 3GPP TR 38.901 3D directional antenna radiation beamforming and elevation fading.
3. Vectorized M/M/c MEC server queuing dynamics (utilization, response latency, deadlines).
4. Dynamic task offloading assignment and sojourn time maximization.
"""

from typing import Optional, Tuple

import torch
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded

from urbanmarl.models.aerodynamics import VectorizedAerodynamics
from urbanmarl.models.channel_advanced import AdvancedChannelModel
from urbanmarl.models.dtlcm import compute_batched_dtlcm_assignment
from urbanmarl.models.mec_queue import VectorizedMECQueue
from urbanmarl.scenarios.base import UrbanScenario


class Scenario(UrbanScenario):
    """Scenario name: UAVMEC_ADVANCED_PHYSICS.

    Objective:
    Multi-UAV MEC networks optimize 3D flight trajectories, offloading matching,
    and server allocations under realistic rotary-wing flight aerodynamics and
    3D directional beamforming propagation.
    """

    def __init__(self, config: dict) -> None:
        """Initializes the high-fidelity UAV-MEC scenario.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config)
        self.has_state = config.get("has_state", True)
        self.has_agent_info = config.get("has_agent_info", False)
        self.has_global_info = config.get("has_global_info", True)

        # Task arrival and computing parameters
        self.task_arrival_rate = config.get("task_arrival_rate", 2.5)
        self.task_data_min = config.get("task_data_min", 0.5e6)
        self.task_data_max = config.get("task_data_max", 2.0e6)
        self.cpu_cycles_per_bit = config.get("cpu_cycles_per_bit", 800.0)
        self.task_deadline = config.get("task_deadline", 0.2)

        self.uav_num_cores = config.get("uav_num_cores", 4)
        self.uav_cpu_freq = config.get("uav_cpu_freq", 2.0e9)

        # Reward weights
        self.w_tasks = config.get("w_tasks", 2.0)
        self.w_delay = config.get("w_delay", 1.5)
        self.w_energy = config.get("w_energy", 0.5)
        self.w_collision = config.get("w_collision", 5.0)
        self.w_sojourn = config.get("w_sojourn", 0.5)

        self.aero: Optional[VectorizedAerodynamics] = None
        self.channel: Optional[AdvancedChannelModel] = None
        self.queue: Optional[VectorizedMECQueue] = None

    def _ensure_models(self, env) -> None:
        """Initializes aerodynamics, channel, and queue models on env device."""
        if self.aero is None or self.aero.device != env.device:
            self.aero = VectorizedAerodynamics(device=env.device)
            self.channel = AdvancedChannelModel(
                frequency_ghz=float(getattr(env, "frequency_ghz", 29.0)),
                bandwidth_hz=float(getattr(env, "bandwidth", 10e6)),
                noise_figure_db=float(getattr(env, "noise_figure_db", 7.0)),
                device=env.device,
            )
            self.queue = VectorizedMECQueue(device=env.device)

    def _reset_all(
        self, env, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets all environment instances."""
        self._ensure_models(env)
        b = env.batch_size[0]

        env.uav_agents_pos = env._env.gen_pos(
            num_pos=env.n_uavs, min_z=20.0, max_z=120.0, outdoor=True
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
        env.uav_ue_sojourn = torch.zeros((b, env.n_uavs, env.n_ues), device=env.device)

        env.uav_mec_utilization = torch.zeros((b, env.n_uavs), device=env.device)
        env.uav_mec_queue_length = torch.zeros((b, env.n_uavs), device=env.device)
        env.uav_mec_waiting_time = torch.zeros((b, env.n_uavs), device=env.device)
        env.uav_mec_system_time = torch.zeros((b, env.n_uavs), device=env.device)
        env.uav_completed_tasks = torch.zeros((b, env.n_uavs), device=env.device)
        env.uav_dropped_tasks = torch.zeros((b, env.n_uavs), device=env.device)

    def _reset_at(
        self, env, env_index: int, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets a single environment instance."""
        self._ensure_models(env)

        env.uav_agents_pos[env_index] = env._env.gen_pos(
            num_pos=env.n_uavs,
            min_z=20.0,
            max_z=120.0,
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
        env.uav_ue_sojourn[env_index].zero_()

        env.uav_mec_utilization[env_index].zero_()
        env.uav_mec_queue_length[env_index].zero_()
        env.uav_mec_waiting_time[env_index].zero_()
        env.uav_mec_system_time[env_index].zero_()
        env.uav_completed_tasks[env_index].zero_()
        env.uav_dropped_tasks[env_index].zero_()

    def process_actions(self, env, tensordict: TensorDictBase) -> None:
        """Executes flight dynamics, aerodynamics power, 3D channel, and MEC queues."""
        self._ensure_models(env)
        b = env.batch_size[0]

        env.uav_collisions.zero_()
        for group, _agent_names in env.group_map.items():
            if group.lower() in ("uav", "agents"):
                group_action = tensordict.get((group, "action"), None)
                if group_action is None:
                    group_action = tensordict.get(
                        ("agents", "action"), tensordict.get("action", None)
                    )
                if group_action is None:
                    continue
                group_action = group_action.detach()

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

                # High-Fidelity Aerodynamic Propulsion Power Dissipation
                propulsion_power = self.aero.compute_propulsion_power(env.uav_velocity)
                # Energy = Power * dt (Joules), converted to battery percentage
                battery_drain = (propulsion_power / 1000.0) * env.dt * 0.1
                env.uav_battery -= battery_drain

        # Check wireless LoS
        env.uav_ue_los = env._env.check_los_batch(env.uav_agents_pos, env.ue_user_pos)

        # Update Sojourn Time
        env.uav_ue_sojourn = torch.where(
            env.uav_ue_los,
            env.uav_ue_sojourn + 1.0,
            torch.zeros_like(env.uav_ue_sojourn),
        )

        # 3D Directional Data Rates
        p_tx = float(getattr(env, "max_power", 2.0))
        env.uav_ue_rates = self.channel.compute_data_rates(
            tx_pos=env.uav_agents_pos,
            rx_pos=env.ue_user_pos,
            tx_power=p_tx,
            los_mask=env.uav_ue_los,
            include_fading=True,
        )

        # Task generation
        data_sizes = (
            torch.rand((b, env.n_ues), device=env.device)
            * (self.task_data_max - self.task_data_min)
            + self.task_data_min
        )

        # Dynamic Task Offloading Matching
        uav_caps = torch.full((b, env.n_uavs), self.uav_cpu_freq, device=env.device)
        uav_loads = env.uav_mec_utilization * uav_caps
        task_cycles = data_sizes * self.cpu_cycles_per_bit
        assignments = compute_batched_dtlcm_assignment(
            uav_pos=env.uav_agents_pos,
            ue_pos=env.ue_user_pos,
            uav_caps=uav_caps,
            uav_loads=uav_loads,
            task_workloads=task_cycles,
            distance_weight=0.7,
        )

        # M/M/c Queuing Dynamics
        assigned_one_hot = torch.nn.functional.one_hot(
            assignments, num_classes=env.n_uavs
        ).float()
        arrival_rates = assigned_one_hot.sum(dim=1) * self.task_arrival_rate

        mean_cycles = (
            (self.task_data_min + self.task_data_max) * 0.5 * self.cpu_cycles_per_bit
        )
        service_rates = torch.full(
            (b, env.n_uavs),
            self.uav_cpu_freq / mean_cycles,
            device=env.device,
        )

        queue_metrics = self.queue.compute_delays(
            arrival_rates=arrival_rates,
            service_rates=service_rates,
            num_cores=self.uav_num_cores,
        )

        env.uav_mec_utilization = queue_metrics["utilization"]
        env.uav_mec_queue_length = queue_metrics["avg_queue_length"]
        env.uav_mec_waiting_time = queue_metrics["avg_waiting_time"]
        env.uav_mec_system_time = queue_metrics["avg_system_time"]

        # Track completed / dropped tasks based on deadline
        deadline_met = env.uav_mec_system_time <= self.task_deadline
        tasks_arrived = arrival_rates * env.dt
        env.uav_completed_tasks = torch.where(
            deadline_met, tasks_arrived, torch.zeros_like(tasks_arrived)
        )
        env.uav_dropped_tasks = torch.where(
            deadline_met, torch.zeros_like(tasks_arrived), tasks_arrived
        )

    def observation_spec(self, env, group: str) -> Composite:
        """Observation spec for high-fidelity UAV-MEC."""
        return Unbounded(shape=torch.Size([8]), dtype=torch.float32, device=env.device)

    def observation(self, env) -> torch.Tensor:
        """Observation tensor: [x, y, z, battery, util, system_time, los_ratio, mean_sojourn]."""
        pos = env.uav_agents_pos
        battery = env.uav_battery
        util = env.uav_mec_utilization.unsqueeze(-1)
        st = env.uav_mec_system_time.unsqueeze(-1)
        los_ratio = env.uav_ue_los.float().mean(dim=-1, keepdim=True)
        sojourn = env.uav_ue_sojourn.mean(dim=-1, keepdim=True)
        return torch.cat([pos, battery, util, st, los_ratio, sojourn], dim=-1)

    def action_spec(self, env, group: str) -> Composite:
        """Continuous velocity action: [v_h, phi, v_z]."""
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
        """Reward spec."""
        return Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=env.device)

    def reward(self, env, group: str) -> torch.Tensor:
        """Multi-objective reward combining completed tasks, delay, energy, and sojourn bonus."""
        throughput = (env.uav_completed_tasks / max(1, env.n_ues)).unsqueeze(-1)
        delay_penalty = (
            torch.clamp(env.uav_mec_system_time, max=self.task_deadline)
            / self.task_deadline
        ).unsqueeze(-1)

        collision_penalty = env.uav_collisions.float()
        sojourn = env.uav_ue_sojourn.mean(dim=-1, keepdim=True)
        sojourn_bonus = torch.tanh(0.1 * sojourn)

        reward = (
            self.w_tasks * throughput
            - self.w_delay * delay_penalty
            - self.w_collision * collision_penalty
            + self.w_sojourn * sojourn_bonus
        )
        return reward

    def done(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Termination flags."""
        terminated = env.uav_collisions.any(dim=1) | (env.uav_battery <= 0.0).any(dim=1)
        truncated = env.current_step >= env.max_steps
        dones = terminated | truncated
        return dones, terminated, truncated

    def state_spec(self, env) -> Optional[Composite]:
        """Global state spec."""
        n_env_param = 4
        obs_dim = 8
        state_dim = n_env_param + (env.n_uavs * obs_dim) + (env.n_ues * 3)
        return Unbounded(
            shape=torch.Size([state_dim]), dtype=torch.float32, device=env.device
        )

    def state(self, env) -> Optional[torch.Tensor]:
        """Global state tensor."""
        b = env.batch_size[0]
        n_env_param = 4
        obs = self.observation(env).view(b, -1)
        ue_pos = env.ue_user_pos.view(b, -1)
        urban_params = env._env.info[:, :n_env_param].view(b, -1)
        return torch.cat([urban_params, obs, ue_pos], dim=-1)

    def info_agent_spec(self, env, group: str) -> Optional[Composite]:
        """Agent-level info spec if enabled."""
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
        """Agent-level info dictionary."""
        if not self.has_agent_info:
            return None
        return {
            "utilization": env.uav_mec_utilization.unsqueeze(-1),
            "system_time": env.uav_mec_system_time.unsqueeze(-1),
        }

    def info_global_spec(self, env) -> Optional[Composite]:
        """Global info spec."""
        return Composite(
            {
                "urban_params": Unbounded(
                    shape=torch.Size([4]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "completed_tasks": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "dropped_tasks": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "mean_utilization": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "mean_system_time": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "sojourn_time": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "collisions": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
            }
        )

    def info_global(self, env) -> Optional[dict]:
        """Global info dictionary."""
        return {
            "urban_params": env._env.info[:, :4].view(env.batch_size[0], -1),
            "completed_tasks": env.uav_completed_tasks.sum(dim=-1, keepdim=True),
            "dropped_tasks": env.uav_dropped_tasks.sum(dim=-1, keepdim=True),
            "mean_utilization": env.uav_mec_utilization.mean(dim=-1, keepdim=True),
            "mean_system_time": env.uav_mec_system_time.mean(dim=-1, keepdim=True),
            "sojourn_time": env.uav_ue_sojourn.mean(dim=(-1, -2)).unsqueeze(-1),
            "collisions": env.uav_collisions.float().sum(dim=1),
        }

    def _render(self, env, mode: str = "rgb_array") -> dict:
        """Render payload."""
        if not hasattr(self, "render_idx"):
            self.render_idx = torch.randint(
                0, env.batch_size[0], (1,), device=env.device
            ).item()

        uav_positions = env.uav_agents_pos[self.render_idx].detach().cpu()
        ue_positions = env.ue_user_pos[self.render_idx].detach().cpu()
        los = env.uav_ue_los[self.render_idx].detach().cpu()

        links = []
        collisions = []

        for uav in range(env.n_uavs):
            if env.uav_collisions[self.render_idx, uav]:
                collisions.append(uav_positions[uav])
            for ue in range(env.n_ues):
                links.append(
                    {
                        "source": uav_positions[uav],
                        "target": ue_positions[ue],
                        "los": los[uav, ue].item(),
                    }
                )

        return {
            "uav_positions": uav_positions,
            "ue_positions": ue_positions,
            "links": links,
            "collisions": collisions,
            "telemetry": {
                "completed_tasks": int(
                    env.uav_completed_tasks[self.render_idx].sum().item()
                ),
                "dropped_tasks": int(
                    env.uav_dropped_tasks[self.render_idx].sum().item()
                ),
                "utilization": float(
                    env.uav_mec_utilization[self.render_idx].mean().item()
                ),
                "system_time": float(
                    env.uav_mec_system_time[self.render_idx].mean().item()
                ),
            },
        }
