"""UrbanMARL Dynamic User Equipment (UE) Mobility Scenario.

NEW standalone scenario introducing dynamic ground user motion (Manhattan street-constrained,
Random Waypoint, or dynamic crowd hotspots) into multi-UAV wireless coverage.

Ground users continuously move along urban corridors each time step, forcing the UAV swarm
to dynamically coordinate 3D trajectories to maximize persistent Line-of-Sight (LoS) coverage
and sojourn time without colliding with 3D buildings.
"""

from typing import Optional, Tuple

import torch
from tensordict import TensorDictBase
from torchrl.data import BoundedContinuous, Composite, Unbounded

from urbanmarl.models.mobility import VectorizedUserMobility
from urbanmarl.scenarios.base import UrbanScenario


class Scenario(UrbanScenario):
    """Scenario name: UAV_MOBILE_UE.

    Objective:
    Multi-UAV base stations collaboratively track and provide wireless coverage
    to dynamic, mobile ground users moving through street corridors.
    """

    def __init__(self, config: dict) -> None:
        """Initializes the UAV Mobile UE scenario.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config)
        self.has_state = config.get("has_state", True)
        self.has_agent_info = config.get("has_agent_info", False)
        self.has_global_info = config.get("has_global_info", True)

        self.mobility_model_type = config.get("mobility_model", "manhattan")
        self.ue_speed_min = config.get("ue_speed_min", 0.5)
        self.ue_speed_max = config.get("ue_speed_max", 3.0)

        self.w_los = config.get("w_los", 1.0)
        self.w_collision = config.get("w_collision", 2.0)
        self.w_sojourn = config.get("w_sojourn", 0.5)

        self.mobility_engine: Optional[VectorizedUserMobility] = None

    def _ensure_mobility_engine(self, env) -> None:
        """Lazily initializes the mobility engine on the environment's device."""
        if self.mobility_engine is None or self.mobility_engine.device != env.device:
            self.mobility_engine = VectorizedUserMobility(
                volume_size=env.volume_size,
                model_type=self.mobility_model_type,
                speed_min=self.ue_speed_min,
                speed_max=self.ue_speed_max,
                device=env.device,
            )

    def _reset_all(
        self, env, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets all environment instances in the batch."""
        self._ensure_mobility_engine(env)
        b = env.batch_size[0]

        env.uav_agents_pos = env._env.gen_pos(
            num_pos=env.n_uavs, min_z=20.0, max_z=150.0, outdoor=True
        )
        env.uav_battery = torch.full((b, env.n_uavs, 1), 100.0, device=env.device)
        env.uav_velocity = torch.zeros((b, env.n_uavs, 3), device=env.device)
        env.uav_collisions = torch.zeros(
            (b, env.n_uavs, 1), dtype=torch.bool, device=env.device
        )

        # Initialize ground users outdoors
        env.ue_user_pos = env._env.gen_pos(
            num_pos=env.n_ues, min_z=1.5, max_z=1.5, outdoor=True
        )
        env.ue_battery = torch.full((b, env.n_ues, 1), 100.0, device=env.device)

        # Initialize dynamic ground user velocities
        env.ue_velocity = self.mobility_engine.initialize_velocities(b, env.n_ues)

        if self.mobility_model_type == "hotspot":
            self.mobility_engine.initialize_hotspots(b, num_hotspots=2)

        env.current_step = torch.zeros((b, 1), dtype=torch.int32, device=env.device)
        env.done = torch.zeros((b, 1), dtype=torch.bool, device=env.device)

        env.uav_ue_los = env._env.check_los_batch(env.uav_agents_pos, env.ue_user_pos)
        env.uav_ue_sojourn = torch.zeros((b, env.n_uavs, env.n_ues), device=env.device)

    def _reset_at(
        self, env, env_index: int, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets a single environment instance."""
        self._ensure_mobility_engine(env)

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

        new_vel = self.mobility_engine.initialize_velocities(1, env.n_ues)[0]
        if hasattr(env, "ue_velocity"):
            env.ue_velocity[env_index] = new_vel

        env.current_step[env_index] = torch.zeros(
            (1,), dtype=torch.int32, device=env.device
        )
        env.done[env_index] = torch.zeros((1,), dtype=torch.bool, device=env.device)

        env.uav_ue_los[env_index] = env._env.check_los_batch(
            env.uav_agents_pos[env_index : env_index + 1],
            env.ue_user_pos[env_index : env_index + 1],
        )[0]
        if hasattr(env, "uav_ue_sojourn"):
            env.uav_ue_sojourn[env_index].zero_()

    def process_actions(self, env, tensordict: TensorDictBase) -> None:
        """Applies agent actions and advances both UAVs and mobile ground UEs."""
        self._ensure_mobility_engine(env)

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

                # Battery dissipation
                horizontal_speed = torch.norm(
                    env.uav_velocity[..., :2], dim=-1, keepdim=True
                )
                vertical_speed = torch.abs(env.uav_velocity[..., 2:])
                propulsion_power = 0.1 * horizontal_speed + 0.2 * vertical_speed
                env.uav_battery -= propulsion_power * env.dt

        # Dynamic Ground User (UE) Mobility Step
        if not hasattr(env, "ue_velocity"):
            env.ue_velocity = self.mobility_engine.initialize_velocities(
                env.batch_size[0], env.n_ues
            )

        new_ue_pos, new_ue_vel = self.mobility_engine.step(
            ue_pos=env.ue_user_pos,
            ue_vel=env.ue_velocity,
            dt=env.dt,
            urban_map=getattr(env, "_env", None),
        )
        env.ue_user_pos = new_ue_pos
        env.ue_velocity = new_ue_vel

        # Recalculate LoS between new UAV positions and new mobile UE positions
        env.uav_ue_los = env._env.check_los_batch(env.uav_agents_pos, env.ue_user_pos)

        # Update Sojourn Time
        if not hasattr(env, "uav_ue_sojourn"):
            env.uav_ue_sojourn = torch.zeros(
                (env.batch_size[0], env.n_uavs, env.n_ues), device=env.device
            )

        env.uav_ue_sojourn = torch.where(
            env.uav_ue_los,
            env.uav_ue_sojourn + 1.0,
            torch.zeros_like(env.uav_ue_sojourn),
        )

    def observation_spec(self, env, group: str) -> Composite:
        """Returns observation spec for UAV agents: [x, y, z, battery, los_ratio, mean_sojourn]."""
        low = torch.tensor(
            [-env.volume_size[0] / 2, -env.volume_size[1] / 2, 0.0, 0.0, 0.0, 0.0],
            device=env.device,
        )
        high = torch.tensor(
            [
                env.volume_size[0] / 2,
                env.volume_size[1] / 2,
                env.volume_size[2],
                100.0,
                1.0,
                100.0,
            ],
            device=env.device,
        )
        return BoundedContinuous(low=low, high=high, shape=torch.Size([6]))

    def observation(self, env) -> torch.Tensor:
        """Observation tensor: [x, y, z, battery, los_ratio, mean_sojourn]."""
        pos = env.uav_agents_pos
        battery = env.uav_battery
        los_ratio = env.uav_ue_los.float().mean(dim=-1, keepdim=True)
        sojourn = (
            env.uav_ue_sojourn.mean(dim=-1, keepdim=True)
            if hasattr(env, "uav_ue_sojourn")
            else torch.zeros_like(los_ratio)
        )
        return torch.cat([pos, battery, los_ratio, sojourn], dim=-1)

    def action_spec(self, env, group: str) -> Composite:
        """Continuous velocity action spec: [v_h, phi, v_z]."""
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
        """Reward balancing mobile user LoS coverage, sojourn persistence, and collision avoidance."""
        los_ratio = env.uav_ue_los.float().mean(dim=2, keepdim=True)
        collision_penalty = env.uav_collisions.float()
        sojourn = (
            env.uav_ue_sojourn.mean(dim=2, keepdim=True)
            if hasattr(env, "uav_ue_sojourn")
            else torch.zeros_like(los_ratio)
        )
        sojourn_bonus = torch.tanh(0.1 * sojourn)

        reward = (
            (self.w_los * los_ratio)
            - (self.w_collision * collision_penalty)
            + (self.w_sojourn * sojourn_bonus)
        )
        return reward

    def done(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Termination flags."""
        terminated = env.uav_collisions.any(dim=1) | (env.uav_battery <= 0.0).any(dim=1)
        truncated = env.current_step >= env.max_steps
        dones = terminated | truncated
        return dones, terminated, truncated

    def state_spec(self, env) -> Optional[Composite]:
        """Global state spec for CTDE algorithms."""
        n_env_param = 4
        pos_dim = 3
        obs_dim = 6
        vel_dim = 3

        state_dim = (
            n_env_param + (env.n_uavs * obs_dim) + (env.n_ues * (pos_dim + vel_dim))
        )
        return Unbounded(
            shape=torch.Size([state_dim]), dtype=torch.float32, device=env.device
        )

    def state(self, env) -> Optional[torch.Tensor]:
        """Global state tensor including UAV and mobile UE states."""
        b = env.batch_size[0]
        n_env_param = 4

        obs = self.observation(env).view(b, -1)
        ue_pos = env.ue_user_pos.view(b, -1)
        ue_vel = (
            env.ue_velocity.view(b, -1)
            if hasattr(env, "ue_velocity")
            else torch.zeros_like(ue_pos)
        )
        urban_params = env._env.info[:, :n_env_param].view(b, -1)

        state = torch.cat([urban_params, obs, ue_pos, ue_vel], dim=-1)
        return state

    def info_agent_spec(self, env, group: str) -> Optional[Composite]:
        """Specifies agent-level info dictionary if enabled."""
        if not self.has_agent_info:
            return None
        return Composite(
            {
                "los_ratio": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "sojourn_time": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
            }
        )

    def info_agent(self, env, group: str) -> Optional[dict]:
        """Returns agent-level info dictionary if enabled."""
        if not self.has_agent_info:
            return None
        los_ratio = env.uav_ue_los.float().mean(dim=-1, keepdim=True)
        sojourn = (
            env.uav_ue_sojourn.mean(dim=-1, keepdim=True)
            if hasattr(env, "uav_ue_sojourn")
            else torch.zeros_like(los_ratio)
        )
        return {
            "los_ratio": los_ratio,
            "sojourn_time": sojourn,
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
                "los": Unbounded(
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
        """Global info dictionary for logging."""
        sojourn = (
            env.uav_ue_sojourn.mean(dim=(-1, -2)).unsqueeze(-1)
            if hasattr(env, "uav_ue_sojourn")
            else torch.zeros((env.batch_size[0], 1), device=env.device)
        )
        return {
            "urban_params": env._env.info[:, :4].view(env.batch_size[0], -1),
            "los": env.uav_ue_los.float().mean(dim=(-1, -2)).unsqueeze(-1),
            "sojourn_time": sojourn,
            "collisions": env.uav_collisions.float().sum(dim=1),
        }

    def _render(self, env, mode: str = "rgb_array") -> dict:
        """Returns renderable state payload including mobile user trajectories."""
        if not hasattr(self, "render_idx"):
            self.render_idx = torch.randint(
                0, env.batch_size[0], (1,), device=env.device
            ).item()
        #
        uav_positions = env.uav_agents_pos[self.render_idx].detach().cpu()
        ue_positions = env.ue_user_pos[self.render_idx].detach().cpu()
        los = env.uav_ue_los[self.render_idx].detach().cpu()

        los_links = []
        collisions = []

        for uav in range(env.n_uavs):
            if env.uav_collisions[self.render_idx, uav]:
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
        }
        return ndt
