"""UrbanMARL LiDAR-Equipped Autonomous UAV Navigation Scenario.

NEW standalone scenario providing POMDP resolution through 360-degree virtual LiDAR rangefinding:

* Each UAV is equipped with a vectorized 8-beam radial LiDAR sensor.
* UAVs observe their 3D coordinates, battery, relative target destination vector,
  and normalized obstacle proximity clearance in 8 azimuth directions.
* UAVs receive continuous target-approach rewards, obstacle proximity margin penalties,
  and goal arrival bonuses.
"""

from typing import Optional, Tuple

import torch
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded

from urbanmarl.models.lidar import VectorizedLiDAR
from urbanmarl.scenarios.base import UrbanScenario


class Scenario(UrbanScenario):
    """Scenario name: UAV_LIDAR_NAVIGATION.

    Objective:
    Decentralized collision-free navigation through complex 3D urban building canyons
    using 360-degree LiDAR proximity sensing and target waypoint guidance.
    """

    def __init__(self, config: dict) -> None:
        """Initializes the UAV LiDAR Navigation scenario.

        Args:
            config (dict): Scenario configuration dictionary.
        """
        super().__init__(config)
        self.has_state = config.get("has_state", True)
        self.has_agent_info = config.get("has_agent_info", False)
        self.has_global_info = config.get("has_global_info", True)

        self.num_lidar_beams = config.get("num_lidar_beams", 8)
        self.lidar_max_range = config.get("lidar_max_range", 60.0)

        self.w_dist = config.get("w_dist", 1.0)
        self.w_reached = config.get("w_reached", 5.0)
        self.w_collision = config.get("w_collision", 5.0)
        self.w_proximity = config.get("w_proximity", 1.0)

        self.lidar: Optional[VectorizedLiDAR] = None

    def _ensure_lidar(self, env) -> None:
        """Lazily initializes the LiDAR sensor on the environment's device."""
        if self.lidar is None or self.lidar.device != env.device:
            self.lidar = VectorizedLiDAR(
                num_beams=self.num_lidar_beams,
                max_range=self.lidar_max_range,
                device=env.device,
            )

    def _reset_all(
        self, env, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets all environments in the batch."""
        self._ensure_lidar(env)
        b = env.batch_size[0]

        env.uav_agents_pos = env._env.gen_pos(
            num_pos=env.n_uavs, min_z=20.0, max_z=120.0, outdoor=True
        )
        env.uav_target_pos = env._env.gen_pos(
            num_pos=env.n_uavs, min_z=20.0, max_z=120.0, outdoor=True
        )

        env.uav_battery = torch.full((b, env.n_uavs, 1), 100.0, device=env.device)
        env.uav_velocity = torch.zeros((b, env.n_uavs, 3), device=env.device)
        env.uav_collisions = torch.zeros(
            (b, env.n_uavs, 1), dtype=torch.bool, device=env.device
        )
        env.uav_reached = torch.zeros(
            (b, env.n_uavs, 1), dtype=torch.bool, device=env.device
        )

        # Ground users if any
        env.ue_user_pos = env._env.gen_pos(
            num_pos=env.n_ues, min_z=1.5, max_z=1.5, outdoor=True
        )
        env.ue_battery = torch.full((b, env.n_ues, 1), 100.0, device=env.device)

        env.current_step = torch.zeros((b, 1), dtype=torch.int32, device=env.device)
        env.done = torch.zeros((b, 1), dtype=torch.bool, device=env.device)

        # Initial LiDAR scan
        env.uav_lidar_ranges = self.lidar.scan(
            env.uav_agents_pos, getattr(env, "_env", None)
        )

    def _reset_at(
        self, env, env_index: int, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets a single environment instance."""
        self._ensure_lidar(env)

        env.uav_agents_pos[env_index] = env._env.gen_pos(
            num_pos=env.n_uavs,
            min_z=20.0,
            max_z=120.0,
            batch_idx=env_index,
            outdoor=True,
        )
        env.uav_target_pos[env_index] = env._env.gen_pos(
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
        env.uav_reached[env_index] = torch.zeros(
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

        env.uav_lidar_ranges[env_index : env_index + 1] = self.lidar.scan(
            env.uav_agents_pos[env_index : env_index + 1],
            getattr(env, "_env", None),
        )

    def process_actions(self, env, tensordict: TensorDictBase) -> None:
        """Processes agent flight actions and computes new LiDAR scans."""
        self._ensure_lidar(env)

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
                h_speed = torch.norm(env.uav_velocity[..., :2], dim=-1, keepdim=True)
                v_speed = torch.abs(env.uav_velocity[..., 2:])
                propulsion_power = 0.1 * h_speed + 0.2 * v_speed
                env.uav_battery -= propulsion_power * env.dt

        # Check target arrival (distance < 5.0 meters)
        target_dist = torch.norm(
            env.uav_agents_pos - env.uav_target_pos, dim=-1, keepdim=True
        )
        env.uav_reached = target_dist < 5.0

        # Respawn target if reached
        if env.uav_reached.any():
            new_targets = env._env.gen_pos(
                num_pos=env.n_uavs, min_z=20.0, max_z=120.0, outdoor=True
            )
            env.uav_target_pos = torch.where(
                env.uav_reached.expand(-1, -1, 3), new_targets, env.uav_target_pos
            )

        # Update 360-degree LiDAR proximity readings
        env.uav_lidar_ranges = self.lidar.scan(
            env.uav_agents_pos, getattr(env, "_env", None)
        )

    def observation_spec(self, env, group: str) -> Composite:
        """Observation specification: [x, y, z, battery, dx_target, dy_target, dz_target, lidar_8]."""
        dim = 4 + 3 + self.num_lidar_beams
        return Unbounded(
            shape=torch.Size([dim]), dtype=torch.float32, device=env.device
        )

    def observation(self, env) -> torch.Tensor:
        """Constructs observation tensor normalized to approximately [-1, 1]."""
        pos_norm = env.uav_agents_pos / torch.tensor(
            [env.volume_size[0] / 2, env.volume_size[1] / 2, env.volume_size[2]],
            device=env.device,
        )
        battery_norm = env.uav_battery / 100.0

        target_disp = env.uav_target_pos - env.uav_agents_pos
        target_disp_norm = target_disp / torch.tensor(
            [env.volume_size[0] / 2, env.volume_size[1] / 2, env.volume_size[2]],
            device=env.device,
        )

        lidar_readings = (
            env.uav_lidar_ranges
            if hasattr(env, "uav_lidar_ranges")
            else torch.ones(
                (env.batch_size[0], env.n_uavs, self.num_lidar_beams), device=env.device
            )
        )

        return torch.cat(
            [pos_norm, battery_norm, target_disp_norm, lidar_readings], dim=-1
        )

    def action_spec(self, env, group: str) -> Composite:
        """Continuous action: [v_h, phi, v_z]."""
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
        """Reward specification."""
        return Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=env.device)

    def reward(self, env, group: str) -> torch.Tensor:
        """Reward incorporating distance reduction, goal arrival, proximity, and collision."""
        target_dist = torch.norm(
            env.uav_agents_pos - env.uav_target_pos, dim=-1, keepdim=True
        )
        max_dist = float(
            (
                env.volume_size[0] ** 2
                + env.volume_size[1] ** 2
                + env.volume_size[2] ** 2
            )
            ** 0.5
        )
        dist_penalty = target_dist / max_dist

        goal_reward = env.uav_reached.float() * self.w_reached
        collision_penalty = env.uav_collisions.float() * self.w_collision

        # Proximity penalty: active if minimum LiDAR range drops below 0.25 (too close to facade)
        if hasattr(env, "uav_lidar_ranges"):
            min_lidar_range = env.uav_lidar_ranges.min(dim=-1, keepdim=True)[0]
            proximity_penalty = (
                torch.clamp(0.25 - min_lidar_range, min=0.0) * self.w_proximity
            )
        else:
            proximity_penalty = torch.zeros_like(collision_penalty)

        reward = (
            -self.w_dist * dist_penalty
            + goal_reward
            - collision_penalty
            - proximity_penalty
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
        obs_dim = 4 + 3 + self.num_lidar_beams
        state_dim = n_env_param + (env.n_uavs * obs_dim) + (env.n_uavs * 3)
        return Unbounded(
            shape=torch.Size([state_dim]), dtype=torch.float32, device=env.device
        )

    def state(self, env) -> Optional[torch.Tensor]:
        """Global state tensor."""
        b = env.batch_size[0]
        n_env_param = 4
        obs = self.observation(env).view(b, -1)
        targets = env.uav_target_pos.view(b, -1)
        urban_params = env._env.info[:, :n_env_param].view(b, -1)
        return torch.cat([urban_params, obs, targets], dim=-1)

    def info_agent_spec(self, env, group: str) -> Optional[Composite]:
        """Agent-level info spec if enabled."""
        if not self.has_agent_info:
            return None
        return Composite(
            {
                "target_dist": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "min_lidar_clearance": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
            }
        )

    def info_agent(self, env, group: str) -> Optional[dict]:
        """Agent-level info dictionary."""
        if not self.has_agent_info:
            return None
        target_dist = torch.norm(
            env.uav_agents_pos - env.uav_target_pos, dim=-1, keepdim=True
        )
        min_clearance = (
            env.uav_lidar_ranges.min(dim=-1, keepdim=True)[0]
            if hasattr(env, "uav_lidar_ranges")
            else torch.ones_like(target_dist)
        )
        return {
            "target_dist": target_dist,
            "min_lidar_clearance": min_clearance,
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
                "reached_goals": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "collisions": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
                "mean_clearance": Unbounded(
                    shape=torch.Size([1]), dtype=torch.float32, device=env.device
                ),
            }
        )

    def info_global(self, env) -> Optional[dict]:
        """Global info dictionary."""
        n_env_param = 4  # alpha, beta, gamma, E
        clearance = (
            env.uav_lidar_ranges.mean(dim=(-1, -2)).unsqueeze(-1)
            if hasattr(env, "uav_lidar_ranges")
            else torch.ones((env.batch_size[0], 1), device=env.device)
        )
        return {
            "urban_params": env._env.info[:, :n_env_param].view(env.batch_size[0], -1),
            "reached_goals": env.uav_reached.float().sum(dim=1),
            "collisions": env.uav_collisions.float().sum(dim=1),
            "mean_clearance": clearance,
        }

    def _render(self, env, mode: str = "rgb_array") -> dict:
        """Render payload."""
        if not hasattr(self, "render_idx"):
            self.render_idx = torch.randint(
                0, env.batch_size[0], (1,), device=env.device
            ).item()

        uav_positions = env.uav_agents_pos[self.render_idx].detach().cpu()
        targets = env.uav_target_pos[self.render_idx].detach().cpu()
        collisions = []

        for uav in range(env.n_uavs):
            if env.uav_collisions[self.render_idx, uav]:
                collisions.append(uav_positions[uav])

        return {
            "uav_positions": uav_positions,
            "base_station_positions": targets,
            "collisions": collisions,
        }
