from typing import Optional, Tuple

import torch
from tensordict import TensorDictBase
from torchrl.data import BoundedContinuous, Composite, Unbounded
from torchrl.data.tensor_specs import Bounded

from .base import UrbanScenario


class Scenario(UrbanScenario):
    """MEC Offloading scenario for UrbanMARL environments.

    UAV base stations navigate an urban 3D environment to provide computation
    offloading coverage to ground user equipments (UEs) while monitoring
    battery levels and avoiding obstacle collisions.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config or {})
        self.has_state = True
        self.has_agent_info = False
        self.has_global_info = True

    def _reset_at(
        self, env, env_index: int, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        env.uav_agents_pos[env_index] = env._env.gen_pos(
            num_pos=env.n_uavs,
            min_z=20.0,
            max_z=150.0,
            batch_idx=env_index,
            outdoor=True,
        )
        env.uav_battery[env_index] = (
            torch.ones((env.n_uavs, 1), device=env.device) * 100.0
        )
        env.uav_velocity[env_index] = torch.zeros((env.n_uavs, 3), device=env.device)
        env.uav_collisions[env_index] = torch.zeros(
            (env.n_uavs, 1), dtype=torch.bool, device=env.device
        )
        env.ue_user_pos[env_index] = env._env.gen_pos(
            num_pos=env.n_ues, min_z=1.5, max_z=1.5, batch_idx=env_index, outdoor=True
        )
        env.ue_battery[env_index] = (
            torch.ones((env.n_ues, 1), device=env.device) * 100.0
        )
        env.current_step[env_index] = torch.zeros(
            (1,), dtype=torch.int32, device=env.device
        )
        env.done[env_index] = torch.zeros((1,), dtype=torch.bool, device=env.device)
        env.uav_ue_los = env._env.check_los_batch(env.uav_agents_pos, env.ue_user_pos)

    def _reset_all(
        self, env, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        env.uav_agents_pos = env._env.gen_pos(
            num_pos=env.n_uavs, min_z=20.0, max_z=150.0, outdoor=True
        )
        env.uav_battery = (
            torch.ones((env.batch_size[0], env.n_uavs, 1), device=env.device) * 100.0
        )
        env.uav_velocity = torch.zeros(
            (env.batch_size[0], env.n_uavs, 3), device=env.device
        )
        env.uav_collisions = torch.zeros(
            (env.batch_size[0], env.n_uavs, 1), dtype=torch.bool, device=env.device
        )
        env.ue_user_pos = env._env.gen_pos(
            num_pos=env.n_ues, min_z=1.5, max_z=1.5, outdoor=True
        )
        env.ue_battery = (
            torch.ones((env.batch_size[0], env.n_ues, 1), device=env.device) * 100.0
        )
        env.current_step = torch.zeros(
            (env.batch_size[0], 1), dtype=torch.int32, device=env.device
        )
        env.done = torch.zeros(
            (env.batch_size[0], 1), dtype=torch.bool, device=env.device
        )
        env.uav_ue_los = env._env.check_los_batch(env.uav_agents_pos, env.ue_user_pos)

    def process_actions(self, env, tensordict: TensorDictBase) -> None:
        env.uav_collisions.zero_()
        for group, _agent_names in env.group_map.items():
            group_action = tensordict.get((group, "action"))
            if group.lower() in ("uav", "agents"):
                # Denormalise
                speed_h = (group_action[..., 0] + 1.0) / 2.0 * env.max_h_speed
                phi = group_action[..., 1] * torch.pi
                speed_v = group_action[..., 2] * env.max_v_speed
                dx = speed_h * torch.cos(phi)
                dy = speed_h * torch.sin(phi)
                dz = speed_v
                delta_pos = torch.stack([dx, dy, dz], dim=-1)
                env.previous_uav_pos = env.uav_agents_pos.clone()
                env.uav_agents_pos += delta_pos
                # Clamp within environment boundaries
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
                # Battery consumption
                horizontal_speed = torch.norm(
                    env.uav_velocity[..., :2], dim=-1, keepdim=True
                )
                vertical_speed = torch.abs(env.uav_velocity[..., 2:])
                power = 0.1 * horizontal_speed + 0.2 * vertical_speed
                env.uav_battery -= power * env.dt

    def reward(self, env, group: Optional[str] = None) -> torch.Tensor:
        # Current reward: LoS ratio minus collision penalty
        los_ratio = env.uav_ue_los.float().mean(dim=2, keepdim=True)
        collision_ratio = env.uav_collisions.float()
        return los_ratio - collision_ratio

    def observation(self, env) -> torch.Tensor:
        # Current observation: position (x,y,z) + battery
        return torch.cat([env.uav_agents_pos, env.uav_battery], dim=-1)

    def done(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        terminated = env.uav_collisions.any(dim=1) | (env.uav_battery <= 0.0).any(dim=1)
        truncated = env.current_step >= env.max_steps
        dones = terminated | truncated
        return dones, terminated, truncated

    def observation_spec(self, env, group: str) -> BoundedContinuous:
        low = torch.tensor(
            [-env.volume_size[0] / 2, -env.volume_size[1] / 2, 0.0, 0.0],
            device=env.device,
        )
        high = torch.tensor(
            [env.volume_size[0] / 2, env.volume_size[1] / 2, env.volume_size[2], 100.0],
            device=env.device,
        )
        return BoundedContinuous(low=low, high=high, shape=torch.Size([4]))

    def action_spec(self, env, group: str) -> Bounded:
        return Bounded(
            low=torch.tensor([-1.0, -1.0, -1.0], device=env.device),
            high=torch.tensor([1.0, 1.0, 1.0], device=env.device),
            shape=torch.Size([3]),
            dtype=torch.float32,
            device=env.device,
        )

    def reward_spec(self, env, group: str) -> Unbounded:
        return Unbounded(
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=env.device,
        )

    def state_spec(self, env) -> Unbounded:
        n_env_param = 4  # alpha, beta, gamma, E
        pos_dim = 3
        observation_dim = 4
        state_per_env_dim = (
            n_env_param + env.n_uavs * observation_dim + env.n_ues * pos_dim
        )
        return Unbounded(
            shape=torch.Size([state_per_env_dim]),
            dtype=torch.float32,
            device=env.device,
        )

    def state(self, env) -> torch.Tensor:
        n_env_param = 4  # alpha, beta, gamma, E
        obs = self.observation(env)
        return torch.cat(
            [
                env._env.info[:, :n_env_param].view(env.batch_size[0], -1),
                obs.view(env.batch_size[0], -1),
                env.ue_user_pos.view(env.batch_size[0], -1),
            ],
            dim=-1,
        )

    def info_global_spec(self, env) -> Optional[Composite]:
        n_env_param = 4  # alpha, beta, gamma, E
        return Composite(
            {
                "urban_params": Unbounded(
                    shape=torch.Size([n_env_param]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "collisions": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "velocity": Unbounded(
                    shape=torch.Size([env.n_uavs]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "los": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "mean_battery": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
            }
        )

    def info_global(self, env) -> Optional[dict]:
        n_env_param = 4  # alpha, beta, gamma, E
        return {
            "urban_params": env._env.info[:, :n_env_param].view(env.batch_size[0], -1),
            "collisions": env.uav_collisions.float().sum(dim=1),
            "velocity": torch.norm(env.uav_velocity, dim=-1),
            "los": env.uav_ue_los.float().mean(dim=(-1, -2)).unsqueeze(-1),
            "mean_battery": env.uav_battery.mean(dim=1),
        }

    def _render(self, env, mode: str = "rgb_array") -> dict:
        """Returns renderable information for network digital twins."""
        if not hasattr(self, "render_idx"):
            self.render_idx = 0

        uav_positions = env.uav_agents_pos[self.render_idx].cpu()
        ue_positions = env.ue_user_pos[self.render_idx].cpu()
        los = env.uav_ue_los[self.render_idx]
        los_links = []
        collisions = []
        if hasattr(env, "uav_ue_los"):
            for uav in range(env.n_uavs):
                if (
                    hasattr(env, "uav_collisions")
                    and env.uav_collisions[self.render_idx, uav]
                ):
                    collisions.append(uav_positions[uav])
                for ue in range(env.n_ues):
                    los_links.append(
                        {
                            "source": uav_positions[uav],
                            "target": ue_positions[ue],
                            "los": los[uav, ue],
                        }
                    )
        return {
            "uav_positions": uav_positions,
            "ue_positions": ue_positions,
            "links": los_links,
            "collisions": collisions,
        }
