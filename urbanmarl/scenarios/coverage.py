import torch
from .base import UrbanScenario
from torchrl.data import Composite, BoundedContinuous

class Scenario(UrbanScenario):
    def reset(self, env, tensordict=None, **kwargs):
        # Same as default, but maybe spread UAVs further apart
        env.uav_agents_pos = env._env.gen_pos(
            num_pos=env.n_uavs, 
            min_z=30.0, 
            max_z=120.0, 
            outdoor=True
        )
        env.uav_battery = torch.ones((env.batch_size[0], env.n_uavs, 1), 
                                    device=env.device) * 100.0
        env.uav_velocity = torch.zeros((env.batch_size[0], env.n_uavs, 3), 
                                    device=env.device)
        env.uav_collisions = torch.zeros((env.batch_size[0], env.n_uavs, 1), 
                                         dtype=torch.bool, device=env.device)
        env.ue_user_pos = env._env.gen_pos(
            num_pos=env.n_ues, 
            min_z=1.5, 
            max_z=1.5, 
            outdoor=True
        )
        env.ue_battery = torch.ones((env.batch_size[0], env.n_ues, 1), 
                                    device=env.device) * 100.0
        env.current_step = torch.zeros((env.batch_size[0], 1), 
                                       dtype=torch.int32, device=env.device)
        env.done = torch.zeros((env.batch_size[0], 1), 
                               dtype=torch.bool, device=env.device)

    def process_actions(self, env, tensordict):
        # Same action processing as default
        env.uav_collisions.zero_()
        for group, agent_names in env.group_map.items():
            group_action = tensordict.get((group, "action"))
            if group == "agents":
                # 1. Denormalize action parameters
                # horizontal distance
                group_action[..., 0] = (group_action[..., 0] + 1.0) / 2.0 * env.max_h_speed
                # angle [-1, 1] -> [-pi, pi]
                group_action[..., 1] = group_action[..., 1] * torch.pi
                # vertical distance
                group_action[..., 2] = group_action[..., 2] * env.max_v_speed
                # update horizontal distance
                dx = group_action[..., 0] * torch.cos(group_action[..., 1])
                dy = group_action[..., 0] * torch.sin(group_action[..., 1])
                dz = group_action[..., 2]
                delta_pos = torch.stack([dx, dy, dz], dim=-1)
                env.previous_uav_pos = env.uav_agents_pos.clone()
                env.uav_agents_pos += delta_pos
                env.uav_agents_pos[..., 0] = torch.clamp(
                    env.uav_agents_pos[..., 0],
                    -env.volume_size[0]/2, env.volume_size[0]/2
                )
                env.uav_agents_pos[..., 1] = torch.clamp(
                    env.uav_agents_pos[..., 1],
                    -env.volume_size[1]/2, env.volume_size[1]/2
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
                horizontal_speed = torch.norm(env.uav_velocity[..., :2], dim=-1, keepdim=True)
                vertical_speed = torch.abs(env.uav_velocity[..., 2:])
                power = 0.1 * horizontal_speed + 0.2 * vertical_speed
                env.uav_battery -= power * env.dt

    def reward(self, env):
        # Reward: cover as many UEs as possible (LoS) and avoid collisions.
        # We'll reward each UAV for the number of UEs it has LoS to, penalise collisions.
        los_count = env.uav_ue_los.float().sum(dim=2, keepdim=True)  # (B, n_uavs, 1)
        collision_penalty = env.uav_collisions.float() * 10.0
        return los_count / env.n_ues - collision_penalty

    def observation(self, env):
        # Include own position, battery, and relative positions to other UAVs and UEs?
        # For simplicity, keep same as default but could be expanded.
        return torch.cat([env.uav_agents_pos, env.uav_battery], dim=-1)

    def done(self, env):
        # Same as default
        terminated = env.uav_collisions.any(dim=1) | (env.uav_battery <= 0.0).any(dim=1)
        truncated = env.current_step >= env.max_steps
        dones = terminated | truncated
        return dones, terminated, truncated

    def observation_spec(self, env, group):
        # Same spec as default (position+battery)
        low = torch.tensor([
            -env.volume_size[0]/2, -env.volume_size[1]/2, 0.0, 0.0
        ], device=env.device)
        high = torch.tensor([
            env.volume_size[0]/2, env.volume_size[1]/2, env.volume_size[2], 100.0
        ], device=env.device)
        if group.lower() == "uav" or group.lower() == "agents":
            return BoundedContinuous(low=low, high=high, shape=torch.Size([4]))
        # return Composite(
        #     {"observation": BoundedContinuous(low=low, high=high, shape=torch.Size([4]))}
        # )

    def action_spec(self, env, group):
        from torchrl.data.tensor_specs import Bounded
        if group.lower() == "uav" or group.lower() == "agents":
            return Bounded(
                low=torch.tensor([-1.0, -1.0, -1.0], device=env.device),
                high=torch.tensor([1.0, 1.0, 1.0], device=env.device),
                shape=torch.Size([3]),
                dtype=torch.float32, 
                device=env.device
            )
        # from urbanmarl.envs.specs import unbatched_uav_action_spec
        # return Composite({"action": unbatched_uav_action_spec(device=env.device)})
    
    def reward_spec(self, env, group):
        from torchrl.data.tensor_specs import Unbounded
        
        if group.lower() == "uav" or group.lower() == "agents":
            return Unbounded(
                # low=-1.0, 
                # high=1.0, 
                shape=torch.Size([1]), 
                dtype=torch.float32, 
                device=env.device
            )
    
    def stat_spec(self, env):
        from torchrl.data.tensor_specs import Unbounded
        # State spec (global CTDE) – can also be scenario-defined, but keep as before
        n_env_param = 3 # alpha, beta, gamma
        pos_dim = 3
        observarion_dim = 4
        state_per_env_dim = (n_env_param 
                             + env.n_uavs * observarion_dim 
                             + env.n_ues * pos_dim)
        return Unbounded(
            shape=torch.Size([state_per_env_dim]),
            dtype=torch.float32,
            device=env.device,
        )
        
