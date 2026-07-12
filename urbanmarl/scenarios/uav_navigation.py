import torch
from .base import UrbanScenario
from torchrl.data import Composite, BoundedContinuous
from urbanmarl.envs.specs import unbatched_uav_action_spec, unbatched_uav_reward_spec

class Scenario(UrbanScenario):
    def __init__(self, config: dict):
        super().__init__(config)
        self.has_state = True
        # agent_info is velocity
        self.has_agent_info = False
        self.has_global_info = True
    
    
    def _reset_at(self, env, env_index, **kwargs):
        #
        env.uav_agents_pos[env_index] = env._env.gen_pos(
            num_pos=env.n_uavs, min_z=20.0, max_z=150.0, batch_idx=env_index, outdoor=True
        )
        env.uav_battery[env_index] = torch.ones((env.n_uavs, 1), device=env.device) * 100.0
        env.uav_velocity[env_index] = torch.zeros((env.n_uavs, 3), device=env.device)
        env.uav_collisions[env_index] = torch.zeros((env.n_uavs, 1), dtype=torch.bool, device=env.device)
        env.ue_user_pos[env_index] = env._env.gen_pos(
            num_pos=env.n_ues, min_z=1.5, max_z=1.5, batch_idx=env_index, outdoor=True
        )
        env.ue_battery[env_index] = torch.ones((env.n_ues, 1), device=env.device) * 100.0
        env.current_step[env_index] = torch.zeros((1), dtype=torch.int32, device=env.device)
        env.done[env_index] = torch.zeros((1), dtype=torch.bool, device=env.device)
        #
        env.uav_ue_los = env._env.check_los_batch(
                    env.uav_agents_pos, env.ue_user_pos
                )
        
    def _reset_all(self, env, tensordict=None, **kwargs):
        # Re‑initialise UAV and UE positions (same as in UrbanEnv._init_env)
        env.uav_agents_pos = env._env.gen_pos(
            num_pos=env.n_uavs, min_z=20.0, max_z=150.0, outdoor=True
        )
        env.uav_battery = torch.ones((env.batch_size[0], env.n_uavs, 1), device=env.device) * 100.0
        env.uav_velocity = torch.zeros((env.batch_size[0], env.n_uavs, 3), device=env.device)
        env.uav_collisions = torch.zeros((env.batch_size[0], env.n_uavs, 1), dtype=torch.bool, device=env.device)
        env.ue_user_pos = env._env.gen_pos(
            num_pos=env.n_ues, min_z=1.5, max_z=1.5, outdoor=True
        )
        env.ue_battery = torch.ones((env.batch_size[0], env.n_ues, 1), device=env.device) * 100.0
        env.current_step = torch.zeros((env.batch_size[0], 1), dtype=torch.int32, device=env.device)
        env.done = torch.zeros((env.batch_size[0], 1), dtype=torch.bool, device=env.device)
        #
        env.uav_ue_los = env._env.check_los_batch(
                    env.uav_agents_pos, env.ue_user_pos
                )
        
    def process_actions(self, env, tensordict):
        # Exactly the same as UrbanEnv._process_actions
        env.uav_collisions.zero_()
        for group, agent_names in env.group_map.items():
            group_action = tensordict.get((group, "action"))
            #
            if group.lower() == "uav" or group.lower() == "agents":
                # Denormalise
                # group_action[..., 0] = (group_action[..., 0] + 1.0) / 2.0 * env.max_h_speed
                # group_action[..., 1] = group_action[..., 1] * torch.pi
                # group_action[..., 2] = group_action[..., 2] * env.max_v_speed
                dx = group_action[..., 0] * torch.cos(group_action[..., 1])
                dy = group_action[..., 0] * torch.sin(group_action[..., 1])
                dz = group_action[..., 2]
                delta_pos = torch.stack([dx, dy, dz], dim=-1)
                env.previous_uav_pos = env.uav_agents_pos.clone()
                env.uav_agents_pos += delta_pos
                # Clamp
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
                # Battery consumption
                horizontal_speed = torch.norm(env.uav_velocity[..., :2], dim=-1, keepdim=True)
                vertical_speed = torch.abs(env.uav_velocity[..., 2:])
                power = 0.1 * horizontal_speed + 0.2 * vertical_speed
                env.uav_battery -= power * env.dt

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
        
    def observation(self, env):
        # Current observation: position (x,y,z) + battery
        return torch.cat([env.uav_agents_pos, env.uav_battery], dim=-1)

    def action_spec(self, env, group):
        from torchrl.data.tensor_specs import Bounded
        max_h_speed = float(env.max_h_speed)
        max_v_speed = float(env.max_v_speed)
        if group.lower() == "uav" or group.lower() == "agents":
            return Bounded(
                low=torch.tensor([0.0, -torch.pi, -max_v_speed], device=env.device),
                high=torch.tensor([max_h_speed, torch.pi, max_v_speed], device=env.device),
                shape=torch.Size([3]),
                dtype=torch.float32, 
                device=env.device
            )
            
        
    def done(self, env):
        terminated = env.uav_collisions.any(dim=1) | (env.uav_battery <= 0.0).any(dim=1)
        truncated = env.current_step >= env.max_steps
        dones = terminated | truncated
        return dones, terminated, truncated


    def reward_spec(self, env, group):
        from torchrl.data.tensor_specs import Unbounded
        if group.lower() == "uav" or group.lower() == "agents":
            return Unbounded(
                shape=torch.Size([1]), 
                dtype=torch.float32, 
                device=env.device
            )
    
    def reward(self, env):
        # Current reward: LoS ratio minus collision penalty
        los_ratio = env.uav_ue_los.float().mean(dim=2, keepdim=True)
        collision_ratio = env.uav_collisions.float()
        return los_ratio - collision_ratio
    
    
    
    def stat_spec(self, env):
        from torchrl.data.tensor_specs import Unbounded
        # State spec (global CTDE) – can also be scenario-defined, but keep as before
        n_env_param = 4 # alpha, beta, gamma
        pos_dim = 3
        observarion_dim = 4
        state_per_env_dim = (
            n_env_param +
            env.n_uavs * observarion_dim +
            env.n_ues * pos_dim
        )
        return Unbounded(
            shape=torch.Size([state_per_env_dim]),
            dtype=torch.float32,
            device=env.device,
        )
        
    def state(self, env):
        n_env_param = 4 # alpha, beta, gamma
        obs = env._get_obs()
        state = torch.cat([
            env._env.info[:, :n_env_param].view(env.batch_size[0], -1),
            obs.view(env.batch_size[0], -1),
            env.ue_user_pos.view(env.batch_size[0], -1)
        ], dim=-1)
        return state
    
    def info_global_spec(self, env):
        from torchrl.data.tensor_specs import Unbounded
        # 
        # alpha, beta, gamma, E
        info_spec_unbatched = Composite(
            {
                "collisions": Unbounded(
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "velocity": Unbounded(
                    shape=torch.Size([3]),
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
        return info_spec_unbatched
    
    def info_global(self, env):
        collision = env.uav_collisions.float().sum(dim=1)
        mean_velocity = torch.norm(env.uav_velocity, dim=-1)
        los = env.uav_ue_los.float().mean(dim=-1).mean(dim=-1, keepdim=True)
        return {
            "collisions": collision,
            "velocity": mean_velocity,
            "los": los,
        }
        
    

