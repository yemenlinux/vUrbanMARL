import torch
from .base import UrbanScenario
from torchrl.data import Composite, BoundedContinuous

class Scenario(UrbanScenario):
    """Scenario name: UAV_NAVIGATION
    
    Objective: 
    We assume that multi-UAV assest base stations to improve the coverage
    of a wireless network. A reward function based on line of sight ratio and collision penality
    is designed to encourage the UAV agents to navigate the simulation 3D volume space 
    to maximize the return value.
    
    Reward function: 
        LoS ratio - collision penalty. 
        The reward is calculated as the mean of the LoS (Line of Sight) ratio 
        between UAVs and UEs (User Equipments) minus a penalty for collisions. 
        The LoS ratio is computed as the mean of the `uav_ue_los` tensor along 
        the last dimension, while the collision penalty is derived from the 
        `uav_collisions` tensor. 
        The final reward is returned as a tensor of shape (batch_size, n_uavs, 1).

    """
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
                low=torch.tensor([-max_h_speed, -torch.pi, -max_v_speed], device=env.device),
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
        """calculate the reward function per UAV-agent.

        Args:
            env (UrbanEnv): urbanMARL environment

        Returns:
            reward (torch.Tensor): reward tensor of shape (batch_size, n_uavs, 1)
            
        Reward function: LoS ratio - collision penalty. 
        The reward is calculated as the mean of the LoS (Line of Sight) ratio 
        between UAVs and UEs (User Equipments) minus a penalty for collisions. 
        The LoS ratio is computed as the mean of the `uav_ue_los` tensor along 
        the last dimension, while the collision penalty is derived from the 
        `uav_collisions` tensor. 
        The final reward is returned as a tensor of shape (batch_size, n_uavs, 1).
        """
        # Current reward: LoS ratio minus collision penalty
        los_ratio = env.uav_ue_los.float().mean(dim=2, keepdim=True)
        collision_ratio = env.uav_collisions.float()
        return los_ratio - collision_ratio
    
    def state_spec(self, env):
        from torchrl.data.tensor_specs import Unbounded
        # State spec (global CTDE) – can also be scenario-defined, but keep as before
        n_env_param = 4 # alpha, beta, gamma, E
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
        n_env_param = 4 # alpha, beta, gamma, E
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
        n_env_param = 4 # alpha, beta, gamma, E
        info_spec_unbatched = Composite(
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
        n_env_param = 4 # alpha, beta, gamma, E
        collision = env.uav_collisions.float().sum(dim=1)
        mean_velocity = torch.norm(env.uav_velocity, dim=-1)
        los = env.uav_ue_los.float().mean(dim=-1).mean(dim=-1, keepdim=True)
        return {
            "urban_params": env._env.info[:, :n_env_param].view(env.batch_size[0], -1),
            "collisions": collision,
            "velocity": mean_velocity,
            "los": los,
        }
        
    def _render(self, env, mode='rgb_array'):
        """Reterns renderable information for network digital twins.
        Args:
            env (UrbanEnv): urbanMARL environment
            mode (str, optional): The rendering mode.
                Available modes are 'rgb_array' and 'human'.
                Defaults to 'rgb_array'.

        Returns:
            _type_: _description_
        """
        if not hasattr(self, 'render_idx'):
            self.render_idx = np.random.randint(env.batch_size[0])
        #
        uav_positions = env.uav_agents_pos[self.render_idx].cpu()
        ue_positions = env.ue_user_pos[self.render_idx].cpu()
        los = env.uav_ue_los[self.render_idx]
        los_links = []
        collisions = []
        if hasattr(env, "uav_ue_los"):
            for uav in range(env.n_uavs):
                if hasattr(env, "uav_collisions") and env.uav_collisions[self.render_idx, uav]:
                    collisions.append(uav_positions[uav])
                for ue in range(env.n_ues):
                    los_links.append({
                        'source': uav_positions[uav], 
                        'target': ue_positions[ue], 
                        'los': los[uav, ue]
                        })
        # 
        ndt = {
            'uav_positions': uav_positions,
            'ue_positions': ue_positions,
            'links': los_links,
            'collisions': collisions
        }
        return ndt

