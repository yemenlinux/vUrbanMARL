import torch
# from .base import UrbanScenario
# from torchrl.data import Composite, BoundedContinuous

from .uav_navigation import Scenario as UAVNavigationScenario
class Scenario(UAVNavigationScenario):
    """Scenario name: COVERAGE
    
    Objective: 
    Similar to UAV_NAVIGATION, but without global state information. 
    The UAVs are trained to navigate in an urban environment with the same dynamics 
    and constraints, while the reward function has additional penality for beeing closer
    than a certain threshold.
    
    
    
    Reward function: 
        LoS ratio - collision penalty. 
        The reward is calculated as the mean of the LoS (Line of Sight) ratio 
        between UAVs and UEs (User Equipments) minus a penalty for collisions. 
        The LoS ratio is computed as the mean of the `uav_ue_los` tensor along 
        the last dimension, while the collision penalty is derived from the 
        `uav_collisions` tensor. 
        Additionally, a distance penalty is applied for UAVs that are closer than 
        a specified threshold from each other. 
        The final reward is returned as a tensor of shape (batch_size, n_uavs, 1).

    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.has_state = False
        self.has_agent_info = False
        self.has_global_info = True
        #
        self.min_distance_threshold = config.get("min_distance_threshold", 100.0)
    
    def state_spec(self, env):
        return None 
    
    def state(self, env):
        pass
    
    def reward(self, env):
        # Calculate the mean LoS ratio
        los_ratio = env.uav_ue_los.float().mean(dim=2, keepdim=True)
        # Calculate the collision penalty
        collision_penalty = env.uav_collisions.float().sum(dim=-1, keepdim=True)
        # Calculate the distance penalty for UAVs that are too close to each other
        distance_penalty = torch.zeros_like(collision_penalty)
        for i in range(env.n_uavs):
            for j in range(i + 1, env.n_uavs):
                distance = torch.norm(env.uav_agents_pos[:, i] - env.uav_agents_pos[:, j], dim=-1, keepdim=True)
                distance_penalty[:, i] += (distance < self.min_distance_threshold).float()
        # Combine the rewards and penalties
        reward = los_ratio - collision_penalty - 5 * distance_penalty
        return reward
    
    

