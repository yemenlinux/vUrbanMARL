# import torch
# from .base import UrbanScenario
# from torchrl.data import Composite, BoundedContinuous

from .uav_navigation import Scenario as UAVNavigationScenario
class Scenario(UAVNavigationScenario):
    """Scenario name: UAV_UE_LOS
    
    Objective: 
    Similar to UAV_NAVIGATION, but without global state information. 
    The UAVs are trained to navigate in an urban environment with the same dynamics 
    and constraints, and reward function.
    
    
    
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
        self.has_state = False
        self.has_agent_info = False
        self.has_global_info = True
    
    def state_spec(self, env):
        return None 
    
    def state(self, env):
        pass
    
    
    
    

