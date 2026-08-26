import torch
from .uav_navigation import Scenario as UAVNavigationScenario


class Scenario(UAVNavigationScenario):
    """UAV MEC Offloading scenario for UrbanMARL environment."""
    def __init__(self, config: dict):
        super().__init__(config)
