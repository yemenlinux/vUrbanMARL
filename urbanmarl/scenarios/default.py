import torch
from .uav_navigation import Scenario as UAVNavigationScenario


class Scenario(UAVNavigationScenario):
    """Default scenario for UrbanMARL environment, inheriting full navigation dynamics."""
    def __init__(self, config: dict):
        super().__init__(config)
