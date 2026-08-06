from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from tensordict import TensorDictBase
from torchrl.data import Composite
import numpy as np

from urbanmarl.envs.rendering import Urban3DRenderer, UrbanRenderConfig

class UrbanScenario(ABC):
    """Base class for all urban scenarios.
    
    A scenario defines:
        - the specifications of the environment:
            - local observation space per agent per environment (unbatched)
            - global state space per environment (unbatched)
            - action space per agent per environment (unbatched)
            - info space per agent per environment (unbatched)
            - reward space per agent per environment (unbatched)
            - done / terminated / truncated flags per environment (unbatched)
        - Scenario controls the dynamics of the environment:
            - initial placement of agents (reset)
            - action processing and state update (step)
            - reward computation
            - observation formation
            - done / terminated / truncated flags
            - observation and action specs (for TorchRL)
    """
    
    def __init__(self, config: dict):
        """Initialize the scenario with the given configuration.
        Args:
            config (dict): configuration dictionary for the scenario.
        """
        self.config = config
        self.has_state = config.get("has_state", False)
        self.has_agent_info = config.get("has_agent_info", False)
        self.has_global_info = config.get("has_global_info", False)

    def reset(self, env, tensordict=None, **kwargs):
        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset").squeeze(-1)
            if _reset.all():
                env._env.reset(return_urban_info=False)
                self._reset_all(env, **kwargs)
                
            else:
                for env_index, to_reset in enumerate(_reset):
                    if to_reset:
                        env._env.reset_at(env_index, return_urban_info=False)
                        self._reset_at(env, env_index, **kwargs)
        else:
            env._env.reset(return_urban_info=False)
            self._reset_all(env, **kwargs)
    
    @abstractmethod
    def _reset_all(self, env, tensordict: Optional[TensorDictBase] = None, **kwargs):
        """Reset the environment's internal state (positions, battery, etc.)."""
        pass
    
    @abstractmethod
    def _reset_at(self, env, tensordict: Optional[TensorDictBase] = None, **kwargs):
        """Reset the environment's internal state (positions, battery, etc.)."""
        pass

    @abstractmethod
    def process_actions(self, env, tensordict: TensorDictBase):
        """Apply actions, update positions, battery, collision flags, etc."""
        pass

    @abstractmethod
    def reward(self, env, group) -> torch.Tensor:
        """Return reward tensor of shape (batch_size, 1)."""
        pass

    @abstractmethod
    def observation(self, env) -> torch.Tensor:
        """Return observation tensor of shape (batch_size, n_agents, obs_dim)."""
        pass

    @abstractmethod
    def done(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (done, terminated, truncated), each bool of shape (batch_size, 1)."""
        pass

    @abstractmethod
    def observation_spec(self, env, group) -> Composite:
        """Return the unbatched observation spec for this scenario."""
        pass

    @abstractmethod
    def action_spec(self, env, group) -> Composite:
        """Return the unbatched action spec for this scenario."""
        pass
    
    def state(self, env):
        pass
    
    def info_global_spec(self, env) -> Optional[Composite]:
        """Info for urban parameters 
        (alpha, beta, gamma, E) and other global info.
        
        Args:
            env (UrbanEnv): The UrbanEnv instance.
        Return:
        an unbatched Composite spec for the global info, or None if not applicable.
        
        This spec is important for logging and analysis of the environment's 
        impact on network digital twins, such as LoS, velocity, 
        communication channel, etc.
        To analyze the impact of urban parameters on network performance,
        the global info spec should include the urban parameters and any other
        relevant global information that can be used for logging and analysis.
        
        For example, the global info spec can be defined as:
        
        from torchrl.data import Composite
        from torchrl.data.tensor_specs import Unbounded
        n_env_param = 4 # alpha, beta, gamma, E
        info_spec_unbatched = Composite(
            {
                # mandatory for urban analysis: urban parameters (alpha, beta, gamma, E)
                "urban_params": Unbounded(
                    shape=torch.Size([n_env_param]),
                    dtype=torch.float32,
                    device=env.device,
                ),
                # optional for network digital twin analysis: other global info
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
        """
        return None
    
    def info_global(self, env):
        """Return a dictionary of global info for logging and analysis.
        """
        pass

    # Optional: info_spec if needed
    def info_agent_spec(self, env, group) -> Optional[Composite]:
        return None
    
    def info_agent(self, env, group) -> Optional[Composite]:
        return None 
    
    
    
    # Optional: reward_spec if needed
    def reward_spec(self, env, group) -> Optional[Composite]:
        return None 
    
    # Optional: state_spec if needed for global state
    def state_spec(self, env):
        return None
    
    # rendering / visualization
    def render(
        self, 
        env,
        algorithm: str = "random",
        mode = 'rgb_array'
    ):
        """Render UrbanMARL
        Args:
            env (UrbanEnv): urbanMARL environment
            algorithm (str): The algorithm name for labeling the render.
            mode (str, optional): The rendering mode.
                Available modes are 'rgb_array' and 'human'.
                Defaults to 'rgb_array'.

        Return:
            np.ndarray: The rendered image as a numpy array if mode is 'rgb_array' 
            or a matplotlib figure if mode is 'human'.
            The UrbanScenario base class provides the implementation 
            of the geospatial digital twin, while the network digital twin
            can be provided in subclasses using the method _render.
            
            The rendering is done using the Urban3DRenderer class, which takes in the environment state
            and renders the 3D visualization of the urban environment, including buildings, UAVs, UEs, and their LoS links.
            
            Urban3DRenderer.render(state, mode=mode) where state is a dict containing:
                1. Geospatial information (geospatial digital twin) of the environment:
                    - volume_size: (x, y, z) size of the environment
                    - buildings: list of building data (x, y, z, width, depth, height)
                    - title: string title for the plot 
                    to show the urban parameters (alpha, beta, gamma, e)
                2. Network information (network digital twin) of the environment:
                    - uav_positions: (n_uavs, 3) positions of UAVs
                    - ue_positions: (n_ues, 3) positions of UEs
                    - links: list of dicts with 'source', 'target', 'los' for each UAV-UE pair
                    - signal strength (RSSI) between UAVs/eNB and UEs/IoT devices
                    - interference levels between UAVs/eNB and UEs/IoT devices
                    
            Example of the state dictionary passed to the renderer:
            state = {
                # Geospatial information (geospatial digital twin)
                'volume_size': env.volume_size,
                'buildings': env._env.building_data[urban_idx],
                'title': title,
                
                # Network information (network digital twin)
                'uav_positions': uav_positions,
                'ue_positions': ue_positions,
                'links': los_links,
                'signal_strength': signal_strength,
                'interference_levels': interference_levels
            }
            The _render method can be overridden in subclasses to provide additional 
            information for rendering the network digital twin, such as signal 
            strength, interference levels, etc.
            
        """
        if not hasattr(self, 'renderer'):
            self.renderer = Urban3DRenderer()
        #
        if not hasattr(self, 'render_idx'):
            self.render_idx = np.random.randint(env.batch_size[0])
        #
        alpha, beta, gamma, e = env._env.info[self.render_idx][:4]
        scenario_name = env.scenario_name.upper()
        title = f"UrbanMARL - {scenario_name} - {algorithm.upper()} - ({alpha:.2f}, {beta}, {gamma:.2f}, {e:.4f})"
        state = {
            'volume_size': env.volume_size,
            'current_frame': env.current_step[self.render_idx],
            'buildings': env._env.building_data[self.render_idx],
            'building_faces': env._env.building_faces[self.render_idx],
            'heatmap': env._env.height_maps[self.render_idx],
            'title': title
        }
        #
        network_digital_twin = self._render(env, mode=mode)
        state.update(network_digital_twin)
        
        return self.renderer.render(state, mode=mode)
    
    @abstractmethod
    def _render(self, env, mode='rgb_array'):
        """Returns a dictionary with additional information for 
        rendering radio digital twins, such as:
            - positions of network nodes (UAVs, eNBs, UEs, IoT devices,...)
            - line of sight (LoS) conditons between UAVs/eNB and UEs/IoT devices
            - signal strength (RSSI) between UAVs/eNB and UEs/IoT devices
            - interference levels between UAVs/eNB and UEs/IoT devices
        This method should be overridden by subclasses to provide 
        custom rendering information.
        """
        return {}
    
