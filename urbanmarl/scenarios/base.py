from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from tensordict import TensorDictBase
from torchrl.data import Composite

class UrbanScenario(ABC):
    """Base class for all urban scenarios.
    
    A scenario defines:
      - initial placement of agents (reset)
      - action processing and state update (step)
      - reward computation
      - observation formation
      - done / terminated / truncated flags
      - observation and action specs (for TorchRL)
    """
    
    def __init__(self, config: dict):
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
    
    abstractmethod
    def _reset_at(self, env, tensordict: Optional[TensorDictBase] = None, **kwargs):
        """Reset the environment's internal state (positions, battery, etc.)."""
        pass

    @abstractmethod
    def process_actions(self, env, tensordict: TensorDictBase):
        """Apply actions, update positions, battery, collision flags, etc."""
        pass

    @abstractmethod
    def reward(self, env) -> torch.Tensor:
        """Return reward tensor of shape (batch_size, n_agents, 1)."""
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

    # Optional: info_spec if needed
    def info_agent_spec(self, env, group) -> Optional[Composite]:
        return None
    
    def info_agent(self, env, group) -> Optional[Composite]:
        return None 
    
    def info_global_spec(self, env) -> Optional[Composite]:
        """For urban parameters vs global reward"""
        return None
    
    def info_global(self, env):
        pass
    
    # Optional: reward_spec if needed
    def reward_spec(self, env, group) -> Optional[Composite]:
        return None 
    
    # Optional: stat_spec if needed for global state
    def stat_spec(self, env):
        return None
