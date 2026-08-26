"""UrbanMARL Base Scenario Abstract Class.

Defines the abstract interface and digital twin rendering workflow for all UrbanMARL scenarios.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.data import Composite

from urbanmarl.envs.rendering import Urban3DRenderer, UrbanRenderConfig


class UrbanScenario(ABC):
    """Abstract base class for all urban environment MARL scenarios.

    A scenario defines:
    - Environment specs (observations, actions, rewards, done flags, info).
    - Environment step/reset dynamics (UAV mobility, UE distribution, energy).
    - Reward calculations and termination criteria.
    - Geospatial & radio network digital twin rendering.

    Attributes:
        config (dict): Scenario configuration dictionary.
        has_state (bool): Whether global state space is provided.
        has_agent_info (bool): Whether per-agent info dictionary is generated.
        has_global_info (bool): Whether global environment info dictionary is generated.
    """

    def __init__(self, config: dict) -> None:
        """Initializes the UrbanScenario.

        Args:
            config (dict): Configuration options dictionary.
        """
        self.config = config
        self.continuous_actions = True
        self.discrete_actions = False
        self.has_state = config.get("has_state", False)
        self.has_agent_info = config.get("has_agent_info", False)
        self.has_global_info = config.get("has_global_info", False)

    def reset(self, env, tensordict=None, **kwargs) -> None:
        """Resets environment state across the whole batch or targeted environment indices.

        Args:
            env: UrbanEnv wrapper object.
            tensordict (TensorDictBase, optional): Input TensorDict with reset flags.
            **kwargs: Extra reset parameters.
        """
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
    def _reset_all(
        self, env, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets internal state for all environment instances in the batch."""
        pass

    @abstractmethod
    def _reset_at(
        self, env, env_index: int, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> None:
        """Resets internal state for a specific environment index in the batch."""
        pass

    @abstractmethod
    def process_actions(self, env, tensordict: TensorDictBase) -> None:
        """Applies agent actions and updates environment positions and battery levels."""
        pass

    @abstractmethod
    def reward(self, env, group: str) -> torch.Tensor:
        """Computes reward tensor of shape (batch_size, n_agents, 1) for an agent group."""
        pass

    @abstractmethod
    def observation(self, env) -> torch.Tensor:
        """Computes observation tensor of shape (batch_size, n_agents, obs_dim)."""
        pass

    @abstractmethod
    def done(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes (done, terminated, truncated) flags of shape (batch_size, 1)."""
        pass

    @abstractmethod
    def observation_spec(self, env, group: str) -> Composite:
        """Returns unbatched observation spec for an agent group."""
        pass

    @abstractmethod
    def action_spec(self, env, group: str) -> Composite:
        """Returns unbatched action spec for an agent group."""
        pass

    def state(self, env) -> Optional[torch.Tensor]:
        """Returns global state tensor if enabled."""
        return None

    def info_global_spec(self, env) -> Optional[Composite]:
        """Returns unbatched spec for global environment info dictionary."""
        return None

    def info_global(self, env) -> Optional[dict]:
        """Returns global environment info dictionary for logging."""
        return None

    def info_agent_spec(self, env, group: str) -> Optional[Composite]:
        """Returns unbatched agent info spec for an agent group."""
        return None

    def info_agent(self, env, group: str) -> Optional[dict]:
        """Returns agent info dictionary for an agent group."""
        return None

    def reward_spec(self, env, group: str) -> Optional[Composite]:
        """Returns unbatched reward spec for an agent group."""
        return None

    def state_spec(self, env) -> Optional[Composite]:
        """Returns unbatched global state spec."""
        return None

    def render(
        self, env, algorithm: str = "random", mode: str = "rgb_array"
    ) -> Optional[Union[np.ndarray, object]]:
        """Renders 3D geospatial and radio network digital twins.

        Args:
            env: UrbanEnv environment instance.
            algorithm (str): Algorithm name for plot labeling.
            mode (str): Rendering mode ('rgb_array' or 'human').

        Returns:
            Optional[Union[np.ndarray, object]]: RGB frame array or Matplotlib Figure handle.
        """
        if not hasattr(self, "renderer"):
            self.renderer = Urban3DRenderer()
        if not hasattr(self, "render_idx"):
            self.render_idx = np.random.randint(env.batch_size[0])

        alpha, beta, gamma, e = env._env.info[self.render_idx][:4]
        scenario_name = env.scenario_name.upper()
        title = (
            f"UrbanMARL - {scenario_name} - {algorithm.upper()} - "
            f"({alpha:.2f}, {beta:.1f}, {gamma:.2f}, {e:.4f})"
        )
        state = {
            "volume_size": env.volume_size,
            "current_frame": env.current_step[self.render_idx],
            "buildings": env._env.building_data[self.render_idx],
            "building_faces": env._env.building_faces[self.render_idx],
            "heatmap": env._env.height_maps[self.render_idx],
            "title": title,
        }

        network_digital_twin = self._render(env, mode=mode)
        state.update(network_digital_twin)

        return self.renderer.render(state, mode=mode)

    @abstractmethod
    def _render(self, env, mode: str = "rgb_array") -> dict:
        """Returns network digital twin state payload (UAV/UE positions, links, etc.)."""
        return {}
