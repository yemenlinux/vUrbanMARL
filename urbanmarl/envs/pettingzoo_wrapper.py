"""UrbanMARL PettingZoo ParallelEnv Standardization Wrapper.

Provides a standard PettingZoo ParallelEnv interface for UrbanMARL:

- Fully compatible with CleanRL, Ray RLlib, Stable-Baselines3 (via shimmy/SuperSuit),
  and standard MARL evaluation pipelines.
- Supports continuous 3D velocity actions and localized agent observation spaces.
"""

from typing import Any, Dict, Optional, Tuple

# import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from urbanmarl.envs.base_env import UrbanEnv


class UrbanPettingZooEnv:
    """PettingZoo ParallelEnv wrapper for UrbanMARL environments.

    Wraps a single-environment UrbanEnv instance into standard PettingZoo format.
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "name": "urbanmarl_v1",
    }

    def __init__(
        self,
        scenario: str = "uav_navigation",
        render_mode: Optional[str] = "rgb_array",
        **env_kwargs,
    ) -> None:
        """Initializes PettingZoo wrapper.

        Args:
            scenario (str): UrbanMARL scenario name. Defaults to 'uav_navigation'.
            render_mode (Optional[str]): Rendering mode ('rgb_array' or 'human').
            **env_kwargs: Arguments passed directly to UrbanEnv.
        """
        self.render_mode = render_mode
        seed = env_kwargs.pop("seed", 42)
        device = env_kwargs.pop("device", "cpu")
        self.env = UrbanEnv(
            num_envs=1,
            continuous_actions=True,
            seed=seed,
            device=device,
            scenario=scenario,
            **env_kwargs,
        )

        self.possible_agents = [f"uav_{i}" for i in range(self.env.n_uavs)]
        self.agents = self.possible_agents.copy()

        # Observation spaces
        # obs_spec = self.env.observation_spec
        # Extract UAV observation dimension
        sample_obs = self.env.reset()
        uav_obs = sample_obs.get(("uav", "observation"))[0, 0].detach().cpu().numpy()
        obs_dim = uav_obs.shape[0]

        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Action spaces: [v_h, phi, v_z]
        max_h = float(self.env.max_h_speed)
        max_v = float(self.env.max_v_speed)
        act_low = np.array([-max_h, -np.pi, -max_v], dtype=np.float32)
        act_high = np.array([max_h, np.pi, max_v], dtype=np.float32)

        self.action_spaces = {
            agent: spaces.Box(low=act_low, high=act_high, dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent: str) -> spaces.Space:
        """Returns the observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Returns the action space for an agent."""
        return self.action_spaces[agent]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the environment.

        Args:
            seed (Optional[int]): Random seed.
            options (Optional[dict]): Configuration options.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: (observations, infos).
        """
        if seed is not None:
            self.env.set_seed(seed)

        self.agents = self.possible_agents.copy()
        td = self.env.reset()

        obs_tensor = td.get(("uav", "observation"))[0].detach().cpu().numpy()
        observations = {agent: obs_tensor[i] for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """Steps all agents in the environment.

        Args:
            actions (Dict[str, np.ndarray]): Dictionary of agent actions.

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos).
        """
        # Pack actions into TensorDict
        act_list = [actions[agent] for agent in self.possible_agents]
        act_tensor = torch.tensor(
            np.stack(act_list), dtype=torch.float32, device=self.env.device
        ).unsqueeze(
            0
        )  # Shape (1, N, 3)

        fake_td = self.env.fake_tensordict()
        fake_td.set(("uav", "action"), act_tensor)

        stepped = self.env.step(fake_td)
        next_td = stepped.get("next")

        obs_tensor = next_td.get(("uav", "observation"))[0].detach().cpu().numpy()
        observations = {
            agent: obs_tensor[i] for i, agent in enumerate(self.possible_agents)
        }

        # Rewards
        if ("uav", "reward") in next_td.keys(True, True):
            r_tensor = (
                next_td.get(("uav", "reward"))[0].detach().cpu().numpy().flatten()
            )
            rewards = {
                agent: float(r_tensor[i])
                for i, agent in enumerate(self.possible_agents)
            }
        else:
            rewards = dict.fromkeys(self.possible_agents, 0.0)

        # Terminations and truncations
        term_val = bool(next_td.get("terminated")[0, 0].item())
        trunc_val = bool(next_td.get("truncated")[0, 0].item())

        terminations = {agent: term_val for agent in self.possible_agents}
        truncations = {agent: trunc_val for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        if term_val or trunc_val:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        """Renders environment frame."""
        return self.env.render(mode=self.render_mode or "rgb_array")

    def close(self) -> None:
        """Closes environment renderer."""
        self.env.close()
