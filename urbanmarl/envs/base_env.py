from __future__ import annotations

"""UrbanMARL Environment Base Wrappers.

Provides TorchRL-compatible multi-agent environment wrappers for 3D urban UAV/MEC
simulations supporting batched execution across CPU and GPU devices.
"""

import importlib.util
from typing import Optional

import torch
from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    DEVICE_TYPING,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    StackedComposite,
    TensorSpec,
    Unbounded,
)
from torchrl.envs.common import _EnvWrapper, EnvBase
from torchrl.envs.utils import (
    MarlGroupMapType,
    _classproperty,
    _selective_unsqueeze,
    check_marl_grouping,
)

from urbanmarl.models.urban_map import VectorizedUrbanMap
from urbanmarl.scenarios import UrbanScenario, load_scenario

_has_urbanmarl = importlib.util.find_spec("urbanmarl") is not None



class UrbanEnvBase(_EnvWrapper):
    """Base TorchRL environment wrapper for UrbanMARL environments.

    Manages multi-agent grouping, scenario initialization, batch environment map
    allocations, tensor specifications, and vectorized step/reset mechanics.

    Attributes:
        git_url (str): Repository URL for UrbanMARL.

        libname (str): Package library identifier name.
        scenario_name (str): Name identifier of the active scenario.
        scenario (UrbanScenario): Active UrbanScenario instance.
        num_envs (int): Number of parallel environments in the batch.
        n_uavs (int): Number of UAV agents per environment.
        n_ues (int): Number of User Equipment entities per environment.
        max_steps (int): Maximum episode horizon step count.
        group_map (dict): Multi-agent grouping configuration dictionary.
    """

    git_url = "https://github.com/yemenlinux/vUrbanMARL.git"
    libname = "urbanmarl"

    @property
    def lib(self):
        """Returns the imported urbanmarl module package reference."""
        import urbanmarl
        return urbanmarl

    @_classproperty
    def available_envs(cls):
        """Lists available environment identifiers registered in the library."""
        if not _has_urbanmarl:
            return []
        raise NotImplementedError("Dynamic import of urbanmarl not implemented yet")

    def __init__(
        self,
        device: DEVICE_TYPING = None,
        batch_size: torch.Size | None = None,
        scenario: str | UrbanScenario = "navigate",
        **kwargs,
    ) -> None:
        """Initializes the UrbanEnvBase environment.

        Args:
            device (DEVICE_TYPING, optional): Computation device (e.g. 'cpu', 'cuda').
            batch_size (torch.Size | None, optional): Tensor batch size of parallel envs.
            scenario (str | UrbanScenario): Scenario name string or UrbanScenario instance.
            **kwargs: Additional scenario and environment configuration parameters.
        """
        if isinstance(scenario, str):
            self.scenario_name = scenario
        if isinstance(scenario, UrbanScenario):
            self.scenario = scenario

        super().__init__(device=device, batch_size=batch_size, **kwargs)

    def _check_kwargs(self, kwargs: dict) -> None:
        """Parses and sets default environment parameters from keyword arguments.

        Args:
            kwargs (dict): Configuration options dictionary.
        """
        self.num_envs = kwargs.get("num_envs", 72)
        self.continuous_actions = kwargs.get("continuous_actions", True)
        self.dt = kwargs.get("dt", 1.0)
        self.n_uavs = kwargs.get("num_uavs", 3)
        self.n_ues = kwargs.get("num_ues", 50)
        self.max_steps = kwargs.get("max_steps", 100)

        if "group_map" in kwargs:
            self.group_map = kwargs["group_map"]
        else:
            self.group_map = self._get_default_group_map(
                [f"uav_{i}" for i in range(self.n_uavs)]
            )

        self.volume_size = kwargs.get("volume_size", (500, 500, 200))
        self.max_h_speed = kwargs.get("max_horizontal_speed", 49.0)
        self.max_v_speed = kwargs.get("max_vertical_speed", 12.0)
        self.max_power = kwargs.get("max_transmit_power", 5.0)
        self.uav_base_cap = 20.0 * 1e9

        self.scenario_config = {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "num_envs",
                "continuous_actions",
                "dt",
                "num_uavs",
                "num_ues",
                "max_steps",
                "agents",
                "group_map",
                "volume_size",
                "max_horizontal_speed",
                "max_vertical_speed",
                "max_transmit_power",
                "uav_base_cap",
            ]
        }

    def _build_env(self, **kwargs) -> VectorizedUrbanMap:
        """Builds and instantiates the underlying vectorized urban map model.

        Args:
            **kwargs: Extra environment build parameters.

        Returns:
            VectorizedUrbanMap: The created batch 3D urban map instance.
        """
        self.scenario = load_scenario(self.scenario_name, self.scenario_config)
        return VectorizedUrbanMap(
            self.batch_size[0], self.volume_size, self.device
        )

    def _get_default_group_map(self, agent_names: list[str]) -> dict:
        """Derives default multi-agent grouping mapping from agent names.

        Args:
            agent_names (list[str]): List of individual agent identifiers.

        Returns:
            dict: Mapping of group names to lists of agent names.
        """
        group_map = {}
        follows_convention = True
        for agent_name in agent_names:
            agent_name_split = agent_name.split("_")
            if len(agent_name_split) == 1:
                follows_convention = False
            follows_convention = (
                follows_convention and agent_name_split[-1].isdigit()
            )
            if not follows_convention:
                break
            group_name = "_".join(agent_name_split[:-1])
            if group_name in group_map:
                group_map[group_name].append(agent_name)
            else:
                group_map[group_name] = [agent_name]
        if not follows_convention:
            group_map = MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(
                agent_names
            )
        if "agent" in group_map and len(group_map) == 1:
            agent_group = group_map["agent"]
            group_map["agents"] = agent_group
            del group_map["agent"]
        return group_map

    def _init_env(self) -> int | None:
        """Initializes internal environment state."""
        pass

    def _set_seed(self, seed: int | None) -> None:
        """Sets random seed for environment components.

        Args:
            seed (int | None): Seed value.
        """
        self._env.seed(seed)

    @property
    def agents(self) -> list[str]:
        """List of all active agent identifiers across groups."""
        agents = []
        for group, agent_names in self.group_map.items():
            agents.extend(agent_names)
        return agents

    @property
    def n_agents(self) -> int:
        """Total number of agents."""
        return len(self.agents)

    @property
    def agent_names_to_indices_map(self) -> dict[str, int]:
        """Mapping from agent name to index position."""
        return {agent: i for i, agent in enumerate(self.agents)}

    @property
    def agent_indices_to_names_map(self) -> dict[int, str]:
        """Mapping from index position to agent name."""
        return {i: agent for i, agent in enumerate(self.agents)}

    @property
    def agent_names(self) -> list[str]:
        """List of agent name strings."""
        return self.agents

    def _make_specs(self, env) -> None:
        """Constructs action, observation, reward, and done specs for TorchRL.

        Args:
            env: Wrapped base environment object.
        """
        check_marl_grouping(self.group_map, self.agent_names)

        full_action_spec_unbatched = Composite(device=self.device)
        full_observation_spec_unbatched = Composite(device=self.device)
        full_reward_spec_unbatched = Composite(device=self.device)

        self.het_specs = False
        self.het_specs_map = {}

        for group, agent_list in self.group_map.items():
            action_specs = []
            observation_specs = []
            reward_specs = []
            info_specs = []
            for agent in agent_list:
                action_specs.append(
                    Composite(
                        {"action": self.scenario.action_spec(self, group)}
                    )
                )
                observation_specs.append(
                    Composite(
                        {
                            "observation": self.scenario.observation_spec(
                                self, group
                            )
                        }
                    )
                )
                reward_specs.append(
                    Composite(
                        {"reward": self.scenario.reward_spec(self, group)}
                    )
                )
                info = self.scenario.info_agent_spec(self, group)
                if info:
                    info_specs.append(Composite(info))

            group_action_spec = torch.stack(action_specs, dim=0)
            group_observation_spec = torch.stack(observation_specs, dim=0)
            group_reward_spec = torch.stack(reward_specs, dim=0)
            group_info_spec = None
            if len(info_specs):
                group_info_spec = torch.stack(info_specs, dim=0)

            full_action_spec_unbatched[group] = group_action_spec
            full_observation_spec_unbatched[group] = group_observation_spec
            full_reward_spec_unbatched[group] = group_reward_spec

            if group_info_spec is not None:
                full_observation_spec_unbatched[(group, "info")] = (
                    group_info_spec
                )

            group_het_specs = isinstance(
                group_observation_spec, StackedComposite
            ) or isinstance(group_action_spec, StackedComposite)
            self.het_specs_map[group] = group_het_specs
            self.het_specs = self.het_specs or group_het_specs

        global_state_spec = self.scenario.state_spec(self)
        if global_state_spec is not None:
            full_observation_spec_unbatched["state"] = global_state_spec

        global_info_spec = self.scenario.info_global_spec(self)
        if global_info_spec is not None:
            full_observation_spec_unbatched["info"] = global_info_spec

        full_done_spec_unbatched = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        )

        self.full_action_spec_unbatched = full_action_spec_unbatched
        self.full_observation_spec_unbatched = full_observation_spec_unbatched
        self.full_reward_spec_unbatched = full_reward_spec_unbatched
        self.full_done_spec_unbatched = full_done_spec_unbatched

    def _get_obs(self) -> torch.Tensor:
        """Retrieves observation tensor from the active scenario."""
        return self.scenario.observation(self)

    def _get_state(self) -> torch.Tensor:
        """Retrieves global state tensor from the active scenario."""
        return self.scenario.state(self)

    def _update_done_flags(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes episode termination flags (done, terminated, truncated)."""
        return self.scenario.done(self)

    def _update_reward() -> None:
        """Computes and updates scenario rewards."""
        self.reward = self.scenario.reward(self)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        """Resets the multi-agent environment state and returns initial TensorDict.

        Args:
            tensordict (TensorDictBase | None): Input TensorDict if any.
            **kwargs: Extra reset parameters.

        Returns:
            TensorDictBase: TensorDict containing initial observations and state.
        """
        self.scenario.reset(self, tensordict, **kwargs)

        source = {
            "done": self.done,
            "terminated": self.done.clone(),
            "truncated": self.done.clone(),
        }
        if (
            self.scenario.has_state
            or "state" in self.full_observation_spec_unbatched.keys()
        ):
            source["state"] = self._get_state()
        if (
            self.scenario.has_global_info
            or "info" in self.full_observation_spec_unbatched.keys()
        ):
            source["info"] = self.scenario.info_global(self)

        obs = self._get_obs()
        for group, agent_names in self.group_map.items():
            indices = [
                self.agent_names_to_indices_map[name] for name in agent_names
            ]
            group_obs = obs[:, indices, :]
            group_batch_size = self.batch_size + torch.Size([len(agent_names)])

            group_dict = {
                "observation": group_obs,
            }

            if self.scenario.has_agent_info:
                agent_info_all = self.scenario.info_agent(self, group)
                group_info = {}
                for key, value in agent_info_all.items():
                    group_info[key] = value[:, indices, ...]
                group_dict["info"] = TensorDict(
                    source=group_info,
                    batch_size=group_batch_size,
                    device=self.device,
                )

            source[group] = TensorDict(
                source=group_dict,
                batch_size=group_batch_size,
                device=self.device,
            )

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Executes a single environment simulation step.

        Args:
            tensordict (TensorDictBase): TensorDict containing agent actions.

        Returns:
            TensorDictBase: Output TensorDict containing updated observations,
                rewards, done flags, and info dictionaries.
        """
        self.scenario.process_actions(self, tensordict)
        self.current_step += 1

        done, terminated, truncated = self._update_done_flags()

        obs = self._get_obs()

        source = {
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }
        if (
            self.scenario.has_state
            or "state" in self.full_observation_spec_unbatched.keys()
        ):
            source["state"] = self._get_state()
        if (
            self.scenario.has_global_info
            or "info" in self.full_observation_spec_unbatched.keys()
        ):
            source["info"] = self.scenario.info_global(self)

        for group, agent_names in self.group_map.items():
            indices = [
                self.agent_names_to_indices_map[name] for name in agent_names
            ]
            group_obs = obs[:, indices, :]
            group_reward = self.scenario.reward(self, group)[:, indices, :]
            group_batch_size = self.batch_size + torch.Size([len(agent_names)])

            group_dict = {
                "observation": group_obs,
                "reward": group_reward,
            }

            if self.scenario.has_agent_info:
                agent_info_all = self.scenario.info_agent(self, group)
                group_info = {}
                for key, value in agent_info_all.items():
                    group_info[key] = value[:, indices, ...]
                group_dict["info"] = TensorDict(
                    source=group_info,
                    batch_size=group_batch_size,
                    device=self.device,
                )

            source[group] = TensorDict(
                source=group_dict,
                batch_size=group_batch_size,
                device=self.device,
            )

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def read_obs(
        self, observations: torch.Tensor | dict
    ) -> torch.Tensor | TensorDict:
        """Formats and unsqueezes observation data matching batch size.

        Args:
            observations (torch.Tensor | dict): Raw observation input.

        Returns:
            torch.Tensor | TensorDict: Unsandwiched batch observation structure.
        """
        if isinstance(observations, torch.Tensor):
            return _selective_unsqueeze(observations, batch_size=self.batch_size)
        return TensorDict(
            source={
                key: self.read_obs(value) for key, value in observations.items()
            },
            batch_size=self.batch_size,
        )

    def info(self, agent: str) -> dict:
        """Retrieves info dictionary for a specific agent name.

        Args:
            agent (str): Agent identifier.

        Returns:
            dict: Agent-specific info payload.
        """
        return {}

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        """Moves environment tensors to specified target device.

        Args:
            device (DEVICE_TYPING): Target device.

        Returns:
            EnvBase: Self reference after device migration.
        """
        self._env.to(device)
        return super().to(device)


class UrbanEnv(UrbanEnvBase):
    """High-level multi-agent Urban MARL environment class.

    Convenience wrapper over UrbanEnvBase that accepts num_envs integer batch sizing.
    """

    def __init__(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: torch.device,
        scenario: str | UrbanScenario = "default",
        **kwargs,
    ) -> None:
        """Initializes an UrbanEnv instance.

        Args:
            num_envs (int): Number of parallel environments in the batch.
            continuous_actions (bool): Whether actions are continuous vector spaces.
            seed (Optional[int]): Random seed.
            device (torch.device): PyTorch compute device.
            scenario (str | UrbanScenario): Scenario identifier or instance.
            **kwargs: Extra environment parameters.
        """
        super().__init__(
            device=device,
            batch_size=torch.Size([num_envs]),
            scenario=scenario,
            **kwargs,
        )

