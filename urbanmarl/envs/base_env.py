from __future__ import annotations

import importlib.util
import torch
from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase
# from torchrl.data import (
#     Composite, 
#     DEVICE_TYPING,
# )
from torchrl.envs.common import _EnvWrapper, EnvBase

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

from torchrl.envs.utils import (
    _classproperty,
    _selective_unsqueeze,
    check_marl_grouping,
    MarlGroupMapType,
)
from typing import Optional

from urbanmarl.models.urban_map import VectorizedUrbanMap
from urbanmarl.scenarios import load_scenario, UrbanScenario
# from urbanmarl.envs.specs import *

_has_urbanmarl = importlib.util.find_spec("urbanmarl") is not None

class UrbanEnvBase(_EnvWrapper):
    git_url = "https://github.com/yemenlinux/vUrbanMARL.git"
    libname = "urbanmarl"

    @property
    def lib(self):
        import urbanmarl
        return urbanmarl

    @_classproperty
    def available_envs(cls):
        if not _has_urbanmarl:
            return []
        raise NotImplementedError("dynamic import of urbanmarl not implemented yet")

    def __init__(
        self,
        device: DEVICE_TYPING = None,
        batch_size: torch.Size | None = None,
        scenario: str | UrbanScenario = "navigate",
        **kwargs,
    ):
        if isinstance(scenario, str):
            self.scenario_name = scenario
        if isinstance(scenario, UrbanScenario):
            self.scenario = scenario
            
        super().__init__(
            device=device, 
            batch_size=batch_size, **kwargs)

    def _check_kwargs(self, kwargs: dict):
        # Same as before
        self.num_envs = kwargs.get("num_envs", 72)
        self.continuous_actions = kwargs.get("continuous_actions", True)
        self.dt = kwargs.get("dt", 1.0)
        self.n_uavs = kwargs.get("num_uavs", 3)
        self.n_ues = kwargs.get("num_ues", 50)
        self.max_steps = kwargs.get("max_steps", 100)
        # group_map
        if "group_map" in kwargs:
            self.group_map = kwargs["group_map"]
        else:
            self.group_map = self._get_default_group_map(
                [f"uav_{i}" for i in range(self.n_uavs)]
                # + [f"ue_{i}" for i in range(self.n_ues)]
            )
        
        # self.agents = kwargs.get("agents", [f"uav_{i}" for i in range(self.n_uavs)])
        # self.n_agents = len(self.agents)
        # self.group_map = kwargs.get("group_map", None)
        self.volume_size = kwargs.get("volume_size", (500, 500, 200))
        self.max_h_speed = kwargs.get("max_horizontal_speed", 49.0)
        self.max_v_speed = kwargs.get("max_vertical_speed", 12.0)
        self.max_power = kwargs.get("max_transmit_power", 5.0)
        self.uav_base_cap = 20.0 * 1e9

        # Store scenario config (everything that the scenario might need)
        self.scenario_config = {k: v for k, v in kwargs.items() if k not in [
            "num_envs", "continuous_actions", "dt", "num_uavs", "num_ues", "max_steps",
            "agents", "group_map", "volume_size", "max_horizontal_speed",
            "max_vertical_speed", "max_transmit_power", "uav_base_cap"
        ]}

    def _build_env(self, **kwargs):
        # Instantiate scenario (done after group_map is set)
        self.scenario = load_scenario(self.scenario_name, self.scenario_config)
        #
        return VectorizedUrbanMap(
            self.batch_size[0],
            self.volume_size,
            self.device
        )

    def _get_default_group_map(self, agent_names: list[str]):
        # Same as before
        group_map = {}
        follows_convention = True
        for agent_name in agent_names:
            agent_name_split = agent_name.split("_")
            if len(agent_name_split) == 1:
                follows_convention = False
            follows_convention = follows_convention and agent_name_split[-1].isdigit()
            if not follows_convention:
                break
            group_name = "_".join(agent_name_split[:-1])
            if group_name in group_map:
                group_map[group_name].append(agent_name)
            else:
                group_map[group_name] = [agent_name]
        if not follows_convention:
            group_map = MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(agent_names)
        if "agent" in group_map and len(group_map) == 1:
            agent_group = group_map["agent"]
            group_map["agents"] = agent_group
            del group_map["agent"]
        return group_map

    def _init_env(self) -> int | None:
        # Now handled by the scenario's reset method
        pass

    def _set_seed(self, seed: int | None) -> None:
        self._env.seed(seed)
        
    @property
    def agents(self) -> list[str]:
        agents = []
        for group, agent_names in self.group_map.items():
            agents.extend(agent_names)
        return agents
    @property
    def n_agents(self) -> int:
        return len(self.agents)
    @property
    def agent_names_to_indices_map(self) -> dict[str, int]:
        return {agent: i for i, agent in enumerate(self.agents)}
    @property
    def agent_indices_to_names_map(self) -> dict[int, str]:
        return {i: agent for i, agent in enumerate(self.agents)}
    @property
    def agent_names(self) -> list[str]:
        return self.agents
    

    def _make_specs(self, env) -> None:
        # Delegate to scenario
        # self.agent_names = [f"agent_{i}" for i in range(len(self.agents))]
        # self.agent_names_to_indices_map = {
        #     agent: i for i, agent in enumerate(self.agent_names)
        # }
        # if self.group_map is None:
        #     self.group_map = self._get_default_group_map(self.agent_names)
        # elif isinstance(self.group_map, MarlGroupMapType):
        #     self.group_map = self.group_map.get_group_map(self.agent_names)
        check_marl_grouping(self.group_map, self.agent_names)

        # # Instantiate scenario (done after group_map is set)
        # self.scenario = load_scenario(self.scenario_name, self.scenario_config)

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
                        {
                            "action": self.scenario.action_spec(self, group)
                        },
                    )
                )
                observation_specs.append(
                    Composite(
                        {
                            "observation": self.scenario.observation_spec(self, group)
                        },
                    )
                )
                reward_specs.append(
                    Composite(
                        {
                            "reward": self.scenario.reward_spec(self, group)
                        }
                    )
                )
                info = self.scenario.info_agent_spec(self, group)
                if info:
                    info_specs.append(
                        Composite(
                            info
                        )
                    )
                
                # if self.scenario.has_agent_info:
                #     info_specs.append(
                #         Composite(
                #             self.scenario.info_agent_spec(self, group)
                #         )
                #     )
            # Build multi-agent stacked specs
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
                full_observation_spec_unbatched[(group, "info")] = group_info_spec
            # Determine heterogeneity (if needed)
            group_het_specs = (isinstance(group_observation_spec, StackedComposite) 
                               or isinstance(group_action_spec, StackedComposite))
            self.het_specs_map[group] = group_het_specs
            self.het_specs = self.het_specs or group_het_specs

        # State spec (global CTDE) – can also be scenario-defined, but keep as before
        global_state_spec = self.scenario.state_spec(self)
        if global_state_spec is not None:
            full_observation_spec_unbatched["state"] = global_state_spec
        # if self.scenario.has_state:
        #     full_observation_spec_unbatched["state"] = self.scenario.stat_spec(self)
        #
        global_info_spec = self.scenario.info_global_spec(self)
        if global_info_spec is not None:
            full_observation_spec_unbatched["info"] = global_info_spec
        # if self.scenario.has_global_info:
        #     full_observation_spec_unbatched["info"] = self.scenario.info_global_spec(self)
        
        full_done_spec_unbatched = Composite(
            {
                "done": Categorical(
                    n=2, 
                    shape=torch.Size((1,)), 
                    dtype=torch.bool, device=self.device),
                "terminated": Categorical(
                    n=2, 
                    shape=torch.Size((1,)), 
                    dtype=torch.bool, device=self.device),
                "truncated": Categorical(
                    n=2, shape=torch.Size((1,)), 
                    dtype=torch.bool, 
                    device=self.device),
            },
        )

        self.full_action_spec_unbatched = full_action_spec_unbatched
        self.full_observation_spec_unbatched = full_observation_spec_unbatched
        self.full_reward_spec_unbatched = full_reward_spec_unbatched
        self.full_done_spec_unbatched = full_done_spec_unbatched
        
            

    # ---- Helper methods used by scenario ----
    def _get_obs(self) -> torch.Tensor:
        return self.scenario.observation(self)

    def _get_state(self):
        return self.scenario.state(self)
        

    def _update_done_flags(self):
        return self.scenario.done(self)

    def _update_reward(self):
        # Store reward in self.reward for later use
        self.reward = self.scenario.reward(self)

    # ---- Override EnvBase methods ----
    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        # Scenario-specific reset
        self.scenario.reset(self, tensordict, **kwargs)

        source = {
            "done": self.done,
            "terminated": self.done.clone(),
            "truncated": self.done.clone(),
        }
        if (self.scenario.has_state 
            or "state" in self.full_observation_spec_unbatched.keys()):
            source["state"] = self._get_state()
        if (self.scenario.has_global_info
            or "info" in self.full_observation_spec_unbatched.keys()):
            source["info"] = self.scenario.info_global(self)

        obs = self._get_obs()
        for group, agent_names in self.group_map.items():
            indices = [self.agent_names_to_indices_map[name] for name in agent_names]
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
        self.scenario.process_actions(self, tensordict)
        self.current_step += 1

        self._update_reward()
        done, terminated, truncated = self._update_done_flags()

        obs = self._get_obs()
        reward = self.reward

        source = {
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }
        if (self.scenario.has_state 
            or "state" in self.full_observation_spec_unbatched.keys()):
            source["state"] = self._get_state()
        if (self.scenario.has_global_info
            or "info" in self.full_observation_spec_unbatched.keys()):
            source["info"] = self.scenario.info_global(self)

        for group, agent_names in self.group_map.items():
            indices = [self.agent_names_to_indices_map[name] for name in agent_names]
            group_obs = obs[:, indices, :]
            group_reward = reward[:, indices, :]
            group_batch_size = self.batch_size + torch.Size([len(agent_names)])

            group_dict = {
                "observation": group_obs,
                "reward": group_reward,
            }

            # ----- Add agent info if requested -----
            if self.scenario.has_agent_info:
                agent_info_all = self.scenario.info_agent(self, group)
                # agent_info_all is a dict with keys and values of shape (B, n_agents, ...)
                group_info = {}
                for key, value in agent_info_all.items():
                    # select agents belonging to this group
                    group_info[key] = value[:, indices, ...]  # keep other dims if any
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
    
    
    

    # ---- Utility methods ----
    def read_obs(self, observations):
        if isinstance(observations, torch.Tensor):
            return _selective_unsqueeze(observations, batch_size=self.batch_size)
        return TensorDict(
            source={key: self.read_obs(value) for key, value in observations.items()},
            batch_size=self.batch_size,
        )

    # def read_done(self, done):
    #     return _selective_unsqueeze(done, batch_size=self.batch_size)

    # def read_reward(self, rewards):
    #     return _selective_unsqueeze(rewards, batch_size=self.batch_size)

    def info(self, agent):
        # Could be scenario-defined
        return {}

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        self._env.to(device)
        return super().to(device)

# For backward compatibility, keep name UrbanEnv
class UrbanEnv(UrbanEnvBase):
    def __init__(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: torch.device,
        scenario: str | UrbanScenario = "default",
        **kwargs
    ):
        super().__init__(
            device=device,
            batch_size=torch.Size([num_envs]),
            # 
            scenario=scenario,
            **kwargs
        )
