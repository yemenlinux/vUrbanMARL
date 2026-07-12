from __future__ import annotations

import importlib.util

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
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs.common import _EnvWrapper, EnvBase
from torchrl.envs.libs.gym import gym_backend, set_gym_backend
from torchrl.envs.utils import (
    _classproperty,
    _selective_unsqueeze,
    check_marl_grouping,
    MarlGroupMapType,
)



from typing import Optional

from urbanmarl.models.urban_map import VectorizedUrbanMap
from urbanmarl.models.channel import VectorizedChannelModel
from urbanmarl.models.dtlcm import compute_batched_dtlcm_assignment
from urbanmarl.envs.specs import *
# import urbanmarl.envs.specs

_has_ma3dus = importlib.util.find_spec("ma3dus") is not None

class UrbanWrapper(_EnvWrapper):
    git_url = "https://github.com/yemenlinux/MultiAgent3DUrbanSimulator"
    libname = "ma3dus"

    @property
    def lib(self):
        import ma3dus

        return ma3dus

    @_classproperty
    def available_envs(cls):
        if not _has_ma3dus:
            return []
        raise NotImplementedError("dynamic import of ma3dus not implemented yet")
    
    def __init__(
        self,
        device: DEVICE_TYPING = None,
        batch_size: torch.Size | None = None,
        # allow_done_after_reset: bool = False,
        # spec_locked: bool = True,
        # env: ma3dus.base_env.BaseEnvironment = None,  # noqa
        # categorical_actions: bool = True,
        # group_map: MarlGroupMapType | dict[str, list[str]] | None = None,
        **kwargs,
    ):
        #
        super().__init__(
            device=device,
            batch_size=batch_size,
            **kwargs)
    
    def _check_kwargs(self, kwargs: dict):
        # self.config = kwargs
        self.num_envs = kwargs.get("num_envs", 72)
        self.continuous_actions = kwargs.get("continuous_actions", True)
        self.dt = kwargs.get("dt", 1.0)
        # Agents
        self.n_uavs = kwargs.get("num_uavs", 3)
        self.n_ues = kwargs.get("num_ues", 50)
        self.max_steps = kwargs.get("max_steps", 100)
        
        # -----------
        if 'agents' in kwargs:
            self.agents = kwargs['agents']
            self.n_agents = len(self.agents)
        else:
            self.n_agents = kwargs.get("num_uavs", 3)
            self.agents = [f"uav_{i}" for i in range(self.n_agents)]
        #
        self.group_map = kwargs.get("group_map", None)
        # ----------
        
        # Operational limits
        self.volume_size = kwargs.get("volume_size", (500, 500, 200))
        self.max_h_speed = kwargs.get("max_horizontal_speed", 49.0)
        self.max_v_speed = kwargs.get("max_vertical_speed", 12.0)
        
        # Communication power
        self.max_power = kwargs.get("max_transmit_power", 5.0)
        
        # UAV-MEC cababilities
        self.uav_base_cap = 20.0 * 1e9 # 20 Gcycles/s default benchmark baseline
        
    
    def _build_env(
        self,
        **kwargs: dict,
    ):
        return VectorizedUrbanMap(
            self.batch_size[0], 
            self.volume_size, 
            self.device)
    
    def _get_default_group_map(self, agent_names: list[str]):
        # This function performs the default grouping in ma3dus.
        # Agents with names "<name>_<int>" will be grouped in group name "<name>".
        # If any of the agents does not follow the naming convention, we fall back
        # back on having all agents in one group named "agents".
        group_map = {}
        follows_convention = True
        for agent_name in agent_names:
            # See if the agent follows the convention "<name>_<int>"
            agent_name_split = agent_name.split("_")
            if len(agent_name_split) == 1:
                follows_convention = False
            follows_convention = follows_convention and agent_name_split[-1].isdigit()

            if not follows_convention:
                break

            # Group it with other agents that follow the same convention
            group_name = "_".join(agent_name_split[:-1])
            if group_name in group_map:
                group_map[group_name].append(agent_name)
            else:
                group_map[group_name] = [agent_name]

        if not follows_convention:
            group_map = MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(agent_names)

        # For BC-compatibility rename the "agent" group to "agents"
        if "agent" in group_map and len(group_map) == 1:
            agent_group = group_map["agent"]
            group_map["agents"] = agent_group
            del group_map["agent"]
        return group_map
    
    
    def _init_env(self) -> int | None:
        # uav 
        # uav agent positions
        self.uav_agents_pos = self._env.gen_pos(
            num_pos=self.n_uavs,
            min_z = 20.0,
            max_z = 150.,
            outdoor=True,
        )
        self.uav_battery = torch.ones((self.batch_size[0], self.n_uavs, 1), device=self.device) * 100.0
        self.uav_velocity = torch.zeros((self.batch_size[0], self.n_uavs, 3), device=self.device)
        self.uav_collisions = torch.zeros((self.batch_size[0], self.n_uavs, 1), dtype=torch.bool, device=self.device)
        # UE positions
        self.ue_user_pos = self._env.gen_pos(
            num_pos=self.n_ues,
            min_z = 1.5,
            max_z = 1.5,
            outdoor=True,
        )
        self.ue_battery = torch.ones((self.batch_size[0], self.n_ues, 1), device=self.device) * 100.0
        
        # done
        self.done = torch.zeros((self.batch_size[0], 1), dtype=torch.bool, device=self.device)
        
        # reward
        self.reward = torch.zeros((self.batch_size[0], self.n_uavs, 1), device=self.device)
        
        # step counter
        self.current_step = torch.zeros(
            (self.batch_size[0], 1), dtype=torch.int32, device=self.device
        )
        

    def _set_seed(self, seed: int | None) -> None:
        self._env.seed(seed)
        
    def _make_specs(
        self,
        env: ma3dus.base_env.BaseEnvironment,  # noqa
    ) -> None:
        self.agent_names = [f"agent_{i}" for i in range(len(self.agents))]
        self.agent_names_to_indices_map = {
            agent: i for i, agent in enumerate(self.agent_names)
        }
        if self.group_map is None:
            self.group_map = self._get_default_group_map(self.agent_names)
        elif isinstance(self.group_map, MarlGroupMapType):
            self.group_map = self.group_map.get_group_map(self.agent_names)
        check_marl_grouping(self.group_map, self.agent_names)

        full_action_spec_unbatched = Composite(device=self.device)
        full_observation_spec_unbatched = Composite(device=self.device)
        full_reward_spec_unbatched = Composite(device=self.device)

        self.het_specs = False
        self.het_specs_map = {}
        
        for group in self.group_map.keys():
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
                group_info_spec,
            ) = self._make_unbatched_group_specs(group)
            
            full_action_spec_unbatched[group] = group_action_spec
            full_observation_spec_unbatched[group] = group_observation_spec
            full_reward_spec_unbatched[group] = group_reward_spec
            
            
            if group_info_spec is not None:
                full_observation_spec_unbatched[(group, "info")] = group_info_spec
                
            group_het_specs = isinstance(group_observation_spec, StackedComposite) or isinstance(group_action_spec, StackedComposite)
            self.het_specs_map[group] = group_het_specs
            self.het_specs = self.het_specs or group_het_specs
        
        # --- state spec (per‑environment shape) ---
        # alpha+beta+gamma + self.n_uavs * 4 + self.n_ues * 3
        state_per_env_dim = 3 + self.n_uavs * 4 + self.n_ues * 3
        state_spec = Unbounded(
            shape=torch.Size([state_per_env_dim]),
            dtype=torch.float32,
            device=self.device,
        )
        
        # part of the output that `check_env_specs` validates.
        full_observation_spec_unbatched["state"] = state_spec
        # urban parameters info
        
        
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
        
        
    
    def _make_unbatched_group_specs(
        self, 
        group: str
    ):
        import torch
        from torchrl.data.tensor_specs import BoundedContinuous
        
        action_specs = []
        observation_specs = []
        reward_specs = []
        info_specs = []
        
        # Define physical limits for an individual agent observation tensor (Size: 4)
        # Elements: [pos_x, pos_y, pos_z, battery]
        low_bounds = torch.tensor(
            [-self.volume_size[0] / 2, -self.volume_size[1] / 2, 0.0], 
            device=self.device, dtype=torch.float32
        )
        high_bounds = torch.tensor(
            [self.volume_size[0] / 2, self.volume_size[1] / 2, self.volume_size[2]], 
            device=self.device, dtype=torch.float32
        )
        obs_pos_tmp = BoundedContinuous(
            low=torch.tensor(
            [-self.volume_size[0] / 2, -self.volume_size[1] / 2, 0.0, 0.0], 
            device=self.device, dtype=torch.float32
            ),
            high=torch.tensor(
            [self.volume_size[0] / 2, self.volume_size[1] / 2, self.volume_size[2], 100.0], 
            device=self.device, dtype=torch.float32
            ),
            shape=torch.Size([4]),
            device=self.device,
            dtype=torch.float32
        )
        obs_pos = BoundedContinuous(
            low=low_bounds,
            high=high_bounds,
            shape=torch.Size([3]),
            device=self.device,
            dtype=torch.float32
        )
        obs_battery = BoundedContinuous(
            low=torch.tensor(0.0, device=self.device),
            high=torch.tensor(100.0, device=self.device),
            shape=torch.Size([1]),
            device=self.device,
            dtype=torch.float32
        )

        for agent_name in self.group_map[group]:
            agent_index = self.agent_names_to_indices_map[agent_name]
            agent = self.agents[agent_index]
            
            action_specs.append(
                Composite(
                    {
                        "action": unbatched_uav_action_spec(device=self.device)
                    },
                )
            )
            
            # OVERRIDE: Deploy explicit physical boundary thresholds
            observation_specs.append(
                Composite(
                    {
                        "observation": obs_pos_tmp,
                    },
                )
            )
            
            reward_specs.append(
                Composite(
                    {
                        "reward": unbatched_uav_reward_spec(device=self.device)
                    }
                )
            )
            
            agent_info = self.info(agent)
            if len(agent_info):
                info_specs.append(
                    Composite(
                        {
                            key: Unbounded(
                                shape=_selective_unsqueeze(
                                    value, batch_size=self.batch_size
                                ).shape[1:],
                                device=self.device,
                                dtype=torch.float32,
                            )
                            for key, value in agent_info.items()
                        },
                    ).to(self.device)
                )

        # Build multi-agent stacked specs
        group_action_spec = torch.stack(action_specs, dim=0)  
        group_observation_spec = torch.stack(observation_specs, dim=0)  
        group_reward_spec = torch.stack(reward_specs, dim=0)  
        group_info_spec = None
        if len(info_specs):
            group_info_spec = torch.stack(info_specs, dim=0)

        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
            group_info_spec,
        )
    
    def _get_obs(self) -> torch.tensor:
        """Get observation. Override this method if required in scenario.
        """
        obs = torch.cat([self.uav_agents_pos, self.uav_battery], dim=-1)
        return self.read_obs(obs)
    
    def _get_state(self):
        # FIXED: Returns a structured TensorDict matching the exact Composite schema
        obs = torch.cat([self.uav_agents_pos, self.uav_battery], dim=-1)
        state = torch.cat(
            [self._env.info[:, :3].view(self.batch_size[0], -1),
            obs.view(self.batch_size[0], -1),
            self.ue_user_pos.view(self.batch_size[0], -1)
            ], dim=-1
        )
        return state
    
    def _update_done_flags(self) -> tuple:
        """Update done, terminated, and truncated flags
        Override this method, if required by the scrnario
        return:
            tuple of 3 torch.tensor of a size (batch_size, 1)
            that represent done, terminated, trucated
            
        """
        terminated = self.uav_collisions.any(dim=1) | (
            self.uav_battery <= 0.0
        ).any(dim=1)
        truncated = self.current_step >= self.max_steps
        dones = terminated | truncated

        dones = self.read_done(dones)
        terminated = self.read_done(terminated)
        truncated = self.read_done(truncated)
        return dones, terminated, truncated
    
    def _update_reward(self):
        """Update reward
        Override this method for scenarios
        """
        # Calculate rewards
        los_ratio = self.uav_ue_los.float().mean(dim=2, keepdim=True)
        collision_ratio = self.uav_collisions.float()
        self.reward = los_ratio - collision_ratio
    
    def _reset(
        self, 
        tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset")
            envs_to_reset = _reset.squeeze(-1)
            if envs_to_reset.all():
                self._env.reset(return_urban_info=False)
                self.done = torch.zeros(
                    self.batch_size + (1,), dtype=torch.bool, device=self.device
                )
                self.current_step = torch.zeros(
                    (self.batch_size[0], 1), dtype=torch.int32, device=self.device
                )
            else:
                for env_index, to_reset in enumerate(envs_to_reset):
                    if to_reset:
                        self._env.reset_at(env_index, return_urban_info=False)
                        self.done[env_index, 0] = False
                        self.current_step[env_index, 0] = 0
        else:
            self._env.reset(return_urban_info=False)
            self.done = torch.zeros(
                self.batch_size + (1,), dtype=torch.bool, device=self.device
            )
            self.current_step = torch.zeros(
                (self.batch_size[0], 1), dtype=torch.int32, device=self.device
            )
            
        
        
        # false_flag = self.done
        state = self._get_state()
        source = {
            "done": self.done,
            "terminated": self.done.clone(),
            "truncated": self.done.clone(),
            "state": state,
            
        }
        
        # Make parallel observations
        # obs = torch.cat([self.uav_agents_pos, self.uav_battery], dim=-1)
        obs = self._get_obs()
        # dones = self.read_done(self.dones)
        for group, agent_names in self.group_map.items():
            indices = [self.agent_names_to_indices_map[name] for name in agent_names]
            group_obs = obs[:, indices, :]
            group_batch_size = self.batch_size + torch.Size([len(agent_names)])
            
            source[group] = TensorDict(
                source={
                    "observation": group_obs,
                },
                batch_size=group_batch_size,
                device=self.device,
            )
            
        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out
    
    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        # Process actions and update internal state
        self._process_actions(tensordict)
        self.current_step += 1
        # Get updated observations, Shape: (B, n_uavs, 4)
        # obs = torch.cat([self.uav_agents_pos, self.uav_battery], dim=-1)
        # obs = self.read_obs(obs)
        obs = self._get_obs()
        
        # Calculate dones
        
        # terminated = self.uav_collisions.any(dim=1) | (
        #     self.uav_battery <= 0.0
        # ).any(dim=1)
        # truncated = self.current_step >= self.max_steps
        # dones = terminated | truncated

        # dones = self.read_done(dones)
        # terminated = self.read_done(terminated)
        # truncated = self.read_done(truncated)
        done, terminated, truncated = self._update_done_flags()

        # Calculate rewards
        los_ratio = self.uav_ue_los.float().mean(dim=2, keepdim=True)
        collision_ratio = self.uav_collisions.float()
        self.reward = los_ratio - collision_ratio
        reward = self.read_reward(self.reward)

        state = self._get_state()
        source = {
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "state": state,
            
        }
        
        # COMPLETED: Construct group observations, rewards, and global states
        state_dict = {}
        for group, agent_names in self.group_map.items():
            indices = [self.agent_names_to_indices_map[name] for name in agent_names]
            group_obs = obs[:, indices, :]
            group_reward = reward[:, indices, :]
            group_batch_size = self.batch_size + torch.Size([len(agent_names)])
            
            source[group] = TensorDict(
                source={
                    "observation": group_obs,
                    "reward": group_reward,
                },
                batch_size=group_batch_size,
                device=self.device,
            )
            

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out
    
    def _process_actions(
        self,
        tensordict: TensorDictBase,
    ) -> None:
        # reset collisions
        self.uav_collisions.zero_()
        #
        for group, agent_names in self.group_map.items():
            group_action = tensordict.get((group, "action"))
            if group == "agents":
                # 1. Denormalize action parameters
                # horizontal distance
                group_action[..., 0] = (group_action[..., 0] + 1.0) / 2.0 * self.max_h_speed  # h_dist
                # angle [-1, 1] -> [-pi, pi]
                group_action[..., 1] = group_action[..., 1] * torch.pi  # angle
                # vertical distance
                group_action[..., 2] = group_action[..., 2] * self.max_v_speed  # v_dist
                # update horizontal distance
                dx = group_action[..., 0] * torch.cos(group_action[..., 1])
                dy = group_action[..., 0] * torch.sin(group_action[..., 1])
                dz = group_action[..., 2]
                delta_pos = torch.stack([dx, dy, dx], dim=-1)
                
                
                # clone current positions
                self.previous_uav_pos = self.uav_agents_pos.clone()
                # new positions
                self.uav_agents_pos += delta_pos
                # clip to volume limits
                self.uav_agents_pos[..., 0] = torch.clamp(
                    self.uav_agents_pos[..., 0],
                    -self.volume_size[0] / 2,
                    self.volume_size[0] / 2,
                )
                self.uav_agents_pos[..., 1] = torch.clamp(
                    self.uav_agents_pos[..., 1],
                    -self.volume_size[1] / 2,
                    self.volume_size[1] / 2,
                )
                self.uav_agents_pos[..., 2] = torch.clamp(
                    self.uav_agents_pos[..., 2],
                    0.0,
                    self.volume_size[2],
                )
                # calculate velocity
                self.uav_velocity = (self.uav_agents_pos - self.previous_uav_pos) / self.dt
                
                # calculate collisions
                self.uav_collisions = self._env.check_collision_batch(
                    self.uav_agents_pos, 
                    self.previous_uav_pos)
                
                # calculate LoS
                self.uav_ue_los = self._env.check_los_batch(
                    self.uav_agents_pos,
                    self.ue_user_pos
                )
                
                # calculate battery consumption
                horizontal_speed = torch.norm(self.uav_velocity[..., :2], dim=-1, keepdim=True)
                vertical_speed = torch.abs(self.uav_velocity[..., 2:])
                power_consumption = 0.1 * horizontal_speed + 0.2 * vertical_speed  # Simplified power model
                self.uav_battery -= power_consumption * self.dt
                
    
    def read_obs(
        self, 
        observations: dict | torch.Tensor
    ) -> dict | torch.Tensor:
        if isinstance(observations, torch.Tensor):
            return _selective_unsqueeze(observations, batch_size=self.batch_size)
        return TensorDict(
            source={key: self.read_obs(value) for key, value in observations.items()},
            batch_size=self.batch_size,
        )

    def read_info(self, infos: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(infos) == 0:
            return None
        infos = TensorDict(
            source={
                key: _selective_unsqueeze(
                    value.to(torch.float32), batch_size=self.batch_size
                )
                for key, value in infos.items()
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return infos

    def read_done(self, done):
        done = _selective_unsqueeze(done, batch_size=self.batch_size)
        return done

    def read_reward(self, rewards):
        rewards = _selective_unsqueeze(rewards, batch_size=self.batch_size)
        return rewards

    def read_action(self, action, group: str = "agents"):
        if not self.continuous_actions and not self.categorical_actions:
            action = self.full_action_spec_unbatched[group, "action"].to_categorical(
                action
            )
        agent_actions = action.unbind(dim=1)
        return agent_actions

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_envs={self.num_envs}, n_agents={self.n_agents},"
            f" batch_size={self.batch_size}, device={self.device})"
        )

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        self._env.to(device)
        return super().to(device)
    
    def info(self, agent: Agent) -> dict:
        """Do not change this"""
        return self._info(agent)
        
    
    def _info(self, agent: Agent) -> dict:
        """This function computes the info dict for ``agent`` in a vectorized way.

        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape ``(n_envs, info_size)``

        By default this function returns an empty dictionary.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the info for

        Returns:
             Union[torch.Tensor, Dict[str, torch.Tensor]]: the info
        """
        # Environment info
        info = {
            "alpha": self._env.info[:, 0].clone(),
            "beta": self._env.info[:, 1].clone(),
            "gamma": self._env.info[:, 2].clone(),
            "E": self._env.info[:, 3].clone(),
        }
        return {}
    
class UrbanEnv(UrbanWrapper):
    """UrbanEnv is a native TorchRL Environment that directly implements 
    the multi-agent urban environment dynamics,
    
    """
    def __init__(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: torch.device,
        **kwargs
    ):
        # Initialize the underlying ma3dus environment
        # env = self.lib.make_env(num_envs=num_envs, **kwargs)
        
        
        super().__init__(
            device=device,
            batch_size=torch.Size([num_envs]),
            # env=env,
            # categorical_actions=not continuous_actions,
            # group_map=self.agents,
            # device=device,
            # seed=seed,
            **kwargs
        )
        
    # 
    

