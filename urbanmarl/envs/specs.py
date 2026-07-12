"""
TensorDict Specification definitions for the UrbanMARL environment.
Strict spec definitions are required for TorchRL and BenchMARL to pre-allocate
memory on the GPU and validate data flow during multi-agent rollouts.
"""

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

___all__ = [
    "build_env_specs",
    "unbatched_uav_action_spec",
    "unbatched_uav_observation_spec",
    "unbatched_uav_reward_spec",
    "unbatched_ue_observation_spec"
]

def unbatched_uav_action_spec(
    device: torch.device
) -> Bounded:
    """Unbached uav action space represents movement in 3D space
    using a polar coordinate system (h_distance, angle, v_distance) 
    horizontal distance, direction angle, and vertical distance.
        Args:
            device (torch.device): Target device (CPU/CUDA).
        Action components:
            action (Composite): Normalized action vector of shape (5,).
                    - h_distance (Bounded): horizontal distance (normalized to [-1, 1])
                                representing [0, max_h_speed]
                    - angle (Bounded): angle (normalized to [-1, 1])
                                representing speed direction [0, 2π]
                    - v_distance (Bounded): vertical distance (normalized to [-1, 1])
                                representing [-max_v_speed, max_v_speed]
        Returns:
            Composite: Unbatched TorchRL action specification for a single UAV agent.
    """
    # action_spec = {
    #     "h_distance": Bounded(
    #         low=-1.0, 
    #         high=1.0, 
    #         shape=torch.Size([1]), 
    #         dtype=torch.float32, 
    #         device=device
    #     ),
    #     "angle": Bounded(
    #         low=-1.0, 
    #         high=1.0, 
    #         shape=torch.Size([1]), 
    #         dtype=torch.float32, 
    #         device=device
    #     ),
    #     "v_distance": Bounded(
    #         low=-1.0, 
    #         high=1.0, 
    #         shape=torch.Size([1]), 
    #         dtype=torch.float32, 
    #         device=device
    #     ),
        
    #     }
    
    # horizontal distance, direction angle, and vertical distance
    action_spec = Bounded(
        low=torch.tensor([-1.0, -1.0, -1.0], device=device),
        high=torch.tensor([1.0, 1.0, 1.0], device=device),
        shape=torch.Size([3]),
        dtype=torch.float32, 
        device=device
    )
    return action_spec

def unbatched_uav_observation_spec(
    device: torch.device
) -> Bounded:
    """Unbatched UAV observation spec represents the local observation for a single UAV agent.
        Each agent observes its own state (position, capability, load, battery, energy) 
        and the positions of all UEs in the environment.
    Args:
        device (torch.device): Target device (CPU/CUDA).
    Observation components:
        - position (x, y, z) (Composite): as 3D coordinates
            - x (Bounded): horizontal position (normalized to [-1, 1])
            - y (Bounded): vertical position (normalized to [-1, 1])
            - z (Bounded): altitude (normalized to [0, 1])
        - battery (Bounded): Computational capability of the UAV (normalized to [0, 1])
        

    Returns:
        Bounded: Unbatched TorchRL observation specification for a single UAV agent.
    """
    obs_spec = Bounded(
        low=torch.tensor([-1.0, -1.0, 0.0, 0.0], device=device),
        high=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
        shape=torch.Size([4]),
        dtype=torch.float32,
        device=device
    )
    return obs_spec

def unbatched_uav_reward_spec(
    device: torch.device
) -> Unbounded:
    """Unbatched UAV reward spec represents the reward signal for a single UAV agent.
        Each agent receives a scalar reward based on its performance in the environment.
    Args:
        device (torch.device): Target device (CPU/CUDA).
    Reward components:
        reward (Unbounded): Scalar reward signal for the UAV agent

    Returns:
        Unbounded: Unbatched TorchRL reward specification for a single UAV agent.
    """
    reward_spec = Unbounded(
        # low=-1.0, 
        # high=1.0, 
        shape=torch.Size([1]), 
        dtype=torch.float32, 
        device=device
    )
    return reward_spec

def unbatched_ue_observation_spec(
    device: torch.device
) -> Composite:
    """Unbatched UE observation spec represents the local observation for a single UE node.
        Each UE observes its own state (position, demand).
    Args:
        device (torch.device): Target device (CPU/CUDA).
    Observation components:
        position (x, y, z) (Composite): as 3D coordinates
            - x (Bounded): horizontal position (normalized to [-1, 1])
            - y (Bounded): vertical position (normalized to [-1, 1])
            - z (Bounded): altitude (normalized to [0, 1])
        demand (Bounded): Service demand of the UE (normalized to [0, 1])

    Returns:
        Composite: Unbatched TorchRL observation specification for a single UE node.
    """
    obs_spec = Bounded(
        low=torch.tensor([-1.0, -1.0, 0.0, 0.0], device=device),
        high=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
        shape=torch.Size([4]),
        dtype=torch.float32,
        device=device
    )
    return obs_spec

def build_env_specs(
    n_agents: int, 
    n_ues: int, 
    device: torch.device, 
    batch_size: torch.Size
) -> tuple:
    """
    Builds the observation, state, action, reward, and done specifications.

    Args:
        n_agents (int): Number of agents.
        n_ues (int): Number of User Equipment nodes.
        device (torch.device): Target device (CPU/CUDA).
        batch_size (torch.Size): Batch size for parallel environment execution.

    Returns:
        tuple: (observation_spec, state_spec, action_spec, reward_spec, done_spec)
    """
    B = batch_size
    
    # -------------------------------------------------------------------------
    # 1. Observation Spec (Local Agent Observations)
    # -------------------------------------------------------------------------
    # Features per agent (7): pos_x, pos_y, pos_z, capability, load, battery, energy
    obs_dim = 7
    observation_spec = Composite(
        agents = Composite(
            observation = Unbounded(
                shape=B + torch.Size([n_agents, obs_dim]),
                dtype=torch.float32, 
                device=device
            ),
            shape=B + torch.Size([n_agents])
        ),
        shape=B
    )

    # -------------------------------------------------------------------------
    # 2. State Spec (Global CTDE State)
    # -------------------------------------------------------------------------
    # Global State = All agent observations + All UE positions (x, y, z)
    global_dim = (n_agents * obs_dim) + (n_ues * 3)
    state_spec = Composite(
        state = Unbounded(
            shape=B + torch.Size([global_dim]),
            dtype=torch.float32, 
            device=device
        ),
        shape=B
    )

    # -------------------------------------------------------------------------
    # 3. Action Spec
    # -------------------------------------------------------------------------
    # Actions per agent (5): [h_dist, angle, v_dist, power, task_partition]
    # Bounded strictly between -1.0 and 1.0 for normalized policy outputs
    action_size = 5
    action_spec = Composite(
        {
            "agents": Composite(
                {
                    # Leaf shape must only contain: [n_agents, action_size]
                    "action": Unbounded(
                        shape=torch.Size([n_agents, action_size])  # torch.Size([3, 5])
                    )
                },
                shape=torch.Size([n_agents])  # torch.Size([3])
            )
        },
        shape=torch.Size([])  # Leave empty! BenchMARL automatically updates this with n_envs (10)
    )
    # action_spec = Composite(
    #     agents = Composite(
    #         action = Unbounded(
    #             shape=torch.Size([n_agents, 5]), 
    #             # shape=torch.Size([5]),  # Per-agent action spec (will be repeated for each agent in the group map)
    #             # dtype=torch.float32, 
    #             device=device
    #     ),
    #     # shape=B + torch.Size([n_agents])
    #     shape=torch.Size([n_agents])
    #     ),
    #     shape=torch.Size([])
    # )

    # -------------------------------------------------------------------------
    # 4. Reward Spec
    # -------------------------------------------------------------------------
    # One scalar reward per agent per step
    reward_spec = Composite(
        agents = Composite(
            reward = Unbounded(
                shape=B + torch.Size([n_agents, 1]), 
                dtype=torch.float32, 
                device=device
            )
        ),
        shape=B
    )

    # -------------------------------------------------------------------------
    # 5. Done Spec & Internal Trackers
    # -------------------------------------------------------------------------
    # TorchRL requires explicit tracking of `done`, `terminated`, and `truncated`.
    # We also include internal environment states (e.g., positions, battery) 
    # to allow the step() function to read and write them natively on the GPU.
    done_spec = Composite(
        done = Bounded(
            low=0, 
            high=1, 
            shape=B + torch.Size([1]), 
            dtype=torch.bool, 
            device=device
        ),
        terminated = Bounded(
            low=0, 
            high=1, 
            shape=B + torch.Size([1]), 
            dtype=torch.bool, 
            device=device
        ),
        truncated = Bounded(
            low=0, 
            high=1, 
            shape=B + torch.Size([1]), 
            dtype=torch.bool, 
            device=device
        ),
        # Internal physical state trackers (hidden from policy, used by environment physics)
        # _uav_pos = Unbounded(
        #     shape=B + torch.Size([n_agents, 3]), 
        #     dtype=torch.float32, 
        #     device=device
        # ),
        # _ue_pos = Unbounded(
        #     shape=B + torch.Size([n_ues, 3]), 
        #     dtype=torch.float32, 
        #     device=device
        # ),
        # _uav_caps = Unbounded(
        #     shape=B + torch.Size([n_agents]), 
        #     dtype=torch.float32, 
        #     device=device
        # ),
        # _uav_loads = Unbounded(
        #     shape=B + torch.Size([n_agents]), 
        #     dtype=torch.float32, 
        #     device=device
        # ),
        # _uav_battery = Unbounded(
        #     shape=B + torch.Size([n_agents]), 
        #     dtype=torch.float32, 
        #     device=device
        # ),
        # _uav_energy = Unbounded(
        #     shape=B + torch.Size([n_agents]), 
        #     dtype=torch.float32, 
        #     device=device
        # ),
        # _steps = Bounded(
        #     shape=B + torch.Size([1]), 
        #     dtype=torch.int64, 
        #     device=device),
        shape=B
    )

    return observation_spec, state_spec, action_spec, reward_spec, done_spec


