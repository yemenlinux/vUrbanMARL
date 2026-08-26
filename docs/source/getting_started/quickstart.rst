Quickstart Tutorial
===================

This tutorial walks through creating an `UrbanMARL` vectorized environment, stepping through environment interactions, and rendering 3D digital twin visualizations.

Creating an Urban Environment
-----------------------------

`UrbanMARL` environments are native `TorchRL` environment wrappers (`UrbanEnv`). They support batched execution across CPU and CUDA devices.


.. code-block:: python

   import torch
   from urbanmarl.envs.base_env import UrbanEnv

   # Instantiate batched urban environment (4 parallel environments)
   env = UrbanEnv(
       num_envs=4,
       scenario="uav_navigation",
       continuous_actions=True,
       seed=42,
       device="cuda" if torch.cuda.is_available() else "cpu",
   )

   # Reset environment
   tensordict = env.reset()
   print("Initial observation Tensordict keys:", tensordict.keys())

Stepping through the Environment
--------------------------------

Actions are supplied as TensorDicts containing tensor tensors for agent action spaces.

.. code-block:: python

   # Sample random action matching full_action_spec
   action = env.full_action_spec.rand()

   # Step environment
   next_tensordict = env.step(action)
   reward = next_tensordict.get(("next", "agents", "reward"))
   done = next_tensordict.get(("next", "done"))

   print(f"Step Reward shape: {reward.shape}")
   print(f"Done shape: {done.shape}")

Rendering 3D Visualizations
---------------------------

`UrbanEnv` provides high-speed vectorized 3D rendering using persistent Matplotlib canvas reuse:

.. code-block:: python

   # Render RGB frame array (H, W, 3)
   frame = env.render(mode="rgb_array")
   print("Rendered 3D frame dimensions:", frame.shape)
