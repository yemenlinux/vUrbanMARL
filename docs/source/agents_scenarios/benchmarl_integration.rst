BenchMARL Integration Guide
===========================

**UrbanMARL** seamlessly integrates with **BenchMARL**, the benchmark library for Multi-Agent Reinforcement Learning built on PyTorch and TorchRL.


Architecture Overview
---------------------

The integration is structured into three main layers:

1. **Environment Layer** (`urbanmarl.envs.base_env.UrbanEnv`):
   Provides TorchRL-compatible vectorized multi-agent environment wrappers supporting GPU execution, line-of-sight ray-casting, radio propagation, and M/M/c queuing dynamics.

2. **Task Class** (`benchmarl.environments.urbanmarl.common.UrbanEnvClass`):
   Implements BenchMARL's `TaskClass` interface. Maps TorchRL environment specs, group maps (`uav`, `ue`), observation specs, action specs, and custom evaluation logging metrics (`log_info`).

3. **Task Registry** (`benchmarl.environments.UrbanEnvTask`):
   Exposes standard urban scenario tasks:
   - `UrbanEnvTask.UAV_NAVIGATION`: UAV 3D obstacle avoidance and waypoint navigation.
   - `UrbanEnvTask.UAV_UE_LOS`: Dynamic UAV-UE line-of-sight link maintenance.
   - `UrbanEnvTask.COVERAGE`: Area coverage and user equipment service maximization.
   - `UrbanEnvTask.UAVMEC_OFFLOADING`: UAV-assisted MEC task offloading and resource allocation.
   - `UrbanEnvTask.UAV_MOBILE_UE`: Dynamic user mobility tracking under Gauss-Markov / Manhattan models.
   - `UrbanEnvTask.UAV_LIDAR_NAVIGATION`: Safe urban canyon navigation with 360° LiDAR ray-casting.
   - `UrbanEnvTask.UAVMEC_ADVANCED_PHYSICS`: Aerodynamics power dissipation and 3D directional antennas.

Code Example: Training MADDPG on Urban Navigation
-------------------------------------------------

The following complete Python script demonstrates running **MADDPG** on the `UAV_NAVIGATION` task:

.. code-block:: python

   import os
   from pathlib import Path
   import torch

   from benchmarl.algorithms import MaddpgConfig
   from benchmarl.environments import UrbanEnvTask
   from benchmarl.experiment import Experiment, ExperimentConfig
   from benchmarl.models.mlp import MlpConfig
   from urbanmarl.callback.evaluate_los_per_urban import EvaluateLoS

   def run_experiment():
       # Configure experiment
       experiment_config = ExperimentConfig.get_from_yaml()
       output_dir = Path("./outputs/uav_navigation_maddpg")
       os.makedirs(output_dir, exist_ok=True)
       
       experiment_config.save_folder = output_dir
       experiment_config.max_n_iters = 100
       experiment_config.parallel_collection = True
       experiment_config.checkpoint_at_end = True

       if torch.cuda.is_available():
           experiment_config.device = "cuda"
           experiment_config.sampling_device = "cuda"
           experiment_config.train_device = "cuda"

       # Select task and algorithms
       task = UrbanEnvTask.UAV_NAVIGATION.get_from_yaml()
       algorithm_config = MaddpgConfig.get_from_yaml()
       model_config = MlpConfig.get_from_yaml()
       critic_model_config = MlpConfig.get_from_yaml()

       # Initialize BenchMARL Experiment
       experiment = Experiment(
           task=task,
           algorithm_config=algorithm_config,
           model_config=model_config,
           critic_model_config=critic_model_config,
           seed=42,
           config=experiment_config,
           callbacks=[EvaluateLoS()],
       )

       # Run training loop
       experiment.run()

   if __name__ == "__main__":
       run_experiment()

Code Example: MAPPO on MEC Offloading
-------------------------------------

To train **MAPPO** on the `UAVMEC_OFFLOADING` task:

.. code-block:: python

   from benchmarl.algorithms import MappoConfig
   from benchmarl.environments import UrbanEnvTask
   from benchmarl.experiment import Experiment, ExperimentConfig
   from benchmarl.models.mlp import MlpConfig

   task = UrbanEnvTask.UAVMEC_OFFLOADING.get_from_yaml()
   algorithm_config = MappoConfig.get_from_yaml()
   model_config = MlpConfig.get_from_yaml()

   experiment = Experiment(
       task=task,
       algorithm_config=algorithm_config,
       model_config=model_config,
       critic_model_config=model_config,
       seed=0,
       config=ExperimentConfig.get_from_yaml(),
   )
   experiment.run()

Code Example: MADDPG on High-Fidelity Physics
----------------------------------------------

Train **MADDPG** with aerodynamic flight power dissipation and 3GPP directional beamforming:

.. code-block:: python

   from benchmarl.algorithms import MaddpgConfig
   from benchmarl.environments import UrbanEnvTask
   from benchmarl.experiment import Experiment, ExperimentConfig
   from benchmarl.models.mlp import MlpConfig

   task = UrbanEnvTask.UAVMEC_ADVANCED_PHYSICS.get_from_yaml()
   algo_config = MaddpgConfig.get_from_yaml()
   model_config = MlpConfig.get_from_yaml()

   experiment = Experiment(
       task=task,
       algorithm_config=algo_config,
       model_config=model_config,
       critic_model_config=model_config,
       seed=0,
       config=ExperimentConfig.get_from_yaml(),
   )
   experiment.run()

Code Example: MASAC on LiDAR Urban Canyon Navigation
-----------------------------------------------------

Train **MASAC** (Multi-Agent Soft Actor-Critic) on POMDP LiDAR rangefinder navigation:

.. code-block:: python

   from benchmarl.algorithms import MasacConfig
   from benchmarl.environments import UrbanEnvTask
   from benchmarl.experiment import Experiment, ExperimentConfig
   from benchmarl.models.mlp import MlpConfig

   task = UrbanEnvTask.UAV_LIDAR_NAVIGATION.get_from_yaml()
   algo_config = MasacConfig.get_from_yaml()
   model_config = MlpConfig.get_from_yaml()

   experiment = Experiment(
       task=task,
       algorithm_config=algo_config,
       model_config=model_config,
       critic_model_config=model_config,
       seed=42,
       config=ExperimentConfig.get_from_yaml(),
   )
   experiment.run()

Evaluating & Plotting Results
-----------------------------

After running experiments, use `urbanmarl.eval_results` to aggregate CSV scalar logs and generate comparative plots across algorithms:

.. code-block:: python

   from pathlib import Path
   from urbanmarl.eval_results import find_experiments, load_metric_over_seeds

   # Load output metrics
   output_dir = Path("./outputs/uav_navigation_maddpg")
   experiments = find_experiments(output_dir)

   print(f"Found {len(experiments)} experiment runs:")
   for exp in experiments:
       print(f"  - Algorithm: {exp.algorithm}, Scenario: {exp.scenario}")
