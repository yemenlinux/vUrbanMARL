# An example experiment script of Victorized Urban Multi-Agent Simulation (VUMAS).
import sys
import os
from pathlib import Path

# Add project root to path (adjust if notebook is in a subfolder)
sys.path.insert(0, str(Path(__file__).parent.parent ))
#
import warnings
# Suppress the specific future warning from torchrl
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="torchrl.modules.mcts.scores"
)
# Suppress the tensordict to_module warning
warnings.filterwarnings(
    "ignore",
    message=".*TensorDict.to_module().*",
    category=FutureWarning,
    module="tensordict"
)
#
import hydra
from omegaconf import DictConfig
import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
# from benchmarl.environments import TaskRegistry

# from urbanmarl.tasks.common import UrbanEnvTask
from benchmarl.environments import UrbanEnvTask
# Callback
from urbanmarl.callback.evaluate_los_per_urban import EvaluateLoS


if __name__ == "__main__":
    # main()

    # Loads from "../conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    # configure absolute save_folder to ../results/ for easier access
    experiment_config.save_folder = Path(__file__).parent.parent /"outputs/test"
    os.makedirs(experiment_config.save_folder, exist_ok=True)
    if torch.cuda.is_available():
        experiment_config.device = "cuda"
        experiment_config.sampling_device = "cuda"
        experiment_config.train_device = "cuda"
        experiment_config.buffer_device = "cuda"
    experiment_config.max_n_iters = 3
    experiment_config.parallel_collection = True
    # Initialize targeted operational context mapping
    # task = UrbanEnvTask.UAVMEC_OFFLOADING.get_from_yaml()
    # task = UrbanEnvTask.UAV_NAVIGATION.get_from_yaml()
    # task = UrbanEnvTask.UAV_UE_LOS.get_from_yaml()
    task = UrbanEnvTask.COVERAGE.get_from_yaml()
    
    # Configure candidate algorithmic structures
    algorithm_config = MappoConfig.get_from_yaml()
    
    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
        callbacks=[EvaluateLoS()],
    )
    experiment.name = f"{experiment.algorithm_name}"
    experiment.folder_name = f"{experiment.algorithm_name}_{experiment.task_name}"
    experiment.run()