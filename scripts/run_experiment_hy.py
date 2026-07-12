# An example experiment script of Victorized Urban Multi-Agent Simulation (VUMAS).
import sys
import os
from pathlib import Path

# Add project root to path (adjust if notebook is in a subfolder)
# project_root = Path.cwd().parent  # if notebook is in experiments/ or similar
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent ))
#
import warnings
# Suppress the specific future warning from torchrl
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="torchrl.modules.mcts.scores"
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


@hydra.main(version_base=None, config_path="../benchmarl/conf", config_name="config")
def main(cfg: DictConfig):
    """
    Standard BenchMARL setup script instantiating validation routines 
    evaluating Multi-Agent performance structures.
    """
    # Loads from "../conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    # configure absolute save_folder to ../results/ for easier access
    experiment_config.save_folder = Path(__file__).parent.parent /"outputs"
    os.makedirs(experiment_config.save_folder, exist_ok=True)
    if torch.cuda.is_available():
        experiment_config.device = "cuda"
        experiment_config.sampling_device = "cuda"
        experiment_config.train_device = "cuda"
        experiment_config.buffer_device = "cuda"
    experiment_config.max_n_iters = 20
    
    # Initialize targeted operational context mapping
    task = UrbanEnvTask.UAVMEC_OFFLOADING.get_from_yaml()
    
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
    )
    experiment.run()


if __name__ == "__main__":
    main()

    