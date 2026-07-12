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
from benchmarl.benchmark import Benchmark
# from benchmarl.experiment import Experiment
from benchmarl.algorithms import MappoConfig, MaddpgConfig
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig

# from urbanmarl.tasks.task import UrbanEnvTask
# from urbanmarl.tasks.common import UrbanEnvTask
from benchmarl.environments import UrbanEnvTask

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Standard BenchMARL setup script instantiating validation routines 
    evaluating Multi-Agent performance structures.
    """
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Initialize targeted operational context mapping
    tasks = [UrbanEnvTask.UAVMEC_OFFLOADING]
    
    # Configure candidate algorithmic structures
    algorithm_configs = [
        MappoConfig.get_from_yaml(),
        MaddpgConfig.get_from_yaml()
    ]
    
    # Loads from "benchmarl/conf/model/layers"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    
    # Launch multi-model benchmarking run
    # benchmark = Benchmark(
    #     tasks=task,
    #     algorithm_configs=algorithms,
    #     config=cfg
    # )
    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds={0, 1},
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )
    benchmark.run_sequential()

if __name__ == "__main__":
    main() 
