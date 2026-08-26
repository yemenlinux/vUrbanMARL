# An example experiment script of Victorized Urban Multi-Agent Simulation (VUMAS).
import sys
import os
from pathlib import Path

# Add project root to path (adjust if notebook is in a subfolder)
# sys.path.insert(0, str(Path(__file__).parent.parent ))

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
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

import torch
# BenchMARL
from benchmarl.algorithms import (
    # full observation in critic
    MappoConfig,
    MaddpgConfig,
    MasacConfig, 
    # no full observation in critic
    IppoConfig,
    IddpgConfig,
    IsacConfig,
    # Discrete only
    #QmixConfig
)
# from benchmarl.benchmark import Benchmark

from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from benchmarl.eval_results import (
    load_and_merge_json_dicts, 
    Plotting, 
    get_raw_dict_from_multirun_folder
)

# VUMAS task for BenchMARL
from benchmarl.environments import UrbanEnvTask


# config experiment
def config_experiment(
    experiment_config,
    num_envs: int = 72,
    max_n_steps: int = 100,
    max_n_iters: int = 10,
    experiment_dir: str = "experiments",
    eval_interval: int = 10,
    episodes_per_batch: int = 1,
):
    # Experiment parameters
    frames_per_batch = int(num_envs * max_n_steps)
    frames_per_batch = int(episodes_per_batch * num_envs * max_n_steps)
    max_n_frames = int(frames_per_batch * max_n_iters)
    #
    output_dir = project_root / "outputs" / experiment_dir
    
    # configure experiment
    if torch.cuda.is_available():
        experiment_config.device = "cuda"
        experiment_config.sampling_device = "cuda"
        experiment_config.train_device = "cuda"
        experiment_config.buffer_device = "cuda"
        
    experiment_config.parallel_collection = True # False
    experiment_config.max_n_iters = max_n_iters
    experiment_config.max_n_frames = max_n_frames
    experiment_config.on_policy_collected_frames_per_batch = frames_per_batch
    experiment_config.on_policy_n_envs_per_worker = num_envs
    experiment_config.off_policy_collected_frames_per_batch = frames_per_batch
    experiment_config.off_policy_n_envs_per_worker = num_envs
    # experiment_config.evaluation = True
    experiment_config.render = False
    
    experiment_config.evaluation_interval = int(eval_interval * frames_per_batch)
    experiment_config.evaluation_episodes = 5
    experiment_config.loggers = ["csv", "tensorboard"]
    
    experiment_config.save_folder = output_dir

    experiment_config.checkpoint_interval = 0 #int(2 * num_envs * max_n_steps)
    
    experiment_config.checkpoint_at_end = True
    # experiment_config.exclude_buffer_from_checkpoint = False


# Algorithm configurations to test
_algorithm_configs = [
    # On-policy algorithms
    MappoConfig.get_from_yaml(),
    IppoConfig.get_from_yaml(),
    
    # Off-policy algorithms
    MaddpgConfig.get_from_yaml(),
    MasacConfig.get_from_yaml(),
    IddpgConfig.get_from_yaml(),
    IsacConfig.get_from_yaml(),
]



_seeds = [
    0, #1, 2, 3, 4
]

output_dir = project_root / "outputs" / "experiments"

if __name__ == "__main__":
    # Experiment parameters
    num_envs = 72
    max_n_steps = 100
    max_n_iters = 10 # for text then can resume using resume_experiments.py
    experiment_dir = "experiments"
    eval_interval = 10
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    config_experiment(
        experiment_config,
        num_envs=num_envs,
        max_n_steps=max_n_steps,
        max_n_iters=max_n_iters,
        experiment_dir=experiment_dir,
        eval_interval=eval_interval
        )
    # create the output folder if it does not exist
    os.makedirs(experiment_config.save_folder, exist_ok=True)
    
    # Loads from "benchmarl/conf/task/urbanmarl"
    tasks = [
        # uncomment the tasks you want to run
        # UrbanEnvTask.UAV_NAVIGATION.get_from_yaml(),
        # UrbanEnvTask.UAV_UE_LOS.get_from_yaml(),
        UrbanEnvTask.COVERAGE.get_from_yaml(),
    ]

    # Loads from "benchmarl/conf/model/layers"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    
    for seed in _seeds:
        for task in tasks:
            for algorithm_config in _algorithm_configs:
                experiment = Experiment(
                    task=task,
                    algorithm_config=algorithm_config,
                    model_config=model_config,
                    critic_model_config=critic_model_config,
                    seed=seed,
                    config=experiment_config,
                    # callbacks=[EvaluateLoS()],
                )
                print("-"*80)
                print(f"Running experiment with seed={seed}, task={task.name}, algorithm={experiment.algorithm_name}")
                print("-"*80)
                experiment.run()
        
    




