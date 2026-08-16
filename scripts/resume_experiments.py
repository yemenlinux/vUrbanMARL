import os
import sys
import re
from pathlib import Path
import warnings
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress standard warnings
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="torchrl.modules.mcts.scores"
)
warnings.filterwarnings(
    "ignore",
    message=".*TensorDict.to_module().*",
    category=FutureWarning,
    module="tensordict"
)

from benchmarl.experiment import Experiment


# NEW_MAX_N_ITERS = 20
global ADDITIONAL_N_ITERS
# ADDITIONAL_N_ITERS = 10

def get_latest_checkpoint(ckpt_dir: Path) -> Path:
    """Finds the checkpoint file with the highest frame count."""
    checkpoints = list(ckpt_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
        
    def extract_frame_number(filepath: Path) -> int:
        match = re.search(r"checkpoint_(\d+)\.pt", filepath.name)
        return int(match.group(1)) if match else -1
        
    return max(checkpoints, key=extract_frame_number)


def get_latest_config(conf_dir: Path) -> Path:
    """Finds the checkpoint file with the highest frame count."""
    conf_files = list(conf_dir.glob("hparams*.txt"))
    if not conf_files:
        return None
        
    def extract_run_number(filepath: Path) -> int:
        match = re.search(r"hparams(\d+)\.txt", filepath.name)
        return int(match.group(1)) if match else -1
        
    return max(conf_files, key=extract_run_number)

def load_experiment_config(filepath: str | Path) -> dict:
    """
    Parses a BenchMARL hparams configuration file containing raw Python 
    objects (like classes and PosixPaths) into a standard dictionary.
    """
    import ast
    
    parsed_config = {}
    
    # Regex patterns to catch and sanitize specific Python objects found in the file
    class_pattern = re.compile(r"<class '(.*?)'>")
    path_pattern = re.compile(r"PosixPath\((.*?)\)")
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            # Split only on the first colon to separate the primary key from its value
            if ': ' in line:
                key, val_str = line.split(': ', 1)
                key = key.strip()
                val_str = val_str.strip()
                
                # Sanitize the value string:
                # Convert <class 'torch.nn...'> to just the string "'torch.nn...'"
                val_str = class_pattern.sub(r"'\1'", val_str)
                # Convert PosixPath('/path/to/dir') to just the string "'/path/to/dir'"
                val_str = path_pattern.sub(r"\1", val_str)
                
                try:
                    # Safely evaluate lists, dicts, booleans, and numbers
                    parsed_val = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    # Fallback to plain string if ast evaluation fails
                    parsed_val = val_str
                    
                parsed_config[key] = parsed_val
                
    return parsed_config

def update_experiment_patch(
    previous_conf_path: str | Path, 
    additional_n_iters: int | None = None) -> dict:
    """Update configuration for resume"""
    config = load_experiment_config(previous_conf_path)
    current_max_n_iters = config['experiment_config']['max_n_iters']
    if additional_n_iters:
        new_max_n_iters = int(current_max_n_iters + additional_n_iters)
    else:
        new_max_n_iters = int(current_max_n_iters + current_max_n_iters)
        
    if config['on_policy']:
        frames_per_batch = config['experiment_config']['on_policy_collected_frames_per_batch']
    else:
        frames_per_batch = config['experiment_config']['off_policy_collected_frames_per_batch']
    #
    max_n_frames = int(frames_per_batch * new_max_n_iters)
    return {
        'max_n_iters': new_max_n_iters,
        'max_n_frames':max_n_frames,
    }

def main():
    parser = argparse.ArgumentParser(description="Resume BenchMARL Experiments")
    parser.add_argument("--experiments_dir", 
                        type=str, 
                        default=str(project_root / "outputs" / "experiments"),
                        help="Directory containing experiment folders to resume.")
    parser.add_argument("--additional_n_iters", type=int, default=10,
                        help="Number of additional iterations to run for each experiment.")
    args = parser.parse_args()
    
    ADDITIONAL_N_ITERS = args.additional_n_iters
    
    experiments_dir = Path(args.experiments_dir)
    
    if not experiments_dir.exists():
        print(f"Target directory does not exist: {experiments_dir}")
        return

    # Iterate through all experiment folders in the directory
    for exp_folder in experiments_dir.iterdir():
        if not exp_folder.is_dir():
            continue
            
        ckpt_dir = exp_folder / "checkpoints"
        if not ckpt_dir.exists():
            print(f"No checkpoints directory found in {exp_folder.name}, skipping.")
            continue
            
        latest_ckpt = get_latest_checkpoint(ckpt_dir)
        if latest_ckpt is None:
            print(f"No checkpoint files found in {ckpt_dir.name}, skipping.")
            continue
            
        previous_config = exp_folder / exp_folder.name / "texts"
        if not previous_config.exists():
            print(f"No config directory found in {exp_folder.name}, skipping.")
            continue
        conf_file = get_latest_config(previous_config)
        # 
        config = load_experiment_config(conf_file)
        updated_conf = update_experiment_patch(conf_file, additional_n_iters=ADDITIONAL_N_ITERS)
            
        # Reload the experiment and patch the configuration to increase the iteration limit
        experiment = Experiment.reload_from_file(
            restore_file=str(latest_ckpt),
            experiment_patch=updated_conf
        )
        #
        print("-"*80)
        print(f"Resuming experiment with seed={experiment.seed}, task={experiment.task_name}, algorithm={experiment.algorithm_name}")
        print("-"*80)
        
        # Resume the training loop
        experiment.run()

if __name__ == "__main__":
    main() 
