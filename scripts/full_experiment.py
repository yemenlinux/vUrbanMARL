# An example experiment script of Victorized Urban Multi-Agent Simulation (VUMAS).
import argparse
import gc
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

# Add project root to path (adjust if notebook is in a subfolder)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress standard future warnings from torchrl / tensordict
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torchrl.modules.mcts.scores"
)
warnings.filterwarnings(
    "ignore",
    message=".*TensorDict.to_module().*",
    category=FutureWarning,
    module="tensordict",
)
warnings.filterwarnings(
    "ignore",
    message=".*torch.jit.script.*",
    category=FutureWarning,
)


import torch

# BenchMARL
from benchmarl.algorithms import (
    IddpgConfig,
    IppoConfig,
    IsacConfig,
    MaddpgConfig,
    MappoConfig,
    MasacConfig,
)
from benchmarl.environments import UrbanEnvTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

TASK_MAP = {
    "uav_navigation": UrbanEnvTask.UAV_NAVIGATION,
    "uav_ue_los": UrbanEnvTask.UAV_UE_LOS,
    "coverage": UrbanEnvTask.COVERAGE,
    "uavmec_offloading": UrbanEnvTask.UAVMEC_OFFLOADING,
    "uav_mobile_ue": UrbanEnvTask.UAV_MOBILE_UE,
    "uav_lidar_navigation": UrbanEnvTask.UAV_LIDAR_NAVIGATION,
    "uavmec_advanced_physics": UrbanEnvTask.UAVMEC_ADVANCED_PHYSICS,
    "mec_offloading": UrbanEnvTask.MEC_OFFLOADING,
}

ALGORITHM_CONFIG_MAP = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "maddpg": MaddpgConfig,
    "masac": MasacConfig,
    "iddpg": IddpgConfig,
    "isac": IsacConfig,
}


def config_experiment(
    experiment_config: ExperimentConfig,
    num_envs: int = 72,
    max_n_steps: int = 100,
    max_n_iters: int = 100,
    episodes_per_batch: int = 1,
    experiment_dir: str = "experiments",
    eval_interval: int = 10,
    render: bool = False,
):
    """Configures the BenchMARL experiment settings."""
    frames_per_batch = int(episodes_per_batch * num_envs * max_n_steps)
    max_n_frames = int(frames_per_batch * max_n_iters)
    output_dir = project_root / "outputs" / experiment_dir

    if torch.cuda.is_available():
        experiment_config.device = "cuda"
        experiment_config.sampling_device = "cuda"
        experiment_config.train_device = "cuda"
        experiment_config.buffer_device = "cuda"

    experiment_config.parallel_collection = True
    experiment_config.max_n_iters = max_n_iters
    experiment_config.max_n_frames = max_n_frames
    experiment_config.on_policy_collected_frames_per_batch = frames_per_batch
    experiment_config.on_policy_n_envs_per_worker = num_envs
    experiment_config.off_policy_collected_frames_per_batch = frames_per_batch
    experiment_config.off_policy_n_envs_per_worker = num_envs

    # Cap replay buffer memory to avoid allocating an excessive 1,000,000 frames
    experiment_config.off_policy_memory_size = min(max_n_frames, 200_000)

    experiment_config.render = render
    experiment_config.evaluation_interval = int(eval_interval * frames_per_batch)
    experiment_config.evaluation_episodes = 5
    experiment_config.loggers = ["csv", "tensorboard"]

    experiment_config.save_folder = output_dir
    experiment_config.checkpoint_interval = 0
    experiment_config.checkpoint_at_end = True
    experiment_config.exclude_buffer_from_checkpoint = True


def extract_frame_number(filepath: Path) -> int:
    """Extracts frame number from checkpoint filename."""
    match = re.search(r"checkpoint_(\d+)\.pt", filepath.name)
    return int(match.group(1)) if match else -1


def get_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Finds the checkpoint file with the highest frame count."""
    checkpoints = list(ckpt_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=extract_frame_number)


def get_latest_config(conf_dir: Path) -> Path | None:
    """Finds the config file with the highest run number in texts/."""
    conf_files = list(conf_dir.glob("hparams*.txt"))
    if not conf_files:
        return None

    def extract_run_number(filepath: Path) -> int:
        match = re.search(r"hparams(\d+)\.txt", filepath.name)
        return int(match.group(1)) if match else -1

    return max(conf_files, key=extract_run_number)


def load_experiment_config(filepath: str | Path) -> dict:
    """Parses a BenchMARL hparams configuration file."""
    import ast

    parsed_config = {}
    class_pattern = re.compile(r"<class '(.*?)'>")
    path_pattern = re.compile(r"PosixPath\((.*?)\)")

    with open(filepath, "r") as f:
        for line in f:
            if not line.strip() or ": " not in line:
                continue
            key, val_str = line.split(": ", 1)
            key = key.strip()
            val_str = val_str.strip()
            val_str = class_pattern.sub(r"'\1'", val_str)
            val_str = path_pattern.sub(r"\1", val_str)
            try:
                parsed_val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                parsed_val = val_str
            parsed_config[key] = parsed_val

    return parsed_config


def get_experiment_info(exp_dir: Path) -> dict | None:
    """Extracts metadata (seed, max_n_iters, max_n_frames, on_policy, latest_checkpoint) from an experiment folder."""
    meta = {}
    config_file = exp_dir / "config.pkl"
    if config_file.exists():
        try:
            with open(config_file, "rb") as f:
                task = pickle.load(f)
                task_config = pickle.load(f)
                algorithm_config = pickle.load(f)
                model_config = pickle.load(f)
                seed = pickle.load(f)
                experiment_config = pickle.load(f)
            meta["seed"] = seed
            meta["max_n_iters"] = getattr(experiment_config, "max_n_iters", None)
            meta["max_n_frames"] = getattr(experiment_config, "max_n_frames", None)
            meta["on_policy"] = getattr(algorithm_config, "on_policy", lambda: True)()
        except Exception:
            pass

    # Fallback to texts/hparams*.txt if config.pkl was missing or unreadable
    if "seed" not in meta:
        texts_dir = exp_dir / exp_dir.name / "texts"
        if not texts_dir.exists():
            texts_dir = exp_dir / "texts"
        if texts_dir.exists():
            conf_file = get_latest_config(texts_dir)
            if conf_file is not None:
                try:
                    parsed = load_experiment_config(conf_file)
                    meta["seed"] = parsed.get("seed")
                    exp_cfg = parsed.get("experiment_config", {})
                    meta["max_n_iters"] = exp_cfg.get("max_n_iters")
                    meta["max_n_frames"] = exp_cfg.get("max_n_frames")
                    meta["on_policy"] = parsed.get("on_policy", True)
                except Exception:
                    pass

    if "seed" not in meta:
        return None

    ckpt_dir = exp_dir / "checkpoints"
    meta["latest_checkpoint"] = (
        get_latest_checkpoint(ckpt_dir) if ckpt_dir.exists() else None
    )
    meta["dir"] = exp_dir
    return meta


def find_matching_experiment(
    output_dir: Path, task_name: str, algo_name: str, seed: int
) -> dict | None:
    """Finds the most advanced existing experiment folder matching task, algo, and seed."""
    prefix = f"{algo_name.lower()}_{task_name.lower()}_"
    matches = sorted(output_dir.glob(f"{prefix}*"))
    best_match = None
    for m in matches:
        if m.is_dir():
            info = get_experiment_info(m)
            if info is not None:
                exp_seed = info.get("seed")
                try:
                    exp_seed = int(exp_seed)
                except (ValueError, TypeError):
                    pass
                if exp_seed == seed:
                    if best_match is None:
                        best_match = info
                    else:
                        prev_iters = best_match.get("max_n_iters") or 0
                        curr_iters = info.get("max_n_iters") or 0
                        if curr_iters >= prev_iters:
                            best_match = info
    return best_match


def run_single_experiment(
    task_name: str,
    algo_name: str,
    seed: int,
    num_envs: int,
    max_n_steps: int,
    max_n_iters: int,
    episodes_per_batch: int,
    experiment_dir: str,
    eval_interval: int,
    render: bool,
    restore_file: str | None = None,
):
    """Executes a single experiment run (either fresh or resumed)."""
    frames_per_batch = int(episodes_per_batch * num_envs * max_n_steps)
    max_n_frames = int(frames_per_batch * max_n_iters)
    eval_interval_frames = int(eval_interval * frames_per_batch)

    if restore_file is not None and os.path.exists(restore_file):
        experiment_patch = {
            "max_n_iters": max_n_iters,
            "max_n_frames": max_n_frames,
            "evaluation_interval": eval_interval_frames,
            "render": render,
        }
        print("-" * 80)
        print(f"Resuming experiment from checkpoint: {restore_file}")
        print(f"Target max_n_iters: {max_n_iters}, Target max_n_frames: {max_n_frames}")
        print("-" * 80)
        experiment = Experiment.reload_from_file(
            restore_file=restore_file,
            experiment_patch=experiment_patch,
        )
    else:
        task_key = task_name.lower()
        algo_key = algo_name.lower()

        if task_key not in TASK_MAP:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(TASK_MAP.keys())}"
            )
        if algo_key not in ALGORITHM_CONFIG_MAP:
            raise ValueError(
                f"Unknown algo '{algo_name}'. Available: {list(ALGORITHM_CONFIG_MAP.keys())}"
            )

        task_enum = TASK_MAP[task_key]
        task = task_enum.get_from_yaml()
        algo_config_class = ALGORITHM_CONFIG_MAP[algo_key]
        algorithm_config = algo_config_class.get_from_yaml()

        experiment_config = ExperimentConfig.get_from_yaml()
        config_experiment(
            experiment_config,
            num_envs=num_envs,
            max_n_steps=max_n_steps,
            max_n_iters=max_n_iters,
            episodes_per_batch=episodes_per_batch,
            experiment_dir=experiment_dir,
            eval_interval=eval_interval,
            render=render,
        )
        os.makedirs(experiment_config.save_folder, exist_ok=True)

        model_config = MlpConfig.get_from_yaml()
        critic_model_config = MlpConfig.get_from_yaml()

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed,
            config=experiment_config,
        )

        print("-" * 80)
        print(
            f"Running experiment with seed={seed}, task={task.name}, algorithm={experiment.algorithm_name}"
        )
        print("-" * 80)

    try:
        experiment.run()
    finally:
        experiment.close()
        del experiment
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_seeds(seed_arg: str | int) -> list[int]:
    """Parses seed argument into a list of integer seeds."""
    if isinstance(seed_arg, int):
        return [seed_arg]
    return [int(s.strip()) for s in str(seed_arg).split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="UrbanMARL Vectorized Multi-Agent RL Experiment Suite with Process Isolation & Resumption.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tasks and algorithms for seed 0 (skips already completed runs):
  python scripts/full_experiment.py --seed 0

  # Run all tasks for a new seed (e.g., seed 1):
  python scripts/full_experiment.py --seed 1

  # Run multiple seeds:
  python scripts/full_experiment.py --seed 0,1,2

  # Run a specific task and algorithm:
  python scripts/full_experiment.py --task uav_lidar_navigation --algo masac --seed 0

  # Resume an existing experiment to run for more iterations (e.g., from 100 to 150):
  python scripts/full_experiment.py --task uav_mobile_ue --algo ippo --seed 0 --max_n_iters 150

  # Customize environment batching parameters:
  python scripts/full_experiment.py --num_envs 72 --max_n_steps 100 --episodes_per_batch 1 --max_n_iters 100

  # Enable or disable rendering:
  python scripts/full_experiment.py --seed 1 --render
  python scripts/full_experiment.py --seed 0 --no-render

  # Force re-running from scratch (ignoring existing checkpoints):
  python scripts/full_experiment.py --task uav_mobile_ue --algo mappo --force
        """,
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Internal worker flag to execute a single experiment directly in this process.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task(s) to run (e.g. 'uav_lidar_navigation', comma-separated 'uav_mobile_ue,coverage', or 'all').",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Specific algorithm(s) to run (e.g. 'masac', comma-separated 'mappo,ippo', or 'all').",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="0",
        help="Random seed(s) for the experiment (single integer or comma-separated integers, e.g. '0' or '0,1,2').",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=72,
        help="Number of vectorized parallel environments per worker.",
    )
    parser.add_argument(
        "--max_n_steps",
        type=int,
        default=100,
        help="Maximum time steps per episode horizon.",
    )
    parser.add_argument(
        "--max_n_iters",
        type=int,
        default=100,
        help="Target maximum training iterations to reach.",
    )
    parser.add_argument(
        "--episodes_per_batch",
        type=int,
        default=1,
        help="Number of episodes per collection batch.",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments",
        help="Subdirectory name inside outputs/ where experiment logs and checkpoints are stored.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Evaluation interval frequency in terms of iteration batches.",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable rendering in the experiment configuration (default: render first seed only).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume existing experiments from their latest checkpoint if target max_n_iters is higher.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running experiments from scratch even if checkpoints already exist.",
    )
    parser.add_argument(
        "--restore_file",
        type=str,
        default=None,
        help="Explicit path to a checkpoint (.pt) file to reload and resume.",
    )

    args = parser.parse_args()

    # Worker mode: execute single experiment directly in this subprocess
    if args.single:
        if args.task is None or args.algo is None:
            parser.error("--single requires both --task and --algo")
        render_val = bool(args.render) if args.render is not None else False
        run_single_experiment(
            task_name=args.task,
            algo_name=args.algo,
            seed=int(args.seed),
            num_envs=args.num_envs,
            max_n_steps=args.max_n_steps,
            max_n_iters=args.max_n_iters,
            episodes_per_batch=args.episodes_per_batch,
            experiment_dir=args.experiment_dir,
            eval_interval=args.eval_interval,
            render=render_val,
            restore_file=args.restore_file,
        )
        sys.exit(0)

    # Launcher mode: parse seeds, tasks, and algorithms
    output_dir = project_root / "outputs" / args.experiment_dir
    os.makedirs(output_dir, exist_ok=True)

    seeds = parse_seeds(args.seed)

    if args.task is None or args.task.lower() == "all":
        tasks = list(TASK_MAP.keys())
    else:
        tasks = [t.strip().lower() for t in args.task.split(",") if t.strip()]
        for t in tasks:
            if t not in TASK_MAP:
                parser.error(
                    f"Unknown task '{t}'. Available tasks: {list(TASK_MAP.keys())}"
                )

    if args.algo is None or args.algo.lower() == "all":
        algos = list(ALGORITHM_CONFIG_MAP.keys())
    else:
        algos = [a.strip().lower() for a in args.algo.split(",") if a.strip()]
        for a in algos:
            if a not in ALGORITHM_CONFIG_MAP:
                parser.error(
                    f"Unknown algo '{a}'. Available algos: {list(ALGORITHM_CONFIG_MAP.keys())}"
                )

    print("=" * 80)
    print("UrbanMARL Multi-Agent RL Experiment Suite (Process-Isolated & Resumable)")
    print(f"Output Directory   : {output_dir}")
    print(f"Tasks ({len(tasks):2d})         : {[t.upper() for t in tasks]}")
    print(f"Algorithms ({len(algos):2d})    : {[a.upper() for a in algos]}")
    print(f"Seeds ({len(seeds):2d})         : {seeds}")
    print(f"Vectorized Envs    : {args.num_envs}")
    print(f"Episodes / Batch   : {args.episodes_per_batch}")
    print(
        f"Max Steps / Horizon: {args.max_n_steps} steps, target {args.max_n_iters} iters"
    )
    print(f"Auto-Resume        : {args.resume}")
    render_status = (
        "Enabled"
        if args.render is True
        else ("Disabled" if args.render is False else "First seed only")
    )
    print(f"Rendering          : {render_status}")
    print("=" * 80)

    for seed in seeds:
        for task_name in tasks:
            for algo_name in algos:
                match = find_matching_experiment(output_dir, task_name, algo_name, seed)

                resume_ckpt = None
                if match is not None and not args.force:
                    prev_iters = match.get("max_n_iters") or 0
                    latest_ckpt = match.get("latest_checkpoint")

                    # Check if already completed all requested iterations
                    if latest_ckpt is not None and prev_iters >= args.max_n_iters:
                        print(
                            f"[SKIP] {task_name.upper():25} {algo_name.upper():8} (seed={seed}) "
                            f"already completed ({prev_iters}/{args.max_n_iters} iters) -> {match['dir'].name}"
                        )
                        continue

                    # Check if we can resume with higher iterations
                    if (
                        latest_ckpt is not None
                        and args.max_n_iters > prev_iters
                        and args.resume
                    ):
                        resume_ckpt = latest_ckpt
                        print(
                            f"[RESUME] {task_name.upper():25} {algo_name.upper():8} (seed={seed}) "
                            f"resuming from {prev_iters} to {args.max_n_iters} iters using {latest_ckpt.name}"
                        )

                if args.render is not None:
                    render_for_seed = args.render
                else:
                    render_for_seed = seed == seeds[0]
                render_arg = "--render" if render_for_seed else "--no-render"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--single",
                    "--task",
                    task_name,
                    "--algo",
                    algo_name,
                    "--seed",
                    str(seed),
                    "--num_envs",
                    str(args.num_envs),
                    "--max_n_steps",
                    str(args.max_n_steps),
                    "--max_n_iters",
                    str(args.max_n_iters),
                    "--episodes_per_batch",
                    str(args.episodes_per_batch),
                    "--experiment_dir",
                    args.experiment_dir,
                    "--eval_interval",
                    str(args.eval_interval),
                    render_arg,
                ]
                if resume_ckpt is not None:
                    cmd.extend(["--restore_file", str(resume_ckpt)])

                mode_str = "RESUMING" if resume_ckpt else "LAUNCHING"
                print(f"\n{'=' * 80}")
                print(
                    f"[{mode_str}] Task={task_name.upper()}, Algorithm={algo_name.upper()}, Seed={seed} (in subprocess)"
                )
                print(f"{'=' * 80}\n")

                result = subprocess.run(cmd)

                if result.returncode != 0:
                    print(
                        f"\n[FAILED] Experiment {task_name.upper()} with {algo_name.upper()} (seed={seed}) "
                        f"exited with code {result.returncode}."
                    )
                else:
                    print(
                        f"\n[COMPLETED] Experiment {task_name.upper()} with {algo_name.upper()} (seed={seed}) "
                        f"finished successfully.\n"
                    )


if __name__ == "__main__":
    main()
