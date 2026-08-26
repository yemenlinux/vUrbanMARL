#!/usr/bin/env python3
"""UrbanMARL Results Evaluation and Plotting Suite.

Aggregates, processes, evaluates, and visualizes BenchMARL training CSV metrics
across algorithms, seeds, and urban environment configurations.
"""

import argparse
from datetime import datetime
import glob
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import sem
import seaborn as sns

from benchmarl.eval_results import (
    Plotting,
    get_raw_dict_from_multirun_folder,
    load_and_merge_json_dicts,
)

FIGURE_SIZE = (7, 6)


class ExperimentResult:
    """Container holding metadata and scalar logs for a single experiment run.

    Attributes:
        path (Path): Path to experiment root directory.
        name (str): Experiment folder name.
        algorithm (str): Algorithm name (e.g. 'mappo', 'maddpg').
        scenario (str): Scenario identifier.
        model (str): Model architecture name (e.g. 'mlp').
        scalars_dir (Path): Directory containing CSV metrics logs.
        csv_files (List[Path]): List of CSV file paths.
    """

    def __init__(self, path: Path) -> None:
        """Initializes ExperimentResult.

        Args:
            path (Path): Path to experiment run output directory.
        """
        self.path = path
        self.name = path.name
        parts = self.name.split('_')
        self.algorithm = parts[0]
        self.scenario = parts[1]
        self.model = parts[2]

        self.scalars_dir = path / self.name / "scalars"
        self.csv_files = (
            list(self.scalars_dir.glob("*.csv"))
            if self.scalars_dir.exists()
            else []
        )

    def get_csv_data(self, pattern: str) -> Optional[pd.DataFrame]:
        """Loads CSV scalar data file matching given filename pattern.

        Args:
            pattern (str): Target CSV filename (e.g. 'collection_agents_reward_episode_reward_mean.csv').

        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if file not found or invalid.
        """
        for csv_path in self.csv_files:
            if csv_path.name == pattern:
                metric = os.path.splitext(pattern)[0]
                try:
                    df = pd.read_csv(
                        csv_path, header=None, names=['step', metric]
                    )
                    return df
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    return None
        return None

    @property
    def has_data(self) -> bool:
        """Checks if scalar CSV files exist for this experiment."""
        return len(self.csv_files) > 0


def find_experiments(root: Path) -> List[ExperimentResult]:
    """Discovers all experiment output directories within root path.

    Args:
        root (Path): Root directory containing output experiment runs.

    Returns:
        List[ExperimentResult]: List of valid ExperimentResult instances.
    """
    experiments = []
    for exp_dir in root.glob("*"):
        if exp_dir.is_dir():
            sub_dir = exp_dir / exp_dir.name
            if sub_dir.exists() and (sub_dir / "scalars").exists():
                experiments.append(ExperimentResult(exp_dir))
            else:
                if (exp_dir / "scalars").exists():
                    experiments.append(ExperimentResult(exp_dir))
    return experiments

def load_metric_over_seeds(
    experiments: List[ExperimentResult], 
    pattern: str, 
    metric_col: str) -> pd.DataFrame:
    """
    Load a specific metric from all experiments and return a DataFrame with columns:
    step, seed, algorithm, value.
    """
    data_frames = []
    for exp in experiments:
        df = exp.get_csv_data(pattern)
        if df is not None and not df.empty:
            # Assume the CSV has columns: Step, Value (or the metric_col)
            # Sometimes the column is named 'Value' or directly the metric name.
            # We'll try to find the column that contains the values.
            value_col = None
            for col in df.columns:
                if col.lower() == metric_col.lower() or col.lower() == 'value':
                    value_col = col
                    break
            if value_col is None:
                # If not found, assume the last column is the value
                value_col = df.columns[-1]
            step_col = 'Step' if 'Step' in df.columns else df.columns[0]  # assume first is step
            # Add algorithm and seed info
            df_exp = df[[step_col, value_col]].copy()
            df_exp.columns = ['step', 'value']
            df_exp['algorithm'] = exp.algorithm
            df_exp['seed'] = exp.path.name  # use full name as seed identifier
            data_frames.append(df_exp)
    if not data_frames:
        return pd.DataFrame()
    return pd.concat(data_frames, ignore_index=True)


def aggregate_by_algorithm(
    df: pd.DataFrame, 
    group_cols: List[str] = ['step', 'algorithm']) -> pd.DataFrame:
    """Compute mean and std across seeds for each algorithm."""
    if df.empty:
        return df
    agg = df.groupby(group_cols).agg(
        mean=('value', 'mean'),
        std=('value', 'std'),
        sem=('value', sem),
        n=('value', 'count')
    ).reset_index()
    return agg


def plot_metric(agg_df: pd.DataFrame, metric_name: str, title: str, save_path: Path):
    """Plot learning curves for each algorithm."""
    if agg_df.empty:
        print(f"No data for {metric_name}")
        return
    plt.figure(figsize=FIGURE_SIZE)
    for algo in agg_df['algorithm'].unique():
        algo_data = agg_df[agg_df['algorithm'] == algo]
        plt.plot(algo_data['step'], algo_data['mean'], label=algo)
        plt.fill_between(
            algo_data['step'],
            algo_data['mean'] - algo_data['sem'],
            algo_data['mean'] + algo_data['sem'],
            alpha=0.2
        )
    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# ----------------------------------------------------------------------
#  Utility functions
# ----------------------------------------------------------------------
def pi_formatter(x, pos):
    """Dynamically formats axis ticks as multiples of pi from -pi to pi."""
    # Normalize the value by pi
    val = x / np.pi
    
    # Explicit checks for common angles in [-pi, pi]
    if val == 0:
        return r"$0$"
    elif val == 1:
        return r"$\pi$"
    elif val == -1:
        return r"$-\pi$"
    elif val == 0.5:
        return r"$\frac{\pi}{2}$"
    elif val == -0.5:
        return r"$-\frac{\pi}{2}$"
    elif val.is_integer():
        return rf"${int(val)}\pi$"
    else:
        # Fallback for arbitrary fractions
        from fractions import Fraction
        frac = Fraction(val).limit_denominator()
        
        # Handle the negative sign placement for better LaTeX rendering
        if frac.numerator < 0:
            if frac.numerator == -1:
                return rf"$-\frac{{\pi}}{{{frac.denominator}}}$"
            else:
                return rf"$-\frac{{{-frac.numerator}\pi}}{{{frac.denominator}}}$"
        else:
            if frac.numerator == 1:
                return rf"$-\frac{{\pi}}{{{frac.denominator}}}$"
            else:
                return rf"$\frac{{{frac.numerator}\pi}}{{{frac.denominator}}}$"

def parse_experiment_folder_name(folder_path: str | Path) -> dict:
    folder_str = Path(folder_path).name
    
    # Split primary configuration from execution metadata
    parts = folder_str.split("__")
    if len(parts) != 2:
        raise ValueError(f"Invalid BenchMARL folder structure: {folder_str}")
    
    config_str, run_meta_str = parts[0], parts[1]
    
    # Extract Algorithm, Task, and Model
    config_tokens = config_str.split("_")
    algorithm = config_tokens[0]
    model = config_tokens[-1]
    task = "_".join(config_tokens[1:-1])
    
    # Extract Hash and Timestamp
    run_meta_tokens = run_meta_str.split("_", 1)
    exp_hash = run_meta_tokens[0]
    timestamp_str = run_meta_tokens[1]
    
    # Parse standard date object
    dt = datetime.strptime(timestamp_str, "%y_%m_%d-%H_%M_%S")
    
    return {
        "algorithm": algorithm,
        "task": task,
        "model": model,
        "hash": exp_hash,
        "datetime": dt,
        "timestamp_raw": timestamp_str
    }

def get_algo_colors(dictionary: Dict) -> Dict[str, tuple]:
    """Generates a consistent color palette for all algorithms present 
    in the dataset."""
    unique_algos = set()
    for metric, dfs in dictionary.items():
        for df in dfs:
            unique_algos.update(df['algorithm'].unique())
    
    unique_algos = sorted(list(unique_algos))
    # Using 'colorblind' palette for accessible and distinct colors
    palette = sns.color_palette("colorblind", n_colors=len(unique_algos))
    return {algo: color for algo, color in zip(unique_algos, palette)}

#
def find_experiment_dirs(root_dir):
    """
    Locate all experiment directories under root_dir.
    Each experiment is a folder like: iddpg_navigate_mlp__4cb37893_26_06_21-22_04_23
    Returns a list of dicts with keys:
        - algorithm : algorithm name (e.g., iddpg, ippo, mappo)
        - path   : full path to the experiment folder
        - name   : folder name
        - task   : task name (e.g., uav_navigate)
        - model  : model name (e.g., mlp)
        - hash   : unique hash for the run
        - datetime : datetime object parsed from the timestamp
        - timestamp_raw : raw timestamp string
    """
    experiments = []
    # The top-level folders are directly under root_dir
    for f in os.listdir(root_dir):
        exp_path = os.path.join(root_dir, f)
        if os.path.isdir(exp_path):
            # Extract algorithm name (e.g., "iddpg" from "iddpg_navigate_mlp__...")
            # algo = f.split('_')[0] if '_' in f else f
            info = parse_experiment_folder_name(exp_path)
            info.update({
                'path': exp_path,
                'name': f,
            })
            experiments.append(info)
    return experiments


def get_scalars_dir(exp_info):
    """
    Given an experiment info dict, return the path to the "scalars" folder.
    Expected structure: exp_path / exp_name / scalars
    """
    exp_path = exp_info['path']
    exp_name = exp_info['name']
    scalars_path = os.path.join(exp_path, exp_name, 'scalars')
    if os.path.isdir(scalars_path):
        return scalars_path
    else:
        # Some experiments might have a different structure; try alternative.
        alt_path = os.path.join(exp_path, 'scalars')
        if os.path.isdir(alt_path):
            return alt_path
        else:
            return None


def parse_metric_from_filename(filename):
    """
    Determine metric name and optional environment id from a CSV filename.

    For standard metrics, the filename is the metric name (e.g. "collection_agents_reward_episode_reward_mean.csv").
    For per-environment rewards, filenames follow: "rwd_ {alpha}_{beta}_{gamma}.csv"
    We extract the beta value as the environment id.

    Returns: (metric_base_name, env_id) where env_id may be None.
    """
    metric_keys = {
        "rwd": "episode_reward_per_env_mean",
        "vel": "episode_velocity_per_env_mean"
    }
    
    base = os.path.splitext(filename)[0]  # remove .csv
    # Check for per-env reward pattern: starts with "rwd_" and contains numbers separated by underscores
    # Example: "rwd_ 0.10_725_ 27.20" -> groups: alpha=0.10, beta=725, gamma=27.20
    # We'll look for pattern: rwd_ (\d+\.?\d*) _ (\d+) _ (\d+\.?\d*)
    # But there may be spaces around underscores.
    pattern1 = re.compile(r'^(.*)_\s*([0-9.]+)\s*_\s*([0-9]+)\s*_\s*([0-9.]+)$')
    pattern2 = re.compile(r'^(.*)_\s*([0-9.]+)\s*_\s*([0-9]+)\s*_\s*([0-9.]+)\s*_\s*(-?[0-9.]+)$')
    pattern = pattern1 if pattern1.match(base) else pattern2
    m = pattern.match(base)
    E = None
    if m:
        if len(m.groups()) == 4:
            metric, alpha, beta, gamma = m.groups()
        if len(m.groups()) == 5:
            metric, alpha, beta, gamma, E = m.groups()
        alpha = float(alpha)
        beta = int(beta)
        gamma = float(gamma)
        if E:
            E = float(E)
        else:
            building_width = 1000 * np.sqrt(alpha / beta)
            street_width = 1000/np.sqrt(beta) - building_width
            # complex representation
            ex = (street_width - building_width) + 1j * (street_width - gamma)
            E = np.round(np.arctan2(ex.imag, ex.real), 4)
        return metric, E
        # 
    else:
        # Standard metric
        return base, None


def load_csv_metric(file_path):
    """
    Load a CSV file with no header, two columns: step and value.
    Returns a pandas DataFrame with columns ['step', 'value'].
    """
    try:
        df = pd.read_csv(file_path, header=None, names=['step', 'value'])
        return df
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return None


def load_all_metrics(experiments, metrics_of_interest=None):
    """
    Load all CSV files from all experiments' scalars directories.

    Parameters:
        experiments : list of experiment dicts
        metrics_of_interest : list of metric names (or None for all)

    Returns a dictionary:
        key : metric name (str) 
        value : list of DataFrames, each with columns ['step', 'value', 'exp_name', 'algorithm']
    """
    data = defaultdict(list)

    for exp in experiments:
        scalars_path = get_scalars_dir(exp)
        if scalars_path is None:
            continue

        # Find all CSV files
        csv_files = glob.glob(os.path.join(scalars_path, '*.csv'))
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            metric_name, env_id = parse_metric_from_filename(filename)

            # If we have a specific list, skip others
            if metrics_of_interest is not None and metric_name not in metrics_of_interest:
                continue

            # Load the data
            df = load_csv_metric(csv_file)
            if df is None:
                continue
            # Add experiment info
            df['exp_name'] = exp['name']
            df['algorithm'] = exp['algorithm']
            df['task'] = exp['task']
            df['model'] = exp['model']
            # df['hash'] = exp['hash']
            # df['datetime'] = exp['datetime']
            # df['timestamp_raw'] = exp['timestamp_raw']
            # For per-env rewards, store env_id as an extra column
            if env_id is not None:
                df['env_id'] = env_id
            data[metric_name].append(df)
    return data


def aggregate_runs(metric_dfs):
    """
    Given a list of DataFrames (each with columns: step, value, algo, exp_name),
    aggregate them by step using interpolation to a common step grid,
    then compute mean and std across runs.

    Returns: DataFrame with columns: step, mean, std, (and optionally algo if all same)
    """
    if not metric_dfs:
        return None

    # Determine the common step range (union of all steps)
    all_steps = set()
    for df in metric_dfs:
        all_steps.update(df['step'].values)
    # Use sorted unique steps, maybe we want a dense grid? We'll use all unique steps.
    steps_sorted = sorted(all_steps)

    # Interpolate each run's value onto this grid
    interpolated = []
    for df in metric_dfs:
        # Sort by step to interpolate
        df_sorted = df.sort_values('step')
        # Interpolate linearly
        interp_vals = np.interp(steps_sorted, df_sorted['step'], df_sorted['value'])
        interpolated.append(interp_vals)

    # Convert to array
    interp_array = np.array(interpolated)  # shape: (num_runs, num_steps)

    # Compute mean and std across runs
    mean_vals = np.mean(interp_array, axis=0)
    std_vals = np.std(interp_array, axis=0)

    result = pd.DataFrame({
        'step': steps_sorted,
        'mean': mean_vals,
        'std': std_vals
    })
    return result

def aggregate_runs_by_envs(metric_dfs):
    """
    Given a list of DataFrames (each with columns: step, value, algo, exp_name),
    aggregate them by step using interpolation to a common step grid,
    then compute mean and std across runs.

    Returns: DataFrame with columns: step, mean, std, (and optionally algo if all same)
    """
    if not metric_dfs:
        return None

    # Determine the common step range (union of all steps)
    all_envs = set()
    for df in metric_dfs:
        all_envs.update(df['env_id'].values)
    # Use sorted unique steps, maybe we want a dense grid? We'll use all unique steps.
    env_sorted = sorted(all_envs)
    env_sorted = np.linspace(-np.pi, np.pi, 100)

    # Interpolate each run's value onto this grid
    interpolated = []
    for df in metric_dfs:
        # Sort by step to interpolate
        df_sorted = df.sort_values('env_id')
        # Interpolate linearly
        interp_vals = np.interp(env_sorted, df_sorted['env_id'], df_sorted['value'])
        interpolated.append(interp_vals)

    # Convert to array
    interp_array = np.array(interpolated)  # shape: (num_runs, num_steps)

    # Compute mean and std across runs
    mean_vals = np.mean(interp_array, axis=0)
    std_vals = np.std(interp_array, axis=0)

    result = pd.DataFrame({
        'E': env_sorted,
        'mean': mean_vals,
        'std': std_vals
    })
    return result

def plot_metric_by_envs(
    metric_name, 
    aggregated_dict, 
    output_dir, 
    algo_names=None,
    pdf=False
    ):
    """
    Plot aggregated metric for each algorithm (from aggregated_dict) and save figure.
    """
    if not aggregated_dict:
        print(f"No data to plot for metric: {metric_name}")
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, subplot_kw={'projection': 'polar'})
    # plt.figure(figsize=(10, 6), projection='polar')
    for algo, df in aggregated_dict.items():
        if df is None:
            continue
        ax.plot(df['E'], df['mean'], label=algo.upper())
        ax.fill_between(df['E'], df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.2)

    
    ax.set_xlabel('E')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if pdf:
            save_path = os.path.join(output_dir, f'{metric_name}.pdf')
        else:
            save_path = os.path.join(output_dir, f'{metric_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_metric(
    metric_name, 
    aggregated_dict, 
    output_dir, 
    algo_names=None,
    pdf=False
):
    """
    Plot aggregated metric for each algorithm (from aggregated_dict) and save figure.
    """
    if not aggregated_dict:
        print(f"No data to plot for metric: {metric_name}")
        return

    plt.figure(figsize=FIGURE_SIZE)
    for algo, df in aggregated_dict.items():
        if df is None:
            continue
        plt.plot(df['step'], df['mean'], label=algo)
        plt.fill_between(df['step'], df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.2)

    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.title(f'Learning curve - {metric_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # save_path = os.path.join(output_dir, f'{metric_name}.png')
        if pdf:
            save_path = os.path.join(output_dir, f'{metric_name}.pdf')
        else:
            save_path = os.path.join(output_dir, f'{metric_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()
    
def calculate_percentile(group, percentile):
    return np.percentile(group, percentile)

percentile_95 = lambda x: calculate_percentile(x, 95)

def metric_labels(
    metric_name,
    info_metrics = ['reward', 'los', 'collisions', 'velocity']
):
    if metric_name in info_metrics:
        x_label = r"$\mathcal{E}$"
        title = f"{metric_name} per Urban Environements".title()
    else:
        x_label = 'Episode'
        title = f"{metric_name}".replace('_', ' ').title()
    if metric_name.startswith('timers_'):
        y_label = f"{metric_name.replace('timers_', '_')}".replace('_', ' ').title() + ' (s)'
    elif metric_name.startswith('train_'):
        y_label = metric_name.replace('train_', '').replace('_', ' ').title()
    elif metric_name == 'eval_reward_episode_len_mean':
        y_label = 'Mean Steps per Episode'.title()
    else:
        y_label = metric_name.replace('_', ' ').title().replace('Los', 'LoS')
    return x_label, y_label, title

def plot_all_metrics(
    dictionary: Dict,
    output_dir, 
    algo_colors: Dict[str, tuple],
    pdf=False,
    fill = True
):
    for metric, value in dictionary.items():
        con_df = pd.concat(dictionary[metric])
        #
        if 'env_id' in con_df.columns:
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=FIGURE_SIZE)  
            ax.set_xlim(-np.pi, np.pi)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
            ax.set_rlabel_position(45)
            ax.yaxis.set_label_coords(0.5, 0.6)
        else:
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)  
        for algo in con_df.algorithm.unique():
            _df = con_df.loc[con_df['algorithm'] == algo]
            #
            max_limit = np.max(_df['value'].values)
            min_limit = np.min(_df['value'].values)
            #
            algo_color = algo_colors.get(algo)
            if 'env_id' in _df.columns:
                # 
                x_label = 'env_id'
                df = _df.groupby([x_label])[['value']].agg(percentile_95)
                # df = _df.groupby(['env_id'])[['value']].agg(['max'])
                df.reset_index(inplace=True)
                df.columns = [x_label, 'mean']
                df['std'] = df['mean'].std()
                E = df[x_label]
                metric_mean = df['mean'] 
                metric_std = df['std'] 
                # 
                ax.plot(E, metric_mean, label= algo.upper(), color=algo_color)
                if fill:
                    lower_fill = np.clip(metric_mean - metric_std, a_min=min_limit, a_max=None)
                    upper_fill = np.clip(metric_mean + metric_std, a_min=None, a_max=max_limit)
                    ax.fill_between(E, lower_fill, upper_fill, alpha=0.2, color=algo_color)
            else:
                x_label = 'step'
                df = _df.groupby([x_label])[['value']].agg(percentile_95)
                # df = df.groupby(['env_id'])[['value']].agg(['max'])
                df.reset_index(inplace=True)
                df.columns = [x_label, 'mean']
                df['std'] = df['mean'].std()
                E = df[x_label]
                metric_mean = df['mean'] 
                metric_std = df['std'] 
                #
                ax.plot(E, metric_mean, label= algo.upper(), color=algo_color)
                if fill:
                    lower_fill = np.clip(metric_mean - metric_std, a_min=min_limit, a_max=None)
                    upper_fill = np.clip(metric_mean + metric_std, a_min=None, a_max=max_limit)
                    ax.fill_between(E, lower_fill, upper_fill, alpha=0.2, color=algo_color)

        #
        x_label, y_label, title = metric_labels(metric)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        #
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 
        # Save
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if pdf:
                save_path = os.path.join(output_dir, f'{metric}.pdf')
            else:
                save_path = os.path.join(output_dir, f'{metric}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
        else:
            plt.show()
        plt.close()
        
def plot_group_metrics(
    dictionary: Dict,
    algo_colors: Dict[str, tuple],
    output_dir = None, 
    metric_list= [],
    n_cols = 2,
    projection = 'cartesian',
    file_name = 'compare_metrics',
    figsize = FIGURE_SIZE,
    pdf=False,
    fill = True
):
    n_metrics = len(metric_list)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    if projection.lower() == 'polar':
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 subplot_kw={'projection': 'polar'},
                                 figsize=(figsize[0]*n_cols, figsize[1]*n_rows+1),
                                 squeeze=False)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 # subplot_kw={'projection': 'polar'},
                                 figsize=(figsize[0]*n_cols, figsize[1]*n_rows),
                                 squeeze=False)
    
    for idx, metric in enumerate(metric_list):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        if projection.lower() == 'polar':
            ax.set_xlim(-np.pi, np.pi)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
            ax.set_rlabel_position(45)
            ax.yaxis.set_label_coords(0.5, 0.6)
        #
        con_df = pd.concat(dictionary[metric])
        #
        for algo in sorted(list(con_df.algorithm.unique())):
            _df = con_df.loc[con_df['algorithm'] == algo]
            #
            max_limit = np.max(_df['value'].values)
            min_limit = np.min(_df['value'].values)
            #
            algo_color = algo_colors.get(algo)
            if 'env_id' in _df.columns:
                # 
                x_label = 'env_id'
                df = _df.groupby([x_label])[['value']].agg(percentile_95)
                # df = _df.groupby(['env_id'])[['value']].agg(['max'])
                df.reset_index(inplace=True)
                df.columns = [x_label, 'mean']
                df['std'] = df['mean'].std()
                E = df[x_label]
                metric_mean = df['mean'] 
                metric_std = df['std'] 
                # 
                ax.plot(E, metric_mean, label= algo.upper(), linewidth=2.0, color=algo_color)
                if fill:
                    lower_fill = np.clip(metric_mean - metric_std, a_min=min_limit, a_max=None)
                    upper_fill = np.clip(metric_mean + metric_std, a_min=None, a_max=max_limit)
                    ax.fill_between(E, lower_fill, upper_fill, alpha=0.2, color=algo_color)
            else:
                x_label = 'step'
                df = _df.groupby([x_label])[['value']].agg(percentile_95)
                # df = df.groupby(['env_id'])[['value']].agg(['max'])
                df.reset_index(inplace=True)
                df.columns = [x_label, 'mean']
                df['std'] = df['mean'].std()
                E = df[x_label]
                metric_mean = df['mean'] 
                metric_std = df['std'] 
                #
                ax.plot(E, metric_mean, label= algo.upper(), linewidth=2.0, color=algo_color)
                if fill:
                    lower_fill = np.clip(metric_mean - metric_std, a_min=min_limit, a_max=None)
                    upper_fill = np.clip(metric_mean + metric_std, a_min=None, a_max=max_limit)
                    ax.fill_between(E, lower_fill, upper_fill, alpha=0.2, color=algo_color)

        #
        x_label, y_label, title = metric_labels(metric)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    #
    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if pdf:
            save_path = os.path.join(output_dir, f'{file_name}.pdf')
        else:
            save_path = os.path.join(output_dir, f'{file_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_group_tasks(
    dictionary: Dict,
    metric,
    algo_colors: Dict[str, tuple],
    output_dir = None, 
    n_cols = 3,
    projection = 'cartesian',
    file_name = 'compare_metrics',
    figsize = FIGURE_SIZE,
    pdf=False,
    fill = True
):
    con_df = pd.concat(dictionary[metric])
    n_tasks = len(list(con_df.task.unique()))
    if n_tasks < 4:
        n_cols = n_tasks
    #
    n_rows = (n_tasks + n_cols - 1) // n_cols
    if projection.lower() == 'polar':
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 subplot_kw={'projection': 'polar'},
                                 figsize=(figsize[0]*n_cols, figsize[1]*n_rows+1),
                                 squeeze=False)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 # subplot_kw={'projection': 'polar'},
                                 figsize=(figsize[0]*n_cols, figsize[1]*n_rows),
                                 squeeze=False)
    
    for idx, task in enumerate(sorted(list(con_df.task.unique()))):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        if projection.lower() == 'polar':
            ax.set_xlim(-np.pi, np.pi)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
            ax.set_rlabel_position(45)
            ax.yaxis.set_label_coords(0.5, 0.6)
        #
        _con_df = con_df.loc[con_df['task'] == task]
        #
        for algo in sorted(list(_con_df.algorithm.unique())):
            _df = _con_df.loc[_con_df['algorithm'] == algo]
            #
            max_limit = np.max(_df['value'].values)
            min_limit = np.min(_df['value'].values)
            #
            algo_color = algo_colors.get(algo)
            if 'env_id' in _df.columns:
                # 
                x_label = 'env_id'
                df = _df.groupby([x_label])[['value']].agg(percentile_95)
                # df = _df.groupby(['env_id'])[['value']].agg(['max'])
                df.reset_index(inplace=True)
                df.columns = [x_label, 'mean']
                df['std'] = df['mean'].std()
                E = df[x_label]
                metric_mean = df['mean'] 
                metric_std = df['std'] 
                # 
                ax.plot(E, metric_mean, label= algo.upper(), linewidth=2.0, color=algo_color)
                if fill:
                    lower_fill = np.clip(metric_mean - metric_std, a_min=min_limit, a_max=None)
                    upper_fill = np.clip(metric_mean + metric_std, a_min=None, a_max=max_limit)
                    ax.fill_between(E, lower_fill, upper_fill, alpha=0.2, color=algo_color)
            else:
                x_label = 'step'
                df = _df.groupby([x_label])[['value']].agg(percentile_95)
                # df = df.groupby(['env_id'])[['value']].agg(['max'])
                df.reset_index(inplace=True)
                df.columns = [x_label, 'mean']
                df['std'] = df['mean'].std()
                E = df[x_label]
                metric_mean = df['mean'] 
                metric_std = df['std'] 
                #
                ax.plot(E, metric_mean, label= algo.upper(), linewidth=2.0, color=algo_color)
                if fill:
                    lower_fill = np.clip(metric_mean - metric_std, a_min=min_limit, a_max=None)
                    upper_fill = np.clip(metric_mean + metric_std, a_min=None, a_max=max_limit)
                    ax.fill_between(E, lower_fill, upper_fill, alpha=0.2, color=algo_color)
            
        #
        x_label, y_label, title = metric_labels(metric)
        title = f"{title} - {task.upper()}"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    #
    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if pdf:
            save_path = os.path.join(output_dir, f'{file_name}.pdf')
        else:
            save_path = os.path.join(output_dir, f'{file_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_catalogue(
    dictionary: Dict,
    output_dir,
    algo_colors: Dict[str, tuple],
    figsize = FIGURE_SIZE,
    pdf=False,
    fill = True
):
    catalog_dir = output_dir / 'catalogue'
    # draw all metrics
    plot_all_metrics(dictionary, catalog_dir, algo_colors=algo_colors, pdf=pdf)
    # plot groups
    time_metrics = [
        'timers_collection_time',
        'timers_training_time',
        'timers_evaluation_time',
        'timers_iteration_time',
    ]
    plot_group_metrics(
        dictionary,
        algo_colors,
        catalog_dir, 
        metric_list= time_metrics,
        n_cols = 2,
        projection = 'cartesian',
        file_name = 'grouped_timers',
        figsize = figsize,
        pdf=pdf
    )
    #
    info_metrics = [
        'los',
        'collisions',
        'velocity',
        'reward',
    ]
    plot_group_metrics(
        dictionary,
        algo_colors,
        catalog_dir, 
        metric_list= info_metrics,
        n_cols = 2,
        projection = 'polar',
        file_name = 'grouped_info_metrics',
        figsize = figsize,
        pdf=pdf
    )

def plot_training_metrics(
    dictionary: Dict,
    output_dir, 
    algo_colors: Dict[str, tuple],
    figsize = FIGURE_SIZE,
    n_cols = 3,
    pdf=True,
    fill = False
):
    training_metrics = [
        'train_uav_ESS',
        'train_uav_alpha',
        'train_uav_clip_fraction',
        'train_uav_entropy',
        'train_uav_explained_variance',
        'train_uav_grad_norm_loss_actor',
        'train_uav_grad_norm_loss_alpha',
        'train_uav_grad_norm_loss_critic',
        'train_uav_grad_norm_loss_objective',
        'train_uav_grad_norm_loss_qvalue',
        'train_uav_grad_norm_loss_value',
        'train_uav_kl_approx',
        'train_uav_loss_actor',
        'train_uav_loss_alpha',
        'train_uav_loss_critic',
        'train_uav_loss_entropy',
        'train_uav_loss_objective',
        'train_uav_loss_qvalue',
        'train_uav_loss_value',
        'train_uav_pred_value',
        'train_uav_pred_value_max',
        'train_uav_target_value',
        'train_uav_target_value_max',
        'train_uav_td_error',
    ]
    plot_group_metrics(
        dictionary,
        algo_colors,
        output_dir, 
        metric_list= training_metrics,
        n_cols = n_cols,
        projection = 'cartesian',
        file_name = 'grouped_training_metrics',
        figsize = figsize,
        pdf=pdf,
        fill = fill
    )


def plot_report(
    dictionary: Dict,
    output_dir, 
    algo_colors: Dict[str, tuple],
    figsize = FIGURE_SIZE,
    pdf=True,
    fill = True
):
    # draw all metrics
    info_metrics = [
        'los',
        'collisions',
        'velocity',
        'reward',
    ]
    new_dict = {metric: dfs for metric, dfs in dictionary.items() if metric in info_metrics}
    plot_all_metrics(new_dict, output_dir, algo_colors=algo_colors, pdf=pdf)
    # plot groups per task
    for metric in info_metrics:
        plot_group_tasks(
            dictionary,
            metric,
            algo_colors,
            output_dir=output_dir, 
            n_cols = 3,
            projection = 'polar',
            file_name = f'grouped_{metric}_per_task',
            figsize = figsize,
            pdf=pdf
        )
    # plot groups
    time_metrics = [
        'timers_collection_time',
        'timers_training_time',
        'timers_evaluation_time',
        'timers_iteration_time',
    ]
    plot_group_metrics(
        dictionary,
        algo_colors,
        output_dir, 
        metric_list= time_metrics,
        n_cols = 2,
        projection = 'cartesian',
        file_name = 'grouped_timers',
        figsize = figsize,
        pdf=pdf
    )
    # plot groups per task
    for metric in time_metrics:
        plot_group_tasks(
            dictionary,
            metric,
            algo_colors,
            output_dir=output_dir, 
            n_cols = 3,
            projection = 'cartesian',
            file_name = f'grouped_{metric}_per_task',
            figsize = figsize,
            pdf=pdf
        )
    #
    info_metrics = [
        'los',
        'collisions',
        'velocity',
        'reward',
    ]
    plot_group_metrics(
        dictionary,
        algo_colors,
        output_dir, 
        metric_list= info_metrics,
        n_cols = 2,
        projection = 'polar',
        file_name = 'grouped_info_metrics',
        figsize = figsize,
        pdf=pdf
    )

def extract_marl_eval_colors(fig) -> dict:
    """Extracts the algorithm color mapping from a marl-eval figure."""
    algo_colors = {}
    
    # Iterate through all axes in the figure
    for ax in fig.axes:
        # Iterate through all lines plotted in the axis
        for line in ax.get_lines():
            label = line.get_label()
            # Ignore hidden lines or Matplotlib artifacts
            if label and not label.startswith('_'):
                # marl-eval often capitalizes labels; convert to lower to match your df
                algo_colors[label.lower()] = line.get_color()
                
    return algo_colors



def main():
    parser = argparse.ArgumentParser(description='Aggregate and plot BenchMARL metrics.')
    parser.add_argument('--exp_dir', '-e',
                        default='experiments',
                        help='Root directory containing experiment folders (e.g., outputs/experiments)')
    parser.add_argument('--output', '-o', 
                        default='plots', help='Directory to save plots (default: ./plots)')
    parser.add_argument('--metrics', '-m', nargs='*', help='List of specific metrics to plot (e.g., collection_agents_reward_episode_reward_mean). If not given, plot a default set.')
    args = parser.parse_args()

    experiments_dir = project_root / "outputs" / args.exp_dir
    plot_dir = project_root / "outputs" / args.output / args.exp_dir
    #
    if not experiments_dir.exists():
        print(f"Error: Experiment directory {experiments_dir} does not exist.")
        return
    #
    os.makedirs(plot_dir, exist_ok=True)
    # draw marl-eval style plots
    raw_dict = get_raw_dict_from_multirun_folder(
        multirun_folder=experiments_dir
    )
    processed_data = Plotting.process_data(raw_dict)
    (
        environment_comparison_matrix,
        sample_efficiency_matrix,
    ) = Plotting.create_matrices(processed_data, env_name="urbanmarl")
    
    performance_profile_figure = Plotting.performance_profile_figure(
        environment_comparison_matrix=environment_comparison_matrix
    )
    save_path = plot_dir / "performance_profile.pdf"
    performance_profile_figure.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    aggregate_scores, mean_table, ci_table = Plotting.aggregate_scores(
        environment_comparison_matrix=environment_comparison_matrix,
        save_tabular_as_latex=True
    )
    save_path = plot_dir / "aggregate_scores.pdf"
    aggregate_scores.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    
    src_csv = Path("aggregated_score_return.csv")
    src_tex = Path("aggregated_score_return_latex.txt")
    if src_csv.exists():
        shutil.move(src_csv, plot_dir / "aggregate_scores_return.csv")
    if src_tex.exists():
        shutil.move(src_tex, plot_dir / "aggregated_score_return_latex.tex")
    
    print(f"Tabular data saved to {plot_dir}")
    
    environemnt_sample_efficiency_curves, _, _ = Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix
    )
    save_path = plot_dir / "environemnt_sample_efficiency_curves.pdf"
    environemnt_sample_efficiency_curves.figure.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    
    for task in processed_data['urbanmarl'].keys():
        task_sample_efficiency_curves = Plotting.task_sample_efficiency_curves(
            processed_data=processed_data, env="urbanmarl", task=task
        )
        #
        save_path = plot_dir / f"{task}_sample_efficiency_curves.pdf"
        # task_sample_efficiency_curves.set_title(f"{task.upper()} Sample Efficiency", fontsize=16)
        task_sample_efficiency_curves.figure.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    
    # -------- draw scallers  -------- 
    experiments = find_experiment_dirs(experiments_dir)
    if not experiments:
        print(f"No experiment directories found in {experiments_dir}")
    print(f"Found {len(experiments)} experiment directories.")
    
    all_data = load_all_metrics(experiments)
    
    # Generate unified colors
    # algo_colors = get_algo_colors(all_data)
    algo_colors = extract_marl_eval_colors(performance_profile_figure)
    
    # draw catalog of metrics
    plot_catalogue(
        all_data,
        plot_dir,
        algo_colors,
        figsize = FIGURE_SIZE,
        pdf=False
    )
    
    plot_training_metrics(
        all_data,
        plot_dir, 
        algo_colors,
        figsize = FIGURE_SIZE,
        n_cols = 4,
        pdf=True,
        fill = False
    )
    
    plot_report(
        all_data,
        plot_dir, 
        algo_colors,
        figsize = FIGURE_SIZE,
        pdf=True
    )
    
    


if __name__ == '__main__':
    main() 
