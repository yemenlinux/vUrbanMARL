#!/usr/bin/env python3
"""UrbanMARL Results Evaluation and Plotting Suite.

Aggregates, processes, evaluates, and visualizes BenchMARL training CSV metrics
across algorithms, seeds, and urban environment configurations using an
object-oriented architecture.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Add project root to sys.path if not present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from benchmarl.eval_results import (
    Plotting,
    get_raw_dict_from_multirun_folder,
    load_and_merge_json_dicts,
)
from scipy.stats import sem

FIGURE_SIZE: Tuple[int, int] = (7, 6)


class MetricParser:
    """Parser and formatter for urban environment and scalar metrics."""

    PER_ENV_PATTERN_1 = re.compile(
        r"^(.*)_\s*([0-9.]+)\s*_\s*([0-9]+)\s*_\s*([0-9.]+)$"
    )
    PER_ENV_PATTERN_2 = re.compile(
        r"^(.*)_\s*([0-9.]+)\s*_\s*([0-9]+)\s*_\s*([0-9.]+)\s*_\s*(-?[0-9.]+)$"
    )

    METRIC_ALIASES: Dict[str, str] = {
        "rwd": "reward",
        "vel": "velocity",
    }

    REVERSE_ALIASES: Dict[str, List[str]] = {
        "reward": ["rwd"],
        "velocity": ["vel"],
    }

    @staticmethod
    def pi_formatter(x: float, pos: Any = None) -> str:
        """Dynamically formats axis ticks as multiples of pi from -pi to pi."""
        val = x / np.pi
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
            frac = Fraction(val).limit_denominator()
            if frac.numerator < 0:
                if frac.numerator == -1:
                    return rf"$-\frac{{\pi}}{{{frac.denominator}}}$"
                else:
                    return rf"$-\frac{{{-frac.numerator}\pi}}{{{frac.denominator}}}$"
            else:
                if frac.numerator == 1:
                    return rf"$\frac{{\pi}}{{{frac.denominator}}}$"
                else:
                    return rf"$\frac{{{frac.numerator}\pi}}{{{frac.denominator}}}$"

    @classmethod
    def calculate_complex_angle(
        cls, alpha: float, beta: int, gamma: float
    ) -> float:
        """Calculates complex angle representation E for urban parameters."""
        building_width = 1000 * np.sqrt(alpha / beta)
        street_width = 1000 / np.sqrt(beta) - building_width
        ex = (street_width - building_width) + 1j * (street_width - gamma)
        return float(np.round(np.arctan2(ex.imag, ex.real), 4))

    @classmethod
    def parse_filename(cls, filename: str) -> Tuple[str, Optional[float]]:
        """Parses a scalar CSV filename into metric name and optional env angle E.

        Args:
            filename: CSV filename (e.g. 'collisions_0.24_710_45.00_-1.5415.csv').

        Returns:
            Tuple of (metric_name, env_id). env_id is float if per-env, else None.
        """
        base = os.path.splitext(filename)[0]
        pattern = (
            cls.PER_ENV_PATTERN_1
            if cls.PER_ENV_PATTERN_1.match(base)
            else cls.PER_ENV_PATTERN_2
        )
        m = pattern.match(base)
        if m:
            groups = m.groups()
            metric = groups[0]
            alpha = float(groups[1])
            beta = int(groups[2])
            gamma = float(groups[3])
            if len(groups) == 5 and groups[4] is not None:
                E = float(groups[4])
            else:
                E = cls.calculate_complex_angle(alpha, beta, gamma)
            metric = cls.METRIC_ALIASES.get(metric, metric)
            return metric, E
        return base, None

    @classmethod
    def get_labels(
        cls,
        metric_name: str,
        info_metrics: Sequence[str] = (
            "reward",
            "rwd",
            "los",
            "collisions",
            "velocity",
            "mean_clearance",
            "reached_goals",
            "mean_battery",
            "completed_tasks",
            "dropped_tasks",
            "mean_system_time",
            "mean_utilization",
            "sojourn_time",
        ),
    ) -> Tuple[str, str, str]:
        """Generates formatting labels and titles for a given metric."""
        if metric_name in info_metrics:
            x_label = r"$\mathcal{E}$"
            title = f"{metric_name} per Urban Environements".title()
        else:
            x_label = "Episode"
            title = f"{metric_name}".replace("_", " ").title()

        if metric_name.startswith("timers_"):
            y_label = (
                f"{metric_name.replace('timers_', '_')}".replace("_", " ").title()
                + " (s)"
            )
        elif metric_name.startswith("train_"):
            y_label = metric_name.replace("train_", "").replace("_", " ").title()
        elif metric_name == "eval_reward_episode_len_mean":
            y_label = "Mean Steps per Episode".title()
        else:
            y_label = metric_name.replace("_", " ").title().replace("Los", "LoS")

        return x_label, y_label, title


class ExperimentRun:
    """Container holding metadata and scalar logs for a single experiment run.

    Attributes:
        path (Path): Path to experiment root directory.
        name (str): Experiment folder name.
        algorithm (str): Algorithm name (e.g. 'mappo', 'iddpg').
        task (str): Task identifier (e.g. 'uav_navigate').
        model (str): Model architecture name (e.g. 'mlp').
        hash (str): Hash identifier for the run.
        datetime (Optional[datetime]): Timestamp datetime object.
        timestamp_raw (str): Raw timestamp string.
        scalars_dir (Optional[Path]): Directory containing CSV metric logs.
        csv_files (List[Path]): List of CSV file paths.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path).resolve()
        self.name = self.path.name
        self.metadata = self._parse_folder_name(self.name)
        self.algorithm: str = self.metadata["algorithm"]
        self.task: str = self.metadata["task"]
        self.model: str = self.metadata["model"]
        self.hash: str = self.metadata["hash"]
        self.datetime: Optional[datetime] = self.metadata["datetime"]
        self.timestamp_raw: str = self.metadata["timestamp_raw"]

        # Backwards compatibility scenario alias
        self.scenario = self.task

        self.scalars_dir = self._find_scalars_dir()
        self.csv_files: List[Path] = (
            list(self.scalars_dir.glob("*.csv"))
            if self.scalars_dir and self.scalars_dir.exists()
            else []
        )

    def _find_scalars_dir(self) -> Optional[Path]:
        nested = self.path / self.name / "scalars"
        if nested.is_dir():
            return nested
        direct = self.path / "scalars"
        if direct.is_dir():
            return direct
        return None

    @staticmethod
    def _parse_folder_name(folder_str: str) -> Dict[str, Any]:
        """Parses algorithm, task, model, hash, and timestamp from folder name."""
        if "__" in folder_str:
            parts = folder_str.split("__")
            config_str, run_meta_str = parts[0], parts[1]
            config_tokens = config_str.split("_")
            algorithm = config_tokens[0]
            model = config_tokens[-1]
            task = "_".join(config_tokens[1:-1])

            run_meta_tokens = run_meta_str.split("_", 1)
            exp_hash = run_meta_tokens[0]
            timestamp_str = run_meta_tokens[1] if len(run_meta_tokens) > 1 else ""
            try:
                dt = datetime.strptime(timestamp_str, "%y_%m_%d-%H_%M_%S")
            except Exception:
                dt = None
        else:
            tokens = folder_str.split("_")
            algorithm = tokens[0] if tokens else "unknown"
            task = tokens[1] if len(tokens) > 1 else "unknown"
            model = tokens[2] if len(tokens) > 2 else "unknown"
            exp_hash = "unknown"
            timestamp_str = ""
            dt = None

        return {
            "algorithm": algorithm,
            "task": task,
            "model": model,
            "hash": exp_hash,
            "datetime": dt,
            "timestamp_raw": timestamp_str,
        }

    @property
    def has_data(self) -> bool:
        """Checks if scalar CSV files exist for this experiment."""
        return len(self.csv_files) > 0

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
                    df = pd.read_csv(csv_path, header=None, names=["step", metric])
                    return df
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    return None
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Returns metadata dictionary for backwards compatibility."""
        return {
            "algorithm": self.algorithm,
            "task": self.task,
            "model": self.model,
            "hash": self.hash,
            "datetime": self.datetime,
            "timestamp_raw": self.timestamp_raw,
            "path": str(self.path),
            "name": self.name,
        }


# Alias for legacy compatibility
ExperimentResult = ExperimentRun


class PaletteManager:
    """Manages consistent color mapping for algorithms across visualizations."""

    def __init__(self, palette_name: str = "colorblind") -> None:
        self.palette_name = palette_name
        self.colors: Dict[str, tuple] = {}

    def extract_from_figure(self, fig: plt.Figure) -> Dict[str, tuple]:
        """Extracts algorithm color mapping from a marl-eval figure."""
        for ax in fig.axes:
            for line in ax.get_lines():
                label = line.get_label()
                if label and not label.startswith("_"):
                    self.colors[label.lower()] = line.get_color()
        return self.colors

    def generate_palette(self, algorithms: Sequence[str]) -> Dict[str, tuple]:
        """Generates colorblind palette for given list of algorithms."""
        unique_algos = sorted(list(set(a.lower() for a in algorithms)))
        palette = sns.color_palette(self.palette_name, n_colors=max(len(unique_algos), 1))
        for algo, color in zip(unique_algos, palette):
            self.colors[algo] = color
        return self.colors

    def get_color(self, algo: str) -> tuple:
        """Gets color for an algorithm, generating on the fly if needed."""
        key = algo.lower()
        if key not in self.colors:
            self.generate_palette([key])
        return self.colors.get(key, (0.2, 0.2, 0.2))


class EvaluationDataset:
    """Manages experiment runs, loading, and structured aggregation of scalar metrics."""

    def __init__(
        self,
        experiments: Sequence[Union[ExperimentRun, Dict[str, Any], Path, str]],
    ) -> None:
        self.runs: List[ExperimentRun] = []
        for exp in experiments:
            if isinstance(exp, ExperimentRun):
                self.runs.append(exp)
            elif isinstance(exp, dict) and "path" in exp:
                self.runs.append(ExperimentRun(exp["path"]))
            elif isinstance(exp, (str, Path)):
                self.runs.append(ExperimentRun(exp))
        self.data: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        self._combined_cache: Dict[str, pd.DataFrame] = {}

    @classmethod
    def from_directory(cls, root: Union[str, Path]) -> "EvaluationDataset":
        """Discovers all experiment runs within root directory."""
        root_path = Path(root).resolve()
        experiments = []
        if root_path.exists():
            for child in sorted(root_path.glob("*")):
                if child.is_dir():
                    try:
                        run = ExperimentRun(child)
                        if run.has_data or (child / child.name / "scalars").exists():
                            experiments.append(run)
                    except Exception:
                        pass
        return cls(experiments)

    def load_all_metrics(
        self, metrics_of_interest: Optional[Sequence[str]] = None
    ) -> Dict[str, List[pd.DataFrame]]:
        """Loads all CSV files from all experiment runs' scalars directories."""
        self.data.clear()
        self._combined_cache.clear()

        for run in self.runs:
            if not run.scalars_dir or not run.scalars_dir.exists():
                continue
            for csv_file in run.csv_files:
                filename = csv_file.name
                metric_name, env_id = MetricParser.parse_filename(filename)

                if (
                    metrics_of_interest is not None
                    and metric_name not in metrics_of_interest
                ):
                    continue

                try:
                    df = pd.read_csv(csv_file, header=None, names=["step", "value"])
                except Exception as e:
                    print(f"Warning: Could not read {csv_file}: {e}")
                    continue

                df["exp_name"] = run.name
                df["algorithm"] = run.algorithm
                df["task"] = run.task
                df["model"] = run.model
                if env_id is not None:
                    df["env_id"] = env_id

                self.data[metric_name].append(df)

        return self.data

    def resolve_metric_key(self, metric_name: str) -> Optional[str]:
        """Resolves metric name or any known alias to the key present in self.data."""
        if metric_name in self.data and len(self.data[metric_name]) > 0:
            return metric_name
        alias = MetricParser.METRIC_ALIASES.get(metric_name)
        if alias and alias in self.data and len(self.data[alias]) > 0:
            return alias
        rev = MetricParser.REVERSE_ALIASES.get(metric_name, [])
        for r in rev:
            if r in self.data and len(self.data[r]) > 0:
                return r
        return None

    def has_metric(self, metric_name: str) -> bool:
        """Checks if a metric has non-empty data in the dataset."""
        key = self.resolve_metric_key(metric_name)
        return key is not None and len(self.data[key]) > 0

    def get_metric_dfs(self, metric_name: str) -> List[pd.DataFrame]:
        """Returns list of DataFrames for given metric name or alias."""
        key = self.resolve_metric_key(metric_name)
        if key is not None:
            return self.data[key]
        return []

    def get_combined_df(self, metric_name: str) -> Optional[pd.DataFrame]:
        """Returns concatenated DataFrame for metric, or None if empty."""
        key = self.resolve_metric_key(metric_name)
        if key is None:
            return None
        if key in self._combined_cache:
            return self._combined_cache[key]
        dfs = self.data[key]
        if not dfs:
            return None
        combined = pd.concat(dfs, ignore_index=True)
        self._combined_cache[key] = combined
        return combined

    @property
    def available_metrics(self) -> List[str]:
        """Returns list of all metrics with non-empty data."""
        return sorted([k for k, v in self.data.items() if len(v) > 0])

    @property
    def algorithms(self) -> List[str]:
        """Returns list of unique algorithms in the dataset."""
        algos = set()
        for run in self.runs:
            algos.add(run.algorithm)
        return sorted(list(algos))

    @property
    def tasks(self) -> List[str]:
        """Returns list of unique tasks in the dataset."""
        tasks = set()
        for run in self.runs:
            tasks.add(run.task)
        return sorted(list(tasks))


class MarlEvalPlotter:
    """Encapsulates BenchMARL and marl-eval aggregate evaluation plotting."""

    def __init__(
        self,
        experiments_dir: Union[str, Path],
        plot_dir: Union[str, Path],
        env_name: str = "urbanmarl",
    ) -> None:
        self.experiments_dir = Path(experiments_dir).resolve()
        self.plot_dir = Path(plot_dir).resolve()
        self.env_name = env_name

    def evaluate(self) -> Tuple[plt.Figure, Dict[str, Any]]:
        """Generates performance profile, aggregate scores, and sample efficiency curves."""
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        raw_dict = get_raw_dict_from_multirun_folder(multirun_folder=self.experiments_dir)
        processed_data = Plotting.process_data(raw_dict)
        (
            environment_comparison_matrix,
            sample_efficiency_matrix,
        ) = Plotting.create_matrices(processed_data, env_name=self.env_name)

        # 1. Performance Profile
        performance_profile_figure = Plotting.performance_profile_figure(
            environment_comparison_matrix=environment_comparison_matrix
        )
        save_path = self.plot_dir / "performance_profile.pdf"
        performance_profile_figure.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

        # 2. Aggregate Scores
        aggregate_scores, _, _ = Plotting.aggregate_scores(
            environment_comparison_matrix=environment_comparison_matrix,
            save_tabular_as_latex=True,
        )
        save_path = self.plot_dir / "aggregate_scores.pdf"
        aggregate_scores.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

        # Move generated tabular files if present
        for src, dst in [
            (
                Path("aggregated_score_return.csv"),
                self.plot_dir / "aggregate_scores_return.csv",
            ),
            (
                Path("aggregated_score_return_latex.txt"),
                self.plot_dir / "aggregated_score_return_latex.tex",
            ),
        ]:
            if src.exists():
                shutil.move(src, dst)

        print(f"Tabular data saved to {self.plot_dir}")

        # 3. Environment Sample Efficiency Curves
        env_curves, _, _ = Plotting.environemnt_sample_efficiency_curves(
            sample_effeciency_matrix=sample_efficiency_matrix
        )
        save_path = self.plot_dir / "environemnt_sample_efficiency_curves.pdf"
        env_curves.figure.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

        # 4. Task Sample Efficiency Curves
        if self.env_name in processed_data:
            for task in processed_data[self.env_name].keys():
                task_curves = Plotting.task_sample_efficiency_curves(
                    processed_data=processed_data, env=self.env_name, task=task
                )
                save_path = self.plot_dir / f"{task}_sample_efficiency_curves.pdf"
                task_curves.figure.savefig(
                    save_path, bbox_inches="tight", pad_inches=0.1
                )

        return performance_profile_figure, processed_data


class UrbanVisualizer:
    """Visualization suite for scalar and urban environment metrics."""

    def __init__(
        self,
        dataset: EvaluationDataset,
        palette_manager: Optional[PaletteManager] = None,
        algo_colors: Optional[Dict[str, tuple]] = None,
        figsize: Tuple[int, int] = FIGURE_SIZE,
    ) -> None:
        self.dataset = dataset
        self.figsize = figsize
        self.palette = palette_manager or PaletteManager()
        if algo_colors:
            self.palette.colors.update(algo_colors)
        elif not self.palette.colors:
            self.palette.generate_palette(self.dataset.algorithms)

    @staticmethod
    def percentile_95(x: Any) -> float:
        """Computes 95th percentile for given array or Series."""
        return float(np.percentile(x, 95))

    def plot_all_metrics(
        self,
        dictionary: Optional[Dict[str, List[pd.DataFrame]]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        pdf: bool = False,
        fill: bool = True,
    ) -> None:
        """Plots every metric in dataset individually."""
        data_dict = dictionary if dictionary is not None else self.dataset.data
        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        for metric, dfs in data_dict.items():
            if not dfs:
                continue
            con_df = pd.concat(dfs, ignore_index=True)
            if con_df.empty:
                continue

            is_polar = "env_id" in con_df.columns
            if is_polar:
                fig, ax = plt.subplots(
                    subplot_kw=dict(projection="polar"), figsize=self.figsize
                )
                ax.set_xlim(-np.pi, np.pi)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(MetricParser.pi_formatter)
                )
                ax.set_rlabel_position(45)
                ax.yaxis.set_label_coords(0.5, 0.6)
            else:
                fig, ax = plt.subplots(figsize=self.figsize)

            for algo in con_df.algorithm.unique():
                _df = con_df.loc[con_df["algorithm"] == algo]
                if _df.empty:
                    continue
                max_limit = np.max(_df["value"].values)
                min_limit = np.min(_df["value"].values)
                algo_color = self.palette.get_color(algo)

                x_label = "env_id" if is_polar else "step"
                df = _df.groupby([x_label])[["value"]].agg(self.percentile_95)
                df.reset_index(inplace=True)
                df.columns = [x_label, "mean"]
                df["std"] = df["mean"].std()
                E = df[x_label]
                metric_mean = df["mean"]
                metric_std = df["std"]

                ax.plot(E, metric_mean, label=algo.upper(), color=algo_color)
                if fill:
                    lower_fill = np.clip(
                        metric_mean - metric_std, a_min=min_limit, a_max=None
                    )
                    upper_fill = np.clip(
                        metric_mean + metric_std, a_min=None, a_max=max_limit
                    )
                    ax.fill_between(
                        E, lower_fill, upper_fill, alpha=0.2, color=algo_color
                    )

            x_label, y_label, title = MetricParser.get_labels(metric)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)

            if out_path:
                ext = "pdf" if pdf else "png"
                save_file = out_path / f"{metric}.{ext}"
                plt.savefig(save_file, dpi=300, bbox_inches="tight")
                print(f"Saved plot: {save_file}")
            else:
                plt.show()
            plt.close()

    def plot_group_metrics(
        self,
        dictionary: Optional[Dict[str, List[pd.DataFrame]]] = None,
        algo_colors: Optional[Dict[str, tuple]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        metric_list: Sequence[str] = (),
        n_cols: int = 2,
        projection: str = "cartesian",
        file_name: str = "compare_metrics",
        figsize: Optional[Tuple[int, int]] = None,
        pdf: bool = False,
        fill: bool = True,
    ) -> Optional[Path]:
        """Plots a grid of multiple metrics across algorithms.

        Gracefully skips metrics that have no data without raising an error.
        """
        data_dict = dictionary if dictionary is not None else self.dataset.data
        if algo_colors:
            self.palette.colors.update(algo_colors)
        fig_sz = figsize or self.figsize
        out_path = Path(output_dir) if output_dir else None

        # Filter metric_list to available metrics with non-empty data
        resolved_metrics: List[Tuple[str, str]] = []
        for m in metric_list:
            if m in data_dict and len(data_dict[m]) > 0:
                resolved_metrics.append((m, m))
            else:
                alias = MetricParser.METRIC_ALIASES.get(m)
                if alias and alias in data_dict and len(data_dict[alias]) > 0:
                    resolved_metrics.append((m, alias))
                else:
                    rev = MetricParser.REVERSE_ALIASES.get(m, [])
                    found = False
                    for r in rev:
                        if r in data_dict and len(data_dict[r]) > 0:
                            resolved_metrics.append((m, r))
                            found = True
                            break
                    if not found:
                        print(
                            f"Notice: Metric '{m}' not found in dataset. Skipping in {file_name}."
                        )

        if not resolved_metrics:
            print(
                f"Notice: No data available for any metric in {list(metric_list)}. Skipping {file_name}."
            )
            return None

        n_metrics = len(resolved_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        is_polar = projection.lower() == "polar"

        subplot_kw = {"projection": "polar"} if is_polar else None
        h_extra = 1 if is_polar else 0
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            subplot_kw=subplot_kw,
            figsize=(fig_sz[0] * n_cols, fig_sz[1] * n_rows + h_extra),
            squeeze=False,
        )

        for idx, (display_metric, data_key) in enumerate(resolved_metrics):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            if is_polar:
                ax.set_xlim(-np.pi, np.pi)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(MetricParser.pi_formatter)
                )
                ax.set_rlabel_position(45)
                ax.yaxis.set_label_coords(0.5, 0.6)

            dfs = data_dict[data_key]
            con_df = pd.concat(dfs, ignore_index=True)

            for algo in sorted(list(con_df.algorithm.unique())):
                _df = con_df.loc[con_df["algorithm"] == algo]
                if _df.empty:
                    continue
                max_limit = np.max(_df["value"].values)
                min_limit = np.min(_df["value"].values)
                algo_color = self.palette.get_color(algo)

                if "env_id" in _df.columns:
                    x_label = "env_id"
                    df = _df.groupby([x_label])[["value"]].agg(self.percentile_95)
                    df.reset_index(inplace=True)
                    df.columns = [x_label, "mean"]
                    df["std"] = df["mean"].std()
                    E = df[x_label]
                    metric_mean = df["mean"]
                    metric_std = df["std"]

                    ax.plot(
                        E,
                        metric_mean,
                        label=algo.upper(),
                        linewidth=2.0,
                        color=algo_color,
                    )
                    if fill:
                        lower_fill = np.clip(
                            metric_mean - metric_std, a_min=min_limit, a_max=None
                        )
                        upper_fill = np.clip(
                            metric_mean + metric_std, a_min=None, a_max=max_limit
                        )
                        ax.fill_between(
                            E, lower_fill, upper_fill, alpha=0.2, color=algo_color
                        )
                else:
                    x_label = "step"
                    df = _df.groupby([x_label])[["value"]].agg(self.percentile_95)
                    df.reset_index(inplace=True)
                    df.columns = [x_label, "mean"]
                    df["std"] = df["mean"].std()
                    E = df[x_label]
                    metric_mean = df["mean"]
                    metric_std = df["std"]

                    ax.plot(
                        E,
                        metric_mean,
                        label=algo.upper(),
                        linewidth=2.0,
                        color=algo_color,
                    )
                    if fill:
                        lower_fill = np.clip(
                            metric_mean - metric_std, a_min=min_limit, a_max=None
                        )
                        upper_fill = np.clip(
                            metric_mean + metric_std, a_min=None, a_max=max_limit
                        )
                        ax.fill_between(
                            E, lower_fill, upper_fill, alpha=0.2, color=algo_color
                        )

            x_label, y_label, title = MetricParser.get_labels(display_metric)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide any unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].set_visible(False)

        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)
            ext = "pdf" if pdf else "png"
            save_path = out_path / f"{file_name}.{ext}"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None

    def plot_group_tasks(
        self,
        metric: str,
        dictionary: Optional[Dict[str, List[pd.DataFrame]]] = None,
        algo_colors: Optional[Dict[str, tuple]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        n_cols: int = 3,
        projection: str = "cartesian",
        file_name: str = "compare_metrics",
        figsize: Optional[Tuple[int, int]] = None,
        pdf: bool = False,
        fill: bool = True,
    ) -> Optional[Path]:
        """Plots a metric separated into a grid across tasks."""
        data_dict = dictionary if dictionary is not None else self.dataset.data
        if algo_colors:
            self.palette.colors.update(algo_colors)
        fig_sz = figsize or self.figsize
        out_path = Path(output_dir) if output_dir else None

        data_key = None
        if metric in data_dict and len(data_dict[metric]) > 0:
            data_key = metric
        else:
            alias = MetricParser.METRIC_ALIASES.get(metric)
            if alias and alias in data_dict and len(data_dict[alias]) > 0:
                data_key = alias
            else:
                rev = MetricParser.REVERSE_ALIASES.get(metric, [])
                for r in rev:
                    if r in data_dict and len(data_dict[r]) > 0:
                        data_key = r
                        break

        if not data_key:
            print(
                f"Notice: Metric '{metric}' not found in dataset. Skipping task group plot {file_name}."
            )
            return None

        con_df = pd.concat(data_dict[data_key], ignore_index=True)
        if con_df.empty or "task" not in con_df.columns:
            print(
                f"Notice: Metric '{metric}' has no task column or empty data. Skipping."
            )
            return None

        unique_tasks = sorted(list(con_df.task.unique()))
        n_tasks = len(unique_tasks)
        if n_tasks < 4:
            n_cols = max(n_tasks, 1)

        n_rows = (n_tasks + n_cols - 1) // n_cols
        is_polar = projection.lower() == "polar"
        subplot_kw = {"projection": "polar"} if is_polar else None
        h_extra = 1 if is_polar else 0

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            subplot_kw=subplot_kw,
            figsize=(fig_sz[0] * n_cols, fig_sz[1] * n_rows + h_extra),
            squeeze=False,
        )

        for idx, task in enumerate(unique_tasks):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            if is_polar:
                ax.set_xlim(-np.pi, np.pi)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(MetricParser.pi_formatter)
                )
                ax.set_rlabel_position(45)
                ax.yaxis.set_label_coords(0.5, 0.6)

            _con_df = con_df.loc[con_df["task"] == task]

            for algo in sorted(list(_con_df.algorithm.unique())):
                _df = _con_df.loc[_con_df["algorithm"] == algo]
                if _df.empty:
                    continue
                max_limit = np.max(_df["value"].values)
                min_limit = np.min(_df["value"].values)
                algo_color = self.palette.get_color(algo)

                if "env_id" in _df.columns:
                    x_label = "env_id"
                    df = _df.groupby([x_label])[["value"]].agg(self.percentile_95)
                    df.reset_index(inplace=True)
                    df.columns = [x_label, "mean"]
                    df["std"] = df["mean"].std()
                    E = df[x_label]
                    metric_mean = df["mean"]
                    metric_std = df["std"]

                    ax.plot(
                        E,
                        metric_mean,
                        label=algo.upper(),
                        linewidth=2.0,
                        color=algo_color,
                    )
                    if fill:
                        lower_fill = np.clip(
                            metric_mean - metric_std, a_min=min_limit, a_max=None
                        )
                        upper_fill = np.clip(
                            metric_mean + metric_std, a_min=None, a_max=max_limit
                        )
                        ax.fill_between(
                            E, lower_fill, upper_fill, alpha=0.2, color=algo_color
                        )
                else:
                    x_label = "step"
                    df = _df.groupby([x_label])[["value"]].agg(self.percentile_95)
                    df.reset_index(inplace=True)
                    df.columns = [x_label, "mean"]
                    df["std"] = df["mean"].std()
                    E = df[x_label]
                    metric_mean = df["mean"]
                    metric_std = df["std"]

                    ax.plot(
                        E,
                        metric_mean,
                        label=algo.upper(),
                        linewidth=2.0,
                        color=algo_color,
                    )
                    if fill:
                        lower_fill = np.clip(
                            metric_mean - metric_std, a_min=min_limit, a_max=None
                        )
                        upper_fill = np.clip(
                            metric_mean + metric_std, a_min=None, a_max=max_limit
                        )
                        ax.fill_between(
                            E, lower_fill, upper_fill, alpha=0.2, color=algo_color
                        )

            x_label, y_label, title = MetricParser.get_labels(metric)
            title = f"{title} - {task.upper()}"
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide any unused subplots
        for idx in range(n_tasks, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].set_visible(False)

        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)
            ext = "pdf" if pdf else "png"
            save_path = out_path / f"{file_name}.{ext}"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None

    def plot_catalogue(
        self,
        output_dir: Union[str, Path],
        figsize: Optional[Tuple[int, int]] = None,
        pdf: bool = False,
        fill: bool = True,
    ) -> None:
        """Generates comprehensive catalogue of all scalar metrics."""
        catalog_dir = Path(output_dir) / "catalogue"
        catalog_dir.mkdir(parents=True, exist_ok=True)

        self.plot_all_metrics(output_dir=catalog_dir, pdf=pdf, fill=fill)

        time_metrics = [
            "timers_collection_time",
            "timers_training_time",
            "timers_evaluation_time",
            "timers_iteration_time",
        ]
        self.plot_group_metrics(
            metric_list=time_metrics,
            output_dir=catalog_dir,
            n_cols=2,
            projection="cartesian",
            file_name="grouped_timers",
            figsize=figsize,
            pdf=pdf,
            fill=fill,
        )

        info_metrics = [
            "los",
            "collisions",
            "velocity",
            "reward",
        ]
        self.plot_group_metrics(
            metric_list=info_metrics,
            output_dir=catalog_dir,
            n_cols=2,
            projection="polar",
            file_name="grouped_info_metrics",
            figsize=figsize,
            pdf=pdf,
            fill=fill,
        )

    def plot_training_metrics(
        self,
        output_dir: Union[str, Path],
        figsize: Optional[Tuple[int, int]] = None,
        n_cols: int = 4,
        pdf: bool = True,
        fill: bool = False,
    ) -> None:
        """Plots training loss and gradient metrics grid."""
        training_metrics = [
            "train_uav_ESS",
            "train_uav_alpha",
            "train_uav_clip_fraction",
            "train_uav_entropy",
            "train_uav_explained_variance",
            "train_uav_grad_norm_loss_actor",
            "train_uav_grad_norm_loss_alpha",
            "train_uav_grad_norm_loss_critic",
            "train_uav_grad_norm_loss_objective",
            "train_uav_grad_norm_loss_qvalue",
            "train_uav_grad_norm_loss_value",
            "train_uav_kl_approx",
            "train_uav_loss_actor",
            "train_uav_loss_alpha",
            "train_uav_loss_critic",
            "train_uav_loss_entropy",
            "train_uav_loss_objective",
            "train_uav_loss_qvalue",
            "train_uav_loss_value",
            "train_uav_pred_value",
            "train_uav_pred_value_max",
            "train_uav_target_value",
            "train_uav_target_value_max",
            "train_uav_td_error",
        ]
        self.plot_group_metrics(
            metric_list=training_metrics,
            output_dir=output_dir,
            n_cols=n_cols,
            projection="cartesian",
            file_name="grouped_training_metrics",
            figsize=figsize,
            pdf=pdf,
            fill=fill,
        )

    def plot_report(
        self,
        output_dir: Union[str, Path],
        figsize: Optional[Tuple[int, int]] = None,
        pdf: bool = True,
        fill: bool = True,
    ) -> None:
        """Generates publication report plots."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        info_metrics = ["los", "collisions", "velocity", "reward"]
        new_dict = {
            m: self.dataset.get_metric_dfs(m)
            for m in info_metrics
            if self.dataset.has_metric(m)
        }
        self.plot_all_metrics(
            dictionary=new_dict, output_dir=out_path, pdf=pdf, fill=fill
        )

        for metric in info_metrics:
            if self.dataset.has_metric(metric):
                self.plot_group_tasks(
                    metric=metric,
                    output_dir=out_path,
                    n_cols=3,
                    projection="polar",
                    file_name=f"grouped_{metric}_per_task",
                    figsize=figsize,
                    pdf=pdf,
                    fill=fill,
                )

        time_metrics = [
            "timers_collection_time",
            "timers_training_time",
            "timers_evaluation_time",
            "timers_iteration_time",
        ]
        self.plot_group_metrics(
            metric_list=time_metrics,
            output_dir=out_path,
            n_cols=2,
            projection="cartesian",
            file_name="grouped_timers",
            figsize=figsize,
            pdf=pdf,
            fill=fill,
        )

        for metric in time_metrics:
            if self.dataset.has_metric(metric):
                self.plot_group_tasks(
                    metric=metric,
                    output_dir=out_path,
                    n_cols=3,
                    projection="cartesian",
                    file_name=f"grouped_{metric}_per_task",
                    figsize=figsize,
                    pdf=pdf,
                    fill=fill,
                )

        self.plot_group_metrics(
            metric_list=info_metrics,
            output_dir=out_path,
            n_cols=2,
            projection="polar",
            file_name="grouped_info_metrics",
            figsize=figsize,
            pdf=pdf,
            fill=fill,
        )


class EvaluationPipeline:
    """Orchestrates end-to-end evaluation, data loading, and visualization."""

    def __init__(
        self,
        exp_dir: str = "experiments",
        output_dir: str = "plots",
        metrics_of_interest: Optional[List[str]] = None,
    ) -> None:
        self.exp_dir_name = exp_dir
        self.output_dir_name = output_dir
        self.metrics_of_interest = metrics_of_interest

        self.experiments_dir = project_root / "outputs" / self.exp_dir_name
        self.plot_dir = project_root / "outputs" / self.output_dir_name / self.exp_dir_name

    def run(self) -> None:
        """Executes the full evaluation and visualization pipeline."""
        if not self.experiments_dir.exists():
            print(f"Error: Experiment directory {self.experiments_dir} does not exist.")
            return

        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. marl-eval benchmarks
        marl_eval = MarlEvalPlotter(self.experiments_dir, self.plot_dir)
        perf_figure, _ = marl_eval.evaluate()

        # 2. Extract or generate color palette
        palette_manager = PaletteManager()
        palette_manager.extract_from_figure(perf_figure)

        # 3. Discover and load scalar metrics
        dataset = EvaluationDataset.from_directory(self.experiments_dir)
        print(f"Found {len(dataset.runs)} experiment directories.")
        dataset.load_all_metrics(self.metrics_of_interest)

        # 4. Generate visual catalogue and reports
        visualizer = UrbanVisualizer(
            dataset=dataset,
            palette_manager=palette_manager,
            figsize=FIGURE_SIZE,
        )

        visualizer.plot_catalogue(output_dir=self.plot_dir, pdf=False)
        visualizer.plot_training_metrics(
            output_dir=self.plot_dir, n_cols=4, pdf=True, fill=False
        )
        visualizer.plot_report(output_dir=self.plot_dir, pdf=True, fill=True)
        print("Evaluation pipeline completed successfully.")


# ----------------------------------------------------------------------
#  Backwards Compatibility Module-Level Functions
# ----------------------------------------------------------------------
pi_formatter = MetricParser.pi_formatter


def parse_experiment_folder_name(folder_path: Union[str, Path]) -> Dict[str, Any]:
    """Parses experiment folder name into metadata dict."""
    folder_name = Path(folder_path).name
    return ExperimentRun._parse_folder_name(folder_name)


def find_experiments(root: Path) -> List[ExperimentResult]:
    """Discovers all experiment output directories within root path."""
    dataset = EvaluationDataset.from_directory(root)
    return dataset.runs


def find_experiment_dirs(root_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Locate all experiment directories under root_dir returning dict list."""
    dataset = EvaluationDataset.from_directory(root_dir)
    return [run.to_dict() for run in dataset.runs]


def get_scalars_dir(exp_info: Union[Dict[str, Any], ExperimentRun]) -> Optional[str]:
    """Returns the path to the 'scalars' folder as a string."""
    if isinstance(exp_info, ExperimentRun):
        return str(exp_info.scalars_dir) if exp_info.scalars_dir else None
    run = ExperimentRun(exp_info["path"])
    return str(run.scalars_dir) if run.scalars_dir else None


def parse_metric_from_filename(filename: str) -> Tuple[str, Optional[float]]:
    """Determine metric name and optional environment id from a CSV filename."""
    return MetricParser.parse_filename(filename)


def load_csv_metric(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a CSV file with no header, two columns: step and value."""
    try:
        df = pd.read_csv(file_path, header=None, names=["step", "value"])
        return df
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return None


def load_all_metrics(
    experiments: Sequence[Union[ExperimentRun, Dict[str, Any], Path, str]],
    metrics_of_interest: Optional[Sequence[str]] = None,
) -> Dict[str, List[pd.DataFrame]]:
    """Loads all CSV files from all experiments' scalars directories."""
    dataset = EvaluationDataset(experiments)
    return dataset.load_all_metrics(metrics_of_interest)


def load_metric_over_seeds(
    experiments: List[ExperimentResult], pattern: str, metric_col: str
) -> pd.DataFrame:
    """Load a specific metric from all experiments."""
    data_frames = []
    for exp in experiments:
        df = exp.get_csv_data(pattern)
        if df is not None and not df.empty:
            value_col = None
            for col in df.columns:
                if col.lower() == metric_col.lower() or col.lower() == "value":
                    value_col = col
                    break
            if value_col is None:
                value_col = df.columns[-1]
            step_col = "Step" if "Step" in df.columns else df.columns[0]
            df_exp = df[[step_col, value_col]].copy()
            df_exp.columns = ["step", "value"]
            df_exp["algorithm"] = exp.algorithm
            df_exp["seed"] = exp.path.name
            data_frames.append(df_exp)
    if not data_frames:
        return pd.DataFrame()
    return pd.concat(data_frames, ignore_index=True)


def aggregate_by_algorithm(
    df: pd.DataFrame, group_cols: Sequence[str] = ("step", "algorithm")
) -> pd.DataFrame:
    """Compute mean and std across seeds for each algorithm."""
    if df.empty:
        return df
    return (
        df.groupby(list(group_cols))
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            sem=("value", sem),
            n=("value", "count"),
        )
        .reset_index()
    )


def aggregate_runs(metric_dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Aggregate a list of DataFrames by step using linear interpolation."""
    if not metric_dfs:
        return None
    all_steps = set()
    for df in metric_dfs:
        all_steps.update(df["step"].values)
    steps_sorted = sorted(all_steps)
    interpolated = []
    for df in metric_dfs:
        df_sorted = df.sort_values("step")
        interp_vals = np.interp(steps_sorted, df_sorted["step"], df_sorted["value"])
        interpolated.append(interp_vals)
    interp_array = np.array(interpolated)
    mean_vals = np.mean(interp_array, axis=0)
    std_vals = np.std(interp_array, axis=0)
    return pd.DataFrame({"step": steps_sorted, "mean": mean_vals, "std": std_vals})


def aggregate_runs_by_envs(metric_dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Aggregate a list of DataFrames by env angle using linear interpolation."""
    if not metric_dfs:
        return None
    env_sorted = np.linspace(-np.pi, np.pi, 100)
    interpolated = []
    for df in metric_dfs:
        df_sorted = df.sort_values("env_id")
        interp_vals = np.interp(env_sorted, df_sorted["env_id"], df_sorted["value"])
        interpolated.append(interp_vals)
    interp_array = np.array(interpolated)
    mean_vals = np.mean(interp_array, axis=0)
    std_vals = np.std(interp_array, axis=0)
    return pd.DataFrame({"E": env_sorted, "mean": mean_vals, "std": std_vals})


def plot_metric(
    metric_name: Any,
    aggregated_dict: Any = None,
    output_dir: Any = None,
    algo_names: Any = None,
    pdf: bool = False,
    agg_df: Any = None,
    title: Any = None,
    save_path: Any = None,
) -> None:
    """Plot learning curves for each algorithm (compatible with both legacy signatures)."""
    # Signature 1: plot_metric(agg_df, metric_name, title, save_path)
    if isinstance(metric_name, pd.DataFrame) or agg_df is not None:
        target_df = metric_name if isinstance(metric_name, pd.DataFrame) else agg_df
        target_metric = aggregated_dict if isinstance(metric_name, pd.DataFrame) else metric_name
        target_title = output_dir if isinstance(metric_name, pd.DataFrame) else title
        target_save = algo_names if isinstance(metric_name, pd.DataFrame) else save_path

        if target_df.empty:
            print(f"No data for {target_metric}")
            return
        plt.figure(figsize=FIGURE_SIZE)
        for algo in target_df["algorithm"].unique():
            algo_data = target_df[target_df["algorithm"] == algo]
            plt.plot(algo_data["step"], algo_data["mean"], label=algo)
            if "sem" in algo_data.columns:
                err = algo_data["sem"]
            elif "std" in algo_data.columns:
                err = algo_data["std"]
            else:
                err = 0
            plt.fill_between(
                algo_data["step"],
                algo_data["mean"] - err,
                algo_data["mean"] + err,
                alpha=0.2,
            )
        plt.xlabel("Step")
        plt.ylabel(str(target_metric))
        plt.title(str(target_title))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if target_save:
            Path(target_save).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(target_save)
        plt.close()
        return

    # Signature 2: plot_metric(metric_name, aggregated_dict, output_dir, algo_names=None, pdf=False)
    if not aggregated_dict:
        print(f"No data to plot for metric: {metric_name}")
        return
    plt.figure(figsize=FIGURE_SIZE)
    for algo, df in aggregated_dict.items():
        if df is None:
            continue
        plt.plot(df["step"], df["mean"], label=algo)
        plt.fill_between(
            df["step"], df["mean"] - df["std"], df["mean"] + df["std"], alpha=0.2
        )
    plt.xlabel("Step")
    plt.ylabel(str(metric_name))
    plt.title(f"Learning curve - {metric_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ext = "pdf" if pdf else "png"
        save_file = os.path.join(output_dir, f"{metric_name}.{ext}")
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_file}")
    else:
        plt.show()
    plt.close()


def plot_metric_by_envs(
    metric_name: str,
    aggregated_dict: Dict[str, pd.DataFrame],
    output_dir: Optional[Union[str, Path]],
    algo_names: Any = None,
    pdf: bool = False,
) -> None:
    """Plot aggregated metric for each algorithm on polar axes."""
    if not aggregated_dict:
        print(f"No data to plot for metric: {metric_name}")
        return
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, subplot_kw={"projection": "polar"})
    for algo, df in aggregated_dict.items():
        if df is None:
            continue
        ax.plot(df["E"], df["mean"], label=algo.upper())
        ax.fill_between(
            df["E"], df["mean"] - df["std"], df["mean"] + df["std"], alpha=0.2
        )
    ax.set_xlabel("E")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ext = "pdf" if pdf else "png"
        save_path = os.path.join(output_dir, f"{metric_name}.{ext}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()


def calculate_percentile(group: Any, percentile: float) -> float:
    """Calculates percentile for a group."""
    return float(np.percentile(group, percentile))


percentile_95 = UrbanVisualizer.percentile_95
metric_labels = MetricParser.get_labels


def get_algo_colors(dictionary: Dict[str, List[pd.DataFrame]]) -> Dict[str, tuple]:
    """Generates consistent colorblind palette for all algorithms in dataset."""
    unique_algos = set()
    for metric, dfs in dictionary.items():
        for df in dfs:
            unique_algos.update(df["algorithm"].unique())
    palette = PaletteManager()
    return palette.generate_palette(list(unique_algos))


def extract_marl_eval_colors(fig: plt.Figure) -> Dict[str, tuple]:
    """Extracts algorithm color mapping from a marl-eval figure."""
    return PaletteManager().extract_from_figure(fig)


def plot_all_metrics(
    dictionary: Dict[str, List[pd.DataFrame]],
    output_dir: Union[str, Path],
    algo_colors: Dict[str, tuple],
    pdf: bool = False,
    fill: bool = True,
) -> None:
    """Wrapper calling UrbanVisualizer.plot_all_metrics."""
    dataset = EvaluationDataset([])
    dataset.data = defaultdict(list, dictionary)
    visualizer = UrbanVisualizer(dataset=dataset, algo_colors=algo_colors)
    visualizer.plot_all_metrics(
        dictionary=dictionary, output_dir=output_dir, pdf=pdf, fill=fill
    )


def plot_group_metrics(
    dictionary: Dict[str, List[pd.DataFrame]],
    algo_colors: Dict[str, tuple],
    output_dir: Optional[Union[str, Path]] = None,
    metric_list: Sequence[str] = (),
    n_cols: int = 2,
    projection: str = "cartesian",
    file_name: str = "compare_metrics",
    figsize: Tuple[int, int] = FIGURE_SIZE,
    pdf: bool = False,
    fill: bool = True,
) -> Optional[Path]:
    """Wrapper calling UrbanVisualizer.plot_group_metrics."""
    dataset = EvaluationDataset([])
    dataset.data = defaultdict(list, dictionary)
    visualizer = UrbanVisualizer(
        dataset=dataset, algo_colors=algo_colors, figsize=figsize
    )
    return visualizer.plot_group_metrics(
        dictionary=dictionary,
        algo_colors=algo_colors,
        output_dir=output_dir,
        metric_list=metric_list,
        n_cols=n_cols,
        projection=projection,
        file_name=file_name,
        figsize=figsize,
        pdf=pdf,
        fill=fill,
    )


def plot_group_tasks(
    dictionary: Dict[str, List[pd.DataFrame]],
    metric: str,
    algo_colors: Dict[str, tuple],
    output_dir: Optional[Union[str, Path]] = None,
    n_cols: int = 3,
    projection: str = "cartesian",
    file_name: str = "compare_metrics",
    figsize: Tuple[int, int] = FIGURE_SIZE,
    pdf: bool = False,
    fill: bool = True,
) -> Optional[Path]:
    """Wrapper calling UrbanVisualizer.plot_group_tasks."""
    dataset = EvaluationDataset([])
    dataset.data = defaultdict(list, dictionary)
    visualizer = UrbanVisualizer(
        dataset=dataset, algo_colors=algo_colors, figsize=figsize
    )
    return visualizer.plot_group_tasks(
        metric=metric,
        dictionary=dictionary,
        algo_colors=algo_colors,
        output_dir=output_dir,
        n_cols=n_cols,
        projection=projection,
        file_name=file_name,
        figsize=figsize,
        pdf=pdf,
        fill=fill,
    )


def plot_catalogue(
    dictionary: Dict[str, List[pd.DataFrame]],
    output_dir: Union[str, Path],
    algo_colors: Dict[str, tuple],
    figsize: Tuple[int, int] = FIGURE_SIZE,
    pdf: bool = False,
    fill: bool = True,
) -> None:
    """Wrapper calling UrbanVisualizer.plot_catalogue."""
    dataset = EvaluationDataset([])
    dataset.data = defaultdict(list, dictionary)
    visualizer = UrbanVisualizer(
        dataset=dataset, algo_colors=algo_colors, figsize=figsize
    )
    visualizer.plot_catalogue(
        output_dir=output_dir, figsize=figsize, pdf=pdf, fill=fill
    )


def plot_training_metrics(
    dictionary: Dict[str, List[pd.DataFrame]],
    output_dir: Union[str, Path],
    algo_colors: Dict[str, tuple],
    figsize: Tuple[int, int] = FIGURE_SIZE,
    n_cols: int = 4,
    pdf: bool = True,
    fill: bool = False,
) -> None:
    """Wrapper calling UrbanVisualizer.plot_training_metrics."""
    dataset = EvaluationDataset([])
    dataset.data = defaultdict(list, dictionary)
    visualizer = UrbanVisualizer(
        dataset=dataset, algo_colors=algo_colors, figsize=figsize
    )
    visualizer.plot_training_metrics(
        output_dir=output_dir, figsize=figsize, n_cols=n_cols, pdf=pdf, fill=fill
    )


def plot_report(
    dictionary: Dict[str, List[pd.DataFrame]],
    output_dir: Union[str, Path],
    algo_colors: Dict[str, tuple],
    figsize: Tuple[int, int] = FIGURE_SIZE,
    pdf: bool = True,
    fill: bool = True,
) -> None:
    """Wrapper calling UrbanVisualizer.plot_report."""
    dataset = EvaluationDataset([])
    dataset.data = defaultdict(list, dictionary)
    visualizer = UrbanVisualizer(
        dataset=dataset, algo_colors=algo_colors, figsize=figsize
    )
    visualizer.plot_report(
        output_dir=output_dir, figsize=figsize, pdf=pdf, fill=fill
    )


def main() -> None:
    """CLI entrypoint for evaluation and plotting."""
    parser = argparse.ArgumentParser(
        description="Aggregate and plot BenchMARL metrics."
    )
    parser.add_argument(
        "--exp_dir",
        "-e",
        default="experiments",
        help="Root directory containing experiment folders (e.g., outputs/experiments)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="plots",
        help="Directory to save plots (default: ./plots)",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="*",
        help="List of specific metrics to plot (e.g., collection_agents_reward_episode_reward_mean). If not given, plot a default set.",
    )
    args = parser.parse_args()

    pipeline = EvaluationPipeline(
        exp_dir=args.exp_dir,
        output_dir=args.output,
        metrics_of_interest=args.metrics,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
