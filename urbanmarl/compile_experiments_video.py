#!/usr/bin/env python3
"""UrbanMARL Experiment Video Compiler.

Concatenates recorded MP4 trajectory videos across experiment runs into
unified compilation videos grouped and named by scenario (task), algorithm,
model, or combinations thereof.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure project root is in sys.path so the module can run from any working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from moviepy.editor import concatenate_videoclips, VideoFileClip
except ImportError:
    from moviepy import concatenate_videoclips, VideoFileClip


def natural_sort_key(filepath: Union[str, Path]) -> list:
    """Splits a file path into strings and integers for natural sorting.

    Ensures numerical filenames are sorted in ascending order (e.g. 'video_2.mp4' before 'video_10.mp4').

    Args:
        filepath: File path object or string.

    Returns:
        list: List of alternating string and integer tokens.
    """
    path_str = Path(filepath).name if isinstance(filepath, Path) else str(filepath)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path_str)
    ]


@dataclass
class VideoClipInfo:
    """Container holding metadata and path for a discovered MP4 video clip.

    Attributes:
        path (Path): Absolute file path to the MP4 file.
        task (str): Task / scenario name (e.g. 'uav_mobile_ue', 'coverage').
        algorithm (str): Algorithm name (e.g. 'maddpg', 'mappo', 'iddpg').
        model (str): Policy/critic model architecture (e.g. 'mlp').
        hash (str): Experiment hash identifier.
        timestamp_raw (str): Timestamp string from the run folder.
        run_name (str): Experiment directory name.
    """

    path: Path
    task: str = "unknown"
    algorithm: str = "unknown"
    model: str = "unknown"
    hash: str = "unknown"
    timestamp_raw: str = ""
    run_name: str = ""


class ExperimentMetadataParser:
    """Extracts scenario, algorithm, and model metadata from paths and folder names."""

    @staticmethod
    def parse_folder_name(folder_str: str) -> Dict[str, str]:
        """Parses algorithm, task, model, hash, and timestamp from BenchMARL folder name.

        Expected folder naming: {algorithm}_{task}_{model}__{hash}_{timestamp}
        """
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
        else:
            tokens = folder_str.split("_")
            algorithm = tokens[0] if tokens else "unknown"
            task = tokens[1] if len(tokens) > 1 else "unknown"
            model = tokens[2] if len(tokens) > 2 else "unknown"
            exp_hash = "unknown"
            timestamp_str = ""

        return {
            "algorithm": algorithm.lower(),
            "task": task.lower(),
            "model": model.lower(),
            "hash": exp_hash,
            "timestamp_raw": timestamp_str,
        }

    @classmethod
    def extract_metadata(
        cls,
        video_path: Path,
        default_task: Optional[str] = None,
        default_algorithm: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> VideoClipInfo:
        """Inspects video path ancestry to extract experiment metadata."""
        resolved = video_path.resolve()
        curr = resolved.parent

        metadata: Optional[Dict[str, str]] = None
        run_name = ""

        # Walk upwards looking for a folder matching BenchMARL pattern
        while curr != curr.parent:
            if "__" in curr.name:
                metadata = cls.parse_folder_name(curr.name)
                run_name = curr.name
                break
            curr = curr.parent

        if metadata is None:
            metadata = cls.parse_folder_name(video_path.parent.name)
            run_name = video_path.parent.name

        task = metadata.get("task", "unknown")
        algorithm = metadata.get("algorithm", "unknown")
        model = metadata.get("model", "unknown")
        exp_hash = metadata.get("hash", "unknown")
        timestamp_raw = metadata.get("timestamp_raw", "")

        # Apply fallback overrides if fields are unknown
        if task == "unknown" and default_task:
            task = default_task.lower()
        if algorithm == "unknown" and default_algorithm:
            algorithm = default_algorithm.lower()
        if model == "unknown" and default_model:
            model = default_model.lower()

        return VideoClipInfo(
            path=resolved,
            task=task,
            algorithm=algorithm,
            model=model,
            hash=exp_hash,
            timestamp_raw=timestamp_raw,
            run_name=run_name,
        )


class ExperimentVideoScanner:
    """Discovers, filters, and groups MP4 video clips from experiment directories."""

    def __init__(
        self,
        source_dir: Union[str, Path],
        task_filter: Optional[str] = None,
        algorithm_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
    ) -> None:
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.task_filter = task_filter.lower() if task_filter else None
        self.algorithm_filter = algorithm_filter.lower() if algorithm_filter else None
        self.model_filter = model_filter.lower() if model_filter else None

    def scan(self) -> List[VideoClipInfo]:
        """Finds all MP4 files and extracts metadata."""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")

        video_files = list(self.source_dir.rglob("*.mp4"))
        clips = []
        for vf in video_files:
            clip = ExperimentMetadataParser.extract_metadata(
                vf,
                default_task=self.task_filter,
                default_algorithm=self.algorithm_filter,
                default_model=self.model_filter,
            )
            # Apply filters
            if self.task_filter and clip.task != self.task_filter:
                continue
            if self.algorithm_filter and clip.algorithm != self.algorithm_filter:
                continue
            if self.model_filter and clip.model != self.model_filter:
                continue
            clips.append(clip)

        return clips

    def group_clips(
        self,
        clips: List[VideoClipInfo],
        group_by: str = "both",
    ) -> Dict[str, List[VideoClipInfo]]:
        """Groups clips according to specified grouping strategy.

        Supported group_by options:
            - 'both' / 'combo' / 'all': Groups by (task, algorithm, model).
              Output name: '{task}_{algorithm}_{model}' (e.g. 'uav_mobile_ue_maddpg_mlp').
            - 'task' / 'scenario': Groups by task.
              Output name: '{task}' (e.g. 'uav_mobile_ue').
            - 'algorithm' / 'algo': Groups by algorithm.
              Output name: '{algorithm}' (e.g. 'maddpg').
            - 'model': Groups by model architecture.
              Output name: '{model}' (e.g. 'mlp').
            - 'task_algorithm' / 'task_algo': Groups by (task, algorithm).
              Output name: '{task}_{algorithm}' (e.g. 'uav_mobile_ue_maddpg').
            - 'none' / 'single': All clips into one group.
              Output name: 'compiled_master'.
        """
        mode = group_by.lower()
        grouped: Dict[str, List[VideoClipInfo]] = defaultdict(list)

        for clip in clips:
            if mode in ("both", "combo", "all", "task_algorithm_model"):
                key = f"{clip.task}_{clip.algorithm}_{clip.model}"
            elif mode in ("task", "scenario"):
                key = f"{clip.task}"
            elif mode in ("algorithm", "algo"):
                key = f"{clip.algorithm}"
            elif mode == "model":
                key = f"{clip.model}"
            elif mode in ("task_algorithm", "task_algo"):
                key = f"{clip.task}_{clip.algorithm}"
            elif mode in ("none", "single"):
                key = "compiled_master"
            else:
                raise ValueError(
                    f"Unknown group_by mode '{group_by}'. Expected one of: "
                    "both, task, algorithm, model, task_algorithm, none."
                )
            grouped[key].append(clip)

        # Sort clips within each group chronologically
        for key, group_items in grouped.items():
            group_items.sort(
                key=lambda c: (
                    c.timestamp_raw,
                    c.run_name,
                    natural_sort_key(c.path.name),
                )
            )

        return dict(grouped)


class VideoCompiler:
    """Concatenates video clips into standardized MP4 videos using MoviePy."""

    def __init__(
        self,
        fps: int = 20,
        codec: str = "libx264",
        method: str = "compose",
        verbose: bool = True,
    ) -> None:
        self.fps = fps
        self.codec = codec
        self.method = method
        self.verbose = verbose

    def compile(
        self,
        video_paths: Sequence[Union[str, Path]],
        output_path: Union[str, Path],
    ) -> Path:
        """Concatenates video files and writes to output_path."""
        paths = [Path(p) for p in video_paths]
        if not paths:
            raise ValueError("No video paths provided for compilation.")

        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Loading {len(paths)} clips for {out.name}...")

        clips = []
        try:
            for p in paths:
                clips.append(VideoFileClip(str(p)))

            if self.verbose:
                print(f"Concatenating clips (method={self.method})...")
            final_clip = concatenate_videoclips(clips, method=self.method)

            has_audio = final_clip.audio is not None
            write_kwargs: Dict[str, Any] = {
                "codec": self.codec,
                "fps": self.fps,
                "logger": "bar" if self.verbose else None,
            }
            if has_audio:
                write_kwargs["audio_codec"] = "aac"
            else:
                write_kwargs["audio"] = False

            if self.verbose:
                print(f"Writing compiled video to: {out}...")
            final_clip.write_videofile(str(out), **write_kwargs)
            final_clip.close()
        finally:
            for clip in clips:
                try:
                    clip.close()
                except Exception:
                    pass

        if self.verbose:
            size_mb = out.stat().st_size / (1024 * 1024)
            print(f"Successfully generated: {out} ({size_mb:.2f} MB)")
        return out


class ExperimentVideoManager:
    """End-to-end experiment video compilation manager."""

    def __init__(
        self,
        source_dir: Union[str, Path],
        output_dir: Union[str, Path],
        group_by: str = "both",
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        model: Optional[str] = None,
        fps: int = 20,
        codec: str = "libx264",
        verbose: bool = True,
    ) -> None:
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.output_target = Path(output_dir).expanduser().resolve()
        self.group_by = group_by
        self.task = task
        self.algorithm = algorithm
        self.model = model
        self.fps = fps
        self.codec = codec
        self.verbose = verbose

        self.scanner = ExperimentVideoScanner(
            source_dir=self.source_dir,
            task_filter=self.task,
            algorithm_filter=self.algorithm,
            model_filter=self.model,
        )
        self.compiler = VideoCompiler(
            fps=self.fps,
            codec=self.codec,
            verbose=self.verbose,
        )

    def run(self, dry_run: bool = False) -> List[Path]:
        """Discovers clips, groups them, and compiles videos."""
        clips = self.scanner.scan()
        if not clips:
            print(f"No matching MP4 files found in {self.source_dir}.")
            return []

        grouped = self.scanner.group_clips(clips, group_by=self.group_by)
        print(f"Found {len(clips)} video files across {len(grouped)} group(s):")
        for group_name, group_clips in sorted(grouped.items()):
            print(f"  [{group_name}] ({len(group_clips)} clips)")

        if dry_run:
            print("\n[Dry Run] Planned output videos:")
            for group_name, group_clips in sorted(grouped.items()):
                target = self._resolve_target_path(group_name, len(grouped))
                print(f"  -> {target} ({len(group_clips)} clips)")
            return []

        compiled_paths = []
        for group_name, group_clips in sorted(grouped.items()):
            target_file = self._resolve_target_path(group_name, len(grouped))
            print(
                f"\nCompiling group '{group_name}' ({len(group_clips)} clips) -> {target_file}..."
            )
            out = self.compiler.compile(
                [c.path for c in group_clips],
                target_file,
            )
            compiled_paths.append(out)

        print(
            f"\nAll video compilation complete! ({len(compiled_paths)} video(s) created)."
        )
        return compiled_paths

    def _resolve_target_path(self, group_name: str, total_groups: int) -> Path:
        """Determines target output video path based on output_target and group count."""
        # If output_target ends in .mp4 and there's only 1 group, use that exact filename
        if self.output_target.suffix.lower() == ".mp4" and total_groups == 1:
            return self.output_target
        elif self.output_target.suffix.lower() == ".mp4":
            # If multiple groups but user passed a .mp4 path, write into its parent folder
            return self.output_target.parent / f"{group_name}.mp4"
        else:
            return self.output_target / f"{group_name}.mp4"


def generate_combined_mp4(source_dir: str, output_path: str) -> None:
    """Finds all MP4 files under source directory and concatenates them naturally.

    Maintained for backwards compatibility.

    Args:
        source_dir: Root directory path containing MP4 video clips.
        output_path: File destination path for the concatenated master video.
    """
    manager = ExperimentVideoManager(
        source_dir=source_dir,
        output_dir=output_path,
        group_by="none",
        verbose=True,
    )
    results = manager.run()
    if not results:
        raise FileNotFoundError(
            f"No MP4 files found in {source_dir} or its subdirectories."
        )


def main() -> None:
    """CLI entrypoint for experiment video compilation."""
    default_source = Path("outputs/experiments")
    if not default_source.exists():
        fallback_source = project_root / "outputs" / "experiments"
        if fallback_source.exists():
            default_source = fallback_source

    default_output = Path("outputs/videos")
    if not default_output.exists():
        fallback_output = project_root / "outputs" / "videos"
        if fallback_output.parent.exists():
            default_output = fallback_output

    parser = argparse.ArgumentParser(
        description="UrbanMARL Experiment Video Compiler: Concatenate and organize evaluation videos."
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default=str(default_source),
        help=f"Root directory containing experiment runs or MP4 files (default: {default_source}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(default_output),
        help=f"Output directory or file path for compiled video(s) (default: {default_output}).",
    )
    parser.add_argument(
        "-g",
        "--group-by",
        "--by",
        type=str,
        default="both",
        choices=[
            "both",
            "combo",
            "all",
            "task",
            "scenario",
            "algorithm",
            "algo",
            "model",
            "task_algorithm",
            "task_algo",
            "none",
            "single",
        ],
        help=(
            "Grouping strategy for compiling videos:\n"
            "  'both' / 'combo': group by task, algorithm, and model (e.g. uav_mobile_ue_maddpg_mlp.mp4) [DEFAULT]\n"
            "  'task' / 'scenario': group by task only (e.g. uav_mobile_ue.mp4)\n"
            "  'algorithm' / 'algo': group by algorithm only (e.g. maddpg.mp4)\n"
            "  'model': group by model architecture only (e.g. mlp.mp4)\n"
            "  'task_algorithm': group by task and algorithm (e.g. uav_mobile_ue_maddpg.mp4)\n"
            "  'none' / 'single': compile all discovered clips into a single master video"
        ),
    )
    parser.add_argument(
        "-t",
        "--task",
        "--scenario",
        type=str,
        default=None,
        help="Filter by task / scenario name (e.g., uav_mobile_ue, coverage).",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        "--algo",
        type=str,
        default=None,
        help="Filter by algorithm name (e.g., maddpg, mappo, iddpg).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Filter by model architecture name (e.g., mlp).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for output video (default: 20).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec for compilation (default: libx264).",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Scan and display discovered video groups without compiling.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress detailed compilation output.",
    )

    args = parser.parse_args()

    manager = ExperimentVideoManager(
        source_dir=args.source,
        output_dir=args.output,
        group_by=args.group_by,
        task=args.task,
        algorithm=args.algorithm,
        model=args.model,
        fps=args.fps,
        codec=args.codec,
        verbose=not args.quiet,
    )
    manager.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
