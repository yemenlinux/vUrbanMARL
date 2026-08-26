"""UrbanMARL Experiment Video Compiler.

Concatenates recorded MP4 trajectory videos across multiple algorithm experiment runs
into a unified compilation video.
"""

import argparse
import os
import re
from pathlib import Path

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips


def natural_sort_key(filepath: Path) -> list:
    """Splits a file path into strings and integers for natural sorting.

    Ensures numerical filenames are sorted in ascending order (e.g. 'video_2.mp4' before 'video_10.mp4').

    Args:
        filepath (Path): File path object.

    Returns:
        list: List of alternating string and integer tokens.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', str(filepath))
    ]


def generate_combined_mp4(source_dir: str, output_path: str) -> None:
    """Finds all MP4 files under source directory and concatenates them naturally.

    Args:
        source_dir (str): Root directory path containing MP4 video clips.
        output_path (str): File destination path for the concatenated master video.

    Raises:
        FileNotFoundError: If no MP4 video files are discovered in the source directory.
    """
    directory = Path(source_dir)
    video_files = list(directory.rglob("*.mp4"))

    if not video_files:
        raise FileNotFoundError(
            f"No MP4 files found in {directory} or its subdirectories."
        )

    video_files.sort(key=natural_sort_key)

    print(f"Found {len(video_files)} video files.")
    for vf in video_files:
        print(f" - {vf.parent.parent.name}/{vf.name}")

    print("\nLoading clips...")
    clips = []
    for vf in video_files:
        clips.append(VideoFileClip(str(vf)))

    print("Concatenating clips (this may take several minutes)...")
    final_clip = concatenate_videoclips(clips, method="compose")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing final video to {output_path}...")
    final_clip.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=20
    )

    for clip in clips:
        clip.close()
    final_clip.close()
    print("Video compilation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple MP4 videos into a single master video."
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help="Root directory to search for MP4 files (e.g., outputs/experiments).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path and filename for the output master video (e.g., outputs/videos/master_eval.mp4).",
    )

    args = parser.parse_args()
    generate_combined_mp4(args.source, args.output)
