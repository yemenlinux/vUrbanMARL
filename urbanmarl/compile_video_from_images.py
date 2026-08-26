"""UrbanMARL PNG Image Sequence to MP4 Video Compiler.

Compiles image frame collections into standardized MP4 videos with auto-scaling canvas sizing.
"""

import argparse
import os
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def generate_mp4_from_pngs(image_dir: str, output_path: str, fps: int) -> None:
    """Collects PNG images, sorts them numerically, scales, and compiles an MP4 video.

    Args:
        image_dir (str): Source directory path containing PNG frames.
        output_path (str): File destination path for the output MP4 video.
        fps (int): Frame rate (frames per second).

    Raises:
        FileNotFoundError: If no PNG files are found in the directory.
    """
    directory = Path(image_dir)
    image_files = list(directory.glob("*.png"))

    if not image_files:
        raise FileNotFoundError(
            f"No PNG files found in the directory: {directory}"
        )

    def extract_number(filepath: Path) -> int:
        numbers = re.findall(r'\d+', filepath.name)
        return int(numbers[-1]) if numbers else 0

    image_files.sort(key=extract_number)

    print("Scanning images to determine the largest dimensions...")
    max_w, max_h = 0, 0
    for image_path in image_files:
        with Image.open(image_path) as img:
            w, h = img.size
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h

    target_w = max_w if max_w % 2 == 0 else max_w + 1
    target_h = max_h if max_h % 2 == 0 else max_h + 1
    target_size = (target_w, target_h)

    print(
        f"Compiling {len(image_files)} images into {output_path} at {fps} FPS..."
    )
    print(
        f"Standardizing all frames to the maximum size: {target_w}x{target_h}..."
    )

    with imageio.get_writer(
        output_path, fps=fps, macro_block_size=None
    ) as writer:
        for image_path in image_files:
            img = imageio.imread(image_path)

            if img.shape[:2] != (target_h, target_w):
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize(target_size, Image.Resampling.LANCZOS)
                img = np.array(img_pil)

            if fps < 20:
                for _ in range(fps // 2):
                    writer.append_data(img)
            else:
                writer.append_data(img)

    print("Video compilation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile a sequence of PNG images into an MP4 video."
    )

    parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help="Path to the directory containing the source PNG images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path and filename for the output MP4 video (e.g., output.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the output video (default: 20).",
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_mp4_from_pngs(args.source, args.output, args.fps)
