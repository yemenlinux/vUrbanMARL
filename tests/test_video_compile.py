"""Unit tests for urbanmarl video compilation utilities."""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from urbanmarl.compile_video_from_images import generate_mp4_from_pngs
from urbanmarl.compile_experiments_video import natural_sort_key


def test_natural_sort_key():
    p1 = Path("video_2.mp4")
    p2 = Path("video_10.mp4")
    p3 = Path("video_1.mp4")
    files = [p1, p2, p3]
    files.sort(key=natural_sort_key)
    assert files == [p3, p1, p2]


def test_generate_mp4_from_pngs_empty(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_file = tmp_path / "out.mp4"
    with pytest.raises(FileNotFoundError):
        generate_mp4_from_pngs(str(empty_dir), str(out_file), fps=20)


def test_generate_mp4_from_pngs(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Create 3 synthetic PNG frames
    for i in range(3):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[i*10:(i+1)*10, :, 0] = 255
        img = Image.fromarray(arr)
        img.save(img_dir / f"frame_{i}.png")
        
    out_file = tmp_path / "output.mp4"
    generate_mp4_from_pngs(str(img_dir), str(out_file), fps=10)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
