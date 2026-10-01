"""Unit tests for urbanmarl video compilation utilities."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from urbanmarl.compile_experiments_video import natural_sort_key
from urbanmarl.compile_video_from_images import generate_mp4_from_pngs


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
        arr[i * 10 : (i + 1) * 10, :, 0] = 255
        img = Image.fromarray(arr)
        img.save(img_dir / f"frame_{i}.png")

    out_file = tmp_path / "output.mp4"
    generate_mp4_from_pngs(str(img_dir), str(out_file), fps=10)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_experiment_metadata_parser():
    from urbanmarl.compile_experiments_video import ExperimentMetadataParser

    folder_name = "maddpg_uav_mobile_ue_mlp__d26e7a96_26_09_25-06_56_55"
    meta = ExperimentMetadataParser.parse_folder_name(folder_name)
    assert meta["algorithm"] == "maddpg"
    assert meta["task"] == "uav_mobile_ue"
    assert meta["model"] == "mlp"
    assert meta["hash"] == "d26e7a96"
    assert meta["timestamp_raw"] == "26_09_25-06_56_55"


def test_experiment_video_scanner_and_grouping(tmp_path):
    from urbanmarl.compile_experiments_video import (
        ExperimentVideoScanner,
        ExperimentVideoManager,
        VideoClipInfo,
    )

    # Setup mock directory structure
    exp1 = tmp_path / "maddpg_uav_mobile_ue_mlp__d26e7a96_26_09_25-06_56_55" / "videos"
    exp2 = tmp_path / "mappo_uav_mobile_ue_mlp__69c3d9e0_26_09_25-06_26_04" / "videos"
    exp3 = tmp_path / "maddpg_coverage_mlp__586cb22a_26_09_25-03_07_09" / "videos"
    for p in (exp1, exp2, exp3):
        p.mkdir(parents=True)

    # Touch mock mp4 files
    (exp1 / "eval_video_0.mp4").touch()
    (exp1 / "eval_video_4.mp4").touch()
    (exp2 / "eval_video_0.mp4").touch()
    (exp3 / "eval_video_0.mp4").touch()

    # 1. Full scan
    scanner = ExperimentVideoScanner(source_dir=tmp_path)
    clips = scanner.scan()
    assert len(clips) == 4

    # 2. Filter by task
    scanner_task = ExperimentVideoScanner(source_dir=tmp_path, task_filter="coverage")
    assert len(scanner_task.scan()) == 1

    # 3. Filter by algorithm
    scanner_algo = ExperimentVideoScanner(source_dir=tmp_path, algorithm_filter="maddpg")
    assert len(scanner_algo.scan()) == 3

    # 4. Grouping by 'both' (task_algo_model)
    grouped_both = scanner.group_clips(clips, group_by="both")
    assert set(grouped_both.keys()) == {
        "uav_mobile_ue_maddpg_mlp",
        "uav_mobile_ue_mappo_mlp",
        "coverage_maddpg_mlp",
    }
    assert len(grouped_both["uav_mobile_ue_maddpg_mlp"]) == 2

    # 5. Grouping by 'task'
    grouped_task = scanner.group_clips(clips, group_by="task")
    assert set(grouped_task.keys()) == {"uav_mobile_ue", "coverage"}
    assert len(grouped_task["uav_mobile_ue"]) == 3

    # 6. Grouping by 'algorithm'
    grouped_algo = scanner.group_clips(clips, group_by="algorithm")
    assert set(grouped_algo.keys()) == {"maddpg", "mappo"}
    assert len(grouped_algo["maddpg"]) == 3

    # 7. Grouping by 'model'
    grouped_model = scanner.group_clips(clips, group_by="model")
    assert set(grouped_model.keys()) == {"mlp"}
    assert len(grouped_model["mlp"]) == 4

    # 8. Target path resolution
    manager = ExperimentVideoManager(
        source_dir=tmp_path,
        output_dir=tmp_path / "videos",
        group_by="both",
    )
    target = manager._resolve_target_path("uav_mobile_ue_maddpg_mlp", 1)
    assert target.name == "uav_mobile_ue_maddpg_mlp.mp4"
    assert target.parent == tmp_path / "videos"
