import os
import re
import argparse
from pathlib import Path

# Handle imports based on the installed moviepy version
try:
    # MoviePy 1.x
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    # MoviePy 2.x
    from moviepy import VideoFileClip, concatenate_videoclips

def natural_sort_key(filepath: Path) -> list:
    """
    Breaks a string into a list of strings and integers for natural sorting.
    Ensures 'video_2.mp4' comes before 'video_10.mp4'.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(filepath))]

def generate_combined_mp4(source_dir: str, output_path: str):
    """
    Recursively finds all MP4 files in the source directory, sorts them naturally,
    and concatenates them into a single video file.
    """
    directory = Path(source_dir)
    
    # Recursively find all mp4 files in subdirectories
    video_files = list(directory.rglob("*.mp4"))
    
    if not video_files:
        raise FileNotFoundError(f"No MP4 files found in {directory} or its subdirectories.")
        
    # Sort files naturally based on their full paths
    video_files.sort(key=natural_sort_key)
    
    print(f"Found {len(video_files)} video files.")
    for vf in video_files:
        # Print the parent experiment folder and the video name for clarity
        print(f" - {vf.parent.parent.name}/{vf.name}")
        
    print("\nLoading clips...")
    clips = []
    for vf in video_files:
        clips.append(VideoFileClip(str(vf)))
        
    print("Concatenating clips (this may take several minutes)...")
    # method="compose" ensures that if videos have different resolutions,
    # they are safely padded/centered on a uniform canvas without crashing.
    final_clip = concatenate_videoclips(clips, method="compose")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing final video to {output_path}...")
    final_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        fps=20 # Adjust if your original videos use a different framerate
    )
    
    # Close clips to free memory
    for clip in clips:
        clip.close()
    final_clip.close()
    print("Video compilation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple MP4 videos into a single master video.")
    
    parser.add_argument(
        "-s", "--source", 
        type=str, 
        required=True, 
        help="Root directory to search for MP4 files (e.g., outputs/experiments)."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        required=True, 
        help="Path and filename for the output master video (e.g., outputs/videos/master_eval.mp4)."
    )
    
    args = parser.parse_args()
    generate_combined_mp4(args.source, args.output) 
