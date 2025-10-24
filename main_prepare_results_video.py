import cv2
import os
from pathlib import Path
import argparse

def create_videos_from_images(results_dir: str, subfolder: str, fps: int = 5, image_exts=('.png', '.jpg', '.jpeg')):
    """
    Recursively creates videos from image sequences in all subdirectories of a given folder.

    Args:
        results_dir (str): Path to the 'results' directory.
        subfolder (str): Subfolder name inside 'results' treated as root.
        fps (int): Frame rate for the generated videos.
        image_exts (tuple): Accepted image extensions.
    """
    root_path = Path(results_dir) / subfolder
    if not root_path.exists():
        print(f"Path does not exist: {root_path}")
        return

    videos_dir = root_path / "videos"
    videos_dir.mkdir(exist_ok=True)

    print(f"Starting video generation from: {root_path}")
    print(f"Videos will be saved in: {videos_dir}\n")
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')

    for dirpath, _, filenames in os.walk(root_path):
        dirpath = Path(dirpath)
        if dirpath == videos_dir:
            continue  # skip the videos folder itself

        # Filter image files and sort them
        images = sorted([f for f in filenames if f.lower().endswith(image_exts)])
        if not images:
            continue

        # Read first image to get frame size
        first_image_path = dirpath / images[0]
        frame = cv2.imread(str(first_image_path))
        if frame is None:
            print(f"Skipping {dirpath}: unable to read first image.")
            continue

        height, width, _ = frame.shape

        # Name the video after the folder
        video_name = f"{dirpath.name}.avi"
        video_path = videos_dir / video_name

        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        print(f"Creating video from: {dirpath}")
        for img_name in images:
            img_path = dirpath / img_name
            img = cv2.imread(str(img_path))
            if img is not None:
                out.write(img)
            else:
                print(f"Could not read {img_path}, skipping frame.")

        out.release()
        print(f"Saved video: {video_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create videos from image sequences in results/<subfolder>.")
    parser.add_argument("subfolder", type=str, help="Subfolder inside 'results' to process")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to the results directory (default: 'results')")
    parser.add_argument("--fps", type=int, default=5, help="Frame rate for the output videos (default: 5)")

    args = parser.parse_args()
    create_videos_from_images(args.results_dir, args.subfolder, args.fps)

