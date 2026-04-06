"""Download and extract datasets for fall detection training."""

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

# Add parent dir to path so falldet package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falldet.data.registry import DATASETS, get_dataset_info

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_git_clone(name: str, url: str, output_dir: Path) -> None:
    """Clone a GitHub repository as a dataset source."""
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"  [{name}] Already exists at {output_dir}, skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{name}] Cloning {url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(output_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [{name}] ERROR: git clone failed:\n{result.stderr}")
        # Clean up failed clone
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return

    # Remove .git directory to save space (force writable on Windows)
    git_dir = output_dir / ".git"
    if git_dir.exists():

        def _force_remove_readonly(func, path, _exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(git_dir, onexc=_force_remove_readonly)

    print(f"  [{name}] Done. Saved to {output_dir}")


def download_manual(name: str, url: str, output_dir: Path) -> None:
    """Print instructions for manually downloading a dataset."""
    print(f"  [{name}] Manual download required.")
    print(f"  Visit: {url}")
    print(f"  Download and extract to: {output_dir}")


def download_dataset(name: str) -> None:
    """Download a single dataset by name."""
    info = get_dataset_info(name)
    output_dir = DATA_DIR / name

    print(f"\n--- {info['name']} ---")

    if info["download_type"] == "git_clone":
        download_git_clone(name, info["url"], output_dir)
    elif info["download_type"] == "manual":
        download_manual(name, info["url"], output_dir)
    else:
        print(f"  [{name}] Unknown download type: {info['download_type']}")


def main():
    parser = argparse.ArgumentParser(description="Download fall detection datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name to download, or 'all' for all datasets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            dl = info["download_type"]
            print(f"  {name:12s}  [{dl:10s}]  {info['name']}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        for name in DATASETS:
            download_dataset(name)
    else:
        download_dataset(args.dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
