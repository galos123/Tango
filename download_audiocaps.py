"""
AudioCaps Download Helper Script
Downloads AudioCaps dataset using the audiocaps-download package.
"""

import argparse
from pathlib import Path


def download_audiocaps(
    root_dir: str = "./data",
    subsets: list = ["train", "val", "test"],
    n_jobs: int = 8,
    format: str = "wav",
    max_samples: int = None
):
    """
    Download AudioCaps dataset.
    
    Args:
        root_dir: Root directory to save data
        subsets: List of subsets to download
        n_jobs: Number of parallel download jobs
        format: Audio format (wav, flac, mp3)
        max_samples: Maximum number of samples to download per subset (None for all)
    """
    try:
        from audiocaps_download import Downloader
    except ImportError:
        print("Error: audiocaps-download package not installed.")
        print("Please install it: pip install audiocaps-download")
        return
    
    root_path = Path(root_dir) / "audiocaps"
    
    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"Downloading AudioCaps {subset} subset")
        if max_samples:
            print(f"Max samples limit: {max_samples}")
        print(f"{'='*60}\n")
        
        subset_dir = root_path / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            downloader = Downloader(
                root_path=str(subset_dir),
                n_jobs=n_jobs
            )
            
            print(f"Downloading to: {subset_dir}")
            print(f"Format: {format}")
            print(f"Parallel jobs: {n_jobs}")
            
            # Note: audiocaps-download doesn't support limiting directly
            # Users would need to limit the metadata first or stop early
            if max_samples:
                print(f"Note: Download will proceed normally. Use --max_samples in")
                print(f"preparation script to limit processing to {max_samples} samples.")
            
            downloader.download(format=format)
            
            # Count downloaded files
            audio_files = list(subset_dir.glob(f"*.{format}"))
            print(f"\nDownloaded {len(audio_files)} audio files for {subset}")
            
        except Exception as e:
            print(f"Error downloading {subset}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Data saved to: {root_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download AudioCaps dataset"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data",
        help="Root directory to save data (default: ./data)"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs='+',
        default=["train", "val", "test"],
        help="Dataset subsets to download (default: train val test)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=8,
        help="Number of parallel download jobs (default: 8)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="wav",
        choices=["wav", "flac", "mp3"],
        help="Audio format (default: wav)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per subset (informational only, limit during processing)"
    )
    
    args = parser.parse_args()
    
    print("\nAudioCaps Download Script")
    print("="*60)
    print(f"Root directory: {args.root_dir}")
    print(f"Subsets: {args.subsets}")
    print(f"Format: {args.format}")
    print(f"Parallel jobs: {args.n_jobs}")
    if args.max_samples:
        print(f"Note: Will download all, but you can limit to {args.max_samples}")
        print(f"      samples per subset during dataset preparation")
    print("="*60)
    print("\nNote: This will download videos from YouTube.")
    print("Some videos may no longer be available.")
    print("Download may take several hours depending on your connection.")
    print("="*60 + "\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    download_audiocaps(
        root_dir=args.root_dir,
        subsets=args.subsets,
        n_jobs=args.n_jobs,
        format=args.format,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()