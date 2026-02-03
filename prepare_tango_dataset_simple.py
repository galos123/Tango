"""
Simplified Tango Dataset Preparation (No HuggingFace)
Downloads AudioCaps and prepares data without using HuggingFace libraries.
"""

import os
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from typing import Dict, List, Optional
import argparse
import urllib.request
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class SimpleTangoDatasetPreparer:
    """Simplified dataset preparer without HuggingFace dependencies."""
    
    def __init__(
        self,
        root_dir: str = "./data",
        output_dir: str = "./data/tango_dataset",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 160,
        max_duration: float = 10.0,
        max_samples: int = None,
    ):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.max_samples = max_samples
        
        # Create output directories
        self.mel_dir = self.output_dir / "mel-spectrogram"
        self.latent_dir = self.output_dir / "latent-vector"
        self.audio_dir = self.output_dir / "original-audio"
        self.caption_dir = self.output_dir / "caption"
        
        for dir_path in [self.mel_dir, self.latent_dir, self.audio_dir, self.caption_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    
    def download_metadata(self, subset: str = "train"):
        """Download AudioCaps metadata CSV files."""
        metadata_dir = self.root_dir / "audiocaps_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = metadata_dir / f"{subset}.csv"
        
        if metadata_file.exists():
            print(f"Metadata for {subset} already exists.")
            return str(metadata_file)
        
        # Download from AudioCaps repository
        base_url = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset"
        url = f"{base_url}/{subset}.csv"
        
        print(f"Downloading metadata for {subset} from {url}...")
        try:
            urllib.request.urlretrieve(url, str(metadata_file))
            print(f"Metadata downloaded: {metadata_file}")
            return str(metadata_file)
        except Exception as e:
            print(f"Failed to download metadata: {e}")
            raise
    
    def load_metadata(self, metadata_file: str) -> List[Dict]:
        """Load metadata from CSV file."""
        metadata = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata.append({
                    'audiocap_id': row.get('audiocap_id', ''),
                    'youtube_id': row.get('youtube_id', ''),
                    'start_time': int(row.get('start_time', 0)),
                    'caption': row.get('caption', ''),
                })
        
        return metadata
    
    def download_audio_yt_dlp(self, youtube_id: str, start_time: int, output_path: Path) -> bool:
        """
        Download audio from YouTube using yt-dlp.
        
        Args:
            youtube_id: YouTube video ID
            start_time: Start time in seconds
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        if output_path.exists():
            return True
        
        try:
            import subprocess
            
            # YouTube URL
            url = f"https://www.youtube.com/watch?v={youtube_id}"
            
            # yt-dlp command to download audio segment
            end_time = start_time + 10  # 10 second clips
            
            cmd = [
                "yt-dlp",
                "-f", "bestaudio",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--postprocessor-args", f"-ss {start_time} -t 10",
                "-o", str(output_path),
                url
            ]
            
            # Run command
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60
            )
            
            return output_path.exists()
            
        except Exception as e:
            print(f"Error downloading {youtube_id}: {e}")
            return False
    
    def compute_mel_spectrogram(self, audio: torch.Tensor) -> np.ndarray:
        """Compute mel-spectrogram from audio."""
        # Ensure mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Compute mel-spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-7)
        
        return mel_spec.squeeze(0).numpy()
    
    def save_mel_spectrogram_image(self, mel_spec: np.ndarray, save_path: Path):
        """
        Save mel-spectrogram as PNG image.
        
        Args:
            mel_spec: Mel-spectrogram array (n_mels, time_steps)
            save_path: Path to save the image
        """
        # Create figure without axes for clean image
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot mel-spectrogram
        im = ax.imshow(
            mel_spec,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        
        # Remove axes and padding for a clean spectrogram image
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Save as PNG
        plt.savefig(
            save_path,
            bbox_inches='tight',
            pad_inches=0,
            dpi=100
        )
        plt.close(fig)
    
    def compute_latent_vector(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Compute latent vector from mel-spectrogram using simple compression.
        
        For actual Tango training, you should use the AudioLDM VAE.
        This is a simplified version using dimensionality reduction.
        """
        # Simple approach: use PCA-like compression
        # Average over time dimension
        temporal_avg = mel_spec.mean(axis=1)
        
        # Also include some temporal information
        # by computing statistics across time
        temporal_std = mel_spec.std(axis=1)
        temporal_max = mel_spec.max(axis=1)
        temporal_min = mel_spec.min(axis=1)
        
        # Concatenate features
        latent = np.concatenate([
            temporal_avg,
            temporal_std,
            temporal_max,
            temporal_min
        ])
        
        return latent
    
    def process_audio_file(
        self,
        audio_path: Path,
        index: int,
        caption: str
    ) -> bool:
        """Process a single audio file and save all components."""
        try:
            # Load audio
            audio, sr = torchaudio.load(str(audio_path))
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Ensure mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Trim or pad to max_duration
            max_samples = int(self.max_duration * self.sample_rate)
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
            elif audio.shape[1] < max_samples:
                padding = max_samples - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            
            # Compute mel-spectrogram
            mel_spec = self.compute_mel_spectrogram(audio)
            
            # Compute latent vector
            latent = self.compute_latent_vector(mel_spec)
            
            # Convert audio to numpy
            audio_np = audio.squeeze().numpy()
            
            # Save all components with the same index
            # Save mel-spectrogram as PNG image
            self.save_mel_spectrogram_image(
                mel_spec,
                self.mel_dir / f"{index:06d}.png"
            )
            
            # Also save as numpy for numerical processing if needed
            np.save(self.mel_dir / f"{index:06d}.npy", mel_spec)
            
            # Save latent vector
            np.save(self.latent_dir / f"{index:06d}.npy", latent)
            
            # Save original audio
            sf.write(
                self.audio_dir / f"{index:06d}.wav",
                audio_np,
                self.sample_rate
            )
            
            # Save caption
            with open(self.caption_dir / f"{index:06d}.txt", 'w', encoding='utf-8') as f:
                f.write(caption)
            
            return True
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return False
    
    def prepare_dataset_from_local(
        self,
        audio_dir: Path,
        metadata_file: str,
        subset_name: str = "train"
    ) -> Dict:
        """
        Prepare dataset from locally available audio files.
        
        Args:
            audio_dir: Directory containing audio files
            metadata_file: Path to metadata CSV
            subset_name: Name of the subset
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n{'='*50}")
        print(f"Processing {subset_name} subset from local files")
        print(f"Audio directory: {audio_dir}")
        if self.max_samples:
            print(f"Max samples limit: {self.max_samples}")
        print(f"{'='*50}\n")
        
        # Load metadata
        metadata = self.load_metadata(metadata_file)
        
        # Limit metadata if max_samples is set
        if self.max_samples and len(metadata) > self.max_samples:
            print(f"Limiting from {len(metadata)} to {self.max_samples} samples")
            metadata = metadata[:self.max_samples]
        
        # Process each sample
        subset_indices = []
        successful = 0
        failed = 0
        
        for idx, item in enumerate(tqdm(metadata, desc=f"Processing {subset_name}")):
            # Construct potential filenames
            audiocap_id = item['audiocap_id']
            start_time = item['start_time']
            caption = item['caption']
            
            # Try multiple naming conventions
            possible_files = [
                audio_dir / f"{audiocap_id}_{start_time}.wav",
                audio_dir / f"{audiocap_id}_{start_time}_{start_time+10000}.wav",
                audio_dir / f"{subset_name}/{audiocap_id}.wav",
            ]
            
            audio_file = None
            for f in possible_files:
                if f.exists():
                    audio_file = f
                    break
            
            if audio_file is None:
                failed += 1
                continue
            
            # Process the audio file
            sample_idx = len(subset_indices)
            if self.process_audio_file(audio_file, sample_idx, caption):
                subset_indices.append(sample_idx)
                successful += 1
            else:
                failed += 1
        
        stats = {
            'subset': subset_name,
            'total': len(metadata),
            'successful': successful,
            'failed': failed,
            'indices': subset_indices
        }
        
        print(f"\n{subset_name} processing complete:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        
        return stats
    
    def prepare_dataset(self, subsets: List[str] = ["train", "val", "test"]):
        """
        Main function to prepare the complete dataset.
        
        Args:
            subsets: List of subsets to process
        """
        all_stats = []
        
        for subset in subsets:
            # Download metadata
            metadata_file = self.download_metadata(subset)
            
            # Expected audio directory from audiocaps-download
            audio_dir = self.root_dir / "audiocaps" / subset
            
            if not audio_dir.exists():
                print(f"\nAudio directory not found: {audio_dir}")
                print(f"Please download AudioCaps using audiocaps-download package:")
                print(f"  from audiocaps_download import Downloader")
                print(f"  d = Downloader(root_path='{audio_dir}/', n_jobs=16)")
                print(f"  d.download(format='wav')")
                continue
            
            # Process subset
            stats = self.prepare_dataset_from_local(
                audio_dir,
                metadata_file,
                subset
            )
            all_stats.append(stats)
        
        # Save overall dataset info
        dataset_info = {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'max_duration': self.max_duration,
            'subsets': {s['subset']: s for s in all_stats}
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Dataset preparation complete!")
        print(f"Dataset info saved to: {info_file}")
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Tango dataset from AudioCaps (Simplified, No HF)"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data",
        help="Root directory for raw data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/tango_dataset",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs='+',
        default=["train"],
        help="Dataset subsets to process (train, val, test)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate (Hz)"
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=64,
        help="Number of mel frequency bins"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per subset (None for all)"
    )
    
    args = parser.parse_args()
    
    print(f"\nTango Dataset Preparation")
    print(f"{'='*50}")
    print(f"Root directory: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Subsets: {args.subsets}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Mel bins: {args.n_mels}")
    if args.max_samples:
        print(f"Max samples per subset: {args.max_samples}")
    print(f"{'='*50}\n")
    
    preparer = SimpleTangoDatasetPreparer(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        max_samples=args.max_samples
    )
    
    preparer.prepare_dataset(subsets=args.subsets)


if __name__ == "__main__":
    main()