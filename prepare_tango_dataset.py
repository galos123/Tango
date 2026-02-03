"""
Tango Dataset Preparation Script
Downloads AudioCaps dataset and prepares it in the required format:
- mel-spectrogram/
- latent-vector/
- original-audio/
- caption/

Each sample has files named with the same index.
"""

import os
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
from typing import Dict, List, Tuple
import pickle
import argparse
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


class TangoDatasetPreparer:
    """Prepares AudioCaps dataset for Tango training."""

    def __init__(
        self,
        root_dir: str = "./data",
        output_dir: str = "./data/tango_dataset",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 160,
        max_duration: float = 10.0,
        use_audioldm_vae: bool = True,
        max_samples: int = None,
    ):
        """
        Args:
            root_dir: Root directory for downloading raw data
            output_dir: Output directory for processed dataset
            sample_rate: Target sample rate (16kHz for Tango)
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Hop length for STFT
            max_duration: Maximum audio duration in seconds
            use_audioldm_vae: Whether to use AudioLDM VAE for latent encoding
            max_samples: Maximum number of samples to process per subset (None for all)
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.use_audioldm_vae = use_audioldm_vae
        self.max_samples = max_samples

        # Create output directories
        self.mel_dir = self.output_dir / "mel-spectrogram"
        self.latent_dir = self.output_dir / "latent-vector"
        self.audio_dir = self.output_dir / "original-audio"
        self.caption_dir = self.output_dir / "caption"

        for dir_path in [
            self.mel_dir,
            self.latent_dir,
            self.audio_dir,
            self.caption_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Initialize VAE for latent encoding if needed
        self.vae = None
        if use_audioldm_vae:
            self._load_audioldm_vae()

    def _load_audioldm_vae(self):
        """Load AudioLDM VAE for latent encoding."""
        try:
            from audioldm.variational_autoencoder.autoencoder import AutoencoderKL
            
            print("Loading AudioLDM VAE...")
            
            # CORRECT CONFIG for AudioLDM-S-Full (Small)
            # The previous error happened because ch_mult was [1, 2, 4, 4]
            ddconfig = {
                "double_z": True,
                "z_channels": 8,
                "resolution": 256,
                "in_channels": 1,
                "out_ch": 1,
                "ch": 128,
                "ch_mult": [1, 2, 4],  # CHANGED: Removed the extra '4' to match the S-Full checkpoint
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
            }
            
            embed_dim = 8
            
            # Initialize VAE with the config
            self.vae = AutoencoderKL(
                ddconfig=ddconfig,
                embed_dim=embed_dim
            )
            
            # Load pretrained weights
            ckpt_path = self._download_audioldm_checkpoint()
            if ckpt_path and os.path.exists(ckpt_path):
                print(f"Loading checkpoint from {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                
                if "state_dict" in checkpoint:
                    vae_state_dict = {
                        k.replace("first_stage_model.", ""): v
                        for k, v in checkpoint["state_dict"].items()
                        if k.startswith("first_stage_model.")
                    }
                    
                    # strict=True ensures the config matches the weights exactly
                    self.vae.load_state_dict(vae_state_dict, strict=True)
                    print("AudioLDM VAE loaded successfully!")
                    
            self.vae.eval()
            
        except Exception as e:
            # This fixes the NameError 'e' by ensuring 'e' is defined in the except block
            print(f"Warning: Could not load AudioLDM VAE: {e}")
            print("Latent vectors will be computed using simple dimensionality reduction.")
            self.use_audioldm_vae = False

    def _download_audioldm_checkpoint(self) -> str:
        """Download AudioLDM checkpoint."""
        import urllib.request

        cache_dir = Path.home() / ".cache" / "audioldm"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cache_dir / "audioldm-s-full.ckpt"

        if not ckpt_path.exists():
            print("Downloading AudioLDM checkpoint...")
            url = "https://zenodo.org/record/7600541/files/audioldm-s-full?download=1"
            try:
                urllib.request.urlretrieve(url, str(ckpt_path))
                print("AudioLDM checkpoint downloaded!")
            except Exception as e:
                print(f"Failed to download AudioLDM checkpoint: {e}")
                return None

        return str(ckpt_path)

    def download_audiocaps(self, subset: str = "train"):
        """
        Download AudioCaps dataset.

        Args:
            subset: Dataset subset ('train', 'val', or 'test')
        """
        try:
            from audiocaps_download import Downloader

            print(f"Downloading AudioCaps {subset} subset...")
            audio_dir = self.root_dir / "audiocaps" / subset
            audio_dir.mkdir(parents=True, exist_ok=True)

            downloader = Downloader(root_path=str(audio_dir), n_jobs=8)
            downloader.download(format="wav")
            print(f"AudioCaps {subset} download complete!")

        except Exception as e:
            print(f"Error downloading AudioCaps: {e}")
            print("Please install audiocaps-download: pip install audiocaps-download")
            raise

    def load_audiocaps_metadata(self, subset: str = "train") -> List[Dict]:
        """
        Load AudioCaps metadata from CSV/JSON files.

        Args:
            subset: Dataset subset

        Returns:
            List of metadata dictionaries
        """
        # Try to load from the original AudioCaps repository format
        metadata_file = self.root_dir / "audiocaps" / f"{subset}.csv"

        if not metadata_file.exists():
            # Download metadata
            import pandas as pd
            import urllib.request

            base_url = (
                "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset"
            )
            url = f"{base_url}/{subset}.csv"

            try:
                print(f"Downloading AudioCaps metadata for {subset}...")
                urllib.request.urlretrieve(url, str(metadata_file))
            except Exception as e:
                print(f"Failed to download metadata: {e}")
                raise

        # Load CSV
        import pandas as pd

        df = pd.read_csv(metadata_file)

        metadata = []
        for idx, row in df.iterrows():
            metadata.append(
                {
                    "audiocap_id": row.get("audiocap_id", idx),
                    "youtube_id": row.get("youtube_id", ""),
                    "start_time": row.get("start_time", 0),
                    "caption": row.get("caption", ""),
                    "file_name": f"{row.get('youtube_id', '')}_{row.get('start_time', 0)}.wav",
                }
            )

        return metadata

    def compute_mel_spectrogram(self, audio: torch.Tensor) -> np.ndarray:
        """
        Compute mel-spectrogram from audio.

        Args:
            audio: Audio tensor (channels, samples)

        Returns:
            Mel-spectrogram as numpy array
        """
        # Ensure audio is mono
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
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )

        # Remove axes and padding for a clean spectrogram image
        ax.axis("off")
        plt.tight_layout(pad=0)

        # Save as PNG
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)

    def compute_latent_vector(self, mel_spec: torch.Tensor) -> np.ndarray:
        """
        Compute latent vector from mel-spectrogram.

        Args:
            mel_spec: Mel-spectrogram tensor

        Returns:
            Latent vector as numpy array
        """
        if self.vae is not None and self.use_audioldm_vae:
            # Use AudioLDM VAE
            with torch.no_grad():
                # Prepare input for VAE
                if mel_spec.dim() == 2:
                    mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
                elif mel_spec.dim() == 3:
                    mel_spec = mel_spec.unsqueeze(0)

                # Encode to latent space
                latent = self.vae.encode(mel_spec)
                if hasattr(latent, "sample"):
                    latent = latent.sample()

                return latent.squeeze().cpu().numpy()
        else:
            # Simple dimensionality reduction using averaging
            # This is a placeholder - in practice you'd want to use the actual VAE
            if isinstance(mel_spec, torch.Tensor):
                mel_spec = mel_spec.numpy()

            # Downsample mel-spectrogram to create a compact representation
            latent = mel_spec.mean(axis=-1)  # Average over time
            return latent

    def process_audio_file(
        self, audio_path: Path, index: int, caption: str, save: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Process a single audio file.

        Args:
            audio_path: Path to audio file
            index: Sample index for naming
            caption: Text caption
            save: Whether to save processed files

        Returns:
            Tuple of (mel_spec, latent, audio, caption)
        """
        # Load audio
        try:
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

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None, None, None, None

        # Compute mel-spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)

        # Compute latent vector
        latent = self.compute_latent_vector(torch.from_numpy(mel_spec))

        # Convert audio to numpy
        audio_np = audio.squeeze().numpy()

        if save:
            # Save all components
            # Save mel-spectrogram as PNG image
            self.save_mel_spectrogram_image(mel_spec, self.mel_dir / f"{index:06d}.png")

            # Also save as numpy for numerical processing if needed
            np.save(self.mel_dir / f"{index:06d}.npy", mel_spec)

            # Save latent vector
            np.save(self.latent_dir / f"{index:06d}.npy", latent)

            # Save original audio
            sf.write(self.audio_dir / f"{index:06d}.wav", audio_np, self.sample_rate)

            # Save caption
            with open(self.caption_dir / f"{index:06d}.txt", "w") as f:
                f.write(caption)

        return mel_spec, latent, audio_np, caption

    def prepare_dataset(self, subsets: List[str] = ["train", "val", "test"]):
        """
        Prepare the complete dataset.

        Args:
            subsets: List of subsets to process
        """
        global_index = 0
        dataset_info = {
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "max_duration": self.max_duration,
            "max_samples_per_subset": self.max_samples,
            "subsets": {},
        }

        for subset in subsets:
            print(f"\n{'='*50}")
            print(f"Processing {subset} subset")
            if self.max_samples:
                print(f"Max samples limit: {self.max_samples}")
            print(f"{'='*50}")

            # Download data if needed
            audio_dir = self.root_dir / "audiocaps" / subset
            if not audio_dir.exists():
                self.download_audiocaps(subset)

            # Load metadata
            metadata = self.load_audiocaps_metadata(subset)

            # Limit metadata if max_samples is set
            if self.max_samples and len(metadata) > self.max_samples:
                print(f"Limiting from {len(metadata)} to {self.max_samples} samples")
                metadata = metadata[: self.max_samples]

            # Process each audio file
            subset_indices = []
            successful_samples = 0

            for item in tqdm(metadata, desc=f"Processing {subset}"):
                # Find audio file
                audio_file = audio_dir / f"{subset}/{item['audiocap_id']}.wav"

                # Try alternative naming conventions
                if not audio_file.exists():
                    # Try with different formats
                    possible_names = [
                        f"{item['audiocap_id']}_{item['start_time']}.wav",
                        f"{item['audiocap_id']}_{item['start_time']}_10000.wav",
                        f"{item['audiocap_id']}_{int(item['start_time'])}_{int(item['start_time'])+10000}.wav",
                    ]

                    for name in possible_names:
                        test_path = audio_dir / name
                        if test_path.exists():
                            audio_file = test_path
                            break

                if not audio_file.exists():
                    print(f"Warning: Audio file not found: {audio_file}")
                    continue

                # Process the file
                result = self.process_audio_file(
                    audio_file, global_index, item["caption"], save=True
                )

                if result[0] is not None:
                    subset_indices.append(global_index)
                    successful_samples += 1
                    global_index += 1

            dataset_info["subsets"][subset] = {
                "indices": subset_indices,
                "num_samples": successful_samples,
            }

            print(f"Successfully processed {successful_samples} samples from {subset}")

        # Save dataset info
        with open(self.output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        print(f"\n{'='*50}")
        print(f"Dataset preparation complete!")
        print(f"Total samples: {global_index}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Tango dataset from AudioCaps")
    parser.add_argument(
        "--root_dir", type=str, default="./data", help="Root directory for raw data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/tango_dataset",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset subsets to process",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--no_vae", action="store_true", help="Disable AudioLDM VAE for latent encoding"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per subset (None for all)",
    )

    args = parser.parse_args()

    preparer = TangoDatasetPreparer(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        use_audioldm_vae=not args.no_vae,
        max_samples=args.max_samples,
    )

    preparer.prepare_dataset(subsets=args.subsets)


if __name__ == "__main__":
    main()
