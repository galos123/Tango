"""
Example script showing how to load and use the prepared Tango dataset.
"""

import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import soundfile as sf
from typing import Dict, Optional
from PIL import Image


class TangoDataset(Dataset):
    """PyTorch Dataset for loading prepared Tango data."""
    
    def __init__(
        self,
        data_dir: str = "./data/tango_dataset",
        subset: Optional[str] = None,
        load_audio: bool = False,
        use_png_mel: bool = False,
        transform=None
    ):
        """
        Args:
            data_dir: Directory containing prepared dataset
            subset: Specific subset to load ('train', 'val', 'test'), or None for all
            load_audio: Whether to load audio waveforms (memory intensive)
            use_png_mel: Whether to load PNG images instead of NPY for mel-spectrograms
            transform: Optional transform to apply to samples
        """
        self.data_dir = Path(data_dir)
        self.load_audio = load_audio
        self.use_png_mel = use_png_mel
        self.transform = transform
        
        # Load dataset info
        info_file = self.data_dir / "dataset_info.json"
        with open(info_file, 'r') as f:
            self.info = json.load(f)
        
        # Get indices for requested subset(s)
        self.indices = []
        if subset is None:
            # Load all subsets
            for subset_info in self.info['subsets'].values():
                self.indices.extend(subset_info['indices'])
        else:
            # Load specific subset
            if subset not in self.info['subsets']:
                raise ValueError(f"Subset '{subset}' not found. Available: {list(self.info['subsets'].keys())}")
            self.indices = self.info['subsets'][subset]['indices']
        
        print(f"Loaded {len(self.indices)} samples from {data_dir}")
        if subset:
            print(f"Subset: {subset}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        
        # Load mel-spectrogram (PNG or NPY)
        if self.use_png_mel:
            # Load PNG image
            mel_path = self.data_dir / "mel-spectrogram" / f"{sample_idx:06d}.png"
            mel_img = Image.open(mel_path)
            mel_spec = np.array(mel_img)
            # Convert RGB to grayscale if needed, then to tensor
            if len(mel_spec.shape) == 3:
                mel_spec = mel_spec.mean(axis=2)
            mel_spec = torch.from_numpy(mel_spec).float()
        else:
            # Load NPY array
            mel_path = self.data_dir / "mel-spectrogram" / f"{sample_idx:06d}.npy"
            mel_spec = np.load(mel_path)
            mel_spec = torch.from_numpy(mel_spec).float()
        
        # Load latent vector
        latent_path = self.data_dir / "latent-vector" / f"{sample_idx:06d}.npy"
        latent = np.load(latent_path)
        latent = torch.from_numpy(latent).float()
        
        # Load caption
        caption_path = self.data_dir / "caption" / f"{sample_idx:06d}.txt"
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        sample = {
            'mel_spectrogram': mel_spec,
            'latent_vector': latent,
            'caption': caption,
            'index': sample_idx
        }
        
        # Optionally load audio
        if self.load_audio:
            audio_path = self.data_dir / "original-audio" / f"{sample_idx:06d}.wav"
            audio, sr = sf.read(audio_path)
            sample['audio'] = torch.from_numpy(audio).float()
            sample['sample_rate'] = sr
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_dataset_info(self):
        """Return dataset configuration information."""
        return self.info


def collate_fn(batch):
    """Custom collate function for batching."""
    mel_specs = torch.stack([item['mel_spectrogram'] for item in batch])
    latents = torch.stack([item['latent_vector'] for item in batch])
    captions = [item['caption'] for item in batch]
    indices = [item['index'] for item in batch]
    
    result = {
        'mel_spectrogram': mel_specs,
        'latent_vector': latents,
        'caption': captions,
        'index': indices
    }
    
    # Include audio if present
    if 'audio' in batch[0]:
        audios = torch.stack([item['audio'] for item in batch])
        result['audio'] = audios
        result['sample_rate'] = batch[0]['sample_rate']
    
    return result


def example_usage():
    """Example of how to use the dataset."""
    
    print("="*60)
    print("Tango Dataset Usage Example")
    print("="*60)
    
    # Create dataset for training
    train_dataset = TangoDataset(
        data_dir="./data/tango_dataset",
        subset="train",
        load_audio=False  # Set to True if you need audio waveforms
    )
    
    # Print dataset info
    info = train_dataset.get_dataset_info()
    print(f"\nDataset Information:")
    print(f"  Sample rate: {info['sample_rate']} Hz")
    print(f"  Mel bins: {info['n_mels']}")
    print(f"  Max duration: {info['max_duration']} seconds")
    print(f"  Total samples: {len(train_dataset)}")
    
    # Get a single sample
    print(f"\nSingle Sample:")
    sample = train_dataset[0]
    print(f"  Caption: {sample['caption']}")
    print(f"  Mel-spectrogram shape: {sample['mel_spectrogram'].shape}")
    print(f"  Latent vector shape: {sample['latent_vector'].shape}")
    
    # Create DataLoader
    print(f"\nCreating DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Iterate through batches
    print(f"\nBatch Example:")
    for batch in train_loader:
        print(f"  Batch size: {len(batch['caption'])}")
        print(f"  Mel-spectrogram batch shape: {batch['mel_spectrogram'].shape}")
        print(f"  Latent vector batch shape: {batch['latent_vector'].shape}")
        print(f"  Captions: {batch['caption'][:2]}")  # First 2 captions
        break
    
    # Example: Create datasets for all splits
    print(f"\n{'='*60}")
    print("Loading All Splits:")
    print("="*60)
    
    splits = {}
    for split in ['train', 'val', 'test']:
        try:
            splits[split] = TangoDataset(
                data_dir="./data/tango_dataset",
                subset=split,
                load_audio=False
            )
            print(f"{split:5s}: {len(splits[split]):5d} samples")
        except ValueError as e:
            print(f"{split:5s}: Not available ({e})")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


def inspect_dataset_structure():
    """Inspect and print dataset directory structure."""
    data_dir = Path("./data/tango_dataset")
    
    print("\n" + "="*60)
    print("Dataset Directory Structure:")
    print("="*60)
    
    subdirs = [
        ("mel-spectrogram", ["*.png", "*.npy"]),
        ("latent-vector", ["*.npy"]),
        ("original-audio", ["*.wav"]),
        ("caption", ["*.txt"])
    ]
    
    for subdir, patterns in subdirs:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            print(f"\n{subdir}/")
            
            for pattern in patterns:
                files = list(subdir_path.glob(pattern))
                ext = pattern.replace("*", "")
                print(f"  {ext} files: {len(files)}")
                
                if files:
                    # Show first few files
                    for f in sorted(files)[:3]:
                        print(f"    {f.name}")
                    if len(files) > 3:
                        print(f"    ...")
        else:
            print(f"\n{subdir}/")
            print(f"  Not found")
    
    # Check dataset info
    info_file = data_dir / "dataset_info.json"
    if info_file.exists():
        print(f"\ndataset_info.json:")
        with open(info_file, 'r') as f:
            info = json.load(f)
        print(f"  Subsets: {list(info['subsets'].keys())}")
        for subset, subset_info in info['subsets'].items():
            print(f"  {subset}: {subset_info.get('num_samples', 'N/A')} samples")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tango Dataset Example")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/tango_dataset",
        help="Path to prepared dataset"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect dataset structure"
    )
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_dataset_structure()
    else:
        example_usage()
