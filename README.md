# Tango Dataset Preparation

This repository contains scripts to download and prepare the AudioCaps dataset for training the Tango text-to-audio model.

## Dataset Structure

The prepared dataset will have the following structure:

```
data/tango_dataset/
├── mel-spectrogram/
│   ├── 000000.png
│   ├── 000000.npy
│   ├── 000001.png
│   ├── 000001.npy
│   └── ...
├── latent-vector/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
├── original-audio/
│   ├── 000000.wav
│   ├── 000001.wav
│   └── ...
├── caption/
│   ├── 000000.txt
│   ├── 000001.txt
│   └── ...
└── dataset_info.json
```

Each sample has files with the same index (e.g., `000000`) across all directories. Mel-spectrograms are saved as both PNG images (for visualization) and NPY arrays (for numerical processing).

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Additional Tools

For downloading AudioCaps:

```bash
# Install yt-dlp for YouTube downloads (optional, if using audiocaps-download)
pip install yt-dlp

# Install ffmpeg (required for audio processing)
# On Ubuntu/Debian:
sudo apt-get install ffmpeg

# On macOS:
brew install ffmpeg

# On Windows:
# Download from https://ffmpeg.org/download.html
```

## Quick Start

For a quick test with a small subset of data:

```bash
# 1. Download AudioCaps (first 100 samples will be processed below)
python download_audiocaps.py --subsets train

# 2. Process first 100 samples for testing
python prepare_tango_dataset_simple.py \
    --subsets train \
    --max_samples 100

# 3. Verify the output
python example_dataset_usage.py --inspect
```

For full dataset preparation, see the detailed usage instructions below.

## Usage

### Option 1: Using Simplified Script (Recommended)

This script works without HuggingFace dependencies and uses locally downloaded AudioCaps data.

#### Step 1: Download AudioCaps

```bash
python download_audiocaps.py
```

Or manually using the audiocaps-download package:

```python
from audiocaps_download import Downloader

# Download train set
d = Downloader(root_path='./data/audiocaps/train/', n_jobs=16)
d.download(format='wav')

# Download validation set
d = Downloader(root_path='./data/audiocaps/val/', n_jobs=16)
d.download(format='wav')

# Download test set
d = Downloader(root_path='./data/audiocaps/test/', n_jobs=16)
d.download(format='wav')
```

#### Step 2: Prepare Dataset

```bash
# Process all subsets
python prepare_tango_dataset_simple.py \
    --root_dir ./data \
    --output_dir ./data/tango_dataset \
    --subsets train val test

# Process only training set
python prepare_tango_dataset_simple.py \
    --root_dir ./data \
    --output_dir ./data/tango_dataset \
    --subsets train

# Limit to first 1000 samples per subset (useful for testing)
python prepare_tango_dataset_simple.py \
    --root_dir ./data \
    --output_dir ./data/tango_dataset \
    --subsets train \
    --max_samples 1000
```

### Option 2: Using Full Script with VAE Encoding

This script includes AudioLDM VAE for proper latent vector encoding.

```bash
# Process all subsets with VAE encoding
python prepare_tango_dataset.py \
    --root_dir ./data \
    --output_dir ./data/tango_dataset \
    --subsets train val test

# Limit to 1000 samples for quick testing
python prepare_tango_dataset.py \
    --root_dir ./data \
    --output_dir ./data/tango_dataset \
    --subsets train \
    --max_samples 1000
```

To disable VAE encoding (use simple dimensionality reduction):

```bash
python prepare_tango_dataset.py \
    --root_dir ./data \
    --output_dir ./data/tango_dataset \
    --subsets train \
    --no_vae
```

## Script Parameters

### Common Parameters

- `--root_dir`: Root directory for raw data (default: `./data`)
- `--output_dir`: Output directory for processed dataset (default: `./data/tango_dataset`)
- `--subsets`: Dataset subsets to process (default: `train val test`)
- `--sample_rate`: Target sample rate in Hz (default: `16000`)
- `--n_mels`: Number of mel frequency bins (default: `64`)
- `--max_samples`: Maximum number of samples to process per subset (default: `None` for all samples)

### Full Script Additional Parameters

- `--no_vae`: Disable AudioLDM VAE encoding (uses simple dimensionality reduction instead)

## Output Files

### 1. Mel-Spectrogram
- **PNG image** (`mel-spectrogram/*.png`): Visual representation of the mel-spectrogram
- **NumPy array** (`mel-spectrogram/*.npy`): Numerical data for model training
  - Shape: `(n_mels, time_steps)`
  - Log-scaled mel-spectrogram
  - Used as input to the diffusion model

### 2. Latent Vector (`latent-vector/*.npy`)
- Compressed representation of the audio
- Encoded using AudioLDM VAE (full script) or simple compression (simplified script)
- Used in the latent diffusion process

### 3. Original Audio (`original-audio/*.wav`)
- Sample rate: 16kHz (default)
- Duration: 10 seconds (padded or trimmed)
- Mono channel
- Used for reference and evaluation

### 4. Caption (`caption/*.txt`)
- Text description of the audio
- Used as conditioning input to the model

### 5. Dataset Info (`dataset_info.json`)
Contains metadata about the dataset:
```json
{
  "sample_rate": 16000,
  "n_mels": 64,
  "n_fft": 1024,
  "hop_length": 160,
  "max_duration": 10.0,
  "subsets": {
    "train": {
      "indices": [0, 1, 2, ...],
      "num_samples": 45000
    },
    ...
  }
}
```

## Dataset Statistics (AudioCaps)

According to the original AudioCaps paper:
- **Training set**: ~49,000 samples (1 caption each)
- **Validation set**: ~495 samples (5 captions each)
- **Test set**: ~957 samples (5 captions each)

Note: Some videos may no longer be available on YouTube, so actual numbers may be lower.

## Loading the Dataset

Example code to load the prepared dataset:

```python
import numpy as np
import json
from pathlib import Path

class TangoDataset:
    def __init__(self, data_dir="./data/tango_dataset"):
        self.data_dir = Path(data_dir)
        
        # Load dataset info
        with open(self.data_dir / "dataset_info.json") as f:
            self.info = json.load(f)
        
        # Get all indices
        self.indices = []
        for subset_info in self.info['subsets'].values():
            self.indices.extend(subset_info['indices'])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        
        # Load mel-spectrogram
        mel_spec = np.load(
            self.data_dir / "mel-spectrogram" / f"{sample_idx:06d}.npy"
        )
        
        # Load latent vector
        latent = np.load(
            self.data_dir / "latent-vector" / f"{sample_idx:06d}.npy"
        )
        
        # Load caption
        with open(self.data_dir / "caption" / f"{sample_idx:06d}.txt") as f:
            caption = f.read()
        
        # Optionally load audio
        # audio, sr = sf.read(
        #     self.data_dir / "original-audio" / f"{sample_idx:06d}.wav"
        # )
        
        return {
            'mel_spectrogram': mel_spec,
            'latent_vector': latent,
            'caption': caption,
            'index': sample_idx
        }

# Usage
dataset = TangoDataset("./data/tango_dataset")
sample = dataset[0]
print(f"Caption: {sample['caption']}")
print(f"Mel-spec shape: {sample['mel_spectrogram'].shape}")
print(f"Latent shape: {sample['latent_vector'].shape}")
```

## Troubleshooting

### Issue: "Audio files not found"

The AudioCaps audio files need to be downloaded separately:

```python
from audiocaps_download import Downloader
d = Downloader(root_path='./data/audiocaps/train/', n_jobs=16)
d.download(format='wav')
```

### Issue: "YouTube download blocked"

YouTube may block automated downloads. Solutions:
1. Use cookies from your browser:
   ```python
   # When using aac-datasets
   from aac_datasets import AudioCaps
   dataset = AudioCaps(ytdlp_opts=["--cookies-from-browser", "firefox"])
   ```

2. Download in smaller batches
3. Use a VPN or different IP address

### Issue: "Out of memory"

Process subsets separately:
```bash
python prepare_tango_dataset_simple.py --subsets train
python prepare_tango_dataset_simple.py --subsets val
python prepare_tango_dataset_simple.py --subsets test
```

### Issue: "VAE loading failed"

Use the simplified script or disable VAE:
```bash
python prepare_tango_dataset.py --no_vae
```

## References

- **Tango Paper**: [Text-to-Audio Generation using Instruction Tuned LLM and Latent Diffusion Model](https://arxiv.org/abs/2304.13731)
- **Tango GitHub**: https://github.com/declare-lab/tango
- **AudioCaps Dataset**: https://audiocaps.github.io/
- **AudioCaps Paper**: [AudioCaps: Generating Captions for Audios in the Wild](https://arxiv.org/abs/1904.06355)

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{ghosal2023tango,
  title={Text-to-Audio Generation using Instruction Tuned LLM and Latent Diffusion Model},
  author={Ghosal, Deepanway and Majumder, Navonil and Mehrish, Ambuj and Poria, Soujanya},
  journal={arXiv preprint arXiv:2304.13731},
  year={2023}
}

@inproceedings{kim2019audiocaps,
  title={Audiocaps: Generating captions for audios in the wild},
  author={Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle={NAACL-HLT},
  year={2019}
}
```

## License

This code is provided for research purposes. Please respect the licenses of the original datasets and models:
- AudioCaps: See https://audiocaps.github.io/
- Tango: MIT License
- AudioLDM: See https://github.com/haoheliu/AudioLDM