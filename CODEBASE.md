# Tango Codebase Explanation

## Overview

Tango is an audio generation project built on **AudioLDM** (Audio Latent Diffusion Model). It implements a pipeline for generating audio from text descriptions using diffusion models conditioned on text embeddings. The project focuses on:

1. **Dataset preparation** - Converting AudioCaps/WavCaps audio datasets into training-ready formats
2. **Latent encoding** - Compressing mel-spectrograms into latent vectors using a VAE
3. **Diffusion-based audio generation** - Using a U-Net denoiser conditioned on text embeddings to generate audio

---

## Architecture

The audio generation pipeline follows this flow:

```
Text prompt
    |
    v
Text Encoder (CLIP / T5)
    |
    v
U-Net Diffusion Model (1000 timesteps, DDPM scheduler)
    |
    v
Latent vector [batch, 8, 16, 250]
    |
    v
VAE Decoder --> Mel-spectrogram [batch, 1, 64, 1000]
    |
    v
HiFi-GAN Vocoder --> Waveform (16 kHz)
```

### Key Parameters

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Mel channels | 64 |
| STFT filter length | 1,024 |
| STFT hop length | 160 |
| Mel frequency range | 0 - 8,000 Hz |
| VAE latent channels | 8 |
| VAE embed dim | 8 |
| Diffusion timesteps | 1,000 |
| Beta schedule | scaled_linear (0.00085 - 0.012) |

---

## File-by-File Breakdown

### Core Scripts

#### `a-full_tango.py`
The complete self-contained pipeline (~1,000 lines). Implements everything from scratch without relying on the `audioldm` package at runtime:

- **Checkpoint download** from Zenodo (`audioldm-s-full`, ~2 GB)
- **STFT / Mel-spectrogram** computation (`TacotronSTFT` class)
- **VAE** (`AutoencoderKL`) for encoding mel-spectrograms into latent vectors and decoding them back
- **HiFi-GAN vocoder** for converting mel-spectrograms to waveforms
- Audio loading, padding to target length (1,024 frames), and normalization

#### `a-latent-creation.py`
Similar to `a-full_tango.py` but focused specifically on the latent encoding step. Processes audio files through the VAE encoder to produce latent vectors for training.

#### `a-latent-creation-wavcaps.py`
Extends the latent creation pipeline for the **WavCaps / AudioSet-SL** dataset format. Takes command-line arguments:

```bash
python a-latent-creation-wavcaps.py \
    --audio_dir ./original_data/AudioSet_SL \
    --json_dir  ./original_data/json_files/AudioSet_SL \
    --output_dir ./new_dataset
```

Produces:
```
new_dataset/
  ├── captions/           (.txt files)
  ├── latent_vectors/     (.pt tensors)
  ├── mel_spectrograms/   (.png images)
  └── original_wavs/      (symlinks to source .wav files)
```

#### `prepare_tango_dataset_simple.py`
`SimpleTangoDatasetPreparer` class - a simplified dataset builder that does **not** depend on HuggingFace libraries. Handles:

- Downloading AudioCaps metadata CSVs from GitHub
- Computing mel-spectrograms via `torchaudio.transforms.MelSpectrogram`
- Saving mel-spectrograms (as `.npy` and `.png`), captions (`.txt`), and audio (`.wav`)
- Generating a `dataset_info.json` manifest with train/val/test split indices

#### `example_dataset_usage.py`
`TangoDataset` - a PyTorch `Dataset` class for loading prepared data. Supports:

- Loading mel-spectrograms from `.npy` or `.png` files
- Loading latent vectors (`.pt` tensors)
- Loading text captions
- Optional audio waveform loading
- Train/val/test subset selection via `dataset_info.json`
- Custom `collate_fn` for batching variable-length samples

#### `download_audiocaps.py`
Wrapper around the `audiocaps-download` package for parallel downloading of AudioCaps audio from YouTube. Supports configurable subset selection, parallel jobs, and audio format.

#### `a-new-data-download.py`
Downloads **WavCaps AudioSet-SL** data from HuggingFace (`cvssp/WavCaps`):

1. Downloads multi-part zip archives and JSON metadata with retry logic
2. Extracts archives using `7z`
3. Transcodes FLAC to WAV (16 kHz mono) using `ffmpeg`
4. Cleans up temporary files

#### `vae_infer.py`
Standalone VAE inference script. Loads a latent vector from disk, decodes it through the VAE, de-normalizes (mean=-4.63, std=2.74), runs the HiFi-GAN vocoder, and saves the resulting waveform.

### Notebooks

| Notebook | Purpose |
|---|---|
| `a-run_model.ipynb` | Interactive model execution and experimentation |
| `a-latent-creation.ipynb` | Interactive latent vector creation |
| `z-tests.ipynb` | Testing and validation |

### Reference Implementations (`original_files/`)

These are reference files from the original Tango/AudioLDM project:

- **`models.py`** - `AudioDiffusion` class: the full diffusion model combining text encoder (CLIP/T5), U-Net denoiser (`UNet2DConditionModel`), DDPM scheduler, and VAE. Implements `build_pretrained_models()` for loading checkpoints.
- **`autoencoder.py`** - Full VAE architecture implementation
- **`utils.py`** - Utilities including `default_audioldm_config()`, `get_metadata()`, checkpoint caching, audio saving, and seed management
- **`diffusion_model_config.json`** - U-Net model configuration

### Configuration

- **`local_config/scheduler/scheduler_config.json`** - DDPM scheduler config (1000 timesteps, scaled_linear beta schedule, v_prediction)
- **`pyproject.toml`** - Project dependencies managed with `uv`
- **`.python-version`** - Python 3.10

---

## Dataset Format

After preparation, the training dataset has this structure:

```
tango_dataset/
  ├── mel-spectrogram/     000000.npy, 000000.png, ...
  ├── latent-vector/       000000.npy (or .pt), ...
  ├── original-audio/      000000.wav, ...
  ├── caption/             000000.txt, ...
  └── dataset_info.json    (manifest with split indices and metadata)
```

Each sample consists of:
- A **mel-spectrogram** (64 mel bins x 1000 frames) as numpy array and/or PNG
- A **latent vector** (8 x 16 x 250 tensor) produced by the VAE encoder
- The **original audio** (16 kHz WAV, up to 10 seconds)
- A **text caption** describing the audio content

---

## Dependencies

Key dependency groups (managed via `pyproject.toml` with `uv`):

| Category | Packages |
|---|---|
| Deep learning | `torch`, `torchaudio`, `torchvision`, `pytorch-lightning` |
| Diffusion | `diffusers`, `transformers`, `accelerate` |
| Audio processing | `librosa`, `soundfile`, `pydub`, `resampy`, `torchlibrosa` |
| Data handling | `datasets`, `pandas`, `numpy`, `scipy`, `h5py` |
| Visualization | `matplotlib`, `pillow` |
| Experiment tracking | `wandb` |
| Configuration | `omegaconf` |

---

## How to Use

### 1. Download audio data

For AudioCaps:
```bash
python download_audiocaps.py --subsets train val test
```

For WavCaps/AudioSet-SL:
```bash
python a-new-data-download.py
```

### 2. Prepare the dataset

Using the simplified preparer (no HuggingFace):
```bash
python prepare_tango_dataset_simple.py --root_dir ./data --output_dir ./data/tango_dataset
```

Or create latent vectors for WavCaps data:
```bash
python a-latent-creation-wavcaps.py \
    --audio_dir ./original_data/AudioSet_SL \
    --json_dir  ./original_data/json_files/AudioSet_SL \
    --output_dir ./new_dataset
```

### 3. Load the dataset for training

```python
from example_dataset_usage import TangoDataset
dataset = TangoDataset(data_dir="./data/tango_dataset", subset="train")
```

### 4. Run VAE inference

```bash
python vae_infer.py
```
