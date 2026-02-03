"""
Main script for Tango Dataset Preparation
Simple and practical interface for preparing the dataset.

Usage:
    python main.py
"""

from pathlib import Path
from prepare_tango_dataset_simple import SimpleTangoDatasetPreparer
from prepare_tango_dataset import TangoDatasetPreparer
from download_audiocaps import download_audiocaps


# ============================================================================
# CONFIGURATION - Edit these variables according to your needs
# ============================================================================

# Directory Configuration
ROOT_DIR = "./data"                      # Where to store raw AudioCaps data
OUTPUT_DIR = "./data/tango_dataset"      # Where to save processed dataset

# Dataset Configuration
SUBSETS = ["train"]                      # Which subsets to process: "train", "val", "test"
MAX_SAMPLES = 100                        # Max samples per subset (None for all)

# Audio Processing Configuration
SAMPLE_RATE = 16000                      # Target sample rate in Hz
N_MELS = 64                              # Number of mel frequency bins
N_FFT = 1024                             # FFT window size
HOP_LENGTH = 160                         # Hop length for STFT
MAX_DURATION = 10.0                      # Max audio duration in seconds
USE_VAE=True
# Download Configuration
DOWNLOAD_AUDIO = False                    # Whether to download audio first
N_DOWNLOAD_JOBS = 16                      # Parallel download jobs
AUDIO_FORMAT = "wav"                     # Audio format: "wav", "flac", "mp3"

# ============================================================================
# MAIN FUNCTION - Don't edit unless you know what you're doing
# ============================================================================

def main():
    """
    Main function to prepare Tango dataset.
    This function orchestrates the entire pipeline.
    """
    
    print("="*70)
    print(" " * 20 + "TANGO DATASET PREPARATION")
    print("="*70)
    print("\n📋 CONFIGURATION:")
    print(f"   Root directory:        {ROOT_DIR}")
    print(f"   Output directory:      {OUTPUT_DIR}")
    print(f"   Subsets:               {', '.join(SUBSETS)}")
    print(f"   Max samples/subset:    {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"   Sample rate:           {SAMPLE_RATE} Hz")
    print(f"   Mel bins:              {N_MELS}")
    print(f"   Download audio:        {DOWNLOAD_AUDIO}")
    print("="*70)
    
    # Step 1: Download AudioCaps (if enabled)
    if DOWNLOAD_AUDIO:
        print("\n🔽 STEP 1: DOWNLOADING AUDIOCAPS")
        print("-"*70)
        
        response = input("Download AudioCaps dataset? This may take a while. (y/n): ")
        if response.lower() == 'y':
            download_audiocaps(
                root_dir=ROOT_DIR,
                subsets=SUBSETS,
                n_jobs=N_DOWNLOAD_JOBS,
                format=AUDIO_FORMAT,
                max_samples=MAX_SAMPLES
            )
            print("✅ Download complete!")
        else:
            print("⏭️  Skipping download (make sure audio files exist)")
    else:
        print("\n⏭️  STEP 1: SKIPPING DOWNLOAD")
        print(f"   Make sure audio files exist in: {ROOT_DIR}/audiocaps/")
    
    # Step 2: Prepare Dataset
    print("\n🔧 STEP 2: PREPARING DATASET")
    print("-"*70)
    
    # Create preparer instance
    preparer = TangoDatasetPreparer(
        root_dir=ROOT_DIR,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        max_duration=MAX_DURATION,
        use_audioldm_vae=USE_VAE,
        max_samples=MAX_SAMPLES
    )
    
    # Process the dataset
    preparer.prepare_dataset(subsets=SUBSETS)
    
    # Step 3: Summary
    print("\n" + "="*70)
    print(" " * 25 + "🎉 ALL DONE!")
    print("="*70)
    print(f"\n📁 Dataset saved to: {OUTPUT_DIR}")
    print("\n📊 Dataset structure:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── mel-spectrogram/    (PNG images + NPY arrays)")
    print(f"   ├── latent-vector/      (NPY arrays)")
    print(f"   ├── original-audio/     (WAV files)")
    print(f"   ├── caption/            (TXT files)")
    print(f"   └── dataset_info.json   (Metadata)")
    print("\n💡 Next steps:")
    print("   1. Inspect dataset: python example_dataset_usage.py --inspect")
    print("   2. Load in PyTorch: from example_dataset_usage import TangoDataset")
    print("   3. Start training your Tango model!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
