"""
Inference Script - Recreate Audio from Latent Vector
Takes a latent vector file and reconstructs:
1. Mel-spectrogram
2. Audio waveform

Output saved to: ./last_inferred/

Usage:
    python infer_from_latent.py /path/to/latent_vector.npy
"""

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import argparse
import shutil


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "./last_inferred"           # Output folder
SAMPLE_RATE = 16000                      # Audio sample rate
N_MELS = 64                              # Number of mel bins
N_FFT = 1024                             # FFT window size
HOP_LENGTH = 160                         # Hop length

# Griffin-Lim parameters for audio reconstruction
GRIFFIN_LIM_ITERS = 32                   # Number of iterations


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_latent_vector(latent_path: str) -> np.ndarray:
    """Load latent vector from file."""
    latent = np.load(latent_path)
    print(f"✓ Loaded latent vector: shape {latent.shape}")
    return latent


def latent_to_mel_spectrogram(latent: np.ndarray) -> np.ndarray:
    """
    Convert latent vector back to mel-spectrogram.
    
    Note: This is a simplified reconstruction. For proper reconstruction,
    you would use the AudioLDM VAE decoder. This expands the latent
    back to mel-spectrogram dimensions.
    """
    # Simple expansion - repeat latent features across time dimension
    # In practice, you'd use a trained VAE decoder here
    
    # Assuming latent has shape (n_features,)
    # We need to expand it to (n_mels, time_steps)
    
    if latent.ndim == 1:
        # Extract mel bins from latent (first N_MELS elements)
        if len(latent) >= N_MELS * 4:
            # Latent contains: avg, std, max, min
            mel_avg = latent[:N_MELS]
            mel_std = latent[N_MELS:N_MELS*2]
            mel_max = latent[N_MELS*2:N_MELS*3]
            mel_min = latent[N_MELS*3:N_MELS*4]
            
            # Reconstruct approximate mel-spectrogram
            # Create time dimension (10 seconds at 16kHz with hop_length=160)
            time_steps = int((SAMPLE_RATE * 10.0) / HOP_LENGTH)
            
            # Initialize with average values
            mel_spec = np.tile(mel_avg.reshape(-1, 1), (1, time_steps))
            
            # Add some temporal variation using std
            variation = np.random.randn(N_MELS, time_steps) * mel_std.reshape(-1, 1) * 0.1
            mel_spec += variation
            
            # Clip to reasonable range based on min/max
            for i in range(N_MELS):
                mel_spec[i] = np.clip(mel_spec[i], mel_min[i], mel_max[i])
        else:
            # Simple case: expand latent to mel-spectrogram
            time_steps = int((SAMPLE_RATE * 10.0) / HOP_LENGTH)
            mel_spec = np.tile(latent[:N_MELS].reshape(-1, 1), (1, time_steps))
    else:
        # If already 2D, assume it's a mel-spectrogram
        mel_spec = latent
    
    print(f"✓ Reconstructed mel-spectrogram: shape {mel_spec.shape}")
    return mel_spec


def mel_to_audio(mel_spec: np.ndarray) -> np.ndarray:
    """
    Convert mel-spectrogram to audio using Griffin-Lim algorithm.
    """
    # Convert to torch tensor
    mel_spec_torch = torch.from_numpy(mel_spec).float()
    
    # Convert from log scale back to linear
    mel_spec_torch = torch.exp(mel_spec_torch) - 1e-7
    
    # Initialize inverse mel-spectrogram transform
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
    )
    
    # Convert mel to linear spectrogram
    linear_spec = inverse_mel(mel_spec_torch)
    
    # Initialize Griffin-Lim
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=GRIFFIN_LIM_ITERS,
    )
    
    # Reconstruct audio
    audio = griffin_lim(linear_spec)
    audio_np = audio.numpy()
    
    # Normalize audio to [-1, 1]
    audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
    
    print(f"✓ Reconstructed audio: {len(audio_np)} samples ({len(audio_np)/SAMPLE_RATE:.2f}s)")
    return audio_np


def save_mel_spectrogram_image(mel_spec: np.ndarray, save_path: Path):
    """Save mel-spectrogram as PNG image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(
        mel_spec,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel Frequency Bin')
    ax.set_title('Reconstructed Mel-Spectrogram')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved mel-spectrogram image: {save_path}")


def save_audio(audio: np.ndarray, save_path: Path):
    """Save audio as WAV file."""
    sf.write(save_path, audio, SAMPLE_RATE)
    print(f"✓ Saved audio: {save_path}")


def save_mel_spectrogram_npy(mel_spec: np.ndarray, save_path: Path):
    """Save mel-spectrogram as NPY file."""
    np.save(save_path, mel_spec)
    print(f"✓ Saved mel-spectrogram array: {save_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def infer(latent_path: str):
    """
    Main inference function.
    
    Args:
        latent_path: Path to latent vector .npy file
    """
    print("="*70)
    print(" " * 15 + "TANGO LATENT VECTOR INFERENCE")
    print("="*70)
    
    # Setup output directory
    output_dir = Path(OUTPUT_DIR)
    if output_dir.exists():
        print(f"\n🗑️  Removing old results from: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Load latent vector
    print(f"\n📥 STEP 1: LOADING LATENT VECTOR")
    print(f"   File: {latent_path}")
    latent = load_latent_vector(latent_path)
    
    # Step 2: Reconstruct mel-spectrogram
    print(f"\n🔄 STEP 2: RECONSTRUCTING MEL-SPECTROGRAM")
    mel_spec = latent_to_mel_spectrogram(latent)
    
    # Step 3: Reconstruct audio
    print(f"\n🔊 STEP 3: RECONSTRUCTING AUDIO")
    audio = mel_to_audio(mel_spec)
    
    # Step 4: Save outputs
    print(f"\n💾 STEP 4: SAVING OUTPUTS")
    save_mel_spectrogram_image(mel_spec, output_dir / "mel_spectrogram.png")
    save_mel_spectrogram_npy(mel_spec, output_dir / "mel_spectrogram.npy")
    save_audio(audio, output_dir / "reconstructed_audio.wav")
    
    # Summary
    print("\n" + "="*70)
    print(" " * 25 + "✅ INFERENCE COMPLETE!")
    print("="*70)
    print(f"\n📂 Results saved to: {output_dir}/")
    print(f"   ├── mel_spectrogram.png      (Visualization)")
    print(f"   ├── mel_spectrogram.npy      (NumPy array)")
    print(f"   └── reconstructed_audio.wav  (Audio file)")
    print("\n💡 Listen to the audio:")
    print(f"   Play: {output_dir / 'reconstructed_audio.wav'}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct audio from latent vector"
    )
    parser.add_argument(
        "latent_path",
        type=str,
        help="Path to latent vector .npy file"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.latent_path).exists():
        print(f"❌ Error: File not found: {args.latent_path}")
        return
    
    # Run inference
    infer(args.latent_path)


if __name__ == "__main__":
    main()
