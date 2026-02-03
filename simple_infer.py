"""
Simple Inference Script
Recreate audio from latent vector in 3 steps.

Usage:
    python simple_infer.py path/to/latent.npy
"""

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import sys
import shutil


def infer(latent_path, output_dir="./last_inferred"):
    """
    Recreate audio from latent vector.
    
    Args:
        latent_path: Path to .npy latent vector file
        output_dir: Where to save outputs (default: ./last_inferred)
    """
    # Configuration
    SAMPLE_RATE = 16000
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 160
    
    # Clear and create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    print(f"Loading: {latent_path}")
    
    # 1. Load latent vector
    latent = np.load(latent_path)
    
    # 2. Recreate mel-spectrogram from latent
    time_steps = int((SAMPLE_RATE * 10.0) / HOP_LENGTH)
    
    if latent.ndim == 1 and len(latent) >= N_MELS * 4:
        # Latent has statistics (avg, std, max, min)
        mel_avg = latent[:N_MELS]
        mel_std = latent[N_MELS:N_MELS*2]
        mel_max = latent[N_MELS*2:N_MELS*3]
        mel_min = latent[N_MELS*3:N_MELS*4]
        
        # Reconstruct with temporal variation
        mel_spec = np.tile(mel_avg.reshape(-1, 1), (1, time_steps))
        variation = np.random.randn(N_MELS, time_steps) * mel_std.reshape(-1, 1) * 0.1
        mel_spec += variation
        
        for i in range(N_MELS):
            mel_spec[i] = np.clip(mel_spec[i], mel_min[i], mel_max[i])
    else:
        # Simple expansion
        mel_spec = np.tile(latent[:N_MELS].reshape(-1, 1), (1, time_steps))
    
    # 3. Save mel-spectrogram as PNG
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Bin')
    plt.tight_layout()
    plt.savefig(output_path / "mel_spectrogram.png", dpi=150)
    plt.close()
    
    # 4. Save mel-spectrogram as NPY
    np.save(output_path / "mel_spectrogram.npy", mel_spec)
    
    # 5. Recreate audio using Griffin-Lim
    mel_tensor = torch.from_numpy(mel_spec).float()
    mel_tensor = torch.exp(mel_tensor) - 1e-7  # Convert from log scale
    
    # Mel to linear spectrogram
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
    )
    linear_spec = inverse_mel(mel_tensor)
    
    # Griffin-Lim reconstruction
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=32,
    )
    audio = griffin_lim(linear_spec).numpy()
    
    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-7)
    
    # 6. Save audio
    sf.write(output_path / "audio.wav", audio, SAMPLE_RATE)
    
    print(f"✓ Mel-spectrogram: {output_path / 'mel_spectrogram.png'}")
    print(f"✓ Mel-spectrogram: {output_path / 'mel_spectrogram.npy'}")
    print(f"✓ Audio: {output_path / 'audio.wav'}")
    print(f"\nDone! Results in: {output_path}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_infer.py path/to/latent.npy")
        sys.exit(1)
    
    infer(sys.argv[1])
