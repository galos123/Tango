"""
VAE-Based Inference Script
Uses AudioLDM VAE for proper encoding/decoding of latent vectors.

Usage:
    python vae_infer.py path/to/latent.npy
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import sys
import shutil


def load_audioldm_vae():
    """Load the AudioLDM VAE model."""
    try:
        # Try importing audioldm modules
        from audioldm.variational_autoencoder.autoencoder import AutoencoderKL
        from audioldm.utils import default_audioldm_config
        
        print("Loading AudioLDM VAE...")
        
        # Get default config
        config = default_audioldm_config()
        
        # Initialize VAE
        vae = AutoencoderKL(
            ddconfig=config["model"]["params"]["first_stage_config"]["params"]["ddconfig"],
            embed_dim=config["model"]["params"]["first_stage_config"]["params"]["embed_dim"]
        )
        
        # Load checkpoint
        import os
        ckpt_path = os.path.expanduser("~/.cache/audioldm/audioldm-s-full.ckpt")
        
        if not os.path.exists(ckpt_path):
            print("Downloading AudioLDM checkpoint...")
            import urllib.request
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            url = "https://zenodo.org/record/7600541/files/audioldm-s-full?download=1"
            urllib.request.urlretrieve(url, ckpt_path)
        
        # Load weights
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            vae_state_dict = {
                k.replace("first_stage_model.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("first_stage_model.")
            }
            vae.load_state_dict(vae_state_dict)
        
        vae.eval()
        print("✓ AudioLDM VAE loaded successfully!")
        return vae
        
    except Exception as e:
        print(f"❌ Error loading AudioLDM VAE: {e}")
        print("\nPlease install audioldm:")
        print("  pip install audioldm")
        return None


def latent_to_mel_vae(latent: np.ndarray, vae) -> np.ndarray:
    """
    Decode latent vector to mel-spectrogram using VAE.
    
    Args:
        latent: Latent vector from VAE encoder
        vae: AudioLDM VAE model
        
    Returns:
        Mel-spectrogram as numpy array
    """
    if vae is None:
        raise ValueError("VAE not loaded. Cannot decode latent.")
    
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        latent_tensor = torch.from_numpy(latent).float()
        
        # Ensure proper shape for VAE decoder
        if latent_tensor.dim() == 1:
            # If 1D, need to reshape to (batch, channels, height, width)
            # This depends on your VAE's latent structure
            raise ValueError("Latent vector shape not compatible. Need 4D tensor (B, C, H, W)")
        elif latent_tensor.dim() == 2:
            # Assume (H, W), add batch and channel dims
            latent_tensor = latent_tensor.unsqueeze(0).unsqueeze(0)
        elif latent_tensor.dim() == 3:
            # Assume (C, H, W), add batch dim
            latent_tensor = latent_tensor.unsqueeze(0)
        
        # Decode using VAE
        mel_spec = vae.decode(latent_tensor)
        
        # Remove batch dimension and convert to numpy
        mel_spec_np = mel_spec.squeeze(0).squeeze(0).cpu().numpy()
    
    return mel_spec_np


def mel_to_audio_vocoder(mel_spec: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Convert mel-spectrogram to audio using vocoder.
    For best results, use HiFi-GAN or another neural vocoder.
    Falls back to Griffin-Lim if vocoder not available.
    """
    try:
        # Try using AudioLDM's vocoder
        from audioldm.hifigan.utilities import get_vocoder
        
        vocoder = get_vocoder(None, "cpu")
        vocoder.eval()
        
        with torch.no_grad():
            mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
            audio = vocoder(mel_tensor)
            audio_np = audio.squeeze().cpu().numpy()
        
        print("✓ Using HiFi-GAN vocoder")
        return audio_np
        
    except Exception as e:
        print(f"⚠️  Vocoder not available, using Griffin-Lim: {e}")
        return mel_to_audio_griffinlim(mel_spec, sample_rate)


def mel_to_audio_griffinlim(mel_spec: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Fallback: Convert mel to audio using Griffin-Lim."""
    import torchaudio
    
    N_FFT = 1024
    HOP_LENGTH = 160
    N_MELS = mel_spec.shape[0]
    
    mel_tensor = torch.from_numpy(mel_spec).float()
    mel_tensor = torch.exp(mel_tensor) - 1e-7
    
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=sample_rate,
    )
    linear_spec = inverse_mel(mel_tensor)
    
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=32,
    )
    audio = griffin_lim(linear_spec).numpy()
    audio = audio / (np.abs(audio).max() + 1e-7)
    
    return audio


def infer(latent_path: str, output_dir: str = "./last_inferred"):
    """
    Reconstruct audio from VAE latent vector.
    
    Args:
        latent_path: Path to latent vector .npy file
        output_dir: Output directory (default: ./last_inferred)
    """
    print("="*70)
    print(" " * 15 + "VAE LATENT VECTOR INFERENCE")
    print("="*70)
    
    # Setup output directory
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"\n🗑️  Clearing: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    # Load VAE
    print("\n📥 STEP 1: LOADING VAE MODEL")
    vae = load_audioldm_vae()
    if vae is None:
        print("❌ Cannot proceed without VAE")
        return
    
    # Load latent vector
    print(f"\n📥 STEP 2: LOADING LATENT VECTOR")
    print(f"   File: {latent_path}")
    latent = np.load(latent_path)
    print(f"   Shape: {latent.shape}")
    
    # Decode latent to mel-spectrogram
    print(f"\n🔄 STEP 3: DECODING LATENT → MEL-SPECTROGRAM")
    try:
        mel_spec = latent_to_mel_vae(latent, vae)
        print(f"   Mel-spec shape: {mel_spec.shape}")
    except Exception as e:
        print(f"❌ Error decoding latent: {e}")
        return
    
    # Save mel-spectrogram
    print(f"\n💾 STEP 4: SAVING MEL-SPECTROGRAM")
    
    # Save as PNG
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('Decoded Mel-Spectrogram (from VAE)')
    plt.xlabel('Time')
    plt.ylabel('Mel Bin')
    plt.tight_layout()
    plt.savefig(output_path / "mel_spectrogram.png", dpi=150)
    plt.close()
    print(f"   ✓ PNG: {output_path / 'mel_spectrogram.png'}")
    
    # Save as NPY
    np.save(output_path / "mel_spectrogram.npy", mel_spec)
    print(f"   ✓ NPY: {output_path / 'mel_spectrogram.npy'}")
    
    # Convert mel to audio
    print(f"\n🔊 STEP 5: CONVERTING MEL → AUDIO")
    audio = mel_to_audio_vocoder(mel_spec)
    
    # Save audio
    sf.write(output_path / "audio.wav", audio, 16000)
    duration = len(audio) / 16000
    print(f"   ✓ Audio: {output_path / 'audio.wav'} ({duration:.2f}s)")
    
    # Summary
    print("\n" + "="*70)
    print(" " * 25 + "✅ INFERENCE COMPLETE!")
    print("="*70)
    print(f"\n📂 Results saved to: {output_path}/")
    print(f"   ├── mel_spectrogram.png")
    print(f"   ├── mel_spectrogram.npy")
    print(f"   └── audio.wav")
    print("="*70 + "\n")


if __name__ == "__main__":
    infer("/home/yitshag/test_uv/data/tango_dataset/latent-vector/000000.npy")
