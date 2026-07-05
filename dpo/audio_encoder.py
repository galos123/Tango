#!/usr/bin/env python3
"""
audio_encoder.py — Encode raw waveforms into Tango-compatible VAE latents.

Only needed when a preference pair already exists as *audio* (e.g. the public
Audio-Alpaca dataset, see `load_audio_alpaca.py`). When a pair is instead
generated directly with the TTA model (see `build_preference_dataset.py`),
`AudioDiffusion.inference()` already returns latents and this module isn't
used.

Reuses the self-contained STFT/VAE implementation from
`a-latent-creation-wavcaps.py` (checkpoint download, mel-spectrogram
extraction, `AutoencoderKL`) so the latent space here is guaranteed to match
the rest of the repo instead of re-implementing the DSP a second time.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WAVCAPS_SCRIPT = _REPO_ROOT / "a-latent-creation-wavcaps.py"


def _load_wavcaps_module():
    """Import a-latent-creation-wavcaps.py by path (hyphens make it an invalid
    module name for a normal `import`) without running its `__main__` block."""
    spec = importlib.util.spec_from_file_location(
        "_wavcaps_latent_creation", _WAVCAPS_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AudioEncoder:
    """Wraps the AudioLDM VAE + STFT so raw audio can be turned into the same
    (1, 8, 256, 16) latent tensors used everywhere else in this repo."""

    def __init__(self, device: str | None = None, cache_dir: str | None = None):
        self._wc = _load_wavcaps_module()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        cache_dir = cache_dir or str(_REPO_ROOT / ".cache" / "dpo_vae")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, "audioldm-s-full.ckpt")
        self._wc.download_file(
            self._wc.AUDIO_LDM_S_FULL_URL, ckpt_path, min_bytes_ok=100_000_000
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        scale_factor = 1.0
        for k in ["scale_factor", "model.scale_factor", "latent_diffusion.scale_factor"]:
            if k in state:
                scale_factor = float(state[k].item() if torch.is_tensor(state[k]) else state[k])
                break

        prefix = "first_stage_model."
        vae_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not vae_state:
            raise RuntimeError(
                "Checkpoint does not contain 'first_stage_model.*' keys; "
                "expected an AudioLDM-style checkpoint (audioldm-s-full)."
            )
        del ckpt, state

        cfg = self._wc.CONFIG
        vae_cfg = cfg["first_stage_config"]["params"]
        self.vae = self._wc.AutoencoderKL(
            ddconfig=vae_cfg["ddconfig"],
            embed_dim=vae_cfg["embed_dim"],
            image_key=vae_cfg["image_key"],
            subband=vae_cfg["subband"],
            scale_factor=scale_factor,
        )
        self.vae.load_state_dict(vae_state, strict=False)
        self.vae = self.vae.float().to(self.device).eval()

        self.fn_STFT = self._wc.TacotronSTFT(
            cfg["preprocessing"]["stft"]["filter_length"],
            cfg["preprocessing"]["stft"]["hop_length"],
            cfg["preprocessing"]["stft"]["win_length"],
            cfg["preprocessing"]["mel"]["n_mel_channels"],
            cfg["preprocessing"]["audio"]["sampling_rate"],
            cfg["preprocessing"]["mel"]["mel_fmin"],
            cfg["preprocessing"]["mel"]["mel_fmax"],
        ).eval()
        self.target_length = cfg["preprocessing"]["mel"]["target_length"]

    def encode_file(self, wav_path: str) -> torch.Tensor:
        """wav file on disk -> latent tensor, shape (1, 8, 256, 16)."""
        mel, _, _ = self._wc.wav_to_fbank(
            wav_path, target_length=self.target_length, fn_STFT=self.fn_STFT
        )
        return self._encode_mel(mel)

    def encode_waveform(self, waveform: np.ndarray, sampling_rate: int) -> torch.Tensor:
        """In-memory waveform (e.g. from a HF `datasets` Audio feature) -> latent."""
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, waveform, sampling_rate)
            return self.encode_file(tmp.name)

    def _encode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        x = mel.unsqueeze(0).unsqueeze(0).to(self.device, dtype=vae_dtype)
        with torch.no_grad():
            posterior = self.vae.encode(x)
            z = self.vae.get_first_stage_encoding(posterior)
        return z.float().cpu()
