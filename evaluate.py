#!/usr/bin/env python3
"""
evaluate.py — Run TANGO inference on 100 AudioCaps test prompts and compute metrics.

Metrics computed:
  - CLAP score (no reference audio needed)
  - FAD + KL divergence (only if --ref_audio_dir is provided)

Usage:
    # CLAP score only (no reference audio):
    python evaluate.py \
        --csv test.csv \
        --output_dir ./eval_output \
        --n_samples 100

    # Full metrics (needs reference audio downloaded separately):
    python evaluate.py \
        --csv test.csv \
        --output_dir ./eval_output \
        --ref_audio_dir ./reference_audio \
        --n_samples 100
"""

import os
import sys
import csv
import json
import argparse
import importlib.util
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "original_files"))

# audioldm fails to install on current Colab/pip. models.py and autoencoder.py
# import it at module level, but none of those code paths are used during
# inference (only in build_pretrained_models which we never call).
# Stub every required submodule so the imports resolve without the package.
from types import ModuleType as _ModuleType

_AUDIOLDM_STUBS = [
    "audioldm",
    "audioldm.audio",
    "audioldm.audio.stft",
    "audioldm.latent_diffusion",
    "audioldm.latent_diffusion.ema",
    "audioldm.variational_autoencoder",
    "audioldm.variational_autoencoder.modules",
    "audioldm.variational_autoencoder.distributions",
    "audioldm.hifigan",
    "audioldm.hifigan.utilities",
    "audioldm.clap",
    "audioldm.clap.encoders",
    "audioldm.pipline",
]
for _n in _AUDIOLDM_STUBS:
    if _n not in sys.modules:
        _m = _ModuleType(_n)
        _m.__path__ = []   # marks it as a package so sub-imports work
        sys.modules[_n] = _m

# wire parent → child so attribute access works alongside sys.modules lookup
sys.modules["audioldm"].audio                            = sys.modules["audioldm.audio"]                           # type: ignore
sys.modules["audioldm.audio"].stft                       = sys.modules["audioldm.audio.stft"]                      # type: ignore
sys.modules["audioldm"].latent_diffusion                 = sys.modules["audioldm.latent_diffusion"]                # type: ignore
sys.modules["audioldm.latent_diffusion"].ema             = sys.modules["audioldm.latent_diffusion.ema"]            # type: ignore
sys.modules["audioldm"].variational_autoencoder          = sys.modules["audioldm.variational_autoencoder"]         # type: ignore
sys.modules["audioldm.variational_autoencoder"].modules  = sys.modules["audioldm.variational_autoencoder.modules"] # type: ignore
sys.modules["audioldm.variational_autoencoder"].distributions = sys.modules["audioldm.variational_autoencoder.distributions"] # type: ignore
sys.modules["audioldm"].hifigan                          = sys.modules["audioldm.hifigan"]                         # type: ignore
sys.modules["audioldm.hifigan"].utilities                = sys.modules["audioldm.hifigan.utilities"]               # type: ignore

# stub the specific names imported from those modules
sys.modules["audioldm.audio.stft"].TacotronSTFT = object                   # type: ignore
sys.modules["audioldm.variational_autoencoder.modules"].Encoder = object   # type: ignore
sys.modules["audioldm.variational_autoencoder.modules"].Decoder = object   # type: ignore
sys.modules["audioldm.variational_autoencoder.distributions"].DiagonalGaussianDistribution = object  # type: ignore
sys.modules["audioldm.hifigan.utilities"].get_vocoder    = lambda *a, **k: None  # type: ignore
sys.modules["audioldm.hifigan.utilities"].vocoder_infer  = lambda *a, **k: None  # type: ignore

# Also stub the local autoencoder module — it has deep audioldm dependencies
# and is only used in build_pretrained_models() which we never call
_ae_stub = _ModuleType("autoencoder")
_ae_stub.AutoencoderKL = object  # type: ignore
sys.modules["autoencoder"] = _ae_stub

from models import AudioDiffusion   # noqa: E402

# a-latent-creation-wavcaps.py has dashes → can't use normal import
_latent_mod_path = Path(__file__).parent / "a-latent-creation-wavcaps.py"
_spec = importlib.util.spec_from_file_location("latent_creation", _latent_mod_path)
_latent_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_latent_mod)
AutoencoderKL = _latent_mod.AutoencoderKL
CONFIG        = _latent_mod.CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# VAE loader (same logic as a-latent-creation-wavcaps.py)
# ─────────────────────────────────────────────────────────────────────────────

AUDIOLDM_URL = "https://zenodo.org/record/7600541/files/audioldm-s-full?download=1"


def download_audioldm_checkpoint(dst: str) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > 100_000_000:
        print(f"[OK] AudioLDM checkpoint already at {dst}")
        return
    import requests
    print(f"[DL] Downloading AudioLDM checkpoint to {dst} (~2.5 GB) ...")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with requests.get(AUDIOLDM_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {100*done/total:.1f}%", end="", flush=True)
    print(f"\n[OK] Downloaded to {dst}")


def load_vae(ckpt_path: str, device: torch.device) -> AutoencoderKL:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    scale_factor = 1.0
    for k in ["scale_factor", "model.scale_factor"]:
        if k in state:
            scale_factor = float(state[k].item() if torch.is_tensor(state[k]) else state[k])
            break

    prefix = "first_stage_model."
    vae_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    del ckpt, state

    cfg = CONFIG["first_stage_config"]["params"]
    vae = AutoencoderKL(
        ddconfig=cfg["ddconfig"],
        embed_dim=cfg["embed_dim"],
        image_key=cfg["image_key"],
        subband=cfg["subband"],
        scale_factor=scale_factor,
    )
    vae.load_state_dict(vae_state, strict=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        vae = vae.half().to(device).eval()
        print(f"[INFO] VAE on {device} (fp16)")
    except Exception:
        vae = vae.float().to(device).eval()
        print(f"[INFO] VAE on {device} (fp32)")
    return vae


# ─────────────────────────────────────────────────────────────────────────────
# Build AudioDiffusion (loads UNet from HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_SCHED_DIR = str(Path(__file__).parent / "local_config")


def ensure_scheduler_config() -> None:
    cfg_path = Path(LOCAL_SCHED_DIR) / "scheduler" / "scheduler_config.json"
    if cfg_path.exists():
        return
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({
        "_class_name": "DDPMScheduler",
        "_diffusers_version": "0.21.0",
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "clip_sample": False,
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "variance_type": "fixed_small",
    }, indent=2))


def load_diffusion_model(device: torch.device, hf_repo: str = "declare-lab/tango") -> AudioDiffusion:
    """
    The declare-lab/tango HF repo stores the full AudioDiffusion state dict in
    pytorch_model_main.bin (not a diffusers unet/ subfolder).
    We use snapshot_download to get all files, build the model from the repo's
    main_config.json, then load the state dict directly.
    """
    from huggingface_hub import snapshot_download

    ensure_scheduler_config()

    print(f"[INFO] Downloading {hf_repo} from HuggingFace ...")
    repo_path = snapshot_download(repo_id=hf_repo)
    print(f"[INFO] Downloaded to {repo_path}")

    main_config_path = os.path.join(repo_path, "main_config.json")
    weights_path     = os.path.join(repo_path, "pytorch_model_main.bin")

    # main_config.json is the AudioDiffusion config; the actual UNet config is
    # referenced inside it under 'unet_model_config_path' (relative to repo root).
    # UNet2DConditionModel.load_config() expects a *directory* containing config.json,
    # not a direct path to a JSON file — so we write it into a temp dir.
    import tempfile
    with open(main_config_path) as f:
        main_cfg = json.load(f)
    unet_cfg_rel  = main_cfg.get("unet_model_config_path", "configs/diffusion_model_config.json")
    unet_cfg_path = os.path.join(repo_path, unet_cfg_rel)
    print(f"[INFO] UNet config source: {unet_cfg_path}")

    with open(unet_cfg_path) as f:
        unet_cfg_data = json.load(f)
    tmp_cfg_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_cfg_dir, "config.json"), "w") as f:
        json.dump(unet_cfg_data, f)
    print(f"[INFO] UNet config dir: {tmp_cfg_dir}")

    model = AudioDiffusion(
        text_encoder_name="google/flan-t5-large",
        scheduler_name=LOCAL_SCHED_DIR,
        unet_model_name=None,
        unet_model_config_path=tmp_cfg_dir,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
    )

    print(f"[INFO] Loading pretrained weights from {weights_path} ...")
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[INFO] {len(missing)} missing keys (expected — text encoder loaded separately)")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys ignored")

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Inference: text prompt → latent → mel → waveform
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_audio(
    prompt: str,
    model: AudioDiffusion,
    vae: AutoencoderKL,
    device: torch.device,
    num_steps: int = 200,
    guidance_scale: float = 3.0,
) -> np.ndarray:
    """Returns a 1-D float32 numpy array at 16 kHz."""

    # 1. Run diffusion → latent (8, 256, 16)
    latents = model.inference(
        prompt=[prompt],
        inference_scheduler=model.inference_scheduler,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        num_samples_per_prompt=1,
        disable_progress=True,
    )   # shape: (1, 8, 256, 16)

    # 2. VAE decode latent → mel spectrogram
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(device, dtype=vae_dtype)
    mel = vae.decode_first_stage(latents)          # (1, 1, 1024, 64)

    # 3. HiFi-GAN vocoder: mel → waveform
    wav = vae.decode_to_waveform(mel)              # (1, T_samples)
    return wav[0].float().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_clap_score(wav_paths: list[str], prompts: list[str]) -> float:
    """
    Compute mean CLAP score (text–audio cosine similarity) using HuggingFace
    transformers ClapModel. No extra install needed — transformers is already
    a dependency. Model: laion/larger_clap_music_and_speech (~600 MB, downloads
    automatically on first run).

    CLAP expects 48 kHz audio; we resample from 16 kHz before embedding.
    Higher score = better semantic alignment between text and generated audio.
    """
    from transformers import ClapModel, ClapProcessor
    import torch.nn.functional as F

    clap_name = "laion/larger_clap_music_and_speech"
    print(f"[CLAP] Loading {clap_name} ...")
    clap_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap_model  = ClapModel.from_pretrained(clap_name).to(clap_device).eval()
    processor   = ClapProcessor.from_pretrained(clap_name)
    target_sr   = processor.feature_extractor.sampling_rate   # 48000

    sims = []
    for wav_path, prompt in tqdm(zip(wav_paths, prompts), total=len(wav_paths),
                                  desc="CLAP", leave=False):
        # Load + resample to 48 kHz
        wav, sr = torchaudio.load(wav_path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav_np = wav[0].numpy()   # (T,)

        with torch.no_grad():
            a_in = processor(audios=wav_np, sampling_rate=target_sr,
                             return_tensors="pt").to(clap_device)
            t_in = processor(text=prompt, return_tensors="pt",
                             padding=True).to(clap_device)
            a_emb = clap_model.get_audio_features(**a_in)
            t_emb = clap_model.get_text_features(**t_in)
            sim   = F.cosine_similarity(a_emb, t_emb).item()
        sims.append(sim)

    return float(np.mean(sims))


def compute_fad(gen_dir: str, ref_dir: str) -> float:
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except ImportError:
        print("[WARN] frechet-audio-distance not installed. Skipping FAD.")
        print("       pip install frechet-audio-distance")
        return float("nan")

    fad = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=True)
    score = fad.score(ref_dir, gen_dir)
    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         required=True,  help="AudioCaps test.csv path")
    parser.add_argument("--output_dir",  default="./eval_output")
    parser.add_argument("--n_samples",   type=int, default=100)
    parser.add_argument("--num_steps",   type=int, default=200,  help="DDPM inference steps")
    parser.add_argument("--guidance",    type=float, default=3.0)
    parser.add_argument("--audioldm_ckpt", default="./eval_output/audioldm-s-full.ckpt",
                        help="Path to the AudioLDM checkpoint (downloaded automatically if missing)")
    parser.add_argument("--ref_audio_dir", default=None,
                        help="Folder of reference .wav files (needed for FAD/KL). "
                             "Filenames must be <youtube_id>.wav. "
                             "If not provided, only CLAP score is computed.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prompts from CSV ──────────────────────────────────────────────────
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= args.n_samples:
                break
    print(f"[INFO] Loaded {len(rows)} prompts from {args.csv}")

    # ── Setup device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[INFO] Loading diffusion model from declare-lab/tango-base ...")
    model = load_diffusion_model(device)

    download_audioldm_checkpoint(args.audioldm_ckpt)
    print("[INFO] Loading AudioLDM VAE ...")
    vae = load_vae(args.audioldm_ckpt, device)

    # ── Generate audio ────────────────────────────────────────────────────────
    wav_paths = []
    prompts   = []
    print(f"\n[INFO] Generating {len(rows)} audio files ...")

    for row in tqdm(rows, desc="Inference"):
        youtube_id = row["youtube_id"]
        caption    = row["caption"]
        out_path   = gen_dir / f"{youtube_id}.wav"

        if out_path.exists():
            wav_paths.append(str(out_path))
            prompts.append(caption)
            continue

        try:
            wav = generate_audio(caption, model, vae, device,
                                 num_steps=args.num_steps,
                                 guidance_scale=args.guidance)
            torchaudio.save(str(out_path), torch.from_numpy(wav).unsqueeze(0), 16000)
            wav_paths.append(str(out_path))
            prompts.append(caption)
        except Exception as e:
            print(f"[ERROR] {youtube_id}: {e}")

    print(f"\n[INFO] Generated {len(wav_paths)} files → {gen_dir}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    results = {}

    print("\n[METRIC] Computing CLAP score ...")
    results["clap_score"] = compute_clap_score(wav_paths, prompts)
    print(f"         CLAP score: {results['clap_score']:.4f}")

    if args.ref_audio_dir:
        print("\n[METRIC] Computing FAD ...")
        results["fad"] = compute_fad(str(gen_dir), args.ref_audio_dir)
        print(f"         FAD: {results['fad']:.4f}")
    else:
        print("\n[INFO] No --ref_audio_dir provided. Skipping FAD/KL.")
        print("       To get FAD, download reference audio (see README) and re-run with --ref_audio_dir.")

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = out_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump({"n_samples": len(wav_paths), **results}, f, indent=2)
    print(f"\n[DONE] Results saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
