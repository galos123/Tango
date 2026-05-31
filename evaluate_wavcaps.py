#!/usr/bin/env python3
"""
evaluate_wavcaps.py — Evaluate declare-lab/tango on WavCaps AudioSet-SL.

Metrics reported:
  FD   Fréchet Distance          (PANNs CNN14 features, lower = better)
  FAD  Fréchet Audio Distance    (VGGish features,      lower = better)
  KL   KL Divergence             (PANNs class probs,    lower = better)
  OVL  Overall quality proxy     (CLAP audio-audio sim, higher = better)
  REL  Relevance proxy           (CLAP text-audio sim,  higher = better)

Note on OVL / REL:
  In the TANGO paper these are human MOS scores.  Here we use CLAP-based
  proxies: REL = cosine_sim(text_embed, gen_audio_embed),
           OVL = cosine_sim(ref_audio_embed, gen_audio_embed).

Runtime estimate on T4 GPU (DDIM 50 steps):
  ~30 s/sample → 1000 samples ≈ 8 h.
  Use --n_samples 200 for a quick sanity-check (~1.5 h).

Usage:
  python evaluate_wavcaps.py \\
      --n_samples 1000 \\
      --steps 50 \\
      --audioldm_ckpt ./audioldm-s-full.ckpt \\
      --output_dir ./wavcaps_eval_out
"""

import os, sys, json, argparse, random
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from types import ModuleType as _ModuleType

# ─────────────────────────────────────────────────────────────────────────────
# audioldm stubs — must come before any models.py import
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "original_files"))

import importlib.util

_AUDIOLDM_STUBS = [
    "audioldm", "audioldm.audio", "audioldm.audio.stft",
    "audioldm.latent_diffusion", "audioldm.latent_diffusion.ema",
    "audioldm.variational_autoencoder", "audioldm.variational_autoencoder.modules",
    "audioldm.variational_autoencoder.distributions",
    "audioldm.hifigan", "audioldm.hifigan.utilities",
    "audioldm.clap", "audioldm.clap.encoders", "audioldm.pipline",
]
for _n in _AUDIOLDM_STUBS:
    if _n not in sys.modules:
        _m = _ModuleType(_n)
        _m.__path__ = []
        sys.modules[_n] = _m

sys.modules["audioldm"].audio                                    = sys.modules["audioldm.audio"]
sys.modules["audioldm.audio"].stft                               = sys.modules["audioldm.audio.stft"]
sys.modules["audioldm"].latent_diffusion                         = sys.modules["audioldm.latent_diffusion"]
sys.modules["audioldm.latent_diffusion"].ema                     = sys.modules["audioldm.latent_diffusion.ema"]
sys.modules["audioldm"].variational_autoencoder                  = sys.modules["audioldm.variational_autoencoder"]
sys.modules["audioldm.variational_autoencoder"].modules          = sys.modules["audioldm.variational_autoencoder.modules"]
sys.modules["audioldm.variational_autoencoder"].distributions    = sys.modules["audioldm.variational_autoencoder.distributions"]
sys.modules["audioldm"].hifigan                                  = sys.modules["audioldm.hifigan"]
sys.modules["audioldm.hifigan"].utilities                        = sys.modules["audioldm.hifigan.utilities"]

sys.modules["audioldm.audio.stft"].TacotronSTFT                                      = object
sys.modules["audioldm.variational_autoencoder.modules"].Encoder                       = object
sys.modules["audioldm.variational_autoencoder.modules"].Decoder                       = object
sys.modules["audioldm.variational_autoencoder.distributions"].DiagonalGaussianDistribution = object
sys.modules["audioldm.hifigan.utilities"].get_vocoder                                 = lambda *a, **k: None
sys.modules["audioldm.hifigan.utilities"].vocoder_infer                               = lambda *a, **k: None

_ae_stub = _ModuleType("autoencoder")
_ae_stub.AutoencoderKL = object
sys.modules["autoencoder"] = _ae_stub

from models import AudioDiffusion  # noqa: E402
from diffusers import DDIMScheduler  # noqa: E402

# import AutoencoderKL + CONFIG from the dashed-filename module
_lc_path = Path(__file__).parent / "a-latent-creation-wavcaps.py"
_lc_spec  = importlib.util.spec_from_file_location("latent_creation", _lc_path)
_lc_mod   = importlib.util.module_from_spec(_lc_spec)
_lc_spec.loader.exec_module(_lc_mod)
AutoencoderKL = _lc_mod.AutoencoderKL
CONFIG        = _lc_mod.CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler + model loading  (identical to evaluate.py)
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
    import tempfile
    from huggingface_hub import snapshot_download
    ensure_scheduler_config()
    print(f"[INFO] Downloading {hf_repo} ...")
    repo_path    = snapshot_download(repo_id=hf_repo)
    weights_path = os.path.join(repo_path, "pytorch_model_main.bin")

    _TANGO_UNET_CFG = {
        "_class_name": "UNet2DConditionModel",
        "act_fn": "silu",
        "attention_head_dim": 8,
        "block_out_channels": [320, 640, 1280, 1280],
        "cross_attention_dim": 1024,
        "down_block_types": ["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
        "downsample_padding": 1, "flip_sin_to_cos": True, "freq_shift": 0,
        "in_channels": 8, "layers_per_block": 2, "mid_block_scale_factor": 1,
        "norm_eps": 1e-5, "norm_num_groups": 32, "out_channels": 8, "sample_size": 64,
        "up_block_types": ["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"],
        "use_linear_projection": True,
    }
    tmp_cfg = tempfile.mkdtemp()
    with open(os.path.join(tmp_cfg, "config.json"), "w") as f:
        json.dump(_TANGO_UNET_CFG, f)

    model = AudioDiffusion(
        text_encoder_name="google/flan-t5-large",
        scheduler_name=LOCAL_SCHED_DIR,
        unet_model_name=None,
        unet_model_config_path=tmp_cfg,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
    )
    print(f"[INFO] Loading weights from {weights_path} ...")
    state = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] {len(missing)} missing keys")
    return model.to(device).eval()


def load_vae(ckpt_path: str, device: torch.device):
    print(f"[INFO] Loading VAE from {ckpt_path} ...")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    scale_factor = 1.0
    for k in ["scale_factor", "model.scale_factor"]:
        if k in state:
            scale_factor = float(state[k].item() if torch.is_tensor(state[k]) else state[k])
            break

    prefix    = "first_stage_model."
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
    del vae_state
    return vae.half().to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Audio generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_audio(prompt, model, vae, device, num_steps=50, guidance_scale=3.0):
    ddim    = DDIMScheduler.from_config(model.noise_scheduler.config)
    latents = model.inference(
        prompt=[prompt],
        inference_scheduler=ddim,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        num_samples_per_prompt=1,
        disable_progress=True,
    )
    vae_dtype = next(vae.parameters()).dtype
    latents   = latents.to(device, dtype=vae_dtype)
    mel       = vae.decode_first_stage(latents)
    wav       = vae.decode_to_waveform(mel)
    return wav[0].float().cpu().numpy()   # 1-D float32 at 16 kHz


# ─────────────────────────────────────────────────────────────────────────────
# WavCaps data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_wavcaps_samples(n_samples: int, seed: int = 42):
    """
    Stream n_samples from WavCaps AudioSet-SL via HuggingFace datasets.
    Returns list of {'caption': str, 'audio_array': np.ndarray, 'sr': int}.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"[INFO] Streaming {n_samples} samples from cvssp/WavCaps AudioSet_SL ...")
    ds = load_dataset(
        "cvssp/WavCaps",
        "AudioSet_SL",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    samples = []
    for item in tqdm(ds, desc="Fetching WavCaps", total=n_samples):
        caption = item.get("caption") or item.get("text") or ""
        audio   = item.get("audio", {})
        arr     = np.array(audio.get("array", []), dtype=np.float32)
        sr      = int(audio.get("sampling_rate", 16000))
        if len(arr) < 1000 or not caption.strip():
            continue
        samples.append({"caption": caption, "audio_array": arr, "sr": sr})
        if len(samples) >= n_samples:
            break

    print(f"[INFO] Collected {len(samples)} valid samples.")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_fd_fad_kl(gen_dir: str, ref_dir: str) -> dict:
    """FAD, FD, KL via audioldm_eval (pip install audioldm_eval)."""
    try:
        from audioldm_eval import EvaluationHelper
    except ImportError:
        print("[WARN] audioldm_eval not installed — skipping FD/FAD/KL.")
        print("       pip install audioldm_eval")
        return {"FD": float("nan"), "FAD": float("nan"), "KL": float("nan")}

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    helper     = EvaluationHelper(sampling_rate=16000, device=device_str)
    raw        = helper.main(gen_dir, ref_dir)

    return {
        "FD":  raw.get("frechet_distance",       float("nan")),
        "FAD": raw.get("frechet_audio_distance",  float("nan")),
        "KL":  raw.get("kl_softmax",              raw.get("kl_sigmoid", float("nan"))),
    }


def compute_ovl_rel(gen_wav_paths, ref_wav_paths, captions) -> dict:
    """
    OVL — CLAP audio-audio cosine similarity (generated vs reference).
    REL — CLAP text-audio cosine similarity (caption vs generated).
    Both use laion/larger_clap_music_and_speech.
    """
    from transformers import ClapModel, ClapProcessor
    import torch.nn.functional as F

    clap_name = "laion/larger_clap_music_and_speech"
    print(f"[CLAP] Loading {clap_name} ...")
    cdev      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap_mdl  = ClapModel.from_pretrained(clap_name).to(cdev).eval()
    proc      = ClapProcessor.from_pretrained(clap_name)
    target_sr = proc.feature_extractor.sampling_rate  # 48000

    def embed_audio(path):
        wav, sr = torchaudio.load(path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        inp = proc(audios=wav[0].numpy(), sampling_rate=target_sr,
                   return_tensors="pt").to(cdev)
        with torch.no_grad():
            return clap_mdl.get_audio_features(**inp)   # (1, D)

    def embed_text(text):
        inp = proc(text=text, return_tensors="pt", padding=True).to(cdev)
        with torch.no_grad():
            return clap_mdl.get_text_features(**inp)    # (1, D)

    ovl_scores, rel_scores = [], []
    for gen_p, ref_p, cap in tqdm(
        zip(gen_wav_paths, ref_wav_paths, captions),
        total=len(gen_wav_paths), desc="CLAP OVL/REL"
    ):
        try:
            gen_emb = embed_audio(gen_p)
            ref_emb = embed_audio(ref_p)
            txt_emb = embed_text(cap)
            ovl_scores.append(F.cosine_similarity(gen_emb, ref_emb).item())
            rel_scores.append(F.cosine_similarity(gen_emb, txt_emb).item())
        except Exception as e:
            print(f"  [WARN] CLAP failed for {Path(gen_p).stem}: {e}")

    return {
        "OVL": float(np.mean(ovl_scores)) if ovl_scores else float("nan"),
        "REL": float(np.mean(rel_scores)) if rel_scores else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate TANGO on WavCaps AudioSet-SL")
    parser.add_argument("--n_samples",     type=int,   default=1000)
    parser.add_argument("--steps",         type=int,   default=50,
                        help="DDIM inference steps (50 ≈ DDPM-200 quality)")
    parser.add_argument("--guidance",      type=float, default=3.0)
    parser.add_argument("--audioldm_ckpt", type=str,   default="./audioldm-s-full.ckpt",
                        help="Path to audioldm-s-full.ckpt (VAE)")
    parser.add_argument("--output_dir",    type=str,   default="./wavcaps_eval_out")
    parser.add_argument("--hf_repo",       type=str,   default="declare-lab/tango")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    gen_dir = os.path.join(args.output_dir, "generated")
    ref_dir = os.path.join(args.output_dir, "reference")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    model = load_diffusion_model(device, hf_repo=args.hf_repo)
    vae   = load_vae(args.audioldm_ckpt, device)

    # ── Load WavCaps samples ──────────────────────────────────────────────────
    samples = load_wavcaps_samples(args.n_samples, seed=args.seed)
    if len(samples) == 0:
        print("[ERROR] No WavCaps samples loaded — check your HF credentials / network.")
        return

    # ── Generate + save reference audio ──────────────────────────────────────
    captions, gen_paths, ref_paths = [], [], []
    skipped = 0

    for i, s in enumerate(tqdm(samples, desc="Generating")):
        stem    = f"{i:05d}"
        gen_p   = os.path.join(gen_dir, f"{stem}.wav")
        ref_p   = os.path.join(ref_dir,  f"{stem}.wav")

        # Save reference (resample to 16 kHz mono if needed)
        if not os.path.exists(ref_p):
            arr = s["audio_array"]
            sr  = s["sr"]
            if sr != 16000:
                t   = torch.from_numpy(arr).unsqueeze(0)
                arr = torchaudio.functional.resample(t, sr, 16000)[0].numpy()
            sf.write(ref_p, arr, 16000)

        # Generate with TANGO (skip if already done)
        if not os.path.exists(gen_p):
            try:
                wav = generate_audio(
                    s["caption"], model, vae, device,
                    num_steps=args.steps, guidance_scale=args.guidance,
                )
                sf.write(gen_p, wav, 16000)
            except Exception as e:
                print(f"  [WARN] Generation failed for sample {i}: {e}")
                skipped += 1
                continue

        captions.append(s["caption"])
        gen_paths.append(gen_p)
        ref_paths.append(ref_p)

    n = len(gen_paths)
    print(f"\n[INFO] Generated {n} samples  ({skipped} skipped)")

    # ── FD / FAD / KL ─────────────────────────────────────────────────────────
    print("\n[INFO] Computing FD / FAD / KL (audioldm_eval) ...")
    dist_metrics = compute_fd_fad_kl(gen_dir, ref_dir)

    # ── OVL / REL ─────────────────────────────────────────────────────────────
    print("\n[INFO] Computing OVL / REL (CLAP) ...")
    clap_metrics = compute_ovl_rel(gen_paths, ref_paths, captions)

    # ── Results ───────────────────────────────────────────────────────────────
    results = {**dist_metrics, **clap_metrics, "n_samples": n}

    print("\n" + "="*50)
    print("  EVALUATION RESULTS — declare-lab/tango")
    print("  Dataset: WavCaps AudioSet-SL")
    print(f"  Samples: {n}")
    print("="*50)
    print(f"  FD   (lower=better) : {results['FD']:.4f}")
    print(f"  FAD  (lower=better) : {results['FAD']:.4f}")
    print(f"  KL   (lower=better) : {results['KL']:.4f}")
    print(f"  OVL  (higher=better): {results['OVL']:.4f}  [CLAP audio-audio proxy]")
    print(f"  REL  (higher=better): {results['REL']:.4f}  [CLAP text-audio proxy]")
    print("="*50)

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {out_json}")


if __name__ == "__main__":
    main()
