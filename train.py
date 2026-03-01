#!/usr/bin/env python3
"""
train.py — Train or fine-tune AudioDiffusion on pre-computed WavCaps latents.

All settings live at the bottom of this file inside  if __name__ == "__main__".
Edit that block, then run:
    python train.py

Checkpoint policy
  checkpoints/best.pt           ← lowest val loss ever seen
  checkpoints/last.pt           ← end of the most recent epoch
  checkpoints/epoch_XXXX.pt    ← periodic save every SAVE_EVERY epochs

TensorBoard
  tensorboard --logdir <OUTPUT_DIR>/tensorboard
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "original_files"))
from models import AudioDiffusion  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WavCapsLatentDataset(Dataset):
    """
    Loads pre-computed latents + captions from a-latent-creation-wavcaps.py.

    Expected layout:
        data_dir/
          latent_vectors/   <file_id>.pt    # shape (1, 8, 256, 16)
          captions/         <file_id>.txt
    """

    def __init__(self, data_dir: str):
        self.latent_dir  = Path(data_dir) / "latent_vectors"
        self.caption_dir = Path(data_dir) / "captions"
        self.file_ids = sorted(p.stem for p in self.latent_dir.glob("*.pt"))
        if not self.file_ids:
            raise RuntimeError(f"No .pt latents found in {self.latent_dir}")
        print(f"[Dataset] {len(self.file_ids):,} samples  ({data_dir})")

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        # Saved with a batch dim (1, 8, 256, 16) — squeeze it out
        latent = torch.load(
            self.latent_dir / f"{fid}.pt", map_location="cpu"
        ).squeeze(0)   # → (8, 256, 16)

        with open(self.caption_dir / f"{fid}.txt", "r", encoding="utf-8") as f:
            caption = f.read().strip()

        return {"latent": latent, "caption": caption}


def collate_fn(batch):
    return {
        "latent":  torch.stack([b["latent"]  for b in batch]),  # (B, 8, 256, 16)
        "caption": [b["caption"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler config (written locally so no SD download is needed)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_scheduler_config(scheduler_root: str) -> None:
    """
    DDPMScheduler.from_pretrained(scheduler_root, subfolder='scheduler') looks
    for  <scheduler_root>/scheduler/scheduler_config.json.
    We write it on first run so you don't need a Stable Diffusion checkpoint.
    """
    cfg_path = Path(scheduler_root) / "scheduler" / "scheduler_config.json"
    if cfg_path.exists():
        return
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "_class_name": "DDPMScheduler",
        "_diffusers_version": "0.21.0",
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "clip_sample": False,
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "variance_type": "fixed_small",
        "thresholding": False,
        "steps_offset": 0,
        "trained_betas": None,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[Config] Wrote scheduler config → {cfg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying model, unwrapping DataParallel if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def load_pretrained_weights(path: str, model: nn.Module) -> None:
    """
    Load model weights from a .pt file for fine-tuning.
    Handles both:
      • our checkpoint format  {"model": state_dict, ...}
      • a raw state_dict saved directly with torch.save()
    Uses strict=False so partial matches work (e.g. if the TANGO checkpoint
    has slightly different keys, missing keys stay randomly initialised).
    """
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)   # unwrap if needed
    missing, unexpected = _unwrap(model).load_state_dict(state_dict, strict=False)
    print(f"[Pretrained] Loaded weights from {path}")
    if missing:
        print(f"  {len(missing)} missing keys → trained from random init")
    if unexpected:
        print(f"  {len(unexpected)} unexpected keys → ignored")


def load_resume_checkpoint(path: str, model: nn.Module, optimizer, lr_sched):
    """Restore full training state (model + optimizer + scheduler + counters)."""
    ckpt = torch.load(path, map_location="cpu")
    _unwrap(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["lr_sched"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["global_step"]


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    lr_sched,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    global_step: int,
    cfg: dict,
) -> None:
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch":         epoch,
            "global_step":   global_step,
            "model":         _unwrap(model).state_dict(),
            "optimizer":     optimizer.state_dict(),
            "lr_sched":      lr_sched.state_dict(),
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "cfg":           cfg,
        },
        tmp,
    )
    os.replace(tmp, path)   # atomic — never leaves a corrupt file


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def _lr(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training or validation
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    lr_sched,
    scaler,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
    grad_accum: int,
    grad_clip: float,
    epoch: int,
    is_train: bool,
):
    model.train(is_train)
    tag        = "train" if is_train else "val"
    total_loss = 0.0
    n_batches  = 0

    if is_train:
        optimizer.zero_grad()

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        pbar = tqdm(loader, desc=f"Epoch {epoch:4d} [{tag}]", leave=False)
        for step, batch in enumerate(pbar):
            latents  = batch["latent"].to(device)   # (B, 8, 256, 16)
            captions = batch["caption"]

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                loss = model(latents, captions, validation_mode=not is_train)

            if is_train:
                scaled = loss / grad_accum
                if scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                if (step + 1) % grad_accum == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip,
                    )
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                    global_step += 1

                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                    writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], global_step)

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = total_loss / max(1, n_batches)
    writer.add_scalar(f"{tag}/loss_epoch", mean_loss, epoch)
    return mean_loss, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    # Paths
    data_dir: str,
    output_dir: str,
    unet_config: str,
    # Fine-tuning / resume  (set at most ONE)
    pretrained_hf: str | None,    # HuggingFace model ID  e.g. "declare-lab/tango"
    pretrained_ckpt: str | None,  # local .pt  — loads weights only, fresh optimizer
    resume: str | None,           # local .pt  — restores weights + optimizer + epoch
    # Model
    text_encoder: str,
    snr_gamma: float,
    uncondition: bool,
    # Training
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    grad_accum: int,
    warmup_steps: int,
    val_split: float,
    num_workers: int,
    save_every: int,
    seed: int,
):
    # Collect cfg for checkpoint serialisation
    cfg = {k: v for k, v in locals().items()}

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    out_dir  = Path(output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_dir = out_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[TensorBoard]  tensorboard --logdir {tb_dir}")

    # ── Local DDPM scheduler config ───────────────────────────────────────────
    sched_dir = str(out_dir / "scheduler_cfg")
    ensure_scheduler_config(sched_dir)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_ds = WavCapsLatentDataset(data_dir)
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"[Split]   train={n_train:,}   val={n_val:,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # pretrained_hf  → pass as unet_model_name (HF downloads the UNet weights)
    # pretrained_ckpt → build from unet_config, then load .pt weights below
    # resume         → build from unet_config, then load full checkpoint below
    # (none)         → build from unet_config, random UNet init

    using_hf_pretrained = pretrained_hf is not None
    unet_model_name     = pretrained_hf if using_hf_pretrained else None
    unet_model_cfg_path = None if using_hf_pretrained else unet_config

    print(
        "[INFO] Building AudioDiffusion —",
        f"fine-tuning from HF '{pretrained_hf}'" if using_hf_pretrained
        else f"fine-tuning from '{pretrained_ckpt}'" if pretrained_ckpt
        else f"resuming from '{resume}'" if resume
        else "training from scratch",
    )

    model = AudioDiffusion(
        text_encoder_name=text_encoder,
        scheduler_name=sched_dir,
        unet_model_name=unet_model_name,
        unet_model_config_path=unet_model_cfg_path,
        snr_gamma=snr_gamma,
        freeze_text_encoder=True,   # text encoder stays frozen; only UNet trains
        uncondition=uncondition,
    ).to(device)

    # Load local pretrained weights BEFORE wrapping with DataParallel
    if pretrained_ckpt:
        load_pretrained_weights(pretrained_ckpt, model)

    # Collect trainable params from the unwrapped model BEFORE DataParallel
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Trainable params: {n_params / 1e6:.1f} M")

    # ── Multi-GPU (DataParallel) ───────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"[INFO] Using {n_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
    elif n_gpus == 1:
        print("[INFO] Single GPU")

    # ── Optimizer + LR schedule ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay
    )
    total_opt_steps = (len(train_loader) // grad_accum) * epochs
    lr_sched = warmup_cosine_scheduler(optimizer, warmup_steps, total_opt_steps)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Resume (restores optimizer + epoch + step) ────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0

    if resume:
        print(f"[INFO] Resuming from {resume}")
        start_epoch, best_val_loss, global_step = load_resume_checkpoint(
            resume, model, optimizer, lr_sched
        )
        start_epoch += 1
        print(f"       Epoch {start_epoch} | best_val={best_val_loss:.4f} | step={global_step}")

    writer.add_text("cfg", json.dumps(cfg, indent=2, default=str), 0)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("[INFO] Training started …\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, lr_sched, scaler,
            writer, global_step, device, grad_accum, grad_clip,
            epoch, is_train=True,
        )
        val_loss, _ = run_epoch(
            model, val_loader, optimizer, lr_sched, scaler,
            writer, global_step, device, grad_accum, grad_clip,
            epoch, is_train=False,
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:4d} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | {elapsed:.0f}s"
        )

        ckpt_kwargs = dict(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            lr_sched=lr_sched,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            global_step=global_step,
            cfg=cfg,
        )

        # 1 — always save last epoch
        save_checkpoint(str(ckpt_dir / "last.pt"), **ckpt_kwargs)

        # 2 — save best epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_kwargs["best_val_loss"] = best_val_loss
            save_checkpoint(str(ckpt_dir / "best.pt"), **ckpt_kwargs)
            print(f"  -> new best val loss {best_val_loss:.4f}  [{ckpt_dir}/best.pt]")

        # 3 — periodic checkpoint every SAVE_EVERY epochs
        if (epoch + 1) % save_every == 0:
            periodic_path = str(ckpt_dir / f"epoch_{epoch:04d}.pt")
            save_checkpoint(periodic_path, **ckpt_kwargs)
            print(f"  -> periodic checkpoint saved  [{periodic_path}]")

    writer.close()
    print(f"\n[DONE] best checkpoint: {ckpt_dir / 'best.pt'}")


# ═════════════════════════════════════════════════════════════════════════════
# EDIT EVERYTHING BELOW THIS LINE
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR   = "/path/to/wavcaps_dataset"   # must contain latent_vectors/ + captions/
    OUTPUT_DIR = "./runs/tango_finetune"       # checkpoints + tensorboard go here
    UNET_CONFIG = "./unet_config.json"         # only used when NOT loading from HF/CKPT

    # ── Fine-tuning / resume ──────────────────────────────────────────────────
    # Set ONE of the three options below (or all None to train from scratch).
    #
    #  PRETRAINED_HF    — fine-tune from a HuggingFace model.
    #                     AudioDiffusion will call
    #                       UNet2DConditionModel.from_pretrained(PRETRAINED_HF, subfolder="unet")
    #                     so the repo must have a unet/ subfolder in diffusers format.
    #                     e.g. "declare-lab/tango"
    #
    #  PRETRAINED_CKPT  — fine-tune from a local .pt checkpoint.
    #                     Loads model weights only; epoch counter + optimizer reset to 0.
    #                     Use this for a downloaded TANGO .pt file or a previous best.pt.
    #                     e.g. "./downloads/tango_full.pt"
    #
    #  RESUME           — fully resume an interrupted training run.
    #                     Restores model weights, optimizer state, epoch counter, and step.
    #                     e.g. "./runs/tango_finetune/checkpoints/last.pt"
    #
    PRETRAINED_HF   = "declare-lab/tango"   # ← HF fine-tune (change or set None)
    PRETRAINED_CKPT = None                  # ← local .pt fine-tune
    RESUME          = None                  # ← resume interrupted run

    # ── Model ─────────────────────────────────────────────────────────────────
    TEXT_ENCODER = "google/flan-t5-large"   # must match cross_attention_dim in unet_config
    SNR_GAMMA    = 5.0                      # min-SNR loss weighting; set None to disable
    UNCONDITION  = False                    # True = CFG training (randomly drops 10% of captions)

    # ── Training hyperparameters ──────────────────────────────────────────────
    EPOCHS       = 10        # fine-tuning needs far fewer epochs than scratch (300)
    #
    # BATCH_SIZE is the TOTAL batch fed per step.  DataParallel splits it evenly
    # across all GPUs, so each GPU receives  BATCH_SIZE / n_gpus  samples.
    #   2× RTX 2080 Ti (11 GB each) → BATCH_SIZE = 16  (8 per GPU) works well.
    #   Single GPU                  → BATCH_SIZE = 8
    BATCH_SIZE   = 16        # 16 total → 8 per GPU on 2× 2080 Ti
    LR           = 3e-5      # learning rate  (use 1e-4 for training from scratch)
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP    = 1.0       # max gradient norm
    GRAD_ACCUM   = 2         # effective batch = BATCH_SIZE × GRAD_ACCUM  (= 32 here)
    WARMUP_STEPS = 200       # steps before LR reaches its peak  (use 500 for scratch)
    VAL_SPLIT    = 0.05      # fraction of dataset held out for validation
    NUM_WORKERS  = 8         # DataLoader workers  (4 per GPU is a good rule of thumb)
    SAVE_EVERY   = 30        # save a periodic checkpoint every N epochs
    SEED         = 42

    # ── Launch ────────────────────────────────────────────────────────────────
    main(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        unet_config=UNET_CONFIG,
        pretrained_hf=PRETRAINED_HF,
        pretrained_ckpt=PRETRAINED_CKPT,
        resume=RESUME,
        text_encoder=TEXT_ENCODER,
        snr_gamma=SNR_GAMMA,
        uncondition=UNCONDITION,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        grad_accum=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        val_split=VAL_SPLIT,
        num_workers=NUM_WORKERS,
        save_every=SAVE_EVERY,
        seed=SEED,
    )
