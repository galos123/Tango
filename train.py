#!/usr/bin/env python3
"""
train.py — Train AudioDiffusion on pre-computed WavCaps latents.

Checkpoint policy:
  checkpoints/best.pt          ← lowest val loss ever
  checkpoints/last.pt          ← end of most recent epoch
  checkpoints/epoch_XXXX.pt   ← every --save_every epochs (default 30)

TensorBoard:
  tensorboard --logdir <output_dir>/tensorboard

Usage:
  python train.py \\
      --data_dir /path/to/wavcaps_dataset \\
      --output_dir ./runs/tango_run1
"""

import argparse
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

# Make original_files importable
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
        # Saved as (1, 8, 256, 16) — drop the batch dim that the VAE added
        latent = torch.load(
            self.latent_dir / f"{fid}.pt", map_location="cpu"
        ).squeeze(0)   # → (8, 256, 16)

        caption_path = self.caption_dir / f"{fid}.txt"
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        return {"latent": latent, "caption": caption}


def collate_fn(batch):
    return {
        "latent":  torch.stack([b["latent"]  for b in batch]),  # (B, 8, 256, 16)
        "caption": [b["caption"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_scheduler_config(scheduler_root: str) -> None:
    """
    Write a local DDPM scheduler config so we don't need to download a full
    Stable Diffusion checkpoint just to get a scheduler object.
    DDPMScheduler.from_pretrained(scheduler_root, subfolder='scheduler') will
    look for <scheduler_root>/scheduler/scheduler_config.json.
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


def warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def _lr(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


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
    args: argparse.Namespace,
) -> None:
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch":         epoch,
            "global_step":   global_step,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "lr_sched":      lr_sched.state_dict(),
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "args":          vars(args),
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic write — never leaves a corrupt checkpoint


def load_checkpoint(path: str, model: nn.Module, optimizer, lr_sched):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["lr_sched"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["global_step"]


# ─────────────────────────────────────────────────────────────────────────────
# Train / val for one epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    lr_sched,
    scaler,          # GradScaler or None
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
    grad_accum: int,
    grad_clip: float,
    epoch: int,
    is_train: bool,
):
    model.train(is_train)
    tag = "train" if is_train else "val"
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
                scaled_loss = loss / grad_accum
                if scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # optimizer step every grad_accum mini-batches
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
# CLI args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AudioDiffusion on WavCaps latents")

    # Paths
    p.add_argument("--data_dir",    required=True,
                   help="Dir with latent_vectors/ and captions/ subdirs")
    p.add_argument("--output_dir",  default="./runs/tango",
                   help="Where to write checkpoints + TensorBoard logs")
    p.add_argument("--unet_config", default="./unet_config.json",
                   help="Path to UNet2DConditionModel config JSON")
    p.add_argument("--resume",      default=None,
                   help="Checkpoint path to resume from")

    # Model
    p.add_argument("--text_encoder", default="google/flan-t5-large",
                   help="HuggingFace text encoder (must match cross_attention_dim in unet_config)")
    p.add_argument("--snr_gamma",    type=float, default=5.0,
                   help="Min-SNR loss weighting gamma (None to disable)")
    p.add_argument("--uncondition",  action="store_true",
                   help="Enable classifier-free guidance training (10%% caption drop)")

    # Training
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--grad_accum",   type=int,   default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--val_split",    type=float, default=0.05,
                   help="Fraction of dataset held out for validation")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--save_every",   type=int,   default=30,
                   help="Save a periodic checkpoint every N epochs")
    p.add_argument("--seed",         type=int,   default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_dir = out_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[TensorBoard] run:  tensorboard --logdir {tb_dir}")

    # ── Local scheduler config (avoids downloading Stable Diffusion) ──────────
    sched_dir = str(out_dir / "scheduler_cfg")
    ensure_scheduler_config(sched_dir)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_ds = WavCapsLatentDataset(args.data_dir)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[Split]   train={n_train:,}   val={n_val:,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[INFO] Building AudioDiffusion …")
    model = AudioDiffusion(
        text_encoder_name=args.text_encoder,
        scheduler_name=sched_dir,
        unet_model_config_path=args.unet_config,
        snr_gamma=args.snr_gamma,
        freeze_text_encoder=True,   # text encoder is frozen; only UNet trains
        uncondition=args.uncondition,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Trainable params: {n_params / 1e6:.1f} M")

    # ── Optimizer + LR schedule ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    total_opt_steps = (len(train_loader) // args.grad_accum) * args.epochs
    lr_sched = warmup_cosine_scheduler(optimizer, args.warmup_steps, total_opt_steps)

    # Mixed-precision scaler (GPU only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0

    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        start_epoch, best_val_loss, global_step = load_checkpoint(
            args.resume, model, optimizer, lr_sched
        )
        start_epoch += 1
        print(f"[INFO] Continuing from epoch {start_epoch} | best_val={best_val_loss:.4f}")

    # Log hyperparams to TensorBoard
    writer.add_text("hparams", json.dumps(vars(args), indent=2), 0)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("[INFO] Training started …\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, lr_sched, scaler,
            writer, global_step, device,
            args.grad_accum, args.grad_clip, epoch, is_train=True,
        )
        val_loss, _ = run_epoch(
            model, val_loader, optimizer, lr_sched, scaler,
            writer, global_step, device,
            args.grad_accum, args.grad_clip, epoch, is_train=False,
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:4d} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | {elapsed:.0f}s"
        )

        # Shared kwargs for every save_checkpoint call this epoch
        ckpt_kwargs = dict(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            lr_sched=lr_sched,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            global_step=global_step,
            args=args,
        )

        # 1. Always save last
        save_checkpoint(str(ckpt_dir / "last.pt"), **ckpt_kwargs)

        # 2. Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_kwargs["best_val_loss"] = best_val_loss
            save_checkpoint(str(ckpt_dir / "best.pt"), **ckpt_kwargs)
            print(f"  -> new best val loss {best_val_loss:.4f}  [{ckpt_dir}/best.pt]")

        # 3. Periodic checkpoint every save_every epochs
        if (epoch + 1) % args.save_every == 0:
            periodic_path = str(ckpt_dir / f"epoch_{epoch:04d}.pt")
            save_checkpoint(periodic_path, **ckpt_kwargs)
            print(f"  -> periodic checkpoint  [{periodic_path}]")

    writer.close()
    print(f"\n[DONE] best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
