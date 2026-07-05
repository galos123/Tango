#!/usr/bin/env python3
"""
train_dpo.py — Align a Tango-style TTA model with Diffusion-DPO.

Trains directly on preference pairs produced by either
`build_preference_dataset.py` (own LLM-augmented-prompt pairs) or
`load_audio_alpaca.py` (public Audio-Alpaca dataset) -- both write the same
on-disk layout, read by `dataset.PreferenceLatentDataset`.

Implements the Diffusion-DPO loss (Wallace et al., 2023 -- the same recipe
used to train TANGO 2 on Audio-Alpaca):

    diff_policy = mse(eps_policy_w, noise) - mse(eps_policy_l, noise)
    diff_ref    = mse(eps_ref_w,    noise) - mse(eps_ref_l,    noise)
    loss = -logsigmoid(-0.5 * beta_dpo * (diff_policy - diff_ref))

Chosen/rejected latents in a pair share the same timestep and the same text
conditioning (their generation prompt); this file only differs from that
reference on which UNet produced the noise prediction, so the loss isolates
the model's own preference signal from the frozen reference model's.

Settings live in the `if __name__ == "__main__"` block, same convention as
../train.py. Run:
    python dpo/train_dpo.py
"""

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "original_files"))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import AudioDiffusion  # noqa: E402
from train import (  # noqa: E402
    ensure_scheduler_config,
    warmup_cosine_scheduler,
    save_checkpoint,
    load_pretrained_weights,
    load_resume_checkpoint,
    _unwrap,
)
from dataset import PreferenceLatentDataset, collate_fn  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion-DPO forward pass
# ─────────────────────────────────────────────────────────────────────────────

def _unet_predict(model: AudioDiffusion, noisy_latents, timesteps, encoder_hidden_states, boolean_encoder_mask):
    if model.set_from == "random":
        return model.unet(
            noisy_latents, timesteps, encoder_hidden_states,
            encoder_attention_mask=boolean_encoder_mask,
        ).sample
    compressed = model.group_in(
        noisy_latents.permute(0, 2, 3, 1).contiguous()
    ).permute(0, 3, 1, 2).contiguous()
    pred = model.unet(
        compressed, timesteps, encoder_hidden_states,
        encoder_attention_mask=boolean_encoder_mask,
    ).sample
    return model.group_out(pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


def _per_sample_mse(pred, target):
    loss = F.mse_loss(pred.float(), target.float(), reduction="none")
    return loss.mean(dim=list(range(1, loss.dim())))


def dpo_losses(policy: AudioDiffusion, reference: AudioDiffusion, chosen, rejected, captions):
    """Returns (model_w, model_l, ref_w, ref_l): per-sample noise-prediction
    MSE for the winning/losing latent under the policy and the frozen
    reference model, computed with a shared timestep + noise per pair."""
    device = chosen.device
    bsz = chosen.shape[0]

    latents = torch.cat([chosen, rejected], dim=0)          # (2B, 8, 256, 16)
    captions2 = list(captions) + list(captions)

    encoder_hidden_states, boolean_encoder_mask = policy.encode_text(captions2)

    num_train_timesteps = policy.noise_scheduler.num_train_timesteps
    timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=device).long().repeat(2)
    noise = torch.randn_like(latents)
    noisy_latents = policy.noise_scheduler.add_noise(latents, noise, timesteps)

    if policy.noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif policy.noise_scheduler.config.prediction_type == "v_prediction":
        target = policy.noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {policy.noise_scheduler.config.prediction_type}")

    model_pred = _unet_predict(policy, noisy_latents, timesteps, encoder_hidden_states, boolean_encoder_mask)
    model_losses = _per_sample_mse(model_pred, target)
    model_w, model_l = model_losses.chunk(2)

    with torch.no_grad():
        ref_pred = _unet_predict(reference, noisy_latents, timesteps, encoder_hidden_states, boolean_encoder_mask)
        ref_losses = _per_sample_mse(ref_pred, target)
        ref_w, ref_l = ref_losses.chunk(2)

    return model_w, model_l, ref_w, ref_l


def dpo_loss_fn(model_w, model_l, ref_w, ref_l, beta_dpo: float):
    model_diff = model_w - model_l
    ref_diff = (ref_w - ref_l).detach()
    inside_term = -0.5 * beta_dpo * (model_diff - ref_diff)
    loss = -F.logsigmoid(inside_term).mean()
    reward_acc = (model_diff < ref_diff).float().mean()   # policy separates chosen/rejected more than reference
    return loss, reward_acc


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    policy, reference, loader, optimizer, lr_sched, writer, global_step, device,
    grad_accum, grad_clip, beta_dpo, sft_weight, epoch, is_train,
):
    policy.train(is_train)
    tag = "train" if is_train else "val"
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    if is_train:
        optimizer.zero_grad()

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        pbar = tqdm(loader, desc=f"Epoch {epoch:4d} [{tag}]", leave=False)
        for step, batch in enumerate(pbar):
            chosen = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)
            captions = batch["caption"]

            model_w, model_l, ref_w, ref_l = dpo_losses(policy, reference, chosen, rejected, captions)
            loss, reward_acc = dpo_loss_fn(model_w, model_l, ref_w, ref_l, beta_dpo)

            if sft_weight > 0:
                loss = loss + sft_weight * model_w.mean()

            if is_train:
                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in policy.parameters() if p.requires_grad], grad_clip
                    )
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                    global_step += 1
                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                    writer.add_scalar("train/reward_acc_step", reward_acc.item(), global_step)
                    writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], global_step)

            total_loss += loss.item()
            total_acc += reward_acc.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{reward_acc.item():.2f}")

    mean_loss = total_loss / max(1, n_batches)
    mean_acc = total_acc / max(1, n_batches)
    writer.add_scalar(f"{tag}/loss_epoch", mean_loss, epoch)
    writer.add_scalar(f"{tag}/reward_acc_epoch", mean_acc, epoch)
    return mean_loss, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    data_dir: str,
    output_dir: str,
    unet_config: str,
    pretrained_hf: str | None,
    pretrained_ckpt: str | None,
    resume: str | None,
    text_encoder: str,
    beta_dpo: float,
    sft_weight: float,
    sft_first_epochs: int,
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
    cfg = {k: v for k, v in locals().items()}

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    out_dir = Path(output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
    print(f"[TensorBoard]  tensorboard --logdir {out_dir / 'tensorboard'}")

    sched_dir = str(out_dir / "scheduler_cfg")
    ensure_scheduler_config(sched_dir)

    full_ds = PreferenceLatentDataset(data_dir)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    print(f"[Split]   train={n_train:,}   val={n_val:,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
    )

    using_hf_pretrained = pretrained_hf is not None
    print(
        "[INFO] Building AudioDiffusion policy —",
        f"from HF '{pretrained_hf}'" if using_hf_pretrained else f"from local ckpt '{pretrained_ckpt}'",
    )
    policy = AudioDiffusion(
        text_encoder_name=text_encoder,
        scheduler_name=sched_dir,
        unet_model_name=pretrained_hf if using_hf_pretrained else None,
        unet_model_config_path=None if using_hf_pretrained else unet_config,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
    ).to(device)

    if pretrained_ckpt:
        load_pretrained_weights(pretrained_ckpt, policy)

    # Reference model: a frozen snapshot of the policy's starting weights.
    print("[INFO] Building frozen reference model (deep copy of the starting policy)")
    reference = copy.deepcopy(policy).to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Trainable params: {n_params / 1e6:.1f} M")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_opt_steps = (len(train_loader) // grad_accum) * epochs
    lr_sched = warmup_cosine_scheduler(optimizer, warmup_steps, total_opt_steps)

    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if resume:
        print(f"[INFO] Resuming from {resume}")
        start_epoch, best_val_loss, global_step = load_resume_checkpoint(resume, policy, optimizer, lr_sched)
        start_epoch += 1

    writer.add_text("cfg", json.dumps(cfg, indent=2, default=str), 0)

    print("[INFO] DPO training started …\n")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        epoch_sft_weight = sft_weight if epoch < sft_first_epochs else 0.0

        train_loss, global_step = run_epoch(
            policy, reference, train_loader, optimizer, lr_sched, writer, global_step,
            device, grad_accum, grad_clip, beta_dpo, epoch_sft_weight, epoch, is_train=True,
        )
        val_loss, _ = run_epoch(
            policy, reference, val_loader, optimizer, lr_sched, writer, global_step,
            device, grad_accum, grad_clip, beta_dpo, epoch_sft_weight, epoch, is_train=False,
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f} | {elapsed:.0f}s")

        ckpt_kwargs = dict(
            epoch=epoch, model=policy, optimizer=optimizer, lr_sched=lr_sched,
            train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val_loss,
            global_step=global_step, cfg=cfg,
        )
        save_checkpoint(str(ckpt_dir / "last.pt"), **ckpt_kwargs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_kwargs["best_val_loss"] = best_val_loss
            save_checkpoint(str(ckpt_dir / "best.pt"), **ckpt_kwargs)
            print(f"  -> new best val loss {best_val_loss:.4f}  [{ckpt_dir}/best.pt]")

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
    DATA_DIR    = "./dpo_dataset"        # must contain captions/, latent_vectors_chosen/, latent_vectors_rejected/
    OUTPUT_DIR  = "./runs/tango_dpo"     # checkpoints + tensorboard go here
    UNET_CONFIG = str(_REPO_ROOT / "unet_config.json")  # only used when PRETRAINED_HF is None

    # ── Starting point for DPO ────────────────────────────────────────────────
    # DPO aligns an already fine-tuned TTA model -- it does not train from scratch.
    PRETRAINED_HF   = "declare-lab/tango"   # or "declare-lab/tango-full-ft"
    PRETRAINED_CKPT = None                  # local .pt to load on top of PRETRAINED_HF init
    RESUME          = None                  # local .pt to fully resume a DPO run

    TEXT_ENCODER = "google/flan-t5-large"

    # ── DPO hyperparameters ────────────────────────────────────────────────────
    # Defaults follow the TANGO 2 recipe (trained on Audio-Alpaca).
    BETA_DPO         = 2000.0
    SFT_WEIGHT        = 1.0   # weight of an auxiliary plain-diffusion loss on the chosen sample
    SFT_FIRST_EPOCHS  = 1     # warm-start with SFT_WEIGHT added for this many epochs, then pure DPO

    # ── Training hyperparameters ──────────────────────────────────────────────
    EPOCHS       = 5
    BATCH_SIZE   = 2
    LR           = 9.6e-7
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP    = 1.0
    GRAD_ACCUM   = 16
    WARMUP_STEPS = 100
    VAL_SPLIT    = 0.05
    NUM_WORKERS  = 2
    SAVE_EVERY   = 1
    SEED         = 42

    main(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        unet_config=UNET_CONFIG,
        pretrained_hf=PRETRAINED_HF,
        pretrained_ckpt=PRETRAINED_CKPT,
        resume=RESUME,
        text_encoder=TEXT_ENCODER,
        beta_dpo=BETA_DPO,
        sft_weight=SFT_WEIGHT,
        sft_first_epochs=SFT_FIRST_EPOCHS,
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
