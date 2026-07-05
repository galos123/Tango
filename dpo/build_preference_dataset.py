#!/usr/bin/env python3
"""
build_preference_dataset.py — Build a DPO preference dataset for the TTA model.

For every input caption:
  1. An LLM (see `prompt_expansion.py`) is asked to rewrite the caption with
     more acoustic detail -> "detailed prompt".
  2. The TTA model (AudioDiffusion) generates one latent from the detailed
     prompt and one latent from the plain, original caption -- in a single
     `inference()` call, so no VAE re-encoding is needed (the model already
     denoises directly in latent space).
  3. The pair is written to disk. By default the generation guided by the
     richer, more explicit prompt is treated as "chosen" and the generation
     from the bare caption as "rejected" (--chosen-source lets you flip
     this). The ORIGINAL caption is stored as the training-time conditioning
     text for both sides of the pair -- that's the prompt the model will
     actually see at inference time, and what DPO teaches it to prefer
     "chosen"-quality output for.

Output layout (consumed by `dataset.py` / `train_dpo.py`):
    output_dir/
      captions/                <id>.txt   original caption (training condition)
      captions_detailed/       <id>.txt   LLM-expanded caption (for reference)
      latent_vectors_chosen/   <id>.pt
      latent_vectors_rejected/ <id>.pt
      dataset_info.json

Usage:
    python dpo/build_preference_dataset.py \
        --captions_file ./captions.txt \
        --output_dir ./dpo_dataset \
        --tta_model declare-lab/tango \
        --llm_model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "original_files"))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import AudioDiffusion  # noqa: E402
from prompt_expansion import PromptExpander, DEFAULT_LLM_MODEL  # noqa: E402
from train import ensure_scheduler_config  # noqa: E402


def load_captions(captions_file: str | None, captions_dir: str | None):
    """Returns a list of (id, caption) pairs."""
    items = []
    if captions_file:
        with open(captions_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    items.append((f"{i:06d}", line))
    elif captions_dir:
        for p in sorted(Path(captions_dir).glob("*.txt")):
            caption = p.read_text(encoding="utf-8").strip()
            if caption:
                items.append((p.stem, caption))
    else:
        raise ValueError("Provide either --captions_file or --captions_dir")
    return items


def main(
    output_dir: str,
    captions_file: str | None,
    captions_dir: str | None,
    tta_model: str,
    text_encoder: str,
    llm_model: str,
    chosen_source: str,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    overwrite: bool,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(output_dir)
    dirs = {
        "captions": out_dir / "captions",
        "captions_detailed": out_dir / "captions_detailed",
        "chosen": out_dir / "latent_vectors_chosen",
        "rejected": out_dir / "latent_vectors_rejected",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    items = load_captions(captions_file, captions_dir)
    print(f"[INFO] {len(items)} captions to process")

    print(f"[INFO] Loading LLM for prompt expansion: {llm_model}")
    expander = PromptExpander(model_name=llm_model, device=str(device))

    print(f"[INFO] Loading TTA model: {tta_model}")
    sched_dir = str(out_dir / "scheduler_cfg")
    ensure_scheduler_config(sched_dir)
    policy = AudioDiffusion(
        text_encoder_name=text_encoder,
        scheduler_name=sched_dir,
        unet_model_name=tta_model,
        snr_gamma=None,
        freeze_text_encoder=True,
    ).to(device).eval()

    manifest = {
        "tta_model": tta_model,
        "text_encoder": text_encoder,
        "llm_model": llm_model,
        "chosen_source": chosen_source,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "count": 0,
        "ids": [],
    }

    pending = [
        (fid, caption) for fid, caption in items
        if overwrite or not (dirs["chosen"] / f"{fid}.pt").exists()
    ]
    print(f"[INFO] {len(pending)} pairs left to generate (skip existing unless --overwrite)")

    for start in tqdm(range(0, len(pending), batch_size), desc="Building preference pairs"):
        batch = pending[start:start + batch_size]
        fids = [fid for fid, _ in batch]
        originals = [caption for _, caption in batch]
        detailed = [expander.expand(caption) for caption in originals]

        prompts = detailed + originals  # first half detailed, second half original
        with torch.no_grad():
            latents = policy.inference(
                prompts,
                policy.inference_scheduler,
                num_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_samples_per_prompt=1,
            )
        b = len(batch)
        latents_detailed = latents[:b].cpu()
        latents_original = latents[b:].cpu()

        for i, fid in enumerate(fids):
            latent_detailed = latents_detailed[i:i + 1]
            latent_original = latents_original[i:i + 1]
            if chosen_source == "detailed":
                chosen, rejected = latent_detailed, latent_original
            else:
                chosen, rejected = latent_original, latent_detailed

            torch.save(chosen, dirs["chosen"] / f"{fid}.pt")
            torch.save(rejected, dirs["rejected"] / f"{fid}.pt")
            (dirs["captions"] / f"{fid}.txt").write_text(originals[i], encoding="utf-8")
            (dirs["captions_detailed"] / f"{fid}.txt").write_text(detailed[i], encoding="utf-8")
            manifest["ids"].append(fid)

    manifest["count"] = len(manifest["ids"])
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] {manifest['count']} preference pairs written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions_file", type=str, default=None, help="One caption per line.")
    parser.add_argument("--captions_dir", type=str, default=None, help="Folder of <id>.txt caption files.")
    parser.add_argument("--output_dir", type=str, default="./dpo_dataset")
    parser.add_argument("--tta_model", type=str, default="declare-lab/tango", help="HF UNet checkpoint to generate with.")
    parser.add_argument("--text_encoder", type=str, default="google/flan-t5-large")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help="HF instruction-tuned LLM used to expand prompts.")
    parser.add_argument("--chosen_source", type=str, default="detailed", choices=["detailed", "original"],
                         help="Which generation becomes the 'chosen' (preferred) sample.")
    parser.add_argument("--batch_size", type=int, default=4, help="Caption pairs generated per inference() call.")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate pairs that already exist on disk.")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        captions_file=args.captions_file,
        captions_dir=args.captions_dir,
        tta_model=args.tta_model,
        text_encoder=args.text_encoder,
        llm_model=args.llm_model,
        chosen_source=args.chosen_source,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        overwrite=args.overwrite,
    )
