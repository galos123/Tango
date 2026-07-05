#!/usr/bin/env python3
"""
load_audio_alpaca.py — Convert the public Audio-Alpaca preference dataset
(https://huggingface.co/datasets/declare-lab/audio-alpaca) into the same
on-disk format used by `build_preference_dataset.py`, so `train_dpo.py` can
train on either source interchangeably.

Audio-Alpaca ships ~15k (prompt, chosen, rejected) triplets where `chosen`
and `rejected` are full waveforms for the SAME prompt. Each waveform is
VAE-encoded here (via `audio_encoder.AudioEncoder`) into the latent format
the rest of this repo trains on.

Output layout (same as build_preference_dataset.py, minus captions_detailed
since both sides of an Audio-Alpaca pair share one prompt):
    output_dir/
      captions/                 <id>.txt
      latent_vectors_chosen/    <id>.pt
      latent_vectors_rejected/  <id>.pt
      dataset_info.json

Usage:
    python dpo/load_audio_alpaca.py --output_dir ./dpo_dataset_audio_alpaca
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from audio_encoder import AudioEncoder


def _audio_to_array(sample):
    """HF `Audio` feature -> (np.ndarray, sampling_rate)."""
    return sample["array"], sample["sampling_rate"]


def main(output_dir: str, split: str, limit: int | None, start: int, streaming: bool, overwrite: bool):
    out_dir = Path(output_dir)
    dirs = {
        "captions": out_dir / "captions",
        "chosen": out_dir / "latent_vectors_chosen",
        "rejected": out_dir / "latent_vectors_rejected",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading declare-lab/audio-alpaca ...")
    ds = load_dataset("declare-lab/audio-alpaca", split=split, streaming=streaming)
    if not streaming:
        end = (start + limit) if limit else None
        ds = ds.select(range(start, end if end is not None else len(ds)))
    elif start:
        ds = ds.skip(start)
    if streaming and limit:
        ds = ds.take(limit)

    encoder = AudioEncoder()

    manifest = {"source": "declare-lab/audio-alpaca", "split": split, "count": 0, "ids": []}

    for i, row in enumerate(tqdm(ds, desc="Encoding audio-alpaca pairs")):
        fid = f"{start + i:06d}"
        chosen_path = dirs["chosen"] / f"{fid}.pt"
        if chosen_path.exists() and not overwrite:
            manifest["ids"].append(fid)
            continue

        prompt = row["prompt"]
        chosen_array, chosen_sr = _audio_to_array(row["chosen"])
        rejected_array, rejected_sr = _audio_to_array(row["rejected"])

        chosen_latent = encoder.encode_waveform(chosen_array, chosen_sr)
        rejected_latent = encoder.encode_waveform(rejected_array, rejected_sr)

        torch.save(chosen_latent, dirs["chosen"] / f"{fid}.pt")
        torch.save(rejected_latent, dirs["rejected"] / f"{fid}.pt")
        (dirs["captions"] / f"{fid}.txt").write_text(prompt, encoding="utf-8")
        manifest["ids"].append(fid)

    manifest["count"] = len(manifest["ids"])
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] {manifest['count']} preference pairs written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=str, default="./dpo_dataset_audio_alpaca")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many rows.")
    parser.add_argument("--start", type=int, default=0, help="Row index to start from (resuming a partial run).")
    parser.add_argument("--streaming", action="store_true", help="Stream rows instead of downloading the full 9.7GB dataset upfront.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args.output_dir, args.split, args.limit, args.start, args.streaming, args.overwrite)
