#!/usr/bin/env python3
"""
dataset.py — PyTorch Dataset for DPO preference pairs.

Reads the common on-disk format produced by either
`build_preference_dataset.py` or `load_audio_alpaca.py`:

    data_dir/
      captions/                 <id>.txt
      latent_vectors_chosen/    <id>.pt   # shape (1, 8, 256, 16)
      latent_vectors_rejected/  <id>.pt   # shape (1, 8, 256, 16)
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class PreferenceLatentDataset(Dataset):
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        self.chosen_dir = data_dir / "latent_vectors_chosen"
        self.rejected_dir = data_dir / "latent_vectors_rejected"
        self.caption_dir = data_dir / "captions"

        chosen_ids = {p.stem for p in self.chosen_dir.glob("*.pt")}
        rejected_ids = {p.stem for p in self.rejected_dir.glob("*.pt")}
        caption_ids = {p.stem for p in self.caption_dir.glob("*.txt")}
        self.file_ids = sorted(chosen_ids & rejected_ids & caption_ids)

        if not self.file_ids:
            raise RuntimeError(f"No complete preference pairs found in {data_dir}")
        print(f"[Dataset] {len(self.file_ids):,} preference pairs  ({data_dir})")

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        chosen = torch.load(self.chosen_dir / f"{fid}.pt", map_location="cpu").squeeze(0)
        rejected = torch.load(self.rejected_dir / f"{fid}.pt", map_location="cpu").squeeze(0)
        with open(self.caption_dir / f"{fid}.txt", "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return {"chosen": chosen, "rejected": rejected, "caption": caption}


def collate_fn(batch):
    return {
        "chosen": torch.stack([b["chosen"] for b in batch]),
        "rejected": torch.stack([b["rejected"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }
