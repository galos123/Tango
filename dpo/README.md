# DPO alignment for the Tango TTA model

This folder adds preference-based alignment (Direct Preference Optimization,
the recipe used to train [TANGO 2](https://tango2-web.github.io/) on
[Audio-Alpaca](https://huggingface.co/datasets/declare-lab/audio-alpaca)) on
top of the diffusion training pipeline in `../train.py`.

DPO starts from an already-trained/fine-tuned TTA checkpoint and teaches it
to prefer "chosen" generations over "rejected" ones for the same prompt. It
does not train a model from scratch.

## 1. Building a preference dataset

Two ways to get a preference dataset, both writing the **same** on-disk
format so `train_dpo.py` doesn't care which one you used:

```
<dataset_dir>/
  captions/                 <id>.txt   # prompt used as conditioning at train time
  latent_vectors_chosen/    <id>.pt    # shape (1, 8, 256, 16)
  latent_vectors_rejected/  <id>.pt    # shape (1, 8, 256, 16)
  dataset_info.json
```

### a) Generate your own pairs (LLM prompt-augmentation)

`build_preference_dataset.py` reproduces the original workflow used for this
project:

1. For each short caption, an instruction-tuned HuggingFace LLM is given
   **two prompts**: a system instruction telling it to rewrite the caption
   with more acoustic/temporal/environmental detail, and the caption itself
   as the user turn (see `prompt_expansion.py`). The reply is the "detailed
   prompt".
2. The TTA model (`AudioDiffusion`) generates one latent conditioned on the
   detailed prompt and one conditioned on the plain original caption, in a
   single `inference()` call — no VAE re-encoding needed, since generation
   already happens in the same latent space training uses.
3. By default the detailed-prompt generation is stored as `chosen` and the
   plain-prompt generation as `rejected` (`--chosen_source original` flips
   this if your own evaluation says otherwise). The **original** caption is
   stored as the conditioning text for the pair, since that's the prompt the
   model will actually see at inference time — DPO then teaches the model
   that, given that plain prompt, it should prefer chosen-quality output.

```bash
python dpo/build_preference_dataset.py \
    --captions_file ./captions.txt \
    --output_dir ./dpo_dataset \
    --tta_model declare-lab/tango \
    --llm_model Qwen/Qwen2.5-7B-Instruct \
    --batch_size 4
```

`--captions_dir` also works if you already have a `captions/*.txt` folder
(e.g. reuse the one produced by `a-latent-creation-wavcaps.py`).

### b) Use the public Audio-Alpaca dataset

`load_audio_alpaca.py` downloads `declare-lab/audio-alpaca` (prompt, chosen
audio, rejected audio triplets) and VAE-encodes both waveforms with the same
STFT/VAE pipeline as `a-latent-creation-wavcaps.py`:

```bash
python dpo/load_audio_alpaca.py --output_dir ./dpo_dataset_audio_alpaca
```

Use `--streaming` to avoid downloading the full ~9.7GB dataset upfront, and
`--limit`/`--start` to process it in chunks.

You can merge outputs from both builders into one directory (they write
disjoint zero-padded ids by construction — rename if you combine them) to
train on a mix of self-generated and Audio-Alpaca pairs.

## 2. Training

```bash
python dpo/train_dpo.py
```

Settings live in the `if __name__ == "__main__":` block at the bottom of
`train_dpo.py`, same convention as `../train.py`. Key ones:

| Setting | Default | Notes |
|---|---|---|
| `PRETRAINED_HF` | `declare-lab/tango` | starting checkpoint to align |
| `BETA_DPO` | 2000.0 | DPO inverse-temperature; TANGO 2's value |
| `SFT_WEIGHT` / `SFT_FIRST_EPOCHS` | 1.0 / 1 | adds a plain diffusion loss on the chosen sample for the first N epochs to warm-start before pure DPO |
| `LR` | 9.6e-7 | TANGO 2's DPO learning rate (much lower than plain fine-tuning) |

Training builds a frozen deep-copy of the starting policy as the DPO
reference model, then for every pair:

1. Samples **one** timestep and noise per pair, shared between chosen and
   rejected (and between the policy and reference passes) so the only
   difference between the two noise predictions is the UNet weights.
2. Computes the noise-prediction MSE for chosen/rejected under both the
   trainable policy and the frozen reference.
3. Loss: `-logsigmoid(-0.5 * beta_dpo * ((mse_policy_w - mse_policy_l) - (mse_ref_w - mse_ref_l)))`.

TensorBoard logs `loss` and `reward_acc` (fraction of the batch where the
policy separates chosen/rejected further than the reference does — the
standard DPO training diagnostic).

Checkpoints (`checkpoints/{last,best,epoch_XXXX}.pt`) only store the policy
model, in the same format `train.py` uses, so they can be loaded with
`load_pretrained_weights` / used as `PRETRAINED_CKPT` for a further DPO run
or plain fine-tuning.

## Files

| File | Purpose |
|---|---|
| `prompt_expansion.py` | Sends the (instruction, caption) prompt pair to an HF LLM to get a detailed rewrite |
| `build_preference_dataset.py` | Builds preference pairs by generating audio from both prompts with the TTA model |
| `audio_encoder.py` | VAE/STFT encoder for turning raw audio into latents (only needed for Audio-Alpaca) |
| `load_audio_alpaca.py` | Converts `declare-lab/audio-alpaca` into the common preference-pair format |
| `dataset.py` | `PreferenceLatentDataset` — reads the common format for training |
| `train_dpo.py` | Diffusion-DPO training loop |

## Limitations

- Single-GPU only (unlike `../train.py`, no `DataParallel` support — the DPO
  loss needs direct access to `encode_text`/`unet`/`group_in`/`group_out` on
  both the policy and reference models, which a `DataParallel` wrapper
  would hide behind `forward()`).
- `chosen`/`rejected` assignment in `build_preference_dataset.py` is a fixed
  heuristic (detailed prompt = better). If you have a scoring model (e.g.
  CLAP similarity) plug it in where the pair is written instead of assuming
  the heuristic always holds.
