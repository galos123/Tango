#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Latent Creation Script for WavCaps / AudioSet-SL Data
# ============================================================
#
# Usage:
#   python a-latent-creation-wavcaps.py \
#       --audio_dir ./original_data/AudioSet_SL \
#       --json_dir  ./original_data/json_files/AudioSet_SL \
#       --output_dir ./new_dataset
#
# Inputs:
#   --audio_dir   Folder containing .wav files (searched recursively)
#   --json_dir    Folder containing WavCaps JSON file(s) with {"data": [...]}
#   --output_dir  Where to create the dataset (default: ./new_dataset)
#
# Output structure:
#   new_dataset/
#     ├── captions/          (.txt files)
#     ├── latent_vectors/    (.pt files)
#     ├── mel_spectrograms/  (.png files)
#     └── original_wavs/     (symlinks to original .wav files)
#
# ============================================================

# ============================================================
# 0) Imports
# ============================================================
import os
import sys
import math
import json
import glob
import argparse
import torch
import requests
import torchaudio
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn


# ============================================================
# 1) Utility: download checkpoint
# ============================================================
AUDIO_LDM_S_FULL_URL = (
    "https://zenodo.org/record/7600541/files/audioldm-s-full?download=1"
)


def download_file(url, dst_path, min_bytes_ok=10_000_000):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) >= min_bytes_ok:
        print(
            f"[OK] checkpoint exists: {dst_path} ({os.path.getsize(dst_path)/1e9:.2f} GB)"
        )
        return
    print(f"[DL] downloading: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        pct = 100.0 * done / total
                        print(
                            f"\r  {pct:6.2f}%  {done/1e9:.2f}/{total/1e9:.2f} GB",
                            end="",
                        )
    print("\n[OK] download finished:", dst_path)


# ============================================================
# 2) Audio processing helpers
# ============================================================
def window_sumsquare(
    window, n_frames, hop_length, win_length, n_fft, dtype=np.float32, norm=None
):
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    win_sq = get_window(window, win_length, fftbins=True)
    if norm is not None:
        win_sq = win_sq / (np.linalg.norm(win_sq, ord=norm) + 1e-12)
    win_sq = win_sq**2
    win_sq = pad_center(win_sq, n_fft)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-5):
    return normalize_fun(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    return torch.exp(x) / C


def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav


def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5


def read_wav_file(filename, segment_length):
    waveform, sr = torchaudio.load(filename, backend="soundfile")
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)

    waveform = waveform / np.max(np.abs(waveform))
    waveform = 0.5 * waveform
    return waveform


def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]
    return fbank


# ============================================================
# 3) STFT + TacotronSTFT
# ============================================================
class STFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        device = self.forward_basis.device
        input_data = input_data.to(device)

        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        device = self.forward_basis.device
        magnitude, phase = magnitude.to(device), phase.to(device)

        recombine = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            approx_nonzero = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            inverse_transform[:, :, approx_nonzero] /= window_sum[approx_nonzero]
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]
        return inverse_transform

    def forward(self, input_data):
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes, normalize_fun):
        return dynamic_range_compression(magnitudes, normalize_fun)

    def spectral_de_normalize(self, magnitudes):
        return dynamic_range_decompression(magnitudes)

    def mel_spectrogram(self, y, normalize_fun=torch.log):
        assert torch.min(y.data) >= -1, torch.min(y.data)
        assert torch.max(y.data) <= 1, torch.max(y.data)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output, normalize_fun)
        energy = torch.norm(magnitudes, dim=1)
        log_magnitudes = self.spectral_normalize(magnitudes, normalize_fun)
        return mel_output, log_magnitudes, energy


def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    log_magnitudes_stft = (
        torch.squeeze(log_magnitudes_stft, 0).numpy().astype(np.float32)
    )
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, log_magnitudes_stft, energy


def wav_to_fbank(filename, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None
    waveform = read_wav_file(filename, target_length * 160)  # hop size 160
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank = _pad_spec(fbank, target_length)
    log_magnitudes_stft = _pad_spec(log_magnitudes_stft, target_length)

    return fbank, log_magnitudes_stft, waveform


# ============================================================
# 4) HiFi-GAN vocoder
# ============================================================
LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


from torch.nn.utils import weight_norm, remove_weight_norm


class ResBlockHifi(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


HIFIGAN_16K_64 = {
    "resblock": "1",
    "upsample_rates": [5, 4, 2, 2, 2],
    "upsample_kernel_sizes": [16, 16, 8, 4, 4],
    "upsample_initial_channel": 1024,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "num_mels": 64,
}


class GeneratorHifi(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(ResBlockHifi(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                xs = rb(x) if xs is None else (xs + rb(x))
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def get_vocoder(device):
    config = AttrDict(HIFIGAN_16K_64)
    vocoder = GeneratorHifi(config)
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


@torch.no_grad()
def vocoder_infer(mels, vocoder):
    wavs = vocoder(mels).squeeze(1)
    return wavs


# ============================================================
# 5) VAE modules (AutoencoderKL)
# ============================================================
def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = (
                self.conv_shortcut(x)
                if self.use_conv_shortcut
                else self.nin_shortcut(x)
            )
        return x + h


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=1,
        resolution=256,
        z_channels=8,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, 1, 1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = None

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, dropout=dropout
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, (2 * z_channels if double_z else z_channels), 3, 1, 1
        )

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hs.append(self.down[i_level].block[i_block](hs[-1], None))
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, None)
        h = self.mid.block_2(h, None)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=1,
        resolution=256,
        z_channels=8,
        give_pre_end=False,
        tanh_out=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, dropout=dropout
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h, None)
        h = self.mid.block_2(h, None)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, None)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(
        self, ddconfig, embed_dim, image_key="fbank", subband=1, scale_factor=1.0
    ):
        super().__init__()
        self.image_key = image_key
        self.subband = int(subband)
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.vocoder = get_vocoder("cpu")
        self.embed_dim = embed_dim
        self.scale_factor = float(scale_factor)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_to_waveform(self, dec):
        dec = dec.squeeze(1).permute(0, 2, 1)
        wav = vocoder_infer(dec, self.vocoder)
        return wav

    @torch.no_grad()
    def get_first_stage_encoding(self, encoder_posterior):
        z = (
            encoder_posterior.sample()
            if isinstance(encoder_posterior, DiagonalGaussianDistribution)
            else encoder_posterior
        )
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = (1.0 / self.scale_factor) * z
        return self.decode(z)


# ============================================================
# 6) Config
# ============================================================
CONFIG = {
    "preprocessing": {
        "audio": {"sampling_rate": 16000, "max_wav_value": 32768},
        "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
        "mel": {
            "n_mel_channels": 64,
            "mel_fmin": 0,
            "mel_fmax": 8000,
            "target_length": 1024,
        },
    },
    "first_stage_config": {
        "params": {
            "image_key": "fbank",
            "subband": 1,
            "embed_dim": 8,
            "ddconfig": {
                "double_z": True,
                "z_channels": 8,
                "resolution": 256,
                "in_channels": 1,
                "out_ch": 1,
                "ch": 128,
                "ch_mult": [1, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
        }
    },
}


# ============================================================
# 7) Load WavCaps JSON metadata from a folder
# ============================================================
def load_wavcaps_metadata(json_dir):
    """
    Loads all WavCaps-format JSON files from a directory.
    Each JSON has structure: {"data": [{"id": "...", "caption": "...", ...}, ...]}
    Returns a DataFrame with columns: [id, caption] (+ any other fields).
    """
    all_records = []
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

    if not json_files:
        print(f"[ERROR] No JSON files found in: {json_dir}")
        sys.exit(1)

    for jf in json_files:
        print(f"[INFO] Loading JSON: {jf}")
        with open(jf, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # WavCaps stores records in the 'data' key
        if "data" in raw:
            records = raw["data"]
        elif isinstance(raw, list):
            records = raw
        else:
            print(f"[WARN] Unexpected JSON structure in {jf}, skipping.")
            continue

        all_records.extend(records)

    df = pd.DataFrame(all_records)

    if "id" not in df.columns:
        print(f"[ERROR] JSON records do not have an 'id' field. Found columns: {list(df.columns)}")
        sys.exit(1)
    if "caption" not in df.columns:
        print(f"[ERROR] JSON records do not have a 'caption' field. Found columns: {list(df.columns)}")
        sys.exit(1)

    df["id"] = df["id"].astype(str)

    # Normalize IDs: strip directory prefixes and file extensions so that
    # e.g. "AudioSet_SL/Y---1_cCGK4.flac" becomes "Y---1_cCGK4"
    df["id_raw"] = df["id"]
    df["id"] = df["id"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    sample_ids = df["id"].head(5).tolist()
    print(f"[INFO] Loaded {len(df)} metadata records from {len(json_files)} JSON file(s).")
    print(f"[DEBUG] Sample normalized JSON IDs: {sample_ids}")
    return df


# ============================================================
# 8) Scan audio folder recursively and build id -> path mapping
# ============================================================
def scan_audio_folder(audio_dir):
    """
    Recursively finds all .wav and .flac files in audio_dir.
    Returns a dict mapping filename_stem -> absolute_path.
    If a stem appears as both .wav and .flac, .wav takes precedence.
    """
    audio_map = {}
    wav_count = 0
    flac_count = 0
    # Walk twice: first flac (lower priority), then wav (higher priority)
    for ext, counter_attr in [(".flac", "flac"), (".wav", "wav")]:
        for root, dirs, files in os.walk(audio_dir):
            for fname in files:
                if fname.lower().endswith(ext):
                    stem = os.path.splitext(fname)[0]
                    audio_map[stem] = os.path.abspath(os.path.join(root, fname))
                    if ext == ".flac":
                        flac_count += 1
                    else:
                        wav_count += 1
    print(
        f"[INFO] Found {len(audio_map)} audio files"
        f" ({wav_count} .wav, {flac_count} .flac) in: {audio_dir}"
    )
    return audio_map


# ============================================================
# 9) Main dataset builder
# ============================================================
def build_wavcaps_dataset(audio_dir, json_dir, output_dir):
    """
    Processes WavCaps audio + JSON metadata into a Tango-compatible dataset.

    Output structure:
        output_dir/
          ├── captions/          (.txt caption files)
          ├── latent_vectors/    (.pt VAE latent tensors)
          ├── mel_spectrograms/  (.png mel images)
          └── original_wavs/     (symlinks -> original wav files)
    """

    # --- Create output directories ---
    dirs = {
        "captions": os.path.join(output_dir, "captions"),
        "latents": os.path.join(output_dir, "latent_vectors"),
        "mels": os.path.join(output_dir, "mel_spectrograms"),
        "wavs": os.path.join(output_dir, "original_wavs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"[INFO] Output directories ready at: {output_dir}")

    # --- Load metadata ---
    df = load_wavcaps_metadata(json_dir)

    # --- Scan audio files ---
    wav_map = scan_audio_folder(audio_dir)

    # --- Filter to entries that have both metadata and audio ---
    matched_ids = set(df["id"].values) & set(wav_map.keys())

    if len(matched_ids) == 0:
        # Diagnostic: show samples from both sides so the user can see the mismatch
        sample_json = list(df["id"].values)[:8]
        sample_audio = list(wav_map.keys())[:8]
        print(f"[DIAG] Sample JSON 'id' values  : {sample_json}")
        print(f"[DIAG] Sample audio file stems  : {sample_audio}")

        # --- Fallback 1: use 'wav' field stem (WavCaps sometimes differs from 'id') ---
        if "wav" in df.columns and len(matched_ids) == 0:
            df["_id_wav"] = df["wav"].apply(
                lambda x: os.path.splitext(os.path.basename(str(x)))[0]
            )
            m = set(df["_id_wav"].values) & set(wav_map.keys())
            if m:
                print(f"[INFO] Fallback 1: matched {len(m)} entries via 'wav' field stem.")
                df["id"] = df["_id_wav"]
                matched_ids = m

        # --- Fallback 2: strip leading 'Y' from JSON ids (AudioSet convention mismatch) ---
        if len(matched_ids) == 0:
            df["_id_strip"] = df["id"].str.replace(r"^Y", "", regex=True)
            m = set(df["_id_strip"].values) & set(wav_map.keys())
            if m:
                print(f"[INFO] Fallback 2: matched {len(m)} entries after stripping leading 'Y' from JSON ids.")
                df["id"] = df["_id_strip"]
                matched_ids = m

        # --- Fallback 3: prepend 'Y' to audio stems (files lack the Y prefix) ---
        if len(matched_ids) == 0:
            wav_map_y = {"Y" + k: v for k, v in wav_map.items()}
            m = set(df["id"].values) & set(wav_map_y.keys())
            if m:
                print(f"[INFO] Fallback 3: matched {len(m)} entries after prepending 'Y' to audio stems.")
                wav_map = wav_map_y
                matched_ids = m

        if len(matched_ids) == 0:
            print(
                "[ERROR] No matching files found after all fallback strategies.\n"
                "        Verify that JSON 'id' (or 'wav') fields match audio filenames (without extension).\n"
                "        See [DIAG] lines above for sample values from each side."
            )
            return

    df_filtered = df[df["id"].isin(matched_ids)].copy()
    print(f"[INFO] Matched {len(df_filtered)} entries (have both JSON metadata and audio file).")

    # --- Setup device ---
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # --- Download / load checkpoint ---
    ckpt_dir = os.path.join(output_dir, ".cache")
    ckpt_path = os.path.join(ckpt_dir, "audioldm-s-full.ckpt")
    download_file(AUDIO_LDM_S_FULL_URL, ckpt_path, min_bytes_ok=100_000_000)

    print("[LOAD] checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Discover scale_factor
    scale_factor = 1.0
    for k in ["scale_factor", "model.scale_factor", "latent_diffusion.scale_factor"]:
        if k in state:
            try:
                scale_factor = (
                    float(state[k].item()) if torch.is_tensor(state[k]) else float(state[k])
                )
                print(f"[INFO] found scale_factor in checkpoint: {k} = {scale_factor}")
                break
            except Exception:
                pass
    print(f"[INFO] using scale_factor: {scale_factor}")

    # --- Build STFT ---
    fn_STFT = TacotronSTFT(
        CONFIG["preprocessing"]["stft"]["filter_length"],
        CONFIG["preprocessing"]["stft"]["hop_length"],
        CONFIG["preprocessing"]["stft"]["win_length"],
        CONFIG["preprocessing"]["mel"]["n_mel_channels"],
        CONFIG["preprocessing"]["audio"]["sampling_rate"],
        CONFIG["preprocessing"]["mel"]["mel_fmin"],
        CONFIG["preprocessing"]["mel"]["mel_fmax"],
    )

    target_length = CONFIG["preprocessing"]["mel"]["target_length"]

    # --- Extract VAE weights then free the full checkpoint before GPU transfer ---
    prefix = "first_stage_model."
    vae_state = {}
    for k, v in state.items():
        if k.startswith(prefix):
            vae_state[k[len(prefix):]] = v

    if not vae_state:
        raise RuntimeError(
            "Checkpoint does not contain 'first_stage_model.*' keys. "
            "Expected an AudioLDM-style checkpoint (audioldm-s-full)."
        )

    del ckpt, state  # free ~2.5 GB before moving model to GPU

    # --- Build VAE ---
    vae_cfg = CONFIG["first_stage_config"]["params"]
    vae = AutoencoderKL(
        ddconfig=vae_cfg["ddconfig"],
        embed_dim=vae_cfg["embed_dim"],
        image_key=vae_cfg["image_key"],
        subband=vae_cfg["subband"],
        scale_factor=scale_factor,
    )
    msg = vae.load_state_dict(vae_state, strict=False)
    del vae_state
    print(f"[LOAD] VAE loaded. missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        vae = vae.to(device).eval()
    except torch.cuda.OutOfMemoryError:
        print("[WARN] CUDA OOM moving VAE to GPU — falling back to CPU")
        device = torch.device("cpu")
        vae = vae.to(device).eval()

    # --- Processing loop ---
    success_count = 0
    skipped_count = 0
    error_count = 0

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Building dataset"):
        try:
            file_id = row["id"]

            # Skip if latent already exists (resumable)
            latent_path = os.path.join(dirs["latents"], f"{file_id}.pt")
            if os.path.exists(latent_path):
                skipped_count += 1
                continue

            caption_text = str(row["caption"])
            src_wav_path = wav_map[file_id]

            # A. Create symlink to original wav
            link_path = os.path.join(dirs["wavs"], f"{file_id}.wav")
            if not os.path.exists(link_path):
                os.symlink(src_wav_path, link_path)

            # B. Save caption
            caption_path = os.path.join(dirs["captions"], f"{file_id}.txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption_text)

            # C. Compute mel spectrogram
            mel, _, _ = wav_to_fbank(src_wav_path, target_length=target_length, fn_STFT=fn_STFT)

            # D. Save mel spectrogram image
            mel_image_path = os.path.join(dirs["mels"], f"{file_id}.png")
            plt.imsave(mel_image_path, mel.cpu().numpy().T, cmap="viridis", origin="lower")

            # E. VAE encode -> latent vector
            x = mel.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                posterior = vae.encode(x)
                z = vae.get_first_stage_encoding(posterior)

            torch.save(z.cpu(), latent_path)
            success_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"[ERROR] {row.get('id', '?')}: {e}")
            continue

    # --- Summary ---
    print("\n" + "=" * 50)
    print("DATASET CREATION COMPLETE")
    print(f"  Processed successfully : {success_count}")
    print(f"  Skipped (already done) : {skipped_count}")
    print(f"  Errors                 : {error_count}")
    print(f"  Output location        : {output_dir}")
    print("=" * 50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 10) CLI entry point
# ============================================================
if __name__ == "__main__":
    # ---- Hardcoded paths (edit these if your layout changes) ----
    AUDIO_DIR  = "/home/yitshag/test_uv/original_data/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac/"
    JSON_DIR   = "/home/yitshag/test_uv/original_data/json_files/AudioSet_SL"
    OUTPUT_DIR = "/home/yitshag/test_uv/new_dataset_wavcaps"
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Create Tango-compatible latent dataset from WavCaps data."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=AUDIO_DIR,
        help="Folder containing WAV/FLAC files (searched recursively).",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default=JSON_DIR,
        help="Folder containing WavCaps JSON metadata file(s).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory.",
    )
    args = parser.parse_args()

    build_wavcaps_dataset(args.audio_dir, args.json_dir, args.output_dir)
