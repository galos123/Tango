#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# 0) Imports
# ============================================================
import os
import math
import json
import torch
import shutil
import requests
import torchaudio
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg') # Forces a non-interactive backend
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn


# In[2]:


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


# In[3]:


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
    # librosa.util.normalize-like behavior (simple L2 norm if needed); here norm=None as in your file
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
    waveform, sr = torchaudio.load(filename)  # Faster!!!
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


# In[4]:


# ===========================================================
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


# In[5]:


# ============================================================
# 4) HiFi-GAN (minimal pieces needed + same architecture)
#    The weights come from the checkpoint via first_stage_model.vocoder.*
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
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
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
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
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
                        k,
                        u,
                        padding=(k - u) // 2,
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
    # mels: (B, n_mels, T)
    wavs = vocoder(mels).squeeze(1)
    return wavs



# In[6]:


# ============================================================
# 5) VAE modules (subset needed for AutoencoderKL)
#    These match the repo structure you pasted.
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

        # IMPORTANT: vocoder weights will be loaded from the checkpoint
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
        # dec: (B,1,T,64) -> (B,64,T)
        dec = dec.squeeze(1).permute(0, 2, 1)
        wav = vocoder_infer(dec, self.vocoder)
        return wav

    # These 2 mimic the scale_factor usage seen in the repo AutoencoderKL helpers
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


# In[7]:


# ============================================================
# 6) Exact config values (from default_audioldm_config) :contentReference[oaicite:3]{index=3}
#    We only need the preprocessing + first_stage_config bits.
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


# In[8]:


# ============================================================
# 7) User inputs
# ============================================================
audio_path = "/home/yitshag/test_uv/data/audiocaps_metadata/train/190.wav"  # <<< change this
out_dir = "./vae_single_test"
os.makedirs(out_dir, exist_ok=True)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

# checkpoint local path
ckpt_path = os.path.join(out_dir, "audioldm-s-full.ckpt")

# download checkpoint (optional)

if not os.path.exists(ckpt_path):
    print("Checkpoint not found. Downloading...")
    # Insert your download function here (e.g., wget or gdown)
    # download_model(url, checkpoint_path)
else:
    print(f"Loading weights from existing local file: {ckpt_path}")
download_file(AUDIO_LDM_S_FULL_URL, ckpt_path, min_bytes_ok=100_000_000)


# In[9]:


# ============================================================
# 8) Load checkpoint
# ============================================================
print("[LOAD] checkpoint:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# Try to discover a scale_factor stored in checkpoint (if present)
scale_factor = 1.0
for k in ["scale_factor", "model.scale_factor", "latent_diffusion.scale_factor"]:
    if k in state:
        try:
            scale_factor = (
                float(state[k].item()) if torch.is_tensor(state[k]) else float(state[k])
            )
            print("[INFO] found scale_factor in checkpoint:", k, "=", scale_factor)
            break
        except Exception:
            pass
print("[INFO] using scale_factor:", scale_factor)


# In[10]:


# ============================================================
# 9) Build STFT + compute mel (exact pipeline path)
# ============================================================
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

mel, _, _ = wav_to_fbank(audio_path, target_length=target_length, fn_STFT=fn_STFT)
# mel returned as (T, 64) torch.FloatTensor
mel_2d = mel.cpu().numpy()

# Save original mel as image
plt.figure()
plt.imshow(mel_2d.T, aspect="auto", origin="lower")
plt.title("Original mel (fbank)")
plt.tight_layout()
orig_mel_img = os.path.join(out_dir, "mel_original.png")
plt.savefig(orig_mel_img, dpi=200)
plt.close()
print("[SAVE]", orig_mel_img)

# Prepare VAE input shape: (B,1,T,64)
x = mel.unsqueeze(0).unsqueeze(0).to(device)



# In[11]:


# ============================================================
# 10) Build VAE and load *only* first_stage_model weights from checkpoint
# ============================================================
vae_cfg = CONFIG["first_stage_config"]["params"]
vae = (
    AutoencoderKL(
        ddconfig=vae_cfg["ddconfig"],
        embed_dim=vae_cfg["embed_dim"],
        image_key=vae_cfg["image_key"],
        subband=vae_cfg["subband"],
        scale_factor=scale_factor,
    )
    .to(device)
    .eval()
)

# Extract sub-keys: "first_stage_model.*" and load into vae
prefix = "first_stage_model."
vae_state = {}
missing_prefix = True
for k, v in state.items():
    if k.startswith(prefix):
        missing_prefix = False
        vae_state[k[len(prefix) :]] = v

if missing_prefix:
    raise RuntimeError(
        "Checkpoint state_dict does not contain keys starting with 'first_stage_model.'. "
        "This script expects an AudioLDM-style checkpoint (audioldm-s-full)."
    )

msg = vae.load_state_dict(vae_state, strict=False)
print("[LOAD] VAE load_state_dict strict=False")
print("  missing keys:", len(msg.missing_keys))
print("  unexpected keys:", len(msg.unexpected_keys))


# In[12]:


# ============================================================
# 11) VAE round-trip: encode -> latent -> decode -> mel image
# ============================================================
with torch.no_grad():
    posterior = vae.encode(x)
    z = vae.get_first_stage_encoding(posterior)  # scaled latent
    x_rec = vae.decode_first_stage(z)  # back to mel (B,1,T,64)

x_rec_2d = x_rec.squeeze(0).squeeze(0).detach().cpu().numpy()

plt.figure()
plt.imshow(x_rec_2d.T, aspect="auto", origin="lower")
plt.title("Reconstructed mel (VAE decode)")
plt.tight_layout()
rec_mel_img = os.path.join(out_dir, "mel_reconstructed.png")
plt.savefig(rec_mel_img, dpi=200)
plt.close()
print("[SAVE]", rec_mel_img)


# In[13]:


# ============================================================
# 12) Decode mel -> waveform via vocoder inside VAE, save wav
# ============================================================
with torch.no_grad():
    wav_rec = vae.decode_to_waveform(x_rec)  # (B, T_wav)
    wav_rec = wav_rec.detach().cpu()

# Save reconstructed audio (16kHz)
rec_wav_path = os.path.join(out_dir, "audio_reconstructed.wav")
torchaudio.save(rec_wav_path, wav_rec[:1, :], 16000)
print("[SAVE]", rec_wav_path)
print("\nDONE.")
print("Outputs in:", out_dir)


# In[14]:


# ============================================================
# Imports for Dataset Creation
# ============================================================

def build_tango_dataset(wav_source_dir, csv_metadata_path, output_base_dir):
    """
    Optimized version for speed using plt.imsave and smart skipping.
    
    Structure created:
    output_base_dir/
      └── tango-dataset/
          ├── latent_vectors/    (saved .pt tensors)
          ├── mel_spectrograms/  (saved .png images)
          ├── original_wavs/     (copied .wav files)
          └── captions/          (saved .txt files)
    """
    
    # 1. Define Output Directories
    dataset_root = os.path.join(output_base_dir, "tango-dataset")
    dirs = {
        "latents": os.path.join(dataset_root, "latent_vectors"),
        "mels": os.path.join(dataset_root, "mel_spectrograms"),
        "wavs": os.path.join(dataset_root, "original_wavs"),
        "captions": os.path.join(dataset_root, "captions")
    }
    
    # Create directories if they don't exist
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    print(f"[INFO] Dataset directories ready at: {dataset_root}")

    # 2. Load Metadata and Filter
    try:
        df = pd.read_csv(csv_metadata_path)
        # Ensure ID is treated as a string
        df['audiocap_id'] = df['audiocap_id'].astype(str)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return

    print(f"[INFO] Scanning {wav_source_dir}...")
    
    # Get set of available files (optimized for lookup speed)
    available_files = set(f for f in os.listdir(wav_source_dir) if f.lower().endswith('.wav'))
    
    # Extract IDs from filenames (remove .wav extension)
    available_ids = set(os.path.splitext(f)[0] for f in available_files)
    
    # Filter DataFrame to include only files that exist on disk
    df_filtered = df[df['audiocap_id'].isin(available_ids)].copy()
    
    print(f"[INFO] Processing {len(df_filtered)} files (found in both CSV and folder).")

    # 3. Processing Loop
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    # Target length for Mel Spectrogram (From Global Config)
    target_length = CONFIG["preprocessing"]["mel"]["target_length"]

    # Iterate with progress bar
    for index, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing"):
        try:
            audiocap_id = row['audiocap_id']
            
            # --- Optimization: Check if output already exists ---
            # If the latent file exists, we assume this file is done.
            latent_path = os.path.join(dirs["latents"], f"{audiocap_id}.pt")
            if os.path.exists(latent_path):
                skipped_count += 1
                continue

            caption_text = str(row['caption'])
            filename = f"{audiocap_id}.wav"
            src_wav_path = os.path.join(wav_source_dir, filename)
            
            # --- A. Save Original Wav (Copy is fast) ---
            dst_wav_path = os.path.join(dirs["wavs"], filename)
            if not os.path.exists(dst_wav_path):
                # shutil.copy is slightly faster than copy2
                shutil.copy(src_wav_path, dst_wav_path) 

            # --- B. Save Caption ---
            caption_path = os.path.join(dirs["captions"], f"{audiocap_id}.txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption_text)

            # --- C. Process Audio (The heavy lifting) ---
            # mel shape: (T, 64)
            mel, _, _ = wav_to_fbank(src_wav_path, target_length=target_length, fn_STFT=fn_STFT)
            
            # --- D. Save Mel Image (OPTIMIZED) ---
            # plt.imsave is much faster than plt.figure + plt.imshow + plt.savefig
            mel_cpu = mel.cpu().numpy()
            mel_image_path = os.path.join(dirs["mels"], f"{audiocap_id}.png")
            
            # Direct save of the array as an image with colormap 'viridis'
            # origin='lower' ensures low frequencies are at the bottom
            plt.imsave(mel_image_path, mel_cpu.T, cmap='viridis', origin='lower')

            # --- E. VAE Encoding (GPU) ---
            # Reshape for VAE: (Batch, Channel, Time, Freq) -> (1, 1, T, 64)
            x = mel.unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                posterior = vae.encode(x)
                # Sample (or take mode) and scale the latent
                z = vae.get_first_stage_encoding(posterior) 
            
            # Save latent vector
            torch.save(z.cpu(), latent_path)
            success_count += 1

        except Exception as e:
            error_count += 1
            # print(f"[ERROR] {audiocap_id}: {e}") # Uncomment to debug specific files
            continue

    print("="*40)
    print("PROCESSING COMPLETE")
    print(f"Processed successfully: {success_count}")
    print(f"Skipped (Already existed): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Output location: {dataset_root}")
    print("="*40)
    
    # Optional: Clear GPU memory after large batch processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# In[15]:


# ============================================================
# Execution Configuration
# ============================================================

# 1. Path to the folder containing your subset of .wav files
INPUT_WAV_FOLDER = "/home/yitshag/test_uv/data/audiocaps_metadata/train" 

# 2. Path to your metadata.csv file
INPUT_CSV_PATH = "/home/yitshag/test_uv/data/audiocaps_metadata/train.csv"

# 3. Path where you want the 'tango-dataset' folder to be created
OUTPUT_DIR = "./output_data"

# Run the function
if __name__ == "__main__":
    # Ensure dependencies are loaded
    if 'vae' not in globals() or 'fn_STFT' not in globals():
        print("ERROR: VAE model or STFT function not found. Please run the setup code first.")
    else:
        build_tango_dataset(INPUT_WAV_FOLDER, INPUT_CSV_PATH, OUTPUT_DIR)


# In[18]:


#debug single file decode
latent_path = "/home/yitshag/test_uv/output_data/tango-dataset/latent_vectors/196.pt"  # Change to your actual file path
debug_dir = "./last_debug"
os.makedirs(debug_dir, exist_ok=True)
print("hi")
# 1. Load the Latent Vector
z = torch.load(latent_path, map_location=device)
if z.dim() == 3: 
    z = z.unsqueeze(0)  # Ensure batch dimension exists
print("hi")

# 2. Decode
with torch.no_grad():
    mel_rec = vae.decode_first_stage(z)      # Latent -> Mel Spectrogram
    wav_rec = vae.decode_to_waveform(mel_rec) # Mel -> Audio Waveform

print("hi")
# 3. Save Outputs
# Save Mel Spectrogram as image
plt.imsave(
    os.path.join(debug_dir, "debug_mel.png"), 
    mel_rec.squeeze().cpu().numpy().T, 
    cmap='viridis', 
    origin='lower'
)
print("hi")

# Save Waveform as .wav file
torchaudio.save(
    os.path.join(debug_dir, "debug_audio.wav"), 
    wav_rec.cpu(), 
    16000
)


# In[ ]:





# In[ ]:




