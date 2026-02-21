"""
VAE Inference - NO ROTATION
Verified Pipeline:
1. Loads Vocoder Weights
2. Uses Slaney Mel Scale
3. Correct Normalization (-4.63, 2.74)
4. NO TRANSPOSE (Fixes geometric distortion)
"""

import numpy as np
import torch
import soundfile as sf
import os
from audioldm.variational_autoencoder.autoencoder import AutoencoderKL
from audioldm.hifigan.utilities import get_vocoder

LATENT_FILE = "./data/tango_dataset/latent-vector/000000.npy"
CKPT_PATH = os.path.expanduser("~/.cache/audioldm/audioldm-s-full.ckpt")

def load_models():
    print("⏳ Loading Models...")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # VAE
    ddconfig = {
        "double_z": True, "z_channels": 8, "resolution": 256, "in_channels": 1, "out_ch": 1,
        "ch": 128, "ch_mult": [1, 2, 4], "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0,
    }
    vae = AutoencoderKL(ddconfig=ddconfig, embed_dim=8)
    vae.load_state_dict({k.replace("first_stage_model.", ""): v for k, v in sd.items() if k.startswith("first_stage_model.")}, strict=False)
    vae.eval()

    # Vocoder (With Weights)
    vocoder = get_vocoder(None, "cpu")
    vocoder_sd = {}
    prefix = "first_stage_model.vocoder."
    for k, v in sd.items():
        if k.startswith(prefix):
            vocoder_sd[k.replace(prefix, "")] = v
    vocoder.load_state_dict(vocoder_sd, strict=True)
    vocoder.eval()
    
    return vae, vocoder

def run_inference():
    if not os.path.exists(LATENT_FILE):
        print("❌ Latent file missing. Run main.py!")
        return

    vae, vocoder = load_models()
    
    # 1. Load Latent [1, 8, 16, 250] (Correct shape now!)
    latent = np.load(LATENT_FILE)
    print(f"📥 Latent Shape: {latent.shape}") 
    # Should be (8, 16, 250) or (16, 250) depends on squeeze. 
    # VAE expects [Batch, 8, 16, 250]
    
    with torch.no_grad():
        z = torch.from_numpy(latent).float()
        if z.dim() == 3: z = z.unsqueeze(0)
        
        # 2. Decode
        print("🔄 Decoding...")
        mel_norm = vae.decode(z) # Output: [1, 1, 64, 1000]
        
        # 3. De-Normalize
        mel = (mel_norm * 2.74) + (-4.63)
        
        # 4. Prepare for Vocoder
        # Current: [1, 1, 64, 1000]
        # Target: [1, 64, 1000]
        # NO TRANSPOSE NEEDED HERE! Just squeeze channel.
        mel = mel.squeeze(1) 
        
        # 5. Vocode
        print("🌊 Vocoding...")
        wav = vocoder(mel).squeeze(1)
        
        # 6. Save
        wav = (np.clip(wav.cpu().numpy(), -1, 1) * 32767).astype("int16")
        sf.write("final_correct.wav", wav[0], 16000)
        print("✅ DONE! Saved to: final_correct.wav")

if __name__ == "__main__":
    run_inference()