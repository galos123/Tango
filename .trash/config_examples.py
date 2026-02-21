"""
Configuration Examples for main.py
Copy the configuration you need and paste it into main.py
"""

# ============================================================================
# EXAMPLE 1: QUICK TEST (100 samples)
# Perfect for testing the pipeline before full processing
# ============================================================================
"""
ROOT_DIR = "./data"
OUTPUT_DIR = "./data/tango_dataset"
SUBSETS = ["train"]
MAX_SAMPLES = 100
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
MAX_DURATION = 10.0
DOWNLOAD_AUDIO = True
N_DOWNLOAD_JOBS = 8
AUDIO_FORMAT = "wav"
"""

# ============================================================================
# EXAMPLE 2: SMALL DATASET (1000 samples per subset)
# Good for prototyping and development
# ============================================================================
"""
ROOT_DIR = "./data"
OUTPUT_DIR = "./data/tango_dataset"
SUBSETS = ["train", "val", "test"]
MAX_SAMPLES = 1000
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
MAX_DURATION = 10.0
DOWNLOAD_AUDIO = True
N_DOWNLOAD_JOBS = 16
AUDIO_FORMAT = "wav"
"""

# ============================================================================
# EXAMPLE 3: FULL DATASET (All samples)
# For final training - this will take several hours
# ============================================================================
"""
ROOT_DIR = "./data"
OUTPUT_DIR = "./data/tango_dataset"
SUBSETS = ["train", "val", "test"]
MAX_SAMPLES = None  # Process all samples
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
MAX_DURATION = 10.0
DOWNLOAD_AUDIO = True
N_DOWNLOAD_JOBS = 16
AUDIO_FORMAT = "wav"
"""

# ============================================================================
# EXAMPLE 4: PROCESS ONLY (Audio already downloaded)
# Use this if you already have AudioCaps downloaded
# ============================================================================
"""
ROOT_DIR = "./data"
OUTPUT_DIR = "./data/tango_dataset"
SUBSETS = ["train", "val", "test"]
MAX_SAMPLES = None
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
MAX_DURATION = 10.0
DOWNLOAD_AUDIO = False  # Skip download step
N_DOWNLOAD_JOBS = 8
AUDIO_FORMAT = "wav"
"""

# ============================================================================
# EXAMPLE 5: HIGH-QUALITY AUDIO (Custom audio settings)
# For experimenting with different audio parameters
# ============================================================================
"""
ROOT_DIR = "./data"
OUTPUT_DIR = "./data/tango_dataset_highres"
SUBSETS = ["train"]
MAX_SAMPLES = 1000
SAMPLE_RATE = 22050  # Higher sample rate
N_MELS = 128         # More mel bins
N_FFT = 2048         # Larger FFT window
HOP_LENGTH = 256     # Larger hop
MAX_DURATION = 10.0
DOWNLOAD_AUDIO = False
N_DOWNLOAD_JOBS = 8
AUDIO_FORMAT = "wav"
"""

# ============================================================================
# EXAMPLE 6: VALIDATION SET ONLY
# For creating a small validation dataset quickly
# ============================================================================
"""
ROOT_DIR = "./data"
OUTPUT_DIR = "./data/tango_dataset_val"
SUBSETS = ["val"]
MAX_SAMPLES = None  # Get all validation samples
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
MAX_DURATION = 10.0
DOWNLOAD_AUDIO = True
N_DOWNLOAD_JOBS = 8
AUDIO_FORMAT = "wav"
"""

# ============================================================================
# HOW TO USE:
# ============================================================================
# 1. Copy the configuration block you want (including the triple quotes)
# 2. Open main.py
# 3. Find the CONFIGURATION section
# 4. Replace the existing configuration with your chosen one
# 5. Remove the triple quotes (""")
# 6. Run: python main.py
