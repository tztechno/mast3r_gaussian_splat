# MASt3R-based Gaussian Splatting Pipeline
# Preserves: DINO pair selection + Biplet-Square Normalization
# Replaces: ALIKED/LightGlue/COLMAP with MASt3R

from .utils import clear_memory, get_memory_info
from .config import Config

import os
import sys
import gc
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import subprocess
from PIL import Image, ImageFilter
import struct

# Transformers for DINO
from transformers import AutoImageProcessor, AutoModel



# ============================================================================
# Environment Setup
# ============================================================================

def run_cmd(cmd, check=True, capture=False):
    """Run command with better error handling"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"❌ Command failed with code {result.returncode}")
        if capture:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
    return result


def setup_base_environment():
    """Setup base Python environment"""
    print("\n=== Setting up Base Environment ===")
    
    # NumPy fix for Python 3.12
    print("\n📦 Fixing NumPy...")
    run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
    run_cmd([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
    
    # PyTorch
    print("\n📦 Installing PyTorch...")
    run_cmd([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio"
    ])
    
    # Core utilities
    print("\n📦 Installing core utilities...")
    run_cmd([
        sys.executable, "-m", "pip", "install",
        "opencv-python",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "plyfile",
        "tqdm",
        "tensorboard",
        "scipy",  # for rotation conversions and image resizing
        "psutil"  # for memory monitoring
    ])
    
    # Transformers for DINO
    print("\n📦 Installing transformers...")
    run_cmd([
        sys.executable, "-m", "pip", "install",
        "transformers==4.40.0"
    ])
    
    # pycolmap for COLMAP format
    print("\n📦 Installing pycolmap...")
    run_cmd([sys.executable, "-m", "pip", "install", "pycolmap"])
    
    print("✓ Base environment setup complete!")


def setup_mast3r():
    """Install and setup MASt3R"""
    print("\n=== Setting up MASt3R ===")
    
    os.chdir('/kaggle/working')
    
    # Remove existing installation
    if os.path.exists('mast3r'):
        print("Removing existing MASt3R installation...")
        os.system('rm -rf mast3r')
    
    # Clone repository
    print("Cloning MASt3R repository...")
    os.system('git clone --recursive https://github.com/naver/mast3r')
    os.chdir('/kaggle/working/mast3r')
    
    # Check dust3r directory
    print("Checking dust3r structure...")
    os.system('ls -la dust3r/')
    
    # Install dust3r
    print("Installing dust3r...")
    os.system('cd dust3r && python -m pip install -e .')
    
    # Install croco
    print("Installing croco...")
    os.system('cd dust3r/croco && python -m pip install -e .')
    
    # Install requirements
    print("Installing MASt3R requirements...")
    os.system('pip install -r requirements.txt')
    
    # Download model weights
    print("Downloading model weights...")
    os.system('mkdir -p checkpoints')
    os.system('wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    
    # Install additional dependencies
    print("Installing additional dependencies...")
    os.system('pip install trimesh matplotlib roma')
    
    # Add to path
    sys.path.insert(0, '/kaggle/working/mast3r')
    sys.path.insert(0, '/kaggle/working/mast3r/dust3r')
    
    # Verification
    print("\n🔍 Verifying MASt3R installation...")
    try:
        from mast3r.model import AsymmetricMASt3R
        print("  ✓ MASt3R import: OK")
    except Exception as e:
        print(f"  ❌ MASt3R import failed: {e}")
        raise
    
    print("✓ MASt3R setup complete!")


def setup_gaussian_splatting():
    """Setup Gaussian Splatting"""
    print("\n=== Setting up Gaussian Splatting ===")
    
    os.chdir('/kaggle/working')
    
    WORK_DIR = "gaussian-splatting"
    
    if not os.path.exists(WORK_DIR):
        print("Cloning Gaussian Splatting repository...")
        run_cmd([
            "git", "clone", "--recursive",
            "https://github.com/graphdeco-inria/gaussian-splatting.git",
            WORK_DIR
        ])
    else:
        print("✓ Repository already exists")
    
    os.chdir(WORK_DIR)
    
    # Install requirements
    print("Installing Gaussian Splatting requirements...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Build submodules
    print("\n📦 Building Gaussian Splatting submodules...")
    
    submodules = {
        "diff-gaussian-rasterization":
            "https://github.com/graphdeco-inria/diff-gaussian-rasterization.git",
        "simple-knn":
            "https://github.com/camenduru/simple-knn.git"
    }
    
    for name, repo in submodules.items():
        print(f"\n📦 Installing {name}...")
        path = os.path.join("submodules", name)
        if not os.path.exists(path):
            run_cmd(["git", "clone", repo, path])
        run_cmd([sys.executable, "-m", "pip", "install", path])
    
    print("✓ Gaussian Splatting setup complete!")



def run(cfg):
    setup_base_environment()
    setup_mast3r()
    setup_gaussian_splatting()
    return cfg
