import os
import torch

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Feature extraction
    N_KEYPOINTS = 8192
    IMAGE_SIZE = 1024

    # Pair selection - CRITICAL for memory
    GLOBAL_TOPK = 20  # Reduced from 50 - each image pairs with top 20
    MIN_MATCHES = 10
    RATIO_THR = 1.2

    # Paths
    DINO_MODEL = "facebook/dinov2-base"
    
    # MASt3R - Reduced size for memory
    MAST3R_MODEL = "/kaggle/working/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    MAST3R_IMAGE_SIZE = 224  # Small size to save memory

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
