import sys
sys.path.append('wild-gaussian-splatting/mast3r/')

import os
import tempfile
import torch
from mast3r.utils.misc import hash_md5
from mast3r.model import AsymmetricMASt3R

DATASET_DIR = "colmap_data"
weights_path = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(CACHE_PATH, exist_ok=True)
EXAMPLE_PATH = os.path.join(CACHE_PATH, 'examples_datasets')
os.makedirs(EXAMPLE_PATH, exist_ok=True)

DEVICE = device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = AsymmetricMASt3R.from_pretrained(weights_path).to(DEVICE)
SILENT = False
