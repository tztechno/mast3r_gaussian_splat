# ============================================================================
# Step 2: MASt3R Reconstruction (REPLACES ALIKED/LIGHTGLUE/COLMAP)
# ============================================================================
import os
from .utils import clear_memory, get_memory_info
from .config import Config

def load_mast3r_model(device='cuda'):
    """Load MASt3R model"""
    from mast3r.model import AsymmetricMASt3R
    
    model = AsymmetricMASt3R.from_pretrained(Config.MAST3R_MODEL).to(device)
    model.eval()
    
    print(f"✓ MASt3R model loaded on {device}")
    return model

def load_images_for_mast3r(image_paths, size=224):
    """Load images using DUSt3R's format with reduced size"""
    print(f"\n=== Loading images for MASt3R (size={size}) ===")
    
    from dust3r.utils.image import load_images
    
    # Load images using DUSt3R's loader with reduced size
    images = load_images(image_paths, size=size, verbose=True)
    
    return images

def run_mast3r_pairs(model, image_paths, pairs, device='cuda', batch_size=1, max_pairs=None):
    """Run MASt3R on selected pairs with memory management"""
    print("\n=== Running MASt3R Reconstruction ===")
    print("Initial memory state:")
    get_memory_info()
    
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    
    # Limit number of pairs if specified
    if max_pairs and len(pairs) > max_pairs:
        print(f"Limiting pairs from {len(pairs)} to {max_pairs}")
        # Select pairs more evenly distributed
        step = max(1, len(pairs) // max_pairs)
        pairs = pairs[::step][:max_pairs]
    
    print(f"Processing {len(pairs)} pairs...")
    
    # Load images in smaller size
    print(f"Loading {len(image_paths)} images at {Config.MAST3R_IMAGE_SIZE}x{Config.MAST3R_IMAGE_SIZE}...")
    images = load_images_for_mast3r(image_paths, size=Config.MAST3R_IMAGE_SIZE)
    
    print(f"Loaded {len(images)} images")
    print("After loading images:")
    get_memory_info()
    
    # Create all image pairs at once
    print(f"Creating {len(pairs)} image pairs...")
    mast3r_pairs = []
    for idx1, idx2 in tqdm(pairs, desc="Preparing pairs"):
        mast3r_pairs.append((images[idx1], images[idx2]))
    
    print(f"Running MASt3R inference on {len(mast3r_pairs)} pairs...")
    
    # Run inference (this returns the dict format we need)
    output = inference(mast3r_pairs, model, device, batch_size=batch_size, verbose=True)
    
    # Clear pairs from memory
    del mast3r_pairs
    clear_memory()
    
    print("✓ MASt3R inference complete")
    print("After inference:")
    get_memory_info()
    
    # Global alignment
    print("Running global alignment...")
    scene = global_aligner(
        output, 
        device=device, 
        mode=GlobalAlignerMode.PointCloudOptimizer
    )
    
    # Clear output after creating scene
    del output
    clear_memory()
    
    print("Computing global alignment...")
    loss = scene.compute_global_alignment(
        init="mst", 
        niter=150,  # Reduced from 300
        schedule='cosine', 
        lr=0.01
    )
    
    print(f"✓ Global alignment complete (final loss: {loss:.6f})")
    print("Final memory state:")
    get_memory_info()
    
    return scene, images



def run(cfg):
    load_mast3r_model(device='cuda')
    load_images_for_mast3r(image_paths, size=224)
    run_mast3r_pairs(model, image_paths, pairs, device='cuda', batch_size=1, max_pairs=None)
    return cfg
