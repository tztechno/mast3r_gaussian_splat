# ============================================================================
# Main Pipeline
# ============================================================================

# ===== standard =====

import os
import shutil

# ===== pipeline steps =====

from pipeline.step01_setup import setup_base_environment, clear_memory
from pipeline.step01_setup import run as setup_run

from pipeline.step02_biplet import normalize_image_sizes_biplet
from pipeline.step03_dino import get_image_pairs_dino
from pipeline.step04_mast3r import (
    load_mast3r_model,
    run_mast3r_pairs
)
from pipeline.step05_process1 import extract_colmap_data, save_colmap_reconstruction
from pipeline.step06_gaussiansplat import train_gaussian_splatting

cfg = {}
cfg = setup_run(cfg)

from .utils import clear_memory, get_memory_info
from .config import Config

# ==========================




def main_pipeline(image_dir, output_dir, square_size=224, iterations=2000, 
                 max_images=None, max_pairs=10000, max_points=1000000):
    """
    Main pipeline for DINO matching -> MASt3R -> Gaussian Splatting
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory for output files
        square_size: Size to resize images for processing
        iterations: Number of training iterations
        max_images: Maximum number of images to process (None = all)
        max_pairs: Maximum number of image pairs for matching
        max_points: Maximum number of 3D points to extract (default: 1M)
    """
    os.makedirs(output_dir, exist_ok=True)

    #setup_base_environment()
    #clear_memory()
    
    #setup_mast3r()
    #clear_memory()
    
    #setup_gaussian_splatting()
    clear_memory()
    
    # Step 1: Normalize images to biplet-square format
    print("\n" + "="*70)
    print("Step 1: Biplet-Square Normalization")
    print("="*70)
    
    processed_image_dir = os.path.join(output_dir, "processed_images")
    
    # Get original images first
    original_image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Limit original images if specified
    if max_images and len(original_image_paths) > max_images:
        print(f"\n⚠️  Limiting to {max_images} original images")
        original_image_paths = original_image_paths[:max_images]
    
    print(f"Processing {len(original_image_paths)} original images → ~{len(original_image_paths)*2} after biplet-square")
    
    # Only process the selected images
    temp_dir = os.path.join(output_dir, "temp_originals")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy selected images to temp directory
    for img_path in original_image_paths:
        import shutil
        shutil.copy(img_path, temp_dir)
    
    # Process the temp directory
    normalize_image_sizes_biplet(
        input_dir=temp_dir,
        output_dir=processed_image_dir,
        size=square_size
    )
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    # Get processed image paths
    image_paths = sorted([
        os.path.join(processed_image_dir, f)
        for f in os.listdir(processed_image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    print(f"\n📸 Processing {len(image_paths)} images (after biplet-square)")
    print(f"⚠️  Will use maximum {max_pairs} pairs to save memory")
    
    # Step 2: DINO-based pair selection
    print("\n" + "="*70)
    print("Step 2: DINO Pair Selection")
    print("="*70)
    
    pairs = get_image_pairs_dino(image_paths, max_pairs=max_pairs)
    clear_memory()
    
    print(f"✓ Using {len(pairs)} pairs for reconstruction")
    
    # Step 3: MASt3R reconstruction
    print("\n" + "="*70)
    print("Step 3: MASt3R Reconstruction")
    print("="*70)
    
    device = Config.DEVICE
    model = load_mast3r_model(device)
    
    scene, mast3r_images = run_mast3r_pairs(
        model, image_paths, pairs, device,
        max_pairs=None  # Already limited in get_image_pairs_dino
    )
    
    # Clear model from memory
    del model
    clear_memory()
    
    # Step 4: Extract COLMAP-compatible data
    print("\n" + "="*70)
    print("Step 4: Converting to COLMAP Format")
    print("="*70)
    
    # Extract COLMAP-compatible data with point limit
    pts3d, colors, cameras, poses = extract_colmap_data(
        scene, image_paths, max_points=max_points  
    )

    # Clear scene from memory
    del scene, mast3r_images
    clear_memory()
    
    # Step 5: Save COLMAP reconstruction
    colmap_dir = os.path.join(output_dir, 'colmap')
    sparse_dir = save_colmap_reconstruction(
        pts3d, colors, cameras, poses, image_paths, colmap_dir
    )
    
    # Clear reconstruction data
    del pts3d, colors, cameras, poses
    clear_memory()
    
    # Step 6: Train Gaussian Splatting
    print("\n" + "="*70)
    print("Step 6: Training Gaussian Splatting")
    print("="*70)
    
    gs_output = train_gaussian_splatting(
        colmap_dir=colmap_dir,
        image_dir=processed_image_dir,
        output_dir=output_dir,
        iterations=iterations
    )
    
    print("\n" + "="*70)
    print("✅ Full Pipeline Successfully Completed!")
    print("="*70)
    print(f"\nGaussian Splatting model saved at: {gs_output}")
    
    return gs_output


if __name__ == "__main__":
    IMAGE_DIR = "/kaggle/input/two-dogs/bike15"
    OUTPUT_DIR = "/kaggle/working/output"
    
    gs_output = main_pipeline(
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        square_size=1024,  
        iterations=1000,   
        max_images=30,
        max_pairs=1000,     
        max_points=1000000        
    )

    print(f"\n{'='*70}")
    print("Pipeline completed successfully!")
    print(f"{'='*70}")
    print(f"Gaussian Splatting output: {gs_output}")
