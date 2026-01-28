"""
CORRECTED main_pipeline_process3 function
==========================================
This version correctly calls convert_mast3r_to_colmap_standalone without colmap_utils_path
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import subprocess
import shutil
from standalone_colmap_converter import convert_mast3r_to_colmap_standalone


def main_pipeline_process3(
    image_dir: str,
    output_dir: str,
    square_size: int = 512,
    iterations: int = 30000,
    max_images = None,
    max_pairs: int = 100000,
    min_conf_thr: float = 2.0,
    clean_depth: bool = False,
    mask_images: bool = True,
    colmap_utils_path = None,  # IGNORED - kept for backward compatibility
    verbose: bool = True
):
    """
    Complete pipeline using Process 3 with standalone COLMAP converter.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Output directory for results
        square_size: Size for image resizing
        iterations: Number of training iterations
        max_images: Maximum number of images to process
        max_pairs: Maximum number of image pairs
        min_conf_thr: Minimum confidence threshold
        clean_depth: Whether to clean depth maps
        mask_images: Whether to save confidence masks
        colmap_utils_path: IGNORED (kept for backward compatibility)
        verbose: Verbose output
        
    Returns:
        Trained Gaussian Splatting model or None if failed
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    mast3r_dir = os.path.join(output_dir, "mast3r_output")
    colmap_dir = os.path.join(output_dir, "colmap_output")
    gs_dir = os.path.join(output_dir, "gaussian_splatting")
    
    print("=" * 70)
    print("Process 3: MASt3R -> COLMAP -> Gaussian Splatting (Standalone)")
    print("=" * 70)
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Square size: {square_size}")
    print(f"Max images: {max_images}")
    print(f"Confidence threshold: {min_conf_thr}")
    if colmap_utils_path:
        print(f"Note: colmap_utils_path is ignored (using standalone converter)")
    print("=" * 70)
    
    # Step 1: Load and prepare images
    print("\n" + "=" * 70)
    print("Step 1: Loading Images")
    print("=" * 70)
    
    try:
        from PIL import Image
        import glob
        
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]g")))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            raise ValueError("No images found in directory")
        
        # Load images
        images = []
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            # Resize if needed
            if square_size:
                img = img.resize((square_size, square_size), Image.Resampling.LANCZOS)
            images.append(np.array(img))
        
        print(f"✓ Loaded {len(images)} images")
        
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 2: Run MASt3R reconstruction
    print("\n" + "=" * 70)
    print("Step 2: Running MASt3R Reconstruction")
    print("=" * 70)
    
    try:
        # Check if MASt3R is available
        try:
            from mast3r.model import AsymmetricMASt3R
            from mast3r.fast_nn import fast_reciprocal_NNs
            import mast3r.utils.path_to_dust3r
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        except ImportError as e:
            print(f"❌ MASt3R not available: {e}")
            print("Please install MASt3R first")
            return None
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        
        # Prepare images for MASt3R
        mast3r_images = [(img, None, img_path) for img, img_path in zip(images, image_files)]
        
        # Create pairs
        pairs = make_pairs(
            mast3r_images,
            scene_graph='complete',
            prefilter=None,
            symmetrize=True
        )
        
        if max_pairs and len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        
        print(f"Processing {len(pairs)} image pairs")
        
        # Run inference
        output = inference(pairs, model, device, batch_size=1, verbose=verbose)
        
        # Global alignment
        scene = global_aligner(
            output,
            device=device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=verbose
        )
        
        # Optimize
        loss = scene.compute_global_alignment(
            init='mst',
            niter=300,
            schedule='cosine',
            lr=0.01
        )
        
        print(f"✓ MASt3R reconstruction completed (loss: {loss:.4f})")
        
    except Exception as e:
        print(f"❌ Error during MASt3R reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 3: Convert to COLMAP format (STANDALONE VERSION)
    print("\n" + "=" * 70)
    print("Step 3: Converting to COLMAP Format (Standalone)")
    print("=" * 70)
    print(f"Confidence threshold: {min_conf_thr}")
    print(f"Clean depth: {clean_depth}")
    print(f"Save masks: {mask_images}")
    print("-" * 70)
    
    try:
        # CORRECTED: No colmap_utils_path argument!
        colmap_output = convert_mast3r_to_colmap_standalone(
            scene=scene,
            output_dir=colmap_dir,
            min_conf_thr=min_conf_thr,
            clean_depth=clean_depth,
            mask_images=mask_images,
            verbose=verbose
        )
        
        print(f"✓ COLMAP conversion completed")
        print(f"  Output: {colmap_output}")
        
    except Exception as e:
        print(f"❌ Error during COLMAP conversion: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 4: Train Gaussian Splatting
    print("\n" + "=" * 70)
    print("Step 4: Training Gaussian Splatting")
    print("=" * 70)
    print(f"Iterations: {iterations}")
    print("-" * 70)
    
    try:
        # Check if gaussian-splatting is available
        gs_available = False
        
        # Try to import gaussian_splatting
        try:
            import gaussian_splatting
            gs_available = True
            print("Using gaussian_splatting Python package")
        except ImportError:
            # Try to find gaussian-splatting executable
            gs_train = shutil.which("gaussian_splatting_train")
            if gs_train:
                gs_available = True
                print(f"Using gaussian-splatting executable: {gs_train}")
        
        if not gs_available:
            print("⚠️  Gaussian Splatting not available, skipping training")
            print("COLMAP output is ready for manual processing")
            print(f"You can use the COLMAP data at: {colmap_output}")
            return None
        
        # Train the model
        if 'gaussian_splatting' in sys.modules:
            # Use Python API
            from gaussian_splatting import train
            
            model = train(
                source_path=colmap_output,
                model_path=gs_dir,
                iterations=iterations,
                test_iterations=[],
                save_iterations=[iterations],
                checkpoint_iterations=[],
                quiet=not verbose
            )
        else:
            # Use command-line interface
            cmd = [
                gs_train,
                "-s", colmap_output,
                "-m", gs_dir,
                "--iterations", str(iterations),
                "--test_iterations", str(iterations),
                "--save_iterations", str(iterations)
            ]
            
            if not verbose:
                cmd.append("--quiet")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Gaussian Splatting training failed")
                print(f"Error: {result.stderr}")
                return None
            
            model = None
        
        print(f"✓ Gaussian Splatting training completed")
        print(f"  Model saved to: {gs_dir}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error during Gaussian Splatting training: {e}")
        import traceback
        traceback.print_exc()
        return None


# Test that the function is loaded
print("✓ Corrected main_pipeline_process3 loaded successfully")
print("✓ Ready to use without colmap_utils_path errors")
