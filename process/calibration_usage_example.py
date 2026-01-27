"""
Example: How to use the calibration test with your existing code

This shows how to integrate the calibration test into your MASt3R pipeline
to verify that 3D coordinates are not being altered during COLMAP conversion.
"""

# ============================================================================
# STEP 1: Import the calibration test modules
# ============================================================================
from test_calibration_pipeline import test_calibration_pipeline, print_scene_coordinates
from calibration_data_generator import CalibrationDataGenerator, verify_transformation


# ============================================================================
# STEP 2: Your existing COLMAP conversion function
# ============================================================================
def mast3r_to_colmap(scene, images, output_path, masks=None):
    """
    Your existing function that converts MASt3R scene to COLMAP format
    
    This is just a placeholder - replace with your actual implementation
    """
    from pathlib import Path
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Example structure - replace with your actual code
    # Extract 3D points from scene
    pts3d = []
    for i in range(len(images)):
        pts_view = scene.get_pts3d(i)
        if torch.is_tensor(pts_view):
            pts_view = pts_view.cpu().numpy()
        pts3d.append(pts_view)
    
    # Convert to COLMAP format
    # ... your conversion code here ...
    
    # Write COLMAP files
    # cameras.txt
    with open(output_path / 'cameras.txt', 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # ... write camera data ...
    
    # images.txt
    with open(output_path / 'images.txt', 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        # ... write image data ...
    
    # points3D.txt
    with open(output_path / 'points3D.txt', 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        # ... write point data ...
    
    print(f"COLMAP data written to {output_path}")


# ============================================================================
# STEP 3: Run the calibration test
# ============================================================================
def run_calibration_test():
    """
    Run a calibration test to verify coordinate transformations
    """
    print("\n" + "="*80)
    print("RUNNING CALIBRATION TEST")
    print("="*80)
    
    # Run the test with your conversion function
    success = test_calibration_pipeline(
        colmap_converter_func=mast3r_to_colmap,
        output_dir='./calibration_test_results'
    )
    
    if success:
        print("\n✓ Calibration test completed successfully")
        print("Check the output for coordinate comparison results")
    else:
        print("\n❌ Calibration test failed")
        print("Check the error messages above")


# ============================================================================
# STEP 4: Alternative - Create calibration data for manual testing
# ============================================================================
def create_calibration_data_only():
    """
    Just create the calibration data without running conversion
    Useful if you want to manually test your pipeline
    """
    print("Creating calibration scene data...")
    
    generator = CalibrationDataGenerator(num_views=6, num_points=100)
    scene = generator.create_mock_scene(device='cpu')
    
    # Save ground truth
    generator.save_ground_truth(scene, './calibration_data')
    
    # Print coordinate details
    print_scene_coordinates(scene, './calibration_data/scene_coordinates.txt')
    
    print("\n✓ Calibration data created")
    print("You can now manually use 'scene' in your pipeline")
    print("After COLMAP conversion, run:")
    print("  verify_transformation('./calibration_data', './your_colmap_output')")
    
    return scene


# ============================================================================
# STEP 5: Verify existing COLMAP output
# ============================================================================
def verify_existing_output(ground_truth_dir, colmap_output_dir):
    """
    Verify an existing COLMAP output against ground truth
    """
    print(f"\nVerifying COLMAP output...")
    print(f"Ground truth: {ground_truth_dir}")
    print(f"COLMAP output: {colmap_output_dir}")
    
    verify_transformation(ground_truth_dir, colmap_output_dir)


# ============================================================================
# MAIN: Choose what to run
# ============================================================================
if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("MASt3R to COLMAP Calibration Test")
    print("="*80)
    print("\nOptions:")
    print("  1. Run full calibration test (recommended)")
    print("  2. Create calibration data only")
    print("  3. Verify existing COLMAP output")
    print()
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Enter choice (1-3) [1]: ").strip() or "1"
    
    if choice == "1":
        run_calibration_test()
    
    elif choice == "2":
        scene = create_calibration_data_only()
        print("\nNext steps:")
        print("1. Use the returned 'scene' object in your pipeline")
        print("2. Run your COLMAP conversion")
        print("3. Call verify_existing_output() to check results")
    
    elif choice == "3":
        gt_dir = input("Ground truth directory [./calibration_data]: ").strip() or "./calibration_data"
        colmap_dir = input("COLMAP output directory: ").strip()
        if colmap_dir:
            verify_existing_output(gt_dir, colmap_dir)
        else:
            print("Error: COLMAP directory required")
    
    else:
        print("Invalid choice")


# ============================================================================
# INTEGRATION WITH YOUR EXISTING CODE
# ============================================================================
"""
To integrate this into your existing notebook or script:

METHOD A: Test your existing function
--------------------------------------
from test_calibration_pipeline import test_calibration_pipeline
from your_module import mast3r_to_colmap

# Run test
test_calibration_pipeline(mast3r_to_colmap, output_dir='./test_output')


METHOD B: Replace real scene with calibration scene
----------------------------------------------------
from calibration_data_generator import CalibrationDataGenerator

# Instead of running MASt3R:
# scene, images = run_mast3r_pairs(...)

# Use calibration scene:
generator = CalibrationDataGenerator(num_views=4, num_points=100)
scene = generator.create_mock_scene(device='cpu')
generator.save_ground_truth(scene, './ground_truth')

# Create mock images
images = [{'idx': i, 'instance': f'view_{i:02d}.jpg', 'true_shape': np.array([512, 512])} 
          for i in range(scene.num_views)]

# Continue with your normal pipeline
mast3r_to_colmap(scene, images, './colmap_output')

# Verify
from calibration_data_generator import verify_transformation
verify_transformation('./ground_truth', './colmap_output')


METHOD C: Compare before and after coordinates
-----------------------------------------------
# Before COLMAP conversion
pts_before = scene.ground_truth_pts3d_world.copy()
print("Before COLMAP:", pts_before[:5])

# After COLMAP conversion
import numpy as np
colmap_points = []
with open('colmap_output/points3D.txt', 'r') as f:
    for line in f:
        if not line.startswith('#') and line.strip():
            parts = line.split()
            colmap_points.append([float(parts[1]), float(parts[2]), float(parts[3])])
pts_after = np.array(colmap_points)
print("After COLMAP:", pts_after[:5])

# Compare
diff = pts_after - pts_before[:len(pts_after)]
print("Difference:", diff[:5])
print("Max difference:", np.abs(diff).max())
"""
