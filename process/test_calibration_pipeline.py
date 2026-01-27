"""
Calibration Pipeline Test
Tests the complete MASt3R to COLMAP pipeline with known coordinates
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from calibration_data_generator import CalibrationDataGenerator, verify_transformation
import numpy as np
import torch


def test_calibration_pipeline(colmap_converter_func, output_dir='./test_calibration'):
    """
    Test the complete pipeline with calibration data
    
    Args:
        colmap_converter_func: Your function that converts scene to COLMAP format
                               Should accept (scene, images, output_path, masks=None)
        output_dir: Directory for test outputs
    """
    print("=" * 70)
    print("CALIBRATION TEST: Verifying 3D Coordinate Transformation")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate calibration data
    print("\n[1/4] Generating calibration data...")
    generator = CalibrationDataGenerator(num_views=6, num_points=120)
    scene = generator.create_mock_scene(device='cpu')
    
    # Save ground truth
    ground_truth_dir = output_dir / 'ground_truth'
    generator.save_ground_truth(scene, ground_truth_dir)
    
    print("\n[2/4] Scene information:")
    print(f"  Views: {scene.num_views}")
    print(f"  Points: {scene.num_points}")
    print(f"  3D points shape: {scene.ground_truth_pts3d_world.shape}")
    print(f"  Coordinate range:")
    print(f"    X: [{scene.ground_truth_pts3d_world[:, 0].min():.3f}, "
          f"{scene.ground_truth_pts3d_world[:, 0].max():.3f}]")
    print(f"    Y: [{scene.ground_truth_pts3d_world[:, 1].min():.3f}, "
          f"{scene.ground_truth_pts3d_world[:, 1].max():.3f}]")
    print(f"    Z: [{scene.ground_truth_pts3d_world[:, 2].min():.3f}, "
          f"{scene.ground_truth_pts3d_world[:, 2].max():.3f}]")
    
    # Create mock images list (required by COLMAP converter)
    print("\n[3/4] Creating mock images...")
    mock_images = []
    for i in range(scene.num_views):
        # Create minimal image dict matching load_images output format
        img_dict = {
            'img': torch.zeros(3, 512, 512),  # Mock image tensor
            'true_shape': np.array([512, 512]),
            'idx': i,
            'instance': f'calibration_view_{i:02d}.jpg'
        }
        mock_images.append(img_dict)
    
    # Run COLMAP conversion
    print("\n[4/4] Running COLMAP conversion...")
    colmap_output_dir = output_dir / 'colmap_output'
    colmap_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Call your COLMAP converter
        colmap_converter_func(scene, mock_images, str(colmap_output_dir))
        print("✓ COLMAP conversion completed")
    except Exception as e:
        print(f"❌ COLMAP conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify transformation
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    verify_transformation(ground_truth_dir, colmap_output_dir)
    
    print("\n" + "=" * 70)
    print(f"Test complete! Results saved to: {output_dir}")
    print(f"  Ground truth: {ground_truth_dir}")
    print(f"  COLMAP output: {colmap_output_dir}")
    print("=" * 70)
    
    return True


def print_scene_coordinates(scene, output_file=None):
    """
    Print detailed coordinate information from scene object
    Useful for debugging
    """
    print("\n=== Scene Coordinate Details ===")
    
    # Ground truth world coordinates
    if hasattr(scene, 'ground_truth_pts3d_world'):
        pts_world = scene.ground_truth_pts3d_world
        print(f"\nGround Truth World Coordinates ({len(pts_world)} points):")
        print(f"  First 5 points:")
        for i in range(min(5, len(pts_world))):
            print(f"    Point {i}: {pts_world[i]}")
        print(f"  Statistics:")
        print(f"    Mean: {pts_world.mean(axis=0)}")
        print(f"    Std:  {pts_world.std(axis=0)}")
    
    # Camera coordinate points
    print(f"\nPer-View Camera Coordinates:")
    for i, pts_cam in enumerate(scene.pts3d):
        pts_np = pts_cam.cpu().numpy() if torch.is_tensor(pts_cam) else pts_cam
        print(f"  View {i}: {len(pts_np)} points")
        print(f"    First point: {pts_np[0]}")
        print(f"    Mean: {pts_np.mean(axis=0)}")
    
    # Camera poses
    if hasattr(scene, 'ground_truth_poses'):
        print(f"\nCamera Poses (World to Camera):")
        for i, pose in enumerate(scene.ground_truth_poses):
            print(f"  Camera {i}:")
            print(f"    Translation: {pose[:3, 3]}")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("Scene Coordinate Details\n")
            f.write("=" * 60 + "\n\n")
            if hasattr(scene, 'ground_truth_pts3d_world'):
                pts_world = scene.ground_truth_pts3d_world
                f.write(f"Ground Truth World Coordinates ({len(pts_world)} points):\n")
                for i, pt in enumerate(pts_world):
                    f.write(f"{i:4d}: {pt[0]:10.6f} {pt[1]:10.6f} {pt[2]:10.6f}\n")
        print(f"\n✓ Detailed coordinates saved to {output_file}")


# Example usage
if __name__ == "__main__":
    print("Calibration Test Module")
    print("=" * 70)
    print("\nTo use this module:")
    print("\n1. Import your COLMAP converter function:")
    print("   from your_module import mast3r_to_colmap")
    print("\n2. Run the test:")
    print("   test_calibration_pipeline(mast3r_to_colmap)")
    print("\n3. Check the verification results to see if coordinates changed")
    print("=" * 70)
