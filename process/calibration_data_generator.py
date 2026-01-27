"""
Calibration Data Generator for MASt3R to COLMAP Verification
Creates synthetic scene data with known 3D coordinates to verify transformation pipeline
"""

import numpy as np
import torch
from pathlib import Path
import json


class CalibrationDataGenerator:
    """Generate synthetic scene data with known 3D coordinates"""
    
    def __init__(self, num_views=4, num_points=100):
        self.num_views = num_views
        self.num_points = num_points
        
    def generate_cube_points(self, size=1.0):
        """Generate 3D points forming a cube"""
        points = []
        # 8 corners of cube
        for x in [-size, size]:
            for y in [-size, size]:
                for z in [-size, size]:
                    points.append([x, y, z])
        
        # Points on edges
        n_edge = (self.num_points - 8) // 12
        for i in range(n_edge):
            t = (i + 1) / (n_edge + 1)
            # X edges
            points.extend([
                [2*size*t - size, -size, -size],
                [2*size*t - size, size, -size],
                [2*size*t - size, -size, size],
                [2*size*t - size, size, size],
            ])
            # Y edges
            points.extend([
                [-size, 2*size*t - size, -size],
                [size, 2*size*t - size, -size],
                [-size, 2*size*t - size, size],
                [size, 2*size*t - size, size],
            ])
            # Z edges
            points.extend([
                [-size, -size, 2*size*t - size],
                [size, -size, 2*size*t - size],
                [-size, size, 2*size*t - size],
                [size, size, 2*size*t - size],
            ])
        
        return np.array(points[:self.num_points])
    
    def generate_circular_camera_poses(self, radius=5.0, height=0.0):
        """Generate camera poses in a circle around the scene"""
        poses = []
        for i in range(self.num_views):
            angle = 2 * np.pi * i / self.num_views
            
            # Camera position
            cam_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])
            
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0, 0, 1])
            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            
            # Rotation matrix (world to camera)
            R = np.stack([right, up, forward], axis=0)
            
            # Translation (world to camera)
            t = -R @ cam_pos
            
            # 4x4 transformation matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            
            poses.append(pose)
            
        return poses
    
    def create_mock_scene(self, device='cpu'):
        """Create a mock scene object similar to MASt3R output"""
        
        # Generate ground truth 3D points
        pts3d_world = self.generate_cube_points(size=1.0)
        
        # Generate camera poses
        camera_poses = self.generate_circular_camera_poses(radius=5.0, height=0.5)
        
        # Create mock scene object
        class MockScene:
            def __init__(self, pts3d_world, camera_poses, device):
                self.device = device
                self.num_views = len(camera_poses)
                self.num_points = len(pts3d_world)
                
                # Store ground truth
                self.ground_truth_pts3d_world = pts3d_world.copy()
                self.ground_truth_poses = [p.copy() for p in camera_poses]
                
                # Create mock outputs for each view
                self.imgs = []
                for i in range(self.num_views):
                    img_data = {
                        'idx': i,
                        'instance': f'view_{i:02d}',
                        'true_shape': np.array([512, 512])
                    }
                    self.imgs.append(img_data)
                
                # Store 3D points per view (in camera coordinates)
                self.pts3d = []
                self.conf = []
                for pose in camera_poses:
                    # Transform world points to camera coordinates
                    pts_cam = self._world_to_camera(pts3d_world, pose)
                    self.pts3d.append(torch.from_numpy(pts_cam).float().to(device))
                    # High confidence for all points
                    self.conf.append(torch.ones(self.num_points).float().to(device))
                
                # Camera intrinsics (mock)
                self.focals = torch.tensor([500.0] * self.num_views).to(device)
                self.principal_points = torch.tensor([[256.0, 256.0]] * self.num_views).to(device)
                
                # Camera poses (initially identity, will be optimized)
                self.im_poses = torch.stack([
                    torch.eye(4) for _ in range(self.num_views)
                ]).float().to(device)
                
                # Set the "optimized" poses to ground truth
                for i, pose in enumerate(camera_poses):
                    self.im_poses[i] = torch.from_numpy(pose).float()
            
            def _world_to_camera(self, pts_world, pose):
                """Transform world points to camera coordinates"""
                pts_homo = np.hstack([pts_world, np.ones((len(pts_world), 1))])
                pts_cam_homo = (pose @ pts_homo.T).T
                return pts_cam_homo[:, :3]
            
            def get_im_poses(self):
                """Return camera poses (world to camera)"""
                return self.im_poses
            
            def get_pts3d(self, i=None):
                """Get 3D points for view i (or all views)"""
                if i is not None:
                    return self.pts3d[i]
                return self.pts3d
            
            def get_conf(self, i=None):
                """Get confidence for view i (or all views)"""
                if i is not None:
                    return self.conf[i]
                return self.conf
            
            def get_focals(self):
                return self.focals
            
            def get_principal_points(self):
                return self.principal_points
        
        scene = MockScene(pts3d_world, camera_poses, device)
        
        return scene
    
    def save_ground_truth(self, scene, output_path):
        """Save ground truth data for verification"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save ground truth 3D points
        np.savetxt(
            output_path / 'ground_truth_points3d.txt',
            scene.ground_truth_pts3d_world,
            header='X Y Z (world coordinates)',
            fmt='%.6f'
        )
        
        # Save ground truth camera poses
        with open(output_path / 'ground_truth_poses.json', 'w') as f:
            poses_list = [p.tolist() for p in scene.ground_truth_poses]
            json.dump({
                'poses': poses_list,
                'description': 'Camera poses as 4x4 transformation matrices (world to camera)'
            }, f, indent=2)
        
        # Save camera intrinsics
        with open(output_path / 'ground_truth_intrinsics.json', 'w') as f:
            json.dump({
                'focals': scene.focals.cpu().numpy().tolist(),
                'principal_points': scene.principal_points.cpu().numpy().tolist(),
                'image_size': [512, 512]
            }, f, indent=2)
        
        print(f"\n✓ Ground truth data saved to {output_path}")
        print(f"  - {len(scene.ground_truth_pts3d_world)} 3D points")
        print(f"  - {len(scene.ground_truth_poses)} camera poses")


def verify_transformation(ground_truth_path, colmap_output_path):
    """Verify that COLMAP output matches ground truth coordinates"""
    ground_truth_path = Path(ground_truth_path)
    colmap_output_path = Path(colmap_output_path)
    
    # Load ground truth
    gt_points = np.loadtxt(ground_truth_path / 'ground_truth_points3d.txt')
    
    # Load COLMAP points3D.txt
    colmap_points_file = colmap_output_path / 'points3D.txt'
    if not colmap_points_file.exists():
        print(f"❌ COLMAP output not found: {colmap_points_file}")
        return
    
    # Parse COLMAP points3D.txt
    colmap_points = []
    with open(colmap_points_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                # Format: POINT3D_ID X Y Z R G B ERROR TRACK[] ...
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                colmap_points.append([x, y, z])
    
    colmap_points = np.array(colmap_points)
    
    print("\n=== Coordinate Transformation Verification ===")
    print(f"Ground truth points: {len(gt_points)}")
    print(f"COLMAP points: {len(colmap_points)}")
    
    if len(colmap_points) == 0:
        print("❌ No points found in COLMAP output")
        return
    
    # Statistics
    print("\nGround Truth Statistics:")
    print(f"  Mean: {gt_points.mean(axis=0)}")
    print(f"  Std:  {gt_points.std(axis=0)}")
    print(f"  Min:  {gt_points.min(axis=0)}")
    print(f"  Max:  {gt_points.max(axis=0)}")
    
    print("\nCOLMAP Output Statistics:")
    print(f"  Mean: {colmap_points.mean(axis=0)}")
    print(f"  Std:  {colmap_points.std(axis=0)}")
    print(f"  Min:  {colmap_points.min(axis=0)}")
    print(f"  Max:  {colmap_points.max(axis=0)}")
    
    # Check if there's a simple transformation (scale, rotation, translation)
    # This would indicate coordinate frame changes
    if len(colmap_points) == len(gt_points):
        # Try to find transformation
        from scipy.spatial import procrustes
        gt_centered = gt_points - gt_points.mean(axis=0)
        colmap_centered = colmap_points - colmap_points.mean(axis=0)
        
        mtx1, mtx2, disparity = procrustes(gt_centered, colmap_centered)
        
        print(f"\nProcrustes alignment disparity: {disparity:.6f}")
        print("(Lower is better; 0 means perfect match after alignment)")
        
        if disparity < 0.01:
            print("✓ Coordinates match after alignment (rigid transformation)")
        elif disparity < 0.1:
            print("⚠ Coordinates approximately match (small differences)")
        else:
            print("❌ Significant coordinate differences detected")
    else:
        print("\n⚠ Point count mismatch - cannot perform direct comparison")
    
    # Save comparison
    comparison_file = colmap_output_path / 'coordinate_comparison.txt'
    with open(comparison_file, 'w') as f:
        f.write("Ground Truth vs COLMAP Coordinate Comparison\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Ground truth points: {len(gt_points)}\n")
        f.write(f"COLMAP points: {len(colmap_points)}\n\n")
        f.write("Ground Truth Statistics:\n")
        f.write(f"  Mean: {gt_points.mean(axis=0)}\n")
        f.write(f"  Std:  {gt_points.std(axis=0)}\n")
        f.write("\nCOLMAP Output Statistics:\n")
        f.write(f"  Mean: {colmap_points.mean(axis=0)}\n")
        f.write(f"  Std:  {colmap_points.std(axis=0)}\n")
    
    print(f"\n✓ Comparison saved to {comparison_file}")


# Example usage
if __name__ == "__main__":
    # Generate calibration data
    generator = CalibrationDataGenerator(num_views=4, num_points=100)
    scene = generator.create_mock_scene(device='cpu')
    
    # Save ground truth
    generator.save_ground_truth(scene, './calibration_data')
    
    print("\nCalibration data generated successfully!")
    print("\nTo use this data:")
    print("1. Use the returned 'scene' object in place of MASt3R output")
    print("2. Run your COLMAP conversion")
    print("3. Call verify_transformation() to check if coordinates changed")
    print("\nExample:")
    print("  verify_transformation('./calibration_data', './colmap_output')")
