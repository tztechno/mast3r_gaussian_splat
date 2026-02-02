#process1_06.py

def extract_colmap_data(scene, image_paths, max_points=1000000):
    """
    Extract COLMAP-compatible camera parameters and 3D points from MASt3R scene
    
    Args:
        scene: MASt3R scene object
        image_paths: List of image paths
        max_points: Maximum number of 3D points to extract (default: 1M)
    """
    print("\n=== Extracting COLMAP-compatible data ===")
    
    # Extract point cloud
    pts_all = scene.get_pts3d()
    print(f"pts_all type: {type(pts_all)}")
    
    if isinstance(pts_all, list):
        print(f"pts_all is a list with {len(pts_all)} elements")
        if len(pts_all) > 0:
            print(f"First element type: {type(pts_all[0])}")
            if hasattr(pts_all[0], 'shape'):
                print(f"First element shape: {pts_all[0].shape}")
        
        pts_all = torch.stack([p if isinstance(p, torch.Tensor) else torch.tensor(p) 
                              for p in pts_all])
        print(f"pts_all shape after conversion: {pts_all.shape}")
    
    if len(pts_all.shape) == 4:
        print(f"Found batched point cloud: {pts_all.shape}")
        B, H, W, _ = pts_all.shape
        pts3d = pts_all.reshape(-1, 3).detach().cpu().numpy()  
        
        # Extract colors
        colors = []
        for img_path in image_paths:
            img = Image.open(img_path).resize((W, H))
            colors.append(np.array(img))
        colors = np.stack(colors).reshape(-1, 3) / 255.0
    else:
        pts3d = pts_all.detach().cpu().numpy() if isinstance(pts_all, torch.Tensor) else pts_all
        colors = np.ones((len(pts3d), 3)) * 0.5
    
    print(f"✓ Extracted {len(pts3d)} 3D points from {len(image_paths)} images")
    
    # **DOWNSAMPLE POINTS TO REDUCE MEMORY USAGE**
    if len(pts3d) > max_points:
        print(f"\n⚠ Downsampling from {len(pts3d)} to {max_points} points to reduce memory usage...")
        
        # Remove invalid points first (NaN or Inf)
        valid_mask = ~(np.isnan(pts3d).any(axis=1) | np.isinf(pts3d).any(axis=1))
        pts3d_valid = pts3d[valid_mask]
        colors_valid = colors[valid_mask]
        
        # Random sampling
        indices = np.random.choice(len(pts3d_valid), size=max_points, replace=False)
        pts3d = pts3d_valid[indices]
        colors = colors_valid[indices]
        
        print(f"✓ Downsampled to {len(pts3d)} points")
    
    # Extract camera parameters
    print("Extracting camera parameters...")
    
    # [IMPORTANT] MASt3R uses camera-to-world format.
    # COLMAP requires world-to-camera format, so the matrix must be inverted.
    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    print(f"Retrieved camera-to-world poses: shape {poses_c2w.shape}")
    
    # Convert camera-to-world to world-to-camera
    poses = []
    for i, pose_c2w in enumerate(poses_c2w):
        # Calculate the inverse of the 4x4 matrix
        pose_w2c = np.linalg.inv(pose_c2w)
        poses.append(pose_w2c)
    
    poses = np.array(poses)
    print(f"Converted to world-to-camera poses for COLMAP")
    
    # Retrieve focal length and principal points
    focals = scene.get_focals().detach().cpu().numpy()
    pp = scene.get_principal_points().detach().cpu().numpy()
    print(f"Focals shape: {focals.shape}")
    print(f"Principal points shape: {pp.shape}")
    
    # MASt3R internal processing size (usually 224x224)
    mast3r_size = 224.0
    
    cameras = []
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        W, H = img.size
        
        # Scale ratio relative to the original image size
        scale = W / mast3r_size
        
        # Focals are in [N, 1] format (Isotropic camera: fx = fy)
        if focals.shape[1] == 1:
            focal_mast3r = float(focals[i, 0])
            fx = fy = focal_mast3r * scale
        else:
            fx = float(focals[i, 0]) * scale
            fy = float(focals[i, 1]) * scale
        
        # Scale principal points as well
        cx = float(pp[i, 0]) * scale
        cy = float(pp[i, 1]) * scale
        
        camera = {
            'camera_id': i + 1,
            'model': 'PINHOLE',
            'width': W,
            'height': H,
            'params': [fx, fy, cx, cy]
        }
        cameras.append(camera)
        
        if i == 0:
            print(f"\nExample camera 0:")
            print(f"  Image size: {W}x{H}")
            print(f"  MASt3R focal: {focal_mast3r:.2f}, pp: ({pp[i,0]:.2f}, {pp[i,1]:.2f})")
            print(f"  Scaled fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            print(f"  Pose (first row): {poses[i][0]}")
    
    print(f"\n✓ Extracted {len(cameras)} cameras and {len(poses)} poses")
    
    return pts3d, colors, cameras, poses


import struct
from pathlib import Path

def save_colmap_reconstruction(pts3d, colors, cameras, poses, image_paths, output_dir):
    """Save reconstruction in COLMAP binary format by writing files directly"""
    print("\n=== Saving COLMAP reconstruction ===")
    
    sparse_dir = Path(output_dir) / 'sparse' / '0'
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Writing COLMAP files directly to {sparse_dir}...")
    
    # Write cameras.bin
    write_cameras_binary(cameras, sparse_dir / 'cameras.bin')
    print(f"  ✓ Wrote {len(cameras)} cameras")
    
    # Write images.bin
    write_images_binary(image_paths, cameras, poses, sparse_dir / 'images.bin')
    print(f"  ✓ Wrote {len(image_paths)} images")
    
    # Write points3D.bin
    num_points = write_points3d_binary(pts3d, colors, sparse_dir / 'points3D.bin')
    print(f"  ✓ Wrote {num_points} 3D points")
    
    print(f"\n✓ COLMAP reconstruction saved to {sparse_dir}")
    print(f"  Cameras: {len(cameras)}")
    print(f"  Images: {len(image_paths)}")
    print(f"  Points: {num_points}")
    
    return sparse_dir


def write_cameras_binary(cameras, output_file):
    """Write cameras.bin in COLMAP binary format"""
    with open(output_file, 'wb') as f:
        # Write number of cameras
        f.write(struct.pack('Q', len(cameras)))
        
        for i, cam in enumerate(cameras):
            camera_id = cam.get('camera_id', i + 1)
            
            # Model ID: 1 = PINHOLE
            model_id = 1
            width = cam['width']
            height = cam['height']
            params = cam['params']  # [fx, fy, cx, cy]
            
            f.write(struct.pack('i', camera_id))
            f.write(struct.pack('i', model_id))
            f.write(struct.pack('Q', width))
            f.write(struct.pack('Q', height))
            
            # Write 4 parameters for PINHOLE model
            for param in params[:4]:
                f.write(struct.pack('d', param))


def write_images_binary(image_paths, cameras, poses, output_file):
    """Write images.bin in COLMAP binary format"""
    with open(output_file, 'wb') as f:
        # Write number of images
        f.write(struct.pack('Q', len(image_paths)))
        
        for i, (img_path, pose) in enumerate(zip(image_paths, poses)):
            image_id = i + 1
            camera_id = cameras[i].get('camera_id', i + 1)
            image_name = os.path.basename(img_path)
            
            # Extract rotation and translation
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Convert rotation matrix to quaternion [w, x, y, z]
            qvec = rotmat2qvec(R)
            tvec = t
            
            # Write image data
            f.write(struct.pack('i', image_id))
            
            # Write quaternion (4 doubles)
            for q in qvec:
                f.write(struct.pack('d', float(q)))
            
            # Write translation vector (3 doubles)
            for tv in tvec:
                f.write(struct.pack('d', float(tv)))
            
            # Write camera ID
            f.write(struct.pack('i', camera_id))
            
            # Write image name (null-terminated string)
            f.write(image_name.encode('utf-8') + b'\x00')
            
            # Write number of 2D points (0 for now, as we don't have 2D-3D correspondences)
            f.write(struct.pack('Q', 0))


def write_points3d_binary(pts3d, colors, output_file):
    """Write points3D.bin in COLMAP binary format"""
    # Filter out invalid points
    valid_indices = []
    for i, pt in enumerate(pts3d):
        if not (np.isnan(pt).any() or np.isinf(pt).any()):
            valid_indices.append(i)
    
    with open(output_file, 'wb') as f:
        # Write number of points
        f.write(struct.pack('Q', len(valid_indices)))
        
        for idx, point_id in enumerate(valid_indices):
            pt = pts3d[point_id]
            color = colors[point_id]
            
            # Write point3D ID
            f.write(struct.pack('Q', point_id))
            
            # Write XYZ coordinates (3 doubles)
            for coord in pt:
                f.write(struct.pack('d', float(coord)))
            
            # Write RGB color (3 unsigned chars)
            col_int = (color * 255).astype(np.uint8)
            for c in col_int:
                f.write(struct.pack('B', int(c)))
            
            # Write error (1 double) - set to 0
            f.write(struct.pack('d', 0.0))
            
            # Write track length (number of images seeing this point)
            # Set to 0 as we don't have track information
            f.write(struct.pack('Q', 0))
            
            # Progress indicator
            if (idx + 1) % 1000000 == 0:
                print(f"    Wrote {idx + 1} / {len(valid_indices)} points...")
    
    return len(valid_indices)


import os
import struct
import numpy as np
import torch
from PIL import Image
from pathlib import Path

def rotmat2qvec(R):
    """Convert rotation matrix to COLMAP quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)




def export_mast3r_to_colmap(scene, image_paths, output_dir, max_points=1_000_000):
    """Extracts data and saves COLMAP binaries in one workflow."""
    print("\n" + "="*50 + "\nSTARTING COLMAP EXPORT\n" + "="*50)
    
    # --- 1. Point Cloud & Color Extraction ---
    pts_all = scene.get_pts3d()
    if isinstance(pts_all, list):
        pts_all = torch.stack([p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in pts_all])
    
    if len(pts_all.shape) == 4: # Batched [B, H, W, 3]
        B, H, W, _ = pts_all.shape
        pts3d = pts_all.reshape(-1, 3).detach().cpu().numpy()
        colors = []
        for path in image_paths:
            img = Image.open(path).resize((W, H))
            colors.append(np.array(img))
        colors = np.stack(colors).reshape(-1, 3) / 255.0
    else:
        pts3d = pts_all.detach().cpu().numpy() if isinstance(pts_all, torch.Tensor) else pts_all
        colors = np.ones((len(pts3d), 3)) * 0.5

    # Filter invalid values and Downsample
    mask = ~(np.isnan(pts3d).any(axis=1) | np.isinf(pts3d).any(axis=1))
    pts3d, colors = pts3d[mask], colors[mask]
    
    if len(pts3d) > max_points:
        print(f"Downsampling from {len(pts3d):,} to {max_points:,} points...")
        idx = np.random.choice(len(pts3d), max_points, replace=False)
        pts3d, colors = pts3d[idx], colors[idx]

    # --- 2. Camera & Pose Extraction ---
    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    pps = scene.get_principal_points().detach().cpu().numpy()
    mast3r_size = 224.0
    
    cameras_data, w2c_poses = [], []
    for i, path in enumerate(image_paths):
        with Image.open(path) as img:
            orig_W, orig_H = img.size
        
        scale = orig_W / mast3r_size
        fx = focals[i, 0] * scale
        fy = (focals[i, 1] if focals.shape[1] > 1 else focals[i, 0]) * scale
        cx, cy = pps[i, 0] * scale, pps[i, 1] * scale
        
        cameras_data.append({'id': i+1, 'W': orig_W, 'H': orig_H, 'params': [fx, fy, cx, cy]})
        w2c_poses.append(np.linalg.inv(poses_c2w[i]))

    # --- 3. Binary Writing ---
    save_path = Path(output_dir) / 'sparse' / '0'
    save_path.mkdir(parents=True, exist_ok=True)

    # cameras.bin
    print("Writing cameras.bin...")
    with open(save_path / 'cameras.bin', 'wb') as f:
        f.write(struct.pack('Q', len(cameras_data)))
        for cam in cameras_data:
            f.write(struct.pack('i', cam['id']))        # camera_id
            f.write(struct.pack('i', 1))                # model_id (1 = PINHOLE)
            f.write(struct.pack('Q', cam['W']))         # width
            f.write(struct.pack('Q', cam['H']))         # height
            f.write(struct.pack('d', cam['params'][0])) # fx
            f.write(struct.pack('d', cam['params'][1])) # fy
            f.write(struct.pack('d', cam['params'][2])) # cx
            f.write(struct.pack('d', cam['params'][3])) # cy
    print(f"  ✓ Wrote {len(cameras_data)} cameras")

    # images.bin
    print("Writing images.bin...")
    with open(save_path / 'images.bin', 'wb') as f:
        f.write(struct.pack('Q', len(image_paths)))
        for i, (path, pose) in enumerate(zip(image_paths, w2c_poses)):
            qvec = rotmat2qvec(pose[:3, :3])
            f.write(struct.pack('i', i+1))              # image_id
            for val in qvec: 
                f.write(struct.pack('d', float(val)))   # quaternion
            for val in pose[:3, 3]: 
                f.write(struct.pack('d', float(val)))   # translation
            f.write(struct.pack('i', i+1))              # camera_id
            f.write(os.path.basename(path).encode('utf-8') + b'\x00')  # name
            f.write(struct.pack('Q', 0))                # num_points2D
    print(f"  ✓ Wrote {len(image_paths)} images")

    # points3D.bin - Fixed version
    print("Writing points3D.bin...")
    
    # Deduplication: Round coordinates and keep only unique points
    unique_points = {}
    for i, (pt, col) in enumerate(zip(pts3d, colors)):
        # Round to 6 decimals to use as a dictionary key
        key = tuple(np.round(pt, decimals=6))
        if key not in unique_points:
            unique_points[key] = (pt, col)
    
    print(f"  Removed {len(pts3d) - len(unique_points)} duplicate points")
    print(f"  Writing {len(unique_points)} unique points...")
    
    with open(save_path / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('Q', len(unique_points)))
        
        point_id = 1  # Sequence starting from 1
        for pt, col in unique_points.values():
            # point3D_id (unsigned long long)
            f.write(struct.pack('Q', point_id))
            
            # XYZ (3 doubles)
            f.write(struct.pack('d', float(pt[0])))
            f.write(struct.pack('d', float(pt[1])))
            f.write(struct.pack('d', float(pt[2])))
            
            # RGB (3 unsigned chars)
            rgb = (col * 255).astype(np.uint8)
            f.write(struct.pack('B', int(rgb[0])))
            f.write(struct.pack('B', int(rgb[1])))
            f.write(struct.pack('B', int(rgb[2])))
            
            # error (1 double)
            f.write(struct.pack('d', 0.0))
            
            # track_length (unsigned long long)
            f.write(struct.pack('Q', 0))
            
            # Progress tracking
            if point_id % 100_000 == 0:
                print(f"    Wrote {point_id} / {len(unique_points)} points...")
            
            point_id += 1

    print(f"\n✓ Exported to {save_path}")
    print(f"  Cameras: {len(cameras_data)}")
    print(f"  Images: {len(image_paths)}")
    print(f"  Points: {len(unique_points):,}")
    return save_path
