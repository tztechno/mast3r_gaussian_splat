import os
from pipeline.config import Config
from utils import clear_memory, get_memory_info

#v26
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
        
        # Remove invalid points first
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
    
    # [IMPORTANT] MASt3R poses are in camera-to-world format.
    # COLMAP requires world-to-camera format, so we need the inverse matrix.
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
    
    # Get focal lengths and principal points
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
        
        # Focals are in [N, 1] format (fx=fy for isotropic cameras)
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


def rotmat2qvec(R):
    """
    Convert rotation matrix to quaternion in COLMAP format [w, x, y, z]
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        qvec: quaternion [w, x, y, z]
    """
    # Ensure R is a numpy array
    R = np.asarray(R, dtype=np.float64)
    
    # Calculate trace
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    qvec = np.array([w, x, y, z], dtype=np.float64)
    
    # Normalize
    qvec = qvec / np.linalg.norm(qvec)
    
    return qvec



def run(cfg):
    extract_colmap_data(scene, image_paths, max_points=1000000)
    save_colmap_reconstruction(pts3d, colors, cameras, poses, image_paths, output_dir)
    write_cameras_binary(cameras, output_file)
    write_images_binary(image_paths, cameras, poses, output_file)    
    write_points3d_binary(pts3d, colors, output_file)
    rotmat2qvec(R)
    return cfg
    
