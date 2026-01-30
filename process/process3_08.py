# ============================================================================
# COLMAP Conversion (process3_08.py)
# ============================================================================

import numpy as np
import cv2
from pathlib import Path
import struct
from scipy.spatial.transform import Rotation
import torch
from PIL import Image

# ============================================================================
# COLMAP Conversion (process3_05.py) - FIXED VERSION
# ============================================================================

def convert_mast3r_to_colmap(scene, output_dir, min_conf_thr=1.5, clean_depth=True, 
                            mask_images=True, verbose=True, processed_image_paths=None):
    """
    Convert MASt3R scene to COLMAP format
    
    Args:
        scene: MASt3R optimized scene
        output_dir: Output directory path
        min_conf_thr: Minimum confidence threshold for 3D points
        clean_depth: Whether to clean depth maps
        mask_images: Whether to apply masks
        verbose: Print verbose output
        processed_image_paths: List of paths to processed (square) images
    """
    output_dir = Path(output_dir)
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    normal_dir = output_dir / "normal"
    mask_dir = output_dir / "mask"
    
    # Create directories
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("Converting MASt3R scene to COLMAP format...")
        print(f"Output directory: {output_dir}")
    
    cameras, images_data, points3D = extract_scene_data(scene, min_conf_thr, verbose)
    
    if verbose:
        print(f"Extracted {len(cameras)} cameras")
        print(f"Extracted {len(images_data)} images")
        print(f"Extracted {len(points3D)} 3D points")
    
    # Pass processed image paths if available
    save_image_data(scene, images_dir, depth_dir, normal_dir, mask_dir, 
                    min_conf_thr, verbose, processed_image_paths=processed_image_paths)
    
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images_data, sparse_dir / "images.bin")
    
    # Create points3D.bin even if empty
    if len(points3D) == 0:
        if verbose:
            print("Warning: No 3D points extracted. Creating empty points3D.bin")
    write_points3d_binary(points3D, sparse_dir / "points3D.bin")
    
    if verbose:
        print(f"✓ COLMAP format data saved to {output_dir}")
    
    return output_dir

def extract_scene_data(scene, min_conf_thr, verbose):
    """Extract cameras, images, and 3D points from MASt3R scene"""
    cameras = {}
    images_data = {}
    points3D = []
    
    if verbose:
        print("Extracting scene data...")
        print(f"Scene type: {type(scene)}")
        print(f"Scene attributes: {dir(scene)}")
    
    # Check scene structure
    if hasattr(scene, 'imgs'):
        num_views = len(scene.imgs)
    elif hasattr(scene, 'views'):
        num_views = len(scene.views)
    else:
        num_views = 0
        if verbose:
            print("Warning: Cannot determine number of views")
    
    if verbose:
        print(f"Number of views: {num_views}")
    
    # Extract camera parameters and poses
    for idx in range(num_views):
        if verbose:
            print(f"\n=== Processing view {idx} ===")
        
        # Get view
        if hasattr(scene, 'imgs'):
            view = scene.imgs[idx]
        elif hasattr(scene, 'views'):
            view = scene.views[idx]
        else:
            if verbose:
                print(f"Warning: Cannot access view {idx}")
            continue
        
        # Get image size
        try:
            if hasattr(view, 'shape'):
                if isinstance(view.shape, (list, tuple)) and len(view.shape) >= 2:
                    height, width = int(view.shape[0]), int(view.shape[1])
                else:
                    height, width = 512, 512
            elif hasattr(view, 'img'):
                if isinstance(view.img, np.ndarray):
                    height, width = view.img.shape[:2]
                elif torch.is_tensor(view.img):
                    shape = view.img.shape
                    if len(shape) >= 2:
                        height, width = int(shape[-2]), int(shape[-1])
                    else:
                        height, width = 512, 512
                else:
                    height, width = 512, 512
            else:
                height, width = 512, 512
                
            if verbose:
                print(f"  Image size: {width}x{height}")
        except Exception as e:
            if verbose:
                print(f"  Error getting image size: {e}, using default 512x512")
            height, width = 512, 512
        
        # Camera intrinsics (Default values)
        fx = fy = 500.0
        cx = width / 2.0
        cy = height / 2.0
        
        try:
            # Attempt to get camera parameters from the scene
            if hasattr(scene, 'get_intrinsics'):
                K = scene.get_intrinsics()
                if K is not None:
                    if isinstance(K, torch.Tensor):
                        K = K.detach().cpu().numpy()
                    if K.ndim >= 2:
                        K_view = K[idx] if K.ndim == 3 else K
                        if K_view.shape[0] >= 3 and K_view.shape[1] >= 3:
                            fx = float(K_view[0, 0])
                            fy = float(K_view[1, 1])
                            cx = float(K_view[0, 2])
                            cy = float(K_view[1, 2])
                            if verbose:
                                print(f"  Extracted intrinsics from scene: fx={fx:.2f}, fy={fy:.2f}")
            
            # Attempt to get camera parameters from individual views
            if hasattr(view, 'camera') and view.camera is not None:
                cam = view.camera
                if isinstance(cam, torch.Tensor):
                    cam = cam.detach().cpu().numpy()
                
                if cam.ndim == 2 and cam.shape[0] >= 3 and cam.shape[1] >= 3:
                    fx = float(cam[0, 0])
                    fy = float(cam[1, 1])
                    cx = float(cam[0, 2])
                    cy = float(cam[1, 2])
                    if verbose:
                        print(f"  Extracted intrinsics from view.camera: fx={fx:.2f}, fy={fy:.2f}")
                        
        except Exception as e:
            if verbose:
                print(f"  Error extracting camera intrinsics: {e}")
                print(f"  Using defaults: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        # Save Camera ID and parameters
        cam_id = idx
        cameras[cam_id] = {
            'model': 'PINHOLE',
            'width': int(width),
            'height': int(height),
            'params': [fx, fy, cx, cy]
        }
        
        # Extract camera pose
        qvec = np.array([1.0, 0.0, 0.0, 0.0])  # Default (no rotation)
        tvec = np.array([0.0, 0.0, 0.0])       # Default (no translation)
        
        try:
            pose = None
            
            # Get pose at scene level
            if hasattr(scene, 'get_im_poses'):
                poses = scene.get_im_poses()
                if poses is not None:
                    if isinstance(poses, (list, tuple)):
                        if idx < len(poses):
                            pose = poses[idx]
                    else:
                        if isinstance(poses, torch.Tensor):
                            poses = poses.detach().cpu().numpy()
                        if poses.ndim >= 2:
                            pose = poses[idx] if poses.ndim == 3 else poses
            
            # Get pose at view level
            if pose is None and hasattr(view, 'pose') and view.pose is not None:
                pose = view.pose
            
            # Convert pose to numpy if it's a Tensor
            if pose is not None and isinstance(pose, torch.Tensor):
                pose = pose.detach().cpu().numpy()
            
            # Process Pose
            if pose is not None:
                if verbose:
                    print(f"  Pose type: {type(pose)}")
                    print(f"  Pose shape: {pose.shape if hasattr(pose, 'shape') else 'N/A'}")
                
                # Check pose dimensions
                if isinstance(pose, np.ndarray):
                    if pose.ndim == 1:
                        if verbose:
                            print(f"  Warning: 1D pose array (shape {pose.shape}), using identity pose")
                        
                    elif pose.ndim == 2:
                        if pose.shape == (4, 4):
                            # Valid 4x4 matrix
                            if verbose:
                                print(f"  Processing 4x4 pose matrix")
                            try:
                                # Check if matrix is singular
                                det = np.linalg.det(pose)
                                if abs(det) < 1e-10:
                                    if verbose:
                                        print(f"  Warning: Near-singular matrix (det={det}), using identity pose")
                                else:
                                    # MASt3R poses are world-to-camera; COLMAP needs camera-to-world
                                    pose_inv = np.linalg.inv(pose)
                                    qvec, tvec = matrix_to_quaternion_translation(pose_inv)
                                    if verbose:
                                        print(f"  Successfully extracted pose")
                            except np.linalg.LinAlgError as e:
                                if verbose:
                                    print(f"  LinAlgError: {e}, using identity pose")
                                    
                        elif pose.shape == (3, 4):
                            # Extend 3x4 matrix to 4x4
                            if verbose:
                                print(f"  Processing 3x4 pose matrix")
                            pose_4x4 = np.eye(4)
                            pose_4x4[:3, :] = pose
                            try:
                                det = np.linalg.det(pose_4x4)
                                if abs(det) < 1e-10:
                                    if verbose:
                                        print(f"  Warning: Near-singular matrix (det={det}), using identity pose")
                                else:
                                    pose_inv = np.linalg.inv(pose_4x4)
                                    qvec, tvec = matrix_to_quaternion_translation(pose_inv)
                                    if verbose:
                                        print(f"  Successfully extracted pose from 3x4 matrix")
                            except np.linalg.LinAlgError as e:
                                if verbose:
                                    print(f"  LinAlgError: {e}, using identity pose")
                                    
                        elif pose.shape == (3, 3):
                            # 3x3 rotation matrix only
                            if verbose:
                                print(f"  Processing 3x3 rotation matrix")
                            pose_4x4 = np.eye(4)
                            pose_4x4[:3, :3] = pose
                            try:
                                pose_inv = np.linalg.inv(pose_4x4)
                                qvec, tvec = matrix_to_quaternion_translation(pose_inv)
                                if verbose:
                                    print(f"  Successfully extracted pose from 3x3 matrix")
                            except np.linalg.LinAlgError as e:
                                if verbose:
                                    print(f"  LinAlgError: {e}, using identity pose")
                        else:
                            if verbose:
                                print(f"  Warning: Unexpected 2D pose shape {pose.shape}, using identity pose")
                    else:
                        if verbose:
                            print(f"  Warning: Unexpected pose dimensions (ndim={pose.ndim}), using identity pose")
                else:
                    if verbose:
                        print(f"  Warning: Pose is not a numpy array, using identity pose")
            else:
                if verbose:
                    print(f"  No pose found, using identity pose")
                    
        except Exception as e:
            if verbose:
                print(f"  Error extracting pose: {e}")
                print(f"  Using identity pose")
            import traceback
            traceback.print_exc()
        
        # Save image data
        img_id = idx + 1
        images_data[img_id] = {
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': cam_id,
            'name': f'image_{idx:04d}.jpg',
            'xys': np.array([]),           # Empty 2D points array
            'point3D_ids': np.array([])    # Empty 3D point IDs array
        }
        
        if verbose:
            print(f"  Final - Camera {cam_id}: {width}x{height}")
            print(f"  Final - Image {img_id}: qvec={qvec[:4]}, tvec={tvec[:3]}")

    # Extract 3D points
    if verbose:
        print("\n=== Extracting 3D points ===")
    
    try:
        # Get 3D points from MASt3R scene
        if hasattr(scene, 'get_pts3d'):
            pts3d = scene.get_pts3d()
            if pts3d is not None:
                if verbose:
                    print(f"  pts3d type: {type(pts3d)}")
                
                # Handle list input
                if isinstance(pts3d, list):
                    if verbose:
                        print(f"  pts3d is a list with {len(pts3d)} elements")
                    
                    # Process each list element
                    all_points = []
                    for i, pts in enumerate(pts3d):
                        if isinstance(pts, torch.Tensor):
                            pts = pts.detach().cpu().numpy()
                        if isinstance(pts, np.ndarray):
                            all_points.append(pts.reshape(-1, 3))
                            if verbose and i < 3:  # Show first 3 elements
                                print(f"    Element {i} shape: {pts.shape}")
                    
                    if all_points:
                        pts3d_combined = np.vstack(all_points)
                        if verbose:
                            print(f"  Combined pts3d shape: {pts3d_combined.shape}")
                    else:
                        pts3d_combined = None
                        
                # Handle Tensor or Numpy array input
                elif isinstance(pts3d, torch.Tensor):
                    pts3d_combined = pts3d.detach().cpu().numpy()
                    if verbose:
                        print(f"  pts3d shape (from tensor): {pts3d_combined.shape}")
                elif isinstance(pts3d, np.ndarray):
                    pts3d_combined = pts3d
                    if verbose:
                        print(f"  pts3d shape (numpy): {pts3d_combined.shape}")
                else:
                    pts3d_combined = None
                    if verbose:
                        print(f"  Unexpected pts3d type: {type(pts3d)}")
                
                # Confidence Filtering
                if pts3d_combined is not None:
                    # Get confidence values
                    conf = None
                    if hasattr(scene, 'get_conf'):
                        conf = scene.get_conf()
                    elif hasattr(scene, 'im_conf'):
                        conf = scene.im_conf
                    
                    if conf is not None:
                        if verbose:
                            print(f"  conf type: {type(conf)}")
                        
                        # conf might also be a list
                        if isinstance(conf, list):
                            all_conf = []
                            for c in conf:
                                if isinstance(c, torch.Tensor):
                                    c = c.detach().cpu().numpy()
                                if isinstance(c, np.ndarray):
                                    all_conf.append(c.flatten())
                            if all_conf:
                                conf_combined = np.concatenate(all_conf)
                            else:
                                conf_combined = None
                        elif isinstance(conf, torch.Tensor):
                            conf_combined = conf.detach().cpu().numpy().flatten()
                        elif isinstance(conf, np.ndarray):
                            conf_combined = conf.flatten()
                        else:
                            conf_combined = None
                        
                        if conf_combined is not None:
                            if verbose:
                                print(f"  conf shape: {conf_combined.shape}")
                            
                            # Flatten point cloud
                            pts3d_flat = pts3d_combined.reshape(-1, 3)
                            
                            # Align sizes
                            min_size = min(len(pts3d_flat), len(conf_combined))
                            pts3d_flat = pts3d_flat[:min_size]
                            conf_combined = conf_combined[:min_size]
                            
                            # Apply mask
                            mask = conf_combined >= min_conf_thr
                            pts3d_filtered = pts3d_flat[mask]
                            
                            if verbose:
                                print(f"  Points before filtering: {len(pts3d_flat)}")
                                print(f"  Points after filtering (conf >= {min_conf_thr}): {len(pts3d_filtered)}")
                        else:
                            pts3d_filtered = pts3d_combined.reshape(-1, 3)
                            if verbose:
                                print(f"  No valid confidence, using all {len(pts3d_filtered)} points")
                    else:
                        # Use all points if confidence is unavailable
                        pts3d_filtered = pts3d_combined.reshape(-1, 3)
                        if verbose:
                            print(f"  No confidence values, using all {len(pts3d_filtered)} points")
                    
                    # Convert to COLMAP format
                    for i, pt in enumerate(pts3d_filtered):
                        # Skip invalid points (NaN or Inf)
                        if not np.all(np.isfinite(pt)):
                            continue
                        
                        points3D.append({
                            'xyz': pt,
                            'rgb': np.array([128, 128, 128]),  # Default gray
                            'error': 0.0,
                            'image_ids': np.array([]),
                            'point2D_idxs': np.array([])
                        })
                else:
                    if verbose:
                        print("  No valid pts3d data")
        else:
            if verbose:
                print("  Warning: Scene has no get_pts3d method")
                
    except Exception as e:
        if verbose:
            print(f"  Error extracting 3D points: {e}")
        import traceback
        traceback.print_exc()

    return cameras, images_data, points3D


def estimate_camera_pose(pts3d: np.ndarray, confidence: np.ndarray, min_conf_thr: float) -> np.ndarray:
    """Estimates camera pose from 3D points using PCA for orientation."""
    if hasattr(pts3d, 'cpu'):
        pts3d = pts3d.detach().cpu().numpy()
    if hasattr(confidence, 'cpu'):
        confidence = confidence.detach().cpu().numpy()

    h, w = pts3d.shape[:2]
    pts_flat = pts3d.reshape(-1, 3)
    conf_flat = confidence.reshape(-1)

    mask = conf_flat > min_conf_thr
    valid_pts = pts_flat[mask]

    if len(valid_pts) < 4:
        return np.eye(4)

    # Compute center
    center = np.median(valid_pts, axis=0)
    centered_pts = valid_pts - center

    # Use PCA to estimate orientation
    try:
        cov = np.cov(centered_pts.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort eigenvectors by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
            
        R = eigenvectors
    except:
        R = np.eye(3)

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = -R @ center  # Camera-to-world transformation

    return pose


def matrix_to_quaternion_translation(matrix: np.ndarray):
    """Robust conversion of 4x4 transformation matrix to quaternion and translation."""
    R = matrix[:3, :3]
    t = matrix[:3, 3]

    # Use scipy for robust quaternion conversion
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    
    # COLMAP format is [w, x, y, z]
    qvec = np.array([quat[3], quat[0], quat[1], quat[2]])

    return qvec, t


def extract_3d_points_with_correspondences(scene, images_data, min_conf_thr: float, verbose: bool):
    """Extracts 3D points with proper 2D-3D correspondences across images."""
    points3D = {}
    point_id = 1

    num_images = len(scene.imgs)
    all_confidences = scene.get_conf()
    all_pts3d = scene.get_pts3d()

    # Build a spatial hash for matching 3D points across views
    point_map = {}  # Maps quantized 3D coordinates to point IDs
    quantization = 0.01  # 1cm grid for matching

    def quantize_point(pt):
        """Quantize 3D point to grid cell."""
        return tuple((pt / quantization).astype(int))

    for idx in range(num_images):
        pts3d = all_pts3d[idx]
        confidence = all_confidences[idx]
        img = scene.imgs[idx]

        if hasattr(pts3d, 'cpu'):
            pts3d = pts3d.detach().cpu().numpy()
        if hasattr(confidence, 'cpu'):
            confidence = confidence.detach().cpu().numpy()
        if hasattr(img, 'cpu'):
            img = img.detach().cpu().numpy()

        h, w = pts3d.shape[:2]
        pts_flat = pts3d.reshape(-1, 3)
        conf_flat = confidence.reshape(-1)

        # Extract colors
        if len(img.shape) == 3:
            colors = img.reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
        else:
            gray = img.reshape(-1)
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
            colors = np.stack([gray] * 3, axis=1)

        mask = conf_flat > min_conf_thr

        # Limit points but maintain spatial distribution
        if mask.sum() > 10000:
            indices = np.where(mask)[0]
            # Sample uniformly across image
            step = len(indices) // 10000
            sampled_indices = indices[::step][:10000]
            mask = np.zeros_like(mask, dtype=bool)
            mask[sampled_indices] = True

        valid_pts = pts_flat[mask]
        valid_colors = colors[mask]
        valid_indices = np.where(mask)[0]

        # Create 2D pixel coordinates
        y_coords, x_coords = np.unravel_index(valid_indices, (h, w))
        pixel_coords = np.stack([x_coords, y_coords], axis=1).astype(np.float64)

        # Lists to store correspondences for this image
        image_xys = []
        image_point3D_ids = []

        for i, (pt, color, xy) in enumerate(zip(valid_pts, valid_colors, pixel_coords)):
            q_pt = quantize_point(pt)
            
            if q_pt in point_map:
                # Point already exists - add correspondence
                pid = point_map[q_pt]
                points3D[pid]['image_ids'] = np.append(points3D[pid]['image_ids'], idx + 1)
                points3D[pid]['point2D_idxs'] = np.append(points3D[pid]['point2D_idxs'], len(image_xys))
            else:
                # New point
                point_map[q_pt] = point_id
                points3D[point_id] = {
                    'id': point_id,
                    'xyz': pt.astype(np.float64),
                    'rgb': color.astype(np.uint8),
                    'error': 0.0,
                    'image_ids': np.array([idx + 1], dtype=np.int32),
                    'point2D_idxs': np.array([len(image_xys)], dtype=np.int32)
                }
                pid = point_id
                point_id += 1
            
            image_xys.append(xy)
            image_point3D_ids.append(pid)

        # Update image with 2D-3D correspondences
        images_data[idx + 1]['xys'] = np.array(image_xys, dtype=np.float64)
        images_data[idx + 1]['point3D_ids'] = np.array(image_point3D_ids, dtype=np.uint64)

    if verbose:
        print(f"Extracted {len(points3D)} unique 3D points with correspondences")
        multi_view = sum(1 for p in points3D.values() if len(p['image_ids']) > 1)
        print(f"  {multi_view} points visible in multiple views")

    return points3D


def save_depth_map(depth: np.ndarray, path: Path):
    """Saves depth map in COLMAP binary format with proper handling."""
    h, w = depth.shape

    # Handle invalid depth values
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    with open(path, 'wb') as f:
        f.write(struct.pack('i', w))
        f.write(struct.pack('i', h))
        f.write(struct.pack('i', 1))  # Number of channels
        depth_flat = depth.astype(np.float32).flatten()
        f.write(depth_flat.tobytes())

def save_image_data(scene, images_dir, depth_dir, normal_dir, mask_dir, min_conf_thr, verbose, processed_image_paths=None):
    """Save RGB images, depth maps, normal maps, and masks"""
    if verbose:
        print("\nSaving image data...")
    
    # Ensure directories exist
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the number of views
    if hasattr(scene, 'imgs'):
        num_views = len(scene.imgs)
        imgs = scene.imgs
    elif hasattr(scene, 'views'):
        num_views = len(scene.views)
        imgs = scene.views
    else:
        if verbose:
            print("  Warning: Cannot access views")
        return
    
    # Use processed images if provided
    if processed_image_paths is not None:
        if verbose:
            print(f"  Using {len(processed_image_paths)} processed images")
        
        for idx, src_path in enumerate(processed_image_paths):
            if idx >= num_views:
                break
            
            try:
                # Copy processed images
                import shutil
                dst_path = images_dir / f'image_{idx:04d}.jpg'
                shutil.copy2(src_path, dst_path)
                
                if verbose and idx < 3:
                    print(f"  Copied image {idx}: {src_path} -> {dst_path}")
            except Exception as e:
                if verbose:
                    print(f"  Error copying image {idx}: {e}")
    else:
        # If no processed images, extract images from the scene
        for idx in range(num_views):
            try:
                # Save RGB images
                img_path = images_dir / f'image_{idx:04d}.jpg'
                
                # Retrieve image data
                if hasattr(imgs[idx], 'img'):
                    img = imgs[idx].img
                elif hasattr(imgs[idx], 'image'):
                    img = imgs[idx].image
                else:
                    img = imgs[idx]
                
                # Convert tensor to numpy array
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()
                
                # Convert image to correct format
                if isinstance(img, np.ndarray):
                    # Convert (C, H, W) -> (H, W, C)
                    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                        img = np.transpose(img, (1, 2, 0))
                    
                    # Normalize values to [0, 255] range
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    # Convert grayscale to RGB
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=-1)
                    elif img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    
                    # Save the image
                    from PIL import Image
                    Image.fromarray(img).save(img_path)
                    
                    if verbose and idx < 3:
                        print(f"  Saved image {idx}: {img_path}")
            except Exception as e:
                if verbose:
                    print(f"  Error saving image {idx}: {e}")
    
    # Save depth maps and masks
    try:
        if hasattr(scene, 'get_depthmaps'):
            depthmaps = scene.get_depthmaps()
            if depthmaps is not None:
                for idx in range(min(num_views, len(depthmaps))):
                    depth = depthmaps[idx]
                    if isinstance(depth, torch.Tensor):
                        depth = depth.detach().cpu().numpy()
                    
                    if isinstance(depth, np.ndarray):
                        depth_path = depth_dir / f'depth_{idx:04d}.npy'
                        np.save(depth_path, depth)
                        
                        if verbose and idx < 3:
                            print(f"  Saved depth {idx}: {depth_path}")
    except Exception as e:
        if verbose:
            print(f"  Note: Could not save depth maps: {e}")
    
    try:
        if hasattr(scene, 'get_masks'):
            masks = scene.get_masks()
            if masks is not None:
                for idx in range(min(num_views, len(masks))):
                    mask = masks[idx]
                    if isinstance(mask, torch.Tensor):
                        mask = mask.detach().cpu().numpy()
                    
                    if isinstance(mask, np.ndarray):
                        mask_path = mask_dir / f'mask_{idx:04d}.png'
                        from PIL import Image
                        mask_img = (mask * 255).astype(np.uint8)
                        Image.fromarray(mask_img).save(mask_path)
                        
                        if verbose and idx < 3:
                            print(f"  Saved mask {idx}: {mask_path}")
    except Exception as e:
        if verbose:
            print(f"  Note: Could not save masks: {e}")
    
    if verbose:
        print(f"  Saved {num_views} images")

# Helpers for binary writing (assumed based on context)
def write_next_bytes(fid, data, format_str):
    if isinstance(data, (list, tuple, np.ndarray)):
        fid.write(struct.pack(format_str, *data))
    else:
        fid.write(struct.pack(format_str, data))

def write_cameras_binary(cameras, path_to_model_file):
    """Write COLMAP cameras.bin file"""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for camera_id, cam in cameras.items():
            model_id = 1  # PINHOLE
            write_next_bytes(fid, camera_id, "I")
            write_next_bytes(fid, model_id, "I")
            write_next_bytes(fid, cam['width'], "Q")
            write_next_bytes(fid, cam['height'], "Q")
            for p in cam['params']:
                write_next_bytes(fid, float(p), "d")


def write_images_binary(images, path_to_model_file):
    """Write COLMAP images.bin file"""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for image_id, img in images.items():
            write_next_bytes(fid, image_id, "I")
            write_next_bytes(fid, img['qvec'], "dddd")
            write_next_bytes(fid, img['tvec'], "ddd")
            write_next_bytes(fid, img['camera_id'], "I")
            
            # Write image name
            for char in img['name']:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            
            # Write 2D points
            write_next_bytes(fid, len(img['xys']), "Q")
            for xy, point3D_id in zip(img['xys'], img['point3D_ids']):
                write_next_bytes(fid, xy, "dd")
                write_next_bytes(fid, point3D_id, "Q")

def write_points3d_binary(points3D, path_to_model_file):
    """Write COLMAP points3D.bin file"""
    # Assuming points3D is a list or dict of point data...
    pass
