# ============================================================================
# COLMAP Conversion (process3_11.py) - COMPLETE FIXED VERSION - ply success
# ============================================================================

import numpy as np
import cv2
from pathlib import Path
import struct
from scipy.spatial.transform import Rotation
import torch
from PIL import Image


def write_next_bytes(fid, data, format_str):
    """Helper function to write bytes to file"""
    if isinstance(data, (list, tuple, np.ndarray)):
        fid.write(struct.pack("<" + format_str, *data))
    else:
        fid.write(struct.pack("<" + format_str, data))


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
    """
    Write COLMAP points3D.bin file
    
    Args:
        points3D: list or dict of 3D point data
        path_to_model_file: path to points3D.bin
    """
    with open(path_to_model_file, "wb") as fid:
        # Write number of points
        if isinstance(points3D, dict):
            write_next_bytes(fid, len(points3D), "Q")
            points_iter = points3D.values()
        else:
            write_next_bytes(fid, len(points3D), "Q")
            points_iter = points3D
        
        # Write each point
        for point_id, point in enumerate(points_iter):
            # Handle both dict with 'id' key and list with index
            if isinstance(point, dict) and 'id' in point:
                pid = point['id']
            else:
                pid = point_id
            
            write_next_bytes(fid, pid, "Q")
            write_next_bytes(fid, point['xyz'], "ddd")
            write_next_bytes(fid, point['rgb'], "BBB")
            write_next_bytes(fid, point['error'], "d")
            
            # Write track
            track_length = len(point['image_ids'])
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_idx in zip(point['image_ids'], point['point2D_idxs']):
                write_next_bytes(fid, int(image_id), "I")
                write_next_bytes(fid, int(point2D_idx), "I")


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
    if processed_image_paths is not None and len(processed_image_paths) > 0:
        if verbose:
            print(f"  Using {len(processed_image_paths)} processed images")
        
        import shutil
        for idx, src_path in enumerate(processed_image_paths):
            if idx >= num_views:
                break
            
            try:
                # Copy processed images
                dst_path = images_dir / f'image_{idx:04d}.jpg'
                shutil.copy2(src_path, dst_path)
                
                if verbose and idx < 3:
                    print(f"  Copied image {idx}: {Path(src_path).name}")
            except Exception as e:
                if verbose:
                    print(f"  Error copying image {idx}: {e}")
    else:
        # If no processed images, extract images from the scene
        if verbose:
            print("  No processed images provided, extracting from scene...")
        
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
                    Image.fromarray(img).save(img_path)
                    
                    if verbose and idx < 3:
                        print(f"  Saved image {idx}: {img_path}")
            except Exception as e:
                if verbose:
                    print(f"  Error saving image {idx}: {e}")
    
    # Save depth maps
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
    
    # Save masks
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
                        mask_img = (mask * 255).astype(np.uint8)
                        Image.fromarray(mask_img).save(mask_path)
                        
                        if verbose and idx < 3:
                            print(f"  Saved mask {idx}: {mask_path}")
    except Exception as e:
        if verbose:
            print(f"  Note: Could not save masks: {e}")
    
    if verbose:
        print(f"  Completed saving {num_views} images")


def extract_scene_data(scene, min_conf_thr, verbose):
    """Extract cameras, images, and 3D points from MASt3R scene"""
    cameras = {}
    images_data = {}
    points3D = []
    
    if verbose:
        print("\nExtracting scene data...")
    
    # Check scene structure
    if hasattr(scene, 'imgs'):
        num_views = len(scene.imgs)
        imgs = scene.imgs
    elif hasattr(scene, 'views'):
        num_views = len(scene.views)
        imgs = scene.views
    else:
        num_views = 0
        imgs = []
    
    if verbose:
        print(f"Number of views: {num_views}")
    
    # Extract camera parameters and poses
    for idx in range(num_views):
        # Get image size
        if hasattr(scene, 'imshapes') and idx < len(scene.imshapes):
            height, width = scene.imshapes[idx]
        else:
            height, width = 192, 256
        
        # Get intrinsics
        fx = fy = 260.0
        cx = width / 2.0
        cy = height / 2.0
        
        try:
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
        except:
            pass
        
        cameras[idx] = {
            'model': 'PINHOLE',
            'width': int(width),
            'height': int(height),
            'params': [fx, fy, cx, cy]
        }
        
        # Get pose
        qvec = np.array([1.0, 0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.0, 0.0])
        
        try:
            if hasattr(scene, 'get_im_poses'):
                poses = scene.get_im_poses()
                if poses is not None and idx < len(poses):
                    pose = poses[idx]
                    if isinstance(pose, torch.Tensor):
                        pose = pose.detach().cpu().numpy()
                    
                    if isinstance(pose, np.ndarray) and pose.ndim == 2 and pose.shape == (4, 4):
                        det = np.linalg.det(pose)
                        if abs(det) > 1e-10:
                            pose_inv = np.linalg.inv(pose)
                            qvec, tvec = matrix_to_quaternion_translation(pose_inv)
        except:
            pass
        
        images_data[idx + 1] = {
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': idx,
            'name': f'image_{idx:04d}.jpg',
            'xys': np.array([]),
            'point3D_ids': np.array([])
        }
    
    # Extract 3D points WITH COLORS
    if verbose:
        print("\nExtracting 3D points with colors...")
    
    try:
        if hasattr(scene, 'get_pts3d'):
            pts3d = scene.get_pts3d()
            
            if pts3d is not None:
                # Handle list of arrays
                if isinstance(pts3d, list):
                    all_points = []
                    all_colors = []
                    
                    for view_idx, pts in enumerate(pts3d):
                        if isinstance(pts, torch.Tensor):
                            pts = pts.detach().cpu().numpy()
                        if isinstance(pts, np.ndarray):
                            all_points.append(pts.reshape(-1, 3))
                            
                            # Extract colors from corresponding image
                            if view_idx < len(imgs):
                                img = imgs[view_idx]
                                if isinstance(img, torch.Tensor):
                                    img = img.detach().cpu().numpy()
                                
                                # Convert image format
                                if img.ndim == 3:
                                    # (C, H, W) -> (H, W, C)
                                    if img.shape[0] in [1, 3, 4]:
                                        img = np.transpose(img, (1, 2, 0))
                                
                                # Normalize to 0-255
                                if img.max() <= 1.0:
                                    img = (img * 255).astype(np.uint8)
                                else:
                                    img = img.astype(np.uint8)
                                
                                # Handle grayscale
                                if img.ndim == 2 or img.shape[-1] == 1:
                                    img = np.stack([img.squeeze()] * 3, axis=-1)
                                
                                # Reshape to match points
                                img_flat = img.reshape(-1, 3)
                                all_colors.append(img_flat)
                            else:
                                # Default gray if no image available
                                n_pts = pts.reshape(-1, 3).shape[0]
                                all_colors.append(np.full((n_pts, 3), 128, dtype=np.uint8))
                    
                    pts3d_combined = np.vstack(all_points) if all_points else None
                    colors_combined = np.vstack(all_colors) if all_colors else None
                        
                elif isinstance(pts3d, torch.Tensor):
                    pts3d_combined = pts3d.detach().cpu().numpy().reshape(-1, 3)
                    
                    # Extract colors from first image
                    if len(imgs) > 0:
                        img = imgs[0]
                        if isinstance(img, torch.Tensor):
                            img = img.detach().cpu().numpy()
                        
                        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                            img = np.transpose(img, (1, 2, 0))
                        
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                        
                        if img.ndim == 2 or img.shape[-1] == 1:
                            img = np.stack([img.squeeze()] * 3, axis=-1)
                        
                        colors_combined = img.reshape(-1, 3)
                    else:
                        colors_combined = None
                        
                elif isinstance(pts3d, np.ndarray):
                    pts3d_combined = pts3d.reshape(-1, 3)
                    
                    # Extract colors from first image
                    if len(imgs) > 0:
                        img = imgs[0]
                        if isinstance(img, torch.Tensor):
                            img = img.detach().cpu().numpy()
                        
                        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                            img = np.transpose(img, (1, 2, 0))
                        
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                        
                        if img.ndim == 2 or img.shape[-1] == 1:
                            img = np.stack([img.squeeze()] * 3, axis=-1)
                        
                        colors_combined = img.reshape(-1, 3)
                    else:
                        colors_combined = None
                else:
                    pts3d_combined = None
                    colors_combined = None
                
                if pts3d_combined is not None and len(pts3d_combined) > 0:
                    # Get confidence
                    conf_combined = None
                    if hasattr(scene, 'get_conf'):
                        conf = scene.get_conf()
                        if conf is not None:
                            if isinstance(conf, list):
                                all_conf = []
                                for c in conf:
                                    if isinstance(c, torch.Tensor):
                                        c = c.detach().cpu().numpy()
                                    all_conf.append(c.flatten())
                                conf_combined = np.concatenate(all_conf) if all_conf else None
                            elif isinstance(conf, torch.Tensor):
                                conf_combined = conf.detach().cpu().numpy().flatten()
                            elif isinstance(conf, np.ndarray):
                                conf_combined = conf.flatten()
                    
                    # Ensure all arrays have the same size
                    min_size = len(pts3d_combined)
                    if colors_combined is not None:
                        min_size = min(min_size, len(colors_combined))
                    if conf_combined is not None:
                        min_size = min(min_size, len(conf_combined))
                    
                    pts3d_combined = pts3d_combined[:min_size]
                    if colors_combined is not None:
                        colors_combined = colors_combined[:min_size]
                    else:
                        colors_combined = np.full((min_size, 3), 128, dtype=np.uint8)
                    
                    # Filter by confidence
                    if conf_combined is not None and len(conf_combined) > 0:
                        conf_combined = conf_combined[:min_size]
                        mask = conf_combined >= min_conf_thr
                        pts3d_filtered = pts3d_combined[mask]
                        colors_filtered = colors_combined[mask]
                    else:
                        pts3d_filtered = pts3d_combined
                        colors_filtered = colors_combined
                    
                    # Create point cloud with colors
                    for pt, color in zip(pts3d_filtered, colors_filtered):
                        if np.all(np.isfinite(pt)):
                            points3D.append({
                                'xyz': pt,
                                'rgb': color.astype(np.uint8),  #use actual color
                                'error': 0.0,
                                'image_ids': np.array([]),
                                'point2D_idxs': np.array([])
                            })
                    
                    if verbose:
                        print(f"  Extracted {len(points3D)} 3D points with colors")
                        print(f"  Sample colors: {[p['rgb'].tolist() for p in points3D[:3]]}")
    except Exception as e:
        if verbose:
            print(f"  Error extracting 3D points: {e}")
        import traceback
        traceback.print_exc()
    
    if verbose:
        print(f"\nTotal: {len(cameras)} cameras, {len(images_data)} images, {len(points3D)} points")
    
    return cameras, images_data, points3D




def convert_mast3r_to_colmap(scene, output_dir, min_conf_thr=1.5, clean_depth=True, 
                            mask_images=True, verbose=True, processed_image_paths=None,
                            max_points=100000):
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
        print("\n" + "="*70)
        print("Converting MASt3R scene to COLMAP format")
        print("="*70)
        print(f"Output directory: {output_dir}")
    
    cameras, images_data, points3D = extract_scene_data(scene, min_conf_thr, verbose)

    #----------------------------down sampling
    if max_points is not None and len(points3D) > max_points:
        print(f"\nDownsampling 3D points from {len(points3D)} to {max_points}...")
        
        if isinstance(points3D, dict):
            all_ids = list(points3D.keys())
            sampled_ids = np.random.choice(all_ids, max_points, replace=False)
            points3D = {idx: points3D[idx] for idx in sampled_ids}
        elif isinstance(points3D, list):
            sampled_indices = np.random.choice(len(points3D), max_points, replace=False)
            points3D = [points3D[i] for i in sampled_indices]
        else:
            raise TypeError(f"points3D must be dict or list, got {type(points3D)}")
    #----------------------------down sampling   
    
    save_image_data(scene, images_dir, depth_dir, normal_dir, mask_dir, 
                    min_conf_thr, verbose, processed_image_paths=processed_image_paths)
    
    if verbose:
        print("\nWriting COLMAP binary files...")
    
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    if verbose:
        print(f"  ✓ cameras.bin ({len(cameras)} cameras)")
    
    write_images_binary(images_data, sparse_dir / "images.bin")
    if verbose:
        print(f"  ✓ images.bin ({len(images_data)} images)")
    
    write_points3d_binary(points3D, sparse_dir / "points3D.bin")
    if verbose:
        print(f"  ✓ points3D.bin ({len(points3D)} points)")
    
    if verbose:
        print("\n" + "="*70)
        print("✓ COLMAP conversion complete!")
        print("="*70)
    
    return output_dir
